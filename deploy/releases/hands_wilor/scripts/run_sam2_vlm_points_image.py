#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_idx}")
    return frame


def put_caption(frame: np.ndarray, caption: str, frame_idx: int) -> None:
    text = f"{frame_idx:04d}  {caption}"
    cv2.rectangle(frame, (0, frame.shape[0] - 34), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def prompt_by_frame(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"point prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def selected_frames(frame_start: int, frame_end: int) -> list[int]:
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    return list(range(frame_start, frame_end + 1))


def prompt_points(prompt: dict, from_size: tuple[int, int], to_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    positives = prompt.get("positive_points", [])
    negatives = prompt.get("negative_points", [])
    if not positives:
        raise RuntimeError(f"prompt frame {prompt['frame_idx']} has no positive points")
    scale = np.asarray([to_size[0] / from_size[0], to_size[1] / from_size[1]], dtype=np.float32)
    pos = np.asarray([[float(point["x"]), float(point["y"])] for point in positives], dtype=np.float32) * scale[None, :]
    neg = np.asarray([[float(point["x"]), float(point["y"])] for point in negatives], dtype=np.float32) * scale[None, :] if negatives else np.zeros((0, 2), dtype=np.float32)
    points = np.vstack([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    return points, labels


def prompt_box(prompt: dict, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray | None:
    box = prompt.get("bbox_xyxy", [])
    if len(box) < 4:
        return None
    scale = np.asarray([to_size[0] / from_size[0], to_size[1] / from_size[1], to_size[0] / from_size[0], to_size[1] / from_size[1]], dtype=np.float32)
    arr = np.asarray([float(v) for v in box[:4]], dtype=np.float32) * scale
    arr[[0, 2]] = np.clip(arr[[0, 2]], 0.0, to_size[0] - 1.0)
    arr[[1, 3]] = np.clip(arr[[1, 3]], 0.0, to_size[1] - 1.0)
    if arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    return arr


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)
    return box, float(xs.size), center


def scale_xyxy(values: list[float], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[float]:
    sx = to_size[0] / float(from_size[0])
    sy = to_size[1] / float(from_size[1])
    return [float(values[0] * sx), float(values[1] * sy), float(values[2] * sx), float(values[3] * sy)]


def scale_xy(values: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> list[float]:
    sx = to_size[0] / float(from_size[0])
    sy = to_size[1] / float(from_size[1])
    return [float(values[0] * sx), float(values[1] * sy)]


def point_hits(mask: np.ndarray, points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return int(mask[y, x].sum())


def prompt_extent_area(points: np.ndarray, image_size: tuple[int, int], margin_px: float) -> float:
    if len(points) == 0:
        return float(image_size[0] * image_size[1])
    lo = np.min(points, axis=0) - float(margin_px)
    hi = np.max(points, axis=0) + float(margin_px)
    lo = np.clip(lo, [0.0, 0.0], [image_size[0] - 1.0, image_size[1] - 1.0])
    hi = np.clip(hi, [0.0, 0.0], [image_size[0] - 1.0, image_size[1] - 1.0])
    return max(1.0, float((hi[0] - lo[0] + 1.0) * (hi[1] - lo[1] + 1.0)))


def select_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    min_area_px: int,
    image_size: tuple[int, int],
    max_prompt_area_ratio: float,
    max_area_fraction: float,
    min_positive_hit_fraction: float,
    max_negative_hits: int,
    prompt_area_margin_px: float,
    score_tie_margin: float,
    selection_mode: str,
) -> tuple[np.ndarray | None, dict]:
    pos = points[labels == 1]
    neg = points[labels == 0]
    prompt_area = prompt_extent_area(pos, image_size, prompt_area_margin_px)
    max_area = min(prompt_area * float(max_prompt_area_ratio), float(image_size[0] * image_size[1]) * float(max_area_fraction))
    required_pos = int(math.ceil(len(pos) * float(min_positive_hit_fraction)))
    candidates = []
    for i, (mask_raw, score) in enumerate(zip(masks, scores)):
        mask = mask_raw.astype(bool)
        area = int(mask.sum())
        pos_hits = point_hits(mask, pos)
        neg_hits = point_hits(mask, neg)
        valid = area >= min_area_px and area <= max_area and pos_hits >= required_pos and neg_hits <= int(max_negative_hits)
        candidates.append(
            {
                "candidate": int(i),
                "sam_score": float(score),
                "area_px": area,
                "max_area_px": float(max_area),
                "max_area_fraction": float(max_area_fraction),
                "prompt_extent_area_px": float(prompt_area),
                "positive_hits": int(pos_hits),
                "positive_points": int(len(pos)),
                "required_positive_hits": int(required_pos),
                "min_positive_hit_fraction": float(min_positive_hit_fraction),
                "negative_hits": int(neg_hits),
                "negative_points": int(len(neg)),
                "max_negative_hits": int(max_negative_hits),
                "accepted_by_prompt_contract": bool(valid),
            }
        )
    valid_indices = [row["candidate"] for row in candidates if row["accepted_by_prompt_contract"]]
    if not valid_indices:
        return None, {"reason": "no_candidate_satisfies_prompt_contract", "candidates": candidates}
    if selection_mode == "prompt_hits":
        best_fraction = max(candidates[idx]["positive_hits"] / max(1, candidates[idx]["positive_points"]) for idx in valid_indices)
        near_best = [
            idx
            for idx in valid_indices
            if candidates[idx]["positive_hits"] / max(1, candidates[idx]["positive_points"]) >= best_fraction - float(score_tie_margin)
        ]
        best = max(
            near_best,
            key=lambda idx: (
                candidates[idx]["positive_hits"] / max(1, candidates[idx]["positive_points"]),
                -candidates[idx]["negative_hits"],
                float(scores[idx]),
                -abs(
                    math.log(
                        max(1.0, float(candidates[idx]["area_px"]))
                        / max(1.0, float(candidates[idx]["prompt_extent_area_px"]))
                    )
                ),
            ),
        )
    elif selection_mode == "sam_score_compact":
        best = max(
            valid_indices,
            key=lambda idx: (
                float(scores[idx]),
                -candidates[idx]["negative_hits"],
                -abs(
                    math.log(
                        max(1.0, float(candidates[idx]["area_px"]))
                        / max(1.0, float(candidates[idx]["prompt_extent_area_px"]))
                    )
                ),
                candidates[idx]["positive_hits"] / max(1, candidates[idx]["positive_points"]),
            ),
        )
    else:
        raise RuntimeError(f"unknown SAM2 selection mode: {selection_mode}")
    return masks[best].astype(bool), {
        "reason": "ok",
        "selected_candidate": int(best),
        "selection_mode": selection_mode,
        "candidates": candidates,
    }


def save_candidate_review(
    output_dir: Path,
    source_idx: int,
    image_bgr: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    report: dict,
) -> list[str]:
    review_dir = output_dir / "sam2_candidate_review"
    mask_dir = output_dir / "sam2_candidate_masks"
    review_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    candidates = report.get("candidates", [])
    paths = []
    for row in candidates:
        idx = int(row["candidate"])
        mask = masks[idx].astype(bool)
        mask_path = mask_dir / f"{source_idx:06d}_candidate_{idx}.png"
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        row["candidate_mask_path"] = str(mask_path)
        panel = image_bgr.copy()
        tint = np.zeros_like(panel)
        tint[:, :, 1] = 220
        tint[:, :, 2] = 255
        panel[mask] = cv2.addWeighted(panel, 0.50, tint, 0.50, 0.0)[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        label = (
            f"{source_idx} cand={idx} score={float(scores[idx]):.3f} "
            f"pos={row['positive_hits']}/{row['positive_points']} "
            f"neg={row['negative_hits']}/{row['negative_points']} "
            f"area={row['area_px']}"
        )
        if row.get("accepted_by_prompt_contract"):
            label += " ACCEPT"
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        path = review_dir / f"{source_idx:06d}_candidate_{idx}.jpg"
        if not cv2.imwrite(str(path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
            raise RuntimeError(f"failed to write {path}")
        paths.append(str(path))
    return paths


def run_predictor(
    args: argparse.Namespace,
    frames: list[int],
    prompts: dict[int, dict],
    prompt_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[dict[int, dict], list[dict], Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 image predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam2(args.model_cfg, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    cap, _ = open_video(args.clip)
    frame_dir = args.output_dir / "sam2_image_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "sam2_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}
    reports: list[dict] = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for source_idx in frames:
                prompt = prompts.get(source_idx)
                if prompt is None or not prompt.get("target_visible") or not prompt.get("positive_points"):
                    results[source_idx] = {"visible": False, "area_px": 0.0, "reason": "no_visible_point_prompt"}
                    continue
                frame = read_video_frame(cap, source_idx)
                image_bgr = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
                image_path = frame_dir / f"{source_idx:06d}.jpg"
                if not cv2.imwrite(str(image_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                    raise RuntimeError(f"failed to write {image_path}")
                points, labels = prompt_points(prompt, prompt_size, image_size)
                box = prompt_box(prompt, prompt_size, image_size) if args.use_box else None
                predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                    normalize_coords=True,
                )
                selected, report = select_mask(
                    masks,
                    scores,
                    points,
                    labels,
                    int(args.min_area_px),
                    image_size,
                    float(args.max_prompt_area_ratio),
                    float(args.max_area_fraction),
                    float(args.min_positive_hit_fraction),
                    int(args.max_negative_hits),
                    float(args.prompt_area_margin_px),
                    float(args.score_tie_margin),
                    str(args.selection_mode),
                )
                report.update({"frame_idx": int(source_idx), "used_box": bool(box is not None)})
                if args.save_candidate_masks:
                    report["candidate_review_paths"] = save_candidate_review(args.output_dir, source_idx, image_bgr, masks, scores, report)
                reports.append(report)
                if selected is None:
                    results[source_idx] = {"visible": False, "area_px": 0.0, "reason": report["reason"]}
                    continue
                box_small, area_small, center_small = mask_box(selected)
                if box_small is None or center_small is None:
                    results[source_idx] = {"visible": False, "area_px": 0.0, "reason": "empty_selected_mask"}
                    continue
                mask_path = mask_dir / f"{source_idx:06d}.png"
                if not cv2.imwrite(str(mask_path), selected.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write {mask_path}")
                results[source_idx] = {
                    "visible": True,
                    "bbox_xyxy": scale_xyxy(box_small, image_size, (int(args.source_width), int(args.source_height))),
                    "center_xy": scale_xy(center_small, image_size, (int(args.source_width), int(args.source_height))),
                    "area_px": float(area_small * (int(args.source_width) / float(image_size[0])) * (int(args.source_height) / float(image_size[1]))),
                    "mask_path": str(mask_path),
                    "reason": "ok",
                }
    finally:
        cap.release()
    return results, reports, frame_dir


def render(args: argparse.Namespace, frames: list[int], results: dict[int, dict], track_id: str, image_size: tuple[int, int]) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_dir / "sam2_image_points_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    try:
        for source_idx in frames:
            image = read_video_frame(cap, source_idx)
            image = cv2.resize(image, (args.render_width, height), interpolation=cv2.INTER_AREA)
            result = results.get(source_idx, {})
            if result.get("visible") and result.get("mask_path"):
                mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read {result['mask_path']}")
                mask = cv2.resize(mask, (args.render_width, height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(image)
                tint[:, :, 0] = 255
                tint[:, :, 2] = 255
                image[mask] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask]
            put_caption(image, f"SAM2 image VLM-point {track_id}", source_idx)
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts, payload = prompt_by_frame(args.point_prompts)
    frames = selected_frames(int(args.frame_start), int(args.frame_end))
    cap, info = open_video(args.clip)
    cap.release()
    source_size = (info.width, info.height)
    if args.source_width is None:
        args.source_width = info.width
    if args.source_height is None:
        args.source_height = info.height
    image_height = int(round(info.height * int(args.sam2_image_width) / info.width))
    image_size = (int(args.sam2_image_width), image_height)
    prompt_size = (int(payload["prompt_image_width"]), int(round(int(payload["prompt_image_width"]) * info.height / info.width)))
    results, reports, _ = run_predictor(args, frames, prompts, prompt_size, image_size)
    video = render(args, frames, results, str(payload["track_id"]), image_size)
    visible = sum(1 for row in results.values() if row.get("visible"))
    qc = {
        "status": "ok",
        "backend": "SAM2 image predictor from VLM point prompts",
        "clip": str(args.clip),
        "source_image_size": [int(source_size[0]), int(source_size[1])],
        "point_prompts": str(args.point_prompts),
        "track_id": payload["track_id"],
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "prompt_frames": sorted(int(idx) for idx in prompts),
        "visible_frames": visible,
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "sam2_image_size": [int(image_size[0]), int(image_size[1])],
        "prompt_image_size": [int(prompt_size[0]), int(prompt_size[1])],
        "use_box": bool(args.use_box),
        "min_area_px": int(args.min_area_px),
        "max_prompt_area_ratio": float(args.max_prompt_area_ratio),
        "max_area_fraction": float(args.max_area_fraction),
        "min_positive_hit_fraction": float(args.min_positive_hit_fraction),
        "max_negative_hits": int(args.max_negative_hits),
        "prompt_area_margin_px": float(args.prompt_area_margin_px),
        "score_tie_margin": float(args.score_tie_margin),
        "selection_mode": str(args.selection_mode),
        "elapsed_s": time.time() - started,
        "outputs": {
            "sam2_track": str(args.output_dir / "sam2_track.json"),
            "sam2_masks": str(args.output_dir / "sam2_masks"),
            "overlay": str(video),
        },
        "reports": reports,
    }
    (args.output_dir / "sam2_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "qc_sam2_image_points.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "reports"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--max-prompt-area-ratio", type=float, default=3.5)
    parser.add_argument("--max-area-fraction", type=float, default=1.0)
    parser.add_argument("--min-positive-hit-fraction", type=float, default=1.0)
    parser.add_argument("--max-negative-hits", type=int, default=0)
    parser.add_argument("--prompt-area-margin-px", type=float, default=40.0)
    parser.add_argument("--score-tie-margin", type=float, default=0.08)
    parser.add_argument("--selection-mode", choices=["prompt_hits", "sam_score_compact"], default="prompt_hits")
    parser.add_argument("--use-box", action="store_true")
    parser.add_argument("--save-candidate-masks", action="store_true")
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--source-height", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
