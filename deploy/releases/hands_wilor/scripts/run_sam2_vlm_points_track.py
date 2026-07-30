#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


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


def extract_frames(clip: Path, frames: list[dict], output_dir: Path, image_width: int) -> Path:
    frame_dir = output_dir / "sam2_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(clip)
    image_height = int(round(info.height * image_width / info.width))
    try:
        for local_idx, frame in enumerate(frames):
            source_idx = int(frame["frame_idx"])
            frame = read_video_frame(cap, source_idx)
            resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
            path = frame_dir / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write {path}")
    finally:
        cap.release()
    return frame_dir


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)
    return box, float(xs.size), center


def prompt_by_frame(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"point prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def selected_frames(frame_start: int, frame_end: int) -> list[dict]:
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    return [{"frame_idx": idx} for idx in range(frame_start, frame_end + 1)]


def prompt_points(prompt: dict, scale: float) -> tuple[np.ndarray, np.ndarray]:
    positives = prompt.get("positive_points", [])
    negatives = prompt.get("negative_points", [])
    if not positives:
        raise RuntimeError(f"prompt frame {prompt['frame_idx']} has no positive points")
    points = [[float(point["x"]) * scale, float(point["y"]) * scale] for point in positives]
    labels = [1] * len(points)
    for point in negatives:
        points.append([float(point["x"]) * scale, float(point["y"]) * scale])
        labels.append(0)
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def run_sam2(args: argparse.Namespace, frames: list[dict], prompts: dict[int, dict], frame_dir: Path, scale: float) -> dict[int, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 video predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=device, vos_optimized=False)
    selected = [int(frame["frame_idx"]) for frame in frames]
    local_by_source = {source_idx: local for local, source_idx in enumerate(selected)}
    prompt_frames = [int(part) for raw in args.prompt_frames for part in raw.split(",") if part.strip()]
    missing = [idx for idx in prompt_frames if idx not in prompts or idx not in local_by_source]
    if missing:
        raise RuntimeError(f"SAM2 prompt frames missing from point prompts or selected range: {missing}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for source_idx in prompt_frames:
            point_coords, point_labels = prompt_points(prompts[source_idx], scale)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_by_source[source_idx],
                obj_id=1,
                points=point_coords,
                labels=point_labels,
            )
        mask_dir = args.output_dir / "sam2_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            ids = [int(v) for v in out_obj_ids]
            source_idx = selected[int(out_frame_idx)]
            if 1 not in ids:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            obj_pos = ids.index(1)
            mask = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0).astype(np.uint8)
            box_small, area_small, center_small = mask_box(mask)
            if box_small is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            inv = 1.0 / scale
            results[source_idx] = {
                "visible": True,
                "bbox_xyxy": [float(v * inv) for v in box_small],
                "center_xy": (center_small * inv).astype(float).tolist(),
                "area_px": float(area_small * inv * inv),
                "mask_path": str(mask_path),
            }
    return results


def render(args: argparse.Namespace, frames: list[dict], results: dict[int, dict], track_id: str) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_dir / "sam2_points_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    for frame in frames:
        source_idx = int(frame["frame_idx"])
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
        put_caption(image, f"SAM2 VLM-point {track_id}", source_idx)
        writer.write(image)
    writer.release()
    cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts, payload = prompt_by_frame(args.point_prompts)
    frames = selected_frames(int(args.frame_start), int(args.frame_end))
    frame_dir = extract_frames(args.clip, frames, args.output_dir, int(args.sam2_image_width))
    cap, info = open_video(args.clip)
    cap.release()
    scale = args.sam2_image_width / float(info.width)
    results = run_sam2(args, frames, prompts, frame_dir, scale)
    video = render(args, frames, results, str(payload["track_id"]))
    visible = sum(1 for row in results.values() if row.get("visible"))
    qc = {
        "status": "ok",
        "backend": "SAM2 propagation from VLM point prompts",
        "clip": str(args.clip),
        "point_prompts": str(args.point_prompts),
        "track_id": payload["track_id"],
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "prompt_frames": [int(part) for raw in args.prompt_frames for part in raw.split(",") if part.strip()],
        "visible_frames": visible,
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "elapsed_s": time.time() - started,
        "outputs": {
            "sam2_track": str(args.output_dir / "sam2_track.json"),
            "sam2_masks": str(args.output_dir / "sam2_masks"),
            "overlay": str(video),
        },
    }
    (args.output_dir / "sam2_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "qc_sam2_vlm_points_track.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
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
    parser.add_argument("--prompt-frames", nargs="+", required=True)
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
