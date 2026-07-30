#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fuse_v1_full_fidelity import DEFAULT_CLIP, load_json, open_video, put_caption, read_video_frame
from run_sam2_object_track import extract_frames, mask_box


def prompt_by_frame(path: Path | None) -> tuple[dict[int, dict], dict]:
    if path is None:
        return {}, {}
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"point prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def selected_frames(frame_start: int, frame_end: int) -> list[dict]:
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    return [{"frame_idx": idx} for idx in range(frame_start, frame_end + 1)]


def points_from_prompt(prompt: dict, prompt_size: tuple[int, int]) -> tuple[list[list[float]], list[int]]:
    width, height = prompt_size
    coords: list[list[float]] = []
    labels: list[int] = []
    for row in prompt.get("positive_points", []):
        coords.append([float(row["x"]) / float(width), float(row["y"]) / float(height)])
        labels.append(1)
    for row in prompt.get("negative_points", []):
        coords.append([float(row["x"]) / float(width), float(row["y"]) / float(height)])
        labels.append(0)
    return coords, labels


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_frame_outputs(outputs: Any) -> list[tuple[int, np.ndarray, float | None, list[float] | None]]:
    if outputs is None:
        return []
    if isinstance(outputs, dict) and "out_binary_masks" in outputs:
        masks = tensor_to_numpy(outputs["out_binary_masks"])
        obj_ids = tensor_to_numpy(outputs.get("out_obj_ids", np.arange(len(masks)))).astype(int).reshape(-1)
        scores_raw = outputs.get("out_probs", outputs.get("out_scores", None))
        scores = None if scores_raw is None else tensor_to_numpy(scores_raw).reshape(-1)
        boxes_raw = outputs.get("out_boxes_xywh", None)
        boxes = None if boxes_raw is None else tensor_to_numpy(boxes_raw).reshape((-1, 4))
        rows = []
        for i, raw_mask in enumerate(masks):
            mask = np.asarray(raw_mask)
            while mask.ndim > 2:
                mask = mask[0]
            rows.append(
                (
                    int(obj_ids[i]) if i < len(obj_ids) else int(i),
                    mask > 0,
                    None if scores is None or i >= len(scores) else float(scores[i]),
                    None if boxes is None or i >= len(boxes) else [float(v) for v in boxes[i]],
                )
            )
        return rows
    if isinstance(outputs, dict):
        rows = []
        for obj_id, raw_mask in outputs.items():
            if isinstance(obj_id, str) and not obj_id.isdigit():
                continue
            mask = tensor_to_numpy(raw_mask)
            while mask.ndim > 2:
                mask = mask[0]
            rows.append((int(obj_id), mask > 0, None, None))
        return rows
    raise RuntimeError(f"unsupported SAM3 output type: {type(outputs)!r}")


def choose_mask(
    outputs: Any,
    previous_obj_id: int | None,
    previous_mask: np.ndarray | None,
) -> tuple[int, np.ndarray, float | None, list[float] | None] | None:
    candidates = []
    for obj_id, mask, score, box in normalize_frame_outputs(outputs):
        area = int(mask.sum())
        if area <= 0:
            continue
        continuity = 0.0
        if previous_mask is not None and previous_mask.shape == mask.shape:
            inter = np.logical_and(previous_mask, mask).sum()
            union = np.logical_or(previous_mask, mask).sum()
            continuity = 0.0 if union == 0 else float(inter / union)
        same_id = 1.0 if previous_obj_id is not None and int(obj_id) == int(previous_obj_id) else 0.0
        score_term = 0.0 if score is None else float(score)
        candidates.append((same_id, continuity, score_term, area, obj_id, mask, score, box))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]), reverse=True)
    _, _, _, _, obj_id, mask, score, box = candidates[0]
    return int(obj_id), mask, score, box


def load_predictor(args: argparse.Namespace):
    if args.sam3_repo and str(args.sam3_repo) not in sys.path:
        sys.path.insert(0, str(args.sam3_repo))
    from sam3.model_builder import build_sam3_predictor  # noqa: PLC0415

    return build_sam3_predictor(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        version=args.sam3_version,
        compile=bool(args.compile),
        max_num_objects=int(args.max_num_objects),
        multiplex_count=int(args.multiplex_count),
    )


def run_sam3(args: argparse.Namespace, frames: list[dict], frame_dir: Path, prompts: dict[int, dict], prompt_payload: dict) -> dict[int, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 referring video predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = load_predictor(args)
    selected = [int(frame["frame_idx"]) for frame in frames]
    local_by_source = {source_idx: local for local, source_idx in enumerate(selected)}
    prompt_frames = [int(part) for raw in args.prompt_frames for part in raw.split(",") if part.strip()]
    missing = [idx for idx in prompt_frames if idx not in local_by_source]
    if missing:
        raise RuntimeError(f"SAM3 prompt frames outside selected range: {missing}")

    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": bool(args.offload_state_to_cpu),
        }
    )
    session_id = response["session_id"]
    previous_obj_id: int | None = None
    previous_mask: np.ndarray | None = None
    try:
        for source_idx in prompt_frames:
            request: dict[str, Any] = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": local_by_source[source_idx],
                "text": args.text_prompt,
                "output_prob_thresh": float(args.output_prob_thresh),
            }
            if source_idx in prompts:
                prompt_size = (
                    int(prompt_payload["prompt_image_width"]),
                    int(round(int(prompt_payload["prompt_image_width"]) * 9 / 16)),
                )
                points, labels = points_from_prompt(prompts[source_idx], prompt_size)
                if points:
                    request["points"] = points
                    request["point_labels"] = labels
            if previous_obj_id is not None and args.reuse_obj_id:
                request["obj_id"] = int(previous_obj_id)
            response = predictor.handle_request(request=request)
            selected_mask = choose_mask(response.get("outputs"), previous_obj_id, previous_mask)
            if selected_mask is not None:
                previous_obj_id, previous_mask = selected_mask[0], selected_mask[1]

        mask_dir = args.output_dir / "sam3_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, dict] = {}
        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "output_prob_thresh": float(args.output_prob_thresh),
            }
        ):
            source_idx = selected[int(response["frame_index"])]
            selected_mask = choose_mask(response.get("outputs"), previous_obj_id, previous_mask)
            if selected_mask is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            obj_id, mask, score, box = selected_mask
            previous_obj_id = obj_id
            previous_mask = mask
            box_small, area_small, center_small = mask_box(mask.astype(np.uint8))
            if box_small is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            results[source_idx] = {
                "visible": True,
                "obj_id": int(obj_id),
                "bbox_xyxy": [float(v) for v in box_small],
                "center_xy": center_small.astype(float).tolist(),
                "area_px": float(area_small),
                "score": None if score is None else float(score),
                "sam3_box_xywh": box,
                "mask_path": str(mask_path),
            }
        return results
    finally:
        predictor.handle_request(
            request={
                "type": "close_session",
                "session_id": session_id,
            }
        )
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()


def render(args: argparse.Namespace, frames: list[dict], results: dict[int, dict]) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_dir / "sam3_referring_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    still_dir = args.output_dir / "review_stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    review_indices = set(int(part) for raw in args.review_frames for part in raw.split(",") if part.strip())
    for frame in frames:
        source_idx = int(frame["frame_idx"])
        image_full = read_video_frame(cap, source_idx)
        image = cv2.resize(image_full, (args.render_width, height), interpolation=cv2.INTER_AREA)
        result = results.get(source_idx, {})
        if result.get("visible") and result.get("mask_path"):
            mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read {result['mask_path']}")
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (args.render_width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0
            tint = np.zeros_like(image)
            tint[:, :, 0] = 255
            tint[:, :, 2] = 255
            image[mask_bool] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask_bool]
        put_caption(image, f"SAM3 text track: {args.text_prompt}", source_idx)
        writer.write(image)
        if source_idx in review_indices:
            if not cv2.imwrite(str(still_dir / f"{source_idx:06d}.jpg"), image):
                raise RuntimeError(f"failed to write review still for {source_idx}")
    writer.release()
    cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts, payload = prompt_by_frame(args.point_prompts)
    frames = selected_frames(int(args.frame_start), int(args.frame_end))
    frame_dir = extract_frames(args.clip, frames, list(range(len(frames))), args.output_dir, int(args.sam3_image_width))
    results = run_sam3(args, frames, frame_dir, prompts, payload)
    video = render(args, frames, results)
    visible = sum(1 for row in results.values() if row.get("visible"))
    areas = [float(row["area_px"]) for row in results.values() if row.get("visible")]
    qc = {
        "status": "ok",
        "backend": f"SAM3 referring video segmentation ({args.sam3_version})",
        "clip": str(args.clip),
        "text_prompt": args.text_prompt,
        "point_prompts": None if args.point_prompts is None else str(args.point_prompts),
        "track_id": payload.get("track_id"),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "prompt_frames": [int(part) for raw in args.prompt_frames for part in raw.split(",") if part.strip()],
        "visible_frames": visible,
        "visible_rate": visible / max(1, len(frames)),
        "area_median": None if not areas else float(np.median(areas)),
        "area_p05": None if not areas else float(np.percentile(areas, 5)),
        "area_p95": None if not areas else float(np.percentile(areas, 95)),
        "sam3_version": args.sam3_version,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "elapsed_s": time.time() - started,
        "outputs": {
            "sam3_track": str(args.output_dir / "sam3_track.json"),
            "sam3_masks": str(args.output_dir / "sam3_masks"),
            "overlay": str(video),
            "review_stills": str(args.output_dir / "review_stills"),
        },
    }
    (args.output_dir / "sam3_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "qc_sam3_referring_track.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--point-prompts", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--prompt-frames", nargs="+", required=True)
    parser.add_argument("--review-frames", nargs="+", default=["678,720,797,858,880,900,918"])
    parser.add_argument("--sam3-repo", type=Path)
    parser.add_argument("--sam3-version", choices=["sam3", "sam3.1"], default="sam3.1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sam3-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--output-prob-thresh", type=float, default=0.35)
    parser.add_argument("--max-num-objects", type=int, default=8)
    parser.add_argument("--multiplex-count", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--reuse-obj-id", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
