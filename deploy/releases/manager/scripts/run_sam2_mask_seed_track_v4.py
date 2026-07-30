#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, list[float] | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    return (
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        float(xs.size),
        [float(xs.mean()), float(ys.mean())],
    )


def extract_frames(clip: Path, source_frames: list[int], output_dir: Path, image_width: int) -> tuple[Path, float]:
    frame_dir = output_dir / "sam2_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = open_video(clip)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid video dimensions")
    scale = float(image_width) / float(width)
    image_height = int(round(height * scale))
    try:
        for local_idx, source_idx in enumerate(source_frames):
            frame = read_frame(cap, source_idx)
            frame = cv2.resize(frame, (int(image_width), image_height), interpolation=cv2.INTER_AREA)
            path = frame_dir / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write {path}")
    finally:
        cap.release()
    return frame_dir, scale


def load_seed_mask(path: Path, scale: float) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read seed mask: {path}")
    if scale != 1.0:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return mask > 0


def render_overlay(
    clip: Path,
    source_frames: list[int],
    results: dict[int, dict],
    output_dir: Path,
    render_width: int,
    track_id: str,
) -> Path:
    cap = open_video(clip)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    render_height = int(round(height * float(render_width) / float(width)))
    out = output_dir / "sam2_mask_seed_track_overlay.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_width, render_height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {out}")
    try:
        for source_idx in source_frames:
            image = read_frame(cap, source_idx)
            image = cv2.resize(image, (render_width, render_height), interpolation=cv2.INTER_AREA)
            row = results.get(source_idx, {})
            if row.get("visible"):
                mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read {row['mask_path']}")
                mask = cv2.resize(mask, (render_width, render_height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(image)
                tint[:, :, 0] = 255
                tint[:, :, 2] = 255
                image[mask] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask]
            cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"frame {source_idx}  SAM2 mask-seed track  {track_id}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_frames = list(range(int(args.frame_start), int(args.frame_end) + 1))
    if int(args.seed_frame) not in source_frames:
        raise RuntimeError("--seed-frame must lie inside frame range")
    frame_dir, scale = extract_frames(args.clip, source_frames, args.output_dir, int(args.sam2_image_width))
    cap = open_video(args.clip)
    try:
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("invalid source video dimensions")
    seed_local = source_frames.index(int(args.seed_frame))
    seed_mask = load_seed_mask(args.seed_mask, scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 mask-seed tracking requires CUDA")
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=device, vos_optimized=False)
    mask_dir = args.output_dir / "sam2_masks"
    mask_dir.mkdir(exist_ok=True)
    results: dict[int, dict] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=seed_local, obj_id=1, mask=seed_mask)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            ids = [int(v) for v in out_obj_ids]
            source_idx = source_frames[int(out_frame_idx)]
            if 1 not in ids:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            obj_pos = ids.index(1)
            mask_small = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0).astype(np.uint8)
            mask = cv2.resize(mask_small, (source_width, source_height), interpolation=cv2.INTER_NEAREST)
            box, area, center = mask_box(mask)
            if box is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            results[source_idx] = {
                "visible": True,
                "source": "sam2_mask_seed_propagation",
                "seed_frame": int(args.seed_frame),
                "bbox_xyxy": box,
                "center_xy": center,
                "area_px": area,
                "mask_path": str(mask_path),
            }
    overlay = render_overlay(args.clip, source_frames, results, args.output_dir, int(args.render_width), args.track_id)
    report = {
        "status": "ok",
        "method": "sam2_mask_seed_track_v4",
        "track_id": args.track_id,
        "clip": str(args.clip),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "seed_frame": int(args.seed_frame),
        "seed_mask": str(args.seed_mask),
        "frames": int(len(source_frames)),
        "visible_frames": int(sum(1 for row in results.values() if row.get("visible"))),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "scale": float(scale),
        "elapsed_s": time.time() - started,
        "outputs": {
            "track": str(args.output_dir / "sam2_mask_seed_track.json"),
            "masks": str(mask_dir),
            "overlay": str(overlay),
        },
    }
    (args.output_dir / "sam2_mask_seed_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "qc_sam2_mask_seed_track_v4.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--seed-mask", type=Path, required=True)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
