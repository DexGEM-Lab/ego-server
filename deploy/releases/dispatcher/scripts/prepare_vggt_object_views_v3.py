#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MEASURED_OBJECT_STATUSES = {
    "measured_plan_sam",
    "measured_plan_sam_vlm_verified",
    "measured_sam2_vlm_points",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_path(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        try:
            rel = direct.relative_to(remote_root)
        except ValueError:
            rel = None
        if rel is not None:
            candidate = local_root / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read video frame {frame_idx}")
    return frame


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("object mask has no positive pixels")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def render_review(raw: np.ndarray, mask: np.ndarray, masked: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros_like(raw)
    mask_rgb[:, :, 1] = mask
    overlay = cv2.addWeighted(raw, 0.72, mask_rgb, 0.28, 0.0)
    return np.concatenate([raw, overlay, masked], axis=1)


def select_frames(frames: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        obj = frame.get("object") or {}
        if str(obj.get("status")) not in MEASURED_OBJECT_STATUSES:
            continue
        if args.track_id is not None and str(obj.get("track_id")) != str(args.track_id):
            continue
        if not obj.get("mask_path"):
            raise RuntimeError(f"frame {frame_idx} is measured but lacks mask_path")
        selected.append(frame)
    if not selected:
        raise RuntimeError("no measured object frames selected")
    return selected


def prepare(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    review_dir = args.output_dir / "review"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    review_dir.mkdir(exist_ok=True)

    payload = load_json(args.annotations)
    selected = select_frames(payload["frames"], args)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_width <= 0 or video_height <= 0:
        raise RuntimeError("video reports invalid dimensions")

    frame_reports = []
    for seq_i, frame in enumerate(selected):
        frame_idx = int(frame["frame_idx"])
        raw = read_frame(cap, frame_idx)
        if raw.shape[:2] != (video_height, video_width):
            raise RuntimeError(f"frame {frame_idx} has unexpected shape {raw.shape}")
        obj = frame["object"]
        mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
        mask_small = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_small is None:
            raise RuntimeError(f"could not read mask: {mask_path}")
        mask = cv2.resize(mask_small, (video_width, video_height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if int((mask > 0).sum()) < int(args.min_mask_pixels):
            raise RuntimeError(f"frame {frame_idx} mask underconstrained after resize")
        masked = np.full_like(raw, int(args.background_value), dtype=np.uint8)
        masked[mask > 0] = raw[mask > 0]

        image_path = image_dir / f"frame_{frame_idx:06d}.png"
        mask_out = mask_dir / f"frame_{frame_idx:06d}.png"
        if not cv2.imwrite(str(image_path), masked):
            raise RuntimeError(f"failed to write {image_path}")
        if not cv2.imwrite(str(mask_out), mask):
            raise RuntimeError(f"failed to write {mask_out}")
        if seq_i % int(args.review_stride) == 0:
            review = render_review(raw, mask, masked)
            review_path = review_dir / f"review_{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(review_path), review):
                raise RuntimeError(f"failed to write {review_path}")

        frame_reports.append(
            {
                "frame_idx": frame_idx,
                "image_path": str(image_path),
                "mask_path": str(mask_out),
                "source_mask_path": str(mask_path),
                "track_id": obj.get("track_id"),
                "label": obj.get("label"),
                "status": obj.get("status"),
                "source_image_size": [video_width, video_height],
                "mask_positive_pixels": int((mask > 0).sum()),
                "mask_bbox_xyxy_source": mask_bbox(mask),
            }
        )

    cap.release()
    report = {
        "status": "ok",
        "video": str(args.video),
        "annotations": str(args.annotations),
        "output_dir": str(args.output_dir),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "track_id": args.track_id,
        "background_value": int(args.background_value),
        "frames": len(frame_reports),
        "image_size": [video_width, video_height],
        "frame_reports": frame_reports,
    }
    (args.output_dir / "vggt_object_views_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frame_reports"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--background-value", type=int, default=255)
    parser.add_argument("--min-mask-pixels", type=int, default=1000)
    parser.add_argument("--review-stride", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
