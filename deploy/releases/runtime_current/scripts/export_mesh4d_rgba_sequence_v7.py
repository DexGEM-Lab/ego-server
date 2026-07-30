#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise RuntimeError(f"{path} must contain a JSON object with a frames list")
    return payload


def select_frames(frames: list[dict], start: int, end: int, max_frames: int) -> list[dict]:
    selected = [row for row in frames if start <= int(row["frame_idx"]) <= end]
    selected.sort(key=lambda row: int(row["frame_idx"]))
    if len(selected) < 6:
        raise RuntimeError(f"Mesh4D requires at least 6 frames; selected {len(selected)}")
    if max_frames <= 0 or len(selected) <= max_frames:
        return selected
    indices = np.linspace(0, len(selected) - 1, max_frames)
    chosen = []
    used = set()
    for value in indices:
        idx = int(round(float(value)))
        while idx in used and idx + 1 < len(selected):
            idx += 1
        while idx in used and idx > 0:
            idx -= 1
        used.add(idx)
        chosen.append(selected[idx])
    chosen.sort(key=lambda row: int(row["frame_idx"]))
    if len(chosen) < 6:
        raise RuntimeError(f"downsampled sequence has {len(chosen)} unique frames")
    return chosen


def read_rgb_mask(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise RuntimeError(f"failed to read RGB image: {entry['rgb']}")
    if mask is None:
        raise RuntimeError(f"failed to read mask image: {entry['mask']}")
    if rgb.shape[:2] != mask.shape:
        raise RuntimeError(f"RGB shape {rgb.shape[:2]} differs from mask shape {mask.shape}: {entry['rgb']}")
    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise RuntimeError(f"empty mask for frame {entry['frame_idx']}: {entry['mask']}")
    return rgb, mask_bool


def write_rgba(path: Path, rgb_bgr: np.ndarray, mask: np.ndarray) -> None:
    white = np.full_like(rgb_bgr, 255)
    isolated = np.where(mask[:, :, None], rgb_bgr, white)
    rgba = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(path), rgba):
        raise RuntimeError(f"failed to write {path}")


def render_review(path: Path, rgbs: list[np.ndarray], masks: list[np.ndarray], frame_ids: list[int]) -> None:
    thumbs = []
    for rgb, mask, frame_idx in zip(rgbs, masks, frame_ids):
        review = rgb.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(review, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        review = cv2.resize(review, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(review, str(frame_idx), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(review, str(frame_idx), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(review)
    cols = min(6, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = np.full((rows * 135, cols * 240, 3), 32, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * 135
        x = (idx % cols) * 240
        canvas[y : y + 135, x : x + 240] = thumb
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")


def run(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.manifest)
    selected = select_frames(manifest["frames"], int(args.frame_start), int(args.frame_end), int(args.max_frames))
    dataset_root = args.output_dir / "DATA"
    sequence_dir = dataset_root / str(args.group_name) / str(args.sequence_name)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frames: list[dict] = []
    for out_idx, entry in enumerate(selected):
        rgb, mask = read_rgb_mask(entry)
        output_path = sequence_dir / f"{out_idx}.png"
        write_rgba(output_path, rgb, mask)
        rgbs.append(rgb)
        masks.append(mask)
        frames.append(
            {
                "sequence_index": int(out_idx),
                "frame_idx": int(entry["frame_idx"]),
                "source_index": int(entry.get("source_index", entry.get("index", out_idx))),
                "rgb": str(entry["rgb"]),
                "mask": str(entry["mask"]),
                "mesh4d_rgba": str(output_path),
                "mask_area_px": int(np.count_nonzero(mask)),
            }
        )

    review_path = args.output_dir / "mesh4d_rgba_sequence_review.jpg"
    render_review(review_path, rgbs, masks, [row["frame_idx"] for row in frames])

    report = {
        "status": "ok",
        "method": "export_mesh4d_rgba_sequence_v7",
        "claim_tested": "existing model-produced masks can be packaged as Mesh4D RGBA video input without category-specific visual logic",
        "manifest": str(args.manifest),
        "dataset_root": str(dataset_root),
        "sequence_dir": str(sequence_dir),
        "group_name": str(args.group_name),
        "sequence_name": str(args.sequence_name),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_count": int(len(frames)),
        "review": str(review_path),
        "frames": frames,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
