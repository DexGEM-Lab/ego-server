#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mask_box(mask_path: Path, args: argparse.Namespace) -> tuple[list[float], int, list[float]]:
    mask = resize_bool_mask(mask_path, (int(args.mask_width), int(args.mask_height)))
    if int(args.source_width) != int(args.mask_width) or int(args.source_height) != int(args.mask_height):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(args.source_width), int(args.source_height)),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    ys, xs = np.nonzero(mask)
    if len(xs) < int(args.min_mask_area_px):
        raise RuntimeError(f"mask too small: {len(xs)} px")
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    width = x1 - x0
    height = y1 - y0
    margin = float(args.box_margin_fraction) * max(width, height)
    x0 = max(0.0, x0 - margin)
    y0 = max(0.0, y0 - margin)
    x1 = min(float(args.source_width - 1), x1 + margin)
    y1 = min(float(args.source_height - 1), y1 + margin)
    return [x0, y0, x1, y1], int(len(xs)), [float(xs.mean()), float(ys.mean())]


def synthetic_keypoints(box: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = [float(v) for v in box]
    xs = np.linspace(x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0), 5)
    ys = np.linspace(y0 + 0.25 * (y1 - y0), y0 + 0.75 * (y1 - y0), 5)
    points = [[0.5 * (x0 + x1), 0.8 * y0 + 0.2 * y1]]
    for col in range(5):
        for row in range(4):
            points.append([float(xs[col]), float(ys[row])])
    return points[:21]


def frame_rows(track_path: Path, track_id: str, side: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    track = load_json(track_path)
    rows = []
    skipped = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        row = track.get(str(frame_idx))
        if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
            skipped.append({"frame_idx": frame_idx, "track_id": track_id, "reason": "missing_visible_mask"})
            continue
        try:
            path = localize_path(str(row["mask_path"]), args.remote_output_root, args.local_output_root)
            box, area, center = mask_box(path, args)
            keypoints = synthetic_keypoints(box)
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_idx": len(rows),
                    "side": side,
                    "track_id": track_id,
                    "bbox_xyxy": box,
                    "keypoints": keypoints,
                    "scores": [1.0] * 21,
                    "mean_score": 1.0,
                    "mask_path": str(path),
                    "mask_area_px": area,
                    "mask_center_xy": center,
                    "source": "vlm_sam_hand_mask_box",
                    "note": "Synthetic keypoints are a crop contract only; downstream measurement must use HaMeR projection, mask, and depth evidence.",
                }
            )
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "track_id": track_id, "reason": str(exc)})
    return rows, skipped


def run(args: argparse.Namespace) -> dict:
    all_rows: dict[int, list[dict]] = {}
    skipped: list[dict] = []
    if args.left_track is not None:
        rows, skip = frame_rows(args.left_track, args.left_track_id, "left", args)
        skipped.extend(skip)
        for row in rows:
            all_rows.setdefault(int(row["frame_idx"]), []).append(row)
    if args.right_track is not None:
        rows, skip = frame_rows(args.right_track, args.right_track_id, "right", args)
        skipped.extend(skip)
        for row in rows:
            all_rows.setdefault(int(row["frame_idx"]), []).append(row)
    frames = [{"frame_idx": frame_idx, "hands": all_rows.get(frame_idx, [])} for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1)]
    output = {
        "status": "ok",
        "backend": "VLM/SAM hand mask boxes",
        "frames": frames,
        "skipped": skipped,
        "contract": "Boxes are visual crop evidence. Synthetic keypoints are not anatomical measurements.",
    }
    save_json(args.output_json, output)
    print(json.dumps({"status": "ok", "frames": len(frames), "hand_rows": sum(len(f["hands"]) for f in frames), "skipped": len(skipped)}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-track", type=Path)
    parser.add_argument("--right-track", type=Path)
    parser.add_argument("--left-track-id", default="left_visible_gloved_hand")
    parser.add_argument("--right-track-id", default="right_visible_gloved_hand")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--mask-width", type=int, default=960)
    parser.add_argument("--mask-height", type=int, default=540)
    parser.add_argument("--box-margin-fraction", type=float, default=0.15)
    parser.add_argument("--min-mask-area-px", type=int, default=200)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/dev/shm/ego_annotation_keyboard_hand_masks/outputs"))
    parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/representative_keyboard/v3_keyboard_hand_sam2_visual_tracks_60_75"),
    )
    args = parser.parse_args()
    if args.left_track is None and args.right_track is None:
        raise RuntimeError("pass at least one hand mask track")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
