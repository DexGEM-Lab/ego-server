#!/usr/bin/env python3
"""Render HOT3D GT hand boxes against HaWoR predicted boxes for review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def gt_box(row: dict[str, Any], side: str, stream_id: str) -> np.ndarray | None:
    hands = row.get("json", {}).get("hands.json")
    if not isinstance(hands, dict) or not isinstance(hands.get(side), dict):
        return None
    boxes = hands[side].get("boxes_amodal")
    if not isinstance(boxes, dict) or stream_id not in boxes:
        return None
    arr = np.asarray(boxes[stream_id], dtype=float)
    return arr if arr.shape == (4,) and np.isfinite(arr).all() else None


def iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else None


def pred_box(npz: Any, side: str, idx: int) -> np.ndarray | None:
    key = f"{side}_det_box_xyxyscore"
    if key not in npz.files or idx >= len(npz[key]):
        return None
    arr = np.asarray(npz[key][idx], dtype=float)
    if arr.shape[0] < 4 or not np.isfinite(arr[:4]).all():
        return None
    return arr[:4]


def draw_box(image: np.ndarray, box: np.ndarray | None, color: tuple[int, int, int], label: str, thickness: int) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    tx = min(max(4, x1), max(4, image.shape[1] - text_w - 4))
    ty = min(max(text_h + 4, y1 - 6), image.shape[0] - 6)
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def tile_for_frame(frame: dict[str, Any], gt_row: dict[str, Any], npz: Any, stream_id: str, width: int) -> np.ndarray:
    image = cv2.imread(str(frame["raw_frame_path"]))
    if image is None:
        raise FileNotFoundError(frame["raw_frame_path"])
    idx = int(frame["frame_idx"])
    canvas = image.copy()
    for side, gt_color, pred_color in (("left", (60, 255, 80), (0, 255, 255)), ("right", (80, 160, 255), (255, 0, 255))):
        gb = gt_box(gt_row, side, stream_id)
        pb = pred_box(npz, side, idx)
        val = iou(gb, pb)
        draw_box(canvas, gb, gt_color, f"GT {side}", 3)
        draw_box(canvas, pb, pred_color, f"HaWoR {side} iou={val:.2f}" if val is not None else f"HaWoR {side}", 2)
    banner_h = 56
    banner = np.zeros((banner_h, canvas.shape[1], 3), dtype=np.uint8)
    banner[:] = (8, 8, 8)
    cv2.putText(banner, f"HOT3D clip box comparison frame {idx} | GT left/right=green/orange, HaWoR left/right=yellow/magenta", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([banner, canvas])
    if out.shape[1] != width:
        scale = width / out.shape[1]
        out = cv2.resize(out, (width, int(round(out.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return out


def render(args: argparse.Namespace) -> None:
    manifest = load_json(args.manifest)
    gt = load_json(args.hot3d_gt)
    frames = {int(row["frame_idx"]): row for row in manifest.get("frames", [])}
    gt_rows = {int(row["frame_idx"]): row for row in gt.get("frames", [])}
    npz = np.load(args.hawor_npz, allow_pickle=True)
    tiles = []
    for idx in args.frames:
        if idx not in frames or idx not in gt_rows:
            raise RuntimeError(f"frame {idx} missing from manifest or GT")
        tiles.append(tile_for_frame(frames[idx], gt_rows[idx], npz, args.stream_id, args.tile_width))
    max_h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)
    rows = []
    for i in range(0, len(padded), args.cols):
        row = padded[i : i + args.cols]
        while len(row) < args.cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write {args.output}")
    print(json.dumps({"status": "ok", "output": str(args.output), "frames": args.frames}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hot3d-gt", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 50, 100, 149])
    parser.add_argument("--tile-width", type=int, default=704)
    parser.add_argument("--cols", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
