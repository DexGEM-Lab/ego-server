#!/usr/bin/env python3
"""Evaluate HaWoR detector boxes against HOT3D-Clips amodal hand boxes.

This is an initial bounded HOT3D comparison for a claim family that does not
require metric fisheye-to-pinhole conversion: 2D hand localization/visibility in
the selected HOT3D camera stream.  It is not a MANO 3D or contact metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else float("nan")


def center_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=float)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=float)
    return float(np.linalg.norm(ac - bc))


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def gt_box(row: dict[str, Any], side: str, stream_id: str, visibility_threshold: float) -> tuple[np.ndarray | None, float | None]:
    hands = row.get("json", {}).get("hands.json")
    if not isinstance(hands, dict):
        return None, None
    hand = hands.get(side)
    if not isinstance(hand, dict):
        return None, None
    visibility = None
    vis = hand.get("visibilities_modeled")
    if isinstance(vis, dict) and stream_id in vis:
        visibility = float(vis[stream_id])
        if visibility < visibility_threshold:
            return None, visibility
    boxes = hand.get("boxes_amodal")
    if not isinstance(boxes, dict) or stream_id not in boxes:
        return None, visibility
    arr = np.asarray(boxes[stream_id], dtype=float)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        return None, visibility
    return arr, visibility


def pred_box(npz: Any, side: str, idx: int, score_threshold: float) -> tuple[np.ndarray | None, float | None, bool]:
    box_key = f"{side}_det_box_xyxyscore"
    valid_key = f"{side}_valid"
    detected_key = f"{side}_detected_same_frame"
    if box_key not in npz.files:
        return None, None, False
    valid = bool(npz[valid_key][idx]) if valid_key in npz.files and idx < len(npz[valid_key]) else True
    detected = bool(npz[detected_key][idx]) if detected_key in npz.files and idx < len(npz[detected_key]) else valid
    raw = np.asarray(npz[box_key][idx], dtype=float)
    if raw.shape[0] < 4 or not np.isfinite(raw[:4]).all():
        return None, None, detected
    score = float(raw[4]) if raw.shape[0] >= 5 and np.isfinite(raw[4]) else None
    if score is not None and score < score_threshold:
        return None, score, detected
    return raw[:4], score, detected


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gt = load_json(args.hot3d_gt)
    frames = gt.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("HOT3D GT sidecar has no frames")
    npz = np.load(args.hawor_npz, allow_pickle=True)
    frame_idx = np.asarray(npz["frame_idx"], dtype=int) if "frame_idx" in npz.files else np.arange(len(frames), dtype=int)
    row_by_frame = {int(row["frame_idx"]): row for row in frames if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for local_i, frame in enumerate(frame_idx.tolist()):
        if local_i >= len(frame_idx):
            continue
        gt_row = row_by_frame.get(int(frame))
        if gt_row is None:
            continue
        for side in ("left", "right"):
            gbox, visibility = gt_box(gt_row, side, args.stream_id, args.visibility_threshold)
            pbox, score, detected = pred_box(npz, side, local_i, args.score_threshold)
            measurable = gbox is not None
            matched = measurable and pbox is not None
            rows.append(
                {
                    "frame_idx": int(frame),
                    "side": side,
                    "gt_visibility": visibility,
                    "gt_box_xyxy": gbox.tolist() if gbox is not None else None,
                    "pred_box_xyxy": pbox.tolist() if pbox is not None else None,
                    "pred_score": score,
                    "pred_detected_same_frame": bool(detected),
                    "measurable": bool(measurable),
                    "matched": bool(matched),
                    "iou": iou_xyxy(gbox, pbox) if matched else None,
                    "center_distance_px": center_distance_px(gbox, pbox) if matched else None,
                }
            )
    measurable_rows = [r for r in rows if r["measurable"]]
    matched_rows = [r for r in measurable_rows if r["matched"]]
    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sr = [r for r in measurable_rows if r["side"] == side]
        sm = [r for r in matched_rows if r["side"] == side]
        by_side[side] = {
            "measurable_frames": len(sr),
            "matched_frames": len(sm),
            "match_rate": float(len(sm) / max(1, len(sr))),
            "iou": summarize([float(r["iou"]) for r in sm if r["iou"] is not None]),
            "center_distance_px": summarize([float(r["center_distance_px"]) for r in sm if r["center_distance_px"] is not None]),
            "pred_score": summarize([float(r["pred_score"]) for r in sm if r["pred_score"] is not None]),
        }
    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_hawor_boxes",
        "claim_scope": "2D hand-box localization against HOT3D amodal boxes; not 3D MANO/contact/object-pose scoring",
        "hot3d_gt": str(args.hot3d_gt),
        "hawor_npz": str(args.hawor_npz),
        "stream_id": args.stream_id,
        "visibility_threshold": float(args.visibility_threshold),
        "score_threshold": float(args.score_threshold),
        "frame_count_gt": len(frames),
        "rows": rows,
        "summary": {
            "measurable_rows": len(measurable_rows),
            "matched_rows": len(matched_rows),
            "match_rate": float(len(matched_rows) / max(1, len(measurable_rows))),
            "iou": summarize([float(r["iou"]) for r in matched_rows if r["iou"] is not None]),
            "center_distance_px": summarize([float(r["center_distance_px"]) for r in matched_rows if r["center_distance_px"] is not None]),
            "by_side": by_side,
        },
    }
    write_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot3d-gt", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--visibility-threshold", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
