#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def run(args: argparse.Namespace) -> dict:
    report = json.loads(args.contact_depth_report.read_text(encoding="utf-8"))
    rows = []
    for frame in report.get("frame_reports", []):
        if frame.get("status") != "ok":
            continue
        hand = frame.get("mask_near_hand_depth_z") or {}
        obj = frame.get("object_depth_z") or {}
        if hand.get("count", 0) <= 0 or obj.get("count", 0) <= 0:
            continue
        hand_z = float(hand["median"])
        obj_z = float(obj["median"])
        if hand_z <= 0 or obj_z <= 0:
            continue
        rows.append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "near_hand_depth_m": hand_z,
                "object_depth_m": obj_z,
                "hand_minus_object_depth_m": hand_z - obj_z,
                "object_over_hand_depth_ratio": obj_z / hand_z,
                "near_hand_vertex_count": int(hand["count"]),
            }
        )
    if not rows:
        raise RuntimeError(f"no near-mask hand/object depth pairs in {args.contact_depth_report}")
    gaps = np.asarray([row["hand_minus_object_depth_m"] for row in rows], dtype=float)
    ratios = np.asarray([row["object_over_hand_depth_ratio"] for row in rows], dtype=float)
    hand_depth = np.asarray([row["near_hand_depth_m"] for row in rows], dtype=float)
    object_depth = np.asarray([row["object_depth_m"] for row in rows], dtype=float)
    counts = np.asarray([row["near_hand_vertex_count"] for row in rows], dtype=float)
    worst = max(rows, key=lambda row: abs(row["hand_minus_object_depth_m"]))
    summary = {
        "status": "ok",
        "contact_depth_report": str(args.contact_depth_report),
        "rows": len(rows),
        "hand_minus_object_depth_m": summarize(gaps),
        "object_over_hand_depth_ratio": summarize(ratios),
        "near_hand_depth_m": summarize(hand_depth),
        "object_depth_m": summarize(object_depth),
        "near_hand_vertex_count": summarize(counts),
        "worst_abs_gap": worst,
        "interpretation": (
            "Depth conflict is large when 2D hand vertices project near the object mask. "
            "The object_over_hand_depth_ratio approximates the hand-depth scale needed "
            "to place the near-mask hand vertices at the object depth for each frame."
        ),
        "rows_preview": rows[:40],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows_preview"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-depth-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
