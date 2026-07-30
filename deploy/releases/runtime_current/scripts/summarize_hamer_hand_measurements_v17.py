#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compact_bbox(raw: object) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    return [float(x) for x in raw]


def as_float(raw: object) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    return value if value == value else None


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.annotations)
    measurements: list[dict[str, Any]] = []
    frame_count = 0
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_count += 1
        idx = int(frame["frame_idx"])
        for hand_i, hand in enumerate(frame.get("hands") or []):
            if not isinstance(hand, dict) or hand.get("backend") != "HaMeR":
                continue
            solve = hand.get("source_camera_solve") or {}
            rtmlib = hand.get("rtmlib_measurement") or {}
            measurement_available = bool(hand.get("measurement_available", False))
            detector_hand_idx = rtmlib.get("hand_idx")
            row = {
                "measurement_id": f"hamer:{idx}:{hand_i}",
                "frame_idx": idx,
                "entity_type": "hand",
                "entity_id": f"hand:{hand.get('side', 'unknown')}",
                "measurement_type": "mano_per_frame_metric_translation",
                "source_model": "HaMeR",
                "coordinate_frame": "source_camera_metric_translation",
                "confidence": as_float(hand.get("detector_score")) if measurement_available else None,
                "bbox_xyxy": compact_bbox(hand.get("bbox_xyxy")),
                "detector_hand_idx": int(detector_hand_idx) if detector_hand_idx is not None else None,
                "side": hand.get("side"),
                "has_joints2d": hand.get("joints2d") is not None,
                "has_joints3d_camera": hand.get("joints3d_source_camera_m") is not None
                or hand.get("joints3d_camera") is not None,
                "has_vertices_camera": hand.get("vertices_source_camera_m") is not None
                or hand.get("vertices_camera") is not None,
                "has_mano_params": hand.get("mano_params") is not None,
                "measurement_available": measurement_available,
                "filter_status": hand.get("filter_status"),
                "projection_residual_px_median": as_float(solve.get("median_reprojection_error_px")),
                "projection_residual_px_p95": as_float(solve.get("p95_reprojection_error_px")),
                "median_depth_m": as_float(solve.get("median_depth_m")),
                "hand_bone_scale_m": as_float(solve.get("hand_bone_scale_m")),
                "source_intrinsics_field": solve.get("source_intrinsics_field"),
                "mano_vertex_count": hand.get("mano_vertex_count"),
                "source_annotation": str(args.annotations),
                "failure_reason": None if measurement_available else "hamer_initial_metric_qc_rejected",
            }
            measurements.append(row)
    measured = [row for row in measurements if row["measurement_available"]]
    return {
        "status": "ok",
        "method": "summarize_hamer_hand_measurements_v17",
        "source_annotations": str(args.annotations),
        "source_qc": str(args.qc) if args.qc is not None else None,
        "frame_count": frame_count,
        "measurement_count": len(measurements),
        "measured_hand_rows": len(measured),
        "measurements": measurements,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--qc", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.output_json, summarize(args))


if __name__ == "__main__":
    main()
