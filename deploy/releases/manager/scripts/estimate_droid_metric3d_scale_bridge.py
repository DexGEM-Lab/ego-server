#!/usr/bin/env python3
"""Estimate one or more DROID/Metric3D scales in a single Metric3D depth pass.

The bridge accepts either its historical single ``--geometry`` input or a
``--multi-geometry`` archive.  Multi-session mode loads Metric3D once and
materializes each source-frame metric depth once, then applies the unchanged
HaWoR ``est_scale_hybrid`` calculation to every session's exact keyframe
mapping.  It intentionally has no per-session subprocess fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

MANAGER_ROOT = Path(__file__).resolve().parents[1]
if str(MANAGER_ROOT) not in sys.path:
    sys.path.insert(0, str(MANAGER_ROOT))

from scripts.adapt_droid_to_hawor import import_scale_helpers, sha256_file  # noqa: E402

THRESHOLDS = ((0.4, 0.7), (0.3, 0.8), (0.2, 1.0), (0.0, 2.0))
est_scale_hybrid: object


def _load_geometries(*, geometry: Path | None, multi_geometry: Path | None) -> list[tuple[np.ndarray, np.ndarray]]:
    if (geometry is None) == (multi_geometry is None):
        raise RuntimeError("provide exactly one of --geometry or --multi-geometry")
    archive_path = geometry or multi_geometry
    assert archive_path is not None
    with np.load(archive_path, allow_pickle=False) as archive:
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int32)
        if multi_geometry is None:
            geometries = [(np.asarray(archive["tstamp"], dtype=np.int32), np.asarray(archive["disps"], dtype=np.float32))]
        else:
            count = int(np.asarray(archive["session_count"]).item())
            geometries = [
                (
                    np.asarray(archive[f"session_{index}_tstamp"], dtype=np.int32),
                    np.asarray(archive[f"session_{index}_disps"], dtype=np.float32),
                )
                for index in range(count)
            ]
    if not geometries:
        raise RuntimeError("Metric3D bridge multi-session geometry is empty")
    for timestamps, disparities in geometries:
        if disparities.ndim != 3 or disparities.shape[0] != len(timestamps):
            raise RuntimeError("Metric3D bridge disparities do not match exact keyframe mapping")
    return [(frame_idx, timestamps, disparities) for timestamps, disparities in geometries]


def _estimate_scale(
    *,
    timestamps: np.ndarray,
    disparities: np.ndarray,
    metric_depth_by_source: dict[int, np.ndarray],
    masks: np.ndarray,
) -> tuple[float, dict[str, object]]:
    scales: list[float] = []
    rows: list[dict[str, object]] = []
    disparity_h, disparity_w = disparities.shape[1:]
    for keyframe_position, source_index in enumerate(timestamps.tolist()):
        if source_index not in metric_depth_by_source or source_index < 0 or source_index >= len(masks):
            raise RuntimeError("Metric3D keyframe source index is outside the source timeline")
        metric_depth = cv2.resize(metric_depth_by_source[source_index], (disparity_w, disparity_h), interpolation=cv2.INTER_LINEAR)
        slam_depth = 1.0 / np.maximum(disparities[keyframe_position], 1.0e-8)
        dynamic_mask = np.asarray(masks[source_index], dtype=np.float32)
        scale = float("nan")
        used_threshold: tuple[float, float] | None = None
        error: str | None = None
        for near, far in THRESHOLDS:
            try:
                candidate = float(est_scale_hybrid(
                    slam_depth, metric_depth, sigma=0.5, msk=dynamic_mask,
                    near_thresh=near, far_thresh=far,
                ))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                scale = candidate
                used_threshold = (near, far)
                break
        rows.append({
            "keyframe_position": keyframe_position,
            "frame_idx": source_index,
            "scale": scale if math.isfinite(scale) else None,
            "threshold": list(used_threshold) if used_threshold else None,
            "error": error,
        })
        if math.isfinite(scale) and scale > 0.0:
            scales.append(scale)
    if not scales:
        raise RuntimeError("Metric3D/DROID disparity scale estimation produced no finite positive scales")
    values = np.asarray(scales, dtype=np.float64)
    scale = float(np.median(values))
    return scale, {
        "status": "ok",
        "depth_source": "Metric3D",
        "scale": scale,
        "keyframes": int(len(timestamps)),
        "finite_keyframe_scales": int(len(scales)),
        "disparity_grid_hw": [int(disparity_h), int(disparity_w)],
        "scale_statistics": {
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10)),
            "median": scale,
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        },
        "per_keyframe": rows,
        "per_keyframe_preview": rows[:16],
    }


def main() -> int:
    global est_scale_hybrid
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--geometry", type=Path)
    inputs.add_argument("--multi-geometry", type=Path)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--metric-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    geometries = _load_geometries(geometry=args.geometry, multi_geometry=args.multi_geometry)
    frame_idx = geometries[0][0]
    if any(not np.array_equal(row[0], frame_idx) for row in geometries[1:]):
        raise RuntimeError("Metric3D bridge session timelines disagree")
    image_paths = {int(path.stem): str(path) for path in sorted(args.image_dir.glob("*.jpg")) if path.stem.isdecimal()}
    masks = np.load(args.masks, allow_pickle=False)
    calib = np.load(args.calib, allow_pickle=False)
    if len(image_paths) != len(frame_idx) or set(image_paths) != set(frame_idx.tolist()) or masks.shape[0] != len(frame_idx):
        raise RuntimeError("Metric3D bridge timeline inputs disagree")

    Metric3D, est_scale_hybrid = import_scale_helpers(args.hawor_root)
    metric = Metric3D(str(args.metric_checkpoint))
    # Metric3D inference is deliberately one full source-timeline pass, shared
    # by all session estimates.  Repeating this in each session would make a
    # long multi-session video scale quadratically expensive.
    metric_depth_by_source = {
        source_index: np.asarray(metric(image_paths[source_index], calib), dtype=np.float32)
        for source_index in frame_idx.tolist()
    }
    session_rows: list[dict[str, object]] = []
    for session_index, (_frame_idx, timestamps, disparities) in enumerate(geometries):
        scale, report = _estimate_scale(
            timestamps=timestamps,
            disparities=disparities,
            metric_depth_by_source=metric_depth_by_source,
            masks=masks,
        )
        report.update({
            "session_index": session_index,
            "exact_keyframe_source_ids": [int(value) for value in timestamps.tolist()],
            "metric_model_load_count": 1,
            "metric_depth_pass_count": len(metric_depth_by_source),
            "metric_checkpoint": str(args.metric_checkpoint.resolve()),
            "metric_checkpoint_sha256": sha256_file(args.metric_checkpoint.resolve()),
        })
        session_rows.append({"scale": scale, "report": report})
    del metric
    payload: dict[str, object]
    if args.multi_geometry is None:
        payload = session_rows[0]
    else:
        payload = {
            "sessions": session_rows,
            "shared_metric3d": {
                "metric_model_load_count": 1,
                "metric_depth_pass_count": len(metric_depth_by_source),
                "source_frame_count": len(frame_idx),
            },
        }
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
