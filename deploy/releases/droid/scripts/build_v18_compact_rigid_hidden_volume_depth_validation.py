#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Validate/quarantine compact-rigid hidden volume against metric depth.

For each fitted object pose, this projects sampled vertices of the completed
compact-rigid mesh into available metric depth frames. A mesh sample is:

- observed_depth_supported: projected depth agrees with observed depth;
- free_space_conflict: mesh lies substantially in front of observed depth, so it
  would have been visible/occluding under the depth image;
- behind_observed_surface: mesh lies behind the observed nearest surface and is
  therefore hidden/occluded rather than directly validated;
- missing_depth/out_of_frame.

This is not a visual annotation deliverable. It is a physical measurement of
which hidden-volume constraints can legitimately support or falsify MANO hand
state. The output remains conservative: free-space conflict or lack of observed
support quarantines hidden-volume nonpenetration acceptance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    load_mesh,
    numeric_summary,
    pose_map,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True, help="Metric depth NPZ. Earlier entries take precedence for duplicate frames.")
    parser.add_argument("--temporal-mano-state", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--support-margin-m", type=float, default=0.015)
    parser.add_argument("--free-space-margin-m", type=float, default=0.025)
    parser.add_argument("--free-space-conflict-fraction", type=float, default=0.02)
    parser.add_argument("--observed-support-fraction", type=float, default=0.01)
    return parser.parse_args()


def load_depth_sources(paths: list[Path]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        frame_idx = np.asarray(data["frame_idx"], dtype=int)
        depth = np.asarray(data["depth"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics_fx_fy_cx_cy"], dtype=np.float32)
        for i, frame in enumerate(frame_idx):
            idx = int(frame)
            if idx in frames:
                continue
            frames[idx] = {
                "source": str(path),
                "depth": depth[i],
                "intrinsics": intrinsics[i],
            }
    return frames


def sample_vertices(vertices: np.ndarray, max_samples: int) -> np.ndarray:
    if len(vertices) <= max_samples:
        return vertices.astype(float)
    idx = np.linspace(0, len(vertices) - 1, int(max_samples), dtype=np.int64)
    return vertices[idx].astype(float)


def project_points(points_camera: np.ndarray, intr: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = [float(x) for x in intr]
    z = points_camera[:, 2]
    valid = z > 1.0e-4
    u_float = fx * points_camera[:, 0] / np.maximum(z, 1.0e-6) + cx
    v_float = fy * points_camera[:, 1] / np.maximum(z, 1.0e-6) + cy
    u = np.rint(u_float).astype(np.int32)
    v = np.rint(v_float).astype(np.int32)
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def classify_frame(
    frame: dict[str, Any],
    object_vertices: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = int(frame["frame_idx"])
    if depth_row is None:
        return {
            "frame_idx": frame_idx,
            "state": "missing_depth",
            "depth_source": None,
            "sample_count": int(len(object_vertices)),
            "projected_count": 0,
            "observed_depth_supported_count": 0,
            "free_space_conflict_count": 0,
            "behind_observed_surface_count": 0,
            "out_of_frame_or_invalid_count": int(len(object_vertices)),
            "free_space_conflict_fraction_projected": None,
            "observed_support_fraction_projected": None,
            "depth_residual_m": {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None},
        }
    r_obj, t_obj = pose
    vertices_world = object_vertices @ r_obj.T + t_obj[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    vertices_camera = (vertices_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    height, width = depth.shape
    u, v, valid = project_points(vertices_camera, np.asarray(depth_row["intrinsics"], dtype=float), width, height)
    projected = int(np.count_nonzero(valid))
    out_count = int(len(object_vertices) - projected)
    if projected == 0:
        return {
            "frame_idx": frame_idx,
            "state": "no_projected_mesh_samples",
            "depth_source": depth_row["source"],
            "sample_count": int(len(object_vertices)),
            "projected_count": 0,
            "observed_depth_supported_count": 0,
            "free_space_conflict_count": 0,
            "behind_observed_surface_count": 0,
            "out_of_frame_or_invalid_count": out_count,
            "free_space_conflict_fraction_projected": None,
            "observed_support_fraction_projected": None,
            "depth_residual_m": {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None},
        }
    z_mesh = vertices_camera[valid, 2]
    z_obs = depth[v[valid], u[valid]].astype(float)
    finite = np.isfinite(z_obs) & (z_obs > 0.0)
    residual = z_mesh[finite] - z_obs[finite]
    supported = np.abs(residual) <= float(args.support_margin_m)
    free_space = residual < -float(args.free_space_margin_m)
    behind = residual > float(args.support_margin_m)
    supported_count = int(np.count_nonzero(supported))
    free_count = int(np.count_nonzero(free_space))
    behind_count = int(np.count_nonzero(behind))
    finite_count = int(np.count_nonzero(finite))
    free_fraction = free_count / finite_count if finite_count else None
    support_fraction = supported_count / finite_count if finite_count else None
    if finite_count == 0:
        state = "projected_but_depth_invalid"
    elif free_fraction is not None and free_fraction >= float(args.free_space_conflict_fraction):
        state = "hidden_volume_free_space_conflict"
    elif support_fraction is not None and support_fraction >= float(args.observed_support_fraction):
        state = "observed_depth_support_with_hidden_uncertainty"
    else:
        state = "hidden_volume_unvalidated_mostly_occluded_or_behind_observed_depth"
    return {
        "frame_idx": frame_idx,
        "state": state,
        "depth_source": depth_row["source"],
        "sample_count": int(len(object_vertices)),
        "projected_count": projected,
        "finite_depth_count": finite_count,
        "observed_depth_supported_count": supported_count,
        "free_space_conflict_count": free_count,
        "behind_observed_surface_count": behind_count,
        "out_of_frame_or_invalid_count": out_count + int(projected - finite_count),
        "free_space_conflict_fraction_projected": free_fraction,
        "observed_support_fraction_projected": support_fraction,
        "depth_residual_m": numeric_summary(residual),
    }


def load_temporal_intervals(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = load_json(path)
    intervals = data.get("intervals") if isinstance(data, dict) else None
    return [row for row in intervals if isinstance(row, dict)] if isinstance(intervals, list) else []


def summarize_interval_validation(intervals: list[dict[str, Any]], rows_by_frame: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for interval in intervals:
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        rows = [rows_by_frame[idx] for idx in range(start, end + 1) if idx in rows_by_frame]
        states = Counter(str(row.get("state")) for row in rows)
        free_values = [float(row["free_space_conflict_fraction_projected"]) for row in rows if isinstance(row.get("free_space_conflict_fraction_projected"), (float, int))]
        support_values = [float(row["observed_support_fraction_projected"]) for row in rows if isinstance(row.get("observed_support_fraction_projected"), (float, int))]
        out.append(
            {
                "interval_id": interval.get("interval_id"),
                "hand_side": interval.get("hand_side"),
                "start_frame": start,
                "end_frame": end,
                "frame_count": int(interval.get("frame_count") or (end - start + 1)),
                "validated_frame_count": len(rows),
                "validation_state_counts": dict(states),
                "free_space_conflict_fraction_projected": numeric_summary(np.asarray(free_values, dtype=float)),
                "observed_support_fraction_projected": numeric_summary(np.asarray(support_values, dtype=float)),
                "mano_constraint_implication": (
                    "nonpenetration_acceptance_quarantined_by_free_space_conflict_or_missing_depth"
                    if states.get("hidden_volume_free_space_conflict", 0) or states.get("missing_depth", 0)
                    else "hidden_volume_still_uncertain_not_sufficient_for_coordinate_acceptance"
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    depth_by_frame = load_depth_sources(args.depth_npz)
    mesh = load_mesh(args.completed_mesh)
    object_vertices = sample_vertices(np.asarray(mesh.vertices, dtype=float), int(args.max_samples))
    frames = [frame for frame in as_list(annotations.get("frames")) if isinstance(frame, dict)]

    rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        pose = poses.get(frame_idx)
        if pose is None:
            continue
        rows.append(classify_frame(frame, object_vertices, pose, depth_by_frame.get(frame_idx), args))

    rows_by_frame = {int(row["frame_idx"]): row for row in rows}
    intervals = load_temporal_intervals(args.temporal_mano_state)
    interval_validation = summarize_interval_validation(intervals, rows_by_frame)
    state_counts = Counter(str(row.get("state")) for row in rows)
    free_values = [float(row["free_space_conflict_fraction_projected"]) for row in rows if isinstance(row.get("free_space_conflict_fraction_projected"), (float, int))]
    support_values = [float(row["observed_support_fraction_projected"]) for row in rows if isinstance(row.get("observed_support_fraction_projected"), (float, int))]
    report = {
        "method": "build_v18_compact_rigid_hidden_volume_depth_validation",
        "status": "ok",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Depth/free-space validation of the posed compact-rigid object mesh. This can quarantine hidden-volume "
            "nonpenetration constraints for MANO but does not by itself solve hand pose."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(path) for path in args.depth_npz],
            "temporal_mano_state": str(args.temporal_mano_state) if args.temporal_mano_state else None,
        },
        "parameters": {
            "max_samples": int(args.max_samples),
            "support_margin_m": float(args.support_margin_m),
            "free_space_margin_m": float(args.free_space_margin_m),
            "free_space_conflict_fraction": float(args.free_space_conflict_fraction),
            "observed_support_fraction": float(args.observed_support_fraction),
        },
        "summary": {
            "pose_frame_count": int(len(poses)),
            "evaluated_frame_count": int(len(rows)),
            "depth_available_frame_count": int(sum(1 for row in rows if row.get("state") != "missing_depth")),
            "state_counts": dict(state_counts),
            "free_space_conflict_fraction_projected": numeric_summary(np.asarray(free_values, dtype=float)),
            "observed_support_fraction_projected": numeric_summary(np.asarray(support_values, dtype=float)),
            "coordinate_correction_acceptance_implication": "hidden_volume_not_accepted_for_mano_coordinate_correction",
        },
        "frame_rows": rows,
        "interval_validation": interval_validation,
        "physical_conclusion": (
            "Observed-depth support and free-space conflict are measured for the posed completed mesh. Frames or "
            "intervals with free-space conflict, missing depth, or mostly behind-observed samples cannot be used as "
            "accepted hidden-volume nonpenetration proof for MANO correction; they remain uncertainty or quarantine."
        ),
    }
    out_path = args.output_dir / "v18_compact_rigid_hidden_volume_depth_validation.json"
    write_json(out_path, report)
    print(json.dumps({"output": str(out_path), "summary": report["summary"], "first_intervals": interval_validation[:3]}, indent=2))


if __name__ == "__main__":
    main()
