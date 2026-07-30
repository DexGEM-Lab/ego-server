#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import diags  # type: ignore[reportMissingTypeStubs]
from scipy.sparse.linalg import spsolve  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_camera_depth_correction"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: Any, fallback: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def percentile(values: list[float], p: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    xs = [v for v in values if math.isfinite(v)]
    return {"count": len(xs), "median": percentile(xs, 50), "p05": percentile(xs, 5), "p95": percentile(xs, 95), "min": min(xs) if xs else None, "max": max(xs) if xs else None}


def solve_temporal_log_scale(obs_frames: list[int], log_obs: np.ndarray, weights: np.ndarray, temporal_weight: float) -> np.ndarray:
    n = len(obs_frames)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    diag = np.maximum(weights.astype(np.float64), 1e-6)
    rhs = diag * log_obs.astype(np.float64)
    lower = np.zeros(max(0, n - 1), dtype=np.float64)
    upper = np.zeros(max(0, n - 1), dtype=np.float64)
    for i in range(1, n):
        dt = max(1, int(obs_frames[i]) - int(obs_frames[i - 1]))
        w = temporal_weight / float(dt * dt)
        diag[i - 1] += w
        diag[i] += w
        upper[i - 1] -= w
        lower[i - 1] -= w
    diag += 1e-9
    if n == 1:
        return rhs / diag
    matrix = diags([lower, diag, upper], offsets=[-1, 0, 1], shape=(n, n), format="csc")  # type: ignore[reportArgumentType]
    return np.asarray(spsolve(matrix, rhs), dtype=np.float64)


def sample_depth_patch(depth_frame: np.ndarray, x: float, y: float, radius: int) -> tuple[float | None, int, dict[str, int]]:
    h, w = depth_frame.shape
    xi = max(0, min(w - 1, int(round(x))))
    yi = max(0, min(h - 1, int(round(y))))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = depth_frame[y0:y1, x0:x1].astype(np.float64)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return None, 0, {"x": xi, "y": yi}
    return float(np.median(vals)), int(vals.size), {"x": xi, "y": yi}


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    ann_path = args.v16_root / case / "annotations_v16_full.json"
    depth_path = args.v16_root / case / "unidepth_metric" / "unidepth_metric_depth_v3.npz"
    ann = load_json(ann_path)
    depth_blob = np.load(depth_path)
    frame_to_i = {int(f): int(i) for i, f in enumerate(depth_blob["frame_idx"])}
    depth = depth_blob["depth"]
    depth_h, depth_w = int(depth.shape[1]), int(depth.shape[2])
    source_w, source_h = int(args.source_width), int(args.source_height)
    frames = [row for row in ann.get("frames", []) if isinstance(row, dict) and isinstance(row.get("frame_idx"), int)]
    observation_rows: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        depth_i = frame_to_i.get(frame_idx)
        obj = frame.get("object") if isinstance(frame.get("object"), dict) else {}
        target_depth = finite_float(obj.get("depth_m")) if isinstance(obj, dict) else float("nan")
        center = obj.get("center_xy") if isinstance(obj, dict) else None
        if depth_i is None:
            reject_counts["missing_depth_backend_frame"] = reject_counts.get("missing_depth_backend_frame", 0) + 1
            continue
        if not math.isfinite(target_depth) or target_depth <= 0:
            reject_counts["missing_v16_object_depth_target"] = reject_counts.get("missing_v16_object_depth_target", 0) + 1
            continue
        if not (isinstance(center, list) and len(center) == 2):
            reject_counts["missing_v16_object_center"] = reject_counts.get("missing_v16_object_center", 0) + 1
            continue
        x = finite_float(center[0])
        y = finite_float(center[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            reject_counts["invalid_v16_object_center"] = reject_counts.get("invalid_v16_object_center", 0) + 1
            continue
        if source_w > 0 and source_h > 0:
            x *= depth_w / float(source_w)
            y *= depth_h / float(source_h)
        raw_depth, valid_count, sample_xy = sample_depth_patch(depth[int(depth_i)], x, y, args.patch_radius)
        if raw_depth is None or raw_depth <= 0:
            reject_counts["invalid_sampled_depth"] = reject_counts.get("invalid_sampled_depth", 0) + 1
            continue
        scale = target_depth / raw_depth
        if not math.isfinite(scale) or scale <= 0:
            reject_counts["invalid_scale_observation"] = reject_counts.get("invalid_scale_observation", 0) + 1
            continue
        observation_rows.append(
            {
                "frame_idx": frame_idx,
                "target_v16_object_depth_m": target_depth,
                "sampled_backend_depth_m": raw_depth,
                "depth_scale_observation": scale,
                "log_depth_scale_observation": math.log(scale),
                "sample_xy_depth_grid": sample_xy,
                "valid_patch_depth_samples": valid_count,
                "object_label": obj.get("label") if isinstance(obj, dict) else None,
                "observation_source": "v16_object_depth_m_over_backend_depth_patch_median",
            }
        )
    obs_frames = [int(r["frame_idx"]) for r in observation_rows]
    log_obs = np.asarray([float(r["log_depth_scale_observation"]) for r in observation_rows], dtype=np.float64)
    weights = np.asarray([max(1.0, min(25.0, float(r.get("valid_patch_depth_samples", 1)))) for r in observation_rows], dtype=np.float64)
    log_est = solve_temporal_log_scale(obs_frames, log_obs, weights, args.temporal_weight)
    for i, row in enumerate(observation_rows):
        row["log_depth_scale_estimate"] = float(log_est[i])
        row["depth_scale_estimate"] = float(math.exp(float(log_est[i])))
        row["observation_residual_log_scale"] = float(log_est[i] - log_obs[i])
        row["identity_prior_residual_log_scale"] = float(0.0 - log_obs[i])
    if obs_frames:
        full_frame_indices = [int(frame["frame_idx"]) for frame in frames]
        interp = np.interp(np.asarray(full_frame_indices, dtype=np.float64), np.asarray(obs_frames, dtype=np.float64), log_est)
    else:
        full_frame_indices = [int(frame["frame_idx"]) for frame in frames]
        interp = np.zeros((len(full_frame_indices),), dtype=np.float64)
    observed_set = set(obs_frames)
    full_rows = []
    for frame_idx, log_value in zip(full_frame_indices, interp):
        full_rows.append(
            {
                "frame_idx": frame_idx,
                "depth_scale_estimate": float(math.exp(float(log_value))),
                "log_depth_scale_estimate": float(log_value),
                "state": "observed_depth_scale_correction" if frame_idx in observed_set else "interpolated_or_nearest_depth_scale_no_direct_observation",
                "has_direct_observation": frame_idx in observed_set,
                "observation": next((row for row in observation_rows if row["frame_idx"] == frame_idx), None),
            }
        )
    initial_energy = float(np.sum(weights * (0.0 - log_obs) ** 2)) if len(log_obs) else 0.0
    after_energy = float(np.sum(weights * (log_est - log_obs) ** 2)) if len(log_obs) else 0.0
    out = {
        "method": "build_v18_camera_depth_correction",
        "status": STATUS,
        "claim": "Estimates a depth-scale correction variable from V16 object depth targets divided by sampled backend depth at object centers, then temporally smooths log-scale observations. This calibrates the reused depth backend; it is not a new SLAM solve or geometric proof.",
        "case": case,
        "sources": {"v16_annotations": str(ann_path), "depth_backend_npz": str(depth_path)},
        "parameters": {"patch_radius": args.patch_radius, "temporal_weight": args.temporal_weight, "source_width": source_w, "source_height": source_h, "depth_grid_width": depth_w, "depth_grid_height": depth_h},
        "frame_count": len(frames),
        "observation_rows": len(observation_rows),
        "full_timeline_rows": len(full_rows),
        "rejection_counts": dict(sorted(reject_counts.items())),
        "depth_scale_observation_stats": stats([float(r["depth_scale_observation"]) for r in observation_rows]),
        "depth_scale_estimate_stats": stats([float(r["depth_scale_estimate"]) for r in full_rows]),
        "objective": {"identity_prior_energy": initial_energy, "temporal_smoothed_observation_energy": after_energy, "energy_delta": initial_energy - after_energy, "energy_units": "weighted_squared_log_depth_scale_residual"},
        "rows": full_rows,
        "direct_observation_rows": observation_rows,
        "camera_depth_correction_variable_ready": len(observation_rows) > 0,
        "camera_depth_correction_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_camera_depth_correction_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_camera_depth_correction",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "frame_count": report["frame_count"],
                "observation_rows": report["observation_rows"],
                "full_timeline_rows": report["full_timeline_rows"],
                "depth_scale_estimate_stats": report["depth_scale_estimate_stats"],
                "camera_depth_correction_variable_ready": report["camera_depth_correction_variable_ready"],
            }
            for report in reports
        ],
        "claim_scope": "depth_scale_correction_observation_for_reused_backend_not_new_camera_slam",
    }
    write_json(args.output_root / "v18_camera_depth_correction_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_camera_depth_correction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
