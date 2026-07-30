#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_contact_depth_scale_v3 import summarize


@dataclass(frozen=True)
class HandRow:
    frame_idx: int
    side: str
    detector_score: float
    local_joints_m: np.ndarray
    local_vertices_m: np.ndarray
    cam_t_m: np.ndarray
    joints2d_px: np.ndarray
    intrinsics: np.ndarray
    metric_joint_depth_m: np.ndarray


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hand_vertex_key(hand: dict) -> str:
    if "vertices_camera" in hand:
        return "vertices_camera"
    if "vertices_camera_sample" in hand:
        return "vertices_camera_sample"
    raise RuntimeError("hand has no local MANO vertices")


def row_from_hand(frame: dict, hand: dict, depth: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> HandRow | None:
    if not hand.get("measurement_available", False):
        return None
    score = float(hand.get("detector_score", np.nan))
    if not np.isfinite(score) or score < args.min_detector_score:
        return None
    local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
    local_vertices = np.asarray(hand[hand_vertex_key(hand)], dtype=float)
    cam_t = np.asarray(hand["cam_t"], dtype=float)
    joints2d = np.asarray(hand["joints2d_raw"], dtype=float)
    intr = np.asarray(hand["source_intrinsics"], dtype=float)
    if local_joints.shape != (21, 3) or local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
        raise RuntimeError("invalid MANO local geometry")
    if cam_t.shape != (3,) or joints2d.shape != (21, 2) or intr.shape != (4,):
        raise RuntimeError("invalid MANO projection fields")
    depth_samples = sample_depth(depth, joints2d, source_size)
    valid = np.isfinite(depth_samples) & (depth_samples > 0.0)
    if np.count_nonzero(valid) < args.min_depth_joints:
        return None
    metric = np.full(21, np.nan, dtype=float)
    metric[valid] = depth_samples[valid]
    return HandRow(
        frame_idx=int(frame["frame_idx"]),
        side=str(hand.get("side", "unknown")),
        detector_score=score,
        local_joints_m=local_joints,
        local_vertices_m=local_vertices,
        cam_t_m=cam_t,
        joints2d_px=joints2d,
        intrinsics=intr,
        metric_joint_depth_m=metric,
    )


def corrected_points(row: HandRow, log_scale: float, ray_shift_m: float, vertices: bool) -> np.ndarray:
    local = row.local_vertices_m if vertices else row.local_joints_m
    base = np.exp(log_scale) * local + row.cam_t_m[None, :]
    center = np.median(base, axis=0)
    if center[2] <= 0.0:
        raise RuntimeError("non-positive hand center depth")
    ray = center / center[2]
    out = base + float(ray_shift_m) * ray[None, :]
    if np.any(out[:, 2] <= 0.0):
        raise RuntimeError("non-positive corrected depth")
    return out


def depth_shift_for_scale(row: HandRow, log_scale: float, args: argparse.Namespace) -> float:
    joints = corrected_points(row, log_scale, 0.0, vertices=False)
    valid = np.isfinite(row.metric_joint_depth_m) & (row.metric_joint_depth_m > 0.0)
    target_shift = float(np.median(row.metric_joint_depth_m[valid] - joints[valid, 2]))
    return float(np.clip(target_shift, -args.max_abs_ray_shift_m, args.max_abs_ray_shift_m))


def row_metrics(params: np.ndarray, rows: list[HandRow]) -> list[dict]:
    log_scale = float(params[0])
    shifts = np.asarray(params[1:], dtype=float)
    out = []
    for i, row in enumerate(rows):
        before = row.local_joints_m + row.cam_t_m[None, :]
        after = corrected_points(row, log_scale, shifts[i], vertices=False)
        projected_before = project_points(before, row.intrinsics)
        projected_after = project_points(after, row.intrinsics)
        valid = np.isfinite(row.metric_joint_depth_m) & (row.metric_joint_depth_m > 0.0)
        span_before = float(np.linalg.norm(before[12] - before[0]))
        span_after = float(np.linalg.norm(after[12] - after[0]))
        out.append(
            {
                "frame_idx": row.frame_idx,
                "side": row.side,
                "detector_score": row.detector_score,
                "ray_shift_m": float(shifts[i]),
                "joint_reprojection_median_before_px": float(np.median(np.linalg.norm(projected_before - row.joints2d_px, axis=1))),
                "joint_reprojection_median_after_px": float(np.median(np.linalg.norm(projected_after - row.joints2d_px, axis=1))),
                "mano_minus_metric_depth_before_m": float(np.median(before[valid, 2] - row.metric_joint_depth_m[valid])),
                "mano_minus_metric_depth_after_m": float(np.median(after[valid, 2] - row.metric_joint_depth_m[valid])),
                "span_before_m": span_before,
                "span_after_m": span_after,
                "span_change_m": span_after - span_before,
            }
        )
    return out


def score_params(params: np.ndarray, rows: list[HandRow], args: argparse.Namespace) -> float:
    metrics = row_metrics(params, rows)
    depth = np.asarray([row["mano_minus_metric_depth_after_m"] for row in metrics], dtype=float)
    reproj = np.asarray([row["joint_reprojection_median_after_px"] for row in metrics], dtype=float)
    span = np.asarray([row["span_after_m"] for row in metrics], dtype=float)
    shifts = np.asarray(params[1:], dtype=float)
    span_low = np.clip(args.min_span_m - span, 0.0, None)
    span_high = np.clip(span - args.max_span_m, 0.0, None)
    return float(
        np.mean((depth / args.sigma_metric_depth_m) ** 2)
        + np.mean((np.clip(reproj, 0.0, args.max_reprojection_px) / args.sigma_reprojection_px) ** 2)
        + (float(params[0]) / args.sigma_log_scale) ** 2
        + np.mean((shifts / args.sigma_ray_shift_m) ** 2)
        + np.mean((span_low / args.sigma_span_m) ** 2)
        + np.mean((span_high / args.sigma_span_m) ** 2)
    )


def solve_by_scale_scan(rows: list[HandRow], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    scales = np.linspace(args.min_hand_scale, args.max_hand_scale, args.scale_grid_count)
    best_params = None
    best_score = float("inf")
    grid = []
    for scale in scales:
        log_scale = float(np.log(scale))
        shifts = np.asarray([depth_shift_for_scale(row, log_scale, args) for row in rows], dtype=float)
        params = np.r_[log_scale, shifts]
        score = score_params(params, rows, args)
        grid.append({"scale": float(scale), "score": score})
        if score < best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError("scale scan produced no candidate")
    return best_params, {"method": "bounded_global_scale_grid_plus_per_row_depth_shift", "best_score": best_score, "grid_preview": grid[:20]}


def summarize_key(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=float)
    return summarize(values)


def metrics_summary(before_rows: list[dict], after_rows: list[dict], prefix: str = "") -> dict:
    _ = prefix
    return {
        "rows": len(after_rows),
        "before": {
            "mano_minus_metric_depth_m": summarize_key(before_rows, "mano_minus_metric_depth_before_m"),
            "joint_reprojection_median_px": summarize_key(before_rows, "joint_reprojection_median_before_px"),
            "span_m": summarize_key(before_rows, "span_before_m"),
        },
        "after": {
            "mano_minus_metric_depth_m": summarize_key(after_rows, "mano_minus_metric_depth_after_m"),
            "joint_reprojection_median_px": summarize_key(after_rows, "joint_reprojection_median_after_px"),
            "span_m": summarize_key(after_rows, "span_after_m"),
            "span_change_m": summarize_key(after_rows, "span_change_m"),
        },
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_frame_idx)) != len(depth_frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    rows: list[HandRow] = []
    skipped = []
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        obj = frame.get("object", {})
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand in frame.get("hands", []):
            row = row_from_hand(frame, hand, depth, source_size, args)
            if row is not None:
                rows.append(row)
    if len(rows) < args.min_rows:
        raise RuntimeError(f"insufficient measured hand rows: {len(rows)}")
    rows.sort(key=lambda r: (r.side, r.frame_idx))
    x0 = np.zeros(1 + len(rows), dtype=float)
    before_rows = row_metrics(x0, rows)
    solved, solver = solve_by_scale_scan(rows, args)
    after_rows = row_metrics(solved, rows)
    good_indices = [
        i
        for i, row in enumerate(before_rows)
        if row["joint_reprojection_median_before_px"] <= args.good_keypoint_reprojection_px
    ]
    good_before = [before_rows[i] for i in good_indices]
    good_after = [after_rows[i] for i in good_indices]
    scale = float(np.exp(solved[0]))
    span_after = np.asarray([row["span_after_m"] for row in after_rows], dtype=float)
    depth_after = np.asarray([abs(row["mano_minus_metric_depth_after_m"]) for row in after_rows], dtype=float)
    reproj_after = np.asarray([row["joint_reprojection_median_after_px"] for row in after_rows], dtype=float)
    scale_at_bound = abs(scale - args.min_hand_scale) <= args.bound_tolerance or abs(scale - args.max_hand_scale) <= args.bound_tolerance
    shift_at_bound = bool(np.max(np.abs(solved[1:])) >= args.max_abs_ray_shift_m - args.bound_tolerance)
    physical_span_ok = bool(np.percentile(span_after, 5.0) >= args.min_span_m and np.percentile(span_after, 95.0) <= args.max_span_m)
    status = "diagnostic_mano_metric_depth_refit"
    if np.median(depth_after) > args.depth_solved_median_m:
        status = "diagnostic_mano_metric_depth_residual_remains"
    elif np.median(reproj_after) > args.reprojection_solved_median_px:
        status = "diagnostic_mano_reprojection_residual_too_large"
    elif scale_at_bound or shift_at_bound:
        status = "diagnostic_mano_refit_requires_bound_saturation"
    elif not physical_span_ok:
        status = "diagnostic_mano_refit_violates_span_prior"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "rows": len(rows),
        "skipped_rows": len(skipped),
        "good_keypoint_reprojection_px": float(args.good_keypoint_reprojection_px),
        "good_keypoint_rows": len(good_after),
        "variables": int(solved.size),
        "solver": solver,
        "hand_scale": scale,
        "scale_at_bound": bool(scale_at_bound),
        "ray_shift_m": summarize(solved[1:]),
        "ray_shift_abs_m": summarize(np.abs(solved[1:])),
        "shift_at_bound": bool(shift_at_bound),
        "before": {
            "mano_minus_metric_depth_m": summarize_key(before_rows, "mano_minus_metric_depth_before_m"),
            "joint_reprojection_median_px": summarize_key(before_rows, "joint_reprojection_median_before_px"),
            "span_m": summarize_key(before_rows, "span_before_m"),
        },
        "after": {
            "mano_minus_metric_depth_m": summarize_key(after_rows, "mano_minus_metric_depth_after_m"),
            "joint_reprojection_median_px": summarize_key(after_rows, "joint_reprojection_median_after_px"),
            "span_m": summarize_key(after_rows, "span_after_m"),
            "span_change_m": summarize_key(after_rows, "span_change_m"),
        },
        "good_keypoint_subset": metrics_summary(good_before, good_after),
        "thresholds": {
            "depth_solved_median_m": float(args.depth_solved_median_m),
            "reprojection_solved_median_px": float(args.reprojection_solved_median_px),
            "min_span_m": float(args.min_span_m),
            "max_span_m": float(args.max_span_m),
        },
        "bounds": {
            "hand_scale": [float(args.min_hand_scale), float(args.max_hand_scale)],
            "max_abs_ray_shift_m": float(args.max_abs_ray_shift_m),
        },
        "priors": {
            "sigma_log_scale": float(args.sigma_log_scale),
            "sigma_ray_shift_m": float(args.sigma_ray_shift_m),
            "sigma_ray_shift_step_m": float(args.sigma_ray_shift_step_m),
            "sigma_metric_depth_m": float(args.sigma_metric_depth_m),
            "sigma_reprojection_px": float(args.sigma_reprojection_px),
            "sigma_span_m": float(args.sigma_span_m),
        },
        "rows_before_preview": before_rows[:80],
        "rows_after_preview": after_rows[:80],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_before_preview", "rows_after_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--min-depth-joints", type=int, default=12)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--sigma-log-scale", type=float, default=0.050)
    parser.add_argument("--sigma-ray-shift-m", type=float, default=0.080)
    parser.add_argument("--sigma-ray-shift-step-m", type=float, default=0.030)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.050)
    parser.add_argument("--sigma-reprojection-px", type=float, default=8.0)
    parser.add_argument("--max-reprojection-px", type=float, default=80.0)
    parser.add_argument("--min-hand-scale", type=float, default=0.85)
    parser.add_argument("--max-hand-scale", type=float, default=1.10)
    parser.add_argument("--max-abs-ray-shift-m", type=float, default=0.220)
    parser.add_argument("--min-span-m", type=float, default=0.110)
    parser.add_argument("--max-span-m", type=float, default=0.210)
    parser.add_argument("--depth-solved-median-m", type=float, default=0.020)
    parser.add_argument("--reprojection-solved-median-px", type=float, default=12.0)
    parser.add_argument("--good-keypoint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--sigma-span-m", type=float, default=0.015)
    parser.add_argument("--scale-grid-count", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
