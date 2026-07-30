#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def rotation_angle_rad(rot: np.ndarray) -> float:
    cos_theta = float((np.trace(rot) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def cloud_support(points: np.ndarray) -> dict:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("invalid point cloud support input")
    center = np.mean(points, axis=0)
    centered = points - center
    radial = np.linalg.norm(centered, axis=1)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size < 3:
        singular = np.pad(singular, (0, 3 - singular.size), constant_values=0.0)
    rank2_ratio = float(singular[1] / singular[0]) if singular[0] > 1e-12 else 0.0
    rank3_ratio = float(singular[2] / singular[0]) if singular[0] > 1e-12 else 0.0
    return {
        "center_world_m": center.astype(float).tolist(),
        "aabb_extent_m": (np.max(points, axis=0) - np.min(points, axis=0)).astype(float).tolist(),
        "radial_extent_m": summarize(radial),
        "singular_values_m": singular.astype(float).tolist(),
        "rank2_ratio": rank2_ratio,
        "rank3_ratio": rank3_ratio,
    }


def finite_summary_value(summary: dict, key: str) -> float:
    value = summary.get(key)
    if value is None:
        return float("nan")
    return float(value)


def load_track_ids(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    edges = data.get("edges")
    if not isinstance(edges, list) or not edges:
        raise RuntimeError(f"sparse edge report has no edges: {path}")
    return np.asarray(sorted({int(edge["track_id"]) for edge in edges}), dtype=np.int64)


def selected_track_ids(path: Path | None, track_count: int) -> np.ndarray:
    if path is None:
        return np.arange(track_count, dtype=np.int64)
    ids = load_track_ids(path)
    if ids.size and int(np.max(ids)) >= track_count:
        raise RuntimeError("sparse edge track ids exceed CoTracker archive track dimension")
    return ids


def weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("invalid Kabsch inputs")
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(source):
        raise RuntimeError("weight length mismatch")
    total = float(np.sum(weights))
    if total <= 1e-12:
        raise RuntimeError("degenerate Kabsch weights")
    src_center = np.sum(source * weights[:, None], axis=0) / total
    tgt_center = np.sum(target * weights[:, None], axis=0) / total
    src = source - src_center
    tgt = target - tgt_center
    cov = (src * weights[:, None]).T @ tgt
    u, _s, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    trans = tgt_center - src_center @ rot
    if np.linalg.norm(src_center @ rot + trans - tgt_center) > 1e-8:
        raise RuntimeError("Kabsch transform failed centroid consistency")
    return rot, trans


def robust_pair_fit(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    weights = np.ones((len(source),), dtype=np.float64)
    rows = []
    rot = np.eye(3)
    trans = np.zeros((3,), dtype=np.float64)
    residual = np.full((len(source),), np.nan, dtype=np.float64)
    for iteration in range(int(args.irls_iterations)):
        rot, trans = weighted_kabsch(source, target, weights)
        aligned = source @ rot + trans
        residual = np.linalg.norm(aligned - target, axis=1)
        delta = float(args.huber_delta_m)
        new_weights = np.minimum(1.0, delta / np.maximum(residual, 1e-9))
        rows.append(
            {
                "iteration": int(iteration),
                "weight": summarize(weights),
                "residual_m": summarize(residual),
            }
        )
        if np.max(np.abs(new_weights - weights)) < 1e-4:
            weights = new_weights
            break
        weights = new_weights
    return rot, trans, residual, rows


def run(args: argparse.Namespace) -> dict:
    tracks = np.load(args.cotracker_npz)
    frame_idx = np.asarray(tracks["frame_idx"], dtype=np.int64)
    accepted = np.asarray(tracks["accepted"], dtype=bool)
    world = np.asarray(tracks["world_xyz"], dtype=np.float64)
    usable_ids = selected_track_ids(args.sparse_edges_json, accepted.shape[1])

    pair_rows = []
    all_inlier_residuals = []
    ready_inlier_residuals = []
    for i in range(len(frame_idx) - 1):
        source_frame = int(frame_idx[i])
        target_frame = int(frame_idx[i + 1])
        keep = usable_ids[
            accepted[i, usable_ids]
            & accepted[i + 1, usable_ids]
            & np.all(np.isfinite(world[i, usable_ids]), axis=1)
            & np.all(np.isfinite(world[i + 1, usable_ids]), axis=1)
        ]
        if len(keep) < int(args.min_pair_tracks):
            pair_rows.append({"source_frame": source_frame, "target_frame": target_frame, "track_count": int(len(keep)), "status": "too_few_tracks"})
            continue
        source = world[i, keep]
        target = world[i + 1, keep]
        rot, trans, residual, solver_rows = robust_pair_fit(source, target, args)
        source_support = cloud_support(source)
        target_support = cloud_support(target)
        source_center = np.asarray(source_support["center_world_m"], dtype=np.float64)
        target_center = np.asarray(target_support["center_world_m"], dtype=np.float64)
        centroid_displacement_m = float(np.linalg.norm(target_center - source_center))
        angle_rad = rotation_angle_rad(rot)
        source_radial_p95_m = finite_summary_value(source_support["radial_extent_m"], "p95")
        target_radial_p95_m = finite_summary_value(target_support["radial_extent_m"], "p95")
        min_radial_p95_m = float(min(source_radial_p95_m, target_radial_p95_m))
        min_rank2_ratio = float(min(source_support["rank2_ratio"], target_support["rank2_ratio"]))
        inliers = residual <= float(args.max_inlier_residual_m)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_p95 = float(np.percentile(residual[inliers], 95)) if np.any(inliers) else float("inf")
        readiness_checks = {
            "min_inlier_tracks_met": bool(inlier_count >= int(args.min_inlier_tracks)),
            "inlier_p95_met": bool(inlier_p95 <= float(args.accept_inlier_p95_m)),
            "max_centroid_displacement_met": bool(centroid_displacement_m <= float(args.max_centroid_displacement_m)),
            "max_rotation_angle_met": bool(angle_rad <= float(args.max_rotation_angle_rad)),
            "min_radial_extent_met": bool(min_radial_p95_m >= float(args.min_radial_extent_m)),
            "min_rank2_ratio_met": bool(min_rank2_ratio >= float(args.min_rank2_ratio)),
        }
        ready = bool(all(readiness_checks.values()))
        all_inlier_residuals.extend(residual[inliers].astype(float).tolist())
        if ready:
            ready_inlier_residuals.extend(residual[inliers].astype(float).tolist())
        pair_rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "track_count": int(len(keep)),
                "inlier_count": inlier_count,
                "inlier_fraction": float(np.mean(inliers)),
                "rigid_factor_ready": bool(ready),
                "readiness_checks": readiness_checks,
                "rotation_angle_rad": angle_rad,
                "centroid_displacement_m": centroid_displacement_m,
                "min_radial_p95_m": min_radial_p95_m,
                "min_rank2_ratio": min_rank2_ratio,
                "source_support": source_support,
                "target_support": target_support,
                "rotation": rot.tolist(),
                "translation_m": trans.tolist(),
                "residual_m": summarize(residual),
                "inlier_residual_m": summarize(residual[inliers]),
                "solver": solver_rows,
            }
        )

    ready_pairs = [row for row in pair_rows if row.get("rigid_factor_ready")]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "fit_cotracker_pairwise_rigid_factors_v6",
        "claim_tested": "robust SE3 factors on learned sparse object tracks identify frame pairs where material motion is coherent enough to use as a graph factor",
        "cotracker_npz": str(args.cotracker_npz),
        "sparse_edges_json": str(args.sparse_edges_json) if args.sparse_edges_json is not None else None,
        "frames": [int(frame) for frame in frame_idx.tolist()],
        "usable_track_count": int(len(usable_ids)),
        "pair_count": int(len(pair_rows)),
        "rigid_factor_ready_pairs": int(len(ready_pairs)),
        "all_pair_inlier_residual_m": summarize(np.asarray(all_inlier_residuals, dtype=np.float64)),
        "ready_pair_inlier_residual_m": summarize(np.asarray(ready_inlier_residuals, dtype=np.float64)),
        "pair_rows": pair_rows,
        "parameters": {
            "min_pair_tracks": int(args.min_pair_tracks),
            "min_inlier_tracks": int(args.min_inlier_tracks),
            "huber_delta_m": float(args.huber_delta_m),
            "max_inlier_residual_m": float(args.max_inlier_residual_m),
            "accept_inlier_p95_m": float(args.accept_inlier_p95_m),
            "max_centroid_displacement_m": float(args.max_centroid_displacement_m),
            "max_rotation_angle_rad": float(args.max_rotation_angle_rad),
            "min_radial_extent_m": float(args.min_radial_extent_m),
            "min_rank2_ratio": float(args.min_rank2_ratio),
            "irls_iterations": int(args.irls_iterations),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument("--sparse-edges-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-pair-tracks", type=int, default=12)
    parser.add_argument("--min-inlier-tracks", type=int, default=12)
    parser.add_argument("--huber-delta-m", type=float, default=0.010)
    parser.add_argument("--max-inlier-residual-m", type=float, default=0.012)
    parser.add_argument("--accept-inlier-p95-m", type=float, default=0.010)
    parser.add_argument("--max-centroid-displacement-m", type=float, default=0.080)
    parser.add_argument("--max-rotation-angle-rad", type=float, default=0.250)
    parser.add_argument("--min-radial-extent-m", type=float, default=0.010)
    parser.add_argument("--min-rank2-ratio", type=float, default=0.050)
    parser.add_argument("--irls-iterations", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
