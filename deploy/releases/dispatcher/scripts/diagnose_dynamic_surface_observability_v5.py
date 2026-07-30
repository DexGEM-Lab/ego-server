#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError(f"{path} offsets do not match frame count")
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, frame in enumerate(frame_idx.tolist()):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        frame_vertices = vertices[v0:v1]
        frame_faces = faces[f0:f1]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"{path} frame {frame} is empty")
        if frame_faces.min() < 0 or frame_faces.max() >= len(frame_vertices):
            raise RuntimeError(f"{path} frame {frame} has invalid face indices")
        meshes[int(frame)] = (frame_vertices, frame_faces)
    return meshes


def manifest_status(path: Path) -> dict[int, str]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    status = {}
    for entry in frames:
        frame_idx = int(entry["frame_idx"])
        status[frame_idx] = str(entry.get("track_status_source", "unlabeled"))
    return status


def zbuffer_rows(path: Path) -> dict[int, dict]:
    rows = load_json(path).get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain nonempty rows")
    return {int(row["frame_idx"]): row for row in rows}


def contact_counts(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    payload = load_json(path)
    rows = payload.get("rows_detail")
    if rows is None:
        rows = payload.get("geometry_backed_rows_preview")
    if rows is None:
        rows = payload.get("rows_preview")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain rows_detail or preview rows")
    counts: dict[int, int] = {}
    for row in rows:
        if bool(row.get("reliable_for_contact", False)):
            frame_idx = int(row["frame_idx"])
            counts[frame_idx] = counts.get(frame_idx, 0) + 1
    return counts


def summarize(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("cannot sample an empty point cloud")
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(max_points), replace=False)]


def pca_extent(points: np.ndarray, quantile: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - center[None, :], full_matrices=False)
    projected = (points - center[None, :]) @ vh.T
    lo = np.quantile(projected, float(quantile), axis=0)
    hi = np.quantile(projected, 1.0 - float(quantile), axis=0)
    return np.sort(hi - lo)[::-1]


def pcd(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if float(voxel_size) > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    if len(cloud.points) == 0:
        raise RuntimeError("point cloud became empty after voxel downsample")
    return cloud


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ homog.T).T[:, :3]


def nearest_summary(source: np.ndarray, target: np.ndarray, threshold_m: float) -> dict:
    distances = cKDTree(target).query(source, k=1)[0]
    return {
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95.0)),
        "within_threshold_fraction": float(np.mean(distances <= float(threshold_m))),
    }


def register_pair(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = target_center - source_center
    result = o3d.pipelines.registration.registration_icp(
        pcd(source, float(args.voxel_size)),
        pcd(target, float(args.voxel_size)),
        float(args.icp_threshold_m),
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(args.icp_iterations)),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    aligned = transform_points(source, transform)
    src_to_tgt = nearest_summary(aligned, target, float(args.surface_overlap_threshold_m))
    tgt_to_src = nearest_summary(target, aligned, float(args.surface_overlap_threshold_m))
    accepted = (
        src_to_tgt["p95_m"] <= float(args.max_pair_p95_m)
        and tgt_to_src["p95_m"] <= float(args.max_pair_p95_m)
        and min(src_to_tgt["within_threshold_fraction"], tgt_to_src["within_threshold_fraction"])
        >= float(args.min_pair_overlap_fraction)
    )
    report = {
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
        "source_to_target": src_to_tgt,
        "target_to_source": tgt_to_src,
        "center_init_translation_m": init[:3, 3].astype(float).tolist(),
        "icp_translation_m": transform[:3, 3].astype(float).tolist(),
        "stable_pair": bool(accepted),
    }
    return transform, report


def contiguous_windows(frame_rows: list[dict], min_length: int) -> list[dict]:
    windows = []
    current = []
    for row in frame_rows:
        if bool(row["map_observable"]):
            current.append(row)
        else:
            if len(current) >= int(min_length):
                windows.append(window_report(current))
            current = []
    if len(current) >= int(min_length):
        windows.append(window_report(current))
    return windows


def window_report(rows: list[dict]) -> dict:
    return {
        "frame_start": int(rows[0]["frame_idx"]),
        "frame_end": int(rows[-1]["frame_idx"]),
        "frames": [int(row["frame_idx"]) for row in rows],
        "length": int(len(rows)),
        "contact_rows": int(sum(int(row["reliable_contact_rows"]) for row in rows)),
        "zbuffer_p95_median_m": float(np.median([float(row["zbuffer_depth_abs_p95_m"]) for row in rows])),
        "stable_neighbor_count_min": int(min(int(row["stable_neighbor_count"]) for row in rows)),
    }


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    statuses = manifest_status(args.manifest)
    zrows = zbuffer_rows(args.zbuffer_qc)
    contact = contact_counts(args.contact_qc)
    frame_ids = [idx for idx in sorted(meshes) if int(args.frame_start) <= idx <= int(args.frame_end)]
    expected = list(range(int(args.frame_start), int(args.frame_end) + 1))
    if frame_ids != expected:
        raise RuntimeError(f"selected mesh frames are not dense: expected {expected}, got {frame_ids}")
    samples: dict[int, np.ndarray] = {}
    extents = {}
    for frame_idx in frame_ids:
        vertices, _faces = meshes[frame_idx]
        points = sample_points(vertices, int(args.max_points_per_frame), int(args.seed) + frame_idx)
        samples[frame_idx] = points
        extents[frame_idx] = pca_extent(points, float(args.extent_quantile))
    extent_matrix = np.vstack([extents[idx] for idx in frame_ids])
    median_extent = np.median(extent_matrix, axis=0)
    pair_rows = []
    stable_neighbors: dict[int, int] = {idx: 0 for idx in frame_ids}
    for prev_idx, cur_idx in zip(frame_ids[:-1], frame_ids[1:], strict=True):
        _transform, report = register_pair(samples[cur_idx], samples[prev_idx], args)
        report.update({"from_frame": int(cur_idx), "to_frame": int(prev_idx)})
        pair_rows.append(report)
        if bool(report["stable_pair"]):
            stable_neighbors[prev_idx] += 1
            stable_neighbors[cur_idx] += 1
    frame_rows = []
    for frame_idx in frame_ids:
        if frame_idx not in zrows:
            raise RuntimeError(f"z-buffer QC lacks frame {frame_idx}")
        zrow = zrows[frame_idx]
        extent_ratio = extents[frame_idx] / np.maximum(median_extent, 1e-9)
        max_extent_log = float(np.max(np.abs(np.log(np.maximum(extent_ratio, 1e-9)))))
        silhouette_ok = float(zrow["silhouette_mask_iou"]) >= float(args.min_silhouette_iou)
        visible_inside_ok = (
            float(zrow["visible_silhouette_inside_mask_fraction"]) >= float(args.min_visible_inside_fraction)
        )
        depth_p95_ok = float(zrow["zbuffer_depth_abs_p95_m"]) <= float(args.max_zbuffer_p95_m)
        zbuffer_ok = bool(silhouette_ok and visible_inside_ok and depth_p95_ok)
        extent_ok = max_extent_log <= float(args.max_extent_log)
        temporal_ok = int(stable_neighbors[frame_idx]) >= int(args.min_stable_neighbors)
        map_observable = zbuffer_ok and extent_ok and temporal_ok
        reasons = []
        if not silhouette_ok:
            reasons.append("low_silhouette_iou")
        if not visible_inside_ok:
            reasons.append("low_visible_inside_fraction")
        if not depth_p95_ok:
            reasons.append("high_zbuffer_p95")
        if not extent_ok:
            reasons.append("extent_outlier")
        if not temporal_ok:
            reasons.append("unstable_temporal_overlap")
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "track_status_source": statuses.get(frame_idx, "unlabeled"),
                "zbuffer_ok": bool(zbuffer_ok),
                "extent_ok": bool(extent_ok),
                "temporal_overlap_ok": bool(temporal_ok),
                "map_observable": bool(map_observable),
                "reject_reasons": reasons,
                "stable_neighbor_count": int(stable_neighbors[frame_idx]),
                "reliable_contact_rows": int(contact.get(frame_idx, 0)),
                "pca_extent_m": extents[frame_idx].astype(float).tolist(),
                "pca_extent_ratio_to_median": extent_ratio.astype(float).tolist(),
                "pca_extent_max_abs_log_to_median": max_extent_log,
                "silhouette_mask_iou": float(zrow["silhouette_mask_iou"]),
                "visible_silhouette_inside_mask_fraction": float(zrow["visible_silhouette_inside_mask_fraction"]),
                "zbuffer_depth_abs_p95_m": float(zrow["zbuffer_depth_abs_p95_m"]),
            }
        )
    observable = [row for row in frame_rows if bool(row["map_observable"])]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "dynamic_surface_observability_v5",
        "claim_tested": "which V4 accepted object mesh frames are observable enough for a dynamic surface map",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "zbuffer_qc": str(args.zbuffer_qc),
        "contact_qc": str(args.contact_qc) if args.contact_qc is not None else None,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(frame_ids)),
        "observable_frames": [int(row["frame_idx"]) for row in observable],
        "rejected_frames": [
            {"frame_idx": int(row["frame_idx"]), "reasons": row["reject_reasons"]}
            for row in frame_rows
            if not bool(row["map_observable"])
        ],
        "candidate_windows": contiguous_windows(frame_rows, int(args.min_window_length)),
        "pca_extent_median_m": median_extent.astype(float).tolist(),
        "pair_stability": {
            "stable_pairs": int(sum(1 for row in pair_rows if bool(row["stable_pair"]))),
            "pairs": int(len(pair_rows)),
            "source_to_target_p95_m": summarize([row["source_to_target"]["p95_m"] for row in pair_rows]),
            "target_to_source_p95_m": summarize([row["target_to_source"]["p95_m"] for row in pair_rows]),
            "source_to_target_overlap": summarize([row["source_to_target"]["within_threshold_fraction"] for row in pair_rows]),
            "target_to_source_overlap": summarize([row["target_to_source"]["within_threshold_fraction"] for row in pair_rows]),
        },
        "thresholds": {
            "min_silhouette_iou": float(args.min_silhouette_iou),
            "min_visible_inside_fraction": float(args.min_visible_inside_fraction),
            "max_zbuffer_p95_m": float(args.max_zbuffer_p95_m),
            "max_extent_log": float(args.max_extent_log),
            "surface_overlap_threshold_m": float(args.surface_overlap_threshold_m),
            "max_pair_p95_m": float(args.max_pair_p95_m),
            "min_pair_overlap_fraction": float(args.min_pair_overlap_fraction),
            "min_stable_neighbors": int(args.min_stable_neighbors),
        },
        "frame_rows": frame_rows,
        "pair_rows": pair_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_rows", "pair_rows"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zbuffer-qc", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--max-points-per-frame", type=int, default=2500)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--icp-threshold-m", type=float, default=0.035)
    parser.add_argument("--icp-iterations", type=int, default=45)
    parser.add_argument("--surface-overlap-threshold-m", type=float, default=0.006)
    parser.add_argument("--max-pair-p95-m", type=float, default=0.018)
    parser.add_argument("--min-pair-overlap-fraction", type=float, default=0.45)
    parser.add_argument("--extent-quantile", type=float, default=0.05)
    parser.add_argument("--max-extent-log", type=float, default=0.55)
    parser.add_argument("--min-silhouette-iou", type=float, default=0.9)
    parser.add_argument("--min-visible-inside-fraction", type=float, default=0.97)
    parser.add_argument("--max-zbuffer-p95-m", type=float, default=0.010)
    parser.add_argument("--min-stable-neighbors", type=int, default=1)
    parser.add_argument("--min-window-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=531)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
