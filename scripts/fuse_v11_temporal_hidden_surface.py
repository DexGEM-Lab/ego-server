#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from append_v10_mesh4d_hidden_faces_to_observed_mesh import annotations_by_frame, manifest_by_frame
from append_v9_hidden_prior_faces_to_observed_mesh import projection_filter
from check_v11_hidden_face_temporal_qc import hidden_submesh, rows_by_frame, sample_surface
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_factor_graph_v6 import report_pair_rows
from fit_cotracker_pairwise_rigid_factors_v6 import summarize
from render_bundlesdf_mesh_qc_v3 import load_depth_archive


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def pair_transforms(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]]:
    report = load_json(path)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]] = {}
    for row in report_pair_rows(report, path):
        source = int(row["source_frame"])
        target = int(row["target_frame"])
        if not row.get("rigid_factor_ready"):
            continue
        rot = np.asarray(row["rotation"], dtype=np.float64)
        trans = np.asarray(row["translation_m"], dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,):
            raise RuntimeError(f"invalid pair transform for {source}->{target}")
        out[(source, target)] = (rot, trans, row)
    return out


def compose_to_reference(frames: list[int], transforms: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]], reference_frame: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if reference_frame not in frames:
        raise RuntimeError(f"reference frame {reference_frame} is absent from selected frames")
    ref_index = frames.index(reference_frame)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {reference_frame: (np.eye(3), np.zeros(3, dtype=np.float64))}
    rot = np.eye(3)
    trans = np.zeros(3, dtype=np.float64)
    for i in range(ref_index - 1, -1, -1):
        source = frames[i]
        target = frames[i + 1]
        if (source, target) not in transforms:
            raise RuntimeError(f"missing motion factor for {source}->{target}")
        pair_rot, pair_trans, _row = transforms[(source, target)]
        inv_rot = pair_rot.T
        inv_trans = -pair_trans @ inv_rot
        rot = rot @ inv_rot
        trans = trans @ inv_rot + inv_trans
        out[source] = (rot.copy(), trans.copy())
    rot = np.eye(3)
    trans = np.zeros(3, dtype=np.float64)
    for i in range(ref_index, len(frames) - 1):
        source = frames[i]
        target = frames[i + 1]
        if (source, target) not in transforms:
            raise RuntimeError(f"missing motion factor for {source}->{target}")
        pair_rot, pair_trans, _row = transforms[(source, target)]
        rot = rot @ pair_rot
        trans = trans @ pair_rot + pair_trans
        out[target] = (rot.copy(), trans.copy())
    return out


def inverse_transform(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - trans[None, :]) @ rot.T


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rot + trans[None, :]


def support_mask(points: np.ndarray, frame_id: np.ndarray, radius_m: float, min_support_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise RuntimeError("no hidden points to support-filter")
    tree = cKDTree(points)
    keep = np.zeros(len(points), dtype=bool)
    support_counts = np.zeros(len(points), dtype=np.int32)
    for i, neighbors in enumerate(tree.query_ball_point(points, r=float(radius_m))):
        support = len(set(frame_id[np.asarray(neighbors, dtype=np.int64)].tolist()))
        support_counts[i] = int(support)
        keep[i] = support >= int(min_support_frames)
    return keep, support_counts


def voxel_downsample(points: np.ndarray, frame_id: np.ndarray, voxel_size_m: float, support_counts: np.ndarray) -> tuple[np.ndarray, dict]:
    if len(points) == 0:
        raise RuntimeError("cannot downsample empty point set")
    voxel = float(voxel_size_m)
    keys = np.floor(points / voxel).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)
    out = []
    frame_support = []
    point_support = []
    for ids in buckets.values():
        idx = np.asarray(ids, dtype=np.int64)
        out.append(np.median(points[idx], axis=0))
        frame_support.append(len(set(frame_id[idx].tolist())))
        point_support.append(int(np.max(support_counts[idx])))
    return np.asarray(out, dtype=np.float64), {
        "input_points": int(len(points)),
        "output_points": int(len(out)),
        "voxel_size_m": float(voxel_size_m),
        "voxel_frame_support": summarize(np.asarray(frame_support, dtype=np.float64)),
        "voxel_point_support": summarize(np.asarray(point_support, dtype=np.float64)),
    }


def knn_surface(points: np.ndarray, k: int, max_edge_m: float) -> tuple[np.ndarray, dict]:
    if len(points) < 3:
        raise RuntimeError("too few points for KNN surface")
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=min(int(k) + 1, len(points)))
    faces: set[tuple[int, int, int]] = set()
    edge_lengths = []
    for center, neighbors in enumerate(np.asarray(idxs[:, 1:], dtype=np.int64)):
        valid = [int(n) for n, d in zip(neighbors, dists[center, 1:]) if np.isfinite(d) and float(d) <= float(max_edge_m)]
        if len(valid) < 2:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            tri = tuple(sorted((int(center), int(a), int(b))))
            if len(set(tri)) == 3:
                faces.add(tri)
                edge_lengths.extend(
                    [
                        float(np.linalg.norm(points[tri[0]] - points[tri[1]])),
                        float(np.linalg.norm(points[tri[1]] - points[tri[2]])),
                        float(np.linalg.norm(points[tri[2]] - points[tri[0]])),
                    ]
                )
    if not faces:
        raise RuntimeError("KNN surface produced no faces")
    return np.asarray(sorted(faces), dtype=np.int32), {
        "knn_k": int(k),
        "max_edge_m": float(max_edge_m),
        "face_count": int(len(faces)),
        "edge_length_m": summarize(np.asarray(edge_lengths, dtype=np.float64)),
    }


def append_hidden(observed_vertices: np.ndarray, observed_faces: np.ndarray, hidden_vertices: np.ndarray, hidden_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack([np.asarray(observed_vertices, dtype=np.float64), np.asarray(hidden_vertices, dtype=np.float64)])
    faces = np.vstack([np.asarray(observed_faces, dtype=np.int32), np.asarray(hidden_faces, dtype=np.int32) + len(observed_vertices)])
    return vertices, faces


def filter_fused_faces(
    frame: int,
    hidden_world: np.ndarray,
    hidden_faces: np.ndarray,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    manifest: dict[int, dict],
    annotations: dict[int, dict],
    depths: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if frame not in manifest or frame not in annotations or frame not in depths:
        raise RuntimeError(f"filter evidence missing for frame {frame}")
    face_keep, row = projection_filter(
        hidden_world,
        hidden_faces,
        observed_vertices,
        observed_faces,
        manifest[frame],
        annotations[frame],
        np.asarray(depths[frame], dtype=np.float64),
        args,
    )
    return hidden_faces[face_keep], row


def write_mesh_archive(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    if not rows:
        raise RuntimeError("no meshes to write")
    frame_idx = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame, vertices, faces in rows:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise RuntimeError(f"invalid vertices for frame {frame}: {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise RuntimeError(f"invalid faces for frame {frame}: {faces.shape}")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise RuntimeError(f"face index out of range for frame {frame}")
        frame_idx.append(int(frame))
        vertices_all.append(vertices.astype(np.float32))
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.concatenate(vertices_all, axis=0),
        faces=np.concatenate(faces_all, axis=0),
    )


def run(args: argparse.Namespace) -> dict:
    observed_meshes = load_mesh_archive(args.observed_mesh_archive)
    hidden_meshes = load_mesh_archive(args.hidden_mesh_archive)
    append_rows = rows_by_frame(load_json(args.append_report))
    manifest = manifest_by_frame(args.manifest) if args.manifest is not None else {}
    annotations = annotations_by_frame(args.annotations) if args.annotations is not None else {}
    depths = load_depth_archive(args.metric_depth_npz) if args.metric_depth_npz is not None else {}
    frames = [frame for frame in sorted(hidden_meshes) if int(args.frame_start) <= frame <= int(args.frame_end)]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames selected")
    transforms = pair_transforms(args.pair_factors_json)
    to_ref = compose_to_reference(frames, transforms, int(args.reference_frame))

    samples = []
    sample_frames = []
    frame_rows = []
    for frame in frames:
        vertices, faces = hidden_submesh(hidden_meshes, append_rows, frame)
        if len(faces) == 0:
            frame_rows.append({"frame_idx": int(frame), "hidden_faces": 0, "sample_points": 0})
            continue
        sample_count = min(int(args.sample_points_per_frame), max(int(args.min_sample_points_per_frame), len(faces)))
        points_world = sample_surface(vertices, faces, sample_count, seed=int(args.seed) + frame)
        rot, trans = to_ref[frame]
        points_ref = inverse_transform(points_world, rot, trans)
        samples.append(points_ref)
        sample_frames.append(np.full(len(points_ref), int(frame), dtype=np.int32))
        frame_rows.append({"frame_idx": int(frame), "hidden_faces": int(len(faces)), "sample_points": int(len(points_ref))})
    if not samples:
        raise RuntimeError("no hidden samples selected")
    points = np.vstack(samples)
    frame_id = np.concatenate(sample_frames)
    keep, support_counts = support_mask(points, frame_id, float(args.support_radius_m), int(args.min_support_frames))
    if int(np.count_nonzero(keep)) < int(args.min_supported_points):
        raise RuntimeError(f"only {int(np.count_nonzero(keep))} supported hidden points")
    supported = points[keep]
    supported_frames = frame_id[keep]
    supported_counts = support_counts[keep]
    fused_vertices_ref, downsample_report = voxel_downsample(supported, supported_frames, float(args.voxel_size_m), supported_counts)
    if len(fused_vertices_ref) > int(args.max_fused_vertices):
        stride = int(np.ceil(len(fused_vertices_ref) / int(args.max_fused_vertices)))
        fused_vertices_ref = fused_vertices_ref[::stride]
        downsample_report["stride_pruned_to_max_vertices"] = int(stride)
        downsample_report["output_points_after_stride"] = int(len(fused_vertices_ref))
    fused_faces, surface_report = knn_surface(fused_vertices_ref, int(args.knn_k), float(args.max_edge_m))

    archive_rows = []
    output_frame_rows = []
    for frame in frames:
        if frame not in observed_meshes:
            raise RuntimeError(f"observed archive lacks frame {frame}")
        observed_vertices, observed_faces = observed_meshes[frame]
        rot, trans = to_ref[frame]
        hidden_world = transform_points(fused_vertices_ref, rot, trans)
        if args.apply_projection_filter:
            retained_hidden_faces, filter_row = filter_fused_faces(
                frame,
                hidden_world,
                fused_faces,
                observed_vertices,
                observed_faces,
                manifest,
                annotations,
                depths,
                args,
            )
        else:
            retained_hidden_faces = fused_faces
            filter_row = {}
        vertices, faces = append_hidden(observed_vertices, observed_faces, hidden_world, retained_hidden_faces)
        archive_rows.append((frame, vertices, faces))
        output_frame_rows.append(
            {
                "frame_idx": int(frame),
                "observed_vertices": int(len(observed_vertices)),
                "observed_faces": int(len(observed_faces)),
                "fused_hidden_vertices": int(len(hidden_world)),
                "fused_hidden_faces": int(len(fused_faces)),
                "retained_hidden_faces": int(len(retained_hidden_faces)),
                "archive_vertices": int(len(vertices)),
                "archive_faces": int(len(faces)),
                "projection_filter": filter_row,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observed_plus_temporal_fused_hidden_meshes_world.npz"
    write_mesh_archive(archive, archive_rows)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "fuse_v11_temporal_hidden_surface",
        "claim_tested": (
            "hidden Mesh4D proposals should be fused into one shared temporal surface before replay; "
            "per-frame append fragments were rejected by hidden-face temporal QC"
        ),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "hidden_mesh_archive": str(args.hidden_mesh_archive),
        "append_report": str(args.append_report),
        "pair_factors_json": str(args.pair_factors_json),
        "manifest": None if args.manifest is None else str(args.manifest),
        "annotations": None if args.annotations is None else str(args.annotations),
        "metric_depth_npz": None if args.metric_depth_npz is None else str(args.metric_depth_npz),
        "mesh_archive": str(archive),
        "output_mesh_archive": str(archive),
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "reference_frame": int(args.reference_frame),
        "input_hidden_samples": int(len(points)),
        "supported_hidden_samples": int(len(supported)),
        "support_sample_fraction": float(len(supported) / len(points)),
        "fused_hidden_vertices": int(len(fused_vertices_ref)),
        "fused_hidden_faces": int(len(fused_faces)),
        "retained_hidden_faces": summarize(np.asarray([row["retained_hidden_faces"] for row in output_frame_rows], dtype=np.float64)),
        "support": {
            "radius_m": float(args.support_radius_m),
            "min_support_frames": int(args.min_support_frames),
            "support_frame_count": summarize(support_counts.astype(np.float64)),
        },
        "downsample": downsample_report,
        "surface": surface_report,
        "input_frames": frame_rows,
        "output_frames": output_frame_rows,
        "parameters": {
            "sample_points_per_frame": int(args.sample_points_per_frame),
            "min_sample_points_per_frame": int(args.min_sample_points_per_frame),
            "voxel_size_m": float(args.voxel_size_m),
            "knn_k": int(args.knn_k),
            "max_edge_m": float(args.max_edge_m),
            "seed": int(args.seed),
            "apply_projection_filter": bool(args.apply_projection_filter),
            "max_raster_filter_iters": int(args.max_raster_filter_iters),
            "min_component_faces": int(args.min_component_faces),
        },
    }
    output_json = args.output_dir / "qc_fuse_v11_temporal_hidden_surface.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"input_frames", "output_frames"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--hidden-mesh-archive", type=Path, required=True)
    parser.add_argument("--append-report", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--metric-depth-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--sample-points-per-frame", type=int, default=12000)
    parser.add_argument("--min-sample-points-per-frame", type=int, default=2000)
    parser.add_argument("--support-radius-m", type=float, default=0.012)
    parser.add_argument("--min-support-frames", type=int, default=2)
    parser.add_argument("--min-supported-points", type=int, default=500)
    parser.add_argument("--voxel-size-m", type=float, default=0.006)
    parser.add_argument("--max-fused-vertices", type=int, default=25000)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--max-edge-m", type=float, default=0.02)
    parser.add_argument("--apply-projection-filter", action="store_true")
    parser.add_argument("--intrinsics-source", choices=("annotation-vggt", "manifest"), default="annotation-vggt")
    parser.add_argument("--min-hidden-distance-m", type=float, default=0.004)
    parser.add_argument("--max-visible-depth-abs-m", type=float, default=0.008)
    parser.add_argument("--max-front-free-space-m", type=float, default=0.004)
    parser.add_argument("--min-hidden-behind-observed-m", type=float, default=0.003)
    parser.add_argument("--max-raster-filter-iters", type=int, default=12)
    parser.add_argument("--allow-visible-mask-fill", action="store_true")
    parser.add_argument("--keep-inside-image-hidden-faces", action="store_true")
    parser.add_argument("--min-component-faces", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9101)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
