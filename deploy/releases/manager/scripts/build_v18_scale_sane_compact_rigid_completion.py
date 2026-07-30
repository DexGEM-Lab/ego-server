#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a scale-sane compact-rigid completion candidate from a TRELLIS prior.

The repair avoids aligning a hidden prior to broad full-timeline fused surfaces. It
aligns a model-produced TRELLIS mesh to one selected frame's visible-depth sample
in the existing graph-canonical object coordinates. The output is a candidate
hidden prior with explicit uncertainty; observed-depth overwrite/free-space
validation still must be run before it can act as a trusted MANO constraint.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree  # type: ignore[import,reportAttributeAccessIssue]

DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "task5_tomato_960/annotations_v18_full.json"
)
DEFAULT_TRELLIS_MESH = Path(
    "/data2/ego_annotation_outputs/v18_compact_rigid_completion_frame929/task5_tomato_960/"
    "object_obj_tomato/trellis_prior_seed42/trellis_tomato_frame929_seed42.ply"
)
DEFAULT_TRELLIS_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_compact_rigid_completion_frame929/task5_tomato_960/"
    "object_obj_tomato/trellis_prior_seed42/qc_trellis_shape_v3_local.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/"
    "object_obj_tomato/completed_mesh_frame929prior_frame806scale_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--trellis-mesh", type=Path, default=DEFAULT_TRELLIS_MESH)
    parser.add_argument("--trellis-report", type=Path, default=DEFAULT_TRELLIS_REPORT)
    parser.add_argument("--object-id", default="object:obj_tomato")
    parser.add_argument("--scale-frame-idx", type=int, default=806)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-sample-count", type=int, default=8000)
    parser.add_argument("--icp-iterations", type=int, default=6)
    parser.add_argument("--observed-band-m", type=float, default=0.010392304845413263)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [mesh for mesh in geom.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No triangle mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh geometry from {path}: {type(geom)}")
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(geom.vertices, dtype=float),
        faces=np.asarray(geom.faces, dtype=np.int64),
        process=False,
    )


def pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return axes


def radius_percentile(points: np.ndarray, percentile: float) -> float:
    radius = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return float(np.percentile(radius, percentile))


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    distances, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "mean_m": float(np.mean(distances)),
        "max_m": float(np.max(distances)),
    }


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError("rigid fit requires matched arrays with >=3 points")
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    centered_src = src - mean_src
    centered_dst = dst - mean_dst
    u, _, vt = np.linalg.svd(centered_src.T @ centered_dst / len(src))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = mean_dst - mean_src @ rotation.T
    return rotation, translation


def selected_visible_points_canonical(annotations: dict[str, Any], frame_idx: int, object_id: str) -> np.ndarray:
    frame = annotations["frames"][frame_idx]
    obj = next((candidate for candidate in frame.get("objects", []) if candidate.get("object_id") == object_id), None)
    if obj is None:
        raise RuntimeError(f"Object {object_id} not found in frame {frame_idx}")
    pose = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
    rotation = np.asarray(pose.get("rotation_world_from_canonical_matrix") or [], dtype=float)
    translation = np.asarray(pose.get("translation_world_m") or [], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError(f"Object {object_id} frame {frame_idx} lacks graph canonical pose")
    geometry = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    observed_world = np.asarray(geometry.get("world_vertices_sample_m") or [], dtype=float)
    if observed_world.ndim != 2 or observed_world.shape[1] != 3 or len(observed_world) < 3:
        raise RuntimeError(f"Object {object_id} frame {frame_idx} lacks visible-depth sample")
    return (observed_world - translation[None, :]) @ rotation


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    mesh = load_mesh(args.trellis_mesh)
    observed_canonical = selected_visible_points_canonical(annotations, args.scale_frame_idx, args.object_id)

    rng = np.random.default_rng(20260620)
    sample_count = min(int(args.surface_sample_count), max(1, len(mesh.faces)))
    mesh_sample, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    mesh_sample = np.asarray(mesh_sample, dtype=float)

    mesh_mean = mesh_sample.mean(axis=0)
    observed_mean = observed_canonical.mean(axis=0)
    mesh_axes = pca_axes(mesh_sample)
    observed_axes = pca_axes(observed_canonical)
    scale = radius_percentile(observed_canonical, 95) / max(radius_percentile(mesh_sample, 95), 1.0e-9)

    best: tuple[tuple[float, float], tuple[float, ...], np.ndarray, dict[str, Any]] | None = None
    for signs in itertools.product([-1.0, 1.0], repeat=3):
        sign_matrix = np.diag(signs)
        rotation = mesh_axes @ sign_matrix @ observed_axes.T
        transformed = (mesh_sample - mesh_mean) @ rotation * scale + observed_mean
        summary = nearest_summary(observed_canonical, transformed)
        key = (float(summary["median_m"]), float(summary["p95_m"]))
        if best is None or key < best[0]:
            best = (key, signs, rotation, summary)
    if best is None:
        raise RuntimeError("failed to initialize TRELLIS alignment")

    _, signs, rotation_total, summary = best
    translation_total = observed_mean - (mesh_mean @ rotation_total) * scale
    sample_current = mesh_sample @ rotation_total * scale + translation_total
    trace: list[dict[str, Any]] = [
        {"stage": "pca_sign_init", "signs": list(signs), "observed_to_mesh": summary}
    ]

    for iteration in range(int(args.icp_iterations)):
        _, indices = cKDTree(sample_current).query(observed_canonical, k=1, workers=-1)
        source_model = mesh_sample[indices]
        source_current = source_model @ rotation_total * scale + translation_total
        delta_rotation, delta_translation = rigid_fit(source_current, observed_canonical)
        rotation_total = rotation_total @ delta_rotation.T
        translation_total = translation_total @ delta_rotation.T + delta_translation
        sample_current = mesh_sample @ rotation_total * scale + translation_total
        trace.append(
            {
                "stage": f"rigid_icp_{iteration + 1}",
                "observed_to_mesh": nearest_summary(observed_canonical, sample_current),
            }
        )

    vertices = np.asarray(mesh.vertices, dtype=float) @ rotation_total * scale + translation_total
    repaired_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / f"{args.object_id.replace(':', '_').replace('/', '_')}_scale_sane_completed_mesh_labeled.ply"
    repaired_mesh.export(mesh_path)

    label_path = args.output_dir / "completed_mesh_face_labels.json"
    label_path.write_text(
        json.dumps(
            {
                "face_label_schema": (
                    "all faces are scale-sane TRELLIS hidden prior; observed-depth overwrite/free-space validation "
                    "not yet rebuilt"
                ),
                "counts": {"trellis_inferred_hidden_surface_uncertain": int(len(repaired_mesh.faces))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    covariance = np.cov((vertices - vertices.mean(axis=0)).T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    source_vertices = np.asarray(mesh.vertices, dtype=float)
    report: dict[str, Any] = {
        "method": "build_v18_scale_sane_compact_rigid_completion",
        "status": "ok",
        "object_id": args.object_id,
        "claim_scope": (
            "Repair attempt: a model-produced TRELLIS prior is similarity-aligned to one compact visible-depth "
            "sample in graph canonical coordinates. Hidden volume remains uncertain until observed-depth overwrite, "
            "free-space validation, and visual artifact inspection pass."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "source_trellis_mesh": str(args.trellis_mesh),
            "source_trellis_report": str(args.trellis_report),
            "scale_frame_idx": int(args.scale_frame_idx),
        },
        "alignment": {
            "scale_from_visible_sample_radius95": float(scale),
            "scale_frame_visible_sample_extent_canonical_m": (
                observed_canonical.max(axis=0) - observed_canonical.min(axis=0)
            ).astype(float).tolist(),
            "source_trellis_extent_model_units": (source_vertices.max(axis=0) - source_vertices.min(axis=0)).astype(float).tolist(),
            "trace": trace,
        },
        "outputs": {
            "completed_mesh_labeled": str(mesh_path),
            "completed_face_labels": str(label_path),
        },
        "observed_band_m": float(args.observed_band_m),
        "face_label_counts": {
            "completed_mesh": {"trellis_inferred_hidden_surface_uncertain_scale_sane": int(len(repaired_mesh.faces))}
        },
        "mesh_counts": {
            "completed_vertices": int(len(repaired_mesh.vertices)),
            "completed_faces": int(len(repaired_mesh.faces)),
        },
        "mesh_geometry_summary": {
            "extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
            "pca_sqrt_variance_m": np.sqrt(np.maximum(eigenvalues, 0.0)).astype(float).tolist(),
            "pca_variance_ratio": (eigenvalues / max(float(eigenvalues[0]), 1.0e-12)).astype(float).tolist(),
        },
        "remaining_gap": (
            "Observed-depth overwrite/free-space validation and visual artifact inspection are still required "
            "before this mesh can constrain MANO."
        ),
    }
    report_path = args.output_dir / "v18_scale_sane_compact_rigid_completion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = build(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(Path(report["outputs"]["completed_mesh_labeled"]).parent / "v18_scale_sane_compact_rigid_completion_report.json"),
                "completed_mesh": report["outputs"]["completed_mesh_labeled"],
                "extent_m": report["mesh_geometry_summary"]["extent_m"],
                "final_alignment": report["alignment"]["trace"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
