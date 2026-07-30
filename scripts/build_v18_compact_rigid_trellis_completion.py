#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_vertices(path: Path) -> np.ndarray:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        parts = [g for g in geom.geometry.values() if hasattr(g, "vertices")]
        if not parts:
            raise RuntimeError(f"no vertex geometry in {path}")
        verts = np.concatenate([np.asarray(g.vertices, dtype=float) for g in parts], axis=0)
    else:
        verts = np.asarray(geom.vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise RuntimeError(f"invalid vertices in {path}")
    if not np.isfinite(verts).all():
        raise RuntimeError(f"non-finite vertices in {path}")
    return verts


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"not a mesh: {path}")
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(1, len(points) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1.0
    return center, vecs, vals


def rotation_candidates(obs_basis: np.ndarray, model_basis: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        perm_mat = np.eye(3)[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            sign_mat = np.diag(signs)
            r = obs_basis @ perm_mat @ sign_mat @ model_basis.T
            if np.linalg.det(r) < 0:
                continue
            candidates.append(r)
    return candidates


def nearest_stats(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    d, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "mean_m": float(np.mean(d)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "p95_m": float(np.percentile(d, 95)),
        "max_m": float(np.max(d)),
    }


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError("Umeyama requires matched arrays with >=3 points")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    cov = (xd.T @ xs) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt
    var_src = np.sum(xs * xs) / len(src)
    scale = float(np.sum(s * d) / max(var_src, 1e-12))
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    count = min(max(1, count), max(1, len(mesh.faces) * 2))
    rng = np.random.default_rng(18)
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(pts, dtype=float)


def align_trellis_to_observed(trellis: trimesh.Trimesh, observed_points: np.ndarray, max_samples: int = 20000) -> dict[str, Any]:
    obs_center, obs_basis, obs_vals = pca_basis(observed_points)
    model_vertices = np.asarray(trellis.vertices, dtype=float)
    model_center, model_basis, model_vals = pca_basis(model_vertices)
    model_samples = deterministic_sample_mesh(trellis, min(max_samples, max(2000, len(trellis.faces))))

    obs_radius = float(np.sqrt(np.mean(np.sum((observed_points - obs_center) ** 2, axis=1))))
    model_radius = float(np.sqrt(np.mean(np.sum((model_samples - model_center) ** 2, axis=1))))
    initial_scale = obs_radius / max(model_radius, 1e-12)

    best: dict[str, Any] | None = None
    for r in rotation_candidates(obs_basis, model_basis):
        transformed_samples = initial_scale * (model_samples @ r.T) + obs_center - initial_scale * (model_center @ r.T)
        stats = nearest_stats(observed_points, transformed_samples)
        score = stats["median_m"] + 0.25 * stats["p90_m"]
        if best is None or score < best["score"]:
            best = {
                "scale": initial_scale,
                "rotation": r,
                "translation": obs_center - initial_scale * (r @ model_center),
                "score": float(score),
                "observed_to_trellis_stats_initial": stats,
            }
    assert best is not None

    # A small ICP refinement updates only the global similarity transform using
    # observed surfels matched to the current TRELLIS surface. The raw source
    # points remain the TRELLIS surface samples, so the model cannot simply copy
    # the observed cloud; residuals expose partial-view misfit.
    r_total = np.asarray(best["rotation"], dtype=float)
    s_total = float(best["scale"])
    t_total = np.asarray(best["translation"], dtype=float)
    refinement_stats = []
    raw_tree_points = model_samples
    for _ in range(6):
        transformed_samples = s_total * (raw_tree_points @ r_total.T) + t_total
        tree = cKDTree(transformed_samples)
        _, idx = tree.query(observed_points, k=1, workers=-1)
        src_match = raw_tree_points[idx]
        s_new, r_new, t_new = umeyama_similarity(src_match, observed_points)
        if not np.isfinite(s_new) or s_new <= 0:
            break
        s_total, r_total, t_total = s_new, r_new, t_new
        refinement_stats.append(nearest_stats(observed_points, s_total * (raw_tree_points @ r_total.T) + t_total))

    final_samples = s_total * (model_samples @ r_total.T) + t_total
    final_stats_obs_to_model = nearest_stats(observed_points, final_samples)
    final_stats_model_to_obs = nearest_stats(final_samples, observed_points)
    return {
        "scale": float(s_total),
        "rotation": r_total.astype(float).tolist(),
        "translation": t_total.astype(float).tolist(),
        "matrix_model_to_canonical": np.vstack([
            np.hstack([s_total * r_total, t_total.reshape(3, 1)]),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
        ]).astype(float).tolist(),
        "initial_scale_from_rms_radius": float(initial_scale),
        "observed_pca_eigenvalues": obs_vals.astype(float).tolist(),
        "trellis_pca_eigenvalues": model_vals.astype(float).tolist(),
        "observed_to_trellis_stats_initial": best["observed_to_trellis_stats_initial"],
        "observed_to_trellis_stats_final": final_stats_obs_to_model,
        "trellis_to_observed_stats_final": final_stats_model_to_obs,
        "icp_refinement_stats": refinement_stats,
    }


def transform_mesh(mesh: trimesh.Trimesh, align: dict[str, Any]) -> trimesh.Trimesh:
    m = np.asarray(align["matrix_model_to_canonical"], dtype=float)
    verts = np.asarray(mesh.vertices, dtype=float)
    hom = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    out = (hom @ m.T)[:, :3]
    return trimesh.Trimesh(vertices=out, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)


def face_center_labels(mesh: trimesh.Trimesh, observed_points: np.ndarray, observed_band_m: float) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(mesh.triangles_center, dtype=float)
    d, _ = cKDTree(observed_points).query(centers, k=1, workers=-1)
    near = d <= observed_band_m
    return d, near


def export_label_sidecar(path: Path, labels: list[str], distances: np.ndarray, source_mesh: str, offset: int = 0) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    sidecar = {
        "source_mesh": source_mesh,
        "face_index_offset_in_combined_mesh": int(offset),
        "face_count": len(labels),
        "label_counts": counts,
        "labels": labels,
        "nearest_observed_surfels_m": distances.astype(float).tolist(),
    }
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def color_for_labels(labels: list[str]) -> np.ndarray:
    colors = {
        "observed_depth_surface": [30, 180, 255, 255],
        "trellis_inferred_hidden_surface": [255, 150, 30, 255],
        "free_space_rejected": [255, 30, 30, 150],
        "unsupported_uncertain": [150, 150, 150, 160],
    }
    return np.asarray([colors.get(l, [255, 255, 255, 255]) for l in labels], dtype=np.uint8)


def resolve_anchor_centroid_world(evidence: dict[str, Any]) -> np.ndarray:
    row = evidence.get("depth_fused_object_row") if isinstance(evidence.get("depth_fused_object_row"), dict) else {}
    mesh_recon = row.get("mesh_reconstruction") if isinstance(row.get("mesh_reconstruction"), dict) else {}
    for value in (
        mesh_recon.get("anchor_centroid_world_m"),
        (evidence.get("selected") or {}).get("visible_geometry_candidate", {}).get("anchor_centroid_world_m") if isinstance(evidence.get("selected"), dict) else None,
        (evidence.get("selected") or {}).get("visible_geometry_candidate", {}).get("centroid_world_m") if isinstance(evidence.get("selected"), dict) else None,
    ):
        arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr
    raise RuntimeError("evidence report lacks anchor_centroid_world_m/centroid_world_m needed for silhouette free-space filtering")


def world_points_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def estimate_projection_source_size(intr: np.ndarray, image_w: int, image_h: int, selected: dict[str, Any]) -> tuple[int, int, str]:
    vg = selected.get("visible_geometry_candidate") if isinstance(selected.get("visible_geometry_candidate"), dict) else {}
    source_w = int(vg.get("source_width") or selected.get("source_width") or 0)
    source_h = int(vg.get("source_height") or selected.get("source_height") or 0)
    if source_w > 0 and source_h > 0:
        return source_w, source_h, "visible_geometry_candidate_source_size"
    # Older evidence reports did not persist source_width/source_height.  The
    # principal point is near the source-frame center, so 2*cx,2*cy reconstructs
    # the source coordinate scale while leaving already-decoded K unchanged.
    fx, fy, cx, cy = [float(x) for x in intr]
    source_w = max(int(image_w), int(round(2.0 * cx)))
    source_h = max(int(image_h), int(round(2.0 * cy)))
    return source_w, source_h, "estimated_from_principal_point_legacy_evidence"


def trellis_planar_slab_keep_mask(
    observed_points: np.ndarray,
    mesh_canonical: trimesh.Trimesh,
    *,
    enabled: bool,
    observed_band_m: float,
    eigenvalue_ratio_max: float,
    min_band_m: float,
    max_band_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    face_count = int(len(mesh_canonical.faces))
    if not enabled:
        return np.ones(face_count, dtype=bool), {"state": "disabled", "kept_faces": face_count, "rejected_faces": 0}
    center, basis, vals = pca_basis(observed_points)
    ratio = float(vals[-1] / max(vals[0], 1e-12))
    if not np.isfinite(ratio) or ratio > float(eigenvalue_ratio_max):
        return np.ones(face_count, dtype=bool), {
            "state": "not_applied_observed_surface_not_planar_enough",
            "observed_pca_eigenvalues": vals.astype(float).tolist(),
            "planarity_ratio_smallest_over_largest": ratio,
            "ratio_threshold": float(eigenvalue_ratio_max),
            "kept_faces": face_count,
            "rejected_faces": 0,
        }
    normal = basis[:, -1]
    obs_dist = np.abs((observed_points - center[None, :]) @ normal)
    data_band = float(np.percentile(obs_dist, 95.0)) if len(obs_dist) else 0.0
    band = max(float(min_band_m), data_band + 2.0 * float(observed_band_m))
    band = min(float(max_band_m), band)
    centers = np.asarray(mesh_canonical.triangles_center, dtype=float)
    dist = np.abs((centers - center[None, :]) @ normal)
    keep = np.isfinite(dist) & (dist <= band)
    return keep, {
        "state": "planar_support_slab_filter_applied",
        "observed_pca_eigenvalues": vals.astype(float).tolist(),
        "planarity_ratio_smallest_over_largest": ratio,
        "ratio_threshold": float(eigenvalue_ratio_max),
        "observed_normal_abs_distance_p95_m": data_band,
        "observed_band_m": float(observed_band_m),
        "slab_half_width_m": float(band),
        "min_band_m": float(min_band_m),
        "max_band_m": float(max_band_m),
        "input_trellis_faces": face_count,
        "kept_by_planar_slab_faces": int(np.count_nonzero(keep)),
        "unsupported_outside_planar_slab_faces": int(np.count_nonzero(~keep)),
        "claim_scope": "When the observed object surface is planar, hidden prior faces far off that observed support plane are not accepted as physical object body.",
    }


def trellis_silhouette_keep_mask(
    evidence: dict[str, Any],
    mesh_canonical: trimesh.Trimesh,
    *,
    enabled: bool,
    dilate_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    face_count = int(len(mesh_canonical.faces))
    if not enabled:
        return np.ones(face_count, dtype=bool), {"state": "disabled", "kept_faces": face_count, "rejected_faces": 0}
    selected = evidence.get("selected") if isinstance(evidence.get("selected"), dict) else {}
    mask_path = selected.get("mask_path")
    if not mask_path and isinstance(selected.get("trellis_conditioning_crop"), dict):
        mask_path = selected["trellis_conditioning_crop"].get("mask")
    if not mask_path:
        raise RuntimeError("evidence selected row lacks mask_path for silhouette free-space filtering")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read silhouette mask for free-space filtering: {mask_path}")
    mask_bool = mask > 0
    if int(dilate_px) > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
    h, w = mask_bool.shape[:2]
    camera = selected.get("camera") if isinstance(selected.get("camera"), dict) else {}
    T = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or [], dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError("evidence selected camera lacks a valid T_world_camera for silhouette free-space filtering")
    intr = np.asarray(camera.get("intrinsics_fx_fy_cx_cy") or [], dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all():
        vg = selected.get("visible_geometry_candidate") if isinstance(selected.get("visible_geometry_candidate"), dict) else {}
        intr = np.asarray(vg.get("intrinsics_fx_fy_cx_cy") or [], dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0.0 or intr[1] <= 0.0:
        raise RuntimeError("evidence selected row lacks valid intrinsics for silhouette free-space filtering")
    source_w, source_h, source_note = estimate_projection_source_size(intr, w, h, selected)
    sx = float(w) / float(source_w)
    sy = float(h) / float(source_h)
    fx, fy, cx, cy = [float(intr[0] * sx), float(intr[1] * sy), float(intr[2] * sx), float(intr[3] * sy)]
    centroid_world = resolve_anchor_centroid_world(evidence)
    centers_canonical = np.asarray(mesh_canonical.triangles_center, dtype=float)
    centers_world = centers_canonical + centroid_world[None, :]
    centers_camera = world_points_to_camera(centers_world, T)
    z = centers_camera[:, 2]
    keep = np.zeros(face_count, dtype=bool)
    valid_z = np.isfinite(z) & (z > 0.01)
    uv = np.full((face_count, 2), np.nan, dtype=float)
    uv[valid_z, 0] = fx * centers_camera[valid_z, 0] / z[valid_z] + cx
    uv[valid_z, 1] = fy * centers_camera[valid_z, 1] / z[valid_z] + cy
    xi = np.rint(uv[:, 0]).astype(np.int64, copy=False)
    yi = np.rint(uv[:, 1]).astype(np.int64, copy=False)
    inside_image = valid_z & np.isfinite(uv).all(axis=1) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    keep[inside_image] = mask_bool[yi[inside_image], xi[inside_image]]
    rejected = ~keep
    return keep, {
        "state": "silhouette_free_space_filter_applied",
        "mask_path": str(mask_path),
        "mask_size": [int(w), int(h)],
        "source_size": [int(source_w), int(source_h)],
        "source_size_note": source_note,
        "scaled_intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
        "dilate_px": int(max(0, dilate_px)),
        "input_trellis_faces": int(face_count),
        "projected_inside_image_faces": int(np.count_nonzero(inside_image)),
        "kept_by_silhouette_faces": int(np.count_nonzero(keep)),
        "free_space_rejected_faces": int(np.count_nonzero(rejected)),
        "claim_scope": "TRELLIS hidden faces whose evidence-frame projection falls outside the object-owned silhouette are not accepted as object body.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--trellis-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observed-band-scale", type=float, default=math.sqrt(3.0), help="multiplier on depth-fusion voxel size; sqrt(3) covers one voxel diagonal")
    parser.add_argument("--silhouette-free-space-filter", action=argparse.BooleanOptionalAction, default=True, help="Reject TRELLIS hidden faces whose evidence-frame projection falls outside the object-owned mask silhouette.")
    parser.add_argument("--silhouette-dilate-px", type=int, default=16, help="Dilation radius in evidence-mask pixels before silhouette/free-space rejection.")
    parser.add_argument("--planar-slab-support-filter", action=argparse.BooleanOptionalAction, default=True, help="When observed surfels are planar, reject hidden-prior faces far off that observed support slab.")
    parser.add_argument("--planar-slab-eigenvalue-ratio-max", type=float, default=0.04, help="Apply planar slab filter only when smallest/largest observed PCA eigenvalue is at most this ratio.")
    parser.add_argument("--planar-slab-min-band-m", type=float, default=0.018, help="Minimum half-width for planar support slab.")
    parser.add_argument("--planar-slab-max-band-m", type=float, default=0.055, help="Maximum half-width for planar support slab.")
    args = parser.parse_args()

    evidence = load_json(args.evidence_report)
    trellis_report = load_json(args.trellis_report)
    partial_paths = evidence.get("partial_metric_geometry_paths", {})
    fused_points_path = Path(partial_paths.get("fused_point_cloud_path") or "")
    poisson_path = Path(partial_paths.get("poisson_mesh_path") or "")
    trellis_mesh_path = Path(trellis_report.get("mesh") or "")
    if not fused_points_path.is_file():
        raise RuntimeError(f"missing fused point cloud: {fused_points_path}")
    if not poisson_path.is_file():
        raise RuntimeError(f"missing Poisson visible mesh: {poisson_path}")
    if not trellis_mesh_path.is_file():
        raise RuntimeError(f"missing TRELLIS mesh: {trellis_mesh_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed_points = load_vertices(fused_points_path)
    trellis = load_mesh(trellis_mesh_path)
    observed_mesh = load_mesh(poisson_path)

    mesh_recon = evidence.get("depth_fused_object_row", {}).get("mesh_reconstruction", {})
    voxel_size_m = float(mesh_recon.get("voxel_size_m") or 0.006)
    observed_band_m = voxel_size_m * float(args.observed_band_scale)

    align = align_trellis_to_observed(trellis, observed_points)
    trellis_canonical = transform_mesh(trellis, align)

    obs_d, obs_near = face_center_labels(observed_mesh, observed_points, observed_band_m)
    observed_labels = ["observed_depth_surface" if x else "unsupported_uncertain" for x in obs_near]
    trellis_d, trellis_near = face_center_labels(trellis_canonical, observed_points, observed_band_m)
    # A TRELLIS face close to observed surfels is an RGB-prior candidate in a
    # region already owned by metric depth; it is omitted from the completed mesh.
    # A TRELLIS hidden face whose evidence-frame projection falls outside the
    # object silhouette is free-space inconsistent and is also omitted.
    silhouette_keep, silhouette_state = trellis_silhouette_keep_mask(
        evidence,
        trellis_canonical,
        enabled=bool(args.silhouette_free_space_filter),
        dilate_px=int(args.silhouette_dilate_px),
    )
    planar_keep, planar_state = trellis_planar_slab_keep_mask(
        observed_points,
        trellis_canonical,
        enabled=bool(args.planar_slab_support_filter),
        observed_band_m=float(observed_band_m),
        eigenvalue_ratio_max=float(args.planar_slab_eigenvalue_ratio_max),
        min_band_m=float(args.planar_slab_min_band_m),
        max_band_m=float(args.planar_slab_max_band_m),
    )
    trellis_labels_all = []
    for near_observed, keep_by_silhouette, keep_by_planar_slab in zip(trellis_near, silhouette_keep, planar_keep):
        if near_observed:
            trellis_labels_all.append("observed_region_overwritten_candidate")
        elif not keep_by_silhouette:
            trellis_labels_all.append("free_space_rejected")
        elif not keep_by_planar_slab:
            trellis_labels_all.append("unsupported_uncertain")
        else:
            trellis_labels_all.append("trellis_inferred_hidden_surface")

    observed_mesh.visual.face_colors = color_for_labels(observed_labels)
    trellis_canonical.visual.face_colors = color_for_labels(trellis_labels_all)

    accepted_observed_faces = np.where(obs_near)[0]
    if len(accepted_observed_faces) == 0:
        raise RuntimeError(
            "observed Poisson mesh has no faces close to fused visible surfels; "
            "refusing to construct accepted object body from unsupported fill"
        )
    accepted_observed = observed_mesh.submesh([accepted_observed_faces], append=True, repair=False)
    accepted_observed_labels = ["observed_depth_surface"] * len(accepted_observed.faces)

    kept_trellis_faces = np.where((~trellis_near) & silhouette_keep & planar_keep)[0]
    kept_trellis = trellis_canonical.submesh([kept_trellis_faces], append=True, repair=False)
    kept_trellis_labels = ["trellis_inferred_hidden_surface"] * len(kept_trellis.faces)

    completed = trimesh.util.concatenate([accepted_observed, kept_trellis])
    completed_labels = accepted_observed_labels + kept_trellis_labels
    completed.visual.face_colors = color_for_labels(completed_labels)

    object_safe = safe_id(str(evidence.get("object_id", "object")).replace("object:", "object_"))
    observed_mesh_path = args.output_dir / f"{object_safe}_observed_depth_surface_labeled.ply"
    trellis_all_path = args.output_dir / f"{object_safe}_trellis_aligned_all_candidate_labeled.ply"
    completed_path = args.output_dir / f"{object_safe}_compact_rigid_completed_mesh_labeled.ply"
    observed_mesh.export(str(observed_mesh_path))
    trellis_canonical.export(str(trellis_all_path))
    completed.export(str(completed_path))

    observed_sidecar = export_label_sidecar(args.output_dir / "observed_depth_surface_face_labels.json", observed_labels, obs_d, str(observed_mesh_path), 0)
    trellis_sidecar = export_label_sidecar(args.output_dir / "trellis_candidate_face_labels.json", trellis_labels_all, trellis_d, str(trellis_all_path), 0)
    completed_sidecar = export_label_sidecar(
        args.output_dir / "completed_mesh_face_labels.json",
        completed_labels,
        np.concatenate([obs_d[accepted_observed_faces], trellis_d[kept_trellis_faces]]),
        str(completed_path),
        0,
    )

    report = {
        "method": "build_v18_compact_rigid_trellis_completion",
        "status": "ok",
        "case": evidence.get("case"),
        "object_id": evidence.get("object_id"),
        "claim_scope": "TRELLIS is metric-aligned as an RGB hidden-surface prior; observed depth-fused surfels remain the source of truth for visible surface regions; unsupported observed Poisson fill is diagnostic uncertainty and is excluded from accepted object body.",
        "inputs": {
            "evidence_report": str(args.evidence_report),
            "trellis_report": str(args.trellis_report),
            "trellis_mesh": str(trellis_mesh_path),
            "observed_fused_points": str(fused_points_path),
            "observed_poisson_mesh": str(poisson_path),
        },
        "metric_alignment": align,
        "observed_band_m": float(observed_band_m),
        "observed_band_derivation": {
            "depth_fusion_voxel_size_m": float(voxel_size_m),
            "scale": float(args.observed_band_scale),
            "reason": "one voxel diagonal around fused surfels distinguishes raw observed support from completion/extrapolation faces",
        },
        "outputs": {
            "observed_depth_surface_labeled_mesh": str(observed_mesh_path),
            "trellis_aligned_all_candidate_labeled_mesh": str(trellis_all_path),
            "completed_mesh_labeled": str(completed_path),
            "observed_face_labels": str(args.output_dir / "observed_depth_surface_face_labels.json"),
            "trellis_candidate_face_labels": str(args.output_dir / "trellis_candidate_face_labels.json"),
            "completed_face_labels": str(args.output_dir / "completed_mesh_face_labels.json"),
        },
        "face_label_counts": {
            "observed_mesh": observed_sidecar["label_counts"],
            "trellis_all_candidate": trellis_sidecar["label_counts"],
            "completed_mesh": completed_sidecar["label_counts"],
            "free_space_rejected": int(trellis_sidecar["label_counts"].get("free_space_rejected", 0)),
        },
        "accepted_body_semantics": {
            "observed_depth_surface_faces_accepted": int(len(accepted_observed_faces)),
            "observed_unsupported_uncertain_faces_excluded": int(len(observed_labels) - len(accepted_observed_faces)),
            "trellis_hidden_surface_faces_accepted": int(len(kept_trellis_faces)),
            "unsupported_uncertain_is_not_object_body": True,
            "claim_scope": "completed_mesh_labeled is the downstream accepted object body; unsupported observed Poisson fill remains only in observed_depth_surface_labeled_mesh and sidecar diagnostics.",
        },
        "free_space_rejection_state": "silhouette_free_space_filter_applied" if bool(args.silhouette_free_space_filter) else "silhouette_free_space_filter_disabled",
        "silhouette_free_space_filter": silhouette_state,
        "planar_slab_support_filter": planar_state,
        "mesh_counts": {
            "observed_mesh_vertices": int(len(observed_mesh.vertices)),
            "observed_mesh_faces": int(len(observed_mesh.faces)),
            "trellis_vertices": int(len(trellis.vertices)),
            "trellis_faces": int(len(trellis.faces)),
            "completed_vertices": int(len(completed.vertices)),
            "completed_faces": int(len(completed.faces)),
        },
    }
    (args.output_dir / "v18_compact_rigid_trellis_completion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
