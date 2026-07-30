#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, safe_id, write_json


def pca_basis(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def robust_extent(points: np.ndarray) -> np.ndarray:
    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    return np.maximum(hi - lo, 1.0e-4)


def collect_points(annotations: dict[str, Any], object_id: str, max_frames: int, max_points_per_frame: int) -> np.ndarray:
    chunks = []
    for frame in annotations.get("frames", []) if isinstance(annotations, dict) else []:
        if len(chunks) >= max_frames:
            break
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if str(obj.get("object_id")) != object_id:
                continue
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            pts = np.asarray(geom.get("camera_vertices_sample_m") or geom.get("points_camera_sample_m") or [], dtype=float)
            pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 and pts.shape[1] == 3 else np.zeros((0, 3), dtype=float)
            if len(pts) > max_points_per_frame:
                pts = pts[np.linspace(0, len(pts) - 1, max_points_per_frame, dtype=int)]
            if len(pts) >= 3:
                chunks.append(pts)
            break
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(chunks)


def fit_one(obj: dict[str, Any], annotations: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    object_id = str(obj["object_id"])
    mesh_path = Path(obj["mesh_path"])
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ContractError(f"invalid_public_cad_mesh: {mesh_path}")
    visible = collect_points(annotations, object_id, int(args.max_frames), int(args.max_points_per_frame))
    if visible.shape[0] < int(args.min_visible_points):
        raise ContractError(f"insufficient_visible_depth_points_for_cad_fit: {object_id} count={visible.shape[0]}")
    verts = np.asarray(mesh.vertices, dtype=float)
    cad_center = verts.mean(axis=0)
    vis_center = visible.mean(axis=0)
    cad_basis = pca_basis(verts)
    vis_basis = pca_basis(visible)
    cad_local = (verts - cad_center) @ cad_basis
    vis_local = (visible - vis_center) @ vis_basis
    cad_extent = robust_extent(cad_local)
    vis_extent = robust_extent(vis_local)
    scale = float(np.median(vis_extent / np.maximum(cad_extent, 1.0e-6)))
    scale = float(np.clip(scale, args.min_scale, args.max_scale))
    transformed = ((verts - cad_center) @ cad_basis @ vis_basis.T) * scale + vis_center
    out_mesh = trimesh.Trimesh(vertices=transformed, faces=np.asarray(mesh.faces), process=False)
    object_dir = args.output_dir / safe_id(object_id)
    object_dir.mkdir(parents=True, exist_ok=True)
    out_path = object_dir / "cad_visible_depth_fit.ply"
    out_mesh.export(out_path)
    T = np.eye(4, dtype=float)
    T[:3, :3] = cad_basis @ vis_basis.T * scale
    T[:3, 3] = vis_center - (cad_center @ cad_basis @ vis_basis.T) * scale
    report = {
        "schema": "v20_cad_visible_depth_fit.v0",
        "object_id": object_id,
        "object_name": obj.get("object_name"),
        "source_mesh": str(mesh_path),
        "output_mesh": str(out_path),
        "visible_point_count": int(visible.shape[0]),
        "fit_method": "pca_extent_scale_centroid_from_prediction_side_visible_depth_surfels",
        "scale_model_to_camera": scale,
        "cad_extent_pca": cad_extent.astype(float).tolist(),
        "visible_extent_pca_m": vis_extent.astype(float).tolist(),
        "T_camera_model_4x4": T.astype(float).tolist(),
        "eval_refs_loaded": False,
        "claim_scope": "Initial metric CAD pose/scale adaptation from SAM2+native-depth visible surfels; validation/promotion remains separate.",
    }
    write_json(object_dir / "cad_visible_depth_fit_report.json", report)
    return report




def target_objects(dataset_manifest: dict[str, Any], object_plan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_payload = load_json(object_plan_path)
    plan = plan_payload.get("plan") if isinstance(plan_payload, dict) and isinstance(plan_payload.get("plan"), dict) else plan_payload
    plan_rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(plan_rows, list) or not plan_rows:
        raise ContractError(f"v20_object_plan_has_no_targets: {object_plan_path}")
    public_rows = dataset_manifest.get("public_object_model_roster") or dataset_manifest.get("objects")
    if not isinstance(public_rows, list) or not public_rows:
        raise ContractError("dataset_manifest_has_no_public_object_model_roster")
    public_by_id = {str(row.get("object_id")): row for row in public_rows if isinstance(row, dict)}
    out = []
    for row in plan_rows:
        if not isinstance(row, dict):
            continue
        object_id = str(row.get("object_id") or row.get("target_object_id") or row.get("model_object_id") or "")
        if object_id not in public_by_id:
            raise ContractError(f"object_plan_target_not_in_public_model_roster: {object_id}")
        merged = dict(public_by_id[object_id])
        merged["track_id"] = str(row.get("track_id") or object_id.replace(":", "_"))
        merged["target_selection_source"] = "object_plan_not_public_model_roster"
        out.append(merged)
    if not out:
        raise ContractError(f"v20_object_plan_produced_no_fit_targets: {object_plan_path}")
    return out, plan_payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    dataset_manifest = load_json(args.dataset_manifest)
    ensure_no_gt_in_prediction(annotations, "annotations")
    ensure_no_gt_in_prediction(dataset_manifest, "dataset_manifest")
    objects, object_plan_payload = target_objects(dataset_manifest, args.object_plan)
    ensure_no_gt_in_prediction(object_plan_payload, "object_plan")
    reports = [fit_one(obj, annotations, args) for obj in objects]
    summary = {
        "status": "ok",
        "method": "fit_v20_cad_mesh_to_visible_depth",
        "annotations": str(args.annotations),
        "dataset_manifest": str(args.dataset_manifest),
        "object_plan": str(args.object_plan),
        "target_selection_policy": "Only object_plan targets are fitted. Dataset public object roster is a model library, not a target list.",
        "output_dir": str(args.output_dir),
        "object_count": len(reports),
        "objects": reports,
        "eval_refs_loaded": False,
    }
    write_json(args.output_report, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit public CAD meshes to prediction-side visible depth surfels.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--max-points-per-frame", type=int, default=256)
    parser.add_argument("--min-visible-points", type=int, default=128)
    parser.add_argument("--min-scale", type=float, default=1.0e-5)
    parser.add_argument("--max-scale", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
