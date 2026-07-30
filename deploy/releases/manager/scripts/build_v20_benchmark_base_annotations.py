#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, project_points, read_depth_image_m, write_json


def intrinsics_vec(manifest: dict[str, Any]) -> np.ndarray:
    raw = manifest.get("camera_intrinsics")
    arr = np.asarray(raw, dtype=float)
    if arr.shape == (3, 3):
        return np.asarray([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=float)
    if arr.shape == (4,):
        return arr.astype(float)
    if isinstance(raw, dict):
        return np.asarray([raw["fx"], raw["fy"], raw["cx"], raw["cy"]], dtype=float)
    raise ContractError("dataset_manifest_missing_camera_intrinsics")


def is_opengl_negative_z(manifest: dict[str, Any]) -> bool:
    return "opengl_negative_z" in str(manifest.get("coordinate_convention", "")).lower()


def load_points(path: Path, max_points: int) -> np.ndarray:
    pts = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not pts:
        raise ContractError(f"object_points_empty: {path}")
    arr = np.asarray(pts, dtype=np.float32)
    if arr.shape[0] > max_points:
        idx = np.linspace(0, arr.shape[0] - 1, max_points).round().astype(int)
        arr = arr[idx]
    return arr


def read_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ContractError(f"could_not_read_prediction_mask: {path}")
    if mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def unproject(mask: np.ndarray, depth_m: np.ndarray, intr: np.ndarray, max_points: int, *, opengl_negative_z: bool) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if xs.size > max_points:
        idx = np.linspace(0, xs.size - 1, max_points).round().astype(int)
        xs = xs[idx]
        ys = ys[idx]
    depth = depth_m[ys, xs].astype(np.float32)
    fx, fy, cx, cy = intr.astype(np.float32)
    x = (xs.astype(np.float32) - cx) * depth / fx
    if opengl_negative_z:
        y = -(ys.astype(np.float32) - cy) * depth / fy
        z = -depth
    else:
        y = (ys.astype(np.float32) - cy) * depth / fy
        z = depth
    return np.stack([x, y, z], axis=1)


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def center_xy_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def summarize_depth(depth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = depth[mask & np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)]
    if values.size == 0:
        return {"count": 0, "median_m": None, "p10_m": None, "p90_m": None}
    return {
        "count": int(values.size),
        "median_m": float(np.median(values)),
        "p10_m": float(np.percentile(values, 10.0)),
        "p90_m": float(np.percentile(values, 90.0)),
    }


def load_tracks(root: Path | None, objects: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    if root is None:
        return out
    for obj in objects:
        object_id = str(obj["object_id"])
        track_id = str(obj.get("track_id") or object_id.replace(":", "_"))
        candidates = [root / track_id / "sam2" / "sam2_track.json", root / object_id.replace(":", "_") / "sam2" / "sam2_track.json"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        payload = load_json(path)
        out[object_id] = {int(k): v for k, v in payload.items() if isinstance(v, dict)}
    return out


def load_mano_npz(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise ContractError(f"missing_prediction_mano_npz: {path}")
    blob = np.load(path, allow_pickle=True)
    required = {"frame_idx", "side", "joints_camera_m", "vertices_camera_m"}
    missing = sorted(required - set(blob.files))
    if missing:
        raise ContractError(f"prediction_mano_npz_missing_keys: {path}:{missing}")
    return {key: np.asarray(blob[key]) for key in blob.files}


def load_geometry_fit_reports(root: Path | None) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in root.glob("*/cad_visible_depth_fit_report.json"):
        payload = load_json(path)
        object_id = str(payload.get("object_id"))
        if object_id and object_id != "None":
            out[object_id] = payload
    return out


def load_target_object_plan(path: Path, public_roster: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload, dict) and isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"v20_target_object_plan_has_no_objects: {path}")
    public_by_id = {str(row.get("object_id")): row for row in public_roster if isinstance(row, dict)}
    targets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        object_id = row.get("object_id") or row.get("target_object_id") or row.get("model_object_id")
        if not object_id:
            raise ContractError(f"v20_target_object_missing_public_model_object_id: {path}")
        object_id = str(object_id)
        public = public_by_id.get(object_id)
        if public is None:
            raise ContractError(f"v20_target_object_not_in_public_model_roster: {object_id}")
        physical_model = row.get("physical_model") if isinstance(row.get("physical_model"), dict) else {}
        target = dict(public)
        target.update({
            "track_id": str(row.get("track_id") or object_id.replace(":", "_")),
            "target_object_plan_path": str(path),
            "target_description": row.get("description"),
            "active_intervals": row.get("active_intervals", []),
            "physical_branch": row.get("physical_branch") or physical_model.get("primary_physical_model"),
            "target_selection_source": "object_plan_not_public_model_roster",
        })
        targets.append(target)
    if not targets:
        raise ContractError(f"v20_target_object_plan_produced_no_targets: {path}")
    return targets, payload


def hand_rows_from_mano(blob: dict[str, Any] | None, frame_idx: int, intr: np.ndarray, coordinate_status: str, convention: str) -> list[dict[str, Any]]:
    if blob is None:
        return []
    frames = np.asarray(blob["frame_idx"], dtype=int)
    sides = np.asarray(blob["side"])
    rows = np.where(frames == frame_idx)[0]
    hands = []
    for row_idx in rows.tolist():
        side_raw = sides[row_idx]
        side = side_raw.decode("utf-8") if isinstance(side_raw, bytes) else str(side_raw)
        joints = np.asarray(blob["joints_camera_m"][row_idx], dtype=float)
        verts = np.asarray(blob["vertices_camera_m"][row_idx], dtype=float)
        if joints.shape != (21, 3) or verts.ndim != 2 or verts.shape[1] != 3:
            continue
        sample_idx = np.linspace(0, verts.shape[0] - 1, min(256, verts.shape[0])).round().astype(int)
        verts_sample = verts[sample_idx]
        uv = project_points(joints, intr, convention)
        hands.append({
            "hand_id": f"hand:{side}",
            "hand_side": side,
            "side": side,
            "hand_geometry_source": "prediction_side_metric_mano_npz",
            "metric_mano_state": {
                "source": "prediction_side_metric_mano_npz",
                "case_frame_idx": int(frame_idx),
                "hand_side": side,
                "coordinate_status": coordinate_status,
                "joint_3d_camera_m": joints.astype(float).tolist(),
                "joints_current_v18_camera_m": joints.astype(float).tolist(),
                "joints_current_v18_world_m": joints.astype(float).tolist(),
                "joints_2d_px": uv.astype(float).tolist(),
                "vertices_camera_sample_m": verts_sample.astype(float).tolist(),
                "vertices_world_sample_m": verts_sample.astype(float).tolist(),
                "vertices_current_v18_world_m": verts_sample.astype(float).tolist(),
                "current_v18_camera_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                "support_state": "prediction_side_mano_observation",
                "physical_factor_weight": 1.0,
                "physical_factor_role": "metric_mano_state_from_prediction_side_model_output",
            },
        })
    return hands


def object_row(obj: dict[str, Any], frame_idx: int, track: dict[str, Any] | None, mask: np.ndarray | None, depth_m: np.ndarray, intr: np.ndarray, points: np.ndarray, geometry_fit: dict[str, Any] | None, args: argparse.Namespace, coordinate_status: str, opengl_negative_z: bool) -> dict[str, Any]:
    object_id = str(obj["object_id"])
    visible = bool(track and track.get("visible") and mask is not None and np.any(mask))
    if args.require_masks and not visible:
        raise ContractError(f"missing_visible_prediction_mask_for_object_frame: {object_id}:{frame_idx}")
    surface = unproject(mask, depth_m, intr, int(args.max_surface_points), opengl_negative_z=opengl_negative_z) if visible and mask is not None else np.zeros((0, 3), dtype=np.float32)
    centroid = surface.mean(axis=0) if surface.size else np.asarray([math.nan, math.nan, math.nan], dtype=np.float32)
    bbox = bbox_from_mask(mask) if visible and mask is not None else None
    center_xy = center_xy_from_mask(mask) if visible and mask is not None else None
    depth_summary = summarize_depth(depth_m, mask) if visible and mask is not None else {"count": 0, "median_m": None, "p10_m": None, "p90_m": None}
    geom = {
        "source": "prediction_mask_native_rgbd_unprojection" if visible else "missing_prediction_visible_surface",
        "object_id": object_id,
        "mask_path": track.get("mask_path") if track else None,
        "points_world_sample_m": surface.astype(float).tolist(),
        "points_camera_sample_m": surface.astype(float).tolist(),
        "camera_vertices_sample_m": surface.astype(float).tolist(),
        "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
        "centroid_world_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
        "centroid_camera_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
        "surface_point_count": int(surface.shape[0]),
        "depth_summary": depth_summary,
        "bbox_xyxy": bbox,
        "center_xy": center_xy,
        "coordinate_status": coordinate_status,
        "claim_scope": "visible object surface from prediction-side mask and native depth; not complete object pose by itself",
    }
    fit_T = np.asarray(geometry_fit.get("T_camera_model_4x4") if isinstance(geometry_fit, dict) else [], dtype=float)
    R_camera_object = fit_T[:3, :3].astype(float).tolist() if fit_T.shape == (4, 4) else None
    t_camera_object = fit_T[:3, 3].astype(float).tolist() if fit_T.shape == (4, 4) else (centroid.astype(float).tolist() if np.isfinite(centroid).all() else None)
    return {
        "object_id": object_id,
        "track_id": str(obj.get("track_id") or object_id.replace(":", "_")),
        "object_name": obj.get("object_name"),
        "R_camera_object": R_camera_object,
        "t_camera_object_m": t_camera_object,
        "object_pose_source": "v20_cad_visible_depth_fit" if R_camera_object is not None else "visible_depth_centroid_no_rotation",
        "dataset_label_id_public_roster": obj.get("dataset_label_id"),
        "mesh_path": obj.get("mesh_path"),
        "points_path": obj.get("points_path"),
        "visible": visible,
        "mask_path": track.get("mask_path") if track else None,
        "bbox_xyxy": bbox,
        "center_xy": center_xy,
        "visible_geometry_candidate": geom,
        "reconstructed_geometry_pose": {
            "renderable_pose_geometry": bool(visible),
            "state": "visible_surface_only_not_complete_pose" if visible else "missing_visible_surface",
            "mesh_path": obj.get("mesh_path"),
            "canonical_points_sample_m": points.astype(float).tolist(),
            "translation_world_from_object_m": t_camera_object,
            "translation_camera_m": t_camera_object,
            "rotation_camera_from_object_matrix": R_camera_object,
            "rotation_world_from_object_rotvec": [0.0, 0.0, 0.0],
            "claim_scope": "CAD mesh is public object model; pose estimate is weak visible-depth centroid prior unless promoted by V20 geometry validation",
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.dataset_manifest)
    input_manifest = load_json(args.input_manifest) if args.input_manifest else {}
    ensure_no_gt_in_prediction(manifest, "dataset_manifest")
    public_roster = manifest.get("public_object_model_roster") or manifest.get("objects")
    if not isinstance(public_roster, list) or not public_roster:
        raise ContractError("dataset_manifest_has_no_public_objects")
    objects, object_plan_payload = load_target_object_plan(args.object_plan, public_roster)
    ensure_no_gt_in_prediction(object_plan_payload, "object_plan")
    intr = intrinsics_vec(manifest)
    dataset = str(manifest.get("dataset") or "unknown")
    coordinate_status = f"{dataset}_camera_as_world_identity_for_benchmark_prediction_{manifest.get('coordinate_convention', 'unknown_coordinate_convention')}"
    opengl_negative_z = is_opengl_negative_z(manifest)
    projection_convention = "opengl_negative_z" if opengl_negative_z else "opencv_positive_z"
    points_by_object = {str(obj["object_id"]): load_points(Path(obj["points_path"]), int(args.max_model_points)) for obj in objects}
    tracks = load_tracks(args.mask_root, objects)
    geometry_fit_reports = load_geometry_fit_reports(args.geometry_fit_root)
    mano = load_mano_npz(args.mano_npz)
    frames_out = []
    missing_masks = []
    hand_count = 0
    visible_count = 0
    for index, row in enumerate(manifest.get("frames", [])):
        frame_idx = int(row.get("frame_index", row.get("frame_idx", index)))
        rgb_path = Path(str(row["rgb_path"]))
        depth_path = Path(str(row["depth_path"]))
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ContractError(f"could_not_read_rgb: {rgb_path}")
        depth_m = read_depth_image_m(depth_path, manifest.get("depth_semantics"))
        if depth_m.shape[:2] != image.shape[:2]:
            depth_m = cv2.resize(depth_m.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        objects_out = []
        for obj in objects:
            object_id = str(obj["object_id"])
            track = tracks.get(object_id, {}).get(frame_idx)
            mask = read_mask(Path(track["mask_path"]), tuple(image.shape[:2])) if track and track.get("mask_path") else None
            if not (track and track.get("visible") and mask is not None and np.any(mask)):
                missing_masks.append({"frame_idx": frame_idx, "object_id": object_id})
            obj_row = object_row(obj, frame_idx, track, mask, depth_m, intr, points_by_object[object_id], geometry_fit_reports.get(object_id), args, coordinate_status, opengl_negative_z)
            if obj_row["visible"]:
                visible_count += 1
            objects_out.append(obj_row)
        hands = hand_rows_from_mano(mano, frame_idx, intr, coordinate_status, projection_convention)
        hand_count += len(hands)
        frames_out.append({
            "frame_idx": frame_idx,
            "source_frame_idx": frame_idx,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera": {
                "T_world_camera_metric": np.eye(4, dtype=float).tolist(),
                "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                "coordinate_status": coordinate_status,
            },
            "hands": hands,
            "objects": objects_out,
            "factor_graph_solution": {
                "variables": {
                    "hand_wrist": [
                        {"variable_id": f"hand_wrist::{hand['hand_side']}", "estimate": hand["metric_mano_state"]["joints_current_v18_world_m"][0], "source": "prediction_side_metric_mano"}
                        for hand in hands
                    ],
                    "object_se3": [
                        {"variable_id": f"object_se3::{obj['object_id']}", "translation_world_from_object_m": obj["visible_geometry_candidate"].get("centroid_world_m"), "rotation_world_from_object_rotvec": [0.0, 0.0, 0.0], "source": obj["visible_geometry_candidate"].get("source")}
                        for obj in objects_out
                    ],
                    "contact_switch": [],
                    "occlusion_owner": [],
                },
                "solution": {"status": "prediction_side_base_state_before_v20_refinement", "active_contact_hypotheses": 0, "unresolved_or_contradicted_contact_hypotheses": 0},
            },
        })
    if args.require_mano and hand_count == 0:
        raise ContractError("missing_prediction_side_metric_mano_output")
    annotations = {
        "schema": "v20_benchmark_prediction_base_annotations.v0",
        "case_id": args.case_id,
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "dataset": dataset,
        "claim_scope": "Prediction-side benchmark base annotations from public RGB-D/CAD inputs, object-plan targets, and optional model outputs only. Eval refs are withheld until evaluator.",
        "fps": float(input_manifest.get("fps", manifest.get("fps_assumed", args.fps))),
        "frame_count": len(frames_out),
        "duration_s": len(frames_out) / float(input_manifest.get("fps", manifest.get("fps_assumed", args.fps))),
        "image_width": int(manifest.get("resolution", {}).get("width", 640)),
        "image_height": int(manifest.get("resolution", {}).get("height", 480)),
        "dataset_manifest": str(args.dataset_manifest),
        "input_manifest": str(args.input_manifest) if args.input_manifest else None,
        "object_plan": str(args.object_plan),
        "target_selection_policy": "Only object-plan targets are rendered/optimized. Dataset public object roster is a model library, not a target list.",
        "coordinate_status": coordinate_status,
        "frames": frames_out,
        "module_counts": {"frames": len(frames_out), "objects_per_frame": len(objects), "visible_object_rows": int(visible_count), "hand_rows": int(hand_count), "missing_mask_rows": int(len(missing_masks))},
        "limitations": {"hand_state": "requires prediction-side MANO npz; absent if hand_rows=0", "object_pose": "weak visible-depth centroid unless later V20 geometry promotion/pose solve improves it", "eval_refs": "not loaded by this builder"},
        "object_plan_payload": object_plan_payload,
    }
    ensure_no_gt_in_prediction(annotations, "v20_benchmark_base_annotations")
    write_json(args.output_annotations, annotations)
    report = {"status": "ok", "method": "build_v20_benchmark_base_annotations", "dataset": dataset, "output_annotations": str(args.output_annotations), "frame_count": len(frames_out), "visible_object_rows": int(visible_count), "hand_rows": int(hand_count), "missing_mask_rows": int(len(missing_masks)), "missing_mask_examples": missing_masks[:20], "eval_refs_loaded": False}
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prediction-side V20 benchmark base annotations from RGB-D/CAD/model outputs only.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--mano-npz", type=Path)
    parser.add_argument("--geometry-fit-root", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-model-points", type=int, default=256)
    parser.add_argument("--max-surface-points", type=int, default=512)
    parser.add_argument("--require-masks", action="store_true")
    parser.add_argument("--require-mano", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
