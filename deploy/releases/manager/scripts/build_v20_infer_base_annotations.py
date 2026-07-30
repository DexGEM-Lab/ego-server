#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, load_mask, write_json


def intrinsics_from_json(path: Path | None, width: int, height: int) -> tuple[np.ndarray, str]:
    if path is not None:
        payload = load_json(path)
        intr = payload.get("intrinsics") if isinstance(payload, dict) else None
        if isinstance(intr, dict):
            fx = float(intr["fx"])
            fy = float(intr["fy"])
            cx = float(intr["ppx"])
            cy = float(intr["ppy"])
            source_width = float(intr.get("width") or payload.get("width") or width)
            source_height = float(intr.get("height") or payload.get("height") or height)
            if source_width > 0 and source_height > 0:
                sx = float(width) / source_width
                sy = float(height) / source_height
                fx *= sx
                fy *= sy
                cx *= sx
                cy *= sy
            return np.asarray([fx, fy, cx, cy], dtype=float), str(path)
    # Weak fallback: only supports rough visualization, not metric geometry.
    focal = max(width, height) * 0.9
    return np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=float), "weak_fallback_from_image_size_not_metric_calibration"


def object_rows_from_plan(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    ensure_no_gt_in_prediction(payload, "object_plan")
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"object_plan_has_no_objects: {path}")
    return [row for row in rows if isinstance(row, dict)]


def load_tracks(root: Path, objects: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id") or obj.get("target_object_id") or obj.get("model_object_id"))
        track_id = str(obj.get("track_id") or object_id).replace(":", "_")
        path = root / track_id / "sam2" / "sam2_track.json"
        if not path.exists():
            raise ContractError(f"missing_sam2_track_for_target: {object_id}:{path}")
        payload = load_json(path)
        out[object_id] = {int(k): v for k, v in payload.items() if isinstance(v, dict)}
    return out


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def center_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def unproject(mask: np.ndarray, depth: np.ndarray | None, intr: np.ndarray, max_points: int, pseudo_depth_m: float) -> tuple[np.ndarray, str]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), "empty_mask"
    if depth is not None:
        if depth.shape[:2] != mask.shape[:2]:
            depth = cv2.resize(depth.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        z_all = depth[ys, xs].astype(np.float32)
        valid = np.isfinite(z_all) & (z_all > 0.05) & (z_all < 10.0)
        xs = xs[valid]
        ys = ys[valid]
        z = z_all[valid]
        source = "native_depth_mask_unprojection"
    else:
        z = np.full(xs.shape, float(pseudo_depth_m), dtype=np.float32)
        source = "pseudo_depth_visualization_plane_not_metric_geometry"
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), "mask_depth_has_no_valid_depth"
    if xs.size > max_points:
        rng = np.random.default_rng(1337 + xs.size)
        keep = rng.choice(xs.size, size=max_points, replace=False)
        xs = xs[keep]
        ys = ys[keep]
        z = z[keep]
    fx, fy, cx, cy = intr.astype(float)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, source


def depth_frames_from_avi(path: Path | None, expected: int, output_dir: Path, depth_scale_m: float) -> dict[int, Path]:
    if path is None:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_depth_avi: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    idx = 0
    try:
        while idx < expected:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            depth = gray.astype(np.float32) * float(depth_scale_m)
            npy = output_dir / f"{idx:06d}.npy"
            np.save(npy, depth)
            out[idx] = npy
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        raise ContractError(f"depth_avi_no_frames: {path}")
    return out


def read_depth_npy(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path).astype(np.float32)


def hand_rows_from_mano_npz(path: Path | None, frame_idx: int, intr: np.ndarray) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    blob = np.load(path, allow_pickle=True)
    frames = np.asarray(blob["frame_idx"], dtype=int)
    rows = np.where(frames == frame_idx)[0]
    out = []
    sides = np.asarray(blob["side"])
    betas = np.asarray(blob["betas"], dtype=float) if "betas" in blob.files else None
    trans = np.asarray(blob["trans_camera_m"], dtype=float) if "trans_camera_m" in blob.files else None
    for ri in rows.tolist():
        side_raw = sides[ri]
        side = side_raw.decode("utf-8") if isinstance(side_raw, bytes) else str(side_raw)
        joints = np.asarray(blob["joints_camera_m"][ri], dtype=float)
        verts = np.asarray(blob["vertices_camera_m"][ri], dtype=float)
        if joints.shape != (21, 3) or verts.ndim != 2 or verts.shape[1] != 3:
            continue
        sample = np.linspace(0, verts.shape[0] - 1, min(256, verts.shape[0])).round().astype(int)
        verts_sample = verts[sample]
        uv = np.full((21, 2), np.nan, dtype=float)
        valid = np.isfinite(joints).all(axis=1) & (joints[:, 2] > 1e-6)
        fx, fy, cx, cy = intr.astype(float)
        uv[valid, 0] = fx * joints[valid, 0] / joints[valid, 2] + cx
        uv[valid, 1] = fy * joints[valid, 1] / joints[valid, 2] + cy
        out.append({
            "hand_id": f"hand:{side}",
            "hand_side": side,
            "side": side,
            "hand_geometry_source": "prediction_side_hawor_mano_npz",
            "metric_mano_state": {
                "source": "prediction_side_hawor_mano_npz",
                "hand_side": side,
                "coordinate_status": "camera_space_prediction_side_mano_uncalibrated_scale_if_no_metric_refit",
                "joints_current_v18_camera_m": joints.astype(float).tolist(),
                "joints_current_v18_world_m": joints.astype(float).tolist(),
                "joints_2d_px": uv.astype(float).tolist(),
                "vertices_camera_sample_m": verts_sample.astype(float).tolist(),
                "vertices_world_sample_m": verts_sample.astype(float).tolist(),
                "vertices_current_v18_camera_m": verts_sample.astype(float).tolist(),
                "vertices_current_v18_world_m": verts_sample.astype(float).tolist(),
                "betas": betas[ri].astype(float).tolist() if betas is not None and betas.ndim == 2 and ri < betas.shape[0] else None,
                "mano_betas": betas[ri].astype(float).tolist() if betas is not None and betas.ndim == 2 and ri < betas.shape[0] else None,
                "trans_camera_m": trans[ri].astype(float).tolist() if trans is not None and trans.ndim == 2 and ri < trans.shape[0] else None,
                "current_v18_camera_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                "support_state": "prediction_side_hawor_observation_uncertain",
            },
        })
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_manifest = load_json(args.raw_frame_manifest)
    frames = raw_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("raw_frame_manifest_has_no_frames")
    first = cv2.imread(str(frames[0]["rgb"]), cv2.IMREAD_COLOR)
    if first is None:
        raise ContractError(f"could_not_read_first_frame: {frames[0]['rgb']}")
    height, width = first.shape[:2]
    intr, intr_source = intrinsics_from_json(args.intrinsics_json, width, height)
    objects = object_rows_from_plan(args.object_plan)
    tracks = load_tracks(args.sam2_root, objects)
    depth_paths = depth_frames_from_avi(args.depth_avi, len(frames), args.output_annotations.parent / "depth_npy", float(args.depth_scale_m))
    depth_available = bool(depth_paths)
    out_frames = []
    visible_rows = 0
    missing_rows = []
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", frame.get("index")))
        rgb_path = str(frame["rgb"])
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ContractError(f"could_not_read_rgb: {rgb_path}")
        depth = read_depth_npy(depth_paths.get(frame_idx))
        frame_objects = []
        for obj in objects:
            object_id = str(obj.get("object_id") or obj.get("target_object_id") or obj.get("model_object_id"))
            track = tracks.get(object_id, {}).get(frame_idx)
            mask = load_mask(Path(track["mask_path"]), tuple(image.shape[:2])) if track and track.get("mask_path") else None
            visible = bool(track and track.get("visible") and mask is not None and np.any(mask))
            if not visible:
                missing_rows.append({"frame_idx": frame_idx, "object_id": object_id})
            pts, source = unproject(mask, depth, intr, int(args.max_points_per_object), float(args.pseudo_depth_m)) if visible and mask is not None else (np.zeros((0, 3), dtype=np.float32), "missing_visible_mask")
            centroid = pts.mean(axis=0) if pts.size else np.asarray([math.nan, math.nan, math.nan], dtype=np.float32)
            bbox = bbox_from_mask(mask) if visible and mask is not None else None
            center = center_from_mask(mask) if visible and mask is not None else None
            if visible:
                visible_rows += 1
            frame_objects.append({
                "object_id": object_id,
                "track_id": str(obj.get("track_id") or object_id),
                "object_name": obj.get("description") or object_id,
                "visible": visible,
                "bbox_xyxy": bbox,
                "center_xy": center,
                "object_pose_source": "visible_surface_centroid_uncertain_not_rigid_pose",
                "t_camera_object_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
                "R_camera_object": None,
                "visible_geometry_candidate": {
                    "source": source,
                    "object_id": object_id,
                    "mask_path": track.get("mask_path") if track else None,
                    "points_world_sample_m": pts.astype(float).tolist(),
                    "points_camera_sample_m": pts.astype(float).tolist(),
                    "camera_vertices_sample_m": pts.astype(float).tolist(),
                    "world_vertices_sample_m": pts.astype(float).tolist(),
                    "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                    "centroid_world_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
                    "centroid_camera_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
                    "surface_point_count": int(pts.shape[0]),
                    "bbox_xyxy": bbox,
                    "center_xy": center,
                    "coordinate_status": "camera_as_world_for_v20_infer_visualization",
                    "claim_scope": "visible surface from SAM2 mask and native/weak depth; not complete object mesh or rigid pose",
                },
                "reconstructed_geometry_pose": {
                    "renderable_pose_geometry": bool(visible),
                    "state": "visible_surface_only_uncertain_pose" if visible else "missing_visible_surface",
                    "translation_camera_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
                    "translation_world_from_object_m": centroid.astype(float).tolist() if np.isfinite(centroid).all() else None,
                    "rotation_camera_from_object_matrix": None,
                    "claim_scope": "not a completed object mesh reconstruction; V20 infer approximation until geometry completion/pose graph is connected",
                },
            })
        hands = hand_rows_from_mano_npz(args.mano_npz, frame_idx, intr)
        out_frames.append({
            "frame_idx": frame_idx,
            "source_frame_idx": frame_idx,
            "rgb_path": rgb_path,
            "source_image_size": [int(width), int(height)],
            "source_size": [int(width), int(height)],
            "camera": {
                "T_world_camera_metric": np.eye(4, dtype=float).tolist(),
                "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                "intrinsics_source": intr_source,
                "coordinate_status": "camera_as_world_for_v20_infer_visualization",
            },
            "hands": hands,
            "objects": frame_objects,
            "factor_graph_solution": {
                "variables": {
                    "object_se3": [
                        {
                            "variable_id": f"object_se3::{obj['object_id']}",
                            "translation_world_from_object_m": obj["visible_geometry_candidate"].get("centroid_world_m"),
                            "rotation_world_from_object_rotvec": [0.0, 0.0, 0.0],
                            "source": obj["visible_geometry_candidate"].get("source"),
                            "optimization_status": "not_optimized_visible_surface_centroid_only",
                        }
                        for obj in frame_objects
                    ],
                    "hand_wrist": [],
                    "contact_switch": [],
                    "occlusion_owner": [],
                },
                "solution": {"status": "v20_infer_base_state_before_full_branch_optimization"},
            },
        })
    annotations = {
        "schema": "v20_infer_base_annotations.v0",
        "case_id": args.case_id,
        "mode": "v20_infer",
        "claim_scope": "Prediction-side V20 infer base annotations from object-plan SAM2 masks, available depth, and optional HaWoR MANO. This is renderable approximate state, not final optimized physical truth.",
        "fps": float(raw_manifest.get("video", {}).get("fps", args.fps)),
        "frame_count": len(out_frames),
        "duration_s": len(out_frames) / float(raw_manifest.get("video", {}).get("fps", args.fps)),
        "image_width": width,
        "image_height": height,
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "object_plan": str(args.object_plan),
        "depth_available": depth_available,
        "depth_semantics": args.depth_semantics if depth_available else "pseudo_depth_or_no_depth",
        "target_selection_policy": "Only object_plan targets are rendered; background objects are not targets.",
        "frames": out_frames,
        "module_counts": {
            "frames": len(out_frames),
            "objects_per_frame": len(objects),
            "visible_object_rows": int(visible_rows),
            "missing_mask_rows": int(len(missing_rows)),
            "hand_rows": int(sum(len(f["hands"]) for f in out_frames)),
        },
        "limitations": {
            "object_pose": "visible surface centroid only until completion/pose graph is connected",
            "hand_state": "optional HaWoR MANO if supplied; absent otherwise",
            "depth": args.depth_semantics if depth_available else "weak pseudo-depth visualization, not metric geometry",
        },
    }
    ensure_no_gt_in_prediction(annotations, "v20_infer_base_annotations")
    write_json(args.output_annotations, annotations)
    report = {
        "status": "ok",
        "method": "build_v20_infer_base_annotations",
        "output_annotations": str(args.output_annotations),
        "frame_count": len(out_frames),
        "visible_object_rows": int(visible_rows),
        "missing_mask_rows": int(len(missing_rows)),
        "hand_rows": int(sum(len(f["hands"]) for f in out_frames)),
        "depth_available": depth_available,
        "intrinsics_source": intr_source,
        "claim_scope": annotations["claim_scope"],
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build approximate renderable V20 infer base annotations from target SAM2 masks and optional depth/MANO.")
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path)
    parser.add_argument("--depth-avi", type=Path)
    parser.add_argument("--depth-scale-m", type=float, default=0.02)
    parser.add_argument("--depth-semantics", default="uint8_depth_avi_scaled_to_m_approximate")
    parser.add_argument("--mano-npz", type=Path)
    parser.add_argument("--pseudo-depth-m", type=float, default=1.0)
    parser.add_argument("--max-points-per-object", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
