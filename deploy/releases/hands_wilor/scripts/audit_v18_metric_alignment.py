#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SIDES = ("left", "right")
HAWOR_FULL_INTRINSICS = np.asarray([2304.0, 2304.0, 960.0, 540.0], dtype=np.float64)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        fx, fy, cx, cy = [float(v) for v in intrinsics.tolist()]
        uv[valid, 0] = fx * points[valid, 0] / points[valid, 2] + cx
        uv[valid, 1] = fy * points[valid, 1] / points[valid, 2] + cy
    return uv, valid


def bbox_fraction(uv: np.ndarray, valid: np.ndarray, bbox: list[Any] | None) -> float | None:
    if not bbox or len(bbox) != 4 or not np.any(valid):
        return None
    x1, y1, x2, y2 = [finite_float(v) for v in bbox]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    pts = uv[valid]
    inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
    return float(np.mean(inside))


def mask_fraction(uv: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    h, w = mask.shape[:2]
    pts = uv[valid]
    in_image = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    out = {
        "projected_count": int(uv.shape[0]),
        "valid_depth_count": int(np.sum(valid)),
        "inside_image_count": int(np.sum(in_image)),
        "inside_image_fraction": float(np.mean(in_image)) if pts.shape[0] else 0.0,
        "inside_mask_count": 0,
        "inside_mask_fraction": 0.0,
    }
    if not np.any(in_image):
        return out
    pix = np.rint(pts[in_image]).astype(np.int64)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    inside_mask = mask[pix[:, 1], pix[:, 0]] > 0
    out["inside_mask_count"] = int(np.sum(inside_mask))
    out["inside_mask_fraction"] = float(np.mean(inside_mask))
    return out


def transform_world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(T_world_camera)
    hom = np.c_[points_world.astype(np.float64), np.ones(points_world.shape[0], dtype=np.float64)]
    return (inv @ hom.T).T[:, :3]


def nearest_pair(a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3 or a.shape[0] == 0 or b.shape[0] == 0:
        return None
    best_i = -1
    best_j = -1
    best_d = float("inf")
    chunk = 512
    for start in range(0, a.shape[0], chunk):
        diff = a[start:start + chunk, None, :] - b[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        flat = int(np.argmin(dist2))
        d = float(math.sqrt(float(dist2.reshape(-1)[flat])))
        if d < best_d:
            local_i, j = divmod(flat, b.shape[0])
            best_i = start + local_i
            best_j = j
            best_d = d
    delta = a[best_i] - b[best_j]
    return {
        "distance_m": float(best_d),
        "hand_point_world_m": [float(v) for v in a[best_i].tolist()],
        "object_point_world_m": [float(v) for v in b[best_j].tolist()],
        "delta_hand_minus_object_m": [float(v) for v in delta.tolist()],
    }


def object_track_dir_name(object_id: str) -> str:
    return object_id.split(":", 1)[-1]


def load_object_dataset_manifest(root: Path, case: str, object_id: str) -> dict[int, dict[str, Any]]:
    path = root / case / object_track_dir_name(object_id) / "manifest.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    rows = payload.get("frames") if isinstance(payload, dict) else []
    return {int(row["frame_idx"]): row for row in rows if isinstance(row, dict) and "frame_idx" in row}


def load_visible_surface_index(path: Path) -> dict[tuple[int, str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    frames = np.asarray(z["frame_idx"])
    object_ids = np.asarray(z["object_id"])
    offsets = np.asarray(z["vertex_offsets"])
    vertices = np.asarray(z["vertices"], dtype=np.float64)
    out: dict[tuple[int, str], np.ndarray] = {}
    for i in range(frames.shape[0]):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        out[(int(frames[i]), str(object_ids[i]))] = vertices[start:end]
    return out


def load_bridge_index(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    frames = np.asarray(z["frame_idx"])
    sides = np.asarray(z["side_labels"])
    cam = np.asarray(z["vertices_hawor_camera_m"], dtype=np.float64)
    world = np.asarray(z["vertices_current_v18_world_from_hawor_camera_local_m"], dtype=np.float64)
    transforms = np.asarray(z["T_world_camera_metric_current_v18"], dtype=np.float64)
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for i in range(frames.shape[0]):
        out[(int(frames[i]), str(sides[i]))] = {"camera": cam[i], "world": world[i], "T": transforms[i]}
    return out


def scaled_bbox(bbox: Any, sx: float, sy: float) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [finite_float(v) for v in bbox]
    if not all(math.isfinite(v) for v in vals):
        return None
    return [vals[0] * sx, vals[1] * sy, vals[2] * sx, vals[3] * sy]


def audit_row(
    frame: dict[str, Any],
    switch: dict[str, Any],
    object_vertices_world: np.ndarray,
    bridge_row: dict[str, np.ndarray],
    object_dataset_row: dict[str, Any] | None,
) -> dict[str, Any]:
    frame_idx = int(frame["frame_idx"])
    side = str(switch.get("hand_side"))
    object_id = str(switch.get("object_id"))
    hands = {str(hand.get("hand_side")): hand for hand in frame.get("hands", []) if isinstance(hand, dict)}
    objects = {str(obj.get("object_id")): obj for obj in frame.get("objects", []) if isinstance(obj, dict)}
    hand = hands.get(side, {})
    obj = objects.get(object_id, {})
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        T = bridge_row["T"]
    intr = None
    mask = None
    mask_path = None
    source_rgb = None
    if object_dataset_row:
        raw_intr = object_dataset_row.get("intrinsics_fx_fy_cx_cy")
        if isinstance(raw_intr, list) and len(raw_intr) == 4:
            intr = np.asarray([float(v) for v in raw_intr], dtype=np.float64)
        mask_path = object_dataset_row.get("source_mask") or object_dataset_row.get("mask")
        source_rgb = object_dataset_row.get("source_rgb")
        if isinstance(mask_path, str) and Path(mask_path).exists():
            mask = np.asarray(Image.open(mask_path))
    if intr is None:
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        raw_intr = geom.get("source_depth_intrinsics_fx_fy_cx_cy")
        if isinstance(raw_intr, list) and len(raw_intr) == 4:
            intr = np.asarray([float(v) for v in raw_intr], dtype=np.float64)
    result: dict[str, Any] = {
        "frame_idx": frame_idx,
        "hand_side": side,
        "object_id": object_id,
        "source_rgb": source_rgb,
        "mask_path": mask_path,
        "annotation_hand_bbox_xyxy_fullres": hand.get("bbox_xyxy"),
        "annotation_object_bbox_xyxy_fullres": obj.get("bbox_xyxy"),
        "switch_effective_metric_contact_distance_m": switch.get("effective_metric_contact_distance_m"),
        "switch_min_box_coverage": switch.get("min_box_coverage"),
        "switch_image_iou": switch.get("image_iou"),
        "switch_mesh_contact_support_score": switch.get("mesh_contact_support_score"),
        "switch_episode_role": switch.get("manipulation_contact_episode_frame_role"),
        "depth_intrinsics_fx_fy_cx_cy": [float(v) for v in intr.tolist()] if intr is not None else None,
        "mask_shape_hw": [int(mask.shape[0]), int(mask.shape[1])] if mask is not None else None,
    }
    hand_world = np.asarray(bridge_row["world"], dtype=np.float64)
    hand_camera = np.asarray(bridge_row["camera"], dtype=np.float64)
    result["nearest_hand_object_world"] = nearest_pair(hand_world, object_vertices_world)
    if intr is not None and mask is not None:
        object_camera = transform_world_to_camera(object_vertices_world, T)
        obj_uv_depth, obj_valid_depth = project(object_camera, intr)
        hand_uv_depth, hand_valid_depth = project(hand_camera, intr)
        sx = mask.shape[1] / 1920.0
        sy = mask.shape[0] / 1080.0
        result["object_world_vertices_projected_with_depth_intrinsics_to_object_mask"] = mask_fraction(obj_uv_depth, obj_valid_depth, mask)
        result["hand_hawor_camera_vertices_projected_with_depth_intrinsics_to_object_mask"] = mask_fraction(hand_uv_depth, hand_valid_depth, mask)
        result["hand_hawor_camera_vertices_projected_with_depth_intrinsics_inside_scaled_hand_bbox_fraction"] = bbox_fraction(hand_uv_depth, hand_valid_depth, scaled_bbox(hand.get("bbox_xyxy"), sx, sy))
        result["object_world_vertices_projected_with_depth_intrinsics_inside_scaled_object_bbox_fraction"] = bbox_fraction(obj_uv_depth, obj_valid_depth, scaled_bbox(obj.get("bbox_xyxy"), sx, sy))
    hand_uv_hawor_full, hand_valid_hawor_full = project(hand_camera, HAWOR_FULL_INTRINSICS)
    result["hand_hawor_camera_vertices_projected_with_hawor_full_intrinsics_inside_fullres_hand_bbox_fraction"] = bbox_fraction(hand_uv_hawor_full, hand_valid_hawor_full, hand.get("bbox_xyxy"))
    result["hand_hawor_camera_projection_fullres_bbox_xyxy"] = projected_bbox(hand_uv_hawor_full, hand_valid_hawor_full)
    return result


def projected_bbox(uv: np.ndarray, valid: np.ndarray) -> list[float] | None:
    pts = uv[valid]
    if pts.shape[0] == 0:
        return None
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return None
    return [float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))]


def select_switches(case: str, ann: dict[str, Any], fixed_task5_frames: set[int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for frame in ann.get("frames", []):
        frame_idx = int(frame.get("frame_idx", -1))
        switches = ((frame.get("factor_graph_solution") or {}).get("variables") or {}).get("contact_switch", [])
        for switch in switches:
            if not isinstance(switch, dict):
                continue
            distance = finite_float(switch.get("effective_metric_contact_distance_m"), float("nan"))
            coverage = finite_float(switch.get("min_box_coverage"), 0.0)
            if case == "task5_tomato_960" and frame_idx in fixed_task5_frames and switch.get("object_id") == "object:obj_tomato":
                selected.append({"frame_idx": frame_idx, "switch": switch, "reason": "fixed_task5_tomato_audit_frame"})
            elif case == "trash_1050" and math.isfinite(distance) and distance >= 0.20 and coverage >= 0.80:
                candidates.append((distance, frame_idx, switch))
    if case == "trash_1050":
        seen: set[tuple[int, str, str]] = set()
        for _, frame_idx, switch in sorted(candidates, key=lambda item: (-item[0], item[1])):
            key = (frame_idx, str(switch.get("hand_side")), str(switch.get("object_id")))
            if key in seen:
                continue
            seen.add(key)
            selected.append({"frame_idx": frame_idx, "switch": switch, "reason": "trash_high_2d_overlap_large_3d_gap"})
            if len(selected) >= 8:
                break
    return selected


def audit_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    ann_path = args.full_pipeline_root / case / "annotations_v18_full.json"
    ann = load_json(ann_path)
    frames = {int(frame["frame_idx"]): frame for frame in ann.get("frames", []) if isinstance(frame, dict) and "frame_idx" in frame}
    surface_index = load_visible_surface_index(args.visible_geometry_root / case / "v18_visible_surfaces_world.npz")
    bridge_index = load_bridge_index(args.hawor_bridge_root / case / "hawor_bridge_candidates_current_v18_camera_local.npz")
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    object_manifests: dict[str, dict[int, dict[str, Any]]] = {}
    selected = select_switches(case, ann, {296, 480, 780, 926})
    for item in selected:
        frame_idx = int(item["frame_idx"])
        switch = item["switch"]
        side = str(switch.get("hand_side"))
        object_id = str(switch.get("object_id"))
        frame = frames.get(frame_idx)
        object_vertices = surface_index.get((frame_idx, object_id))
        bridge_row = bridge_index.get((frame_idx, side))
        if object_id not in object_manifests:
            object_manifests[object_id] = load_object_dataset_manifest(args.object_dataset_root, case, object_id)
        dataset_row = object_manifests[object_id].get(frame_idx)
        blockers = []
        if frame is None:
            blockers.append("missing_frame")
        if object_vertices is None or object_vertices.shape[0] == 0:
            blockers.append("missing_visible_surface_vertices")
        if bridge_row is None:
            blockers.append("missing_hawor_bridge_row")
        if dataset_row is None:
            blockers.append("missing_object_dataset_mask_intrinsics_row")
        if blockers:
            missing.append({"frame_idx": frame_idx, "hand_side": side, "object_id": object_id, "reason": item["reason"], "blockers": blockers})
            continue
        row = audit_row(frame, switch, object_vertices, bridge_row, dataset_row)
        row["selection_reason"] = item["reason"]
        rows.append(row)
    return {
        "case": case,
        "annotation_path": str(ann_path),
        "row_count": len(rows),
        "missing_count": len(missing),
        "rows": rows,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_unidepth_extension/v18_visible_geometry_archive_complete_depth"))
    parser.add_argument("--hawor-bridge-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_bridge_state"))
    parser.add_argument("--object-dataset-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"))
    parser.add_argument("--output", type=Path, default=Path("/data2/ego_annotation_outputs/v18_metric_alignment_audit/v18_metric_alignment_audit_report.json"))
    args = parser.parse_args()
    cases = ["task5_tomato_960", "trash_1050"]
    reports = [audit_case(case, args) for case in cases]
    payload = {
        "status": "ok",
        "method": "audit_v18_metric_alignment",
        "claim": "Frame-local projection and nearest-distance audit for MANO/object metric alignment before contact or occlusion inference.",
        "reports": reports,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "ok", "output": str(args.output), "case_rows": {r["case"]: r["row_count"] for r in reports}, "missing": {r["case"]: r["missing_count"] for r in reports}}, indent=2))


if __name__ == "__main__":
    main()
