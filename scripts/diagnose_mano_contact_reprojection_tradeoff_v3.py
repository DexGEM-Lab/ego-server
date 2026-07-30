#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_contact_depth_conflict_v3 import mesh_frame_vertices, summarize
from diagnose_hand_reprojection_depth_v3 import bbox_corners, project_points
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, project_world, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def required_points(hand: dict, key: str) -> np.ndarray:
    if key not in hand:
        raise RuntimeError(f"hand row is missing required field {key}")
    points = np.asarray(hand[key], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"{key} must have shape Nx3")
    if len(points) == 0:
        raise RuntimeError(f"{key} is empty")
    if not np.all(np.isfinite(points)):
        raise RuntimeError(f"{key} contains non-finite values")
    if np.any(points[:, 2] <= 0.0):
        raise RuntimeError(f"{key} contains non-positive depth")
    return points


def required_intrinsics(hand: dict) -> np.ndarray:
    if "source_intrinsics" not in hand:
        raise RuntimeError("hand row is missing source_intrinsics")
    intr = np.asarray(hand["source_intrinsics"], dtype=float)
    if intr.shape != (4,) or not np.all(np.isfinite(intr)):
        raise RuntimeError("source_intrinsics must have shape 4 with finite values")
    return intr


def required_bbox(hand: dict) -> np.ndarray:
    if "bbox_xyxy" not in hand:
        raise RuntimeError("hand row is missing bbox_xyxy")
    box = np.asarray(hand["bbox_xyxy"], dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise RuntimeError("bbox_xyxy must have shape 4 with finite values")
    return box


def bbox_metrics(points_camera_m: np.ndarray, intrinsics: np.ndarray, box_xyxy: np.ndarray) -> dict:
    projected = project_points(points_camera_m, intrinsics)
    proj_min = projected.min(axis=0)
    proj_max = projected.max(axis=0)
    target = bbox_corners(box_xyxy.tolist())
    residual = np.r_[proj_min - target[0], proj_max - target[1]]
    return {
        "projected_bbox_xyxy": [float(proj_min[0]), float(proj_min[1]), float(proj_max[0]), float(proj_max[1])],
        "detector_bbox_xyxy": [float(v) for v in box_xyxy],
        "residual_l2_px": float(np.linalg.norm(residual)),
        "residual_max_abs_px": float(np.max(np.abs(residual))),
        "residual_px": [float(v) for v in residual],
    }


def apply_z_axis_shift(points_camera_m: np.ndarray, shift_m: float) -> np.ndarray:
    shifted = points_camera_m.copy()
    shifted[:, 2] += float(shift_m)
    if np.any(shifted[:, 2] <= 0.0):
        raise RuntimeError("z-axis shift produced non-positive hand depth")
    return shifted


def apply_center_ray_shift(points_camera_m: np.ndarray, near_depth_shift_m: float) -> np.ndarray:
    center = np.median(points_camera_m, axis=0)
    if center[2] <= 0.0:
        raise RuntimeError("hand center has non-positive depth")
    direction = center / center[2]
    shifted = points_camera_m + float(near_depth_shift_m) * direction[None, :]
    if np.any(shifted[:, 2] <= 0.0):
        raise RuntimeError("center-ray shift produced non-positive hand depth")
    return shifted


def center_ray_translation_norm(points_camera_m: np.ndarray, near_depth_shift_m: float) -> float:
    center = np.median(points_camera_m, axis=0)
    if center[2] <= 0.0:
        raise RuntimeError("hand center has non-positive depth")
    direction = center / center[2]
    return float(np.linalg.norm(float(near_depth_shift_m) * direction))


def apply_camera_origin_scale(points_camera_m: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        raise RuntimeError("camera-origin scale must be positive")
    scaled = points_camera_m * float(scale)
    if np.any(scaled[:, 2] <= 0.0):
        raise RuntimeError("camera-origin scale produced non-positive hand depth")
    return scaled


def hand_extent_diag(points_camera_m: np.ndarray) -> float:
    extent = points_camera_m.max(axis=0) - points_camera_m.min(axis=0)
    return float(np.linalg.norm(extent))


def near_mask_vertices(
    vertices_camera_m: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    source_size: np.ndarray,
    contact_distance_px: float,
) -> np.ndarray:
    uv = project_points(vertices_camera_m, intrinsics)
    scale = np.asarray(mask.shape[::-1], dtype=float) / source_size.astype(float)
    xy = uv * scale[None, :]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(vertices_camera_m[:, 2]) & (vertices_camera_m[:, 2] > 0.0)
    if not np.any(valid):
        return np.zeros(len(vertices_camera_m), dtype=bool)
    distance = mask_distance_map(mask)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return valid & (distance[y, x] <= float(contact_distance_px) * float(scale.mean()))


def summarize_key(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=float)
    return summarize(values)


def summarize_by_side(rows: list[dict], key: str) -> dict:
    sides = sorted({str(row.get("side")) for row in rows})
    return {side: summarize_key([row for row in rows if str(row.get("side")) == side], key) for side in sides}


def group_summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "contact_abs_gap_m": summarize_key(rows, "contact_abs_gap_m"),
        "center_ray_translation_norm_m": summarize_key(rows, "center_ray_translation_norm_m"),
        "current_bbox_l2_px": summarize_key(rows, "current_bbox_l2_px"),
        "camera_origin_scale": summarize_key(rows, "camera_origin_scale"),
        "camera_origin_scale_hand_extent_change_m": summarize_key(rows, "camera_origin_scale_hand_extent_change_m"),
        "by_side_contact_abs_gap_m": summarize_by_side(rows, "contact_abs_gap_m") if rows else {},
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    if intrinsics.shape != (4,):
        raise RuntimeError("DROID intrinsics_source must have shape 4")
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    rows: list[dict] = []
    skipped: list[dict] = []

    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        obj = frame.get("object")
        if not isinstance(obj, dict) or "mask_path" not in obj:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        try:
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            _, object_z = project_world(object_vertices, T_world_camera, intrinsics)
            object_z = object_z[np.isfinite(object_z) & (object_z > 0.0)]
            if len(object_z) == 0:
                raise RuntimeError("object mesh has no positive camera depth")
            object_depth_m = float(np.median(object_z))
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            if source_size.shape != (2,):
                raise RuntimeError("object source_image_size must have shape 2")
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue

        for hand_idx, hand in enumerate(frame.get("hands", [])):
            try:
                vertices = required_points(hand, "vertices_source_camera_m")
                joints = required_points(hand, "joints3d_source_camera_m")
                intr = required_intrinsics(hand)
                if float(np.max(np.abs(intr - intrinsics))) > args.intrinsics_tolerance:
                    raise RuntimeError("hand and DROID source intrinsics disagree")
                box = required_bbox(hand)
                near = near_mask_vertices(vertices, intr, mask, source_size, args.contact_distance_px)
                near_count = int(np.count_nonzero(near))
                if near_count < args.min_near_vertices:
                    skipped.append(
                        {
                            "frame_idx": frame_idx,
                            "hand_idx": hand_idx,
                            "side": hand.get("side"),
                            "reason": "too_few_near_mask_vertices",
                            "near_vertices": near_count,
                        }
                    )
                    continue
                bbox_points = np.vstack([joints, vertices])
                current_bbox = bbox_metrics(bbox_points, intr, box)
                near_depth_m = float(np.median(vertices[near, 2]))
                signed_gap_m = near_depth_m - object_depth_m
                z_axis_points = apply_z_axis_shift(bbox_points, -signed_gap_m)
                z_axis_bbox = bbox_metrics(z_axis_points, intr, box)
                center_ray_points = apply_center_ray_shift(bbox_points, -signed_gap_m)
                center_ray_bbox = bbox_metrics(center_ray_points, intr, box)
                center_ray_norm_m = center_ray_translation_norm(bbox_points, -signed_gap_m)
                scale_to_contact = object_depth_m / near_depth_m
                camera_scale_points = apply_camera_origin_scale(bbox_points, scale_to_contact)
                camera_scale_bbox = bbox_metrics(camera_scale_points, intr, box)
                extent_diag_m = hand_extent_diag(vertices)
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_idx": int(hand_idx),
                        "side": hand.get("side"),
                        "detector_score": float(hand.get("detector_score", np.nan)),
                        "measurement_available": bool(hand.get("measurement_available", False)),
                        "high_confidence_measured": bool(
                            hand.get("measurement_available", False)
                            and np.isfinite(float(hand.get("detector_score", np.nan)))
                            and float(hand.get("detector_score", np.nan)) >= args.report_min_score
                        ),
                        "filter_status": hand.get("filter_status"),
                        "object_depth_m": object_depth_m,
                        "near_hand_depth_m": near_depth_m,
                        "near_vertices": near_count,
                        "hand_extent_diag_m": extent_diag_m,
                        "current_bbox_l2_px": current_bbox["residual_l2_px"],
                        "current_bbox_max_abs_px": current_bbox["residual_max_abs_px"],
                        "contact_signed_gap_m": signed_gap_m,
                        "contact_abs_gap_m": abs(signed_gap_m),
                        "z_axis_shift_m": -signed_gap_m,
                        "z_axis_shift_abs_m": abs(signed_gap_m),
                        "z_axis_bbox_l2_px": z_axis_bbox["residual_l2_px"],
                        "z_axis_bbox_delta_l2_px": z_axis_bbox["residual_l2_px"] - current_bbox["residual_l2_px"],
                        "z_axis_bbox_max_abs_px": z_axis_bbox["residual_max_abs_px"],
                        "center_ray_shift_m": float(abs(signed_gap_m)),
                        "center_ray_translation_norm_m": center_ray_norm_m,
                        "center_ray_bbox_l2_px": center_ray_bbox["residual_l2_px"],
                        "center_ray_bbox_delta_l2_px": center_ray_bbox["residual_l2_px"] - current_bbox["residual_l2_px"],
                        "center_ray_bbox_max_abs_px": center_ray_bbox["residual_max_abs_px"],
                        "camera_origin_scale": scale_to_contact,
                        "camera_origin_scale_abs_error": abs(scale_to_contact - 1.0),
                        "camera_origin_scale_hand_extent_change_m": abs(scale_to_contact - 1.0) * extent_diag_m,
                        "camera_origin_scale_bbox_l2_px": camera_scale_bbox["residual_l2_px"],
                        "camera_origin_scale_bbox_delta_l2_px": camera_scale_bbox["residual_l2_px"] - current_bbox["residual_l2_px"],
                    }
                )
            except Exception as exc:
                skipped.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_idx": hand_idx,
                        "side": hand.get("side"),
                        "reason": str(exc),
                    }
                )

    if len(rows) == 0:
        raise RuntimeError("no valid hand-object tradeoff rows were produced")

    median_abs_gap = float(np.median(np.asarray([row["contact_abs_gap_m"] for row in rows], dtype=float)))
    median_scale_extent_change = float(
        np.median(np.asarray([row["camera_origin_scale_hand_extent_change_m"] for row in rows], dtype=float))
    )
    status = "diagnostic_contact_requires_large_mano_depth_change"
    if median_abs_gap <= args.contact_solved_m and median_scale_extent_change <= args.hand_extent_change_solved_m:
        status = "diagnostic_contact_depth_already_consistent"

    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "contact_distance_px": float(args.contact_distance_px),
        "min_near_vertices": int(args.min_near_vertices),
        "tradeoff_rows": len(rows),
        "skipped_rows": len(skipped),
        "report_min_score": float(args.report_min_score),
        "all_rows_summary": group_summary(rows),
        "measured_rows_summary": group_summary([row for row in rows if row.get("measurement_available")]),
        "high_confidence_measured_rows_summary": group_summary([row for row in rows if row.get("high_confidence_measured")]),
        "contact_abs_gap_m": summarize_key(rows, "contact_abs_gap_m"),
        "contact_signed_gap_m": summarize_key(rows, "contact_signed_gap_m"),
        "current_bbox_l2_px": summarize_key(rows, "current_bbox_l2_px"),
        "z_axis_shift_m": summarize_key(rows, "z_axis_shift_m"),
        "z_axis_shift_abs_m": summarize_key(rows, "z_axis_shift_abs_m"),
        "z_axis_bbox_delta_l2_px": summarize_key(rows, "z_axis_bbox_delta_l2_px"),
        "z_axis_bbox_l2_px": summarize_key(rows, "z_axis_bbox_l2_px"),
        "center_ray_translation_norm_m": summarize_key(rows, "center_ray_translation_norm_m"),
        "center_ray_bbox_delta_l2_px": summarize_key(rows, "center_ray_bbox_delta_l2_px"),
        "center_ray_bbox_l2_px": summarize_key(rows, "center_ray_bbox_l2_px"),
        "camera_origin_scale": summarize_key(rows, "camera_origin_scale"),
        "camera_origin_scale_abs_error": summarize_key(rows, "camera_origin_scale_abs_error"),
        "camera_origin_scale_hand_extent_change_m": summarize_key(rows, "camera_origin_scale_hand_extent_change_m"),
        "camera_origin_scale_bbox_delta_l2_px": summarize_key(rows, "camera_origin_scale_bbox_delta_l2_px"),
        "camera_origin_scale_bbox_l2_px": summarize_key(rows, "camera_origin_scale_bbox_l2_px"),
        "by_side": {
            "contact_abs_gap_m": summarize_by_side(rows, "contact_abs_gap_m"),
            "current_bbox_l2_px": summarize_by_side(rows, "current_bbox_l2_px"),
            "center_ray_translation_norm_m": summarize_by_side(rows, "center_ray_translation_norm_m"),
            "camera_origin_scale": summarize_by_side(rows, "camera_origin_scale"),
            "camera_origin_scale_hand_extent_change_m": summarize_by_side(rows, "camera_origin_scale_hand_extent_change_m"),
        },
        "interpretation": (
            "Each row asks what would be required to make current near-mask MANO vertices match the object mesh depth. "
            "A z-axis or center-ray rigid hand shift preserves MANO size but changes the projected hand box. "
            "Camera-origin scaling preserves 2D projection but changes metric hand size. "
            "Large values in both families mean the contact conflict cannot be hidden inside a harmless smoother."
        ),
        "rows_preview": rows[:120],
        "skipped_preview": skipped[:120],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--intrinsics-tolerance", type=float, default=1e-3)
    parser.add_argument("--report-min-score", type=float, default=0.50)
    parser.add_argument("--contact-solved-m", type=float, default=0.005)
    parser.add_argument("--hand-extent-change-solved-m", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
