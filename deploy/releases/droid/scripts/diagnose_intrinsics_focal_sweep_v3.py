#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from diagnose_contact_depth_conflict_v3 import mesh_frame_vertices
from diagnose_hand_contact_reliability_v3 import (
    condition_counts,
    depth_patch_iqr_ratio,
    hand_bone_scale_m,
    hand_vertex_key,
    resize_mask_to_depth,
    summarize_key,
)
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def solve_source_camera_translation(local_points_m: np.ndarray, points2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(float)
    qx = (points2d[:, 0] - cx) / fx
    qy = (points2d[:, 1] - cy) / fy
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y, z), u, v in zip(local_points_m, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), rcond=None)
    return trans.astype(float)


def source_local_vertices(hand: dict) -> np.ndarray:
    key = hand_vertex_key(hand)
    if key not in {"vertices_source_camera_m", "vertices_source_camera_m_sample"}:
        raise RuntimeError(f"hand vertex key {key} is not source-camera geometry")
    vertices_source = np.asarray(hand[key], dtype=float)
    trans = np.asarray(hand.get("cam_t", []), dtype=float)
    if vertices_source.ndim != 2 or vertices_source.shape[1] != 3 or trans.shape != (3,):
        raise RuntimeError("invalid source vertices or cam_t")
    return vertices_source - trans[None, :]


def hand_row(
    frame: dict,
    hand_i: int,
    hand: dict,
    depth: np.ndarray,
    mask: np.ndarray,
    object_vertices: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    frame_idx = int(frame["frame_idx"])
    obj = frame["object"]
    source_size = np.asarray(obj["source_image_size"], dtype=float)
    local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if local_joints.shape != (21, 3) or raw2d.shape != (21, 2):
        raise RuntimeError("invalid local hand or raw 2D keypoint fields")
    if not bool(hand.get("measurement_available", False)):
        raise RuntimeError("focal sweep uses measured rows only")
    local_vertices = source_local_vertices(hand)
    trans = solve_source_camera_translation(local_joints, raw2d, intrinsics)
    joints = local_joints + trans[None, :]
    vertices = local_vertices + trans[None, :]
    if np.any(vertices[:, 2] <= 0.0) or np.any(joints[:, 2] <= 0.0):
        raise RuntimeError("candidate focal produced non-positive hand depth")

    projected = project_points(joints, intrinsics)
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    metric_depth = sample_depth(depth, raw2d, source_size)
    valid_depth = np.isfinite(metric_depth) & (metric_depth > 0.0)
    good_depth = valid_depth & (reproj <= args.good_joint_reprojection_px)
    mano_minus_metric = joints[good_depth, 2] - metric_depth[good_depth]

    depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    depth_xy = raw2d * depth_scale[None, :]
    patch_ratios = np.asarray([depth_patch_iqr_ratio(depth, xy, args.patch_radius) for xy in depth_xy[good_depth]], dtype=float)
    stable_depth = patch_ratios[np.isfinite(patch_ratios)] <= args.max_depth_iqr_ratio

    mask_depth = resize_mask_to_depth(mask, depth)
    dist = mask_distance_map(mask_depth)
    uv = project_points(vertices, intrinsics)
    xy = uv * depth_scale[None, :]
    valid_uv = np.isfinite(xy).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = valid_uv & (dist[y, x] <= args.contact_distance_px)
    near_vertices = vertices[near]

    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    homog = np.c_[object_vertices, np.ones(len(object_vertices), dtype=float)]
    object_camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    object_z = object_camera[:, 2]
    object_z = object_z[np.isfinite(object_z) & (object_z > 0.0)]
    if len(object_z) == 0:
        raise RuntimeError("object mesh has no positive camera depth")
    object_depth = float(np.median(object_z))
    contact_gap = near_vertices[:, 2] - object_depth if len(near_vertices) else np.asarray([], dtype=float)

    original_trans = np.asarray(hand["cam_t"], dtype=float)
    score = float(hand.get("detector_score", np.nan))
    projection_ok = float(np.median(reproj)) <= args.max_good_median_reprojection_px
    depth_ok = len(mano_minus_metric) >= args.min_good_depth_joints and abs(float(np.median(mano_minus_metric))) <= args.max_good_depth_bias_m
    patch_ok = len(stable_depth) >= args.min_good_depth_joints and float(np.mean(stable_depth)) >= args.min_stable_depth_fraction
    bone_scale = hand_bone_scale_m(joints)
    bone_scale_ok = args.min_bone_scale_m <= bone_scale <= args.max_bone_scale_m
    contact_ok = len(near_vertices) >= args.min_near_vertices and abs(float(np.median(contact_gap))) <= args.max_good_contact_gap_m
    detector_ok = np.isfinite(score) and score >= args.min_detector_score
    reliable_for_contact = bool(detector_ok and projection_ok and depth_ok and patch_ok and bone_scale_ok and contact_ok)

    return {
        "frame_idx": frame_idx,
        "hand_idx": int(hand_i),
        "side": hand.get("side"),
        "filter_status": hand.get("filter_status"),
        "measurement_available": True,
        "detector_score": score,
        "median_joint_reprojection_px": float(np.median(reproj)),
        "p95_joint_reprojection_px": float(np.percentile(reproj, 95.0)),
        "good_depth_joints": int(np.count_nonzero(good_depth)),
        "mano_minus_metric_depth_median_m": None if len(mano_minus_metric) == 0 else float(np.median(mano_minus_metric)),
        "mano_minus_metric_depth_p95_abs_m": None if len(mano_minus_metric) == 0 else float(np.percentile(np.abs(mano_minus_metric), 95.0)),
        "stable_depth_fraction": None if len(stable_depth) == 0 else float(np.mean(stable_depth)),
        "hand_bone_scale_m": bone_scale,
        "near_mask_vertices": int(len(near_vertices)),
        "near_mask_hand_minus_object_depth_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "near_mask_hand_minus_object_depth_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
        "cam_t_z_m": float(trans[2]),
        "original_cam_t_z_m": float(original_trans[2]),
        "cam_t_shift_m": float(np.linalg.norm(trans - original_trans)),
        "detector_ok": bool(detector_ok),
        "projection_ok": bool(projection_ok),
        "depth_ok": bool(depth_ok),
        "stable_depth_ok": bool(patch_ok),
        "bone_scale_ok": bool(bone_scale_ok),
        "contact_ok": bool(contact_ok),
        "reliable_for_contact": reliable_for_contact,
    }


def row_sort_key(row: dict) -> tuple[int, float, float]:
    near = int(row.get("near_mask_vertices", 0))
    gap = row.get("near_mask_hand_minus_object_depth_median_m")
    gap_value = abs(float(gap)) if gap is not None and np.isfinite(float(gap)) else math.inf
    reproj = float(row.get("median_joint_reprojection_px", math.inf))
    return (-near, gap_value, reproj)


def compact_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    keys = [
        "frame_idx",
        "hand_idx",
        "side",
        "detector_score",
        "median_joint_reprojection_px",
        "good_depth_joints",
        "mano_minus_metric_depth_median_m",
        "stable_depth_fraction",
        "near_mask_vertices",
        "near_mask_hand_minus_object_depth_median_m",
        "cam_t_z_m",
        "cam_t_shift_m",
        "projection_ok",
        "depth_ok",
        "stable_depth_ok",
        "bone_scale_ok",
        "contact_ok",
        "reliable_for_contact",
    ]
    out = []
    for row in sorted(rows, key=row_sort_key)[:limit]:
        out.append({key: row.get(key) for key in keys})
    return out


def focal_report(rows: list[dict], focal: float, width: int, height: int, row_preview_limit: int) -> dict:
    high_score = [row for row in rows if row["detector_ok"]]
    reliable = [row for row in rows if row["reliable_for_contact"]]
    return {
        "focal_px": float(focal),
        "horizontal_fov_deg": float(2.0 * math.degrees(math.atan(width / (2.0 * focal)))),
        "vertical_fov_deg": float(2.0 * math.degrees(math.atan(height / (2.0 * focal)))),
        "rows": len(rows),
        "high_score_rows": len(high_score),
        "reliable_contact_rows": len(reliable),
        "summary_high_score": {
            "joint_reprojection_px": summarize_key(high_score, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(high_score, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(high_score, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(high_score, "hand_bone_scale_m"),
            "cam_t_z_m": summarize_key(high_score, "cam_t_z_m"),
            "cam_t_shift_m": summarize_key(high_score, "cam_t_shift_m"),
        },
        "condition_counts_high_score": condition_counts(high_score),
        "summary_reliable_contact": {
            "joint_reprojection_px": summarize_key(reliable, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(reliable, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(reliable, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(reliable, "hand_bone_scale_m"),
        },
        "contact_rows_preview": compact_rows([row for row in high_score if int(row.get("near_mask_vertices", 0)) > 0], row_preview_limit),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_frame_idx)) != len(depth_frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)

    reports = []
    skipped: list[dict] = []
    for focal in args.focals:
        rows: list[dict] = []
        intrinsics = np.asarray([float(focal), float(focal), args.cx, args.cy], dtype=float)
        for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
            frame = frames.get(frame_idx)
            if frame is None:
                skipped.append({"focal_px": float(focal), "frame_idx": frame_idx, "reason": "missing_annotation_frame"})
                continue
            obj = frame.get("object", {})
            if not obj.get("mask_path"):
                skipped.append({"focal_px": float(focal), "frame_idx": frame_idx, "reason": "missing_object_mask"})
                continue
            try:
                depth = depth_frame(depths, frame_to_depth_i, frame_idx)
                mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
                mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
                object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            except Exception as exc:
                skipped.append({"focal_px": float(focal), "frame_idx": frame_idx, "reason": str(exc)})
                continue
            for hand_i, hand in enumerate(frame.get("hands", [])):
                if not bool(hand.get("measurement_available", False)):
                    continue
                try:
                    rows.append(hand_row(frame, hand_i, hand, depth, mask, object_vertices, intrinsics, args))
                except Exception as exc:
                    skipped.append({"focal_px": float(focal), "frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
        if not rows:
            raise RuntimeError(f"no focal-sweep rows for focal {focal}")
        reports.append(focal_report(rows, float(focal), args.width, args.height, int(args.row_preview_limit)))

    ranked_by_depth = sorted(
        reports,
        key=lambda item: abs(float(item["summary_high_score"]["mano_minus_metric_depth_m"].get("median", math.inf))),
    )
    ranked_by_contact = sorted(
        reports,
        key=lambda item: abs(float(item["summary_high_score"]["contact_gap_m"].get("median", math.inf))),
    )
    report = {
        "status": "ok",
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "source_size": [int(args.width), int(args.height)],
        "principal_point": [float(args.cx), float(args.cy)],
        "tested_focals_px": [float(x) for x in args.focals],
        "reports": reports,
        "best_depth_bias_focal_px": ranked_by_depth[0]["focal_px"],
        "best_contact_gap_focal_px": ranked_by_contact[0]["focal_px"],
        "skipped_preview": skipped[:200],
        "interpretation": (
            "This diagnostic varies only the pinhole focal length used to translate existing MANO local geometry "
            "from observed 2D keypoints into source-camera metric coordinates. A focal value is useful only if it "
            "improves depth/contact while retaining keypoint projection and bone-scale reliability."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k not in {"skipped_preview"}}
    print(json.dumps(printable, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--focals", type=float, nargs="+", default=[500.0, 600.0, 700.0, 800.0, 900.0, 960.0, 1100.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2304.0, 2600.0])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--max-good-median-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-good-depth-bias-m", type=float, default=0.030)
    parser.add_argument("--max-good-contact-gap-m", type=float, default=0.030)
    parser.add_argument("--min-good-depth-joints", type=int, default=12)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--min-stable-depth-fraction", type=float, default=0.75)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--row-preview-limit", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
