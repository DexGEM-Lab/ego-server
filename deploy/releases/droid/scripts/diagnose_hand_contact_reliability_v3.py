#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_contact_depth_conflict_v3 import mesh_frame_vertices, summarize
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hand_vertex_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    if "vertices_world_m" in hand:
        return "vertices_world_m"
    if "vertices_source_camera_m_sample" in hand:
        return "vertices_source_camera_m_sample"
    raise RuntimeError("hand has no source/world MANO vertices")


def camera_points_from_hand(hand: dict, T_world_camera: np.ndarray) -> np.ndarray:
    key = hand_vertex_key(hand)
    pts = np.asarray(hand[key], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"invalid {key}")
    if key == "vertices_world_m":
        T_camera_world = np.linalg.inv(T_world_camera)
        homog = np.c_[pts, np.ones(len(pts), dtype=float)]
        pts = (T_camera_world @ homog.T).T[:, :3]
    return pts


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def summarize_key(rows: list[dict], key: str) -> dict:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            vals.append(value_f)
    values = np.asarray(vals, dtype=float)
    return summarize(values)


def count_true(rows: list[dict], key: str) -> int:
    return int(sum(1 for row in rows if bool(row.get(key, False))))


def condition_counts(rows: list[dict]) -> dict:
    keys = [
        "measurement_available",
        "detector_ok",
        "projection_ok",
        "depth_ok",
        "stable_depth_ok",
        "bone_scale_ok",
        "contact_ok",
        "reliable_for_contact",
    ]
    return {key: count_true(rows, key) for key in keys}


def hand_tip_spread_m(joints: np.ndarray) -> float:
    if joints.shape != (21, 3):
        return float("nan")
    fingertips = joints[[4, 8, 12, 16, 20]]
    return float(np.max(np.linalg.norm(fingertips[:, None, :] - fingertips[None, :, :], axis=2)))


def hand_bone_scale_m(joints: np.ndarray) -> float:
    if joints.shape != (21, 3):
        return float("nan")
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    lengths = []
    for chain in chains:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def depth_patch_iqr_ratio(depth: np.ndarray, xy: np.ndarray, radius: int) -> float:
    x = int(np.clip(round(float(xy[0])), 0, depth.shape[1] - 1))
    y = int(np.clip(round(float(xy[1])), 0, depth.shape[0] - 1))
    patch = depth[max(0, y - radius) : min(depth.shape[0], y + radius + 1), max(0, x - radius) : min(depth.shape[1], x + radius + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0.0)]
    if len(vals) == 0:
        return float("nan")
    med = float(np.median(vals))
    return float((np.percentile(vals, 75.0) - np.percentile(vals, 25.0)) / max(1e-6, med))


def hand_row(frame: dict, hand_i: int, hand: dict, depth: np.ndarray, mask: np.ndarray, object_vertices: np.ndarray, args: argparse.Namespace) -> dict:
    frame_idx = int(frame["frame_idx"])
    obj = frame["object"]
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    source_size = np.asarray(obj["source_image_size"], dtype=float)
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if joints.shape != (21, 3) or raw2d.shape != (21, 2) or intr.shape != (4,):
        raise RuntimeError("invalid hand keypoint fields")
    vertices = camera_points_from_hand(hand, T_world_camera)
    if np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("hand vertices contain non-positive depth")

    projected = project_points(joints, intr)
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
    uv = project_points(vertices, intr)
    xy = uv * depth_scale[None, :]
    valid_uv = np.isfinite(xy).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = valid_uv & (dist[y, x] <= args.contact_distance_px)
    near_vertices = vertices[near]

    homog = np.c_[object_vertices, np.ones(len(object_vertices), dtype=float)]
    object_camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    object_z = object_camera[:, 2]
    object_z = object_z[np.isfinite(object_z) & (object_z > 0.0)]
    if len(object_z) == 0:
        raise RuntimeError("object mesh has no positive camera depth")
    object_depth = float(np.median(object_z))
    contact_gap = near_vertices[:, 2] - object_depth if len(near_vertices) else np.asarray([], dtype=float)

    score = float(hand.get("detector_score", np.nan))
    projection_ok = float(np.median(reproj)) <= args.max_good_median_reprojection_px
    depth_ok = len(mano_minus_metric) >= args.min_good_depth_joints and abs(float(np.median(mano_minus_metric))) <= args.max_good_depth_bias_m
    patch_ok = len(stable_depth) >= args.min_good_depth_joints and float(np.mean(stable_depth)) >= args.min_stable_depth_fraction
    tip_spread = hand_tip_spread_m(joints)
    bone_scale = hand_bone_scale_m(joints)
    bone_scale_ok = args.min_bone_scale_m <= bone_scale <= args.max_bone_scale_m
    contact_ok = len(near_vertices) >= args.min_near_vertices and abs(float(np.median(contact_gap))) <= args.max_good_contact_gap_m
    measured = bool(hand.get("measurement_available", False))
    detector_ok = np.isfinite(score) and score >= args.min_detector_score
    reliable_for_contact = bool(measured and detector_ok and projection_ok and depth_ok and patch_ok and bone_scale_ok and contact_ok)

    return {
        "frame_idx": frame_idx,
        "hand_idx": int(hand_i),
        "side": hand.get("side"),
        "filter_status": hand.get("filter_status"),
        "measurement_available": measured,
        "detector_score": score,
        "median_joint_reprojection_px": float(np.median(reproj)),
        "p95_joint_reprojection_px": float(np.percentile(reproj, 95.0)),
        "good_depth_joints": int(np.count_nonzero(good_depth)),
        "mano_minus_metric_depth_median_m": None if len(mano_minus_metric) == 0 else float(np.median(mano_minus_metric)),
        "mano_minus_metric_depth_p95_abs_m": None if len(mano_minus_metric) == 0 else float(np.percentile(np.abs(mano_minus_metric), 95.0)),
        "stable_depth_fraction": None if len(stable_depth) == 0 else float(np.mean(stable_depth)),
        "hand_bone_scale_m": bone_scale,
        "hand_tip_spread_m": tip_spread,
        "near_mask_vertices": int(len(near_vertices)),
        "near_mask_hand_minus_object_depth_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "near_mask_hand_minus_object_depth_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
        "detector_ok": bool(detector_ok),
        "projection_ok": bool(projection_ok),
        "depth_ok": bool(depth_ok),
        "stable_depth_ok": bool(patch_ok),
        "bone_scale_ok": bool(bone_scale_ok),
        "contact_ok": bool(contact_ok),
        "reliable_for_contact": reliable_for_contact,
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
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        obj = frame.get("object", {})
        if not obj.get("mask_path"):
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                rows.append(hand_row(frame, hand_i, hand, depth, mask, object_vertices, args))
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
    if not rows:
        raise RuntimeError("no hand reliability rows")

    measured = [row for row in rows if row["measurement_available"]]
    reliable = [row for row in rows if row["reliable_for_contact"]]
    high_score = [row for row in rows if row["detector_ok"]]
    measured_high_score = [row for row in measured if row["detector_ok"]]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "rows": len(rows),
        "measured_rows": len(measured),
        "high_score_rows": len(high_score),
        "measured_high_score_rows": len(measured_high_score),
        "reliable_contact_rows": len(reliable),
        "summary_all": {
            "joint_reprojection_px": summarize_key(rows, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(rows, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(rows, "hand_tip_spread_m"),
        },
        "summary_measured_high_score": {
            "joint_reprojection_px": summarize_key(measured_high_score, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(measured_high_score, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(measured_high_score, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(measured_high_score, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(measured_high_score, "hand_tip_spread_m"),
        },
        "summary_reliable_contact": {
            "joint_reprojection_px": summarize_key(reliable, "median_joint_reprojection_px"),
            "mano_minus_metric_depth_m": summarize_key(reliable, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(reliable, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(reliable, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(reliable, "hand_tip_spread_m"),
        },
        "condition_counts_all": condition_counts(rows),
        "condition_counts_measured_high_score": condition_counts(measured_high_score),
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "max_good_median_reprojection_px": float(args.max_good_median_reprojection_px),
            "max_good_depth_bias_m": float(args.max_good_depth_bias_m),
            "max_good_contact_gap_m": float(args.max_good_contact_gap_m),
            "min_good_depth_joints": int(args.min_good_depth_joints),
            "min_near_vertices": int(args.min_near_vertices),
            "min_bone_scale_m": float(args.min_bone_scale_m),
            "max_bone_scale_m": float(args.max_bone_scale_m),
        },
        "interpretation": (
            "A contact factor is trustworthy only when the measured hand has good 2D keypoint reprojection, "
            "stable metric-depth samples at its keypoints, plausible MANO bone scale, and a small hand-object depth gap. "
            "Tip spread is reported as a pose descriptor but is not used as a size test because a grasping hand can be closed. "
            "Rows that pass detector score alone but fail these tests are perception conflicts, not contact evidence."
        ),
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:120],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
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
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
