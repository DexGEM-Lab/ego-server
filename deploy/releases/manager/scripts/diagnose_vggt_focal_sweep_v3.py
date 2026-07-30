#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from diagnose_intrinsics_focal_sweep_v3 import (
    hand_bone_scale_m,
    mask_distance_map,
    project_points,
    solve_source_camera_translation,
    source_local_vertices,
)
from diagnose_vggt_mano_contact_v3 import (
    points_to_vggt_frame,
    resize_mask,
    vggt_affine_from_source,
    vggt_frame_points,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float | None]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def condition_counts(rows: list[dict]) -> dict:
    names = [
        "detector_ok",
        "projection_ok",
        "bone_scale_ok",
        "contact_ok",
        "contact_p95_ok",
        "reliable_for_contact",
        "strict_reliable_for_contact",
    ]
    return {name: int(sum(bool(row.get(name, False)) for row in rows)) for name in names}


def gap_stats(gap: np.ndarray) -> dict:
    if gap.size == 0:
        return {
            "contact_vertices_005m": 0,
            "contact_vertices_010m": 0,
            "contact_vertices_020m": 0,
            "contact_vertices_030m": 0,
            "contact_fraction_010m": 0.0,
            "contact_fraction_030m": 0.0,
            "penetration_vertices_010m": 0,
            "penetration_vertices_030m": 0,
            "penetration_fraction_010m": 0.0,
            "penetration_fraction_030m": 0.0,
            "positive_gap_p95_m": None,
            "negative_gap_p05_m": None,
        }
    abs_gap = np.abs(gap)
    positive = np.maximum(gap, 0.0)
    return {
        "contact_vertices_005m": int(np.count_nonzero(abs_gap <= 0.005)),
        "contact_vertices_010m": int(np.count_nonzero(abs_gap <= 0.010)),
        "contact_vertices_020m": int(np.count_nonzero(abs_gap <= 0.020)),
        "contact_vertices_030m": int(np.count_nonzero(abs_gap <= 0.030)),
        "contact_fraction_010m": float(np.mean(abs_gap <= 0.010)),
        "contact_fraction_030m": float(np.mean(abs_gap <= 0.030)),
        "penetration_vertices_010m": int(np.count_nonzero(gap > 0.010)),
        "penetration_vertices_030m": int(np.count_nonzero(gap > 0.030)),
        "penetration_fraction_010m": float(np.mean(gap > 0.010)),
        "penetration_fraction_030m": float(np.mean(gap > 0.030)),
        "positive_gap_p95_m": float(np.percentile(positive, 95)),
        "negative_gap_p05_m": float(np.percentile(gap, 5)),
    }


def source_focal_to_vggt_intrinsics(
    focal_px: float,
    source_width: int,
    source_height: int,
    source_cx: float,
    source_cy: float,
    target_size: int,
) -> np.ndarray:
    scale_x, scale_y, pad_top = vggt_affine_from_source(source_width, source_height, target_size)
    return np.asarray(
        [
            float(focal_px) * scale_x,
            float(focal_px) * scale_y,
            float(source_cx) * scale_x,
            float(source_cy) * scale_y + pad_top,
        ],
        dtype=float,
    )


def vggt_predicted_source_focals(archive: np.lib.npyio.NpzFile, source_width: int, source_height: int, target_size: int) -> dict:
    intrinsic = archive["intrinsic"].astype(float)
    scale_x, scale_y, _ = vggt_affine_from_source(source_width, source_height, target_size)
    fx_source = intrinsic[:, 0, 0] / scale_x
    fy_source = intrinsic[:, 1, 1] / scale_y
    return {
        "fx_source_px": summarize(fx_source.tolist()),
        "fy_source_px": summarize(fy_source.tolist()),
        "median_mean_source_focal_px": float(np.median((fx_source + fy_source) * 0.5)),
        "per_frame_source_focal_px": ((fx_source + fy_source) * 0.5).astype(float).tolist(),
    }


def object_depth_for_frame(archive: np.lib.npyio.NpzFile, frame_i: int) -> float:
    points = vggt_frame_points(archive, frame_i)
    extrinsic = archive["extrinsic"].astype(float)[frame_i]
    camera = (points @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if z.size == 0:
        raise RuntimeError(f"VGGT object points have no positive camera depth at frame index {frame_i}")
    return float(np.median(z))


def hand_row(
    frame: dict,
    hand_i: int,
    hand: dict,
    mask: np.ndarray,
    object_depth: float,
    intrinsics_vggt: np.ndarray,
    target_size: int,
    args: argparse.Namespace,
) -> dict | None:
    if not bool(hand.get("measurement_available", False)):
        return None
    local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if local_joints.shape != (21, 3) or raw2d.shape != (21, 2):
        return None
    local_vertices = source_local_vertices(hand)
    raw2d_vggt = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], target_size)
    trans = solve_source_camera_translation(local_joints, raw2d_vggt, intrinsics_vggt)
    joints = local_joints + trans[None, :]
    vertices = local_vertices + trans[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        return None
    projected = project_points(joints, intrinsics_vggt)
    reproj = np.linalg.norm(projected - raw2d_vggt, axis=1)
    dist = mask_distance_map(mask)
    uv = project_points(vertices, intrinsics_vggt)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, target_size - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, target_size - 1)
    near = valid & (dist[y, x] <= float(args.contact_distance_px))
    gap = vertices[near, 2] - float(object_depth) if np.any(near) else np.asarray([], dtype=float)
    detector_score = float(hand.get("detector_score", np.nan))
    detector_ok = bool(np.isfinite(detector_score) and detector_score >= float(args.min_detector_score))
    projection_ok = bool(float(np.median(reproj)) <= float(args.max_good_median_reprojection_px))
    bone_scale = float(hand_bone_scale_m(joints))
    bone_scale_ok = bool(float(args.min_bone_scale_m) <= bone_scale <= float(args.max_bone_scale_m))
    stats = gap_stats(gap)
    contact_ok = bool(
        int(np.count_nonzero(near)) >= int(args.min_near_vertices)
        and gap.size
        and abs(float(np.median(gap))) <= float(args.max_good_contact_gap_m)
    )
    contact_p95_ok = bool(
        int(np.count_nonzero(near)) >= int(args.min_near_vertices)
        and gap.size
        and float(np.percentile(np.abs(gap), 95)) <= float(args.max_good_contact_gap_m)
    )
    return {
        "frame_idx": int(frame["frame_idx"]),
        "hand_idx": int(hand_i),
        "side": hand.get("side"),
        "detector_score": detector_score,
        "median_joint_reprojection_px_vggt": float(np.median(reproj)),
        "p95_joint_reprojection_px_vggt": float(np.percentile(reproj, 95)),
        "cam_t_z_vggt": float(trans[2]),
        "object_depth_vggt": float(object_depth),
        "near_mask_vertices": int(np.count_nonzero(near)),
        "near_mask_hand_minus_object_depth_median_vggt": float(np.median(gap)) if gap.size else None,
        "near_mask_hand_minus_object_depth_p95_abs_vggt": float(np.percentile(np.abs(gap), 95)) if gap.size else None,
        **stats,
        "hand_bone_scale_m": bone_scale,
        "detector_ok": detector_ok,
        "projection_ok": projection_ok,
        "bone_scale_ok": bone_scale_ok,
        "contact_ok": contact_ok,
        "contact_p95_ok": contact_p95_ok,
        "reliable_for_contact": bool(detector_ok and projection_ok and bone_scale_ok and contact_ok),
        "strict_reliable_for_contact": bool(detector_ok and projection_ok and bone_scale_ok and contact_p95_ok),
    }


def compact_vggt_contact_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    keys = [
        "frame_idx",
        "hand_idx",
        "side",
        "detector_score",
        "median_joint_reprojection_px_vggt",
        "near_mask_vertices",
        "near_mask_hand_minus_object_depth_median_vggt",
        "near_mask_hand_minus_object_depth_p95_abs_vggt",
        "contact_vertices_010m",
        "contact_vertices_030m",
        "contact_fraction_010m",
        "contact_fraction_030m",
        "penetration_fraction_010m",
        "penetration_fraction_030m",
        "positive_gap_p95_m",
        "negative_gap_p05_m",
        "cam_t_z_vggt",
        "object_depth_vggt",
        "hand_bone_scale_m",
        "projection_ok",
        "bone_scale_ok",
        "contact_ok",
        "contact_p95_ok",
        "reliable_for_contact",
        "strict_reliable_for_contact",
    ]
    def sort_key(row: dict) -> tuple[float, float]:
        p95 = row.get("near_mask_hand_minus_object_depth_p95_abs_vggt")
        median = row.get("near_mask_hand_minus_object_depth_median_vggt")
        p95_value = abs(float(p95)) if p95 is not None and np.isfinite(float(p95)) else math.inf
        median_value = abs(float(median)) if median is not None and np.isfinite(float(median)) else math.inf
        return (p95_value, median_value)
    return [{key: row.get(key) for key in keys} for row in sorted(rows, key=sort_key)[:limit]]


def report_for_focal(rows: list[dict], focal_px: float, width: int, height: int, row_preview_limit: int) -> dict:
    high = [row for row in rows if row["detector_ok"]]
    contact_rows = [row for row in high if int(row["near_mask_vertices"]) > 0]
    supported_contact = [row for row in high if int(row["near_mask_vertices"]) >= 80]
    reliable = [row for row in rows if row["reliable_for_contact"]]
    strict_reliable = [row for row in rows if row["strict_reliable_for_contact"]]
    return {
        "source_focal_px": float(focal_px),
        "horizontal_fov_deg": float(2.0 * math.degrees(math.atan(width / (2.0 * float(focal_px))))),
        "vertical_fov_deg": float(2.0 * math.degrees(math.atan(height / (2.0 * float(focal_px))))),
        "rows": int(len(rows)),
        "high_score_rows": int(len(high)),
        "contact_projecting_rows": int(len(contact_rows)),
        "supported_contact_rows": int(len(supported_contact)),
        "reliable_contact_rows": int(len(reliable)),
        "strict_reliable_contact_rows": int(len(strict_reliable)),
        "summary_high_score": {
            "joint_reprojection_px_vggt": summarize([row["median_joint_reprojection_px_vggt"] for row in high]),
            "cam_t_z_vggt": summarize([row["cam_t_z_vggt"] for row in high]),
            "object_depth_vggt": summarize([row["object_depth_vggt"] for row in high]),
            "near_mask_vertices": summarize([row["near_mask_vertices"] for row in high]),
            "contact_gap_vggt": summarize([row["near_mask_hand_minus_object_depth_median_vggt"] for row in supported_contact]),
            "contact_gap_abs_p95_vggt": summarize([row["near_mask_hand_minus_object_depth_p95_abs_vggt"] for row in supported_contact]),
            "contact_vertices_010m": summarize([row["contact_vertices_010m"] for row in supported_contact]),
            "contact_vertices_030m": summarize([row["contact_vertices_030m"] for row in supported_contact]),
            "contact_fraction_010m": summarize([row["contact_fraction_010m"] for row in supported_contact]),
            "contact_fraction_030m": summarize([row["contact_fraction_030m"] for row in supported_contact]),
            "penetration_fraction_010m": summarize([row["penetration_fraction_010m"] for row in supported_contact]),
            "penetration_fraction_030m": summarize([row["penetration_fraction_030m"] for row in supported_contact]),
            "positive_gap_p95_m": summarize([row["positive_gap_p95_m"] for row in supported_contact]),
            "negative_gap_p05_m": summarize([row["negative_gap_p05_m"] for row in supported_contact]),
            "hand_bone_scale_m": summarize([row["hand_bone_scale_m"] for row in high]),
        },
        "condition_counts_high_score": condition_counts(high),
        "contact_rows_preview": compact_vggt_contact_rows(contact_rows, row_preview_limit),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    archive = np.load(args.vggt_archive)
    frames = archive["frame_idx"].astype(int)
    masks = {int(row["frame_idx"]): Path(row["mask"]) for row in load_json(args.dataset_manifest)["frames"]}
    source_focals = [float(v) for v in args.source_focals]
    predicted = vggt_predicted_source_focals(archive, int(args.width), int(args.height), int(args.target_size))
    if args.include_vggt_predicted_focal:
        focal = float(predicted["median_mean_source_focal_px"])
        if all(abs(focal - existing) > 1e-6 for existing in source_focals):
            source_focals.append(focal)
    source_focals = sorted(source_focals)
    per_frame = []
    for i, frame_idx in enumerate(frames.tolist()):
        frame = frame_by_idx.get(int(frame_idx))
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        mask_path = masks.get(int(frame_idx))
        if mask_path is None:
            raise RuntimeError(f"dataset manifest missing frame {frame_idx}")
        per_frame.append((frame, resize_mask(mask_path, int(args.target_size)), object_depth_for_frame(archive, i)))
    reports = []
    detail_by_focal = {}
    for focal_px in source_focals:
        K4 = source_focal_to_vggt_intrinsics(
            focal_px,
            int(args.width),
            int(args.height),
            float(args.cx),
            float(args.cy),
            int(args.target_size),
        )
        rows = []
        for frame, mask, object_depth in per_frame:
            for hand_i, hand in enumerate(frame.get("hands", [])):
                row = hand_row(frame, hand_i, hand, mask, object_depth, K4, int(args.target_size), args)
                if row is not None:
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"no valid VGGT focal-sweep rows for focal {focal_px}")
        reports.append(report_for_focal(rows, focal_px, int(args.width), int(args.height), int(args.row_preview_limit)))
        if args.keep_detail:
            detail_by_focal[f"{focal_px:.6f}"] = rows
    ranked_contact = sorted(
        reports,
        key=lambda row: abs(float(row["summary_high_score"]["contact_gap_vggt"].get("median", math.inf))),
    )
    ranked_projection = sorted(
        reports,
        key=lambda row: float(row["summary_high_score"]["joint_reprojection_px_vggt"].get("median", math.inf)),
    )
    report = {
        "status": "ok",
        "diagnostic_only": True,
        "method": "diagnose_vggt_focal_sweep_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "source_size": [int(args.width), int(args.height)],
        "target_size": int(args.target_size),
        "principal_point_source_px": [float(args.cx), float(args.cy)],
        "vggt_predicted_source_focal_px": predicted,
        "reports": reports,
        "best_contact_gap_source_focal_px": ranked_contact[0]["source_focal_px"],
        "best_projection_source_focal_px": ranked_projection[0]["source_focal_px"],
        "interpretation": (
            "This diagnostic holds VGGT object depth fixed and varies only the source-frame focal length "
            "used to translate MANO local joints from 2D keypoints into the VGGT camera frame. A useful "
            "focal must reduce hand-object contact gap while preserving detector reprojection and hand bone scale."
        ),
    }
    if args.keep_detail:
        report["rows_detail_by_focal"] = detail_by_focal
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "rows_detail_by_focal"}
    print(json.dumps(printable, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-focals", type=float, nargs="+", default=[900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1600.0, 1800.0, 2000.0, 2304.0, 2600.0])
    parser.add_argument("--include-vggt-predicted-focal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--max-good-median-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-good-contact-gap-m", type=float, default=0.030)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--row-preview-limit", type=int, default=12)
    parser.add_argument("--keep-detail", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
