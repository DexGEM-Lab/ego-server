#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v20_common import (
    ContractError,
    finite_depth_values,
    load_depth_candidate_depth,
    load_json,
    load_mask,
    load_registry_candidates,
    numeric_summary,
    robust_abs_median,
    sample_depth_at_points,
    write_json,
)


def frame_source_size(frame: dict[str, Any], candidate_depth: np.ndarray) -> list[int]:
    for key in ("source_image_size", "source_size", "resolution"):
        raw = frame.get(key)
        if isinstance(raw, list) and len(raw) >= 2:
            return [int(raw[0]), int(raw[1])]
    return [int(candidate_depth.shape[1]), int(candidate_depth.shape[0])]


def collect_object_masks(annotations: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in annotations.get("frames", []) if isinstance(annotations, dict) else []:
        if len(rows) >= max_frames:
            break
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        if frame_idx < 0:
            continue
        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        for obj in objects:
            mask_path = obj.get("mask_path")
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            if mask_path or geom:
                rows.append({"frame_idx": frame_idx, "frame": frame, "object": obj, "mask_path": mask_path, "visible_geometry": geom})
                break
    return rows


def collect_hand_rows(annotations: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in annotations.get("frames", []) if isinstance(annotations, dict) else []:
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        if frame_idx < 0:
            continue
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else hand
            joints2d = metric.get("joints2d_raw") or metric.get("joints_2d_px") or hand.get("joints2d_raw") or hand.get("joints_2d_px")
            joints3d = metric.get("joints_current_v18_camera_m") or metric.get("joints3d_camera") or hand.get("joints3d_camera")
            if joints2d is not None and joints3d is not None:
                rows.append({"frame_idx": frame_idx, "frame": frame, "hand": hand, "joints2d": joints2d, "joints3d": joints3d})
                if len(rows) >= max_rows:
                    return rows
    return rows


def object_mask_continuity(candidate_data: dict[str, Any], object_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    medians = []
    p95s = []
    valid_fracs = []
    for row in object_rows:
        frame_idx = int(row["frame_idx"])
        depth_i = candidate_data["frame_to_i"].get(frame_idx)
        if depth_i is None:
            continue
        depth = candidate_data["depth"][depth_i]
        mask_path = row.get("mask_path")
        if not mask_path:
            continue
        mask = load_mask(Path(mask_path), depth.shape)
        vals = finite_depth_values(depth, mask)
        if vals.size < int(args.min_mask_depth_pixels):
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        p95 = float(np.percentile(np.abs(vals - med), 95.0))
        medians.append(mad)
        p95s.append(p95)
        valid_fracs.append(float(vals.size / max(1, int(mask.sum()))))
    return {
        "evaluated_rows": len(medians),
        "mask_depth_mad_m": numeric_summary(medians),
        "mask_depth_p95_absdev_m": numeric_summary(p95s),
        "mask_valid_fraction": numeric_summary(valid_fracs),
    }


def visible_surface_depth_residual(candidate_data: dict[str, Any], object_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    residuals = []
    for row in object_rows:
        geom = row.get("visible_geometry") if isinstance(row.get("visible_geometry"), dict) else {}
        camera_points = geom.get("camera_vertices_sample_m")
        intr = geom.get("intrinsics_fx_fy_cx_cy")
        if camera_points is None or intr is None:
            continue
        frame_idx = int(row["frame_idx"])
        depth_i = candidate_data["frame_to_i"].get(frame_idx)
        if depth_i is None:
            continue
        pts = np.asarray(camera_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue
        if len(pts) > int(args.max_surface_points_per_frame):
            pts = pts[np.linspace(0, len(pts) - 1, int(args.max_surface_points_per_frame), dtype=int)]
        depth = candidate_data["depth"][depth_i]
        xy = np.empty((len(pts), 2), dtype=float)
        fx, fy, cx, cy = np.asarray(intr, dtype=float).reshape(4)
        xy[:, 0] = fx * pts[:, 0] / np.maximum(pts[:, 2], 1.0e-9) + cx
        xy[:, 1] = fy * pts[:, 1] / np.maximum(pts[:, 2], 1.0e-9) + cy
        sampled = sample_depth_at_points(depth, xy, frame_source_size(row["frame"], depth))
        valid = np.isfinite(sampled) & (sampled > 0) & np.isfinite(pts[:, 2]) & (pts[:, 2] > 0)
        residuals.extend((sampled[valid] - pts[valid, 2]).astype(float).tolist())
    return {"evaluated_points": len(residuals), "candidate_minus_visible_surface_depth_m": numeric_summary(residuals), "abs_median_m": robust_abs_median(residuals)}


def hand_depth_residual(candidate_data: dict[str, Any], hand_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    residuals = []
    for row in hand_rows:
        frame_idx = int(row["frame_idx"])
        depth_i = candidate_data["frame_to_i"].get(frame_idx)
        if depth_i is None:
            continue
        depth = candidate_data["depth"][depth_i]
        joints2d = np.asarray(row["joints2d"], dtype=float)
        joints3d = np.asarray(row["joints3d"], dtype=float)
        if joints2d.ndim != 2 or joints2d.shape[1] != 2 or joints3d.ndim != 2 or joints3d.shape[1] != 3:
            continue
        n = min(len(joints2d), len(joints3d), 21)
        sampled = sample_depth_at_points(depth, joints2d[:n], frame_source_size(row["frame"], depth))
        model_depth = joints3d[:n, 2]
        valid = np.isfinite(sampled) & (sampled > 0) & np.isfinite(model_depth) & (model_depth > 0)
        if np.count_nonzero(valid) < int(args.min_hand_depth_joints):
            continue
        residuals.extend((sampled[valid] - model_depth[valid]).astype(float).tolist())
    return {"evaluated_joint_samples": len(residuals), "candidate_minus_hand_depth_m": numeric_summary(residuals), "abs_median_m": robust_abs_median(residuals)}


def temporal_smoothness(candidate_data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    depths = candidate_data["depth"]
    frame_idx = candidate_data["frame_idx"]
    deltas = []
    max_pairs = min(depths.shape[0] - 1, int(args.max_temporal_pairs))
    for i in range(max_pairs):
        if int(frame_idx[i + 1]) != int(frame_idx[i]) + 1:
            continue
        a = depths[i]
        b = depths[i + 1]
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if np.count_nonzero(valid) < int(args.min_temporal_pixels):
            continue
        vals = np.abs(a[valid] - b[valid])
        if vals.size > int(args.max_temporal_pixels):
            vals = vals[np.linspace(0, vals.size - 1, int(args.max_temporal_pixels), dtype=int)]
        deltas.append(float(np.median(vals)))
    return {"evaluated_pairs": len(deltas), "median_abs_frame_delta_m": numeric_summary(deltas), "abs_median_m": robust_abs_median(deltas)}


def contact_gap_residual(candidate_data: dict[str, Any], contact_report: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    if contact_report is None:
        return {"evaluated_rows": 0, "reason": "no_contact_report"}
    rows = contact_report.get("rows") or contact_report.get("contact_rows") or contact_report.get("constraint_rows")
    if not isinstance(rows, list):
        return {"evaluated_rows": 0, "reason": "contact_report_has_no_rows"}
    gaps = []
    for row in rows[: int(args.max_contact_rows)]:
        if not isinstance(row, dict):
            continue
        frame_idx = row.get("frame_idx")
        hand_xy = row.get("hand_point_2d_px") or row.get("hand_xy_px")
        object_xy = row.get("object_point_2d_px") or row.get("object_xy_px")
        if frame_idx is None or hand_xy is None or object_xy is None:
            continue
        depth_i = candidate_data["frame_to_i"].get(int(frame_idx))
        if depth_i is None:
            continue
        depth = candidate_data["depth"][depth_i]
        hand_depth = sample_depth_at_points(depth, np.asarray(hand_xy, dtype=float).reshape(1, 2), [depth.shape[1], depth.shape[0]])[0]
        obj_depth = sample_depth_at_points(depth, np.asarray(object_xy, dtype=float).reshape(1, 2), [depth.shape[1], depth.shape[0]])[0]
        if np.isfinite(hand_depth) and np.isfinite(obj_depth) and hand_depth > 0 and obj_depth > 0:
            gaps.append(float(abs(hand_depth - obj_depth)))
    return {"evaluated_rows": len(gaps), "contact_or_near_contact_depth_gap_m": numeric_summary(gaps), "abs_median_m": robust_abs_median(gaps)}


def score_candidate(candidate: dict[str, Any], residuals: dict[str, Any], args: argparse.Namespace) -> tuple[float, list[str]]:
    terms: list[float] = []
    reasons: list[str] = []
    valid_fraction = candidate.get("valid_fraction")
    if valid_fraction is not None:
        terms.append(float(args.w_valid_fraction) * (1.0 - float(valid_fraction)))
        reasons.append("valid_fraction")
    mask_mad = residuals["object_mask_continuity"].get("mask_depth_mad_m", {}).get("median")
    if mask_mad is not None:
        terms.append(float(args.w_mask_continuity) * min(float(mask_mad), float(args.clip_depth_residual_m)) / float(args.sigma_mask_continuity_m))
        reasons.append("object_mask_continuity")
    surface_abs = residuals["visible_surface_depth_residual"].get("abs_median_m")
    if surface_abs is not None:
        terms.append(float(args.w_visible_surface) * min(float(surface_abs), float(args.clip_depth_residual_m)) / float(args.sigma_visible_surface_m))
        reasons.append("visible_surface")
    hand_abs = residuals["hand_depth_residual"].get("abs_median_m")
    if hand_abs is not None:
        terms.append(float(args.w_hand_depth) * min(float(hand_abs), float(args.clip_depth_residual_m)) / float(args.sigma_hand_depth_m))
        reasons.append("hand_depth")
    temporal_abs = residuals["temporal_smoothness"].get("abs_median_m")
    if temporal_abs is not None:
        terms.append(float(args.w_temporal) * min(float(temporal_abs), float(args.clip_depth_residual_m)) / float(args.sigma_temporal_m))
        reasons.append("temporal_smoothness")
    contact_abs = residuals["contact_gap_residual"].get("abs_median_m")
    if contact_abs is not None:
        terms.append(float(args.w_contact_gap) * min(float(contact_abs), float(args.clip_depth_residual_m)) / float(args.sigma_contact_gap_m))
        reasons.append("contact_gap")
    prior_weight = max(float(candidate.get("prior_weight", 1.0)), 1.0e-6)
    if not terms:
        return float("inf"), ["no_discriminating_residual_available"]
    return float(sum(terms) / prior_weight), reasons


def choose_status(score: float, residuals: dict[str, Any], args: argparse.Namespace) -> str:
    if not np.isfinite(score):
        return "unresolved_no_residuals"
    surface_abs = residuals["visible_surface_depth_residual"].get("abs_median_m")
    hand_abs = residuals["hand_depth_residual"].get("abs_median_m")
    if surface_abs is not None and float(surface_abs) <= float(args.primary_surface_threshold_m):
        return "primary_for_visible_surface"
    if hand_abs is not None and float(hand_abs) <= float(args.primary_hand_threshold_m):
        return "primary_for_hand_depth"
    if score <= float(args.retained_score_threshold):
        return "retained_uncertain"
    return "rejected_residual_conflict"


def select(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_registry_candidates(args.registry)
    if not candidates:
        raise ContractError("v20_depth_selector_failed: registry_has_no_candidates")
    annotations = load_json(args.annotations) if args.annotations else None
    contact_report = load_json(args.contact_report) if args.contact_report else None
    object_rows = collect_object_masks(annotations, int(args.max_object_frames)) if isinstance(annotations, dict) else []
    hand_rows = collect_hand_rows(annotations, int(args.max_hand_rows)) if isinstance(annotations, dict) else []
    evaluated = []
    for candidate in candidates:
        if candidate.get("evaluation_reference_allowed_in_prediction") is True or "gt" in str(candidate.get("candidate_id", "")).lower() or "ground_truth" in str(candidate.get("candidate_id", "")).lower() or "oracle" in str(candidate.get("candidate_id", "")).lower():
            raise ContractError(f"reference_depth_candidate_forbidden_in_selector: {candidate.get('candidate_id')}")
        data = load_depth_candidate_depth(candidate)
        residuals = {
            "object_mask_continuity": object_mask_continuity(data, object_rows, args),
            "visible_surface_depth_residual": visible_surface_depth_residual(data, object_rows, args),
            "hand_depth_residual": hand_depth_residual(data, hand_rows, args),
            "temporal_smoothness": temporal_smoothness(data, args),
            "contact_gap_residual": contact_gap_residual(data, contact_report, args),
        }
        score, score_terms = score_candidate(candidate, residuals, args)
        status = choose_status(score, residuals, args)
        evaluated.append({
            "candidate_id": candidate.get("candidate_id"),
            "score": score if np.isfinite(score) else None,
            "score_terms": score_terms,
            "selection_status": status,
            "candidate": candidate,
            "residuals": residuals,
        })
    finite_rows = [row for row in evaluated if row["score"] is not None]
    finite_rows.sort(key=lambda row: float(row["score"]))
    selected: dict[str, str | None] = {
        "primary_for_visible_surface": None,
        "primary_for_hand_depth": None,
        "secondary_scale_anchor": None,
    }
    if finite_rows:
        best_id = str(finite_rows[0]["candidate_id"])
        selected["primary_for_visible_surface"] = best_id
        selected["primary_for_hand_depth"] = best_id
        if len(finite_rows) > 1:
            selected["secondary_scale_anchor"] = str(finite_rows[1]["candidate_id"])
    if args.require_selected and selected["primary_for_visible_surface"] is None:
        raise ContractError("v20_depth_selector_failed: no_candidate_had_discriminating_residuals")
    report = {
        "schema": "v20_depth_selection_report.v0",
        "claim_scope": "Selector compares prediction-side depth candidates using physical residuals; weak candidates are retained or downweighted, not hidden.",
        "selected": selected,
        "evaluated_candidate_count": len(evaluated),
        "object_rows_evaluated": len(object_rows),
        "hand_rows_evaluated": len(hand_rows),
        "candidates": evaluated,
        "selection_policy": {
            "best_score_primary": True,
            "no_eval_ref_prediction": True,
            "weak_measurements_continue_as_retained_uncertain": True,
        },
    }
    bundle = {
        "schema": "v20_observation_bundle.v0",
        "depth": {
            "candidate_registry": str(args.registry),
            "selection_report": str(args.output_report),
            "primary_depth_by_scope": selected,
            "evaluated_candidate_count": len(evaluated),
        },
    }
    write_json(args.output_report, report)
    write_json(args.output_bundle, bundle)
    return {"report": report, "bundle": bundle}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V20 depth candidates and select the observation bundle depth by residuals.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--contact-report", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--require-selected", action="store_true")
    parser.add_argument("--max-object-frames", type=int, default=80)
    parser.add_argument("--max-hand-rows", type=int, default=160)
    parser.add_argument("--max-surface-points-per-frame", type=int, default=512)
    parser.add_argument("--min-mask-depth-pixels", type=int, default=64)
    parser.add_argument("--min-hand-depth-joints", type=int, default=8)
    parser.add_argument("--max-temporal-pairs", type=int, default=80)
    parser.add_argument("--min-temporal-pixels", type=int, default=4096)
    parser.add_argument("--max-temporal-pixels", type=int, default=20000)
    parser.add_argument("--max-contact-rows", type=int, default=1000)
    parser.add_argument("--clip-depth-residual-m", type=float, default=0.25)
    parser.add_argument("--sigma-mask-continuity-m", type=float, default=0.08)
    parser.add_argument("--sigma-visible-surface-m", type=float, default=0.04)
    parser.add_argument("--sigma-hand-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-temporal-m", type=float, default=0.08)
    parser.add_argument("--sigma-contact-gap-m", type=float, default=0.025)
    parser.add_argument("--w-valid-fraction", type=float, default=0.5)
    parser.add_argument("--w-mask-continuity", type=float, default=1.0)
    parser.add_argument("--w-visible-surface", type=float, default=2.0)
    parser.add_argument("--w-hand-depth", type=float, default=1.5)
    parser.add_argument("--w-temporal", type=float, default=0.5)
    parser.add_argument("--w-contact-gap", type=float, default=1.0)
    parser.add_argument("--primary-surface-threshold-m", type=float, default=0.04)
    parser.add_argument("--primary-hand-threshold-m", type=float, default=0.05)
    parser.add_argument("--retained-score-threshold", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = select(parse_args())
    print(result["report"]["selected"])
