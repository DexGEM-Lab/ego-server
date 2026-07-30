#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from v20_common import ContractError, load_json, numeric_summary, write_json


def hand_rows_from_annotations(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = []
    for frame in payload.get("frames", []) if isinstance(payload, dict) else []:
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        if frame_idx < 0:
            continue
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else hand
            side = str(hand.get("side") or metric.get("side") or "unknown")
            track_id = str(hand.get("hand_track_id") or hand.get("hand_id") or f"hand:{side}")
            betas = metric.get("betas") or metric.get("mano_betas") or hand.get("betas") or hand.get("mano_betas")
            scale = metric.get("hand_scale") or metric.get("scale") or hand.get("hand_scale") or hand.get("scale")
            joints = metric.get("joints_current_v18_camera_m") or metric.get("joints3d_camera") or hand.get("joints3d_camera")
            vertices = metric.get("vertices_current_v18_camera_m") or metric.get("vertices_camera") or hand.get("vertices_camera")
            rows.append({
                "frame_idx": frame_idx,
                "side": side,
                "track_id": track_id,
                "betas": betas,
                "scale": scale,
                "joints": joints,
                "vertices": vertices,
                "visibility": hand.get("visibility") or metric.get("visibility") or "unresolved",
            })
    return rows


def load_depth_refit_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ContractError(f"depth_refit_report_not_object: {path}")
    return payload


def span_from_joints(joints_raw: Any) -> float | None:
    joints = np.asarray(joints_raw if joints_raw is not None else [], dtype=float)
    if joints.shape != (21, 3) or not np.isfinite(joints).all():
        return None
    return float(np.linalg.norm(joints[12] - joints[0]))


def width_from_vertices(vertices_raw: Any) -> float | None:
    verts = np.asarray(vertices_raw if vertices_raw is not None else [], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 16 or not np.isfinite(verts).all():
        return None
    extent = verts.max(axis=0) - verts.min(axis=0)
    return float(np.partition(extent, -2)[-2])


def track_support(rows: list[dict[str, Any]], depth_refit: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    betas = []
    scales = []
    spans = []
    widths = []
    support_frames = []
    occluded_frames = []
    for row in rows:
        visibility = str(row.get("visibility") or "")
        if visibility in {"occluded", "out_of_frame"}:
            occluded_frames.append(int(row["frame_idx"]))
            continue
        beta = np.asarray(row.get("betas") if row.get("betas") is not None else [], dtype=float)
        if beta.ndim == 1 and beta.size > 0 and np.isfinite(beta).all():
            betas.append(beta[: int(args.max_betas)])
        if row.get("scale") is not None:
            try:
                val = float(row["scale"])
                if np.isfinite(val) and val > 0:
                    scales.append(val)
            except Exception:
                pass
        span = span_from_joints(row.get("joints"))
        if span is not None:
            spans.append(span)
        width = width_from_vertices(row.get("vertices"))
        if width is not None:
            widths.append(width)
        support_frames.append(int(row["frame_idx"]))
    depth_summary = None
    if isinstance(depth_refit, dict):
        depth_summary = {
            "status": depth_refit.get("status"),
            "hand_scale": depth_refit.get("hand_scale"),
            "after": depth_refit.get("after"),
            "good_keypoint_subset": depth_refit.get("good_keypoint_subset"),
        }
        if depth_refit.get("hand_scale") is not None:
            scales.append(float(depth_refit["hand_scale"]))
    beta_estimate = None
    beta_unc = None
    if betas:
        max_len = max(len(beta) for beta in betas)
        padded = []
        for beta in betas:
            if len(beta) < max_len:
                beta = np.pad(beta, (0, max_len - len(beta)))
            padded.append(beta)
        stack = np.vstack(padded)
        beta_estimate = np.median(stack, axis=0)
        beta_unc = np.std(stack, axis=0) if len(stack) > 1 else np.full(max_len, float(args.default_beta_sigma))
    scale_estimate = float(np.median(scales)) if scales else 1.0
    scale_sigma = float(np.std(scales)) if len(scales) > 1 else float(args.default_scale_sigma)
    residual_summary = {
        "visible_span_m": numeric_summary(spans),
        "visible_width_proxy_m": numeric_summary(widths),
        "depth_refit": depth_summary,
    }
    support_ok = len(support_frames) >= int(args.min_support_frames)
    beta_ok = beta_estimate is not None
    span_values = np.asarray(spans, dtype=float)
    span_ok = bool(span_values.size == 0 or (np.percentile(span_values, 5) >= args.min_span_m and np.percentile(span_values, 95) <= args.max_span_m))
    scale_ok = bool(args.min_scale <= scale_estimate <= args.max_scale)
    if not support_ok:
        status = "unresolved_insufficient_visible_shape_support"
    elif not beta_ok:
        status = "unresolved_no_mano_betas_candidate"
    elif not span_ok or not scale_ok:
        status = "retained_shape_posterior_candidate_physical_prior_conflict"
    else:
        status = "retained_shape_posterior_candidate"
    return {
        "betas_estimate": beta_estimate,
        "betas_sigma": beta_unc,
        "scale_estimate": scale_estimate,
        "scale_sigma": scale_sigma,
        "support_frames": sorted(set(support_frames)),
        "occluded_frames_not_constraining_shape": sorted(set(occluded_frames)),
        "residual_summary": residual_summary,
        "promotion_status": status,
    }


def solve(args: argparse.Namespace) -> dict[str, Any]:
    rows = hand_rows_from_annotations(args.annotations)
    if not rows:
        raise ContractError("v20_hand_shape_solve_failed: annotations_contain_no_hand_rows")
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)
    depth_refit = load_depth_refit_report(args.depth_refit_report)
    reports = []
    arrays: dict[str, np.ndarray] = {}
    for track_id, track_rows in sorted(by_track.items()):
        support = track_support(track_rows, depth_refit, args)
        beta = support.pop("betas_estimate")
        beta_sigma = support.pop("betas_sigma")
        record = {
            "hand_track_id": track_id,
            "side": track_rows[0].get("side", "unknown"),
            "betas_estimate": beta.astype(float).tolist() if beta is not None else None,
            "betas_sigma": beta_sigma.astype(float).tolist() if beta_sigma is not None else None,
            **support,
            "uncertainty": {
                "scale_sigma": support["scale_sigma"],
                "source": "track_level_non_gt_mano_candidates_depth_refit_and_visible_geometry",
                "occlusion_policy": "occluded/out-of-frame rows do not constrain shape",
            },
        }
        if beta is not None:
            arrays[f"{track_id.replace(':', '_').replace('/', '_')}_betas"] = beta.astype(np.float32)
        reports.append(record)
    output = {
        "schema": "v20_hand_shape_solve_report.v0",
        "claim_scope": "Track-level MANO shape posterior from non-GT MANO candidates and residual summaries; it is a prior for later MANO interval solve, not a certain hand state.",
        "tracks": reports,
        "track_count": len(reports),
    }
    write_json(args.output_report, output)
    if arrays:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, **arrays)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve V20 track-level MANO betas/scale posterior from non-GT hand candidates.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-refit-report", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--min-support-frames", type=int, default=5)
    parser.add_argument("--max-betas", type=int, default=10)
    parser.add_argument("--default-beta-sigma", type=float, default=0.75)
    parser.add_argument("--default-scale-sigma", type=float, default=0.04)
    parser.add_argument("--min-scale", type=float, default=0.80)
    parser.add_argument("--max-scale", type=float, default=1.20)
    parser.add_argument("--min-span-m", type=float, default=0.10)
    parser.add_argument("--max-span-m", type=float, default=0.23)
    return parser.parse_args()


if __name__ == "__main__":
    print(solve(parse_args())["track_count"])
