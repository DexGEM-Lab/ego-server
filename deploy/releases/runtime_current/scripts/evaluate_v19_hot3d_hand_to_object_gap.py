#!/usr/bin/env python3
"""Attribute V19 hand/object gap to hand error versus object/scene error.

Given a V19 surface-posterior state, this script compares three quantities on the
same frame/side/object-target correspondences:

1. V19 source MANO selected vertices to V19 object targets.
2. HOT3D GT MANO selected vertices, with the same MANO vertex ids, to the same
   V19 object targets after both are transformed to camera coordinates.
3. V19 source selected vertices to HOT3D GT MANO selected vertices.

If (2) remains large while (3) is much smaller, the source hand estimate is not
large enough to explain the hand/object gap.  The remaining mechanism is object
pose/geometry/camera alignment or true non-contact, not a contact-biased MANO
correction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_v19_hot3d_hawor_mano3d import (  # type: ignore
    load_hand_shape,
    load_json,
    load_smplx_mano,
    replay_hot3d_mano,
    se3_from_hot3d_dict,
    summarize,
    world_to_camera,
)


def load_annotation_cameras(annotation_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ann = load_json(annotation_path)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pos, fr in enumerate(ann.get("frames", [])):
        if not isinstance(fr, dict):
            continue
        cam = fr.get("camera") if isinstance(fr.get("camera"), dict) else {}
        T_raw = cam.get("T_world_camera_metric") or cam.get("T_world_camera")
        if T_raw is None:
            continue
        T = np.asarray(T_raw, dtype=np.float64)
        if T.shape != (4, 4):
            continue
        out[int(fr.get("frame_idx", pos))] = (T[:3, :3].copy(), T[:3, 3].copy())
    if not out:
        raise RuntimeError(f"no camera transforms in {annotation_path}")
    return out


def values_from_rows(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return vals


def compatibility(gap: float, sigma: float) -> float:
    if not math.isfinite(gap) or sigma <= 0:
        return float("nan")
    return math.exp(-0.5 * (gap / sigma) ** 2)


def quantile(vals: list[float], p: float) -> float | None:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-state", required=True, type=Path)
    ap.add_argument("--hot3d-gt", required=True, type=Path)
    ap.add_argument("--mano-left", required=True, type=Path)
    ap.add_argument("--mano-right", required=True, type=Path)
    ap.add_argument("--stream-id", default="214-1")
    ap.add_argument("--output-report", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hand-sigma-m", type=float, default=0.027)
    ap.add_argument("--object-sigma-m", type=float, default=0.010)
    ap.add_argument("--depth-order-sigma-m", type=float, default=0.010)
    args = ap.parse_args()

    state = load_json(args.surface_state)
    state_inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    annotation_path = Path(str(state_inputs.get("annotations") or state.get("annotations") or ""))
    if not annotation_path.exists():
        raise FileNotFoundError(f"missing annotations path from state inputs: {annotation_path}")
    v19_cameras = load_annotation_cameras(annotation_path)

    gt = load_json(args.hot3d_gt)
    gt_frames = {int(fr["frame_idx"]): fr for fr in gt.get("frames", []) if isinstance(fr, dict)}
    beta, hand_shape_source = load_hand_shape(gt, args.hot3d_gt)
    layers = load_smplx_mano(args)
    device = torch.device(args.device)
    for layer in layers.values():
        layer.to(device)
        layer.eval()

    sigma = math.sqrt(args.hand_sigma_m**2 + args.object_sigma_m**2 + args.depth_order_sigma_m**2)
    rows_out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    gt_cache: dict[tuple[int, str], np.ndarray] = {}
    gt_cam_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for row in state.get("per_frame_states", []):
        if not isinstance(row, dict):
            continue
        frame = int(row.get("frame_idx", -1))
        side = str(row.get("hand_side"))
        if side not in {"left", "right"}:
            continue
        if frame not in v19_cameras or frame not in gt_frames:
            skipped.append({"frame_idx": frame, "side": side, "reason": "missing_camera_or_gt_frame"})
            continue
        src_ids = np.asarray(row.get("object_surface_posterior_source_mano_vertex_ids") or [], dtype=np.int64)
        source_w = np.asarray(row.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
        target_w = np.asarray(row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
        n = min(len(src_ids), len(source_w), len(target_w))
        if n <= 0:
            skipped.append({"frame_idx": frame, "side": side, "reason": "missing_source_target_vertices"})
            continue
        src_ids = src_ids[:n]
        source_w = source_w[:n]
        target_w = target_w[:n]
        if np.any(src_ids < 0):
            skipped.append({"frame_idx": frame, "side": side, "reason": "invalid_vertex_ids"})
            continue

        gt_row = gt_frames[frame]
        hand = gt_row.get("json", {}).get("hands.json", {}).get(side)
        hot3d_cam = gt_row.get("json", {}).get("cameras.json", {}).get(args.stream_id)
        if not isinstance(hand, dict) or "mano_pose" not in hand or not isinstance(hot3d_cam, dict):
            skipped.append({"frame_idx": frame, "side": side, "reason": "missing_hot3d_hand_or_camera"})
            continue
        if frame not in gt_cam_cache:
            gt_cam_cache[frame] = se3_from_hot3d_dict(hot3d_cam["T_world_from_camera"])
        key = (frame, side)
        if key not in gt_cache:
            mano = hand["mano_pose"]
            gt_verts_w, _, _ = replay_hot3d_mano(
                layers[side],
                beta,
                np.asarray(mano["thetas"], dtype=np.float32),
                np.asarray(mano["wrist_xform"], dtype=np.float32),
                device,
            )
            gt_cache[key] = gt_verts_w.astype(np.float64)
        gt_verts_w = gt_cache[key]
        if int(src_ids.max()) >= gt_verts_w.shape[0]:
            skipped.append({"frame_idx": frame, "side": side, "reason": "vertex_id_out_of_range", "max_vertex_id": int(src_ids.max())})
            continue

        R_v19, t_v19 = v19_cameras[frame]
        source_cam = world_to_camera(source_w, R_v19, t_v19)
        target_cam = world_to_camera(target_w, R_v19, t_v19)
        R_gt, t_gt = gt_cam_cache[frame]
        gt_selected_cam = world_to_camera(gt_verts_w[src_ids], R_gt, t_gt)

        source_gap = np.linalg.norm(source_cam - target_cam, axis=1)
        gt_gap = np.linalg.norm(gt_selected_cam - target_cam, axis=1)
        source_to_gt = np.linalg.norm(source_cam - gt_selected_cam, axis=1)
        delta = gt_gap - source_gap
        out = {
            "frame_idx": frame,
            "side": side,
            "vertex_count": int(n),
            "v19_source_gap_median_m": float(np.median(source_gap)),
            "hot3d_gt_hand_to_v19_object_gap_median_m": float(np.median(gt_gap)),
            "gt_minus_v19_source_gap_median_delta_m": float(np.median(delta)),
            "v19_source_to_hot3d_gt_selected_vertex_shift_median_m": float(np.median(source_to_gt)),
            "v19_source_gap_p10_m": float(np.percentile(source_gap, 10)),
            "hot3d_gt_gap_p10_m": float(np.percentile(gt_gap, 10)),
            "v19_source_gap_min_m": float(np.min(source_gap)),
            "hot3d_gt_gap_min_m": float(np.min(gt_gap)),
            "v19_source_contact_compatibility": compatibility(float(np.median(source_gap)), sigma),
            "hot3d_gt_contact_compatibility": compatibility(float(np.median(gt_gap)), sigma),
            "gt_visibility_modeled": hand.get("visibilities_modeled", {}).get(args.stream_id) if isinstance(hand.get("visibilities_modeled"), dict) else None,
        }
        rows_out.append(out)

    if not rows_out:
        raise RuntimeError(f"no evaluable rows; skipped preview {skipped[:10]}")

    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sr = [r for r in rows_out if r["side"] == side]
        by_side[side] = {
            "row_count": len(sr),
            "frame_min": min([r["frame_idx"] for r in sr], default=None),
            "frame_max": max([r["frame_idx"] for r in sr], default=None),
            "v19_source_gap_median_m": summarize(values_from_rows(sr, "v19_source_gap_median_m")),
            "hot3d_gt_hand_to_v19_object_gap_median_m": summarize(values_from_rows(sr, "hot3d_gt_hand_to_v19_object_gap_median_m")),
            "v19_source_to_hot3d_gt_selected_vertex_shift_median_m": summarize(values_from_rows(sr, "v19_source_to_hot3d_gt_selected_vertex_shift_median_m")),
            "gt_minus_v19_source_gap_median_delta_m": summarize(values_from_rows(sr, "gt_minus_v19_source_gap_median_delta_m")),
        }

    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_hand_to_object_gap",
        "claim_scope": "attribution only: compares V19 source MANO and HOT3D GT MANO selected vertices to the same V19 object targets in camera coordinates; does not certify object pose/contact/nonpenetration",
        "inputs": {
            "surface_state": str(args.surface_state),
            "annotations": str(annotation_path),
            "hot3d_gt": str(args.hot3d_gt),
            "stream_id": args.stream_id,
            "mano_left": str(args.mano_left),
            "mano_right": str(args.mano_right),
            "hand_shape_source": hand_shape_source,
        },
        "compatibility_model": {
            "basis": "sqrt(hand_sigma^2 + object_sigma^2 + depth_order_sigma^2)",
            "combined_sigma_m": sigma,
            "hand_sigma_m": args.hand_sigma_m,
            "object_sigma_m": args.object_sigma_m,
            "depth_order_sigma_m": args.depth_order_sigma_m,
        },
        "row_count": len(rows_out),
        "skipped_count": len(skipped),
        "summary": {
            "v19_source_gap_median_m": summarize(values_from_rows(rows_out, "v19_source_gap_median_m")),
            "hot3d_gt_hand_to_v19_object_gap_median_m": summarize(values_from_rows(rows_out, "hot3d_gt_hand_to_v19_object_gap_median_m")),
            "gt_minus_v19_source_gap_median_delta_m": summarize(values_from_rows(rows_out, "gt_minus_v19_source_gap_median_delta_m")),
            "v19_source_to_hot3d_gt_selected_vertex_shift_median_m": summarize(values_from_rows(rows_out, "v19_source_to_hot3d_gt_selected_vertex_shift_median_m")),
            "v19_source_contact_compatibility": summarize(values_from_rows(rows_out, "v19_source_contact_compatibility")),
            "hot3d_gt_contact_compatibility": summarize(values_from_rows(rows_out, "hot3d_gt_contact_compatibility")),
            "v19_rows_with_gap_over_3sigma": int(sum(r["v19_source_gap_median_m"] > 3 * sigma for r in rows_out)),
            "hot3d_gt_rows_with_gap_over_3sigma": int(sum(r["hot3d_gt_hand_to_v19_object_gap_median_m"] > 3 * sigma for r in rows_out)),
            "hot3d_gt_rows_with_gap_below_v19_source_by_more_than_30mm": int(sum(r["gt_minus_v19_source_gap_median_delta_m"] < -0.030 for r in rows_out)),
            "hot3d_gt_rows_with_gap_above_v19_source_by_more_than_30mm": int(sum(r["gt_minus_v19_source_gap_median_delta_m"] > 0.030 for r in rows_out)),
        },
        "by_side": by_side,
        "interpretation_notes": [
            "If HOT3D GT hand-to-V19-object gap remains close to V19 source gap while source-to-GT hand shift is much smaller, hand localization error is not sufficient to explain the contact gap.",
            "If HOT3D GT gap collapses while V19 source gap is large, the hand foundation is a plausible root cause for the gap.",
            "This keeps object targets fixed; it does not use HOT3D object CAD and cannot certify object geometry or contact.",
        ],
        "rows": rows_out,
        "skipped_preview": skipped[:40],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "row_count": len(rows_out), "summary": report["summary"], "by_side": by_side}, indent=2))


if __name__ == "__main__":
    main()
