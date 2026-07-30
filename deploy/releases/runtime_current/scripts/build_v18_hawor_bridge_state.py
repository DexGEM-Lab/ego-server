#!/usr/bin/env python3
"""Build a HaWoR-only bridge candidate into the current V18 camera/world convention.

This stage does not run HaWoR and does not accept a V18 physical hand state. It
uses existing HaWoR world MANO outputs where available, converts them through the
HaWoR camera-local frame, then places that camera-local state under the current
V18 per-frame camera pose. The output is candidate evidence plus residuals only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

SIDES = ("left", "right")
SIDE_TO_INT = {"left": 0, "right": 1}
INT_TO_SIDE = {0: "left", 1: "right"}
EXPECTED_VERTICES = 778
EXPECTED_JOINTS = 21
DEFAULT_HAWOR_OUTPUTS = {
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/trash_1050_tailrepair_padded/hawor_world_hands_trimmed_1050_with_track_support.npz"),
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands_with_track_support.npz"),
}
DEFAULT_COMPLETE_DEPTH_ROOT = Path("/data2/ego_annotation_outputs/v18_unidepth_extension/complete_depth_root")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    if points_camera.ndim != 2 or points_camera.shape[1] != 3 or intrinsics.shape != (4,):
        return None
    if np.any(points_camera[:, 2] <= 1e-6):
        return None
    fx, fy, cx, cy = intrinsics.astype(float)
    out = np.empty((len(points_camera), 2), dtype=np.float64)
    out[:, 0] = fx * points_camera[:, 0] / points_camera[:, 2] + cx
    out[:, 1] = fy * points_camera[:, 1] / points_camera[:, 2] + cy
    if not np.isfinite(out).all():
        return None
    return out


def project_current_mano_candidate(hand: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else None
    if mano is None:
        return None, None
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics")
    source = str(mano.get("source")) if mano.get("source") is not None else None
    if source and source.startswith("HaWoR_metric_MANO"):
        return None, "self_hawor_candidate_not_independent_reference"
    if not (isinstance(joints, list) and len(joints) == EXPECTED_JOINTS and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return None, source
    points = np.asarray(joints, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)[None, :]
    return project(points, np.asarray(intr, dtype=np.float64)), source


def current_hands_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        side = str(hand.get("hand_side") or hand.get("side") or "").lower()
        if side in SIDES:
            out[side] = hand
    return out


def bbox_contains_fraction(points2d: np.ndarray, bbox: Any) -> float | None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    vals = np.asarray([finite_float(v) for v in bbox], dtype=np.float64)
    if not np.isfinite(vals).all():
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    inside = (points2d[:, 0] >= x0) & (points2d[:, 0] <= x1) & (points2d[:, 1] >= y0) & (points2d[:, 1] <= y1)
    return float(np.mean(inside))


def load_complete_depth(case: str, root: Path) -> dict[str, Any]:
    path = root / case / "unidepth_metric" / "unidepth_metric_depth_v3.npz"
    if not path.exists():
        return {"path": str(path), "frame_to_row": {}, "depth": None, "intrinsics": None}
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "frame_to_row": {int(frame): i for i, frame in enumerate(np.asarray(z["frame_idx"]))},
        "depth": np.asarray(z["depth"], dtype=np.float64),
        "intrinsics": np.asarray(z["intrinsics_fx_fy_cx_cy"], dtype=np.float64),
    }


def scaled_bbox_for_depth(bbox: Any, depth_shape: tuple[int, int]) -> np.ndarray | None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    vals = np.asarray([finite_float(v) for v in bbox], dtype=np.float64)
    if not np.isfinite(vals).all():
        return None
    scale = np.asarray([depth_shape[1] / 1920.0, depth_shape[0] / 1080.0, depth_shape[1] / 1920.0, depth_shape[0] / 1080.0], dtype=np.float64)
    return vals * scale


def estimate_hawor_to_v18_depth_scale(vertices_camera: np.ndarray, depth_row: np.ndarray | None, intrinsics: np.ndarray | None, bbox_fullres: Any) -> dict[str, Any]:
    if depth_row is None or intrinsics is None or vertices_camera.ndim != 2 or vertices_camera.shape[1] != 3:
        return {"scale": 1.0, "status": "missing_depth_or_vertices", "sample_count": 0}
    projected = project(vertices_camera, intrinsics)
    if projected is None:
        return {"scale": 1.0, "status": "projection_failed", "sample_count": 0}
    h, w = depth_row.shape[:2]
    valid = np.isfinite(projected).all(axis=1) & np.isfinite(vertices_camera).all(axis=1) & (vertices_camera[:, 2] > 1e-6)
    valid &= (projected[:, 0] >= 0.0) & (projected[:, 0] < float(w)) & (projected[:, 1] >= 0.0) & (projected[:, 1] < float(h))
    scaled_bbox = scaled_bbox_for_depth(bbox_fullres, (h, w))
    if scaled_bbox is not None:
        x0, y0, x1, y1 = scaled_bbox.tolist()
        valid &= (projected[:, 0] >= x0) & (projected[:, 0] <= x1) & (projected[:, 1] >= y0) & (projected[:, 1] <= y1)
    if not np.any(valid):
        return {"scale": 1.0, "status": "no_projected_vertices_in_depth_image", "sample_count": 0}
    pix = np.rint(projected[valid]).astype(np.int64)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    depth_values = depth_row[pix[:, 1], pix[:, 0]].astype(np.float64)
    z_values = vertices_camera[valid, 2].astype(np.float64)
    keep = np.isfinite(depth_values) & np.isfinite(z_values) & (depth_values > 1e-4) & (z_values > 1e-4)
    if int(np.sum(keep)) < 40:
        return {"scale": 1.0, "status": "too_few_depth_samples", "sample_count": int(np.sum(keep))}
    ratios = depth_values[keep] / z_values[keep]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.15) & (ratios < 2.5)]
    if ratios.size < 40:
        return {"scale": 1.0, "status": "too_few_bounded_depth_ratios", "sample_count": int(ratios.size)}
    scale = float(np.median(ratios))
    before = z_values[keep] - depth_values[keep]
    after = z_values[keep] * scale - depth_values[keep]
    return {
        "scale": scale,
        "status": "depth_scaled_from_projected_hawor_vertices_to_unidepth",
        "sample_count": int(ratios.size),
        "scale_p10": float(np.percentile(ratios, 10.0)),
        "scale_p90": float(np.percentile(ratios, 90.0)),
        "median_depth_residual_before_m": float(np.median(before)),
        "median_abs_depth_residual_before_m": float(np.median(np.abs(before))),
        "median_depth_residual_after_m": float(np.median(after)),
        "median_abs_depth_residual_after_m": float(np.median(np.abs(after))),
        "scope": "camera_origin_scale_to_place_hawor_mano_in_current_v18_unidepth_metric_frame_preserves_2d_projection_not_contact_acceptance",
    }


def image_inside_fraction(points2d: np.ndarray, width: float = 1920.0, height: float = 1080.0) -> float:
    inside = (points2d[:, 0] >= 0.0) & (points2d[:, 0] < width) & (points2d[:, 1] >= 0.0) & (points2d[:, 1] < height)
    return float(np.mean(inside))


def visibility_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("current_visibility_state"))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def residual_tail_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row.get("projection_residual_px_median", -1.0)) > threshold]
    if not selected:
        return {"threshold_px": threshold, "count": 0}
    return {
        "threshold_px": threshold,
        "count": int(len(selected)),
        "frame_min": int(min(int(row["frame_idx"]) for row in selected)),
        "frame_max": int(max(int(row["frame_idx"]) for row in selected)),
        "residual_median_px": float(np.median([float(row["projection_residual_px_median"]) for row in selected])),
        "current_visibility_counts": visibility_counts(selected),
        "hawor_projected_inside_image_fraction": summarize([float(row["hawor_projected_inside_image_fraction"]) for row in selected]),
        "reference_projected_inside_image_fraction": summarize([float(row["reference_projected_inside_image_fraction"]) for row in selected]),
        "preview": selected[:24],
    }


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise RuntimeError("Sim3 alignment requires matching Nx3 arrays with at least 3 points")
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    X = source - src_mean
    Y = target - tgt_mean
    src_var = float(np.mean(np.sum(X * X, axis=1)))
    if src_var <= 0.0:
        raise RuntimeError("source trajectory has zero variance")
    cov = (Y.T @ X) / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / src_var)
    t = tgt_mean - scale * (R @ src_mean)
    return scale, R, t


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def camera_alignment_report(frames: list[dict[str, Any]], hawor_npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    current: list[np.ndarray] = []
    hawor: list[np.ndarray] = []
    frame_ids: list[int] = []
    t_c2w = np.asarray(hawor_npz["t_c2w"], dtype=np.float64)
    for frame in frames:
        idx = int(frame.get("frame_idx", -1))
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        pos = camera.get("position_world_m")
        if idx < 0 or idx >= len(t_c2w) or not (isinstance(pos, list) and len(pos) == 3):
            continue
        current.append(np.asarray(pos, dtype=np.float64))
        hawor.append(t_c2w[idx])
        frame_ids.append(idx)
    cur = np.asarray(current, dtype=np.float64)
    haw = np.asarray(hawor, dtype=np.float64)
    if len(cur) < 3:
        return {"status": "insufficient_camera_positions", "frame_count": int(len(cur))}
    scale, rotation, translation = estimate_sim3(haw, cur)
    aligned = apply_sim3(haw, scale, rotation, translation)
    err = np.linalg.norm(aligned - cur, axis=1)
    windows: list[dict[str, Any]] = []
    window = 90
    stride = 30
    for start in range(0, len(frame_ids), stride):
        stop = min(len(frame_ids), start + window)
        if stop - start < 10:
            continue
        try:
            s, r, t = estimate_sim3(haw[start:stop], cur[start:stop])
        except RuntimeError:
            continue
        e = np.linalg.norm(apply_sim3(haw[start:stop], s, r, t) - cur[start:stop], axis=1)
        windows.append({
            "frame_start": int(frame_ids[start]),
            "frame_end": int(frame_ids[stop - 1]),
            "align_frames": int(stop - start),
            "scale": float(s),
            "error_m": summarize(e),
        })
    return {
        "status": "global_sim3_diagnostic_only_not_bridge_acceptance",
        "align_frames": int(len(cur)),
        "global_sim3": {
            "scale": float(scale),
            "rotation": rotation.astype(float).tolist(),
            "translation": translation.astype(float).tolist(),
            "error_m": summarize(err),
        },
        "local_window_sim3_error_median_m": summarize([float(w["error_m"].get("median", float("nan"))) for w in windows]),
        "local_windows_preview": windows[:12],
        "interpretation": "single_global_sim3_is_diagnostic_only; per_frame_camera_local_bridge_uses_current_V18_camera_pose_instead_of_accepting_HaWoR_world_as_current_V18_world",
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann_path = args.source_root / case / "annotations_v18_full.json"
    ann = load_json(ann_path)
    frames = ann.get("frames") if isinstance(ann.get("frames"), list) else []
    frame_count = int(ann.get("frame_count") or len(frames))
    out_dir = args.output_root / "hawor_bridge_state" / case
    out_dir.mkdir(parents=True, exist_ok=True)
    hawor_path = DEFAULT_HAWOR_OUTPUTS.get(case)
    base = {
        "method": "build_v18_hawor_bridge_state",
        "case": case,
        "source_annotations": str(ann_path),
        "expected_frame_count": frame_count,
        "expected_frame_side_rows": frame_count * 2,
        "hard_requirement": "HaWoR_full_timeline_metric_MANO_required_for_V18_physical_hand_state",
        "claim_scope": "HaWoR_bridge_candidate_only_no_WiLoR_substitution_no_contact_or_occlusion_acceptance",
        "accepted_v18_hawor_foundation": False,
        "downstream_physical_modules_recomputed_from_bridge": False,
        "hawor_npz": str(hawor_path) if hawor_path is not None else None,
    }
    if hawor_path is None or not hawor_path.exists():
        report = {
            **base,
            "status": "blocked_no_hawor_npz_for_case",
            "bridge_candidate_npz": None,
            "bridge_candidate_rows": 0,
            "blocking_reasons": ["case_hawor_world_hands_npz_missing", "HaWoR_repo_weights_or_MANO_assets_missing_locally"],
            "elapsed_s": time.perf_counter() - start,
        }
        write_json(out_dir / "v18_hawor_bridge_state_report.json", report)
        return report

    z = np.load(hawor_path, allow_pickle=True)
    depth_bundle = load_complete_depth(case, args.complete_depth_root)
    img_focal = float(np.asarray(z["img_focal"]).reshape(-1)[0]) if "img_focal" in z.files else 2304.0
    intr = np.asarray([img_focal, img_focal, 960.0, 540.0], dtype=np.float64)
    frame_indices: list[int] = []
    side_indices: list[int] = []
    vertices_camera_rows: list[np.ndarray] = []
    joints_camera_rows: list[np.ndarray] = []
    vertices_v18_world_rows: list[np.ndarray] = []
    joints_v18_world_rows: list[np.ndarray] = []
    T_rows: list[np.ndarray] = []
    depth_scale_rows: list[float] = []
    depth_scale_status_rows: list[str] = []
    depth_scale_sample_count_rows: list[int] = []
    depth_scale_reports: list[dict[str, Any]] = []
    projection_median: list[float] = []
    projection_p95: list[float] = []
    bbox_inside: list[float] = []
    median_depth: list[float] = []
    reference_sources: list[str] = []
    row_reports: list[dict[str, Any]] = []
    residual_classification_rows: list[dict[str, Any]] = []
    rows_without_reference = 0
    nonpositive_depth_rows = 0
    skipped_rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx >= len(np.asarray(z["frame_idx"])):
            continue
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            skipped_rows.append({"frame_idx": frame_idx, "side": "all", "reason": "missing_current_v18_T_world_camera_metric"})
            continue
        R_h = np.asarray(z["R_c2w"], dtype=np.float64)[frame_idx]
        t_h = np.asarray(z["t_c2w"], dtype=np.float64)[frame_idx]
        current_hands = current_hands_by_side(frame)
        for side in SIDES:
            if int(np.asarray(z[f"{side}_valid"])[frame_idx]) == 0:
                skipped_rows.append({"frame_idx": frame_idx, "side": side, "reason": "hawor_invalid_row"})
                continue
            vertices_world_h = np.asarray(z[f"{side}_vertices_world_m"], dtype=np.float64)[frame_idx]
            joints_world_h = np.asarray(z[f"{side}_joints_world_m"], dtype=np.float64)[frame_idx]
            vertices_camera_raw = (R_h.T @ (vertices_world_h - t_h[None, :]).T).T
            joints_camera_raw = (R_h.T @ (joints_world_h - t_h[None, :]).T).T
            if vertices_camera_raw.shape != (EXPECTED_VERTICES, 3) or joints_camera_raw.shape != (EXPECTED_JOINTS, 3) or np.any(vertices_camera_raw[:, 2] <= 1e-6) or np.any(joints_camera_raw[:, 2] <= 1e-6):
                nonpositive_depth_rows += 1
                skipped_rows.append({"frame_idx": frame_idx, "side": side, "reason": "invalid_or_nonpositive_hawor_camera_depth"})
                continue
            depth_row = None
            depth_intrinsics = None
            depth_frame_to_row = depth_bundle.get("frame_to_row") if isinstance(depth_bundle.get("frame_to_row"), dict) else {}
            depth_i = depth_frame_to_row.get(frame_idx)
            if isinstance(depth_i, int) and depth_bundle.get("depth") is not None and depth_bundle.get("intrinsics") is not None:
                depth_row = np.asarray(depth_bundle["depth"])[depth_i]
                depth_intrinsics = np.asarray(depth_bundle["intrinsics"])[depth_i]
            depth_scale = estimate_hawor_to_v18_depth_scale(vertices_camera_raw, depth_row, depth_intrinsics, current_hands.get(side, {}).get("bbox_xyxy") if side in current_hands else None)
            scale = finite_float(depth_scale.get("scale"), 1.0)
            vertices_camera = vertices_camera_raw * scale
            joints_camera = joints_camera_raw * scale
            homog_v = np.c_[vertices_camera, np.ones(EXPECTED_VERTICES, dtype=np.float64)]
            homog_j = np.c_[joints_camera, np.ones(EXPECTED_JOINTS, dtype=np.float64)]
            vertices_v18_world = (T @ homog_v.T).T[:, :3]
            joints_v18_world = (T @ homog_j.T).T[:, :3]
            projected = project(joints_camera, intr)
            row: dict[str, Any] = {
                "frame_idx": frame_idx,
                "side": side,
                "median_hawor_camera_depth_m": float(np.median(joints_camera[:, 2])),
                "raw_median_hawor_camera_depth_m": float(np.median(joints_camera_raw[:, 2])),
                "hawor_to_v18_depth_scale": scale,
                "hawor_to_v18_depth_scale_status": depth_scale.get("status"),
                "hawor_to_v18_depth_scale_sample_count": depth_scale.get("sample_count"),
                "hawor_to_v18_depth_scale_report": depth_scale,
            }
            depth_scale_reports.append({"frame_idx": frame_idx, "side": side, **depth_scale})
            median_depth.append(row["median_hawor_camera_depth_m"])
            reference, ref_source = project_current_mano_candidate(current_hands.get(side, {})) if side in current_hands else (None, None)
            if ref_source:
                reference_sources.append(ref_source)
            if projected is not None and reference is not None:
                residual = np.linalg.norm(projected - reference, axis=1)
                med = float(np.median(residual))
                p95 = float(np.percentile(residual, 95.0))
                h_inside = image_inside_fraction(projected)
                r_inside = image_inside_fraction(reference)
                projection_median.append(med)
                projection_p95.append(p95)
                row.update({
                    "reference_projection_available": True,
                    "reference_projection_source_family": "current_v18_visible_hand_candidate_projection_not_HaWoR_substitute",
                    "reference_projection_source_backend": ref_source,
                    "projection_residual_px_median": med,
                    "projection_residual_px_p95": p95,
                    "hawor_projected_inside_image_fraction": h_inside,
                    "reference_projected_inside_image_fraction": r_inside,
                    "current_visibility_state": current_hands.get(side, {}).get("visibility_state"),
                })
                residual_classification_rows.append({
                    "frame_idx": frame_idx,
                    "side": side,
                    "projection_residual_px_median": med,
                    "projection_residual_px_p95": p95,
                    "hawor_projected_inside_image_fraction": h_inside,
                    "reference_projected_inside_image_fraction": r_inside,
                    "current_visibility_state": current_hands.get(side, {}).get("visibility_state"),
                })
            else:
                rows_without_reference += 1
                row["reference_projection_available"] = False
            if projected is not None and side in current_hands:
                inside = bbox_contains_fraction(projected, current_hands[side].get("bbox_xyxy"))
                if inside is not None:
                    bbox_inside.append(inside)
                    row["projected_hawor_joints_inside_current_bbox_fraction"] = inside
            frame_indices.append(frame_idx)
            side_indices.append(SIDE_TO_INT[side])
            vertices_camera_rows.append(vertices_camera.astype(np.float32))
            joints_camera_rows.append(joints_camera.astype(np.float32))
            vertices_v18_world_rows.append(vertices_v18_world.astype(np.float32))
            joints_v18_world_rows.append(joints_v18_world.astype(np.float32))
            T_rows.append(T.astype(np.float32))
            depth_scale_rows.append(float(scale))
            depth_scale_status_rows.append(str(depth_scale.get("status")))
            depth_scale_sample_count_rows.append(int(depth_scale.get("sample_count") or 0))
            if len(row_reports) < 240:
                row_reports.append(row)
    bridge_npz = out_dir / "hawor_bridge_candidates_current_v18_camera_local.npz"
    if frame_indices:
        np.savez_compressed(
            bridge_npz,
            frame_idx=np.asarray(frame_indices, dtype=np.int32),
            side=np.asarray(side_indices, dtype=np.int8),
            side_labels=np.asarray([INT_TO_SIDE[int(s)] for s in side_indices]),
            vertices_hawor_camera_m=np.stack(vertices_camera_rows).astype(np.float32),
            joints_hawor_camera_m=np.stack(joints_camera_rows).astype(np.float32),
            vertices_current_v18_world_from_hawor_camera_local_m=np.stack(vertices_v18_world_rows).astype(np.float32),
            joints_current_v18_world_from_hawor_camera_local_m=np.stack(joints_v18_world_rows).astype(np.float32),
            T_world_camera_metric_current_v18=np.stack(T_rows).astype(np.float32),
            hawor_to_v18_depth_scale=np.asarray(depth_scale_rows, dtype=np.float32),
            hawor_to_v18_depth_scale_status=np.asarray(depth_scale_status_rows),
            hawor_to_v18_depth_scale_sample_count=np.asarray(depth_scale_sample_count_rows, dtype=np.int32),
            source_hawor_npz=np.asarray([str(hawor_path)]),
            source_complete_depth_npz=np.asarray([str(depth_bundle.get("path"))]),
            coordinate_status=np.asarray(["hawor_camera_local_depth_scaled_to_current_v18_unidepth_metric_then_transformed_by_current_v18_T_world_camera_metric_candidate_only"]),
        )
    residual_summary = summarize(projection_median)
    residual_threshold_counts = {
        "reference_rows": int(len(projection_median)),
        "median_px_le_25": int(sum(v <= 25.0 for v in projection_median)),
        "median_px_le_50": int(sum(v <= 50.0 for v in projection_median)),
        "median_px_le_100": int(sum(v <= 100.0 for v in projection_median)),
        "median_px_gt_200": int(sum(v > 200.0 for v in projection_median)),
        "median_px_gt_500": int(sum(v > 500.0 for v in projection_median)),
    }
    blockers = [
        "bridge_candidate_not_consumed_by_contact_occlusion_nonpenetration",
    ]
    if len(frame_indices) < frame_count * 2:
        blockers.append("hawor_bridge_candidate_rows_not_full_timeline")
    if residual_summary.get("count", 0) and float(residual_summary.get("p95", 0.0)) > 200.0:
        blockers.append("projection_residual_tail_too_large_for_foundation_acceptance")
    camera_alignment = camera_alignment_report(frames, z)
    global_err = camera_alignment.get("global_sim3", {}).get("error_m", {}) if isinstance(camera_alignment.get("global_sim3"), dict) else {}
    if float(global_err.get("median", 999.0)) > 0.05:
        blockers.append("single_global_HaWoR_to_V18_world_sim3_alignment_too_loose_for_physical_contact")
    report = {
        **base,
        "status": "trash_hawor_bridge_candidate_built_not_accepted" if case == "trash_1050" else "hawor_bridge_candidate_built_not_accepted",
        "bridge_candidate_npz": str(bridge_npz),
        "bridge_candidate_rows": int(len(frame_indices)),
        "expected_frame_side_rows": int(frame_count * 2),
        "valid_hawor_frame_side_rows": int(len(frame_indices)),
        "rows_without_current_reference_projection": int(rows_without_reference),
        "nonpositive_depth_rows": int(nonpositive_depth_rows),
        "projection_reference_source_family": "current_v18_visible_hand_candidate_projection_used_only_for_residual_measurement_not_requirement_substitution",
        "reference_projection_source_backend_counts": {src: int(reference_sources.count(src)) for src in sorted(set(reference_sources))},
        "reference_projection_residual_px_median_per_row": residual_summary,
        "reference_projection_residual_px_p95_per_row": summarize(projection_p95),
        "reference_projection_residual_threshold_counts": residual_threshold_counts,
        "projection_residual_tail_localization": {
            "median_px_gt_200": residual_tail_summary(residual_classification_rows, 200.0),
            "median_px_gt_500": residual_tail_summary(residual_classification_rows, 500.0),
            "median_px_gt_1000": residual_tail_summary(residual_classification_rows, 1000.0),
        },
        "projected_hawor_joints_inside_current_bbox_fraction": summarize(bbox_inside),
        "median_hawor_camera_depth_m": summarize(median_depth),
        "hawor_to_v18_depth_scale": summarize(depth_scale_rows),
        "hawor_to_v18_depth_scale_status_counts": {status: int(depth_scale_status_rows.count(status)) for status in sorted(set(depth_scale_status_rows))},
        "hawor_to_v18_depth_scale_sample_count": summarize(depth_scale_sample_count_rows),
        "hawor_to_v18_depth_scale_reports_preview": depth_scale_reports[:24],
        "camera_trajectory_alignment": camera_alignment,
        "rows_preview": row_reports,
        "skipped_preview": skipped_rows[:120],
        "blocking_reasons": blockers,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(out_dir / "v18_hawor_bridge_state_report.json", report)
    return report


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V18 HaWoR bridge state",
        "",
        "This is HaWoR-only bridge evidence. It does not run HaWoR, does not use WiLoR/HaMeR/MANO2D/depth probes as substitutes, and does not accept physical contact, occlusion ownership, nonpenetration, or V18 closure.",
        "",
        f"Status: `{summary['status']}`",
        f"All cases bridge accepted: `{summary['all_cases_hawor_bridge_accepted']}`",
        f"V18 physical hand state valid from bridge: `{summary['v18_physical_hand_state_valid_from_bridge']}`",
        "",
    ]
    for case in summary["cases"]:
        lines += [
            f"## {case['case']}",
            "",
            f"Status: `{case['status']}`",
            f"Bridge candidate rows: `{case.get('bridge_candidate_rows')}/{case.get('expected_frame_side_rows')}`",
            f"Accepted V18 HaWoR foundation: `{case.get('accepted_v18_hawor_foundation')}`",
            f"Bridge NPZ: `{case.get('bridge_candidate_npz')}`",
            f"Projection residual median summary: `{case.get('reference_projection_residual_px_median_per_row')}`",
            f"Projection threshold counts: `{case.get('reference_projection_residual_threshold_counts')}`",
            f"Blocking reasons: `{case.get('blocking_reasons')}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [build_case(case, args) for case in args.cases]
    all_accepted = all(case.get("accepted_v18_hawor_foundation") is True for case in cases)
    any_blocked = any(case.get("status") == "blocked_no_hawor_npz_for_case" for case in cases)
    summary = {
        "method": "build_v18_hawor_bridge_state",
        "status": "trash_bridge_candidate_built_task5_blocked_not_v18_foundation" if any_blocked else "hawor_bridge_candidates_built_not_v18_foundation",
        "claim_scope": "HaWoR_bridge_candidate_state_no_model_substitution_no_full_V18_closure",
        "output_root": str(args.output_root),
        "all_cases_hawor_bridge_accepted": all_accepted,
        "v18_physical_hand_state_valid_from_bridge": False,
        "cases": cases,
        "blocking_reasons": sorted({reason for case in cases for reason in case.get("blocking_reasons", []) if isinstance(reason, str)}),
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_bridge_state"
    write_json(out_dir / "v18_hawor_bridge_state_summary.json", summary)
    write_summary_markdown(out_dir / "V18_HAWOR_BRIDGE_STATE.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--complete-depth-root", type=Path, default=DEFAULT_COMPLETE_DEPTH_ROOT)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
