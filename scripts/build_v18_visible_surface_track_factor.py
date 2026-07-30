#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build reusable visible-surface track factors from model masks and metric depth.

A visible-surface track is a first-surface observation: a model-produced mask plus
metric depth samples.  It is not hidden geometry, object pose, contact, or signed
nonpenetration.  Solver consumption may use it for MANO depth-order/occlusion
constraints and to quarantine hidden-volume constraints at frames where the
visible first surface is the only trusted object evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources  # noqa: E402
from build_v18_part_visible_surfaces import resize_bool_mask  # noqa: E402
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    numeric_summary,
    write_json,
)

DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_visible_surface_track_factor_v1")

STATE_ACTIVE = "active_visible_surface"
STATE_NO_MASK = "no_visible_surface_observation"
STATE_EMPTY_MASK = "empty_visible_surface_mask"
STATE_MISSING_DEPTH = "missing_metric_depth"
STATE_NO_DEPTH_SUPPORT = "visible_mask_without_metric_depth_support"


def coherent_first_surface_mask(
    *,
    mask: np.ndarray,
    depth_row: dict[str, Any],
    enabled: bool,
    min_gap_m: float,
    min_retained_fraction: float,
    min_removed_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the nearest coherent depth component inside a model mask.

    A visible-surface mask is an object/part support region, but depth inside it
    can contain holes, background, or mixed transparent/edge pixels. The first
    physical surface that can occlude a hand is the nearest coherent component,
    not a disconnected far-depth tail. Smooth depth variation is preserved; only
    a clear upper-tail depth discontinuity creates a filtered solver mask.
    """
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    mask_d = resize_bool_mask(mask.astype(bool), depth.shape)
    valid = mask_d & np.isfinite(depth) & (depth > 1.0e-5)
    z = np.sort(depth[valid].astype(float))
    diag: dict[str, Any] = {
        "enabled": bool(enabled),
        "input_mask_area_px_depth_grid": int(mask_d.sum()),
        "input_valid_depth_pixels": int(z.size),
        "state": "disabled" if not enabled else "no_filter_applied",
        "cutoff_depth_m": None,
        "retained_valid_depth_pixels": int(z.size),
        "removed_valid_depth_pixels": 0,
        "removed_valid_depth_fraction": 0.0,
        "depth_m_before": numeric_summary(z),
    }
    if not enabled or z.size < 32:
        return mask_d, diag
    gaps = np.diff(z)
    if gaps.size == 0:
        return mask_d, diag
    lo = max(0, int(0.50 * z.size) - 1)
    hi = max(lo + 1, int(0.995 * z.size) - 1)
    cand = gaps[lo:hi]
    if cand.size == 0:
        return mask_d, diag
    gap_idx = lo + int(np.argmax(cand))
    max_gap = float(gaps[gap_idx])
    positive = cand[cand > 0]
    median_gap = float(np.median(positive)) if positive.size else 0.0
    required_gap = max(float(min_gap_m), 25.0 * median_gap)
    cutoff = float(z[gap_idx])
    retained = depth[valid] <= cutoff
    retained_count = int(np.count_nonzero(retained))
    removed_count = int(z.size - retained_count)
    retained_fraction = float(retained_count / max(1, z.size))
    removed_fraction = float(removed_count / max(1, z.size))
    q90 = float(np.percentile(z, 90.0))
    q95 = float(np.percentile(z, 95.0))
    q90_removed = int(np.count_nonzero(z > q90))
    q90_removed_fraction = float(q90_removed / max(1, z.size))
    diag.update(
        {
            "candidate_gap_m": max_gap,
            "median_upper_tail_gap_m": median_gap,
            "required_gap_m": required_gap,
            "candidate_cutoff_depth_m": cutoff,
            "candidate_retained_fraction": retained_fraction,
            "candidate_removed_fraction": removed_fraction,
            "q90_depth_m": q90,
            "q95_depth_m": q95,
            "q95_minus_q90_m": float(q95 - q90),
        }
    )
    if max_gap < required_gap or retained_fraction < float(min_retained_fraction) or removed_fraction < float(min_removed_fraction):
        if (q95 - q90) >= float(min_gap_m) and (1.0 - q90_removed_fraction) >= float(min_retained_fraction) and q90_removed_fraction >= float(min_removed_fraction):
            cutoff = q90
            retained = depth[valid] <= cutoff
            retained_count = int(np.count_nonzero(retained))
            removed_count = int(z.size - retained_count)
            retained_fraction = float(retained_count / max(1, z.size))
            removed_fraction = float(removed_count / max(1, z.size))
            diag["state"] = "applied_upper_tail_quantile_depth_filter"
        else:
            diag["state"] = "no_clear_disconnected_far_depth_component"
            return mask_d, diag
    filtered = np.zeros_like(mask_d, dtype=bool)
    valid_indices = np.argwhere(valid)
    keep_indices = valid_indices[retained]
    filtered[keep_indices[:, 0], keep_indices[:, 1]] = True
    applied_state = str(diag.get("state") or "")
    if not applied_state.startswith("applied_"):
        applied_state = "applied_disconnected_far_depth_tail_filter"
    diag.update(
        {
            "state": applied_state,
            "cutoff_depth_m": cutoff,
            "retained_valid_depth_pixels": retained_count,
            "removed_valid_depth_pixels": removed_count,
            "removed_valid_depth_fraction": removed_fraction,
            "depth_m_after": numeric_summary(depth[filtered].astype(float)),
        }
    )
    return filtered, diag


def write_mask(mask: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))
    return path


def parse_spans(values: list[list[int]] | None) -> list[tuple[int, int]]:
    if not values:
        raise RuntimeError("at least one --frame-span START END is required")
    spans: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise RuntimeError(f"invalid --frame-span value: {pair}")
        a, b = int(pair[0]), int(pair[1])
        if b < a:
            raise RuntimeError(f"decreasing frame span {a}:{b}")
        spans.append((a, b))
    return spans


def frame_ids_from_spans(spans: list[tuple[int, int]]) -> list[int]:
    out: set[int] = set()
    for a, b in spans:
        out.update(range(a, b + 1))
    return sorted(out)


def load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(f["frame_idx"]): f for f in as_list(annotations.get("frames")) if isinstance(f, dict) and f.get("frame_idx") is not None}


def entity_tokens(entity_id: str) -> set[str]:
    raw = str(entity_id)
    tokens = {raw}
    if raw.startswith("object:"):
        tokens.add(raw.split(":", 1)[1])
    else:
        tokens.add(f"object:{raw}")
    return tokens


def row_matches_target(row: dict[str, Any], target_entity_id: str) -> bool:
    tokens = entity_tokens(target_entity_id)
    for key in ("target_entity_id", "object_id", "entity_id", "track_id"):
        value = row.get(key)
        if isinstance(value, str):
            return value in tokens
    return True


def load_mask_rows(report_path: Path, target_entity_id: str) -> dict[int, dict[str, Any]]:
    if not report_path.exists():
        raise FileNotFoundError(f"missing visible mask report: {report_path}")
    payload = load_json(report_path)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("saved_mask_rows_after_start", "target_mask_rows", "surface_rows"):
            rows.extend([r for r in as_list(payload.get(key)) if isinstance(r, dict) and row_matches_target(r, target_entity_id)])
            if rows:
                break
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("frame_idx") is None:
            continue
        raw = row.get("saved_mask_path") or row.get("mask_path")
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.exists():
            frame_idx = int(row["frame_idx"])
            if frame_idx in out:
                raise RuntimeError(f"duplicate visible mask row for target {target_entity_id} frame {frame_idx} in {report_path}")
            out[frame_idx] = dict(row)
    return out


def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"could not decode visible surface mask: {path}")
    return arr > 0


def load_ownership_rows(report_path: Path | None, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if report_path is None:
        return {}
    payload = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get("ownership_rows") if isinstance(payload, dict) else None):
        if not isinstance(row, dict) or not row_matches_target(row, target_entity_id):
            continue
        if row.get("frame_idx") is None or row.get("hand_side") is None:
            continue
        key = (int(row["frame_idx"]), str(row["hand_side"]))
        if key in out:
            raise RuntimeError(f"duplicate ownership row for target {target_entity_id} frame/side {key} in {report_path}")
        out[key] = dict(row)
    return out


def lift_mask_depth_samples(
    *,
    frame: dict[str, Any],
    frame_idx: int,
    mask: np.ndarray,
    depth_row: dict[str, Any],
    stride: int,
    max_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    mask_d = resize_bool_mask(mask.astype(bool), depth.shape)
    valid = mask_d & np.isfinite(depth) & (depth > 1.0e-5)
    ys_all, xs_all = np.nonzero(valid)
    if xs_all.size == 0:
        empty = {
            "pixel_uv": np.zeros((0, 2), dtype=np.float32),
            "depth_m": np.zeros((0,), dtype=np.float32),
            "points_camera_m": np.zeros((0, 3), dtype=np.float32),
            "points_world_m": np.zeros((0, 3), dtype=np.float32),
        }
        return empty, {
            "surface_state": STATE_NO_DEPTH_SUPPORT,
            "mask_area_px_raw": int(mask.sum()),
            "mask_area_px_depth_grid": int(mask_d.sum()),
            "valid_depth_pixels": 0,
            "sample_count": 0,
            "depth_m": numeric_summary(np.zeros((0,), dtype=float)),
        }
    stride = max(1, int(stride))
    sampled_grid = np.zeros_like(valid, dtype=bool)
    sampled_grid[::stride, ::stride] = True
    sampled = valid & sampled_grid
    ys, xs = np.nonzero(sampled)
    if xs.size == 0:
        ys, xs = ys_all, xs_all
    if int(max_samples) > 0 and xs.size > int(max_samples):
        order = np.linspace(0, xs.size - 1, int(max_samples), dtype=np.int64)
        ys = ys[order]
        xs = xs[order]
    z = depth[ys, xs].astype(np.float64)
    intr = np.asarray(depth_row["intrinsics"], dtype=np.float64).reshape(-1)
    if intr.size == 9:
        fx, fy, cx, cy = float(intr[0]), float(intr[4]), float(intr[2]), float(intr[5])
    else:
        fx, fy, cx, cy = [float(v) for v in intr[:4]]
    x_cam = (xs.astype(np.float64) - cx) * z / fx
    y_cam = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.column_stack([x_cam, y_cam, z])
    r_c2w, t_c2w = frame_camera_pose(frame)
    pts_world = pts_cam @ r_c2w.T + t_c2w[None, :]
    arrays = {
        "pixel_uv": np.column_stack([xs, ys]).astype(np.float32),
        "depth_m": z.astype(np.float32),
        "points_camera_m": pts_cam.astype(np.float32),
        "points_world_m": pts_world.astype(np.float32),
    }
    summary = {
        "surface_state": STATE_ACTIVE,
        "mask_area_px_raw": int(mask.sum()),
        "mask_area_px_depth_grid": int(mask_d.sum()),
        "valid_depth_pixels": int(xs_all.size),
        "sample_count": int(xs.size),
        "depth_m": numeric_summary(z.astype(float)),
        "bbox_world_min_m": pts_world.min(axis=0).astype(float).tolist(),
        "bbox_world_max_m": pts_world.max(axis=0).astype(float).tolist(),
        "extent_world_m": (pts_world.max(axis=0) - pts_world.min(axis=0)).astype(float).tolist(),
    }
    return arrays, summary


def render_review(
    *,
    frame: dict[str, Any],
    frame_idx: int,
    mask: np.ndarray,
    samples: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> bool:
    raw = frame.get("raw_frame_path")
    image = cv2.imread(str(raw), cv2.IMREAD_COLOR) if isinstance(raw, str) else None
    if image is None:
        return False
    mask_img = resize_bool_mask(mask.astype(bool), image.shape[:2])
    overlay = image.copy()
    overlay[mask_img] = (180, 0, 180)
    canvas = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    uv = np.asarray(samples.get("pixel_uv", np.zeros((0, 2))), dtype=float)
    if uv.size:
        if mask.shape[:2] != image.shape[:2]:
            sx = image.shape[1] / float(mask.shape[1])
            sy = image.shape[0] / float(mask.shape[0])
            uv = uv * np.asarray([sx, sy], dtype=float)[None, :]
        max_pts = 4500
        if len(uv) > max_pts:
            uv = uv[np.linspace(0, len(uv) - 1, max_pts, dtype=np.int64)]
        for u, v in uv.astype(int):
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(canvas, (int(u), int(v)), 1, (0, 220, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))


def build(args: argparse.Namespace) -> dict[str, Any]:
    spans = parse_spans(args.frame_span)
    frame_ids = frame_ids_from_spans(spans)
    annotations = load_json(args.annotations)
    frames = load_frames(annotations)
    mask_rows = load_mask_rows(args.visible_mask_report, str(args.target_entity_id))
    ownership_rows = load_ownership_rows(args.visible_ownership_factor_report, str(args.target_entity_id))
    depth_rows = load_depth_sources(list(args.depth_npz))
    case_root = args.output_root / args.case
    sample_dir = case_root / "visible_surface_samples"
    review_dir = case_root / "review_frames"
    filtered_mask_dir = case_root / "first_surface_masks"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sides = list(args.sides)
    frame_surface_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    review_frames = set(int(f) for f in args.review_frames)

    for frame_idx in frame_ids:
        frame = frames.get(frame_idx)
        mask_row = mask_rows.get(frame_idx)
        depth_row = depth_rows.get(frame_idx)
        frame_surface: dict[str, Any]
        surface_npz_path: Path | None = None
        mask_path: Path | None = None
        if frame is None:
            frame_surface = {"frame_idx": int(frame_idx), "surface_state": "missing_annotation_frame"}
        elif mask_row is None:
            frame_surface = {"frame_idx": int(frame_idx), "surface_state": STATE_NO_MASK}
        else:
            mask_path = Path(str(mask_row.get("saved_mask_path") or mask_row.get("mask_path")))
            mask = load_mask(mask_path)
            if not np.any(mask):
                frame_surface = {
                    "frame_idx": int(frame_idx),
                    "surface_state": STATE_EMPTY_MASK,
                    "surface_mask_path": str(mask_path),
                    "mask_area_px_raw": 0,
                    "valid_depth_pixels": 0,
                    "sample_count": 0,
                }
            elif depth_row is None:
                frame_surface = {
                    "frame_idx": int(frame_idx),
                    "surface_state": STATE_MISSING_DEPTH,
                    "surface_mask_path": str(mask_path),
                    "mask_area_px_raw": int(mask.sum()),
                    "valid_depth_pixels": 0,
                    "sample_count": 0,
                }
            else:
                depth_mask, depth_filter_diag = coherent_first_surface_mask(
                    mask=mask,
                    depth_row=depth_row,
                    enabled=bool(args.coherent_first_surface_depth_filter),
                    min_gap_m=float(args.first_surface_depth_gap_min_m),
                    min_retained_fraction=float(args.first_surface_min_retained_fraction),
                    min_removed_fraction=float(args.first_surface_min_removed_fraction),
                )
                factor_mask_path = mask_path
                if str(depth_filter_diag.get("state") or "").startswith("applied_"):
                    factor_mask_path = write_mask(depth_mask, filtered_mask_dir / f"{frame_idx:06d}_first_surface_depth_filtered.png")
                samples, summary = lift_mask_depth_samples(
                    frame=frame,
                    frame_idx=frame_idx,
                    mask=depth_mask,
                    depth_row=depth_row,
                    stride=int(args.sample_stride),
                    max_samples=int(args.max_samples_per_frame),
                )
                frame_surface = {"frame_idx": int(frame_idx), "surface_mask_path": str(factor_mask_path), "raw_surface_mask_path": str(mask_path), "depth_filter": depth_filter_diag, **summary}
                if frame_surface["surface_state"] == STATE_ACTIVE:
                    surface_npz_path = sample_dir / f"{frame_idx:06d}_visible_surface_samples.npz"
                    np.savez_compressed(surface_npz_path, **samples, metadata_json=json.dumps({"frame_idx": int(frame_idx), "surface_mask_path": str(factor_mask_path), "raw_surface_mask_path": str(mask_path), "depth_filter": depth_filter_diag, "claim": "visible first-surface samples only; no hidden geometry or object pose"}))
                    frame_surface["visible_surface_npz_path"] = str(surface_npz_path)
                    if frame_idx in review_frames:
                        review_path = review_dir / f"{frame_idx:06d}_visible_surface_track.jpg"
                        ok = render_review(frame=frame, frame_idx=frame_idx, mask=depth_mask, samples=samples, output_path=review_path, title=f"visible surface track frame {frame_idx}")
                        if ok:
                            frame_surface["review_path"] = str(review_path)
                else:
                    frame_surface["visible_surface_npz_path"] = None
        state_counts[str(frame_surface.get("surface_state"))] += 1
        frame_surface_rows.append(frame_surface)
        for side in sides:
            factor_surface = dict(frame_surface)
            ownership_row = ownership_rows.get((int(frame_idx), str(side)))
            ownership_mask_path = None
            if (
                frame is not None
                and depth_row is not None
                and ownership_row is not None
                and isinstance(ownership_row.get("adjusted_entity_mask_path"), str)
                and Path(str(ownership_row.get("adjusted_entity_mask_path"))).exists()
            ):
                ownership_mask_path = Path(str(ownership_row["adjusted_entity_mask_path"]))
                ownership_mask = load_mask(ownership_mask_path)
                if np.any(ownership_mask):
                    ownership_depth_mask, ownership_depth_filter_diag = coherent_first_surface_mask(
                        mask=ownership_mask,
                        depth_row=depth_row,
                        enabled=bool(args.coherent_first_surface_depth_filter),
                        min_gap_m=float(args.first_surface_depth_gap_min_m),
                        min_retained_fraction=float(args.first_surface_min_retained_fraction),
                        min_removed_fraction=float(args.first_surface_min_removed_fraction),
                    )
                    factor_mask_path = ownership_mask_path
                    if str(ownership_depth_filter_diag.get("state") or "").startswith("applied_"):
                        factor_mask_path = write_mask(ownership_depth_mask, filtered_mask_dir / f"{frame_idx:06d}_{side}_ownership_first_surface_depth_filtered.png")
                    ownership_samples, ownership_summary = lift_mask_depth_samples(
                        frame=frame,
                        frame_idx=frame_idx,
                        mask=ownership_depth_mask,
                        depth_row=depth_row,
                        stride=int(args.sample_stride),
                        max_samples=int(args.max_samples_per_frame),
                    )
                    factor_surface = {"frame_idx": int(frame_idx), "surface_mask_path": str(factor_mask_path), "raw_surface_mask_path": str(ownership_mask_path), "depth_filter": ownership_depth_filter_diag, **ownership_summary}
                    if factor_surface["surface_state"] == STATE_ACTIVE:
                        side_npz_path = sample_dir / f"{frame_idx:06d}_{side}_ownership_filtered_visible_surface_samples.npz"
                        np.savez_compressed(
                            side_npz_path,
                            **ownership_samples,
                            metadata_json=json.dumps({
                                "frame_idx": int(frame_idx),
                                "hand_side": str(side),
                                "surface_mask_path": str(factor_mask_path),
                                "raw_surface_mask_path": str(ownership_mask_path),
                                "depth_filter": ownership_depth_filter_diag,
                                "visible_ownership_factor_report": str(args.visible_ownership_factor_report),
                                "claim": "visible first-surface samples after side-specific hand-owned-pixel quarantine; no hidden geometry or object pose",
                            }),
                        )
                        factor_surface["visible_surface_npz_path"] = str(side_npz_path)
                        if frame_idx in review_frames:
                            review_path = review_dir / f"{frame_idx:06d}_{side}_ownership_filtered_visible_surface_track.jpg"
                            ok = render_review(frame=frame, frame_idx=frame_idx, mask=ownership_depth_mask, samples=ownership_samples, output_path=review_path, title=f"ownership-filtered visible surface f{frame_idx} {side}")
                            if ok:
                                factor_surface["review_path"] = str(review_path)
                    else:
                        factor_surface["visible_surface_npz_path"] = None
                else:
                    factor_surface = {"frame_idx": int(frame_idx), "surface_state": STATE_EMPTY_MASK, "surface_mask_path": str(ownership_mask_path), "mask_area_px_raw": 0, "valid_depth_pixels": 0, "sample_count": 0, "visible_surface_npz_path": None}
            active = factor_surface.get("surface_state") == STATE_ACTIVE
            factor_rows.append(
                {
                    "factor_family": "visible_surface_track",
                    "target_entity_id": args.target_entity_id,
                    "frame_idx": int(frame_idx),
                    "hand_side": side,
                    "variable_affected": "H_t_and_constraint_eligibility",
                    "observation_type": "model_mask_metric_depth_visible_first_surface" if ownership_mask_path is None else "ownership_filtered_model_mask_metric_depth_visible_first_surface",
                    "surface_state": str(factor_surface.get("surface_state")),
                    "residual_or_quarantine_rule": "if active, MANO vertices projecting into the visible first-surface mask are constrained by one-sided depth order; hidden signed-volume nonpenetration is quarantined for this target/frame",
                    "quarantine_hidden_volume": bool(active and args.quarantine_hidden_volume),
                    "surface_mask_path": factor_surface.get("surface_mask_path"),
                    "visible_surface_npz_path": factor_surface.get("visible_surface_npz_path"),
                    "valid_depth_pixels": int(factor_surface.get("valid_depth_pixels", 0) or 0),
                    "sample_count": int(factor_surface.get("sample_count", 0) or 0),
                    "mask_area_px_raw": int(factor_surface.get("mask_area_px_raw", 0) or 0),
                    "raw_surface_mask_path": factor_surface.get("raw_surface_mask_path"),
                    "depth_filter": factor_surface.get("depth_filter"),
                    "depth_order_margin_m": float(args.depth_order_margin_m),
                    "provenance": {
                        "annotations": str(args.annotations),
                        "visible_mask_report": str(args.visible_mask_report),
                        "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report),
                        "depth_npz": [str(p) for p in args.depth_npz],
                        "mask_source_row": mask_row if mask_row is not None else None,
                        "ownership_source_row": ownership_row if ownership_row is not None else None,
                    },
                    "rendered_uncertainty_channel": "visible first-surface mask/depth and MANO depth-order residual; no hidden volume accepted",
                }
            )
    active_rows = [r for r in frame_surface_rows if r.get("surface_state") == STATE_ACTIVE]
    report = {
        "method": "build_v18_visible_surface_track_factor",
        "status": "ok",
        "claim_scope": "Visible first-surface track factor only: model-produced mask plus metric depth samples. It does not assert hidden geometry, object pose, contact, or signed nonpenetration.",
        "case": args.case,
        "target_entity_id": args.target_entity_id,
        "inputs": {
            "annotations": str(args.annotations),
            "visible_mask_report": str(args.visible_mask_report),
            "depth_npz": [str(p) for p in args.depth_npz],
        },
        "parameters": {
            "frame_spans": [[int(a), int(b)] for a, b in spans],
            "sides": sides,
            "sample_stride": int(args.sample_stride),
            "max_samples_per_frame": int(args.max_samples_per_frame),
            "quarantine_hidden_volume": bool(args.quarantine_hidden_volume),
            "depth_order_margin_m": float(args.depth_order_margin_m),
            "coherent_first_surface_depth_filter": bool(args.coherent_first_surface_depth_filter),
            "first_surface_depth_gap_min_m": float(args.first_surface_depth_gap_min_m),
            "first_surface_min_retained_fraction": float(args.first_surface_min_retained_fraction),
            "first_surface_min_removed_fraction": float(args.first_surface_min_removed_fraction),
        },
        "summary": {
            "frame_count_requested": int(len(frame_ids)),
            "surface_frame_count": int(len(frame_surface_rows)),
            "factor_row_count": int(len(factor_rows)),
            "state_counts": dict(state_counts),
            "valid_depth_pixels": numeric_summary(np.asarray([r.get("valid_depth_pixels", 0) for r in active_rows], dtype=float)),
            "sample_count": numeric_summary(np.asarray([r.get("sample_count", 0) for r in active_rows], dtype=float)),
            "mask_area_px_raw": numeric_summary(np.asarray([r.get("mask_area_px_raw", 0) for r in active_rows], dtype=float)),
        },
        "surface_rows": frame_surface_rows,
        "factor_rows": factor_rows,
    }
    out = case_root / "v18_visible_surface_track_factor_report.json"
    write_json(out, report)
    print(json.dumps({"status": "ok", "report": str(out), "summary": report["summary"]}, indent=2)[:6000])
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--target-entity-id", required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--visible-mask-report", type=Path, required=True)
    p.add_argument("--visible-ownership-factor-report", type=Path, default=None, help="Optional visible ownership factor report. When a frame/side has adjusted_entity_mask_path, build side-specific visible surface samples from object-owned pixels only.")
    p.add_argument("--depth-npz", type=Path, action="append", required=True)
    p.add_argument("--frame-span", nargs=2, type=int, action="append", required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-stride", type=int, default=6)
    p.add_argument("--max-samples-per-frame", type=int, default=6000)
    p.add_argument("--depth-order-margin-m", type=float, default=0.01)
    p.add_argument("--coherent-first-surface-depth-filter", action=argparse.BooleanOptionalAction, default=False, help="When a model mask contains a disconnected far-depth tail, replace its solver mask with the nearest coherent first-surface component before emitting visible-surface factors.")
    p.add_argument("--first-surface-depth-gap-min-m", type=float, default=0.05, help="Minimum metric depth discontinuity required before pruning far-depth mask pixels as non-first-surface outliers.")
    p.add_argument("--first-surface-min-retained-fraction", type=float, default=0.50, help="Do not apply first-surface pruning if the nearest component would retain less than this fraction of valid mask depth pixels.")
    p.add_argument("--first-surface-min-removed-fraction", type=float, default=0.01, help="Do not write a filtered mask unless the far-depth component removes at least this fraction of valid mask depth pixels.")
    p.add_argument("--quarantine-hidden-volume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--review-frames", nargs="*", type=int, default=[])
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
