#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build reusable surface-eligibility factors for MANO/object constraints.

This factor is object/category agnostic.  It classifies posed mesh faces or
surface patches by whether the projected surface is physically eligible to act
as a hard MANO nonpenetration/depth-order constraint.  Hidden, free-space, and
unresolved surfaces are not accepted as hard object geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources, project_points  # noqa: E402
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    load_mesh,
    numeric_summary,
    pose_map,
    write_json,
)

DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_surface_eligibility_factor_v1")

STATE_OBSERVED = "observed_depth_supported"
STATE_HAND_QUARANTINED = "hand_owned_depth_quarantined"
STATE_FREE = "free_space_rejected"
STATE_HIDDEN = "hidden_unvalidated"
STATE_OUTSIDE = "outside_view"
STATE_UNRESOLVED = "unresolved"


def parse_spans(values: list[list[int]] | None) -> list[tuple[int, int]]:
    if not values:
        raise RuntimeError("at least one --frame-span START END is required")
    spans: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise RuntimeError(f"invalid frame span: {pair}")
        a, b = int(pair[0]), int(pair[1])
        if b < a:
            raise RuntimeError(f"invalid decreasing span {a}:{b}")
        spans.append((a, b))
    return spans


def frame_ids_from_spans(spans: list[tuple[int, int]]) -> list[int]:
    ids: set[int] = set()
    for a, b in spans:
        ids.update(range(a, b + 1))
    return sorted(ids)


def load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame["frame_idx"]): frame
        for frame in as_list(annotations.get("frames"))
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }


def face_support_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    return np.stack(
        [
            tri[:, 0],
            tri[:, 1],
            tri[:, 2],
            0.5 * (tri[:, 0] + tri[:, 1]),
            0.5 * (tri[:, 1] + tri[:, 2]),
            0.5 * (tri[:, 2] + tri[:, 0]),
            tri.mean(axis=1),
        ],
        axis=1,
    )


def classify_faces_against_depth(
    *,
    frame: dict[str, Any],
    vertices_object: np.ndarray,
    faces: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    support_margin_m: float,
    free_space_margin_m: float,
    min_supported_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_faces = int(len(faces))
    if depth_row is None:
        masks = {
            STATE_OBSERVED: np.zeros(n_faces, dtype=bool),
            STATE_FREE: np.zeros(n_faces, dtype=bool),
            STATE_HIDDEN: np.zeros(n_faces, dtype=bool),
            STATE_OUTSIDE: np.ones(n_faces, dtype=bool),
            STATE_UNRESOLVED: np.zeros(n_faces, dtype=bool),
        }
        return masks, {"state": "missing_depth", "frame_idx": int(frame["frame_idx"]), "face_count": n_faces}

    r_obj, t_obj = pose
    samples_obj = face_support_samples(vertices_object, faces)
    flat_obj = samples_obj.reshape(-1, 3)
    flat_world = flat_obj @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    flat_cam = (flat_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    height, width = depth.shape
    u, v, valid = project_points(flat_cam, np.asarray(depth_row["intrinsics"], dtype=float), width, height)
    z_obs = np.full((flat_cam.shape[0],), np.nan, dtype=float)
    if np.any(valid):
        z = depth[v[valid], u[valid]].astype(float)
        finite = np.isfinite(z) & (z > 1.0e-5)
        valid_ids = np.where(valid)[0]
        z_obs[valid_ids[finite]] = z[finite]
    finite = np.isfinite(z_obs)
    residual = flat_cam[:, 2].astype(float) - z_obs
    finite_2d = finite.reshape(n_faces, -1)
    residual_2d = residual.reshape(n_faces, -1)
    supported = finite_2d & (np.abs(residual_2d) <= float(support_margin_m))
    free = finite_2d & (residual_2d < -float(free_space_margin_m))
    behind = finite_2d & (residual_2d > float(support_margin_m))
    supported_count = np.count_nonzero(supported, axis=1)
    free_count = np.count_nonzero(free, axis=1)
    behind_count = np.count_nonzero(behind, axis=1)
    finite_count = np.count_nonzero(finite_2d, axis=1)

    free_mask = free_count > 0
    observed_mask = (supported_count >= int(min_supported_samples)) & ~free_mask
    outside_mask = finite_count == 0
    hidden_mask = (finite_count > 0) & (behind_count >= np.maximum(1, finite_count - supported_count)) & ~observed_mask & ~free_mask
    unresolved_mask = ~(observed_mask | free_mask | outside_mask | hidden_mask)
    masks = {
        STATE_OBSERVED: observed_mask.astype(bool),
        STATE_FREE: free_mask.astype(bool),
        STATE_HIDDEN: hidden_mask.astype(bool),
        STATE_OUTSIDE: outside_mask.astype(bool),
        STATE_UNRESOLVED: unresolved_mask.astype(bool),
    }
    summary = {
        "state": "face_support_depth_classified",
        "frame_idx": int(frame["frame_idx"]),
        "depth_source": str(depth_row.get("source")),
        "face_count": n_faces,
        "sample_count": int(flat_cam.shape[0]),
        "finite_sample_count": int(np.count_nonzero(finite)),
        "face_state_counts": {key: int(np.count_nonzero(mask)) for key, mask in masks.items()},
        "sample_depth_residual_m": numeric_summary(residual[finite]),
        "supported_sample_count": numeric_summary(supported_count.astype(float)),
        "free_space_sample_count": numeric_summary(free_count.astype(float)),
        "behind_sample_count": numeric_summary(behind_count.astype(float)),
    }
    return masks, summary


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_pose_support_uncertainty(pose_report: dict[str, Any], *, stat: str, default_m: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in as_list(pose_report.get("pose_rows")) if isinstance(pose_report, dict) else []:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        raw = nested_get(row, stat)
        if raw is None:
            raw = default_m
        try:
            val = float(raw)
        except Exception:
            val = float(default_m)
        out[int(row["frame_idx"])] = max(0.0, val)
    return out


def load_visible_ownership_rows(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get("ownership_rows")) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("frame_idx") is not None and row.get("hand_side") is not None:
            rows[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return rows


def load_bool_mask(path: str | None) -> np.ndarray | None:
    if not isinstance(path, str) or not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"explicit ownership mask path is missing: {mask_path}")
    import cv2
    arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"explicit ownership mask path could not be decoded: {mask_path}")
    return arr > 0


def mask_membership(mask: np.ndarray, uv: np.ndarray, dilation_px: int) -> np.ndarray:
    if mask is None or uv.ndim != 2 or uv.shape[1] != 2:
        return np.zeros((len(uv),), dtype=bool)
    h, w = mask.shape
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = np.zeros((len(uv),), dtype=bool)
    if not np.any(valid):
        return inside
    radius = max(0, int(dilation_px))
    x0 = np.clip(u[valid] - radius, 0, w - 1)
    x1 = np.clip(u[valid] + radius, 0, w - 1)
    y0 = np.clip(v[valid] - radius, 0, h - 1)
    y1 = np.clip(v[valid] + radius, 0, h - 1)
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    area = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
    inside[np.where(valid)[0]] = area > 0
    return inside


def ownership_face_quarantine(
    *,
    frame: dict[str, Any],
    vertices_object: np.ndarray,
    faces: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    ownership_row: dict[str, Any] | None,
    dilation_px: int,
) -> np.ndarray:
    mask = load_bool_mask(None if ownership_row is None else ownership_row.get("non_object_owned_mask_path"))
    if mask is None or depth_row is None:
        return np.zeros((len(faces),), dtype=bool)
    r_obj, t_obj = pose
    samples_obj = face_support_samples(vertices_object, faces)
    flat_world = samples_obj.reshape(-1, 3) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    flat_cam = (flat_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    height, width = depth.shape
    u, v, valid = project_points(flat_cam, np.asarray(depth_row["intrinsics"], dtype=float), width, height)
    uv = np.column_stack([u.astype(float), v.astype(float)])
    inside = mask_membership(mask, uv, int(dilation_px)) & valid
    if inside.shape[0] != flat_cam.shape[0]:
        return np.zeros((len(faces),), dtype=bool)
    return inside.reshape(len(faces), -1).any(axis=1)


def write_face_masks(path: Path, masks: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: mask.astype(np.bool_) for key, mask in masks.items()})


def build(args: argparse.Namespace) -> dict[str, Any]:
    spans = parse_spans(args.frame_span)
    frame_ids = frame_ids_from_spans(spans)
    annotations = load_json(args.annotations)
    frames = load_frames(annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    support_uncertainty_by_frame = load_pose_support_uncertainty(
        pose_report,
        stat=str(args.surface_support_uncertainty_stat),
        default_m=float(args.default_surface_support_uncertainty_m),
    )
    mesh = load_mesh(args.completed_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    depth_rows = load_depth_sources(list(args.depth_npz))
    ownership_rows = load_visible_ownership_rows(args.visible_ownership_factor_report)
    output_case = args.output_root / args.case
    output_case.mkdir(parents=True, exist_ok=True)

    factor_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    state_counter: Counter[str] = Counter()
    for frame_idx in frame_ids:
        frame = frames.get(frame_idx)
        pose = poses.get(frame_idx)
        if frame is None or pose is None:
            state_counter["missing_frame_or_pose"] += 1
            continue
        base_masks, summary = classify_faces_against_depth(
            frame=frame,
            vertices_object=vertices_object,
            faces=faces,
            pose=pose,
            depth_row=depth_rows.get(frame_idx),
            support_margin_m=float(args.support_margin_m),
            free_space_margin_m=float(args.free_space_margin_m),
            min_supported_samples=int(args.min_supported_samples),
        )
        frame_summaries.append(summary)
        surface_support_uncertainty_m = float(support_uncertainty_by_frame.get(frame_idx, float(args.default_surface_support_uncertainty_m)))
        for side in args.sides:
            hand_owned = ownership_face_quarantine(
                frame=frame,
                vertices_object=vertices_object,
                faces=faces,
                pose=pose,
                depth_row=depth_rows.get(frame_idx),
                ownership_row=ownership_rows.get((frame_idx, side)),
                dilation_px=int(args.ownership_face_overlap_dilation_px),
            )
            masks = {key: value.copy() for key, value in base_masks.items()}
            if np.any(hand_owned):
                masks[STATE_HAND_QUARANTINED] = hand_owned
                masks[STATE_OBSERVED] &= ~hand_owned
            else:
                masks[STATE_HAND_QUARANTINED] = hand_owned
            eligible_hard = masks[STATE_OBSERVED].copy()
            mask_path = output_case / "face_state_masks" / side / f"{frame_idx:06d}_surface_eligibility_faces.npz"
            write_face_masks(mask_path, {**masks, "eligible_hard_observed": eligible_hard})
            counts = {key: int(np.count_nonzero(value)) for key, value in masks.items()}
            counts["eligible_hard_observed"] = int(np.count_nonzero(eligible_hard))
            factor_rows.append(
                {
                    "factor_family": "surface_eligibility",
                    "target_entity_id": args.object_id,
                    "frame_idx": int(frame_idx),
                    "hand_side": side,
                    "variable_affected": "constraint_eligibility",
                    "observation_type": "posed_mesh_face_support_samples_x_metric_depth",
                    "residual_or_quarantine_rule": "only eligible_hard_observed faces may produce MANO nonpenetration; observed-surface residuals are hard only beyond observed_surface_support_uncertainty_m, while free-space, hidden, outside-view, unresolved, and hand-owned faces are not hard constraints",
                    "provenance": {
                        "annotations": str(args.annotations),
                        "pose_report": str(args.pose_report),
                        "completed_mesh": str(args.completed_mesh),
                        "depth_npz": [str(p) for p in args.depth_npz],
                        "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report),
                    },
                    "rendered_uncertainty_channel": "surface eligibility should render eligible observed faces separately from free-space rejected, hidden, outside-view, unresolved, hand-owned, and support-uncertain faces",
                    "face_state_npz_path": str(mask_path),
                    "observed_surface_support_uncertainty_m": float(surface_support_uncertainty_m),
                    "surface_support_uncertainty_m": float(surface_support_uncertainty_m),
                    "surface_support_uncertainty_stat": str(args.surface_support_uncertainty_stat),
                    "face_count": int(len(faces)),
                    "counts": counts,
                }
            )
            for key, value in counts.items():
                state_counter[f"{side}:{key}"] += int(value)
    report = {
        "method": "build_v18_surface_eligibility_factor",
        "status": "ok",
        "claim_scope": "Reusable face/patch eligibility factor records; not a MANO correction by themselves.",
        "case": args.case,
        "target_entity_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(p) for p in args.depth_npz],
            "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report),
        },
        "parameters": {
            "frame_spans": [[int(a), int(b)] for a, b in spans],
            "sides": list(args.sides),
            "support_margin_m": float(args.support_margin_m),
            "free_space_margin_m": float(args.free_space_margin_m),
            "min_supported_samples": int(args.min_supported_samples),
            "ownership_face_overlap_dilation_px": int(args.ownership_face_overlap_dilation_px),
            "surface_support_uncertainty_stat": str(args.surface_support_uncertainty_stat),
            "default_surface_support_uncertainty_m": float(args.default_surface_support_uncertainty_m),
        },
        "summary": {
            "frame_count_requested": int(len(frame_ids)),
            "factor_row_count": int(len(factor_rows)),
            "state_counts": dict(state_counter),
            "frame_face_state_counts": {
                key: numeric_summary(np.asarray([s.get("face_state_counts", {}).get(key, 0) for s in frame_summaries], dtype=float))
                for key in [STATE_OBSERVED, STATE_FREE, STATE_HIDDEN, STATE_OUTSIDE, STATE_UNRESOLVED]
            },
            "observed_surface_support_uncertainty_m": numeric_summary(np.asarray([float(r.get("observed_surface_support_uncertainty_m", 0.0)) for r in factor_rows], dtype=float)),
        },
        "frame_summaries": frame_summaries,
        "factor_rows": factor_rows,
    }
    report_path = output_case / "v18_surface_eligibility_factor_report.json"
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path), "factor_rows": len(factor_rows)}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--depth-npz", type=Path, action="append", required=True)
    p.add_argument("--frame-span", nargs=2, type=int, action="append", required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--visible-ownership-factor-report", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--support-margin-m", type=float, default=0.015)
    p.add_argument("--free-space-margin-m", type=float, default=0.025)
    p.add_argument("--min-supported-samples", type=int, default=2)
    p.add_argument("--ownership-face-overlap-dilation-px", type=int, default=2)
    p.add_argument("--surface-support-uncertainty-stat", default="observed_to_mesh_final.p95_m", help="Dotted pose_rows[] field emitted as observed_surface_support_uncertainty_m for solver nonpenetration slack.")
    p.add_argument("--default-surface-support-uncertainty-m", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
