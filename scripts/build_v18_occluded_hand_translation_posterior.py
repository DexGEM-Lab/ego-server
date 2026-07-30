#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a bounded occluded-hand translation posterior for V18 Trash zero-observation rows.

This is not hidden-hand reconstruction.  It reuses the interval solver's current
state as the zero trajectory and estimates a one-dimensional feasible/energy
profile for additional camera-z translation on rows whose MANO observation was
explicitly zeroed by a hand_observation_visibility factor.  The live question is
whether temporal dynamics plus selected visible-surface depth-order constraints
produce a narrow hidden-hand set or whether the feasible set remains saturated and
conflicted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_temporal_mano_translation_interval_state import as_list, frame_camera_pose, load_json, numeric_summary, write_json  # noqa: E402

DEFAULT_STATE = Path("/data2/ego_annotation_outputs/v18_trash_joint_mano_latent_transition_v2_slack_solver_v1/frames_931_1003/trash_1050/v18_joint_mano_interval_trajectory_state.json")
DEFAULT_ANNOTATIONS = Path("/data2/ego_annotation_outputs/v18_full_pipeline_sanitized_base_for_hprime/trash_1050/annotations_v18_full.json")
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_trash_occluded_hand_translation_posterior_v1/trash_1050/v18_occluded_hand_translation_posterior_report.json")
REJECTED_ANNOTATION_PATH_MARKERS = (
    "v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard",
    "verified_hprime_final",
    "hprime_final",
)


def reject_rejected_annotation_path(path_or_payload: Any, *, context: str) -> None:
    text = str(path_or_payload)
    hits = [marker for marker in REJECTED_ANNOTATION_PATH_MARKERS if marker in text]
    if hits:
        raise ValueError(f"{context} contains rejected H-prime/final-v7 annotation marker(s) {hits}; use sanitized non-H-prime sources")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, action="append", default=None, help="Interval solver state JSON. Repeat to build one posterior across adjacent interval-state files; frame/side rows must be unique.")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--zero-observation-weight", type=float, default=0.0)
    p.add_argument("--zero-observation-eps", type=float, default=1.0e-9)
    p.add_argument("--sample-count", type=int, default=201, help="Per-row grid samples for reporting the one-dimensional energy profile around the MAP. This is diagnostic, not a calibrated probability integral.")
    return p.parse_args()


@dataclass
class PosteriorRow:
    frame_idx: int
    hand_side: str
    row: dict[str, Any]
    camera_z_axis_world: np.ndarray
    optimized_translation_world_m: np.ndarray
    selected_final_delta_m: np.ndarray
    lower_s_m: float
    upper_s_m: float
    bound_state: str
    variable_index: int | None


def load_frames_by_idx(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(f["frame_idx"]): f for f in as_list(annotations.get("frames")) if isinstance(f, dict) and f.get("frame_idx") is not None}


def camera_z_axis_world(frame: dict[str, Any]) -> np.ndarray:
    r_c2w, _t_c2w = frame_camera_pose(frame)
    axis = np.asarray(r_c2w, dtype=float)[:, 2]
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        raise ValueError(f"frame {frame.get('frame_idx')} has degenerate camera z axis")
    return axis / norm


def translation_axis_bounds(translation_world: np.ndarray, axis_world: np.ndarray, max_translation_m: float) -> tuple[float, float, str]:
    t = np.asarray(translation_world, dtype=float)
    a = np.asarray(axis_world, dtype=float)
    a = a / max(np.linalg.norm(a), 1.0e-12)
    max_t = max(0.0, float(max_translation_m))
    dot = float(np.dot(t, a))
    c = float(np.dot(t, t) - max_t * max_t)
    disc = dot * dot - c
    if disc < -1.0e-10:
        return 0.0, 0.0, "current_translation_outside_bound"
    root = math.sqrt(max(0.0, disc))
    lower = -dot - root
    upper = -dot + root
    state = "inside_translation_bound" if c <= 1.0e-9 else "on_or_slightly_outside_translation_bound"
    return float(lower), float(upper), state


FINGERPRINT_FIELDS = (
    "optimized_joints_world_m",
    "optimized_translation_world_m",
    "optimized_root_delta_axis_angle_rad",
    "optimized_hand_pose_delta_axis_angle_rad",
)


def row_state_fingerprint(row: dict[str, Any]) -> str:
    payload = {field: row.get(field) for field in FINGERPRINT_FIELDS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_depth_stats(delta: np.ndarray, shift_m: float, margin_m: float) -> dict[str, Any]:
    arr = np.asarray(delta, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "selected_vertex_count": 0,
            "in_front_count": 0,
            "min_delta_hand_minus_surface_m": None,
            "clearance_required_camera_z_m": 0.0,
            "residual_energy_unweighted_m2": 0.0,
        }
    shifted = arr + float(shift_m)
    residual = np.maximum(0.0, -float(margin_m) - shifted)
    active = residual > 0.0
    return {
        "selected_vertex_count": int(arr.size),
        "in_front_count": int(np.count_nonzero(shifted < -float(margin_m))),
        "min_delta_hand_minus_surface_m": float(np.min(shifted)),
        "median_delta_hand_minus_surface_m": float(np.median(shifted)),
        "clearance_required_camera_z_m": float(max(0.0, -float(margin_m) - float(np.min(arr)))),
        "residual_energy_unweighted_m2": float(np.sum(residual * residual) / max(1, int(np.count_nonzero(active)))),
    }


def build_side_rows(state: dict[str, Any], frames_by_idx: dict[int, dict[str, Any]], side: str, args: argparse.Namespace) -> list[PosteriorRow]:
    params = state.get("parameters", {}) if isinstance(state.get("parameters"), dict) else {}
    max_translation_m = float(params.get("max_translation_m", 0.045) or 0.045)
    out: list[PosteriorRow] = []
    var_i = 0
    for st in sorted([r for r in as_list(state.get("per_frame_states")) if isinstance(r, dict) and str(r.get("hand_side")) == side], key=lambda r: int(r["frame_idx"])):
        frame_idx = int(st["frame_idx"])
        frame = frames_by_idx.get(frame_idx)
        if frame is None:
            raise KeyError(f"annotations missing frame {frame_idx}")
        axis = camera_z_axis_world(frame)
        trans = np.asarray(st.get("optimized_translation_world_m") or [0.0, 0.0, 0.0], dtype=float)
        lower, upper, bound_state = translation_axis_bounds(trans, axis, max_translation_m)
        obs_weight = float(st.get("hand_observation_visibility_weight_multiplier", 1.0) or 0.0)
        is_zero = abs(obs_weight - float(args.zero_observation_weight)) <= float(args.zero_observation_eps)
        selected_delta = np.asarray(st.get("visible_surface_depth_order_selected_final_delta_values_m") or [], dtype=float)
        out.append(
            PosteriorRow(
                frame_idx=frame_idx,
                hand_side=side,
                row=st,
                camera_z_axis_world=axis,
                optimized_translation_world_m=trans,
                selected_final_delta_m=selected_delta,
                lower_s_m=lower,
                upper_s_m=upper,
                bound_state=bound_state,
                variable_index=var_i if is_zero else None,
            )
        )
        if is_zero:
            var_i += 1
    return out


def optimize_side(rows: list[PosteriorRow], params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    nvar = 1 + max([r.variable_index for r in rows if r.variable_index is not None], default=-1)
    if nvar == 0:
        return np.zeros((0,), dtype=float), {"state": "no_zero_observation_rows", "success": True}
    margin = float(params.get("visible_surface_depth_order_margin_m", 0.010) or 0.010)
    depth_w = float(params.get("visible_surface_depth_order_weight", 2.0e4) or 2.0e4)
    smooth_w = float(params.get("smooth_weight", 5.0e3) or 5.0e3)
    accel_w = float(params.get("accel_weight", 1.0e4) or 1.0e4)
    bounds = [(r.lower_s_m, r.upper_s_m) for r in rows if r.variable_index is not None]
    current_zero = np.zeros((nvar,), dtype=float)
    feasible_start = np.asarray([float(np.clip(0.0, lo, hi)) for lo, hi in bounds], dtype=float)

    def full_shift_vectors(x: np.ndarray) -> np.ndarray:
        vecs = []
        for r in rows:
            s = 0.0 if r.variable_index is None else float(x[int(r.variable_index)])
            vecs.append(s * r.camera_z_axis_world)
        return np.asarray(vecs, dtype=float)

    def energy(x: np.ndarray) -> float:
        loss = 0.0
        for r in rows:
            if r.variable_index is None:
                continue
            s = float(x[int(r.variable_index)])
            arr = r.selected_final_delta_m[np.isfinite(r.selected_final_delta_m)]
            if arr.size:
                residual = np.maximum(0.0, -margin - (arr + s))
                active = max(1, int(np.count_nonzero(residual > 0.0)))
                loss += depth_w * float(np.sum(residual * residual) / active)
        vec = full_shift_vectors(x)
        if len(vec) > 1:
            vel = vec[1:] - vec[:-1]
            loss += smooth_w * float(np.mean(vel * vel))
        if len(vec) > 2:
            acc = vec[2:] - 2.0 * vec[1:-1] + vec[:-2]
            loss += accel_w * float(np.mean(acc * acc))
        return float(loss)

    result = minimize(energy, feasible_start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1.0e-12, "gtol": 1.0e-10})
    e_current = float(energy(current_zero))
    e_start = float(energy(feasible_start))
    candidate = np.asarray(result.x if np.isfinite(result.fun) else feasible_start, dtype=float)
    ec = float(energy(candidate))
    use_candidate = bool(result.success and np.isfinite(ec) and ec <= e_start + 1.0e-9)
    x = candidate if use_candidate else feasible_start.copy()
    emap = float(energy(x))
    map_in_bounds = bool(all(lo - 1.0e-9 <= float(x[i]) <= hi + 1.0e-9 for i, (lo, hi) in enumerate(bounds)))
    if not map_in_bounds:
        raise ValueError("occluded translation posterior MAP escaped hard translation bounds")
    return x, {
        "state": "optimized_occluded_translation_energy_profile",
        "success": bool(result.success),
        "message": str(result.message),
        "variable_count": int(nvar),
        "energy_at_current_zero_shift": e_current,
        "energy_at_feasible_start": e_start,
        "raw_optimizer_energy": None if not np.isfinite(ec) else ec,
        "used_feasible_start_fallback": bool(not use_candidate),
        "map_in_translation_bounds": map_in_bounds,
        "energy_at_map": emap,
        "energy_reduction_vs_feasible_start": float(e_start - emap),
        "energy_reduction_vs_current_zero_shift": float(e_current - emap),
    }


def side_report(rows: list[PosteriorRow], x: np.ndarray, opt_diag: dict[str, Any], params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    margin = float(params.get("visible_surface_depth_order_margin_m", 0.010) or 0.010)
    sample_count = max(5, int(args.sample_count))
    out_rows: list[dict[str, Any]] = []
    map_shifts: list[float] = []
    clear_required: list[float] = []
    upper_bounds: list[float] = []
    in_front_zero: list[float] = []
    in_front_map: list[float] = []
    state_counts: dict[str, int] = {}
    for r in rows:
        zero_stats = selected_depth_stats(r.selected_final_delta_m, 0.0, margin)
        s_map = 0.0 if r.variable_index is None else float(x[int(r.variable_index)])
        map_stats = selected_depth_stats(r.selected_final_delta_m, s_map, margin)
        req = float(zero_stats.get("clearance_required_camera_z_m", 0.0) or 0.0)
        can_clear = bool(req <= r.upper_s_m + 1.0e-9)
        map_in_bounds = bool(r.lower_s_m - 1.0e-9 <= s_map <= r.upper_s_m + 1.0e-9) if r.variable_index is not None else True
        if not map_in_bounds:
            raise ValueError(f"posterior MAP shift outside bounds for frame={r.frame_idx} side={r.hand_side}: s={s_map} bounds=({r.lower_s_m},{r.upper_s_m})")
        representative_state = "optimized_map" if bool(opt_diag.get("success")) and not bool(opt_diag.get("used_feasible_start_fallback")) else "feasible_bound_projected_fallback_after_optimizer_failure"
        if r.variable_index is None:
            posterior_state = "visible_or_nonzero_observation_row_fixed"
        elif not can_clear:
            posterior_state = "depth_order_conflict_exceeds_translation_bound"
        elif int(map_stats["in_front_count"]) == 0 and representative_state == "optimized_map":
            posterior_state = "map_clears_selected_depth_order"
        elif int(map_stats["in_front_count"]) == 0:
            posterior_state = "fallback_point_clears_selected_depth_order"
        elif bool(opt_diag.get("used_feasible_start_fallback")):
            posterior_state = "clearable_by_bound_but_optimizer_fallback_retains_residual"
        else:
            posterior_state = "temporal_posterior_prefers_residual_over_required_clearance"
        state_counts[posterior_state] = state_counts.get(posterior_state, 0) + 1
        grid = np.linspace(r.lower_s_m, r.upper_s_m, sample_count) if r.variable_index is not None and r.upper_s_m >= r.lower_s_m else np.asarray([0.0])
        grid_stats = [selected_depth_stats(r.selected_final_delta_m, float(s), margin) for s in grid]
        grid_counts = [int(gs["in_front_count"]) for gs in grid_stats]
        grid_residual_energy = [float(gs["residual_energy_unweighted_m2"]) for gs in grid_stats]
        out_rows.append(
            {
                "frame_idx": int(r.frame_idx),
                "hand_side": r.hand_side,
                "posterior_state": posterior_state,
                "hand_observation_visibility_factor_state": r.row.get("hand_observation_visibility_factor_state"),
                "hand_observation_visibility_weight_multiplier": float(r.row.get("hand_observation_visibility_weight_multiplier", 1.0) or 0.0),
                "camera_z_axis_world": r.camera_z_axis_world.astype(float).tolist(),
                "base_state_fingerprint_sha256": row_state_fingerprint(r.row),
                "base_state_fingerprint_fields": list(FINGERPRINT_FIELDS),
                "optimized_translation_world_m": r.optimized_translation_world_m.astype(float).tolist(),
                "additional_camera_z_shift_representative_m": float(s_map),
                "additional_camera_z_shift_representative_state": representative_state,
                "additional_camera_z_shift_map_m": float(s_map),
                "additional_camera_z_shift_map_semantics": "legacy alias for the representative shift; it is an optimized MAP only when additional_camera_z_shift_representative_state == 'optimized_map'",
                "additional_camera_z_shift_lower_bound_m": float(r.lower_s_m),
                "additional_camera_z_shift_upper_bound_m": float(r.upper_s_m),
                "translation_bound_state": r.bound_state,
                "clearance_required_camera_z_m": req,
                "can_clear_selected_depth_order_inside_translation_bound": can_clear,
                "clearance_interval_camera_z_m": [float(max(req, r.lower_s_m)), float(r.upper_s_m)] if can_clear else None,
                "selected_depth_order_final_delta_values_m": r.selected_final_delta_m.astype(float).tolist(),
                "selected_depth_order_selected_vertex_ids": r.row.get("visible_surface_depth_order_selected_vertex_ids") if isinstance(r.row.get("visible_surface_depth_order_selected_vertex_ids"), list) else None,
                "selected_depth_order_selected_surface_depth_m": r.row.get("visible_surface_depth_order_selected_surface_depth_m") if isinstance(r.row.get("visible_surface_depth_order_selected_surface_depth_m"), list) else None,
                "selected_depth_order_zero_shift": zero_stats,
                "selected_depth_order_map_shift": map_stats,
                "grid_profile": {
                    "camera_z_shift_m": grid.astype(float).tolist(),
                    "selected_in_front_count": [int(v) for v in grid_counts],
                    "selected_residual_energy_unweighted_m2": grid_residual_energy,
                },
            }
        )
        if r.variable_index is not None:
            map_shifts.append(s_map)
            clear_required.append(req)
            upper_bounds.append(r.upper_s_m)
            in_front_zero.append(float(zero_stats["in_front_count"]))
            in_front_map.append(float(map_stats["in_front_count"]))
    return {
        "hand_side": rows[0].hand_side if rows else None,
        "optimization": opt_diag,
        "summary": {
            "row_count": len(rows),
            "zero_observation_row_count": int(sum(r.variable_index is not None for r in rows)),
            "posterior_state_counts": dict(sorted(state_counts.items())),
            "map_additional_camera_z_shift_m": numeric_summary(np.asarray(map_shifts, dtype=float)),
            "clearance_required_camera_z_m": numeric_summary(np.asarray(clear_required, dtype=float)),
            "translation_upper_bound_additional_camera_z_m": numeric_summary(np.asarray(upper_bounds, dtype=float)),
            "selected_in_front_count_zero_shift": numeric_summary(np.asarray(in_front_zero, dtype=float)),
            "selected_in_front_count_representative_shift": numeric_summary(np.asarray(in_front_map, dtype=float)),
            "selected_in_front_count_map_shift": numeric_summary(np.asarray(in_front_map, dtype=float)),
            "representative_shift_semantics": "Rows report a representative in-bound shift. It is an optimized MAP only when the side optimization succeeds without feasible-start fallback; otherwise it is a feasible bound-projected fallback point used to expose conflict.",
            "cannot_clear_inside_translation_bound_count": int(sum(1 for r in out_rows if r.get("posterior_state") == "depth_order_conflict_exceeds_translation_bound")),
        },
        "rows": out_rows,
    }


POSTERIOR_PARAM_KEYS = (
    "max_translation_m",
    "visible_surface_depth_order_margin_m",
    "visible_surface_depth_order_weight",
    "smooth_weight",
    "accel_weight",
)


def state_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths = list(args.state or [])
    return paths if paths else [DEFAULT_STATE]


def merge_state_payloads(state_paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    seen: dict[tuple[int, str], Path] = {}
    merged_rows: list[dict[str, Any]] = []
    param_ref: dict[str, Any] | None = None
    for path in state_paths:
        reject_rejected_annotation_path(path, context="state path")
        payload = load_json(path)
        reject_rejected_annotation_path(payload.get("inputs", {}).get("annotations", ""), context=f"state annotation input {path}")
        params = payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {}
        if param_ref is None:
            param_ref = dict(params)
        else:
            for key in POSTERIOR_PARAM_KEYS:
                a = float(param_ref.get(key, 0.0) or 0.0)
                b = float(params.get(key, 0.0) or 0.0)
                if not np.isclose(a, b, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"posterior parameter {key} differs across state files: {a} vs {b} at {path}")
        payloads.append(payload)
        for row in as_list(payload.get("per_frame_states")):
            if not isinstance(row, dict):
                continue
            key = (int(row["frame_idx"]), str(row["hand_side"]))
            if key in seen:
                raise ValueError(f"duplicate posterior state row {key} in {path}; already seen in {seen[key]}")
            seen[key] = path
            merged_rows.append(row)
    merged_rows.sort(key=lambda r: (int(r["frame_idx"]), str(r["hand_side"])))
    if not payloads:
        raise ValueError("no state payloads were loaded")
    merged = dict(payloads[0])
    merged["per_frame_states"] = merged_rows
    raw_inputs = merged.get("inputs")
    base_inputs: dict[str, Any] = dict(raw_inputs) if isinstance(raw_inputs, dict) else {}
    merged["inputs"] = {
        **base_inputs,
        "state_paths": [str(p) for p in state_paths],
    }
    if param_ref is not None:
        merged["parameters"] = param_ref
    return merged, payloads


def build(args: argparse.Namespace) -> dict[str, Any]:
    state_paths = state_paths_from_args(args)
    reject_rejected_annotation_path(args.annotations, context="annotations path")
    state, _payloads = merge_state_payloads(state_paths)
    annotations = load_json(args.annotations)
    reject_rejected_annotation_path(annotations.get("source_annotations") or "", context="annotations source metadata")
    frames_by_idx = load_frames_by_idx(annotations)
    params = state.get("parameters", {}) if isinstance(state.get("parameters"), dict) else {}
    side_reports = []
    for side in sorted({str(r.get("hand_side")) for r in as_list(state.get("per_frame_states")) if isinstance(r, dict)}):
        rows = build_side_rows(state, frames_by_idx, side, args)
        x, diag = optimize_side(rows, params)
        side_reports.append(side_report(rows, x, diag, params, args))
    report = {
        "method": "v18_occluded_hand_translation_posterior_energy_profile_v1",
        "case": "trash_1050",
        "claim_scope": "One-dimensional additional camera-z translation posterior/feasible-set profile for zero-observation MANO rows. It uses the current solved MANO state as the base trajectory and reuses selected visible-surface depth-order residuals, translation bounds, and temporal smoothness/acceleration terms. It is not hidden-hand articulation reconstruction, contact proof, object pose proof, or a calibrated probability distribution.",
        "inputs": {
            "state": str(state_paths[0]) if len(state_paths) == 1 else None,
            "state_paths": [str(p) for p in state_paths],
            "annotations": str(args.annotations),
        },
        "parameters": {
            "zero_observation_weight": float(args.zero_observation_weight),
            "zero_observation_eps": float(args.zero_observation_eps),
            "sample_count": int(args.sample_count),
            "visible_surface_depth_order_margin_m": float(params.get("visible_surface_depth_order_margin_m", 0.010) or 0.010),
            "visible_surface_depth_order_weight": float(params.get("visible_surface_depth_order_weight", 2.0e4) or 2.0e4),
            "max_translation_m": float(params.get("max_translation_m", 0.045) or 0.045),
            "smooth_weight": float(params.get("smooth_weight", 5.0e3) or 5.0e3),
            "accel_weight": float(params.get("accel_weight", 1.0e4) or 1.0e4),
        },
        "side_reports": side_reports,
        "summary": {
            "side_count": len(side_reports),
            "zero_observation_row_count": int(sum(sr["summary"]["zero_observation_row_count"] for sr in side_reports)),
            "posterior_state_counts": dict(sorted({k: sum(sr["summary"]["posterior_state_counts"].get(k, 0) for sr in side_reports) for sr in side_reports for k in sr["summary"]["posterior_state_counts"]}.items())),
            "cannot_clear_inside_translation_bound_count": int(sum(sr["summary"]["cannot_clear_inside_translation_bound_count"] for sr in side_reports)),
        },
    }
    return report


def main() -> None:
    args = parse_args()
    report = build(args)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
