#!/usr/bin/env python3
"""Build a first-order latent MANO depth-uncertainty report from solver slack.

This script consumes a joint interval MANO solver state that records the selected
visible-surface depth-order vertices and their final hand-minus-surface deltas.
For zero-observation latent rows, it computes a camera-z interval: how much the
hand can move toward/away from the visible first surface before selected depth
constraints or the solver translation bound change.  The result bounds hidden
hand depth ambiguity; it does not reconstruct hidden hand pose, contact, object
pose, or nonpenetration.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_temporal_mano_translation_interval_state import frame_camera_pose, load_json, write_json  # noqa: E402


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in vals if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--zero-observation-weight", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = load_json(args.solver_state)
    annotations_path = Path(str(state.get("inputs", {}).get("annotations") or ""))
    if not annotations_path.exists():
        raise FileNotFoundError(f"solver state does not name a readable annotations file: {annotations_path}")
    annotations = load_json(annotations_path)
    frames_by_idx = {int(f["frame_idx"]): f for f in annotations.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    margin = float(state.get("parameters", {}).get("visible_surface_depth_order_margin_m", 0.010))
    rows: list[dict[str, Any]] = []
    for st in state.get("per_frame_states", []):
        if not isinstance(st, dict):
            continue
        try:
            obs_weight = float(st.get("hand_observation_visibility_weight_multiplier", 1.0) or 0.0)
        except Exception:
            obs_weight = 1.0
        if obs_weight != float(args.zero_observation_weight):
            continue
        final = np.asarray(st.get("visible_surface_depth_order_selected_final_delta_values_m") or [], dtype=float)
        initial = np.asarray(st.get("visible_surface_depth_order_selected_initial_delta_hand_minus_surface_m") or [], dtype=float)
        if final.size == 0:
            continue
        frame_idx = int(st["frame_idx"])
        side = str(st["hand_side"])
        frame = frames_by_idx.get(frame_idx)
        if frame is None:
            raise ValueError(f"annotations missing frame {frame_idx} required by solver state")
        r_c2w, _t_c2w = frame_camera_pose(frame)
        camera_z_axis_world = np.asarray(r_c2w[:, 2], dtype=float)
        min_delta = float(np.min(final))
        current_in_front = int(np.count_nonzero(final < -margin))
        additional_to_clear = max(0.0, -margin - min_delta)
        toward_camera_before_violation = max(0.0, min_delta + margin)
        remaining_farther = float(st.get("translation_bound_remaining_camera_z_m", 0.0) or 0.0)
        row = {
            "frame_idx": frame_idx,
            "hand_side": side,
            "state": "latent_observation_invalid_depth_interval_probe",
            "hand_observation_weight_multiplier": obs_weight,
            "selected_visible_surface_vertex_count": int(final.size),
            "selected_current_in_front_count": current_in_front,
            "selected_initial_in_front_count": int(np.count_nonzero(initial < -margin)) if initial.size else None,
            "selected_min_delta_hand_minus_surface_m": min_delta,
            "selected_median_delta_hand_minus_surface_m": float(np.median(final)),
            "additional_camera_z_shift_to_clear_selected_m": float(additional_to_clear),
            "camera_z_shift_toward_camera_before_new_selected_violation_m": float(toward_camera_before_violation),
            "optimized_translation_camera_z_m": float(st.get("optimized_translation_camera_z_m", 0.0) or 0.0),
            "optimized_translation_lateral_norm_m": float(st.get("optimized_translation_lateral_norm_m", 0.0) or 0.0),
            "translation_bound_remaining_farther_camera_z_m": remaining_farther,
            "camera_z_axis_world": camera_z_axis_world.astype(float).tolist(),
            "farther_bound_translation_world_m": (max(0.0, remaining_farther) * camera_z_axis_world).astype(float).tolist(),
            "clearance_translation_world_m": (additional_to_clear * camera_z_axis_world).astype(float).tolist(),
            "can_clear_selected_depth_order_within_translation_bound": bool(additional_to_clear <= max(0.0, remaining_farther) + 1.0e-9),
            "bound_state": "current_translation_exceeds_or_saturates_bound" if remaining_farther < 0 else "has_farther_depth_slack",
            "claim_scope": "First-order selected-vertex camera-depth interval for latent/zero-observation MANO rows. Positive camera-z shift moves the hand farther behind the visible first surface. This bounds hidden-depth ambiguity; it does not reconstruct hidden articulation/contact/object pose.",
        }
        rows.append(row)
    by_side: dict[str, Any] = {}
    for side in sorted({r["hand_side"] for r in rows}):
        rr = [r for r in rows if r["hand_side"] == side]
        by_side[side] = {
            "row_count": len(rr),
            "frames": [min(r["frame_idx"] for r in rr), max(r["frame_idx"] for r in rr)] if rr else None,
            "selected_current_in_front_count": numeric_summary([r["selected_current_in_front_count"] for r in rr]),
            "additional_camera_z_shift_to_clear_selected_m": numeric_summary([r["additional_camera_z_shift_to_clear_selected_m"] for r in rr]),
            "near_front_slack_m": numeric_summary([r["camera_z_shift_toward_camera_before_new_selected_violation_m"] for r in rr]),
            "translation_bound_remaining_farther_camera_z_m": numeric_summary([r["translation_bound_remaining_farther_camera_z_m"] for r in rr]),
            "optimized_translation_camera_z_m": numeric_summary([r["optimized_translation_camera_z_m"] for r in rr]),
            "cannot_clear_inside_translation_bound_count": sum(1 for r in rr if not r["can_clear_selected_depth_order_within_translation_bound"]),
            "bound_saturated_or_exceeded_count": sum(1 for r in rr if r["translation_bound_remaining_farther_camera_z_m"] < 0.0),
        }
    report = {
        "method": "v18_latent_depth_uncertainty_from_selected_visible_surface_slack",
        "case": str(state.get("case")),
        "object_id": str(state.get("object_id")),
        "inputs": {"solver_state": str(args.solver_state), "annotations": str(annotations_path)},
        "parameters": {
            "visible_surface_depth_order_margin_m": margin,
            "max_translation_m": state.get("parameters", {}).get("max_translation_m"),
            "zero_observation_weight": float(args.zero_observation_weight),
        },
        "claim_scope": "Computes a first-order camera-depth uncertainty interval for zero-observation latent MANO rows using solver-selected visible first-surface vertices and translation-bound slack. It is not a hidden-hand reconstruction and not a contact/nonpenetration claim.",
        "summary": {"row_count": len(rows), "by_side": by_side},
        "rows": rows,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2)[:6000])


if __name__ == "__main__":
    main()
