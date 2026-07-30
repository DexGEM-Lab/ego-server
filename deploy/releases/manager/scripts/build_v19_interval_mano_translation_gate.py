#!/usr/bin/env python3
"""Build a support-gated V19 interval MANO state.

This is a prediction-side ablation/repair primitive. It does not consume HOT3D GT.
When an interval row has too little selected visible-surface support, it preserves
source HaWoR wrist/root translation and keeps the interval state's wrist-relative
joint articulation. The mechanism tests whether interval MANO regressions come
from ungrounded global translation rather than articulation.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_v19_hot3d_hawor_mano3d import localize_prediction_path, write_json  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def source_npz_for_state(state: dict[str, Any], args: argparse.Namespace) -> Path:
    rows = state.get("per_frame_states")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("interval state lacks nonempty per_frame_states")
    candidates = [r.get("source_hawor_npz") for r in rows if isinstance(r, dict) and r.get("source_hawor_npz")]
    if not candidates:
        raise RuntimeError("interval state rows lack source_hawor_npz; cannot preserve source wrist translation")
    unique = sorted(set(str(c) for c in candidates))
    if len(unique) != 1:
        raise RuntimeError(f"interval state references multiple source_hawor_npz paths: {unique}")
    return localize_prediction_path(unique[0], args)


def build_frame_index(npz: Any, expected_len: int) -> np.ndarray:
    if "frame_idx" in npz.files:
        return np.asarray(npz["frame_idx"], dtype=int)
    return np.arange(expected_len, dtype=int)


def gate_state(args: argparse.Namespace) -> dict[str, Any]:
    interval_path = localize_prediction_path(args.interval_state, args)
    state = load_json(interval_path)
    rows = state.get("per_frame_states")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{interval_path} lacks nonempty per_frame_states")
    source_npz_path = source_npz_for_state(state, args)
    npz = np.load(source_npz_path, allow_pickle=True)
    frame_idx = build_frame_index(npz, len(rows))
    frame_to_i = {int(f): i for i, f in enumerate(frame_idx.tolist())}
    out = copy.deepcopy(state)
    out_rows = out["per_frame_states"]
    counts = {
        "rows_total": 0,
        "rows_gated": 0,
        "rows_kept_positive_support": 0,
        "rows_skipped_side_filter": 0,
        "rows_missing_support_count": 0,
        "rows_missing_source_frame": 0,
        "rows_invalid_joints": 0,
        "rows_missing_baseline": 0,
    }
    displacement_norms: list[float] = []
    for row in out_rows:
        if not isinstance(row, dict):
            continue
        counts["rows_total"] += 1
        side = str(row.get("hand_side"))
        if args.sides and side not in args.sides:
            counts["rows_skipped_side_filter"] += 1
            continue
        if side not in ("left", "right"):
            counts["rows_invalid_joints"] += 1
            continue
        if "visible_surface_depth_order_selected_vertex_count" not in row:
            counts["rows_missing_support_count"] += 1
            if args.require_support_count:
                raise RuntimeError(f"row frame={row.get('frame_idx')} side={side} lacks visible_surface_depth_order_selected_vertex_count")
            support_count = 0
        else:
            support_count = int(row.get("visible_surface_depth_order_selected_vertex_count") or 0)
        if support_count > args.min_support_vertices:
            counts["rows_kept_positive_support"] += 1
            row["translation_gate"] = {"applied": False, "reason": "support_count_above_threshold", "min_support_vertices": args.min_support_vertices}
            continue
        frame = int(row.get("frame_idx"))
        if frame not in frame_to_i:
            counts["rows_missing_source_frame"] += 1
            raise RuntimeError(f"source HaWoR NPZ lacks frame {frame}")
        i = frame_to_i[frame]
        valid_key = f"{side}_valid"
        if valid_key in npz.files and not bool(np.asarray(npz[valid_key]).astype(bool)[i]):
            counts["rows_missing_baseline"] += 1
            raise RuntimeError(f"source HaWoR prediction invalid for frame {frame} side {side}")
        base_key = f"{side}_joints_world_m"
        if base_key not in npz.files:
            raise RuntimeError(f"source HaWoR NPZ lacks {base_key}")
        baseline_joints = np.asarray(npz[base_key][i], dtype=np.float64)
        interval_joints = np.asarray(row.get("optimized_joints_world_m") or [], dtype=np.float64)
        if baseline_joints.shape != (21, 3) or interval_joints.shape != (21, 3):
            counts["rows_invalid_joints"] += 1
            raise RuntimeError(f"invalid joint shape for frame {frame} side {side}: baseline {baseline_joints.shape}, interval {interval_joints.shape}")
        if not np.isfinite(baseline_joints).all() or not np.isfinite(interval_joints).all():
            counts["rows_invalid_joints"] += 1
            raise RuntimeError(f"nonfinite joints for frame {frame} side {side}")
        shift = baseline_joints[0] - interval_joints[0]
        gated_joints = interval_joints + shift[None, :]
        row["optimized_joints_world_m_before_translation_gate"] = interval_joints.tolist() if args.keep_original_joints else None
        if not args.keep_original_joints:
            row.pop("optimized_joints_world_m_before_translation_gate", None)
        row["optimized_joints_world_m"] = gated_joints.tolist()
        samples = row.get("optimized_vertices_world_sample_m")
        if isinstance(samples, list) and samples:
            sample_arr = np.asarray(samples, dtype=np.float64)
            if sample_arr.ndim == 2 and sample_arr.shape[1] == 3 and np.isfinite(sample_arr).all():
                row["optimized_vertices_world_sample_m"] = (sample_arr + shift[None, :]).tolist()
        displacement_norms.append(float(np.linalg.norm(shift)))
        row["translation_gate"] = {
            "applied": True,
            "reason": "visible_surface_support_at_or_below_threshold",
            "min_support_vertices": args.min_support_vertices,
            "support_count": support_count,
            "source_hawor_npz": str(source_npz_path),
            "baseline_wrist_world_m": baseline_joints[0].tolist(),
            "original_interval_wrist_world_m": interval_joints[0].tolist(),
            "applied_world_shift_m": shift.tolist(),
            "applied_world_shift_norm_m": float(np.linalg.norm(shift)),
            "articulation_policy": "preserve interval wrist-relative joints; preserve HaWoR wrist/root translation",
        }
        counts["rows_gated"] += 1
    out.setdefault("postprocess_history", [])
    out["postprocess_history"].append(
        {
            "name": "v19_interval_mano_translation_gate",
            "input_interval_state": str(interval_path),
            "source_hawor_npz": str(source_npz_path),
            "min_support_vertices": args.min_support_vertices,
            "sides": args.sides if args.sides else ["left", "right"],
            "counts": counts,
            "shift_norm_m": {
                "count": len(displacement_norms),
                "median": float(np.median(displacement_norms)) if displacement_norms else None,
                "mean": float(np.mean(displacement_norms)) if displacement_norms else None,
                "p90": float(np.percentile(displacement_norms, 90)) if displacement_norms else None,
                "max": float(np.max(displacement_norms)) if displacement_norms else None,
            },
            "claim_scope": "prediction-side support-gated wrist/root translation ablation; no GT consumed",
        }
    )
    out["translation_gate_summary"] = out["postprocess_history"][-1]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--min-support-vertices", type=int, default=0)
    parser.add_argument("--sides", nargs="*", choices=["left", "right"], default=[])
    parser.add_argument("--keep-original-joints", action="store_true")
    parser.add_argument("--require-support-count", action="store_true")
    parser.add_argument("--remote-root", type=Path, default=None)
    parser.add_argument("--local-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = gate_state(args)
    report = state["translation_gate_summary"]
    write_json(args.output_state, state)
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
