#!/usr/bin/env python3
"""Demote a contact-coupled interval MANO refit into a split metric/surface state.

A contact optimizer may reduce hand-object surface residuals by moving the MANO
hand in world/camera space.  If that optimized hand state is not supported as a
metric MANO correction, the physically honest V19 representation is split:

- accepted metric MANO joints remain the source annotation hand state;
- optimized/contact vertices and object-surface correspondences remain as an
  uncertain contact-surface posterior;
- downstream renderers/evaluators can consume the state without mistaking the
  contact-coupled transform for accepted hand pose.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def annotation_joint_map(annotations: dict[str, Any]) -> dict[tuple[int, str], list[list[float]]]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations lacks frames list")
    out: dict[tuple[int, str], list[list[float]]] = {}
    for pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", pos))
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = metric.get("joints_current_v18_world_m") or metric.get("joints_world_m")
            arr = np.asarray(joints if joints is not None else [], dtype=np.float64)
            if side in {"left", "right"} and arr.shape == (21, 3) and np.isfinite(arr).all():
                out[(frame_idx, side)] = arr.tolist()
    if not out:
        raise RuntimeError("annotations contain no usable metric MANO joints")
    return out


def demote(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.interval_state)
    annotations = load_json(args.annotations)
    source_joints = annotation_joint_map(annotations)
    rows = state.get("per_frame_states")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{args.interval_state} lacks nonempty per_frame_states")

    out = copy.deepcopy(state)
    out_rows = out["per_frame_states"]
    replaced = 0
    missing: list[dict[str, Any]] = []
    source_to_candidate_wrist_shift: list[float] = []
    source_to_candidate_joint_shift: list[float] = []
    for row in out_rows:
        if not isinstance(row, dict):
            continue
        key = (int(row.get("frame_idx")), str(row.get("hand_side")))
        src = source_joints.get(key)
        if src is None:
            missing.append({"frame_idx": key[0], "hand_side": key[1]})
            continue
        candidate = np.asarray(row.get("optimized_joints_world_m") or [], dtype=np.float64)
        src_arr = np.asarray(src, dtype=np.float64)
        if candidate.shape == (21, 3) and np.isfinite(candidate).all():
            diff = candidate - src_arr
            source_to_candidate_wrist_shift.append(float(np.linalg.norm(diff[0])))
            source_to_candidate_joint_shift.append(float(np.mean(np.linalg.norm(diff, axis=1))))
            row["demoted_metric_candidate_joints_world_m"] = candidate.tolist()
        row["optimized_joints_world_m"] = src
        row["accepted_metric_joints_world_m"] = src
        row["joint_state_policy"] = "metric_mano_preserved_contact_surface_posterior"
        row["metric_mano_preserved"] = True
        row["temporal_mano_state"] = "v19_contact_surface_posterior_metric_mano_preserved"
        row["metric_correction_demoted_reason"] = args.reason
        row["surface_posterior_source"] = str(args.interval_state)
        row["source_metric_annotations"] = str(args.annotations)
        replaced += 1

    if missing and not args.allow_missing:
        preview = missing[:8]
        raise RuntimeError(f"missing source metric joints for {len(missing)} interval rows; preview={preview}")

    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    summary = dict(summary)
    summary.update(
        {
            "split_state_policy": "metric_mano_preserved_contact_surface_posterior",
            "metric_rows_replaced_from_annotations": int(replaced),
            "missing_source_metric_rows": int(len(missing)),
            "source_to_demoted_candidate_wrist_shift_m": summarize(source_to_candidate_wrist_shift),
            "source_to_demoted_candidate_joint_shift_m": summarize(source_to_candidate_joint_shift),
            "demotion_claim_scope": "accepted metric MANO is copied from source annotations; contact-coupled optimized joints are retained only under demoted_metric_candidate_joints_world_m and must not be scored as accepted hand pose",
        }
    )
    out["summary"] = summary
    out["method"] = "v19_metric_mano_preserved_contact_surface_posterior_state"
    out["claim_scope"] = (
        "Split state: accepted metric MANO remains the source annotation hand; "
        "contact-coupled optimized vertices/correspondences remain as uncertain surface posterior. "
        "This is not contact acceptance, nonpenetration certification, or object-pose scoring."
    )
    out["split_state_adapter"] = {
        "method": "demote_v19_interval_mano_to_surface_posterior",
        "created_unix_s": time.time(),
        "input_interval_state": str(args.interval_state),
        "annotations": str(args.annotations),
        "reason": args.reason,
        "metric_rows_replaced_from_annotations": int(replaced),
        "missing_source_metric_rows": int(len(missing)),
    }
    write_json(args.output_state, out)
    report = {
        "status": "ok",
        "method": "demote_v19_interval_mano_to_surface_posterior",
        "claim_scope": out["claim_scope"],
        "inputs": {"interval_state": str(args.interval_state), "annotations": str(args.annotations)},
        "outputs": {"output_state": str(args.output_state)},
        "summary": summary,
    }
    if args.output_report is not None:
        write_json(args.output_report, report)
    print(json.dumps(report, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-state", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--reason", default="HOT3D metric evaluation or visual review did not support promoting the contact-coupled optimized joints as accepted metric MANO")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    demote(parse_args())


if __name__ == "__main__":
    main()
