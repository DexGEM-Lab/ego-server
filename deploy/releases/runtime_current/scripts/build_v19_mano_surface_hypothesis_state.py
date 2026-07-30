#!/usr/bin/env python3
"""Build a V19 temporal state that separates metric MANO joints from contact surface hypotheses.

The input contact state may contain a point-to-plane similarity correction whose
surface samples look physically useful but whose optimized MANO joints are not a
valid metric hand state.  This adapter preserves a chosen source MANO joint state
for evaluation/rendered skeletons and carries the optimized surface samples as an
explicit uncertain contact-surface hypothesis.

It does not accept contact, nonpenetration, or ownership.  It encodes a separate
state variable: a local MANO-surface posterior constrained by object geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contact-state", type=Path, required=True, help="Temporal state containing optimized surface/contact rows")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--joint-source", choices=("hawor_npz", "annotations"), default="hawor_npz")
    p.add_argument("--hawor-npz", type=Path, help="HaWoR NPZ used when --joint-source=hawor_npz and stored for evaluator camera trajectory")
    p.add_argument("--annotations", type=Path, help="Annotations JSON used when --joint-source=annotations")
    p.add_argument("--case", default=None)
    p.add_argument("--object-id", default=None)
    return p.parse_args()


def annotation_joint_map(path: Path) -> dict[tuple[int, str], np.ndarray]:
    data = load_json(path)
    rows: dict[tuple[int, str], np.ndarray] = {}
    for pos, frame in enumerate(data.get("frames") if isinstance(data.get("frames"), list) else []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", pos))
        for hand in frame.get("hands") if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side", ""))
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
            if side in {"left", "right"} and joints.shape == (21, 3) and np.isfinite(joints).all():
                rows[(frame_idx, side)] = joints
    return rows


def hawor_joint_map(path: Path) -> dict[tuple[int, str], np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        frame_idx = np.asarray(z["frame_idx"], dtype=int) if "frame_idx" in z.files else None
        if frame_idx is None:
            # Infer from joint array length if frame_idx is missing.
            n = int(np.asarray(z["left_joints_world_m"]).shape[0])
            frame_idx = np.arange(n, dtype=int)
        rows: dict[tuple[int, str], np.ndarray] = {}
        for side in ("left", "right"):
            key = f"{side}_joints_world_m"
            valid_key = f"{side}_valid"
            if key not in z.files:
                continue
            joints_all = np.asarray(z[key], dtype=np.float64)
            valid = np.asarray(z[valid_key]).astype(bool) if valid_key in z.files else np.ones((len(frame_idx),), dtype=bool)
            for i, frame in enumerate(frame_idx.tolist()):
                if i >= len(joints_all) or i >= len(valid) or not bool(valid[i]):
                    continue
                joints = np.asarray(joints_all[i], dtype=np.float64)
                if joints.shape == (21, 3) and np.isfinite(joints).all():
                    rows[(int(frame), side)] = joints
        return rows


def zero_summary() -> dict[str, Any]:
    return {"count": 21, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def row_contact_median(row: dict[str, Any], field: str) -> float | None:
    contact = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else {}
    report = contact.get(field)
    if isinstance(report, dict) and report.get("median") is not None:
        try:
            return float(report["median"])
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    args = parse_args()
    if args.joint_source == "hawor_npz" and args.hawor_npz is None:
        raise SystemExit("--hawor-npz is required with --joint-source=hawor_npz")
    if args.joint_source == "annotations" and args.annotations is None:
        raise SystemExit("--annotations is required with --joint-source=annotations")

    contact_state = load_json(args.contact_state)
    contact_rows = contact_state.get("per_frame_states")
    if not isinstance(contact_rows, list) or not contact_rows:
        raise SystemExit(f"{args.contact_state} lacks nonempty per_frame_states")

    if args.joint_source == "hawor_npz":
        source_joints = hawor_joint_map(args.hawor_npz)  # type: ignore[arg-type]
        source_desc = str(args.hawor_npz)
    else:
        source_joints = annotation_joint_map(args.annotations)  # type: ignore[arg-type]
        source_desc = str(args.annotations)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    normal_vals: list[float] = []
    tangent_vals: list[float] = []
    distance_vals: list[float] = []
    for row in contact_rows:
        if not isinstance(row, dict):
            continue
        try:
            key = (int(row["frame_idx"]), str(row["hand_side"]))
        except Exception:
            continue
        joints = source_joints.get(key)
        if joints is None:
            skipped.append({"frame_idx": key[0], "hand_side": key[1], "reason": "missing_source_joints"})
            continue
        out = dict(row)
        original_joints = np.asarray(row.get("optimized_joints_world_m") or [], dtype=np.float64)
        if original_joints.shape == (21, 3) and np.isfinite(original_joints).all():
            out["surface_fit_joints_world_m"] = original_joints.astype(float).tolist()
            out["surface_fit_joint_delta_from_source_m"] = numeric_summary(np.linalg.norm(original_joints - joints, axis=1).astype(float).tolist())
        out["optimized_joints_world_m"] = joints.astype(float).tolist()
        out["joint_state_policy"] = f"{args.joint_source}_metric_mano_preserved"
        out["temporal_mano_state"] = "v19_source_metric_mano_plus_uncertain_contact_surface_hypothesis"
        out["source_metric_mano_state"] = {"kind": args.joint_source, "path": source_desc}
        if args.hawor_npz is not None:
            out["source_hawor_npz"] = str(args.hawor_npz)
        out["contact_surface_hypothesis_state"] = "uncertain_not_contact_ownership"
        out["contact_surface_vertices_world_sample_m"] = out.get("optimized_vertices_world_sample_m") or []
        out["metric_joint_shift_px"] = zero_summary()
        out["visible_joint_shift_px"] = zero_summary()
        out["optimized_similarity_scale"] = 1.0
        out["optimized_rotation_norm_rad"] = 0.0
        out["optimized_rotation_vector_camera_rad"] = [0.0, 0.0, 0.0]
        out["optimized_rotation_vector_world_rad"] = [0.0, 0.0, 0.0]
        out["optimized_translation_camera_m"] = [0.0, 0.0, 0.0]
        out["optimized_translation_world_m"] = [0.0, 0.0, 0.0]
        for field, vals in (
            ("contact_normal_abs_after_m", normal_vals),
            ("contact_tangent_after_m", tangent_vals),
            ("contact_distance_after_m", distance_vals),
        ):
            val = row_contact_median(row, field)
            if val is not None:
                vals.append(val)
        rows.append(out)

    if not rows:
        raise SystemExit(f"no rows could be built from {args.contact_state}; skipped={skipped[:10]}")

    payload = {
        "method": "v19_source_metric_mano_plus_contact_surface_hypothesis_state",
        "case": args.case or contact_state.get("case"),
        "object_id": args.object_id or contact_state.get("object_id"),
        "claim_scope": (
            "Metric MANO joints are preserved from the selected source; optimized point-to-plane vertices are rendered only as "
            "uncertain contact-surface hypotheses. This state does not accept contact ownership or nonpenetration."
        ),
        "inputs": {
            "contact_state": str(args.contact_state),
            "joint_source": args.joint_source,
            "source_path": source_desc,
            "source_hawor_npz": str(args.hawor_npz) if args.hawor_npz is not None else None,
            "annotations": str(args.annotations) if args.annotations is not None else None,
        },
        "summary": {
            "contact_rows_in": len(contact_rows),
            "rows_out": len(rows),
            "skipped_count": len(skipped),
            "contact_normal_abs_after_median": numeric_summary(normal_vals),
            "contact_tangent_after_median": numeric_summary(tangent_vals),
            "contact_distance_after_median": numeric_summary(distance_vals),
            "metric_joint_shift_px": zero_summary(),
        },
        "skipped_preview": skipped[:50],
        "per_frame_states": rows,
    }
    write_json(args.output, payload)
    report = {k: v for k, v in payload.items() if k != "per_frame_states"}
    write_json(args.output.with_name(args.output.stem + "_report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
