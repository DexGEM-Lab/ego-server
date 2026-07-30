#!/usr/bin/env python3
"""Attach source-gap contact likelihoods to a V19 surface-posterior state.

Contact requires the source MANO surface and object surface to be coincident
within measurement uncertainty.  Direct object-surface posterior states already
store paired source hand vertices and object-surface targets; this script turns
that physical gap into an explicit compatibility score while preserving all
metric MANO, object, and posterior geometry fields.

The score is not contact ownership, nonpenetration, a calibrated posterior
probability, or a metric correction. It is a Gaussian residual compatibility
score under a stated positional uncertainty scale.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def finite(vals: list[float]) -> list[float]:
    return [float(v) for v in vals if np.isfinite(float(v))]


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    xs = sorted(finite(vals))
    if not xs:
        return {"count": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr[0]),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(arr[-1]),
        "mean": float(np.mean(arr)),
    }


def row_gap_samples(row: dict[str, Any]) -> np.ndarray:
    source = np.asarray(row.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
    target = np.asarray(row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
    if source.ndim == 2 and target.ndim == 2 and source.shape[1:] == (3,) and target.shape[1:] == (3,):
        n = min(len(source), len(target))
        if n:
            return np.linalg.norm(source[:n] - target[:n], axis=1)
    refit = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else {}
    dist = refit.get("source_hand_to_object_surface_distance_m") or refit.get("contact_distance_after_m")
    if isinstance(dist, dict) and isinstance(dist.get("median"), (int, float)):
        return np.asarray([float(dist["median"])], dtype=np.float64)
    return np.zeros((0,), dtype=np.float64)


def state_label(z: float | None, p: float | None) -> str:
    if z is None or p is None:
        return "unresolved_no_gap_measurement"
    if z >= 3.0:
        return "contact_unlikely_source_gap_exceeds_3sigma"
    if z >= 2.0:
        return "contact_low_likelihood_source_gap_exceeds_2sigma"
    if z >= 1.0:
        return "near_contact_uncertain_within_2sigma"
    return "near_contact_compatible_by_source_gap_only"


def build(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.input_state)
    out = copy.deepcopy(payload)
    rows = out.get("per_frame_states") if isinstance(out.get("per_frame_states"), list) else []
    sigma = math.sqrt(float(args.hand_sigma_m) ** 2 + float(args.object_sigma_m) ** 2 + float(args.depth_order_sigma_m) ** 2)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise RuntimeError(f"invalid combined sigma {sigma}")
    row_median_gaps: list[float] = []
    row_median_z: list[float] = []
    row_median_p: list[float] = []
    row_states: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        gaps = row_gap_samples(row)
        probs = np.exp(-0.5 * np.square(gaps / sigma)) if len(gaps) else np.zeros((0,), dtype=np.float64)
        z_vals = gaps / sigma if len(gaps) else np.zeros((0,), dtype=np.float64)
        gap_summary = numeric_summary(gaps.astype(float).tolist())
        p_summary = numeric_summary(probs.astype(float).tolist())
        z_summary = numeric_summary(z_vals.astype(float).tolist())
        z_med = z_summary.get("median") if isinstance(z_summary.get("median"), (int, float)) else None
        p_med = p_summary.get("median") if isinstance(p_summary.get("median"), (int, float)) else None
        label = state_label(float(z_med) if z_med is not None else None, float(p_med) if p_med is not None else None)
        row_states[label] = row_states.get(label, 0) + 1
        if isinstance(gap_summary.get("median"), (int, float)):
            row_median_gaps.append(float(gap_summary["median"]))
        if z_med is not None:
            row_median_z.append(float(z_med))
        if p_med is not None:
            row_median_p.append(float(p_med))
        likelihood = {
            "model": "source_gap_gaussian_contact_compatibility",
            "claim_scope": "Gaussian compatibility score for paired source MANO and object-surface samples being geometrically coincident within stated independent positional uncertainty; not a calibrated probability, contact ownership, or nonpenetration.",
            "hand_sigma_m": float(args.hand_sigma_m),
            "object_sigma_m": float(args.object_sigma_m),
            "depth_order_sigma_m": float(args.depth_order_sigma_m),
            "combined_sigma_m": float(sigma),
            "source_gap_m": gap_summary,
            "source_gap_z": z_summary,
            "contact_compatibility_score": p_summary,
            "likelihood_state": label,
        }
        row["contact_likelihood"] = likelihood
        refit = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else None
        if isinstance(refit, dict):
            refit["contact_likelihood"] = likelihood
    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    summary = dict(summary)
    summary.update(
        {
            "contact_compatibility_score_median": numeric_summary(row_median_p),
            "source_gap_z_median": numeric_summary(row_median_z),
            "source_gap_for_likelihood_median_m": numeric_summary(row_median_gaps),
            "contact_likelihood_state_counts": row_states,
            "contact_likelihood_model": {
                "model": "source_gap_gaussian_contact_compatibility",
                "hand_sigma_m": float(args.hand_sigma_m),
                "object_sigma_m": float(args.object_sigma_m),
                "depth_order_sigma_m": float(args.depth_order_sigma_m),
                "combined_sigma_m": float(sigma),
                "basis": str(args.basis),
            },
        }
    )
    out["summary"] = summary
    out["method"] = str(out.get("method") or "v19_surface_posterior_state") + "+source_gap_contact_likelihood"
    out["source_gap_contact_likelihood_claim_scope"] = (
        "Adds an explicit source-gap contact compatibility likelihood. Metric MANO joints, object pose, and surface posterior samples are unchanged. "
        "Low compatibility means the current source geometry is physically inconsistent with contact under the stated uncertainty scale."
    )
    inputs = out.get("inputs") if isinstance(out.get("inputs"), dict) else {}
    inputs = dict(inputs)
    inputs["source_gap_contact_likelihood_input_state"] = str(args.input_state)
    out["inputs"] = inputs
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-state", type=Path, required=True)
    p.add_argument("--output-state", type=Path, required=True)
    p.add_argument("--hand-sigma-m", type=float, default=0.027, help="Metric hand positional uncertainty scale; default from clip001849 WiLoR/HaWoR median MPJPE scale.")
    p.add_argument("--object-sigma-m", type=float, default=0.010, help="Object surface/pose uncertainty scale in meters.")
    p.add_argument("--depth-order-sigma-m", type=float, default=0.010, help="Depth/order ambiguity scale in meters.")
    p.add_argument("--basis", default="sqrt(hand_sigma^2 + object_sigma^2 + depth_order_sigma^2); contact requires source and object surfaces to coincide within this metric uncertainty")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    write_json(args.output_state, payload)
    report = {k: v for k, v in payload.items() if k != "per_frame_states"}
    write_json(args.output_state.with_name(args.output_state.stem + "_report.json"), report)
    print(json.dumps({"status": "ok", "output": str(args.output_state), "summary": payload.get("summary")}, indent=2)[:20000])


if __name__ == "__main__":
    main()
