#!/usr/bin/env python3
"""Compare two V19 object-surface posterior states.

The intended use is a mechanism test, not an acceptance validator: compare a
baseline posterior that chooses global nearest object-surface targets with a
candidate posterior that restricts targets to a projected local patch.  Metric
MANO may be preserved in both states; this script measures whether target
selection actually changed the physical surface correspondences and source
hand-to-object gaps.
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


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = payload.get("per_frame_states")
    if isinstance(out, list):
        return [r for r in out if isinstance(r, dict)]
    out = payload.get("frames")
    if isinstance(out, list):
        return [r for r in out if isinstance(r, dict)]
    return []


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row.get("frame_idx", row.get("source_frame_index", -1))), str(row.get("hand_side") or row.get("side") or "")


def finite(vals: list[float]) -> list[float]:
    return [float(v) for v in vals if np.isfinite(float(v))]


def summary(vals: list[float]) -> dict[str, Any]:
    xs = sorted(finite(vals))
    if not xs:
        return {"count": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr[0]),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr[-1]),
        "mean": float(np.mean(arr)),
    }


def metric_summary(row: dict[str, Any], name: str) -> dict[str, Any]:
    refit = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else {}
    val = refit.get(name)
    return val if isinstance(val, dict) else {"count": 0}


def metric_median(row: dict[str, Any], name: str) -> float | None:
    val = metric_summary(row, name).get("median")
    if isinstance(val, (int, float)) and np.isfinite(float(val)):
        return float(val)
    return None


def sample_map(row: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ids = row.get("object_surface_posterior_source_mano_vertex_ids")
    source = np.asarray(row.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
    target = np.asarray(row.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
    if not isinstance(ids, list) or source.ndim != 2 or target.ndim != 2 or source.shape[1:] != (3,) or target.shape[1:] != (3,):
        return {}
    n = min(len(ids), len(source), len(target))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n):
        try:
            vid = int(ids[i])
        except Exception:
            continue
        out[vid] = (source[i], target[i])
    return out


def candidate_stat(row: dict[str, Any], name: str) -> Any:
    refit = row.get("contact_similarity_refit") if isinstance(row.get("contact_similarity_refit"), dict) else {}
    stats = refit.get("candidate_stats") if isinstance(refit.get("candidate_stats"), dict) else {}
    return stats.get(name)


def compare_states(baseline: dict[str, Any], candidate: dict[str, Any], label_baseline: str, label_candidate: str) -> dict[str, Any]:
    base_rows = {row_key(r): r for r in rows(baseline)}
    cand_rows = {row_key(r): r for r in rows(candidate)}
    keys = sorted(set(base_rows) & set(cand_rows))
    base_only = sorted(set(base_rows) - set(cand_rows))
    cand_only = sorted(set(cand_rows) - set(base_rows))

    per_row: list[dict[str, Any]] = []
    base_gap_medians: list[float] = []
    cand_gap_medians: list[float] = []
    base_normal_medians: list[float] = []
    cand_normal_medians: list[float] = []
    base_tangent_medians: list[float] = []
    cand_tangent_medians: list[float] = []
    gap_delta_medians: list[float] = []
    target_delta_shared_medians: list[float] = []
    source_delta_shared_medians: list[float] = []
    shared_vertex_counts: list[float] = []
    jaccards: list[float] = []
    base_sample_counts: list[float] = []
    cand_sample_counts: list[float] = []
    cand_localized_vertices: list[float] = []

    for key in keys:
        br = base_rows[key]
        cr = cand_rows[key]
        bm = sample_map(br)
        cm = sample_map(cr)
        b_ids = set(bm)
        c_ids = set(cm)
        shared = sorted(b_ids & c_ids)
        union = b_ids | c_ids
        target_delta = []
        source_delta = []
        for vid in shared:
            bs, bt = bm[vid]
            cs, ct = cm[vid]
            target_delta.append(float(np.linalg.norm(bt - ct)))
            source_delta.append(float(np.linalg.norm(bs - cs)))
        b_gap = metric_median(br, "source_hand_to_object_surface_distance_m")
        c_gap = metric_median(cr, "source_hand_to_object_surface_distance_m")
        b_normal = metric_median(br, "source_hand_to_object_surface_normal_abs_m")
        c_normal = metric_median(cr, "source_hand_to_object_surface_normal_abs_m")
        b_tangent = metric_median(br, "source_hand_to_object_surface_tangent_m")
        c_tangent = metric_median(cr, "source_hand_to_object_surface_tangent_m")
        if b_gap is not None:
            base_gap_medians.append(b_gap)
        if c_gap is not None:
            cand_gap_medians.append(c_gap)
        if b_gap is not None and c_gap is not None:
            gap_delta_medians.append(c_gap - b_gap)
        if b_normal is not None:
            base_normal_medians.append(b_normal)
        if c_normal is not None:
            cand_normal_medians.append(c_normal)
        if b_tangent is not None:
            base_tangent_medians.append(b_tangent)
        if c_tangent is not None:
            cand_tangent_medians.append(c_tangent)
        if target_delta:
            target_delta_shared_medians.append(float(np.median(target_delta)))
            source_delta_shared_medians.append(float(np.median(source_delta)))
        shared_vertex_counts.append(float(len(shared)))
        jaccards.append(float(len(shared) / len(union)) if union else 0.0)
        base_sample_counts.append(float(len(b_ids)))
        cand_sample_counts.append(float(len(c_ids)))
        loc = candidate_stat(cr, "localized_target_vertices")
        if isinstance(loc, (int, float)) and np.isfinite(float(loc)):
            cand_localized_vertices.append(float(loc))
        per_row.append(
            {
                "frame_idx": key[0],
                "hand_side": key[1],
                "baseline_vertex_count": len(b_ids),
                "candidate_vertex_count": len(c_ids),
                "shared_vertex_count": len(shared),
                "source_vertex_jaccard": float(len(shared) / len(union)) if union else None,
                "baseline_gap_m": b_gap,
                "candidate_gap_m": c_gap,
                "gap_delta_m": (c_gap - b_gap) if b_gap is not None and c_gap is not None else None,
                "baseline_normal_abs_m": b_normal,
                "candidate_normal_abs_m": c_normal,
                "baseline_tangent_m": b_tangent,
                "candidate_tangent_m": c_tangent,
                "shared_target_delta_m": summary(target_delta),
                "shared_source_delta_m": summary(source_delta),
                "candidate_target_locality_px": candidate_stat(cr, "target_locality_px"),
                "candidate_localized_target_vertices": loc,
                "candidate_selected_proximity_px": candidate_stat(cr, "selected_proximity_px"),
                "candidate_selected_current_surface_distance_m": candidate_stat(cr, "selected_current_surface_distance_m"),
            }
        )

    return {
        "method": "v19_surface_posterior_target_locality_comparison",
        "claim_scope": (
            "Compares object-surface posterior target selection while treating metric MANO as a fixed source state. "
            "It does not validate contact ownership, occlusion, nonpenetration, or object-pose correctness."
        ),
        "labels": {"baseline": label_baseline, "candidate": label_candidate},
        "row_counts": {
            "baseline": len(base_rows),
            "candidate": len(cand_rows),
            "matched": len(keys),
            "baseline_only": len(base_only),
            "candidate_only": len(cand_only),
        },
        "summary": {
            "baseline_gap_median_m": summary(base_gap_medians),
            "candidate_gap_median_m": summary(cand_gap_medians),
            "gap_delta_candidate_minus_baseline_m": summary(gap_delta_medians),
            "baseline_normal_abs_median_m": summary(base_normal_medians),
            "candidate_normal_abs_median_m": summary(cand_normal_medians),
            "baseline_tangent_median_m": summary(base_tangent_medians),
            "candidate_tangent_median_m": summary(cand_tangent_medians),
            "shared_vertex_count": summary(shared_vertex_counts),
            "source_vertex_jaccard": summary(jaccards),
            "baseline_vertex_count": summary(base_sample_counts),
            "candidate_vertex_count": summary(cand_sample_counts),
            "shared_target_delta_m": summary(target_delta_shared_medians),
            "shared_source_delta_m": summary(source_delta_shared_medians),
            "candidate_localized_target_vertices": summary(cand_localized_vertices),
        },
        "matched_row_preview": per_row[:80],
        "baseline_only_preview": [{"frame_idx": k[0], "hand_side": k[1]} for k in base_only[:80]],
        "candidate_only_preview": [{"frame_idx": k[0], "hand_side": k[1]} for k in cand_only[:80]],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-state", type=Path, required=True)
    p.add_argument("--candidate-state", type=Path, required=True)
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--candidate-label", default="candidate")
    p.add_argument("--output-report", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_state)
    candidate = load_json(args.candidate_state)
    report = compare_states(baseline, candidate, args.baseline_label, args.candidate_label)
    report["inputs"] = {"baseline_state": str(args.baseline_state), "candidate_state": str(args.candidate_state)}
    write_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "matched_row_preview"}, indent=2)[:20000])


if __name__ == "__main__":
    main()
