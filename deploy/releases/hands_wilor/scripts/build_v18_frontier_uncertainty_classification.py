#!/usr/bin/env python3
"""Classify remaining V18 frontier MANO uncertainty by physical cause.

This is a final-artifact consumption step, not a validator and not a new solver.
It reads the current interval-MANO frontier backing states and records why the
important unresolved hand-state intervals remain uncertain: normal measurement
/support noise, physical information limits, or implementation/dataflow defects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FRONTIER_ROOT = Path("/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def span_ranges(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    vals = sorted(set(int(f) for f in frames))
    spans: list[list[int]] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        spans.append([start, prev])
        start = prev = value
    spans.append([start, prev])
    return spans


def numeric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def rows_by_condition(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def frame_set(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["frame_idx"]) for row in rows if row.get("frame_idx") is not None})


def sides(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("hand_side")) for row in rows if row.get("hand_side") is not None})


def frame_spans_by_hand_side(rows: list[dict[str, Any]]) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    for side in sides(rows):
        out[side] = span_ranges(frame_set([row for row in rows if str(row.get("hand_side")) == side]))
    return out


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["frame_idx"]), str(row.get("hand_side")))


def residual_max(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if not isinstance(raw, dict):
        return None
    return optional_float(raw.get("max"))


def classify_case(case: str, case_dir: Path, manifest_case: dict[str, Any]) -> dict[str, Any]:
    backing_path = case_dir / "frontier_interval_mano_states.json"
    backing = load_json(backing_path)
    rows = [row for row in backing.get("per_frame_states", []) if isinstance(row, dict)]
    state_summary = backing.get("state_summary") if isinstance(backing.get("state_summary"), dict) else {}
    all_frames = frame_set(rows)
    optimized_frame_count = len(all_frames)
    full_frame_count = int(state_summary.get("full_video_frame_count") or 0)
    context_count = int(state_summary.get("context_passthrough_frame_count") or max(0, full_frame_count - optimized_frame_count))

    support_rows = rows_by_condition(
        rows,
        lambda row: (optional_float(row.get("observed_surface_support_uncertainty_m")) or 0.0) > 0.0,
    )
    contact_support_rows = rows_by_condition(
        rows,
        lambda row: (optional_float(row.get("contact_patch_support_uncertainty_m")) or 0.0) > 0.0,
    )
    def is_zero_observation(row: dict[str, Any]) -> bool:
        multiplier = optional_float(row.get("hand_observation_visibility_weight_multiplier"))
        return multiplier is not None and multiplier <= 1.0e-9

    zero_obs_rows = rows_by_condition(rows, is_zero_observation)
    visible_depth_order_rows = rows_by_condition(
        rows,
        lambda row: ((optional_float(row.get("visible_surface_depth_order_selected_final_in_front_count")) or 0.0) > 0.0)
        and not is_zero_observation(row),
    )

    classified_keys: set[tuple[int, str]] = set()
    classifications: list[dict[str, Any]] = []
    if support_rows:
        classified_keys.update(row_key(row) for row in support_rows)
        vals = [optional_float(row.get("observed_surface_support_uncertainty_m")) or 0.0 for row in support_rows]
        residual_over_support: list[dict[str, Any]] = []
        residual_over_support_values: list[float] = []
        for row in support_rows:
            support = optional_float(row.get("observed_surface_support_uncertainty_m")) or 0.0
            trusted = residual_max(row, "full_observed_surface_penetration_after_solver_m")
            if trusted is not None and trusted > support:
                diff = float(trusted - support)
                residual_over_support_values.append(diff)
                residual_over_support.append({
                    "frame_idx": int(row["frame_idx"]),
                    "hand_side": str(row.get("hand_side")),
                    "residual_minus_support_m": diff,
                })
        classifications.append({
            "cause_class": "normal_measurement_noise_carried_by_object_support_uncertainty",
            "phenomenon": "The MANO/object residual being corrected is at or below the independently measured object-surface support scale; forcing a sharper hand correction would overclaim object pose/depth precision.",
            "frame_spans": span_ranges(frame_set(support_rows)),
            "frame_spans_by_hand_side": frame_spans_by_hand_side(support_rows),
            "hand_sides": sides(support_rows),
            "state_count": len(support_rows),
            "support_uncertainty_m": numeric_summary(vals),
            "residual_exceeds_support_count": len(residual_over_support),
            "residual_minus_support_positive_m": numeric_summary(residual_over_support_values),
            "residual_exceeds_support_examples": residual_over_support[:12],
            "scope_note": "Support-bounded does not mean every scalar residual is below the support bound; small residual-above-support exceptions remain bounded visually and must not be used as confident contact/nonpenetration closure.",
            "rendered_consequence": "Magenta support-bounded MANO uncertainty on optimized interval frames rather than confident exact-surface/nonpenetration correction.",
            "would_be_falsified_by": "A rendered interval where the visible hand/object relation is physically impossible while the backing state labels it only as support noise, or a new independent object/patch support source tighter than the residual scale.",
        })
    if contact_support_rows and not support_rows:
        classified_keys.update(row_key(row) for row in contact_support_rows)
        vals = [optional_float(row.get("contact_patch_support_uncertainty_m")) or 0.0 for row in contact_support_rows]
        classifications.append({
            "cause_class": "normal_measurement_noise_carried_by_contact_patch_support_uncertainty",
            "phenomenon": "Contact-patch evidence reaches H_t but remains support-bounded; contact state is not a sharper hand-pose observation by itself.",
            "frame_spans": span_ranges(frame_set(contact_support_rows)),
            "frame_spans_by_hand_side": frame_spans_by_hand_side(contact_support_rows),
            "hand_sides": sides(contact_support_rows),
            "state_count": len(contact_support_rows),
            "contact_patch_support_uncertainty_m": numeric_summary(vals),
            "rendered_consequence": "Magenta contact/support uncertainty, not solved contact closure.",
            "would_be_falsified_by": "Stable, independently supported contact-manifold observations that make the patch support tighter than the MANO residuals and visibly move H_t coherently.",
        })
    if zero_obs_rows:
        classified_keys.update(row_key(row) for row in zero_obs_rows)
        classifications.append({
            "cause_class": "physical_information_limit_from_occlusion_and_invalid_hand_observation",
            "phenomenon": "The hand observation is explicitly zero-weighted because RGB/ownership/depth evidence does not support a visible MANO observation; the hidden hand cannot be reconstructed as a certain pose from available sensors.",
            "frame_spans": span_ranges(frame_set(zero_obs_rows)),
            "frame_spans_by_hand_side": frame_spans_by_hand_side(zero_obs_rows),
            "hand_sides": sides(zero_obs_rows),
            "state_count": len(zero_obs_rows),
            "rendered_consequence": "Latent/occluded MANO hypothesis rendered with magenta uncertainty; in the current Trash frontier this includes hard-bound camera-z posterior endpoints where available.",
            "would_be_falsified_by": "Visible hand evidence through the alleged occlusion, an independent depth-order/contact observation that narrows the hidden trajectory, or a render showing a confident hidden pose without uncertainty.",
        })
    if visible_depth_order_rows:
        classified_keys.update(row_key(row) for row in visible_depth_order_rows)
        classifications.append({
            "cause_class": "bounded_depth_order_measurement_noise_or_partial_occlusion",
            "phenomenon": "Visible first-surface depth-order factors constrain H_t, but remaining conflicts are carried as bounded uncertainty rather than contact/nonpenetration closure.",
            "frame_spans": span_ranges(frame_set(visible_depth_order_rows)),
            "frame_spans_by_hand_side": frame_spans_by_hand_side(visible_depth_order_rows),
            "hand_sides": sides(visible_depth_order_rows),
            "state_count": len(visible_depth_order_rows),
            "selected_final_in_front_count": numeric_summary([optional_float(row.get("visible_surface_depth_order_selected_final_in_front_count")) or 0.0 for row in visible_depth_order_rows]),
            "rendered_consequence": "Bounded occlusion/first-surface trajectory, not accepted hidden-volume nonpenetration.",
            "would_be_falsified_by": "Residual clusters shown to be stale/mis-owned first-surface pixels or a visually impossible hand/object relation after optimization.",
        })

    posterior_info = manifest_case.get("occluded_translation_posterior_report") if isinstance(manifest_case.get("occluded_translation_posterior_report"), dict) else None
    if posterior_info:
        summary = ((posterior_info.get("summary") or {}).get("summary") if isinstance(posterior_info.get("summary"), dict) else {}) or {}
        classifications.append({
            "cause_class": "physical_information_limit_from_saturated_fixed_base_camera_z_posterior",
            "phenomenon": "For zero-observation occluded-hand rows, shifting the current solved MANO along camera z usually cannot clear selected visible first-surface conflicts within the solver translation bound. The available evidence supports broad/conflicted uncertainty, not hidden-hand reconstruction.",
            "frame_spans": "see source_occluded_translation_posterior_report.json side rows; summarized over all zero-observation rows in the current posterior report, which may span multiple interval state files",
            "posterior_state_counts": summary.get("posterior_state_counts"),
            "zero_observation_row_count": summary.get("zero_observation_row_count"),
            "cannot_clear_inside_translation_bound_count": summary.get("cannot_clear_inside_translation_bound_count"),
            "rendered_consequence": "Magenta lower/upper hard-bound camera-z skeleton endpoints on hard occlusion rows.",
            "would_be_falsified_by": "A new observation that constrains lateral/root/articulation hidden modes or a valid optimized posterior that narrows the hidden trajectory without conflicting with visible first-surface evidence.",
        })

    ordinary_rows = [row for row in rows if row_key(row) not in classified_keys]
    if ordinary_rows:
        classifications.append({
            "cause_class": "ordinary_optimized_interval_state_no_frontier_uncertainty_flag",
            "phenomenon": "These optimized interval MANO rows have no active support-bound, zero-observation, or nonzero visible-depth conflict flag in the current backing state. They are ordinary optimized interval states, not context-only frames and not a separate unresolved frontier mechanism.",
            "frame_spans": span_ranges(frame_set(ordinary_rows)),
            "frame_spans_by_hand_side": frame_spans_by_hand_side(ordinary_rows),
            "hand_sides": sides(ordinary_rows),
            "state_count": len(ordinary_rows),
            "rendered_consequence": "Cyan/yellow optimized interval MANO without extra magenta frontier uncertainty beyond the base interval hypothesis.",
            "would_be_falsified_by": "A visible contradiction in these rows, or backing evidence that should have activated a support/occlusion/depth-order uncertainty mechanism but did not.",
        })

    optimized_frames = set(all_frames)
    context_frames = [f for f in range(full_frame_count) if f not in optimized_frames] if full_frame_count > 0 else []
    if context_count:
        classifications.append({
            "cause_class": "context_only_no_new_interval_mano_claim",
            "phenomenon": "Frames without interval solver states are rendered as full-video context/passthrough frames. They preserve raw-duration consumption but do not claim new MANO correction.",
            "frame_spans": span_ranges(context_frames),
            "state_count": context_count,
            "rendered_consequence": "Original/context MANO evidence only; not a solved interval correction.",
            "would_be_falsified_by": "A closure claim that treats context-only frames as optimized/corrected interval MANO states.",
        })

    return {
        "case": case,
        "backing_state": str(backing_path),
        "frontier_claim_scope": backing.get("frontier_claim_scope"),
        "optimized_state_count": len(rows),
        "unique_optimized_frame_count": optimized_frame_count,
        "full_video_frame_count": full_frame_count,
        "implementation_dataflow_defect_assessment": {
            "defects_found_in_inspected_scope": [],
            "basis": [
                "Representative current-frontier overlay/world/side-by-side sheets were consumed as annotation.",
                "The current frontier manifest and render manifests preserve sanitized annotation roots and posterior provenance.",
                "The classification is descriptive; this field is not an exhaustive proof that no possible dataflow defect exists.",
            ],
            "would_be_revised_by": "A rendered frame or backing state where H_t contradicts visible evidence, uses rejected provenance, omits a required uncertainty mechanism, or treats context-only frames as corrected interval states.",
        },
        "uncertainty_classifications": classifications,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontier-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    frontier_root = args.frontier_root
    manifest_path = frontier_root / "v18_current_frontier_interval_mano_artifact_manifest.json"
    manifest = load_json(manifest_path)
    cases_in = manifest.get("cases") if isinstance(manifest.get("cases"), dict) else {}
    cases: dict[str, Any] = {}
    for case, data in cases_in.items():
        if not isinstance(data, dict):
            continue
        cases[str(case)] = classify_case(str(case), frontier_root / str(case), data)

    report = {
        "method": "build_v18_frontier_uncertainty_classification",
        "purpose": "Attach physical causes to the current frontier's remaining MANO uncertainty so support limits, occlusion information limits, and context-only frames are not mistaken for solved pose.",
        "frontier_root": str(frontier_root),
        "frontier_manifest": str(manifest_path),
        "claim_scope": {
            "not_a_validator": True,
            "does_not_change_H_t": True,
            "supports_scoped_v18_bounded_mano_closure": True,
            "closure_support_role": "causal classification of the rendered support and occlusion uncertainty that remains after the reusable workbench mechanisms have been consumed by the final MANO artifact",
            "does_not_claim_solved_contact_object_pose_nonpenetration_or_hidden_hand": True,
        },
        "visual_consumption_evidence": {
            "task5_sheet": "/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/task5_tomato_960/current_frontier_interval_mano_review.jpg",
            "trash_sheet": "/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/trash_1050/current_frontier_interval_mano_review.jpg",
            "observation": "Representative v5 overlay/world/side-by-side frames were consumed as annotation. Task5 support spans show plausible visible hand/object alignment with magenta support uncertainty; Trash 970-971 remain visible, 972 transitions to latent uncertainty, 988-1002 show broad hard-bound posterior endpoints, left 1004-1008 continue as bounded observation-invalid/posterior rows, and 1009/1020 return to ordinary visible/context behavior.",
        },
        "cases": cases,
    }
    out = args.output or (frontier_root / "v18_frontier_uncertainty_classification.json")
    write_json(out, report)
    manifest["uncertainty_classification"] = str(out)
    write_json(manifest_path, manifest)
    print(json.dumps({"output": str(out), "case_count": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
