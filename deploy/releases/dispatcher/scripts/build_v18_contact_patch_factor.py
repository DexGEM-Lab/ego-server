#!/usr/bin/env python3
"""Build generic V18 contact_patch factor records from existing contact hypotheses.

This script does not infer contact from labels alone. It promotes contact
hypotheses into solver inputs that can affect H_t: the solver will select
current MANO vertices near eligible observed object surface and add a
near-contact patch residual. In latent weighting mode, supported contacts and
raw near-contact proposals both survive as false-positive-tolerant candidates;
their row weights are derived from current annotation evidence such as visual
association, object-owned contact-patch depth compatibility, latent contact
state support, temporal ownership, and metric proximity. When an independent
object-pose support report is provided, the emitted residual is bounded by that
support uncertainty so contact acts as a latent/sliding patch likelihood rather
than a hard current-surface anchor. The factor is object-agnostic; target
differences are data fields, not code branches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    finite = sorted(float(v) for v in vals if isinstance(v, (int, float)))
    if not finite:
        return {"count": 0}
    def q(frac: float) -> float:
        idx = min(len(finite) - 1, max(0, int(round(frac * (len(finite) - 1)))))
        return finite[idx]
    return {
        "count": len(finite),
        "min": finite[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "max": finite[-1],
    }


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_object_support_uncertainty(path: Path | None, *, stat: str, default_m: float) -> dict[int, float]:
    if path is None:
        return {}
    payload = load_json(path)
    pose_rows = payload.get("pose_rows")
    if not isinstance(pose_rows, list):
        raise ValueError(f"object pose fit report has no pose_rows list: {path}")
    out: dict[int, float] = {}
    for row in pose_rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        raw = nested_get(row, stat)
        if raw is None:
            raw = default_m
        try:
            val = float(raw)
        except Exception:
            val = float(default_m)
        if val < 0.0:
            val = float(default_m)
        out[int(row["frame_idx"])] = val
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--target-entity-id", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weight", type=float, default=5.0e4)
    p.add_argument("--contact-patch-band-m", type=float, default=0.020)
    p.add_argument("--contact-patch-target-margin-m", type=float, default=0.0025)
    p.add_argument("--max-vertices", type=int, default=96)
    p.add_argument("--object-pose-fit-report", type=Path, default=None, help="Optional independent object pose/support report. When supplied, per-frame support uncertainty is added to the contact deadband so the factor cannot force sub-support-scale MANO motion.")
    p.add_argument("--object-support-uncertainty-stat", default="observed_to_mesh_final.p95_m", help="Dotted field in pose_rows[] used as object_support_uncertainty_m. Default uses visible-depth-to-mesh p95 support.")
    p.add_argument("--default-object-support-uncertainty-m", type=float, default=0.0, help="Fallback support uncertainty when the pose report is absent or lacks the selected stat.")
    p.add_argument("--contact-evidence-report", type=Path, default=None, help="Optional external contact evidence report for diagnostics. Current annotation evidence remains the primary V18 contact source.")
    p.add_argument("--require-independent-contact-evidence", action="store_true", help="Diagnostic strict mode: reject rows unless the external/current evidence has matching visual association plus metric-depth compatibility or an accepted contact owner. Do not use as the default latent-contact method.")
    p.add_argument("--include-unsupported-near", action="store_true", help="Include raw near-contact proposals without final support as candidate latent contacts. In latent weighting mode these receive lower weights instead of hard acceptance.")
    p.add_argument("--latent-contact-weighting", action="store_true", help="Use current annotation evidence to emit evidence-weighted latent contact candidates rather than uniform-weight supported-contact rows.")
    p.add_argument("--raw-proposal-weight-factor", type=float, default=0.35, help="Weight multiplier for raw near-contact proposals in latent-contact mode.")
    p.add_argument("--depth-conflict-weight-factor", type=float, default=0.10, help="Weight multiplier when current annotation evidence marks a depth contradiction in latent-contact mode; rows are downweighted, not automatically deleted.")
    p.add_argument("--min-latent-contact-weight-fraction", type=float, default=0.05, help="Minimum nonzero weight fraction for emitted latent contact candidates.")
    return p.parse_args()


def load_contact_evidence(path: Path | None, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"contact evidence report has no rows list: {path}")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("object_id")) != str(target_entity_id):
            continue
        side = str(row.get("hand_side") or "")
        if side not in {"left", "right"}:
            continue
        key = (int(row["frame_idx"]), side)
        if key in out:
            raise ValueError(f"duplicate contact evidence row for {key} and target {target_entity_id}: {path}")
        out[key] = row
    return out


def independent_contact_evidence_supported(row: dict[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(row, dict):
        return False, "missing_independent_contact_evidence_row"
    ev = as_dict(row.get("source_contact_evidence"))
    graph = as_dict(ev.get("contact_ownership_graph"))
    accepted_owner = bool(graph.get("accepted_contact_owner")) or str(row.get("contact_owner_claim") or "").startswith("accepted")
    image_supported = bool(ev.get("image_overlap_candidate") or ev.get("pair_contact_image_candidate"))
    metric_supported = bool(ev.get("metric_depth_compatible_candidate"))
    depth_gap_state = str(ev.get("pair_depth_gap_state") or "")
    source_state = str(row.get("source_contact_state") or "")
    if source_state == "image_contact_rejected_by_metric_depth" or depth_gap_state in {"hand_behind_object_depth", "object_behind_hand_depth"}:
        return False, f"metric_depth_rejects_visual_contact:{source_state}:{depth_gap_state}"
    if accepted_owner:
        return True, "accepted_contact_owner"
    if image_supported and metric_supported:
        return True, "visual_association_and_metric_depth_compatible"
    if image_supported and not metric_supported:
        return False, "visual_association_without_metric_depth_support"
    if metric_supported and not image_supported:
        return False, "metric_depth_without_visual_association"
    return False, "no_independent_visual_metric_contact_support"


def current_annotation_contact_evidence(row: dict[str, Any]) -> dict[str, Any]:
    evidence = as_dict(row.get("evidence"))
    final_switch = as_dict(row.get("final_contact_switch"))
    coupling = as_dict(row.get("active_contact_coupling_state"))
    metric = as_dict(row.get("final_metric_contact_evidence"))
    raw_depth = as_dict(evidence.get("raw_depth_conflict_strength"))
    local_support = as_dict(coupling.get("local_rigid_visible_surface_contact_state_support"))
    return {
        "evidence": evidence,
        "final_switch": final_switch,
        "coupling": coupling,
        "metric": metric,
        "raw_depth": raw_depth,
        "local_support": local_support,
    }


def current_contact_evidence_supported(row: dict[str, Any]) -> tuple[bool, str]:
    parts = current_annotation_contact_evidence(row)
    evidence = as_dict(parts.get("evidence"))
    final_switch = as_dict(parts.get("final_switch"))
    coupling = as_dict(parts.get("coupling"))
    local_support = as_dict(parts.get("local_support"))
    depth_state = str(evidence.get("pair_depth_gap_state") or "")
    image_supported = bool(evidence.get("image_overlap_candidate") or evidence.get("pair_contact_image_candidate"))
    metric_supported = bool(evidence.get("metric_depth_compatible_candidate")) or depth_state.endswith("depth_compatible")
    accepted_owner = bool(as_dict(evidence.get("contact_ownership_graph")).get("accepted_contact_owner"))
    latent_supported = bool(final_switch.get("post_graph_latent_rigid_contact_supported") or coupling.get("contact_state_affects_latent_contact_state") or local_support.get("supported"))
    if image_supported and metric_supported:
        return True, "current_annotation_visual_and_object_owned_depth_compatible"
    if accepted_owner and metric_supported:
        return True, "current_annotation_owner_and_depth_compatible"
    if latent_supported and metric_supported:
        return True, "current_annotation_latent_contact_and_depth_compatible"
    return False, "current_annotation_lacks_visual_metric_latent_contact_support"


def contact_supported(row: dict[str, Any], *, include_unsupported_near: bool) -> bool:
    if row.get("physical_contact_claim_supported") is True:
        return True
    if not include_unsupported_near:
        return False
    state = str(row.get("state") or row.get("contact_physical_mode") or "")
    evidence = as_dict(row.get("final_metric_contact_evidence"))
    return state == "raw_contact_proposal_without_final_validated_physical_support" and evidence.get("contact_switch_observation") == "near"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def latent_contact_score(row: dict[str, Any], *, raw_proposal_weight_factor: float, depth_conflict_weight_factor: float, min_weight_fraction: float) -> dict[str, Any]:
    parts = current_annotation_contact_evidence(row)
    evidence = as_dict(parts.get("evidence"))
    final_switch = as_dict(parts.get("final_switch"))
    coupling = as_dict(parts.get("coupling"))
    metric = as_dict(parts.get("metric"))
    raw_depth = as_dict(parts.get("raw_depth"))
    local_support = as_dict(parts.get("local_support"))
    reasons: list[str] = []
    score = 0.0
    state = str(row.get("state") or "")
    if row.get("physical_contact_claim_supported") is True:
        score += 0.25
        reasons.append("supported_physical_contact_claim")
    if final_switch.get("post_graph_latent_rigid_contact_supported") is True:
        score += 0.20
        reasons.append("post_graph_latent_rigid_contact_supported")
    if coupling.get("contact_state_affects_latent_contact_state") is True:
        score += 0.15
        reasons.append("contact_state_affects_latent_contact_state")
    if local_support.get("supported") is True:
        score += 0.15
        reasons.append("local_visible_surface_contact_state_supported")
    depth_state = str(evidence.get("pair_depth_gap_state") or raw_depth.get("raw_pair_depth_gap_state") or "")
    if bool(evidence.get("metric_depth_compatible_candidate")) or depth_state.endswith("depth_compatible"):
        score += 0.15
        reasons.append("object_owned_contact_patch_depth_compatible")
    if bool(evidence.get("pair_contact_image_candidate") or evidence.get("image_overlap_candidate")):
        score += 0.10
        reasons.append("visual_contact_or_overlap_candidate")
    try:
        raw_min_distance = metric.get("min_distance_m")
        raw_near_band = metric.get("near_contact_band_m")
        min_distance = float(raw_min_distance) if raw_min_distance is not None else float("nan")
        near_band = float(raw_near_band) if raw_near_band is not None else 0.0
    except Exception:
        min_distance = float("nan")
        near_band = 0.0
    if near_band > 0.0 and min_distance == min_distance:
        proximity = clamp01(1.0 - min_distance / near_band)
        score += 0.10 * proximity
        reasons.append("metric_near_contact_distance")
    raw_candidate = state == "raw_contact_proposal_without_final_validated_physical_support"
    if raw_candidate:
        score *= max(0.0, float(raw_proposal_weight_factor))
        reasons.append("raw_near_contact_proposal_downweighted")
    current_depth_conflict = bool(raw_depth.get("raw_depth_contradiction")) or "depth_contradicted" in state or ("incompatible" in depth_state and not depth_state.endswith("depth_compatible"))
    if current_depth_conflict:
        score *= max(0.0, float(depth_conflict_weight_factor))
        reasons.append("current_depth_conflict_downweighted_not_deleted")
    score = clamp01(score)
    if score > 0.0:
        score = max(score, max(0.0, float(min_weight_fraction)))
    return {
        "latent_contact_confidence": score,
        "latent_contact_weight_fraction": score,
        "latent_contact_reasons": reasons,
        "latent_contact_depth_state": depth_state or None,
        "latent_contact_raw_candidate": raw_candidate,
        "latent_contact_current_depth_conflict": current_depth_conflict,
        "latent_contact_variable_id": coupling.get("latent_contact_state_variable_id") or final_switch.get("latent_contact_state_variable_id"),
        "local_contact_patch_variable_id": coupling.get("local_rigid_visible_contact_patch_variable_id"),
    }


def main() -> None:
    args = parse_args()
    payload = load_json(args.annotations)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"annotations file has no frames list: {args.annotations}")
    support_by_frame = load_object_support_uncertainty(
        args.object_pose_fit_report,
        stat=str(args.object_support_uncertainty_stat),
        default_m=float(args.default_object_support_uncertainty_m),
    )
    contact_evidence = load_contact_evidence(args.contact_evidence_report, str(args.target_entity_id))
    if bool(args.require_independent_contact_evidence) and args.contact_evidence_report is None:
        raise ValueError("--require-independent-contact-evidence requires --contact-evidence-report")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.start_frame) or frame_idx > int(args.end_frame):
            continue
        for hyp in frame.get("contact_hypotheses") or []:
            if not isinstance(hyp, dict):
                continue
            if str(hyp.get("object_id")) != str(args.target_entity_id):
                continue
            side = str(hyp.get("hand_side") or "")
            if side not in {"left", "right"}:
                skipped.append({"frame_idx": frame_idx, "reason": "missing_or_invalid_hand_side", "hypothesis": hyp})
                continue
            if not contact_supported(hyp, include_unsupported_near=bool(args.include_unsupported_near)):
                skipped.append({"frame_idx": frame_idx, "hand_side": side, "reason": "contact_not_supported", "state": hyp.get("state"), "physical_contact_claim_supported": hyp.get("physical_contact_claim_supported")})
                continue
            independent_evidence_row = contact_evidence.get((frame_idx, side)) if contact_evidence else None
            external_supported, external_reason = independent_contact_evidence_supported(independent_evidence_row)
            current_supported, current_reason = current_contact_evidence_supported(hyp)
            independent_supported = bool(current_supported or external_supported)
            independent_reason = current_reason if current_supported else external_reason
            if bool(args.require_independent_contact_evidence) and not independent_supported:
                skipped.append({
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "reason": "independent_contact_evidence_rejected",
                    "independent_contact_evidence_reason": independent_reason,
                    "annotation_state": hyp.get("state"),
                    "annotation_contact_owner_hypothesis": hyp.get("contact_owner_hypothesis"),
                })
                continue
            latent = latent_contact_score(
                hyp,
                raw_proposal_weight_factor=float(args.raw_proposal_weight_factor),
                depth_conflict_weight_factor=float(args.depth_conflict_weight_factor),
                min_weight_fraction=float(args.min_latent_contact_weight_fraction),
            ) if bool(args.latent_contact_weighting) else {
                "latent_contact_confidence": 1.0,
                "latent_contact_weight_fraction": 1.0,
                "latent_contact_reasons": ["uniform_supported_contact_weight"],
                "latent_contact_depth_state": None,
                "latent_contact_raw_candidate": False,
                "latent_contact_current_depth_conflict": False,
                "latent_contact_variable_id": None,
                "local_contact_patch_variable_id": None,
            }
            if bool(args.latent_contact_weighting) and float(latent.get("latent_contact_weight_fraction", 0.0)) <= 0.0:
                skipped.append({"frame_idx": frame_idx, "hand_side": side, "reason": "latent_contact_zero_weight", "state": hyp.get("state")})
                continue
            evidence = as_dict(hyp.get("final_metric_contact_evidence"))
            coupling = as_dict(hyp.get("active_contact_coupling_state"))
            stable_anchor_candidate = bool(coupling.get("stable_contact_pose_anchor_candidate"))
            stable_anchor_emitted = bool(coupling.get("stable_contact_pose_anchor_factor_emitted"))
            contact_anchor_state = (
                "stable_pose_anchor_emitted"
                if stable_anchor_emitted
                else ("stable_pose_anchor_candidate_not_emitted" if stable_anchor_candidate else "local_visible_surface_contact_only_no_stable_pose_anchor")
            )
            support_uncertainty_m = float(support_by_frame.get(frame_idx, float(args.default_object_support_uncertainty_m)))
            contact_deadband_m = float(args.contact_patch_target_margin_m) + max(0.0, support_uncertainty_m)
            row_weight = float(args.weight) * float(latent.get("latent_contact_weight_fraction", 1.0))
            rows.append(
                {
                    "factor_family": "contact_patch",
                    "target_entity_id": str(args.target_entity_id),
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "variable_affected": "H_t",
                    "observation_type": "evidence_weighted_latent_contact_to_uncertain_observed_visible_surface_patch" if bool(args.latent_contact_weighting) else "supported_active_contact_to_uncertain_observed_visible_surface_patch",
                    "residual_or_quarantine_rule": "select current MANO vertices near eligible observed object surface and penalize surface-normal distance only beyond contact_patch_target_margin_m + object_support_uncertainty_m while allowing tangential sliding; do not consume as a persistent object-frame A_t pose anchor unless contact_anchor_residual_allowed is true",
                    "rendered_uncertainty_channel": "bounded latent/sliding contact patch MANO hypothesis; no object pose, hidden geometry, or persistent point-anchor claim unless stable anchor fields explicitly allow it",
                    "state": "active_contact_patch",
                    "weight": float(row_weight),
                    "contact_patch_base_weight": float(args.weight),
                    "contact_patch_band_m": float(args.contact_patch_band_m),
                    "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
                    "object_support_uncertainty_m": max(0.0, support_uncertainty_m),
                    "contact_patch_support_uncertainty_m": max(0.0, support_uncertainty_m),
                    "contact_patch_deadband_m": contact_deadband_m,
                    "contact_anchor_state": contact_anchor_state,
                    "contact_anchor_residual_allowed": bool(stable_anchor_emitted),
                    "contact_anchor_blockers": coupling.get("blockers") if isinstance(coupling.get("blockers"), list) else [],
                    "contact_pose_anchor_key": coupling.get("contact_pose_anchor_key"),
                    "max_vertices": int(args.max_vertices),
                    "source_contact_state": hyp.get("state"),
                    "source_contact_owner_hypothesis": hyp.get("contact_owner_hypothesis"),
                    "source_min_distance_m": evidence.get("min_distance_m"),
                    "source_near_contact_band_m": evidence.get("near_contact_band_m"),
                    "source_object_support_uncertainty_stat": str(args.object_support_uncertainty_stat),
                    "source_object_pose_fit_report": str(args.object_pose_fit_report) if args.object_pose_fit_report else None,
                    "source_contact_coupling_state": coupling.get("coupling_state"),
                    "source_stable_contact_pose_anchor_candidate": stable_anchor_candidate,
                    "source_stable_contact_pose_anchor_factor_emitted": stable_anchor_emitted,
                    "independent_contact_evidence_supported": bool(independent_supported),
                    "independent_contact_evidence_reason": independent_reason,
                    "current_annotation_contact_evidence_supported": bool(current_supported),
                    "current_annotation_contact_evidence_reason": current_reason,
                    "external_contact_evidence_supported": bool(external_supported),
                    "external_contact_evidence_reason": external_reason,
                    "source_contact_evidence_report": str(args.contact_evidence_report) if args.contact_evidence_report else None,
                    **latent,
                    "provenance": {
                        "annotations": str(args.annotations),
                        "frame_contact_hypothesis_key": "frames[].contact_hypotheses[]",
                        "selection_rule": "target object, supported active or included raw near-contact candidate, side in left/right, evidence-weighted latent contact when requested, strict independent evidence only in diagnostic mode",
                        "final_metric_contact_evidence": evidence,
                        "active_contact_coupling_state": coupling,
                        "independent_contact_evidence_row": independent_evidence_row,
                    },
                }
            )
    seen: set[tuple[int, str]] = set()
    deduped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["frame_idx"]), str(row["hand_side"]))
        if key in seen:
            duplicates.append(row)
            continue
        seen.add(key)
        deduped.append(row)
    if duplicates:
        raise ValueError(f"duplicate contact_patch rows for target/side/frame: {duplicates[:3]}")
    report = {
        "method": "v18_contact_patch_factor_from_supported_contact_hypotheses",
        "case": str(args.case),
        "target_entity_id": str(args.target_entity_id),
        "claim_scope": "Generic H_t contact patch residual input. It can constrain MANO hand state near observed eligible surface patches; it does not by itself prove object pose, hidden geometry, contact closure, or nonpenetration.",
        "inputs": {
            "annotations": str(args.annotations),
            "object_pose_fit_report": str(args.object_pose_fit_report) if args.object_pose_fit_report else None,
            "contact_evidence_report": str(args.contact_evidence_report) if args.contact_evidence_report else None,
        },
        "parameters": {
            "start_frame": int(args.start_frame),
            "end_frame": int(args.end_frame),
            "weight": float(args.weight),
            "contact_patch_band_m": float(args.contact_patch_band_m),
            "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
            "max_vertices": int(args.max_vertices),
            "object_support_uncertainty_stat": str(args.object_support_uncertainty_stat),
            "default_object_support_uncertainty_m": float(args.default_object_support_uncertainty_m),
            "include_unsupported_near": bool(args.include_unsupported_near),
            "require_independent_contact_evidence": bool(args.require_independent_contact_evidence),
            "latent_contact_weighting": bool(args.latent_contact_weighting),
            "raw_proposal_weight_factor": float(args.raw_proposal_weight_factor),
            "depth_conflict_weight_factor": float(args.depth_conflict_weight_factor),
            "min_latent_contact_weight_fraction": float(args.min_latent_contact_weight_fraction),
        },
        "summary": {
            "factor_row_count": len(deduped),
            "skipped_count": len(skipped),
            "frames": sorted({int(r["frame_idx"]) for r in deduped}),
            "sides": sorted({str(r["hand_side"]) for r in deduped}),
            "object_support_uncertainty_m": numeric_summary([float(r.get("object_support_uncertainty_m", 0.0)) for r in deduped]),
            "contact_patch_deadband_m": numeric_summary([float(r.get("contact_patch_deadband_m", 0.0)) for r in deduped]),
            "latent_contact_confidence": numeric_summary([float(r.get("latent_contact_confidence", 0.0)) for r in deduped]),
            "row_weight": numeric_summary([float(r.get("weight", 0.0)) for r in deduped]),
            "latent_raw_candidate_count": sum(1 for r in deduped if r.get("latent_contact_raw_candidate") is True),
            "latent_current_depth_conflict_count": sum(1 for r in deduped if r.get("latent_contact_current_depth_conflict") is True),
            "contact_anchor_state_counts": {state: sum(1 for r in deduped if str(r.get("contact_anchor_state")) == state) for state in sorted({str(r.get("contact_anchor_state")) for r in deduped})},
            "contact_anchor_residual_allowed_count": sum(1 for r in deduped if r.get("contact_anchor_residual_allowed") is True),
            "current_annotation_contact_evidence_supported_count": sum(1 for r in deduped if r.get("current_annotation_contact_evidence_supported") is True),
            "independent_contact_evidence_rejected_count": sum(1 for r in skipped if r.get("reason") == "independent_contact_evidence_rejected"),
        },
        "factor_rows": deduped,
        "skipped_rows_sample": skipped[:50],
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
