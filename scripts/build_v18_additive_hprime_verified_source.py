#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_key(raw: str) -> tuple[int, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("expected FRAME_IDX:HAND_SIDE")
    frame, side = raw.split(":", 1)
    side = side.strip()
    if not side:
        raise argparse.ArgumentTypeError("empty hand side")
    return int(frame), side


def index_hands(ann: dict[str, Any]) -> dict[tuple[int, str], tuple[int, int, dict[str, Any]]]:
    out: dict[tuple[int, str], tuple[int, int, dict[str, Any]]] = {}
    for frame_i, frame in enumerate(ann.get("frames", []) if isinstance(ann.get("frames"), list) else []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx"))
        for hand_i, hand in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
            if not isinstance(hand, dict):
                continue
            key = (frame_idx, str(hand.get("hand_side")))
            if key in out:
                raise RuntimeError(f"duplicate hand key {key}")
            out[key] = (frame_i, hand_i, hand)
    return out


def index_rows(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("constraint_rows", []) if isinstance(report.get("constraint_rows"), list) else []:
        if not isinstance(row, dict):
            continue
        try:
            key = (int(row.get("frame_idx")), str(row.get("hand_side")))
        except Exception:
            continue
        out[key] = row
    return out


def compact_update(hand: dict[str, Any]) -> dict[str, Any]:
    update = hand.get("compact_rigid_object_mano_constraint_update")
    if isinstance(update, dict):
        return update
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    update = metric.get("compact_rigid_object_constraint_update")
    if isinstance(update, dict):
        return update
    raise RuntimeError("hand has no compact-rigid update")


def is_coordinate_hprime(hand: dict[str, Any]) -> bool:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    update = compact_update(hand)
    return bool(update.get("coordinate_update_applied") is True or metric.get("compact_rigid_object_corrected_h_prime") is True)


def state_counts(ann: dict[str, Any]) -> dict[str, int]:
    counts = {"corrected": 0, "uncertainty": 0, "validated_no_change": 0, "other_update": 0}
    for _, _, hand in index_hands(ann).values():
        try:
            update = compact_update(hand)
        except RuntimeError:
            continue
        state = str(update.get("h_prime_state") or "")
        if is_coordinate_hprime(hand):
            counts["corrected"] += 1
        elif state == "unchanged_with_compact_rigid_object_overlap_uncertainty":
            counts["uncertainty"] += 1
        elif state == "validated_no_compact_rigid_object_coordinate_change":
            counts["validated_no_change"] += 1
        else:
            counts["other_update"] += 1
    return counts


def proof_payload(row: dict[str, Any], proof_path: Path, primary_state: str) -> dict[str, Any]:
    return {
        "application_state": row.get("candidate_application_state"),
        "combined_remeasurement_source": row.get("combined_remeasurement_source"),
        "remeasure_report": str(proof_path),
        "sign_mesh_path": row.get("sign_mesh_path"),
        "sign_mesh_source_report": row.get("sign_mesh_source_report"),
        "nearest_surface_unsigned_m": row.get("nearest_surface_unsigned_m"),
        "signed_distance_m": row.get("signed_distance_m"),
        "penetrating_vertex_count": row.get("penetrating_vertex_count"),
        "verified_no_additional_coordinate_change": True,
        "primary_prior_noncontradiction_state": primary_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified H-prime source annotation by adding selected post-verified rows from an alternate prior.")
    parser.add_argument("--base-annotations", type=Path, required=True, help="Existing verified-source annotation to start from, e.g. final-v1 source")
    parser.add_argument("--candidate-annotations", type=Path, required=True, help="Annotation containing coordinate-updated candidate rows")
    parser.add_argument("--primary-remeasure-report", type=Path, required=True, help="Accepted prior/sign remeasurement report used for existing corrected rows")
    parser.add_argument("--secondary-remeasure-report", type=Path, required=True, help="Alternate prior/sign remeasurement report after applying candidate rows")
    parser.add_argument("--add-key", action="append", type=parse_key, required=True, help="FRAME_IDX:HAND_SIDE row to copy from candidate annotation")
    parser.add_argument("--primary-allowed-state", action="append", default=["uncertainty_sign_mesh_missing_near_surface_support"], help="Primary-prior row states allowed as non-contradiction for added rows")
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--combined-remeasure-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-cause", required=True)
    parser.add_argument("--rejected-candidates-note", required=True)
    args = parser.parse_args()

    base = load_json(args.base_annotations)
    candidate = load_json(args.candidate_annotations)
    primary_report = load_json(args.primary_remeasure_report)
    secondary_report = load_json(args.secondary_remeasure_report)
    primary_rows = index_rows(primary_report)
    secondary_rows = index_rows(secondary_report)
    base_index = index_hands(base)
    candidate_index = index_hands(candidate)
    add_keys = set(args.add_key)

    output = copy.deepcopy(base)
    output_index = index_hands(output)
    added: list[dict[str, Any]] = []
    for key in sorted(add_keys):
        if key not in output_index:
            raise RuntimeError(f"selected key {key} missing from base/output annotations")
        if key not in candidate_index:
            raise RuntimeError(f"selected key {key} missing from candidate annotations")
        primary_row = primary_rows.get(key)
        secondary_row = secondary_rows.get(key)
        if not isinstance(primary_row, dict) or not isinstance(secondary_row, dict):
            raise RuntimeError(f"selected key {key} missing primary or secondary remeasurement row")
        primary_state = str(primary_row.get("candidate_application_state") or "")
        secondary_state = str(secondary_row.get("candidate_application_state") or "")
        if primary_state not in set(args.primary_allowed_state):
            raise RuntimeError(f"selected key {key}: primary state {primary_state!r} is not an allowed non-contradiction state")
        if secondary_state != "no_penetration_no_coordinate_change_needed" or secondary_row.get("penetrating_vertex_count") not in {0, 0.0}:
            raise RuntimeError(f"selected key {key}: secondary proof did not verify no-penetration")
        new_hand = copy.deepcopy(candidate_index[key][2])
        if not is_coordinate_hprime(new_hand):
            raise RuntimeError(f"selected key {key}: candidate hand is not a coordinate H-prime row")
        update = compact_update(new_hand)
        update["additive_hprime_selection"] = {
            "selected_for_verified_source": True,
            "selection_cause": args.selection_cause,
            "primary_prior_noncontradiction_state": primary_state,
            "secondary_prior_post_hprime_state": secondary_state,
            "rejected_candidates_note": args.rejected_candidates_note,
        }
        update["post_hprime_verification"] = proof_payload(secondary_row, args.combined_remeasure_output, primary_state)
        update["row_level_signed_proof"] = "combined row-level signed remeasurement with existing corrected rows from primary prior and selected added rows from secondary prior"
        new_hand["compact_rigid_object_mano_constraint_update"] = update
        if isinstance(new_hand.get("metric_mano_state"), dict):
            new_hand["metric_mano_state"]["compact_rigid_object_constraint_update"] = update
            new_hand["metric_mano_state"]["compact_rigid_object_corrected_h_prime"] = True
        frame_i, hand_i, _ = output_index[key]
        output["frames"][frame_i]["hands"][hand_i] = new_hand
        added.append({
            "frame_idx": key[0],
            "hand_side": key[1],
            "primary_state": primary_state,
            "secondary_state": secondary_state,
            "candidate_translation_norm_m": update.get("candidate_translation_norm_m"),
        })

    combined = copy.deepcopy(primary_report)
    combined["method"] = "combined_additive_hprime_remeasurement"
    combined["status"] = "ok"
    combined["claim_scope"] = "Row-level signed proof for corrected H-prime rows: existing corrected rows verified by the primary report; selected additive rows verified by the secondary report after candidate application."
    combined["source_reports"] = {
        "primary_remeasure_report": str(args.primary_remeasure_report),
        "secondary_remeasure_report": str(args.secondary_remeasure_report),
    }
    out_rows: list[dict[str, Any]] = []
    for row in combined.get("constraint_rows", []) if isinstance(combined.get("constraint_rows"), list) else []:
        key = (int(row.get("frame_idx")), str(row.get("hand_side")))
        if key in add_keys:
            replacement = copy.deepcopy(secondary_rows[key])
            replacement["combined_remeasurement_source"] = "secondary_remeasure_report"
            out_rows.append(replacement)
        else:
            kept = copy.deepcopy(row)
            kept["combined_remeasurement_source"] = "primary_remeasure_report"
            out_rows.append(kept)
    combined["constraint_rows"] = out_rows
    combined["added_hprime_keys"] = [{"frame_idx": k[0], "hand_side": k[1]} for k in sorted(add_keys)]

    write_json(args.output_annotations, output)
    write_json(args.combined_remeasure_output, combined)
    manifest = {
        "method": "build_v18_additive_hprime_verified_source",
        "status": "ok",
        "base_annotations": str(args.base_annotations),
        "candidate_annotations": str(args.candidate_annotations),
        "primary_remeasure_report": str(args.primary_remeasure_report),
        "secondary_remeasure_report": str(args.secondary_remeasure_report),
        "output_annotations": str(args.output_annotations),
        "combined_remeasure_output": str(args.combined_remeasure_output),
        "added_hprime_rows": added,
        "state_counts": state_counts(output),
        "claim_scope": "Builds a verified H-prime source by adding only selected rows where the primary prior is noncontradictory and the secondary prior post-verifies no penetration.",
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
