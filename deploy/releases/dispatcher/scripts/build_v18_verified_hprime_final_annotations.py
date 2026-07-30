#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_VERIFIED_ANNOTATIONS = {
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_temporal_guard_v1/task5_tomato_960/object_obj_tomato/surface806_sign929_full_bridge_all_signed_temporal_guard/iter1_select/annotations_v18_full_with_verified_tomato_full_signed_temporal_guard_hprime.json"),
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_temporal_guard_v1/trash_1050/object_pink_lid_trash_can_second/frame872_full_bridge_all_signed_temporal_guard/iter1_select/annotations_v18_full_with_verified_trash_full_signed_temporal_guard_hprime.json"),
}

DOMINANT_TOKEN = "dominant_visible_part"

POINT_METRIC_FIELDS = {
    "joints_current_v18_world_m": (21, 3),
    "joints_current_v18_camera_m": (21, 3),
    "vertices_world_sample_m": (None, 3),
    "vertices_camera_sample_m": (None, 3),
}
VECTOR_METRIC_FIELDS = {
    "wrist_current_v18_world_m": (3,),
    "current_v18_camera_intrinsics_fx_fy_cx_cy": (4,),
}
COPY_METRIC_FIELDS = [
    *POINT_METRIC_FIELDS.keys(),
    *VECTOR_METRIC_FIELDS.keys(),
    "mano_params",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_case_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected CASE=/path/to/annotations.json")
    case, path = raw.split("=", 1)
    case = case.strip()
    if not case:
        raise argparse.ArgumentTypeError("empty case name")
    return case, Path(path)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def require_int_field(node: dict[str, Any], field: str, label: str) -> int:
    value = node.get(field)
    if value is None:
        raise RuntimeError(f"{label}: missing {field}")
    return int(value)


def compact_update(hand: dict[str, Any]) -> dict[str, Any] | None:
    update = hand.get("compact_rigid_object_mano_constraint_update")
    if isinstance(update, dict):
        return update
    metric = as_dict(hand.get("metric_mano_state"))
    update = metric.get("compact_rigid_object_constraint_update")
    return update if isinstance(update, dict) else None


def hprime_state(update: dict[str, Any] | None) -> str:
    return str(update.get("h_prime_state") or "") if isinstance(update, dict) else ""


def is_coordinate_hprime(hand: dict[str, Any], update: dict[str, Any] | None = None) -> bool:
    metric = as_dict(hand.get("metric_mano_state"))
    return bool(
        (isinstance(update, dict) and update.get("coordinate_update_applied") is True)
        or metric.get("compact_rigid_object_corrected_h_prime") is True
    )


def hand_key(frame: dict[str, Any], hand: dict[str, Any]) -> tuple[int, str]:
    return require_int_field(frame, "frame_idx", "frame"), str(hand.get("hand_side"))


def index_hands(ann: dict[str, Any], label: str) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for frame_raw in as_list(ann.get("frames")):
        if not isinstance(frame_raw, dict):
            continue
        frame = frame_raw
        seen_sides: set[str] = set()
        frame_idx = require_int_field(frame, "frame_idx", f"{label} frame")
        for hand_raw in as_list(frame.get("hands")):
            if not isinstance(hand_raw, dict):
                continue
            hand = hand_raw
            side = str(hand.get("hand_side"))
            if side in seen_sides:
                raise RuntimeError(f"{label}: duplicate hand side {side!r} in frame {frame_idx}")
            seen_sides.add(side)
            key = (frame_idx, side)
            if key in out:
                raise RuntimeError(f"{label}: duplicate hand key {key}")
            out[key] = hand
    return out


def finite_json_numbers(node: Any, path: str = "root") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            finite_json_numbers(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            finite_json_numbers(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if not math.isfinite(float(node)):
            raise RuntimeError(f"non-finite numeric value at {path}")


def require_array(raw: Any, shape: tuple[int | None, ...], label: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != len(shape):
        raise RuntimeError(f"{label}: expected {len(shape)} dimensions, got shape {arr.shape}")
    for dim, expected in zip(arr.shape, shape):
        if expected is not None and dim != expected:
            raise RuntimeError(f"{label}: expected shape {shape}, got {arr.shape}")
    if any(expected is None for expected in shape) and any(dim <= 0 for dim in arr.shape):
        raise RuntimeError(f"{label}: expected non-empty shape, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{label}: contains non-finite values")
    return arr


def validate_corrected_verified_metric(metric: dict[str, Any], *, case: str, key: tuple[int, str]) -> None:
    for field, shape in POINT_METRIC_FIELDS.items():
        if field not in metric:
            raise RuntimeError(f"{case} {key}: verified corrected H' missing metric field {field}")
        require_array(metric[field], shape, f"{case} {key} verified {field}")
    for field, shape in VECTOR_METRIC_FIELDS.items():
        if field not in metric:
            raise RuntimeError(f"{case} {key}: verified corrected H' missing metric field {field}")
        require_array(metric[field], shape, f"{case} {key} verified {field}")
    params = metric.get("mano_params")
    if not isinstance(params, dict):
        raise RuntimeError(f"{case} {key}: verified corrected H' missing mano_params dict")
    finite_json_numbers(params, f"{case} {key} mano_params")
    trans = np.asarray(params.get("trans_world_m"), dtype=float)
    if trans.shape != (3,) or not np.isfinite(trans).all():
        raise RuntimeError(f"{case} {key}: verified corrected H' mano_params.trans_world_m is not finite 3-vector")


def frame_alignment_signature(ann: dict[str, Any]) -> list[tuple[int, str, float]]:
    sig: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for frame_raw in as_list(ann.get("frames")):
        if not isinstance(frame_raw, dict):
            continue
        frame = frame_raw
        idx = require_int_field(frame, "frame_idx", "frame alignment")
        if idx in seen:
            raise RuntimeError(f"duplicate frame_idx {idx}")
        seen.add(idx)
        time_raw = frame.get("time_s")
        time_s = float(time_raw) if isinstance(time_raw, (int, float)) else float("nan")
        if not math.isfinite(time_s):
            raise RuntimeError(f"frame {idx}: non-finite time_s")
        sig.append((idx, str(frame.get("raw_frame_path") or ""), round(time_s, 9)))
    return sig


def assert_source_alignment(case: str, base_ann: dict[str, Any], verified_ann: dict[str, Any]) -> None:
    if str(base_ann.get("case")) != str(verified_ann.get("case")) or str(base_ann.get("case")) != case:
        raise RuntimeError(f"{case}: case identity mismatch between base and verified annotations")
    if base_ann.get("raw_video") != verified_ann.get("raw_video"):
        raise RuntimeError(f"{case}: raw_video identity mismatch between base and verified annotations")
    base_sig = frame_alignment_signature(base_ann)
    verified_sig = frame_alignment_signature(verified_ann)
    if base_sig != verified_sig:
        raise RuntimeError(f"{case}: frame alignment mismatch between base and verified annotations")
    index_hands(base_ann, f"{case} base")
    index_hands(verified_ann, f"{case} verified")


def is_dominant_artifact_dict(node: dict[str, Any]) -> bool:
    for key in ["method", "type", "status", "pose_source", "frame_local_validation_phase"]:
        if DOMINANT_TOKEN in str(node.get(key) or ""):
            return True
    return bool(node.get("dominant_visible_part_surface_only") is True)


def scrub_dominant_visible_part_artifacts(node: Any) -> tuple[Any, int]:
    """Remove/demote the falsified dominant-visible-part mechanism recursively.

    The rejected mechanism may appear as explicit part rows, contact evidence,
    factor-graph support paths, or nested diagnostic structures. Final H' work
    must not consume any of it. This scrub preserves unrelated geometry/contact
    state while deleting fields and list entries that carry the invalid support.
    """
    removed = 0
    if isinstance(node, dict):
        if is_dominant_artifact_dict(node):
            return None, 1
        out: dict[str, Any] = {}
        removed_from_this_dict = False
        for key, value in node.items():
            if DOMINANT_TOKEN in str(key):
                removed += 1
                removed_from_this_dict = True
                continue
            new_value, child_removed = scrub_dominant_visible_part_artifacts(value)
            removed += child_removed
            if child_removed:
                removed_from_this_dict = True
            if new_value is not None:
                out[key] = new_value
        if removed_from_this_dict and out.get("physical_contact_mode") == "active_physical_contact":
            paths = out.get("physical_contact_mode_support_paths")
            if not isinstance(paths, list) or not paths:
                out["physical_contact_mode"] = "articulated_part_contact_unresolved"
                out["physical_contact_mode_renderable"] = True
                out["invalid_part_surface_support_removed"] = True
                out["contact_state_affects_object_or_part_pose"] = False
        return out, removed
    if isinstance(node, list):
        out_list: list[Any] = []
        for value in node:
            if isinstance(value, str) and DOMINANT_TOKEN in value:
                removed += 1
                continue
            new_value, child_removed = scrub_dominant_visible_part_artifacts(value)
            removed += child_removed
            if new_value is not None:
                out_list.append(new_value)
        return out_list, removed
    if isinstance(node, str) and DOMINANT_TOKEN in node:
        return None, 1
    return node, 0


def recursive_dominant_occurrences(node: Any) -> int:
    if isinstance(node, dict):
        total = 0
        for key, value in node.items():
            if DOMINANT_TOKEN in str(key):
                total += 1
            total += recursive_dominant_occurrences(value)
        return total
    if isinstance(node, list):
        return sum(recursive_dominant_occurrences(v) for v in node)
    if isinstance(node, str):
        return int(DOMINANT_TOKEN in node)
    return 0


def update_key_sets(ann: dict[str, Any], label: str) -> dict[str, set[tuple[int, str]]]:
    out = {"corrected": set(), "uncertainty": set(), "validated_no_change": set(), "other_update": set()}
    for key, hand in index_hands(ann, label).items():
        update = compact_update(hand)
        if not isinstance(update, dict):
            continue
        state = hprime_state(update)
        if is_coordinate_hprime(hand, update):
            out["corrected"].add(key)
        elif state == "unchanged_with_compact_rigid_object_overlap_uncertainty":
            out["uncertainty"].add(key)
        elif state == "validated_no_compact_rigid_object_coordinate_change":
            out["validated_no_change"].add(key)
        else:
            out["other_update"].add(key)
    return out


def transplant_hand_state(case: str, key: tuple[int, str], base_hand: dict[str, Any], verified_hand: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Copy only the verified compact-rigid hand-state consequence.

    Corrected rows carry coordinate-bearing metric MANO fields from the verified
    H' annotation. Uncertainty/no-change rows keep the refreshed base MANO
    coordinates and receive only the compact-rigid constraint/uncertainty state.
    """
    update = compact_update(verified_hand)
    if not isinstance(update, dict):
        return base_hand, "no_verified_update"

    out = copy.deepcopy(base_hand)
    copied_update = copy.deepcopy(update)
    out["compact_rigid_object_mano_constraint_update"] = copied_update
    verified_metric = as_dict(verified_hand.get("metric_mano_state"))
    base_metric = as_dict(out.get("metric_mano_state"))
    if not isinstance(base_metric, dict):
        base_metric = {}
        out["metric_mano_state"] = base_metric
    base_metric["compact_rigid_object_constraint_update"] = copy.deepcopy(copied_update)

    state = hprime_state(copied_update)
    if is_coordinate_hprime(verified_hand, copied_update):
        validate_corrected_verified_metric(verified_metric, case=case, key=key)
        for field in COPY_METRIC_FIELDS:
            base_metric[field] = copy.deepcopy(verified_metric[field])
        base_metric["compact_rigid_object_corrected_h_prime"] = True
        base_metric["compact_rigid_object_hprime_transplant_source"] = "verified_compact_rigid_post_signed_remeasurement"
        base_metric["compact_rigid_object_hprime_transplant_scope"] = "coordinate-bearing metric MANO fields copied only for post-verified H-prime rows"
        base_candidate = as_dict(out.get("mano_candidate"))
        verified_candidate = as_dict(verified_hand.get("mano_candidate"))
        verified_candidate_params = verified_candidate.get("mano_params")
        if isinstance(verified_candidate_params, dict):
            base_candidate["mano_params"] = copy.deepcopy(verified_candidate_params)
        base_candidate["compact_rigid_object_corrected_h_prime_available_in_metric_state"] = True
        out["mano_candidate"] = base_candidate
        out["hand_geometry_source"] = "HaWoR_metric_MANO_plus_verified_compact_rigid_Hprime_translation"
        out["uncertainty"] = verified_hand.get("uncertainty", out.get("uncertainty"))
        return out, "coordinate_hprime_transplanted"

    base_metric.pop("compact_rigid_object_corrected_h_prime", None)
    base_metric.pop("compact_rigid_object_hprime_transplant_source", None)
    base_metric.pop("compact_rigid_object_hprime_transplant_scope", None)
    if state == "unchanged_with_compact_rigid_object_overlap_uncertainty":
        out["uncertainty"] = verified_hand.get("uncertainty", out.get("uncertainty"))
        return out, "uncertainty_attached"
    if state == "validated_no_compact_rigid_object_coordinate_change":
        return out, "validated_no_change_attached"
    return out, "noncoordinate_update_attached"


def merge_case(case: str, base_ann: dict[str, Any], verified_ann: dict[str, Any], verified_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_source_alignment(case, base_ann, verified_ann)
    before_dominant_occurrences = recursive_dominant_occurrences(base_ann)
    scrubbed_base, removed_dominant = scrub_dominant_visible_part_artifacts(base_ann)
    if not isinstance(scrubbed_base, dict):
        raise RuntimeError(f"{case}: dominant-visible-part scrub removed the annotation root")
    if recursive_dominant_occurrences(scrubbed_base) != 0:
        raise RuntimeError(f"{case}: dominant-visible-part artifacts remain after recursive scrub")

    verified_hands = index_hands(verified_ann, f"{case} verified")
    base_hands = index_hands(scrubbed_base, f"{case} scrubbed base")
    verified_sets = update_key_sets(verified_ann, f"{case} verified")
    out = copy.deepcopy(scrubbed_base)
    transplant_counts: dict[str, int] = {}
    used_keys: set[tuple[int, str]] = set()
    for frame_raw in as_list(out.get("frames")):
        if not isinstance(frame_raw, dict):
            continue
        frame = frame_raw
        hands = as_list(frame.get("hands"))
        for idx, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            key = hand_key(frame, hand)
            vhand = verified_hands.get(key)
            if not isinstance(vhand, dict) or compact_update(vhand) is None:
                continue
            new_hand, action = transplant_hand_state(case, key, hand, vhand)
            hands[idx] = new_hand
            transplant_counts[action] = transplant_counts.get(action, 0) + 1
            used_keys.add(key)
    update_keys = set().union(*verified_sets.values())
    missing_base = sorted(key for key in update_keys if key not in base_hands)
    missing_output = sorted(key for key in update_keys if key not in used_keys)
    if missing_base or missing_output:
        raise RuntimeError(f"{case}: verified compact-rigid update keys missing in base/output: base={missing_base[:10]} output={missing_output[:10]}")

    final_sets = update_key_sets(out, f"{case} merged")
    for label in ["corrected", "uncertainty", "validated_no_change", "other_update"]:
        if final_sets[label] != verified_sets[label]:
            raise RuntimeError(f"{case}: final {label} key set differs from verified source")
    after_dominant_occurrences = recursive_dominant_occurrences(out)
    if after_dominant_occurrences != 0:
        raise RuntimeError(f"{case}: merged annotations contain dominant-visible-part artifacts")

    out.setdefault("sources", {})["verified_compact_rigid_hprime_hand_state"] = str(verified_path)
    out["method"] = "run_v18_full_pipeline_plus_verified_compact_rigid_hprime_hand_state"
    out["compact_rigid_hprime_finalization"] = {
        "method": "build_v18_verified_hprime_final_annotations",
        "hand_state_semantics": "base final annotation with verified compact-rigid H-prime hand-state consequences transplanted per frame/hand; object/contact backing state is not copied from verified source annotations",
        "coordinate_transplant_rule": "copy coordinate-bearing metric MANO fields only for post-verified H-prime rows; uncertainty/no-change rows keep refreshed base MANO coordinates",
        "rejected_part_surface_shortcut_removed_recursively": True,
    }
    summary = {
        "case": case,
        "frame_count": len(out.get("frames", [])),
        "verified_update_key_counts": {k: len(v) for k, v in sorted(verified_sets.items())},
        "final_update_key_counts": {k: len(v) for k, v in sorted(final_sets.items())},
        "transplant_counts": dict(sorted(transplant_counts.items())),
        "dominant_visible_part_occurrences_before_scrub": before_dominant_occurrences,
        "dominant_visible_part_occurrences_removed": removed_dominant,
        "dominant_visible_part_occurrences_after_merge": after_dominant_occurrences,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True, help="Root containing sanitized base case annotations_v18_full.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verified-annotation", action="append", type=parse_case_path, default=[])
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()

    verified_paths = dict(DEFAULT_VERIFIED_ANNOTATIONS)
    for case, path in args.verified_annotation:
        verified_paths[case] = path

    cases_summary: dict[str, Any] = {}
    for case in args.cases:
        base_path = args.base_root / case / "annotations_v18_full.json"
        verified_path = verified_paths.get(case)
        if verified_path is None:
            raise RuntimeError(f"{case}: no verified compact-rigid annotation source configured")
        base_ann = load_json(base_path)
        verified_ann = load_json(verified_path)
        merged, summary = merge_case(case, base_ann, verified_ann, verified_path)
        summary["base_annotations"] = str(base_path)
        summary["verified_annotations"] = str(verified_path)
        output_path = args.output_root / case / "annotations_v18_full.json"
        write_json(output_path, merged)
        summary["output_annotations"] = str(output_path)
        cases_summary[case] = summary

    manifest = {
        "method": "build_v18_verified_hprime_final_annotations",
        "status": "ok",
        "base_root": str(args.base_root),
        "output_root": str(args.output_root),
        "cases": cases_summary,
        "claim_scope": "Final annotations use the latest base object/contact state after recursive removal of the falsified dominant-visible-part mechanism and only the verified compact-rigid H-prime hand-state consequences from the accepted MANO constraint branches.",
    }
    write_json(args.output_root / "v18_verified_hprime_final_annotations_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
