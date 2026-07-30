#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    "task5_tomato_960": {
        "frames": 960,
        "corrected": 0,
        "uncertainty": 5,
        "validated_no_change": 1315,
        "overlay_counts": {"compact_rigid_mano_update_uncertainty": 5},
        "world_counts": {"world_compact_rigid_mano_update_uncertainty": 5},
    },
    "trash_1050": {
        "frames": 1050,
        "corrected": 32,
        "uncertainty": 134,
        "validated_no_change": 578,
        "overlay_counts": {"hand_metric_hprime_corrected_skeletons": 32},
        "world_counts": {"world_compact_rigid_mano_update_corrected": 32, "world_compact_rigid_mano_update_uncertainty": 134},
    },
}

DEFAULT_VERIFIED = {
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_temporal_guard_v1/task5_tomato_960/object_obj_tomato/surface806_sign929_full_bridge_all_signed_temporal_guard/iter1_select/annotations_v18_full_with_verified_tomato_full_signed_temporal_guard_hprime.json"),
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_temporal_guard_v1/trash_1050/object_pink_lid_trash_can_second/frame872_full_bridge_all_signed_temporal_guard/iter1_select/annotations_v18_full_with_verified_trash_full_signed_temporal_guard_hprime.json"),
}

DEFAULT_TASK5_REMEASURE = Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/task5_tomato_960/object_obj_tomato/surface806_sign929_full_bridge_all_signed/iter1_remeasure/v18_mano_object_constraint_state_full_bridge.json")
DEFAULT_TRASH_REMEASURE = Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/trash_1050/object_pink_lid_trash_can_second/frame872_full_bridge_all_signed/iter1_remeasure/v18_mano_object_constraint_state_full_bridge.json")
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


def ffprobe_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return int(out)
    except ValueError:
        return None


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


def is_corrected(hand: dict[str, Any], update: dict[str, Any] | None = None) -> bool:
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
        frame_idx = require_int_field(frame, "frame_idx", f"{label} frame")
        seen_sides: set[str] = set()
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


def json_compare(a: Any, b: Any, path: str, errors: list[str], tol: float = 1e-9) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            errors.append(f"{path}: bool mismatch {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa = float(a); fb = float(b)
        if not math.isfinite(fa) or not math.isfinite(fb):
            errors.append(f"{path}: non-finite numeric value")
        elif abs(fa - fb) > tol:
            errors.append(f"{path}: numeric mismatch {fa} != {fb}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            errors.append(f"{path}: dict keys differ")
            return
        for key in sorted(a.keys(), key=str):
            json_compare(a[key], b[key], f"{path}.{key}", errors, tol=tol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(f"{path}: list length differs {len(a)} != {len(b)}")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            json_compare(av, bv, f"{path}[{i}]", errors, tol=tol)
        return
    if a != b:
        errors.append(f"{path}: value mismatch {a!r} != {b!r}")


def state_sets_updates(ann: dict[str, Any], label: str) -> tuple[dict[str, set[tuple[int, str]]], dict[tuple[int, str], dict[str, Any]], list[str]]:
    sets = {"corrected": set(), "uncertainty": set(), "validated_no_change": set(), "other_update": set()}
    updates: dict[tuple[int, str], dict[str, Any]] = {}
    errors: list[str] = []
    for key, hand in index_hands(ann, label).items():
        top_update = hand.get("compact_rigid_object_mano_constraint_update") if isinstance(hand.get("compact_rigid_object_mano_constraint_update"), dict) else None
        metric = as_dict(hand.get("metric_mano_state"))
        metric_update_raw = metric.get("compact_rigid_object_constraint_update")
        metric_update = metric_update_raw if isinstance(metric_update_raw, dict) else None
        update = top_update or metric_update
        if not isinstance(update, dict):
            continue
        if not isinstance(top_update, dict):
            errors.append(f"{label} {key}: compact update missing at hand top level")
        if not isinstance(metric_update, dict):
            errors.append(f"{label} {key}: compact update missing inside metric_mano_state")
        if isinstance(top_update, dict) and isinstance(metric_update, dict):
            cmp_errors: list[str] = []
            json_compare(top_update, metric_update, f"{label} {key} top_vs_metric_update", cmp_errors)
            errors.extend(cmp_errors[:3])
        updates[key] = update
        state = str(update.get("h_prime_state") or "")
        if is_corrected(hand, update):
            sets["corrected"].add(key)
        elif state == "unchanged_with_compact_rigid_object_overlap_uncertainty":
            sets["uncertainty"].add(key)
        elif state == "validated_no_compact_rigid_object_coordinate_change":
            sets["validated_no_change"].add(key)
        else:
            sets["other_update"].add(key)
    return sets, updates, errors


def verify_source_alignment(case: str, final_ann: dict[str, Any], verified_ann: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(final_ann.get("case")) != str(verified_ann.get("case")) or str(final_ann.get("case")) != case:
        errors.append(f"{case}: case identity mismatch against verified source")
    if final_ann.get("raw_video") != verified_ann.get("raw_video"):
        errors.append(f"{case}: raw_video identity mismatch against verified source")
    try:
        if frame_alignment_signature(final_ann) != frame_alignment_signature(verified_ann):
            errors.append(f"{case}: frame alignment mismatch against verified source")
    except Exception as exc:
        errors.append(f"{case}: frame alignment check failed: {exc}")
    return errors


def verify_state_sets_and_updates(case: str, final_ann: dict[str, Any], verified_ann: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[tuple[int, str], dict[str, Any]], set[tuple[int, str]]]:
    errors: list[str] = []
    final_sets, final_updates, final_update_errors = state_sets_updates(final_ann, f"{case} final")
    verified_sets, verified_updates, verified_update_errors = state_sets_updates(verified_ann, f"{case} verified")
    errors.extend(final_update_errors)
    errors.extend(verified_update_errors)
    set_summary: dict[str, Any] = {}
    for name in ["corrected", "uncertainty", "validated_no_change", "other_update"]:
        missing = sorted(verified_sets[name] - final_sets[name])
        extra = sorted(final_sets[name] - verified_sets[name])
        set_summary[name] = {"final_count": len(final_sets[name]), "verified_count": len(verified_sets[name]), "missing": missing, "extra": extra}
        if missing or extra:
            errors.append(f"{case}: {name} key set differs from verified source")
    for key in sorted(set().union(*verified_sets.values())):
        if key not in final_updates or key not in verified_updates:
            errors.append(f"{case}: update key {key} missing from final or verified updates")
            continue
        cmp_errors: list[str] = []
        json_compare(final_updates[key], verified_updates[key], f"{case} {key} compact_update", cmp_errors)
        if cmp_errors:
            errors.append(f"{case}: compact update payload differs at {key}: {cmp_errors[0]}")
    return set_summary, errors, index_hands(final_ann, f"{case} final"), final_sets["corrected"]


def verify_corrected_coordinates(case: str, final_hands: dict[tuple[int, str], dict[str, Any]], corrected_keys: set[tuple[int, str]], verified_ann: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    verified_hands = index_hands(verified_ann, f"{case} verified")
    max_deltas: dict[str, float] = {}
    for key in sorted(corrected_keys):
        fhand = final_hands[key]
        vhand = verified_hands.get(key)
        if not isinstance(vhand, dict):
            errors.append(f"{case}: corrected key {key} absent in verified source")
            continue
        fmetric = as_dict(fhand.get("metric_mano_state"))
        vmetric = as_dict(vhand.get("metric_mano_state"))
        if fhand.get("hand_geometry_source") != "HaWoR_metric_MANO_plus_verified_compact_rigid_Hprime_translation":
            errors.append(f"{case} {key}: corrected hand_geometry_source does not identify verified H-prime")
        if fmetric.get("compact_rigid_object_hprime_transplant_source") != "verified_compact_rigid_post_signed_remeasurement":
            errors.append(f"{case} {key}: corrected metric state missing H-prime transplant provenance")
        for field, shape in POINT_METRIC_FIELDS.items():
            try:
                farr = require_array(fmetric.get(field), shape, f"{case} {key} final {field}")
                varr = require_array(vmetric.get(field), shape, f"{case} {key} verified {field}")
                if farr.shape != varr.shape:
                    errors.append(f"{case} {key}: {field} shape mismatch {farr.shape} != {varr.shape}")
                    continue
                delta = float(np.max(np.abs(farr - varr)))
                max_deltas[field] = max(max_deltas.get(field, 0.0), delta)
                if delta > 1e-9:
                    errors.append(f"{case} {key}: {field} differs from verified H-prime by {delta}")
            except Exception as exc:
                errors.append(str(exc))
        for field, shape in VECTOR_METRIC_FIELDS.items():
            try:
                farr = require_array(fmetric.get(field), shape, f"{case} {key} final {field}")
                varr = require_array(vmetric.get(field), shape, f"{case} {key} verified {field}")
                delta = float(np.max(np.abs(farr - varr)))
                max_deltas[field] = max(max_deltas.get(field, 0.0), delta)
                if delta > 1e-9:
                    errors.append(f"{case} {key}: {field} differs from verified H-prime by {delta}")
            except Exception as exc:
                errors.append(str(exc))
        fparams = fmetric.get("mano_params")
        vparams = vmetric.get("mano_params")
        if not isinstance(fparams, dict) or not isinstance(vparams, dict):
            errors.append(f"{case} {key}: final/verified mano_params missing")
        else:
            param_errors: list[str] = []
            json_compare(fparams, vparams, f"{case} {key} mano_params", param_errors)
            if param_errors:
                errors.append(param_errors[0])
            try:
                require_array(fparams.get("trans_world_m"), (3,), f"{case} {key} final mano_params.trans_world_m")
            except Exception as exc:
                errors.append(str(exc))
    return {"corrected_count": len(corrected_keys), "max_abs_coordinate_delta_vs_verified": max_deltas}, errors


def expected_from_verified(case: str, verified_ann: dict[str, Any]) -> dict[str, Any]:
    if case not in EXPECTED:
        raise RuntimeError(f"{case}: no frame-count expectation configured")
    sets, _, errors = state_sets_updates(verified_ann, f"{case} verified expected")
    if errors:
        raise RuntimeError(f"{case}: verified source update consistency errors: {errors[:3]}")
    corrected = len(sets["corrected"])
    uncertainty = len(sets["uncertainty"])
    expected = {
        "frames": EXPECTED[case]["frames"],
        "corrected": corrected,
        "uncertainty": uncertainty,
        "validated_no_change": len(sets["validated_no_change"]),
        "overlay_counts": {
            "hand_metric_hprime_corrected_skeletons": corrected,
        },
        "world_counts": {
            "world_compact_rigid_mano_update_corrected": corrected,
            "world_compact_rigid_mano_update_uncertainty": uncertainty,
        },
    }
    return expected


def verify_accepted_remeasurement(case: str, corrected_keys: set[tuple[int, str]], report_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not corrected_keys:
        return {
            "remeasure_report": str(report_path) if report_path is not None else None,
            "corrected_count": 0,
            "accepted_remeasure_states": {},
            "bad_rows": [],
            "match": True,
        }, errors
    if report_path is None:
        return {
            "remeasure_report": None,
            "corrected_count": len(corrected_keys),
            "accepted_remeasure_states": {},
            "bad_rows": [{"reason": "missing_case_remeasurement_report"}],
            "match": False,
        }, [f"{case}: corrected H-prime rows require a signed post-correction remeasurement report"]
    report = load_json(report_path)
    rows = {}
    for row in report.get("constraint_rows", []) if isinstance(report.get("constraint_rows"), list) else []:
        try:
            rows[(int(row.get("frame_idx")), str(row.get("hand_side")))] = row
        except Exception:
            continue
    bad = []
    states: dict[str, int] = {}
    for key in sorted(corrected_keys):
        row = rows.get(key)
        if not isinstance(row, dict):
            bad.append({"key": key, "reason": "missing_remeasurement_row"})
            continue
        state = str(row.get("candidate_application_state") or "")
        states[state] = states.get(state, 0) + 1
        penetrating = row.get("penetrating_vertex_count")
        signed_summary = as_dict(row.get("signed_distance_m"))
        signed_count = signed_summary.get("count")
        hand_vertex_count = row.get("hand_vertex_count")
        signed_domain_ok = False
        if isinstance(signed_count, (int, float)) and isinstance(hand_vertex_count, (int, float)):
            signed_domain_ok = int(signed_count) == int(hand_vertex_count)
        if state != "no_penetration_no_coordinate_change_needed" or penetrating not in {0, 0.0} or not signed_domain_ok:
            bad.append({
                "key": key,
                "state": state,
                "penetrating_vertex_count": penetrating,
                "hand_vertex_count": hand_vertex_count,
                "signed_distance_count": signed_count,
                "signed_distance_query_scope": row.get("signed_distance_query_scope"),
                "reason": "accepted row must post-remeasure no penetration with signed distances over every bridge hand vertex",
            })
    if bad:
        errors.append(f"{case}: corrected H-prime rows failed signed remeasurement")
    return {
        "remeasure_report": str(report_path),
        "corrected_count": len(corrected_keys),
        "accepted_remeasure_states": dict(sorted(states.items())),
        "bad_rows": bad,
        "match": not bad,
    }, errors


def verify_case(root: Path, case: str, verified_path: Path, remeasure_reports: dict[str, Path | None]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    ann_path = root / case / "annotations_v18_full.json"
    ann = load_json(ann_path)
    verified_ann = load_json(verified_path)
    expected = expected_from_verified(case, verified_ann)
    errors.extend(verify_source_alignment(case, ann, verified_ann))
    dominant_occurrences = recursive_dominant_occurrences(ann)
    if dominant_occurrences:
        errors.append(f"{case}: dominant-visible-part artifact occurrences remain recursively: {dominant_occurrences}")
    if len(ann.get("frames", [])) != expected["frames"]:
        errors.append(f"{case}: frame count {len(ann.get('frames', []))} != {expected['frames']}")

    set_summary, set_errors, final_hands, corrected_keys = verify_state_sets_and_updates(case, ann, verified_ann)
    errors.extend(set_errors)
    for key in ["corrected", "uncertainty", "validated_no_change"]:
        if set_summary[key]["final_count"] != expected[key]:
            errors.append(f"{case}: {key} {set_summary[key]['final_count']} != {expected[key]}")
    coordinate_summary, coordinate_errors = verify_corrected_coordinates(case, final_hands, corrected_keys, verified_ann)
    errors.extend(coordinate_errors)

    render_summary_path = root / case / "render_from_annotations_summary.json"
    render_summary = load_json(render_summary_path)
    if render_summary.get("frame_count_match") is not True:
        errors.append(f"{case}: render summary frame_count_match is not true")
    direct_counts = {}
    for name, filename in [("overlay", "v18_overlay.mp4"), ("world", "v18_world.mp4"), ("side_by_side", "v18_side_by_side.mp4")]:
        n = ffprobe_frame_count(root / case / filename)
        direct_counts[name] = n
        if n != expected["frames"]:
            errors.append(f"{case}: {filename} has {n} frames, expected {expected['frames']}")
    overlay_counts = render_summary.get("overlay", {}).get("draw_counts", {}) if isinstance(render_summary.get("overlay"), dict) else {}
    world_counts = render_summary.get("world", {}).get("draw_counts", {}) if isinstance(render_summary.get("world"), dict) else {}
    for draw_key, draw_expected in expected["overlay_counts"].items():
        if overlay_counts.get(draw_key, 0) != draw_expected:
            errors.append(f"{case}: overlay draw {draw_key}={overlay_counts.get(draw_key, 0)} expected {draw_expected}")
    for draw_key, draw_expected in expected["world_counts"].items():
        if world_counts.get(draw_key, 0) != draw_expected:
            errors.append(f"{case}: world draw {draw_key}={world_counts.get(draw_key, 0)} expected {draw_expected}")

    summary = {
        "case": case,
        "annotations": str(ann_path),
        "verified_annotations": str(verified_path),
        "frame_count": len(ann.get("frames", [])),
        "dynamic_expected_from_verified_source": expected,
        "dominant_visible_part_recursive_occurrences": dominant_occurrences,
        "state_key_set_summary": set_summary,
        "corrected_coordinate_match": coordinate_summary,
        "render_summary": str(render_summary_path),
        "direct_ffprobe_frame_counts": direct_counts,
        "overlay_compact_counts": {k: overlay_counts.get(k, 0) for k in sorted(expected["overlay_counts"].keys())},
        "world_compact_counts": {k: world_counts.get(k, 0) for k in sorted(expected["world_counts"].keys())},
    }
    remeasure, remeasure_errors = verify_accepted_remeasurement(case, corrected_keys, remeasure_reports.get(case))
    summary["accepted_hprime_remeasurement"] = remeasure
    errors.extend(remeasure_errors)
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--task5-remeasure-report", type=Path, default=DEFAULT_TASK5_REMEASURE)
    parser.add_argument("--trash-remeasure-report", type=Path, default=DEFAULT_TRASH_REMEASURE)
    parser.add_argument("--verified-annotation", action="append", type=parse_case_path, default=[])
    args = parser.parse_args()

    verified_paths = dict(DEFAULT_VERIFIED)
    for case, path in args.verified_annotation:
        verified_paths[case] = path
    remeasure_reports: dict[str, Path | None] = {
        "task5_tomato_960": args.task5_remeasure_report,
        "trash_1050": args.trash_remeasure_report,
    }

    case_summaries: dict[str, Any] = {}
    errors: list[str] = []
    for case, verified_path in verified_paths.items():
        try:
            summary, case_errors = verify_case(args.root, case, verified_path, remeasure_reports)
        except Exception as exc:
            summary = {"case": case, "exception": str(exc)}
            case_errors = [f"{case}: verifier exception: {exc}"]
        case_summaries[case] = summary
        errors.extend(case_errors)

    out = {
        "method": "verify_v18_verified_hprime_final_artifact",
        "status": "ok" if not errors else "failed",
        "root": str(args.root),
        "cases": case_summaries,
        "errors": errors,
        "case_remeasure_reports": {case: (str(path) if path is not None else None) for case, path in remeasure_reports.items()},
        "trash_remeasure_report": str(args.trash_remeasure_report),
        "verified_annotations": {case: str(path) for case, path in verified_paths.items()},
        "claim_scope": "Verifies the final artifact's consumed metric MANO H-prime/uncertainty hand states by exact key-set and coordinate matching, absence of dominant-visible-part support anywhere in consumed state, signed post-correction remeasurement over every bridge hand vertex for every case with accepted corrections, and full-video render frame counts.",
    }
    write_json(args.summary, out)
    print(json.dumps(out, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
