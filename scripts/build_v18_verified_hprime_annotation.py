#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def index_hands(ann: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in ann.get('frames', []) if isinstance(ann.get('frames'), list) else []:
        try:
            frame_idx = int(frame.get('frame_idx'))
        except Exception:
            continue
        for hand in frame.get('hands', []) if isinstance(frame.get('hands'), list) else []:
            if isinstance(hand, dict):
                out[(frame_idx, str(hand.get('hand_side')))] = hand
    return out


def index_rows(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get('constraint_rows', []) if isinstance(report.get('constraint_rows'), list) else []:
        try:
            out[(int(row.get('frame_idx')), str(row.get('hand_side')))] = row
        except Exception:
            continue
    return out


def corrected_flag(hand: dict[str, Any]) -> bool:
    update_raw = hand.get('compact_rigid_object_mano_constraint_update')
    update = update_raw if isinstance(update_raw, dict) else {}
    metric_raw = hand.get('metric_mano_state')
    metric = metric_raw if isinstance(metric_raw, dict) else {}
    return bool(update.get('coordinate_update_applied') is True or metric.get('compact_rigid_object_corrected_h_prime') is True)


def full_signed_domain(row: dict[str, Any]) -> bool:
    signed_summary = row.get('signed_distance_m') if isinstance(row.get('signed_distance_m'), dict) else {}
    signed_count = signed_summary.get('count') if isinstance(signed_summary, dict) else None
    hand_vertex_count = row.get('hand_vertex_count')
    if isinstance(signed_count, (int, float)) and isinstance(hand_vertex_count, (int, float)):
        return int(signed_count) == int(hand_vertex_count)
    return False


def wrist_by_key(ann: dict[str, Any]) -> dict[tuple[int, str], np.ndarray]:
    out: dict[tuple[int, str], np.ndarray] = {}
    for frame in ann.get('frames', []) if isinstance(ann.get('frames'), list) else []:
        if not isinstance(frame, dict):
            continue
        frame_idx_raw = frame.get('frame_idx')
        if not isinstance(frame_idx_raw, (int, float, str)):
            continue
        try:
            frame_idx = int(frame_idx_raw)
        except Exception:
            continue
        for hand in frame.get('hands', []) if isinstance(frame.get('hands'), list) else []:
            if not isinstance(hand, dict):
                continue
            metric_raw = hand.get('metric_mano_state')
            metric = metric_raw if isinstance(metric_raw, dict) else {}
            wrist = np.asarray(metric.get('wrist_current_v18_world_m') or [], dtype=float)
            if wrist.shape == (3,) and np.isfinite(wrist).all():
                out[(frame_idx, str(hand.get('hand_side')))] = wrist
    return out


def temporal_guard_for_candidate(
    key: tuple[int, str],
    row: dict[str, Any],
    candidate_hand: dict[str, Any],
    original_wrist: dict[tuple[int, str], np.ndarray],
    candidate_wrist: dict[tuple[int, str], np.ndarray],
) -> tuple[bool, dict[str, Any]]:
    visible_raw = row.get('candidate_visible_2d_consistency')
    visible = visible_raw if isinstance(visible_raw, dict) else {}
    guard = {
        'method': 'local_wrist_temporal_discontinuity_guard',
        'trigger': 'no_same_frame_detector_box_constraint',
        'evaluated': False,
        'accepted': True,
        'visible_2d_state': visible.get('state'),
    }
    if visible.get('state') != 'no_same_frame_detector_box_constraint':
        guard['reason'] = 'same-frame 2D evidence was available or another visible-2D predicate handled compatibility'
        return True, guard
    frame_idx, side = key
    required = [(frame_idx - 1, side), (frame_idx, side), (frame_idx + 1, side)]
    if not all(k in original_wrist and k in candidate_wrist for k in required):
        guard['reason'] = 'missing adjacent wrist states; temporal guard did not reject'
        return True, guard
    orig_prev = float(np.linalg.norm(original_wrist[(frame_idx, side)] - original_wrist[(frame_idx - 1, side)]))
    orig_next = float(np.linalg.norm(original_wrist[(frame_idx + 1, side)] - original_wrist[(frame_idx, side)]))
    cand_prev = float(np.linalg.norm(candidate_wrist[(frame_idx, side)] - candidate_wrist[(frame_idx - 1, side)]))
    cand_next = float(np.linalg.norm(candidate_wrist[(frame_idx + 1, side)] - candidate_wrist[(frame_idx, side)]))
    orig_max = max(orig_prev, orig_next)
    cand_max = max(cand_prev, cand_next)
    update_raw = candidate_hand.get('compact_rigid_object_mano_constraint_update')
    update = update_raw if isinstance(update_raw, dict) else {}
    trans = update.get('candidate_translation_norm_m')
    if not isinstance(trans, (int, float)):
        trans = row.get('candidate_translation_norm_m')
    trans_norm = float(trans) if isinstance(trans, (int, float)) else 0.0
    rejected = cand_max > 2.0 * max(orig_max, 1e-9) and trans_norm > orig_max
    guard.update({
        'evaluated': True,
        'accepted': not rejected,
        'original_adjacent_wrist_motion_m': {'prev_to_current': orig_prev, 'current_to_next': orig_next, 'max': orig_max},
        'candidate_adjacent_wrist_motion_m': {'prev_to_current': cand_prev, 'current_to_next': cand_next, 'max': cand_max},
        'candidate_translation_norm_m': trans_norm,
        'rejection_predicate': 'candidate_max_adjacent_motion > 2 * original_max_adjacent_motion and candidate_translation_norm > original_max_adjacent_motion',
        'reason': 'candidate would create an unsupported one-frame wrist discontinuity' if rejected else 'candidate does not worsen local wrist temporal continuity beyond the data-adaptive guard',
    })
    return not rejected, guard


def object_constraint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'object_id': row.get('object_id'),
        'surface_mesh_path': row.get('surface_mesh_path') or row.get('mesh_path'),
        'sign_mesh_path': row.get('sign_mesh_path'),
        'sign_mesh_source_report': row.get('sign_mesh_source_report'),
        'completed_surface_mesh_watertight': row.get('completed_surface_mesh_watertight', row.get('completed_mesh_watertight')),
        'sign_mesh_watertight': row.get('sign_mesh_watertight'),
        'hand_vertex_count': row.get('hand_vertex_count'),
        'signed_distance_query_scope': row.get('signed_distance_query_scope'),
        'near_surface_gate_applied_to_signed_distance': row.get('near_surface_gate_applied_to_signed_distance'),
        'sign_aabb_gate_applied_to_signed_distance': row.get('sign_aabb_gate_applied_to_signed_distance'),
        'observed_band_m': row.get('observed_band_m'),
        'near_surface_vertex_count': row.get('near_surface_vertex_count'),
        'near_surface_vertex_fraction': row.get('near_surface_vertex_fraction'),
        'surface_aabb_candidate_vertex_count': row.get('surface_aabb_candidate_vertex_count', row.get('aabb_candidate_vertex_count')),
        'surface_aabb_candidate_vertex_fraction': row.get('surface_aabb_candidate_vertex_fraction', row.get('aabb_candidate_vertex_fraction')),
        'sign_aabb_candidate_vertex_count': row.get('sign_aabb_candidate_vertex_count'),
        'sign_aabb_candidate_vertex_fraction': row.get('sign_aabb_candidate_vertex_fraction'),
        'outside_sign_aabb_vertex_count': row.get('outside_sign_aabb_vertex_count'),
        'outside_sign_aabb_vertex_fraction': row.get('outside_sign_aabb_vertex_fraction'),
        'signed_query_candidate_vertex_count': row.get('signed_query_candidate_vertex_count'),
        'signed_query_candidate_vertex_fraction': row.get('signed_query_candidate_vertex_fraction'),
        'penetrating_vertex_count': row.get('penetrating_vertex_count'),
        'nearest_surface_unsigned_m': row.get('nearest_surface_unsigned_m'),
        'signed_distance_m': row.get('signed_distance_m'),
        'application_state': row.get('candidate_application_state'),
        'visible_2d_consistency': row.get('candidate_visible_2d_consistency'),
        'translation_solver': row.get('candidate_translation_solver'),
        'reason': row.get('reason'),
    }


def uncertainty_update(row: dict[str, Any]) -> dict[str, Any]:
    state = str(row.get('candidate_application_state') or '')
    if state == 'no_penetration_no_coordinate_change_needed' and full_signed_domain(row):
        h_state = 'validated_no_compact_rigid_object_coordinate_change'
        uncertainty: list[str] = []
    else:
        h_state = 'unchanged_with_compact_rigid_object_overlap_uncertainty'
        if state == 'no_penetration_no_coordinate_change_needed' and not full_signed_domain(row):
            uncertainty = [
                'post-correction signed nonpenetration was reported only on a subset of the bridge hand vertices',
                'hand state remains original MANO with object-constraint uncertainty until signed inside/outside evidence covers every bridge hand vertex',
            ]
        else:
            uncertainty = [
                'compact-rigid object evidence was measured but no coordinate-changing H-prime is accepted for this row',
                'coordinate update is withheld unless a bounded correction is post-verified to eliminate full-bridge signed penetration without degrading visible 2D or local temporal evidence',
            ]
    return {
        'method': 'build_v18_verified_hprime_annotation',
        'h_prime_state': h_state,
        'h_prime_equals_input_h': True,
        'coordinate_update_applied': False,
        'candidate_translation_world_m': row.get('candidate_translation_world_m'),
        'candidate_translation_norm_m': row.get('candidate_translation_norm_m'),
        'candidate_joint_reprojection_shift_px': row.get('candidate_joint_reprojection_shift_px'),
        'object_constraint': object_constraint_from_row(row),
        'uncertainty_added': uncertainty,
        'scope': 'Verified H-prime selector: unverified or nonconverged compact-rigid corrections remain uncertainty on the original MANO state.',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-annotations', type=Path, required=True)
    ap.add_argument('--candidate-annotations', type=Path, required=True)
    ap.add_argument('--post-verify-report', type=Path, required=True)
    ap.add_argument('--output-annotations', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()

    original = load_json(args.original_annotations)
    candidate = load_json(args.candidate_annotations)
    report = load_json(args.post_verify_report)
    candidate_hands = index_hands(candidate)
    verify_rows = index_rows(report)
    original_wrist = wrist_by_key(original)
    candidate_wrist = wrist_by_key(candidate)

    accepted_keys: set[tuple[int, str]] = set()
    temporal_rejected_keys: set[tuple[int, str]] = set()
    temporal_guard_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    corrected_keys = {key for key, hand in candidate_hands.items() if corrected_flag(hand)}
    for key in corrected_keys:
        row = verify_rows.get(key)
        if row and row.get('candidate_application_state') == 'no_penetration_no_coordinate_change_needed' and full_signed_domain(row):
            candidate_hand_raw = candidate_hands.get(key)
            candidate_hand = candidate_hand_raw if isinstance(candidate_hand_raw, dict) else {}
            temporal_ok, temporal_guard = temporal_guard_for_candidate(key, row, candidate_hand, original_wrist, candidate_wrist)
            temporal_guard_by_key[key] = temporal_guard
            if temporal_ok:
                accepted_keys.add(key)
            else:
                temporal_rejected_keys.add(key)

    output = copy.deepcopy(original)
    accepted = 0
    uncertainty = 0
    validated_no_change = 0
    corrected_rejected = 0
    for frame in output.get('frames', []) if isinstance(output.get('frames'), list) else []:
        frame_idx = int(frame.get('frame_idx'))
        hands = frame.get('hands', []) if isinstance(frame.get('hands'), list) else []
        for i, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            key = (frame_idx, str(hand.get('hand_side')))
            row = verify_rows.get(key)
            if key in accepted_keys:
                new_hand = copy.deepcopy(candidate_hands[key])
                update_raw = new_hand.get('compact_rigid_object_mano_constraint_update')
                update = update_raw if isinstance(update_raw, dict) else {}
                update['post_hprime_verification'] = {
                    'application_state': row.get('candidate_application_state') if row else None,
                    'nearest_surface_unsigned_m': row.get('nearest_surface_unsigned_m') if row else None,
                    'signed_distance_m': row.get('signed_distance_m') if row else None,
                    'penetrating_vertex_count': row.get('penetrating_vertex_count') if row else None,
                    'hand_vertex_count': row.get('hand_vertex_count') if row else None,
                    'signed_distance_query_scope': row.get('signed_distance_query_scope') if row else None,
                    'signed_query_candidate_vertex_count': row.get('signed_query_candidate_vertex_count') if row else None,
                    'near_surface_gate_applied_to_signed_distance': row.get('near_surface_gate_applied_to_signed_distance') if row else None,
                    'sign_aabb_gate_applied_to_signed_distance': row.get('sign_aabb_gate_applied_to_signed_distance') if row else None,
                    'temporal_guard': temporal_guard_by_key.get(key),
                    'verified_no_additional_coordinate_change': True,
                }
                new_hand['compact_rigid_object_mano_constraint_update'] = update
                if isinstance(new_hand.get('metric_mano_state'), dict):
                    new_hand['metric_mano_state']['compact_rigid_object_constraint_update'] = update
                    new_hand['metric_mano_state']['compact_rigid_object_corrected_h_prime'] = True
                hands[i] = new_hand
                accepted += 1
            elif row is not None:
                row_for_update = row
                if key in temporal_rejected_keys:
                    row_for_update = copy.deepcopy(row)
                    row_for_update['candidate_application_state'] = 'not_applied_temporal_discontinuity_guard'
                    row_for_update['reason'] = 'candidate cleared full signed volume but created an unsupported one-frame wrist discontinuity without same-frame 2D detector support'
                    row_for_update['temporal_guard'] = temporal_guard_by_key.get(key)
                upd = uncertainty_update(row_for_update)
                if key in temporal_rejected_keys:
                    corrected_rejected += 1
                    upd['rejected_candidate_hprime_reason'] = 'candidate correction cleared full signed remeasurement but failed local temporal continuity guard and was reverted to original MANO coordinates'
                elif key in corrected_keys:
                    corrected_rejected += 1
                    upd['rejected_candidate_hprime_reason'] = 'candidate correction existed in iterative annotation but failed post-H-prime verification and was reverted to original MANO coordinates'
                hand['compact_rigid_object_mano_constraint_update'] = upd
                if isinstance(hand.get('metric_mano_state'), dict):
                    hand['metric_mano_state']['compact_rigid_object_constraint_update'] = upd
                if upd['h_prime_state'] == 'validated_no_compact_rigid_object_coordinate_change':
                    validated_no_change += 1
                else:
                    uncertainty += 1

    summary = {
        'method': 'build_v18_verified_hprime_annotation',
        'status': 'ok',
        'original_annotations': str(args.original_annotations),
        'candidate_annotations': str(args.candidate_annotations),
        'post_verify_report': str(args.post_verify_report),
        'output_annotations': str(args.output_annotations),
        'corrected_candidate_rows_seen': len(corrected_keys),
        'accepted_verified_hprime_rows': accepted,
        'corrected_candidate_rows_reverted_to_uncertainty': corrected_rejected,
        'temporal_guard_rejected_rows': len(temporal_rejected_keys),
        'temporal_guard_rejections': [
            {'frame_idx': key[0], 'hand_side': key[1], 'temporal_guard': temporal_guard_by_key.get(key)}
            for key in sorted(temporal_rejected_keys)
        ],
        'uncertainty_rows': uncertainty,
        'validated_no_change_rows': validated_no_change,
        'claim_scope': 'Only candidate H-prime rows that remain no-penetration under full-bridge post-correction signed testing and pass the local temporal-discontinuity guard are kept as coordinate updates; all other measured rows keep original MANO coordinates with uncertainty.',
    }
    write_json(args.output_annotations, output)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
