#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_zero_observation_interval(raw: Any, *, source: str) -> dict[str, Any]:
    """Normalize a visual-boundary interval into a durable factor-input record."""
    if isinstance(raw, str):
        parts = raw.split(':', 3)
        if len(parts) != 4:
            raise ValueError(f"zero-observation interval must be side:start:end:reason, got {raw!r}")
        side, start_s, end_s, reason = parts
        record: dict[str, Any] = {
            'hand_side': side,
            'start_frame': int(start_s),
            'end_frame': int(end_s),
            'reason': reason,
            'spec': raw,
        }
    elif isinstance(raw, dict):
        record = dict(raw)
        if 'side' in record and 'hand_side' not in record:
            record['hand_side'] = record['side']
        record.setdefault('spec', f"{record.get('hand_side')}:{record.get('start_frame')}:{record.get('end_frame')}:{record.get('reason')}")
    else:
        raise TypeError(f"zero-observation interval must be string or dict, got {type(raw).__name__}")
    side = str(record.get('hand_side') or '').strip()
    if side not in {'left', 'right'}:
        raise ValueError(f"zero-observation interval side must be left/right, got {side!r} from {source}")
    start = int(record.get('start_frame'))
    end = int(record.get('end_frame'))
    if end < start:
        raise ValueError(f"zero-observation interval end before start: {record!r} from {source}")
    reason = str(record.get('reason') or '').strip()
    if not reason:
        raise ValueError(f"zero-observation interval reason must be nonempty: {record!r} from {source}")
    record['hand_side'] = side
    record['start_frame'] = start
    record['end_frame'] = end
    record['reason'] = reason
    review_frames = record.get('raw_rgb_review_frames')
    if review_frames is not None:
        if not isinstance(review_frames, list) or not all(isinstance(p, str) and Path(p).exists() for p in review_frames):
            raise ValueError(f"zero-observation interval raw_rgb_review_frames must be existing paths: {record!r} from {source}")
    record['source'] = source
    record.setdefault('observation_type', 'explicit_visual_boundary_invalidates_visible_mano_observation_interval')
    record.setdefault('residual_or_quarantine_rule', 'set MANO joint/root/pose visible-observation weights to zero for this frame/side; preserve temporal smoothness and eligible visible object first-surface constraints')
    record.setdefault('rendered_uncertainty_channel', 'latent occluded-hand hypothesis; no contact, object ownership, object pose, nonpenetration, or known hidden-hand pose claim')
    record.setdefault('candidate_px_source_field', 'current_reproject_all_in_front_count')
    record.setdefault('candidate_unit', 'mano_vertices_in_front_of_visible_first_surface_at_current_projection')
    return record


def parse_zero_observation_interval(spec: str) -> dict[str, Any]:
    """Parse side:start:end:reason into a durable factor-input record."""
    return normalize_zero_observation_interval(spec, source='cli:--zero-observation-interval')


def load_zero_observation_interval_files(paths: list[Path]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, dict):
            raw_items = payload.get('intervals', payload.get('zero_observation_intervals'))
        else:
            raw_items = payload
        if not isinstance(raw_items, list):
            raise ValueError(f"zero-observation interval file must contain a list or intervals list: {path}")
        for raw in raw_items:
            intervals.append(normalize_zero_observation_interval(raw, source=str(path)))
    return intervals


def load_remeasurement_cache(path: Path, cache: dict[Path, dict[tuple[int, str], dict[str, Any]]]) -> dict[tuple[int, str], dict[str, Any]]:
    if path not in cache:
        payload = load_json(path)
        rows = {}
        for row in as_list(payload.get('rows')) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            rows[(int(row['frame_idx']), str(row['hand_side']))] = row
        cache[path] = rows
    return cache[path]


def transition_metric_evidence(interval: dict[str, Any], frame_idx: int, hand_side: str, cache: dict[Path, dict[tuple[int, str], dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
    raw_path = interval.get('optimized_projection_remeasurement_report') or interval.get('remeasurement_report') or interval.get('metric_evidence_report')
    if raw_path in (None, ''):
        return 0, {}
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"zero-observation interval metric evidence report does not exist: {path}")
    rows = load_remeasurement_cache(path, cache)
    key = (int(frame_idx), str(hand_side))
    if key not in rows:
        raise KeyError(f"zero-observation interval metric evidence report {path} lacks row {key}")
    row = rows[key]
    field = str(interval.get('candidate_px_source_field') or 'current_reproject_all_in_front_count')
    if field not in row:
        raise KeyError(f"zero-observation interval metric evidence row {key} lacks field {field!r} in {path}")
    candidate_px = int(row.get(field) or 0)
    return candidate_px, {
        'optimized_projection_remeasurement_report': str(path),
        'optimized_projection_candidate_px_field': field,
        'optimized_projection_candidate_px': candidate_px,
        'optimized_projection_mask_path': row.get('mask_path'),
        'optimized_projection_visible_surface_state': row.get('visible_surface_state'),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--ownership-factor-report', type=Path, required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--joint-observation-weight-multiplier', type=float, default=0.12)
    p.add_argument('--zero-observation-interval', action='append', default=[], help='Optional durable visual-boundary interval as side:start:end:reason. Emits zero-observation hand_observation_visibility rows from explicit evidence input rather than ownership candidates. Repeatable.')
    p.add_argument('--zero-observation-intervals-json', type=Path, action='append', default=[], help='JSON file containing visual-boundary zero-observation interval records. Each record may include metric evidence/provenance fields such as optimized_projection_remeasurement_report and raw_rgb_review_frames.')
    return p.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    ownership = load_json(args.ownership_factor_report)
    factor_rows: list[dict[str, Any]] = []
    skipped = Counter()
    for row in as_list(ownership.get('ownership_rows')) if isinstance(ownership, dict) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get('target_entity_id')) != str(args.target_entity_id):
            skipped['wrong_target'] += 1
            continue
        counts = row.get('counts') if isinstance(row.get('counts'), dict) else {}
        candidate_px = int(counts.get('candidate_non_object_owned_px') or counts.get('aligned_hand_entity_overlap_px') or 0)
        hard_px = int(counts.get('non_object_owned_px') or 0)
        hard_state = str(row.get('hard_ownership_state') or '')
        prompt_independent = bool(row.get('hard_ownership_prompt_independent'))
        if candidate_px <= 0:
            skipped['no_candidate'] += 1
            continue
        if hard_px > 0 or prompt_independent:
            skipped['independent_hard_ownership_not_occlusion_conflict'] += 1
            continue
        state = 'active_hand_observation_visibility'
        factor_rows.append({
            'factor_family': 'hand_observation_visibility',
            'target_entity_id': str(args.target_entity_id),
            'frame_idx': int(row['frame_idx']),
            'hand_side': str(row['hand_side']),
            'variable_affected': 'H_t',
            'observation_type': 'visible_object_surface_conflicts_with_mano_prompted_hand_candidate_without_independent_visible_hand',
            'residual_or_quarantine_rule': 'multiply MANO joint/pose visible-observation weights by joint_observation_weight_multiplier for this frame/side; keep visible object first-surface constraints active',
            'rendered_uncertainty_channel': 'ownership review frame renders MANO-prompt self-confirmed hand candidate as orange occluded_or_unresolved, not hard hand-owned',
            'state': state,
            'candidate_px': int(candidate_px),
            'hard_non_object_owned_px': int(hard_px),
            'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier),
            'source_hard_ownership_state': hard_state,
            'provenance': {
                'ownership_factor_report': str(args.ownership_factor_report),
                'ownership_review_frame_path': row.get('review_frame_path'),
                'ownership_hand_prompt_source': ((row.get('provenance') or {}).get('hand_prompt_source') if isinstance(row.get('provenance'), dict) else None),
                'reason': 'candidate visible-hand/entity overlap came from MANO-seeded prompt without independent visible-hand confirmation',
            },
        })
    transition_intervals = [parse_zero_observation_interval(spec) for spec in as_list(args.zero_observation_interval)]
    transition_intervals.extend(load_zero_observation_interval_files(list(args.zero_observation_intervals_json or [])))
    rows_by_key = {(int(r['frame_idx']), str(r['hand_side'])): r for r in factor_rows}
    transition_extension_row_count = 0
    transition_existing_row_count = 0
    remeasurement_cache: dict[Path, dict[tuple[int, str], dict[str, Any]]] = {}
    for interval in transition_intervals:
        if interval.get('target_entity_id') not in (None, '', args.target_entity_id):
            raise ValueError(f"zero-observation interval target {interval.get('target_entity_id')!r} does not match {args.target_entity_id!r}")
        for frame_idx in range(int(interval['start_frame']), int(interval['end_frame']) + 1):
            hand_side = str(interval['hand_side'])
            key = (frame_idx, hand_side)
            candidate_px, metric_provenance = transition_metric_evidence(interval, frame_idx, hand_side, remeasurement_cache)
            provenance = {
                'ownership_factor_report': str(args.ownership_factor_report),
                'visual_boundary_interval_spec': str(interval['spec']),
                'visual_boundary_interval_source': str(interval.get('source')),
                'visual_boundary_summary': interval.get('visual_boundary_summary'),
                'raw_rgb_review_frames': interval.get('raw_rgb_review_frames'),
                'reason': str(interval['reason']),
                **metric_provenance,
            }
            if key in rows_by_key:
                skipped['zero_observation_interval_existing_row_updated'] += 1
                transition_existing_row_count += 1
                row = rows_by_key[key]
                row['observation_type'] = str(interval['observation_type'])
                row['residual_or_quarantine_rule'] = str(interval['residual_or_quarantine_rule'])
                row['rendered_uncertainty_channel'] = str(interval['rendered_uncertainty_channel'])
                row['state'] = 'active_hand_observation_visibility'
                row['candidate_px'] = int(candidate_px if metric_provenance else row.get('candidate_px', 0) or 0)
                row['candidate_unit'] = str(interval.get('candidate_unit') or row.get('candidate_unit') or '')
                row['hard_non_object_owned_px'] = 0
                row['joint_observation_weight_multiplier'] = 0.0
                row['source_hard_ownership_state'] = str(interval.get('source_hard_ownership_state') or 'explicit_visual_boundary_interval')
                old_provenance = row.get('provenance') if isinstance(row.get('provenance'), dict) else {}
                row['provenance'] = {
                    'pre_existing_hand_observation_visibility_provenance': old_provenance,
                    **provenance,
                }
                continue
            transition_extension_row_count += 1
            row = {
                'factor_family': 'hand_observation_visibility',
                'target_entity_id': str(args.target_entity_id),
                'frame_idx': int(frame_idx),
                'hand_side': hand_side,
                'variable_affected': 'H_t',
                'observation_type': str(interval['observation_type']),
                'residual_or_quarantine_rule': str(interval['residual_or_quarantine_rule']),
                'rendered_uncertainty_channel': str(interval['rendered_uncertainty_channel']),
                'state': 'active_hand_observation_visibility',
                'candidate_px': int(candidate_px),
                'candidate_unit': str(interval.get('candidate_unit') or ''),
                'hard_non_object_owned_px': 0,
                'joint_observation_weight_multiplier': 0.0,
                'source_hard_ownership_state': str(interval.get('source_hard_ownership_state') or 'explicit_visual_boundary_interval'),
                'provenance': provenance,
            }
            rows_by_key[key] = row
            factor_rows.append(row)
    factor_rows.sort(key=lambda r: (int(r['frame_idx']), str(r['hand_side'])))
    return {
        'method': 'v18_hand_observation_visibility_factor',
        'case': args.case,
        'target_entity_id': args.target_entity_id,
        'claim_scope': 'This factor downweights or zeros MANO observation anchoring for frame/side hand hypotheses that conflict with visible object first surfaces without independent visible-hand ownership evidence, including explicit visual-boundary intervals with metric first-surface conflict evidence. It does not claim contact, object pose, nonpenetration, known hidden-hand pose, or hand-owned object pixels.',
        'inputs': {
            'ownership_factor_report': str(args.ownership_factor_report),
            'zero_observation_intervals_json': [str(p) for p in list(args.zero_observation_intervals_json or [])],
        },
        'parameters': {
            'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier),
            'zero_observation_intervals': transition_intervals,
        },
        'summary': {
            'factor_row_count': int(len(factor_rows)),
            'transition_interval_frame_count': int(sum(int(i['end_frame']) - int(i['start_frame']) + 1 for i in transition_intervals)),
            'transition_extension_row_count': int(transition_extension_row_count),
            'transition_existing_row_count': int(transition_existing_row_count),
            'zero_observation_row_count': int(sum(float(r.get('joint_observation_weight_multiplier', 1.0)) == 0.0 for r in factor_rows)),
            'candidate_px_sum': int(sum(int(r['candidate_px']) for r in factor_rows)),
            'candidate_px_sum_note': 'legacy diagnostic only; explicit visual-boundary rows may use candidate_unit other than pixels, so unit-specific sums below are authoritative',
            'ownership_candidate_px_sum': int(sum(int(r.get('candidate_px') or 0) for r in factor_rows if not (isinstance(r.get('provenance'), dict) and r['provenance'].get('optimized_projection_remeasurement_report')))),
            'metric_transition_candidate_count_sum': int(sum(int(r.get('candidate_px') or 0) for r in factor_rows if isinstance(r.get('provenance'), dict) and r['provenance'].get('optimized_projection_remeasurement_report'))),
            'state_counts': dict(Counter(str(r.get('state')) for r in factor_rows)),
            'skipped_counts': dict(skipped),
        },
        'factor_rows': factor_rows,
    }


def main() -> None:
    args = parse_args()
    payload = build(args)
    out = args.output_root / args.case / 'v18_hand_observation_visibility_factor_report.json'
    write_json(out, payload)
    print(json.dumps({'status': 'ok', 'report': str(out), 'factor_rows': len(payload['factor_rows'])}, indent=2))


if __name__ == '__main__':
    main()
