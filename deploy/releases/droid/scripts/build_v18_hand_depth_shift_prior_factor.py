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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--base-solver-state', type=Path, required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--depth-order-margin-m', type=float, default=0.010)
    p.add_argument('--max-camera-z-shift-m', type=float, default=0.045)
    p.add_argument('--weight', type=float, default=2500.0)
    return p.parse_args()


def depth_summary(row: dict[str, Any]) -> dict[str, Any]:
    val = row.get('visible_surface_depth_order_selected_final_delta_hand_minus_surface_m')
    if not isinstance(val, dict):
        val = row.get('visible_lid_depth_order_selected_final_delta_hand_minus_lid_m')
    return val if isinstance(val, dict) else {}


def selected_front_count(row: dict[str, Any]) -> int:
    return int(row.get('visible_surface_depth_order_selected_final_in_front_count') or row.get('visible_lid_depth_order_selected_final_in_front_count') or 0)


def build(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.base_solver_state)
    rows=[]; skipped=Counter()
    for row in as_list(state.get('per_frame_states')) if isinstance(state, dict) else []:
        if not isinstance(row, dict):
            continue
        front = selected_front_count(row)
        summary = depth_summary(row)
        median = summary.get('median')
        if front <= 0 or not isinstance(median, (int, float)) or float(median) >= -float(args.depth_order_margin_m):
            skipped['no_strong_front_conflict'] += 1
            continue
        shift = min(float(args.max_camera_z_shift_m), max(0.0, -float(median) + float(args.depth_order_margin_m)))
        if shift <= 0.0:
            skipped['zero_shift'] += 1
            continue
        rows.append({
            'factor_family': 'hand_depth_shift_prior',
            'target_entity_id': str(args.target_entity_id),
            'frame_idx': int(row['frame_idx']),
            'hand_side': str(row['hand_side']),
            'variable_affected': 'H_t',
            'observation_type': 'visible_surface_depth_order_camera_z_shift_prior',
            'residual_or_quarantine_rule': 'apply camera_z_shift_m as a hand translation prior along the camera optical axis; positive moves hand behind/near visible first surface while temporal smoothness arbitrates the interval',
            'rendered_uncertainty_channel': 'rendered MANO trajectory should move or remain bounded relative to visible surface; this factor does not change object ownership',
            'state': 'active_hand_depth_shift_prior',
            'camera_z_shift_m': float(shift),
            'weight': float(args.weight),
            'selected_final_in_front_count': int(front),
            'source_selected_delta_median_m': float(median),
            'provenance': {
                'base_solver_state': str(args.base_solver_state),
                'visible_surface_track_factor_state': row.get('visible_surface_track_factor_state'),
                'visible_surface_track_mask_path': row.get('visible_surface_track_mask_path'),
                'reason': 'selected MANO vertices remain in front of visible first surface without hard hand-owned ownership',
            },
        })
    return {
        'method': 'v18_hand_depth_shift_prior_factor',
        'case': args.case,
        'target_entity_id': args.target_entity_id,
        'claim_scope': 'Camera-z hand translation priors derived from visible-surface depth-order conflicts. They are occlusion/trajectory hypotheses for H_t, not contact or object ownership.',
        'inputs': {'base_solver_state': str(args.base_solver_state)},
        'parameters': {'depth_order_margin_m': float(args.depth_order_margin_m), 'max_camera_z_shift_m': float(args.max_camera_z_shift_m), 'weight': float(args.weight)},
        'summary': {'factor_row_count': len(rows), 'camera_z_shift_m_max': max([r['camera_z_shift_m'] for r in rows], default=0.0), 'camera_z_shift_m_sum': sum(r['camera_z_shift_m'] for r in rows), 'skipped_counts': dict(skipped)},
        'factor_rows': rows,
    }


def main() -> None:
    args=parse_args(); payload=build(args)
    out=args.output_root/args.case/'v18_hand_depth_shift_prior_factor_report.json'
    write_json(out,payload)
    print(json.dumps({'status':'ok','report':str(out),'factor_rows':len(payload['factor_rows'])}, indent=2))

if __name__=='__main__':
    main()
