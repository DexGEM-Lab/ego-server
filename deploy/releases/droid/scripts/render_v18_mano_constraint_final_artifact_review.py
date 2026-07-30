#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def hand_update(hand: dict[str, Any]) -> dict[str, Any] | None:
    upd = hand.get('compact_rigid_object_mano_constraint_update') if isinstance(hand.get('compact_rigid_object_mano_constraint_update'), dict) else None
    if upd is not None:
        return upd
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    upd = metric.get('compact_rigid_object_constraint_update') if isinstance(metric.get('compact_rigid_object_constraint_update'), dict) else None
    return upd


def updated_keys(ann: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in ann.get('frames', []) if isinstance(ann.get('frames'), list) else []:
        frame_idx = int(frame.get('frame_idx'))
        for hand in frame.get('hands', []) if isinstance(frame.get('hands'), list) else []:
            if not isinstance(hand, dict):
                continue
            upd = hand_update(hand)
            if upd is not None:
                out[(frame_idx, str(hand.get('hand_side')))] = upd
    return out


def pick_rows(ann: dict[str, Any], constraints: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    rows = constraints.get('constraint_rows', []) if isinstance(constraints.get('constraint_rows'), list) else []
    state_by_key = {(int(r['frame_idx']), str(r['hand_side'])): r for r in rows if isinstance(r, dict) and 'frame_idx' in r and 'hand_side' in r}
    updates = updated_keys(ann)
    selected: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    # Corrected H' rows first: these are the primary deliverable.
    for key, upd in sorted(updates.items()):
        if upd.get('coordinate_update_applied') is True:
            row = state_by_key.get(key)
            if row is not None:
                selected.append((row, upd))
        if len(selected) >= 6:
            break
    # Then rows that explicitly remain uncertain or rejected, so the sheet tests
    # that the artifact is not painting all measured rows as corrected.
    for wanted in ['not_applied_local_escape_amplified', 'candidate_coordinate_correction_visible_2d_compatible', 'uncertainty_sign_mesh_missing_near_surface_support', 'not_applied_escape_solver_failed']:
        for row in rows:
            if row.get('candidate_application_state') == wanted:
                key = (int(row['frame_idx']), str(row['hand_side']))
                selected.append((row, updates.get(key)))
                break
    # Add representative no-change rows with no coordinate update.
    for row in rows:
        key = (int(row['frame_idx']), str(row['hand_side']))
        if row.get('candidate_application_state') == 'no_penetration_no_coordinate_change_needed' and key not in updates:
            selected.append((row, None))
            if len(selected) >= 12:
                break
    return selected[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-root', type=Path, required=True)
    ap.add_argument('--case', default='task5_tomato_960')
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--constraint-report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    ann = load_json(args.annotations)
    constraints = load_json(args.constraint_report)
    selected = pick_rows(ann, constraints)
    panels = []
    for row, upd in selected:
        idx = int(row['frame_idx'])
        overlay_path = args.render_root / args.case / 'overlay_frames' / f'{idx:06d}.jpg'
        world_path = args.render_root / args.case / 'world_frames' / f'{idx:06d}.jpg'
        if not overlay_path.exists() or not world_path.exists():
            continue
        overlay = Image.open(overlay_path).convert('RGB').resize((480, 270), Image.Resampling.BILINEAR)
        world = Image.open(world_path).convert('RGB').resize((480, 270), Image.Resampling.BILINEAR)
        panel = Image.new('RGB', (960, 324), (20, 20, 20))
        panel.paste(overlay, (0, 54)); panel.paste(world, (480, 54))
        d = ImageDraw.Draw(panel)
        upd_state = upd.get('h_prime_state') if isinstance(upd, dict) else 'no_compact_update'
        applied = upd.get('coordinate_update_applied') if isinstance(upd, dict) else False
        txt = (
            f"{args.case} f{idx} {row['hand_side']} verify={row['candidate_application_state']} "
            f"H'={upd_state} applied={applied} near={row.get('near_surface_vertex_count')} "
            f"pen={row.get('penetrating_vertex_count')}"
        )
        d.rectangle((0, 0, 960, 54), fill=(0, 0, 0)); d.text((5, 8), txt[:155], fill=(255, 255, 255))
        panels.append(panel)
    if not panels:
        raise RuntimeError('no review panels rendered')
    sheet = Image.new('RGB', (960, 324 * len(panels)), (20, 20, 20))
    for i, p in enumerate(panels):
        sheet.paste(p, (0, 324 * i))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    summary = {
        'method': 'render_v18_mano_constraint_final_artifact_review',
        'status': 'ok',
        'case': args.case,
        'output': str(args.output),
        'panel_count': len(panels),
        'reviewed_rows': [
            {
                'frame_idx': int(r['frame_idx']),
                'hand_side': r['hand_side'],
                'verify_state': r['candidate_application_state'],
                'h_prime_state': (u or {}).get('h_prime_state'),
                'coordinate_update_applied': bool((u or {}).get('coordinate_update_applied')),
            }
            for r, u in selected[:len(panels)]
        ],
        'claim_scope': 'Review sheet samples final rendered overlay/world frames to verify compact-rigid MANO H-prime corrections and uncertainty states are visible in the consumed artifact.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
