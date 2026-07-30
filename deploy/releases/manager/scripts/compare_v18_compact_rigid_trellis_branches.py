#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def mesh_stats(path: Path) -> dict[str, Any]:
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        meshes = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        m = trimesh.util.concatenate(meshes)
    ext = np.asarray(m.extents, dtype=float)
    return {
        'vertices': int(len(m.vertices)),
        'faces': int(len(m.faces)),
        'watertight': bool(m.is_watertight),
        'euler_number': int(m.euler_number),
        'extents_m': ext.astype(float).tolist(),
        'min_extent_over_max_extent': float(np.min(ext) / max(np.max(ext), 1e-12)),
        'volume_m3_if_watertight': float(m.volume) if bool(m.is_watertight) else None,
    }


def row_for(report_path: Path) -> dict[str, Any]:
    r = load_json(report_path)
    trellis_report = load_json(Path(r['inputs']['trellis_report']))
    all_mesh = Path(r['outputs']['trellis_aligned_all_candidate_labeled_mesh'])
    completed_mesh = Path(r['outputs']['completed_mesh_labeled'])
    label_counts = r['face_label_counts']
    trellis_counts = label_counts['trellis_all_candidate']
    hidden = int(trellis_counts.get('trellis_inferred_hidden_surface', 0))
    overwritten = int(trellis_counts.get('observed_region_overwritten_candidate', trellis_counts.get('free_space_rejected', 0)))
    total = hidden + overwritten
    final = r['metric_alignment']['observed_to_trellis_stats_final']
    all_stats = mesh_stats(all_mesh)
    completed_stats = mesh_stats(completed_mesh)
    volumetricity = all_stats['min_extent_over_max_extent']
    residual = float(final['median_m']) + 0.25 * float(final['p95_m'])
    hidden_fraction = hidden / total if total else 0.0
    observed_band_m = float(r.get('observed_band_m') or 0.0)
    return {
        'report': str(report_path),
        'case': r.get('case'),
        'object_id': r.get('object_id'),
        'trellis_image': trellis_report.get('image'),
        'trellis_raw_extent_model_units': trellis_report.get('extent_model_units'),
        'observed_to_trellis_stats_final': final,
        'trellis_to_observed_stats_final': r['metric_alignment']['trellis_to_observed_stats_final'],
        'trellis_aligned_all_candidate_mesh': str(all_mesh),
        'completed_mesh': str(completed_mesh),
        'trellis_all_candidate_mesh_stats': all_stats,
        'completed_mesh_stats': completed_stats,
        'trellis_hidden_face_count': hidden,
        'trellis_observed_region_overwritten_face_count': overwritten,
        'trellis_hidden_face_fraction': hidden_fraction,
        'completed_face_label_counts': label_counts['completed_mesh'],
        'free_space_rejection_state': r.get('free_space_rejection_state'),
        'observed_band_m': observed_band_m,
        'selection_terms': {
            'residual_median_plus_quarter_p95_m': residual,
            'volumetricity_min_extent_over_max_extent': volumetricity,
            'hidden_face_fraction': hidden_fraction,
            'residual_indifference_band_m': observed_band_m,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = [row_for(p) for p in args.reports]
    min_residual = min(r['selection_terms']['residual_median_plus_quarter_p95_m'] for r in rows)
    residual_band = max(r['selection_terms']['residual_indifference_band_m'] for r in rows)
    admissible_surface = [r for r in rows if r['selection_terms']['residual_median_plus_quarter_p95_m'] <= min_residual + residual_band]
    best_surface_fit = sorted(rows, key=lambda x: x['selection_terms']['residual_median_plus_quarter_p95_m'])[0]
    # MANO nonpenetration needs a sign-supporting hidden volume. A non-watertight
    # prior may fit the visible surface but cannot determine inside/outside for
    # hand correction. Sign-supporting branch discovery is therefore independent
    # of best visible-surface fit; it must be validated downstream rather than
    # hidden by a surface-fit ranking.
    sign_supporting = [r for r in rows if r['trellis_all_candidate_mesh_stats']['watertight']]
    best = sorted(sign_supporting, key=lambda x: (x['selection_terms']['residual_median_plus_quarter_p95_m'], -x['selection_terms']['volumetricity_min_extent_over_max_extent']))[0] if sign_supporting else None
    rows_sorted = sorted(rows, key=lambda x: (not x['trellis_all_candidate_mesh_stats']['watertight'], -x['selection_terms']['volumetricity_min_extent_over_max_extent'], x['selection_terms']['residual_median_plus_quarter_p95_m']))
    result = {
        'method': 'compare_v18_compact_rigid_trellis_branches',
        'status': 'ok',
        'selection_claim': 'Branch selection compares metric residual and volumetricity after alignment. Residual differences below the observed-surface voxel band are treated as physically indistinguishable; within that band, the more volumetric prior is selected for hidden-surface reasoning.',
        'residual_indifference_band_m': residual_band,
        'best_surface_fit_report': best_surface_fit['report'],
        'best_surface_fit_completed_mesh': best_surface_fit['completed_mesh'],
        'best_sign_supporting_report': best['report'] if best else None,
        'best_sign_supporting_completed_mesh': best['completed_mesh'] if best else None,
        'best_sign_supporting_trellis_all_candidate_mesh': best['trellis_aligned_all_candidate_mesh'] if best else None,
        'sign_supporting_branch_count': len(sign_supporting),
        'rows': rows_sorted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
