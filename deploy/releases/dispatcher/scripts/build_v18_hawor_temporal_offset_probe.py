#!/usr/bin/env python3
"""Probe whether a fixed temporal offset explains trash HaWoR contact mismatch.

This is a candidate-only mechanism diagnostic. It compares strict contact rows at
frame f to same-side HaWoR bridge hand vertices from f + offset, while keeping the
object visible surface and camera frame at f. A strong, consistent offset would
support a frame-index alignment mechanism. The output never accepts contact,
nonpenetration, or HaWoR foundation state.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from build_v18_hawor_strict_contact_probe import bridge_vertices_by_key, load_json, summarize, visible_surface_index, world_to_camera, write_json


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def strict_probe_rows(path: Path) -> list[dict[str, Any]]:
    report = load_json(path)
    rows = report.get("rows") if isinstance(report.get("rows"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def min_distance_tree(hand_vertices_world: np.ndarray, tree: cKDTree) -> float:
    distances, _idx = tree.query(np.asarray(hand_vertices_world, dtype=np.float64), k=1, workers=-1)
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    return float(np.min(distances)) if distances.size else float("nan")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    case = args.case
    out_dir = args.output_root / "hawor_bridge_state" / case
    strict_probe_path = out_dir / "v18_hawor_strict_contact_probe_report.json"
    bridge_report_path = out_dir / "v18_hawor_bridge_state_report.json"
    surface_report_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    strict_report = load_json(strict_probe_path)
    bridge_report = load_json(bridge_report_path)
    bridge_npz = Path(str(bridge_report.get("bridge_candidate_npz")))
    bridge_vertices = bridge_vertices_by_key(bridge_npz) if bridge_npz.exists() else {}
    surface_idx, surface_vertices = visible_surface_index(surface_report_path)

    offsets = list(range(-int(args.max_offset), int(args.max_offset) + 1))
    per_offset_distances: dict[int, list[float]] = defaultdict(list)
    per_offset_depth_gaps: dict[int, list[float]] = defaultdict(list)
    per_offset_abs_depth_gaps: dict[int, list[float]] = defaultdict(list)
    per_offset_available: Counter[int] = Counter()
    best_distance_counts: Counter[int] = Counter()
    best_abs_depth_counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    baseline_distances: list[float] = []
    best_distances: list[float] = []
    baseline_abs_depth_gaps: list[float] = []
    best_abs_depth_gaps: list[float] = []
    missing_surface = 0
    missing_baseline_hand = 0

    tree_cache: dict[tuple[int, str], tuple[cKDTree, np.ndarray, float]] = {}
    for source_row in strict_probe_rows(strict_probe_path):
        frame_idx = int(source_row.get("frame_idx", -1))
        side = str(source_row.get("hand_side"))
        object_id = str(source_row.get("object_id"))
        baseline_key = (frame_idx, side)
        baseline_entry = bridge_vertices.get(baseline_key)
        if baseline_entry is None:
            missing_baseline_hand += 1
            continue
        surface_key = (frame_idx, object_id)
        if surface_key not in tree_cache:
            surf_slice = surface_idx.get(surface_key)
            if surf_slice is None:
                tree_cache[surface_key] = (None, np.empty((0, 3), dtype=np.float32), float("nan"))  # type: ignore[assignment]
            else:
                lo, hi = surf_slice
                obj_vertices = np.asarray(surface_vertices[lo:hi], dtype=np.float64)
                object_camera = world_to_camera(obj_vertices, baseline_entry["T_world_camera"]) if len(obj_vertices) else np.empty((0, 3), dtype=np.float64)
                object_depth = float(np.median(object_camera[:, 2])) if len(object_camera) else float("nan")
                tree_cache[surface_key] = (cKDTree(obj_vertices), obj_vertices, object_depth)
        tree, obj_vertices, object_depth = tree_cache[surface_key]
        if tree is None or len(obj_vertices) == 0 or not math.isfinite(object_depth):
            missing_surface += 1
            continue

        offset_rows: list[dict[str, Any]] = []
        for offset in offsets:
            shifted_key = (frame_idx + offset, side)
            shifted_entry = bridge_vertices.get(shifted_key)
            if shifted_entry is None:
                offset_rows.append({"offset": offset, "available": False})
                continue
            hverts_world = shifted_entry["world"]
            dist = min_distance_tree(hverts_world, tree)
            hverts_in_frame_camera = world_to_camera(hverts_world, baseline_entry["T_world_camera"])
            hand_depth = float(np.median(hverts_in_frame_camera[:, 2])) if len(hverts_in_frame_camera) else float("nan")
            depth_gap = float(hand_depth - object_depth) if math.isfinite(hand_depth) else float("nan")
            abs_depth_gap = abs(depth_gap) if math.isfinite(depth_gap) else float("nan")
            per_offset_available[offset] += 1
            if math.isfinite(dist):
                per_offset_distances[offset].append(dist)
            if math.isfinite(depth_gap):
                per_offset_depth_gaps[offset].append(depth_gap)
                per_offset_abs_depth_gaps[offset].append(abs_depth_gap)
            offset_rows.append({
                "offset": offset,
                "available": True,
                "hawor_hand_to_visible_object_surface_min_m": dist if math.isfinite(dist) else None,
                "hawor_hand_camera_depth_in_frame_f_m": hand_depth if math.isfinite(hand_depth) else None,
                "object_visible_surface_camera_median_depth_in_frame_f_m": object_depth,
                "camera_depth_gap_hawor_minus_object_m": depth_gap if math.isfinite(depth_gap) else None,
                "abs_camera_depth_gap_m": abs_depth_gap if math.isfinite(abs_depth_gap) else None,
            })
        available_offsets = [r for r in offset_rows if r.get("available")]
        if not available_offsets:
            continue
        best_distance_row = min(available_offsets, key=lambda r: finite_float(r.get("hawor_hand_to_visible_object_surface_min_m")) if finite_float(r.get("hawor_hand_to_visible_object_surface_min_m")) is not None else float("inf"))
        best_abs_depth_row = min(available_offsets, key=lambda r: finite_float(r.get("abs_camera_depth_gap_m")) if finite_float(r.get("abs_camera_depth_gap_m")) is not None else float("inf"))
        best_distance_offset = int(best_distance_row["offset"])
        best_abs_depth_offset = int(best_abs_depth_row["offset"])
        best_distance_counts[best_distance_offset] += 1
        best_abs_depth_counts[best_abs_depth_offset] += 1
        baseline_distance = next((finite_float(r.get("hawor_hand_to_visible_object_surface_min_m")) for r in offset_rows if r.get("offset") == 0), None)
        baseline_abs_depth = next((finite_float(r.get("abs_camera_depth_gap_m")) for r in offset_rows if r.get("offset") == 0), None)
        best_distance = finite_float(best_distance_row.get("hawor_hand_to_visible_object_surface_min_m"))
        best_abs_depth = finite_float(best_abs_depth_row.get("abs_camera_depth_gap_m"))
        if baseline_distance is not None:
            baseline_distances.append(baseline_distance)
        if best_distance is not None:
            best_distances.append(best_distance)
        if baseline_abs_depth is not None:
            baseline_abs_depth_gaps.append(baseline_abs_depth)
        if best_abs_depth is not None:
            best_abs_depth_gaps.append(best_abs_depth)
        rows.append({
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": object_id,
            "source_contact_category": source_row.get("source_contact_category"),
            "offset_rows": offset_rows,
            "best_distance_offset": best_distance_offset,
            "best_abs_depth_gap_offset": best_abs_depth_offset,
            "baseline_offset0_distance_m": baseline_distance,
            "best_distance_m": best_distance,
            "baseline_offset0_abs_depth_gap_m": baseline_abs_depth,
            "best_abs_depth_gap_m": best_abs_depth,
            "contact_acceptance_from_probe": False,
            "nonpenetration_acceptance_from_probe": False,
            "state_role": "HaWoR_temporal_offset_mechanism_probe_not_contact_acceptance",
        })

    offset_summaries = []
    for offset in offsets:
        offset_summaries.append({
            "offset": offset,
            "available_rows": int(per_offset_available[offset]),
            "distance_m": summarize(per_offset_distances[offset]),
            "camera_depth_gap_hawor_minus_object_m": summarize(per_offset_depth_gaps[offset]),
            "abs_camera_depth_gap_m": summarize(per_offset_abs_depth_gaps[offset]),
        })
    baseline_distance_summary = summarize(baseline_distances)
    best_distance_summary = summarize(best_distances)
    baseline_abs_depth_summary = summarize(baseline_abs_depth_gaps)
    best_abs_depth_summary = summarize(best_abs_depth_gaps)
    dominant_best_distance_offset, dominant_best_distance_count = (None, 0)
    if best_distance_counts:
        dominant_best_distance_offset, dominant_best_distance_count = best_distance_counts.most_common(1)[0]
    dominant_best_abs_depth_offset, dominant_best_abs_depth_count = (None, 0)
    if best_abs_depth_counts:
        dominant_best_abs_depth_offset, dominant_best_abs_depth_count = best_abs_depth_counts.most_common(1)[0]
    evaluated_rows = len(rows)
    dominant_distance_fraction = float(dominant_best_distance_count / evaluated_rows) if evaluated_rows else 0.0
    dominant_abs_depth_fraction = float(dominant_best_abs_depth_count / evaluated_rows) if evaluated_rows else 0.0
    baseline_median = baseline_distance_summary.get("median")
    best_median = best_distance_summary.get("median")
    baseline_abs_depth_median = baseline_abs_depth_summary.get("median")
    best_abs_depth_median = best_abs_depth_summary.get("median")
    temporal_offset_support = bool(
        evaluated_rows
        and dominant_best_distance_offset not in (None, 0)
        and dominant_distance_fraction >= 0.6
        and isinstance(best_median, float)
        and best_median <= 0.10
        and isinstance(best_abs_depth_median, float)
        and best_abs_depth_median <= 0.10
    )
    if temporal_offset_support:
        interpretation = "candidate_fixed_temporal_offset_mechanism_supported_but_not_physical_acceptance"
    elif dominant_distance_fraction >= 0.6 and dominant_best_distance_offset not in (None, 0):
        interpretation = "dominant_offset_exists_but_distance_or_depth_gap_remains_too_large_for_temporal_offset_explanation"
    else:
        interpretation = "no_consistent_temporal_offset_explains_strict_contact_mismatch"

    report = {
        "method": "build_v18_hawor_temporal_offset_probe",
        "case": case,
        "status": "candidate_temporal_offset_probe_not_acceptance",
        "claim_scope": "temporal_offset_mechanism_probe_only_no_contact_acceptance_no_foundation_acceptance",
        "max_offset": int(args.max_offset),
        "source_strict_contact_probe": str(strict_probe_path),
        "source_bridge_report": str(bridge_report_path),
        "source_visible_geometry_report": str(surface_report_path),
        "strict_contact_rows_input": int(len(strict_report.get("rows", [])) if isinstance(strict_report.get("rows"), list) else 0),
        "rows_evaluated": int(evaluated_rows),
        "rows_missing_surface": int(missing_surface),
        "rows_missing_baseline_hand": int(missing_baseline_hand),
        "offset_summaries": offset_summaries,
        "best_distance_offset_counts": {str(k): int(v) for k, v in sorted(best_distance_counts.items())},
        "best_abs_depth_gap_offset_counts": {str(k): int(v) for k, v in sorted(best_abs_depth_counts.items())},
        "dominant_best_distance_offset": dominant_best_distance_offset,
        "dominant_best_distance_fraction": dominant_distance_fraction,
        "dominant_best_abs_depth_gap_offset": dominant_best_abs_depth_offset,
        "dominant_best_abs_depth_gap_fraction": dominant_abs_depth_fraction,
        "baseline_offset0_distance_m": baseline_distance_summary,
        "best_any_offset_distance_m": best_distance_summary,
        "baseline_offset0_abs_depth_gap_m": baseline_abs_depth_summary,
        "best_any_offset_abs_depth_gap_m": best_abs_depth_summary,
        "temporal_offset_supports_contact_mismatch_explanation": temporal_offset_support,
        "interpretation": interpretation,
        "contact_acceptance_from_probe": False,
        "nonpenetration_acceptance_from_probe": False,
        "foundation_acceptance_from_probe": False,
        "blocking_reasons": [
            "probe_reuses_candidate_only_trash_HaWoR_bridge_rows",
            "task5_hawor_absent_blocks_all_cases_requirement",
            "visible_surface_open_geometry_cannot_prove_contact_or_nonpenetration",
            "temporal_offset_probe_is_mechanism_diagnostic_not_downstream_recompute",
        ],
        "rows": rows,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(out_dir / "v18_hawor_temporal_offset_probe_report.json", report)
    summary = {
        "method": "build_v18_hawor_temporal_offset_probe",
        "status": report["status"],
        "claim_scope": report["claim_scope"],
        "output_root": str(args.output_root),
        "contact_acceptance_from_probe": False,
        "nonpenetration_acceptance_from_probe": False,
        "foundation_acceptance_from_probe": False,
        "cases": [report],
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "hawor_bridge_state" / "v18_hawor_temporal_offset_probe_summary.json", summary)
    md = args.output_root / "hawor_bridge_state" / "V18_HAWOR_TEMPORAL_OFFSET_PROBE.md"
    md.write_text(
        "# V18 HaWoR temporal offset probe\n\n"
        "This is a candidate-only mechanism probe over trash strict contact rows. It tests whether using HaWoR bridge hands from nearby frames explains the contact distance/depth mismatch. It does not accept contact, nonpenetration, or HaWoR foundation state.\n\n"
        f"Status: `{report['status']}`\n"
        f"Rows evaluated: `{report['rows_evaluated']}`\n"
        f"Dominant best-distance offset: `{report['dominant_best_distance_offset']}` fraction `{report['dominant_best_distance_fraction']}`\n"
        f"Dominant best-abs-depth-gap offset: `{report['dominant_best_abs_depth_gap_offset']}` fraction `{report['dominant_best_abs_depth_gap_fraction']}`\n"
        f"Baseline distance: `{report['baseline_offset0_distance_m']}`\n"
        f"Best any-offset distance: `{report['best_any_offset_distance_m']}`\n"
        f"Baseline abs depth gap: `{report['baseline_offset0_abs_depth_gap_m']}`\n"
        f"Best any-offset abs depth gap: `{report['best_any_offset_abs_depth_gap_m']}`\n"
        f"Temporal offset supports mismatch explanation: `{report['temporal_offset_supports_contact_mismatch_explanation']}`\n"
        f"Interpretation: `{report['interpretation']}`\n"
        f"Contact acceptance from probe: `{report['contact_acceptance_from_probe']}`\n"
        f"Nonpenetration acceptance from probe: `{report['nonpenetration_acceptance_from_probe']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--case", default="trash_1050")
    parser.add_argument("--max-offset", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
