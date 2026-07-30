#!/usr/bin/env python3
"""Probe trash contact rows using strict HaWoR bridge candidates.

This is a proximity-only diagnostic. It uses HaWoR bridge hand vertices in current
V18 world coordinates and depth-backed visible object surface vertices. Visible
open surfaces cannot prove contact or nonpenetration, so every output row remains
candidate-only and non-accepted.
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

STRICT_TIER = "strict_candidate_recompute_only"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def summarize(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def strict_policy_keys(policy_report: dict[str, Any]) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for row in policy_report.get("policy_rows", []) if isinstance(policy_report.get("policy_rows"), list) else []:
        if isinstance(row, dict) and row.get("policy_tier") == STRICT_TIER:
            keys.add((int(row.get("frame_idx", -1)), str(row.get("side"))))
    return keys


def bridge_vertices_by_key(bridge_npz: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    z = np.load(bridge_npz)
    frames = np.asarray(z["frame_idx"], dtype=np.int32)
    sides = np.asarray(z["side"], dtype=np.int8)
    world = np.asarray(z["vertices_current_v18_world_from_hawor_camera_local_m"], dtype=np.float32)
    camera = np.asarray(z["vertices_hawor_camera_m"], dtype=np.float32)
    transforms = np.asarray(z["T_world_camera_metric_current_v18"], dtype=np.float32)
    side_name = {0: "left", 1: "right"}
    return {(int(f), side_name[int(s)]): {"world": world[i], "camera": camera[i], "T_world_camera": transforms[i]} for i, (f, s) in enumerate(zip(frames, sides)) if int(s) in side_name}


def visible_surface_index(report_path: Path) -> tuple[dict[tuple[int, str], tuple[int, int]], np.ndarray]:
    report = load_json(report_path)
    npz_path = Path(str(report.get("archive_npz")))
    z = np.load(npz_path)
    frames = np.asarray(z["frame_idx"], dtype=np.int32)
    object_ids = np.asarray(z["object_id"]).astype(str)
    offsets = np.asarray(z["vertex_offsets"], dtype=np.int64)
    vertices = np.asarray(z["vertices"], dtype=np.float32)
    idx: dict[tuple[int, str], tuple[int, int]] = {}
    for i, (frame_idx, oid) in enumerate(zip(frames, object_ids)):
        idx[(int(frame_idx), str(oid))] = (int(offsets[i]), int(offsets[i + 1]))
    return idx, vertices


def contact_rows(path: Path) -> list[dict[str, Any]]:
    report = load_json(path)
    return [row for row in report.get("rows", []) if isinstance(row, dict)] if isinstance(report.get("rows"), list) else []


def world_to_camera(vertices_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    T = np.asarray(T_world_camera, dtype=np.float64)
    inv = np.linalg.inv(T)
    verts = np.asarray(vertices_world, dtype=np.float64)
    homog = np.c_[verts, np.ones(len(verts), dtype=np.float64)]
    return (inv @ homog.T).T[:, :3]


def min_distance(hand_vertices: np.ndarray, object_vertices: np.ndarray, chunk: int = 96) -> float:
    best = float("inf")
    hv = np.asarray(hand_vertices, dtype=np.float32)
    ov = np.asarray(object_vertices, dtype=np.float32)
    for start in range(0, hv.shape[0], chunk):
        part = hv[start : start + chunk]
        diff = part[:, None, :] - ov[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        local = float(np.min(d2)) if d2.size else float("inf")
        if local < best:
            best = local
    return float(math.sqrt(best)) if math.isfinite(best) else float("nan")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    case = args.case
    out_dir = args.output_root / "hawor_bridge_state" / case
    policy_path = out_dir / "v18_hawor_bridge_subset_policy_report.json"
    bridge_report_path = out_dir / "v18_hawor_bridge_state_report.json"
    contact_path = args.output_root / case / "contact_acceptance_audit" / "v18_contact_acceptance_audit_report.json"
    surface_report_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    policy = load_json(policy_path)
    bridge_report = load_json(bridge_report_path)
    strict_keys = strict_policy_keys(policy)
    bridge_npz = Path(str(bridge_report.get("bridge_candidate_npz")))
    bridge_vertices = bridge_vertices_by_key(bridge_npz) if bridge_npz.exists() else {}
    surface_idx, surface_vertices = visible_surface_index(surface_report_path)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    distances: list[float] = []
    deltas: list[float] = []
    depth_gaps: list[float] = []
    hawor_depths: list[float] = []
    object_depths: list[float] = []
    by_object: dict[str, list[float]] = defaultdict(list)
    by_original_category: Counter[str] = Counter()
    missing_surface = 0
    missing_hand = 0
    for row in contact_rows(contact_path):
        frame_idx = int(row.get("frame_idx", -1))
        side = str(row.get("hand_side"))
        key = (frame_idx, side)
        if key not in strict_keys:
            continue
        oid = str(row.get("object_id"))
        bridge_entry = bridge_vertices.get(key)
        surf_slice = surface_idx.get((frame_idx, oid))
        if bridge_entry is None:
            missing_hand += 1
            continue
        hverts = bridge_entry["world"]
        hverts_camera = bridge_entry["camera"]
        T_world_camera = bridge_entry["T_world_camera"]
        hawor_depth = float(np.median(hverts_camera[:, 2]))
        object_depth = None
        depth_gap = None
        if surf_slice is None:
            missing_surface += 1
            probe_status = "strict_policy_contact_row_missing_visible_object_surface"
            dist = None
            vertex_count = 0
        else:
            lo, hi = surf_slice
            obj_vertices = surface_vertices[lo:hi]
            vertex_count = int(len(obj_vertices))
            obj_camera = world_to_camera(obj_vertices, T_world_camera) if vertex_count else np.empty((0, 3), dtype=np.float64)
            object_depth = float(np.median(obj_camera[:, 2])) if vertex_count else None
            depth_gap = float(hawor_depth - object_depth) if object_depth is not None and math.isfinite(object_depth) else None
            if object_depth is not None:
                hawor_depths.append(hawor_depth)
                object_depths.append(object_depth)
            if depth_gap is not None:
                depth_gaps.append(depth_gap)
            dist_value = min_distance(hverts, obj_vertices) if vertex_count else float("nan")
            dist = dist_value if math.isfinite(dist_value) else None
            if dist is None:
                probe_status = "strict_policy_contact_row_invalid_distance"
            else:
                distances.append(float(dist))
                by_object[oid].append(float(dist))
                if dist <= 0.01:
                    counts["distance_le_1cm"] += 1
                if dist <= 0.03:
                    counts["distance_le_3cm"] += 1
                if dist <= 0.05:
                    counts["distance_le_5cm"] += 1
                if dist <= 0.10:
                    counts["distance_le_10cm"] += 1
                probe_status = "visible_surface_proximity_probe_not_contact_acceptance"
        source_min = finite_float(row.get("min_hand_surface_to_object_mesh_m"))
        delta = None
        if dist is not None and source_min is not None:
            delta = float(dist - source_min)
            deltas.append(delta)
        category = str(row.get("category"))
        by_original_category[category] += 1
        rows.append({
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": oid,
            "policy_tier": STRICT_TIER,
            "source_contact_category": category,
            "probe_status": probe_status,
            "hawor_hand_to_visible_object_surface_min_m": dist,
            "source_graph_min_hand_surface_to_object_mesh_m": source_min,
            "distance_delta_hawor_minus_source_m": delta,
            "visible_surface_vertex_count": vertex_count,
            "hawor_hand_camera_median_depth_m": hawor_depth,
            "object_visible_surface_camera_median_depth_m": object_depth,
            "camera_depth_gap_hawor_minus_object_m": depth_gap,
            "object_geometry_scope": "depth_backed_visible_surface_only_open_mesh_not_complete_geometry",
            "contact_acceptance_from_probe": False,
            "nonpenetration_acceptance_from_probe": False,
            "state_role": "HaWoR_strict_bridge_visible_surface_proximity_probe_not_contact_or_nonpenetration_acceptance",
        })
    report = {
        "method": "build_v18_hawor_strict_contact_probe",
        "case": case,
        "status": "candidate_strict_contact_probe_not_acceptance",
        "claim_scope": "strict_HaWoR_bridge_contact_proximity_probe_only_no_contact_acceptance_no_nonpenetration_proof",
        "source_policy_report": str(policy_path),
        "source_bridge_report": str(bridge_report_path),
        "source_contact_audit": str(contact_path),
        "source_visible_geometry_report": str(surface_report_path),
        "strict_policy_hand_rows": int(len(strict_keys)),
        "strict_contact_rows_evaluated": int(len(rows)),
        "strict_contact_rows_missing_hand_vertices": int(missing_hand),
        "strict_contact_rows_missing_visible_surface": int(missing_surface),
        "source_contact_rows_by_category": dict(sorted(by_original_category.items())),
        "distance_threshold_counts": dict(sorted(counts.items())),
        "hawor_hand_to_visible_object_surface_min_m": summarize(distances),
        "distance_delta_hawor_minus_source_m": summarize(deltas),
        "hawor_hand_camera_median_depth_m": summarize(hawor_depths),
        "object_visible_surface_camera_median_depth_m": summarize(object_depths),
        "camera_depth_gap_hawor_minus_object_m": summarize(depth_gaps),
        "per_object_distance_m": {oid: summarize(vals) for oid, vals in sorted(by_object.items())},
        "contact_acceptance_from_probe": False,
        "nonpenetration_acceptance_from_probe": False,
        "downstream_physics_recomputed_or_accepted": False,
        "blocking_reasons": [
            "probe_uses_visible_open_object_surfaces_only_not_complete_mesh_or_sdf",
            "task5_hawor_absent_blocks_all_cases_requirement",
            "contact_not_accepted_without_complete_nonpenetration_and_foundation_state",
            "probe_restricted_to_trash_strict_candidate_queue_only",
        ],
        "rows": rows,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(out_dir / "v18_hawor_strict_contact_probe_report.json", report)
    summary = {
        "method": "build_v18_hawor_strict_contact_probe",
        "status": "candidate_strict_contact_probe_not_acceptance",
        "claim_scope": "strict_HaWoR_bridge_contact_proximity_probe_only_no_contact_acceptance_no_nonpenetration_proof",
        "output_root": str(args.output_root),
        "contact_acceptance_from_probe": False,
        "nonpenetration_acceptance_from_probe": False,
        "cases": [report],
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "hawor_bridge_state" / "v18_hawor_strict_contact_probe_summary.json", summary)
    md = args.output_root / "hawor_bridge_state" / "V18_HAWOR_STRICT_CONTACT_PROBE.md"
    md.write_text(
        "# V18 HaWoR strict contact proximity probe\n\n"
        "This is a candidate-only proximity probe over trash strict HaWoR bridge rows. It uses visible open object surfaces and cannot accept contact or nonpenetration.\n\n"
        f"Status: `{summary['status']}`\n"
        f"Strict contact rows evaluated: `{report['strict_contact_rows_evaluated']}`\n"
        f"Distance summary: `{report['hawor_hand_to_visible_object_surface_min_m']}`\n"
        f"Camera depth gap HaWoR minus object: `{report['camera_depth_gap_hawor_minus_object_m']}`\n"
        f"Threshold counts: `{report['distance_threshold_counts']}`\n"
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
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
