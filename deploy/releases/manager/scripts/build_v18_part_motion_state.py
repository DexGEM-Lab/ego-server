#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_motion_state"
CLAIM = (
    "This artifact reduces part visible-surface centers into bounded relative-motion evidence. It does not "
    "estimate part pose, articulation parameters, hidden geometry, or object pose."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def finite_vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not (isinstance(value, list) and len(value) == 3):
        raise RuntimeError(f"{label} must be a length-3 list")
    out = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(v) for v in out):
        raise RuntimeError(f"{label} must be finite")
    return out


def center_from_row(row: dict[str, Any]) -> tuple[float, float, float]:
    mn = finite_vec3(row.get("bbox_camera_min_m"), "bbox_camera_min_m")
    mx = finite_vec3(row.get("bbox_camera_max_m"), "bbox_camera_max_m")
    return ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": percentile(values, 50.0),
        "p05": percentile(values, 5.0),
        "p95": percentile(values, 95.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_part(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [require_int(row.get("frame_idx"), "frame_idx") for row in rows]
    centers = [center_from_row(row) for row in rows]
    extents = [finite_vec3(row.get("extent_camera_m"), "extent_camera_m") for row in rows]
    displacements = [distance(centers[i - 1], centers[i]) for i in range(1, len(centers))]
    return {
        "surface_frame_count": len(rows),
        "frame_min": min(frames),
        "frame_max": max(frames),
        "center_camera_m_median": [statistics.median([c[i] for c in centers]) for i in range(3)],
        "extent_camera_m_median": [statistics.median([e[i] for e in extents]) for i in range(3)],
        "adjacent_center_displacement_m": stats(displacements),
        "vertex_count_total": sum(require_int(row.get("vertices"), "vertices") for row in rows),
        "face_count_total": sum(require_int(row.get("faces"), "faces") for row in rows),
    }


def pair_state(distance_range_m: float | None, overlap_count: int, args: argparse.Namespace) -> str:
    if overlap_count < int(args.min_pair_overlap_frames):
        return "insufficient_pair_overlap"
    if distance_range_m is None:
        return "insufficient_pair_distance"
    if distance_range_m <= float(args.stable_distance_range_m):
        return "relative_distance_stable_candidate"
    return "relative_distance_variable_or_mask_inconsistent"


def object_motion_state(pair_counts: Counter[str], part_count: int) -> str:
    if part_count == 0:
        return "no_part_surface_motion_evidence"
    if part_count == 1:
        return "single_part_surface_motion_only"
    stable = pair_counts.get("relative_distance_stable_candidate", 0)
    variable = pair_counts.get("relative_distance_variable_or_mask_inconsistent", 0)
    if stable and variable:
        return "mixed_part_motion_evidence_requires_articulation_or_mask_qc"
    if stable and not variable:
        return "part_relative_distances_stable_visible_surface_only"
    if variable:
        return "part_relative_distances_variable_articulation_or_mask_qc"
    return "part_motion_underconstrained"


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    part_surface_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    part_surfaces = require_dict(load_json(part_surface_path), f"{case} part visible surfaces")
    rows = [require_dict(raw, "surface row") for raw in require_list(part_surfaces.get("surface_rows"), "surface_rows")]
    by_object_part: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_object_part[str(row.get("object_id"))][str(row.get("part_track_label"))].append(row)
    object_rows: list[dict[str, Any]] = []
    object_state_counts: Counter[str] = Counter()
    pair_state_counts: Counter[str] = Counter()
    for object_id, parts in sorted(by_object_part.items()):
        part_summaries = {label: summarize_part(sorted(part_rows, key=lambda item: require_int(item.get("frame_idx"), "frame_idx"))) for label, part_rows in sorted(parts.items())}
        centers_by_part = {
            label: {require_int(row.get("frame_idx"), "frame_idx"): center_from_row(row) for row in part_rows}
            for label, part_rows in parts.items()
        }
        pair_rows: list[dict[str, Any]] = []
        pair_counts: Counter[str] = Counter()
        labels = sorted(parts)
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                shared = sorted(set(centers_by_part[a]) & set(centers_by_part[b]))
                distances = [distance(centers_by_part[a][frame], centers_by_part[b][frame]) for frame in shared]
                dist_stats = stats(distances)
                p05 = dist_stats.get("p05")
                p95 = dist_stats.get("p95")
                distance_range = float(p95 - p05) if isinstance(p05, float) and isinstance(p95, float) else None
                state = pair_state(distance_range, len(shared), args)
                pair_counts[state] += 1
                pair_state_counts[state] += 1
                pair_rows.append(
                    {
                        "part_a": a,
                        "part_b": b,
                        "shared_frame_count": len(shared),
                        "frame_min": min(shared) if shared else None,
                        "frame_max": max(shared) if shared else None,
                        "center_distance_m": dist_stats,
                        "p95_minus_p05_distance_m": distance_range,
                        "pair_motion_state": state,
                        "part_pose_ready": False,
                    }
                )
        obj_state = object_motion_state(pair_counts, len(parts))
        object_state_counts[obj_state] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "part_count": len(parts),
                "surface_frame_count": sum(len(part_rows) for part_rows in parts.values()),
                "part_motion_state": obj_state,
                "part_summaries": part_summaries,
                "pair_motion_state_counts": dict(sorted(pair_counts.items())),
                "pair_rows": pair_rows,
                "part_pose_ready": False,
                "articulation_model_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_part_visible_surfaces": str(part_surface_path)},
        "object_count_with_part_surfaces": len(object_rows),
        "part_motion_state_counts": dict(sorted(object_state_counts.items())),
        "pair_motion_state_counts": dict(sorted(pair_state_counts.items())),
        "object_rows": object_rows,
        "part_pose_ready_count": 0,
        "articulation_model_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_motion_state_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    object_state_counts: Counter[str] = Counter()
    pair_state_counts: Counter[str] = Counter()
    for report in reports:
        object_state_counts.update(report["part_motion_state_counts"])
        pair_state_counts.update(report["pair_motion_state_counts"])
    summary = {
        "method": "build_v18_part_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count_with_part_surfaces": sum(require_int(report.get("object_count_with_part_surfaces"), "object count") for report in reports),
        "part_motion_state_counts": dict(sorted(object_state_counts.items())),
        "pair_motion_state_counts": dict(sorted(pair_state_counts.items())),
        "part_pose_ready_count": 0,
        "articulation_model_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_motion_state_report.json"),
                "object_count_with_part_surfaces": report["object_count_with_part_surfaces"],
                "part_motion_state_counts": report["part_motion_state_counts"],
                "pair_motion_state_counts": report["pair_motion_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_motion_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_state"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-pair-overlap-frames", type=int, default=10)
    parser.add_argument("--stable-distance-range-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
