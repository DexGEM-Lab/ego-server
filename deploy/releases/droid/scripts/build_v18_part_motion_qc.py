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

STATUS = "v18_part_motion_qc"
CLAIM = (
    "This artifact audits V18 part-motion diagnostics for confounds. It separates variable part-pair distances "
    "that are supported by robust part surfaces from variation involving sparse/unstable part tracks. It does not "
    "fit an articulation model or mark part pose ready."
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


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


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


def part_surface_quality(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, list[str], dict[str, Any]]:
    vertices = [finite_float(row.get("vertices"), "vertices") for row in rows]
    faces = [finite_float(row.get("faces"), "faces") for row in rows]
    containment = [finite_float(row.get("part_containment_in_object"), "containment") for row in rows]
    frame_count = len(rows)
    median_vertices = statistics.median(vertices) if vertices else 0.0
    median_faces = statistics.median(faces) if faces else 0.0
    median_containment = statistics.median(containment) if containment else 0.0
    blockers: list[str] = []
    if frame_count < int(args.min_robust_frames):
        blockers.append("short_part_surface_track")
    if median_vertices < float(args.min_robust_median_vertices):
        blockers.append("sparse_part_surface_vertices")
    if median_faces < float(args.min_robust_median_faces):
        blockers.append("sparse_part_surface_faces")
    if median_containment < float(args.min_robust_median_containment):
        blockers.append("low_part_object_containment")
    if blockers:
        quality = "sparse_or_unstable_part_surface_track"
    else:
        quality = "robust_part_surface_track"
    metrics = {
        "surface_frame_count": frame_count,
        "vertices": {"median": median_vertices, "p05": percentile(vertices, 5.0), "p95": percentile(vertices, 95.0)},
        "faces": {"median": median_faces, "p05": percentile(faces, 5.0), "p95": percentile(faces, 95.0)},
        "part_containment_in_object": {
            "median": median_containment,
            "p05": percentile(containment, 5.0),
            "p95": percentile(containment, 95.0),
        },
    }
    return quality, blockers, metrics


def pair_qc_state(pair: dict[str, Any], part_quality: dict[str, str]) -> tuple[str, list[str]]:
    state = str(pair.get("pair_motion_state"))
    part_a = str(pair.get("part_a"))
    part_b = str(pair.get("part_b"))
    qa = part_quality.get(part_a, "unknown_part_quality")
    qb = part_quality.get(part_b, "unknown_part_quality")
    blockers: list[str] = []
    if state == "relative_distance_stable_candidate":
        if qa == "robust_part_surface_track" and qb == "robust_part_surface_track":
            return "stable_pair_supported_by_robust_surfaces", blockers
        blockers.append("stable_pair_involves_sparse_or_unknown_part")
        return "stable_pair_low_confidence_due_part_quality", blockers
    if state == "relative_distance_variable_or_mask_inconsistent":
        if qa == "robust_part_surface_track" and qb == "robust_part_surface_track":
            return "variable_pair_between_robust_surfaces_articulation_hypothesis", blockers
        blockers.append("variable_pair_involves_sparse_or_unstable_part_surface")
        return "variable_pair_confounded_by_part_surface_quality", blockers
    blockers.append("pair_motion_underconstrained")
    return "pair_qc_underconstrained", blockers


def object_qc_state(pair_qc_counts: Counter[str]) -> str:
    robust_variable = pair_qc_counts.get("variable_pair_between_robust_surfaces_articulation_hypothesis", 0)
    confounded_variable = pair_qc_counts.get("variable_pair_confounded_by_part_surface_quality", 0)
    robust_stable = pair_qc_counts.get("stable_pair_supported_by_robust_surfaces", 0)
    if robust_variable:
        return "articulation_hypothesis_supported_by_robust_variable_pairs_not_fitted"
    if confounded_variable and robust_stable:
        return "part_motion_confounded_by_sparse_tracks_with_some_stable_support"
    if confounded_variable:
        return "part_motion_confounded_by_sparse_tracks"
    if robust_stable:
        return "only_stable_part_pairs_no_articulation_evidence"
    return "part_motion_qc_underconstrained"


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    motion_path = args.part_motion_root / case / "v18_part_motion_state_report.json"
    surfaces = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    motion = require_dict(load_json(motion_path), f"{case} part motion")
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(surfaces.get("surface_rows"), "surface_rows")]
    rows_by_object_part: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in surface_rows:
        rows_by_object_part[str(row.get("object_id"))][str(row.get("part_track_label"))].append(row)
    object_rows: list[dict[str, Any]] = []
    object_qc_counts: Counter[str] = Counter()
    pair_qc_counts_all: Counter[str] = Counter()
    part_quality_counts: Counter[str] = Counter()
    for raw_obj in require_list(motion.get("object_rows"), "motion object rows"):
        obj = require_dict(raw_obj, "motion object")
        object_id = str(obj.get("object_id"))
        part_quality: dict[str, str] = {}
        part_rows: list[dict[str, Any]] = []
        for label, rows in sorted(rows_by_object_part.get(object_id, {}).items()):
            quality, blockers, metrics = part_surface_quality(rows, args)
            part_quality[label] = quality
            part_quality_counts[quality] += 1
            part_rows.append({"part_track_label": label, "part_surface_quality": quality, "quality_blockers": blockers, "quality_metrics": metrics})
        pair_rows: list[dict[str, Any]] = []
        pair_qc_counts: Counter[str] = Counter()
        for raw_pair in require_list(obj.get("pair_rows"), "pair rows"):
            pair = require_dict(raw_pair, "pair row")
            qc_state, blockers = pair_qc_state(pair, part_quality)
            pair_qc_counts[qc_state] += 1
            pair_qc_counts_all[qc_state] += 1
            pair_rows.append({**pair, "pair_qc_state": qc_state, "qc_blockers": blockers})
        obj_state = object_qc_state(pair_qc_counts)
        object_qc_counts[obj_state] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "source_part_motion_state": obj.get("part_motion_state"),
                "part_motion_qc_state": obj_state,
                "part_surface_quality_counts": dict(sorted(Counter(part_quality.values()).items())),
                "pair_qc_state_counts": dict(sorted(pair_qc_counts.items())),
                "part_rows": part_rows,
                "pair_rows": pair_rows,
                "articulation_model_ready": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_motion_qc",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_part_visible_surfaces": str(surfaces_path), "v18_part_motion_state": str(motion_path)},
        "object_count": len(object_rows),
        "part_motion_qc_state_counts": dict(sorted(object_qc_counts.items())),
        "pair_qc_state_counts": dict(sorted(pair_qc_counts_all.items())),
        "part_surface_quality_counts": dict(sorted(part_quality_counts.items())),
        "object_rows": object_rows,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_motion_qc_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    object_qc_counts: Counter[str] = Counter()
    pair_qc_counts: Counter[str] = Counter()
    part_quality_counts: Counter[str] = Counter()
    for report in reports:
        object_qc_counts.update(report["part_motion_qc_state_counts"])
        pair_qc_counts.update(report["pair_qc_state_counts"])
        part_quality_counts.update(report["part_surface_quality_counts"])
    summary = {
        "method": "build_v18_part_motion_qc",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "part_motion_qc_state_counts": dict(sorted(object_qc_counts.items())),
        "pair_qc_state_counts": dict(sorted(pair_qc_counts.items())),
        "part_surface_quality_counts": dict(sorted(part_quality_counts.items())),
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_motion_qc_report.json"),
                "object_count": report["object_count"],
                "part_motion_qc_state_counts": report["part_motion_qc_state_counts"],
                "pair_qc_state_counts": report["pair_qc_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_motion_qc_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--part-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_state"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-robust-frames", type=int, default=30)
    parser.add_argument("--min-robust-median-vertices", type=float, default=100.0)
    parser.add_argument("--min-robust-median-faces", type=float, default=100.0)
    parser.add_argument("--min-robust-median-containment", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
