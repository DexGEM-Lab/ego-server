#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_split_evidence_audit"
CLAIM = (
    "This artifact audits model-produced part/segment tracks for objects that require part or articulation handling. "
    "The default V18 candidate pool comes from the part-track source manifest, currently generated OWLv2->SAM2 "
    "tracks only; explicit extra cached roots are debug inputs and make the source pool non-uniform. Within the "
    "selected candidate pool, a part track is assigned only by mask overlap/containment with the whole-object mask. "
    "This does not create part geometry, estimate part pose, or complete object pose."
)

PART_TRACK_SOURCE_SCOPE = "v18_owlv2_sam2_generated_only"
PART_REQUIRED_ACTIONS = {"candidate_requires_part_model_not_run", "single_rigid_completion_not_allowed"}
PART_REQUIRED_STATES = {
    "part_motion_requires_part_split_no_single_rigid_completion",
    "articulated_requires_part_model_no_single_pose",
}


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


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def normalize_path(path: str) -> Path:
    candidates = [
        Path(path),
        Path(path.replace("/mnt/user-home/yiwen/ego_annotation_remote/data", "/data2/ego_annotation_outputs")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
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


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": percentile(values, 50.0),
        "p05": percentile(values, 5.0),
        "p95": percentile(values, 95.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def load_mask(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not path.exists():
        raise RuntimeError(f"mask path missing: {path}")
    arr = np.asarray(Image.open(path).convert("L")) > 0
    cache[key] = arr
    return arr


def resize_bool_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape_hw:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def sample_frames(frames: list[int], max_samples: int) -> list[int]:
    frames = sorted(set(frames))
    if len(frames) <= max_samples:
        return frames
    if max_samples <= 1:
        return [frames[len(frames) // 2]]
    positions = [round(i * (len(frames) - 1) / (max_samples - 1)) for i in range(max_samples)]
    return [frames[int(pos)] for pos in positions]


def object_mask_index(annotation: dict[str, Any], object_ids: set[str]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_obj in require_list(frame.get("objects"), "objects"):
            obj = require_dict(raw_obj, "object")
            object_id = require_str(obj.get("object_id"), "object_id")
            if object_id not in object_ids:
                continue
            mask_path = obj.get("mask_path")
            if isinstance(mask_path, str) and obj.get("renderable_mask") is True:
                out[(frame_idx, object_id)] = {
                    "mask_path": str(normalize_path(mask_path)),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "geometry_scope": obj.get("geometry_scope"),
                }
    return out


def is_part_required(row: dict[str, Any]) -> bool:
    return str(row.get("completion_action")) in PART_REQUIRED_ACTIONS or str(row.get("completion_gate_state")) in PART_REQUIRED_STATES


def part_required_objects(gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in require_list(gate.get("object_rows"), "gate object_rows"):
        row = require_dict(raw, "gate object row")
        if is_part_required(row):
            rows.append(row)
    return rows


def track_label(path: Path) -> str:
    if path.parent.name == "sam2":
        return path.parent.parent.name
    return path.parent.name


def discover_tracks(roots: list[Path]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for track_path in sorted(root.rglob("sam2_track.json")):
            resolved = track_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = require_dict(load_json(track_path), f"track {track_path}")
            visible_frames: list[int] = []
            mask_paths: dict[int, Path] = {}
            area_px: dict[int, float] = {}
            bbox_by_frame: dict[int, Any] = {}
            for key, raw in payload.items():
                try:
                    frame_idx = int(key)
                except ValueError:
                    continue
                row = require_dict(raw, f"track row {track_path}:{key}")
                if row.get("visible") is not True or not isinstance(row.get("mask_path"), str):
                    continue
                mask_path = normalize_path(str(row["mask_path"]))
                if not mask_path.exists():
                    continue
                visible_frames.append(frame_idx)
                mask_paths[frame_idx] = mask_path
                if row.get("area_px") is not None:
                    area_px[frame_idx] = finite_float(row.get("area_px"), "area_px")
                bbox_by_frame[frame_idx] = row.get("bbox_xyxy")
            if visible_frames:
                tracks.append(
                    {
                        "track_path": str(track_path),
                        "track_label": track_label(track_path),
                        "root": str(root),
                        "visible_frame_count": len(visible_frames),
                        "visible_frame_min": min(visible_frames),
                        "visible_frame_max": max(visible_frames),
                        "visible_frames": sorted(visible_frames),
                        "mask_paths": mask_paths,
                        "area_px": area_px,
                        "bbox_by_frame": bbox_by_frame,
                    }
                )
    return tracks


def evaluate_track_for_object(
    track: dict[str, Any],
    object_id: str,
    object_masks: dict[tuple[int, str], dict[str, Any]],
    mask_cache: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    visible_frames = [int(x) for x in track["visible_frames"]]
    shared = [idx for idx in visible_frames if (idx, object_id) in object_masks]
    sampled = sample_frames(shared, int(args.max_sample_frames))
    containments: list[float] = []
    object_coverages: list[float] = []
    ious: list[float] = []
    samples: list[dict[str, Any]] = []
    for idx in sampled:
        part_path = track["mask_paths"][idx]
        object_path = Path(require_str(object_masks[(idx, object_id)].get("mask_path"), "object mask_path"))
        part_mask = load_mask(part_path, mask_cache)
        object_mask = load_mask(object_path, mask_cache)
        if part_mask.shape != object_mask.shape:
            part_mask = resize_bool_mask(part_mask, object_mask.shape)
        inter = int(np.logical_and(part_mask, object_mask).sum())
        part_area = int(part_mask.sum())
        object_area = int(object_mask.sum())
        union = int(np.logical_or(part_mask, object_mask).sum())
        if part_area <= 0 or object_area <= 0 or union <= 0:
            continue
        containment = inter / part_area
        object_coverage = inter / object_area
        iou = inter / union
        containments.append(float(containment))
        object_coverages.append(float(object_coverage))
        ious.append(float(iou))
        samples.append(
            {
                "frame_idx": idx,
                "part_mask_path": str(part_path),
                "object_mask_path": str(object_path),
                "part_area_px": part_area,
                "object_area_px": object_area,
                "intersection_px": inter,
                "part_containment_in_object": float(containment),
                "object_coverage_by_part": float(object_coverage),
                "iou": float(iou),
            }
        )
    containment_median = statistics.median(containments) if containments else None
    object_coverage_median = statistics.median(object_coverages) if object_coverages else None
    if len(samples) < int(args.min_sample_frames):
        assignment_state = "insufficient_shared_mask_samples"
    elif containment_median is not None and containment_median < float(args.min_part_containment):
        assignment_state = "low_part_containment_in_object_mask"
    elif object_coverage_median is not None and object_coverage_median > float(args.max_object_coverage_by_part):
        assignment_state = "whole_object_like_track_not_part_evidence"
    else:
        assignment_state = "accepted_part_track_overlap_evidence"
    return {
        "track_path": track["track_path"],
        "track_label": track["track_label"],
        "visible_frame_count": track["visible_frame_count"],
        "visible_frame_min": track["visible_frame_min"],
        "visible_frame_max": track["visible_frame_max"],
        "shared_object_frame_count": len(shared),
        "sampled_frame_count": len(samples),
        "sampled_frames": [row["frame_idx"] for row in samples],
        "part_containment_in_object": stats(containments),
        "object_coverage_by_part": stats(object_coverages),
        "iou": stats(ious),
        "assignment_state": assignment_state,
        "accepted_as_part_evidence": assignment_state == "accepted_part_track_overlap_evidence",
        "samples": samples,
    }


def case_track_source(case: str, args: argparse.Namespace) -> dict[str, Any]:
    path = args.part_track_source_root / case / "v18_part_track_source_manifest_report.json"
    if not path.exists():
        raise RuntimeError(f"missing V18 part-track source manifest for {case}: {path}")
    report = require_dict(load_json(path), f"{case} part-track source manifest")
    roots = [Path(require_str(row.get("root"), "source root")) for row in require_list(report.get("root_records"), "root_records")]
    extra_roots: list[Path] = []
    for item in args.extra_part_track_root:
        if item.startswith(f"{case}="):
            extra_roots.append(Path(item.split("=", 1)[1]))
        elif "=" not in item:
            extra_roots.append(Path(item))
    if extra_roots:
        report = {
            **report,
            "part_track_candidate_source_scope": "source_manifest_plus_cli_extra_roots",
            "extra_part_track_roots": [str(root) for root in extra_roots],
            "uniform_part_track_generation_ready": False,
        }
    report["manifest_path"] = str(path)
    report["resolved_roots_for_audit"] = [str(root) for root in [*roots, *extra_roots]]
    return report


def part_split_state(accepted_count: int) -> str:
    if accepted_count >= 2:
        return "part_mask_evidence_available_geometry_extraction_pending"
    if accepted_count == 1:
        return "single_part_mask_evidence_insufficient_for_split"
    return "no_part_mask_overlap_evidence"


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = args.completion_gate_root / case / "v18_object_completion_gate_report.json"
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    gate = require_dict(load_json(gate_path), f"{case} completion gate")
    annotation = require_dict(load_json(annotation_path), f"{case} annotation state")
    objects = part_required_objects(gate)
    object_ids = {require_str(row.get("object_id"), "part required object_id") for row in objects}
    object_masks = object_mask_index(annotation, object_ids)
    source_manifest = case_track_source(case, args)
    roots = [Path(str(root)) for root in require_list(source_manifest.get("resolved_roots_for_audit"), "resolved roots for audit")]
    tracks = discover_tracks(roots)
    mask_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    accepted_track_count = 0
    for obj in objects:
        object_id = require_str(obj.get("object_id"), "object_id")
        evaluations = [evaluate_track_for_object(track, object_id, object_masks, mask_cache, args) for track in tracks]
        accepted = [row for row in evaluations if row["accepted_as_part_evidence"]]
        accepted_track_count += len(accepted)
        state = part_split_state(len(accepted))
        state_counts[state] += 1
        rows.append(
            {
                "object_id": object_id,
                "track_id": obj.get("track_id"),
                "name": obj.get("name"),
                "completion_gate_state": obj.get("completion_gate_state"),
                "completion_action": obj.get("completion_action"),
                "model_physical_state_type": obj.get("model_physical_state_type"),
                "fast_motion_state": obj.get("fast_motion_state"),
                "object_visible_mask_frame_count": sum(1 for key in object_masks if key[1] == object_id),
                "part_split_evidence_state": state,
                "accepted_part_track_count": len(accepted),
                "accepted_part_track_labels": [row["track_label"] for row in accepted],
                "part_geometry_extraction_ready": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
                "blockers": [] if accepted else ["no_accepted_part_track_overlap_evidence"],
                "required_next_evidence": [
                    "extract depth-backed visible surfaces for accepted part masks",
                    "estimate part/articulation state under bounded residual checks",
                    "validate contact ownership against part geometry and metric depth",
                ]
                if accepted
                else ["run model-produced part/segment plan and SAM2 tracking for candidate object", "then repeat overlap audit"],
                "candidate_part_track_evaluations": evaluations,
            }
        )
    report = {
        "method": "build_v18_part_split_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_object_completion_gate": str(gate_path),
            "v18_annotation_state": str(annotation_path),
            "v18_part_track_source_manifest": str(source_manifest.get("manifest_path")),
            "part_track_roots": [str(root) for root in roots],
        },
        "part_track_candidate_source_scope": source_manifest.get("part_track_candidate_source_scope", PART_TRACK_SOURCE_SCOPE),
        "candidate_assignment_semantics": source_manifest.get(
            "candidate_assignment_semantics_required_downstream", "overlap_and_containment_with_whole_object_mask_after_candidate_pool_selection"
        ),
        "uniform_part_track_generation_ready": bool(source_manifest.get("uniform_part_track_generation_ready")),
        "part_track_source_manifest_ready": bool(source_manifest.get("candidate_source_manifest_ready")),
        "part_required_object_count": len(objects),
        "discovered_part_track_count": len(tracks),
        "accepted_part_track_assignment_count": accepted_track_count,
        "part_split_evidence_state_counts": dict(sorted(state_counts.items())),
        "object_rows": rows,
        "part_geometry_extraction_ready_count": 0,
        "part_pose_ready_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_split_evidence_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["part_split_evidence_state_counts"])
    summary = {
        "method": "build_v18_part_split_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "part_required_object_count": sum(require_int(report.get("part_required_object_count"), "part required object count") for report in reports),
        "discovered_part_track_count": sum(require_int(report.get("discovered_part_track_count"), "discovered track count") for report in reports),
        "accepted_part_track_assignment_count": sum(
            require_int(report.get("accepted_part_track_assignment_count"), "accepted track count") for report in reports
        ),
        "part_split_evidence_state_counts": dict(sorted(state_counts.items())),
        "part_track_candidate_source_scope": "source_manifest_mixed" if len({str(report.get("part_track_candidate_source_scope")) for report in reports}) > 1 else str(reports[0].get("part_track_candidate_source_scope")),
        "candidate_assignment_semantics": "overlap_and_containment_with_whole_object_mask_after_candidate_pool_selection",
        "uniform_part_track_generation_ready": all(bool(report.get("uniform_part_track_generation_ready")) for report in reports),
        "part_track_source_manifest_ready_all_cases": all(bool(report.get("part_track_source_manifest_ready")) for report in reports),
        "part_geometry_extraction_ready_count": 0,
        "part_pose_ready_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_split_evidence_report.json"),
                "part_required_object_count": report["part_required_object_count"],
                "discovered_part_track_count": report["discovered_part_track_count"],
                "accepted_part_track_assignment_count": report["accepted_part_track_assignment_count"],
                "part_split_evidence_state_counts": report["part_split_evidence_state_counts"],
                "part_track_candidate_source_scope": report.get("part_track_candidate_source_scope"),
                "uniform_part_track_generation_ready": report.get("uniform_part_track_generation_ready"),
                "part_track_source_manifest_ready": report.get("part_track_source_manifest_ready"),
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_split_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_object_completion_gate"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--part-track-source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_track_source_manifest"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--extra-part-track-root", action="append", default=[])
    parser.add_argument("--max-sample-frames", type=int, default=12)
    parser.add_argument("--min-sample-frames", type=int, default=3)
    parser.add_argument("--min-part-containment", type=float, default=0.50)
    parser.add_argument("--max-object-coverage-by-part", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
