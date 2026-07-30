#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
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

STATUS = "v18_part_track_source_manifest"
CLAIM = (
    "This artifact is the V18 source-of-truth for part-track candidate inputs. The default V18 pool contains only "
    "generated OWLv2->SAM2 accepted part-track roots. Legacy cached roots are allowed only through explicit debug "
    "CLI arguments and make the candidate source pool non-uniform. Downstream assignment must still use geometric "
    "overlap/containment with whole-object masks and must not treat candidate source availability as pose, hidden "
    "geometry, or contact evidence."
)

DEFAULT_CACHED_PART_TRACK_ROOTS_BY_CASE: dict[str, list[Path]] = {
    "trash_1050": [],
    "task5_tomato_960": [],
}
DEFAULT_V18_OWLV2_SAM2_TRACK_ROOT = Path("/data2/ego_annotation_outputs/v18_owlv2_sam2_part_tracks")


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


def normalize_path(path: str) -> Path:
    candidates = [
        Path(path),
        Path(path.replace("/mnt/user-home/yiwen/ego_annotation_remote/data", "/data2/ego_annotation_outputs")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def track_label(path: Path) -> str:
    if path.parent.name == "sam2":
        return path.parent.parent.name
    return path.parent.name


def discover_track_record(track_path: Path, root: Path) -> dict[str, Any]:
    payload = require_dict(load_json(track_path), f"track {track_path}")
    visible_frames: list[int] = []
    existing_mask_count = 0
    missing_mask_count = 0
    for key, raw in payload.items():
        try:
            frame_idx = int(key)
        except ValueError:
            continue
        row = require_dict(raw, f"track row {track_path}:{key}")
        if row.get("visible") is not True or not isinstance(row.get("mask_path"), str):
            continue
        visible_frames.append(frame_idx)
        if normalize_path(str(row["mask_path"])).exists():
            existing_mask_count += 1
        else:
            missing_mask_count += 1
    return {
        "track_path": str(track_path),
        "track_label": track_label(track_path),
        "root": str(root),
        "visible_frame_count": len(visible_frames),
        "visible_frame_min": min(visible_frames) if visible_frames else None,
        "visible_frame_max": max(visible_frames) if visible_frames else None,
        "existing_visible_mask_count": existing_mask_count,
        "missing_visible_mask_count": missing_mask_count,
        "usable_for_overlap_audit": bool(visible_frames and existing_mask_count > 0),
    }


def discover_tracks(roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for track_path in sorted(root.rglob("sam2_track.json")):
            resolved = track_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(discover_track_record(track_path, root))
    return records


def generated_owlv2_sam2_summary(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.owlv2_sam2_tracks_root / "v18_owlv2_sam2_part_tracks_summary.json"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "summary_exists": False, "accepted_track_count": 0}
    summary = require_dict(load_json(summary_path), "OWLv2 SAM2 part-track summary")
    accepted = int(summary.get("accepted_track_count") or 0)
    return {"summary_path": str(summary_path), "summary_exists": True, "accepted_track_count": accepted}


def env_probe(args: argparse.Namespace) -> dict[str, Any]:
    cv2_available = importlib.util.find_spec("cv2") is not None
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    torch_version = None
    if torch_available:
        try:
            import torch  # type: ignore[import-not-found]

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - diagnostic only
            torch_version = f"import_error:{type(exc).__name__}:{exc}"
    repo_candidates = [Path(raw) for raw in args.samwise_repo_candidates]
    checkpoint_candidates = [Path(raw) for raw in args.samwise_checkpoint_candidates]
    existing_repos = [str(path) for path in repo_candidates if path.exists()]
    existing_checkpoints = [str(path) for path in checkpoint_candidates if path.exists()]
    generated_summary = generated_owlv2_sam2_summary(args)
    owlv2_sam2_ready = bool(generated_summary.get("summary_exists")) and int(generated_summary.get("accepted_track_count") or 0) > 0
    backend_blockers: list[str] = []
    if not cv2_available:
        backend_blockers.append("python_cv2_unavailable_for_existing_samwise_runner")
    if not cuda_available:
        backend_blockers.append("cuda_unavailable_for_existing_samwise_runner")
    if not existing_repos:
        backend_blockers.append("samwise_repo_not_found_in_known_paths")
    if not existing_checkpoints:
        backend_blockers.append("samwise_checkpoint_not_found_in_known_paths")
    samwise_ready = not backend_blockers
    if not owlv2_sam2_ready:
        backend_blockers.append("v18_owlv2_sam2_part_tracks_not_generated_or_no_accepted_tracks")
    uniform_ready = samwise_ready or owlv2_sam2_ready
    backend = "owlv2_sam2_part_tracks" if owlv2_sam2_ready else ("samwise_referring_masks_candidate" if samwise_ready else None)
    return {
        "cv2_available": cv2_available,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "samwise_repo_candidates_checked": [str(path) for path in repo_candidates],
        "samwise_checkpoint_candidates_checked": [str(path) for path in checkpoint_candidates],
        "existing_samwise_repos": existing_repos,
        "existing_samwise_checkpoints": existing_checkpoints,
        "owlv2_sam2_generated_summary": generated_summary,
        "owlv2_sam2_part_tracks_ready": owlv2_sam2_ready,
        "uniform_part_track_generation_ready": uniform_ready,
        "uniform_generation_backend": backend,
        "uniform_generation_blockers": [] if uniform_ready else backend_blockers,
    }


def source_type_for_root(root: Path) -> tuple[str, str]:
    if "v18_owlv2_sam2_part_tracks" in str(root):
        return "v18_owlv2_sam2_generated_part_track_root", "uniform_v18_owlv2_sam2_generation"
    return "cached_model_sam2_part_track_root", "case_configured_cached_evidence"


def roots_for_case(case: str, args: argparse.Namespace) -> list[Path]:
    roots = [args.owlv2_sam2_tracks_root / case / "accepted_tracks"]
    roots.extend(DEFAULT_CACHED_PART_TRACK_ROOTS_BY_CASE.get(case, []))
    for item in args.extra_cached_part_track_root:
        if item.startswith(f"{case}="):
            roots.append(Path(item.split("=", 1)[1]))
        elif "=" not in item:
            roots.append(Path(item))
    return roots


def candidate_source_scope(root_records: list[dict[str, Any]]) -> str:
    has_generated = any(row.get("source_type") == "v18_owlv2_sam2_generated_part_track_root" and row.get("exists") for row in root_records)
    has_cached = any(row.get("source_type") == "cached_model_sam2_part_track_root" and row.get("exists") for row in root_records)
    if has_generated and not has_cached:
        return "v18_owlv2_sam2_generated_only"
    if has_generated and has_cached:
        return "v18_owlv2_sam2_generated_plus_explicit_cached_debug_roots"
    return "cached_case_configured_roots_not_uniform_generation_backend"


def case_report(case: str, args: argparse.Namespace, env: dict[str, Any]) -> dict[str, Any]:
    roots = roots_for_case(case, args)
    root_records = []
    for root in roots:
        source_type, source_scope = source_type_for_root(root)
        root_records.append(
            {
                "root": str(root),
                "exists": root.exists(),
                "source_type": source_type,
                "source_scope": source_scope,
            }
        )
    tracks = discover_tracks(roots)
    label_counts = Counter(str(track.get("track_label")) for track in tracks)
    usable_tracks = [track for track in tracks if bool(track.get("usable_for_overlap_audit"))]
    source_scope = candidate_source_scope(root_records)
    uniform_source_pool = source_scope == "v18_owlv2_sam2_generated_only"
    report = {
        "method": "build_v18_part_track_source_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "part_track_candidate_source_scope": source_scope,
        "candidate_source_manifest_ready": True,
        "uniform_part_track_generation_ready": bool(env.get("uniform_part_track_generation_ready")) and uniform_source_pool,
        "uniform_generation_blockers": env.get("uniform_generation_blockers") if uniform_source_pool else ["candidate_pool_contains_explicit_cached_debug_roots"],
        "candidate_assignment_semantics_required_downstream": "overlap_and_containment_with_whole_object_mask_after_candidate_pool_selection",
        "root_count": len(root_records),
        "existing_root_count": sum(1 for root in root_records if bool(root.get("exists"))),
        "root_records": root_records,
        "track_count": len(tracks),
        "usable_track_count": len(usable_tracks),
        "track_label_counts": dict(sorted(label_counts.items())),
        "track_records": tracks,
        "mask_evidence_created": False,
        "part_geometry_created": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_track_source_manifest_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    env = env_probe(args)
    reports = [case_report(case, args, env) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary_scope = "v18_owlv2_sam2_generated_only" if reports and all(str(report.get("part_track_candidate_source_scope")) == "v18_owlv2_sam2_generated_only" for report in reports) else ("v18_owlv2_sam2_generated_plus_explicit_cached_debug_roots" if any(str(report.get("part_track_candidate_source_scope")) == "v18_owlv2_sam2_generated_plus_explicit_cached_debug_roots" for report in reports) else "cached_case_configured_roots_not_uniform_generation_backend")
    uniform_source_pool = summary_scope == "v18_owlv2_sam2_generated_only"
    summary = {
        "method": "build_v18_part_track_source_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "environment": env,
        "part_track_candidate_source_scope": summary_scope,
        "candidate_source_manifest_ready": True,
        "uniform_part_track_generation_ready": bool(env.get("uniform_part_track_generation_ready")) and uniform_source_pool,
        "uniform_generation_blockers": env.get("uniform_generation_blockers") if uniform_source_pool else ["candidate_pool_contains_explicit_cached_debug_roots"],
        "root_count": sum(int(report["root_count"]) for report in reports),
        "existing_root_count": sum(int(report["existing_root_count"]) for report in reports),
        "track_count": sum(int(report["track_count"]) for report in reports),
        "usable_track_count": sum(int(report["usable_track_count"]) for report in reports),
        "mask_evidence_created_count": 0,
        "part_geometry_created_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_track_source_manifest_report.json"),
                "root_count": report["root_count"],
                "existing_root_count": report["existing_root_count"],
                "track_count": report["track_count"],
                "usable_track_count": report["usable_track_count"],
                "uniform_part_track_generation_ready": report["uniform_part_track_generation_ready"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_track_source_manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_track_source_manifest"))
    parser.add_argument("--owlv2-sam2-tracks-root", type=Path, default=DEFAULT_V18_OWLV2_SAM2_TRACK_ROOT)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--extra-cached-part-track-root", action="append", default=[])
    parser.add_argument(
        "--samwise-repo-candidates",
        nargs="+",
        default=["/home/yiwen/SAMWISE", "/home/yiwen/samwise", "/data2/SAMWISE", "/data2/samwise"],
    )
    parser.add_argument(
        "--samwise-checkpoint-candidates",
        nargs="+",
        default=[
            "/data2/checkpoints/samwise.pth",
            "/data2/ego_annotation_models/samwise.pth",
            "/home/yiwen/models/samwise.pth",
            "/home/yiwen/checkpoints/samwise.pth",
        ],
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
