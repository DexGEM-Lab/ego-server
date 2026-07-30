#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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

STATUS = "v18_part_mask_acquisition_plan"
CLAIM = (
    "This artifact records what evidence is needed to acquire missing or improved part masks for V18 "
    "part/relative-motion objects, whether the local V18 OWLv2->SAM2 path has generated accepted tracks, "
    "and whether mask evidence exists. It does not create geometry, contact ownership, or pose."
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


def module_available(module: str, extra_paths: list[Path] | None = None) -> bool:
    added: list[str] = []
    for path in extra_paths or []:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    try:
        return importlib.util.find_spec(module) is not None
    finally:
        for text in added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def generated_owlv2_sam2_summary(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.owlv2_sam2_tracks_root / "v18_owlv2_sam2_part_tracks_summary.json"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "summary_exists": False, "accepted_track_count": 0}
    summary = require_dict(load_json(summary_path), "OWLv2 SAM2 summary")
    return {
        "summary_path": str(summary_path),
        "summary_exists": True,
        "accepted_track_count": int(summary.get("accepted_track_count") or 0),
        "mask_evidence_created_count": int(summary.get("mask_evidence_created_count") or 0),
    }


def accepted_tracks_by_object(case: str, args: argparse.Namespace) -> dict[str, int]:
    report_path = args.owlv2_sam2_tracks_root / case / "v18_owlv2_sam2_part_tracks_report.json"
    if not report_path.exists():
        return {}
    report = require_dict(load_json(report_path), f"{case} OWLv2 SAM2 part tracks")
    counts: dict[str, int] = {}
    for raw in require_list(report.get("track_records", []), "OWLv2 SAM2 track records"):
        row = require_dict(raw, "track row")
        if row.get("accepted_as_semantic_temporal_part_track") is True:
            object_id = str(row.get("object_id"))
            counts[object_id] = counts.get(object_id, 0) + 1
    return counts


def env_probe(args: argparse.Namespace) -> dict[str, Any]:
    cv2_available = module_available("cv2")
    torch_available = module_available("torch")
    cuda_available = False
    torch_version = None
    if torch_available:
        try:
            import torch  # type: ignore[import-not-found]

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - diagnostic only
            torch_version = f"import_error:{type(exc).__name__}:{exc}"
            cuda_available = False
    samwise_repo_candidates = [Path(raw) for raw in args.samwise_repo_candidates]
    samwise_checkpoint_candidates = [Path(raw) for raw in args.samwise_checkpoint_candidates]
    existing_samwise_repos = [str(path) for path in samwise_repo_candidates if path.exists()]
    existing_samwise_checkpoints = [str(path) for path in samwise_checkpoint_candidates if path.exists()]
    samwise_blockers: list[str] = []
    if not cv2_available:
        samwise_blockers.append("python_cv2_unavailable_for_existing_samwise_runner")
    if not cuda_available:
        samwise_blockers.append("cuda_unavailable_for_existing_samwise_runner")
    if not existing_samwise_repos:
        samwise_blockers.append("samwise_repo_not_found_in_known_paths")
    if not existing_samwise_checkpoints:
        samwise_blockers.append("samwise_checkpoint_not_found_in_known_paths")
    samwise_ready = not samwise_blockers

    sam2_repo_candidates = [Path(raw) for raw in args.sam2_repo_candidates]
    sam2_checkpoint_candidates = [Path(raw) for raw in args.sam2_checkpoint_candidates]
    existing_sam2_repos = [str(path) for path in sam2_repo_candidates if path.exists()]
    existing_sam2_checkpoints = [str(path) for path in sam2_checkpoint_candidates if path.exists()]
    sam2_import_available = module_available("sam2.build_sam", sam2_repo_candidates) and module_available("sam2.sam2_image_predictor", sam2_repo_candidates)
    segment_anything_available = module_available("segment_anything")
    sam_v1_checkpoint_candidates = [Path(raw) for raw in args.sam_v1_checkpoint_candidates]
    existing_sam_v1_checkpoints = [str(path) for path in sam_v1_checkpoint_candidates if path.exists()]
    groundingdino_available = module_available("groundingdino")
    transformers_available = module_available("transformers")
    ultralytics_available = module_available("ultralytics")
    owlv2_model_cache_candidates = [Path(raw) for raw in args.owlv2_model_cache_candidates]
    existing_owlv2_model_caches = [str(path) for path in owlv2_model_cache_candidates if path.exists()]
    owlv2_transformers_class_available = False
    if transformers_available:
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: F401  # type: ignore[import-not-found]

            owlv2_transformers_class_available = True
        except Exception:  # pragma: no cover - diagnostic only
            owlv2_transformers_class_available = False
    open_vocab_detector_backend_cached_available = owlv2_transformers_class_available and bool(existing_owlv2_model_caches)
    promptable_sam2_ready = cuda_available and sam2_import_available and bool(existing_sam2_checkpoints)
    promptable_sam_v1_ready = cuda_available and segment_anything_available and bool(existing_sam_v1_checkpoints)
    # V18 baseline is OWLv2 -> SAM2 video tracking. SAM v1 availability is diagnostic only and must not satisfy readiness.
    promptable_segmentation_backend_available = promptable_sam2_ready
    open_vocab_or_referring_prompt_backend_available = samwise_ready or groundingdino_available or open_vocab_detector_backend_cached_available
    generated_summary = generated_owlv2_sam2_summary(args)
    owlv2_sam2_tracks_ready = bool(generated_summary.get("summary_exists")) and int(generated_summary.get("accepted_track_count") or 0) > 0
    model_produced_part_prompt_plan_ready = owlv2_sam2_tracks_ready
    local_new_mask_generation_ready = samwise_ready or (
        promptable_segmentation_backend_available
        and open_vocab_or_referring_prompt_backend_available
        and model_produced_part_prompt_plan_ready
    )

    blockers: list[str] = []
    blockers.extend(samwise_blockers)
    if promptable_segmentation_backend_available and open_vocab_detector_backend_cached_available and not model_produced_part_prompt_plan_ready:
        blockers.append("open_vocab_detector_cached_but_model_produced_part_prompt_plan_not_ready")
    elif promptable_segmentation_backend_available and not open_vocab_or_referring_prompt_backend_available:
        blockers.append("promptable_sam_backend_available_but_no_open_vocab_or_referring_part_prompt_backend")
    if not promptable_segmentation_backend_available and not samwise_ready:
        blockers.append("no_promptable_sam_backend_ready")
    return {
        "cv2_available": cv2_available,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "samwise_repo_candidates_checked": [str(path) for path in samwise_repo_candidates],
        "samwise_checkpoint_candidates_checked": [str(path) for path in samwise_checkpoint_candidates],
        "existing_samwise_repos": existing_samwise_repos,
        "existing_samwise_checkpoints": existing_samwise_checkpoints,
        "existing_samwise_runner_locally_ready": samwise_ready,
        "sam2_repo_candidates_checked": [str(path) for path in sam2_repo_candidates],
        "sam2_checkpoint_candidates_checked": [str(path) for path in sam2_checkpoint_candidates],
        "existing_sam2_repos": existing_sam2_repos,
        "existing_sam2_checkpoints": existing_sam2_checkpoints,
        "sam2_import_available": sam2_import_available,
        "promptable_sam2_ready": promptable_sam2_ready,
        "segment_anything_available": segment_anything_available,
        "sam_v1_checkpoint_candidates_checked": [str(path) for path in sam_v1_checkpoint_candidates],
        "existing_sam_v1_checkpoints": existing_sam_v1_checkpoints,
        "promptable_sam_v1_ready": promptable_sam_v1_ready,
        "groundingdino_available": groundingdino_available,
        "transformers_available": transformers_available,
        "ultralytics_available": ultralytics_available,
        "owlv2_model_cache_candidates_checked": [str(path) for path in owlv2_model_cache_candidates],
        "existing_owlv2_model_caches": existing_owlv2_model_caches,
        "owlv2_transformers_class_available": owlv2_transformers_class_available,
        "open_vocab_detector_backend_cached_available": open_vocab_detector_backend_cached_available,
        "owlv2_sam2_generated_summary": generated_summary,
        "owlv2_sam2_part_tracks_ready": owlv2_sam2_tracks_ready,
        "promptable_segmentation_backend_available": promptable_segmentation_backend_available,
        "open_vocab_or_referring_prompt_backend_available": open_vocab_or_referring_prompt_backend_available,
        "model_produced_part_prompt_plan_ready": model_produced_part_prompt_plan_ready,
        "local_new_mask_generation_ready": local_new_mask_generation_ready,
        "local_generation_blockers": blockers,
    }


def acquisition_state(row: dict[str, Any]) -> tuple[str, list[str]]:
    state = str(row.get("part_object_blocker_state"))
    blockers = set(str(item) for item in row.get("blockers", []) if isinstance(row.get("blockers"), list))
    if state == "blocked_missing_part_mask_evidence":
        return "requires_new_model_produced_part_masks", sorted(blockers | {"no_accepted_part_mask_evidence"})
    if state == "partial_visible_subset_only_blocked_no_pose":
        return "requires_improved_sparse_part_masks_or_visible_subset_only_model", sorted(
            blockers | {"partial_visible_subset_not_full_part_model"}
        )
    if state == "blocked_part_model_residual_probes_rejected":
        return "requires_repair_rejected_part_model_residual_probes", sorted(blockers | {"part_model_residual_probes_rejected"})
    if state == "blocked_articulation_hypothesis_not_fitted":
        return "requires_bounded_articulation_fit_after_generated_masks", sorted(
            blockers | {"articulation_hypothesis_not_fitted", "articulation_parameter_fit_not_implemented"}
        )
    if state == "blocked_articulation_fit_supported_no_pose":
        return "requires_full_part_pose_validation_after_articulation_fit", sorted(
            blockers | {"articulation_fit_supported_but_not_pose", "full_part_se3_not_estimated"}
        )
    if state == "blocked_articulation_fit_residual_rejected":
        return "requires_repair_articulation_fit_residuals", sorted(blockers | {"articulation_fit_residual_rejected"})
    if state == "blocked_articulation_fit_underconstrained":
        return "requires_more_shared_part_frames_for_articulation_fit", sorted(blockers | {"articulation_fit_underconstrained"})
    if state == "blocked_part_se3_surface_residual_rejected":
        return "requires_repair_part_se3_surface_residuals", sorted(blockers | {"part_se3_surface_residual_rejected"})
    if state == "blocked_part_se3_supported_no_silhouette_depth_pose":
        return "requires_silhouette_depth_hidden_geometry_validation_after_part_se3", sorted(
            blockers | {"part_se3_supported_visible_only_not_pose", "silhouette_residual_not_evaluated", "hidden_geometry_not_completed"}
        )
    if state == "blocked_part_se3_surface_residual_not_supported":
        return "requires_repair_part_se3_surface_residuals", sorted(blockers | {"part_se3_surface_residual_not_supported"})
    if state == "blocked_no_part_model_candidate":
        return "requires_part_surface_model_candidate_after_generated_masks", sorted(blockers | {"part_model_candidate_missing"})
    return "requires_manual_triage", sorted(blockers | {"unclassified_part_object_blocker_state"})


def case_report(case: str, args: argparse.Namespace, env: dict[str, Any]) -> dict[str, Any]:
    blocker_path = args.part_object_blockers_root / case / "v18_part_object_blocker_manifest_report.json"
    blocker_report = require_dict(load_json(blocker_path), f"{case} blocker report")
    generated_counts = accepted_tracks_by_object(case, args)
    object_rows: list[dict[str, Any]] = []
    for raw_row in require_list(blocker_report.get("object_rows"), "blocker object rows"):
        row = require_dict(raw_row, "blocker row")
        object_id = str(row.get("object_id"))
        state, blockers = acquisition_state(row)
        generated_track_count = int(generated_counts.get(object_id, 0))
        if generated_track_count > 0:
            state = "model_produced_owlv2_sam2_part_masks_available"
            blockers = [item for item in blockers if item != "no_accepted_part_mask_evidence"]
        locally_runnable = bool(env.get("local_new_mask_generation_ready"))
        next_actions = list(row.get("required_next_evidence", [])) if isinstance(row.get("required_next_evidence"), list) else []
        if generated_track_count > 0:
            next_actions.append("feed generated OWLv2->SAM2 tracks into part-surface geometry and residual checks")
        elif not locally_runnable:
            if env.get("promptable_segmentation_backend_available") is True:
                next_actions.append("run or repair OWLv2->SAM2 part prompt/tracking path or provide precomputed part tracks; promptable SAM assets are present")
            else:
                next_actions.append("provision runnable open-vocabulary/referring video segmentation backend or provide precomputed part tracks")
        object_rows.append(
            {
                "object_id": row.get("object_id"),
                "track_id": row.get("track_id"),
                "case": case,
                "source_blocker_state": row.get("part_object_blocker_state"),
                "part_mask_acquisition_state": state,
                "accepted_part_track_count": row.get("accepted_part_track_count"),
                "visible_subset_candidate_count": row.get("visible_subset_candidate_count"),
                "local_new_mask_generation_ready": locally_runnable,
                "generated_owlv2_sam2_track_count": generated_track_count,
                "acquisition_blockers": blockers + ([] if generated_track_count > 0 else list(env.get("local_generation_blockers", []))),
                "required_next_actions": sorted(set(str(item) for item in next_actions)),
                "mask_evidence_created": generated_track_count > 0,
                "part_geometry_created": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_mask_acquisition_plan",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"part_object_blockers": str(blocker_path), "owlv2_sam2_part_tracks": str(args.owlv2_sam2_tracks_root / case / "v18_owlv2_sam2_part_tracks_report.json")},
        "environment": env,
        "object_count": len(object_rows),
        "local_new_mask_generation_ready_count": sum(1 for row in object_rows if row["local_new_mask_generation_ready"]),
        "mask_evidence_created_count": sum(int(row.get("generated_owlv2_sam2_track_count") or 0) for row in object_rows),
        "object_rows": object_rows,
        "acquisition_blocker_counts": dict(sorted(Counter(item for row in object_rows for item in row.get("acquisition_blockers", [])).items())),
        "unclassified_acquisition_blocker_count": sum(1 for row in object_rows for item in row.get("acquisition_blockers", []) if item == "unclassified_part_object_blocker_state"),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_mask_acquisition_plan_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    env = env_probe(args)
    reports = [case_report(case, args, env) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_part_mask_acquisition_plan",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "environment": env,
        "object_count": sum(int(report["object_count"]) for report in reports),
        "local_new_mask_generation_ready_count": sum(int(report["local_new_mask_generation_ready_count"]) for report in reports),
        "mask_evidence_created_count": sum(int(report["mask_evidence_created_count"]) for report in reports),
        "acquisition_blocker_counts": dict(sorted(sum((Counter(report.get("acquisition_blocker_counts", {})) for report in reports), Counter()).items())),
        "unclassified_acquisition_blocker_count": sum(int(report.get("unclassified_acquisition_blocker_count", 0)) for report in reports),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_mask_acquisition_plan_report.json"),
                "object_count": report["object_count"],
                "local_new_mask_generation_ready_count": report["local_new_mask_generation_ready_count"],
                "unclassified_acquisition_blocker_count": report.get("unclassified_acquisition_blocker_count", 0),
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_mask_acquisition_plan_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan"))
    parser.add_argument("--owlv2-sam2-tracks-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_owlv2_sam2_part_tracks"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
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
    parser.add_argument(
        "--sam2-repo-candidates",
        nargs="+",
        default=[
            "third_party/sam2",
            "/home/yiwen/ego_annotation/third_party/sam2",
            "/home/yiwen/sam2",
            "/data2/sam2",
        ],
    )
    parser.add_argument(
        "--sam2-checkpoint-candidates",
        nargs="+",
        default=[
            "/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt",
            "/data2/checkpoints/sam2.1_hiera_small.pt",
            "/home/yiwen/ego_annotation/checkpoints/sam2.1_hiera_small.pt",
        ],
    )
    parser.add_argument(
        "--sam-v1-checkpoint-candidates",
        nargs="+",
        default=[
            "/home/yiwen/ego_annotation/checkpoints/sam_vit_b_01ec64.pth",
            "/data2/checkpoints/sam_vit_b_01ec64.pth",
            "/data2/ego_annotation_outputs/checkpoints/sam_vit_b_01ec64.pth",
        ],
    )
    parser.add_argument(
        "--owlv2-model-cache-candidates",
        nargs="+",
        default=[
            "/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d",
            "/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble",
        ],
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
