#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

STATUS = "v18_part_object_blocker_manifest"
CLAIM = (
    "This manifest records object-level blockers for V18 part/relative-motion objects. It prevents partial "
    "part evidence from being interpreted as hidden geometry, part pose, contact ownership, or final object pose."
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


def rows_by_object(rows: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = require_dict(raw, "row")
        out[str(row.get("object_id"))] = row
    return out


def list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def blocker_state(
    part_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    subset_records: list[dict[str, Any]],
    qc_row: dict[str, Any] | None,
    articulation_row: dict[str, Any] | None,
    part_se3_row: dict[str, Any] | None,
    part_depth_rows: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    blockers = set(list_str(part_row.get("blockers")))
    next_evidence = set(list_str(part_row.get("required_next_evidence")))
    accepted_tracks = require_int(part_row.get("accepted_part_track_count"), "accepted part track count")
    if accepted_tracks == 0:
        blockers.add("missing_accepted_part_mask_evidence")
        next_evidence.add("obtain model-produced part plan and tracked part masks overlapping the object")
        return "blocked_missing_part_mask_evidence", sorted(blockers), sorted(next_evidence)
    if candidate_row is not None and candidate_row.get("part_model_candidate_state") == "articulation_hypothesis_not_fitted":
        blockers.add("articulation_hypothesis_not_fitted")
        blockers.add("part_pose_not_estimated")
        blockers.add("hidden_geometry_not_completed")
        if articulation_row is not None:
            fit_state = str(articulation_row.get("articulation_fit_state"))
            fit_blockers = set(list_str(articulation_row.get("fit_blockers")))
            if fit_state == "articulation_fit_residual_supported_visible_center_only_not_pose":
                if part_se3_row is not None:
                    se3_state = str(part_se3_row.get("part_se3_pair_state"))
                    se3_blockers = set(list_str(part_se3_row.get("part_se3_pair_blockers")))
                    if se3_state == "part_se3_surface_residual_rejected":
                        blockers.update({"part_se3_surface_residual_rejected", *se3_blockers})
                        next_evidence.update({"repair part surface SE(3) residual outliers before part pose", "do not promote center-fit articulation without surface SE(3) support"})
                        return "blocked_part_se3_surface_residual_rejected", sorted(blockers), sorted(next_evidence)
                    if se3_state == "part_se3_surface_residual_supported_visible_only_not_pose":
                        part_mesh_count = sum(1 for item in part_depth_rows if item.get("hidden_part_geometry_reconstructed") is True)
                        if part_mesh_count > 0:
                            blockers.update({"part_se3_surface_supported_visible_only_not_pose", "part_depth_fused_geometry_candidate_visible_only_not_complete", "silhouette_residual_not_evaluated", "contact_ownership_not_validated"})
                            next_evidence.update({"validate silhouette/depth residuals for articulated parts", "validate hidden/occluded part geometry before contact or object-pose promotion"})
                            return "blocked_part_depth_fused_geometry_no_silhouette_depth_pose", sorted(blockers), sorted(next_evidence)
                        blockers.update({"part_se3_surface_supported_visible_only_not_pose", "silhouette_residual_not_evaluated", "hidden_geometry_not_completed"})
                        next_evidence.update({"validate silhouette/depth residuals for articulated parts", "complete hidden part geometry before contact or object-pose promotion"})
                        return "blocked_part_se3_supported_no_silhouette_depth_pose", sorted(blockers), sorted(next_evidence)
                    blockers.update({"part_se3_surface_residual_not_supported", *se3_blockers})
                    next_evidence.add("repair part surface SE(3) residual diagnostics")
                    return "blocked_part_se3_surface_residual_not_supported", sorted(blockers), sorted(next_evidence)
                blockers.update({"articulation_fit_visible_center_supported_not_pose", "full_part_se3_not_estimated", "silhouette_residual_not_evaluated"})
                next_evidence.update({"fit full part SE(3) and articulation parameter", "validate silhouette/depth residuals for articulated parts before contact or pose promotion"})
                return "blocked_articulation_fit_supported_no_pose", sorted(blockers), sorted(next_evidence)
            if fit_state == "articulation_fit_residual_rejected":
                blockers.update({"articulation_fit_residual_rejected", *fit_blockers})
                next_evidence.update({"repair articulation fit residuals with better part geometry or a different bounded articulation model", "do not promote rejected center-circle fit to part pose"})
                return "blocked_articulation_fit_residual_rejected", sorted(blockers), sorted(next_evidence)
            blockers.update({"articulation_fit_underconstrained", *fit_blockers})
            next_evidence.add("collect enough robust shared-frame part centers for articulation fitting")
            return "blocked_articulation_fit_underconstrained", sorted(blockers), sorted(next_evidence)
        next_evidence.add("fit bounded articulation parameter and joint axis from robust part surfaces")
        next_evidence.add("validate articulated part residuals before contact or object-pose promotion")
        return "blocked_articulation_hypothesis_not_fitted", sorted(blockers), sorted(next_evidence)
    if candidate_row is not None and require_int(candidate_row.get("rejected_candidate_count", 0), "rejected candidate count") > 0:
        blockers.add("part_model_residual_probes_rejected")
        next_evidence.add("repair rejected part residual probes by improving sparse masks or collecting more shared-frame part surfaces")
        return "blocked_part_model_residual_probes_rejected", sorted(blockers), sorted(next_evidence)
    if candidate_row is not None and subset_records:
        blockers.update(
            {
                "visible_subset_only_not_whole_object",
                "hidden_geometry_not_completed",
                "part_pose_not_estimated",
                "contact_ownership_not_validated",
            }
        )
        if qc_row is not None and qc_row.get("part_motion_qc_state") == "part_motion_confounded_by_sparse_tracks_with_some_stable_support":
            blockers.add("variable_part_motion_confounded_by_sparse_tracks")
            next_evidence.add("improve sparse part masks before interpreting variable part-pair distances")
        next_evidence.add("validate or model only robust visible subset without promoting to object pose")
        return "partial_visible_subset_only_blocked_no_pose", sorted(blockers), sorted(next_evidence)
    blockers.add("part_model_candidate_missing")
    next_evidence.add("extract depth-backed part surfaces and repeat part-motion QC")
    return "blocked_no_part_model_candidate", sorted(blockers), sorted(next_evidence)


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    part_split_path = args.part_split_root / case / "v18_part_split_evidence_report.json"
    completion_path = args.completion_gate_root / case / "v18_object_completion_gate_report.json"
    qc_path = args.part_motion_qc_root / case / "v18_part_motion_qc_report.json"
    candidates_path = args.part_model_candidates_root / case / "v18_part_model_candidates_report.json"
    articulation_path = args.articulation_fit_root / case / "v18_articulation_fit_candidates_report.json"
    part_se3_path = args.part_se3_root / case / "v18_part_se3_surface_residuals_report.json"
    subset_path = args.visible_part_subset_root / case / "v18_visible_part_subset_archive_report.json"
    part_depth_path = args.part_depth_fused_root / case / "v18_part_depth_fused_reconstruction_report.json"
    part_split = require_dict(load_json(part_split_path), f"{case} part split")
    completion = require_dict(load_json(completion_path), f"{case} completion gate")
    qc = require_dict(load_json(qc_path), f"{case} part motion qc")
    candidates = require_dict(load_json(candidates_path), f"{case} part model candidates")
    articulation = require_dict(load_json(articulation_path), f"{case} articulation fit candidates")
    part_se3 = require_dict(load_json(part_se3_path), f"{case} part SE3 surface residuals")
    subset = require_dict(load_json(subset_path), f"{case} visible part subset")
    part_depth = require_dict(load_json(part_depth_path), f"{case} part depth fused reconstruction") if part_depth_path.exists() else {"part_rows": []}
    completion_by_object = rows_by_object(require_list(completion.get("object_rows"), "completion rows"))
    qc_by_object = rows_by_object(require_list(qc.get("object_rows"), "qc rows"))
    candidate_by_object = rows_by_object(require_list(candidates.get("object_rows"), "candidate object rows"))
    articulation_by_object = rows_by_object(require_list(articulation.get("rows"), "articulation fit rows"))
    part_se3_by_object = rows_by_object(require_list(part_se3.get("rows"), "part se3 rows"))
    subset_records_by_object: dict[str, list[dict[str, Any]]] = {}
    for raw_record in require_list(subset.get("candidate_records"), "subset candidate records"):
        record = require_dict(raw_record, "subset candidate record")
        subset_records_by_object.setdefault(str(record.get("object_id")), []).append(record)
    part_depth_by_object: dict[str, list[dict[str, Any]]] = {}
    for raw_record in require_list(part_depth.get("part_rows"), "part depth fused rows"):
        record = require_dict(raw_record, "part depth fused row")
        part_depth_by_object.setdefault(str(record.get("object_id")), []).append(record)
    object_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for raw_part_row in require_list(part_split.get("object_rows"), "part split object rows"):
        part_row = require_dict(raw_part_row, "part split object row")
        object_id = str(part_row.get("object_id"))
        completion_row = completion_by_object.get(object_id, {})
        qc_row = qc_by_object.get(object_id)
        candidate_row = candidate_by_object.get(object_id)
        articulation_row = articulation_by_object.get(object_id)
        part_se3_row = part_se3_by_object.get(object_id)
        subset_records = subset_records_by_object.get(object_id, [])
        part_depth_rows = part_depth_by_object.get(object_id, [])
        state, blockers, next_evidence = blocker_state(part_row, candidate_row, subset_records, qc_row, articulation_row, part_se3_row, part_depth_rows)
        state_counts[state] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "track_id": part_row.get("track_id"),
                "name": part_row.get("name"),
                "model_physical_state_type": part_row.get("model_physical_state_type"),
                "completion_gate_state": completion_row.get("completion_gate_state", part_row.get("completion_gate_state")),
                "part_split_evidence_state": part_row.get("part_split_evidence_state"),
                "part_object_blocker_state": state,
                "accepted_part_track_count": part_row.get("accepted_part_track_count"),
                "accepted_part_track_labels": part_row.get("accepted_part_track_labels"),
                "rejected_part_model_candidate_count": require_int(candidate_row.get("rejected_candidate_count", 0), "rejected candidate count") if candidate_row else 0,
                "surface_icp_probe_count": require_int(candidate_row.get("surface_icp_probe_count", 0), "surface icp probe count") if candidate_row else 0,
                "surface_icp_probe_state_counts": candidate_row.get("surface_icp_probe_state_counts", {}) if candidate_row else {},
                "articulation_hypothesis_pair_count": require_int(candidate_row.get("articulation_hypothesis_pair_count", 0), "articulation hypothesis pair count") if candidate_row else 0,
                "articulation_fit_state": articulation_row.get("articulation_fit_state") if articulation_row else None,
                "articulation_fit_shared_frame_count": articulation_row.get("shared_frame_count") if articulation_row else 0,
                "part_se3_pair_state": part_se3_row.get("part_se3_pair_state") if part_se3_row else None,
                "part_se3_surface_supported_count": part_se3_row.get("part_se3_surface_supported_count") if part_se3_row else 0,
                "part_se3_surface_rejected_count": part_se3_row.get("part_se3_surface_rejected_count") if part_se3_row else 0,
                "visible_subset_candidate_count": len(subset_records),
                "visible_subset_rows": sum(require_int(record.get("archive_row_count"), "archive row count") for record in subset_records),
                "visible_subset_vertices": sum(require_int(record.get("vertex_count"), "vertex count") for record in subset_records),
                "visible_subset_faces": sum(require_int(record.get("face_count"), "face count") for record in subset_records),
                "part_depth_fused_mesh_candidate_count": sum(1 for record in part_depth_rows if record.get("hidden_part_geometry_reconstructed") is True),
                "part_depth_fused_source_frame_count": sum(require_int(record.get("source_frame_count"), "part depth source frame count") for record in part_depth_rows),
                "part_motion_qc_state": qc_row.get("part_motion_qc_state") if qc_row else None,
                "blockers": blockers,
                "required_next_evidence": next_evidence,
                "hidden_geometry_reconstructed": bool(any(record.get("hidden_part_geometry_reconstructed") is True for record in part_depth_rows)),
                "articulation_model_ready": False,
                "part_pose_ready": False,
                "contact_ownership_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_object_blocker_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "part_split_evidence": str(part_split_path),
            "object_completion_gate": str(completion_path),
            "part_motion_qc": str(qc_path),
            "part_model_candidates": str(candidates_path),
            "articulation_fit_candidates": str(articulation_path),
            "part_se3_surface_residuals": str(part_se3_path),
            "visible_part_subset_archive": str(subset_path),
            "part_depth_fused_reconstruction": str(part_depth_path),
        },
        "required_part_object_count": len(object_rows),
        "part_object_blocker_state_counts": dict(sorted(state_counts.items())),
        "object_rows": object_rows,
        "rejected_part_model_candidate_count": sum(require_int(row.get("rejected_part_model_candidate_count"), "rejected candidate count") for row in object_rows),
        "surface_icp_probe_count": sum(require_int(row.get("surface_icp_probe_count"), "surface icp probe count") for row in object_rows),
        "surface_icp_probe_state_counts": dict(sorted(sum((Counter(require_dict(row.get("surface_icp_probe_state_counts"), "surface icp state counts")) for row in object_rows), Counter()).items())),
        "articulation_hypothesis_pair_count": sum(require_int(row.get("articulation_hypothesis_pair_count"), "articulation hypothesis pair count") for row in object_rows),
        "articulation_fit_state_counts": dict(sorted(Counter(str(row.get("articulation_fit_state")) for row in object_rows if row.get("articulation_fit_state") is not None).items())),
        "part_se3_pair_state_counts": dict(sorted(Counter(str(row.get("part_se3_pair_state")) for row in object_rows if row.get("part_se3_pair_state") is not None).items())),
        "part_depth_fused_mesh_candidate_count": sum(require_int(row.get("part_depth_fused_mesh_candidate_count"), "part depth candidate count") for row in object_rows),
        "hidden_geometry_reconstructed_count": sum(1 for row in object_rows if row.get("hidden_geometry_reconstructed") is True),
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_object_blocker_manifest_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["part_object_blocker_state_counts"])
    summary = {
        "method": "build_v18_part_object_blocker_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "required_part_object_count": sum(require_int(report.get("required_part_object_count"), "required part object count") for report in reports),
        "part_object_blocker_state_counts": dict(sorted(state_counts.items())),
        "rejected_part_model_candidate_count": sum(require_int(report.get("rejected_part_model_candidate_count"), "rejected candidate count") for report in reports),
        "surface_icp_probe_count": sum(require_int(report.get("surface_icp_probe_count"), "surface icp probe count") for report in reports),
        "surface_icp_probe_state_counts": dict(sorted(sum((Counter(require_dict(report.get("surface_icp_probe_state_counts"), "surface icp state counts")) for report in reports), Counter()).items())),
        "articulation_hypothesis_pair_count": sum(require_int(report.get("articulation_hypothesis_pair_count"), "articulation hypothesis pair count") for report in reports),
        "articulation_fit_state_counts": dict(sorted(sum((Counter(require_dict(report.get("articulation_fit_state_counts"), "articulation fit state counts")) for report in reports), Counter()).items())),
        "part_se3_pair_state_counts": dict(sorted(sum((Counter(require_dict(report.get("part_se3_pair_state_counts"), "part se3 pair state counts")) for report in reports), Counter()).items())),
        "part_depth_fused_mesh_candidate_count": sum(require_int(report.get("part_depth_fused_mesh_candidate_count"), "part depth candidate count") for report in reports),
        "hidden_geometry_reconstructed_count": sum(require_int(report.get("hidden_geometry_reconstructed_count"), "hidden geometry reconstructed count") for report in reports),
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_object_blocker_manifest_report.json"),
                "required_part_object_count": report["required_part_object_count"],
                "part_object_blocker_state_counts": report["part_object_blocker_state_counts"],
                "rejected_part_model_candidate_count": report.get("rejected_part_model_candidate_count"),
                "surface_icp_probe_count": report.get("surface_icp_probe_count"),
                "surface_icp_probe_state_counts": report.get("surface_icp_probe_state_counts"),
                "articulation_hypothesis_pair_count": report.get("articulation_hypothesis_pair_count"),
                "articulation_fit_state_counts": report.get("articulation_fit_state_counts"),
                "part_se3_pair_state_counts": report.get("part_se3_pair_state_counts"),
                "part_depth_fused_mesh_candidate_count": report.get("part_depth_fused_mesh_candidate_count"),
                "hidden_geometry_reconstructed_count": report.get("hidden_geometry_reconstructed_count"),
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_object_blocker_manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-split-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--completion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_object_completion_gate"))
    parser.add_argument("--part-motion-qc-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--articulation-fit-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--part-se3-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_se3_surface_residuals"))
    parser.add_argument("--visible-part-subset-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_part_subset_archive"))
    parser.add_argument("--part-depth-fused-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_depth_fused_reconstruction"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
