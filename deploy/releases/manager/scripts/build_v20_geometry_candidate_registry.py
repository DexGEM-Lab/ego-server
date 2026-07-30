#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from v20_common import ContractError, load_json, safe_id, write_json


def load_mesh_stats(path: Path) -> dict[str, Any]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ContractError(f"invalid_geometry_candidate_mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if not np.isfinite(extent).all() or float(extent.max()) <= 0.0:
        raise ContractError(f"invalid_geometry_candidate_extent: {path}")
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(mesh.faces)),
        "extent_model_units": extent.astype(float).tolist(),
        "center_model_units": vertices.mean(axis=0).astype(float).tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def object_records_from_plan(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if isinstance(plan, dict) and isinstance(plan.get("objects"), list):
        return [row for row in plan["objects"] if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
        return [row for row in payload["objects"] if isinstance(row, dict)]
    raise ContractError(f"object_plan_has_no_objects: {path}")


def conditioning_packet(object_row: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    physical_model = object_row.get("physical_model") if isinstance(object_row.get("physical_model"), dict) else {}
    branch = physical_model.get("primary_physical_model") or object_row.get("physical_branch") or "unresolved"
    return {
        "object_id": candidate["object_id"],
        "semantic_name": object_row.get("description") or object_row.get("track_id") or candidate["object_id"],
        "physical_branch": branch,
        "agent_shape_description": object_row.get("physical_notes") or object_row.get("description") or "",
        "visible_evidence": {
            "keyframes": object_row.get("sampled_frames", []),
            "masks": object_row.get("mask_paths", []),
            "visible_surfaces": candidate.get("visible_surface_sources", []),
            "depth_candidate_ids": args.depth_candidate_ids or [],
        },
        "size_scale_hints": object_row.get("size_scale_hints", {"metric_extent_range_m": None, "supporting_depth_sources": args.depth_candidate_ids or []}),
        "occlusion_notes": object_row.get("occlusion_notes", []),
        "contact_notes": object_row.get("contact_notes", []),
        "negative_constraints": [
            "must_not_fill_observed_free_space",
            "must_not_replace_visible_depth_surface",
            "must_not_ignore_part_required_state",
            "must_not_use_eval_ref_geometry_or_pose_for_prediction",
        ],
    }


def parse_candidate(raw: str) -> dict[str, Any]:
    parts = raw.split("|")
    if len(parts) < 5:
        raise ContractError("--candidate format: object_id|candidate_id|method_family|mesh_path|source_report[|conditioning_object_track_id]")
    object_id, candidate_id, method_family, mesh_path, source_report = parts[:5]
    if "gt" in candidate_id.lower() or "ground_truth" in candidate_id.lower() or "oracle" in candidate_id.lower() or "gt" in method_family.lower() or "ground_truth" in method_family.lower() or "oracle" in method_family.lower():
        raise ContractError(f"reference_or_oracle_geometry_candidate_forbidden: {candidate_id}")
    return {
        "object_id": object_id,
        "candidate_id": candidate_id,
        "method_family": method_family,
        "mesh_path": str(Path(mesh_path)),
        "source_report": None if source_report in {"", "none", "None"} else str(Path(source_report)),
        "conditioning_track_id": parts[5] if len(parts) > 5 else object_id.split(":")[-1],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    object_plan_rows = object_records_from_plan(args.object_plan)
    object_plan_by_track = {str(row.get("track_id", row.get("object_id", ""))): row for row in object_plan_rows}
    object_plan_by_object_id = {str(row.get("object_id", row.get("target_object_id", row.get("model_object_id", "")))): row for row in object_plan_rows}
    candidates: list[dict[str, Any]] = []
    packet_paths: list[str] = []
    for raw in args.candidate or []:
        candidate = parse_candidate(raw)
        mesh_path = Path(candidate["mesh_path"])
        if not mesh_path.exists():
            raise ContractError(f"missing_geometry_candidate_mesh: {mesh_path}")
        stats = load_mesh_stats(mesh_path)
        object_row = object_plan_by_track.get(str(candidate["conditioning_track_id"])) or object_plan_by_object_id.get(str(candidate["object_id"]))
        if object_row is None:
            raise ContractError(f"geometry_candidate_not_tied_to_selected_object_plan_target: {candidate['object_id']}:{candidate['candidate_id']}")
        packet = conditioning_packet(object_row, candidate, args)
        packet_path = args.output_dir / "agent_conditioning_packets" / f"{safe_id(candidate['object_id'])}.json"
        write_json(packet_path, packet)
        record = {
            "schema": "v20_geometry_candidate.v0",
            "candidate_id": candidate["candidate_id"],
            "object_id": candidate["object_id"],
            "method_family": candidate["method_family"],
            "mesh_path": str(mesh_path),
            "source_report": candidate["source_report"],
            "agent_conditioning_packet": str(packet_path),
            "mesh_stats": stats,
            "accepted_geometry": False,
            "promotion_status": "retained_prior_unvalidated",
            "evaluation_reference_allowed_in_prediction": False,
        }
        candidate_dir = args.output_dir / safe_id(candidate["object_id"]) / safe_id(candidate["candidate_id"])
        write_json(candidate_dir / "geometry_candidate.json", record)
        candidates.append(record)
        packet_paths.append(str(packet_path))
    if not candidates and args.require_candidate:
        raise ContractError("v20_geometry_candidate_registry_failed: no_real_geometry_candidates_provided")
    registry = {
        "schema": "v20_geometry_candidate_registry.v0",
        "claim_scope": "Registry contains prediction-side generated or reconstructed mesh candidates only. Promotion requires validation against visible evidence.",
        "evaluation_reference_policy": "Reference geometry/pose is forbidden in prediction candidates and may be read only by evaluation.",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "agent_conditioning_packets": packet_paths,
    }
    write_json(args.output_dir / "geometry_candidate_registry.json", registry)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V20 prediction-side geometry candidate registry.")
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--depth-candidate-id", dest="depth_candidate_ids", action="append")
    parser.add_argument("--candidate", action="append", help="object_id|candidate_id|method_family|mesh_path|source_report[|conditioning_object_track_id]")
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args())["candidate_count"])
