#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from build_v18_triangle_nonpenetration_evidence import (
    final_hand_points,
    frame_depth_fused_mesh,
    load_depth_fused_mesh_index,
    load_final_frames,
    load_physical_schema_index,
    prepare_triangle_geometry,
    strict_nonpenetration_eligibility,
)

STATUS = "v18_signed_nonpenetration_evidence"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def signed_stats_prepared(points: np.ndarray, prepared: dict[str, Any], max_query_points: int) -> dict[str, Any]:
    if prepared.get("blocker"):
        return {"blocker": prepared.get("blocker")}
    if len(points) > max_query_points:
        step = max(1, int(math.ceil(len(points) / max_query_points)))
        points = points[::step]
    tree = prepared["tree"]
    centroids = prepared["centroids"]
    normals = prepared["normals"]
    _, face_idx = tree.query(points, k=1)
    nearest_centroids = centroids[np.asarray(face_idx, dtype=np.int64)]
    nearest_normals = normals[np.asarray(face_idx, dtype=np.int64)]
    signed = np.sum((points - nearest_centroids) * nearest_normals, axis=1)
    abs_signed = np.abs(signed)
    negative = signed < 0.0
    return {
        "sampled_hand_points": int(len(points)),
        "mesh_face_count": int(prepared["mesh_face_count"]),
        "mesh_vertex_count": int(prepared["mesh_vertex_count"]),
        "mesh_watertight_by_edges": bool(prepared["mesh_watertight_by_edges"]),
        "boundary_edge_count": int(prepared["boundary_edge_count"]),
        "nonmanifold_edge_count": int(prepared["nonmanifold_edge_count"]),
        "min_local_signed_distance_m": float(np.min(signed)),
        "median_local_signed_distance_m": float(np.median(signed)),
        "min_abs_local_signed_distance_m": float(np.min(abs_signed)),
        "negative_signed_distance_count": int(np.sum(negative)),
        "negative_signed_distance_fraction": float(np.mean(negative)),
        "local_signed_distance_semantics": "nearest_face_centroid_normal_projection_on_depth_fused_completion_mesh_not_ground_truth_sdf",
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    contact_path = args.contact_ownership_root / case / "v18_contact_ownership_graph_report.json"
    contact = load_json(contact_path)
    final_frames, final_ann_path = load_final_frames(case, args)
    mesh_index, depth_report_path = load_depth_fused_mesh_index(case, args)
    physical_schema, physical_schema_path = load_physical_schema_index(case, args)
    mesh_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None, str | None]] = {}
    hand_cache: dict[tuple[int, str], tuple[np.ndarray | None, str | None, str | None, str | None]] = {}
    ref_cache: dict[tuple[str, str, int], np.ndarray] = {}
    prepared_mesh_cache: dict[tuple[int, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    for raw in contact.get("rows", []):
        if not isinstance(raw, dict):
            continue
        frame_idx = raw.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        hand_side = str(raw.get("hand_side"))
        object_id = str(raw.get("object_id"))
        final_frame = final_frames.get(frame_idx)
        schema_row = physical_schema.get(object_id)
        strict_eligible, eligibility_state, eligibility_blockers = strict_nonpenetration_eligibility(schema_row)
        vertices, faces, mesh_blocker, mesh_meta = frame_depth_fused_mesh(final_frame, object_id, mesh_index, mesh_cache) if strict_eligible else (None, None, "object_not_strict_rigid_nonpenetration_eligible", mesh_index.get(object_id, {}))
        hand_key = (frame_idx, hand_side)
        if hand_key not in hand_cache:
            hand_cache[hand_key] = final_hand_points(final_frame, hand_side, args, ref_cache)
        points, hand_blocker, hand_source, hand_support_state = hand_cache[hand_key]
        row = {
            "frame_idx": frame_idx,
            "hand_side": hand_side,
            "object_id": object_id,
            "source_contact_owner_claim": raw.get("contact_owner_claim"),
            "source_accepted_contact_owner": bool(raw.get("accepted_contact_owner") is True),
            "source_min_unsigned_distance_m": raw.get("min_hand_surface_to_v16_object_mesh_m"),
            "source_contact_graph_v16_mesh_match": raw.get("v16_mesh_match"),
            "signed_nonpenetration_claim": "not_evaluated",
            "signed_nonpenetration_complete": False,
            "hand_support_state": hand_support_state,
            "require_observed_hawor_support": bool(args.require_observed_hawor_support),
            "object_mesh_backend": "depth_fused_convex_hull_visible_completion_candidate" if mesh_meta and mesh_meta.get("convex_hull_mesh_path") else "depth_fused_poisson_visible_completion_candidate" if mesh_meta else None,
            "object_mesh_path": (mesh_meta.get("convex_hull_mesh_path") or mesh_meta.get("poisson_mesh_path")) if mesh_meta else None,
            "object_mesh_status": mesh_meta.get("mesh_status") if mesh_meta else None,
            "object_physical_state_type": schema_row.get("model_physical_state_type") if isinstance(schema_row, dict) else None,
            "object_requires_part_or_relative_motion_model": bool(schema_row.get("requires_part_or_relative_motion_model")) if isinstance(schema_row, dict) else None,
            "object_secondary_deformable_or_surface_component": bool(schema_row.get("secondary_deformable_or_surface_component")) if isinstance(schema_row, dict) else None,
            "strict_nonpenetration_eligibility": eligibility_state,
            "strict_nonpenetration_eligibility_blockers": eligibility_blockers,
        }
        if mesh_blocker or hand_blocker or vertices is None or faces is None or points is None:
            blocker = mesh_blocker or hand_blocker or "missing_geometry"
            blockers[str(blocker)] = blockers.get(str(blocker), 0) + 1
            claim = "not_evaluated_object_not_strict_rigid_nonpenetration_eligible" if blocker == "object_not_strict_rigid_nonpenetration_eligible" else "blocked"
            row.update({"blocker": blocker, "signed_nonpenetration_claim": claim})
            rows.append(row)
            continue
        prepared_key = (frame_idx, object_id)
        if prepared_key not in prepared_mesh_cache:
            prepared_mesh_cache[prepared_key] = prepare_triangle_geometry(vertices, faces, args)
        stats = signed_stats_prepared(points, prepared_mesh_cache[prepared_key], args.max_query_hand_points)
        if stats.get("blocker"):
            blocker = str(stats["blocker"])
            blockers[blocker] = blockers.get(blocker, 0) + 1
            row.update({"blocker": blocker, "signed_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        min_signed = finite_float(stats.get("min_local_signed_distance_m"), 0.0)
        penetration = min_signed < -args.penetration_tolerance_m
        row.update(
            {
                **stats,
                "hand_geometry_source": hand_source,
                "penetration_tolerance_m": args.penetration_tolerance_m,
                "local_penetration_detected": penetration,
                "signed_nonpenetration_claim": "depth_fused_mesh_normal_penetration_evidence" if penetration else "depth_fused_mesh_normal_no_penetration_beyond_tolerance_evidence",
                "signed_nonpenetration_complete": False,
                "signed_nonpenetration_scope": "depth_fused_visible_point_completion_mesh_against_support_gated_hawor_mano_vertices_not_complete_object_ground_truth_sdf",
            }
        )
        rows.append(row)
    penetration_rows = sum(1 for row in rows if row.get("local_penetration_detected") is True)
    evaluated_rows = sum(1 for row in rows if row.get("signed_nonpenetration_claim") in {"depth_fused_mesh_normal_penetration_evidence", "depth_fused_mesh_normal_no_penetration_beyond_tolerance_evidence"})
    support_blocked_rows = sum(1 for row in rows if row.get("blocker") == "hand_not_observed_hawor_support_for_nonpenetration_claim")
    watertight_rows = sum(1 for row in rows if row.get("mesh_watertight_by_edges") is True)
    out = {
        "method": "build_v18_signed_nonpenetration_evidence",
        "status": STATUS,
        "claim": "Computes local signed hand-object distance evidence for accepted contact-owner rows using support-gated HaWoR MANO hand vertices and depth-fused object completion mesh normals. This is nearest-face normal projection evidence, not a complete ground-truth SDF solve.",
        "case": case,
        "sources": {"contact_ownership_graph": str(contact_path), "v18_full_annotations": final_ann_path, "depth_fused_reconstruction_report": depth_report_path, "physical_state_schema_report": physical_schema_path},
        "source_contact_rows": len(contact.get("rows", [])) if isinstance(contact.get("rows"), list) else None,
        "accepted_contact_rows": int(contact.get("contact_ownership_accepted_rows", 0)),
        "signed_rows": len(rows),
        "evaluated_signed_rows": evaluated_rows,
        "support_blocked_rows": support_blocked_rows,
        "local_penetration_detected_rows": penetration_rows,
        "mesh_watertight_rows": watertight_rows,
        "blocker_counts": dict(sorted(blockers.items())),
        "parameters": {"max_hand_points": args.max_hand_points, "max_query_hand_points": args.max_query_hand_points, "penetration_tolerance_m": args.penetration_tolerance_m, "require_observed_hawor_support": args.require_observed_hawor_support},
        "rows": rows,
        "signed_nonpenetration_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_signed_nonpenetration_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_signed_nonpenetration_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {"case": r["case"], "signed_rows": r["signed_rows"], "evaluated_signed_rows": r["evaluated_signed_rows"], "support_blocked_rows": r["support_blocked_rows"], "local_penetration_detected_rows": r["local_penetration_detected_rows"], "mesh_watertight_rows": r["mesh_watertight_rows"], "signed_nonpenetration_complete": r["signed_nonpenetration_complete"]}
            for r in reports
        ],
        "claim_scope": "support_gated_hawor_to_depth_fused_completion_mesh_signed_normal_evidence_not_complete_sdf",
    }
    write_json(args.output_root / "v18_signed_nonpenetration_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-ownership-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--depth-fused-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-query-hand-points", type=int, default=128)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    parser.add_argument("--nearest-triangle-candidates", type=int, default=32)
    parser.add_argument("--require-observed-hawor-support", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
