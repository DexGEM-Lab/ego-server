#!/usr/bin/env python3
"""Build a direct V19 object-surface contact posterior without moving metric MANO.

The point-to-plane Sim(3) refit is useful as evidence that the hand is near an
object surface, but after the metric/surface split its fitted joints are not the
accepted hand state.  This builder removes that systematic inefficiency: it keeps
metric MANO joints from a selected source and writes cyan render samples directly
on the nearest rigid-object surface, conditioned on source-MANO surface vertices
that are image/depth/proximity compatible with the object.

The output is not accepted contact ownership and not a MANO correction.  It is a
bounded object-surface posterior: "if contact exists, these are the plausible
object-surface support locations, with this source hand-to-surface gap."  The
renderer can consume it through the existing temporal-MANO state contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_v19_mano_surface_hypothesis_state import hawor_joint_map, numeric_summary, zero_summary
from refit_v19_mano_contact_similarity_interval import (
    build_observations,
    camera_to_world,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hawor-npz", type=Path, required=True, help="Metric MANO joint source to preserve exactly")
    p.add_argument("--object-mask-dilation-px", type=int, default=8)
    p.add_argument(
        "--candidate-mode",
        choices=("object_mask", "near_surface", "object_mask_or_near_surface"),
        default="object_mask_or_near_surface",
    )
    p.add_argument("--object-proximity-px", type=float, default=65.0)
    p.add_argument("--mesh-projection-dilation-px", type=int, default=10)
    p.add_argument("--max-current-surface-distance-m", type=float, default=0.22)
    p.add_argument("--max-hand-behind-surface-m", type=float, default=0.08)
    p.add_argument("--contact-proximity-weight-px", type=float, default=45.0)
    p.add_argument("--contact-distance-weight-m", type=float, default=0.12)
    p.add_argument("--min-contact-vertices", type=int, default=16)
    p.add_argument("--max-contact-vertices", type=int, default=96)
    p.add_argument("--mesh-sample-stride", type=int, default=8)
    p.add_argument(
        "--target-locality-px",
        type=float,
        default=0.0,
        help=(
            "If positive, choose object-surface posterior targets only from mesh samples projected within this many "
            "mask pixels of each source MANO vertex. This constrains broad-object nearest-neighbor links to local patches."
        ),
    )
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def source_gap_summaries(o: Any) -> dict[str, Any]:
    if len(o.contact_idx) == 0:
        empty = numeric_summary([])
        return {"distance": empty, "normal_abs": empty, "tangent": empty}
    source_contact = o.verts_cam[o.contact_idx]
    diff = source_contact - o.contact_targets_cam
    distance = np.linalg.norm(diff, axis=1)
    normal_abs = np.abs(np.sum(diff * o.contact_normals_cam, axis=1))
    tangent = np.sqrt(np.maximum(0.0, distance * distance - normal_abs * normal_abs))
    return {
        "distance": numeric_summary(distance.astype(float).tolist()),
        "normal_abs": numeric_summary(normal_abs.astype(float).tolist()),
        "tangent": numeric_summary(tangent.astype(float).tolist()),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    observations, skipped = build_observations(args)
    if not observations:
        raise RuntimeError(f"no direct contact-surface observations; skipped={skipped[:20]}")
    source_joints = hawor_joint_map(args.hawor_npz)
    rows: list[dict[str, Any]] = []
    source_distance_medians: list[float] = []
    source_normal_medians: list[float] = []
    source_tangent_medians: list[float] = []
    missing_joints: list[dict[str, Any]] = []
    for o in observations:
        key = (int(o.frame_idx), str(o.side))
        joints_world = source_joints.get(key)
        if joints_world is None:
            missing_joints.append({"frame_idx": key[0], "hand_side": key[1], "reason": "missing_hawor_source_joints"})
            continue
        posterior_surface_world = camera_to_world(o.contact_targets_cam, o.r_c2w, o.t_c2w)
        source_contact_world = camera_to_world(o.verts_cam[o.contact_idx], o.r_c2w, o.t_c2w)
        gaps = source_gap_summaries(o)
        for field, target in (
            ("distance", source_distance_medians),
            ("normal_abs", source_normal_medians),
            ("tangent", source_tangent_medians),
        ):
            median = gaps[field].get("median") if isinstance(gaps[field], dict) else None
            if isinstance(median, (int, float)) and np.isfinite(float(median)):
                target.append(float(median))
        no_solver_residual = {
            "count": 0,
            "not_applicable": True,
            "reason": (
                "direct_object_surface_posterior does not run a contact/nonpenetration solver; "
                "use contact_similarity_refit.source_hand_to_object_surface_distance_m for source gap"
            ),
        }
        rows.append(
            {
                "frame_idx": int(o.frame_idx),
                "source_frame_index": int(o.frame_idx),
                "hand_side": str(o.side),
                "interval_id": f"{o.side}_{o.frame_idx:04d}_direct_object_surface_posterior",
                "temporal_mano_state": "v19_source_metric_mano_plus_direct_object_surface_contact_posterior",
                "joint_state_policy": "hawor_npz_metric_mano_preserved",
                "optimized_joints_world_m": np.asarray(joints_world, dtype=float).tolist(),
                "optimized_vertices_world_sample_m": posterior_surface_world.astype(float).tolist(),
                "optimized_vertices_sample_ids": [],
                "object_surface_posterior_source_mano_vertex_ids": [int(x) for x in o.contact_idx.tolist()],
                "source_contact_vertices_world_sample_m": source_contact_world.astype(float).tolist(),
                "contact_surface_vertices_world_sample_m": posterior_surface_world.astype(float).tolist(),
                "contact_surface_hypothesis_state": "uncertain_object_surface_posterior_not_contact_ownership",
                "source_metric_mano_state": {"kind": "hawor_npz", "path": str(args.hawor_npz)},
                "source_hawor_npz": str(args.hawor_npz),
                "optimized_translation_world_m": [0.0, 0.0, 0.0],
                "optimized_translation_camera_m": [0.0, 0.0, 0.0],
                "optimized_rotation_vector_camera_rad": [0.0, 0.0, 0.0],
                "optimized_rotation_vector_world_rad": [0.0, 0.0, 0.0],
                "optimized_similarity_scale": 1.0,
                "optimized_rotation_norm_rad": 0.0,
                "metric_joint_shift_px": zero_summary(),
                "visible_joint_shift_px": zero_summary(),
                "contact_similarity_refit": {
                    "contact_residual_mode": "direct_object_surface_posterior",
                    "solver_stage": "none_source_gaps_only",
                    "contact_solver_applied": False,
                    "contact_vertex_count": int(len(o.contact_idx)),
                    "source_hand_to_object_surface_distance_m": gaps["distance"],
                    "source_hand_to_object_surface_normal_abs_m": gaps["normal_abs"],
                    "source_hand_to_object_surface_tangent_m": gaps["tangent"],
                    # Existing render/publish summaries read *_after_m.  For this direct posterior there is no moved hand,
                    # so the honest "after" gap remains the source hand-to-surface gap, not zero at the posterior point.
                    "contact_distance_after_m": gaps["distance"],
                    "contact_normal_abs_after_m": gaps["normal_abs"],
                    "contact_tangent_after_m": gaps["tangent"],
                    "contact_weight": numeric_summary(o.contact_weights.astype(float).tolist()),
                    "candidate_stats": o.candidate_stats,
                },
                "direct_object_surface_source_gap_m": gaps["distance"],
                "full_observed_surface_penetration_after_solver_m": no_solver_residual,
                "final_active_constraint_residual_after_solver_m": no_solver_residual,
            }
        )
    if not rows:
        raise RuntimeError(f"no direct posterior rows after preserving HaWoR joints; missing={missing_joints[:20]}")
    payload = {
        "method": "v19_direct_object_surface_contact_posterior_state",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Metric MANO joints are preserved from HaWoR. Rendered posterior samples are nearest rigid-object surface points conditioned on "
            "source MANO proximity and image/object-mask evidence. They are uncertain contact-surface support hypotheses, "
            "not accepted contact ownership, nonpenetration, or MANO correction."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "hawor_npz": str(args.hawor_npz),
        },
        "parameters": vars(args) | {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "output": str(args.output),
            "hawor_npz": str(args.hawor_npz),
        },
        "summary": {
            "rows_out": len(rows),
            "skipped_count": len(skipped),
            "missing_hawor_joint_rows": len(missing_joints),
            "source_hand_to_object_surface_distance_median": numeric_summary(source_distance_medians),
            "source_hand_to_object_surface_normal_abs_median": numeric_summary(source_normal_medians),
            "source_hand_to_object_surface_tangent_median": numeric_summary(source_tangent_medians),
            "contact_distance_after_median": numeric_summary(source_distance_medians),
            "contact_normal_abs_after_median": numeric_summary(source_normal_medians),
            "contact_tangent_after_median": numeric_summary(source_tangent_medians),
            "metric_joint_shift_px": zero_summary(),
        },
        "skipped_preview": skipped[:160],
        "missing_hawor_joint_rows_preview": missing_joints[:50],
        "per_frame_states": rows,
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args)
    write_json(args.output, payload)
    write_json(args.output.with_name(args.output.stem + "_report.json"), {k: v for k, v in payload.items() if k != "per_frame_states"})
    print(json.dumps({k: v for k, v in payload.items() if k != "per_frame_states"}, indent=2)[:20000])


if __name__ == "__main__":
    main()
