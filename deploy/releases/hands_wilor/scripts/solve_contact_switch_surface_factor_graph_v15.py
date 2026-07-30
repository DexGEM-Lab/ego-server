#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from diagnose_mesh_surface_contact_v3 import load_mesh_archive
from solve_contact_dynamics_factor_graph_v12 import (
    anchor_rows,
    build_initial_state,
    contact_observations,
    frame_map,
    load_json,
    pair_transforms,
    reliable_contact_rows,
    save_json,
    summarize,
    unpack_state,
)
from solve_contact_mode_dynamics_factor_graph_v13 import (
    augment_observations,
    contact_mode_segments,
    edge_metrics,
    filter_surface_supported_observations,
    object_edge_rows_by_contact_mode,
)
from solve_contact_handoff_factor_graph_v14 import (
    adjacent_segment_lengths,
    build_handoff_edges,
    handoff_metrics,
    segment_length_by_frame,
)


def residual_vector(x: np.ndarray, obs: list[dict], continuous_edges: list[dict], switch_edges: list[dict], args: argparse.Namespace) -> np.ndarray:
    object_points, gaps = unpack_state(x, len(obs))
    residuals: list[np.ndarray] = []
    for i, row in enumerate(obs):
        observed_object = np.asarray(row["object_center_world_m"], dtype=np.float64)
        observed_hand = np.asarray(row["hand_center_world_m"], dtype=np.float64)
        residuals.append(float(args.w_object_anchor) * (object_points[i] - observed_object))
        residuals.append(float(args.w_hand_anchor) * (object_points[i] + gaps[i] - observed_hand))
        residuals.append(float(args.w_contact_gap) * gaps[i])
    for edge in continuous_edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        transported_object = object_points[s] @ rot + trans
        transported_hand = (object_points[s] + gaps[s]) @ rot + trans
        slip = object_points[t] - transported_object
        residuals.append(float(args.w_object_motion) * (object_points[t] - transported_object))
        residuals.append(float(args.w_relative_contact) * (((object_points[t] + gaps[t]) - transported_hand) - slip))
        residuals.append(float(args.w_slip_prior) * slip)
    for edge in switch_edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        residuals.append(float(args.w_switch_gap_continuity) * (gaps[t] - gaps[s]))
    return np.concatenate([np.ravel(r) for r in residuals]).astype(np.float64)


def nearest_surface_distance(meshes: dict[int, tuple[np.ndarray, np.ndarray]], frame: int, point: np.ndarray) -> float:
    vertices, _faces = meshes[int(frame)]
    distance, _idx = cKDTree(np.asarray(vertices, dtype=np.float64)).query(point.reshape(1, 3), k=1)
    return float(distance[0])


def local_surface_transport_rows(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    obs: list[dict],
    object_points: np.ndarray,
    switch_edges: list[dict],
    radius_m: float,
) -> list[dict]:
    rows = []
    for edge in switch_edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        source_frame = int(edge["source_frame"])
        target_frame = int(edge["target_frame"])
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        source_vertices, _source_faces = meshes[source_frame]
        target_vertices, _target_faces = meshes[target_frame]
        source_vertices = np.asarray(source_vertices, dtype=np.float64)
        target_vertices = np.asarray(target_vertices, dtype=np.float64)
        source_tree = cKDTree(source_vertices)
        target_tree = cKDTree(target_vertices)
        source_ids = np.asarray(source_tree.query_ball_point(object_points[s], r=radius_m), dtype=np.int64)
        target_ids = np.asarray(target_tree.query_ball_point(object_points[t], r=radius_m), dtype=np.int64)
        if source_ids.size == 0:
            raise RuntimeError(f"no source surface vertices within {radius_m}m for frame {source_frame}")
        if target_ids.size == 0:
            raise RuntimeError(f"no target surface vertices within {radius_m}m for frame {target_frame}")
        transported_source = source_vertices[source_ids] @ rot + trans
        source_to_target_distance, _ = target_tree.query(transported_source, k=1)
        target_preimage = (target_vertices[target_ids] - trans) @ rot.T
        target_to_source_distance, _ = source_tree.query(target_preimage, k=1)
        transported_center = object_points[s] @ rot + trans
        target_preimage_center = (object_points[t] - trans) @ rot.T
        rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "source_selected_patch_region": edge["source_selected_patch_region"],
                "target_selected_patch_region": edge["target_selected_patch_region"],
                "source_selected_patch_anchor_joint": edge["source_selected_patch_anchor_joint"],
                "target_selected_patch_anchor_joint": edge["target_selected_patch_anchor_joint"],
                "surface_radius_m": float(radius_m),
                "source_surface_vertices": int(source_ids.size),
                "target_surface_vertices": int(target_ids.size),
                "source_contact_transport_surface_distance_m": nearest_surface_distance(meshes, target_frame, transported_center),
                "target_contact_preimage_surface_distance_m": nearest_surface_distance(meshes, source_frame, target_preimage_center),
                "source_neighborhood_to_target_surface_m": summarize(source_to_target_distance),
                "target_neighborhood_to_source_surface_m": summarize(target_to_source_distance),
                "object_factor_inlier_p95_m": edge["pair_row"].get("inlier_residual_m", {}).get("p95"),
                "object_factor_ready": bool(edge["pair_row"].get("rigid_factor_ready", False)),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    annotations = frame_map(load_json(args.annotations))
    meshes = load_mesh_archive(args.object_mesh_npz)
    contact_report = load_json(args.contact_report)
    rows = reliable_contact_rows(contact_report, args.contact_row_mode)
    obs_all = augment_observations(contact_observations(annotations, meshes, rows), rows)
    obs, rejected_obs = filter_surface_supported_observations(obs_all, args)
    transforms = pair_transforms(args.pair_factor_report)
    continuous_edges, transitions = object_edge_rows_by_contact_mode(obs, transforms)
    switch_edges = build_handoff_edges(transitions, obs, transforms)
    x0 = build_initial_state(obs)
    before_objects, before_gaps = unpack_state(x0, len(obs))
    before_continuous = edge_metrics(before_objects, before_gaps, continuous_edges, float(args.fps))
    before_switch = handoff_metrics(before_objects, before_gaps, switch_edges, float(args.fps))
    before_anchor_rows = anchor_rows(obs, before_objects, before_gaps)
    result = least_squares(
        lambda x: residual_vector(x, obs, continuous_edges, switch_edges, args),
        x0,
        loss=str(args.loss),
        f_scale=float(args.loss_f_scale),
        max_nfev=int(args.max_nfev),
        ftol=float(args.ftol),
        xtol=float(args.xtol),
        gtol=float(args.gtol),
    )
    after_objects, after_gaps = unpack_state(result.x, len(obs))
    after_continuous = edge_metrics(after_objects, after_gaps, continuous_edges, float(args.fps))
    after_switch = handoff_metrics(after_objects, after_gaps, switch_edges, float(args.fps))
    after_switch_surface = local_surface_transport_rows(meshes, obs, after_objects, switch_edges, float(args.switch_surface_radius_m))
    after_anchor_rows = anchor_rows(obs, after_objects, after_gaps)
    gap_norms = np.linalg.norm(after_gaps, axis=1)
    relative_contact = [row["relative_contact_residual_m"] for row in after_continuous]
    slip_speed = [row["slip_speed_m_s"] for row in after_continuous]
    switch_gap_delta = [row["gap_delta_m"] for row in after_switch]
    switch_source_surface_p95 = [
        float(row["source_neighborhood_to_target_surface_m"].get("p95", np.inf))
        for row in after_switch_surface
    ]
    switch_target_surface_p95 = [
        float(row["target_neighborhood_to_source_surface_m"].get("p95", np.inf))
        for row in after_switch_surface
    ]
    switch_source_center_surface = [row["source_contact_transport_surface_distance_m"] for row in after_switch_surface]
    switch_target_center_surface = [row["target_contact_preimage_surface_distance_m"] for row in after_switch_surface]
    pair_p95 = [
        row["object_factor_inlier_p95_m"]
        for row in [*after_continuous, *after_switch]
        if row["object_factor_inlier_p95_m"] is not None
    ]
    object_anchor_shift = [row["object_anchor_shift_m"] for row in after_anchor_rows]
    hand_anchor_shift = [row["hand_anchor_shift_m"] for row in after_anchor_rows]
    segments = contact_mode_segments(obs, after_gaps, args)
    handoff_adjacent_lengths = adjacent_segment_lengths(after_switch, segment_length_by_frame(segments))
    pass_rows = {
        "min_contact_frames": bool(len(obs) >= int(args.min_contact_frames)),
        "switch_edges_available": bool(len(after_switch) > 0),
        "continuous_edges_available": bool(len(after_continuous) > 0),
        "switch_adjacent_segment_frames": bool(handoff_adjacent_lengths and min(handoff_adjacent_lengths) >= int(args.min_switch_adjacent_segment_frames)),
        "max_contact_gap": bool(float(np.max(gap_norms)) <= float(args.max_contact_gap_m)),
        "relative_contact_residual_p95": bool((not relative_contact) or float(np.percentile(relative_contact, 95.0)) <= float(args.max_relative_contact_residual_p95_m)),
        "continuous_slip_speed_p95": bool((not slip_speed) or float(np.percentile(slip_speed, 95.0)) <= float(args.max_sliding_speed_p95_m_s)),
        "switch_gap_delta_p95": bool((not switch_gap_delta) or float(np.percentile(switch_gap_delta, 95.0)) <= float(args.max_switch_gap_delta_p95_m)),
        "switch_source_surface_transport_p95": bool(switch_source_surface_p95 and float(np.percentile(switch_source_surface_p95, 95.0)) <= float(args.max_switch_surface_transport_p95_m)),
        "switch_target_surface_preimage_p95": bool(switch_target_surface_p95 and float(np.percentile(switch_target_surface_p95, 95.0)) <= float(args.max_switch_surface_transport_p95_m)),
        "switch_source_center_surface": bool(switch_source_center_surface and float(np.percentile(switch_source_center_surface, 95.0)) <= float(args.max_switch_center_surface_m)),
        "switch_target_center_surface": bool(switch_target_center_surface and float(np.percentile(switch_target_center_surface, 95.0)) <= float(args.max_switch_center_surface_m)),
        "object_factor_inlier_p95": bool((not pair_p95) or float(np.percentile(pair_p95, 95.0)) <= float(args.max_pair_factor_inlier_p95_m)),
        "object_anchor_shift_p95": bool(float(np.percentile(object_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "hand_anchor_shift_p95": bool(float(np.percentile(hand_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "least_squares_converged": bool(result.success),
    }
    report = {
        "status": "accepted" if all(pass_rows.values()) else "rejected",
        "annotation_ready": bool(all(pass_rows.values())),
        "method": "solve_contact_switch_surface_factor_graph_v15",
        "claim_tested": "surface-supported contact-mode switches can be explained as a change of contact point on one coherently moving object surface",
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "contact_report": str(args.contact_report),
        "pair_factor_report": str(args.pair_factor_report),
        "contact_row_mode": str(args.contact_row_mode),
        "factor_graph": {
            "nodes": {
                "object_contact_point_per_frame": len(obs),
                "hand_contact_gap_per_frame": len(obs),
            },
            "edges": {
                "object_anchor": len(obs),
                "hand_anchor": len(obs),
                "contact_gap": len(obs),
                "continuous_object_motion": len(continuous_edges),
                "continuous_relative_contact": len(continuous_edges),
                "continuous_slip_prior": len(continuous_edges),
                "switch_gap_continuity": len(switch_edges),
                "switch_surface_transport_qc": len(after_switch_surface),
            },
            "weights": {
                "object_anchor": float(args.w_object_anchor),
                "hand_anchor": float(args.w_hand_anchor),
                "contact_gap": float(args.w_contact_gap),
                "object_motion": float(args.w_object_motion),
                "relative_contact": float(args.w_relative_contact),
                "slip_prior": float(args.w_slip_prior),
                "switch_gap_continuity": float(args.w_switch_gap_continuity),
            },
        },
        "solver": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "final_residual_l2": float(np.linalg.norm(result.fun)),
            "loss": str(args.loss),
            "loss_f_scale": float(args.loss_f_scale),
        },
        "thresholds": {
            "min_contact_frames": int(args.min_contact_frames),
            "min_switch_adjacent_segment_frames": int(args.min_switch_adjacent_segment_frames),
            "max_input_surface_distance_p95_m": float(args.max_input_surface_distance_p95_m),
            "switch_surface_radius_m": float(args.switch_surface_radius_m),
            "max_switch_surface_transport_p95_m": float(args.max_switch_surface_transport_p95_m),
            "max_switch_center_surface_m": float(args.max_switch_center_surface_m),
            "max_contact_gap_m": float(args.max_contact_gap_m),
            "max_relative_contact_residual_p95_m": float(args.max_relative_contact_residual_p95_m),
            "max_sliding_speed_p95_m_s": float(args.max_sliding_speed_p95_m_s),
            "max_switch_gap_delta_p95_m": float(args.max_switch_gap_delta_p95_m),
            "max_pair_factor_inlier_p95_m": float(args.max_pair_factor_inlier_p95_m),
            "max_anchor_shift_p95_m": float(args.max_anchor_shift_p95_m),
        },
        "pass": pass_rows,
        "contact_motion_regime": "contact_switch",
        "summary": {
            "contact_gap_m": summarize(gap_norms),
            "relative_contact_residual_m": summarize(relative_contact),
            "slip_speed_m_s": summarize(slip_speed),
            "switch_gap_delta_m": summarize(switch_gap_delta),
            "switch_source_surface_transport_p95_m": summarize(switch_source_surface_p95),
            "switch_target_surface_preimage_p95_m": summarize(switch_target_surface_p95),
            "switch_source_center_surface_m": summarize(switch_source_center_surface),
            "switch_target_center_surface_m": summarize(switch_target_center_surface),
            "switch_adjacent_segment_frames": summarize(handoff_adjacent_lengths),
            "object_factor_inlier_p95_m": summarize(pair_p95),
            "object_anchor_shift_m": summarize(object_anchor_shift),
            "hand_anchor_shift_m": summarize(hand_anchor_shift),
        },
        "contact_mode_segments": segments,
        "contact_mode_transitions": transitions,
        "observations": obs,
        "input_rejected_observations": rejected_obs,
        "before": {
            "contact_gap_m": summarize(np.linalg.norm(before_gaps, axis=1)),
            "anchor_rows": before_anchor_rows,
            "edge_rows": before_continuous,
            "switch_rows": before_switch,
        },
        "after": {
            "contact_gap_m": gap_norms.astype(float).tolist(),
            "object_contact_point_world_m": after_objects.astype(float).tolist(),
            "hand_contact_gap_world_m": after_gaps.astype(float).tolist(),
            "anchor_rows": after_anchor_rows,
            "edge_rows": after_continuous,
            "switch_rows": after_switch,
            "switch_surface_rows": after_switch_surface,
            "acceleration_rows": [],
        },
    }
    save_json(args.output_json, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--pair-factor-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--contact-row-mode", choices=["geometry_backed_temporal", "reliable_temporal"], default="geometry_backed_temporal")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--w-object-anchor", type=float, default=800.0)
    parser.add_argument("--w-hand-anchor", type=float, default=800.0)
    parser.add_argument("--w-contact-gap", type=float, default=260.0)
    parser.add_argument("--w-object-motion", type=float, default=160.0)
    parser.add_argument("--w-relative-contact", type=float, default=220.0)
    parser.add_argument("--w-slip-prior", type=float, default=30.0)
    parser.add_argument("--w-switch-gap-continuity", type=float, default=80.0)
    parser.add_argument("--max-input-surface-distance-p95-m", type=float, default=0.006)
    parser.add_argument("--switch-surface-radius-m", type=float, default=0.04)
    parser.add_argument("--max-switch-surface-transport-p95-m", type=float, default=0.012)
    parser.add_argument("--max-switch-center-surface-m", type=float, default=0.008)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.006)
    parser.add_argument("--max-relative-contact-residual-p95-m", type=float, default=0.002)
    parser.add_argument("--max-sliding-speed-p95-m-s", type=float, default=0.45)
    parser.add_argument("--max-switch-gap-delta-p95-m", type=float, default=0.006)
    parser.add_argument("--max-pair-factor-inlier-p95-m", type=float, default=0.012)
    parser.add_argument("--max-anchor-shift-p95-m", type=float, default=0.006)
    parser.add_argument("--min-contact-frames", type=int, default=3)
    parser.add_argument("--min-switch-adjacent-segment-frames", type=int, default=2)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--loss-f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--xtol", type=float, default=1e-10)
    parser.add_argument("--gtol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
