#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from solve_contact_mode_dynamics_factor_graph_v13 import (
    augment_observations,
    contact_mode_segments,
    edge_metrics,
    filter_surface_supported_observations,
    object_edge_rows_by_contact_mode,
)
from solve_contact_dynamics_factor_graph_v12 import (
    anchor_rows,
    build_initial_state,
    contact_observations,
    frame_map,
    load_json,
    load_mesh_archive,
    pair_transforms,
    reliable_contact_rows,
    save_json,
    summarize,
    unpack_state,
)


def residual_vector(x: np.ndarray, obs: list[dict], continuous_edges: list[dict], handoff_edges: list[dict], args: argparse.Namespace) -> np.ndarray:
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
    for edge in handoff_edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        transported_source = object_points[s] @ rot + trans
        residuals.append(float(args.w_handoff_object_motion) * (object_points[t] - transported_source))
        residuals.append(float(args.w_handoff_surface_continuity) * (gaps[t] - gaps[s]))
    return np.concatenate([np.ravel(r) for r in residuals]).astype(np.float64)


def build_handoff_edges(transitions: list[dict], obs: list[dict], transforms: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]]) -> list[dict]:
    index_by_frame = {int(row["frame_idx"]): i for i, row in enumerate(obs)}
    out = []
    for transition in transitions:
        source = int(transition["source_frame"])
        target = int(transition["target_frame"])
        if (source, target) not in transforms:
            raise RuntimeError(f"missing object motion factor for handoff edge {source}->{target}")
        rot, trans, pair_row = transforms[(source, target)]
        row = dict(transition)
        row.update(
            {
                "source_index": index_by_frame[source],
                "target_index": index_by_frame[target],
                "rotation": rot,
                "translation_m": trans,
                "pair_row": pair_row,
            }
        )
        out.append(row)
    return out


def handoff_metrics(object_points: np.ndarray, gaps: np.ndarray, edges: list[dict], fps: float) -> list[dict]:
    rows = []
    for edge in edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        source = int(edge["source_frame"])
        target = int(edge["target_frame"])
        dt = (target - source) / float(fps)
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        object_residual = object_points[t] - (object_points[s] @ rot + trans)
        gap_delta = gaps[t] - gaps[s]
        rows.append(
            {
                "source_frame": source,
                "target_frame": target,
                "source_selected_patch_region": edge["source_selected_patch_region"],
                "target_selected_patch_region": edge["target_selected_patch_region"],
                "source_selected_patch_anchor_joint": edge["source_selected_patch_anchor_joint"],
                "target_selected_patch_anchor_joint": edge["target_selected_patch_anchor_joint"],
                "object_motion_residual_m": float(np.linalg.norm(object_residual)),
                "object_motion_speed_m_s": float(np.linalg.norm(object_residual) / dt),
                "gap_delta_m": float(np.linalg.norm(gap_delta)),
                "object_factor_inlier_p95_m": edge["pair_row"].get("inlier_residual_m", {}).get("p95"),
                "object_factor_ready": bool(edge["pair_row"].get("rigid_factor_ready", False)),
            }
        )
    return rows


def segment_length_by_frame(segments: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for segment in segments:
        frames = [int(frame) for frame in segment.get("frames", [])]
        for frame in frames:
            out[frame] = len(frames)
    return out


def adjacent_segment_lengths(handoff_rows: list[dict], segment_lengths: dict[int, int]) -> list[int]:
    return [
        max(
            segment_lengths.get(int(row["source_frame"]), 0),
            segment_lengths.get(int(row["target_frame"]), 0),
        )
        for row in handoff_rows
    ]


def run(args: argparse.Namespace) -> dict:
    annotations = frame_map(load_json(args.annotations))
    meshes = load_mesh_archive(args.object_mesh_npz)
    contact_report = load_json(args.contact_report)
    rows = reliable_contact_rows(contact_report, args.contact_row_mode)
    obs_all = augment_observations(contact_observations(annotations, meshes, rows), rows)
    obs, rejected_obs = filter_surface_supported_observations(obs_all, args)
    transforms = pair_transforms(args.pair_factor_report)
    continuous_edges, transitions = object_edge_rows_by_contact_mode(obs, transforms)
    handoff_edges = build_handoff_edges(transitions, obs, transforms)
    x0 = build_initial_state(obs)
    before_objects, before_gaps = unpack_state(x0, len(obs))
    before_continuous = edge_metrics(before_objects, before_gaps, continuous_edges, float(args.fps))
    before_handoff = handoff_metrics(before_objects, before_gaps, handoff_edges, float(args.fps))
    before_anchor_rows = anchor_rows(obs, before_objects, before_gaps)
    before_residual_norm = float(np.linalg.norm(residual_vector(x0, obs, continuous_edges, handoff_edges, args)))
    result = least_squares(
        lambda x: residual_vector(x, obs, continuous_edges, handoff_edges, args),
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
    after_handoff = handoff_metrics(after_objects, after_gaps, handoff_edges, float(args.fps))
    after_anchor_rows = anchor_rows(obs, after_objects, after_gaps)
    gap_norms = np.linalg.norm(after_gaps, axis=1)
    relative_contact = [row["relative_contact_residual_m"] for row in after_continuous]
    slip_speed = [row["slip_speed_m_s"] for row in after_continuous]
    continuous_pair_p95 = [row["object_factor_inlier_p95_m"] for row in after_continuous if row["object_factor_inlier_p95_m"] is not None]
    handoff_object_motion = [row["object_motion_residual_m"] for row in after_handoff]
    handoff_speed = [row["object_motion_speed_m_s"] for row in after_handoff]
    handoff_gap_delta = [row["gap_delta_m"] for row in after_handoff]
    handoff_pair_p95 = [row["object_factor_inlier_p95_m"] for row in after_handoff if row["object_factor_inlier_p95_m"] is not None]
    object_anchor_shift = [row["object_anchor_shift_m"] for row in after_anchor_rows]
    hand_anchor_shift = [row["hand_anchor_shift_m"] for row in after_anchor_rows]
    segments = contact_mode_segments(obs, after_gaps, args)
    segment_lengths = segment_length_by_frame(segments)
    handoff_adjacent_lengths = adjacent_segment_lengths(after_handoff, segment_lengths)
    pair_p95 = continuous_pair_p95 + handoff_pair_p95
    pass_rows = {
        "min_contact_frames": bool(len(obs) >= int(args.min_contact_frames)),
        "handoff_edges_available": bool(len(after_handoff) > 0),
        "continuous_edges_available": bool(len(after_continuous) > 0),
        "handoff_adjacent_segment_frames": bool(handoff_adjacent_lengths and min(handoff_adjacent_lengths) >= int(args.min_handoff_adjacent_segment_frames)),
        "max_contact_gap": bool(float(np.max(gap_norms)) <= float(args.max_contact_gap_m)),
        "relative_contact_residual_p95": bool((not relative_contact) or float(np.percentile(relative_contact, 95.0)) <= float(args.max_relative_contact_residual_p95_m)),
        "continuous_slip_speed_p95": bool((not slip_speed) or float(np.percentile(slip_speed, 95.0)) <= float(args.max_sliding_speed_p95_m_s)),
        "handoff_object_speed_p95": bool((not handoff_speed) or float(np.percentile(handoff_speed, 95.0)) <= float(args.max_handoff_object_speed_p95_m_s)),
        "handoff_gap_delta_p95": bool((not handoff_gap_delta) or float(np.percentile(handoff_gap_delta, 95.0)) <= float(args.max_handoff_gap_delta_p95_m)),
        "object_factor_inlier_p95": bool((not pair_p95) or float(np.percentile(pair_p95, 95.0)) <= float(args.max_pair_factor_inlier_p95_m)),
        "object_anchor_shift_p95": bool(float(np.percentile(object_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "hand_anchor_shift_p95": bool(float(np.percentile(hand_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "least_squares_converged": bool(result.success),
    }
    report = {
        "status": "accepted" if all(pass_rows.values()) else "rejected",
        "annotation_ready": bool(all(pass_rows.values())),
        "method": "solve_contact_handoff_factor_graph_v14",
        "claim_tested": "surface-supported contact rows can be explained as contact-mode segments plus explicit handoff edges; acceleration consistency is not claimed for segments shorter than three frames",
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
                "handoff_object_motion": len(handoff_edges),
                "handoff_surface_continuity": len(handoff_edges),
            },
            "weights": {
                "object_anchor": float(args.w_object_anchor),
                "hand_anchor": float(args.w_hand_anchor),
                "contact_gap": float(args.w_contact_gap),
                "object_motion": float(args.w_object_motion),
                "relative_contact": float(args.w_relative_contact),
                "slip_prior": float(args.w_slip_prior),
                "handoff_object_motion": float(args.w_handoff_object_motion),
                "handoff_surface_continuity": float(args.w_handoff_surface_continuity),
            },
        },
        "solver": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "initial_residual_l2": before_residual_norm,
            "final_residual_l2": float(np.linalg.norm(result.fun)),
            "loss": str(args.loss),
            "loss_f_scale": float(args.loss_f_scale),
        },
        "thresholds": {
            "min_contact_frames": int(args.min_contact_frames),
            "min_handoff_adjacent_segment_frames": int(args.min_handoff_adjacent_segment_frames),
            "max_input_surface_distance_p95_m": float(args.max_input_surface_distance_p95_m),
            "max_contact_gap_m": float(args.max_contact_gap_m),
            "max_relative_contact_residual_p95_m": float(args.max_relative_contact_residual_p95_m),
            "max_sliding_speed_p95_m_s": float(args.max_sliding_speed_p95_m_s),
            "max_handoff_object_speed_p95_m_s": float(args.max_handoff_object_speed_p95_m_s),
            "max_handoff_gap_delta_p95_m": float(args.max_handoff_gap_delta_p95_m),
            "max_pair_factor_inlier_p95_m": float(args.max_pair_factor_inlier_p95_m),
            "max_anchor_shift_p95_m": float(args.max_anchor_shift_p95_m),
        },
        "pass": pass_rows,
        "contact_motion_regime": "handoff",
        "summary": {
            "contact_gap_m": summarize(gap_norms),
            "relative_contact_residual_m": summarize(relative_contact),
            "slip_speed_m_s": summarize(slip_speed),
            "handoff_object_motion_residual_m": summarize(handoff_object_motion),
            "handoff_object_speed_m_s": summarize(handoff_speed),
            "handoff_gap_delta_m": summarize(handoff_gap_delta),
            "handoff_adjacent_segment_frames": summarize(handoff_adjacent_lengths),
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
            "handoff_rows": before_handoff,
        },
        "after": {
            "contact_gap_m": gap_norms.astype(float).tolist(),
            "object_contact_point_world_m": after_objects.astype(float).tolist(),
            "hand_contact_gap_world_m": after_gaps.astype(float).tolist(),
            "anchor_rows": after_anchor_rows,
            "edge_rows": after_continuous,
            "handoff_rows": after_handoff,
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
    parser.add_argument("--w-handoff-object-motion", type=float, default=130.0)
    parser.add_argument("--w-handoff-surface-continuity", type=float, default=80.0)
    parser.add_argument("--max-input-surface-distance-p95-m", type=float, default=0.006)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.006)
    parser.add_argument("--max-relative-contact-residual-p95-m", type=float, default=0.002)
    parser.add_argument("--max-sliding-speed-p95-m-s", type=float, default=0.45)
    parser.add_argument("--max-handoff-object-speed-p95-m-s", type=float, default=0.45)
    parser.add_argument("--max-handoff-gap-delta-p95-m", type=float, default=0.006)
    parser.add_argument("--max-pair-factor-inlier-p95-m", type=float, default=0.012)
    parser.add_argument("--max-anchor-shift-p95-m", type=float, default=0.006)
    parser.add_argument("--min-contact-frames", type=int, default=3)
    parser.add_argument("--min-handoff-adjacent-segment-frames", type=int, default=2)
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
