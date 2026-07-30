#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from solve_contact_dynamics_factor_graph_v12 import (
    anchor_rows,
    build_initial_state,
    contact_observations,
    edge_metrics,
    frame_map,
    load_json,
    load_mesh_archive,
    pair_transforms,
    reliable_contact_rows,
    save_json,
    summarize,
    unpack_state,
)


def contact_mode_key(row: dict) -> tuple[Any, ...]:
    required = ["hand_idx", "side", "selected_patch_source", "selected_patch_region", "selected_patch_anchor_joint"]
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise RuntimeError(f"contact row for frame {row.get('frame_idx')} lacks contact-mode fields: {missing}")
    return tuple(row[key] for key in required)


def augment_observations(obs: list[dict], rows: list[dict]) -> list[dict]:
    if len(obs) != len(rows):
        raise RuntimeError("observation/contact row count mismatch")
    out = []
    for observed, raw in zip(obs, rows):
        row = dict(observed)
        key = contact_mode_key(raw)
        row["selected_patch_anchor_joint"] = int(raw["selected_patch_anchor_joint"])
        row["selected_patch_track_key"] = raw.get("selected_patch_track_key")
        row["geometry_backed_selected_patch_track_key"] = raw.get("geometry_backed_selected_patch_track_key")
        row["contact_mode_key"] = [str(part) for part in key]
        out.append(row)
    return out


def filter_surface_supported_observations(obs: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    kept = []
    rejected = []
    for row in obs:
        p95 = row.get("surface_distance_m", {}).get("p95")
        ok = p95 is not None and float(p95) <= float(args.max_input_surface_distance_p95_m)
        target = kept if ok else rejected
        target.append(row)
    if len(kept) == 0:
        raise RuntimeError("no selected contact observations pass input surface-distance support")
    return kept, rejected


def frame_index_map(obs: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, row in enumerate(obs):
        frame = int(row["frame_idx"])
        if frame in out:
            raise RuntimeError(f"duplicate selected contact row for frame {frame}")
        out[frame] = i
    return out


def object_edge_rows_by_contact_mode(
    obs: list[dict],
    transforms: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]],
) -> tuple[list[dict], list[dict]]:
    by_frame = frame_index_map(obs)
    frames = sorted(by_frame)
    edges = []
    transitions = []
    for source, target in zip(frames[:-1], frames[1:]):
        source_obs = obs[by_frame[source]]
        target_obs = obs[by_frame[target]]
        same_mode = source_obs["contact_mode_key"] == target_obs["contact_mode_key"]
        if not same_mode:
            transitions.append(
                {
                    "source_frame": source,
                    "target_frame": target,
                    "source_contact_mode_key": source_obs["contact_mode_key"],
                    "target_contact_mode_key": target_obs["contact_mode_key"],
                    "source_selected_patch_region": source_obs["selected_patch_region"],
                    "target_selected_patch_region": target_obs["selected_patch_region"],
                    "source_selected_patch_anchor_joint": source_obs["selected_patch_anchor_joint"],
                    "target_selected_patch_anchor_joint": target_obs["selected_patch_anchor_joint"],
                }
            )
            continue
        if (source, target) not in transforms:
            raise RuntimeError(f"missing object motion factor for continuous contact edge {source}->{target}")
        rot, trans, pair_row = transforms[(source, target)]
        edges.append(
            {
                "source_frame": source,
                "target_frame": target,
                "source_index": by_frame[source],
                "target_index": by_frame[target],
                "rotation": rot,
                "translation_m": trans,
                "pair_row": pair_row,
                "contact_mode_key": source_obs["contact_mode_key"],
            }
        )
    return edges, transitions


def residual_vector(x: np.ndarray, obs: list[dict], edges: list[dict], args: argparse.Namespace) -> np.ndarray:
    object_points, gaps = unpack_state(x, len(obs))
    residuals: list[np.ndarray] = []
    for i, row in enumerate(obs):
        observed_object = np.asarray(row["object_center_world_m"], dtype=np.float64)
        observed_hand = np.asarray(row["hand_center_world_m"], dtype=np.float64)
        residuals.append(float(args.w_object_anchor) * (object_points[i] - observed_object))
        residuals.append(float(args.w_hand_anchor) * (object_points[i] + gaps[i] - observed_hand))
        residuals.append(float(args.w_contact_gap) * gaps[i])
    for edge in edges:
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
    for prev_edge, cur_edge in continuous_edge_triplets(edges):
        prev_frame = int(prev_edge["source_frame"])
        mid_frame = int(prev_edge["target_frame"])
        next_frame = int(cur_edge["target_frame"])
        dt_prev = (mid_frame - prev_frame) / float(args.fps)
        dt_cur = (next_frame - mid_frame) / float(args.fps)
        if dt_prev <= 0.0 or dt_cur <= 0.0:
            raise RuntimeError("frames must be strictly increasing")
        ip = int(prev_edge["source_index"])
        im = int(prev_edge["target_index"])
        inext = int(cur_edge["target_index"])
        object_vel_prev = (object_points[im] - object_points[ip]) / dt_prev
        object_vel_cur = (object_points[inext] - object_points[im]) / dt_cur
        hand_points = object_points + gaps
        hand_vel_prev = (hand_points[im] - hand_points[ip]) / dt_prev
        hand_vel_cur = (hand_points[inext] - hand_points[im]) / dt_cur
        denom = (dt_prev + dt_cur) * 0.5
        object_acc = (object_vel_cur - object_vel_prev) / denom
        hand_acc = (hand_vel_cur - hand_vel_prev) / denom
        residuals.append(float(args.w_acceleration_consistency) * (hand_acc - object_acc))
    return np.concatenate([np.ravel(r) for r in residuals]).astype(np.float64)


def continuous_edge_triplets(edges: list[dict]) -> list[tuple[dict, dict]]:
    edge_by_source = {int(edge["source_frame"]): edge for edge in edges}
    triplets = []
    for edge in edges:
        next_edge = edge_by_source.get(int(edge["target_frame"]))
        if next_edge is None:
            continue
        if edge["contact_mode_key"] != next_edge["contact_mode_key"]:
            continue
        triplets.append((edge, next_edge))
    return triplets


def acceleration_metrics_by_edges(object_points: np.ndarray, gaps: np.ndarray, edges: list[dict], fps: float) -> list[dict]:
    rows = []
    hand_points = object_points + gaps
    for prev_edge, cur_edge in continuous_edge_triplets(edges):
        prev_frame = int(prev_edge["source_frame"])
        mid_frame = int(prev_edge["target_frame"])
        next_frame = int(cur_edge["target_frame"])
        dt_prev = (mid_frame - prev_frame) / float(fps)
        dt_cur = (next_frame - mid_frame) / float(fps)
        ip = int(prev_edge["source_index"])
        im = int(prev_edge["target_index"])
        inext = int(cur_edge["target_index"])
        object_vel_prev = (object_points[im] - object_points[ip]) / dt_prev
        object_vel_cur = (object_points[inext] - object_points[im]) / dt_cur
        hand_vel_prev = (hand_points[im] - hand_points[ip]) / dt_prev
        hand_vel_cur = (hand_points[inext] - hand_points[im]) / dt_cur
        denom = (dt_prev + dt_cur) * 0.5
        object_acc = (object_vel_cur - object_vel_prev) / denom
        hand_acc = (hand_vel_cur - hand_vel_prev) / denom
        rows.append(
            {
                "center_frame": mid_frame,
                "contact_mode_key": prev_edge["contact_mode_key"],
                "object_acceleration_m_s2": float(np.linalg.norm(object_acc)),
                "hand_acceleration_m_s2": float(np.linalg.norm(hand_acc)),
                "acceleration_consistency_residual_m_s2": float(np.linalg.norm(hand_acc - object_acc)),
            }
        )
    return rows


def contact_mode_segments(obs: list[dict], gaps: np.ndarray, args: argparse.Namespace) -> list[dict]:
    segments = []
    current: dict[str, Any] | None = None
    for i, row in enumerate(obs):
        key = row["contact_mode_key"]
        gap = float(np.linalg.norm(gaps[i]))
        new_segment = current is None or current["contact_mode_key"] != key or gap > float(args.max_contact_gap_m)
        if new_segment:
            if current is not None:
                segments.append(current)
            current = {
                "contact_mode_key": key,
                "selected_patch_region": row["selected_patch_region"],
                "selected_patch_anchor_joint": row["selected_patch_anchor_joint"],
                "start_frame": int(row["frame_idx"]),
                "end_frame": int(row["frame_idx"]),
                "frames": [int(row["frame_idx"])],
                "max_gap_m": gap,
            }
        else:
            current["end_frame"] = int(row["frame_idx"])
            current["frames"].append(int(row["frame_idx"]))
            current["max_gap_m"] = max(float(current["max_gap_m"]), gap)
    if current is not None:
        segments.append(current)
    return segments


def run(args: argparse.Namespace) -> dict:
    annotations = frame_map(load_json(args.annotations))
    meshes = load_mesh_archive(args.object_mesh_npz)
    contact_report = load_json(args.contact_report)
    rows = reliable_contact_rows(contact_report, args.contact_row_mode)
    obs_all = augment_observations(contact_observations(annotations, meshes, rows), rows)
    obs, rejected_obs = filter_surface_supported_observations(obs_all, args)
    transforms = pair_transforms(args.pair_factor_report)
    edges, transitions = object_edge_rows_by_contact_mode(obs, transforms)
    x0 = build_initial_state(obs)
    before_objects, before_gaps = unpack_state(x0, len(obs))
    before_edges = edge_metrics(before_objects, before_gaps, edges, float(args.fps))
    before_acc = acceleration_metrics_by_edges(before_objects, before_gaps, edges, float(args.fps))
    before_anchor_rows = anchor_rows(obs, before_objects, before_gaps)
    before_residual_norm = float(np.linalg.norm(residual_vector(x0, obs, edges, args)))
    result = least_squares(
        lambda x: residual_vector(x, obs, edges, args),
        x0,
        loss=str(args.loss),
        f_scale=float(args.loss_f_scale),
        max_nfev=int(args.max_nfev),
        ftol=float(args.ftol),
        xtol=float(args.xtol),
        gtol=float(args.gtol),
    )
    after_objects, after_gaps = unpack_state(result.x, len(obs))
    after_edges = edge_metrics(after_objects, after_gaps, edges, float(args.fps))
    after_acc = acceleration_metrics_by_edges(after_objects, after_gaps, edges, float(args.fps))
    after_anchor_rows = anchor_rows(obs, after_objects, after_gaps)
    gap_norms = np.linalg.norm(after_gaps, axis=1)
    relative_contact = [row["relative_contact_residual_m"] for row in after_edges]
    slip = [row["slip_m"] for row in after_edges]
    slip_speed = [row["slip_speed_m_s"] for row in after_edges]
    object_motion = [row["object_motion_residual_m"] for row in after_edges]
    acc_residual = [row["acceleration_consistency_residual_m_s2"] for row in after_acc]
    object_anchor_shift = [row["object_anchor_shift_m"] for row in after_anchor_rows]
    hand_anchor_shift = [row["hand_anchor_shift_m"] for row in after_anchor_rows]
    pair_p95 = [row["object_factor_inlier_p95_m"] for row in after_edges if row["object_factor_inlier_p95_m"] is not None]
    slip_p95 = float(np.percentile(slip, 95.0)) if slip else 0.0
    contact_motion_regime = "segmented_sticking" if slip_p95 <= float(args.max_sticking_slip_p95_m) else "segmented_sliding"
    segments = contact_mode_segments(obs, after_gaps, args)
    longest_segment = max((len(segment["frames"]) for segment in segments), default=0)
    pass_rows = {
        "min_contact_frames": bool(len(obs) >= int(args.min_contact_frames)),
        "min_continuous_contact_frames": bool(longest_segment >= int(args.min_continuous_contact_frames)),
        "temporal_contact_edges_available": bool(len(after_edges) > 0),
        "acceleration_evidence_available": bool(len(after_acc) > 0),
        "max_contact_gap": bool(float(np.max(gap_norms)) <= float(args.max_contact_gap_m)),
        "relative_contact_residual_p95": bool((not relative_contact) or float(np.percentile(relative_contact, 95.0)) <= float(args.max_relative_contact_residual_p95_m)),
        "slip_speed_p95": bool((not slip_speed) or float(np.percentile(slip_speed, 95.0)) <= float(args.max_sliding_speed_p95_m_s)),
        "object_factor_inlier_p95": bool((not pair_p95) or float(np.percentile(pair_p95, 95.0)) <= float(args.max_pair_factor_inlier_p95_m)),
        "acceleration_consistency_p95": bool((not acc_residual) or float(np.percentile(acc_residual, 95.0)) <= float(args.max_acceleration_consistency_p95_m_s2)),
        "object_anchor_shift_p95": bool(float(np.percentile(object_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "hand_anchor_shift_p95": bool(float(np.percentile(hand_anchor_shift, 95.0)) <= float(args.max_anchor_shift_p95_m)),
        "least_squares_converged": bool(result.success),
    }
    report = {
        "status": "accepted" if all(pass_rows.values()) else "rejected",
        "annotation_ready": bool(all(pass_rows.values())),
        "method": "solve_contact_mode_dynamics_factor_graph_v13",
        "claim_tested": "geometry-backed contact rows decompose into contact-mode-continuous segments whose measured anchors, object motion factors, contact gaps, slip, and acceleration residuals are physically consistent; patch transfers are reported as transitions instead of forced into one contact trajectory",
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
                "object_motion": len(edges),
                "relative_contact": len(edges),
                "slip_prior": len(edges),
                "acceleration_consistency": len(after_acc),
                "contact_mode_transition_report_only": len(transitions),
            },
            "weights": {
                "object_anchor": float(args.w_object_anchor),
                "hand_anchor": float(args.w_hand_anchor),
                "contact_gap": float(args.w_contact_gap),
                "object_motion": float(args.w_object_motion),
                "relative_contact": float(args.w_relative_contact),
                "slip_prior": float(args.w_slip_prior),
                "acceleration_consistency": float(args.w_acceleration_consistency),
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
            "min_continuous_contact_frames": int(args.min_continuous_contact_frames),
            "min_acceleration_rows": 1,
            "max_input_surface_distance_p95_m": float(args.max_input_surface_distance_p95_m),
            "max_contact_gap_m": float(args.max_contact_gap_m),
            "max_relative_contact_residual_p95_m": float(args.max_relative_contact_residual_p95_m),
            "max_sticking_slip_p95_m": float(args.max_sticking_slip_p95_m),
            "max_sliding_speed_p95_m_s": float(args.max_sliding_speed_p95_m_s),
            "max_pair_factor_inlier_p95_m": float(args.max_pair_factor_inlier_p95_m),
            "max_acceleration_consistency_p95_m_s2": float(args.max_acceleration_consistency_p95_m_s2),
            "max_anchor_shift_p95_m": float(args.max_anchor_shift_p95_m),
        },
        "pass": pass_rows,
        "contact_motion_regime": contact_motion_regime,
        "summary": {
            "contact_gap_m": summarize(gap_norms),
            "relative_contact_residual_m": summarize(relative_contact),
            "slip_m": summarize(slip),
            "slip_speed_m_s": summarize(slip_speed),
            "object_motion_residual_m": summarize(object_motion),
            "object_factor_inlier_p95_m": summarize(pair_p95),
            "acceleration_consistency_residual_m_s2": summarize(acc_residual),
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
            "edge_rows": before_edges,
            "acceleration_rows": before_acc,
        },
        "after": {
            "contact_gap_m": gap_norms.astype(float).tolist(),
            "object_contact_point_world_m": after_objects.astype(float).tolist(),
            "hand_contact_gap_world_m": after_gaps.astype(float).tolist(),
            "anchor_rows": after_anchor_rows,
            "edge_rows": after_edges,
            "acceleration_rows": after_acc,
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
    parser.add_argument("--w-acceleration-consistency", type=float, default=0.18)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.006)
    parser.add_argument("--max-input-surface-distance-p95-m", type=float, default=0.006)
    parser.add_argument("--max-relative-contact-residual-p95-m", type=float, default=0.002)
    parser.add_argument("--max-sticking-slip-p95-m", type=float, default=0.003)
    parser.add_argument("--max-sliding-speed-p95-m-s", type=float, default=0.45)
    parser.add_argument("--max-pair-factor-inlier-p95-m", type=float, default=0.011)
    parser.add_argument("--max-acceleration-consistency-p95-m-s2", type=float, default=1.2)
    parser.add_argument("--max-anchor-shift-p95-m", type=float, default=0.006)
    parser.add_argument("--min-contact-frames", type=int, default=3)
    parser.add_argument("--min-continuous-contact-frames", type=int, default=3)
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
