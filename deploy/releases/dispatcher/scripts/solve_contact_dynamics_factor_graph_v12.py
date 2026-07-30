#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from diagnose_contact_kinematics_v3 import frame_map, hand_vertices_camera, source_to_world
from diagnose_mesh_surface_contact_v3 import load_mesh_archive
from fit_cotracker_factor_graph_v6 import report_pair_rows


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def selected_vertex_ids(row: dict) -> np.ndarray:
    source = str(row.get("selected_patch_source"))
    if source == "anatomical_patch":
        key = "anatomical_patch_vertex_ids"
    elif source == "best_patch":
        key = "best_patch_vertex_ids"
    else:
        raise RuntimeError(f"unsupported selected patch source {source!r} for frame {row.get('frame_idx')}")
    ids = np.asarray(row.get(key, []), dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise RuntimeError(f"contact row for frame {row.get('frame_idx')} has no selected patch vertex ids")
    return ids


def reliable_contact_rows(report: dict, mode: str) -> list[dict]:
    if mode == "geometry_backed_temporal":
        rows = [row for row in report.get("rows_detail", []) if bool(row.get("geometry_backed_temporal_contact", False))]
    elif mode == "reliable_temporal":
        rows = [row for row in report.get("rows_detail", []) if bool(row.get("reliable_for_contact", False))]
    else:
        raise RuntimeError(f"unsupported contact row mode: {mode}")
    if not rows:
        raise RuntimeError(f"no contact rows selected by mode {mode}")
    return sorted(rows, key=lambda row: (int(row["frame_idx"]), int(row["hand_idx"]), str(row.get("selected_patch_region"))))


def pair_transforms(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]]:
    report = load_json(path)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]] = {}
    for row in report_pair_rows(report, path):
        source = int(row["source_frame"])
        target = int(row["target_frame"])
        if not bool(row.get("rigid_factor_ready", False)):
            continue
        rot = np.asarray(row["rotation"], dtype=np.float64)
        trans = np.asarray(row["translation_m"], dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,):
            raise RuntimeError(f"invalid object motion factor for {source}->{target}")
        out[(source, target)] = (rot, trans, row)
    return out


def contact_observations(
    annotations: dict[int, dict],
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    rows: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        frame = int(row["frame_idx"])
        hand_idx = int(row["hand_idx"])
        if frame not in annotations:
            raise RuntimeError(f"annotations lack frame {frame}")
        if frame not in meshes:
            raise RuntimeError(f"mesh archive lacks frame {frame}")
        ann = annotations[frame]
        hands = ann.get("hands", [])
        if hand_idx < 0 or hand_idx >= len(hands):
            raise RuntimeError(f"frame {frame} lacks hand index {hand_idx}")
        T_world_camera = np.asarray(ann["camera"]["T_world_camera_metric"], dtype=np.float64)
        hand_camera = hand_vertices_camera(hands[hand_idx], T_world_camera)
        patch_ids = selected_vertex_ids(row)
        if int(np.max(patch_ids)) >= len(hand_camera):
            raise RuntimeError(f"frame {frame} selected patch index exceeds hand vertices")
        patch_world = source_to_world(hand_camera[patch_ids], T_world_camera)
        hand_center_world = np.median(patch_world, axis=0)
        object_vertices, _faces = meshes[frame]
        object_vertices = np.asarray(object_vertices, dtype=np.float64)
        distances, indices = cKDTree(object_vertices).query(patch_world, k=1)
        object_patch_world = object_vertices[np.asarray(indices, dtype=np.int64)]
        object_center_world = np.median(object_patch_world, axis=0)
        out.append(
            {
                "frame_idx": frame,
                "hand_idx": hand_idx,
                "side": str(row.get("side")),
                "selected_patch_source": str(row.get("selected_patch_source")),
                "selected_patch_region": str(row.get("selected_patch_region")),
                "patch_vertex_ids": patch_ids.astype(int).tolist(),
                "hand_center_world_m": hand_center_world.astype(float).tolist(),
                "object_center_world_m": object_center_world.astype(float).tolist(),
                "surface_distance_m": {
                    "median": float(np.median(distances)),
                    "p95": float(np.percentile(distances, 95.0)),
                    "max": float(np.max(distances)),
                },
                "contact_row_metrics": {
                    "best_patch_distance_p95_m": row.get("best_patch_distance_p95_m"),
                    "best_patch_signed_gap_p95_abs_m": row.get("best_patch_signed_gap_p95_abs_m"),
                    "best_patch_penetration_fraction_010m": row.get("best_patch_penetration_fraction_010m"),
                    "median_joint_reprojection_px": row.get("median_joint_reprojection_px"),
                    "mano_minus_metric_depth_median_m": row.get("mano_minus_metric_depth_median_m"),
                },
            }
        )
    return out


def frame_index_map(obs: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, row in enumerate(obs):
        frame = int(row["frame_idx"])
        if frame in out:
            raise RuntimeError(f"duplicate selected contact row for frame {frame}")
        out[frame] = i
    return out


def object_edge_rows(obs: list[dict], transforms: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]]) -> list[dict]:
    by_frame = frame_index_map(obs)
    frames = sorted(by_frame)
    edges = []
    for source, target in zip(frames[:-1], frames[1:]):
        if (source, target) not in transforms:
            raise RuntimeError(f"missing object motion factor for selected contact edge {source}->{target}")
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
            }
        )
    return edges


def build_initial_state(obs: list[dict]) -> np.ndarray:
    state = []
    for row in obs:
        object_point = np.asarray(row["object_center_world_m"], dtype=np.float64)
        gap = np.asarray(row["hand_center_world_m"], dtype=np.float64) - object_point
        state.extend(object_point.tolist())
        state.extend(gap.tolist())
    return np.asarray(state, dtype=np.float64)


def unpack_state(x: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64).reshape(count, 6)
    return arr[:, :3], arr[:, 3:]


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
    if len(edges) >= 2:
        by_source = {int(edge["source_frame"]): edge for edge in edges}
        frames = [int(row["frame_idx"]) for row in obs]
        for prev_frame, mid_frame, next_frame in zip(frames[:-2], frames[1:-1], frames[2:]):
            prev_edge = by_source[prev_frame]
            cur_edge = by_source[mid_frame]
            dt_prev = (mid_frame - prev_frame) / float(args.fps)
            dt_cur = (next_frame - mid_frame) / float(args.fps)
            if dt_prev <= 0.0 or dt_cur <= 0.0:
                raise RuntimeError("frames must be strictly increasing")
            ip = int(prev_edge["source_index"])
            im = int(prev_edge["target_index"])
            inext = int(cur_edge["target_index"])
            object_vel_prev = (object_points[im] - object_points[ip]) / dt_prev
            object_vel_cur = (object_points[inext] - object_points[im]) / dt_cur
            hand_vel_prev = ((object_points[im] + gaps[im]) - (object_points[ip] + gaps[ip])) / dt_prev
            hand_vel_cur = ((object_points[inext] + gaps[inext]) - (object_points[im] + gaps[im])) / dt_cur
            object_acc = (object_vel_cur - object_vel_prev) / ((dt_prev + dt_cur) * 0.5)
            hand_acc = (hand_vel_cur - hand_vel_prev) / ((dt_prev + dt_cur) * 0.5)
            residuals.append(float(args.w_acceleration_consistency) * (hand_acc - object_acc))
    return np.concatenate([np.ravel(r) for r in residuals]).astype(np.float64)


def edge_metrics(object_points: np.ndarray, gaps: np.ndarray, edges: list[dict], fps: float) -> list[dict]:
    rows = []
    for edge in edges:
        s = int(edge["source_index"])
        t = int(edge["target_index"])
        source = int(edge["source_frame"])
        target = int(edge["target_frame"])
        dt = (target - source) / float(fps)
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        object_motion_residual = object_points[t] - (object_points[s] @ rot + trans)
        hand_motion_residual = (object_points[t] + gaps[t]) - ((object_points[s] + gaps[s]) @ rot + trans)
        relative_contact_residual = hand_motion_residual - object_motion_residual
        object_step = object_points[t] - object_points[s]
        hand_step = (object_points[t] + gaps[t]) - (object_points[s] + gaps[s])
        rows.append(
            {
                "source_frame": source,
                "target_frame": target,
                "object_motion_residual_m": float(np.linalg.norm(object_motion_residual)),
                "slip_m": float(np.linalg.norm(object_motion_residual)),
                "slip_speed_m_s": float(np.linalg.norm(object_motion_residual) / dt),
                "transported_hand_residual_m": float(np.linalg.norm(hand_motion_residual)),
                "relative_contact_residual_m": float(np.linalg.norm(relative_contact_residual)),
                "relative_step_m": float(np.linalg.norm(hand_step - object_step)),
                "object_speed_m_s": float(np.linalg.norm(object_step) / dt),
                "hand_speed_m_s": float(np.linalg.norm(hand_step) / dt),
                "object_factor_inlier_p95_m": edge["pair_row"].get("inlier_residual_m", {}).get("p95"),
                "object_factor_ready": bool(edge["pair_row"].get("rigid_factor_ready", False)),
            }
        )
    return rows


def acceleration_metrics(object_points: np.ndarray, gaps: np.ndarray, obs: list[dict], fps: float) -> list[dict]:
    frames = [int(row["frame_idx"]) for row in obs]
    rows = []
    if len(frames) < 3:
        return rows
    for i, (prev_frame, mid_frame, next_frame) in enumerate(zip(frames[:-2], frames[1:-1], frames[2:]), start=1):
        dt_prev = (mid_frame - prev_frame) / float(fps)
        dt_cur = (next_frame - mid_frame) / float(fps)
        object_vel_prev = (object_points[i] - object_points[i - 1]) / dt_prev
        object_vel_cur = (object_points[i + 1] - object_points[i]) / dt_cur
        hand_points = object_points + gaps
        hand_vel_prev = (hand_points[i] - hand_points[i - 1]) / dt_prev
        hand_vel_cur = (hand_points[i + 1] - hand_points[i]) / dt_cur
        denom = (dt_prev + dt_cur) * 0.5
        object_acc = (object_vel_cur - object_vel_prev) / denom
        hand_acc = (hand_vel_cur - hand_vel_prev) / denom
        rows.append(
            {
                "center_frame": mid_frame,
                "object_acceleration_m_s2": float(np.linalg.norm(object_acc)),
                "hand_acceleration_m_s2": float(np.linalg.norm(hand_acc)),
                "acceleration_consistency_residual_m_s2": float(np.linalg.norm(hand_acc - object_acc)),
            }
        )
    return rows


def contact_mode_segments(obs: list[dict], object_points: np.ndarray, gaps: np.ndarray, args: argparse.Namespace) -> list[dict]:
    segments = []
    current: dict[str, Any] | None = None
    for i, row in enumerate(obs):
        label = str(row["selected_patch_region"])
        gap = float(np.linalg.norm(gaps[i]))
        if current is None or current["selected_patch_region"] != label or gap > float(args.max_contact_gap_m):
            if current is not None:
                segments.append(current)
            current = {
                "selected_patch_region": label,
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


def anchor_rows(obs: list[dict], object_points: np.ndarray, gaps: np.ndarray) -> list[dict]:
    rows = []
    for i, row in enumerate(obs):
        object_anchor = np.asarray(row["object_center_world_m"], dtype=np.float64)
        hand_anchor = np.asarray(row["hand_center_world_m"], dtype=np.float64)
        solved_hand = object_points[i] + gaps[i]
        rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "object_anchor_shift_m": float(np.linalg.norm(object_points[i] - object_anchor)),
                "hand_anchor_shift_m": float(np.linalg.norm(solved_hand - hand_anchor)),
                "contact_gap_m": float(np.linalg.norm(gaps[i])),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    annotations = frame_map(load_json(args.annotations))
    meshes = load_mesh_archive(args.object_mesh_npz)
    contact_report = load_json(args.contact_report)
    rows = reliable_contact_rows(contact_report, args.contact_row_mode)
    obs = contact_observations(annotations, meshes, rows)
    transforms = pair_transforms(args.pair_factor_report)
    edges = object_edge_rows(obs, transforms)
    x0 = build_initial_state(obs)
    before_objects, before_gaps = unpack_state(x0, len(obs))
    before_edges = edge_metrics(before_objects, before_gaps, edges, float(args.fps))
    before_acc = acceleration_metrics(before_objects, before_gaps, obs, float(args.fps))
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
    after_acc = acceleration_metrics(after_objects, after_gaps, obs, float(args.fps))
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
    contact_motion_regime = "sticking" if slip_p95 <= float(args.max_sticking_slip_p95_m) else "sliding"
    pass_rows = {
        "min_contact_frames": bool(len(obs) >= int(args.min_contact_frames)),
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
        "method": "solve_contact_dynamics_factor_graph_v12",
        "claim_tested": "geometry-backed contact rows, object motion factors, MANO patch centers, and object surface points admit a short-window contact-dynamics explanation with measured contact anchors preserved",
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
            "min_acceleration_rows": 1,
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
        "contact_mode_segments": contact_mode_segments(obs, after_objects, after_gaps, args),
        "observations": obs,
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
    parser.add_argument("--max-relative-contact-residual-p95-m", type=float, default=0.002)
    parser.add_argument("--max-sticking-slip-p95-m", type=float, default=0.003)
    parser.add_argument("--max-sliding-speed-p95-m-s", type=float, default=0.45)
    parser.add_argument("--max-pair-factor-inlier-p95-m", type=float, default=0.011)
    parser.add_argument("--max-acceleration-consistency-p95-m-s2", type=float, default=1.2)
    parser.add_argument("--max-anchor-shift-p95-m", type=float, default=0.006)
    parser.add_argument("--min-contact-frames", type=int, default=3)
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
