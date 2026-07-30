#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, numeric_summary, write_json


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_vec3(raw: Any) -> np.ndarray | None:
    arr = np.asarray(raw if raw is not None else [], dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr.astype(float)


def finite_array(raw: Any, shape_tail: tuple[int, ...] | None = None) -> np.ndarray | None:
    arr = np.asarray(raw if raw is not None else [], dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    if shape_tail is not None and tuple(arr.shape[-len(shape_tail):]) != shape_tail:
        return None
    return arr.astype(float)


def build_quadratic_matrix(count: int, gaps: list[int], smooth_lambda: float, observation_weight: float) -> np.ndarray:
    if count <= 0:
        raise ContractError("temporal_graph_requires_at_least_one_observation")
    mat = np.eye(count, dtype=float) * float(observation_weight)
    for index, gap in enumerate(gaps):
        lam = float(smooth_lambda) / max(1.0, float(gap))
        mat[index, index] += lam
        mat[index + 1, index + 1] += lam
        mat[index, index + 1] -= lam
        mat[index + 1, index] -= lam
    return mat


def optimize_matrix(values: np.ndarray, frame_indices: list[int], smooth_lambda: float, observation_weight: float) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ContractError(f"temporal_graph_values_must_be_TxD: {values.shape}")
    if values.shape[0] == 1:
        return values.copy(), {
            "observation_residual_norm_m": numeric_summary([0.0]),
            "smooth_step_norm_before_m": numeric_summary([]),
            "smooth_step_norm_after_m": numeric_summary([]),
        }
    gaps = [int(frame_indices[i + 1]) - int(frame_indices[i]) for i in range(len(frame_indices) - 1)]
    mat = build_quadratic_matrix(values.shape[0], gaps, smooth_lambda, observation_weight)
    rhs = values * float(observation_weight)
    optimized = np.linalg.solve(mat, rhs)
    residual = np.linalg.norm(optimized - values, axis=1)
    before = np.linalg.norm(np.diff(values, axis=0), axis=1)
    after = np.linalg.norm(np.diff(optimized, axis=0), axis=1)
    return optimized, {
        "observation_residual_norm_m": numeric_summary(residual),
        "smooth_step_norm_before_m": numeric_summary(before),
        "smooth_step_norm_after_m": numeric_summary(after),
    }


def project_joints(joints: np.ndarray, intrinsics: Any) -> list[list[float]]:
    intr = np.asarray(intrinsics if intrinsics is not None else [], dtype=float).reshape(-1)
    uv = np.full((21, 2), np.nan, dtype=float)
    if joints.shape != (21, 3) or intr.shape != (4,):
        return uv.tolist()
    fx, fy, cx, cy = intr
    valid = np.isfinite(joints).all(axis=1) & (joints[:, 2] > 1e-6)
    uv[valid, 0] = fx * joints[valid, 0] / joints[valid, 2] + cx
    uv[valid, 1] = fy * joints[valid, 1] / joints[valid, 2] + cy
    return uv.tolist()


def shift_points(raw: Any, delta: np.ndarray) -> Any:
    arr = finite_array(raw, (3,))
    if arr is None:
        return raw
    return (arr + delta.reshape(1, 3)).astype(float).tolist()


def object_centroid(obj: dict[str, Any]) -> np.ndarray | None:
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    for key in ("centroid_world_m", "centroid_camera_m"):
        vec = finite_vec3(geom.get(key))
        if vec is not None:
            return vec
    return finite_vec3(obj.get("t_camera_object_m"))


def optimize_objects(frames: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_object: dict[str, list[tuple[int, int, int, np.ndarray]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        frame_idx = int(frame.get("frame_idx", frame.get("index", frame_index)))
        for obj_index, obj in enumerate(frame.get("objects", []) if isinstance(frame.get("objects"), list) else []):
            if not isinstance(obj, dict) or not obj.get("visible", True):
                continue
            object_id = str(obj.get("object_id") or obj.get("track_id") or f"object:{obj_index}")
            center = object_centroid(obj)
            if center is not None:
                by_object[object_id].append((frame_index, obj_index, frame_idx, center))
    reports = []
    for object_id, rows in sorted(by_object.items()):
        values = np.stack([row[3] for row in rows], axis=0)
        frame_indices = [row[2] for row in rows]
        optimized, residuals = optimize_matrix(values, frame_indices, args.object_smooth_lambda, args.observation_weight)
        deltas = optimized - values
        for row, smoothed, delta in zip(rows, optimized, deltas):
            frame_index, obj_index, _, _ = row
            obj = frames[frame_index]["objects"][obj_index]
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            obj["t_camera_object_m"] = smoothed.astype(float).tolist()
            obj["object_pose_source"] = "v20_temporal_observation_graph_smoothed_visible_surface_centroid"
            recon = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
            recon["translation_camera_m"] = smoothed.astype(float).tolist()
            recon["translation_world_from_object_m"] = smoothed.astype(float).tolist()
            recon["state"] = "temporally_smoothed_visible_surface_only_uncertain_pose"
            recon["claim_scope"] = "temporal graph smooths visible surface observations; this remains incomplete geometry, not a complete mesh pose"
            obj["reconstructed_geometry_pose"] = recon
            geom["centroid_world_m"] = smoothed.astype(float).tolist()
            geom["centroid_camera_m"] = smoothed.astype(float).tolist()
            for key in ("points_world_sample_m", "points_camera_sample_m", "camera_vertices_sample_m", "world_vertices_sample_m"):
                geom[key] = shift_points(geom.get(key), delta)
            geom["temporal_graph_delta_m"] = delta.astype(float).tolist()
            geom["source"] = f"{geom.get('source', 'visible_surface')}_plus_temporal_observation_graph"
            geom["claim_scope"] = "visible surface from SAM2/depth with temporal smoothing; not complete object mesh or rigid pose"
            obj["visible_geometry_candidate"] = geom
        reports.append({
            "object_id": object_id,
            "observation_count": len(rows),
            "optimization_status": "optimized_visible_surface_centroid_temporal_graph",
            **residuals,
        })
    return reports


def hand_key(hand: dict[str, Any]) -> str:
    return str(hand.get("hand_track_id") or hand.get("hand_id") or hand.get("hand_side") or hand.get("side") or "hand:unknown")


def optimize_hands(frames: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_hand: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        frame_idx = int(frame.get("frame_idx", frame.get("index", frame_index)))
        for hand_index, hand in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = finite_array(metric.get("joints_current_v18_camera_m"), (3,))
            verts = finite_array(metric.get("vertices_current_v18_world_m") or metric.get("vertices_world_sample_m"), (3,))
            if joints is None or joints.shape != (21, 3):
                continue
            by_hand[hand_key(hand)].append({
                "frame_index": frame_index,
                "frame_idx": frame_idx,
                "hand_index": hand_index,
                "joints": joints,
                "vertices": verts,
            })
    reports = []
    for key, rows in sorted(by_hand.items()):
        rows = sorted(rows, key=lambda row: row["frame_idx"])
        frame_indices = [int(row["frame_idx"]) for row in rows]
        joint_values = np.stack([row["joints"].reshape(-1) for row in rows], axis=0)
        joint_opt, joint_residuals = optimize_matrix(joint_values, frame_indices, args.hand_smooth_lambda, args.observation_weight)
        vertex_opt = None
        vertex_residuals = None
        if all(row["vertices"] is not None for row in rows):
            vertex_values = np.stack([row["vertices"].reshape(-1) for row in rows], axis=0)
            vertex_opt, vertex_residuals = optimize_matrix(vertex_values, frame_indices, args.hand_smooth_lambda, args.observation_weight)
        for row_index, row in enumerate(rows):
            hand = frames[row["frame_index"]]["hands"][row["hand_index"]]
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = joint_opt[row_index].reshape(21, 3)
            metric["joints_current_v18_camera_m"] = joints.astype(float).tolist()
            metric["joints_current_v18_world_m"] = joints.astype(float).tolist()
            metric["joints_2d_px"] = project_joints(joints, metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy"))
            if vertex_opt is not None and row["vertices"] is not None:
                verts = vertex_opt[row_index].reshape(row["vertices"].shape)
                for name in ("vertices_camera_sample_m", "vertices_world_sample_m", "vertices_current_v18_world_m", "vertices_current_v18_camera_m"):
                    if metric.get(name) is not None:
                        metric[name] = verts.astype(float).tolist()
            metric["temporal_graph_status"] = "optimized_mano_observation_sequence"
            metric["support_state"] = "prediction_side_hawor_observation_temporally_smoothed_uncertain"
            hand["metric_mano_state"] = metric
            hand["hand_geometry_source"] = "prediction_side_hawor_mano_npz_plus_temporal_observation_graph"
        report = {
            "hand_track_id": key,
            "observation_count": len(rows),
            "optimization_status": "optimized_mano_joint_and_surface_temporal_graph",
            "joint_observation_residual_norm_m": joint_residuals["observation_residual_norm_m"],
            "joint_smooth_step_norm_before_m": joint_residuals["smooth_step_norm_before_m"],
            "joint_smooth_step_norm_after_m": joint_residuals["smooth_step_norm_after_m"],
        }
        if vertex_residuals is not None:
            report["surface_observation_residual_norm_m"] = vertex_residuals["observation_residual_norm_m"]
            report["surface_smooth_step_norm_before_m"] = vertex_residuals["smooth_step_norm_before_m"]
            report["surface_smooth_step_norm_after_m"] = vertex_residuals["smooth_step_norm_after_m"]
        reports.append(report)
    return reports


def annotate_frames(frames: list[dict[str, Any]], object_reports: list[dict[str, Any]], hand_reports: list[dict[str, Any]]) -> None:
    object_status = {row["object_id"]: row["optimization_status"] for row in object_reports}
    hand_status = {row["hand_track_id"]: row["optimization_status"] for row in hand_reports}
    for frame in frames:
        variables = {
            "object_se3": [],
            "hand_wrist": [],
            "contact_switch": [],
            "occlusion_owner": [],
        }
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            object_id = str(obj.get("object_id") or obj.get("track_id") or "object:unknown")
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            variables["object_se3"].append({
                "variable_id": f"object_surface_center::{object_id}",
                "translation_world_from_object_m": geom.get("centroid_world_m"),
                "rotation_world_from_object_rotvec": [0.0, 0.0, 0.0],
                "source": geom.get("source"),
                "optimization_status": object_status.get(object_id, "not_observed_by_temporal_graph"),
            })
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            key = hand_key(hand)
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = finite_array(metric.get("joints_current_v18_world_m"), (3,))
            wrist = joints[0].astype(float).tolist() if joints is not None and joints.shape == (21, 3) else None
            variables["hand_wrist"].append({
                "variable_id": f"hand_wrist::{key}",
                "translation_world_m": wrist,
                "source": metric.get("source"),
                "optimization_status": hand_status.get(key, "not_observed_by_temporal_graph"),
            })
        frame["factor_graph_solution"] = {
            "variables": variables,
            "solution": {
                "status": "v20_temporal_observation_graph_optimized",
                "claim_scope": "Quadratic temporal smoothing over prediction-side visible surface and MANO observations; no complete object mesh or contact ownership is inferred here.",
            },
        }


def solve(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    ensure_no_gt_in_prediction(annotations, "temporal_graph_input_annotations")
    frames = annotations.get("frames") if isinstance(annotations, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ContractError("temporal_graph_annotations_have_no_frames")
    object_reports = optimize_objects(frames, args)
    hand_reports = optimize_hands(frames, args)
    annotate_frames(frames, object_reports, hand_reports)
    annotations["schema"] = "v20_infer_temporal_optimized_annotations.v0"
    annotations["optimization"] = {
        "method": "solve_v20_infer_temporal_observation_graph",
        "object_smooth_lambda": float(args.object_smooth_lambda),
        "hand_smooth_lambda": float(args.hand_smooth_lambda),
        "observation_weight": float(args.observation_weight),
        "claim_scope": "Renderable V20 infer annotations after temporal observation graph smoothing of non-reference measurements.",
    }
    report = {
        "schema": "v20_infer_temporal_observation_graph_report.v0",
        "status": "ok",
        "method": "solve_v20_infer_temporal_observation_graph",
        "annotation_ready": True,
        "output_annotations": str(args.output_annotations),
        "graph_frame_count": len(frames),
        "object_reports": object_reports,
        "hand_reports": hand_reports,
        "object_count": len(object_reports),
        "hand_track_count": len(hand_reports),
        "claim_scope": "Temporal graph result is the optimized renderable state for available V20 infer observations; object geometry remains visible-surface-only unless a completed mesh candidate is connected.",
    }
    ensure_no_gt_in_prediction(annotations, "temporal_graph_output_annotations")
    ensure_no_gt_in_prediction(report, "temporal_graph_report")
    write_json(args.output_annotations, annotations)
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a lightweight V20 infer temporal observation graph and write optimized renderable annotations.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--object-smooth-lambda", type=float, default=3.0)
    parser.add_argument("--hand-smooth-lambda", type=float, default=1.5)
    parser.add_argument("--observation-weight", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    solve(parse_args())
