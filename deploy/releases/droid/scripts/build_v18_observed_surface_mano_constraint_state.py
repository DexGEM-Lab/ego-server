#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Separate observed-surface MANO constraints from quarantined hidden volume.

The articulated V18 interval state already tests whether MANO pose deltas can
reduce object/hand penetration. This script asks the next physical question:
when penetration remains, is it against object surface that is supported by the
current metric depth image, or only against hidden/free-space-conflicted parts of
the compact-rigid completion?

It does not accept coordinate corrections. It remeasures full 778-vertex current
and articulated MANO surfaces and labels the residual's object-surface provenance
so observed geometry can constrain/falsify H_{t,h} while invalid hidden volume is
quarantined.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources, project_points  # noqa: E402
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    bridge_vertices_and_joints,
    load_source_arrays,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    rotvec_to_matrix,
    similarity_from_to,
    source_npz_for_hand,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    inverse_object,
    load_json,
    load_mesh,
    numeric_summary,
    pose_map,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path("/data2/ego_annotation_outputs/v18_task5_observed_surface_mano_constraints_v1")

# Per-object-vertex depth provenance classes.
VERTEX_OUT_OR_INVALID = 0
VERTEX_OBSERVED_SUPPORTED = 1
VERTEX_FREE_SPACE_CONFLICT = 2
VERTEX_BEHIND_OBSERVED = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True)
    parser.add_argument("--articulated-mano-state", type=Path, required=True)
    parser.add_argument("--hidden-volume-validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path, default=None)
    parser.add_argument("--wilor-mano-left", type=Path, default=None)
    parser.add_argument(
        "--hawor-left-shapedirs-x-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply HaWoR's MANO_LEFT shapedirs[:,0,:] *= -1 convention before left replay.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-margin-m", type=float, default=0.015)
    parser.add_argument("--free-space-margin-m", type=float, default=0.025)
    parser.add_argument("--penetration-epsilon-m", type=float, default=1.0e-5)
    parser.add_argument("--accepted-observed-residual-m", type=float, default=0.0015)
    parser.add_argument("--visible-shift-limit-px", type=float, default=8.0)
    parser.add_argument("--depth-shift-limit-m", type=float, default=0.025)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.35)
    return parser.parse_args()


def temporal_state_map(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload.get("per_frame_states", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("frame_idx") is not None and row.get("hand_side") is not None:
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out, payload


def hidden_state_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("frame_rows", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            out[int(row["frame_idx"])] = row
    return out


def classify_object_vertices_against_depth(
    *,
    frame: dict[str, Any],
    vertices_object: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    support_margin_m: float,
    free_space_margin_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    classes = np.full((len(vertices_object),), VERTEX_OUT_OR_INVALID, dtype=np.uint8)
    frame_idx = int(frame["frame_idx"])
    if depth_row is None:
        return classes, {
            "frame_idx": frame_idx,
            "state": "missing_depth",
            "projected_count": 0,
            "finite_depth_count": 0,
            "vertex_class_counts": {"out_or_invalid": int(len(vertices_object))},
        }
    r_obj, t_obj = pose
    vertices_world = vertices_object @ r_obj.T + t_obj[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    vertices_camera = (vertices_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    height, width = depth.shape
    u, v, valid = project_points(vertices_camera, np.asarray(depth_row["intrinsics"], dtype=float), width, height)
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return classes, {
            "frame_idx": frame_idx,
            "state": "no_projected_object_vertices",
            "projected_count": 0,
            "finite_depth_count": 0,
            "vertex_class_counts": {"out_or_invalid": int(len(vertices_object))},
        }
    z_mesh = vertices_camera[valid_indices, 2]
    z_obs = depth[v[valid_indices], u[valid_indices]].astype(float)
    finite = np.isfinite(z_obs) & (z_obs > 0.0)
    finite_indices = valid_indices[finite]
    residual = z_mesh[finite] - z_obs[finite]
    supported = np.abs(residual) <= float(support_margin_m)
    free_space = residual < -float(free_space_margin_m)
    behind = residual > float(support_margin_m)
    classes[finite_indices[supported]] = VERTEX_OBSERVED_SUPPORTED
    classes[finite_indices[free_space]] = VERTEX_FREE_SPACE_CONFLICT
    classes[finite_indices[behind]] = VERTEX_BEHIND_OBSERVED
    counter = Counter(int(x) for x in classes.tolist())
    class_counts = {
        "out_or_invalid": int(counter.get(VERTEX_OUT_OR_INVALID, 0)),
        "observed_supported": int(counter.get(VERTEX_OBSERVED_SUPPORTED, 0)),
        "free_space_conflict": int(counter.get(VERTEX_FREE_SPACE_CONFLICT, 0)),
        "behind_observed": int(counter.get(VERTEX_BEHIND_OBSERVED, 0)),
    }
    finite_count = int(finite_indices.size)
    support_fraction = class_counts["observed_supported"] / finite_count if finite_count else None
    free_fraction = class_counts["free_space_conflict"] / finite_count if finite_count else None
    state = "object_surface_depth_classified"
    return classes, {
        "frame_idx": frame_idx,
        "state": state,
        "depth_source": str(depth_row["source"]),
        "projected_count": int(valid_indices.size),
        "finite_depth_count": finite_count,
        "vertex_class_counts": class_counts,
        "observed_support_fraction_finite": support_fraction,
        "free_space_conflict_fraction_finite": free_fraction,
        "depth_residual_m": numeric_summary(residual),
    }


def face_provenance(vertex_classes: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    tri = vertex_classes[np.asarray(faces, dtype=np.int64)]
    supported_count = np.count_nonzero(tri == VERTEX_OBSERVED_SUPPORTED, axis=1)
    free_count = np.count_nonzero(tri == VERTEX_FREE_SPACE_CONFLICT, axis=1)
    behind_count = np.count_nonzero(tri == VERTEX_BEHIND_OBSERVED, axis=1)
    invalid_count = np.count_nonzero(tri == VERTEX_OUT_OR_INVALID, axis=1)
    observed_strict = (supported_count >= 2) & (free_count == 0)
    observed_any = (supported_count >= 1) & (free_count == 0)
    free_any = free_count >= 1
    return {
        "observed_supported_strict": observed_strict,
        "observed_supported_any": observed_any,
        "free_space_conflict_any": free_any,
        "supported_count": supported_count,
        "free_count": free_count,
        "behind_count": behind_count,
        "invalid_count": invalid_count,
    }


def measure_hand_against_object(
    *,
    vertices_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    scene: Any,
    face_prov: dict[str, np.ndarray],
    face_count: int,
    penetration_epsilon_m: float,
) -> dict[str, Any]:
    r_obj, t_obj = pose
    vertices_object = inverse_object(vertices_world, r_obj, t_obj)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(penetration_epsilon_m))[0]
    depths = signed[penetrating]
    if penetrating.size:
        closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
        primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
        valid_primitive = (primitive_ids >= 0) & (primitive_ids < int(face_count))
    else:
        primitive_ids = np.zeros((0,), dtype=np.int64)
        valid_primitive = np.zeros((0,), dtype=bool)
    observed_strict_mask = np.zeros_like(valid_primitive, dtype=bool)
    observed_any_mask = np.zeros_like(valid_primitive, dtype=bool)
    free_mask = np.zeros_like(valid_primitive, dtype=bool)
    hidden_mask = np.zeros_like(valid_primitive, dtype=bool)
    if penetrating.size:
        valid_ids = primitive_ids[valid_primitive]
        observed_strict_mask[valid_primitive] = face_prov["observed_supported_strict"][valid_ids]
        observed_any_mask[valid_primitive] = face_prov["observed_supported_any"][valid_ids]
        free_mask[valid_primitive] = face_prov["free_space_conflict_any"][valid_ids]
        hidden_mask = ~(observed_strict_mask | free_mask)
    def masked_summary(mask: np.ndarray) -> dict[str, Any]:
        return numeric_summary(depths[mask]) if depths.size else numeric_summary(np.asarray([], dtype=float))
    max_depth = float(np.max(depths)) if depths.size else 0.0
    return {
        "hand_vertex_count": int(len(vertices_world)),
        "penetrating_vertex_count": int(penetrating.size),
        "penetration_depth_m": numeric_summary(depths),
        "max_penetration_m": max_depth,
        "closest_face_provenance_counts": {
            "observed_supported_strict": int(np.count_nonzero(observed_strict_mask)),
            "observed_supported_any": int(np.count_nonzero(observed_any_mask)),
            "free_space_conflict_any": int(np.count_nonzero(free_mask)),
            "hidden_or_unvalidated": int(np.count_nonzero(hidden_mask)),
            "invalid_closest_face": int(np.count_nonzero(~valid_primitive)) if penetrating.size else 0,
        },
        "observed_supported_strict_penetration_m": masked_summary(observed_strict_mask),
        "observed_supported_any_penetration_m": masked_summary(observed_any_mask),
        "free_space_conflict_penetration_m": masked_summary(free_mask),
        "hidden_or_unvalidated_penetration_m": masked_summary(hidden_mask),
    }


def list_to_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if shape is not None and arr.shape != shape:
        return None
    return arr


def make_candidate_vertices(
    *,
    models_by_side: dict[str, Any],
    temporal_row: dict[str, Any],
    hand: dict[str, Any],
    current_vertices: np.ndarray,
    bridge_cache: dict[Path, Any],
    source_cache: dict[Path, Any],
    device: torch.device,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    side = str(hand.get("hand_side"))
    model = models_by_side.get(side)
    if model is None:
        return None, {"state": f"no_candidate_{side}_mano_model_unavailable"}
    delta = list_to_float_array(temporal_row.get("optimized_hand_pose_delta_axis_angle_rad"), (45,))
    if delta is None:
        return None, {"state": "no_candidate_pose_delta_in_temporal_state"}
    source_info = source_npz_for_hand(hand)
    if source_info is None:
        return None, {"state": "missing_source_npz_for_candidate_replay"}
    source_path, source_frame = source_info
    source = load_source_arrays(source_cache, source_path)
    required = [
        f"{side}_vertices_world_m",
        f"{side}_root_orient_axis_angle",
        f"{side}_hand_pose_axis_angle",
        f"{side}_betas",
        f"{side}_trans_world_m",
    ]
    missing = [key for key in required if key not in source]
    if missing:
        return None, {"state": "source_npz_missing_arrays", "missing": missing}
    raw_vertices = np.asarray(source[f"{side}_vertices_world_m"][source_frame], dtype=float)
    scale, rot, _trans, sim_err = similarity_from_to(raw_vertices, current_vertices)
    root = torch.tensor(np.asarray(source[f"{side}_root_orient_axis_angle"][source_frame], dtype=float).reshape(1, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.asarray(source[f"{side}_hand_pose_axis_angle"][source_frame], dtype=float).reshape(1, 15, 3), dtype=torch.float32, device=device)
    pose_delta = torch.tensor(delta.reshape(1, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.asarray(source[f"{side}_betas"][source_frame], dtype=float).reshape(1, 10), dtype=torch.float32, device=device)
    trans = torch.tensor(np.asarray(source[f"{side}_trans_world_m"][source_frame], dtype=float).reshape(1, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        base_root = rotvec_to_matrix(root)
        base_pose = rotvec_to_matrix(pose)
        base_out = model(global_orient=base_root, hand_pose=base_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        new_pose = rotvec_to_matrix(pose_delta) @ base_pose
        hyp_out = model(global_orient=base_root, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_delta = (hyp_out.vertices - base_out.vertices).detach().cpu().numpy().astype(float)[0]
    mapped_delta = float(scale) * (raw_delta @ rot.T)
    candidate = current_vertices + mapped_delta
    return candidate.astype(float), {
        "state": "candidate_reconstructed_full_778_vertices",
        "source_hawor_npz": str(source_path),
        "source_frame_index": int(source_frame),
        "raw_to_current_similarity_error_m": {
            "median": float(np.median(sim_err)),
            "p95": float(np.percentile(sim_err, 95)),
            "scope": "used only to map articulated MANO displacement onto current V18 bridge zero state",
        },
    }


def scalar_from_summary(summary: Any, key: str = "max") -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    return float(value) if isinstance(value, (float, int)) else None


def classify_constraint_state(
    *,
    candidate_measure: dict[str, Any] | None,
    current_measure: dict[str, Any],
    candidate_info: dict[str, Any],
    temporal_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    if candidate_measure is None:
        return "observed_surface_current_only_no_coordinate_candidate", [str(candidate_info.get("state", "no_candidate"))]
    blockers: list[str] = []
    temporal_state = str(temporal_row.get("temporal_mano_state", "missing_temporal_state")) if temporal_row else "missing_temporal_state"
    visible_max = scalar_from_summary(temporal_row.get("visible_joint_shift_px") if temporal_row else None)
    depth_max = scalar_from_summary(temporal_row.get("joint_camera_depth_shift_m") if temporal_row else None)
    pose_max = scalar_from_summary(temporal_row.get("pose_delta_norm_rad") if temporal_row else None)
    if visible_max is None or visible_max > float(args.visible_shift_limit_px):
        blockers.append("visible_2d_shift_not_compatible")
    if depth_max is None or depth_max > float(args.depth_shift_limit_m):
        blockers.append("joint_depth_shift_not_compatible")
    if pose_max is None or pose_max > float(args.max_pose_delta_rad):
        blockers.append("pose_delta_bound_not_compatible")
    obs_summary = candidate_measure["observed_supported_strict_penetration_m"]
    obs_max = scalar_from_summary(obs_summary)
    full_max = float(candidate_measure.get("max_penetration_m", 0.0))
    obs_count = int(obs_summary.get("count", 0)) if isinstance(obs_summary, dict) else 0
    if obs_max is not None and obs_max > float(args.accepted_observed_residual_m):
        blockers.append("candidate_penetrates_observed_supported_surface")
    if blockers:
        return "candidate_blocked_by_observed_surface_or_visibility", blockers
    if full_max <= float(args.accepted_observed_residual_m):
        return "candidate_full_signed_clear_but_object_volume_still_unaccepted", ["hidden_volume_not_accepted_for_coordinate_level_proof"]
    if obs_count == 0:
        return "candidate_residual_only_on_hidden_or_unvalidated_surface", ["no_observed_supported_penetrating_faces", "hidden_volume_quarantined"]
    return "candidate_observed_surface_compatible_hidden_volume_residual_quarantined", ["hidden_or_free_space_volume_still_has_signed_residual"]


def main() -> None:
    args = parse_args()
    patch_legacy_mano_loader()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    mesh = load_mesh(args.completed_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    depth_by_frame = load_depth_sources(args.depth_npz)
    temporal_by_key, temporal_payload = temporal_state_map(args.articulated_mano_state)
    hidden_by_frame = hidden_state_map(args.hidden_volume_validation)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_object.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))

    mano_right_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_right_path.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT model: {mano_right_path}")
    mano_cls = load_wilor_mano_class(args.wilor_root)
    device = torch.device(args.device)
    models_by_side: dict[str, Any] = {
        "right": mano_cls(model_path=str(mano_right_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    }
    model_paths: dict[str, str] = {"right": str(mano_right_path)}
    left_model_status = "not_provided"
    if args.wilor_mano_left is not None:
        if not args.wilor_mano_left.exists():
            left_model_status = "missing_mano_left_path"
        else:
            left_model = mano_cls(model_path=str(args.wilor_mano_left), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
            if bool(args.hawor_left_shapedirs_x_fix):
                with torch.no_grad():
                    left_model.shapedirs[:, 0, :] *= -1
            models_by_side["left"] = left_model
            model_paths["left"] = str(args.wilor_mano_left)
            left_model_status = "loaded_with_hawor_shapedirs_x_fix" if bool(args.hawor_left_shapedirs_x_fix) else "loaded_without_hawor_shapedirs_x_fix"
    for model in models_by_side.values():
        model.eval()

    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    object_depth_rows: dict[int, dict[str, Any]] = {}
    object_face_prov: dict[int, dict[str, np.ndarray]] = {}
    per_frame_states: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        pose = poses.get(frame_idx)
        if pose is None:
            continue
        vertex_classes, depth_row = classify_object_vertices_against_depth(
            frame=frame,
            vertices_object=vertices_object,
            pose=pose,
            depth_row=depth_by_frame.get(frame_idx),
            support_margin_m=float(args.support_margin_m),
            free_space_margin_m=float(args.free_space_margin_m),
        )
        object_depth_rows[frame_idx] = depth_row
        object_face_prov[frame_idx] = face_provenance(vertex_classes, faces)
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            temporal_row = temporal_by_key.get((frame_idx, side))
            arrays = bridge_vertices_and_joints(hand, bridge_cache)
            if arrays is None:
                skipped.append({"frame_idx": frame_idx, "hand_side": side, "reason": "missing_current_bridge_surface"})
                continue
            current_vertices, _current_joints = arrays
            current_measure = measure_hand_against_object(
                vertices_world=current_vertices,
                pose=pose,
                scene=scene,
                face_prov=object_face_prov[frame_idx],
                face_count=len(faces),
                penetration_epsilon_m=float(args.penetration_epsilon_m),
            )
            candidate_vertices, candidate_info = make_candidate_vertices(
                models_by_side=models_by_side,
                temporal_row=temporal_row or {},
                hand=hand,
                current_vertices=current_vertices,
                bridge_cache=bridge_cache,
                source_cache=source_cache,
                device=device,
            )
            candidate_measure = None
            if candidate_vertices is not None:
                candidate_measure = measure_hand_against_object(
                    vertices_world=candidate_vertices,
                    pose=pose,
                    scene=scene,
                    face_prov=object_face_prov[frame_idx],
                    face_count=len(faces),
                    penetration_epsilon_m=float(args.penetration_epsilon_m),
                )
            state, blockers = classify_constraint_state(
                candidate_measure=candidate_measure,
                current_measure=current_measure,
                candidate_info=candidate_info,
                temporal_row=temporal_row,
                args=args,
            )
            hidden_row = hidden_by_frame.get(frame_idx, {})
            per_frame_states.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "temporal_mano_state_input": temporal_row.get("temporal_mano_state") if temporal_row else None,
                    "observed_surface_mano_state": state,
                    "coordinate_correction_accepted": False,
                    "blocking_mechanisms": blockers,
                    "object_depth_provenance": object_depth_rows[frame_idx],
                    "hidden_volume_state_input": hidden_row.get("state"),
                    "current_full_778_measurement": current_measure,
                    "candidate_full_778_measurement": candidate_measure,
                    "candidate_reconstruction": candidate_info,
                }
            )

    # Summarize by contiguous intervals for states that have temporal MANO rows.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_frame_states:
        grouped[(str(row["hand_side"]), str(row["observed_surface_mano_state"]))].append(row)
    intervals: list[dict[str, Any]] = []
    for (side, state), rows in sorted(grouped.items()):
        frames = sorted(int(row["frame_idx"]) for row in rows)
        if not frames:
            continue
        start = end = frames[0]
        chunks: list[tuple[int, int]] = []
        for idx in frames[1:]:
            if idx == end + 1:
                end = idx
            else:
                chunks.append((start, end))
                start = end = idx
        chunks.append((start, end))
        rows_by_frame = {int(row["frame_idx"]): row for row in rows}
        for start, end in chunks:
            seg = [rows_by_frame[idx] for idx in range(start, end + 1) if idx in rows_by_frame]
            cand_obs_max = []
            cand_full_max = []
            cur_obs_max = []
            for row in seg:
                cm = row.get("candidate_full_778_measurement")
                if isinstance(cm, dict):
                    value = scalar_from_summary(cm.get("observed_supported_strict_penetration_m"))
                    if value is not None:
                        cand_obs_max.append(value)
                    cand_full_max.append(float(cm.get("max_penetration_m", 0.0)))
                cur = row.get("current_full_778_measurement")
                if isinstance(cur, dict):
                    value = scalar_from_summary(cur.get("observed_supported_strict_penetration_m"))
                    if value is not None:
                        cur_obs_max.append(value)
            intervals.append(
                {
                    "hand_side": side,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "frame_count": int(len(seg)),
                    "observed_surface_mano_interval_state": state,
                    "coordinate_correction_accepted": False,
                    "candidate_observed_supported_penetration_max_m": numeric_summary(np.asarray(cand_obs_max, dtype=float)),
                    "candidate_full_signed_penetration_max_m": numeric_summary(np.asarray(cand_full_max, dtype=float)),
                    "current_observed_supported_penetration_max_m": numeric_summary(np.asarray(cur_obs_max, dtype=float)),
                    "state_counts": dict(Counter(str(row.get("observed_surface_mano_state")) for row in seg)),
                }
            )

    state_counts = Counter(str(row.get("observed_surface_mano_state")) for row in per_frame_states)
    candidate_summary_by_side: dict[str, dict[str, Any]] = {}
    for side in sorted(set(str(row.get("hand_side")) for row in per_frame_states)):
        candidate_rows = [row for row in per_frame_states if row.get("hand_side") == side and isinstance(row.get("candidate_full_778_measurement"), dict)]
        candidate_obs_max = [
            scalar_from_summary(row["candidate_full_778_measurement"].get("observed_supported_strict_penetration_m"))
            for row in candidate_rows
        ]
        candidate_obs_max = [float(x) for x in candidate_obs_max if x is not None]
        candidate_summary_by_side[side] = {
            "candidate_frame_count": int(len(candidate_rows)),
            "candidate_observed_supported_penetration_max_m": numeric_summary(np.asarray(candidate_obs_max, dtype=float)),
        }
    report = {
        "method": "build_v18_observed_surface_mano_constraint_state",
        "status": "ok",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Full 778-vertex MANO remeasurement separating depth-observed object surface constraints from hidden/free-space-conflicted "
            "compact-rigid volume. This can tighten or falsify interval MANO uncertainty but does not accept coordinate correction."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(path) for path in args.depth_npz],
            "articulated_mano_state": str(args.articulated_mano_state),
            "hidden_volume_validation": str(args.hidden_volume_validation) if args.hidden_volume_validation else None,
            "wilor_mano_right": str(mano_right_path),
            "wilor_mano_left": str(args.wilor_mano_left) if args.wilor_mano_left is not None else None,
            "loaded_mano_models": model_paths,
            "left_model_status": left_model_status,
        },
        "parameters": {
            "support_margin_m": float(args.support_margin_m),
            "free_space_margin_m": float(args.free_space_margin_m),
            "penetration_epsilon_m": float(args.penetration_epsilon_m),
            "accepted_observed_residual_m": float(args.accepted_observed_residual_m),
            "observed_supported_face_rule": "strict: closest face has at least two depth-supported vertices and no free-space-conflict vertex",
            "hawor_left_shapedirs_x_fix": bool(args.hawor_left_shapedirs_x_fix),
        },
        "summary": {
            "annotation_frame_count": int(len(as_list(annotations.get("frames")))),
            "pose_frame_count": int(len(poses)),
            "evaluated_hand_frame_count": int(len(per_frame_states)),
            "state_counts": dict(state_counts),
            "candidate_summary_by_side": candidate_summary_by_side,
            "coordinate_correction_accepted": False,
        },
        "source_temporal_mano_summary": temporal_payload.get("summary") if isinstance(temporal_payload, dict) else None,
        "skipped": skipped[:200],
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "physical_conclusion": (
            "Observed-supported surface residuals identify whether MANO conflicts are constrained by visible metric object evidence or only by "
            "quarantined hidden/free-space compact-rigid volume. Coordinate correction remains false unless a later mechanism supplies full "
            "object volume validity, visible/depth-compatible MANO, and temporal coherence."
        ),
    }
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_observed_surface_mano_constraint_state.json"
    write_json(out_path, report)
    print(json.dumps({"output": str(out_path), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
