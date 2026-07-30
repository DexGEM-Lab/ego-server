#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Measure local object/contact-patch support for V18 contact_patch rows.

The output is still a generic ``contact_patch`` factor report.  This script does
not infer a contact label and does not create a new residual family.  It takes
existing contact-patch rows, measures the object surface support at the actual
MANO interaction neighborhood from model-produced visible masks plus metric
depth, expresses those depth samples in the current object/mesh pose frame, and
emits a row-level support uncertainty.  The interval solver can then use the
existing sliding local-manifold residual: if the local patch is tighter than the
frame-global object support, H_t can move; if it is not, the row remains an
explicit support-bounded hand-state hypothesis.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources, project_points  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import classify_object_vertices_against_depth, face_provenance  # noqa: E402
from build_v18_part_visible_surfaces import resize_bool_mask  # noqa: E402
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
from solve_v18_joint_mano_interval_trajectory import (  # noqa: E402
    contact_patch_targets_from_vertices,
    hand_owned_object_depth_quarantine,
    load_surface_eligibility_rows as solver_load_surface_eligibility_rows,
    load_visible_ownership_rows as solver_load_visible_ownership_rows,
    surface_eligibility_mask_for_row,
    visible_object_mask_face_gate,
    visible_ownership_masks_for_row,
    visible_ownership_quarantine_faces,
)

REJECTED_ANNOTATION_PATH_MARKERS = (
    "v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard",
    "verified_hprime_final",
    "hprime_final",
)

DEFAULT_ANNOTATIONS = Path("/data2/ego_annotation_outputs/v18_full_pipeline_sanitized_base_for_hprime/task5_tomato_960/annotations_v18_full.json")
DEFAULT_DEPTH = Path("/data2/ego_annotation_outputs/v18_unidepth_extension/complete_depth_root/task5_tomato_960/unidepth_metric/unidepth_metric_depth_v3.npz")
DEFAULT_VISIBLE_MASK_REPORT = Path("/data2/ego_annotation_outputs/v18_unidepth_extension/v17_visible_surfaces_complete_depth/task5_tomato_960/v17_multi_object_visible_surface_report.json")
DEFAULT_POSE_REPORT = Path("/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json")
DEFAULT_MESH = Path("/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply")


def reject_rejected_annotation_path(path_or_payload: Any, *, context: str) -> None:
    text = str(path_or_payload)
    hits = [marker for marker in REJECTED_ANNOTATION_PATH_MARKERS if marker in text]
    if hits:
        raise ValueError(f"{context} contains rejected H-prime/final-v7 annotation marker(s) {hits}; use sanitized non-H-prime sources")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-contact-patch-factor-report", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--case", default="task5_tomato_960")
    p.add_argument("--target-entity-id", default="object:obj_tomato")
    p.add_argument("--visible-mask-report", type=Path, default=DEFAULT_VISIBLE_MASK_REPORT)
    p.add_argument("--visible-ownership-factor-report", type=Path, default=None, help="Optional ownership factor used only to choose the visible depth mask measured for local support. It is not applied to the solver residual selector unless --solver-visible-ownership-factor-report is also supplied.")
    p.add_argument("--solver-visible-ownership-factor-report", type=Path, default=None, help="Optional visible_ownership factor that mirrors the interval solver's residual-face quarantine for selector parity.")
    p.add_argument("--surface-eligibility-factor-report", type=Path, default=None, help="Surface eligibility factor that mirrors the interval solver's eligible_hard_observed face mask for selector parity.")
    p.add_argument("--surface-eligibility-mode", choices=("replace", "intersect"), default="intersect")
    p.add_argument("--depth-npz", type=Path, action="append", default=None)
    p.add_argument("--object-pose-fit-report", type=Path, default=DEFAULT_POSE_REPORT)
    p.add_argument("--completed-mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--support-margin-m", type=float, default=0.015)
    p.add_argument("--free-space-margin-m", type=float, default=0.025)
    p.add_argument("--hand-owned-object-depth-quarantine", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hand-owned-quarantine-radius-px", type=float, default=3.0)
    p.add_argument("--hand-owned-quarantine-depth-margin-m", type=float, default=0.005)
    p.add_argument("--hand-owned-quarantine-hand-depth-support-m", type=float, default=0.030)
    p.add_argument("--visible-ownership-face-overlap-dilation-px", type=int, default=2)
    p.add_argument("--visible-object-mask-gate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--visible-mask-quarantine-signed-mesh", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--visible-object-mask-dilation-px", type=int, default=2)
    p.add_argument("--max-contact-patch-vertices", type=int, default=96)
    p.add_argument("--local-pixel-radius-px", type=float, default=20.0, help="Visible object depth pixels must lie this many depth-grid pixels from projected MANO surface vertices.")
    p.add_argument("--local-world-radius-m", type=float, default=0.060, help="Visible object depth points must lie this close to the MANO surface in metric 3D before they are treated as the interaction patch.")
    p.add_argument("--temporal-window-frames", type=int, default=5, help="Use neighboring rows in this +/- frame window to test same-patch object-frame consistency.")
    p.add_argument("--temporal-object-radius-m", type=float, default=0.035, help="Neighbor-frame local samples must be within this object-frame radius of the current patch center to count as temporal support for the same patch.")
    p.add_argument("--min-local-samples", type=int, default=32)
    p.add_argument("--min-temporal-samples", type=int, default=48)
    p.add_argument("--support-floor-m", type=float, default=0.0010)
    p.add_argument("--consume-local-only-if-tighter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-temporal-support", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-mask-depth-pixels", type=int, default=120000)
    return p.parse_args()


def entity_tokens(entity_id: str) -> set[str]:
    raw = str(entity_id)
    tokens = {raw}
    if raw.startswith("object:"):
        tokens.add(raw.split(":", 1)[1])
    else:
        tokens.add(f"object:{raw}")
    return tokens


def row_matches_target(row: dict[str, Any], target_entity_id: str) -> bool:
    tokens = entity_tokens(target_entity_id)
    seen_explicit = False
    for key in ("target_entity_id", "object_id", "entity_id"):
        value = row.get(key)
        if isinstance(value, str):
            seen_explicit = True
            if value in tokens:
                return True
    # A single-object report may omit target fields; multi-object reports should
    # carry explicit object_id/target_entity_id and will be filtered above.
    return not seen_explicit


def load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame["frame_idx"]): frame
        for frame in as_list(annotations.get("frames"))
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }


def load_visible_mask_rows(report_path: Path, target_entity_id: str) -> dict[int, dict[str, Any]]:
    payload = load_json(report_path)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("surface_rows", "visible_object_frame_rows", "saved_mask_rows_after_start", "target_mask_rows"):
            rows.extend([r for r in as_list(payload.get(key)) if isinstance(r, dict) and row_matches_target(r, target_entity_id)])
            if rows:
                break
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("frame_idx") is None:
            continue
        raw = row.get("saved_mask_path") or row.get("mask_path") or row.get("surface_mask_path")
        if not isinstance(raw, str) or not Path(raw).exists():
            continue
        frame_idx = int(row["frame_idx"])
        if frame_idx in out:
            raise ValueError(f"duplicate visible mask row for target={target_entity_id} frame={frame_idx}: {report_path}")
        out[frame_idx] = dict(row)
    if not out:
        raise ValueError(f"visible mask report produced zero rows for target={target_entity_id}: {report_path}")
    return out


def load_ownership_rows(report_path: Path | None, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if report_path is None:
        return {}
    payload = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get("ownership_rows") if isinstance(payload, dict) else None):
        if not isinstance(row, dict) or not row_matches_target(row, target_entity_id):
            continue
        if row.get("frame_idx") is None or row.get("hand_side") is None:
            continue
        key = (int(row["frame_idx"]), str(row["hand_side"]))
        if key in out:
            raise ValueError(f"duplicate ownership row for target={target_entity_id} frame/side={key}: {report_path}")
        out[key] = dict(row)
    return out


def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"could not decode mask: {path}")
    return arr > 0


def mask_path_for_row(mask_row: dict[str, Any]) -> Path | None:
    raw = mask_row.get("constraint_eligible_entity_mask_path") or mask_row.get("adjusted_entity_mask_path") or mask_row.get("surface_mask_path") or mask_row.get("saved_mask_path") or mask_row.get("mask_path")
    if not isinstance(raw, str):
        return None
    path = Path(raw)
    return path if path.exists() else None


def load_hand_vertices(frame: dict[str, Any], side: str, cache: dict[Path, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    for hand in as_list(frame.get("hands")):
        if not isinstance(hand, dict) or str(hand.get("hand_side")) != str(side):
            continue
        metric_raw = hand.get("metric_mano_state")
        metric: dict[str, Any] = metric_raw if isinstance(metric_raw, dict) else {}
        reference_raw = metric.get("vertices_reference")
        reference: dict[str, Any] = reference_raw if isinstance(reference_raw, dict) else {}
        candidates = [
            (reference.get("bridge_npz"), reference.get("bridge_vertices_world_array"), reference.get("bridge_row_index"), "bridge_npz"),
            (reference.get("source_hawor_npz"), reference.get("source_vertices_world_array"), reference.get("source_frame_index"), "source_hawor_npz"),
        ]
        for raw_path, array_name, row_index, source_kind in candidates:
            if isinstance(raw_path, str) and isinstance(array_name, str) and row_index is not None and Path(raw_path).exists():
                path = Path(raw_path)
                if path not in cache:
                    cache[path] = np.load(path, allow_pickle=True)
                arr = np.asarray(cache[path][array_name][int(row_index)], dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 100:
                    return arr, {"state": "loaded_full_mano_vertices", "source_kind": source_kind, "source_path": str(path), "array": array_name, "row_index": int(row_index), "vertex_count": int(len(arr))}
        sample = np.asarray(metric.get("vertices_world_sample_m") or [], dtype=float)
        if sample.ndim == 2 and sample.shape[1] == 3 and len(sample) >= 8:
            return sample, {"state": "loaded_sparse_mano_vertex_sample_only", "vertex_count": int(len(sample))}
    return None, {"state": "missing_hand_vertices"}


def compute_face_normals_object(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1.0e-12
    normals[valid] /= norms[valid, None]
    normals[~valid] = 0.0
    return normals.astype(float)


def solver_residual_contact_patch_selection(
    *,
    frame: dict[str, Any],
    side: str,
    hand_vertices_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    visible_mask_for_gate: np.ndarray | None,
    solver_ownership_row: dict[str, Any] | None,
    surface_row: dict[str, Any] | None,
    vertices_object: np.ndarray,
    faces: np.ndarray,
    face_normals_object: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene,
    row: dict[str, Any],
    args: argparse.Namespace,
    mask_cache: dict[Path, np.ndarray],
    surface_cache: dict[Path, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproduce the interval solver's contact_patch residual selector.

    The contact residual is applied only to MANO vertices whose closest completed-
    mesh primitive is in the solver's current trusted observed-face set, then it
    is sorted by hand-to-target distance and capped by max_vertices.  Local support
    measurements that ignore this selector can be dominated by unrelated visible
    object surface and are not evidence about the residual that moves H_t.
    """
    vertex_classes, obj_summary = classify_object_vertices_against_depth(
        frame=frame,
        vertices_object=vertices_object,
        pose=pose,
        depth_row=depth_row,
        support_margin_m=float(args.support_margin_m),
        free_space_margin_m=float(args.free_space_margin_m),
    )
    prov = face_provenance(vertex_classes, faces)
    strict_raw = np.asarray(prov["observed_supported_strict"], dtype=bool)
    strict, hand_owned_quarantined = hand_owned_object_depth_quarantine(
        frame=frame,
        side=side,
        object_vertices=vertices_object,
        object_faces=faces,
        object_pose=pose,
        hand_vertices_world=hand_vertices_world,
        face_strict_observed=strict_raw,
        depth_row=depth_row,
        args=args,
    )
    ownership_diag: dict[str, Any] = {"state": "missing_visible_ownership_row"}
    ownership_quarantined = 0
    solver_gate_mask = visible_mask_for_gate
    if solver_ownership_row is not None:
        non_object_mask, constraint_mask, ownership_diag = visible_ownership_masks_for_row(solver_ownership_row, mask_cache)
        strict, ownership_quarantined = visible_ownership_quarantine_faces(
            frame=frame,
            side=side,
            object_vertices=vertices_object,
            object_faces=faces,
            object_pose=pose,
            face_strict_observed=strict,
            non_object_owned_mask=non_object_mask,
            args=args,
        )
        solver_gate_mask = constraint_mask if constraint_mask is not None else solver_gate_mask
    strict, visible_mask_face_count_raw, visible_mask_face_count = visible_object_mask_face_gate(
        frame=frame,
        side=side,
        object_vertices=vertices_object,
        object_faces=faces,
        object_pose=pose,
        face_strict_observed=strict,
        mask=solver_gate_mask,
        args=args,
    )
    surface_mask, surface_diag = surface_eligibility_mask_for_row(surface_row, len(faces), surface_cache)
    surface_input_face_count = int(np.count_nonzero(strict))
    if args.surface_eligibility_factor_report is not None and surface_mask is None:
        raise ValueError(f"surface eligibility factor could not supply exact contact-patch selector row frame={row.get('frame_idx')} side={side}: {surface_diag}")
    if surface_mask is not None:
        if str(args.surface_eligibility_mode) == "replace":
            strict = surface_mask.astype(bool)
        else:
            strict = strict & surface_mask.astype(bool)
    try:
        band_m = float(row.get("contact_patch_band_m", row.get("band_m", 0.020)) or 0.020)
    except Exception:
        band_m = 0.020
    try:
        max_vertices = int(row.get("max_vertices", args.max_contact_patch_vertices) or args.max_contact_patch_vertices)
    except Exception:
        max_vertices = int(args.max_contact_patch_vertices)
    idx, targets, normals, distances = contact_patch_targets_from_vertices(
        np.asarray(hand_vertices_world, dtype=float),
        scene,
        np.asarray(pose[0], dtype=float),
        np.asarray(pose[1], dtype=float),
        strict.astype(bool),
        np.asarray(face_normals_object, dtype=float),
        max_vertices=max_vertices,
        band_m=band_m,
    )
    diag = {
        "state": "solver_residual_patch_selected" if idx.size else "no_solver_residual_patch_vertices",
        "selector": "interval_solver.contact_patch_targets_from_vertices",
        "contact_patch_band_m": float(band_m),
        "max_vertices": int(max_vertices),
        "raw_strict_observed_face_count": int(np.count_nonzero(strict_raw)),
        "hand_owned_quarantined_face_count": int(hand_owned_quarantined),
        "solver_visible_ownership_quarantined_face_count": int(ownership_quarantined),
        "visible_object_mask_face_count_raw": int(visible_mask_face_count_raw),
        "visible_object_mask_face_count": int(visible_mask_face_count),
        "surface_eligibility_state": surface_diag.get("state"),
        "surface_eligibility_npz_path": surface_diag.get("face_state_npz_path"),
        "surface_eligible_face_count": int(surface_diag.get("eligible_hard_observed_count", 0) or 0),
        "surface_input_face_count": int(surface_input_face_count),
        "final_strict_observed_face_count": int(np.count_nonzero(strict)),
        "solver_ownership_state": ownership_diag.get("state"),
        "contact_patch_selected_vertex_count": int(idx.size),
        "contact_patch_selected_vertex_ids": idx.astype(int).tolist(),
        "contact_patch_target_count": int(len(targets)),
        "contact_patch_initial_distance_m": numeric_summary(np.asarray(distances, dtype=float)),
        "target_points_world_m": np.asarray(targets, dtype=float).tolist(),
        "target_normals_world": np.asarray(normals, dtype=float).tolist(),
        "object_depth_summary": obj_summary,
    }
    return idx.astype(np.int64), targets.astype(float), normals.astype(float), distances.astype(float), diag


def project_world_to_depth_grid(points_world: np.ndarray, frame: dict[str, Any], depth_shape: tuple[int, int], intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_c2w, t_c2w = frame_camera_pose(frame)
    pts_cam = (np.asarray(points_world, dtype=float) - t_c2w[None, :]) @ r_c2w
    h, w = depth_shape
    u, v, valid = project_points(pts_cam, np.asarray(intrinsics, dtype=float), w, h)
    uv = np.column_stack([u.astype(float), v.astype(float)])
    return uv, valid


def lift_pixels_to_world(xs: np.ndarray, ys: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    z = depth[ys, xs].astype(np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if intr.size == 9:
        fx, fy, cx, cy = float(intr[0]), float(intr[4]), float(intr[2]), float(intr[5])
    else:
        fx, fy, cx, cy = [float(v) for v in intr[:4]]
    x_cam = (xs.astype(np.float64) - cx) * z / fx
    y_cam = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.column_stack([x_cam, y_cam, z])
    r_c2w, t_c2w = frame_camera_pose(frame)
    return pts_cam @ r_c2w.T + t_c2w[None, :]


def contact_neighborhood_vertices(
    *,
    hand_vertices_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    scene: o3d.t.geometry.RaycastingScene,
    contact_band_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if hand_vertices_world.ndim != 2 or hand_vertices_world.shape[1] != 3 or len(hand_vertices_world) == 0:
        return np.zeros((0, 3), dtype=float), {"state": "missing_hand_vertices", "raw_hand_vertex_count": 0, "contact_vertex_count": 0}
    r_obj, t_obj = pose
    vertices_object = inverse_object(np.asarray(hand_vertices_world, dtype=float), np.asarray(r_obj, dtype=float), np.asarray(t_obj, dtype=float))
    closest = scene.compute_closest_points(o3d.core.Tensor(vertices_object.astype(np.float32)))
    primitive = closest["primitive_ids"].numpy().astype(np.int64)
    closest_object = closest["points"].numpy().astype(float)
    valid = primitive >= 0
    closest_world = closest_object @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    dist = np.linalg.norm(np.asarray(hand_vertices_world, dtype=float) - closest_world, axis=1)
    near = valid & np.isfinite(dist) & (dist <= float(contact_band_m))
    diag = {
        "state": "contact_neighborhood_vertices_selected" if np.any(near) else "no_mano_vertices_within_contact_patch_band",
        "raw_hand_vertex_count": int(len(hand_vertices_world)),
        "contact_vertex_count": int(np.count_nonzero(near)),
        "contact_band_m": float(contact_band_m),
        "nearest_mesh_distance_m": numeric_summary(dist[np.isfinite(dist)]),
    }
    return np.asarray(hand_vertices_world, dtype=float)[near], diag


def local_visible_depth_points(
    *,
    frame: dict[str, Any],
    contact_vertices_world: np.ndarray,
    mask: np.ndarray,
    depth_row: dict[str, Any],
    local_pixel_radius_px: float,
    local_world_radius_m: float,
    max_mask_depth_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    mask_d = resize_bool_mask(mask.astype(bool), depth.shape)
    valid = mask_d & np.isfinite(depth) & (depth > 1.0e-5)
    ys_all, xs_all = np.nonzero(valid)
    if xs_all.size == 0:
        return np.zeros((0, 3), dtype=float), {"state": "visible_mask_without_valid_depth", "valid_depth_pixels": 0}
    if int(max_mask_depth_pixels) > 0 and xs_all.size > int(max_mask_depth_pixels):
        order = np.linspace(0, xs_all.size - 1, int(max_mask_depth_pixels), dtype=np.int64)
        xs_all = xs_all[order]
        ys_all = ys_all[order]
    if contact_vertices_world.ndim != 2 or contact_vertices_world.shape[1] != 3 or len(contact_vertices_world) == 0:
        return np.zeros((0, 3), dtype=float), {"state": "no_contact_neighborhood_vertices_for_local_patch", "valid_depth_pixels": int(xs_all.size)}
    uv_hand, hand_valid = project_world_to_depth_grid(contact_vertices_world, frame, depth.shape, np.asarray(depth_row["intrinsics"], dtype=float))
    uv_hand = uv_hand[hand_valid]
    if uv_hand.size == 0:
        return np.zeros((0, 3), dtype=float), {"state": "contact_vertices_do_not_project_to_depth_grid", "valid_depth_pixels": int(xs_all.size), "contact_vertex_count": int(len(contact_vertices_world))}
    pixel_tree = KDTree(uv_hand)
    pixel_uv = np.column_stack([xs_all.astype(float), ys_all.astype(float)])
    pix_dist, _ = pixel_tree.query(pixel_uv, k=1)
    near_pixel = pix_dist <= float(local_pixel_radius_px)
    if not np.any(near_pixel):
        return np.zeros((0, 3), dtype=float), {"state": "no_object_depth_pixels_near_projected_hand", "valid_depth_pixels": int(xs_all.size), "projected_hand_vertex_count": int(len(uv_hand)), "nearest_pixel_distance_px": numeric_summary(pix_dist.astype(float))}
    xs = xs_all[near_pixel]
    ys = ys_all[near_pixel]
    pts_world = lift_pixels_to_world(xs, ys, depth, np.asarray(depth_row["intrinsics"], dtype=float), frame)
    hand_tree = KDTree(np.asarray(contact_vertices_world, dtype=float))
    world_dist, _ = hand_tree.query(pts_world, k=1)
    near_world = world_dist <= float(local_world_radius_m)
    pts_world = pts_world[near_world]
    diag = {
        "state": "local_visible_depth_points_selected" if len(pts_world) else "no_near_metric_object_patch_after_world_radius",
        "valid_depth_pixels": int(xs_all.size),
        "projected_contact_vertex_count": int(len(uv_hand)),
        "near_pixel_depth_pixels": int(np.count_nonzero(near_pixel)),
        "local_patch_depth_sample_count": int(len(pts_world)),
        "nearest_pixel_distance_px": numeric_summary(pix_dist[near_pixel].astype(float)),
        "nearest_hand_world_distance_m": numeric_summary(world_dist.astype(float)),
    }
    return pts_world.astype(float), diag


def mesh_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
    return scene


def measure_patch_against_mesh(
    *,
    points_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    scene: o3d.t.geometry.RaycastingScene,
    face_normals_object: np.ndarray,
) -> dict[str, Any]:
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) == 0:
        return {"state": "no_local_points", "points_object": np.zeros((0, 3), dtype=float), "signed_normal_m": np.zeros((0,), dtype=float), "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float), "patch_center_object_m": None, "patch_normal_object": None, "sample_count": 0}
    r_obj, t_obj = pose
    points_object = inverse_object(np.asarray(points_world, dtype=float), np.asarray(r_obj, dtype=float), np.asarray(t_obj, dtype=float))
    closest = scene.compute_closest_points(o3d.core.Tensor(points_object.astype(np.float32)))
    primitive = closest["primitive_ids"].numpy().astype(np.int64)
    closest_object = closest["points"].numpy().astype(float)
    valid = (primitive >= 0) & (primitive < len(face_normals_object))
    points_object_v = points_object[valid]
    closest_object_v = closest_object[valid]
    primitive_v = primitive[valid]
    if points_object_v.size == 0:
        return {"state": "no_valid_mesh_closest_points", "points_object": points_object, "signed_normal_m": np.zeros((0,), dtype=float), "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float), "patch_center_object_m": None, "patch_normal_object": None, "sample_count": 0}
    normals = np.asarray(face_normals_object, dtype=float)[primitive_v]
    nn = np.linalg.norm(normals, axis=1)
    good = nn > 1.0e-9
    points_object_v = points_object_v[good]
    closest_object_v = closest_object_v[good]
    normals = normals[good] / nn[good, None]
    residual_vec = points_object_v - closest_object_v
    signed = np.einsum("ij,ij->i", residual_vec, normals)
    euclidean = np.linalg.norm(residual_vec, axis=1)
    center = np.median(points_object_v, axis=0)
    plane_abs = np.zeros((len(points_object_v),), dtype=float)
    patch_normal = None
    if len(points_object_v) >= 3:
        centered = points_object_v - center[None, :]
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        patch_normal = vh[-1]
        if np.dot(patch_normal, np.median(normals, axis=0)) < 0:
            patch_normal = -patch_normal
        plane_abs = np.abs(centered @ patch_normal)
    return {
        "state": "measured_local_patch_support",
        "points_object": points_object_v.astype(float),
        "signed_normal_m": signed.astype(float),
        "normal_abs_m": np.abs(signed).astype(float),
        "euclidean_m": euclidean.astype(float),
        "plane_abs_m": plane_abs.astype(float),
        "patch_center_object_m": center.astype(float).tolist(),
        "patch_normal_object": None if patch_normal is None else patch_normal.astype(float).tolist(),
        "sample_count": int(len(points_object_v)),
    }


def summarize_measurement(meas: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": meas.get("state"),
        "sample_count": int(meas.get("sample_count", 0) or 0),
        "patch_center_object_m": meas.get("patch_center_object_m"),
        "patch_normal_object": meas.get("patch_normal_object"),
        "observed_to_mesh_normal_abs_m": numeric_summary(np.asarray(meas.get("normal_abs_m", []), dtype=float)),
        "observed_to_mesh_euclidean_m": numeric_summary(np.asarray(meas.get("euclidean_m", []), dtype=float)),
        "local_plane_abs_m": numeric_summary(np.asarray(meas.get("plane_abs_m", []), dtype=float)),
    }


def p95_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, 95.0))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    reject_rejected_annotation_path(args.annotations, context="annotations path")
    annotations = load_json(args.annotations)
    reject_rejected_annotation_path(annotations.get("source_annotations") or "", context="annotations source metadata")
    frames = load_frames(annotations)
    base_report = load_json(args.base_contact_patch_factor_report)
    reject_rejected_annotation_path(base_report, context="base contact_patch factor report")
    base_rows = [r for r in as_list(base_report.get("factor_rows")) if isinstance(r, dict)]
    if not base_rows:
        raise ValueError(f"base contact_patch factor report has zero factor_rows: {args.base_contact_patch_factor_report}")
    for row in base_rows:
        if str(row.get("factor_family")) != "contact_patch":
            raise ValueError(f"base factor row is not contact_patch: {row.get('factor_family')}")
        if str(row.get("target_entity_id")) != str(args.target_entity_id):
            raise ValueError(f"base factor row target mismatch {row.get('target_entity_id')} != {args.target_entity_id}")

    depth_rows = load_depth_sources(list(args.depth_npz or [DEFAULT_DEPTH]))
    visible_rows = load_visible_mask_rows(args.visible_mask_report, str(args.target_entity_id))
    ownership_rows = load_ownership_rows(args.visible_ownership_factor_report, str(args.target_entity_id))
    solver_ownership_rows = solver_load_visible_ownership_rows(args.solver_visible_ownership_factor_report, target_entity_id=str(args.target_entity_id))
    surface_rows = solver_load_surface_eligibility_rows(args.surface_eligibility_factor_report, target_entity_id=str(args.target_entity_id))
    pose_report = load_json(args.object_pose_fit_report)
    poses = pose_map(pose_report)
    pose_rows = {int(r["frame_idx"]): r for r in as_list(pose_report.get("pose_rows")) if isinstance(r, dict) and r.get("frame_idx") is not None}
    mesh = load_mesh(args.completed_mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_normals = compute_face_normals_object(vertices, faces)
    scene = mesh_scene(vertices, faces)

    bridge_cache: dict[Path, Any] = {}
    solver_mask_cache: dict[Path, np.ndarray] = {}
    surface_cache: dict[Path, np.ndarray] = {}
    row_measurements: dict[tuple[int, str], dict[str, Any]] = {}
    selection_diagnostics: dict[tuple[int, str], dict[str, Any]] = {}

    for row in base_rows:
        frame_idx = int(row["frame_idx"])
        side = str(row["hand_side"])
        key = (frame_idx, side)
        frame = frames.get(frame_idx)
        pose = poses.get(frame_idx)
        depth_row = depth_rows.get(frame_idx)
        visible_row = visible_rows.get(frame_idx)
        ownership_row = ownership_rows.get(key)
        source_mask_row = ownership_row if ownership_row is not None and mask_path_for_row(ownership_row) is not None else visible_row
        if frame is None or pose is None or depth_row is None or source_mask_row is None:
            row_measurements[key] = {"state": "missing_required_source", "points_object": np.zeros((0, 3), dtype=float), "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float), "sample_count": 0, "missing": {"frame": frame is None, "pose": pose is None, "depth": depth_row is None, "mask": source_mask_row is None}}
            continue
        hand_vertices, hand_diag = load_hand_vertices(frame, side, bridge_cache)
        if hand_vertices is None:
            row_measurements[key] = {"state": "missing_hand_vertices", "points_object": np.zeros((0, 3), dtype=float), "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float), "sample_count": 0, "hand_vertex_source": hand_diag}
            continue
        mask_path = mask_path_for_row(source_mask_row)
        if mask_path is None:
            row_measurements[key] = {"state": "missing_readable_visible_mask", "points_object": np.zeros((0, 3), dtype=float), "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float), "sample_count": 0, "hand_vertex_source": hand_diag}
            continue
        mask = load_mask(mask_path)
        solver_ownership_row = solver_ownership_rows.get(key)
        surface_row = surface_rows.get(key)
        solver_idx, solver_targets, solver_normals, solver_distances, solver_patch_diag = solver_residual_contact_patch_selection(
            frame=frame,
            side=side,
            hand_vertices_world=hand_vertices,
            pose=pose,
            depth_row=depth_row,
            visible_mask_for_gate=mask,
            solver_ownership_row=solver_ownership_row,
            surface_row=surface_row,
            vertices_object=vertices,
            faces=faces,
            face_normals_object=face_normals,
            scene=scene,
            row=row,
            args=args,
            mask_cache=solver_mask_cache,
            surface_cache=surface_cache,
        )
        pts_world, selection_diag = local_visible_depth_points(
            frame=frame,
            contact_vertices_world=solver_targets,
            mask=mask,
            depth_row=depth_row,
            local_pixel_radius_px=float(args.local_pixel_radius_px),
            local_world_radius_m=float(args.local_world_radius_m),
            max_mask_depth_pixels=int(args.max_mask_depth_pixels),
        )
        selection_diag.update(
            {
                "mask_path": str(mask_path),
                "ownership_filtered_for_measurement_mask": bool(ownership_row is not None and source_mask_row is ownership_row),
                "solver_visible_ownership_factor_report": None if args.solver_visible_ownership_factor_report is None else str(args.solver_visible_ownership_factor_report),
                "surface_eligibility_factor_report": None if args.surface_eligibility_factor_report is None else str(args.surface_eligibility_factor_report),
                "hand_vertex_source": hand_diag,
                "solver_residual_patch_selection": solver_patch_diag,
                "solver_residual_patch_vertex_ids": solver_idx.astype(int).tolist(),
                "solver_residual_patch_vertex_count": int(solver_idx.size),
                "solver_residual_patch_target_count": int(len(solver_targets)),
                "solver_residual_patch_initial_distance_m": numeric_summary(np.asarray(solver_distances, dtype=float)),
            }
        )
        selection_diagnostics[key] = selection_diag
        meas = measure_patch_against_mesh(points_world=pts_world, pose=pose, scene=scene, face_normals_object=face_normals)
        row_measurements[key] = meas

    # Temporal same-patch support: use neighboring local samples only if they land
    # near the current patch center in the canonical/object frame.
    temporal_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in base_rows:
        frame_idx = int(row["frame_idx"])
        side = str(row["hand_side"])
        key = (frame_idx, side)
        meas = row_measurements.get(key, {})
        center_raw = meas.get("patch_center_object_m")
        if not isinstance(center_raw, list):
            temporal_by_key[key] = {"state": "missing_current_patch_center", "sample_count": 0, "normal_abs_m": np.zeros((0,), dtype=float), "euclidean_m": np.zeros((0,), dtype=float), "plane_abs_m": np.zeros((0,), dtype=float)}
            continue
        center = np.asarray(center_raw, dtype=float)
        normals: list[np.ndarray] = []
        euclidean: list[np.ndarray] = []
        plane: list[np.ndarray] = []
        contributing: list[dict[str, Any]] = []
        for other_key, other in row_measurements.items():
            other_frame, other_side = other_key
            if other_side != side or abs(int(other_frame) - frame_idx) > int(args.temporal_window_frames):
                continue
            pts_obj = np.asarray(other.get("points_object", np.zeros((0, 3))), dtype=float)
            if pts_obj.ndim != 2 or pts_obj.shape[1] != 3 or len(pts_obj) == 0:
                continue
            near = np.linalg.norm(pts_obj - center[None, :], axis=1) <= float(args.temporal_object_radius_m)
            if not np.any(near):
                continue
            normal_abs = np.asarray(other.get("normal_abs_m", []), dtype=float)
            euclidean_m = np.asarray(other.get("euclidean_m", []), dtype=float)
            plane_abs = np.asarray(other.get("plane_abs_m", []), dtype=float)
            # All arrays are produced over the valid points_object subset.
            n = min(len(near), len(normal_abs), len(euclidean_m), len(plane_abs))
            near = near[:n]
            if np.any(near):
                normals.append(normal_abs[:n][near])
                euclidean.append(euclidean_m[:n][near])
                plane.append(plane_abs[:n][near])
                contributing.append({"frame_idx": int(other_frame), "sample_count": int(np.count_nonzero(near))})
        normal_cat = np.concatenate(normals).astype(float) if normals else np.zeros((0,), dtype=float)
        euclidean_cat = np.concatenate(euclidean).astype(float) if euclidean else np.zeros((0,), dtype=float)
        plane_cat = np.concatenate(plane).astype(float) if plane else np.zeros((0,), dtype=float)
        temporal_by_key[key] = {
            "state": "temporal_same_patch_samples_selected" if len(normal_cat) else "no_temporal_same_patch_samples",
            "sample_count": int(len(normal_cat)),
            "contributing_frames": contributing,
            "normal_abs_m": normal_cat,
            "euclidean_m": euclidean_cat,
            "plane_abs_m": plane_cat,
        }

    output_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    consumed_count = 0
    for row in base_rows:
        frame_idx = int(row["frame_idx"])
        side = str(row["hand_side"])
        key = (frame_idx, side)
        meas = row_measurements.get(key, {})
        temp = temporal_by_key.get(key, {})
        local_count = int(meas.get("sample_count", 0) or 0)
        temporal_count = int(temp.get("sample_count", 0) or 0)
        local_p95 = p95_or_none(np.asarray(meas.get("normal_abs_m", []), dtype=float))
        temporal_p95 = p95_or_none(np.asarray(temp.get("normal_abs_m", []), dtype=float))
        plane_p95 = p95_or_none(np.asarray(meas.get("plane_abs_m", []), dtype=float))
        # The solver contact residual uses per-face mesh normals and allows tangential sliding.
        # Local plane breadth/curvature is therefore a geometry diagnostic, not an added
        # normal-support uncertainty.  The consumed deadband should represent measured
        # observed-to-mesh normal support in the current and temporally consistent patch.
        components = [v for v in (local_p95, temporal_p95, float(args.support_floor_m)) if v is not None]
        local_uncertainty = max(components) if components and local_count >= int(args.min_local_samples) else None
        global_uncertainty = finite_float(row.get("object_support_uncertainty_m", row.get("contact_patch_support_uncertainty_m", 0.0)), 0.0)
        target_margin = finite_float(row.get("contact_patch_target_margin_m", row.get("target_margin_m", 0.0025)), 0.0025)
        local_supported = local_uncertainty is not None and local_count >= int(args.min_local_samples)
        temporal_supported = temporal_count >= int(args.min_temporal_samples)
        if local_supported and (temporal_supported or not bool(args.require_temporal_support)):
            local_state = "current_and_temporal_local_patch_supported" if temporal_supported else "current_local_patch_supported_temporal_insufficient"
        elif local_supported:
            local_state = "current_local_patch_supported_temporal_insufficient"
        else:
            local_state = str(meas.get("state") or "local_patch_support_not_measured")
        tighter = bool(local_uncertainty is not None and (not bool(args.consume_local_only_if_tighter) or local_uncertainty < global_uncertainty or global_uncertainty <= 0.0))
        consumed = bool(local_supported and (temporal_supported or not bool(args.require_temporal_support)) and tighter)
        support_used = float(local_uncertainty if consumed and local_uncertainty is not None else global_uncertainty)
        if consumed:
            consumed_count += 1
        state_counts[local_state] += 1
        new_row = dict(row)
        new_row.update(
            {
                "object_support_uncertainty_m": max(0.0, support_used),
                "contact_patch_support_uncertainty_m": max(0.0, support_used),
                "contact_patch_deadband_m": float(target_margin + max(0.0, support_used)),
                "global_object_support_uncertainty_m": max(0.0, global_uncertainty),
                "local_patch_support_uncertainty_m": None if local_uncertainty is None else float(local_uncertainty),
                "local_patch_plane_breadth_p95_m": None if plane_p95 is None else float(plane_p95),
                "local_patch_support_state": local_state,
                "local_patch_support_consumed": bool(consumed),
                "local_patch_support_tighter_than_global": bool(tighter),
                "local_patch_sample_count": int(local_count),
                "local_patch_temporal_sample_count": int(temporal_count),
                "contact_patch_support_uncertainty_source": "local_visible_mask_depth_temporal_patch_support" if consumed else "global_object_pose_support_fallback",
                "source_object_support_uncertainty_stat": "local_patch_visible_mask_depth_normal_p95_with_temporal_consistency" if consumed else row.get("source_object_support_uncertainty_stat"),
                "residual_or_quarantine_rule": "select current MANO vertices near eligible observed object surface and penalize surface-normal distance only beyond contact_patch_target_margin_m + local measured patch support uncertainty when current+temporal patch support is tighter than global pose support; otherwise fall back to global support and render a support-bounded H_t hypothesis",
                "rendered_uncertainty_channel": "local object/contact-patch support-bounded MANO hypothesis; local support consumed only if mask/depth/temporal object-frame evidence is tighter than global object support",
            }
        )
        raw_provenance = new_row.get("provenance")
        provenance: dict[str, Any] = dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
        provenance["local_patch_support"] = {
            "annotations": str(args.annotations),
            "visible_mask_report": str(args.visible_mask_report),
            "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report),
            "solver_visible_ownership_factor_report": None if args.solver_visible_ownership_factor_report is None else str(args.solver_visible_ownership_factor_report),
            "surface_eligibility_factor_report": None if args.surface_eligibility_factor_report is None else str(args.surface_eligibility_factor_report),
            "object_pose_fit_report": str(args.object_pose_fit_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(p) for p in (args.depth_npz or [DEFAULT_DEPTH])],
            "selection_parameters": {
                "local_pixel_radius_px": float(args.local_pixel_radius_px),
                "local_world_radius_m": float(args.local_world_radius_m),
                "measurement_patch_source": "exact interval-solver contact_patch target points after eligible-face/ownership/hand-owned gates and max_vertices ordering",
                "temporal_window_frames": int(args.temporal_window_frames),
                "temporal_object_radius_m": float(args.temporal_object_radius_m),
                "min_local_samples": int(args.min_local_samples),
                "min_temporal_samples": int(args.min_temporal_samples),
                "support_floor_m": float(args.support_floor_m),
                "require_temporal_support": bool(args.require_temporal_support),
                "consume_local_only_if_tighter": bool(args.consume_local_only_if_tighter),
                "surface_eligibility_mode": str(args.surface_eligibility_mode),
                "hand_owned_object_depth_quarantine": bool(args.hand_owned_object_depth_quarantine),
                "visible_object_mask_gate": bool(args.visible_object_mask_gate),
                "visible_mask_quarantine_signed_mesh": bool(args.visible_mask_quarantine_signed_mesh),
            },
            "selection_diagnostics": selection_diagnostics.get(key),
            "current_frame_measurement": summarize_measurement(meas),
            "temporal_measurement": {
                "state": temp.get("state"),
                "sample_count": int(temp.get("sample_count", 0) or 0),
                "contributing_frames": temp.get("contributing_frames", []),
                "observed_to_mesh_normal_abs_m": numeric_summary(np.asarray(temp.get("normal_abs_m", []), dtype=float)),
                "observed_to_mesh_euclidean_m": numeric_summary(np.asarray(temp.get("euclidean_m", []), dtype=float)),
                "local_plane_abs_m": numeric_summary(np.asarray(temp.get("plane_abs_m", []), dtype=float)),
            },
            "global_object_support_uncertainty_m": max(0.0, global_uncertainty),
            "local_patch_support_uncertainty_m": None if local_uncertainty is None else float(local_uncertainty),
            "local_patch_plane_breadth_p95_m": None if plane_p95 is None else float(plane_p95),
            "local_patch_support_consumed": bool(consumed),
            "claim": "support uncertainty for the visible object depth samples near the exact solver-consumed contact_patch target points; not a stable point anchor and not contact truth by itself",
        }
        new_row["provenance"] = provenance
        output_rows.append(new_row)

    local_uncertainties = [r.get("local_patch_support_uncertainty_m") for r in output_rows if isinstance(r.get("local_patch_support_uncertainty_m"), (int, float))]
    global_uncertainties = [float(r.get("global_object_support_uncertainty_m", 0.0) or 0.0) for r in output_rows]
    used_uncertainties = [float(r.get("contact_patch_support_uncertainty_m", 0.0) or 0.0) for r in output_rows]
    support_delta = [float(r.get("global_object_support_uncertainty_m", 0.0) or 0.0) - float(r.get("contact_patch_support_uncertainty_m", 0.0) or 0.0) for r in output_rows]
    plane_breadths = [r.get("local_patch_plane_breadth_p95_m") for r in output_rows if isinstance(r.get("local_patch_plane_breadth_p95_m"), (int, float))]
    solver_patch_counts = [int((selection_diagnostics.get((int(r["frame_idx"]), str(r["hand_side"]))) or {}).get("solver_residual_patch_vertex_count", 0) or 0) for r in output_rows]
    valid_depth_pixel_counts = [int((selection_diagnostics.get((int(r["frame_idx"]), str(r["hand_side"]))) or {}).get("valid_depth_pixels", 0) or 0) for r in output_rows]
    sample_to_valid_depth_ratios = [
        float(int(r.get("local_patch_sample_count", 0) or 0) / max(1, valid_depth_pixel_counts[i]))
        for i, r in enumerate(output_rows)
    ]
    report = {
        "method": "v18_local_contact_patch_support_factor_from_visible_mask_depth_temporal_object_frame_consistency",
        "case": str(args.case),
        "target_entity_id": str(args.target_entity_id),
        "claim_scope": "Solver-consumed local support uncertainty for existing contact_patch rows. It can narrow or bound H_t only through the existing sliding contact-manifold residual; it is not a contact label, point anchor, object pose proof, hidden geometry, or validator result.",
        "inputs": {
            "base_contact_patch_factor_report": str(args.base_contact_patch_factor_report),
            "annotations": str(args.annotations),
            "visible_mask_report": str(args.visible_mask_report),
            "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report),
            "solver_visible_ownership_factor_report": None if args.solver_visible_ownership_factor_report is None else str(args.solver_visible_ownership_factor_report),
            "surface_eligibility_factor_report": None if args.surface_eligibility_factor_report is None else str(args.surface_eligibility_factor_report),
            "depth_npz": [str(p) for p in (args.depth_npz or [DEFAULT_DEPTH])],
            "object_pose_fit_report": str(args.object_pose_fit_report),
            "completed_mesh": str(args.completed_mesh),
        },
        "parameters": {
            "local_pixel_radius_px": float(args.local_pixel_radius_px),
            "local_world_radius_m": float(args.local_world_radius_m),
            "measurement_patch_source": "exact interval-solver contact_patch target points after eligible-face/ownership/hand-owned gates and max_vertices ordering",
            "surface_eligibility_mode": str(args.surface_eligibility_mode),
            "hand_owned_object_depth_quarantine": bool(args.hand_owned_object_depth_quarantine),
            "visible_object_mask_gate": bool(args.visible_object_mask_gate),
            "visible_mask_quarantine_signed_mesh": bool(args.visible_mask_quarantine_signed_mesh),
            "temporal_window_frames": int(args.temporal_window_frames),
            "temporal_object_radius_m": float(args.temporal_object_radius_m),
            "min_local_samples": int(args.min_local_samples),
            "min_temporal_samples": int(args.min_temporal_samples),
            "support_floor_m": float(args.support_floor_m),
            "consume_local_only_if_tighter": bool(args.consume_local_only_if_tighter),
            "require_temporal_support": bool(args.require_temporal_support),
        },
        "summary": {
            "factor_row_count": len(output_rows),
            "local_patch_support_consumed_count": int(consumed_count),
            "local_patch_support_state_counts": dict(sorted(state_counts.items())),
            "local_patch_support_uncertainty_m": numeric_summary(np.asarray(local_uncertainties, dtype=float)),
            "global_object_support_uncertainty_m": numeric_summary(np.asarray(global_uncertainties, dtype=float)),
            "consumed_contact_patch_support_uncertainty_m": numeric_summary(np.asarray(used_uncertainties, dtype=float)),
            "support_uncertainty_reduction_m": numeric_summary(np.asarray(support_delta, dtype=float)),
            "local_patch_sample_count": numeric_summary(np.asarray([int(r.get("local_patch_sample_count", 0) or 0) for r in output_rows], dtype=float)),
            "local_patch_temporal_sample_count": numeric_summary(np.asarray([int(r.get("local_patch_temporal_sample_count", 0) or 0) for r in output_rows], dtype=float)),
            "solver_residual_patch_vertex_count": numeric_summary(np.asarray(solver_patch_counts, dtype=float)),
            "local_sample_to_valid_object_depth_pixel_ratio": numeric_summary(np.asarray(sample_to_valid_depth_ratios, dtype=float)),
            "local_patch_plane_breadth_p95_m": numeric_summary(np.asarray(plane_breadths, dtype=float)),
            "frames": sorted({int(r["frame_idx"]) for r in output_rows}),
            "sides": sorted({str(r["hand_side"]) for r in output_rows}),
        },
        "factor_rows": output_rows,
    }
    return report


def main() -> None:
    args = parse_args()
    report = build(args)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
