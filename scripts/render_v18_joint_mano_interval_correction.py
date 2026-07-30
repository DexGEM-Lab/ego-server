#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Render a continuous joint MANO interval correction sequence.

This renderer is intentionally narrow: it shows the visual consequence of the
joint MANO trajectory solver.  It draws original current MANO and the optimized
continuous MANO trajectory over the same raw frames and in a local metric world
view.  It is for consuming the correction as a hand annotation, not for reporting
pipeline containers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

REJECTED_HPRIME_ROOT = "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard"
REJECTED_ANNOTATION_PATH_MARKERS = (
    REJECTED_HPRIME_ROOT,
    "verified_hprime_final",
    "hprime_final",
)


def reject_rejected_annotation_path(path_or_payload: Any, *, context: str) -> None:
    text = str(path_or_payload)
    hits = [marker for marker in REJECTED_ANNOTATION_PATH_MARKERS if marker in text]
    if hits:
        raise ValueError(f"{context} contains rejected H-prime/final-v7 annotation marker(s) {hits}; use sanitized non-H-prime sources")

DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_sanitized_base_for_hprime/"
    "task5_tomato_960/annotations_v18_full.json"
)
DEFAULT_POSE_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json"
)
DEFAULT_MESH = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply"
)
DEFAULT_STATE = Path(
    "/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_v1/task5_tomato_960/"
    "v18_joint_mano_interval_trajectory_state.json"
)
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_render_v1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="task5_tomato_960")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    p.add_argument("--completed-mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--joint-mano-state", type=Path, action="append", default=None)
    p.add_argument("--full-video", action="store_true", help="Render every raw video frame; frames without optimized interval state show only the original MANO/object context.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mesh-stride", type=int, default=18)
    p.add_argument("--vertex-stride", type=int, default=2)
    p.add_argument("--ownership-uncertainty-overlay", action=argparse.BooleanOptionalAction, default=True, help="Mark corrected MANO samples in magenta when raw all-observed object residual exceeds ownership-trusted residual.")
    p.add_argument("--ownership-uncertainty-threshold-m", type=float, default=2.0e-4)
    p.add_argument("--occluded-translation-posterior-report", type=Path, default=None, help="Optional report from build_v18_occluded_hand_translation_posterior.py. Draws camera-z lower/upper posterior skeletons for zero-observation occluded MANO rows.")
    p.add_argument("--padding-m", type=float, default=0.08)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_vertices(path: Path) -> np.ndarray:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [m for m in geom.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(geom)}")
    return np.asarray(geom.vertices, dtype=float)


ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
    "completed_temporal_rigid_pose_uncertain",
}


def pose_map(report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in report.get("pose_rows", []) if isinstance(report, dict) else []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status.startswith("fit_to_visible_depth") and status not in ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES:
            raise ValueError(f"unrecognized visible-depth pose status {status!r} for frame {row.get('frame_idx')}")
        if status not in ACCEPTED_VISIBLE_DEPTH_POSE_STATUSES:
            continue
        r = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or [], dtype=float)
        t = np.asarray(row.get("translation_world_m") or [], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,):
            raise ValueError(
                f"pose row frame={row.get('frame_idx')} status={status!r} has invalid pose shapes "
                f"rotation={r.shape} translation={t.shape}"
            )
        out[int(row["frame_idx"])] = (r, t)
    return out


def state_map(state: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in state.get("per_frame_states", []) if isinstance(state, dict) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


POSTERIOR_FINGERPRINT_FIELDS = (
    "optimized_joints_world_m",
    "optimized_translation_world_m",
    "optimized_root_delta_axis_angle_rad",
    "optimized_hand_pose_delta_axis_angle_rad",
)


def state_fingerprint(row: dict[str, Any]) -> str:
    payload = {field: row.get(field) for field in POSTERIOR_FINGERPRINT_FIELDS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_state_maps(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        out.update(state_map(load_json(path)))
    return out


def load_posterior_report(path: Path | None) -> tuple[dict[str, Any] | None, dict[tuple[int, str], dict[str, Any]]]:
    if path is None:
        return None, {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"occluded translation posterior report is not a JSON object: {path}")
    reject_rejected_annotation_path(payload, context="occluded translation posterior report")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for side_report in payload.get("side_reports", []):
        if not isinstance(side_report, dict):
            continue
        for row in side_report.get("rows", []):
            if not isinstance(row, dict):
                continue
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return payload, out


def resolved_path_equal(a: Any, b: Path) -> bool:
    if not isinstance(a, str) or not a:
        return False
    pa = Path(a)
    try:
        return pa.resolve() == b.resolve()
    except Exception:
        return str(pa) == str(b)


def frame_camera_z_axis_world(frame: dict[str, Any]) -> np.ndarray:
    camera_raw = frame.get("camera")
    camera: dict[str, Any] = camera_raw if isinstance(camera_raw, dict) else {}
    T = np.asarray(camera.get("T_world_camera_metric") or np.eye(4), dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"frame {frame.get('frame_idx')} has invalid T_world_camera_metric")
    axis = T[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError(f"frame {frame.get('frame_idx')} has degenerate camera z axis")
    return axis / norm


def validate_posterior_matches_states(
    posterior_report: dict[str, Any] | None,
    posteriors: dict[tuple[int, str], dict[str, Any]],
    states: dict[tuple[int, str], dict[str, Any]],
    frames_by_idx: dict[int, dict[str, Any]],
    annotations_path: Path,
) -> None:
    if not posteriors:
        return
    if posterior_report is None:
        raise ValueError("posterior rows were loaded without a posterior report payload")
    inputs_raw = posterior_report.get("inputs")
    inputs: dict[str, Any] = inputs_raw if isinstance(inputs_raw, dict) else {}
    posterior_annotations = inputs.get("annotations")
    if not resolved_path_equal(posterior_annotations, annotations_path):
        raise ValueError(
            "occluded translation posterior annotations do not match render annotations: "
            f"posterior={posterior_annotations} render={annotations_path}"
        )
    params_raw = posterior_report.get("parameters")
    params: dict[str, Any] = params_raw if isinstance(params_raw, dict) else {}
    max_translation_raw = params.get("max_translation_m")
    max_translation = 0.045 if max_translation_raw is None else float(max_translation_raw)
    for key, posterior in posteriors.items():
        frame_idx, _side = key
        frame = frames_by_idx.get(frame_idx)
        if frame is None:
            raise ValueError(f"occluded translation posterior row {key} has no matching annotation frame")
        state = states.get(key)
        if state is None:
            raise ValueError(f"occluded translation posterior row {key} has no matching rendered MANO state")
        expected = posterior.get("base_state_fingerprint_sha256")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"occluded translation posterior row {key} lacks base_state_fingerprint_sha256; refusing stale overlay")
        actual = state_fingerprint(state)
        if actual != expected:
            raise ValueError(
                f"occluded translation posterior row {key} does not match rendered MANO state: "
                f"posterior fingerprint {expected} != state fingerprint {actual}"
            )
        state_name = str(posterior.get("posterior_state") or "")
        if state_name == "visible_or_nonzero_observation_row_fixed":
            continue
        deltas = posterior.get("selected_depth_order_final_delta_values_m")
        grid_raw = posterior.get("grid_profile")
        grid: dict[str, Any] = grid_raw if isinstance(grid_raw, dict) else {}
        if not isinstance(deltas, list):
            raise ValueError(f"occluded translation posterior row {key} lacks selected per-vertex deltas")
        if not isinstance(grid.get("selected_residual_energy_unweighted_m2"), list):
            raise ValueError(f"occluded translation posterior row {key} lacks grid residual energies")
        ids = posterior.get("selected_depth_order_selected_vertex_ids")
        depths = posterior.get("selected_depth_order_selected_surface_depth_m")
        if ids is not None and (not isinstance(ids, list) or len(ids) != len(deltas)):
            raise ValueError(f"occluded translation posterior row {key} has selected id/delta length mismatch")
        if depths is not None and (not isinstance(depths, list) or len(depths) != len(deltas)):
            raise ValueError(f"occluded translation posterior row {key} has selected depth/delta length mismatch")
        report_axis = posterior_axis(posterior)
        frame_axis = frame_camera_z_axis_world(frame)
        if report_axis is None or float(np.linalg.norm(report_axis - frame_axis)) > 1.0e-6:
            raise ValueError(f"occluded translation posterior row {key} camera-z axis does not match render annotations")
        trans = np.asarray(posterior.get("optimized_translation_world_m") or [], dtype=float)
        if trans.shape != (3,) or not np.isfinite(trans).all():
            raise ValueError(f"occluded translation posterior row {key} has invalid optimized translation")
        shifts: list[float] = []
        for shift_key in (
            "additional_camera_z_shift_lower_bound_m",
            "additional_camera_z_shift_upper_bound_m",
            "additional_camera_z_shift_representative_m",
            "additional_camera_z_shift_map_m",
        ):
            raw_shift = posterior.get(shift_key)
            if raw_shift is not None:
                shifts.append(float(raw_shift))
        for s in shifts:
            shifted_norm = float(np.linalg.norm(trans + s * report_axis))
            if shifted_norm > max_translation + 1.0e-6:
                raise ValueError(
                    f"occluded translation posterior row {key} shift {s} exceeds max translation sphere: "
                    f"norm={shifted_norm} max={max_translation}"
                )


def posterior_axis(row: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(row, dict):
        return None
    axis = np.asarray(row.get("camera_z_axis_world") or [], dtype=float)
    if axis.shape != (3,):
        return None
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return None
    return axis / norm


def posterior_shift_values(row: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    if str(row.get("posterior_state") or "") == "visible_or_nonzero_observation_row_fixed":
        return None
    raw_lo = row.get("additional_camera_z_shift_lower_bound_m")
    raw_hi = row.get("additional_camera_z_shift_upper_bound_m")
    if raw_lo is None or raw_hi is None:
        return None
    try:
        lo = float(raw_lo)
        hi = float(raw_hi)
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return lo, hi


def project_camera(points_camera: np.ndarray, intr: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intr
    z = points_camera[:, 2]
    valid = z > 1.0e-4
    u = (fx * points_camera[:, 0] / np.maximum(z, 1.0e-6) + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / np.maximum(z, 1.0e-6) + cy).astype(np.int32)
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def colors_for_side(side: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if side == "left":
        return (255, 80, 0), (255, 255, 0)  # original blue, corrected cyan in BGR
    return (0, 120, 255), (0, 255, 255)  # original orange, corrected yellow in BGR


def residual_max_m(st: dict[str, Any], key: str) -> float:
    val = ((st.get(key) or {}).get("max")) if isinstance(st, dict) else None
    return float(val) if isinstance(val, (int, float)) else 0.0


def has_ownership_uncertainty(st: dict[str, Any], threshold_m: float) -> bool:
    trusted = residual_max_m(st, "full_observed_surface_penetration_after_solver_m")
    raw = residual_max_m(st, "full_raw_observed_surface_penetration_after_solver_m")
    return raw > trusted + float(threshold_m)


def has_latent_occlusion_uncertainty(st: dict[str, Any]) -> bool:
    state = str(st.get("hand_observation_visibility_factor_state") or "")
    raw_multiplier = st.get("hand_observation_visibility_weight_multiplier")
    try:
        multiplier = 1.0 if raw_multiplier is None else float(raw_multiplier)
    except Exception:
        multiplier = 1.0
    return state == "active_hand_observation_visibility" and multiplier <= 1.0e-6


def has_surface_support_uncertainty(st: dict[str, Any], threshold_m: float) -> bool:
    try:
        support_unc = float(st.get("observed_surface_support_uncertainty_m") or 0.0)
    except Exception:
        support_unc = 0.0
    trusted = residual_max_m(st, "full_observed_surface_penetration_after_solver_m")
    return support_unc > 0.0 and trusted > float(threshold_m) and trusted <= support_unc + float(threshold_m)


def has_contact_patch_uncertainty(st: dict[str, Any], threshold_m: float) -> bool:
    if str(st.get("contact_patch_factor_state") or "") != "active_contact_patch":
        return False
    try:
        support_unc = float(st.get("contact_patch_support_uncertainty_m") or 0.0)
    except Exception:
        support_unc = 0.0
    posterior = st.get("contact_patch_posterior_probability")
    try:
        posterior_f = 1.0 if posterior is None else float(posterior)
    except Exception:
        posterior_f = 1.0
    return bool(st.get("contact_patch_state_optimized")) or support_unc > float(threshold_m) or posterior_f < 1.0 - 1.0e-6


def draw_skeleton(image: np.ndarray, joints_camera: np.ndarray, intr: tuple[float, float, float, float], color: tuple[int, int, int], width_px: int) -> None:
    h, w = image.shape[:2]
    u, v, valid = project_camera(joints_camera, intr, w, h)
    for a, b in HAND_EDGES:
        if valid[a] and valid[b]:
            cv2.line(image, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), color, width_px)
    for i in range(min(21, len(u))):
        if valid[i]:
            cv2.circle(image, (int(u[i]), int(v[i])), max(2, width_px), color, -1)


def world_bounds(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray]:
    valid = [p.reshape(-1, 3) for p in points if isinstance(p, np.ndarray) and p.size and p.reshape(-1, 3).shape[0] > 0]
    if not valid:
        return np.array([-1, -1, -1], dtype=float), np.array([1, 1, 1], dtype=float)
    allp = np.vstack(valid)
    return allp.min(axis=0) - padding, allp.max(axis=0) + padding


def world_point(p: np.ndarray, mn: np.ndarray, mx: np.ndarray, w: int, h: int) -> tuple[int, int] | None:
    extent = np.maximum(mx - mn, 1.0e-6)
    x = int(round((p[0] - mn[0]) / extent[0] * w))
    y = int(round(h - (p[2] - mn[2]) / extent[2] * h))
    if 0 <= x < w and 0 <= y < h:
        return x, y
    return None


def draw_world_skeleton(image: np.ndarray, joints: np.ndarray, mn: np.ndarray, mx: np.ndarray, color: tuple[int, int, int], width_px: int) -> None:
    h, w = image.shape[:2]
    for a, b in HAND_EDGES:
        pa = world_point(joints[a], mn, mx, w, h)
        pb = world_point(joints[b], mn, mx, w, h)
        if pa is not None and pb is not None:
            cv2.line(image, pa, pb, color, width_px)
    for j in joints:
        p = world_point(j, mn, mx, w, h)
        if p is not None:
            cv2.circle(image, p, max(2, width_px), color, -1)


def encode(frame_dir: Path, out: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)
    ], check=True)


def render(args: argparse.Namespace) -> dict[str, Any]:
    reject_rejected_annotation_path(args.annotations, context="render annotations")
    reject_rejected_annotation_path(args.pose_report, context="render pose report")
    reject_rejected_annotation_path(args.completed_mesh, context="render completed mesh")
    state_paths = list(args.joint_mano_state or [DEFAULT_STATE])
    for state_path in state_paths:
        reject_rejected_annotation_path(state_path, context="render joint MANO state path")
    annotations = load_json(args.annotations)
    reject_rejected_annotation_path(annotations, context="render annotations payload")
    poses = pose_map(load_json(args.pose_report))
    mesh = load_mesh_vertices(args.completed_mesh)
    states = load_state_maps(state_paths)
    posterior_report, posteriors = load_posterior_report(args.occluded_translation_posterior_report)
    frames = [f for f in annotations.get("frames", []) if isinstance(f, dict)]
    frames_by_idx = {int(f["frame_idx"]): f for f in frames}
    validate_posterior_matches_states(posterior_report, posteriors, states, frames_by_idx, args.annotations)
    frame_ids = sorted(frames_by_idx) if bool(args.full_video) else sorted({k[0] for k in states})
    if not frame_ids:
        raise RuntimeError("joint state has no per-frame states")
    case_dir = args.output_root / str(args.case)
    overlay_dir = case_dir / "overlay_frames"
    world_dir = case_dir / "world_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)
    fps = float((annotations.get("raw_video") or {}).get("fps", 30.0))
    rendered = 0
    for out_i, frame_idx in enumerate(frame_ids):
        frame = frames_by_idx[frame_idx]
        raw = cv2.imread(str(frame.get("raw_frame_path", "")))
        if raw is None:
            raw = np.zeros((1080, 1920, 3), dtype=np.uint8)
        overlay = raw.copy()
        height, width = overlay.shape[:2]
        T = np.asarray((frame.get("camera") or {}).get("T_world_camera_metric", np.eye(4)), dtype=float)
        object_points: np.ndarray | None = None
        if frame_idx in poses:
            R, t = poses[frame_idx]
            object_points = mesh[:: max(1, int(args.mesh_stride))] @ R.T + t[None, :]
            cam = world_to_camera(object_points, T)
            # Use first hand intrinsics for object dots.
            intr_any = None
            for hand in frame.get("hands", []):
                intr = ((hand.get("metric_mano_state") or {}).get("current_v18_camera_intrinsics_fx_fy_cx_cy"))
                if isinstance(intr, list) and len(intr) == 4:
                    intr_any = (float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]))
                    break
            if intr_any is not None:
                u, v, valid = project_camera(cam, intr_any, width, height)
                for x, y in zip(u[valid], v[valid]):
                    cv2.circle(overlay, (int(x), int(y)), 1, (40, 210, 60), -1)
        world_chunks: list[np.ndarray] = []
        if object_points is not None:
            world_chunks.append(object_points)
        ownership_uncertain_on_frame = False
        for hand in frame.get("hands", []):
            side = str(hand.get("hand_side"))
            st = states.get((frame_idx, side))
            if st is None and not bool(args.full_video):
                continue
            metric = hand.get("metric_mano_state") or {}
            joints_cam = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if not (isinstance(intr, list) and len(intr) == 4 and joints_cam.shape == (21, 3) and joints_world.shape == (21, 3)):
                continue
            intr_tuple = (float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]))
            original_color, corrected_color = colors_for_side(side)
            draw_skeleton(overlay, joints_cam, intr_tuple, original_color, 4)  # original current MANO
            world_chunks.append(joints_world)
            if st is not None:
                opt_world = np.asarray(st.get("optimized_joints_world_m") or [], dtype=float)
                opt_verts = np.asarray(st.get("optimized_vertices_world_sample_m") or [], dtype=float)
                ownership_uncertain = bool(args.ownership_uncertainty_overlay) and has_ownership_uncertainty(st, float(args.ownership_uncertainty_threshold_m))
                latent_occlusion_uncertain = has_latent_occlusion_uncertainty(st)
                surface_support_uncertain = has_surface_support_uncertainty(st, float(args.ownership_uncertainty_threshold_m))
                contact_patch_uncertain = has_contact_patch_uncertainty(st, float(args.ownership_uncertainty_threshold_m))
                posterior_row = posteriors.get((frame_idx, side))
                posterior_shifts = posterior_shift_values(posterior_row)
                posterior_uncertain = posterior_shifts is not None
                uncertainty_color = (255, 0, 255)
                any_uncertain = ownership_uncertain or latent_occlusion_uncertain or surface_support_uncertain or contact_patch_uncertain or posterior_uncertain
                ownership_uncertain_on_frame = ownership_uncertain_on_frame or any_uncertain
                if opt_world.shape == (21, 3):
                    opt_cam = world_to_camera(opt_world, T)
                    draw_skeleton(overlay, opt_cam, intr_tuple, corrected_color, 3)  # optimized trajectory
                    if posterior_uncertain:
                        axis = posterior_axis(posterior_row)
                        if axis is not None and posterior_shifts is not None:
                            lo, hi = posterior_shifts
                            for shift in (lo, hi):
                                shifted_cam = world_to_camera(opt_world + shift * axis[None, :], T)
                                draw_skeleton(overlay, shifted_cam, intr_tuple, uncertainty_color, 1)
                            world_chunks.append(opt_world + lo * axis[None, :])
                            world_chunks.append(opt_world + hi * axis[None, :])
                    if any_uncertain:
                        draw_skeleton(overlay, opt_cam, intr_tuple, uncertainty_color, 1 if ownership_uncertain and not (latent_occlusion_uncertain or surface_support_uncertain or posterior_uncertain) else 2)
                    world_chunks.append(opt_world)
                if opt_verts.ndim == 2 and opt_verts.shape[1] == 3:
                    vc = world_to_camera(opt_verts[:: max(1, int(args.vertex_stride))], T)
                    u, v, valid = project_camera(vc, intr_tuple, width, height)
                    for x, y in zip(u[valid], v[valid]):
                        cv2.circle(overlay, (int(x), int(y)), 1, corrected_color, -1)
                    if any_uncertain:
                        for x, y in zip(u[valid], v[valid]):
                            cv2.circle(overlay, (int(x), int(y)), 2, uncertainty_color, 1)
                    world_chunks.append(opt_verts)
        cv2.putText(overlay, f"frame {frame_idx}: original left/right = blue/orange; interval H_t hypothesis left/right = cyan/yellow", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
        cv2.putText(overlay, f"frame {frame_idx}: original left/right = blue/orange; interval H_t hypothesis left/right = cyan/yellow", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if ownership_uncertain_on_frame:
            cv2.putText(overlay, "magenta = unresolved ownership/support/contact or occluded-hand posterior interval", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5)
            cv2.putText(overlay, "magenta = unresolved ownership/support/contact or occluded-hand posterior interval", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        cv2.imwrite(str(overlay_dir / f"{out_i:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        world = np.zeros((720, 1280, 3), dtype=np.uint8)
        mn, mx = world_bounds(world_chunks, float(args.padding_m))
        if object_points is not None:
            for p in object_points:
                q = world_point(p, mn, mx, 1280, 720)
                if q is not None:
                    cv2.circle(world, q, 1, (40, 210, 60), -1)
        world_ownership_uncertain = False
        for hand in frame.get("hands", []):
            side = str(hand.get("hand_side"))
            st = states.get((frame_idx, side))
            if st is None and not bool(args.full_video):
                continue
            metric = hand.get("metric_mano_state") or {}
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            original_color, corrected_color = colors_for_side(side)
            if joints_world.shape == (21, 3):
                draw_world_skeleton(world, joints_world, mn, mx, original_color, 3)
            if st is not None:
                opt_world = np.asarray(st.get("optimized_joints_world_m") or [], dtype=float)
                opt_verts = np.asarray(st.get("optimized_vertices_world_sample_m") or [], dtype=float)
                ownership_uncertain = bool(args.ownership_uncertainty_overlay) and has_ownership_uncertainty(st, float(args.ownership_uncertainty_threshold_m))
                latent_occlusion_uncertain = has_latent_occlusion_uncertainty(st)
                surface_support_uncertain = has_surface_support_uncertainty(st, float(args.ownership_uncertainty_threshold_m))
                posterior_row = posteriors.get((frame_idx, side))
                posterior_shifts = posterior_shift_values(posterior_row)
                posterior_uncertain = posterior_shifts is not None
                any_uncertain = ownership_uncertain or latent_occlusion_uncertain or surface_support_uncertain or posterior_uncertain
                world_ownership_uncertain = world_ownership_uncertain or any_uncertain
                if opt_world.shape == (21, 3):
                    draw_world_skeleton(world, opt_world, mn, mx, corrected_color, 2)
                    if posterior_uncertain:
                        axis = posterior_axis(posterior_row)
                        if axis is not None and posterior_shifts is not None:
                            lo, hi = posterior_shifts
                            draw_world_skeleton(world, opt_world + lo * axis[None, :], mn, mx, (255, 0, 255), 1)
                            draw_world_skeleton(world, opt_world + hi * axis[None, :], mn, mx, (255, 0, 255), 1)
                    if any_uncertain:
                        draw_world_skeleton(world, opt_world, mn, mx, (255, 0, 255), 1 if ownership_uncertain and not (latent_occlusion_uncertain or surface_support_uncertain or posterior_uncertain) else 2)
                if opt_verts.ndim == 2 and opt_verts.shape[1] == 3:
                    for p in opt_verts[:: max(1, int(args.vertex_stride))]:
                        q = world_point(p, mn, mx, 1280, 720)
                        if q is not None:
                            cv2.circle(world, q, 1, corrected_color, -1)
                            if any_uncertain:
                                cv2.circle(world, q, 2, (255, 0, 255), 1)
        cv2.putText(world, f"local metric world frame {frame_idx}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if world_ownership_uncertain:
            cv2.putText(world, "magenta = unresolved ownership/support or occluded-hand posterior interval", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imwrite(str(world_dir / f"{out_i:06d}.jpg"), world, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rendered += 1
    stem = "joint_mano_full_video_correction" if bool(args.full_video) else "joint_mano_interval_correction"
    overlay_video = case_dir / f"v18_overlay_{stem}.mp4"
    world_video = case_dir / f"v18_world_{stem}.mp4"
    side_video = case_dir / f"v18_side_by_side_{stem}.mp4"
    encode(overlay_dir, overlay_video, fps)
    encode(world_dir, world_video, fps)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(overlay_video), "-i", str(world_video),
        "-filter_complex", "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[l];[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[r];[l][r]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(side_video)
    ], check=True)
    manifest = {
        "case": args.case,
        "full_video": bool(args.full_video),
        "annotations": str(args.annotations),
        "pose_report": str(args.pose_report),
        "completed_mesh": str(args.completed_mesh),
        "state_paths": [str(p) for p in state_paths],
        "occluded_translation_posterior_report": None if args.occluded_translation_posterior_report is None else str(args.occluded_translation_posterior_report),
        "posterior_state_count": int(len(posteriors)),
        "posterior_render_semantics": "magenta posterior endpoints are hard additional camera-z translation-bound endpoints for zero-observation rows; they are a bounded/conflicted uncertainty envelope, not a calibrated credible interval or hidden-hand reconstruction",
        "optimized_state_count": int(len(states)),
        "frame_count": rendered,
        "frame_ids": frame_ids,
        "overlay_video": str(overlay_video),
        "world_video": str(world_video),
        "side_by_side_video": str(side_video),
    }
    (case_dir / "v18_joint_mano_interval_correction_render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
