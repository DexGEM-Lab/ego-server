#!/usr/bin/env python3
"""Render a V19 rigid-object state as full-duration physical artifacts.

The renderer consumes the explicit state JSON produced by
``build_v19_rigid_render_state.py``.  Unlike the earlier compact-rigid diagnostic
renderer, this script rasterizes mesh faces as a visible rigid body and applies
an explicit source-size -> render-size intrinsics scaling rule before projecting
onto each decoded raw frame.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Optional source frame_idx values for a diagnostic subset render.")
    parser.add_argument("--mesh-face-budget", type=int, default=50000)
    parser.add_argument("--world-face-budget", type=int, default=50000)
    parser.add_argument("--wireframe-face-budget", type=int, default=1800)
    parser.add_argument("--surface-alpha", type=float, default=0.50)
    parser.add_argument("--world-view", choices=("local", "global"), default="local")
    parser.add_argument("--local-world-padding-m", type=float, default=0.08)
    parser.add_argument("--render-style", choices=("diagnostic", "presentation"), default="diagnostic", help="diagnostic preserves detailed state labels; presentation keeps the same state but reduces overpaint/text clutter for user-facing review.")
    parser.add_argument("--presentation-surface-alpha", type=float, default=0.28, help="Overlay object opacity used by --render-style presentation.")
    parser.add_argument("--presentation-world-alpha", type=float, default=0.42, help="World-view object opacity used by --render-style presentation.")
    parser.add_argument("--presentation-wireframe-face-budget", type=int, default=600, help="Wireframe face budget used by --render-style presentation.")
    parser.add_argument("--presentation-mesh-face-budget", type=int, default=3500, help="Filled overlay mesh face cap used by --render-style presentation; <=0 disables the presentation cap.")
    parser.add_argument("--presentation-world-face-budget", type=int, default=3500, help="Filled world mesh face cap used by --render-style presentation; <=0 disables the presentation cap.")
    parser.add_argument("--path-rewrite", action="append", default=[], metavar="OLD=NEW")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise RuntimeError(f"invalid --path-rewrite {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        if not old:
            raise RuntimeError(f"invalid --path-rewrite {item!r}; OLD is empty")
        out.append((old.rstrip("/"), new.rstrip("/")))
    return out


def rewrite_path(path: Path | str | None, rewrites: list[tuple[str, str]]) -> Path | None:
    if path is None:
        return None
    text = str(path)
    for old, new in rewrites:
        if text == old or text.startswith(old + "/"):
            text = new + text[len(old):]
            break
    return Path(text)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mesh_geom = trimesh.load(path, process=False)
    if isinstance(mesh_geom, trimesh.Scene):
        sub_meshes = [m for m in mesh_geom.geometry.values() if isinstance(m, trimesh.Trimesh) and len(m.vertices) and len(m.faces)]
        if not sub_meshes:
            raise RuntimeError(f"no triangular mesh geometry in scene: {path}")
        mesh_geom = trimesh.util.concatenate(sub_meshes)
    if not isinstance(mesh_geom, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type from {path}: {type(mesh_geom)}")
    vertices = np.asarray(mesh_geom.vertices, dtype=np.float64)
    faces = np.asarray(mesh_geom.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid vertices from {path}: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid triangular faces from {path}: {faces.shape}")
    return vertices, faces, {"vertices": int(len(vertices)), "faces": int(len(faces)), "path": str(path)}


def pose_map(state: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray, str]]:
    pose_state = state.get("object_pose_trajectory") if isinstance(state.get("object_pose_trajectory"), dict) else {}
    rows = pose_state.get("pose_rows") if isinstance(pose_state.get("pose_rows"), list) else []
    out: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        frame_idx = int(raw["frame_idx"])
        rot = np.asarray(raw["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        trans = np.asarray(raw["translation_world_m"], dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,):
            raise RuntimeError(f"invalid pose row in render state for frame {frame_idx}")
        out[frame_idx] = (rot, trans, str(raw.get("status", "unknown_pose_status")))
    return out


def constraint_map(state: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    block = state.get("mano_constraint_state") if isinstance(state.get("mano_constraint_state"), dict) else {}
    rows = block.get("constraint_rows") if isinstance(block.get("constraint_rows"), list) else []
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in rows:
        if isinstance(raw, dict) and "frame_idx" in raw and "hand_side" in raw:
            out[(int(raw["frame_idx"]), str(raw["hand_side"]))] = raw
    return out


def temporal_map(state: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any] | None]:
    block = state.get("temporal_mano_state") if isinstance(state.get("temporal_mano_state"), dict) else {}
    payload = block.get("payload") if isinstance(block.get("payload"), dict) else None
    out: dict[tuple[int, str], dict[str, Any]] = {}
    if payload is None:
        return out, None
    rows = payload.get("per_frame_states") if isinstance(payload.get("per_frame_states"), list) else []
    for raw in rows:
        if isinstance(raw, dict) and "frame_idx" in raw and "hand_side" in raw:
            out[(int(raw["frame_idx"]), str(raw["hand_side"]))] = raw
    return out, payload


def hidden_validation_map(state: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    block = state.get("hidden_volume_validation") if isinstance(state.get("hidden_volume_validation"), dict) else {}
    payload = block.get("payload") if isinstance(block.get("payload"), dict) else None
    out: dict[int, dict[str, Any]] = {}
    if payload is None:
        return out, None
    rows = payload.get("frame_rows") if isinstance(payload.get("frame_rows"), list) else []
    for raw in rows:
        if isinstance(raw, dict) and "frame_idx" in raw:
            out[int(raw["frame_idx"])] = raw
    return out, payload


def frame_id(frame: dict[str, Any], fallback: int) -> int:
    return int(frame.get("frame_idx", fallback))


def raw_intrinsics(frame: dict[str, Any]) -> tuple[float, float, float, float] | None:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    for key in ("intrinsics_fx_fy_cx_cy", "current_v18_camera_intrinsics_fx_fy_cx_cy"):
        value = camera.get(key)
        if isinstance(value, list) and len(value) == 4:
            return tuple(float(x) for x in value)
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        for key in ("v19_camera_intrinsics_fx_fy_cx_cy", "current_v18_camera_intrinsics_fx_fy_cx_cy"):
            value = metric.get(key)
            if isinstance(value, list) and len(value) == 4:
                return tuple(float(x) for x in value)
    return None


def scaled_intrinsics_for_frame(
    frame: dict[str, Any],
    image_w: int,
    image_h: int,
    raw_video: dict[str, Any] | None,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    intr = raw_intrinsics(frame)
    if intr is None:
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks camera/hand intrinsics for projection")
    fx, fy, cx, cy = intr
    if not all(math.isfinite(v) and v > 0.0 for v in (fx, fy, cx, cy)):
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid intrinsics {intr}")
    source_w = int(frame.get("source_width") or 0)
    source_h = int(frame.get("source_height") or 0)
    if (source_w <= 0 or source_h <= 0) and isinstance(raw_video, dict):
        source_w = int(raw_video.get("width") or 0)
        source_h = int(raw_video.get("height") or 0)
    if source_w <= 0 or source_h <= 0:
        # This is allowed only when the K already appears to be in decoded-image coordinates.
        if cx > image_w or cy > image_h or fx > 3.0 * image_w or fy > 3.0 * image_h:
            raise RuntimeError(
                f"frame {frame.get('frame_idx')} lacks source size and K={intr} is not in decoded-image coordinates {image_w}x{image_h}"
            )
        source_w, source_h = image_w, image_h
        source_note = "source_size_missing_assumed_decoded_image_size"
    else:
        source_note = "source_size_from_annotation_or_raw_video"
    sx = float(image_w) / float(source_w)
    sy = float(image_h) / float(source_h)
    scaled = (fx * sx, fy * sy, cx * sx, cy * sy)
    return scaled, {
        "raw_intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy],
        "scaled_intrinsics_fx_fy_cx_cy": [float(x) for x in scaled],
        "source_size": [int(source_w), int(source_h)],
        "render_size": [int(image_w), int(image_h)],
        "scale_xy": [float(sx), float(sy)],
        "source_note": source_note,
    }


def world_points_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (points_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def project_camera_points(points_camera: np.ndarray, intr: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intr
    z = points_camera[:, 2].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.01)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv[valid, 0] = fx * points_camera[valid, 0] / z[valid] + cx
        uv[valid, 1] = fy * points_camera[valid, 1] / z[valid] + cy
        valid = valid & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -width) & (uv[:, 0] < 2 * width) & (uv[:, 1] >= -height) & (uv[:, 1] < 2 * height)
    return uv[:, 0], uv[:, 1], z, valid


def choose_face_ids(face_count: int, budget: int) -> np.ndarray:
    if int(budget) <= 0 or face_count <= int(budget):
        return np.arange(face_count, dtype=np.int64)
    return np.linspace(0, face_count - 1, int(budget), dtype=np.int64)


def rasterize_image_mesh(
    image: np.ndarray,
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    *,
    face_budget: int,
    wire_budget: int,
    color: tuple[int, int, int],
    alpha: float,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    if len(faces) == 0:
        return {"rasterized_faces": 0, "rasterized_pixels": 0}
    valid_face = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.01, axis=1)
    candidate_ids = np.flatnonzero(valid_face)
    if len(candidate_ids) == 0:
        return {"rasterized_faces": 0, "rasterized_pixels": 0}
    selected_local = choose_face_ids(len(candidate_ids), int(face_budget))
    face_ids = candidate_ids[selected_local]
    order = np.argsort(z[faces[face_ids]].mean(axis=1))[::-1]
    layer = image.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    rasterized = 0
    base = np.asarray(color, dtype=np.float64)
    for face_id in face_ids[order]:
        tri_uv = uv[faces[int(face_id)]]
        if np.any(tri_uv[:, 0] < -width) or np.any(tri_uv[:, 0] > 2 * width):
            continue
        if np.any(tri_uv[:, 1] < -height) or np.any(tri_uv[:, 1] > 2 * height):
            continue
        poly = np.round(tri_uv).astype(np.int32)
        if np.unique(poly, axis=0).shape[0] < 3:
            continue
        # Depth shading gives a body surface instead of a flat opaque sticker.
        z_mean = float(np.mean(z[faces[int(face_id)]]))
        shade = 0.70 + 0.30 * (1.0 / max(1.0, z_mean))
        shaded = tuple(int(np.clip(c * min(shade, 1.25), 0, 255)) for c in base)
        cv2.fillConvexPoly(layer, poly, shaded, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_AA)
        rasterized += 1
    object_pixels = mask > 0
    if np.any(object_pixels):
        blended = cv2.addWeighted(layer, float(alpha), image, 1.0 - float(alpha), 0)
        image[object_pixels] = blended[object_pixels]
    if rasterized > 0 and int(wire_budget) != 0:
        edge_ids = face_ids[choose_face_ids(len(face_ids), int(wire_budget))]
        for face_id in edge_ids:
            poly = np.round(uv[faces[int(face_id)]]).astype(np.int32)
            if np.any(poly[:, 0] < -width) or np.any(poly[:, 0] > 2 * width) or np.any(poly[:, 1] < -height) or np.any(poly[:, 1] > 2 * height):
                continue
            cv2.polylines(image, [poly], True, (20, 120, 50), 1, cv2.LINE_AA)
    return {"rasterized_faces": int(rasterized), "rasterized_pixels": int(np.count_nonzero(object_pixels))}


def world_to_screen(point_world: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    extent = np.maximum(max_xyz - min_xyz, 1.0e-6)
    x = int(round((point_world[0] - min_xyz[0]) / extent[0] * width))
    y = int(round(height - (point_world[2] - min_xyz[2]) / extent[2] * height))
    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def world_uv(points_world: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    extent = np.maximum(max_xyz - min_xyz, 1.0e-6)
    uv = np.zeros((len(points_world), 2), dtype=np.float64)
    uv[:, 0] = (points_world[:, 0] - min_xyz[0]) / extent[0] * width
    uv[:, 1] = height - (points_world[:, 2] - min_xyz[2]) / extent[2] * height
    return uv


def rasterize_world_mesh(
    image: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    *,
    face_budget: int,
    wire_budget: int,
    alpha: float,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    uv = world_uv(vertices_world, min_xyz, max_xyz, width, height)
    z_order = vertices_world[:, 1]
    return rasterize_image_mesh(
        image,
        uv,
        np.maximum(z_order - np.min(z_order) + 1.0, 0.01),
        faces,
        face_budget=face_budget,
        wire_budget=wire_budget,
        color=(40, 255, 80),
        alpha=float(alpha),
    )


def constraint_style(state: str) -> tuple[tuple[int, int, int], int, str]:
    if "not_applied" in state or "candidate" in state:
        return (0, 255, 255), 4, "CONSTRAINT CONFLICT"
    if "uncertainty" in state:
        return (0, 200, 255), 3, "UNCERTAIN"
    if state == "no_penetration_no_coordinate_change_needed":
        return (180, 180, 180), 2, "no conflict"
    return (150, 150, 150), 2, "not measured"


def summary_stat(report: Any, name: str = "median") -> float | None:
    if isinstance(report, dict):
        value = report.get(name)
        if value is None and name != "median":
            value = report.get("median")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    try:
        return float(report)
    except (TypeError, ValueError):
        return None


def fmt_mm(value_m: float | None) -> str:
    return "?" if value_m is None else f"{value_m * 1000.0:.0f}mm"


def fmt_px(value_px: float | None) -> str:
    return "?" if value_px is None else f"{value_px:.0f}px"


def temporal_hypothesis_promotes_metric_mano(temporal: dict[str, Any]) -> bool:
    """Return whether an interval row may be drawn as an accepted metric hand.

    Contact-coupled interval optimizers can reduce an object-surface residual by
    moving the hand in ways that corrupt the camera-coordinate MANO metric.  A
    presentation render must therefore require an explicit promotion flag before
    drawing the optimized full skeleton as if it were the hand state.  Diagnostic
    renders still expose the optimized skeleton for debugging.
    """
    policy = str(temporal.get("joint_state_policy") or "")
    state = str(temporal.get("temporal_mano_state") or "")
    if "metric_mano_preserved" in policy:
        return False
    promote_tokens = (
        "metric_mano_promoted",
        "accepted_metric_mano",
        "accepted_mano_correction",
        "metric_correction_accepted",
    )
    return any(token in policy or token in state for token in promote_tokens)


def temporal_contact_label(temporal: dict[str, Any], *, presentation: bool) -> tuple[str, str, str]:
    contact = temporal.get("contact_similarity_refit") if isinstance(temporal.get("contact_similarity_refit"), dict) else {}
    mode = str(contact.get("contact_residual_mode") or "contact")
    policy = str(temporal.get("joint_state_policy") or "")
    normal = summary_stat(contact.get("contact_normal_abs_after_m"), "median")
    tangent = summary_stat(contact.get("contact_tangent_after_m"), "median")
    distance = summary_stat(contact.get("contact_distance_after_m"), "median")
    source_distance = summary_stat(contact.get("contact_distance_before_m"), "median")
    source_normal = summary_stat(contact.get("contact_normal_abs_before_m"), "median")
    shift = summary_stat(temporal.get("metric_joint_shift_px") or temporal.get("visible_joint_shift_px"), "median")
    likelihood = temporal.get("contact_likelihood") if isinstance(temporal.get("contact_likelihood"), dict) else contact.get("contact_likelihood") if isinstance(contact.get("contact_likelihood"), dict) else {}
    contact_prob = summary_stat(likelihood.get("contact_compatibility_score") or likelihood.get("contact_compatibility_probability"), "median") if likelihood else None
    source_gap_z = summary_stat(likelihood.get("source_gap_z"), "median") if likelihood else None
    likelihood_note = ""
    if contact_prob is not None:
        likelihood_note = f", compat~{contact_prob:.3f}"
        if source_gap_z is not None:
            likelihood_note += f", z={source_gap_z:.1f}"
    if presentation:
        if "metric_mano_preserved" in policy:
            if mode == "direct_object_surface_posterior":
                text = "source MANO + object-surface interval"
                text2 = f"source gap {fmt_mm(distance)}, normal {fmt_mm(normal)}{likelihood_note}, joint shift {fmt_px(shift)}"
                return text, text2, "magenta=source hand, yellow=object surface, orange=gap; contact not accepted"
            text = "source MANO + uncertain contact surface"
            if mode == "point_to_plane":
                text2 = f"source gap {fmt_mm(source_distance)}, posterior normal {fmt_mm(normal)}, tangent {fmt_mm(tangent)}"
            else:
                text2 = f"source gap {fmt_mm(source_distance)}, posterior contact {fmt_mm(distance)}"
            return text, text2, "metric hand preserved; posterior surface is separate from metric joints"
        if temporal_hypothesis_promotes_metric_mano(temporal):
            text = "accepted interval MANO correction"
            if mode == "point_to_plane":
                text2 = f"normal {fmt_mm(normal)}, tangent {fmt_mm(tangent)}, shift {fmt_px(shift)}"
            else:
                text2 = f"contact {fmt_mm(distance)}, shift {fmt_px(shift)}"
            return text, text2, "yellow skeleton is promoted metric correction"
        text = "source MANO + near-surface hypothesis"
        if mode == "point_to_plane":
            text2 = f"surface normal {fmt_mm(normal)}, tangent {fmt_mm(tangent)}, candidate shift {fmt_px(shift)}"
        else:
            text2 = f"surface contact {fmt_mm(distance)}, candidate shift {fmt_px(shift)}"
        return text, text2, "contact not accepted; optimized skeleton hidden in presentation"
    residual = summary_stat(temporal.get("full_observed_surface_penetration_after_solver_m"), "max")
    if residual is None:
        residual = summary_stat(temporal.get("final_active_constraint_residual_after_solver_m"), "max")
    text = f"INTERVAL MANO UNCERTAIN | {str(temporal.get('temporal_mano_state', 'interval_state'))[:42]}"
    if "metric_mano_preserved" in policy:
        if mode == "direct_object_surface_posterior":
            text2 = f"source_joints_preserved source_gap={fmt_mm(distance)} normal_med={fmt_mm(normal)}{likelihood_note}"
        else:
            text2 = f"source_joints_preserved source_gap={fmt_mm(source_distance)} source_normal={fmt_mm(source_normal)} posterior_normal={fmt_mm(normal)} tangent_med={fmt_mm(tangent)}"
    elif mode == "point_to_plane":
        text2 = f"normal_med={fmt_mm(normal)} tangent_med={fmt_mm(tangent)} shift_med={fmt_px(shift)}"
    else:
        text2 = f"pen_res={fmt_mm(residual)} contact_med={fmt_mm(distance)} shift_med={fmt_px(shift)}"
    return text, text2, ""


def put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.58,
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(origin[0]), int(origin[1])
    (tw, th), baseline = cv2.getTextSize(text, font, float(font_scale), int(thickness))
    height, width = image.shape[:2]
    x0 = max(0, x - 5)
    y0 = max(0, y - th - baseline - 5)
    x1 = min(width, x + tw + 5)
    y1 = min(height, y + baseline + 5)
    if x1 > x0 and y1 > y0:
        layer = image.copy()
        cv2.rectangle(layer, (x0, y0), (x1, y1), bg_color, -1)
        image[y0:y1, x0:x1] = cv2.addWeighted(layer[y0:y1, x0:x1], float(bg_alpha), image[y0:y1, x0:x1], 1.0 - float(bg_alpha), 0)
    cv2.putText(image, text, (x, y), font, float(font_scale), color, int(thickness), cv2.LINE_AA)


def draw_projected_skeleton(image: np.ndarray, joints_camera: np.ndarray, intr: tuple[float, float, float, float], color: tuple[int, int, int], line_width: int) -> None:
    height, width = image.shape[:2]
    u, v, _z, valid = project_camera_points(joints_camera, intr, width, height)
    for a, b in HAND_EDGES:
        if a < len(valid) and b < len(valid) and valid[a] and valid[b]:
            cv2.line(image, (int(round(u[a])), int(round(v[a]))), (int(round(u[b])), int(round(v[b]))), color, line_width, cv2.LINE_AA)


def draw_projected_points(image: np.ndarray, points_camera: np.ndarray, intr: tuple[float, float, float, float], color: tuple[int, int, int], radius: int, max_points: int) -> None:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return
    if len(points) > int(max_points):
        ids = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int32)
        points = points[ids]
    height, width = image.shape[:2]
    u, v, _z, valid = project_camera_points(points, intr, width, height)
    for x, y, ok in zip(u, v, valid):
        if ok and 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(round(x)), int(round(y))), int(radius), color, -1, cv2.LINE_AA)


def draw_world_points(image: np.ndarray, points_world: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, color: tuple[int, int, int], radius: int, max_points: int) -> None:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return
    if len(points) > int(max_points):
        ids = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int32)
        points = points[ids]
    for point in points:
        xy = world_to_screen(point, min_xyz, max_xyz, image.shape[1], image.shape[0])
        if xy is not None:
            cv2.circle(image, xy, int(radius), color, -1, cv2.LINE_AA)


def draw_projected_segments(
    image: np.ndarray,
    source_camera: np.ndarray,
    target_camera: np.ndarray,
    intr: tuple[float, float, float, float],
    *,
    line_color: tuple[int, int, int],
    source_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
    max_segments: int,
) -> None:
    source = np.asarray(source_camera, dtype=np.float64)
    target = np.asarray(target_camera, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        return
    count = min(len(source), len(target))
    if count == 0:
        return
    if count > int(max_segments):
        ids = np.linspace(0, count - 1, int(max_segments), dtype=np.int32)
        source = source[ids]
        target = target[ids]
    else:
        source = source[:count]
        target = target[:count]
    height, width = image.shape[:2]
    su, sv, _sz, svalid = project_camera_points(source, intr, width, height)
    tu, tv, _tz, tvalid = project_camera_points(target, intr, width, height)
    for x0, y0, ok0, x1, y1, ok1 in zip(su, sv, svalid, tu, tv, tvalid):
        if ok0 and ok1 and 0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height:
            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))
            cv2.line(image, p0, p1, line_color, 1, cv2.LINE_AA)
            cv2.circle(image, p0, 2, source_color, -1, cv2.LINE_AA)
            cv2.circle(image, p1, 2, target_color, -1, cv2.LINE_AA)


def draw_world_segments(
    image: np.ndarray,
    source_world: np.ndarray,
    target_world: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    *,
    line_color: tuple[int, int, int],
    source_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
    max_segments: int,
) -> None:
    source = np.asarray(source_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        return
    count = min(len(source), len(target))
    if count == 0:
        return
    if count > int(max_segments):
        ids = np.linspace(0, count - 1, int(max_segments), dtype=np.int32)
        source = source[ids]
        target = target[ids]
    else:
        source = source[:count]
        target = target[:count]
    for a, b in zip(source, target):
        p0 = world_to_screen(a, min_xyz, max_xyz, image.shape[1], image.shape[0])
        p1 = world_to_screen(b, min_xyz, max_xyz, image.shape[1], image.shape[0])
        if p0 is not None and p1 is not None:
            cv2.line(image, p0, p1, line_color, 1, cv2.LINE_AA)
            cv2.circle(image, p0, 2, source_color, -1, cv2.LINE_AA)
            cv2.circle(image, p1, 2, target_color, -1, cv2.LINE_AA)


def draw_world_skeleton(image: np.ndarray, joints_world: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, color: tuple[int, int, int], line_width: int) -> None:
    height, width = image.shape[:2]
    for a, b in HAND_EDGES:
        if a >= len(joints_world) or b >= len(joints_world):
            continue
        pa = world_to_screen(joints_world[a], min_xyz, max_xyz, width, height)
        pb = world_to_screen(joints_world[b], min_xyz, max_xyz, width, height)
        if pa is not None and pb is not None:
            cv2.line(image, pa, pb, color, line_width, cv2.LINE_AA)


def apply_temporal_hypothesis(joints_world: np.ndarray, temporal: dict[str, Any]) -> np.ndarray | None:
    contact = temporal.get("contact_similarity_refit") if isinstance(temporal.get("contact_similarity_refit"), dict) else {}
    if contact.get("contact_residual_mode") == "direct_object_surface_posterior":
        return None
    articulated_joints = np.asarray(temporal.get("optimized_joints_world_m") or [], dtype=float)
    if articulated_joints.shape == (21, 3):
        if joints_world.shape == (21, 3) and np.allclose(articulated_joints, joints_world, rtol=0.0, atol=1.0e-10):
            return None
        return articulated_joints
    delta_world = np.asarray(temporal.get("optimized_translation_world_m") or [], dtype=float)
    if delta_world.shape != (3,):
        return None
    candidate = joints_world + delta_world[None, :]
    rotation_world = np.asarray(temporal.get("optimized_rotation_vector_world_rad") or [], dtype=float)
    center_world = np.asarray(temporal.get("hand_center_world_m") or [], dtype=float)
    if rotation_world.shape == (3,) and center_world.shape == (3,):
        candidate = candidate + np.cross(rotation_world[None, :], joints_world - center_world[None, :])
    return candidate


def collect_world_bounds(frames: list[dict[str, Any]], vertices: np.ndarray, faces_by_frame: dict[int, tuple[np.ndarray, np.ndarray, str]]) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for pos, frame in enumerate(frames):
        idx = frame_id(frame, pos)
        if idx in faces_by_frame:
            rot, trans, _ = faces_by_frame[idx]
            chunks.append(vertices[:: max(1, len(vertices) // 3000)] @ rot.T + trans[None, :])
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
            if joints.shape == (21, 3):
                chunks.append(joints)
    if not chunks:
        raise RuntimeError("cannot render world view: no object poses or hand joints found")
    pts = np.vstack(chunks)
    return pts.min(axis=0) - 0.10, pts.max(axis=0) + 0.10


def frame_world_bounds(
    frame: dict[str, Any],
    idx: int,
    vertices: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray, str]],
    global_min: np.ndarray,
    global_max: np.ndarray,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    if idx in poses:
        rot, trans, _ = poses[idx]
        chunks.append(vertices[:: max(1, len(vertices) // 3000)] @ rot.T + trans[None, :])
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        joints = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
        if joints.shape == (21, 3):
            chunks.append(joints)
    if not chunks:
        return global_min, global_max
    pts = np.vstack(chunks)
    return pts.min(axis=0) - float(padding), pts.max(axis=0) + float(padding)


def encode_video(frame_dir: Path, output_path: Path, fps: float, frame_count: int | None = None) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-threads", "1",
        "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.jpg"),
    ]
    if frame_count is not None:
        cmd.extend(["-frames:v", str(int(frame_count))])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path),
    ])
    subprocess.run(cmd, check=True)


def render(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rewrites = parse_rewrites(list(args.path_rewrite or []))
    render_state_path = rewrite_path(args.render_state, rewrites)
    output_root = rewrite_path(args.output_root, rewrites)
    if render_state_path is None or output_root is None:
        raise RuntimeError("render state/output path resolved to None")
    state = load_json(render_state_path)
    if not isinstance(state, dict):
        raise RuntimeError(f"render state must be a JSON object: {render_state_path}")
    if state.get("status") != "ok":
        raise RuntimeError(f"render state status is not ok: {state.get('status')}")
    annotation_path = rewrite_path((state.get("annotation_backbone") or {}).get("path"), rewrites)
    mesh_path = rewrite_path((state.get("object_geometry") or {}).get("completed_mesh_path"), rewrites)
    if annotation_path is None or mesh_path is None:
        raise RuntimeError("render state lacks annotation path or completed mesh path")
    annotations = load_json(annotation_path)
    if not isinstance(annotations, dict) or not isinstance(annotations.get("frames"), list):
        raise RuntimeError(f"annotations contain no frames: {annotation_path}")
    frames = list(annotations["frames"])
    if args.frames:
        requested = {int(x) for x in args.frames}
        frames = [frame for pos, frame in enumerate(frames) if isinstance(frame, dict) and frame_id(frame, pos) in requested]
    if args.max_frames is not None:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise RuntimeError("no frames selected for rendering")
    vertices, faces, mesh_summary = load_mesh(mesh_path)
    poses = pose_map(state)
    constraints = constraint_map(state)
    temporal_states, temporal_report = temporal_map(state)
    hidden_validation, hidden_report = hidden_validation_map(state)

    case = str(state.get("case") or annotations.get("case") or "v19_case")
    label = str(state.get("object_label") or state.get("object_id") or "rigid_object")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "rigid_object"
    output_case_dir = output_root / case
    overlay_dir = output_case_dir / "overlay_frames"
    world_dir = output_case_dir / "world_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)

    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else None
    fps = float((raw_video or {}).get("fps") or 30.0)
    canvas_w, canvas_h = 1280, 720
    global_min, global_max = collect_world_bounds(frames, vertices, poses)
    presentation = str(args.render_style) == "presentation"
    overlay_alpha = float(args.presentation_surface_alpha if presentation else args.surface_alpha)
    world_alpha = float(args.presentation_world_alpha if presentation else 0.70)
    wireframe_face_budget = int(min(args.wireframe_face_budget, args.presentation_wireframe_face_budget) if presentation else args.wireframe_face_budget)
    mesh_face_budget = int(args.mesh_face_budget)
    world_face_budget = int(args.world_face_budget)
    if presentation and int(args.presentation_mesh_face_budget) > 0:
        mesh_face_budget = min(mesh_face_budget, int(args.presentation_mesh_face_budget))
    if presentation and int(args.presentation_world_face_budget) > 0:
        world_face_budget = min(world_face_budget, int(args.presentation_world_face_budget))
    projection_examples: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"frame {pos} is not a JSON object")
        idx = frame_id(frame, pos)
        raw_path = rewrite_path(frame.get("raw_frame_path", ""), rewrites)
        overlay = cv2.imread(str(raw_path)) if raw_path is not None and raw_path.exists() else None
        if overlay is None:
            raise RuntimeError(f"failed to read raw frame {raw_path} for frame {idx}")
        height, width = overlay.shape[:2]
        intr, intr_report = scaled_intrinsics_for_frame(frame, width, height, raw_video)
        if len(projection_examples) < 8:
            projection_examples.append({"frame_idx": idx, **intr_report})
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        T_world_camera = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or np.eye(4), dtype=np.float64)
        if T_world_camera.shape != (4, 4):
            raise RuntimeError(f"frame {idx} has invalid T_world_camera shape {T_world_camera.shape}")

        object_stats: dict[str, Any] = {"frame_idx": idx, "object_pose_present": idx in poses}
        if idx in poses:
            rot, trans, pose_status = poses[idx]
            vertices_world = vertices @ rot.T + trans[None, :]
            vertices_camera = world_points_to_camera(vertices_world, T_world_camera)
            u, v, z, valid = project_camera_points(vertices_camera, intr, width, height)
            uv = np.c_[u, v]
            mesh_stats = rasterize_image_mesh(
                overlay,
                uv,
                z,
                faces,
                face_budget=mesh_face_budget,
                wire_budget=wireframe_face_budget,
                color=(40, 255, 80),
                alpha=overlay_alpha,
            )
            object_stats.update(mesh_stats)
            object_stats["pose_status"] = pose_status
            object_stats["projected_vertex_count_in_extended_bounds"] = int(np.count_nonzero(valid))
            if presentation:
                object_text = f"{label}: reconstructed rigid body (uncertain)"
                object_text2 = "green mesh = state-driven object; hand contact remains weak/uncertain"
            else:
                object_text = f"{label} BODY mesh {mesh_summary['vertices']}v/{mesh_summary['faces']}f | {pose_status}"
                object_text2 = ""
            color = (40, 255, 80) if mesh_stats.get("rasterized_pixels", 0) else (0, 165, 255)
        else:
            object_text = f"{label}: rigid pose missing in render state"
            object_text2 = ""
            color = (0, 165, 255)
        if presentation:
            put_text_with_bg(overlay, object_text[:95], (12, 30), font_scale=0.46, color=color, thickness=1)
            put_text_with_bg(overlay, object_text2[:105], (12, 54), font_scale=0.40, color=(210, 255, 210), thickness=1)
        else:
            cv2.putText(overlay, object_text[:150], (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        volume_row = hidden_validation.get(idx)
        if volume_row is not None:
            volume_state = str(volume_row.get("state", "hidden_volume_unmeasured"))
            if presentation and "unmeasured" not in volume_state:
                put_text_with_bg(overlay, f"hidden volume {volume_state}"[:110], (12, 78), font_scale=0.36, color=(0, 180, 255), thickness=1)
            elif not presentation:
                cv2.putText(overlay, f"hidden volume {volume_state}"[:130], (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1, cv2.LINE_AA)

        for hand_idx, hand in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
            if not isinstance(hand, dict):
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            side = str(hand.get("hand_side", metric.get("hand_side", "hand")))
            row = constraints.get((idx, side))
            temporal = temporal_states.get((idx, side))
            state_label_raw = str((row or {}).get("candidate_application_state", "not_measured"))
            style_color, line_width, state_label = constraint_style(state_label_raw)
            interval_uncertain = temporal is not None or "uncertainty" in state_label_raw or "not_applied" in state_label_raw or "candidate" in state_label_raw
            penetrating = row.get("penetrating_vertex_count", "?") if row else "?"
            label_y = (88 + hand_idx * 50) if presentation else (92 + hand_idx * 132)
            mano_text3 = ""
            if temporal is not None:
                mano_text, mano_text2, mano_text3 = temporal_contact_label(temporal, presentation=presentation)
                if presentation:
                    text = f"{side} MANO: {mano_text}"
                    text2 = mano_text2
                else:
                    text = f"{side} {mano_text}"
                    text2 = f"{mano_text2} penverts={penetrating}"
                text_color = (0, 180, 255)
            else:
                text = f"{side} {state_label} | {state_label_raw[:48]}"
                text2 = f"penetrating verts={penetrating}"
                text_color = style_color
            if presentation:
                put_text_with_bg(overlay, text[:90], (12, label_y), font_scale=0.38, color=text_color, thickness=1)
                put_text_with_bg(overlay, text2[:105], (12, label_y + 20), font_scale=0.34, color=text_color, thickness=1)
                if mano_text3:
                    put_text_with_bg(overlay, mano_text3[:105], (12, label_y + 38), font_scale=0.32, color=text_color, thickness=1)
            else:
                cv2.putText(overlay, text[:130], (12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 2, cv2.LINE_AA)
                cv2.putText(overlay, text2[:130], (12, label_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)
            joints_camera = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=np.float64)
            if joints_camera.shape == (21, 3):
                if interval_uncertain:
                    draw_projected_skeleton(overlay, joints_camera, intr, (0, 120, 255), max(8, line_width + 4))
                draw_projected_skeleton(overlay, joints_camera, intr, style_color, line_width)
            if temporal is not None:
                temporal_vertices_world = np.asarray(temporal.get("optimized_vertices_world_sample_m") or [], dtype=np.float64)
                source_contact_world = np.asarray(temporal.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
                target_contact_world = np.asarray(temporal.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
                if temporal_vertices_world.ndim == 2 and temporal_vertices_world.shape[1] == 3 and len(temporal_vertices_world) > 0:
                    temporal_vertices_camera = world_points_to_camera(temporal_vertices_world, T_world_camera)
                    draw_projected_points(overlay, temporal_vertices_camera, intr, (255, 255, 0), 2 if presentation else 2, 220)
                if (
                    source_contact_world.ndim == 2
                    and target_contact_world.ndim == 2
                    and source_contact_world.shape[1] == 3
                    and target_contact_world.shape[1] == 3
                    and len(source_contact_world) > 0
                    and len(target_contact_world) > 0
                ):
                    draw_projected_segments(
                        overlay,
                        world_points_to_camera(source_contact_world, T_world_camera),
                        world_points_to_camera(target_contact_world, T_world_camera),
                        intr,
                        line_color=(255, 180, 40),
                        source_color=(255, 80, 220),
                        target_color=(255, 255, 0),
                        max_segments=28 if presentation else 48,
                    )
                joints_world = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
                if joints_world.shape == (21, 3):
                    candidate_world = apply_temporal_hypothesis(joints_world, temporal)
                    if candidate_world is not None and (not presentation or temporal_hypothesis_promotes_metric_mano(temporal)):
                        candidate_camera = world_points_to_camera(candidate_world, T_world_camera)
                        draw_projected_skeleton(overlay, candidate_camera, intr, (255, 255, 0), 2)
        cv2.imwrite(str(overlay_dir / f"{pos:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        world = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        if args.world_view == "local":
            world_min, world_max = frame_world_bounds(frame, idx, vertices, poses, global_min, global_max, float(args.local_world_padding_m))
            world_label = f"local metric world  frame {idx:04d}"
        else:
            world_min, world_max = global_min, global_max
            world_label = f"global metric world  frame {idx:04d}"
        if idx in poses:
            rot, trans, _ = poses[idx]
            vertices_world = vertices @ rot.T + trans[None, :]
            world_stats = rasterize_world_mesh(
                world,
                vertices_world,
                faces,
                world_min,
                world_max,
                face_budget=world_face_budget,
                wire_budget=wireframe_face_budget,
                alpha=world_alpha,
            )
            object_stats["world_rasterized_pixels"] = world_stats.get("rasterized_pixels", 0)
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            side = str(hand.get("hand_side", metric.get("hand_side", "hand")))
            row = constraints.get((idx, side))
            temporal = temporal_states.get((idx, side))
            state_label_raw = str((row or {}).get("candidate_application_state", "not_measured"))
            style_color, line_width, _ = constraint_style(state_label_raw)
            interval_uncertain = temporal is not None or "uncertainty" in state_label_raw or "not_applied" in state_label_raw or "candidate" in state_label_raw
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or metric.get("joints_world_m") or [], dtype=np.float64)
            if joints_world.shape == (21, 3):
                if interval_uncertain:
                    draw_world_skeleton(world, joints_world, world_min, world_max, (0, 120, 255), max(8, line_width + 4))
                draw_world_skeleton(world, joints_world, world_min, world_max, style_color, max(2, line_width - 1))
                if temporal is not None:
                    temporal_vertices_world = np.asarray(temporal.get("optimized_vertices_world_sample_m") or [], dtype=np.float64)
                    source_contact_world = np.asarray(temporal.get("source_contact_vertices_world_sample_m") or [], dtype=np.float64)
                    target_contact_world = np.asarray(temporal.get("contact_surface_vertices_world_sample_m") or [], dtype=np.float64)
                    if temporal_vertices_world.ndim == 2 and temporal_vertices_world.shape[1] == 3 and len(temporal_vertices_world) > 0:
                        draw_world_points(world, temporal_vertices_world, world_min, world_max, (255, 255, 0), 2, 220)
                    if (
                        source_contact_world.ndim == 2
                        and target_contact_world.ndim == 2
                        and source_contact_world.shape[1] == 3
                        and target_contact_world.shape[1] == 3
                        and len(source_contact_world) > 0
                        and len(target_contact_world) > 0
                    ):
                        draw_world_segments(
                            world,
                            source_contact_world,
                            target_contact_world,
                            world_min,
                            world_max,
                            line_color=(255, 180, 40),
                            source_color=(255, 80, 220),
                            target_color=(255, 255, 0),
                            max_segments=28 if presentation else 48,
                        )
                    candidate_world = apply_temporal_hypothesis(joints_world, temporal)
                    if candidate_world is not None and (not presentation or temporal_hypothesis_promotes_metric_mano(temporal)):
                        draw_world_skeleton(world, candidate_world, world_min, world_max, (255, 255, 0), 2)
        if presentation:
            put_text_with_bg(world, world_label, (20, 30), font_scale=0.48, color=(255, 255, 255), thickness=1, bg_alpha=0.50)
            direct_surface_posterior = any(
                isinstance(t, dict)
                and isinstance(t.get("contact_similarity_refit"), dict)
                and t["contact_similarity_refit"].get("contact_residual_mode") == "direct_object_surface_posterior"
                for (f, _side), t in temporal_states.items()
                if f == idx
            )
            if direct_surface_posterior:
                put_text_with_bg(world, f"green={label} rigid mesh; yellow=object surface, magenta=source hand", (20, canvas_h - 48), font_scale=0.43, color=(210, 255, 210), thickness=1, bg_alpha=0.50)
                put_text_with_bg(world, "orange links show source-gap correspondence; contact not accepted", (20, canvas_h - 22), font_scale=0.40, color=(0, 200, 255), thickness=1, bg_alpha=0.50)
            else:
                put_text_with_bg(world, f"green={label} rigid mesh; cyan/yellow/orange=uncertain surface posterior", (20, canvas_h - 48), font_scale=0.43, color=(210, 255, 210), thickness=1, bg_alpha=0.50)
                put_text_with_bg(world, "metric MANO stays source unless an interval correction is explicitly promoted", (20, canvas_h - 22), font_scale=0.40, color=(0, 200, 255), thickness=1, bg_alpha=0.50)
        else:
            cv2.putText(world, world_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(world, f"green filled surface = rigid object body ({label})", (20, canvas_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 255, 80), 1, cv2.LINE_AA)
        cv2.imwrite(str(world_dir / f"{pos:06d}.jpg"), world, [cv2.IMWRITE_JPEG_QUALITY, 90])
        render_rows.append(object_stats)
        if pos % 120 == 0:
            print(f"rendered frame {pos}/{len(frames)}")

    overlay_video = output_case_dir / f"v19_overlay_{safe_label}.mp4"
    world_video = output_case_dir / f"v19_world_{safe_label}.mp4"
    side_by_side_video = output_case_dir / f"v19_side_by_side_{safe_label}.mp4"
    frame_count = len(frames)
    encode_video(overlay_dir, overlay_video, fps, frame_count=frame_count)
    encode_video(world_dir, world_video, fps, frame_count=frame_count)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1",
            "-i", str(overlay_video), "-i", str(world_video),
            "-filter_complex",
            "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[l];"
            "[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[r];[l][r]hstack=inputs=2[v]",
            "-map", "[v]", "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-crf", "23", str(side_by_side_video),
        ],
        check=True,
    )
    raster_pixels = np.asarray([row.get("rasterized_pixels", 0) for row in render_rows], dtype=np.float64)
    manifest = {
        "status": "ok",
        "method": "render_v19_rigid_state_artifact",
        "case": case,
        "object_id": str(state.get("object_id")),
        "object_label": label,
        "claim_scope": "state-driven rigid body render with mesh-face rasterization and explicit source-to-render intrinsics scaling",
        "inputs": {
            "render_state": str(render_state_path),
            "annotations": str(annotation_path),
            "completed_mesh": str(mesh_path),
        },
        "outputs": {
            "overlay": str(overlay_video),
            "world": str(world_video),
            "side_by_side": str(side_by_side_video),
            "manifest": str(output_case_dir / "v19_rigid_state_render_manifest.json"),
        },
        "rendered_state": {
            "rigid_object_body_rasterized_from_mesh_faces": True,
            "object_pose_source": "render_state.object_pose_trajectory.pose_rows",
            "mesh_source": "render_state.object_geometry.completed_mesh_path",
            "mano_constraint_state_consumed": bool(constraints),
            "temporal_mano_state_consumed": temporal_report is not None,
            "hidden_volume_validation_consumed": hidden_report is not None,
            "world_view": str(args.world_view),
            "render_style": str(args.render_style),
            "surface_alpha": float(overlay_alpha),
            "world_surface_alpha": float(world_alpha),
            "wireframe_face_budget": int(wireframe_face_budget),
            "mesh_face_budget": int(mesh_face_budget),
            "world_face_budget": int(world_face_budget),
        },
        "projection_contract": {
            "rule": "scaled K = raw source-coordinate K times decoded_render_size/source_size per axis",
            "examples": projection_examples,
        },
        "mesh_summary": mesh_summary,
        "evidence": {
            "frames_rendered": len(frames),
            "frames_with_object_pose": int(sum(1 for row in render_rows if row.get("object_pose_present"))),
            "frames_with_rasterized_body_pixels": int(np.count_nonzero(raster_pixels > 0)),
            "rasterized_body_pixels_median": float(np.median(raster_pixels)) if len(raster_pixels) else 0.0,
            "rasterized_body_pixels_min": float(np.min(raster_pixels)) if len(raster_pixels) else 0.0,
            "rasterized_body_pixels_max": float(np.max(raster_pixels)) if len(raster_pixels) else 0.0,
        },
        "visual_inspection_required": True,
        "total_elapsed_s": time.time() - started,
    }
    write_json(output_case_dir / "v19_rigid_state_render_manifest.json", manifest)
    return manifest


def main() -> None:
    manifest = render(parse_args())
    print(json.dumps(manifest, indent=2)[:20000])


if __name__ == "__main__":
    main()
