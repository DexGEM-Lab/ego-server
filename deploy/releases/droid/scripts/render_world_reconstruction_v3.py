#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from render_mesh_alignment_v3 import project, view_basis
from render_mesh_surface_contact_review_v3 import (
    FrameSource,
    HAND_EDGES,
    draw_contact_patch,
    draw_hand,
    draw_mesh_projection,
    draw_object_mask,
)


STATE_LABELS = {
    "map_observable_measured_geometry": "observable mesh",
    "ambiguous_measured_geometry": "ambiguous mesh",
    "ambiguous_contact_geometry": "contact-ambiguous mesh",
    "completed_geometry": "completed mesh",
    "segmentation_repaired_geometry": "repaired mesh",
}

STATE_COLORS = {
    "map_observable_measured_geometry": (72, 138, 72),
    "ambiguous_measured_geometry": (70, 118, 186),
    "ambiguous_contact_geometry": (72, 72, 198),
    "completed_geometry": (154, 95, 42),
    "segmentation_repaired_geometry": (42, 142, 154),
}


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise RuntimeError("cannot normalize degenerate vector")
    return np.asarray(vector, dtype=float) / norm


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def monkeypatch_chumpy_numpy() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def load_mano_faces(path: Path) -> np.ndarray:
    monkeypatch_chumpy_numpy()
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    faces = np.asarray(data.get("f"), dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0:
        raise RuntimeError(f"invalid MANO face topology in {path}")
    return faces


def load_state_rows(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    data = load_json(path)
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"V5 state JSON has no rows list: {path}")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            raise RuntimeError(f"invalid V5 state row in {path}")
        out[int(row["frame_idx"])] = row
    return out


def load_append_rows(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    data = load_json(path)
    rows = data.get("rows")
    if rows is None:
        rows = data.get("output_frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"append report has no rows/output_frames list: {path}")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            raise RuntimeError(f"invalid append report row in {path}")
        out[int(row["frame_idx"])] = row
    return out


def load_dynamics_rows(path: Path | None) -> tuple[dict[int, dict], dict]:
    if path is None:
        return {}, {}
    data = load_json(path)
    observations = data.get("observations")
    after = data.get("after")
    if not isinstance(observations, list) or not isinstance(after, dict):
        raise RuntimeError(f"dynamics report lacks observations/after objects: {path}")
    object_points = np.asarray(after.get("object_contact_point_world_m", []), dtype=float)
    gaps = np.asarray(after.get("hand_contact_gap_world_m", []), dtype=float)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise RuntimeError(f"dynamics report has invalid object contact points: {path}")
    if gaps.shape != object_points.shape or len(observations) != len(object_points):
        raise RuntimeError(f"dynamics report observation/state count mismatch: {path}")
    edge_by_source = {int(row["source_frame"]): row for row in after.get("edge_rows", [])}
    edge_by_target = {int(row["target_frame"]): row for row in after.get("edge_rows", [])}
    handoff_by_source = {int(row["source_frame"]): row for row in after.get("handoff_rows", [])}
    switch_by_source = {int(row["source_frame"]): row for row in after.get("switch_rows", [])}
    switch_surface_by_source = {int(row["source_frame"]): row for row in after.get("switch_surface_rows", [])}
    acc_by_center = {int(row["center_frame"]): row for row in after.get("acceleration_rows", [])}
    out = {}
    for i, obs in enumerate(observations):
        frame = int(obs["frame_idx"])
        out[frame] = {
            "frame_idx": frame,
            "selected_patch_region": str(obs.get("selected_patch_region")),
            "object_contact_point_world_m": object_points[i].astype(float).tolist(),
            "hand_contact_point_world_m": (object_points[i] + gaps[i]).astype(float).tolist(),
            "contact_gap_m": float(np.linalg.norm(gaps[i])),
            "edge": edge_by_source.get(frame),
            "incoming_edge": edge_by_target.get(frame),
            "handoff": handoff_by_source.get(frame),
            "switch": switch_by_source.get(frame),
            "switch_surface": switch_surface_by_source.get(frame),
            "acceleration": acc_by_center.get(frame),
        }
    return out, data


def reliable_contact_rows(contact: dict) -> dict[int, dict]:
    rows = [
        row
        for row in contact.get("rows_detail", [])
        if bool(row.get("reliable_for_contact", False))
        or bool(row.get("geometry_backed_temporal_contact", False))
    ]
    return {int(row["frame_idx"]): row for row in rows}


def world_joints(hand: dict) -> np.ndarray:
    arr = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        raise RuntimeError("hand lacks finite 21x3 joints3d_world_m")
    return arr


def world_vertices(hand: dict) -> np.ndarray:
    arr = np.asarray(hand.get("vertices_world_m", []), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise RuntimeError("hand lacks finite vertices_world_m")
    return arr


def camera_frustum_points(t_world_camera: np.ndarray, scale: float) -> np.ndarray:
    width = 0.65 * scale
    height = 0.42 * scale
    points_camera = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-width, -height, scale],
            [width, -height, scale],
            [width, height, scale],
            [-width, height, scale],
        ],
        dtype=float,
    )
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=float)]
    return (t_world_camera @ homog.T).T[:, :3]


def draw_polyline_3d(
    image: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool = False,
) -> None:
    xy, _ = project(points, center, basis, radius, (image.shape[1], image.shape[0]))
    pts = xy.astype(np.int32)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(image, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
    if closed and len(pts) > 2:
        cv2.line(image, tuple(pts[-1]), tuple(pts[0]), color, thickness, cv2.LINE_AA)


def simplify_mesh_for_display(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if max_faces <= 0 or len(faces) <= max_faces:
        return vertices, faces
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    out_vertices = np.asarray(mesh.vertices, dtype=float)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    if out_vertices.ndim != 2 or out_vertices.shape[1] != 3 or out_faces.ndim != 2 or out_faces.shape[1] != 3:
        raise RuntimeError("display mesh simplification produced invalid geometry")
    if len(out_faces) == 0:
        raise RuntimeError("display mesh simplification produced no faces")
    return out_vertices, out_faces


def draw_triangle_mesh_world(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    max_faces: int,
    base_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
    alpha: float,
    edge_count: int,
    shadow: bool,
) -> None:
    vertices, faces = simplify_mesh_for_display(vertices, faces, max_faces)
    face_ids = np.arange(len(faces), dtype=int)
    xy, depth = project(vertices, center, basis, radius, (image.shape[1], image.shape[0]))
    if shadow:
        hull = cv2.convexHull(xy.astype(np.float32)).astype(np.int32)
        shadow_poly = hull + np.asarray([14, 16], dtype=np.int32)[None, None, :]
        shadow_overlay = image.copy()
        cv2.fillConvexPoly(shadow_overlay, shadow_poly, (210, 213, 206), cv2.LINE_AA)
        cv2.addWeighted(shadow_overlay, 0.46, image, 0.54, 0.0, image)
    face_depth = depth[faces[face_ids]].mean(axis=1)
    order = face_ids[np.argsort(face_depth)]
    overlay = image.copy()
    face_vertices = vertices[faces[order]]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_norm[:, None], 1e-12)
    light = unit(-0.70 * basis[2] - 0.45 * basis[1] + 0.22 * basis[0])
    shade = 0.48 + 0.48 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray(base_bgr, dtype=float)
    for rank, face_id in enumerate(order):
        poly = xy[faces[int(face_id)]]
        if np.any(poly[:, 0] < -image.shape[1]) or np.any(poly[:, 0] > 2 * image.shape[1]):
            continue
        if np.any(poly[:, 1] < -image.shape[0]) or np.any(poly[:, 1] > 2 * image.shape[0]):
            continue
        color = tuple(np.clip(base * shade[rank], 0, 255).astype(np.uint8).tolist())
        cv2.fillConvexPoly(overlay, poly.astype(np.int32), color, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, image)
    hull = cv2.convexHull(xy.astype(np.float32)).astype(np.int32)
    cv2.polylines(image, [hull], True, edge_bgr, 2, cv2.LINE_AA)
    if edge_count > 0:
        edge_ids = order[np.linspace(0, len(order) - 1, min(len(order), edge_count), dtype=int)]
        for face_id in edge_ids:
            poly = xy[faces[int(face_id)]].astype(np.int32)
            cv2.polylines(image, [poly], True, edge_bgr, 1, cv2.LINE_AA)


def draw_mesh_world(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    max_faces: int,
) -> None:
    draw_triangle_mesh_world(
        image,
        vertices,
        faces,
        center,
        basis,
        radius,
        max_faces,
        (76, 98, 224),
        (43, 48, 154),
        0.70,
        110,
        True,
    )


def draw_completed_mesh_world(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    append_row: dict | None,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    max_faces: int,
) -> None:
    if append_row is None:
        draw_mesh_world(image, vertices, faces, center, basis, radius, max_faces)
        return
    observed_faces = int(append_row.get("observed_faces", len(faces)))
    if observed_faces <= 0 or observed_faces >= len(faces):
        draw_mesh_world(image, vertices, faces, center, basis, radius, max_faces)
        return
    draw_triangle_mesh_world(
        image,
        vertices,
        faces[:observed_faces],
        center,
        basis,
        radius,
        max_faces,
        (76, 98, 224),
        (43, 48, 154),
        0.68,
        90,
        True,
    )
    draw_triangle_mesh_world(
        image,
        vertices,
        faces[observed_faces:],
        center,
        basis,
        radius,
        max(200, int(max_faces * 0.45)),
        (155, 174, 226),
        (86, 104, 174),
        0.28,
        38,
        False,
    )


def draw_metric_axes(image: np.ndarray, center: np.ndarray, basis: np.ndarray, radius: float) -> None:
    origin = center - 0.76 * radius * basis[0] - 0.68 * radius * basis[1]
    scale = max(0.035, 0.16 * radius)
    axes = [
        ("X", np.asarray([1.0, 0.0, 0.0]), (40, 40, 210)),
        ("Y", np.asarray([0.0, 1.0, 0.0]), (40, 150, 60)),
        ("Z", np.asarray([0.0, 0.0, 1.0]), (210, 95, 35)),
    ]
    xy0, _ = project(origin[None, :], center, basis, radius, (image.shape[1], image.shape[0]))
    p0 = tuple(xy0[0].astype(int))
    for label, direction, color in axes:
        xy1, _ = project((origin + scale * direction)[None, :], center, basis, radius, (image.shape[1], image.shape[0]))
        p1 = tuple(xy1[0].astype(int))
        cv2.arrowedLine(image, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(image, label, (p1[0] + 4, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_camera_path(
    image: np.ndarray,
    annotations: dict[int, dict],
    frame_idx: int,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    *,
    full_color: tuple[int, int, int] = (122, 122, 116),
    current_color: tuple[int, int, int] = (25, 25, 25),
    full_thickness: int = 2,
    current_thickness: int = 4,
) -> None:
    frames = sorted(annotations)
    path = np.asarray([annotations[f]["camera"]["position_world_m"] for f in frames], dtype=float)
    xy, _ = project(path, center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.polylines(image, [xy.astype(np.int32)], False, full_color, full_thickness, cv2.LINE_AA)
    cur_i = frames.index(int(frame_idx))
    if cur_i > 0:
        cv2.polylines(image, [xy[: cur_i + 1].astype(np.int32)], False, current_color, current_thickness, cv2.LINE_AA)
    cv2.circle(image, tuple(xy[cur_i].astype(int)), max(5, current_thickness + 2), current_color, -1, cv2.LINE_AA)


def draw_camera(
    image: np.ndarray,
    t_world_camera: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    scale: float,
    label: bool = True,
    color: tuple[int, int, int] = (36, 36, 36),
    thickness: int = 2,
) -> None:
    frustum = camera_frustum_points(t_world_camera, scale)
    origin = frustum[0]
    corners = frustum[1:]
    for corner in corners:
        draw_polyline_3d(image, np.vstack([origin, corner]), center, basis, radius, color, thickness)
    draw_polyline_3d(image, np.vstack([corners, corners[0]]), center, basis, radius, color, thickness, closed=False)
    xy, _ = project(frustum[:1], center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.circle(image, tuple(xy[0].astype(int)), max(6, thickness + 4), color, -1, cv2.LINE_AA)
    if label:
        cv2.putText(image, "head camera", tuple((xy[0] + np.asarray([9, -10])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)


def draw_head_legend(image: np.ndarray) -> None:
    x0 = image.shape[1] - 244
    y0 = image.shape[0] - 104
    cv2.rectangle(image, (x0, y0), (image.shape[1] - 24, image.shape[0] - 24), (244, 246, 241), -1, cv2.LINE_AA)
    cv2.line(image, (x0 + 18, y0 + 32), (x0 + 72, y0 + 32), (25, 25, 25), 4, cv2.LINE_AA)
    cv2.circle(image, (x0 + 72, y0 + 32), 7, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.putText(image, "head trajectory", (x0 + 86, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (25, 25, 25), 1, cv2.LINE_AA)
    glyph = np.asarray(
        [
            [x0 + 26, y0 + 58],
            [x0 + 54, y0 + 49],
            [x0 + 58, y0 + 76],
            [x0 + 26, y0 + 58],
            [x0 + 54, y0 + 76],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [glyph], False, (36, 36, 36), 3, cv2.LINE_AA)
    cv2.circle(image, (x0 + 26, y0 + 58), 5, (36, 36, 36), -1, cv2.LINE_AA)
    cv2.putText(image, "head camera", (x0 + 86, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (25, 25, 25), 1, cv2.LINE_AA)


def draw_object_surface_legend(image: np.ndarray, append_row: dict | None) -> None:
    if append_row is None:
        return
    observed_faces = int(append_row.get("observed_faces", 0))
    archive_faces = int(append_row.get("archive_faces", 0))
    if observed_faces <= 0 or archive_faces <= 0:
        return
    x0 = 18
    y0 = 42
    cv2.rectangle(image, (x0 - 10, y0 - 12), (x0 + 278, y0 + 58), (244, 246, 241), -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + 38, y0 + 16), (76, 98, 224), -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0 + 30), (x0 + 38, y0 + 46), (155, 174, 226), -1, cv2.LINE_AA)
    cv2.putText(image, "observed object surface", (x0 + 52, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "completed hidden surface", (x0 + 52, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)


def draw_dynamics_world(
    image: np.ndarray,
    row: dict | None,
    dynamics_summary: dict,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
) -> None:
    if row is None:
        return
    object_point = np.asarray(row["object_contact_point_world_m"], dtype=float)
    hand_point = np.asarray(row["hand_contact_point_world_m"], dtype=float)
    points = np.vstack([object_point, hand_point])
    xy, _ = project(points, center, basis, radius, (image.shape[1], image.shape[0]))
    start = tuple(xy[0].astype(np.int32))
    end = tuple(xy[1].astype(np.int32))
    cv2.circle(image, start, 8, (18, 92, 195), -1, cv2.LINE_AA)
    cv2.circle(image, end, 8, (200, 42, 166), -1, cv2.LINE_AA)
    cv2.arrowedLine(image, start, end, (78, 38, 174), 3, cv2.LINE_AA, tipLength=0.25)
    edge = row.get("edge") or {}
    incoming_edge = row.get("incoming_edge") or {}
    handoff = row.get("handoff") or {}
    switch = row.get("switch") or {}
    switch_surface = row.get("switch_surface") or {}
    acc = row.get("acceleration") or {}
    report_regime = str(dynamics_summary.get("contact_motion_regime", "contact"))
    active_edge = edge or incoming_edge
    regime = "switch" if switch else ("handoff" if handoff else ("sliding" if active_edge else report_regime if row.get("acceleration") else "contact"))
    gap_mm = 1000.0 * float(row.get("contact_gap_m", 0.0))
    motion_cm_s = 100.0 * float(
        handoff.get("object_motion_speed_m_s", active_edge.get("slip_speed_m_s", 0.0))
    )
    motion_label = "switch" if switch else ("handoff" if handoff else "slip")
    handoff_gap_mm = 1000.0 * float(handoff.get("gap_delta_m", 0.0))
    switch_gap_mm = 1000.0 * float(switch.get("gap_delta_m", 0.0))
    switch_surface_mm = 1000.0 * float((switch_surface.get("source_neighborhood_to_target_surface_m") or {}).get("p95", 0.0))
    acc_res = float(acc.get("acceleration_consistency_residual_m_s2", 0.0))
    x0 = image.shape[1] - 316
    y0 = 46
    cv2.rectangle(image, (x0 - 12, y0 - 16), (image.shape[1] - 22, y0 + 86), (244, 246, 241), -1, cv2.LINE_AA)
    cv2.putText(image, f"contact: {regime}", (x0, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, f"patch: {row.get('selected_patch_region')}", (x0, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)
    if switch:
        metric = f"{motion_label} surf p95 {switch_surface_mm:.1f} mm"
    else:
        metric = f"{motion_label} {motion_cm_s:.1f} cm/s"
    cv2.putText(image, f"gap {gap_mm:.2f} mm  {metric}", (x0, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)
    if switch:
        line = f"switch gap delta {switch_gap_mm:.2f} mm"
    elif handoff:
        line = f"handoff gap delta {handoff_gap_mm:.2f} mm"
    else:
        line = f"dyn residual {acc_res:.3f} m/s2"
    cv2.putText(image, line, (x0, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 1, cv2.LINE_AA)


def draw_egocentric_view_ray(
    image: np.ndarray,
    annotations: dict[int, dict],
    frame_idx: int,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
) -> None:
    camera_pose = np.asarray(annotations[int(frame_idx)]["camera"]["T_world_camera_metric"], dtype=float)
    camera_origin = camera_pose[:3, 3]
    focus = center
    direction = focus - camera_origin
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9 or not np.isfinite(distance):
        return
    direction /= distance
    segment_start = focus - min(distance, 0.55 * radius) * direction
    points = np.vstack([segment_start, focus])
    xy, _ = project(points, center, basis, radius, (image.shape[1], image.shape[0]))
    overlay = image.copy()
    cv2.arrowedLine(overlay, tuple(xy[0]), tuple(xy[1]), (54, 54, 54), 3, cv2.LINE_AA, tipLength=0.13)
    xy_focus, _ = project(np.asarray([focus]), center, basis, radius, (image.shape[1], image.shape[0]))
    cv2.circle(overlay, tuple(xy_focus[0]), 6, (54, 54, 54), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, dst=image)


def draw_camera_inset(
    image: np.ndarray,
    annotations: dict[int, dict],
    frame_idx: int,
    args: argparse.Namespace,
) -> None:
    frames = sorted(annotations)
    path = np.asarray([annotations[f]["camera"]["position_world_m"] for f in frames], dtype=float)
    current = np.asarray(annotations[int(frame_idx)]["camera"]["T_world_camera_metric"], dtype=float)
    frustum = camera_frustum_points(current, float(args.frustum_scale_m) * 3.6)
    center, basis, radius = frame_view([path, frustum], 1.65)
    h, w = 156, 238
    x0 = image.shape[1] - w - 24
    y0 = image.shape[0] - h - 24
    panel = np.full((h, w, 3), (252, 253, 249), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 70, 70), 1, cv2.LINE_AA)
    draw_camera_path(panel, annotations, int(frame_idx), center, basis, radius)
    draw_camera(panel, current, center, basis, radius, float(args.frustum_scale_m) * 3.6, label=False)
    cv2.putText(panel, "head path", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 1, cv2.LINE_AA)
    image[y0 : y0 + h, x0 : x0 + w] = panel


def draw_state_badge(image: np.ndarray, state_row: dict | None, frame_idx: int) -> None:
    (tw, _), _ = cv2.getTextSize(f"world reconstruction  frame {frame_idx}", cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
    cv2.rectangle(image, (18, 12), (38 + tw, 44), (244, 246, 241), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"world reconstruction  frame {frame_idx}",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    if state_row is None:
        return
    state = str(state_row.get("geometry_state", "unknown"))
    label = STATE_LABELS.get(state, state)
    color = STATE_COLORS.get(state, (70, 70, 70))
    x0, y0 = 22, 52
    text = f"V5 state: {label}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
    cv2.rectangle(image, (x0, y0), (x0 + min(tw + 28, image.shape[1] - x0 - 24), y0 + 34), color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x0 + 12, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    reasons = state_row.get("state_reasons") or []
    if reasons:
        text = "evidence flags: " + ", ".join(str(item) for item in reasons[:3])
        cv2.putText(image, text, (x0, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)


def draw_scale_bar(image: np.ndarray, radius: float, args: argparse.Namespace) -> None:
    scale_px = int(round(0.10 * 0.42 * min(image.shape[1], image.shape[0]) / radius))
    scale_px = max(28, min(scale_px, 220))
    sx, sy = 32, args.panel_height - 38
    cv2.line(image, (sx, sy), (sx + scale_px, sy), (25, 25, 25), 5, cv2.LINE_AA)
    cv2.putText(image, "0.10 m", (sx, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (25, 25, 25), 1, cv2.LINE_AA)


def draw_hand_world(
    image: np.ndarray,
    hand: dict,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    contact_ids: list[int] | None,
    mano_faces: np.ndarray | None,
    max_mano_faces: int,
) -> None:
    joints = world_joints(hand)
    measured = bool(hand.get("measurement_available", False))
    if not measured:
        return
    side = str(hand.get("side", "unknown"))
    color = (38, 172, 70) if side == "right" else (220, 136, 46)
    vertices = world_vertices(hand)
    if mano_faces is not None:
        if int(mano_faces.max()) >= len(vertices):
            raise RuntimeError("MANO face topology references vertices not present in annotation")
        edge = (26, 120, 46) if side == "right" else (164, 91, 35)
        draw_triangle_mesh_world(
            image,
            vertices,
            mano_faces,
            center,
            basis,
            radius,
            int(max_mano_faces),
            color,
            edge,
            0.34,
            36,
            False,
        )
    xy, _ = project(joints, center, basis, radius, (image.shape[1], image.shape[0]))
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (18, 18, 18), 5, cv2.LINE_AA)
        cv2.line(image, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 3, cv2.LINE_AA)
    for point in xy:
        cv2.circle(image, tuple(point.astype(int)), 5, (18, 18, 18), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(point.astype(int)), 3, color, -1, cv2.LINE_AA)
    if contact_ids:
        contact_vertices = vertices[np.asarray(contact_ids, dtype=int)]
        uv, _ = project(contact_vertices, center, basis, radius, (image.shape[1], image.shape[0]))
        for point in uv:
            p = tuple(point.astype(int))
            cv2.circle(image, p, 13, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, p, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, p, 7, (255, 0, 255), -1, cv2.LINE_AA)


def draw_world_panel(
    annotations: dict[int, dict],
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    append_by_frame: dict[int, dict],
    dynamics_by_frame: dict[int, dict],
    dynamics_summary: dict,
    contact_by_frame: dict[int, dict],
    state_by_frame: dict[int, dict],
    mano_faces: np.ndarray | None,
    frame_idx: int,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    args: argparse.Namespace,
) -> np.ndarray:
    image = np.full((args.panel_height, args.panel_width, 3), (244, 246, 241), dtype=np.uint8)
    ann = annotations[int(frame_idx)]
    vertices, faces = meshes[int(frame_idx)]
    draw_completed_mesh_world(
        image,
        vertices,
        faces,
        append_by_frame.get(int(frame_idx)),
        center,
        basis,
        radius,
        int(args.max_mesh_faces),
    )
    camera_pose = np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float)
    draw_camera_path(image, annotations, int(frame_idx), center, basis, radius, full_thickness=2, current_thickness=4)
    world_frustum_scale = float(args.frustum_scale_m) * float(args.world_frustum_visual_scale)
    draw_camera(image, camera_pose, center, basis, radius, world_frustum_scale, label=False, thickness=2)
    draw_egocentric_view_ray(image, annotations, int(frame_idx), center, basis, radius)
    row = contact_by_frame.get(int(frame_idx))
    for i, hand in enumerate(ann.get("hands", [])):
        ids = row.get("best_patch_vertex_ids", []) if row is not None and int(row["hand_idx"]) == i else None
        draw_hand_world(image, hand, center, basis, radius, ids, mano_faces, int(args.max_mano_faces))
    if bool(args.show_camera_inset):
        draw_camera_inset(image, annotations, int(frame_idx), args)
    draw_dynamics_world(image, dynamics_by_frame.get(int(frame_idx)), dynamics_summary, center, basis, radius)
    draw_head_legend(image)
    draw_object_surface_legend(image, append_by_frame.get(int(frame_idx)))
    draw_metric_axes(image, center, basis, radius)
    draw_scale_bar(image, radius, args)
    draw_state_badge(image, state_by_frame.get(int(frame_idx)), int(frame_idx))
    return image


def frame_view(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.vstack(points)
    center, basis, radius = view_basis(cloud)
    return center, basis, max(radius * padding, 1e-4)


def oblique_frame_view(points: list[np.ndarray], padding: float) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.vstack(points)
    center = np.median(cloud, axis=0)
    centered = cloud - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if np.linalg.det(vh) < 0:
        vh[-1] *= -1.0
    screen_x = unit(vh[0])
    plane_normal = unit(vh[2])
    in_plane = unit(vh[1])
    view_dir = unit(0.58 * plane_normal + 0.82 * in_plane)
    screen_y = unit(np.cross(view_dir, screen_x))
    basis = np.vstack([screen_x, screen_y, view_dir])
    if np.linalg.det(basis) < 0:
        basis[1] *= -1.0
    q = centered @ basis.T
    radius = float(np.max(np.linalg.norm(q[:, :2], axis=1)))
    return center, basis, max(radius * padding, 1e-4)


def current_focus_view(
    annotations: dict[int, dict],
    ann: dict,
    mesh: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = []
    vertices = mesh[0]
    points.append(vertices[np.linspace(0, len(vertices) - 1, min(len(vertices), 900), dtype=int)])
    for hand in ann.get("hands", []):
        if bool(hand.get("measurement_available", False)):
            hand_vertices = world_vertices(hand)
            points.append(hand_vertices[np.linspace(0, len(hand_vertices) - 1, min(len(hand_vertices), 240), dtype=int)])
    if len(points) == 1:
        for hand in ann.get("hands", []):
            points.append(world_joints(hand))
    center, basis, radius = oblique_frame_view(points, float(args.focus_radius_scale))
    if bool(getattr(args, "include_camera_in_focus", False)):
        camera_pose = np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float)
        frustum = camera_frustum_points(camera_pose, float(args.frustum_scale_m) * float(args.world_frustum_visual_scale))
        path = np.asarray([annotations[f]["camera"]["position_world_m"] for f in sorted(annotations)], dtype=float)
        center, basis, radius = oblique_frame_view(points + [path, frustum], float(args.focus_radius_scale))
    return center, basis, radius


def render_overlay_frame(
    frame_source: FrameSource,
    ann: dict,
    mesh: tuple[np.ndarray, np.ndarray],
    row: dict | None,
    state_row: dict | None,
    frame_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    image = frame_source.read(int(frame_idx))
    draw_object_mask(image, ann, args, frame_source.mask(int(frame_idx), image.shape[:2]))
    draw_mesh_projection(image, ann, mesh, int(args.max_overlay_mesh_edges))
    for hand in ann.get("hands", []):
        draw_hand(image, hand)
    if row is not None:
        draw_contact_patch(image, ann["hands"][int(row["hand_idx"])], row)
    label = f"frame {frame_idx}"
    status_source = frame_source.status(int(frame_idx))
    if status_source:
        label += f"  {status_source}"
    if state_row is not None:
        state = str(state_row.get("geometry_state", "unknown"))
        label += f"  {STATE_LABELS.get(state, state)}"
    if row is None:
        label += "  no reliable mesh-surface contact"
    else:
        contact_label = str(row.get("display_contact_label") or "mesh")
        label += (
            f"  {row['side']} hand {contact_label} contact  reproj {row['median_joint_reprojection_px']:.1f}px"
            f"  surface p95 {row['best_patch_distance_p95_m'] * 1000.0:.1f}mm"
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def wrap_caption_lines(text: str, width_px: int, font_scale: float, thickness: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    overflow = False
    for word_i, word in enumerate(words):
        candidate = word if not cur else f"{cur} {word}"
        candidate_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if candidate_width <= int(width_px):
            cur = candidate
            continue
        if cur:
            lines.append(cur)
            cur = word
        else:
            lines.append(word)
            cur = ""
        if len(lines) == int(max_lines):
            overflow = bool(cur or word_i + 1 < len(words))
            break
    if cur and len(lines) < int(max_lines):
        lines.append(cur)
    if not lines:
        return [""]
    if overflow:
        while cv2.getTextSize(lines[-1] + "...", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] > int(width_px) and lines[-1]:
            lines[-1] = lines[-1][:-1].rstrip()
        if lines[-1]:
            lines[-1] = lines[-1] + "..."
    return lines


def combine_panels(overlay: np.ndarray, world: np.ndarray, caption: str, args: argparse.Namespace) -> np.ndarray:
    half = args.output_width // 2
    panel_h = args.panel_height
    left = cv2.resize(overlay, (half, panel_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(world, (args.output_width - half, panel_h), interpolation=cv2.INTER_AREA)
    joined = np.hstack([left, right])
    bar = np.zeros((args.caption_height, args.output_width, 3), dtype=np.uint8)
    prefix = str(getattr(args, "caption_prefix", "") or "").strip()
    text = f"{prefix}: {caption}" if prefix else caption
    thickness = 2
    font_scale = 0.60 if int(args.caption_height) < 84 else 0.66
    line_gap = int(round(25 * font_scale + 8))
    max_lines = max(1, min(2, (int(args.caption_height) - 12) // max(1, line_gap)))
    lines = wrap_caption_lines(text, int(args.output_width) - 40, font_scale, thickness, max_lines)
    y0 = 25 if len(lines) == 1 else 24
    for line_i, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (20, int(y0 + line_i * line_gap)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return np.vstack([joined, bar])


def run(args: argparse.Namespace) -> dict:
    annotations = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    meshes = load_mesh_archive(args.object_mesh_npz)
    missing_mesh = sorted(set(annotations).difference(meshes))
    if missing_mesh:
        raise RuntimeError(f"mesh archive missing frames: {missing_mesh[:8]}")
    contact_by_frame = reliable_contact_rows(load_json(args.contact_report))
    state_by_frame = load_state_rows(args.v5_state_json)
    append_by_frame = load_append_rows(args.append_report)
    dynamics_by_frame, dynamics_summary = load_dynamics_rows(args.dynamics_report)
    frames = list(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))
    missing_state = sorted(set(frames).difference(state_by_frame)) if args.v5_state_json is not None else []
    if missing_state:
        raise RuntimeError(f"V5 state JSON missing rendered frames: {missing_state[:8]}")
    mano_faces = load_mano_faces(args.mano_model) if args.mano_model is not None else None
    frame_source = FrameSource(args.video, args.manifest)
    fps = float(args.output_fps) if args.output_fps is not None else frame_source.fps()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    world_still_dir = args.output_dir / "stills_world_3d"
    still_dir.mkdir(exist_ok=True)
    world_still_dir.mkdir(exist_ok=True)
    video_path = args.output_dir / "world_reconstruction_side_by_side.mp4"
    world_video_path = args.output_dir / "world_reconstruction_3d.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.output_width), int(args.panel_height + args.caption_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer {video_path}")
    world_writer = cv2.VideoWriter(
        str(world_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.panel_width), int(args.panel_height)),
    )
    if not world_writer.isOpened():
        writer.release()
        raise RuntimeError(f"failed to open writer {world_video_path}")
    written_stills = []
    written_world_stills = []
    try:
        for frame_idx in frames:
            ann = annotations[int(frame_idx)]
            row = contact_by_frame.get(int(frame_idx))
            state_row = state_by_frame.get(int(frame_idx))
            overlay = render_overlay_frame(frame_source, ann, meshes[int(frame_idx)], row, state_row, int(frame_idx), args)
            center, basis, radius = current_focus_view(annotations, ann, meshes[int(frame_idx)], args)
            world = draw_world_panel(
                annotations,
                meshes,
                append_by_frame,
                dynamics_by_frame,
                dynamics_summary,
                contact_by_frame,
                state_by_frame,
                mano_faces,
                int(frame_idx),
                center,
                basis,
                radius,
                args,
            )
            world_writer.write(world)
            caption = str(ann.get("caption", "")).strip()
            if not caption:
                raise RuntimeError(f"frame {frame_idx} has no semantic caption")
            frame = combine_panels(overlay, world, caption, args)
            writer.write(frame)
            if row is not None or int(frame_idx) in set(args.still_frames):
                path = still_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {path}")
                written_stills.append(str(path))
                world_path = world_still_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(world_path), world, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {world_path}")
                written_world_stills.append(str(world_path))
    finally:
        writer.release()
        world_writer.release()
        frame_source.close()
    report = {
        "status": "ok",
        "method": "render_world_reconstruction_v3",
        "video": str(video_path),
        "world_video": str(world_video_path),
        "stills_dir": str(still_dir),
        "world_stills_dir": str(world_still_dir),
        "written_stills": written_stills,
        "written_world_stills": written_world_stills,
        "frames": frames,
        "fps": fps,
        "contact_frames": sorted(contact_by_frame),
        "state_frames": sorted(state_by_frame) if state_by_frame else [],
        "world_view": "world-coordinate manipulation reconstruction with head-camera trajectory, current head-camera frustum, view ray, MANO surfaces, and object mesh",
        "interpretation": "The right panel renders the reconstructed object mesh, MANO surfaces, current head-camera frustum, view ray, and head trajectory in metric world coordinates.",
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "append_report": str(args.append_report) if args.append_report is not None else None,
        "dynamics_report": str(args.dynamics_report) if args.dynamics_report is not None else None,
        "contact_report": str(args.contact_report),
        "v5_state_json": str(args.v5_state_json) if args.v5_state_json is not None else None,
        "mano_model": str(args.mano_model) if args.mano_model is not None else None,
        "video_source": str(args.video) if args.video is not None else None,
        "manifest_source": str(args.manifest) if args.manifest is not None else None,
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--append-report", type=Path)
    parser.add_argument("--dynamics-report", type=Path)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--v5-state-json", type=Path)
    parser.add_argument("--mano-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--caption-height", type=int, default=58)
    parser.add_argument("--focus-radius-scale", type=float, default=1.28)
    parser.add_argument("--frustum-scale-m", type=float, default=0.045)
    parser.add_argument("--world-frustum-visual-scale", type=float, default=1.35)
    parser.add_argument("--include-camera-in-focus", action="store_true")
    parser.add_argument("--show-camera-inset", action="store_true")
    parser.add_argument("--max-mesh-faces", type=int, default=1700)
    parser.add_argument("--max-mano-faces", type=int, default=650)
    parser.add_argument("--max-overlay-mesh-edges", type=int, default=260)
    parser.add_argument("--caption-prefix", default="")
    parser.add_argument("--still-frames", type=int, nargs="*", default=[858, 866, 867, 868, 879, 880])
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
