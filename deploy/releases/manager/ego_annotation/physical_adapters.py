"""Physical-state and render adapter for the typed full-video intermediate.

The adapter consumes metric MANO/camera tensors produced by the timeline driver.
It does not infer poses, replace missing state with boxes, or promote diagnostic
monocular output to metric acceptance. Its output is one full-duration combined
video plus a reproducible numeric state bundle.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ego_annotation.cosmos_semantics import validate_semantic_coverage
from ego_annotation.full_video_timeline import FrameSource, FullVideoAlgorithmState
from ego_annotation.semantic_adapters import caption_for_frame, draw_semantic_caption, semantic_row_anomalies


class PhysicalAdapterError(RuntimeError):
    pass


HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
HAND_COLORS = ((76, 220, 92), (42, 205, 255))  # BGR: left green, right yellow.
CAMERA_PATH_COLOR = (230, 125, 55)  # blue in BGR
CAMERA_FRUSTUM_COLOR = (60, 80, 255)
WORLD_UP = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class PhysicalArtifactResult:
    output_root: str
    state_npz: str
    combined_video: str
    report_json: str
    frame_count: int
    duration_s: float
    metric_state: bool
    diagnostic_only: bool
    local_draw_assembly_s: float
    local_write_encode_s: float
    total_wall_s: float


def transform_points(points_camera: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32)
    transform = np.asarray(world_from_camera, dtype=np.float32)
    if points.shape[-1] != 3 or transform.shape != (4, 4):
        raise PhysicalAdapterError("points must end in xyz and pose must be 4x4")
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1), dtype=np.float32)), axis=1)
    lifted = (transform @ homogeneous.T).T[:, :3]
    return lifted.reshape(points.shape)


def project_points(points_camera: np.ndarray, k: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32)
    intrinsics = np.asarray(k, dtype=np.float32)
    if points.shape[-1] != 3 or intrinsics.shape != (3, 3):
        raise PhysicalAdapterError("projection expects xyz points and 3x3 K")
    z = points[..., 2]
    valid = np.isfinite(points).all(axis=-1) & (z > 1e-5)
    uvw = points @ intrinsics.T
    uv = uvw[..., :2] / np.maximum(uvw[..., 2:], 1e-5)
    return np.concatenate((uv, valid[..., None].astype(np.float32)), axis=-1)


def _pose_array_with_validity(tensor: Any, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(tensor.array, dtype=np.float32)
    valid = np.asarray(tensor.provenance.get("droid_pose_valid"), dtype=bool)
    if value.shape != (frame_count, 4, 4) or valid.shape != (frame_count,):
        raise PhysicalAdapterError("DROID state must expose [N,4,4] poses and an N-frame validity mask")
    if not np.isfinite(value[valid]).all() or np.isfinite(value[~valid]).any():
        raise PhysicalAdapterError("DROID pose validity does not match finite/missing pose values")
    return value, valid


def _valid_projected(points: np.ndarray, width: int, height: int, *, margin: int = 80) -> np.ndarray:
    projected = np.asarray(points, dtype=np.float32)
    return (
        (projected[:, 2] > 0)
        & np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= -margin)
        & (projected[:, 0] < width + margin)
        & (projected[:, 1] >= -margin)
        & (projected[:, 1] < height + margin)
    )


def _draw_projected(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int], width: int, height: int) -> int:
    """Compatibility point renderer used by older callers and focused tests."""
    count = 0
    for point in np.asarray(points):
        if point.shape[0] < 3 or not np.isfinite(point).all() or point[2] <= 0:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)
            count += 1
    return count


def _draw_projected_hand_2d(
    canvas: np.ndarray,
    vertices_source_px: np.ndarray,
    joints_source_px: np.ndarray,
    color: tuple[int, int, int],
) -> dict[str, int]:
    """Draw MANO directly from WiLoR crop-to-source pixel projections."""

    height, width = canvas.shape[:2]
    vertices_xy = np.asarray(vertices_source_px, dtype=np.float32)
    joints_xy = np.asarray(joints_source_px, dtype=np.float32)
    if vertices_xy.shape != (778, 2) or joints_xy.ndim != 2 or joints_xy.shape[1] != 2:
        raise PhysicalAdapterError("direct MANO projection must provide [778,2] vertices and [J,2] joints")
    vertices_uv = np.concatenate((vertices_xy, np.ones((len(vertices_xy), 1), dtype=np.float32)), axis=1)
    joints_uv = np.concatenate((joints_xy, np.ones((len(joints_xy), 1), dtype=np.float32)), axis=1)
    vertex_valid = _valid_projected(vertices_uv, width, height)
    joint_valid = _valid_projected(joints_uv, width, height)
    return _draw_projected_uv(canvas, vertices_uv, joints_uv, vertex_valid, joint_valid, color)


def _draw_projected_hand(
    canvas: np.ndarray,
    vertices_camera: np.ndarray,
    joints_camera: np.ndarray,
    k: np.ndarray,
    color: tuple[int, int, int],
) -> dict[str, int]:
    """Draw a metric MANO surface through a source-camera pinhole."""
    height, width = canvas.shape[:2]
    vertices_uv = project_points(vertices_camera, k)
    joints_uv = project_points(joints_camera, k)
    vertex_valid = _valid_projected(vertices_uv, width, height)
    joint_valid = _valid_projected(joints_uv, width, height)
    return _draw_projected_uv(canvas, vertices_uv, joints_uv, vertex_valid, joint_valid, color)


def _draw_projected_uv(
    canvas: np.ndarray,
    vertices_uv: np.ndarray,
    joints_uv: np.ndarray,
    vertex_valid: np.ndarray,
    joint_valid: np.ndarray,
    color: tuple[int, int, int],
) -> dict[str, int]:
    """Shared rasterization for direct crop and metric pinhole projections."""

    height, width = canvas.shape[:2]
    counts = {"surface_vertices": 0, "mesh_outlines": 0, "skeleton_edges": 0, "joints": 0}

    in_frame_vertices = vertices_uv[vertex_valid, :2]
    if len(in_frame_vertices) >= 3:
        hull = cv2.convexHull(np.rint(in_frame_vertices).astype(np.int32))
        cv2.polylines(canvas, [hull], True, (3, 3, 3), 4, cv2.LINE_AA)
        cv2.polylines(canvas, [hull], True, color, 2, cv2.LINE_AA)
        counts["mesh_outlines"] = 1
    # The typed state carries MANO vertices but not the face table. Surface
    # points plus the silhouette expose the real reconstructed mesh without
    # inventing a topology.
    for point in in_frame_vertices[::6]:
        cv2.circle(canvas, (int(round(float(point[0]))), int(round(float(point[1])))), 1, color, -1, cv2.LINE_AA)
        counts["surface_vertices"] += 1

    for parent, child in HAND_EDGES:
        if parent >= len(joints_uv) or child >= len(joints_uv) or not (joint_valid[parent] and joint_valid[child]):
            continue
        a = tuple(np.rint(joints_uv[parent, :2]).astype(int))
        b = tuple(np.rint(joints_uv[child, :2]).astype(int))
        cv2.line(canvas, a, b, (3, 3, 3), 5, cv2.LINE_AA)
        cv2.line(canvas, a, b, color, 3, cv2.LINE_AA)
        counts["skeleton_edges"] += 1
    for point, valid in zip(joints_uv[:, :2], joint_valid):
        if not valid:
            continue
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, xy, 4, (3, 3, 3), -1, cv2.LINE_AA)
        cv2.circle(canvas, xy, 2, color, -1, cv2.LINE_AA)
        counts["joints"] += 1
    return counts


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-8:
        value = np.asarray(fallback, dtype=np.float64)
        norm = max(float(np.linalg.norm(value)), 1e-8)
    return value / norm


@dataclass(frozen=True)
class _WorldView:
    center: np.ndarray
    span: float
    low: np.ndarray
    high: np.ndarray
    eye: np.ndarray
    target: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    trajectory_forward: np.ndarray
    focal_scale: float
    camera_anchor: np.ndarray
    camera_scale: float


def _estimate_forward(camera_centers: np.ndarray) -> np.ndarray:
    finite = camera_centers[np.isfinite(camera_centers).all(axis=1)]
    if len(finite) >= 2:
        motion = finite[-1] - finite[0]
        motion[1] = 0.0
        if np.linalg.norm(motion) > 0.1:
            return _normalize(motion, np.asarray([0.0, 0.0, 1.0]))
    if len(finite) >= 2:
        centered = finite - np.mean(finite, axis=0, keepdims=True)
        centered[:, 1] = 0.0
        try:
            axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        except np.linalg.LinAlgError:
            axis = np.asarray([0.0, 0.0, 1.0])
        if np.dot(axis, finite[-1] - finite[0]) < 0:
            axis = -axis
        axis[1] = 0.0
        return _normalize(axis, np.asarray([0.0, 0.0, 1.0]))
    return np.asarray([0.0, 0.0, 1.0])


def _finite_world_points(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    return flat[np.isfinite(flat).all(axis=1)]


def _required_frame_points(camera_centers: np.ndarray, world_joints: list[np.ndarray], frame_index: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    if 0 <= frame_index < len(camera_centers) and np.isfinite(camera_centers[frame_index]).all():
        parts.append(np.asarray(camera_centers[frame_index], dtype=np.float64).reshape(1, 3))
    for values in world_joints:
        array = np.asarray(values)
        if array.ndim >= 2 and frame_index < array.shape[0]:
            points = _finite_world_points(array[frame_index])
            if len(points):
                parts.append(points)
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float64)


def _local_extent(
    camera_centers: np.ndarray,
    world_joints: list[np.ndarray],
    frame_index: int | None,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return robust local bounds, camera anchor, and current hand span.

    Percentile bounds suppress a remote trajectory outlier.  Current-frame
    camera/hands are then explicitly reintroduced so the state being rendered
    cannot be discarded by the robust estimator.
    """
    cameras = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    finite_camera = cameras[np.isfinite(cameras).all(axis=1)]
    if frame_index is None:
        indices = np.arange(len(cameras), dtype=np.int64)
    else:
        radius = max(int(window_frames) // 2, 0)
        start = max(0, int(frame_index) - radius)
        stop = min(len(cameras), int(frame_index) + radius + 1)
        indices = np.arange(start, stop, dtype=np.int64)
    parts: list[np.ndarray] = []
    local_camera = cameras[indices] if len(indices) else cameras[:0]
    local_camera = local_camera[np.isfinite(local_camera).all(axis=1)]
    if len(local_camera):
        parts.append(local_camera)
    for values in world_joints:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim >= 2 and len(indices):
            available = indices[indices < array.shape[0]]
            local = _finite_world_points(array[available]) if len(available) else np.empty((0, 3), dtype=np.float64)
            if len(local):
                parts.append(local)
    merged = np.concatenate(parts, axis=0) if parts else np.zeros((1, 3), dtype=np.float64)
    if frame_index is None:
        low, high = np.min(merged, axis=0), np.max(merged, axis=0)
        anchor = finite_camera[-1] if len(finite_camera) else np.zeros(3, dtype=np.float64)
        required = merged
    else:
        robust_low = np.percentile(merged, 5.0, axis=0) if len(merged) >= 8 else np.min(merged, axis=0)
        robust_high = np.percentile(merged, 95.0, axis=0) if len(merged) >= 8 else np.max(merged, axis=0)
        required = _required_frame_points(cameras, world_joints, int(frame_index))
        if len(required):
            required_low, required_high = np.min(required, axis=0), np.max(required, axis=0)
            robust_low = np.minimum(robust_low, required_low)
            robust_high = np.maximum(robust_high, required_high)
            has_current_hand = any(
                np.asarray(values).ndim >= 2
                and frame_index < np.asarray(values).shape[0]
                and len(_finite_world_points(np.asarray(values)[frame_index]))
                for values in world_joints
            )
            # A trajectory jump in the local window is context, not a reason
            # to make the current physical interaction microscopic. Once a
            # valid hand exists, cap the robust window around the current
            # camera+hand cluster; the complete path remains context only.
            if has_current_hand:
                current_extent = max(float(np.max(required_high - required_low)), 0.5)
                focus_margin = 0.10 * current_extent
                robust_low = np.maximum(robust_low, required_low - focus_margin)
                robust_high = np.minimum(robust_high, required_high + focus_margin)
                robust_low = np.minimum(robust_low, required_low)
                robust_high = np.maximum(robust_high, required_high)
        low, high = robust_low, robust_high
        valid_window_cameras = cameras[indices]
        valid_window_cameras = valid_window_cameras[np.isfinite(valid_window_cameras).all(axis=1)]
        if len(valid_window_cameras):
            anchor = cameras[frame_index] if np.isfinite(cameras[frame_index]).all() else valid_window_cameras[-1]
        elif len(finite_camera):
            anchor = finite_camera[-1]
        else:
            anchor = np.zeros(3, dtype=np.float64)
    hand_span = 0.0
    if frame_index is not None:
        for values in world_joints:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim >= 2 and frame_index < array.shape[0]:
                hand = _finite_world_points(array[frame_index])
                if len(hand):
                    hand_span = max(hand_span, float(np.max(np.ptp(hand, axis=0))))
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        low, high = np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    span = max(float(np.max(high - low)), 0.5, 4.0 * hand_span) * 1.20
    center = (low + high) / 2.0
    half = span / 2.0
    return center - half, center + half, np.asarray(anchor, dtype=np.float64), hand_span


def _make_world_view(
    low: np.ndarray,
    high: np.ndarray,
    trajectory_forward: np.ndarray,
    camera_anchor: np.ndarray,
    hand_span: float,
    size: tuple[int, int],
) -> _WorldView:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    center = (low + high) / 2.0
    span = max(float(np.max(high - low)), 0.5)
    side = _normalize(np.cross(WORLD_UP, trajectory_forward), np.asarray([1.0, 0.0, 0.0]))
    target = center + trajectory_forward * (0.10 * span)
    eye = center - trajectory_forward * (2.20 * span) + side * (1.05 * span) + WORLD_UP * (0.70 * span)
    forward = _normalize(target - eye, trajectory_forward)
    right = _normalize(np.cross(WORLD_UP, forward), side)
    up = _normalize(np.cross(forward, right), WORLD_UP)
    merged = np.asarray([
        [x, y, z]
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ] + [camera_anchor], dtype=np.float64)
    relative = merged - eye.reshape(1, 3)
    depth = relative @ forward
    if np.any(depth <= 1e-6):
        eye = eye - forward * (abs(float(np.min(depth))) + 0.5 * span)
        relative = merged - eye.reshape(1, 3)
        depth = relative @ forward
    x_ratio = float(np.max(np.abs(relative @ right) / np.maximum(depth, 1e-6)))
    y_ratio = float(np.max(np.abs(relative @ up) / np.maximum(depth, 1e-6)))
    width, height = size
    min_dimension = float(min(width, height))
    x_budget = 0.43 * float(width) / min_dimension
    y_budget = 0.41 * float(height) / min_dimension
    focal_scale = min(3.0, x_budget / max(x_ratio, 1e-6), y_budget / max(y_ratio, 1e-6))
    # The frustum is a physical visual anchor.  Its size follows the local
    # extent and observed hand scale, so it remains visible after reframing.
    camera_scale = max(0.10 * span, 0.75 * hand_span)
    return _WorldView(center, span, low, high, eye, target, forward, right, up, trajectory_forward, float(focal_scale), camera_anchor, float(camera_scale))


def _enforce_hand_pixel_span(
    view: _WorldView,
    camera_centers: np.ndarray,
    world_joints: list[np.ndarray],
    frame_index: int,
    size: tuple[int, int],
    target_px: float = 50.0,
) -> _WorldView:
    """Increase focal scale when possible while retaining current anchors."""
    observed_spans: list[float] = []
    for values in world_joints:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or frame_index >= array.shape[0]:
            continue
        hand = _finite_world_points(array[frame_index])
        if len(hand) < 2:
            continue
        pixels, valid = _project_world(hand, view, size)
        if valid.all():
            observed_spans.append(float(np.max(np.ptp(pixels, axis=0))))
    if not observed_spans or min(observed_spans) >= target_px:
        return view
    requested = view.focal_scale * target_px / max(min(observed_spans), 1e-6)
    required = _required_frame_points(np.asarray(camera_centers), world_joints, frame_index)
    if not len(required):
        return replace(view, focal_scale=float(requested))
    relative = required - view.eye.reshape(1, 3)
    depth = relative @ view.forward
    valid = np.isfinite(required).all(axis=1) & (depth > 1e-6)
    if not valid.all():
        return view
    x_ratio = (relative @ view.right) / depth
    y_ratio = (relative @ view.up) / depth
    width, height = size
    base = float(min(width, height))
    limits = [requested]
    for ratio in x_ratio:
        if ratio > 1e-9:
            limits.append((width * 0.50 - 2.0) / (base * ratio))
        elif ratio < -1e-9:
            limits.append((width * 0.50 - 2.0) / (base * -ratio))
    for ratio in y_ratio:
        if ratio > 1e-9:
            limits.append((height * 0.56 - 2.0) / (base * ratio))
        elif ratio < -1e-9:
            limits.append((height * 0.44 - 2.0) / (base * -ratio))
    return replace(view, focal_scale=float(max(view.focal_scale, min(limits))))


def _build_world_view(
    camera_centers: np.ndarray,
    world_joints: list[np.ndarray],
    size: tuple[int, int] = (1280, 720),
    *,
    frame_index: int | None = None,
    window_frames: int = 300,
) -> _WorldView:
    """Fit a robust full or local perspective view without changing world data."""
    low, high, anchor, hand_span = _local_extent(camera_centers, world_joints, frame_index, window_frames)
    finite_camera = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    if frame_index is None:
        direction_source = finite_camera[np.isfinite(finite_camera).all(axis=1)]
    else:
        radius = max(int(window_frames) // 2, 0)
        direction_source = finite_camera[max(0, frame_index - radius):min(len(finite_camera), frame_index + radius + 1)]
    trajectory_forward = _estimate_forward(direction_source)
    view = _make_world_view(low, high, trajectory_forward, anchor, hand_span, size)
    return view if frame_index is None else _enforce_hand_pixel_span(view, camera_centers, world_joints, frame_index, size)


def _build_world_views(
    camera_centers: np.ndarray,
    world_joints: list[np.ndarray],
    size: tuple[int, int] = (1280, 720),
    *,
    window_frames: int = 300,
    smoothing_frames: int = 9,
) -> list[_WorldView]:
    """Build smoothed sliding views; current-frame points always remain in fit."""
    count = len(np.asarray(camera_centers).reshape(-1, 3))
    raw = [_build_world_view(camera_centers, world_joints, size, frame_index=i, window_frames=window_frames) for i in range(count)]
    views: list[_WorldView] = []
    alpha = 2.0 / (max(int(smoothing_frames), 1) + 1.0)
    for index, candidate in enumerate(raw):
        if index == 0:
            smoothed_center, smoothed_span = candidate.center, candidate.span
        else:
            previous = views[-1]
            smoothed_center = previous.center + alpha * (candidate.center - previous.center)
            smoothed_span = math.exp(math.log(max(previous.span, 1e-6)) + alpha * (math.log(max(candidate.span, 1e-6)) - math.log(max(previous.span, 1e-6))))
        low = smoothed_center - smoothed_span / 2.0
        high = smoothed_center + smoothed_span / 2.0
        view = _make_world_view(
            low,
            high,
            candidate.trajectory_forward,
            candidate.camera_anchor,
            candidate.camera_scale * smoothed_span / max(candidate.span, 1e-6),
            size,
        )
        view = _enforce_hand_pixel_span(view, camera_centers, world_joints, index, size)
        required = _required_frame_points(np.asarray(camera_centers), world_joints, index)
        pixels, valid = _project_world(required, view, size) if len(required) else (np.empty((0, 2)), np.empty((0,), dtype=bool))
        in_frame = len(required) == 0 or (valid.all() and np.all((pixels[:, 0] >= 2) & (pixels[:, 0] < size[0] - 2) & (pixels[:, 1] >= 2) & (pixels[:, 1] < size[1] - 2)))
        views.append(view if in_frame else candidate)
    return views


def _world_to_camera_display(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    display_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Express world points in the current camera frame for the stable display.

    `world_from_camera` is a conventional world-from-camera transform.  The
    inverse rotation makes the current camera exactly the display origin; an
    optional *constant* display rotation is available solely for presentation.
    """
    values = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if values.shape[-1] != 3 or pose.shape != (4, 4):
        raise PhysicalAdapterError("camera-relative display expects xyz points and a 4x4 world-from-camera pose")
    output = np.full_like(values, np.nan, dtype=np.float64)
    if not np.isfinite(pose).all():
        return output
    valid = np.isfinite(values).all(axis=-1)
    if not np.any(valid):
        return output
    rotation = pose[:3, :3]
    display = np.eye(3, dtype=np.float64) if display_rotation is None else np.asarray(display_rotation, dtype=np.float64)
    if display.shape != (3, 3) or not np.isfinite(display).all():
        raise PhysicalAdapterError("display rotation must be finite 3x3")
    flattened = values.reshape(-1, 3)
    flattened_valid = valid.reshape(-1)
    # Row-vector form of R_wc.T @ (p_world - t_wc).
    output.reshape(-1, 3)[flattened_valid] = (flattened[flattened_valid] - pose[:3, 3]) @ rotation @ display.T
    return output


def _camera_relative_hand_timeline(
    world_from_camera: np.ndarray,
    world_joints: list[np.ndarray],
    display_rotation: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Transform each valid MANO frame into its own current-camera coordinates."""
    poses = np.asarray(world_from_camera, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise PhysicalAdapterError("camera-relative display requires [N,4,4] poses")
    result: list[np.ndarray] = []
    for values in world_joints:
        source = np.asarray(values, dtype=np.float64)
        local = np.full_like(source, np.nan, dtype=np.float64)
        for frame_index in range(min(len(source), len(poses))):
            local[frame_index] = _world_to_camera_display(source[frame_index], poses[frame_index], display_rotation)
        result.append(local)
    return result


def _camera_left_display_basis(bilateral_centroid_camera: np.ndarray) -> np.ndarray:
    """Make the central camera-to-hands direction the fixed display +X axis."""
    x_axis = _normalize(bilateral_centroid_camera, np.asarray([1.0, 0.0, 0.0]))
    camera_up = np.asarray([0.0, 1.0, 0.0])
    y_axis = _normalize(camera_up - x_axis * float(np.dot(camera_up, x_axis)), np.asarray([0.0, 0.0, 1.0]))
    z_axis = _normalize(np.cross(x_axis, y_axis), np.asarray([0.0, 0.0, 1.0]))
    return np.stack((x_axis, y_axis, z_axis))


def _make_camera_left_view(hand_distance: float, robust_hand_extent: float, size: tuple[int, int]) -> _WorldView:
    """Fixed front presentation: origin/frustum left, bilateral hands right."""
    span = max(2.0 * hand_distance, hand_distance + 2.0 * robust_hand_extent, 0.8)
    center = np.asarray([0.72 * hand_distance, 0.0, 0.0], dtype=np.float64)
    half = np.asarray([0.60 * span, 0.50 * span, 0.50 * span], dtype=np.float64)
    low, high = center - half, center + half
    eye = center + np.asarray([0.0, 0.0, -2.50 * span])
    target = center
    forward, right, up = np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])
    return _WorldView(
        center, span, low, high, eye, target, forward, right, up,
        np.asarray([1.0, 0.0, 0.0]), 1.0, np.zeros(3, dtype=np.float64), max(0.10 * span, 0.12),
    )


def _camera_centric_world_view(
    world_from_camera: np.ndarray,
    world_joints: list[np.ndarray],
    size: tuple[int, int] = (1280, 720),
) -> tuple[_WorldView, list[np.ndarray], int | None, np.ndarray]:
    """Build one camera-left view from a central bilateral-hand observation."""
    camera_local_joints = _camera_relative_hand_timeline(world_from_camera, world_joints)
    both_valid = np.ones(len(world_from_camera), dtype=bool)
    for values in camera_local_joints[:2]:
        both_valid &= np.asarray([len(_finite_world_points(frame)) >= 2 for frame in values], dtype=bool)
    candidates = np.flatnonzero(both_valid)
    target_index = int(candidates[len(candidates) // 2]) if len(candidates) else None
    if target_index is None:
        display_rotation = np.eye(3, dtype=np.float64)
        local_joints = camera_local_joints
        central_centroid = np.asarray([0.8, 0.0, 0.0])
    else:
        central_points = np.concatenate([camera_local_joints[side][target_index] for side in range(2)], axis=0)
        display_rotation = _camera_left_display_basis(np.mean(central_points, axis=0))
        local_joints = [values @ display_rotation.T for values in camera_local_joints]
        central_centroid = np.mean(np.concatenate([local_joints[side][target_index] for side in range(2)], axis=0), axis=0)

    finite_offsets: list[np.ndarray] = []
    for values in local_joints:
        finite = _finite_world_points(values)
        if len(finite):
            finite_offsets.append(finite - central_centroid)
    offsets = np.concatenate(finite_offsets, axis=0) if finite_offsets else np.zeros((1, 3), dtype=np.float64)
    if len(offsets) >= 8:
        low_offset, high_offset = np.percentile(offsets, 10.0, axis=0), np.percentile(offsets, 90.0, axis=0)
    else:
        low_offset, high_offset = np.min(offsets, axis=0), np.max(offsets, axis=0)
    robust_hand_extent = float(max(np.max(np.abs(low_offset)), np.max(np.abs(high_offset)), 0.15))
    hand_distance = max(float(central_centroid[0]), 0.35)
    view = _make_camera_left_view(hand_distance, robust_hand_extent, size)

    if target_index is not None:
        observed: list[float] = []
        for values in local_joints[:2]:
            pixels, valid = _project_world(values[target_index], view, size)
            if valid.all():
                observed.append(float(np.max(np.ptp(pixels, axis=0))))
        if observed and min(observed) < 50.0:
            view = replace(view, focal_scale=float(view.focal_scale * 50.0 / max(min(observed), 1e-6)))
    return view, local_joints, target_index, display_rotation


def _project_world(points: np.ndarray, view: _WorldView, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    relative = values - view.eye.reshape(1, 3)
    x = relative @ view.right
    y = relative @ view.up
    depth = relative @ view.forward
    valid = np.isfinite(values).all(axis=1) & np.isfinite(depth) & (depth > 1e-4)
    pixels = np.full((len(values), 2), np.nan, dtype=np.float64)
    focal = view.focal_scale * min(width, height)
    pixels[valid, 0] = width * 0.50 + focal * x[valid] / depth[valid]
    pixels[valid, 1] = height * 0.56 - focal * y[valid] / depth[valid]
    return pixels, valid


def _draw_line3d(canvas: np.ndarray, points: np.ndarray, view: _WorldView, color: tuple[int, int, int], thickness: int = 2) -> int:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) < 2:
        return 0
    pixels, valid = _project_world(values, view, (canvas.shape[1], canvas.shape[0]))
    drawn = 0
    for index in range(len(values) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        a, b = pixels[index], pixels[index + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        first = tuple(np.rint(a).astype(int))
        second = tuple(np.rint(b).astype(int))
        if not all(-200 <= value <= bound + 200 for point in (first, second) for value, bound in zip(point, (canvas.shape[1], canvas.shape[0]))):
            continue
        cv2.line(canvas, first, second, color, thickness, cv2.LINE_AA)
        drawn += 1
    return drawn


def _draw_floor_grid(canvas: np.ndarray, view: _WorldView) -> None:
    floor_y = float(view.low[1] - 0.08 * view.span)
    step = 0.25 if view.span <= 2.5 else 0.5 if view.span <= 6.0 else 1.0
    x0 = math.floor((view.low[0] - 0.20 * view.span) / step) * step
    x1 = math.ceil((view.high[0] + 0.20 * view.span) / step) * step
    z0 = math.floor((view.low[2] - 0.20 * view.span) / step) * step
    z1 = math.ceil((view.high[2] + 0.20 * view.span) / step) * step
    for x in np.arange(x0, x1 + step * 0.5, step):
        _draw_line3d(canvas, np.asarray([[x, floor_y, z0], [x, floor_y, z1]]), view, (58, 68, 78), 1)
    for z in np.arange(z0, z1 + step * 0.5, step):
        _draw_line3d(canvas, np.asarray([[x0, floor_y, z], [x1, floor_y, z]]), view, (58, 68, 78), 1)
    origin = np.asarray([view.center[0], floor_y, view.center[2]])
    axis = min(max(view.span * 0.18, 0.25), 1.0)
    _draw_line3d(canvas, np.asarray([origin, origin + [axis, 0.0, 0.0]]), view, (70, 90, 240), 3)
    _draw_line3d(canvas, np.asarray([origin, origin + [0.0, axis, 0.0]]), view, (70, 220, 90), 3)
    _draw_line3d(canvas, np.asarray([origin, origin + [0.0, 0.0, axis]]), view, (230, 190, 70), 3)


def _draw_camera(canvas: np.ndarray, center: np.ndarray, rotation: np.ndarray | None, view: _WorldView) -> None:
    if rotation is not None and np.asarray(rotation).shape == (3, 3) and np.isfinite(rotation).all():
        x_axis, y_axis, z_axis = (_normalize(rotation[:, i], fallback) for i, fallback in enumerate((view.right, WORLD_UP, view.forward)))
    else:
        x_axis, y_axis, z_axis = view.right, WORLD_UP, view.forward
    scale = view.camera_scale
    front = center + z_axis * scale * 1.15
    corners = np.asarray([
        front + x_axis * scale * 0.55 + y_axis * scale * 0.36,
        front - x_axis * scale * 0.55 + y_axis * scale * 0.36,
        front - x_axis * scale * 0.55 - y_axis * scale * 0.36,
        front + x_axis * scale * 0.55 - y_axis * scale * 0.36,
    ])
    for corner in corners:
        _draw_line3d(canvas, np.asarray([center, corner]), view, (3, 3, 3), 4)
        _draw_line3d(canvas, np.asarray([center, corner]), view, CAMERA_FRUSTUM_COLOR, 2)
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        _draw_line3d(canvas, np.asarray([corners[first], corners[second]]), view, (3, 3, 3), 4)
        _draw_line3d(canvas, np.asarray([corners[first], corners[second]]), view, CAMERA_FRUSTUM_COLOR, 2)
    for axis, color in ((x_axis, (70, 100, 255)), (y_axis, (70, 230, 90)), (z_axis, (255, 210, 80))):
        _draw_line3d(canvas, np.asarray([center, center + axis * scale]), view, color, 2)


def _draw_world_hand(canvas: np.ndarray, joints: np.ndarray, vertices: np.ndarray | None, view: _WorldView, color: tuple[int, int, int]) -> int:
    if vertices is not None:
        pixels, valid = _project_world(np.asarray(vertices)[::10], view, (canvas.shape[1], canvas.shape[0]))
        for point in pixels[valid]:
            x, y = tuple(np.rint(point).astype(int))
            if -30 <= x <= canvas.shape[1] + 30 and -30 <= y <= canvas.shape[0] + 30:
                cv2.circle(canvas, (x, y), 1, color, -1, cv2.LINE_AA)
    values = np.asarray(joints, dtype=np.float64)
    if values.shape != (21, 3):
        return 0
    drawn = 0
    for parent, child in HAND_EDGES:
        edge = values[[parent, child]]
        _draw_line3d(canvas, edge, view, (3, 3, 3), 5)
        drawn += _draw_line3d(canvas, edge, view, color, 3)
    return drawn


def _draw_hand_inset(canvas: np.ndarray, hands: Mapping[int, np.ndarray]) -> bool:
    valid_hands = {side: np.asarray(joints, dtype=np.float64) for side, joints in hands.items() if np.asarray(joints).shape == (21, 3) and np.isfinite(joints).all()}
    if not valid_hands:
        return False
    height, width = canvas.shape[:2]
    inset_width = min(520, max(420, int(width * 0.40)))
    inset_height = min(310, max(240, int(height * 0.40)))
    x0, y0 = width - inset_width - 22, 54
    x1, y1 = x0 + inset_width, y0 + inset_height
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0.0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 105), 1, cv2.LINE_AA)
    cv2.putText(canvas, "MANO 21-joint skeletons", (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    available = [side for side in (0, 1) if side in valid_hands]
    split = (x0 + x1) // 2
    for order, side in enumerate(available):
        left = x0 + 18 if len(available) == 1 or order == 0 else split + 10
        right = x1 - 18 if len(available) == 1 or order == 1 else split - 10
        top, bottom = y0 + 48, y1 - 16
        joints = valid_hands[side]
        centered = joints - np.mean(joints, axis=0, keepdims=True)
        try:
            axes = np.linalg.svd(centered, full_matrices=False)[2][:2]
        except np.linalg.LinAlgError:
            axes = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coordinates = centered @ axes.T
        low, high = np.min(coordinates, axis=0), np.max(coordinates, axis=0)
        scale = 0.82 * min(right - left, bottom - top) / max(float(np.max(high - low)), 1e-4)
        origin = np.asarray([(left + right) / 2, (top + bottom) / 2])
        pixels = origin + (coordinates - (low + high) / 2) * np.asarray([scale, -scale])
        cv2.putText(canvas, "L" if side == 0 else "R", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.50, HAND_COLORS[side], 1, cv2.LINE_AA)
        for parent, child in HAND_EDGES:
            cv2.line(canvas, tuple(np.rint(pixels[parent]).astype(int)), tuple(np.rint(pixels[child]).astype(int)), HAND_COLORS[side], 2, cv2.LINE_AA)
    if len(available) == 2:
        cv2.line(canvas, (split, y0 + 40), (split, y1 - 8), (62, 68, 75), 1, cv2.LINE_AA)
    return True


def _world_canvas(
    points: list[np.ndarray],
    camera_centers: np.ndarray,
    size: tuple[int, int],
    *,
    view: _WorldView | None = None,
    vertices: list[np.ndarray] | None = None,
    camera_rotation: np.ndarray | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    """Render a perspective metric world view; retained signature supports tests."""
    width, height = size
    canvas = np.full((height, width, 3), (28, 31, 34), dtype=np.uint8)
    finite_camera = np.asarray(camera_centers)[np.isfinite(camera_centers).all(axis=1)]
    if view is None:
        synthetic = [np.asarray(item, dtype=np.float64).reshape(1, -1, 3) for item in points]
        view = _build_world_view(np.asarray(camera_centers, dtype=np.float64), synthetic, size)
    _draw_floor_grid(canvas, view)
    if len(finite_camera) >= 2:
        _draw_line3d(canvas, finite_camera, view, CAMERA_PATH_COLOR, 3)
    if np.isfinite(view.camera_anchor).all():
        _draw_camera(canvas, view.camera_anchor, camera_rotation, view)
    inset_hands: dict[int, np.ndarray] = {}
    for side, joints in enumerate(points[:2]):
        values = np.asarray(joints, dtype=np.float64)
        surface = None if vertices is None or side >= len(vertices) else np.asarray(vertices[side], dtype=np.float64)
        if values.shape == (21, 3):
            _draw_world_hand(canvas, values, surface, view, HAND_COLORS[side])
            inset_hands[side] = values
        else:
            projected, valid = _project_world(values.reshape(-1, 3), view, size)
            for point in projected[valid]:
                x, y = tuple(np.rint(point).astype(int))
                if -30 <= x <= width + 30 and -30 <= y <= height + 30:
                    cv2.circle(canvas, (x, y), 2, HAND_COLORS[side], -1, cv2.LINE_AA)
    _draw_hand_inset(canvas, inset_hands)
    cv2.putText(canvas, "3D global-world view: fixed clip presentation; physical camera per frame", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2, cv2.LINE_AA)
    label = "world gauge: T_world_camera + world MANO | units: meters"
    if frame_index is not None:
        label = f"frame {frame_index:06d} | {label}"
    cv2.putText(canvas, label, (16, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "blue=camera path | red=physical camera frustum | green=left MANO | yellow=right MANO", (16, height - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 218, 228), 1, cv2.LINE_AA)
    return canvas


def _add_counts(total: dict[str, int], update: Mapping[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def resize_to_cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize and center-crop an image so every pane pixel is occupied."""
    width, height = size
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or width <= 0 or height <= 0:
        raise PhysicalAdapterError("cover resize expects HWC RGB/BGR image and positive size")
    scale = max(width / value.shape[1], height / value.shape[0])
    resized = cv2.resize(value, (max(width, int(round(value.shape[1] * scale))), max(height, int(round(value.shape[0] * scale)))), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return np.ascontiguousarray(resized[y0:y0 + height, x0:x0 + width])


class PhysicalArtifactAdapter:
    """Lift and render the full source timeline from a driver state."""

    def __init__(self, *, render_size: tuple[int, int] = (1280, 720)) -> None:
        width, height = render_size
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise PhysicalAdapterError("render size must be positive and even")
        self.render_size = render_size

    def render(self, state: FullVideoAlgorithmState, source: FrameSource, output_root: str | Path) -> PhysicalArtifactResult:
        render_started = time.monotonic()
        local_write_encode_s = 0.0
        timeline = state.source_timeline
        if source.timeline.source_sha256 != timeline.source_sha256:
            raise PhysicalAdapterError("source timeline identity changed before physical rendering")
        droid = state.droid_records.final.output
        droid_scale_provenance = dict(getattr(droid, "scale_provenance", {}))
        poses, droid_pose_valid = _pose_array_with_validity(droid.T_world_camera, state.frame_count)
        if not np.array_equal(droid_pose_valid, np.asarray(state.droid_records.coverage.pose_valid, dtype=bool)):
            raise PhysicalAdapterError("DROID record coverage diverges from pose validity")
        k = np.asarray(state.canonical_K.k_canonical, dtype=np.float32)
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise PhysicalAdapterError("canonical K is not finite 3x3")
        root = Path(output_root).resolve()
        renders = root / "renders"
        state_dir = root / "state"
        renders.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        camera_centers = poses[:, :3, 3]
        world_vertices: list[np.ndarray] = []
        world_joints: list[np.ndarray] = []
        for side_index in range(2):
            valid = np.asarray(state.timeline_inference.valid.array[side_index], dtype=bool)
            camera_vertices = np.asarray(state.timeline_inference.vertices_camera_m.array[side_index], dtype=np.float32)
            camera_joints = np.asarray(state.timeline_inference.joints_camera_m.array[side_index], dtype=np.float32)
            vertices_world = np.full_like(camera_vertices, np.nan)
            joints_world = np.full_like(camera_joints, np.nan)
            for frame_index in range(state.frame_count):
                if valid[frame_index] and droid_pose_valid[frame_index]:
                    vertices_world[frame_index] = transform_points(camera_vertices[frame_index], poses[frame_index])
                    joints_world[frame_index] = transform_points(camera_joints[frame_index], poses[frame_index])
            world_vertices.append(vertices_world)
            world_joints.append(joints_world)
        # Fit one clip-level world view.  The presentation camera is fixed; only
        # the physical camera marker and world trajectory advance per frame.
        camera_centric_view = _build_world_view(camera_centers, world_joints, self.render_size)
        camera_centric_reference_frame = None
        camera_centric_display_rotation = np.eye(3, dtype=np.float64)
        # Keep both anatomical sides in the pose contract.  A single frame can
        # contain two valid wrists; collapsing to one side would silently lose
        # part of the reference-aligned state.
        camera_wrist = np.full((2, state.frame_count, 4, 4), np.nan, dtype=np.float32)
        world_wrist = np.full((2, state.frame_count, 4, 4), np.nan, dtype=np.float32)
        wrist_valid = np.zeros((2, state.frame_count), dtype=bool)
        root_orient_tensor = getattr(state.timeline_inference, "root_orient", None)
        for side in range(2):
            for frame_index in range(state.frame_count):
                joints = np.asarray(world_joints[side][frame_index])
                if joints.shape != (21, 3) or not np.isfinite(joints[0]).all() or not droid_pose_valid[frame_index]:
                    continue
                wrist_world = np.asarray(joints[0], dtype=np.float32)
                wrist_camera = transform_points(wrist_world[None], np.linalg.inv(poses[frame_index]))[0]
                camera_wrist[side, frame_index] = np.eye(4, dtype=np.float32)
                root_orient = np.eye(3, dtype=np.float32)
                if root_orient_tensor is not None:
                    root_orient = np.asarray(root_orient_tensor.array[side, frame_index], dtype=np.float32)
                if root_orient.shape == (3, 3) and np.isfinite(root_orient).all() and np.allclose(root_orient.T @ root_orient, np.eye(3), atol=2e-3) and np.linalg.det(root_orient) > 0.0:
                    camera_wrist[side, frame_index, :3, :3] = root_orient
                camera_wrist[side, frame_index, :3, 3] = wrist_camera
                world_wrist[side, frame_index] = poses[frame_index] @ camera_wrist[side, frame_index]
                wrist_valid[side, frame_index] = True

        state_npz = state_dir / "v22_physical_state.npz"
        npz_write_started = time.monotonic()
        np.savez_compressed(
            state_npz,
            frame_idx=np.arange(state.frame_count, dtype=np.int32),
            T_world_camera=poses,
            droid_pose_valid=droid_pose_valid.astype(np.uint8),
            droid_pose_sampled=np.asarray(
                state.droid_records.coverage.pose_sampled
                if state.droid_records.coverage.pose_sampled is not None
                else state.droid_records.coverage.pose_valid,
                dtype=np.uint8,
            ),
            droid_source_frame_count=np.asarray([state.droid_records.coverage.source_frame_count], dtype=np.int32),
            droid_submitted_count=np.asarray([state.droid_records.coverage.submitted_count], dtype=np.int32),
            droid_unannotated_range=np.asarray(
                getattr(state.droid_records.coverage, "unannotated_range", None) or [-1, -1],
                dtype=np.int32,
            ),
            droid_coverage_status=np.asarray([state.droid_records.coverage.to_wire()["status"]]),
            droid_coverage_reason=np.asarray([state.droid_records.coverage.to_wire().get("reason") or ""]),
            pose_frame_idx=np.arange(state.frame_count, dtype=np.int32),
            pose_timestamp_s=np.asarray(timeline.timestamps_s, dtype=np.float64),
            T_camera_wrist=camera_wrist,
            T_world_wrist=world_wrist,
            wrist_pose_valid=wrist_valid.astype(np.uint8),
            pose_convention=np.asarray(["T_world_wrist = T_world_camera @ T_camera_wrist"]),
            pose_provenance=np.asarray(["same-frame MANO joint-0 wrist; camera/world rigid transform"]),
            droid_scale_scalar=np.asarray([float(droid_scale_provenance.get("scale", np.nan))], dtype=np.float64),
            droid_scale_provenance_json=np.asarray([json.dumps(droid_scale_provenance, sort_keys=True)]),
            K_canonical=k,
            left_vertices_world_m=world_vertices[0], right_vertices_world_m=world_vertices[1],
            left_joints_world_m=world_joints[0], right_joints_world_m=world_joints[1],
            left_vertices_source_px=np.asarray(state.timeline_inference.vertices_source_px.array[0], dtype=np.float32) if hasattr(state.timeline_inference, "vertices_source_px") else np.full((state.frame_count, 778, 2), np.nan, dtype=np.float32),
            right_vertices_source_px=np.asarray(state.timeline_inference.vertices_source_px.array[1], dtype=np.float32) if hasattr(state.timeline_inference, "vertices_source_px") else np.full((state.frame_count, 778, 2), np.nan, dtype=np.float32),
            left_joints_source_px=np.asarray(state.timeline_inference.joints_source_px.array[0], dtype=np.float32) if hasattr(state.timeline_inference, "joints_source_px") else np.full((state.frame_count, 21, 2), np.nan, dtype=np.float32),
            right_joints_source_px=np.asarray(state.timeline_inference.joints_source_px.array[1], dtype=np.float32) if hasattr(state.timeline_inference, "joints_source_px") else np.full((state.frame_count, 21, 2), np.nan, dtype=np.float32),
            left_valid=np.asarray(state.timeline_inference.valid.array[0], dtype=np.uint8),
            right_valid=np.asarray(state.timeline_inference.valid.array[1], dtype=np.uint8),
            left_uncertainty_m=np.asarray(state.timeline_inference.uncertainty_m.array[0], dtype=np.float32),
            right_uncertainty_m=np.asarray(state.timeline_inference.uncertainty_m.array[1], dtype=np.float32),
        )
        local_write_encode_s += time.monotonic() - npz_write_started

        validate_semantic_coverage(state.semantic_rows, state.frame_count)
        anomaly_count = sum(len(semantic_row_anomalies(row)) for row in state.semantic_rows)
        width, height = self.render_size
        combined_size = (width * 2, height)
        combined_path = renders / "v22_combined.mp4"
        writer = cv2.VideoWriter(str(combined_path), cv2.VideoWriter_fourcc(*"mp4v"), timeline.fps, combined_size)
        if not writer.isOpened():
            raise PhysicalAdapterError("could not open combined render writer")

        overlay_counts: dict[str, int] = {}
        world_counts = {"frames": 0, "frames_with_camera_pose": 0, "left_skeleton_primitives": 0, "right_skeleton_primitives": 0}
        semantic_cursor = 0
        rendered_anomaly_frames = 0
        try:
            for frame_index in range(state.frame_count):
                rgb = np.asarray(source.read_rgb(frame_index)).copy()
                overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                current_world_joints: list[np.ndarray] = [np.empty((0, 3), dtype=np.float32) for _ in range(2)]
                current_world_vertices: list[np.ndarray] = [np.empty((0, 3), dtype=np.float32) for _ in range(2)]

                for side_index, color in enumerate(HAND_COLORS):
                    if not bool(state.timeline_inference.valid.array[side_index, frame_index]):
                        continue
                    camera_vertices = np.asarray(state.timeline_inference.vertices_camera_m.array[side_index, frame_index])
                    camera_joints = np.asarray(state.timeline_inference.joints_camera_m.array[side_index, frame_index])
                    vertices_source_tensor = getattr(state.timeline_inference, "vertices_source_px", None)
                    joints_source_tensor = getattr(state.timeline_inference, "joints_source_px", None)
                    vertices_source_px = None if vertices_source_tensor is None else np.asarray(vertices_source_tensor.array[side_index, frame_index])
                    joints_source_px = None if joints_source_tensor is None else np.asarray(joints_source_tensor.array[side_index, frame_index])
                    if vertices_source_px is not None and joints_source_px is not None and np.isfinite(vertices_source_px).all() and np.isfinite(joints_source_px).all():
                        _add_counts(overlay_counts, _draw_projected_hand_2d(overlay, vertices_source_px, joints_source_px, color))
                    else:
                        _add_counts(overlay_counts, _draw_projected_hand(overlay, camera_vertices, camera_joints, k, color))
                    if droid_pose_valid[frame_index]:
                        current_world_joints[side_index] = world_joints[side_index][frame_index]
                        current_world_vertices[side_index] = world_vertices[side_index][frame_index]

                caption, semantic_cursor = caption_for_frame(state.semantic_rows, frame_index, semantic_cursor)
                anomalies = semantic_row_anomalies(state.semantic_rows[semantic_cursor])
                if anomalies:
                    rendered_anomaly_frames += 1
                draw_semantic_caption(overlay, caption, anomalies)

                # Keep the presentation view in one global world gauge; the
                # current physical camera and its oriented frustum move through it.
                relative_path = camera_centers[: frame_index + 1] if droid_pose_valid[frame_index] else np.empty((0, 3), dtype=np.float64)
                frame_view = replace(camera_centric_view, camera_anchor=poses[frame_index, :3, 3]) if droid_pose_valid[frame_index] else camera_centric_view
                world = _world_canvas(
                    current_world_joints, relative_path, (width, height), view=frame_view,
                    vertices=current_world_vertices,
                    camera_rotation=poses[frame_index, :3, :3] if droid_pose_valid[frame_index] else None,
                    frame_index=frame_index,
                )
                if droid_pose_valid[frame_index]:
                    world_counts["frames_with_camera_pose"] += 1
                    for side, joints in enumerate(current_world_joints):
                        if np.asarray(joints).shape == (21, 3):
                            world_counts[f"{'left' if side == 0 else 'right'}_skeleton_primitives"] += len(HAND_EDGES)
                else:
                    marker = f"DROID CAPACITY EXCEEDED: camera unavailable frames {state.droid_records.coverage.submitted_count}-{state.frame_count - 1}"
                    for canvas, canvas_width in ((overlay, timeline.width_px), (world, width)):
                        cv2.rectangle(canvas, (0, 0), (canvas_width, 40), (0, 0, 0), -1)
                        cv2.putText(canvas, marker, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 215, 255), 1, cv2.LINE_AA)

                combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
                combined[:, :width] = resize_to_cover(overlay, (width, height))
                combined[:, width:] = resize_to_cover(world, (width, height))
                write_started = time.monotonic()
                writer.write(combined)
                local_write_encode_s += time.monotonic() - write_started
                world_counts["frames"] += 1
        finally:
            release_started = time.monotonic()
            writer.release()
            local_write_encode_s += time.monotonic() - release_started

        report_view = camera_centric_view
        world_report = {
            "status": "ok",
            "pane": "right",
            "claim_scope": "Metric world-coordinate perspective visualization driven by DROID camera trajectory and timeline MANO hands; it is not a metric accuracy certificate or contact proof.",
            "frame_count": state.frame_count,
            "video_frame_count": world_counts["frames"],
            "render_size": [width, height],
            "view_mode": "fixed_clip_level_global_world_perspective_with_per_frame_physical_camera",
            "world_gauge_source": "DROID_T_world_camera_inverse_transform_and_timeline_MANO_world_m",
            "metric_extent_center_xyz_m": report_view.center.astype(float).tolist(),
            "metric_extent_span_m": float(report_view.span),
            "view_eye_xyz_m": report_view.eye.astype(float).tolist(),
            "view_target_xyz_m": report_view.target.astype(float).tolist(),
            "view_trajectory_forward_xyz": report_view.trajectory_forward.astype(float).tolist(),
            "view_focal_scale": float(report_view.focal_scale),
            "view_fit_extent": "fixed_clip_camera_and_hand_2nd_to_98th_percentile_world_extent",
            "camera_centric_reference_frame": camera_centric_reference_frame,
            "pose_state": {"frame_identity": "frame_idx", "timestamp": "frame_idx/fps", "wrist_valid_count": int(np.count_nonzero(wrist_valid)), "convention": "T_world_wrist = T_world_camera @ T_camera_wrist"},
            "camera_centric_display_rotation": camera_centric_display_rotation.astype(float).tolist(),
            "framing_window_frames": None,
            "framing_smoothing_frames": None,
            "camera_geometry_scale_m": float(report_view.camera_scale),
            "hand_render_style": "mano_21_bone_lines_with_world_vertex_cloud_and_joint_zoom_inset",
            "head_render_style": "camera_frustum_axes_and_blue_trajectory",
            "draw_counts": world_counts,
        }
        report = {
            "schema": "v22_combined_physical_semantic_render.v1",
            "status": "ok" if state.acceptance.accepted else "diagnostic_or_uncertain",
            "output_video": str(combined_path),
            "frame_count": state.frame_count,
            "video_frame_count": world_counts["frames"],
            "fps": timeline.fps,
            "duration_s": timeline.duration_s,
            "combined_size": list(combined_size),
            "pane_layout": {"left": "source_overlay_mano_and_cosmos_caption", "right": "metric_world_camera_and_mano"},
            "metric_state": bool(state.acceptance.accepted and not state.acceptance.diagnostic_only),
            "diagnostic_only": bool(state.acceptance.diagnostic_only),
            "acceptance_reasons": list(state.acceptance.reasons),
            "droid_coverage": state.droid_records.coverage.to_wire(),
            "droid_scale_provenance": droid_scale_provenance,
            "droid_sampled_frame_count": int(np.count_nonzero(np.asarray(state.droid_records.coverage.pose_sampled if state.droid_records.coverage.pose_sampled is not None else state.droid_records.coverage.pose_valid, dtype=bool))),
            "droid_capacity_marker_frame_count": int(np.count_nonzero(~droid_pose_valid)),
            "hawor_geometry_diagnostics": state.hawor_geometry_diagnostics,
            "overlay_render_style": "wilor_weak_perspective_crop_to_source_projection_when_available_else_metric_camera_pinhole",
            "overlay_draw_counts": overlay_counts,
            "semantic_render": {
                "row_count": len(state.semantic_rows),
                "anomaly_count": anomaly_count,
                "anomaly_annotated_frame_count": rendered_anomaly_frames,
                "coverage": {"start_frame": 0, "end_frame": state.frame_count, "fraction": 1.0},
                "source": "validated Cosmos semantic rows",
            },
            "world_render": world_report,
            "source_sha256": timeline.source_sha256,
            "state_npz": str(state_npz),
        }
        report_path = renders / "physical_adapter_report.json"
        report_write_started = time.monotonic()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        local_write_encode_s += time.monotonic() - report_write_started
        total_wall_s = time.monotonic() - render_started
        return PhysicalArtifactResult(
            str(root), str(state_npz), str(combined_path), str(report_path), state.frame_count, timeline.duration_s,
            bool(report["metric_state"]), bool(report["diagnostic_only"]),
            max(0.0, total_wall_s - local_write_encode_s), local_write_encode_s, total_wall_s,
        )


__all__ = ["HAND_EDGES", "PhysicalAdapterError", "PhysicalArtifactAdapter", "PhysicalArtifactResult", "project_points", "transform_points"]
