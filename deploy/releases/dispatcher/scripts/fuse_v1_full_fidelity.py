#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.optimize import minimize
from tqdm import tqdm

from run_v1_wilor_colmap import HAND_EDGES, caption_for_frame, load_actions, open_video


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)

LEFT_COLOR = (0, 210, 0)
RIGHT_COLOR = (0, 135, 255)
OBJECT_COLOR = (35, 45, 235)
TIP_IDS = [4, 8, 12, 16, 20]
HAND_SPAN_TARGET_M = 0.175
DEFAULT_MANO_RIGHT = Path("third_party/WiLoR/mano_data/MANO_RIGHT.pkl")
MANO_EDGE_STRIDE = 3

OBJECT_PROFILES: dict[str, dict] = {
    "tomato": {
        "prompts": ("tomato", "red tomato", "cut tomato", "tomato pieces"),
        "keywords": ("tomato",),
        "color_refinement": "tomato_red",
        "pose_type": "deformable_object_centroid_with_spherical_extent",
    },
    "mop": {
        "prompts": ("mop", "long-handled mop", "mop pole", "mop head"),
        "keywords": ("mop",),
        "pose_type": "articulated_or_long_tool_centroid_with_extent",
    },
    "pet_bed": {
        "prompts": ("pet bed", "animal bed", "floor pet bed"),
        "keywords": ("pet bed", "drag pet bed"),
        "pose_type": "deformable_or_rigid_object_centroid_with_extent",
    },
    "knife": {
        "prompts": ("knife", "kitchen knife", "cutting knife"),
        "keywords": ("knife",),
        "pose_type": "thin_rigid_tool_centroid_with_extent",
    },
    "cutting_board": {
        "prompts": ("cutting board", "wooden cutting board", "round cutting board"),
        "keywords": ("cutting board",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "cloth": {
        "prompts": ("cloth", "cleaning cloth", "rag", "blue cloth", "grey cloth"),
        "keywords": ("cloth", "wipe"),
        "pose_type": "deformable_object_centroid_with_extent",
    },
    "keyboard": {
        "prompts": ("keyboard", "black keyboard", "computer keyboard"),
        "keywords": ("keyboard",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "jar": {
        "prompts": ("jar", "small jar", "round jar"),
        "keywords": ("jar",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "box": {
        "prompts": ("box", "small box", "round box", "rectangular object"),
        "keywords": ("box", "round box", "small box", "rectangular object"),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "crochet": {
        "prompts": ("crochet hook", "hook", "yarn", "crocheted fabric", "white fabric"),
        "keywords": ("crochet", "hook", "yarn", "fabric"),
        "pose_type": "fine_tool_and_deformable_material_centroid_with_extent",
    },
    "clothing": {
        "prompts": ("clothing", "clothes", "coat", "black coat"),
        "keywords": ("clothing", "clothes", "coat"),
        "pose_type": "deformable_object_centroid_with_extent",
    },
    "suitcase": {
        "prompts": ("suitcase", "purple suitcase", "luggage"),
        "keywords": ("suitcase", "zip suitcase", "close suitcase"),
        "pose_type": "rigid_container_centroid_with_extent",
    },
    "plant": {
        "prompts": ("plant", "potted plant", "branch", "scissors", "wooden stick"),
        "keywords": ("plant", "branch", "prune", "scissors", "soil", "stick"),
        "pose_type": "thin_object_or_clutter_centroid_with_extent",
    },
    "dehumidifier": {
        "prompts": ("dehumidifier", "white dehumidifier"),
        "keywords": ("dehumidifier",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "fan": {
        "prompts": ("floor fan", "fan"),
        "keywords": ("fan", "floor fan"),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "trash_bag": {
        "prompts": ("trash bag", "garbage bag", "black trash bag", "white trash bag"),
        "keywords": ("trash_bag", "trash bag", "garbage bag", "line_trash_can", "open_trash_bag", "adjust_trash_bag"),
        "pose_type": "deformable_bag_or_container_centroid_with_extent",
    },
    "trash_can": {
        "prompts": ("trash can", "garbage can", "bin"),
        "keywords": ("move_trash_can", "trash can", "garbage can", "bin"),
        "pose_type": "rigid_container_centroid_with_extent",
    },
    "bowl": {
        "prompts": ("bowl", "white bowl", "porcelain bowl"),
        "keywords": ("bowl",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
    "phone": {
        "prompts": ("phone", "mobile phone", "smartphone"),
        "keywords": ("phone",),
        "pose_type": "rigid_object_centroid_with_extent",
    },
}


def hand_vertices_field(hand: dict, suffix: str = "") -> str:
    full = f"vertices{suffix}"
    sample = f"vertices{suffix}_sample"
    if full in hand:
        return full
    if sample in hand:
        return sample
    raise RuntimeError(f"hand record has neither {full} nor {sample}")


def hand_vertices(hand: dict, suffix: str = "") -> np.ndarray:
    return np.asarray(hand[hand_vertices_field(hand, suffix)], dtype=float)


def mano_faces(path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"MANO topology file does not exist: {path}")
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    with path.open("rb") as f, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Please import `csc_matrix` from the `scipy.sparse` namespace.*",
        )
        data = pickle.load(f, encoding="latin1")
    return np.asarray(data["f"], dtype=int)


def mano_edges_from_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    if faces.size == 0 or vertex_count < 3:
        return np.empty((0, 2), dtype=int)
    if int(faces.max()) >= vertex_count:
        return np.empty((0, 2), dtype=int)
    pairs = set()
    for tri in faces:
        a, b, c = map(int, tri)
        if a < vertex_count and b < vertex_count and c < vertex_count:
            pairs.add(tuple(sorted((a, b))))
            pairs.add(tuple(sorted((b, c))))
            pairs.add(tuple(sorted((c, a))))
    edges = np.asarray(sorted(pairs), dtype=int)
    return edges[::MANO_EDGE_STRIDE] if len(edges) > MANO_EDGE_STRIDE else edges


@dataclass(frozen=True)
class RenderSpec:
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class WorldProjector:
    basis: np.ndarray
    q_center: np.ndarray
    pixels_per_meter: float
    screen_center: tuple[float, float]
    size: tuple[int, int]


@dataclass(frozen=True)
class ObjectMeshFrame:
    vertices: np.ndarray
    faces: np.ndarray


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_idx}")
    return frame


def choose_hand_by_side(hands: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for hand in hands:
        side = hand["side"]
        cur = out.get(side)
        if cur is None or float(hand.get("detector_score", 0.0)) > float(cur.get("detector_score", 0.0)):
            out[side] = hand
    return out


def hand_vector(hand: dict) -> np.ndarray:
    vertices_key = hand_vertices_field(hand, "_camera")
    fields = [
        np.asarray(hand["bbox_xyxy"], dtype=float).reshape(-1),
        np.asarray(hand["cam_t"], dtype=float).reshape(-1),
        np.asarray(hand["joints3d_camera"], dtype=float).reshape(-1),
        np.asarray(hand[vertices_key], dtype=float).reshape(-1),
    ]
    return np.concatenate(fields)


def project_points(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = np.clip(points_camera_m[:, 2], 1e-6, None)
    return np.c_[fx * points_camera_m[:, 0] / z + cx, fy * points_camera_m[:, 1] / z + cy]


def solve_source_camera_translation(local_points_m: np.ndarray, points2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    qx = (points2d[:, 0] - cx) / fx
    qy = (points2d[:, 1] - cy) / fy
    rows = []
    rhs = []
    for (x, y, z), u, v in zip(local_points_m, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), rcond=None)
    return trans.astype(float)


def hand_metric_scale_from_raw(raw_frames: list[dict]) -> dict:
    spans = []
    for frame in raw_frames:
        for hand in frame["raw_hands"]:
            joints = np.asarray(hand["joints3d_camera"], dtype=float)
            dist = float(np.linalg.norm(joints[12] - joints[0]))
            if 0.04 < dist < 0.20:
                spans.append(dist)
    if not spans:
        raise RuntimeError("no plausible WiLoR MANO spans for source-camera metric solve")
    spans_arr = np.asarray(spans, dtype=float)
    median_span = float(np.median(spans_arr))
    scale = HAND_SPAN_TARGET_M / median_span
    residual = spans_arr * scale - HAND_SPAN_TARGET_M
    return {
        "status": "wilor_local_hand_geometry_scaled_by_wrist_to_middle_tip",
        "target_wrist_to_middle_tip_m": HAND_SPAN_TARGET_M,
        "median_wilor_wrist_to_middle_tip": median_span,
        "wilor_local_to_meters": scale,
        "sample_count": len(spans),
        "residual_iqr_m": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
    }


def normalize_hand_to_source_camera(hand: dict, intrinsics: np.ndarray, wilor_to_meters: float) -> dict | None:
    local_joints_m = np.asarray(hand["joints3d_camera"], dtype=float) * wilor_to_meters
    raw_vertex_key = hand_vertices_field(hand, "_camera")
    output_vertex_key = "vertices_camera" if raw_vertex_key == "vertices_camera" else "vertices_camera_sample"
    output_source_key = "vertices_source_camera_m" if output_vertex_key == "vertices_camera" else "vertices_source_camera_m_sample"
    local_verts_m = np.asarray(hand[raw_vertex_key], dtype=float) * wilor_to_meters
    raw_joints2d = np.asarray(hand["joints2d_raw"], dtype=float)
    trans_m = solve_source_camera_translation(local_joints_m, raw_joints2d, intrinsics)
    joints_camera_m = local_joints_m + trans_m
    projected = project_points(joints_camera_m, intrinsics)
    reproj = np.linalg.norm(projected - raw_joints2d, axis=1)
    depth = float(np.median(joints_camera_m[:, 2]))
    median_err = float(np.median(reproj))
    p95_err = float(np.percentile(reproj, 95))
    if not (0.2 <= depth <= 3.5 and median_err <= 45.0 and np.isfinite(joints_camera_m).all()):
        return None
    return {
        "backend": hand.get("backend", "WiLoR"),
        "side": hand["side"],
        "detector_score": float(hand.get("detector_score", 0.0)),
        "bbox_xyxy": np.asarray(hand["bbox_xyxy"], dtype=float).tolist(),
        "cam_t": trans_m.astype(float).tolist(),
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "joints3d_camera": local_joints_m.astype(float).tolist(),
        "joints3d_source_camera_m": joints_camera_m.astype(float).tolist(),
        "joints2d_raw": raw_joints2d.astype(float).tolist(),
        "joints2d": projected.astype(float).tolist(),
        "mano_params": hand.get("mano_params", {}),
        output_vertex_key: local_verts_m.astype(float).tolist(),
        output_source_key: (local_verts_m + trans_m).astype(float).tolist(),
        "filter_status": "measured_source_camera_solve",
        "source_camera_solve": {
            "status": "least_squares_translation_from_mano_local_geometry_and_2d_keypoints",
            "wilor_virtual_focal_length": float(hand.get("focal_length", 0.0)),
            "wilor_virtual_cam_t": np.asarray(hand.get("cam_t", [0.0, 0.0, 0.0]), dtype=float).tolist(),
            "median_reprojection_error_px": median_err,
            "p95_reprojection_error_px": p95_err,
            "median_depth_m": depth,
        },
    }


def vector_to_hand(template: dict, vec: np.ndarray, status: str, measurement_score: float, intrinsics: np.ndarray) -> dict:
    cursor = 0

    def take(shape):
        nonlocal cursor
        n = int(np.prod(shape))
        arr = vec[cursor : cursor + n].reshape(shape)
        cursor += n
        return arr

    template_vertex_key = hand_vertices_field(template, "_camera")
    template_vertex_shape = np.asarray(template[template_vertex_key], dtype=float).shape
    output_source_key = "vertices_source_camera_m" if template_vertex_key == "vertices_camera" else "vertices_source_camera_m_sample"
    output_world_key = "vertices_world_m" if template_vertex_key == "vertices_camera" else "vertices_world_m_sample"
    hand = {
        "backend": template.get("backend", "WiLoR"),
        "side": template["side"],
        "detector_score": float(measurement_score),
        "bbox_xyxy": take((4,)).astype(float).tolist(),
        "cam_t": take((3,)).astype(float).tolist(),
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "joints3d_camera": take((21, 3)).astype(float).tolist(),
        template_vertex_key: take(template_vertex_shape).astype(float).tolist(),
        "mano_params": template.get("mano_params", {}),
        "filter_status": status,
        "source_camera_solve": template.get("source_camera_solve", {}),
        "mano_vertex_count": int(template_vertex_shape[0]),
    }
    cam_t = np.asarray(hand["cam_t"], dtype=float)
    joints_camera_m = np.asarray(hand["joints3d_camera"], dtype=float) + cam_t
    verts_camera_m = np.asarray(hand[template_vertex_key], dtype=float) + cam_t
    hand["joints3d_source_camera_m"] = joints_camera_m.astype(float).tolist()
    hand[output_source_key] = verts_camera_m.astype(float).tolist()
    hand["mano_surface_status"] = "full_mano_vertices" if output_world_key == "vertices_world_m" else "sampled_mano_vertices"
    projected = project_points(joints_camera_m, intrinsics)
    hand["joints2d"] = projected.astype(float).tolist()
    if template.get("joints2d_raw") is not None:
        raw = np.asarray(template["joints2d_raw"], dtype=float)
        hand["joints2d_raw"] = raw.astype(float).tolist()
        err = np.linalg.norm(projected - raw, axis=1)
        hand["projection_residual_to_measurement_px"] = {
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)),
        }
    return hand


def kalman_rts(
    measurements: list[np.ndarray | None],
    confidences: list[float],
    fps: float,
    measurement_sigma: np.ndarray | float,
    process_position_sigma: np.ndarray | float,
    process_velocity_sigma: np.ndarray | float,
) -> tuple[list[np.ndarray], list[str]]:
    measured = [i for i, x in enumerate(measurements) if x is not None]
    if not measured:
        raise RuntimeError("no measurements for Kalman smoothing")
    dim = int(measurements[measured[0]].shape[0])  # type: ignore[index,union-attr]
    n = len(measurements)
    dt = 1.0 / fps
    meas_sigma = np.broadcast_to(np.asarray(measurement_sigma, dtype=float), (dim,))
    proc_pos = np.broadcast_to(np.asarray(process_position_sigma, dtype=float), (dim,))
    proc_vel = np.broadcast_to(np.asarray(process_velocity_sigma, dtype=float), (dim,))

    pos = np.asarray(measurements[measured[0]], dtype=float).copy()  # type: ignore[index]
    vel = np.zeros(dim, dtype=float)
    P00 = np.full(dim, 10.0, dtype=float)
    P01 = np.zeros(dim, dtype=float)
    P11 = np.full(dim, 10.0, dtype=float)

    pos_f = np.zeros((n, dim), dtype=float)
    vel_f = np.zeros((n, dim), dtype=float)
    P00_f = np.zeros((n, dim), dtype=float)
    P01_f = np.zeros((n, dim), dtype=float)
    P11_f = np.zeros((n, dim), dtype=float)
    pos_p = np.zeros((n, dim), dtype=float)
    vel_p = np.zeros((n, dim), dtype=float)
    P00_p = np.zeros((n, dim), dtype=float)
    P01_p = np.zeros((n, dim), dtype=float)
    P11_p = np.zeros((n, dim), dtype=float)

    for i in range(n):
        pred_pos = pos + dt * vel
        pred_vel = vel.copy()
        pred_P00 = P00 + 2.0 * dt * P01 + dt * dt * P11 + proc_pos * proc_pos
        pred_P01 = P01 + dt * P11
        pred_P11 = P11 + proc_vel * proc_vel
        pos_p[i], vel_p[i] = pred_pos, pred_vel
        P00_p[i], P01_p[i], P11_p[i] = pred_P00, pred_P01, pred_P11
        z = measurements[i]
        if z is None:
            pos, vel = pred_pos, pred_vel
            P00, P01, P11 = pred_P00, pred_P01, pred_P11
        else:
            conf = max(0.05, float(confidences[i]))
            R = (meas_sigma / conf) ** 2
            innovation = np.asarray(z, dtype=float) - pred_pos
            S = pred_P00 + R
            K0 = pred_P00 / S
            K1 = pred_P01 / S
            pos = pred_pos + K0 * innovation
            vel = pred_vel + K1 * innovation
            P00 = (1.0 - K0) * pred_P00
            P01 = (1.0 - K0) * pred_P01
            P11 = pred_P11 - K1 * pred_P01
        pos_f[i], vel_f[i] = pos, vel
        P00_f[i], P01_f[i], P11_f[i] = P00, P01, P11

    pos_s = pos_f.copy()
    vel_s = vel_f.copy()
    for i in range(n - 2, -1, -1):
        m00 = P00_f[i] + dt * P01_f[i]
        m01 = P01_f[i]
        m10 = P01_f[i] + dt * P11_f[i]
        m11 = P11_f[i]
        a = P00_p[i + 1]
        b = P01_p[i + 1]
        c = P11_p[i + 1]
        det = np.maximum(a * c - b * b, 1e-12)
        C00 = (m00 * c - m01 * b) / det
        C01 = (-m00 * b + m01 * a) / det
        C10 = (m10 * c - m11 * b) / det
        C11 = (-m10 * b + m11 * a) / det
        dpos = pos_s[i + 1] - pos_p[i + 1]
        dvel = vel_s[i + 1] - vel_p[i + 1]
        pos_s[i] = pos_f[i] + C00 * dpos + C01 * dvel
        vel_s[i] = vel_f[i] + C10 * dpos + C11 * dvel

    statuses = ["measured_kalman_rts" if measurements[i] is not None else "predicted_kalman_rts" for i in range(n)]
    return [pos_s[i] for i in range(n)], statuses


def hand_kalman_sigmas(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_dim = 4 + 3 + 21 * 3
    vertex_dim = dim - fixed_dim
    if vertex_dim <= 0 or vertex_dim % 3 != 0:
        raise RuntimeError(f"unexpected hand state dimension {dim}")
    measurement = np.concatenate(
        [
            np.full(4, 18.0),
            np.full(3, 0.035),
            np.full(21 * 3, 0.018),
            np.full(vertex_dim, 0.018),
        ]
    )
    process_position = np.concatenate(
        [
            np.full(4, 4.0),
            np.full(3, 0.006),
            np.full(21 * 3, 0.004),
            np.full(vertex_dim, 0.004),
        ]
    )
    process_velocity = np.concatenate(
        [
            np.full(4, 60.0),
            np.full(3, 0.22),
            np.full(21 * 3, 0.12),
            np.full(vertex_dim, 0.12),
        ]
    )
    return measurement, process_position, process_velocity


def smooth_hands(raw_frames: list[dict], fps: float, intrinsics: np.ndarray, wilor_to_meters: float) -> tuple[list[dict], dict]:
    rejected = {"left": 0, "right": 0, "unknown": 0}
    normalized_frames = []
    for frame in raw_frames:
        normalized = []
        for raw_hand in frame["raw_hands"]:
            hand = normalize_hand_to_source_camera(raw_hand, intrinsics, wilor_to_meters)
            if hand is None:
                rejected[raw_hand.get("side", "unknown")] = rejected.get(raw_hand.get("side", "unknown"), 0) + 1
            else:
                normalized.append(hand)
        normalized_frames.append({"raw_hands": normalized})
    chosen = [choose_hand_by_side(frame["raw_hands"]) for frame in normalized_frames]
    out_frames = [{"frame_idx": f["frame_idx"], "time_s": f["time_s"], "caption": f["caption"], "hands": []} for f in raw_frames]
    stats: dict[str, dict] = {"rejected_source_camera_solves": rejected}
    for side in ["left", "right"]:
        templates = [c[side] for c in chosen if side in c]
        if not templates:
            continue
        template = templates[0]
        measurements = []
        confs = []
        best_by_frame = []
        for c in chosen:
            hand = c.get(side)
            best_by_frame.append(hand)
            measurements.append(hand_vector(hand) if hand is not None else None)
            confs.append(float(hand.get("detector_score", 0.0)) if hand is not None else 0.0)
        measured_indices = [i for i, m in enumerate(measurements) if m is not None]
        first_measured = measured_indices[0]
        last_measured = measured_indices[-1]
        dim = int(measurements[first_measured].shape[0])  # type: ignore[union-attr]
        meas_sigma, proc_pos, proc_vel = hand_kalman_sigmas(dim)
        smoothed, statuses = kalman_rts(measurements, confs, fps, meas_sigma, proc_pos, proc_vel)
        measured_count = 0
        predicted_count = 0
        outside_visibility = 0
        projection_medians = []
        for i, vec in enumerate(smoothed):
            if i < first_measured or i > last_measured:
                outside_visibility += 1
                continue
            source = best_by_frame[i] or template
            score = confs[i]
            hand = vector_to_hand(source, vec, statuses[i], score, intrinsics)
            hand["measurement_available"] = best_by_frame[i] is not None
            if hand["measurement_available"]:
                measured_count += 1
                projection_medians.append(float(hand.get("projection_residual_to_measurement_px", {}).get("median", math.nan)))
            else:
                predicted_count += 1
            out_frames[i]["hands"].append(hand)
        stats[side] = {
            "measured_frames": measured_count,
            "predicted_frames": predicted_count,
            "outside_visibility_frames": outside_visibility,
            "coverage": measured_count / max(1, len(raw_frames)),
            "median_projection_residual_px": float(np.nanmedian(projection_medians)) if projection_medians else None,
        }
    return out_frames, stats


def load_droid_reconstruction(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu")
    required = {"tstamps", "disps", "intrinsics"}
    missing = sorted(required.difference(blob))
    if missing:
        raise RuntimeError(f"DROID reconstruction missing keys: {missing}")
    return {
        "tstamps": blob["tstamps"].detach().cpu().numpy().astype(int),
        "disps": blob["disps"].detach().cpu().numpy().astype(float),
        "intrinsics": blob["intrinsics"].detach().cpu().numpy().astype(float),
        "depth_level": blob.get("depth_level", "unknown"),
    }


def sample_droid_depth_relative(recon: dict, frame_idx: int, point_xy: np.ndarray, image_size: tuple[int, int], max_keyframe_gap: int) -> tuple[float, int] | None:
    tstamps = recon["tstamps"]
    nearest = int(np.argmin(np.abs(tstamps - frame_idx)))
    source_idx = int(tstamps[nearest])
    if abs(source_idx - frame_idx) > max_keyframe_gap:
        return None
    disps = recon["disps"][nearest]
    width, height = image_size
    x = float(point_xy[0]) / width * disps.shape[1]
    y = float(point_xy[1]) / height * disps.shape[0]
    xi = int(np.clip(round(x), 0, disps.shape[1] - 1))
    yi = int(np.clip(round(y), 0, disps.shape[0] - 1))
    patch = disps[max(0, yi - 1) : min(disps.shape[0], yi + 2), max(0, xi - 1) : min(disps.shape[1], xi + 2)]
    valid = patch[np.isfinite(patch) & (patch > 1e-4)]
    if valid.size == 0:
        return None
    return 1.0 / float(np.median(valid)), source_idx


def estimate_droid_metric_scale(frames: list[dict], recon: dict, image_size: tuple[int, int], max_keyframe_gap: int) -> dict:
    ratios = []
    samples = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        for hand in frame["hands"]:
            if not hand.get("measurement_available", False):
                continue
            points2d = np.asarray(hand["joints2d"], dtype=float)
            center = np.median(points2d, axis=0)
            sampled = sample_droid_depth_relative(recon, frame_idx, center, image_size, max_keyframe_gap)
            if sampled is None:
                continue
            depth_rel, keyframe_idx = sampled
            hand_depth_m = float(np.median(np.asarray(hand["joints3d_source_camera_m"], dtype=float)[:, 2]))
            if 0.2 <= hand_depth_m <= 3.5 and 0.05 <= depth_rel <= 10.0:
                ratio = hand_depth_m / depth_rel
                ratios.append(ratio)
                samples.append(
                    {
                        "frame_idx": frame_idx,
                        "keyframe_idx": keyframe_idx,
                        "side": hand["side"],
                        "hand_depth_m": hand_depth_m,
                        "droid_depth_relative": depth_rel,
                        "droid_to_meters": ratio,
                    }
                )
    if len(ratios) < 12:
        raise RuntimeError(f"insufficient DROID depth scale samples: {len(ratios)}")
    arr = np.asarray(ratios, dtype=float)
    median = float(np.median(arr))
    residual = arr / median - 1.0
    return {
        "status": "droid_relative_depth_scaled_to_source_camera_mano_depth",
        "droid_to_meters": median,
        "sample_count": len(ratios),
        "ratio_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "relative_residual_iqr": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
        "samples_preview": samples[:20],
    }


def estimate_droid_metric_scale_from_raw(
    raw_frames: list[dict],
    intrinsics: np.ndarray,
    wilor_to_meters: float,
    recon: dict,
    image_size: tuple[int, int],
    max_keyframe_gap: int,
) -> dict:
    keyframes = recon["tstamps"]
    ratios = []
    samples = []
    for frame in raw_frames:
        frame_idx = int(frame["frame_idx"])
        if int(np.min(np.abs(keyframes - frame_idx))) > max_keyframe_gap:
            continue
        normalized = []
        for raw_hand in frame["raw_hands"]:
            hand = normalize_hand_to_source_camera(raw_hand, intrinsics, wilor_to_meters)
            if hand is not None:
                normalized.append(hand)
        for hand in choose_hand_by_side(normalized).values():
            points2d = np.asarray(hand["joints2d"], dtype=float)
            center = np.median(points2d, axis=0)
            sampled = sample_droid_depth_relative(recon, frame_idx, center, image_size, max_keyframe_gap)
            if sampled is None:
                continue
            depth_rel, keyframe_idx = sampled
            hand_depth_m = float(np.median(np.asarray(hand["joints3d_source_camera_m"], dtype=float)[:, 2]))
            if 0.2 <= hand_depth_m <= 3.5 and 0.05 <= depth_rel <= 10.0:
                ratio = hand_depth_m / depth_rel
                ratios.append(ratio)
                samples.append(
                    {
                        "frame_idx": frame_idx,
                        "keyframe_idx": keyframe_idx,
                        "side": hand["side"],
                        "hand_depth_m": hand_depth_m,
                        "droid_depth_relative": depth_rel,
                        "droid_to_meters": ratio,
                    }
                )
    if len(ratios) < 12:
        raise RuntimeError(f"insufficient full-clip DROID depth scale samples: {len(ratios)}")
    arr = np.asarray(ratios, dtype=float)
    median = float(np.median(arr))
    residual = arr / median - 1.0
    return {
        "status": "full_clip_droid_relative_depth_scaled_to_source_camera_mano_depth",
        "droid_to_meters": median,
        "sample_count": len(ratios),
        "ratio_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "relative_residual_iqr": [
            float(np.percentile(residual, 25)),
            float(np.percentile(residual, 75)),
        ],
        "samples_preview": samples[:20],
    }


def transform_hands_to_world(frames: list[dict], droid_npz: Path, droid_to_meters: float) -> np.ndarray:
    droid = np.load(droid_npz)
    dense = droid["T_world_camera"].astype(float)
    if len(dense) < max(int(frame["frame_idx"]) for frame in frames) + 1:
        raise RuntimeError(f"DROID frames {len(dense)} do not cover requested source frames")
    T = dense[np.asarray([int(frame["frame_idx"]) for frame in frames], dtype=int)]
    T_metric = T.copy()
    T_metric[:, :3, 3] *= droid_to_meters
    for i, frame in enumerate(frames):
        frame["camera"] = {
            "T_world_camera_metric": T_metric[i].tolist(),
            "position_world_m": T_metric[i, :3, 3].astype(float).tolist(),
        }
        for hand in frame["hands"]:
            joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
            verts = hand_vertices(hand, "_source_camera_m")
            joints_h = np.c_[joints, np.ones(len(joints))]
            verts_h = np.c_[verts, np.ones(len(verts))]
            hand["joints3d_world_m"] = (T_metric[i] @ joints_h.T).T[:, :3].astype(float).tolist()
            world_key = "vertices_world_m" if len(verts) > 100 else "vertices_world_m_sample"
            hand[world_key] = (T_metric[i] @ verts_h.T).T[:, :3].astype(float).tolist()
            hand["world_coordinate_status"] = "source_camera_mano_metric_transformed_by_droid_metric_camera_pose"
    return T_metric


def object_profile(label: str) -> dict:
    normalized = label.lower().replace(" ", "_")
    if normalized in OBJECT_PROFILES:
        return OBJECT_PROFILES[normalized]
    prompts = tuple(dict.fromkeys((label, label.replace("_", " "))))
    return {
        "prompts": prompts,
        "keywords": tuple(token for token in label.lower().replace("_", " ").split() if len(token) > 2),
        "pose_type": "object_centroid_with_extent",
    }


def infer_object_label(actions: list[dict], requested: str) -> str:
    if requested != "auto":
        return requested
    scores: dict[str, float] = {label: 0.0 for label in OBJECT_PROFILES}
    for action in actions:
        label, confidence = infer_action_object_label(action)
        duration = max(1, int(action.get("end_frame", 0)) - int(action.get("start_frame", 0)))
        scores[label] = scores.get(label, 0.0) + confidence * duration
    best_label = max(scores, key=lambda label: scores[label])
    if scores[best_label] <= 0.0:
        raise RuntimeError("could not infer manipulated object label from action metadata; pass --object-label")
    return best_label


def infer_action_object_label(action: dict) -> tuple[str, float]:
    text = f"{action.get('action', '')} {action.get('description', '')}".lower().replace("_", " ")
    best_label = None
    best_score = 0.0
    for label, profile in OBJECT_PROFILES.items():
        score = float(sum(text.count(keyword.replace("_", " ")) for keyword in profile["keywords"]))
        if action.get("action", "").lower().endswith(label):
            score += 2.0
        if label in action.get("action", "").lower().replace("_", " "):
            score += 1.5
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return "unknown", 0.0
    verb_only = action.get("action", "").lower().replace("_", " ")
    ambiguous_object_words = {"object", "item", "items", "desk", "table", "surface", "floor"}
    if best_label == "cloth" and "wipe" in verb_only and not any(word in text for word in ("cloth", "rag", "towel")):
        return "unknown", 0.0
    if best_score <= 1.0 and any(word in verb_only.split() for word in ambiguous_object_words):
        return "unknown", 0.0
    return best_label, best_score


def object_label_for_action(action: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    label, score = infer_action_object_label(action)
    if score <= 0.0:
        raise RuntimeError(f"could not infer manipulated object for action: {action}")
    return label


def action_relevance(action: dict, object_label: str) -> int:
    profile = object_profile(object_label)
    text = f"{action.get('action', '')} {action.get('description', '')}".lower().replace("_", " ")
    return sum(text.count(keyword.replace("_", " ")) for keyword in profile["keywords"])


def manipulated_object_interval(actions: list[dict], label: str) -> tuple[int, int]:
    relevant = [
        (int(action["start_frame"]), int(action["end_frame"]))
        for action in actions
        if action_relevance(action, label) > 0
    ]
    if not relevant:
        raise RuntimeError(f"no semantic interval mentions object label: {label}")
    return min(s for s, _ in relevant), max(e for _, e in relevant)


def active_object_hands(frame_ann: dict) -> list[dict]:
    caption = frame_ann.get("caption", "").lower()
    wanted: set[str] = set()
    if "both hands" in caption or "both hand" in caption:
        wanted = {"left", "right"}
    if "left hand" in caption:
        wanted.add("left")
    if "right hand" in caption:
        wanted.add("right")
    selected = [hand for hand in frame_ann["hands"] if hand["side"] in wanted]
    return selected if selected else frame_ann["hands"]


def hand_association_geometry(frame_ann: dict) -> dict:
    hands = active_object_hands(frame_ann)
    points = []
    boxes = []
    tip_points = []
    for hand in hands:
        joints = np.asarray(hand["joints2d"], dtype=float)
        points.extend(joints)
        tip_points.extend(joints[TIP_IDS])
        boxes.append(np.asarray(hand["bbox_xyxy"], dtype=float))
    return {
        "points": np.asarray(points, dtype=float) if points else np.zeros((0, 2), dtype=float),
        "tips": np.asarray(tip_points, dtype=float) if tip_points else np.zeros((0, 2), dtype=float),
        "boxes": boxes,
    }


def caption_phase(frame_ann: dict) -> str:
    caption = frame_ann.get("caption", "").lower()
    if "scrape" in caption:
        return "scrape"
    if "chop" in caption:
        return "chop"
    if "place" in caption:
        return "place"
    return "other"


def point_box_distance(point: np.ndarray, box: np.ndarray, pad: float) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - pad - point[0], 0.0, point[0] - x2 - pad)
    dy = max(y1 - pad - point[1], 0.0, point[1] - y2 - pad)
    return float(math.hypot(dx, dy))


def association_distance(center: np.ndarray, geom: dict) -> float:
    dists = []
    points = geom["points"]
    if points.size:
        dists.append(float(np.linalg.norm(points - center[None, :], axis=1).min()))
    for box in geom["boxes"]:
        dists.append(point_box_distance(center, box, pad=90.0))
    return min(dists) if dists else math.inf


def tomato_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 120, 70), (12, 255, 255)),
        cv2.inRange(hsv, (168, 120, 70), (180, 255, 255)),
    ).astype(bool)


def red_mask_boxes(frame: np.ndarray, geom: dict, prev_box: list[float] | None) -> list[dict]:
    mask = tomato_red_mask(frame).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict] = []
    prev_center = None
    if prev_box is not None:
        p = np.asarray(prev_box, dtype=float)
        prev_center = 0.5 * (p[:2] + p[2:])
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60 or area > 18000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 4.5:
            continue
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        assoc_dist = association_distance(center, geom)
        prev_dist = math.inf if prev_center is None else float(np.linalg.norm(prev_center - center))
        if min(assoc_dist, prev_dist) > 230.0:
            continue
        peri = cv2.arcLength(c, True)
        circularity = float(4.0 * math.pi * area / (peri * peri + 1e-6))
        score = 0.05 + 0.18 * circularity + 0.28 * math.exp(-min(assoc_dist, 260.0) / 95.0)
        if prev_center is not None:
            score += 0.12 * math.exp(-min(prev_dist, 260.0) / 90.0)
        pad = 18.0
        box = [
            float(max(0.0, x - pad)),
            float(max(0.0, y - pad)),
            float(min(frame.shape[1] - 1.0, x + w + pad)),
            float(min(frame.shape[0] - 1.0, y + h + pad)),
        ]
        boxes.append(
            {
                "box": box,
                "score": score,
                "label": "compact_red_contact_component",
                "source": "red_contact_component",
                "association_dist_px": assoc_dist,
                "prev_dist_px": prev_dist,
                "red_component_area_px": float(area),
                "red_component_circularity": circularity,
            }
        )
    boxes.sort(key=lambda item: item["score"], reverse=True)
    return boxes[:8]


def refine_deformable_tomato_mask(frame: np.ndarray, mask: np.ndarray, frame_ann: dict, geom: dict) -> tuple[np.ndarray, str]:
    phase = caption_phase(frame_ann)
    if phase not in {"chop", "scrape"}:
        return mask, "sam_single_mask"
    red = tomato_red_mask(frame).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_ys, base_xs = np.where(mask)
    if len(base_xs) == 0:
        return mask, "sam_single_mask"
    base_center = np.asarray([base_xs.mean(), base_ys.mean()], dtype=float)
    selected = np.zeros_like(red, dtype=np.uint8)
    selected[mask] = 1
    radius = 260.0 if phase == "scrape" else 340.0
    max_area_ratio = 6.0 if phase == "scrape" else 4.5
    max_width = 420 if phase == "scrape" else 460
    max_height = 360 if phase == "scrape" else 620
    max_component_area = 15000 if phase == "scrape" else 24000
    for c in contours:
        area = cv2.contourArea(c)
        if area < 45 or area > max_component_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        assoc = association_distance(center, geom)
        base_dist = float(np.linalg.norm(center - base_center))
        if assoc <= 135.0 and base_dist <= radius:
            cv2.drawContours(selected, [c], -1, 1, thickness=-1)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    if int(selected.sum()) < int(mask.sum()):
        return mask, "sam_single_mask"
    if int(selected.sum()) > max_area_ratio * max(1, int(mask.sum())):
        return mask, "sam_single_mask"
    ys, xs = np.where(selected)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    if (xs.max() - xs.min() + 1) > max_width or (ys.max() - ys.min() + 1) > max_height:
        return mask, "sam_single_mask"
    return selected, f"{phase}_red_component_union"


def warp_mask_forward(
    flow_estimator,
    prev_frame: np.ndarray,
    frame: np.ndarray,
    prev_mask: np.ndarray,
) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = flow_estimator.calc(prev_gray, gray, None)
    ys, xs = np.where(prev_mask)
    warped = np.zeros(prev_mask.shape, dtype=np.uint8)
    if len(xs) == 0:
        return warped.astype(bool)
    moved_x = np.rint(xs.astype(np.float32) + flow[ys, xs, 0]).astype(np.int32)
    moved_y = np.rint(ys.astype(np.float32) + flow[ys, xs, 1]).astype(np.int32)
    valid = (0 <= moved_x) & (moved_x < prev_mask.shape[1]) & (0 <= moved_y) & (moved_y < prev_mask.shape[0])
    warped[moved_y[valid], moved_x[valid]] = 1
    warped = cv2.dilate(warped, np.ones((3, 3), np.uint8), iterations=2)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return warped.astype(bool)


def fuse_temporal_object_mask(
    flow_estimator,
    prev_frame: np.ndarray | None,
    frame: np.ndarray,
    prev_mask: np.ndarray | None,
    mask: np.ndarray,
    frame_ann: dict,
    geom: dict,
) -> tuple[np.ndarray, str]:
    phase = caption_phase(frame_ann)
    if phase != "scrape" or prev_frame is None or prev_mask is None:
        return mask, "sam_single_mask"
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    current_center = np.asarray([xs.mean(), ys.mean()], dtype=float)
    warped = warp_mask_forward(flow_estimator, prev_frame, frame, prev_mask)
    if int(warped.sum()) < 80:
        return mask, "sam_single_mask"
    selected = mask.astype(np.uint8)
    current_dilated = cv2.dilate(mask.astype(np.uint8), np.ones((71, 71), np.uint8), iterations=1).astype(bool)
    contours, _ = cv2.findContours(warped.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_center_dist = 280.0 if phase == "scrape" else 340.0
    max_width = 520 if phase == "scrape" else 560
    max_height = 520 if phase == "scrape" else 660
    for c in contours:
        area = cv2.contourArea(c)
        if area < 45 or area > 90000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        center = np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)
        component = np.zeros_like(selected, dtype=np.uint8)
        cv2.drawContours(component, [c], -1, 1, thickness=-1)
        center_dist = float(np.linalg.norm(center - current_center))
        has_current_overlap = bool(np.logical_and(component.astype(bool), current_dilated).any())
        if center_dist > max_center_dist and not has_current_overlap:
            continue
        if association_distance(center, geom) > 180.0 and center_dist > 140.0:
            continue
        selected[component.astype(bool)] = 1
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    if int(selected.sum()) <= int(mask.sum()) * 1.05:
        return mask, "sam_single_mask"
    ys, xs = np.where(selected)
    if len(xs) == 0:
        return mask, "sam_single_mask"
    if (xs.max() - xs.min() + 1) > max_width or (ys.max() - ys.min() + 1) > max_height:
        return mask, "sam_single_mask"
    if int(selected.sum()) > 120000:
        return mask, "sam_single_mask"
    return selected, f"{phase}_optical_flow_temporal_union"


def hand_contact_points(frame_ann: dict) -> np.ndarray:
    pts = []
    for hand in frame_ann["hands"]:
        joints = np.asarray(hand["joints2d"], dtype=float)
        pts.extend(joints[TIP_IDS])
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2), dtype=float)


def mask_contact_score(mask: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    if points.size == 0:
        return 0.0, math.inf
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, math.inf
    center_dist = []
    inside = 0
    dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    for p in points:
        x = int(np.clip(round(p[0]), 0, mask.shape[1] - 1))
        y = int(np.clip(round(p[1]), 0, mask.shape[0] - 1))
        if mask[y, x]:
            inside += 1
            center_dist.append(0.0)
        else:
            center_dist.append(float(dt[y, x]))
    return inside / max(1, len(points)), min(center_dist)


def load_owl_detector(device: str):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = "google/owlv2-base-patch16-ensemble"
    local_only = os.environ.get("EGO_LOCAL_FILES_ONLY", "0") == "1"
    processor = Owlv2Processor.from_pretrained(model_id, local_files_only=local_only)
    model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=local_only).to(device).eval()
    return processor, model


def owl_boxes(processor, model, frame: np.ndarray, threshold: float, object_label: str) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    profile = object_profile(object_label)
    texts = [list(profile["prompts"])]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        out,
        target_sizes=torch.tensor([image.size[::-1]], device=model.device),
        threshold=threshold,
        text_labels=texts,
    )[0]
    boxes = []
    labels = result.get("text_labels", [])
    for i, score in enumerate(result["scores"].detach().cpu().tolist()):
        label = labels[i] if isinstance(labels, list) and i < len(labels) else str(int(result["labels"][i]))
        box = result["boxes"][i].detach().cpu().numpy().astype(float)
        boxes.append({"box": box.tolist(), "score": float(score), "label": label, "source": "owlv2"})
    return boxes


def load_sam(checkpoint: Path, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device).eval()
    return SamPredictor(sam)


def sam_mask_from_boxes(
    predictor,
    frame: np.ndarray,
    boxes: list[dict],
    contact_points: np.ndarray,
    geom: dict,
    prev_box: list[float] | None,
    object_label: str,
) -> dict | None:
    if not boxes:
        return None
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 60, 35), (14, 255, 255)),
        cv2.inRange(hsv, (165, 60, 35), (180, 255, 255)),
    ).astype(bool)
    scored = []
    profile = object_profile(object_label)
    use_red = profile.get("color_refinement") == "tomato_red"
    prev_center = None
    if prev_box is not None:
        p = np.asarray(prev_box, dtype=float)
        prev_center = 0.5 * (p[:2] + p[2:])
    for box_info in boxes[:8]:
        box = np.asarray(box_info["box"], dtype=np.float32)
        box_area = max(1.0, float((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)))
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        for mask, sam_score in zip(masks, scores):
            area = int(mask.sum())
            if area < 80:
                continue
            ys, xs = np.where(mask)
            bbox = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
            center = np.asarray([xs.mean(), ys.mean()], dtype=float)
            contact_ratio, min_tip_dist = mask_contact_score(mask, contact_points)
            red_fraction = float(red[mask].mean()) if area else 0.0
            assoc_dist = association_distance(center, geom)
            prev_dist = math.inf if prev_center is None else float(np.linalg.norm(prev_center - center))
            if min(assoc_dist, prev_dist) > 230.0:
                continue
            if area < 0.18 * box_area and contact_ratio <= 0.0 and prev_dist > 90.0:
                continue
            if area < 1200 and min(assoc_dist, prev_dist) > 55.0:
                continue
            if use_red and red_fraction < 0.04 and min(assoc_dist, prev_dist) > 90.0:
                continue
            temporal_bonus = 0.18 * math.exp(-min(prev_dist, 220.0) / 95.0) if prev_center is not None else 0.0
            score = (
                float(box_info["score"])
                + 0.35 * float(sam_score)
                + 0.45 * contact_ratio
                + (0.18 * red_fraction if use_red else 0.0)
                + temporal_bonus
                - 0.0016 * min(assoc_dist, 180.0)
                - 0.001 * min(prev_dist, 160.0)
            )
            scored.append((score, mask, bbox, box_info, float(sam_score), contact_ratio, min_tip_dist, red_fraction, prev_dist, assoc_dist))
    if not scored:
        return None
    score, mask, bbox, box_info, sam_score, contact_ratio, min_tip_dist, red_fraction, prev_dist, assoc_dist = max(scored, key=lambda item: item[0])
    m = mask.astype(np.uint8)
    moments = cv2.moments(m)
    if moments["m00"] <= 0:
        return None
    center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
    edge_touch = bool(bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= frame.shape[1] - 2 or bbox[3] >= frame.shape[0] - 2)
    box_w = float(bbox[2] - bbox[0] + 1.0)
    box_h = float(bbox[3] - bbox[1] + 1.0)
    if edge_touch and (box_w < 28.0 or box_h < 28.0 or min_tip_dist > 30.0):
        return None
    return {
        "mask": mask,
        "bbox_xyxy": bbox.astype(float).tolist(),
        "center_xy": center,
        "area_px": int(mask.sum()),
        "score": float(score),
        "owl_score": float(box_info["score"]),
        "owl_label": box_info["label"],
        "sam_score": sam_score,
        "contact_ratio": float(contact_ratio),
        "min_tip_dist_px": float(min_tip_dist),
        "association_dist_px": float(assoc_dist),
        "prev_center_dist_px": float(prev_dist),
        "red_fraction": red_fraction,
        "proposal_source": box_info.get("source", "unknown"),
        "edge_touch": edge_touch,
    }


def action_for_frame(actions: list[dict], source_idx: int) -> dict | None:
    for action in actions:
        if int(action.get("start_frame", -1)) <= source_idx < int(action.get("end_frame", -1)):
            return action
    return None


def run_object_masks(args: argparse.Namespace, frames: list[dict], actions: list[dict], render: RenderSpec) -> tuple[list[dict], dict]:
    available = {int(frame["frame_idx"]): i for i, frame in enumerate(frames)}
    frame_start = min(available)
    frame_end = max(available)
    intervals = []
    action_labels: list[dict] = []
    for action in actions:
        if args.object_label != "auto" and action_relevance(action, args.object_label) <= 0:
            continue
        try:
            label = object_label_for_action(action, args.object_label)
        except RuntimeError:
            continue
        start = max(int(action["start_frame"]), frame_start)
        end = min(int(action["end_frame"]) - 1, frame_end)
        if args.frame_start is not None:
            start = max(start, int(args.frame_start))
        if args.frame_end is not None:
            end = min(end, int(args.frame_end))
        if start <= end:
            intervals.append((start, end, label, action))
            action_labels.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "label": label,
                    "action": action.get("action"),
                    "description": action.get("description"),
                }
            )
    if not intervals:
        raise RuntimeError("no object intervals after frame limits")
    step = max(1, int(args.object_stride))
    cap, info = open_video(args.clip)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_processor, owl_model = load_owl_detector(device)
    sam = load_sam(args.sam_checkpoint, device)

    object_meas: list[dict | None] = [None] * len(frames)
    prev_box: list[float] | None = None
    prev_mask: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    prev_source_idx: int | None = None
    prev_label: str | None = None
    flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    processed = 0
    detected = 0
    try:
        source_indices = []
        frame_label: dict[int, str] = {}
        for start, end, label, _ in intervals:
            for idx in range(start, end + 1, step):
                if idx in available:
                    source_indices.append(idx)
                    frame_label[idx] = label
        source_indices = sorted(set(source_indices))
        for source_idx in tqdm(source_indices, desc="object_sam"):
            local_idx = available[source_idx]
            object_label = frame_label[source_idx]
            profile = object_profile(object_label)
            use_tomato_color = profile.get("color_refinement") == "tomato_red"
            if prev_label != object_label:
                prev_box = None
                prev_mask = None
                prev_frame = None
                prev_source_idx = None
                prev_label = object_label
            frame = read_video_frame(cap, source_idx)
            geom = hand_association_geometry(frames[local_idx])
            contact = geom["tips"]
            boxes = []
            if use_tomato_color:
                boxes.extend(red_mask_boxes(frame, geom, prev_box))
            boxes.extend(owl_boxes(owl_processor, owl_model, frame, args.owl_threshold, object_label))
            if prev_box is not None:
                boxes.append({"box": prev_box, "score": 0.12, "label": f"{object_label}_temporal_prior", "source": "temporal_prior"})
            boxes.sort(key=lambda item: float(item["score"]), reverse=True)
            mask_info = sam_mask_from_boxes(sam, frame, boxes, contact, geom, prev_box, object_label)
            processed += 1
            if mask_info is not None:
                refined_mask, refined_status = (
                    refine_deformable_tomato_mask(frame, mask_info["mask"], frames[local_idx], geom)
                    if use_tomato_color
                    else (mask_info["mask"], "sam_single_mask")
                )
                if prev_source_idx is not None and source_idx - prev_source_idx == step:
                    temporal_mask, temporal_status = fuse_temporal_object_mask(
                        flow_estimator,
                        prev_frame,
                        frame,
                        prev_mask,
                        refined_mask,
                        frames[local_idx],
                        geom,
                    )
                    if temporal_mask is not refined_mask:
                        refined_mask = temporal_mask
                        refined_status = temporal_status
                if refined_mask is not mask_info["mask"]:
                    m = refined_mask.astype(np.uint8)
                    moments = cv2.moments(m)
                    if moments["m00"] > 0:
                        ys, xs = np.where(refined_mask)
                        mask_info["mask"] = refined_mask
                        mask_info["bbox_xyxy"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                        mask_info["center_xy"] = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
                        mask_info["area_px"] = int(refined_mask.sum())
                        mask_info["mask_refinement"] = refined_status
                prev_box = mask_info["bbox_xyxy"]
                prev_mask = mask_info["mask"].copy()
                prev_frame = frame.copy()
                prev_source_idx = source_idx
                object_meas[local_idx] = {k: v for k, v in mask_info.items() if k != "mask"}
                object_meas[local_idx]["label"] = object_label
                object_meas[local_idx]["profile"] = "tomato_color_refined" if use_tomato_color else "general_prompt_contact_temporal"
                object_meas[local_idx]["prompts"] = list(profile["prompts"])
                mask_small = cv2.resize(mask_info["mask"].astype(np.uint8) * 255, (render.width, render.height), interpolation=cv2.INTER_NEAREST)
                mask_path = args.output_dir / "object_masks" / f"{source_idx:06d}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), mask_small)
                object_meas[local_idx]["mask_path"] = str(mask_path)
                object_meas[local_idx]["mask_image_size"] = [int(render.width), int(render.height)]
                object_meas[local_idx]["source_image_size"] = [int(frame.shape[1]), int(frame.shape[0])]
                detected += 1
    finally:
        cap.release()
        del owl_model, sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return object_meas, {
        "label": "per_action_auto" if args.object_label == "auto" else args.object_label,
        "requested_label": args.object_label,
        "action_labels": action_labels,
        "semantic_interval": [min(start for start, _, _, _ in intervals), max(end for _, end, _, _ in intervals)],
        "stride": step,
        "processed_frames": processed,
        "detected_frames": detected,
        "detection_rate_on_processed": detected / max(1, processed),
        "backend": "OWLv2 prompt proposals + SAM ViT-B masks + hand-contact and temporal proposal scoring",
        "profile": "per_action_prompt_contact_temporal",
    }


def fill_object_track(
    frames: list[dict],
    object_meas: list[dict | None],
    fps: float,
    image_size: tuple[int, int],
    object_segments: list[dict],
) -> dict:
    for frame in frames:
        frame["object"] = {"label": None, "status": "outside_semantic_interval"}
    invalid_measurements = 0
    meas_sigma = np.asarray([16.0, 16.0, 28.0, 28.0, 28.0, 28.0, 18.0], dtype=float)
    proc_pos = np.asarray([3.5, 3.5, 6.0, 6.0, 6.0, 6.0, 3.5], dtype=float)
    proc_vel = np.asarray([80.0, 80.0, 120.0, 120.0, 120.0, 120.0, 80.0], dtype=float)
    max_edge_prediction = max(3, int(round(0.50 * fps)))
    observed = 0
    predicted = 0
    outside_visibility = 0
    edge_predicted = 0
    contact_frames = 0
    width, height = image_size
    measured_source_frames: list[int] = []
    segment_qc = []
    for segment in object_segments:
        label = str(segment["label"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        indices = [i for i, frame in enumerate(frames) if start <= int(frame["frame_idx"]) <= end]
        if not indices:
            continue
        pose_type = str(object_profile(label)["pose_type"])
        large_deformable = "bag" in pose_type or "deformable" in pose_type
        area_samples = [
            float(object_meas[i]["area_px"])
            for i in indices
            if object_meas[i] is not None and str(object_meas[i].get("label")) == label
        ]
        segment_area_median = float(np.median(area_samples)) if area_samples else 0.0
        meas = []
        conf = []
        valid_measurements: list[dict | None] = []
        prev_valid_center: np.ndarray | None = None
        for i in indices:
            m = object_meas[i]
            if m is None or str(m.get("label")) != label:
                meas.append(None)
                conf.append(0.0)
                valid_measurements.append(None)
                continue
            bbox = np.asarray(m["bbox_xyxy"], dtype=float)
            center = np.asarray(m["center_xy"], dtype=float)
            edge = bool(m.get("edge_touch", False))
            box_width = bbox[2] - bbox[0] + 1.0
            box_height = bbox[3] - bbox[1] + 1.0
            prev_jump = math.inf if prev_valid_center is None else float(np.linalg.norm(center - prev_valid_center))
            tomato_guard = m.get("profile") == "tomato_color_refined"
            area_px = float(m["area_px"])
            valid = not (edge and (box_width < 36.0 or box_height < 36.0 or (tomato_guard and float(m.get("red_fraction", 0.0)) < 0.20)))
            if prev_valid_center is not None and prev_jump > 520.0 and float(m.get("min_tip_dist_px", math.inf)) > 60.0:
                valid = False
            if large_deformable and segment_area_median > 0.0:
                if area_px < 0.06 * segment_area_median:
                    valid = False
                if area_px < 0.14 * segment_area_median and float(m.get("contact_ratio", 0.0)) <= 0.0 and float(m.get("min_tip_dist_px", math.inf)) > 80.0:
                    valid = False
            if valid:
                meas.append(np.asarray(m["center_xy"] + m["bbox_xyxy"] + [math.sqrt(max(1.0, float(m["area_px"])))], dtype=float))
                conf.append(float(max(0.05, m["score"])))
                valid_measurements.append(m)
                prev_valid_center = center
            else:
                meas.append(None)
                conf.append(0.0)
                valid_measurements.append(None)
                invalid_measurements += 1
        measured_positions = [pos for pos, m in enumerate(meas) if m is not None]
        if not measured_positions:
            for i in indices:
                frames[i]["object"] = {"label": label, "status": "unobserved_no_valid_object_measurement"}
                outside_visibility += 1
            segment_qc.append({"label": label, "start_frame": start, "end_frame": end, "measured_frames": 0, "predicted_frames": 0})
            continue
        smoothed, _ = kalman_rts(meas, conf, fps, meas_sigma, proc_pos, proc_vel)
        first_measured = measured_positions[0]
        last_measured = measured_positions[-1]
        segment_observed = 0
        segment_predicted = 0
        for pos, i in enumerate(indices):
            frame = frames[i]
            before = pos < first_measured
            after = pos > last_measured
            edge_gap = first_measured - pos if before else pos - last_measured if after else 0
            if (before or after) and edge_gap > max_edge_prediction:
                outside_visibility += 1
                frame["object"] = {"label": label, "status": "unobserved_before_or_after_object_track"}
                continue
            vec = smoothed[pos]
            center = vec[:2]
            bbox = vec[2:6]
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
            center[0] = float(np.clip(center[0], 0, width - 1))
            center[1] = float(np.clip(center[1], 0, height - 1))
            box_w = float(bbox[2] - bbox[0])
            box_h = float(bbox[3] - bbox[1])
            m = valid_measurements[pos]
            predicted_contact_ratio = 0.0
            predicted_min_tip = math.inf
            if m is None:
                contact = hand_contact_points(frame)
                if contact.size:
                    d = np.linalg.norm(contact - center[None, :], axis=1)
                    predicted_min_tip = float(d.min())
                    predicted_contact_ratio = float(predicted_min_tip < 45.0)
            area_px = float(max(1.0, vec[6] * vec[6]))
            collapsed_large_prediction = (
                large_deformable
                and m is None
                and segment_area_median > 0.0
                and area_px < 0.14 * segment_area_median
                and predicted_contact_ratio <= 0.0
                and predicted_min_tip > 80.0
            )
            if (box_w < 12.0 or box_h < 12.0) or (center[0] <= 1.0 and box_w < 60.0) or collapsed_large_prediction:
                frame["object"] = {"label": label, "status": "unobserved_degenerate_track_state"}
                if object_meas[i] is not None:
                    invalid_measurements += 1
                continue
            status = "measured_sam_kalman" if m is not None else "predicted_kalman"
            if m is not None:
                observed += 1
                segment_observed += 1
                measured_source_frames.append(int(frame["frame_idx"]))
                contact_ratio = float(m.get("contact_ratio", 0.0))
                min_tip = float(m.get("min_tip_dist_px", math.inf))
                red_fraction = float(m.get("red_fraction", 0.0))
            else:
                predicted += 1
                segment_predicted += 1
                if before or after:
                    edge_predicted += 1
                min_tip = predicted_min_tip
                contact_ratio = predicted_contact_ratio
                red_fraction = 0.0
            if contact_ratio > 0 or min_tip < 45.0:
                contact_frames += 1
            frame["object"] = {
                "label": label,
                "status": status,
                "bbox_xyxy": bbox.astype(float).tolist(),
                "center_xy": center.astype(float).tolist(),
                "area_px": area_px,
                "measurement_available": m is not None,
                "mask_path": m.get("mask_path") if m else None,
                "contact_ratio": contact_ratio,
                "min_tip_dist_px": min_tip,
                "red_fraction": red_fraction,
                "mask_refinement": m.get("mask_refinement") if m else None,
                "proposal_source": m.get("proposal_source") if m else None,
                "sam_score": float(m["sam_score"]) if m and "sam_score" in m else None,
                "pose_status": "pending_world_ray_depth_optimization",
            }
        segment_qc.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "measured_frames": segment_observed,
                "predicted_frames": segment_predicted,
                "measurement_area_median_px": segment_area_median,
            }
        )
    if not measured_source_frames:
        raise RuntimeError("object module produced no valid SAM masks in any active segment")
    return {
        "measured_frames": observed,
        "predicted_frames": predicted,
        "outside_visibility_frames": outside_visibility,
        "edge_predicted_frames": edge_predicted,
        "contact_frames": contact_frames,
        "first_measured_source_frame": int(min(measured_source_frames)),
        "last_measured_source_frame": int(max(measured_source_frames)),
        "invalid_measurements_rejected": invalid_measurements,
        "segments": segment_qc,
    }


def source_camera_ray(center_xy: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    x, y = np.asarray(center_xy, dtype=float)
    return np.asarray([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=float)


def contact_anchor_world(frame: dict, center_xy: np.ndarray, radius_px: float) -> tuple[np.ndarray | None, float]:
    candidates = []
    for hand in frame["hands"]:
        all_points = radius_px >= 120.0
        points2d = np.asarray(hand["joints2d"], dtype=float) if all_points else np.asarray(hand["joints2d"], dtype=float)[TIP_IDS]
        points3d = np.asarray(hand["joints3d_world_m"], dtype=float) if all_points else np.asarray(hand["joints3d_world_m"], dtype=float)[TIP_IDS]
        dists = np.linalg.norm(points2d - center_xy[None, :], axis=1)
        for dist, point in zip(dists, points3d):
            if dist <= radius_px:
                candidates.append((float(dist), point))
    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda item: item[0])
    selected = np.asarray([point for _, point in candidates[:4]], dtype=float)
    return selected.mean(axis=0), candidates[0][0]


def object_depth_bounds(label: str) -> tuple[float, float]:
    pose_type = str(object_profile(label)["pose_type"])
    if "bag" in pose_type or "deformable" in pose_type or "tool" in pose_type:
        return 0.15, 2.20
    return 0.20, 3.20


def add_sparse_row(rows: list[int], cols: list[int], vals: list[float], rhs: list[float], row: int, terms: list[tuple[int, float]], target: float, sigma: float) -> int:
    weight = 1.0 / sigma
    for col, val in terms:
        rows.append(row)
        cols.append(col)
        vals.append(float(val) * weight)
    rhs.append(float(target) * weight)
    return row + 1


def attach_object_world(
    frames: list[dict],
    T_metric: np.ndarray,
    intrinsics: np.ndarray,
    recon: dict,
    droid_to_meters: float,
    image_size: tuple[int, int],
    max_keyframe_gap: int,
) -> dict:
    active = []
    droid_depth_samples = {}
    contact_samples = {}
    for i, frame in enumerate(frames):
        obj = frame.get("object", {})
        if obj.get("center_xy") is None or obj.get("status") == "outside_semantic_interval":
            continue
        active.append(i)
        center = np.asarray(obj["center_xy"], dtype=float)
        source_idx = int(frame["frame_idx"])
        sampled = sample_droid_depth_relative(recon, source_idx, center, image_size, max_keyframe_gap)
        if sampled is not None:
            droid_depth_samples[i] = (sampled[0] * droid_to_meters, sampled[1])
        contact_radius = max(85.0, min(260.0, 0.35 * math.sqrt(max(1.0, float(obj.get("area_px", 1.0))))))
        contact_radius = max(contact_radius, min(220.0, float(obj.get("min_tip_dist_px", 150.0)) + 55.0))
        anchor, min_dist = contact_anchor_world(frame, center, contact_radius)
        if anchor is not None:
            contact_samples[i] = (anchor, min_dist)
    if not active:
        raise RuntimeError("no active object frames for world pose optimization")

    index = {frame_idx: j for j, frame_idx in enumerate(active)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0
    absolute_rows = 0
    initial_depths: dict[int, float] = {}
    for frame_idx in active:
        col = index[frame_idx]
        if frame_idx in droid_depth_samples:
            depth_m, _ = droid_depth_samples[frame_idx]
            row = add_sparse_row(rows, cols, vals, rhs, row, [(col, 1.0)], depth_m, sigma=0.18)
            absolute_rows += 1
            initial_depths[frame_idx] = float(depth_m)
        if frame_idx in contact_samples:
            anchor, min_dist = contact_samples[frame_idx]
            center = np.asarray(frames[frame_idx]["object"]["center_xy"], dtype=float)
            ray = source_camera_ray(center, intrinsics)
            origin = T_metric[frame_idx, :3, 3]
            direction = T_metric[frame_idx, :3, :3] @ ray
            label = str(frames[frame_idx]["object"].get("label", ""))
            pose_type = str(object_profile(label)["pose_type"])
            sigma = 0.030 + 0.00022 * min(min_dist, 220.0)
            if "bag" in pose_type or "deformable" in pose_type:
                sigma *= 0.65
            depth_from_anchor = float(np.dot(anchor - origin, direction) / max(1e-9, np.dot(direction, direction)))
            lo, hi = object_depth_bounds(label)
            if lo <= depth_from_anchor <= hi:
                initial_depths.setdefault(frame_idx, depth_from_anchor)
            for axis in range(3):
                row = add_sparse_row(
                    rows,
                    cols,
                    vals,
                    rhs,
                    row,
                    [(col, float(direction[axis]))],
                    float(anchor[axis] - origin[axis]),
                    sigma=sigma,
                )
                absolute_rows += 1
    for a, b, c in zip(active[:-2], active[1:-1], active[2:]):
        if b - a != 1 or c - b != 1:
            continue
        if (
            frames[a]["object"].get("label") != frames[b]["object"].get("label")
            or frames[b]["object"].get("label") != frames[c]["object"].get("label")
        ):
            continue
        for axis in range(3):
            terms = []
            for frame_idx, coeff in [(a, 1.0), (b, -2.0), (c, 1.0)]:
                center = np.asarray(frames[frame_idx]["object"]["center_xy"], dtype=float)
                ray = source_camera_ray(center, intrinsics)
                direction = T_metric[frame_idx, :3, :3] @ ray
                terms.append((index[frame_idx], coeff * float(direction[axis])))
            origin_term = (
                T_metric[c, axis, 3]
                - 2.0 * T_metric[b, axis, 3]
                + T_metric[a, axis, 3]
            )
            row = add_sparse_row(rows, cols, vals, rhs, row, terms, -float(origin_term), sigma=0.020)
    if row == 0:
        raise RuntimeError("object world optimizer has no depth/contact/smoothness rows")
    if absolute_rows == 0:
        raise RuntimeError("object world optimizer has no absolute DROID-depth or contact-anchor rows")
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(row, len(active))).tocsr()
    b = np.asarray(rhs, dtype=float)
    if initial_depths:
        median_initial = float(np.median(list(initial_depths.values())))
    else:
        median_initial = 1.40
    bounds = [object_depth_bounds(str(frames[frame_idx]["object"].get("label", ""))) for frame_idx in active]
    lo_arr = np.asarray([lo for lo, _ in bounds], dtype=float)
    hi_arr = np.asarray([hi for _, hi in bounds], dtype=float)
    x0 = np.asarray([initial_depths.get(frame_idx, median_initial) for frame_idx in active], dtype=float)
    x0 = np.clip(x0, lo_arr, hi_arr)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = A @ x - b
        value = 0.5 * float(np.dot(residual, residual))
        grad = A.T @ residual
        return value, np.asarray(grad, dtype=float)

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0,
        jac=lambda x: objective(x)[1],
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"object world depth optimization failed: {result.message}")
    depths = np.asarray(result.x, dtype=float)
    fx, fy, _, _ = intrinsics
    focal = 0.5 * (float(fx) + float(fy))
    for frame_idx, depth_m in zip(active, depths):
        obj = frames[frame_idx]["object"]
        center = np.asarray(obj["center_xy"], dtype=float)
        point_cam = source_camera_ray(center, intrinsics) * depth_m
        point_world = (T_metric[frame_idx] @ np.r_[point_cam, 1.0])[:3]
        radius_px = math.sqrt(max(1.0, float(obj["area_px"])) / math.pi)
        obj["center_source_camera_m"] = point_cam.astype(float).tolist()
        obj["center_world_m"] = point_world.astype(float).tolist()
        obj["depth_m"] = float(depth_m)
        obj["radius_m"] = float(radius_px * depth_m / focal)
        obj["pose_type"] = object_profile(str(obj["label"]))["pose_type"]
        obj["pose_status"] = "world_ray_depth_optimized_from_droid_depth_contact_and_temporal_smoothness"
        obj["depth_evidence"] = {
            "droid_depth": frame_idx in droid_depth_samples,
            "contact_anchor": frame_idx in contact_samples,
            "temporal_smoothness": True,
        }
        if frame_idx in droid_depth_samples:
            obj["droid_depth_keyframe"] = int(droid_depth_samples[frame_idx][1])
        if frame_idx in contact_samples:
            obj["contact_anchor_min_tip_dist_px"] = float(contact_samples[frame_idx][1])
    return {
        "active_frames": len(active),
        "variables": len(active),
        "linear_rows": row,
        "absolute_rows": absolute_rows,
        "droid_depth_frames": len(droid_depth_samples),
        "contact_anchor_frames": len(contact_samples),
        "cost": float(result.fun),
        "optimizer_iterations": int(result.nit),
        "projected_gradient_inf_norm": float(np.max(np.abs(result.jac))) if result.jac is not None else None,
        "depth_m_iqr": [float(np.percentile(depths, 25)), float(np.percentile(depths, 75))],
        "depth_m_minmax": [float(np.min(depths)), float(np.max(depths))],
    }


def apply_hand_object_contact_correction(frames: list[dict], fps: float) -> dict:
    measurements: dict[str, list[np.ndarray | None]] = {"left": [None] * len(frames), "right": [None] * len(frames)}
    confidences: dict[str, list[float]] = {"left": [0.0] * len(frames), "right": [0.0] * len(frames)}
    candidates = 0
    accepted = 0
    for i, frame in enumerate(frames):
        obj = frame.get("object", {})
        if obj.get("center_world_m") is None or obj.get("radius_m") is None:
            continue
        center = np.asarray(obj["center_world_m"], dtype=float)
        radius = float(obj["radius_m"])
        pose_type = str(obj.get("pose_type", ""))
        contact_limit = 0.12 if ("bag" in pose_type or "deformable" in pose_type) else 0.07
        active_sides = {hand["side"] for hand in active_object_hands(frame)}
        for hand in frame.get("hands", []):
            side = str(hand["side"])
            if side not in active_sides:
                continue
            joints = np.asarray(hand["joints3d_world_m"], dtype=float)
            if not joints.size:
                continue
            d = np.linalg.norm(joints - center[None, :], axis=1) - radius
            tip_ids = np.asarray(TIP_IDS, dtype=int)
            tip_gap = float(np.min(np.abs(d[tip_ids])))
            all_gap = float(np.min(np.abs(d)))
            min_gap = min(tip_gap, all_gap)
            candidates += 1
            if min_gap > contact_limit:
                continue
            closest = tip_ids[int(np.argmin(np.abs(d[tip_ids])))] if tip_gap <= all_gap else int(np.argmin(np.abs(d)))
            joint = joints[closest]
            direction = joint - center
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            target = center + direction * (radius / norm)
            delta_world = target - joint
            T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            joint_source = np.asarray(hand["joints3d_source_camera_m"], dtype=float)[closest]
            depth = float(joint_source[2])
            if depth <= 0.15:
                continue
            ray = T[:3, :3] @ (joint_source / depth)
            denom = float(np.dot(ray, ray))
            if denom < 1e-9:
                continue
            depth_delta = float(np.dot(delta_world, ray) / denom)
            if abs(depth_delta) > 0.28:
                continue
            measurements[side][i] = np.asarray([depth_delta], dtype=float)
            confidences[side][i] = max(0.05, 1.0 - min_gap / contact_limit)
            accepted += 1
    applied = 0
    offsets_by_side: dict[str, int] = {}
    for side in ("left", "right"):
        if not any(m is not None for m in measurements[side]):
            offsets_by_side[side] = 0
            continue
        smoothed, _ = kalman_rts(
            measurements[side],
            confidences[side],
            fps,
            measurement_sigma=np.asarray([0.025], dtype=float),
            process_position_sigma=np.asarray([0.010], dtype=float),
            process_velocity_sigma=np.asarray([0.045], dtype=float),
        )
        side_applied = 0
        for i, offset_vec in enumerate(smoothed):
            if measurements[side][i] is None:
                continue
            depth_delta = float(np.clip(offset_vec[0], -0.35, 0.35))
            if abs(depth_delta) < 1e-4:
                continue
            frame = frames[i]
            T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            for hand in frame.get("hands", []):
                if hand["side"] != side:
                    continue
                for suffix in ("joints3d", "vertices"):
                    source_key = f"{suffix}_source_camera_m"
                    if source_key not in hand:
                        source_key = f"{suffix}_source_camera_m_sample"
                    world_key = f"{suffix}_world_m"
                    if world_key not in hand:
                        world_key = f"{suffix}_world_m_sample"
                    if source_key not in hand or world_key not in hand:
                        continue
                    source = np.asarray(hand[source_key], dtype=float)
                    z = np.maximum(source[:, 2:3], 1e-6)
                    corrected = source * ((z + depth_delta) / z)
                    world = (T @ np.c_[corrected, np.ones(len(corrected))].T).T[:, :3]
                    hand[source_key] = corrected.astype(float).tolist()
                    hand[world_key] = world.astype(float).tolist()
                hand["hand_object_contact_correction"] = {
                    "depth_delta_m": depth_delta,
                    "status": "semantic_contact_depth_residual_smoothed",
                }
                applied += 1
                side_applied += 1
                break
        offsets_by_side[side] = side_applied
    return {
        "candidate_hand_frames": candidates,
        "accepted_measurements": accepted,
        "applied_hand_frames": applied,
        "applied_by_side": offsets_by_side,
        "max_abs_depth_delta_m": 0.35,
        "contact_limit_m": {"rigid": 0.07, "deformable": 0.12},
    }


def draw_hand_overlay(frame: np.ndarray, frame_ann: dict, sx: float, sy: float, mano_edges: dict[int, np.ndarray]) -> None:
    for hand in frame_ann["hands"]:
        color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
        pts = np.asarray(hand["joints2d"], dtype=float) * np.asarray([sx, sy])
        verts_camera = hand_vertices(hand, "_source_camera_m")
        verts2d = project_points(verts_camera, np.asarray(hand["source_intrinsics"], dtype=float)) * np.asarray([sx, sy])
        box = np.asarray(hand["bbox_xyxy"], dtype=float) * np.asarray([sx, sy, sx, sy])
        x1, y1, x2, y2 = box.astype(int)
        measured = bool(hand.get("measurement_available", False))
        thickness = 2 if measured else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        edges = mano_edges.get(len(verts_camera), np.empty((0, 2), dtype=int))
        for a, b in edges:
            pa = tuple(np.clip(verts2d[a], [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            pb = tuple(np.clip(verts2d[b], [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            cv2.line(frame, pa, pb, color, max(1, thickness), cv2.LINE_AA)
        for p in verts2d[:: max(1, len(verts2d) // 120)]:
            q = tuple(np.clip(p, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
            cv2.circle(frame, q, 1, color, -1, cv2.LINE_AA)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thickness + 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p.astype(int)), 4, color, 1, cv2.LINE_AA)


def source_intrinsics_from_frame(frame_ann: dict) -> np.ndarray | None:
    obj = frame_ann.get("object", {})
    if obj.get("source_intrinsics") is not None:
        intrinsics = np.asarray(obj["source_intrinsics"], dtype=float)
        if intrinsics.shape == (4,) and np.isfinite(intrinsics).all():
            return intrinsics
    for hand in frame_ann.get("hands", []):
        if hand.get("source_intrinsics") is not None:
            intrinsics = np.asarray(hand["source_intrinsics"], dtype=float)
            if intrinsics.shape == (4,) and np.isfinite(intrinsics).all():
                return intrinsics
    return None


def world_to_camera_points(points_world_m: np.ndarray, frame_ann: dict) -> np.ndarray:
    T_wc = camera_transform(frame_ann)
    T_cw = np.linalg.inv(T_wc)
    hom = np.c_[np.asarray(points_world_m, dtype=float), np.ones(len(points_world_m), dtype=float)]
    return (hom @ T_cw.T)[:, :3]


def draw_projected_object_mesh_overlay(
    frame: np.ndarray,
    frame_ann: dict,
    object_mesh: ObjectMeshFrame | None,
    sx: float,
    sy: float,
) -> None:
    if object_mesh is None or len(object_mesh.vertices) == 0 or len(object_mesh.faces) == 0:
        return
    intrinsics = source_intrinsics_from_frame(frame_ann)
    if intrinsics is None:
        return
    camera_vertices = world_to_camera_points(object_mesh.vertices, frame_ann)
    visible = np.isfinite(camera_vertices).all(axis=1) & (camera_vertices[:, 2] > 1e-4)
    if int(visible.sum()) < 3:
        return
    projected = project_points(camera_vertices, intrinsics) * np.asarray([sx, sy], dtype=float)
    faces = np.asarray(object_mesh.faces, dtype=np.int32)
    face_visible = visible[faces].all(axis=1)
    visible_faces = faces[face_visible]
    if len(visible_faces) == 0:
        return
    edge_budget = min(len(visible_faces), 1200)
    face_ids = np.linspace(0, len(visible_faces) - 1, edge_budget, dtype=int)
    for face in visible_faces[face_ids]:
        poly = projected[face].astype(np.int32)
        if np.any(poly[:, 0] < -200) or np.any(poly[:, 0] >= frame.shape[1] + 200):
            continue
        if np.any(poly[:, 1] < -200) or np.any(poly[:, 1] >= frame.shape[0] + 200):
            continue
        cv2.polylines(frame, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, OBJECT_COLOR, 1, cv2.LINE_AA)
    mesh_center = project_points(camera_vertices[visible].mean(axis=0, keepdims=True), intrinsics)[0] * np.asarray([sx, sy], dtype=float)
    center_xy = tuple(np.clip(mesh_center + np.asarray([8.0, 18.0]), [0, 18], [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
    cv2.putText(frame, "OBJECT MESH", center_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "OBJECT MESH", center_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.42, OBJECT_COLOR, 1, cv2.LINE_AA)


def draw_object_overlay(frame: np.ndarray, frame_ann: dict, sx: float, sy: float, object_mesh: ObjectMeshFrame | None = None) -> None:
    obj = frame_ann.get("object", {})
    if obj.get("bbox_xyxy") is None or obj.get("status") == "not_visible":
        return
    if obj.get("mask_path"):
        mask = cv2.imread(obj["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            color_layer = np.zeros_like(frame)
            color_layer[:, :] = OBJECT_COLOR
            frame[:] = np.where(mask[..., None] > 0, (0.55 * frame + 0.45 * color_layer).astype(np.uint8), frame)
    box = np.asarray(obj["bbox_xyxy"], dtype=float) * np.asarray([sx, sy, sx, sy])
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_COLOR, 2)
    center = (np.asarray(obj["center_xy"], dtype=float) * np.asarray([sx, sy])).astype(int)
    cv2.drawMarker(frame, tuple(center), OBJECT_COLOR, cv2.MARKER_CROSS, 14, 2)


def put_caption(frame: np.ndarray, caption: str, frame_idx: int) -> None:
    words = f"{frame_idx:04d}  {caption}".split()
    lines: list[str] = []
    cur = ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 2
    max_width = frame.shape[1] - 24
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    lines = lines[:2]
    band_h = 20 + 24 * len(lines)
    cv2.rectangle(frame, (0, frame.shape[0] - band_h), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    y = frame.shape[0] - band_h + 26
    for line in lines:
        cv2.putText(frame, line, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 24


def unit_vector(vec: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-9:
        raise RuntimeError(f"cannot normalize {name}")
    return arr / norm


def camera_transform(frame: dict) -> np.ndarray:
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError("camera T_world_camera_metric must be finite 4x4")
    return T


def camera_axes(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = unit_vector(T[:3, 0], "camera right")
    up = unit_vector(-T[:3, 1], "camera up")
    forward = unit_vector(T[:3, 2], "camera forward")
    return right, up, forward


def world_display_basis(frames: list[dict], camera_positions: np.ndarray) -> np.ndarray:
    step = max(1, len(frames) // 160)
    ups = []
    forwards = []
    for frame in frames[::step]:
        _, up, forward = camera_axes(camera_transform(frame))
        ups.append(up)
        forwards.append(forward)
    up = unit_vector(np.median(np.asarray(ups, dtype=float), axis=0), "median camera up")
    forward_raw = np.median(np.asarray(forwards, dtype=float), axis=0)
    forward = forward_raw - up * float(np.dot(forward_raw, up))
    if np.linalg.norm(forward) < 1e-6:
        travel = np.asarray(camera_positions[-1] - camera_positions[0], dtype=float)
        forward = travel - up * float(np.dot(travel, up))
    forward = unit_vector(forward, "display forward")
    right = unit_vector(np.cross(forward, up), "display right")
    forward = unit_vector(np.cross(up, right), "display forward orthogonalized")
    basis = np.stack([right, forward, up], axis=0)
    hand_offsets = []
    sample_step = max(1, len(frames) // 240)
    for frame in frames[::sample_step]:
        head = np.asarray(frame["camera"]["position_world_m"], dtype=float)
        for hand in frame.get("hands", []):
            joints = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
            if joints.size:
                hand_offsets.append(float(np.median((joints - head[None, :]) @ basis[2])))
    if hand_offsets and float(np.median(hand_offsets)) > 0.0:
        basis[1:] *= -1.0
    return basis


def camera_display_basis(T: np.ndarray, frame: dict) -> np.ndarray:
    right, up, forward = camera_axes(T)
    basis = np.stack([right, forward, up], axis=0)
    offsets = []
    head = np.asarray(frame["camera"]["position_world_m"], dtype=float)
    for hand in frame.get("hands", []):
        joints = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
        if joints.size:
            offsets.append(float(np.median((joints - head[None, :]) @ basis[2])))
    if offsets and float(np.median(offsets)) > 0.0:
        basis[1:] *= -1.0
    return basis


def build_world_projector(points: np.ndarray, basis: np.ndarray, size: tuple[int, int]) -> WorldProjector:
    width, height = size
    q = np.asarray(points, dtype=float) @ basis.T
    xy = np.stack([q[:, 0] + 0.34 * q[:, 1], q[:, 2] - 0.18 * q[:, 1]], axis=1)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, np.asarray([0.20, 0.20], dtype=float))
    left, right, top, bottom = 44.0, 38.0, 78.0, 34.0
    available = np.asarray([width - left - right, height - top - bottom], dtype=float)
    if np.any(available <= 0):
        raise RuntimeError("render size is too small for world projector")
    pixels_per_meter = 0.92 * float(np.min(available / span))
    xy_center = 0.5 * (lo + hi)
    qy = float(np.median(q[:, 1]))
    q_center = np.asarray([xy_center[0] - 0.34 * qy, qy, xy_center[1] + 0.18 * qy], dtype=float)
    return WorldProjector(
        basis=basis,
        q_center=q_center,
        pixels_per_meter=pixels_per_meter,
        screen_center=(float(left + 0.5 * available[0]), float(top + 0.5 * available[1])),
        size=size,
    )


def project_world(points: np.ndarray, projector: WorldProjector) -> np.ndarray:
    q = np.asarray(points, dtype=float) @ projector.basis.T
    q_center = projector.q_center
    x_metric = q[:, 0] + 0.34 * q[:, 1] - (q_center[0] + 0.34 * q_center[1])
    y_metric = q[:, 2] - 0.18 * q[:, 1] - (q_center[2] - 0.18 * q_center[1])
    x = projector.screen_center[0] + projector.pixels_per_meter * x_metric
    y = projector.screen_center[1] - projector.pixels_per_meter * y_metric
    xy = np.stack([x, y], axis=1)
    return np.clip(np.rint(xy), -100000, 100000).astype(int)


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    projector: WorldProjector,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool = False,
) -> None:
    if len(points) < 2:
        return
    xy = project_world(points, projector)
    count = len(xy)
    limit = count if closed else count - 1
    for i in range(limit):
        a = tuple(xy[i])
        b = tuple(xy[(i + 1) % count])
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)


def draw_scale_bar(image: np.ndarray, projector: WorldProjector, meters: float = 0.25) -> None:
    width, height = image.shape[1], image.shape[0]
    length_px = int(round(projector.pixels_per_meter * meters))
    if length_px < 24:
        return
    length_px = min(length_px, 180)
    x0 = width - length_px - 36
    y0 = height - 30
    cv2.line(image, (x0, y0), (x0 + length_px, y0), (35, 35, 35), 3, cv2.LINE_AA)
    cv2.line(image, (x0, y0 - 7), (x0, y0 + 7), (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(image, (x0 + length_px, y0 - 7), (x0 + length_px, y0 + 7), (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(image, f"{int(round(meters * 100))} cm", (x0, y0 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 2, cv2.LINE_AA)


def load_object_mesh_archive(path: Path | None) -> dict[int, ObjectMeshFrame]:
    if path is None:
        return {}
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"object mesh archive missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError("object mesh archive offsets do not match frame_idx length")
    out: dict[int, ObjectMeshFrame] = {}
    for i, source_idx in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        frame_vertices = vertices[v0:v1]
        frame_faces = faces[f0:f1]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"object mesh archive contains empty mesh for frame {source_idx}")
        if frame_faces.min() < 0 or frame_faces.max() >= len(frame_vertices):
            raise RuntimeError(f"object mesh archive face index out of range for frame {source_idx}")
        out[int(source_idx)] = ObjectMeshFrame(vertices=frame_vertices, faces=frame_faces)
    return out


def draw_object_mesh(image: np.ndarray, mesh: ObjectMeshFrame, projector: WorldProjector) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        return
    q = vertices @ projector.basis.T
    face_depth = q[faces].mean(axis=1)[:, 1]
    order = np.argsort(face_depth)
    xy = project_world(vertices, projector)
    overlay = image.copy()
    for face_id in order:
        poly = xy[faces[int(face_id)]]
        if np.any(poly[:, 0] < -1000) or np.any(poly[:, 0] > image.shape[1] + 1000):
            continue
        if np.any(poly[:, 1] < -1000) or np.any(poly[:, 1] > image.shape[0] + 1000):
            continue
        cv2.fillConvexPoly(overlay, poly.astype(np.int32), (70, 92, 220), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0, image)
    edge_budget = min(len(faces), 900)
    edge_ids = np.linspace(0, len(faces) - 1, edge_budget, dtype=int)
    for face_id in edge_ids:
        poly = xy[faces[int(face_id)]]
        cv2.polylines(image, [poly.astype(np.int32)], True, OBJECT_COLOR, 1, cv2.LINE_AA)
    center = project_world(vertices.mean(axis=0, keepdims=True), projector)[0]
    cv2.drawMarker(image, tuple(center), OBJECT_COLOR, cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
    cv2.putText(image, "OBJECT MESH", tuple((center + np.asarray([9, 14])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, OBJECT_COLOR, 2, cv2.LINE_AA)


def hand_mesh_gap_m(frame: dict, object_mesh: ObjectMeshFrame | None) -> tuple[float | None, str | None]:
    if object_mesh is None or len(object_mesh.vertices) == 0:
        return None, None
    vertices = np.asarray(object_mesh.vertices, dtype=float)
    best = math.inf
    best_side: str | None = None
    for hand in frame.get("hands", []):
        hv = hand_vertices(hand, "_world_m")
        if len(hv) == 0:
            continue
        d = np.linalg.norm(hv[:, None, :] - vertices[None, :, :], axis=2)
        gap = float(d.min())
        if gap < best:
            best = gap
            best_side = str(hand.get("side", "hand"))
    if not np.isfinite(best):
        return None, None
    return best, best_side


def draw_hand_world(
    image: np.ndarray,
    hand: dict,
    projector: WorldProjector,
    mano_edges: dict[int, np.ndarray],
    *,
    mesh_stride: int,
    joint_radius: int,
    label: bool,
) -> None:
    joints = np.asarray(hand["joints3d_world_m"], dtype=float)
    verts = hand_vertices(hand, "_world_m")
    color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
    mesh_color = tuple(int(0.50 * c + 0.50 * 244) for c in color)
    edges = mano_edges.get(len(verts), np.empty((0, 2), dtype=int))
    if len(edges):
        for a, b in edges[:: max(1, mesh_stride)]:
            draw_polyline(image, verts[[int(a), int(b)]], projector, mesh_color, 1)
    joint_xy = project_world(joints, projector)
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(joint_xy[a]), tuple(joint_xy[b]), color, 3, cv2.LINE_AA)
    for point in joint_xy:
        cv2.circle(image, tuple(point), joint_radius, color, -1, cv2.LINE_AA)
    if label:
        text = "L MANO" if hand["side"] == "left" else "R MANO"
        cv2.putText(image, text, tuple((joint_xy[0] + np.asarray([7, -7])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 2, cv2.LINE_AA)


def draw_manipulation_inset(
    image: np.ndarray,
    frame: dict,
    object_mesh: ObjectMeshFrame | None,
    mano_edges: dict[int, np.ndarray],
    view_basis: np.ndarray,
) -> None:
    if object_mesh is None or not frame.get("hands"):
        return
    near_points = [np.asarray(object_mesh.vertices, dtype=float)]
    for hand in frame["hands"]:
        near_points.append(hand_vertices(hand, "_world_m"))
        near_points.append(np.asarray(hand["joints3d_world_m"], dtype=float))
    pts = np.concatenate(near_points, axis=0)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return
    width, height = image.shape[1], image.shape[0]
    inset_w = min(330, width - 48)
    inset_h = min(250, height - 128)
    if inset_w < 220 or inset_h < 160:
        return
    x0 = width - inset_w - 18
    y0 = 72
    roi = np.full((inset_h, inset_w, 3), (251, 251, 247), dtype=np.uint8)
    projector = build_world_projector(pts[finite], view_basis, (inset_w, inset_h))
    draw_reference_grid(roi, projector, 0.35)
    draw_object_mesh(roi, object_mesh, projector)
    for hand in frame["hands"]:
        draw_hand_world(roi, hand, projector, mano_edges, mesh_stride=2, joint_radius=2, label=False)
    gap, side = hand_mesh_gap_m(frame, object_mesh)
    cv2.rectangle(roi, (0, 0), (inset_w - 1, inset_h - 1), (55, 55, 55), 1)
    cv2.putText(roi, "MANIPULATION DETAIL", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 2, cv2.LINE_AA)
    if gap is not None:
        gap_mm = 1000.0 * gap
        text = f"nearest {side} hand gap {gap_mm:.1f} mm"
        cv2.putText(roi, text, (12, inset_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25, 25, 25), 2, cv2.LINE_AA)
    image[y0 : y0 + inset_h, x0 : x0 + inset_w] = roi


def draw_reference_grid(image: np.ndarray, projector: WorldProjector, radius: float) -> None:
    extent = max(0.25, radius * 0.9)
    q_center = projector.q_center
    z = q_center[2] - 0.5 * radius
    values = np.linspace(-extent, extent, 9)
    for value in values:
        x = q_center[0] + value
        draw_polyline(
            image,
            np.asarray([[x, q_center[1] - extent, z], [x, q_center[1] + extent, z]], dtype=float) @ projector.basis,
            projector,
            (224, 226, 220),
            1,
        )
        y = q_center[1] + value
        draw_polyline(
            image,
            np.asarray([[q_center[0] - extent, y, z], [q_center[0] + extent, y, z]], dtype=float) @ projector.basis,
            projector,
            (224, 226, 220),
            1,
        )


def camera_frustum_points(T: np.ndarray, radius: float) -> np.ndarray:
    right, up, forward = camera_axes(T)
    center = T[:3, 3]
    length = max(0.08, min(0.22, 0.18 * radius))
    width = length * 0.72
    height = length * 0.46
    plane = center + forward * length
    corners = np.asarray(
        [
            plane - right * width - up * height,
            plane + right * width - up * height,
            plane + right * width + up * height,
            plane - right * width + up * height,
        ],
        dtype=float,
    )
    return np.vstack([center[None, :], corners])


def draw_camera_frustum(
    image: np.ndarray,
    T: np.ndarray,
    projector: WorldProjector,
    radius: float,
) -> None:
    pts = camera_frustum_points(T, radius)
    cam = pts[0]
    corners = pts[1:]
    draw_polyline(image, corners, projector, (20, 20, 20), 2, closed=True)
    for corner in corners:
        draw_polyline(image, np.vstack([cam, corner]), projector, (20, 20, 20), 2)
    right, up, forward = camera_axes(T)
    axis_len = max(0.06, min(0.16, 0.13 * radius))
    axes = [
        (cam + forward * axis_len, (190, 70, 20)),
        (cam + up * axis_len, (35, 120, 35)),
        (cam + right * axis_len, (70, 70, 70)),
    ]
    cam_xy = project_world(cam[None, :], projector)[0]
    cv2.circle(image, tuple(cam_xy), 6, (15, 15, 15), -1, cv2.LINE_AA)
    for end, color in axes:
        end_xy = project_world(end[None, :], projector)[0]
        cv2.arrowedLine(image, tuple(cam_xy), tuple(end_xy), color, 2, cv2.LINE_AA, tipLength=0.24)
    label_xy = tuple((cam_xy + np.asarray([9, -9])).astype(int))
    cv2.putText(image, "HEAD CAM", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 2, cv2.LINE_AA)


def object_extent_points(obj: dict, basis: np.ndarray) -> np.ndarray:
    p = np.asarray(obj["center_world_m"], dtype=float)
    radius = float(obj.get("radius_m", 0.0) or 0.0)
    if radius <= 0.0:
        return p[None, :]
    offsets = radius * np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=float,
    )
    return p[None, :] + offsets @ basis


def mask_patch_pixels(obj: dict) -> np.ndarray | None:
    mask_path = obj.get("mask_path")
    if not mask_path:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = mask > 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20.0:
        return None
    hull = cv2.convexHull(contour)
    epsilon = max(1.5, 0.018 * cv2.arcLength(hull, True))
    poly = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(float)
    if len(poly) < 3:
        return None
    center = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
    poly = poly[np.argsort(angles)]
    if len(poly) > 10:
        step = int(math.ceil(len(poly) / 10))
        poly = poly[::step][:10]
    size = obj.get("mask_image_size")
    if size is None:
        xs = [float(poly[:, 0].max()), float(obj.get("bbox_xyxy", [0, 0, mask.shape[1] - 1, mask.shape[0] - 1])[2])]
        ys = [float(poly[:, 1].max()), float(obj.get("bbox_xyxy", [0, 0, mask.shape[1] - 1, mask.shape[0] - 1])[3])]
        size = [mask.shape[1], mask.shape[0]]
        if max(xs) <= mask.shape[1] and max(ys) <= mask.shape[0]:
            size = [mask.shape[1], mask.shape[0]]
    sx = float(obj.get("source_image_size", [mask.shape[1], mask.shape[0]])[0]) / float(size[0])
    sy = float(obj.get("source_image_size", [mask.shape[1], mask.shape[0]])[1]) / float(size[1])
    return poly * np.asarray([sx, sy], dtype=float)


def object_patch_points(obj: dict, frame: dict) -> np.ndarray | None:
    pose_type = str(obj.get("pose_type", ""))
    if "bag" not in pose_type and "deformable" not in pose_type:
        return None
    intrinsics = obj.get("source_intrinsics")
    if intrinsics is None:
        for hand in frame.get("hands", []):
            if hand.get("source_intrinsics") is not None:
                intrinsics = hand["source_intrinsics"]
                break
    if obj.get("bbox_xyxy") is None or intrinsics is None or obj.get("depth_m") is None:
        return None
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=float)
    points_2d = mask_patch_pixels(obj)
    if points_2d is None:
        x1, y1, x2, y2 = np.asarray(obj["bbox_xyxy"], dtype=float)
        w = max(1.0, x2 - x1 + 1.0)
        h = max(1.0, y2 - y1 + 1.0)
        inset = 0.08
        points_2d = np.asarray(
            [
                [x1 + inset * w, y1 + inset * h],
                [x2 - inset * w, y1 + inset * h],
                [x2 - inset * w, y2 - inset * h],
                [x1 + inset * w, y2 - inset * h],
            ],
            dtype=float,
        )
    center_2d = np.asarray(obj["center_xy"], dtype=float)
    points_2d = np.vstack([points_2d, center_2d[None, :]])
    rays = np.c_[(points_2d[:, 0] - cx) / fx, (points_2d[:, 1] - cy) / fy, np.ones(len(points_2d))]
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    pts = (T @ np.c_[rays * float(obj["depth_m"]), np.ones(len(rays))].T).T[:, :3]
    center = np.asarray(obj["center_world_m"], dtype=float)
    span = float(np.max(np.linalg.norm(pts[:-1] - center[None, :], axis=1)))
    if not np.isfinite(span) or span < 1e-6:
        return None
    scale = min(1.0, 0.55 / span)
    return center[None, :] + scale * (pts - center[None, :])


def draw_object_extent(
    image: np.ndarray,
    obj: dict,
    projector: WorldProjector,
    patch: np.ndarray | None = None,
) -> None:
    p = np.asarray(obj["center_world_m"], dtype=float)
    radius = float(obj.get("radius_m", 0.0) or 0.0)
    p_xy = project_world(p[None, :], projector)[0]
    if patch is not None:
        patch_xy = project_world(patch[:-1], projector)
        overlay = image.copy()
        cv2.fillConvexPoly(overlay, patch_xy, (70, 80, 220), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
        cv2.polylines(image, [patch_xy], True, OBJECT_COLOR, 3, cv2.LINE_AA)
        patch_center_xy = patch_xy.mean(axis=0).astype(int)
        object_center_xy = project_world(patch[-1:], projector)[0]
        cv2.drawMarker(image, tuple(object_center_xy), OBJECT_COLOR, cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        cv2.putText(image, "MASK RAY PATCH", tuple((patch_center_xy + np.asarray([9, 14])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, OBJECT_COLOR, 2, cv2.LINE_AA)
        return
    cv2.circle(image, tuple(p_xy), 7, OBJECT_COLOR, -1, cv2.LINE_AA)
    if radius > 0.0:
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        rings = [
            np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1),
            np.stack([radius * np.cos(theta), np.zeros_like(theta), radius * np.sin(theta)], axis=1),
            np.stack([np.zeros_like(theta), radius * np.cos(theta), radius * np.sin(theta)], axis=1),
        ]
        for ring in rings:
            draw_polyline(image, p[None, :] + ring @ projector.basis, projector, OBJECT_COLOR, 2, closed=True)
    cv2.putText(image, "OBJECT", tuple((p_xy + np.asarray([9, 14])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OBJECT_COLOR, 2, cv2.LINE_AA)


def render_3d_frame(
    frames: list[dict],
    index: int,
    camera_positions: np.ndarray,
    size: tuple[int, int],
    mano_edges: dict[int, np.ndarray],
    display_basis: np.ndarray,
    object_meshes: dict[int, ObjectMeshFrame] | None = None,
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), (244, 245, 240), dtype=np.uint8)
    frame = frames[index]
    T = camera_transform(frame)
    view_basis = camera_display_basis(T, frame)
    local_path = camera_positions[max(0, index - 30) : index + 1]
    scene_pts = [camera_frustum_points(T, 0.9)]
    for hand in frame["hands"]:
        joints = np.asarray(hand["joints3d_world_m"], dtype=float)
        verts = hand_vertices(hand, "_world_m")
        scene_pts.extend([joints, verts])
    obj = frame.get("object", {})
    object_mesh = (object_meshes or {}).get(int(frame["frame_idx"]))
    if object_mesh is not None:
        scene_pts.append(object_mesh.vertices)
    if obj.get("center_world_m") is not None:
        patch = object_patch_points(obj, frame)
        if patch is not None:
            scene_pts.append(patch)
        else:
            scene_pts.append(object_extent_points(obj, view_basis))
    pts = np.concatenate(scene_pts, axis=0)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise RuntimeError("3D renderer received no finite scene points")
    pts = pts[finite]
    projector = build_world_projector(pts, view_basis, size)
    q = pts @ view_basis.T
    radius = max(0.20, float(np.percentile(np.linalg.norm(q - projector.q_center[None, :], axis=1), 92)))

    draw_reference_grid(image, projector, radius)
    draw_polyline(image, local_path, projector, (112, 112, 112), 2)
    past_path = camera_positions[max(0, index - 30) : index + 1]
    if len(past_path) > 1:
        draw_polyline(image, past_path, projector, (15, 15, 15), 4)
    if object_mesh is not None:
        draw_object_mesh(image, object_mesh, projector)
    elif obj.get("center_world_m") is not None:
        draw_object_extent(image, obj, projector, object_patch_points(obj, frame))
    for hand in frame["hands"]:
        draw_hand_world(image, hand, projector, mano_edges, mesh_stride=2, joint_radius=3, label=True)
    draw_camera_frustum(image, T, projector, radius)
    draw_scale_bar(image, projector, 0.25)
    draw_manipulation_inset(image, frame, object_mesh, mano_edges, view_basis)
    gap, side = hand_mesh_gap_m(frame, object_mesh)
    cv2.putText(image, "3D WORLD RECONSTRUCTION", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "head camera trajectory, MANO hands, object mesh", (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (85, 85, 85), 1, cv2.LINE_AA)
    if gap is not None:
        cv2.putText(
            image,
            f"nearest {side} hand-object gap {1000.0 * gap:.1f} mm",
            (16, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(image, f"frame {int(frame['frame_idx']):04d}", (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return image


def render_outputs(args: argparse.Namespace, frames: list[dict], render: RenderSpec) -> None:
    cap, info = open_video(args.clip)
    faces = mano_faces(args.mano_right)
    vertex_counts = sorted(
        {
            len(hand_vertices(hand, "_source_camera_m"))
            for frame in frames
            for hand in frame.get("hands", [])
            if "source_intrinsics" in hand
        }
    )
    mano_edges = {count: mano_edges_from_faces(faces, count) for count in vertex_counts}
    overlay_path = args.output_dir / "overlay_mano_object.mp4"
    recon_path = args.output_dir / "reconstruction_3d_world.mp4"
    side_path = args.output_dir / "side_by_side.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay = cv2.VideoWriter(str(overlay_path), fourcc, render.fps, (render.width, render.height))
    recon = cv2.VideoWriter(str(recon_path), fourcc, render.fps, (render.width, render.height))
    side = cv2.VideoWriter(str(side_path), fourcc, render.fps, (render.width * 2, render.height))
    if not overlay.isOpened() or not recon.isOpened() or not side.isOpened():
        raise RuntimeError("failed to open video writers")
    camera_positions = np.asarray([frame["camera"]["position_world_m"] for frame in frames], dtype=float)
    display_basis = world_display_basis(frames, camera_positions)
    object_meshes = load_object_mesh_archive(args.object_mesh_npz)
    sx, sy = render.width / info.width, render.height / info.height
    try:
        for i, frame_ann in enumerate(tqdm(frames, desc="render")):
            frame = read_video_frame(cap, int(frame_ann["frame_idx"]))
            frame = cv2.resize(frame, (render.width, render.height), interpolation=cv2.INTER_AREA)
            object_mesh = object_meshes.get(int(frame_ann["frame_idx"]))
            draw_object_overlay(frame, frame_ann, sx, sy, object_mesh)
            draw_hand_overlay(frame, frame_ann, sx, sy, mano_edges)
            draw_projected_object_mesh_overlay(frame, frame_ann, object_mesh, sx, sy)
            put_caption(frame, frame_ann["caption"], frame_ann["frame_idx"])
            panel = render_3d_frame(frames, i, camera_positions, (render.width, render.height), mano_edges, display_basis, object_meshes)
            overlay.write(frame)
            recon.write(panel)
            side.write(np.concatenate([frame, panel], axis=1))
    finally:
        cap.release()
        overlay.release()
        recon.release()
        side.release()


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.render_only_annotations is not None:
        frames = load_json(args.render_only_annotations)["frames"]
        localize_render_only_mask_paths(frames, args.remote_output_root, args.local_output_root)
        if args.frame_start is not None or args.frame_end is not None:
            start = int(frames[0]["frame_idx"]) if args.frame_start is None else int(args.frame_start)
            end = int(frames[-1]["frame_idx"]) if args.frame_end is None else int(args.frame_end)
            frames = [frame for frame in frames if start <= int(frame["frame_idx"]) <= end]
            if not frames:
                raise RuntimeError(f"render-only slice {start}:{end} contains no frames")
            expected = np.arange(start, end + 1, dtype=int)
            actual = np.asarray([int(frame["frame_idx"]) for frame in frames], dtype=int)
            if len(actual) != len(expected) or not np.array_equal(actual, expected):
                raise RuntimeError("render-only annotations are not source-contiguous over requested slice")
        cap, info = open_video(args.clip)
        cap.release()
        if not frames or any("camera" not in frame for frame in frames):
            raise RuntimeError("render-only annotations must contain camera/world fields")
        render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
        render_outputs(args, frames, render)
        qc = {
            "status": "ok",
            "mode": "render_only",
            "clip": str(args.clip),
            "processed_frames": len(frames),
            "render": render.__dict__,
            "elapsed_s": time.time() - started,
            "outputs": {
                "overlay": str(args.output_dir / "overlay_mano_object.mp4"),
                "reconstruction_3d": str(args.output_dir / "reconstruction_3d_world.mp4"),
                "side_by_side": str(args.output_dir / "side_by_side.mp4"),
            },
        }
        (args.output_dir / "render_only_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        return qc
    if not args.allow_legacy_object_heuristics:
        raise RuntimeError(
            "full fusion uses retired category-specific object heuristics; "
            "run V2 object-plan scripts for new object annotations, or pass "
            "--allow-legacy-object-heuristics only to reproduce old V1 outputs"
        )
    raw_all = load_json(args.wilor_raw)["frames"]
    raw = raw_all
    full_raw_count = len(raw_all)
    if args.frame_start is not None or args.frame_end is not None:
        start = 0 if args.frame_start is None else int(args.frame_start)
        end = len(raw) - 1 if args.frame_end is None else int(args.frame_end)
        if start < 0 or end < start or end >= len(raw):
            raise RuntimeError(f"invalid frame slice {start}:{end} for {len(raw)} raw frames")
        raw = raw[start : end + 1]
        for expected, frame in enumerate(raw, start=start):
            if int(frame["frame_idx"]) != expected:
                raise RuntimeError("raw WiLoR frames are not source-contiguous")
    droid = np.load(args.droid_npz)
    actions_path = args.actions_json if args.actions_json is not None else args.clip.with_suffix(".json")
    actions = load_actions(actions_path)
    cap, info = open_video(args.clip)
    cap.release()
    intrinsics = droid["intrinsics_source"].astype(float)
    recon = load_droid_reconstruction(args.droid_reconstruction)
    render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
    hand_scale_qc = hand_metric_scale_from_raw(raw_all)
    frames, hand_stats = smooth_hands(raw, info.fps, intrinsics, float(hand_scale_qc["wilor_local_to_meters"]))
    droid_scale_qc = estimate_droid_metric_scale_from_raw(
        raw_all,
        intrinsics,
        float(hand_scale_qc["wilor_local_to_meters"]),
        recon,
        (info.width, info.height),
        int(args.max_keyframe_gap),
    )
    T = transform_hands_to_world(frames, args.droid_npz, float(droid_scale_qc["droid_to_meters"]))

    object_meas, object_measure_qc = run_object_masks(args, frames, actions, render)
    semantic_interval = tuple(object_measure_qc["semantic_interval"])
    object_track_qc = fill_object_track(
        frames,
        object_meas,
        info.fps,
        (info.width, info.height),
        list(object_measure_qc["action_labels"]),
    )
    object_world_qc = attach_object_world(
        frames,
        T,
        intrinsics,
        recon,
        float(droid_scale_qc["droid_to_meters"]),
        (info.width, info.height),
        int(args.max_keyframe_gap),
    )
    hand_object_qc = apply_hand_object_contact_correction(frames, info.fps)

    annotations_path = args.output_dir / "annotations_v1_full.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    render_outputs(args, frames, render)

    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "video": info.__dict__,
        "processed_frames": len(frames),
        "source_frame_range": [int(frames[0]["frame_idx"]), int(frames[-1]["frame_idx"])],
        "camera": {
            "backend": "DROID-SLAM",
            "dense_frames": int(len(T)),
            "source_dense_frames_available": int(len(droid["T_world_camera"])),
            "full_source_timeline": bool(len(T) == len(frames) == info.frame_count and full_raw_count == info.frame_count),
            "trajectory_npz": str(args.droid_npz),
        },
        "hands": hand_stats,
        "scale": {
            "hand": hand_scale_qc,
            "droid": droid_scale_qc,
            "intrinsics_source": intrinsics.astype(float).tolist(),
            "droid_reconstruction": str(args.droid_reconstruction),
            "droid_depth_level": recon["depth_level"],
        },
        "object_measurement": object_measure_qc,
        "object_track": object_track_qc,
        "object_world": object_world_qc,
        "hand_object_contact_correction": hand_object_qc,
        "render": render.__dict__,
        "elapsed_s": time.time() - started,
        "outputs": {
            "annotations": str(annotations_path),
            "overlay": str(args.output_dir / "overlay_mano_object.mp4"),
            "reconstruction_3d": str(args.output_dir / "reconstruction_3d_world.mp4"),
            "side_by_side": str(args.output_dir / "side_by_side.mp4"),
        },
    }
    (args.output_dir / "qc_v1_full.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def localize_render_only_mask_paths(frames: list[dict], remote_root: Path | None, local_root: Path | None) -> None:
    if remote_root is None and local_root is None:
        return
    if remote_root is None or local_root is None:
        raise RuntimeError("--remote-output-root and --local-output-root must be provided together")
    for frame in frames:
        obj = frame.get("object", {})
        mask_path = obj.get("mask_path")
        if not mask_path:
            continue
        path = Path(str(mask_path))
        if path.exists():
            continue
        try:
            rel = path.relative_to(remote_root)
        except ValueError as exc:
            raise RuntimeError(f"mask path is outside --remote-output-root: {path}") from exc
        candidate = local_root / rel
        if not candidate.exists():
            raise RuntimeError(f"localized mask path is missing: {candidate}")
        obj["mask_path"] = str(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--wilor-raw", type=Path, default=Path("outputs/examples/tomato_v1_full/wilor/wilor_raw.json"))
    parser.add_argument("--droid-npz", type=Path, default=Path("outputs/examples/tomato_v1_full/droid/droid_dense_trajectory.npz"))
    parser.add_argument("--droid-reconstruction", type=Path, default=Path("outputs/examples/tomato_v1_full/droid/droid_keyframe_reconstruction.pth"))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/fused"))
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    parser.add_argument("--object-label", default="auto")
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--owl-threshold", type=float, default=0.03)
    parser.add_argument("--max-keyframe-gap", type=int, default=15)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-only-annotations", type=Path)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--actions-json", type=Path)
    parser.add_argument("--object-mesh-npz", type=Path)
    parser.add_argument("--allow-legacy-object-heuristics", action="store_true")
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
