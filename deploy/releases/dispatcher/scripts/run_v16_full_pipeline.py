#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.frame_count <= 0 or info.fps <= 0.0 or info.width <= 0 or info.height <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata for {path}: {info}")
    return cap, info


def video_info(path: Path) -> VideoInfo:
    cap, info = open_video(path)
    cap.release()
    return info


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def run_command(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}")


def check_video(path: Path, expected: VideoInfo) -> dict:
    info = video_info(path)
    frame_match = int(info.frame_count) == int(expected.frame_count)
    fps_match = abs(float(info.fps) - float(expected.fps)) <= 0.02
    if not frame_match:
        raise RuntimeError(f"{path} has {info.frame_count} frames, expected {expected.frame_count}")
    if not fps_match:
        raise RuntimeError(f"{path} has fps {info.fps}, expected {expected.fps}")
    return {
        "path": str(path),
        "frame_count": int(info.frame_count),
        "fps": float(info.fps),
        "width": int(info.width),
        "height": int(info.height),
        "frame_count_match": bool(frame_match),
        "fps_match": bool(fps_match),
    }


def save_mesh_archive(path: Path, frame_indices: list[int], vertices: list[np.ndarray], faces: list[np.ndarray]) -> None:
    if len(frame_indices) != len(vertices) or len(frame_indices) != len(faces):
        raise RuntimeError("mesh archive frame/vertex/face list length mismatch")
    vertex_offsets = [0]
    face_offsets = [0]
    for v, f in zip(vertices, faces, strict=True):
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError("cannot save empty mesh in V16 archive")
        vertex_offsets.append(vertex_offsets[-1] + int(len(v)))
        face_offsets.append(face_offsets[-1] + int(len(f)))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices).astype(np.float32),
        faces=np.vstack(faces).astype(np.int32),
    )


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {missing}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError(f"{path} has invalid mesh offsets")
    for i, idx in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        v = vertices[v0:v1]
        f = faces[f0:f1]
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError(f"{path} contains empty mesh for frame {idx}")
        if int(f.min()) < 0 or int(f.max()) >= len(v):
            raise RuntimeError(f"{path} contains out-of-range face index for frame {idx}")
        out[int(idx)] = (v, f)
    return out


def raw_frame_manifest(clip: Path, output_dir: Path, *, render_width: int) -> dict:
    cap, info = open_video(clip)
    render_height = int(round(render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    rgb_dir = output_dir / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for frame_idx in range(info.frame_count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {frame_idx}, expected {info.frame_count}")
            rgb = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write {path}")
            frames.append(
                {
                    "index": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx / info.fps),
                    "rgb": str(path),
                    "source_width": int(info.width),
                    "source_height": int(info.height),
                    "manifest_width": int(render_width),
                    "manifest_height": int(render_height),
                }
            )
    finally:
        cap.release()
    manifest = {
        "status": "ok",
        "method": "v16_raw_frame_manifest",
        "clip": str(clip),
        "video": info.__dict__,
        "frames": frames,
    }
    path = output_dir / "raw_frame_manifest" / "manifest.json"
    write_json(path, manifest)
    return {"manifest": str(path), "frames": int(len(frames)), "video": info.__dict__}


def make_depth_manifest_from_masks(annotations_path: Path, raw_manifest_path: Path, output_dir: Path) -> dict:
    annotations = load_json(annotations_path)
    raw_manifest = load_json(raw_manifest_path)
    raw_by_frame = {int(row["frame_idx"]): row for row in raw_manifest["frames"]}
    mask_dir = output_dir / "depth_request_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    rows = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        obj = frame.get("object", {})
        mask_path = obj.get("mask_path")
        status = str(obj.get("status", ""))
        if mask_path and Path(mask_path).exists() and status not in {"outside_semantic_interval", "not_visible"}:
            raw_row = raw_by_frame.get(idx)
            if raw_row is None:
                raise RuntimeError(f"raw manifest missing frame {idx}")
            mask_src = Path(mask_path)
            mask_dst = mask_dir / f"{idx:06d}.png"
            if not mask_dst.exists() or mask_dst.stat().st_size != mask_src.stat().st_size:
                shutil.copy2(mask_src, mask_dst)
            entries.append(
                {
                    "index": int(len(entries)),
                    "frame_idx": int(idx),
                    "rgb": raw_row["rgb"],
                    "mask": str(mask_dst),
                    "source_mask": str(mask_src),
                    "source_object_status": status,
                    "source_object_label": obj.get("label"),
                }
            )
            rows.append({"frame_idx": idx, "status": "depth_requested", "mask": str(mask_dst), "source_mask": str(mask_src)})
        else:
            rows.append({"frame_idx": idx, "status": "inactive_or_missing_mask"})
    if not entries:
        raise RuntimeError("no masked object frames available for V16 depth")
    manifest = {
        "status": "ok",
        "method": "v16_masked_depth_request_manifest",
        "annotations": str(annotations_path),
        "raw_manifest": str(raw_manifest_path),
        "frames": entries,
        "rows": rows,
    }
    path = output_dir / "depth_request_manifest.json"
    write_json(path, manifest)
    return {
        "manifest": str(path),
        "requested_frames": int(len(entries)),
        "first_frame": int(entries[0]["frame_idx"]),
        "last_frame": int(entries[-1]["frame_idx"]),
    }


def build_annotation_camera_qc(annotations_path: Path, clip: Path, output_dir: Path, info: VideoInfo) -> dict:
    payload = load_json(annotations_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != info.frame_count:
        raise RuntimeError(f"{annotations_path} does not contain one camera row per source frame")
    positions = []
    for expected_idx, frame in enumerate(frames):
        idx = int(frame.get("frame_idx", -1))
        if idx != expected_idx:
            raise RuntimeError(f"annotation camera timeline is not source-contiguous at row {expected_idx}: {idx}")
        camera = frame.get("camera", {})
        T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        pos = np.asarray(camera.get("position_world_m"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"frame {idx} has invalid T_world_camera_metric")
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise RuntimeError(f"frame {idx} has invalid position_world_m")
        if np.linalg.norm(T[:3, 3] - pos) > 1e-4:
            raise RuntimeError(f"frame {idx} camera position does not match T_world_camera_metric translation")
        positions.append(pos)
    xyz = np.vstack(positions)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(0, dtype=np.float64)
    report = {
        "status": "ok",
        "method": "v16_annotation_camera_trajectory_qc",
        "clip": str(clip),
        "annotations": str(annotations_path),
        "video": info.__dict__,
        "processed_frames": int(len(frames)),
        "dense_trajectory_frames": int(len(frames)),
        "full_source_timeline": True,
        "pose_convention": "T_world_camera_metric from annotation source, position_world_m equals translation",
        "calibration_source": "inherited from full annotation camera stream",
        "trajectory_path_length": float(steps.sum()),
        "median_step": float(np.median(steps)) if len(steps) else 0.0,
        "p95_step": float(np.percentile(steps, 95.0)) if len(steps) else 0.0,
        "max_step": float(np.max(steps)) if len(steps) else 0.0,
        "position_median_world_m": np.median(xyz, axis=0).astype(float).tolist(),
        "position_extent_world_m": (xyz.max(axis=0) - xyz.min(axis=0)).astype(float).tolist(),
    }
    path = output_dir / "camera_qc_annotation_trajectory.json"
    write_json(path, report)
    return {"path": str(path), "report": report}


def load_metric_depth(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing depth keys: {missing}")
    frame_idx = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float32)
    intrinsics = blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if len(frame_idx) != depth.shape[0] or len(frame_idx) != intrinsics.shape[0]:
        raise RuntimeError(f"{path} has inconsistent depth rows")
    return {
        "frame_idx": frame_idx,
        "depth": depth,
        "intrinsics": intrinsics,
        "frame_to_i": {int(idx): int(i) for i, idx in enumerate(frame_idx)},
    }


def build_faces(index_grid: np.ndarray, vertices: np.ndarray, max_edge_m: float) -> np.ndarray:
    faces = []
    rows, cols = index_grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = (
                int(index_grid[r, c]),
                int(index_grid[r, c + 1]),
                int(index_grid[r + 1, c]),
                int(index_grid[r + 1, c + 1]),
            )
            for tri in ((corners[0], corners[1], corners[2]), (corners[1], corners[3], corners[2])):
                if min(tri) < 0:
                    continue
                pts = vertices[np.asarray(tri, dtype=np.int64)]
                edge = max(
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[0]),
                )
                if edge <= max_edge_m:
                    faces.append(tri)
    return np.asarray(faces, dtype=np.int32)


def remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0], faces
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (hom @ matrix.T)[:, :3]


def mesh_from_mask_depth(
    frame: dict,
    depth: dict,
    *,
    mask_stride: int,
    mask_erode_px: int,
    max_triangle_edge_m: float,
    min_vertices: int,
    min_faces: int,
    min_depth_m: float,
    max_depth_m: float,
    depth_low_quantile: float,
    depth_high_quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    idx = int(frame["frame_idx"])
    obj = frame.get("object", {})
    mask_path = Path(str(obj.get("mask_path", "")))
    if not mask_path.exists():
        raise RuntimeError(f"frame {idx} has no mask path for active mesh state")
    depth_i = depth["frame_to_i"].get(idx)
    if depth_i is None:
        raise RuntimeError(f"frame {idx} has object mask but no metric depth")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object mask {mask_path}")
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    if depth_m.shape != mask.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_erode_px > 0:
        kernel = np.ones((2 * mask_erode_px + 1, 2 * mask_erode_px + 1), dtype=np.uint8)
        mask_bool = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1) > 0
    else:
        mask_bool = mask > 0
    valid = mask_bool & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    values = depth_m[valid]
    if values.size < min_vertices:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            {
                "frame_idx": int(idx),
                "status": "rejected_underconstrained_mask_depth",
                "reason": "too_few_valid_masked_depth_pixels",
                "valid_masked_depth_pixels": int(values.size),
                "min_vertices": int(min_vertices),
                "mask_path": str(mask_path),
            },
        )
    lo = float(np.quantile(values, depth_low_quantile))
    hi = float(np.quantile(values, depth_high_quantile))
    keep = valid & (depth_m >= lo) & (depth_m <= hi)
    ys = np.arange(0, depth_m.shape[0], int(mask_stride), dtype=np.int32)
    xs = np.arange(0, depth_m.shape[1], int(mask_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled_keep = keep[np.ix_(ys, xs)]
    flat_x = grid_x[sampled_keep].astype(np.float64)
    flat_y = grid_y[sampled_keep].astype(np.float64)
    flat_z = depth_m[np.ix_(ys, xs)][sampled_keep].astype(np.float64)
    if len(flat_z) < min_vertices:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            {
                "frame_idx": int(idx),
                "status": "rejected_underconstrained_mask_depth",
                "reason": "too_few_sampled_mesh_vertices",
                "sampled_vertices": int(len(flat_z)),
                "min_vertices": int(min_vertices),
                "mask_path": str(mask_path),
            },
        )
    fx, fy, cx, cy = depth["intrinsics"][int(depth_i)].astype(float).tolist()
    source_intrinsics = None
    for hand in frame.get("hands", []):
        if hand.get("source_intrinsics") is not None:
            source_intrinsics = np.asarray(hand["source_intrinsics"], dtype=np.float64)
            break
    if source_intrinsics is None:
        raise RuntimeError(f"frame {idx} has no source_intrinsics for source-camera object mesh")
    if source_intrinsics.shape != (4,) or not np.isfinite(source_intrinsics).all():
        raise RuntimeError(f"frame {idx} has invalid source_intrinsics")
    sfx, sfy, scx, scy = source_intrinsics.astype(float).tolist()
    source_w = float(frame.get("source_width", 0.0) or 0.0)
    source_h = float(frame.get("source_height", 0.0) or 0.0)
    if source_w <= 0.0 or source_h <= 0.0:
        source_w = max(float(mask.shape[1]), float(scx * 2.0))
        source_h = max(float(mask.shape[0]), float(scy * 2.0))
    scale_x = source_w / float(mask.shape[1])
    scale_y = source_h / float(mask.shape[0])
    source_x = flat_x * scale_x
    source_y = flat_y * scale_y
    depth_ref = obj.get("depth_m")
    if depth_ref is None:
        depth_ref = float(np.median(values))
    depth_ref = float(depth_ref)
    if not np.isfinite(depth_ref) or depth_ref <= 0.0:
        raise RuntimeError(f"frame {idx} has invalid object depth_m for source-camera mesh")
    flat_z = flat_z * (depth_ref / max(1e-9, float(np.median(flat_z))))
    camera_vertices = np.column_stack(((source_x - scx) * flat_z / sfx, (source_y - scy) * flat_z / sfy, flat_z))
    index_grid = np.full(sampled_keep.shape, -1, dtype=np.int32)
    index_grid[sampled_keep] = np.arange(len(camera_vertices), dtype=np.int32)
    faces = build_faces(index_grid, camera_vertices, max_triangle_edge_m)
    camera_vertices, faces = remove_unreferenced(camera_vertices, faces)
    if len(camera_vertices) < min_vertices or len(faces) < min_faces:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            {
                "frame_idx": int(idx),
                "status": "rejected_underconstrained_mask_depth",
                "reason": "too_few_vertices_or_faces_after_surface_connectivity",
                "vertices": int(len(camera_vertices)),
                "faces": int(len(faces)),
                "min_vertices": int(min_vertices),
                "min_faces": int(min_faces),
                "mask_path": str(mask_path),
            },
        )
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError(f"frame {idx} has invalid camera pose")
    vertices_world = transform_points(camera_vertices, T)
    row = {
        "frame_idx": int(idx),
        "status": "measured_mesh_from_mask_metric_depth",
        "vertices": int(len(vertices_world)),
        "faces": int(len(faces)),
        "mask_path": str(mask_path),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "depth_median_m": float(np.median(values)),
        "annotation_depth_anchor_m": depth_ref,
        "surface_depth_model": "unidepth_relative_depth_scaled_to_annotation_source_camera_depth",
        "source_intrinsics": [float(sfx), float(sfy), float(scx), float(scy)],
        "world_extent_m": (vertices_world.max(axis=0) - vertices_world.min(axis=0)).astype(float).tolist(),
    }
    return vertices_world.astype(np.float32), faces.astype(np.int32), row


def nearest_measured_mesh(target_idx: int, measured_frames: list[int], max_gap: int) -> int | None:
    if not measured_frames:
        return None
    nearest = min(measured_frames, key=lambda frame_idx: abs(frame_idx - target_idx))
    if abs(nearest - target_idx) > max_gap:
        return None
    return int(nearest)


def row_by_frame_status(rows: list[dict], frame_idx: int) -> str:
    for row in rows:
        if int(row.get("frame_idx", -1)) == int(frame_idx):
            return str(row.get("status", "unknown"))
    return "unknown"


def build_full_mesh_stream(args: argparse.Namespace, annotations_path: Path, depth_npz: Path, output_dir: Path) -> dict:
    annotations = load_json(annotations_path)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{annotations_path} has no frames")
    depth = load_metric_depth(depth_npz)
    measured_vertices: dict[int, np.ndarray] = {}
    measured_faces: dict[int, np.ndarray] = {}
    frame_indices = []
    vertices_all = []
    faces_all = []
    rows = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        obj = frame.get("object", {})
        status = str(obj.get("status", ""))
        if obj.get("mask_path") and status not in {"outside_semantic_interval", "not_visible"}:
            v, f, row = mesh_from_mask_depth(
                frame,
                depth,
                mask_stride=int(args.object_mesh_stride),
                mask_erode_px=int(args.object_mesh_erode_px),
                max_triangle_edge_m=float(args.object_mesh_max_edge_m),
                min_vertices=int(args.object_mesh_min_vertices),
                min_faces=int(args.object_mesh_min_faces),
                min_depth_m=float(args.object_mesh_min_depth_m),
                max_depth_m=float(args.object_mesh_max_depth_m),
                depth_low_quantile=float(args.object_mesh_depth_low_quantile),
                depth_high_quantile=float(args.object_mesh_depth_high_quantile),
            )
            if row["status"] == "measured_mesh_from_mask_metric_depth":
                measured_vertices[idx] = v
                measured_faces[idx] = f
                frame_indices.append(idx)
                vertices_all.append(v)
                faces_all.append(f)
                row["delivered_state"] = "measured"
            rows.append(row)
        elif status == "outside_semantic_interval":
            rows.append({"frame_idx": idx, "status": "inactive_outside_delivered_object_stream"})
        else:
            rows.append({"frame_idx": idx, "status": "no_mesh_measurement"})
    measured_frame_indices = sorted(measured_vertices)
    if not measured_frame_indices:
        raise RuntimeError("V16 object mesh stream has zero measured mesh frames")
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    active_without_mesh = []
    for row in rows:
        row_idx = int(row["frame_idx"])
        row_status = str(row["status"])
        frame = frame_by_idx[row_idx]
        label = str(frame["object"].get("label", ""))
        if row_status in {"no_mesh_measurement", "rejected_underconstrained_mask_depth"} and label not in {"", "None"}:
            active_without_mesh.append(row_idx)
    prediction_rows = []
    for idx in active_without_mesh:
        nearest = nearest_measured_mesh(int(idx), measured_frame_indices, int(args.object_mesh_prediction_max_gap))
        if nearest is None:
            raise RuntimeError(f"active object frame {idx} has no mesh measurement within prediction gap")
        v = measured_vertices[nearest].copy()
        f = measured_faces[nearest].copy()
        frame_indices.append(int(idx))
        vertices_all.append(v)
        faces_all.append(f)
        prediction_rows.append(
            {
                "frame_idx": int(idx),
                "status": "predicted_mesh_from_nearest_measured_surface",
                "source_frame_idx": int(nearest),
                "gap_frames": int(abs(nearest - int(idx))),
                "reason": row_by_frame_status(rows, int(idx)),
                "vertices": int(len(v)),
                "faces": int(len(f)),
            }
        )
    order = np.argsort(np.asarray(frame_indices, dtype=np.int64))
    frame_indices_sorted = [int(frame_indices[int(i)]) for i in order]
    vertices_sorted = [vertices_all[int(i)] for i in order]
    faces_sorted = [faces_all[int(i)] for i in order]
    archive = output_dir / "object_meshes_full_timeline.npz"
    save_mesh_archive(archive, frame_indices_sorted, vertices_sorted, faces_sorted)
    extents = [row["world_extent_m"] for row in rows if row.get("world_extent_m")]
    report = {
        "status": "ok",
        "method": "v16_mask_metric_depth_mesh_stream",
        "annotations": str(annotations_path),
        "metric_depth_npz": str(depth_npz),
        "archive": str(archive),
        "raw_frames": int(len(frames)),
        "mesh_frames": int(len(frame_indices_sorted)),
        "measured_mesh_frames": int(len(measured_frame_indices)),
        "predicted_mesh_frames": int(len(prediction_rows)),
        "first_mesh_frame": int(frame_indices_sorted[0]),
        "last_mesh_frame": int(frame_indices_sorted[-1]),
        "prediction_max_gap_frames": int(args.object_mesh_prediction_max_gap),
        "world_extent_median_m": np.median(np.asarray(extents, dtype=np.float64), axis=0).astype(float).tolist() if extents else None,
        "rows": rows,
        "prediction_rows": prediction_rows,
    }
    write_json(output_dir / "object_mesh_qc.json", report)
    return report


def patch_annotations_with_mesh_state(annotations_path: Path, mesh_report: dict, output_dir: Path) -> dict:
    payload = load_json(annotations_path)
    frames = payload["frames"]
    mesh_frames = set(map(int, load_mesh_archive(Path(mesh_report["archive"])).keys()))
    row_by_frame = {int(row["frame_idx"]): row for row in mesh_report.get("rows", [])}
    pred_by_frame = {int(row["frame_idx"]): row for row in mesh_report.get("prediction_rows", [])}
    unresolved = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        obj = frame.get("object", {})
        row = pred_by_frame.get(idx) or row_by_frame.get(idx)
        if idx in mesh_frames:
            obj["mesh_state"] = "measured" if idx not in pred_by_frame else "predicted"
            obj["mesh_archive"] = str(mesh_report["archive"])
            if row is not None:
                obj["mesh_qc"] = {k: v for k, v in row.items() if k not in {"mask_path"}}
        elif str(obj.get("status", "")) == "outside_semantic_interval":
            obj["mesh_state"] = "inactive"
        else:
            obj["mesh_state"] = "unresolved"
            unresolved.append(idx)
        frame["object"] = obj
    out = output_dir / "annotations_v16_full.json"
    write_json(out, {"frames": frames})
    report = {
        "status": "ok" if not unresolved else "unresolved_frames",
        "annotations": str(out),
        "raw_frames": int(len(frames)),
        "mesh_frames": int(len(mesh_frames)),
        "unresolved_mesh_frames": [int(x) for x in unresolved],
    }
    write_json(output_dir / "hand_camera_object_timeline_qc.json", report)
    if unresolved:
        raise RuntimeError(f"V16 annotations contain unresolved active mesh frames: {unresolved[:20]}")
    return report


def summarize(values: list[float]) -> dict:
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


def hand_vertices_from_frame(frame: dict) -> list[np.ndarray]:
    out = []
    for hand in frame.get("hands", []):
        for key in ("vertices_world_m", "vertices3d_world_m"):
            if key in hand:
                verts = np.asarray(hand[key], dtype=np.float64)
                if verts.ndim == 2 and verts.shape[1] == 3 and np.isfinite(verts).all():
                    out.append(verts)
                    break
    return out


def point_mesh_distance_sample(points: np.ndarray, vertices: np.ndarray, max_points: int = 160, max_vertices: int = 2000) -> np.ndarray:
    if len(points) == 0 or len(vertices) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(points) > max_points:
        take = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[take]
    if len(vertices) > max_vertices:
        take = np.linspace(0, len(vertices) - 1, max_vertices, dtype=np.int64)
        vertices = vertices[take]
    diff = points[:, None, :] - vertices[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def build_timeline_qc(annotations_path: Path, mesh_archive: Path, output_dir: Path, info: VideoInfo) -> dict:
    frames = load_json(annotations_path)["frames"]
    if len(frames) != info.frame_count:
        raise RuntimeError(f"annotation frame count {len(frames)} does not match raw frame count {info.frame_count}")
    expected = np.arange(info.frame_count, dtype=np.int64)
    actual = np.asarray([int(frame["frame_idx"]) for frame in frames], dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise RuntimeError("V16 annotations are not source-contiguous")
    meshes = load_mesh_archive(mesh_archive)
    camera_positions = []
    mesh_extents = []
    active_mesh_missing = []
    hand_object_min_dist = []
    hand_rows = 0
    for frame in frames:
        idx = int(frame["frame_idx"])
        camera = frame.get("camera", {})
        if "position_world_m" not in camera:
            raise RuntimeError(f"frame {idx} missing camera.position_world_m")
        camera_positions.append(np.asarray(camera["position_world_m"], dtype=np.float64))
        obj = frame.get("object", {})
        mesh_state = str(obj.get("mesh_state", ""))
        if mesh_state in {"measured", "predicted"} and idx not in meshes:
            active_mesh_missing.append(idx)
        if idx in meshes:
            vertices, _faces = meshes[idx]
            mesh_extents.append((vertices.max(axis=0) - vertices.min(axis=0)).astype(float))
            hand_sets = hand_vertices_from_frame(frame)
            for hand_vertices in hand_sets:
                hand_rows += 1
                distances = point_mesh_distance_sample(hand_vertices, vertices)
                if distances.size:
                    hand_object_min_dist.append(float(np.min(distances)))
    if active_mesh_missing:
        raise RuntimeError(f"active mesh states missing archive rows: {active_mesh_missing[:20]}")
    cam = np.vstack(camera_positions)
    steps = np.linalg.norm(np.diff(cam, axis=0), axis=1) if len(cam) > 1 else np.zeros(0, dtype=np.float64)
    ext = np.vstack(mesh_extents) if mesh_extents else np.zeros((0, 3), dtype=np.float64)
    report = {
        "status": "ok",
        "method": "v16_full_timeline_residual_qc",
        "annotations": str(annotations_path),
        "mesh_archive": str(mesh_archive),
        "raw_frame_count": int(info.frame_count),
        "annotation_frames": int(len(frames)),
        "mesh_frames": int(len(meshes)),
        "camera_step_m": summarize(steps.astype(float).tolist()),
        "mesh_extent_x_m": summarize(ext[:, 0].astype(float).tolist()) if len(ext) else {"count": 0},
        "mesh_extent_y_m": summarize(ext[:, 1].astype(float).tolist()) if len(ext) else {"count": 0},
        "mesh_extent_z_m": summarize(ext[:, 2].astype(float).tolist()) if len(ext) else {"count": 0},
        "hand_object_sampled_min_distance_m": summarize(hand_object_min_dist),
        "hand_mesh_distance_rows": int(hand_rows),
        "claim": "timeline contains contiguous camera states, MANO states from the source annotations, and mesh-backed object states for every active delivered-object frame",
    }
    write_json(output_dir / "v16_full_timeline_qc.json", report)
    return report


def visual_inspection_sheet(video: Path, output: Path, frames: list[int]) -> dict:
    cap, info = open_video(video)
    thumbs = []
    try:
        for frame_idx in frames:
            if frame_idx < 0 or frame_idx >= info.frame_count:
                continue
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
                raise RuntimeError(f"failed to seek {video} to frame {frame_idx}")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read {video} frame {frame_idx}")
            thumb_w = 480
            thumb_h = int(round(thumb_w * frame.shape[0] / frame.shape[1]))
            thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"frame {frame_idx}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(thumb, f"frame {frame_idx}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(thumb)
    finally:
        cap.release()
    if not thumbs:
        raise RuntimeError("no visual inspection frames selected")
    cols = min(3, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    h = max(thumb.shape[0] for thumb in thumbs)
    w = max(thumb.shape[1] for thumb in thumbs)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h : r * h + thumb.shape[0], c * w : c * w + thumb.shape[1]] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write {output}")
    return {"path": str(output), "sampled_frames": [int(x) for x in frames]}


def render_v16(args: argparse.Namespace, annotations_path: Path, mesh_archive: Path, output_dir: Path, info: VideoInfo) -> dict:
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(args.python),
            "scripts/fuse_v1_full_fidelity.py",
            "--clip",
            str(args.clip),
            "--output-dir",
            str(render_dir),
            "--render-only-annotations",
            str(annotations_path),
            "--object-mesh-npz",
            str(mesh_archive),
            "--render-width",
            str(args.render_width),
        ],
        args.repo_root,
    )
    overlay = render_dir / "overlay_mano_object.mp4"
    world = render_dir / "reconstruction_3d_world.mp4"
    side = render_dir / "side_by_side.mp4"
    return {
        "overlay": check_video(overlay, info),
        "world": check_video(world, info),
        "side_by_side": check_video(side, info),
        "render_dir": str(render_dir),
    }


def unresolved_vlm_tracks(object_plan_path: Path | None, delivered_label: str) -> list[dict]:
    if object_plan_path is None or not object_plan_path.exists():
        return []
    plan = load_json(object_plan_path).get("plan", {})
    rows = []
    for obj in plan.get("objects", []):
        track_id = str(obj.get("track_id", ""))
        if delivered_label and delivered_label in track_id:
            continue
        rows.append(
            {
                "track_id": track_id,
                "description": obj.get("description"),
                "active_intervals": obj.get("active_intervals", []),
                "v16_status": "not_delivered_in_first_v16_run",
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.repo_root = Path(args.repo_root).resolve()
    args.python = Path(args.python)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(args.clip)
    require_path(args.annotations, "full annotations")
    if args.droid_qc is not None:
        require_path(args.droid_qc, "DROID QC")
    require_path(args.wilor_qc, "WiLoR QC")
    if args.actions_json is not None:
        require_path(args.actions_json, "actions JSON")
    if args.object_plan is not None:
        require_path(args.object_plan, "object plan")
    raw_manifest = raw_frame_manifest(args.clip, args.output_dir, render_width=int(args.depth_manifest_width))
    annotation_camera_qc = None
    if args.droid_qc is None:
        annotation_camera_qc = build_annotation_camera_qc(args.annotations, args.clip, args.output_dir, info)
    depth_request = make_depth_manifest_from_masks(args.annotations, Path(raw_manifest["manifest"]), args.output_dir)
    if args.depth_npz is None:
        raise RuntimeError(
            "V16 requires full metric depth for masked object frames. "
            f"Run scripts/run_unidepth_metric_source_v3.py on {depth_request['manifest']} and pass --depth-npz."
        )
    depth_npz = require_path(args.depth_npz, "metric depth npz")
    mesh_report = build_full_mesh_stream(args, args.annotations, depth_npz, args.output_dir)
    ann_report = patch_annotations_with_mesh_state(args.annotations, mesh_report, args.output_dir)
    timeline_qc = build_timeline_qc(Path(ann_report["annotations"]), Path(mesh_report["archive"]), args.output_dir, info)
    render_report = render_v16(args, Path(ann_report["annotations"]), Path(mesh_report["archive"]), args.output_dir, info)
    sample_frames = sorted(
        set(
            [
                0,
                max(0, info.frame_count // 4),
                max(0, info.frame_count // 2),
                max(0, (3 * info.frame_count) // 4),
                info.frame_count - 1,
                int(mesh_report["first_mesh_frame"]),
                int(mesh_report["last_mesh_frame"]),
            ]
        )
    )
    inspection = visual_inspection_sheet(
        Path(render_report["side_by_side"]["path"]),
        args.output_dir / "visual_inspection" / "side_by_side_sheet.jpg",
        sample_frames,
    )
    raw_match = all(
        section["frame_count_match"]
        for section in (render_report["overlay"], render_report["world"], render_report["side_by_side"])
    )
    droid_qc = load_json(args.droid_qc) if args.droid_qc is not None else annotation_camera_qc["report"]
    wilor_qc = load_json(args.wilor_qc)
    unresolved_tracks = unresolved_vlm_tracks(args.object_plan, args.delivered_object_label or "")
    manifest = {
        "status": "ok" if raw_match else "failed",
        "method": "run_v16_full_pipeline",
        "clip": str(args.clip),
        "actions_json": str(args.actions_json) if args.actions_json else None,
        "raw_video": info.__dict__,
        "raw_frame_count": int(info.frame_count),
        "raw_fps": float(info.fps),
        "output_frame_count": int(render_report["side_by_side"]["frame_count"]),
        "output_fps": float(render_report["side_by_side"]["fps"]),
        "frame_count_match": bool(raw_match),
        "annotations": ann_report["annotations"],
        "object_mesh_archive": mesh_report["archive"],
        "overlay_video": render_report["overlay"]["path"],
        "world_video": render_report["world"]["path"],
        "side_by_side_video": render_report["side_by_side"]["path"],
        "camera_qc": str(args.droid_qc) if args.droid_qc is not None else annotation_camera_qc["path"],
        "hand_qc": str(args.wilor_qc),
        "object_mask_qc": str(args.object_mask_qc) if args.object_mask_qc else None,
        "object_mesh_qc": str(args.output_dir / "object_mesh_qc.json"),
        "timeline_qc": str(args.output_dir / "v16_full_timeline_qc.json"),
        "timeline_residual_qc": str(args.output_dir / "v16_full_timeline_qc.json"),
        "render_qc": render_report,
        "visual_inspection_sheet": inspection,
        "delivered_object_stream": {
            "label": args.delivered_object_label,
            "source": "full V1 object mask stream with V16 mesh-backed replacement",
            "measured_mesh_frames": int(mesh_report["measured_mesh_frames"]),
            "predicted_mesh_frames": int(mesh_report["predicted_mesh_frames"]),
            "mesh_frames": int(mesh_report["mesh_frames"]),
        },
        "unresolved_vlm_tracks": unresolved_tracks,
        "camera_summary": droid_qc.get("camera", droid_qc),
        "hand_summary": wilor_qc,
        "timeline_summary": timeline_qc,
        "failure_rows": [],
        "elapsed_s": float(time.time() - started),
    }
    write_json(args.output_dir / "v16_full_pipeline_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--actions-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-profile", default="manual")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-qc", type=Path)
    parser.add_argument("--wilor-qc", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path)
    parser.add_argument("--object-mask-qc", type=Path)
    parser.add_argument("--depth-npz", type=Path)
    parser.add_argument("--delivered-object-label", default="trash_bag")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(__file__).resolve().parents[1] / ".venv/bin/python")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--depth-manifest-width", type=int, default=960)
    parser.add_argument("--object-mesh-stride", type=int, default=8)
    parser.add_argument("--object-mesh-erode-px", type=int, default=1)
    parser.add_argument("--object-mesh-max-edge-m", type=float, default=0.050)
    parser.add_argument("--object-mesh-min-vertices", type=int, default=180)
    parser.add_argument("--object-mesh-min-faces", type=int, default=180)
    parser.add_argument("--object-mesh-min-depth-m", type=float, default=0.20)
    parser.add_argument("--object-mesh-max-depth-m", type=float, default=4.00)
    parser.add_argument("--object-mesh-depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--object-mesh-depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--object-mesh-prediction-max-gap", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
