#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ContractError(RuntimeError):
    pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "unnamed"


def numeric_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise ContractError(f"missing_{label}: {path}")
    return path


def optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    raw = str(path)
    if raw == "" or raw.lower() == "none":
        return None
    return Path(raw)


def decode_ho3d_depth_m(path: Path) -> np.ndarray:
    depth_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ContractError(f"could_not_read_depth: {path}")
    if depth_img.ndim == 2:
        raw = depth_img.astype(np.float32)
    elif depth_img.shape[2] >= 3:
        raw = depth_img[:, :, 2].astype(np.float32) + depth_img[:, :, 1].astype(np.float32) * 256.0
    else:
        raise ContractError(f"ho3d_depth_decode_failed_unexpected_channels: {path} shape={depth_img.shape}")
    return raw * 0.00012498664727900177


def read_depth_image_m(path: Path, semantics: str | None = None, scale: float | None = None) -> np.ndarray:
    semantics_l = (semantics or "").lower()
    if "ho3d" in semantics_l:
        return decode_ho3d_depth_m(path)
    depth_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ContractError(f"could_not_read_depth: {path}")
    raw_is_integer = depth_img.dtype.kind in {"u", "i"}
    depth = depth_img.astype(np.float32)
    if scale is not None:
        depth = depth * float(scale)
    elif "mm" in semantics_l or "millimeter" in semantics_l or raw_is_integer:
        depth = depth * 0.001
    return depth


def load_npz_depth(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    if "depth" not in blob.files:
        raise ContractError(f"depth_npz_missing_depth: {path}")
    depth = np.asarray(blob["depth"], dtype=np.float32)
    if depth.ndim != 3:
        raise ContractError(f"depth_npz_depth_must_be_NHW: {path} shape={depth.shape}")
    if "frame_idx" in blob.files:
        frame_idx = np.asarray(blob["frame_idx"], dtype=np.int64)
    else:
        frame_idx = np.arange(depth.shape[0], dtype=np.int64)
    if frame_idx.shape[0] != depth.shape[0]:
        raise ContractError(f"depth_npz_frame_count_mismatch: {path}")
    intr = None
    if "intrinsics_fx_fy_cx_cy" in blob.files:
        intr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    elif "intrinsics" in blob.files:
        raw = np.asarray(blob["intrinsics"], dtype=np.float64)
        if raw.ndim == 3 and raw.shape[1:] == (3, 3):
            intr = np.stack([raw[:, 0, 0], raw[:, 1, 1], raw[:, 0, 2], raw[:, 1, 2]], axis=1)
    source_size = np.asarray(blob["source_size"], dtype=float) if "source_size" in blob.files else None
    return {
        "path": str(path),
        "depth": depth,
        "frame_idx": frame_idx,
        "frame_to_i": {int(frame): int(i) for i, frame in enumerate(frame_idx)},
        "intrinsics_fx_fy_cx_cy": intr,
        "source_size": source_size,
    }


def load_depth_candidate_depth(candidate: dict[str, Any]) -> dict[str, Any]:
    path_raw = candidate.get("depth_npz") or candidate.get("depth_candidate_npz") or candidate.get("npz_path")
    if path_raw:
        return load_npz_depth(Path(path_raw))
    paths = candidate.get("depth_paths")
    frames = candidate.get("frame_indices")
    if isinstance(paths, list) and paths:
        depths = []
        frame_idx = []
        semantics = str(candidate.get("depth_semantics") or candidate.get("decode") or "")
        scale = candidate.get("depth_scale_to_m")
        for i, raw_path in enumerate(paths):
            depths.append(read_depth_image_m(Path(raw_path), semantics, float(scale) if scale is not None else None))
            frame_idx.append(int(frames[i]) if isinstance(frames, list) and i < len(frames) else i)
        shapes = {tuple(depth.shape) for depth in depths}
        if len(shapes) != 1:
            raise ContractError(f"depth_candidate_images_shape_mismatch: {path_raw or paths[:2]}")
        return {
            "path": None,
            "depth": np.stack(depths, axis=0).astype(np.float32),
            "frame_idx": np.asarray(frame_idx, dtype=np.int64),
            "frame_to_i": {int(frame): int(i) for i, frame in enumerate(frame_idx)},
            "intrinsics_fx_fy_cx_cy": None,
            "source_size": None,
        }
    raise ContractError(f"depth_candidate_lacks_depth_data: {candidate.get('candidate_id')}")


def load_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ContractError(f"could_not_read_mask: {path}")
    mask_bool = mask > 0
    if shape_hw is not None and mask_bool.shape != shape_hw:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask_bool


def sample_depth_at_points(depth: np.ndarray, points_xy: np.ndarray, source_size: Any | None = None) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ContractError(f"points_xy_must_be_Nx2: shape={pts.shape}")
    scaled = pts.copy()
    if source_size is not None:
        source = np.asarray(source_size, dtype=float).reshape(-1)
        if source.size >= 2 and source[0] > 0 and source[1] > 0:
            scaled[:, 0] *= depth.shape[1] / source[0]
            scaled[:, 1] *= depth.shape[0] / source[1]
    xs = np.clip(np.rint(scaled[:, 0]).astype(int), 0, depth.shape[1] - 1)
    ys = np.clip(np.rint(scaled[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return depth[ys, xs]


def finite_depth_values(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    return np.asarray(depth[valid], dtype=float)


def robust_abs_median(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(np.abs(arr)))


def load_registry_candidates(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise ContractError(f"registry_has_no_candidates_list: {path}")
    return [row for row in candidates if isinstance(row, dict)]


def transform_points(points: np.ndarray, matrix4: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    hom = np.concatenate([pts, np.ones((len(pts), 1), dtype=float)], axis=1)
    return (hom @ np.asarray(matrix4, dtype=float).T)[:, :3]


def project_points(points_camera: np.ndarray, intrinsics: Any, convention: str = "opencv_positive_z") -> np.ndarray:
    pts = np.asarray(points_camera, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ContractError(f"points_camera_must_be_Nx3: {pts.shape}")
    intr = np.asarray(intrinsics, dtype=float)
    if intr.shape == (3, 3):
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    elif intr.shape == (4,):
        fx, fy, cx, cy = intr
    else:
        raise ContractError(f"intrinsics_must_be_3x3_or_4: {intr.shape}")
    if convention == "opengl_negative_z":
        depth = -pts[:, 2]
        u = fx * pts[:, 0] / np.maximum(depth, 1.0e-9) + cx
        v = cy - fy * pts[:, 1] / np.maximum(depth, 1.0e-9)
    else:
        depth = pts[:, 2]
        u = fx * pts[:, 0] / np.maximum(depth, 1.0e-9) + cx
        v = fy * pts[:, 1] / np.maximum(depth, 1.0e-9) + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def depth_at_projected_points(depth: np.ndarray, points_camera: np.ndarray, intrinsics: Any, convention: str = "opencv_positive_z") -> tuple[np.ndarray, np.ndarray]:
    uv = project_points(points_camera, intrinsics, convention)
    sampled = sample_depth_at_points(depth, uv, [depth.shape[1], depth.shape[0]])
    model_depth = -points_camera[:, 2] if convention == "opengl_negative_z" else points_camera[:, 2]
    return sampled, model_depth


def mean_or_none(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def ensure_no_gt_in_prediction(payload: Any, path_label: str = "payload") -> None:
    forbidden = []

    def visit(value: Any, breadcrumb: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lower = str(key).lower()
                if lower.startswith("gt") or "ground_truth" in lower or "oracle" in lower or "reference_label" in lower or "reference-label" in lower:
                    forbidden.append(f"{breadcrumb}.{key}")
                visit(item, f"{breadcrumb}.{key}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                visit(item, f"{breadcrumb}[{idx}]")
        elif isinstance(value, str):
            lower = value.lower()
            if "/gt" in lower or "ground_truth" in lower or "oracle" in lower or "reference_label" in lower or "reference-label" in lower or "reference labels" in lower:
                forbidden.append(breadcrumb)

    visit(payload, path_label)
    if forbidden:
        raise ContractError(f"gt_forbidden_in_prediction_payload: {forbidden[:20]}")
