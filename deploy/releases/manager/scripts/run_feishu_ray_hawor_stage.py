#!/usr/bin/env python3
"""Feishu Ray HaWoR track/infiller adapter for the complete V22 DAG.

The service returns MANO parameters, not accepted surfaces. This adapter owns
source-grid crops, DROID/depth scale fusion, temporal joins, deterministic MANO
replay, and the legacy hawor_world_hands.npz publication consumed by D7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.adapt_droid_to_hawor import load_shared_geometry
from scripts.call_feishu_ray_service import ServiceCallerError, call_service_arrays
from scripts.run_feishu_ray_annotation_stage import (
    FeishuRayAdapterError,
    call_typed,
    decode_array,
    load_json_object,
    load_profile,
    make_ownership,
    ownership_matches,
    profile_base_url,
    require_success,
    resolve_manifest_rgb,
    sha256_file,
    utc_now,
    write_json,
)

TRACK_ROUTE = "/hawor.infer_tracks"
INFILLER_ROUTE = "/hawor_infiller.fill"
TRACK_MODEL_REVISION = "hawor-v1"
INFILLER_MODEL_REVISION = "hawor-infiller-v1"
CHUNK_SIZE = 16
INFILLER_WINDOW = 120
MANO_VERTICES = 778
MANO_JOINTS = 21

ServiceCall = Callable[..., dict[str, Any]]


def _error(code: str, message: str) -> FeishuRayAdapterError:
    return FeishuRayAdapterError(code, message)


@contextmanager
def hawor_root_cwd(hawor_root: Path) -> Iterator[None]:
    root = hawor_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HaWoR root is missing: {root}")
    previous = Path.cwd()
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(previous)


def _finite_float(value: Any, *, code: str, label: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(code, f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise _error(code, f"{label} is not finite/valid: {value!r}")
    return result


def _array_tuple(array: np.ndarray, dtype: str | None = None) -> tuple[bytes, tuple[int, ...], str]:
    value = np.ascontiguousarray(array, dtype=dtype)
    return value.tobytes(), tuple(int(x) for x in value.shape), value.dtype.name


def _service_droid_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_scale": _finite_float(value.get("metric_scale"), code="hawor_droid_evidence_invalid", label="metric_scale", positive=True),
        "scale_residual": _finite_float(value.get("scale_residual"), code="hawor_droid_evidence_invalid", label="scale_residual"),
        "scale_confidence": _finite_float(value.get("scale_confidence"), code="hawor_droid_evidence_invalid", label="scale_confidence"),
        "source": str(value.get("source") or ""),
    }


def _validate_rotation_matrices(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-2:] != (3, 3) or not np.isfinite(array).all():
        raise _error("hawor_rotation_invalid", f"{label} must contain finite 3x3 matrices, got {array.shape}")
    flat = array.reshape(-1, 3, 3)
    gram = np.matmul(np.swapaxes(flat, 1, 2), flat)
    det = np.linalg.det(flat)
    if not np.allclose(gram, np.eye(3), atol=2.0e-4, rtol=2.0e-4) or not np.allclose(det, 1.0, atol=2.0e-4, rtol=2.0e-4):
        raise _error("hawor_rotation_invalid", f"{label} contains non-proper rotations")
    return array.astype(np.float32)


def matrix_to_axis_angle(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices without requiring torch."""
    value = _validate_rotation_matrices(matrices, "rotation")
    flat = value.reshape(-1, 3, 3).astype(np.float64)
    result = np.zeros((len(flat), 3), dtype=np.float64)
    for index, matrix in enumerate(flat):
        cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle < 1.0e-7:
            result[index] = 0.5 * np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
            continue
        sine = math.sin(angle)
        if abs(sine) > 1.0e-6:
            axis = np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]) / (2.0 * sine)
        else:
            values, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
            axis = vectors[:, int(np.argmax(values))]
        if axis[0] < -1.0e-10 or (abs(axis[0]) <= 1.0e-10 and axis[1] < -1.0e-10) or (abs(axis[0]) <= 1.0e-10 and abs(axis[1]) <= 1.0e-10 and axis[2] < 0.0):
            axis = -axis
        result[index] = axis * angle
    return result.reshape(value.shape[:-2] + (3,)).astype(np.float32)


def quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise _error("droid_pose_invalid", f"quaternions have invalid shape/values: {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise _error("droid_pose_invalid", "DROID returned a zero quaternion")
    q = q / norm
    x, y, z, w = [q[..., i] for i in range(4)]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3)).astype(np.float32)


def estimate_metric_scale_from_depth_disparity(
    depth_m: np.ndarray,
    depth_frame_idx: np.ndarray,
    keyframe_idx: np.ndarray,
    disparities: np.ndarray,
    dynamic_masks: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Estimate scale in the required direction: metric depth * DROID disparity."""
    import cv2

    depth = np.asarray(depth_m, dtype=np.float64)
    frame_ids = np.asarray(depth_frame_idx, dtype=np.int64).reshape(-1)
    keyframes = np.asarray(keyframe_idx, dtype=np.int64).reshape(-1)
    disp = np.asarray(disparities, dtype=np.float64)
    masks = np.asarray(dynamic_masks, dtype=np.float64)
    if depth.ndim != 3 or len(frame_ids) != depth.shape[0] or disp.ndim != 3 or disp.shape[0] != len(keyframes):
        raise _error("hawor_scale_shape_invalid", f"depth/frame/disparity shapes are incompatible: {depth.shape}/{frame_ids.shape}/{disp.shape}")
    if masks.shape != depth.shape:
        raise _error("hawor_scale_shape_invalid", f"dynamic masks {masks.shape} do not match source depth {depth.shape}")
    by_frame = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    per_keyframe: list[dict[str, Any]] = []
    candidates: list[float] = []
    for position, frame in enumerate(keyframes.tolist()):
        if int(frame) not in by_frame:
            per_keyframe.append({"keyframe_position": position, "frame_idx": int(frame), "status": "missing_depth"})
            continue
        source_index = by_frame[int(frame)]
        h, w = disp[position].shape
        depth_grid = cv2.resize(depth[source_index].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        mask_grid = cv2.resize(masks[source_index].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
        valid = np.isfinite(depth_grid) & (depth_grid > 0.0) & np.isfinite(disp[position]) & (disp[position] > 0.0) & np.isfinite(mask_grid) & (mask_grid < 0.5)
        values = (depth_grid * disp[position])[valid]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            per_keyframe.append({"keyframe_position": position, "frame_idx": int(frame), "status": "no_static_finite_support"})
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        cutoff = max(3.0 * mad, 1.0e-6)
        robust = values[np.abs(values - median) <= cutoff]
        if robust.size == 0:
            robust = values
        estimate = float(np.median(robust))
        candidates.append(estimate)
        per_keyframe.append(
            {
                "keyframe_position": int(position),
                "frame_idx": int(frame),
                "status": "ok",
                "sample_count": int(values.size),
                "robust_sample_count": int(robust.size),
                "median_depth_times_disparity": median,
                "mad": mad,
                "scale": estimate,
                "residual_p95": float(np.percentile(np.abs(robust - estimate), 95)) if robust.size else None,
            }
        )
    if not candidates:
        raise _error("hawor_scale_no_support", "D3 depth and D4 disparity have no finite static support")
    candidate_array = np.asarray(candidates, dtype=np.float64)
    scale = float(np.median(candidate_array))
    residuals = np.abs(candidate_array - scale)
    residual = float(np.median(residuals))
    relative = residual / max(scale, 1.0e-12)
    confidence = float(np.clip((len(candidates) / max(1, len(keyframes))) * math.exp(-relative), 0.0, 1.0))
    report = {
        "status": "ok",
        "direction": "metric_scale = median(depth_m * disparity)",
        "scale": scale,
        "scale_residual": residual,
        "scale_confidence": confidence,
        "finite_keyframe_count": int(len(candidates)),
        "keyframe_count": int(len(keyframes)),
        "scale_statistics": {"min": float(np.min(candidate_array)), "median": scale, "max": float(np.max(candidate_array))},
        "per_keyframe": per_keyframe,
    }
    return scale, report


def _load_dynamic_masks(run_root: Path, frame_count: int, source_height: int, source_width: int, selected_frames: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    detector_path = run_root / "measurements" / "hand_detections" / "feishu_ray_hands" / "hands_detector_timeline.json"
    if not detector_path.is_file():
        raise _error("hawor_detector_missing", f"D6 detector timeline is required: {detector_path}")
    detector = load_json_object(detector_path)
    rows = detector.get("frames")
    archive_spec = detector.get("mask_archive")
    if detector.get("schema") != "ego.annotation.hands_detector_timeline.v1" or not isinstance(rows, list) or len(rows) != frame_count or not isinstance(archive_spec, Mapping):
        raise _error("hawor_detector_invalid", f"D6 detector timeline is malformed: {detector_path}")
    archive_path = Path(str(archive_spec.get("path"))).expanduser()
    if not archive_path.is_absolute():
        archive_path = (detector_path.parent / archive_path).resolve()
    if not archive_path.is_file() or sha256_file(archive_path) != str(archive_spec.get("sha256")):
        raise _error("hawor_detector_mask_invalid", f"D6 mask archive missing or hash mismatched: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"masks_packbits", "frame_idx", "detection_idx", "side", "source_size", "mask_bit_count", "packbits_bitorder"}
        if not required.issubset(set(archive.files)):
            raise _error("hawor_detector_mask_invalid", f"D6 mask archive lacks {sorted(required.difference(archive.files))}")
        packed = np.asarray(archive["masks_packbits"], dtype=np.uint8)
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64).reshape(-1)
        source_size = np.asarray(archive["source_size"], dtype=np.int64).reshape(-1)
        bitorder = str(np.asarray(archive["packbits_bitorder"]).reshape(-1)[0])
    if source_size.tolist() != [source_width, source_height] or bitorder != "little" or packed.ndim != 2 or frame_idx.shape != (packed.shape[0],):
        raise _error("hawor_detector_mask_invalid", f"D6 mask archive grid/encoding mismatch: {archive_path}")
    selected = np.asarray(selected_frames, dtype=np.int64).reshape(-1)
    selected_positions = {int(frame): position for position, frame in enumerate(selected.tolist())}
    if len(selected_positions) != len(selected) or np.any(selected < 0) or np.any(selected >= frame_count):
        raise _error("hawor_detector_mask_invalid", "selected DROID keyframes are invalid or duplicated")
    masks = np.zeros((len(selected), source_height, source_width), dtype=np.uint8)
    for index, frame in enumerate(frame_idx.tolist()):
        if frame < 0 or frame >= frame_count:
            raise _error("hawor_detector_mask_invalid", f"D6 mask archive frame index out of range: {frame}")
        selected_position = selected_positions.get(int(frame))
        if selected_position is None:
            continue
        decoded = np.unpackbits(packed[index], bitorder="little", count=source_width * source_height).reshape(source_height, source_width)
        masks[selected_position] = np.maximum(masks[selected_position], decoded.astype(np.uint8))
    return masks, {"path": str(archive_path), "sha256": sha256_file(archive_path), "source": str(detector_path), "source_sha256": sha256_file(detector_path), "decoded_keyframe_count": int(len(selected))}


def _load_camera_depth_and_geometry(run_root: Path, expected_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from scripts.adapt_droid_to_hawor import load_reconstruction

    shared_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json"
    if not shared_path.is_file():
        raise _error("hawor_shared_droid_missing", f"shared DROID manifest is required: {shared_path}")
    geometry = load_shared_geometry(shared_path, expected_frames=expected_frames)
    dense = np.load(geometry["dense_path"], allow_pickle=False)
    if "T_world_camera" in dense.files:
        raw_T = np.asarray(dense["T_world_camera"], dtype=np.float64)
    else:
        raw_T = np.tile(np.eye(4, dtype=np.float64)[None], (expected_frames, 1, 1))
        raw_T[:, :3, :3] = quaternion_xyzw_to_matrix(np.asarray(geometry["traj"][:, 3:], dtype=np.float64))
        raw_T[:, :3, 3] = np.asarray(geometry["traj"][:, :3], dtype=np.float64)
    if raw_T.shape != (expected_frames, 4, 4) or not np.isfinite(raw_T).all() or not np.allclose(raw_T[:, 3], [0, 0, 0, 1], atol=1.0e-5):
        raise _error("hawor_droid_pose_invalid", f"D4 T_world_camera has invalid shape/values: {raw_T.shape}")
    depth_path = run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"
    if not depth_path.is_file():
        raise _error("hawor_unidepth_missing", f"D3 depth archive is required: {depth_path}")
    with np.load(depth_path, allow_pickle=False) as depth_blob:
        required = {"depth", "frame_idx", "intrinsics_fx_fy_cx_cy", "source_size"}
        if not required.issubset(set(depth_blob.files)):
            raise _error("hawor_unidepth_invalid", f"D3 depth archive lacks {sorted(required.difference(depth_blob.files))}")
        depth_idx_all = np.asarray(depth_blob["frame_idx"], dtype=np.int64).reshape(-1)
        intrinsics = np.asarray(depth_blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        source_size = np.asarray(depth_blob["source_size"], dtype=np.int64).reshape(-1)
        depth_all = depth_blob["depth"]
        if depth_all.ndim != 3 or len(depth_idx_all) != depth_all.shape[0] or not np.array_equal(depth_idx_all, np.arange(expected_frames)):
            raise _error("hawor_unidepth_invalid", f"D3 depth archive does not cover contiguous source timeline: {depth_all.shape}/{depth_idx_all.shape}")
        keyframes = np.asarray(geometry["tstamp"], dtype=np.int64)
        depth = np.asarray(depth_all[keyframes], dtype=np.float32)
        depth_idx = keyframes.copy()
    if source_size.shape != (2,) or source_size.tolist() != [depth.shape[2], depth.shape[1]] or intrinsics.shape[0] != expected_frames or intrinsics.shape[1] != 4:
        raise _error("hawor_unidepth_invalid", "D3 depth archive intrinsics/source size disagree with depth")
    source_height, source_width = int(depth.shape[1]), int(depth.shape[2])
    masks, mask_provenance = _load_dynamic_masks(run_root, expected_frames, source_height, source_width, depth_idx)
    scale, scale_report = estimate_metric_scale_from_depth_disparity(depth, depth_idx, geometry["tstamp"], geometry["disps"], masks)
    metric_T = np.array(raw_T, copy=True)
    metric_T[:, :3, 3] *= float(scale)
    if not np.isfinite(metric_T).all():
        raise _error("hawor_droid_pose_invalid", "scaled DROID poses became nonfinite")
    K_values = intrinsics[0]
    K_px = np.asarray([[K_values[0], 0.0, K_values[2]], [0.0, K_values[1], K_values[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    timeline_manifest = load_json_object(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    timeline_fps = float(timeline_manifest.get("fps") or (timeline_manifest.get("video") or {}).get("fps") or 30.0)
    timestamps = np.asarray([float(row.get("time_s", int(row["frame_idx"]) / timeline_fps)) for row in timeline_manifest["frames"]], dtype=np.float64)
    evidence = {"metric_scale": float(scale), "scale_residual": float(scale_report["scale_residual"]), "scale_confidence": float(scale_report["scale_confidence"]), "source": str(depth_path), "scale_report": scale_report, "droid_manifest": str(shared_path), "droid_manifest_sha256": sha256_file(shared_path), "mask_provenance": mask_provenance}
    return metric_T, timestamps, depth, masks, K_px, {"geometry": geometry, "scale": scale, "scale_report": scale_report, "droid_evidence": evidence, "source_width": source_width, "source_height": source_height, "depth_path": depth_path}


def _detector_observations(run_root: Path, frames: list[dict[str, Any]], side: str) -> dict[int, dict[str, Any]]:
    detector_path = run_root / "measurements" / "hand_detections" / "feishu_ray_hands" / "hands_detector_timeline.json"
    detector = load_json_object(detector_path)
    rows = detector.get("frames")
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise _error("hawor_detector_invalid", f"D6 detector timeline frame count does not match source: {detector_path}")
    result: dict[int, dict[str, Any]] = {}
    for expected, row in zip(frames, rows):
        frame_idx = int(expected["frame_idx"])
        if not isinstance(row, Mapping) or int(row.get("frame_idx", -1)) != frame_idx or not isinstance(row.get("observations"), list):
            raise _error("hawor_detector_invalid", f"D6 detector row does not match source frame {frame_idx}")
        candidates = [item for item in row["observations"] if isinstance(item, Mapping) and str(item.get("side")) == side]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda item: float(item.get("score", 0.0)))
        box = np.asarray(chosen.get("bbox_xyxy_source"), dtype=np.float32).reshape(-1)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise _error("hawor_detector_invalid", f"D6 {side} box is invalid at frame {frame_idx}")
        result[frame_idx] = {"box": box, "score": _finite_float(chosen.get("score", 0.0), code="hawor_detector_invalid", label="detector score"), "visibility": _finite_float(chosen.get("visibility", 0.0), code="hawor_detector_invalid", label="detector visibility"), "uncertainty": _finite_float(chosen.get("uncertainty", 1.0), code="hawor_detector_invalid", label="detector uncertainty"), "occlusion_state": str(chosen.get("occlusion_state") or ("visible" if float(chosen.get("visibility", 0.0)) >= 0.5 else "partially_visible")), "track_id": str(chosen.get("track_id") or f"d6-{side}"), "detection_index": int(chosen.get("detection_index", -1))}
    return result


def _nearest_observation(observations: Mapping[int, dict[str, Any]], frame_idx: int) -> tuple[dict[str, Any] | None, bool]:
    if frame_idx in observations:
        return observations[frame_idx], True
    if not observations:
        return None, False
    nearest = min(observations, key=lambda item: (abs(int(item) - frame_idx), int(item)))
    return observations[nearest], False


def _crop_transform(center: np.ndarray, scale: float, *, width: int, height: int, do_flip: bool) -> dict[str, Any]:
    source_center = np.asarray(center, dtype=np.float64)
    sx = 256.0 / float(scale)
    if do_flip:
        source_to_crop = np.asarray(
            [[-sx, 0.0, 128.0 + sx * source_center[0]], [0.0, sx, 128.0 - sx * source_center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        source_to_crop = np.asarray(
            [[sx, 0.0, 128.0 - sx * source_center[0]], [0.0, sx, 128.0 - sx * source_center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return {
        "source_to_model": source_to_crop.tolist(),
        "model_to_source": np.linalg.inv(source_to_crop).tolist(),
        "resize_mode": "hawor_crop_flip" if do_flip else "hawor_crop",
        "crop_xywh": None,
        "pad_ltrb": None,
    }


def _load_track_dataset(hawor_root: Path) -> Any:
    root = hawor_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HaWoR root is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        with hawor_root_cwd(root):
            from lib.datasets.track_dataset import TrackDatasetEval  # type: ignore
    except Exception as exc:
        raise _error("hawor_preprocessor_import_failed", f"could not import TrackDatasetEval from {root}: {exc}") from exc
    return TrackDatasetEval


def _make_track_chunks(run_root: Path, repo_root: Path, frames: list[dict[str, Any]], side: str, observations: Mapping[int, dict[str, Any]], metric_T: np.ndarray, timestamps: np.ndarray, droid_evidence: Mapping[str, Any], K_px: np.ndarray, hawor_root: Path) -> list[dict[str, Any]]:
    if not observations:
        return []
    from PIL import Image

    dataset_class = _load_track_dataset(hawor_root)
    first_path = resolve_manifest_rgb(run_root, repo_root, str(frames[0]["rgb"]))
    with Image.open(first_path) as image:
        width, height = image.size
    img_focal = float(math.sqrt(max(K_px[0, 0], 1.0e-8) * max(K_px[1, 1], 1.0e-8)))
    img_center = [float(K_px[0, 2]), float(K_px[1, 2])]
    chunks: list[dict[str, Any]] = []
    do_flip = side == "left"
    for start in range(0, len(frames), CHUNK_SIZE):
        source_positions = list(range(start, min(start + CHUNK_SIZE, len(frames))))
        if not any(int(frames[position]["frame_idx"]) in observations for position in source_positions):
            continue
        real_length = len(source_positions)
        while len(source_positions) < CHUNK_SIZE:
            source_positions.append(source_positions[-1])
        boxes: list[np.ndarray] = []
        observed_flags: list[bool] = []
        crop_meta: list[dict[str, Any]] = []
        image_paths: list[str] = []
        for local_position, position in enumerate(source_positions):
            frame_idx = int(frames[position]["frame_idx"])
            observation, observed = _nearest_observation(observations, frame_idx)
            if observation is None:
                raise _error("hawor_missing_side_anchor", f"no crop anchor exists for {side} chunk starting {start}")
            box = np.asarray(observation["box"], dtype=np.float32)
            center = np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)
            scale = max(float(box[2] - box[0]), float(box[3] - box[1])) * 1.2
            if not math.isfinite(scale) or scale <= 0.0:
                raise _error("hawor_crop_invalid", f"invalid crop scale at {side} frame {frame_idx}")
            boxes.append(box)
            observed_flags.append(bool(observed and local_position < real_length))
            crop_meta.append({"center": center.astype(float).tolist(), "scale": float(scale), "img_focal": img_focal, "img_center": img_center, "do_flip": bool(do_flip), "source_size": {"width": width, "height": height}, "pixel_transform": _crop_transform(center, scale, width=width, height=height, do_flip=do_flip)})
            image_paths.append(str(resolve_manifest_rgb(run_root, repo_root, str(frames[position]["rgb"]))))
        crops: list[np.ndarray] = []
        transforms: list[dict[str, Any]] = []
        with hawor_root_cwd(hawor_root):
            dataset = dataset_class(image_paths, np.stack(boxes, axis=0), img_focal=img_focal, img_center=img_center, normalization=True, dilate=1.2, do_flip=do_flip)
            for local, item in enumerate(dataset):
                crop = np.asarray(item["img"].detach().cpu().numpy() if hasattr(item["img"], "detach") else item["img"], dtype=np.float32)
                if crop.shape != (3, 256, 256) or not np.isfinite(crop).all():
                    raise _error("hawor_crop_invalid", f"TrackDatasetEval returned invalid {side} crop {crop.shape}")
                crops.append(crop)
                center_value = np.asarray(item.get("center", np.asarray(crop_meta[local]["center"])), dtype=np.float32).reshape(2)
                scale_value = float(np.asarray(item.get("scale", crop_meta[local]["scale"])).reshape(-1)[0])
                transforms.append({**crop_meta[local], "center": center_value.astype(float).tolist(), "scale": scale_value})
        actual_indices = [int(frames[position]["frame_idx"]) for position in source_positions]
        chunks.append({"side": side, "start": int(start), "frame_indices": actual_indices, "observed_flags": observed_flags, "crop_batch": np.stack(crops, axis=0).astype(np.float32), "crop_transforms": transforms, "observations": [observations.get(frame_idx) for frame_idx in actual_indices], "img_focal": img_focal, "img_center": img_center, "source_size": {"width": width, "height": height}, "metric_T": metric_T[actual_indices], "timestamps": timestamps[actual_indices], "track_id": str(next(iter(observations.values())).get("track_id") or f"d6-{side}"), "K_px": K_px})
    return chunks


def _track_response(report: Mapping[str, Any], ownership: Mapping[str, Any], expected_length: int = CHUNK_SIZE) -> dict[str, np.ndarray]:
    _, result, arrays = require_success(dict(report), expected_ownership=ownership, route=TRACK_ROUTE)
    if not ownership_matches(ownership, result.get("ownership")) or result.get("model_revision") != TRACK_MODEL_REVISION:
        raise _error("hawor_response_contract_invalid", f"{TRACK_ROUTE}: result ownership/model revision mismatch")
    decoded: dict[str, np.ndarray] = {}
    decoded["root_orient"] = _validate_rotation_matrices(decode_array(arrays, "root_orient", shape=(expected_length, 3, 3), dtypes=("float32", "float64")), "root_orient")
    decoded["hand_pose"] = _validate_rotation_matrices(decode_array(arrays, "hand_pose", shape=(expected_length, 15, 3, 3), dtypes=("float32", "float64")), "hand_pose")
    decoded["trans"] = decode_array(arrays, "trans", shape=(expected_length, 3), dtypes=("float32", "float64"))
    decoded["betas"] = decode_array(arrays, "betas", shape=(expected_length, 10), dtypes=("float32", "float64"))
    decoded["joints"] = decode_array(arrays, "joints", dtypes=("float32", "float64"))
    if decoded["joints"].ndim != 3 or decoded["joints"].shape[0] != expected_length or decoded["joints"].shape[2] != 3 or decoded["joints"].shape[1] not in {16, 21}:
        raise _error(
            "hawor_response_shape_mismatch",
            f"{TRACK_ROUTE}: joints shape {decoded['joints'].shape}, expected ({expected_length},16|21,3)",
        )
    if "vertices" in arrays:
        placeholder = decode_array(arrays, "vertices", shape=(expected_length, MANO_VERTICES, 3), dtypes=("float32", "float64"))
        if not np.isfinite(placeholder).all():
            raise _error("hawor_response_nonfinite", f"{TRACK_ROUTE}: service vertices contain nonfinite values")
    decoded["observed"] = decode_array(arrays, "observed", shape=(expected_length,), dtypes=("bool", "uint8", "int8"))
    decoded["uncertainty"] = decode_array(arrays, "uncertainty", shape=(expected_length,), dtypes=("float32", "float64"))
    for name, value in decoded.items():
        if name not in {"observed"} and not np.isfinite(value).all():
            raise _error("hawor_response_nonfinite", f"{TRACK_ROUTE}: {name} contains nonfinite values")
    if np.any(decoded["uncertainty"] < 0.0):
        raise _error("hawor_response_invalid_uncertainty", f"{TRACK_ROUTE}: uncertainty is negative")
    return decoded


def _infiller_response(report: Mapping[str, Any], ownership: Mapping[str, Any], length: int) -> dict[str, np.ndarray]:
    _, result, arrays = require_success(dict(report), expected_ownership=ownership, route=INFILLER_ROUTE)
    if not ownership_matches(ownership, result.get("ownership")) or result.get("model_revision") != INFILLER_MODEL_REVISION:
        raise _error("hawor_response_contract_invalid", f"{INFILLER_ROUTE}: result ownership/model revision mismatch")
    decoded = {
        "root_orient": _validate_rotation_matrices(decode_array(arrays, "root_orient", shape=(2, length, 3, 3), dtypes=("float32", "float64")), "infiller root_orient"),
        "hand_pose": _validate_rotation_matrices(decode_array(arrays, "hand_pose", shape=(2, length, 15, 3, 3), dtypes=("float32", "float64")), "infiller hand_pose"),
        "trans": decode_array(arrays, "trans", shape=(2, length, 3), dtypes=("float32", "float64")),
        "betas": decode_array(arrays, "betas", shape=(2, length, 10), dtypes=("float32", "float64")),
        "observed": decode_array(arrays, "observed", shape=(2, length), dtypes=("bool", "uint8", "int8")),
        "inferred": decode_array(arrays, "inferred", shape=(2, length), dtypes=("bool", "uint8", "int8")),
        "uncertainty": decode_array(arrays, "uncertainty", shape=(2, length), dtypes=("float32", "float64")),
        "timestamps_s": decode_array(arrays, "timestamps_s", shape=(length,), dtypes=("float64", "float32")),
    }
    for name, value in decoded.items():
        if name not in {"observed", "inferred"} and not np.isfinite(value).all():
            raise _error("hawor_response_nonfinite", f"{INFILLER_ROUTE}: {name} contains nonfinite values")
    if np.any(decoded["uncertainty"] < 0.0):
        raise _error("hawor_response_invalid_uncertainty", f"{INFILLER_ROUTE}: uncertainty is negative")
    return decoded


def _load_mano_functions(hawor_root: Path) -> tuple[Any, Any, Any]:
    root = hawor_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        with hawor_root_cwd(root):
            from hawor.utils.process import get_mano_faces, run_mano, run_mano_left  # type: ignore
    except Exception as exc:
        raise _error("hawor_mano_assets_missing", f"could not import MANO assets from {root}: {exc}") from exc
    return get_mano_faces, run_mano, run_mano_left


def mano_asset_provenance(hawor_root: Path, *, required: bool) -> dict[str, Any]:
    root = hawor_root.expanduser().resolve()
    candidates = {
        "right": [root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl", root / "_DATA" / "data" / "mano" / "mano" / "MANO_RIGHT.pkl"],
        "left": [root / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl", root / "_DATA" / "data_left" / "mano_left" / "mano" / "MANO_LEFT.pkl"],
    }
    files: dict[str, dict[str, Any] | None] = {}
    for side, options in candidates.items():
        path = next((candidate for candidate in options if candidate.is_file()), None)
        if path is None:
            if required:
                raise _error("hawor_mano_assets_missing", f"{side} MANO asset is missing under {root / '_DATA'}")
            files[side] = None
        else:
            files[side] = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {"hawor_root": str(root), "hawor_root_exists": root.is_dir(), "files": files, "required": required}


def replay_mano_parameters(side: str, root_orient: np.ndarray, hand_pose: np.ndarray, trans: np.ndarray, betas: np.ndarray, hawor_root: Path, runner: Callable[..., Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay returned parameters and reject zero/degenerate surfaces."""
    root = _validate_rotation_matrices(root_orient, "MANO root_orient")
    pose = _validate_rotation_matrices(hand_pose, "MANO hand_pose")
    translation = np.asarray(trans, dtype=np.float32)
    shape = np.asarray(betas, dtype=np.float32)
    if translation.shape != (len(root), 3) or shape.shape != (len(root), 10) or not np.isfinite(translation).all() or not np.isfinite(shape).all():
        raise _error("hawor_mano_parameters_invalid", "MANO translation/betas shape or values are invalid")
    root_aa = matrix_to_axis_angle(root)
    pose_aa = matrix_to_axis_angle(pose).reshape(len(root), 45)
    if runner is not None:
        output = runner(side=side, root_orient_axis_angle=root_aa, hand_pose_axis_angle=pose_aa.reshape(len(root), 15, 3), trans=translation, betas=shape, hawor_root=hawor_root)
        if isinstance(output, Mapping):
            vertices = np.asarray(output["vertices"], dtype=np.float32)
            joints = np.asarray(output["joints"], dtype=np.float32)
            faces = np.asarray(output.get("faces", np.empty((0, 3), dtype=np.int32)), dtype=np.int32)
        else:
            vertices, joints = output[:2]
            faces = np.asarray(output[2], dtype=np.int32) if len(output) > 2 else np.empty((0, 3), dtype=np.int32)
    else:
        get_faces, run_right, run_left = _load_mano_functions(hawor_root)
        try:
            import torch
        except ImportError as exc:
            raise _error("hawor_mano_runtime_missing", "torch is required for deterministic MANO replay") from exc
        trans_t = torch.from_numpy(translation[None])
        root_t = torch.from_numpy(root_aa[None])
        pose_t = torch.from_numpy(pose_aa.reshape(len(root), 15, 3)[None])
        betas_t = torch.from_numpy(shape[None])
        previous_cwd = Path.cwd()
        try:
            os.chdir(hawor_root.expanduser().resolve())
            output = (run_left if side == "left" else run_right)(trans_t, root_t, pose_t, betas=betas_t, use_cuda=False)
            faces = np.asarray(get_faces(), dtype=np.int32)
        finally:
            os.chdir(previous_cwd)
        vertices = np.asarray(output["vertices"].detach().cpu().numpy() if hasattr(output["vertices"], "detach") else output["vertices"], dtype=np.float32).reshape(len(root), -1, 3)
        joints = np.asarray(output["joints"].detach().cpu().numpy() if hasattr(output["joints"], "detach") else output["joints"], dtype=np.float32).reshape(len(root), -1, 3)
    if vertices.shape != (len(root), MANO_VERTICES, 3) or joints.shape != (len(root), MANO_JOINTS, 3) or not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise _error("hawor_mano_replay_invalid", f"MANO replay returned invalid surface/joint shapes: {vertices.shape}/{joints.shape}")
    extents = np.ptp(vertices, axis=1)
    if np.any(np.max(extents, axis=1) <= 1.0e-7) or np.any(np.linalg.norm(vertices - vertices[:, :1], axis=2).max(axis=1) <= 1.0e-7):
        raise _error("hawor_mano_zero_vertices", "MANO replay produced zero or degenerate 778-vertex surfaces")
    if faces.size == 0:
        get_faces, _, _ = _load_mano_functions(hawor_root)
        previous_cwd = Path.cwd()
        try:
            os.chdir(hawor_root.expanduser().resolve())
            faces = np.asarray(get_faces(), dtype=np.int32)
        finally:
            os.chdir(previous_cwd)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise _error("hawor_mano_faces_invalid", f"MANO faces have invalid shape: {faces.shape}")
    if side == "left":
        faces = faces[:, [0, 2, 1]]
    return vertices, joints, faces


# Alias kept explicit for focused tests and callers.
replay_mano = replay_mano_parameters


def validate_legacy_hawor_archive(path: Path, frame_count: int) -> dict[str, Any]:
    required = {"frame_idx", "R_c2w", "t_c2w"}
    for side in ("left", "right"):
        required.update({f"{side}_vertices_world_m", f"{side}_joints_world_m", f"{side}_trans_world_m", f"{side}_root_orient_axis_angle", f"{side}_hand_pose_axis_angle", f"{side}_betas", f"{side}_valid", f"{side}_detected_same_frame", f"{side}_det_box_xyxyscore", f"{side}_track_id", f"{side}_faces"})
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise _error("hawor_output_schema_invalid", f"legacy HaWoR archive lacks {missing}")
        frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
        R_c2w = np.asarray(archive["R_c2w"], dtype=np.float64)
        t_c2w = np.asarray(archive["t_c2w"], dtype=np.float64)
        if not np.array_equal(frame_idx, np.arange(frame_count)) or R_c2w.shape != (frame_count, 3, 3) or t_c2w.shape != (frame_count, 3) or not np.isfinite(t_c2w).all():
            raise _error("hawor_output_schema_invalid", "legacy HaWoR camera timeline is incomplete or invalid")
        _validate_rotation_matrices(R_c2w, "legacy R_c2w")
        valid_counts: dict[str, int] = {}
        for side in ("left", "right"):
            vertices = np.asarray(archive[f"{side}_vertices_world_m"], dtype=np.float64)
            joints = np.asarray(archive[f"{side}_joints_world_m"], dtype=np.float64)
            trans = np.asarray(archive[f"{side}_trans_world_m"], dtype=np.float64)
            root = np.asarray(archive[f"{side}_root_orient_axis_angle"], dtype=np.float64)
            pose = np.asarray(archive[f"{side}_hand_pose_axis_angle"], dtype=np.float64)
            betas = np.asarray(archive[f"{side}_betas"], dtype=np.float64)
            valid = np.asarray(archive[f"{side}_valid"]).astype(bool)
            boxes = np.asarray(archive[f"{side}_det_box_xyxyscore"], dtype=np.float64)
            track_id = np.asarray(archive[f"{side}_track_id"])
            faces = np.asarray(archive[f"{side}_faces"], dtype=np.int64)
            expected_shapes = (vertices.shape == (frame_count, MANO_VERTICES, 3) and joints.shape == (frame_count, MANO_JOINTS, 3) and trans.shape == (frame_count, 3) and root.shape == (frame_count, 3) and pose.shape == (frame_count, 45) and betas.shape == (frame_count, 10) and valid.shape == (frame_count,) and boxes.shape == (frame_count, 5) and track_id.shape == (frame_count,) and faces.ndim == 2 and faces.shape[1] == 3)
            if not expected_shapes or faces.size == 0 or np.any(faces < 0) or np.any(faces >= MANO_VERTICES):
                raise _error("hawor_output_schema_invalid", f"legacy HaWoR {side} array shapes/faces are invalid")
            if np.any(valid):
                if not (np.isfinite(vertices[valid]).all() and np.isfinite(joints[valid]).all() and np.isfinite(trans[valid]).all() and np.isfinite(root[valid]).all() and np.isfinite(pose[valid]).all() and np.isfinite(betas[valid]).all()):
                    raise _error("hawor_output_schema_invalid", f"legacy HaWoR {side} valid rows are nonfinite")
                if np.any(np.max(np.ptp(vertices[valid], axis=1), axis=1) <= 1.0e-7):
                    raise _error("hawor_mano_zero_vertices", f"legacy HaWoR {side} contains a degenerate valid surface")
            valid_counts[side] = int(np.count_nonzero(valid))
    return {"status": "ok", "frame_count": frame_count, "valid_counts": valid_counts, "required_keys": sorted(required)}


def _default_timeline(frame_count: int) -> dict[str, dict[str, np.ndarray]]:
    return {side: {"root_orient": np.tile(np.eye(3, dtype=np.float32)[None], (frame_count, 1, 1)), "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None], (frame_count, 15, 1, 1)), "trans": np.zeros((frame_count, 3), dtype=np.float32), "betas": np.zeros((frame_count, 10), dtype=np.float32), "observed": np.zeros(frame_count, dtype=np.bool_), "inferred": np.zeros(frame_count, dtype=np.bool_), "uncertainty": np.ones(frame_count, dtype=np.float32)} for side in ("left", "right")}


def _compose_world(root_camera: np.ndarray, pose_camera: np.ndarray, trans_camera: np.ndarray, camera_T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_R = camera_T[:, :3, :3]
    world_root = np.einsum("nij,njk->nik", camera_R, root_camera)
    world_pose = np.asarray(pose_camera, dtype=np.float32)
    world_trans = np.einsum("nij,nj->ni", camera_R, trans_camera) + camera_T[:, :3, 3]
    return world_root.astype(np.float32), world_pose.astype(np.float32), world_trans.astype(np.float32)


def _merge_state(target: dict[str, np.ndarray], indices: list[int], state: dict[str, np.ndarray], observed: np.ndarray, inferred: np.ndarray, uncertainty: np.ndarray, source_observed: list[bool] | None = None) -> None:
    for local, frame_idx in enumerate(indices):
        if frame_idx < 0 or frame_idx >= len(target["observed"]):
            continue
        if source_observed is not None and not source_observed[local]:
            continue
        is_valid = bool(observed[local] or inferred[local])
        if not is_valid:
            continue
        score = float(uncertainty[local])
        if not math.isfinite(score) or score < 0.0:
            continue
        if target["observed"][frame_idx] or target["inferred"][frame_idx]:
            if score >= float(target["uncertainty"][frame_idx]):
                continue
        target["root_orient"][frame_idx] = state["root_orient"][local]
        target["hand_pose"][frame_idx] = state["hand_pose"][local]
        target["trans"][frame_idx] = state["trans"][local]
        target["betas"][frame_idx] = state["betas"][local]
        target["observed"][frame_idx] = bool(observed[local])
        target["inferred"][frame_idx] = bool(inferred[local])
        target["uncertainty"][frame_idx] = score


def _make_infiller_frames(timeline: Mapping[str, Mapping[str, np.ndarray]], frames: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in indices:
        frame_idx = int(frames[position]["frame_idx"])
        timestamp = float(frames[position].get("time_s", frame_idx / 30.0))
        for side in ("left", "right"):
            state = timeline[side]
            observed = bool(state["observed"][frame_idx])
            rows.append({"frame_index": frame_idx, "source_timestamp_s": timestamp, "side": side, "root_orient": state["root_orient"][frame_idx].astype(float).tolist(), "hand_pose": matrix_to_axis_angle(state["hand_pose"][frame_idx]).astype(float).tolist(), "trans": state["trans"][frame_idx].astype(float).tolist(), "betas": state["betas"][frame_idx].astype(float).tolist(), "observed": observed, "uncertainty": float(state["uncertainty"][frame_idx]) if observed else 1.0})
    return rows


def _preserve_invalid_response(run_root: Path, route: str, ownership: Mapping[str, Any], report: Mapping[str, Any], error: Exception) -> Path:
    request_id = str(ownership.get("request_id") or f"failure-{time.time_ns()}")
    failure_dir = run_root / "failures" / "feishu_ray_hawor" / request_id
    failure_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = failure_dir / "response_metadata.json"
    metadata_path.write_text(json.dumps(report.get("metadata"), indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    artifacts: list[dict[str, Any]] = []
    rows = report.get("arrays")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("data"), (bytes, bytearray, memoryview)):
                continue
            name = str(row.get("name") or f"array-{index}")
            safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in name)
            raw_path = failure_dir / f"{index:02d}_{safe}.bin"
            raw_path.write_bytes(bytes(row["data"]))
            artifacts.append({"name": name, "path": str(raw_path), "sha256": sha256_file(raw_path), "shape": list(row.get("shape", ())), "dtype": row.get("dtype")})
    evidence_path = failure_dir / "invalid_response.json"
    write_json(evidence_path, {"schema": "ego.annotation.feishu_ray_hawor_failure.v1", "status": "failed_response_validation", "route": route, "ownership": dict(ownership), "error": {"type": type(error).__name__, "code": getattr(error, "code", "unexpected"), "message": str(error)}, "http_status": report.get("http_status"), "content_type": report.get("content_type"), "response_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)}, "typed_arrays": artifacts, "successful_hawor_artifacts_published": False})
    try:
        setattr(error, "request_id", request_id)
        setattr(error, "failure_path", str(evidence_path))
    except Exception:
        pass
    return evidence_path


def _write_failure(run_root: Path, error: Exception, stage: str = "hawor") -> Path:
    request_id = getattr(error, "request_id", None) or f"failure-{time.time_ns()}"
    failure_dir = run_root / "failures" / "feishu_ray_hawor" / str(request_id)
    failure_dir.mkdir(parents=True, exist_ok=True)
    raw_response = getattr(error, "raw_response_bytes", None)
    raw_artifact = None
    if isinstance(raw_response, (bytes, bytearray, memoryview)):
        raw_path = failure_dir / "raw_response.bin"
        raw_path.write_bytes(bytes(raw_response))
        raw_artifact = {"path": str(raw_path), "sha256": sha256_file(raw_path), "size_bytes": raw_path.stat().st_size}
    headers = getattr(error, "response_headers", None)
    headers_artifact = None
    if isinstance(headers, Mapping):
        headers_path = failure_dir / "response_headers.json"
        write_json(headers_path, dict(headers))
        headers_artifact = {"path": str(headers_path), "sha256": sha256_file(headers_path)}
    path = failure_dir / "failure.json"
    write_json(path, {"schema": "ego.annotation.feishu_ray_hawor_failure.v1", "status": "failed", "stage": stage, "error": {"type": type(error).__name__, "code": getattr(error, "code", "unexpected"), "message": str(error)}, "response_received": getattr(error, "response_received", None), "response_status": getattr(error, "response_status", None), "response_headers": headers_artifact, "raw_response": raw_artifact, "successful_hawor_artifacts_published": False})
    return path


def run_hawor(args: argparse.Namespace, *, caller: ServiceCall = call_service_arrays, mano_runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    started = time.time()
    try:
        manifest = load_json_object(run_root / "input" / "raw_frame_manifest" / "manifest.json")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise _error("hawor_source_timeline_invalid", "raw frame manifest has no frames")
        fps = float(manifest.get("fps") or (manifest.get("video") or {}).get("fps") or 30.0)
        if not math.isfinite(fps) or fps <= 0.0:
            raise _error("hawor_source_timeline_invalid", f"raw frame fps is invalid: {fps!r}")
        previous_timestamp = -float("inf")
        for index, row in enumerate(frames):
            if not isinstance(row, Mapping) or int(row.get("frame_idx", -1)) != index:
                raise _error("hawor_source_timeline_invalid", f"raw frame index {index} is not contiguous")
            timestamp = _finite_float(row.get("time_s", index / fps), code="hawor_source_timeline_invalid", label=f"frame {index} timestamp")
            if timestamp <= previous_timestamp:
                raise _error("hawor_source_timeline_invalid", f"raw frame timestamp is not strictly increasing at frame {index}")
            previous_timestamp = timestamp
            if row.get("rgb") is None:
                raise _error("hawor_source_timeline_invalid", f"raw frame {index} lacks rgb reference")
            try:
                resolve_manifest_rgb(run_root, repo_root, str(row["rgb"]))
            except FileNotFoundError as exc:
                raise _error("hawor_source_timeline_invalid", str(exc)) from exc
        frame_count = len(frames)
        profile = load_profile(args.profile.expanduser().resolve())
        base_url = profile_base_url(profile, "hawor", args.base_url)
        profile_row = profile.get("services", {}).get("hawor") if isinstance(profile.get("services"), Mapping) else None
        if not isinstance(profile_row, Mapping) or profile_row.get("routes") != [TRACK_ROUTE, INFILLER_ROUTE]:
            raise _error("invalid_service_profile", "HaWoR profile must pin track and infiller routes")
        job_id = str(args.job_id or manifest.get("case_id") or run_root.name)
        metric_T, timestamps, depth, dynamic_masks, K_px, geometry_info = _load_camera_depth_and_geometry(run_root, frame_count)
        for row, timestamp in zip(frames, timestamps.tolist()):
            if isinstance(row, dict):
                row.setdefault("time_s", float(timestamp))
        timeline = _default_timeline(frame_count)
        detector_by_side = {side: _detector_observations(run_root, frames, side) for side in ("left", "right")}
        chunks_by_side = {side: _make_track_chunks(run_root, repo_root, frames, side, detector_by_side[side], metric_T, timestamps, geometry_info["droid_evidence"], K_px, args.hawor_root) for side in ("left", "right")}
        service_calls = {TRACK_ROUTE: 0, INFILLER_ROUTE: 0}
        retry_events: list[dict[str, Any]] = []
        chunk_reports: list[dict[str, Any]] = []
        for side in ("left", "right"):
            for chunk in chunks_by_side[side]:
                start_frame = int(chunk["start"])
                ownership = make_ownership(job_id=job_id, item_id=f"{job_id}-{side}-{start_frame:06d}", stage_id=TRACK_ROUTE.lstrip("/"), source_id=f"{side}-chunk-{start_frame:06d}", source_timestamp_s=float(chunk["timestamps"][0]))
                observations: list[dict[str, Any]] = []
                for local, frame_idx in enumerate(chunk["frame_indices"]):
                    source = chunk["observations"][local]
                    observed = bool(chunk["observed_flags"][local] and source is not None)
                    observations.append({"frame_index": frame_idx, "source_timestamp_s": float(chunk["timestamps"][local]), "occlusion_state": str(source.get("occlusion_state", "unresolved")) if observed else "unresolved", "detection_confidence": float(source.get("score", 0.0)) if observed else 0.0, "side": side})
                metadata = {"ownership": ownership, "track_id": str(chunk["track_id"]), "side": side, "crop_transforms": chunk["crop_transforms"], "observations": observations, "unidepth": {"K_px": K_px.astype(float).tolist(), "img_focal": float(chunk["img_focal"]), "img_center": [float(x) for x in chunk["img_center"]], "source_size": chunk["source_size"], "metric_scale": float(geometry_info["scale"]), "source": str(geometry_info["depth_path"])}, "droid_evidence": _service_droid_evidence(geometry_info["droid_evidence"]), "model_revision": TRACK_MODEL_REVISION}
                report = call_typed(caller, base_url=base_url, route=TRACK_ROUTE, metadata=metadata, arrays={"droid_poses": _array_tuple(chunk["metric_T"], "float32"), "droid_timestamps": _array_tuple(chunk["timestamps"], "float64"), "crop_batch": _array_tuple(chunk["crop_batch"], "float32")}, timeout_s=float(args.timeout_s), retry_events=retry_events, retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)), retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)))
                service_calls[TRACK_ROUTE] += 1
                try:
                    decoded = _track_response(report, ownership)
                except Exception as exc:
                    _preserve_invalid_response(run_root, TRACK_ROUTE, ownership, report, exc)
                    raise
                # Keep the timeline in the service's metric camera frame. The
                # infiller consumes this same frame and applies the DROID lift
                # internally; world lifting happens once, after MANO replay.
                state = {"root_orient": decoded["root_orient"], "hand_pose": decoded["hand_pose"], "trans": decoded["trans"], "betas": decoded["betas"]}
                _merge_state(timeline[side], chunk["frame_indices"], state, decoded["observed"].astype(bool), np.zeros(CHUNK_SIZE, dtype=np.bool_), decoded["uncertainty"], chunk["observed_flags"])
                chunk_reports.append({"side": side, "start_frame": start_frame, "ownership": ownership, "observed_count": int(np.count_nonzero(decoded["observed"])), "padded_count": int(CHUNK_SIZE - sum(chunk["observed_flags"])), "service_vertices_used": False})
        infiller_reports: list[dict[str, Any]] = []
        skipped_windows: list[dict[str, Any]] = []
        if frame_count <= INFILLER_WINDOW:
            infiller_starts = [0]
        else:
            infiller_starts = sorted(set(range(0, frame_count - INFILLER_WINDOW + 1, 90)) | {frame_count - INFILLER_WINDOW})
        for start in infiller_starts:
            indices = list(range(start, min(start + INFILLER_WINDOW, frame_count)))
            left_anchor = any(bool(timeline["left"]["observed"][idx]) for idx in indices)
            right_anchor = any(bool(timeline["right"]["observed"][idx]) for idx in indices)
            if not (left_anchor and right_anchor):
                skipped_windows.append({"start_frame": start, "end_frame": indices[-1], "reason": "both_side_anchors_required", "left_anchor": left_anchor, "right_anchor": right_anchor})
                continue
            ownership = make_ownership(job_id=job_id, item_id=f"{job_id}-window-{start:06d}", stage_id=INFILLER_ROUTE.lstrip("/"), source_id=f"window-{start:06d}", source_timestamp_s=float(timestamps[indices[0]]))
            metadata = {"ownership": ownership, "window_id": f"window-{start:06d}-{indices[-1]:06d}", "frames": _make_infiller_frames(timeline, frames, indices), "droid_evidence": _service_droid_evidence(geometry_info["droid_evidence"]), "unidepth": {"K_px": K_px.astype(float).tolist(), "img_focal": float(math.sqrt(K_px[0, 0] * K_px[1, 1])), "img_center": [float(K_px[0, 2]), float(K_px[1, 2])], "source_size": {"width": int(depth.shape[2]), "height": int(depth.shape[1])}, "metric_scale": float(geometry_info["scale"]), "source": str(geometry_info["depth_path"])}, "model_revision": INFILLER_MODEL_REVISION}
            T = metric_T[indices]
            report = call_typed(caller, base_url=base_url, route=INFILLER_ROUTE, metadata=metadata, arrays={"droid_poses": _array_tuple(T, "float32"), "droid_timestamps": _array_tuple(timestamps[indices], "float64")}, timeout_s=float(args.timeout_s), retry_events=retry_events, retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)), retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)))
            service_calls[INFILLER_ROUTE] += 1
            try:
                decoded = _infiller_response(report, ownership, len(indices))
            except FeishuRayAdapterError as exc:
                failure_path = _preserve_invalid_response(run_root, INFILLER_ROUTE, ownership, report, exc)
                skipped_windows.append({
                    "window_id": metadata["window_id"],
                    "start_frame": int(start),
                    "end_frame": int(indices[-1]),
                    "reason": "invalid_service_response",
                    "error_code": exc.code,
                    "error": str(exc),
                    "failure_path": str(failure_path),
                })
                continue
            if not np.allclose(decoded["timestamps_s"], timestamps[indices], atol=1.0e-7, rtol=0.0):
                raise _error("hawor_infiller_timeline_mismatch", f"{INFILLER_ROUTE}: response timestamps disagree with source window")
            for side_index, side in enumerate(("left", "right")):
                state = {"root_orient": decoded["root_orient"][side_index], "hand_pose": decoded["hand_pose"][side_index], "trans": decoded["trans"][side_index], "betas": decoded["betas"][side_index]}
                _merge_state(timeline[side], [int(frames[idx]["frame_idx"]) for idx in indices], state, decoded["observed"][side_index].astype(bool), decoded["inferred"][side_index].astype(bool), decoded["uncertainty"][side_index])
            infiller_reports.append({"window_id": metadata["window_id"], "start_frame": start, "frame_count": len(indices), "ownership": ownership, "service_vertices_used": False})
        if not any(bool(np.any(timeline[side]["observed"] | timeline[side]["inferred"])) for side in ("left", "right")):
            raise _error("hawor_no_hand_evidence", "neither hand side has an accepted HaWoR anchor")
        output_dir = run_root / "measurements" / "hand_candidates" / "hawor_world"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_npz = output_dir / "hawor_world_hands.npz"
        if output_npz.exists():
            raise _error("hawor_output_not_fresh", f"refusing to overwrite existing HaWoR archive: {output_npz}")
        mano_assets = mano_asset_provenance(args.hawor_root, required=mano_runner is None)
        get_faces = None
        hands: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            valid = timeline[side]["observed"] | timeline[side]["inferred"]
            vertices = np.full((frame_count, MANO_VERTICES, 3), np.nan, dtype=np.float32)
            joints = np.full((frame_count, MANO_JOINTS, 3), np.nan, dtype=np.float32)
            world_trans = np.full((frame_count, 3), np.nan, dtype=np.float32)
            world_root = np.tile(np.eye(3, dtype=np.float32)[None], (frame_count, 1, 1))
            world_pose = np.asarray(timeline[side]["hand_pose"], dtype=np.float32).copy()
            if np.any(valid):
                valid_idx = np.flatnonzero(valid)
                camera_vertices, camera_joints, faces = replay_mano_parameters(
                    side,
                    timeline[side]["root_orient"][valid_idx],
                    timeline[side]["hand_pose"][valid_idx],
                    timeline[side]["trans"][valid_idx],
                    timeline[side]["betas"][valid_idx],
                    args.hawor_root,
                    runner=mano_runner,
                )
                camera_R = metric_T[valid_idx, :3, :3]
                camera_t = metric_T[valid_idx, :3, 3]
                vertices[valid_idx] = np.einsum("nij,nvj->nvi", camera_R, camera_vertices) + camera_t[:, None, :]
                joints[valid_idx] = np.einsum("nij,nvj->nvi", camera_R, camera_joints) + camera_t[:, None, :]
                world_root[valid_idx], world_pose[valid_idx], world_trans[valid_idx] = _compose_world(
                    timeline[side]["root_orient"][valid_idx],
                    timeline[side]["hand_pose"][valid_idx],
                    timeline[side]["trans"][valid_idx],
                    metric_T[valid_idx],
                )
            else:
                if mano_runner is None:
                    get_faces, _, _ = _load_mano_functions(args.hawor_root)
                    previous_cwd = Path.cwd()
                    try:
                        os.chdir(args.hawor_root.expanduser().resolve())
                        faces = np.asarray(get_faces(), dtype=np.int32)
                    finally:
                        os.chdir(previous_cwd)
                    if side == "left":
                        faces = faces[:, [0, 2, 1]]
                else:
                    faces = np.empty((0, 3), dtype=np.int32)
            hands[side] = {"vertices": vertices, "joints": joints, "trans": world_trans, "root_axis": matrix_to_axis_angle(world_root), "pose_axis": matrix_to_axis_angle(world_pose).reshape(frame_count, 45), "betas": timeline[side]["betas"].astype(np.float32), "valid": valid.astype(np.uint8), "observed": timeline[side]["observed"].astype(np.uint8), "inferred": timeline[side]["inferred"].astype(np.uint8), "uncertainty": timeline[side]["uncertainty"].astype(np.float32), "faces": faces.astype(np.int32)}
        if hands["left"]["faces"].size == 0 and hands["right"]["faces"].size:
            hands["left"]["faces"] = hands["right"]["faces"][:, [0, 2, 1]]
        if hands["right"]["faces"].size == 0 and hands["left"]["faces"].size:
            hands["right"]["faces"] = hands["left"]["faces"][:, [0, 2, 1]]
        if hands["left"]["faces"].size == 0 or hands["right"]["faces"].size == 0:
            raise _error("hawor_mano_faces_invalid", "MANO replay produced no canonical faces for publication")
        frame_idx = np.arange(frame_count, dtype=np.int32)
        staging_npz = output_dir / ".hawor_world_hands.staging.npz"
        try:
            np.savez_compressed(staging_npz, frame_idx=frame_idx, R_c2w=metric_T[:, :3, :3].astype(np.float32), t_c2w=metric_T[:, :3, 3].astype(np.float32), left_vertices_world_m=hands["left"]["vertices"], left_joints_world_m=hands["left"]["joints"], left_trans_world_m=hands["left"]["trans"], left_root_orient_axis_angle=hands["left"]["root_axis"], left_hand_pose_axis_angle=hands["left"]["pose_axis"], left_betas=hands["left"]["betas"], left_valid=hands["left"]["valid"], left_detected_same_frame=np.asarray([int(i in detector_by_side["left"]) for i in frame_idx], dtype=np.uint8), left_det_box_xyxyscore=np.asarray([np.r_[detector_by_side["left"][int(i)]["box"], detector_by_side["left"][int(i)]["score"]] if int(i) in detector_by_side["left"] else [np.nan] * 5 for i in frame_idx], dtype=np.float32), left_track_id=np.asarray([detector_by_side["left"].get(int(i), {}).get("track_id", "") for i in frame_idx], dtype="<U64"), left_faces=hands["left"]["faces"], right_vertices_world_m=hands["right"]["vertices"], right_joints_world_m=hands["right"]["joints"], right_trans_world_m=hands["right"]["trans"], right_root_orient_axis_angle=hands["right"]["root_axis"], right_hand_pose_axis_angle=hands["right"]["pose_axis"], right_betas=hands["right"]["betas"], right_valid=hands["right"]["valid"], right_detected_same_frame=np.asarray([int(i in detector_by_side["right"]) for i in frame_idx], dtype=np.uint8), right_det_box_xyxyscore=np.asarray([np.r_[detector_by_side["right"][int(i)]["box"], detector_by_side["right"][int(i)]["score"]] if int(i) in detector_by_side["right"] else [np.nan] * 5 for i in frame_idx], dtype=np.float32), right_track_id=np.asarray([detector_by_side["right"].get(int(i), {}).get("track_id", "") for i in frame_idx], dtype="<U64"), right_faces=hands["right"]["faces"], left_observed=hands["left"]["observed"], right_observed=hands["right"]["observed"], left_inferred=hands["left"]["inferred"], right_inferred=hands["right"]["inferred"], left_uncertainty=hands["left"]["uncertainty"], right_uncertainty=hands["right"]["uncertainty"], timestamps_s=timestamps.astype(np.float64), metric_scale=np.asarray([float(geometry_info["scale"])], dtype=np.float64), service_vertices_used=np.asarray([False], dtype=np.bool_), legacy_hawor_droid_executed=np.asarray([False], dtype=np.bool_), droid_invocation_count=np.asarray([1], dtype=np.int32))
            validate_legacy_hawor_archive(staging_npz, frame_count)
            os.replace(staging_npz, output_npz)
        except Exception:
            try:
                staging_npz.unlink()
            except FileNotFoundError:
                pass
            raise
        detector_timeline_path = run_root / "measurements" / "hand_detections" / "feishu_ray_hands" / "hands_detector_timeline.json"
        detector_provenance = {"path": str(detector_timeline_path), "sha256": sha256_file(detector_timeline_path)}
        adapter_report = {"schema": "ego.annotation.feishu_ray_hawor_adapter.v1", "status": "ok", "method": "feishu_ray_hawor_track_infiller_adapter", "output_npz": str(output_npz), "output_sha256": sha256_file(output_npz), "service_profile": profile.get("profile"), "service_base_url": base_url, "service_calls": service_calls, "retry_events": retry_events, "track_chunks": chunk_reports, "infiller_windows": infiller_reports, "skipped_infiller_windows": skipped_windows, "detector_timeline": detector_provenance, "metric_scale": geometry_info["scale_report"], "droid_evidence": geometry_info["droid_evidence"], "service_vertices_used": False, "legacy_hawor_droid_executed": False, "droid_invocation_count": 1, "mano_replay": {"asset_root": str(args.hawor_root), "assets": mano_assets, "finite_nonzero_surfaces": {side: int(np.count_nonzero(hands[side]["valid"])) for side in ("left", "right")}, "vertices_per_hand": MANO_VERTICES, "joints_per_hand": MANO_JOINTS}}
        adapter_path = output_dir / "hawor_slam_adapter_report.json"
        write_json(adapter_path, adapter_report)
        qc = {"schema": "ego.annotation.feishu_ray_hawor_qc.v1", "status": "ok", "frames": frame_count, "output_npz": str(output_npz), "valid_hand_frames": {side: int(np.count_nonzero(hands[side]["valid"])) for side in ("left", "right")}, "detected_same_frame_hand_frames": {side: int(np.count_nonzero(np.asarray([int(i in detector_by_side[side]) for i in frame_idx], dtype=np.uint8))) for side in ("left", "right")}, "metric_scale": geometry_info["scale_report"], "droid_evidence": geometry_info["droid_evidence"], "service_calls": service_calls, "retry_events": retry_events, "service_vertices_used": False, "legacy_hawor_droid_executed": False, "droid_invocation_count": 1, "adapter_report": str(adapter_path), "adapter_report_sha256": sha256_file(adapter_path), "claim_scope": "Feishu Ray HaWoR MANO parameters replayed through remote HaWoR MANO assets; missing/infiller rows remain uncertain and do not establish contact/occlusion ownership/nonpenetration."}
        qc_path = output_dir / "qc_hawor_world_hands.json"
        write_json(qc_path, qc)
        provenance = {"schema": "ego.annotation.feishu_ray_hawor_provenance.v1", "created_at": utc_now(), "run_root": str(run_root), "raw_frame_manifest": {"path": str(run_root / "input" / "raw_frame_manifest" / "manifest.json"), "sha256": sha256_file(run_root / "input" / "raw_frame_manifest" / "manifest.json")}, "detector_timeline": detector_provenance, "unidepth": {"path": str(geometry_info["depth_path"]), "sha256": sha256_file(geometry_info["depth_path"])}, "droid_shared_manifest": {"path": geometry_info["droid_evidence"]["droid_manifest"], "sha256": geometry_info["droid_evidence"]["droid_manifest_sha256"]}, "hawor_world_hands": {"path": str(output_npz), "sha256": sha256_file(output_npz)}, "mano_assets": mano_assets, "qc": {"path": str(qc_path), "sha256": sha256_file(qc_path)}, "adapter": {"path": str(adapter_path), "sha256": sha256_file(adapter_path)}, "service_profile": profile.get("profile"), "service_base_url": base_url, "model_revisions": {"track": TRACK_MODEL_REVISION, "infiller": INFILLER_MODEL_REVISION}, "legacy_hawor_droid_executed": False, "droid_invocation_count": 1}
        provenance_path = output_dir / "hawor_hash_provenance.json"
        write_json(provenance_path, provenance)
        stage = {"schema": "v22_hawor_metric_hand_stage.v0", "status": "ok", "method": "feishu_ray_hawor_adapter", "run_root": str(run_root), "hawor_output_dir": str(output_dir), "hawor_world_hands": str(output_npz), "qc": str(qc_path), "adapter_report": str(adapter_path), "provenance": str(provenance_path), "service_profile": profile.get("profile"), "service_base_url": base_url, "processed_frames": frame_count, "service_calls": service_calls, "retry_events": retry_events, "elapsed_s": float(time.time() - started), "legacy_hawor_droid_executed": False, "droid_invocation_count": 1, "claim_scope": qc["claim_scope"]}
        write_json(output_dir / "v22_hawor_metric_hand_stage.json", stage)
        return stage
    except Exception as exc:
        failure_path = _write_failure(run_root, exc)
        try:
            setattr(exc, "failure_path", str(failure_path))
        except Exception:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--profile", type=Path, default=REPO_ROOT / "configs" / "feishu_ray_services.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-max-wait-s", type=float, default=0.0, help="Maximum cumulative wait for explicit retryable service responses; 0 means wait indefinitely")
    parser.add_argument("--retry-initial-delay-s", type=float, default=1.0)
    parser.add_argument("--job-id", default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        report = run_hawor(args)
    except (FeishuRayAdapterError, ServiceCallerError, FileNotFoundError) as exc:
        print(json.dumps({"status": "failed", "code": getattr(exc, "code", "file_not_found"), "error": str(exc), "failure_path": getattr(exc, "failure_path", None)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
