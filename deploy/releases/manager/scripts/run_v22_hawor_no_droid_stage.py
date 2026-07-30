#!/usr/bin/env python3
"""Resident HaWoR temporal-stage adapter with no DROID-SLAM dependency.

Detection/tracking state is kept per request. Equal-length 16-frame chunks from
multiple requests are assembled into one `[B,16,...]` HaWoR model forward. The
output is transformed with an explicitly supplied upstream camera artifact when
one is available; no DROID module, manifest, or adapter is imported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _camera_artifact(payload: dict[str, Any]) -> Path | None:
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("role") == "camera_artifact" and row.get("path"):
            path = Path(str(row["path"])).expanduser().resolve()
            if path.is_file():
                return path
    return None


def _load_camera(path: Path | None, frame_count: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load world-camera evidence without creating world state past coverage.

    Legacy sparse camera artifacts without a coverage mask retain their existing
    interpolation behavior.  A DROID/camera validity mask is authoritative:
    only explicitly valid source frames may enter the world transform; all other
    camera slots remain NaN rather than endpoint-filled or identity-filled.
    """
    if path is None:
        return np.full((frame_count, 4, 4), np.nan, dtype=np.float32), np.zeros(frame_count, dtype=bool), "camera_absent_world_unresolved"
    data = np.load(path, allow_pickle=False)

    if "T_world_camera" in data:
        poses = np.asarray(data["T_world_camera"], dtype=np.float32)
    elif "pose_world_camera_xyzw" in data:
        from scipy.spatial.transform import Rotation
        pose = np.asarray(data["pose_world_camera_xyzw"], dtype=np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32)[None], (len(pose), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(pose[:, 3:]).as_matrix().astype(np.float32)
        poses[:, :3, 3] = pose[:, :3]
    else:
        raise RuntimeError(f"camera artifact lacks T_world_camera or pose_world_camera_xyzw: {path}")
    if len(poses) == 0:
        raise RuntimeError(f"camera artifact has no poses: {path}")
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64).reshape(-1) if "frame_idx" in data else np.linspace(0, frame_count - 1, len(poses), dtype=np.int64)
    if len(frame_idx) != len(poses) or np.any(frame_idx < 0) or np.any(frame_idx >= frame_count) or len(set(frame_idx.tolist())) != len(frame_idx):
        raise RuntimeError(f"camera frame_idx is invalid for the request timeline: {path}")
    coverage_key = next((key for key in ("droid_pose_valid", "camera_valid") if key in data), None)
    if coverage_key is not None:
        raw_valid = np.asarray(data[coverage_key], dtype=bool).reshape(-1)
        if raw_valid.shape == (len(poses),):
            pose_valid = raw_valid
        elif raw_valid.shape == (frame_count,):
            pose_valid = raw_valid[frame_idx]
        else:
            raise RuntimeError(f"{coverage_key} length does not match camera poses or source timeline: {path}")
        dense = np.full((frame_count, 4, 4), np.nan, dtype=np.float32)
        camera_valid = np.zeros(frame_count, dtype=bool)
        for source_index, pose, valid in zip(frame_idx.tolist(), poses, pose_valid.tolist()):
            if not valid:
                continue
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1.0e-5):
                raise RuntimeError(f"camera validity mask marks a non-finite/invalid pose valid: {path}")
            dense[source_index] = pose
            camera_valid[source_index] = True
        # An explicit coverage mask forbids interpolation and endpoint fill.
        # It is the source of truth for DROID's [0,1024) service prefix.
        return dense, camera_valid, f"upstream_camera_artifact_masked:{coverage_key}"

    if "droid" in path.name.lower():
        raise RuntimeError(f"DROID camera artifact requires authoritative droid_pose_valid/camera_valid coverage: {path}")

    order = np.argsort(frame_idx)
    frame_idx = frame_idx[order].astype(np.float32)
    poses = poses[order]
    if not np.isfinite(poses).all():
        raise RuntimeError(f"unmasked camera artifact contains non-finite poses: {path}")
    # VGGT emits sparse sequence windows. This legacy path is only available
    # without an explicit DROID coverage mask.
    if len(poses) < frame_count or not np.array_equal(frame_idx.astype(np.int32), np.arange(frame_count, dtype=np.int32)):
        from scipy.spatial.transform import Rotation, Slerp
        target_idx = np.arange(frame_count, dtype=np.float32)
        clipped = np.clip(target_idx, frame_idx[0], frame_idx[-1])
        translations = np.stack([np.interp(clipped, frame_idx, poses[:, axis, 3]) for axis in range(3)], axis=1)
        rotations = Slerp(frame_idx, Rotation.from_matrix(poses[:, :3, :3]))(clipped).as_matrix().astype(np.float32)
        dense = np.tile(np.eye(4, dtype=np.float32)[None], (frame_count, 1, 1))
        dense[:, :3, :3] = rotations
        dense[:, :3, 3] = translations.astype(np.float32)
        return dense, np.ones(frame_count, dtype=bool), "upstream_camera_artifact_sparse_interpolated_without_coverage_mask"
    return poses[:frame_count], np.ones(frame_count, dtype=bool), "upstream_camera_artifact_full_timeline"


def _world_validity(model_valid: np.ndarray, camera_valid: np.ndarray) -> np.ndarray:
    model = np.asarray(model_valid, dtype=bool).reshape(-1)
    camera = np.asarray(camera_valid, dtype=bool).reshape(-1)
    if model.shape != camera.shape:
        raise RuntimeError("model and camera validity must cover the same timeline")
    return model & camera


def _best_boxes(det_result: Any) -> dict[int, np.ndarray]:
    best: dict[int, np.ndarray] = {}
    for det in det_result:
        boxes = det.boxes
        if boxes is None or len(boxes) == 0:
            continue
        raw = boxes.data.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        for row, side_value in zip(raw, classes):
            side = int(round(float(side_value)))
            if side not in (0, 1) or row.shape[0] < 5:
                continue
            candidate = np.asarray(row[:5], dtype=np.float32)
            if side not in best or float(candidate[4]) > float(best[side][4]):
                best[side] = candidate
    return best


def _nearest_box(boxes: list[np.ndarray | None], index: int) -> tuple[np.ndarray | None, bool]:
    if boxes[index] is not None:
        return np.asarray(boxes[index], dtype=np.float32), True
    for distance in range(1, len(boxes)):
        left = index - distance
        right = index + distance
        if left >= 0 and boxes[left] is not None:
            return np.asarray(boxes[left], dtype=np.float32), False
        if right < len(boxes) and boxes[right] is not None:
            return np.asarray(boxes[right], dtype=np.float32), False
    return None, False


def _move(value: Any, device: str, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _cached_mano_output(adapter: Any, side: int, trans: Any, root_axis: Any, pose_axis: Any, betas: Any, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    """Reuse resident MANO layers while preserving the existing output geometry."""
    layer_name = "_mano_left_layer" if side == 0 else "_mano_right_layer"
    layer = getattr(adapter, layer_name, None)
    if layer is None:
        if side == 1:
            layer = adapter.model.mano
        else:
            from lib.models.mano_wrapper import MANO
            layer = MANO(
                data_dir="_DATA/data_left/",
                model_path="_DATA/data_left/mano_left",
                gender="neutral",
                num_hand_joints=15,
                create_body_pose=False,
                is_rhand=False,
            )
            layer = layer.to(adapter.device).eval()
            with torch.no_grad():
                layer.shapedirs[:, 0, :] *= -1
        setattr(adapter, layer_name, layer)
    trans = trans.reshape(1, 1, 3).to(adapter.device)
    root_axis = root_axis.reshape(1, 1, 3).to(adapter.device)
    pose_axis = pose_axis.reshape(1, 1, 45).to(adapter.device)
    betas = betas.reshape(1, 1, 10).to(adapter.device)
    from hawor.utils.geometry import aa_to_rotmat
    root_flat = root_axis.reshape(-1, 3)
    pose_flat = pose_axis.reshape(-1, 45)
    mano_params = {
        "global_orient": aa_to_rotmat(root_flat).view(-1, 1, 3, 3),
        "hand_pose": aa_to_rotmat(pose_flat.reshape(-1, 3)).view(-1, 15, 3, 3),
        "betas": betas.reshape(-1, 10),
        "transl": trans.reshape(-1, 3),
    }
    with torch.inference_mode():
        mano_output = layer(**mano_params, pose2rot=False)
    return mano_output.vertices[0].detach().cpu().numpy().astype(np.float32), mano_output.joints[0].detach().cpu().numpy().astype(np.float32)


def _make_chunk_items(*, adapter: Any, entry: Any, frames: list[dict[str, Any]], boxes_by_side: dict[int, list[np.ndarray | None]], side: int, focal: float, center: list[float], dataset_cls: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(frames), 16):
        indices = list(range(start, min(len(frames), start + 16)))
        if not indices:
            continue
        selected_boxes: list[np.ndarray] = []
        valid: list[bool] = []
        for index in indices:
            box, observed = _nearest_box(boxes_by_side[side], index)
            if box is None:
                selected_boxes = []
                break
            selected_boxes.append(box)
            valid.append(observed)
        if not selected_boxes:
            continue
        while len(indices) < 16:
            indices.append(indices[-1])
            selected_boxes.append(selected_boxes[-1])
            valid.append(False)
        image_paths = [str(Path(str(frames[index].get("rgb") or frames[index].get("raw_frame_path"))).expanduser().resolve()) for index in indices]
        if any(not Path(path).is_file() for path in image_paths):
            raise FileNotFoundError(f"HaWoR staged image missing in request {entry.request_id}")
        boxes = np.stack(selected_boxes, axis=0).astype(np.float32)
        dataset = dataset_cls(image_paths, boxes, img_focal=focal, img_center=center, normalization=True, dilate=1.2, do_flip=(side == 0))
        items = [dataset[i] for i in range(len(dataset))]
        chunks.append({"entry": entry, "side": side, "indices": indices, "valid": valid, "items": items})
    return chunks


def run_hawor_service_batch(adapter: Any, entries: list[Any], default_collate: Any) -> dict[str, Any]:
    import torch
    from hawor.utils.process import get_mano_faces
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    from lib.datasets.track_dataset import TrackDatasetEval

    per_request: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry.payload
        frames = [row for row in _frames(payload) if isinstance(row, dict)]
        camera = payload.get("camera") if isinstance(payload.get("camera"), dict) else {}
        intrinsics = camera.get("intrinsics_px") if isinstance(camera.get("intrinsics_px"), list) else None
        focal = float(np.sqrt(float(intrinsics[0]) * float(intrinsics[1]))) if intrinsics and len(intrinsics) >= 2 else 600.0
        width = int((payload.get("video_meta") or {}).get("width") or frames[0].get("source_width") or frames[0].get("manifest_width"))
        height = int((payload.get("video_meta") or {}).get("height") or frames[0].get("source_height") or frames[0].get("manifest_height"))
        center = [width / 2.0, height / 2.0]
        side_boxes: dict[int, list[np.ndarray | None]] = {0: [], 1: []}
        for frame in frames:
            image = cv2.imread(str(Path(str(frame.get("rgb") or frame.get("raw_frame_path"))).expanduser().resolve()))
            if image is None:
                raise FileNotFoundError(f"HaWoR cannot read staged frame {frame}")
            detection = adapter.detector(image, conf=0.2, verbose=False)[0]
            best = _best_boxes([detection])
            for side in (0, 1):
                side_boxes[side].append(best.get(side))
        camera_poses, camera_valid, camera_status = _load_camera(_camera_artifact(payload), len(frames))
        per_request[entry.request_id] = {"entry": entry, "frames": frames, "camera": (camera_poses, camera_valid, camera_status), "camera_vertices": {0: {}, 1: {}}, "camera_joints": {0: {}, 1: {}}, "camera_trans": {0: {}, 1: {}}, "rot": {0: {}, 1: {}}, "pose": {0: {}, 1: {}}, "betas": {0: {}, 1: {}}, "valid": {0: np.zeros(len(frames), dtype=np.uint8), 1: np.zeros(len(frames), dtype=np.uint8)}, "detected": {0: np.asarray([box is not None for box in side_boxes[0]], dtype=np.uint8), 1: np.asarray([box is not None for box in side_boxes[1]], dtype=np.uint8)}, "boxes": side_boxes}
        for side in (0, 1):
            chunks.extend(_make_chunk_items(adapter=adapter, entry=entry, frames=frames, boxes_by_side=side_boxes, side=side, focal=focal, center=center, dataset_cls=TrackDatasetEval))
    native_forward_count = 0
    native_shapes: list[list[int]] = []
    for side in (0, 1):
        side_chunks = [chunk for chunk in chunks if chunk["side"] == side]
        for start in range(0, len(side_chunks), int(adapter.native_batch_cap)):
            group = side_chunks[start : start + int(adapter.native_batch_cap)]
            if not group:
                continue
            flat_items = [item for chunk in group for item in chunk["items"]]
            collated = default_collate(flat_items)
            batch: dict[str, Any] = {}
            for key, value in collated.items():
                if not torch.is_tensor(value):
                    continue
                shape = tuple(value.shape)
                if shape[0] != len(flat_items):
                    continue
                batch[key] = value.reshape(len(group), 16, *shape[1:]).to(adapter.device)
            with torch.inference_mode():
                output = adapter.model.forward(batch)
            output = output.get("out", output)
            native_forward_count += 1
            native_shapes.append([int(x) for x in batch["img"].shape])
            total = len(group) * 16
            rotmat = output["pred_rotmat"].detach().cpu()
            trans = output["trans_full"].detach().cpu()
            shape_coeff = output["pred_shape"].detach().cpu()
            root_axis = rotation_matrix_to_angle_axis(rotmat[:, 0]).detach().cpu()
            pose_axis = rotation_matrix_to_angle_axis(rotmat[:, 1:]).reshape(total, 45).detach().cpu()
            for chunk_index, chunk in enumerate(group):
                target = per_request[chunk["entry"].request_id]
                for local in range(16):
                    global_index = chunk_index * 16 + local
                    frame_index = int(chunk["indices"][local])
                    if not chunk["valid"][local]:
                        continue
                    t = trans[global_index : global_index + 1]
                    r = root_axis[global_index : global_index + 1]
                    p = pose_axis[global_index : global_index + 1]
                    b = shape_coeff[global_index : global_index + 1]
                    vertices, joints = _cached_mano_output(adapter, side, t, r, p, b, torch)
                    target["camera_vertices"][side][frame_index] = vertices
                    target["camera_joints"][side][frame_index] = joints
                    target["camera_trans"][side][frame_index] = t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    target["rot"][side][frame_index] = r[0].numpy().astype(np.float32)
                    target["pose"][side][frame_index] = p[0].numpy().astype(np.float32)
                    target["betas"][side][frame_index] = b[0].numpy().astype(np.float32)
                    target["valid"][side][frame_index] = 1
    results = []
    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    for request_id, target in per_request.items():
        entry = target["entry"]
        frame_count = len(target["frames"])
        poses, camera_valid, world_status = target["camera"]
        output_dir = Path(str(entry.payload.get("output_dir"))).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "frame_idx": np.arange(frame_count, dtype=np.int32),
            "T_world_camera": poses.astype(np.float32),
            "R_c2w": poses[:, :3, :3].astype(np.float32),
            "t_c2w": poses[:, :3, 3].astype(np.float32),
            "camera_valid": camera_valid.astype(np.uint8),
            "hawor_droid_slam_executed": np.asarray([False], dtype=np.bool_),
            "hawor_world_frame_status": np.asarray([world_status]),
        }
        for side, name in ((0, "left"), (1, "right")):
            n_vertices = len(next(iter(target["camera_vertices"][side].values()))) if target["camera_vertices"][side] else 778
            n_joints = len(next(iter(target["camera_joints"][side].values()))) if target["camera_joints"][side] else 21
            vertices = np.full((frame_count, n_vertices, 3), np.nan, dtype=np.float32)
            joints = np.full((frame_count, n_joints, 3), np.nan, dtype=np.float32)
            trans = np.full((frame_count, 3), np.nan, dtype=np.float32)
            model_valid = target["valid"][side]
            world_valid = _world_validity(model_valid, camera_valid).astype(np.uint8)
            for frame_index in np.flatnonzero(world_valid):
                T = poses[int(frame_index)]
                points = np.concatenate([target["camera_vertices"][side][int(frame_index)], np.ones((n_vertices, 1), dtype=np.float32)], axis=1)
                vertices[int(frame_index)] = (T @ points.T).T[:, :3]
                points = np.concatenate([target["camera_joints"][side][int(frame_index)], np.ones((n_joints, 1), dtype=np.float32)], axis=1)
                joints[int(frame_index)] = (T @ points.T).T[:, :3]
                p = np.concatenate([target["camera_trans"][side][int(frame_index)], np.ones(1, dtype=np.float32)])
                trans[int(frame_index)] = (T @ p)[:3]
            arrays[f"{name}_vertices_world_m"] = vertices
            arrays[f"{name}_joints_world_m"] = joints
            arrays[f"{name}_trans_world_m"] = trans
            arrays[f"{name}_valid"] = world_valid
            arrays[f"{name}_model_valid_camera_independent"] = model_valid.astype(np.uint8)
            arrays[f"{name}_detected_same_frame"] = target["detected"][side].astype(np.uint8)
            arrays[f"{name}_det_box_xyxyscore"] = np.full((frame_count, 5), np.nan, dtype=np.float32)
            arrays[f"{name}_faces"] = faces if side == 1 else faces[:, [0, 2, 1]]
        archive = output_dir / "hawor_world_hands.npz"
        np.savez_compressed(archive, **arrays)
        qc = {"schema": "v22_hawor_no_droid_service_qc.v1", "status": "completed_with_partial_camera_coverage" if not bool(np.all(camera_valid)) else "ok", "request_id": request_id, "frame_count": frame_count, "output_npz": str(archive), "droid_slam_executed": False, "world_frame_status": world_status, "camera_valid_count": int(np.count_nonzero(camera_valid)), "camera_invalid_count": int(frame_count - np.count_nonzero(camera_valid)), "native_batch_shapes": native_shapes, "claim_scope": "resident HaWoR temporal local-model inference; world camera/hand geometry exists only where upstream camera coverage is explicitly valid; DROID-SLAM excluded"}
        qc_path = output_dir / "qc_hawor_world_hands.json"
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
        results.append({"status": "completed_with_partial_camera_coverage" if not bool(np.all(camera_valid)) else "ok", "output_artifacts": {"hawor_world_hands": str(archive), "qc": str(qc_path)}, "frame_count": frame_count, "camera_valid_count": int(np.count_nonzero(camera_valid)), "native_forward_count": native_forward_count, "native_batch_shapes": native_shapes, "droid_slam_executed": False})
    return {"results": results, "native_forward_count": native_forward_count, "native_batch_shapes": native_shapes, "rows_processed": sum(len(target["frames"]) for target in per_request.values())}


# Imported lazily by services.model_adapters to avoid importing torch/HaWoR in the CLI client.
try:
    from services.model_adapters import _frames  # type: ignore
except Exception:  # pragma: no cover
    _frames = None
