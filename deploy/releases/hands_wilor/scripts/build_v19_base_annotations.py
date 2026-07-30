#!/usr/bin/env python3
"""Build V19 base annotations from fresh V19 measurement outputs.

Inputs are produced under the current V19 run root: raw frame manifest, camera
trajectory, HaWoR/MANO export, agent object plan, and SAM2 tracks. The output is
one frame per source frame in the V18-compatible annotation shape needed by
existing geometry/constraint/render components, but no cached V16/V17/V18
annotation root is consumed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def finite_matrix4(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
        return arr
    return None


def load_raw_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    frames = [f for f in as_list(payload.get("frames")) if isinstance(f, dict)]
    if not frames:
        raise RuntimeError(f"raw manifest has no frame rows: {path}")
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for frame in frames:
        if frame.get("frame_idx") is None:
            raise RuntimeError(f"raw manifest row lacks frame_idx: {frame}")
        idx = int(frame["frame_idx"])
        if idx in seen:
            raise RuntimeError(f"duplicate raw frame index {idx} in {path}")
        seen.add(idx)
        rgb = frame.get("rgb") or frame.get("raw_frame_path")
        if not isinstance(rgb, str) or not Path(rgb).exists():
            raise FileNotFoundError(f"raw frame {idx} image missing: {rgb}")
        row = dict(frame)
        row["rgb"] = str(rgb)
        row["raw_frame_path"] = str(frame.get("raw_frame_path") or rgb)
        out.append(row)
    out.sort(key=lambda r: int(r["frame_idx"]))
    return out, payload


def load_camera_npz(path: Path | None) -> tuple[dict[int, tuple[np.ndarray, str]], dict[int, tuple[list[float], str]], str | None]:
    if path is None:
        return {}, {}, None
    if not path.exists():
        raise FileNotFoundError(f"missing camera npz: {path}")
    blob = np.load(path, allow_pickle=True)
    if "frame_idx" not in blob.files:
        raise RuntimeError(f"camera npz lacks frame_idx: {path}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    poses: dict[int, tuple[np.ndarray, str]] = {}
    intrinsics: dict[int, tuple[list[float], str]] = {}
    if "T_world_camera" in blob.files:
        mats = np.asarray(blob["T_world_camera"], dtype=float)
        source = "camera_npz_T_world_camera"
    elif "T_world_camera_metric_current_v18" in blob.files:
        mats = np.asarray(blob["T_world_camera_metric_current_v18"], dtype=float)
        source = "camera_npz_T_world_camera_metric_current_v18"
    elif "R_c2w" in blob.files and "t_c2w" in blob.files:
        r = np.asarray(blob["R_c2w"], dtype=float)
        t = np.asarray(blob["t_c2w"], dtype=float)
        if len(r) != len(frame_idx) or len(t) != len(frame_idx):
            raise RuntimeError(f"camera npz R_c2w/t_c2w length mismatch: {path}")
        mats = np.repeat(np.eye(4, dtype=float)[None, :, :], len(frame_idx), axis=0)
        mats[:, :3, :3] = r
        mats[:, :3, 3] = t
        source = "camera_npz_R_c2w_t_c2w"
    else:
        raise RuntimeError(
            f"camera npz lacks supported pose keys: {path}; expected T_world_camera or R_c2w/t_c2w"
        )
    if mats.shape != (len(frame_idx), 4, 4):
        raise RuntimeError(f"camera npz pose shape mismatch {mats.shape} for {path}")
    for idx, mat in zip(frame_idx, mats, strict=True):
        if not np.isfinite(mat).all():
            raise RuntimeError(f"non-finite camera pose for frame {int(idx)} in {path}")
        poses[int(idx)] = (mat.astype(float), source)
    if "intrinsics_source" in blob.files:
        intr = np.asarray(blob["intrinsics_source"], dtype=float).reshape(-1)
        if intr.shape[0] == 4 and np.all(np.isfinite(intr)):
            for idx in frame_idx:
                intrinsics[int(idx)] = ([float(v) for v in intr.tolist()], "camera_npz_intrinsics_source")
    if "intrinsics_fx_fy_cx_cy" in blob.files:
        intr_arr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=float)
        if intr_arr.shape == (len(frame_idx), 4):
            for idx, intr in zip(frame_idx, intr_arr, strict=True):
                if np.all(np.isfinite(intr)):
                    intrinsics[int(idx)] = ([float(v) for v in intr.tolist()], "camera_npz_intrinsics_fx_fy_cx_cy")
    return poses, intrinsics, str(path)


def load_depth_intrinsics(path: Path | None) -> dict[int, tuple[list[float], str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"missing depth npz: {path}")
    blob = np.load(path, allow_pickle=True)
    if "frame_idx" not in blob.files or "intrinsics_fx_fy_cx_cy" not in blob.files:
        raise RuntimeError(f"depth npz lacks frame_idx/intrinsics_fx_fy_cx_cy: {path}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    intr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=float)
    if intr.shape != (len(frame_idx), 4):
        raise RuntimeError(f"depth intrinsics shape mismatch {intr.shape} for {path}")
    out: dict[int, tuple[list[float], str]] = {}
    for idx, row in zip(frame_idx, intr, strict=True):
        if np.all(np.isfinite(row)):
            out[int(idx)] = ([float(v) for v in row.tolist()], "depth_npz_intrinsics_fx_fy_cx_cy")
    return out


def load_calibration_contract(path: Path | None, frame_ids: list[int]) -> tuple[dict[int, tuple[list[float], str]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    if not path.exists():
        raise FileNotFoundError(f"missing calibration contract: {path}")
    payload = load_json(path)
    intr = np.asarray(payload.get("intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or float(intr[0]) <= 0.0 or float(intr[1]) <= 0.0:
        raise RuntimeError(f"calibration contract has invalid intrinsics_fx_fy_cx_cy: {path}")
    source = str(payload.get("intrinsics_source") or payload.get("method") or "v19_calibration_contract")
    source = f"calibration_contract:{source}"
    rows = {int(idx): ([float(v) for v in intr.tolist()], source) for idx in frame_ids}
    summary = {
        "path": str(path),
        "intrinsics_fx_fy_cx_cy": [float(v) for v in intr.tolist()],
        "intrinsics_source": source,
        "fov_degrees": payload.get("fov_degrees"),
        "aggregation": payload.get("aggregation"),
    }
    return rows, summary


def load_hawor_npz(path: Path | None) -> tuple[dict[str, np.ndarray], str | None]:
    if path is None:
        return {}, None
    if not path.exists():
        raise FileNotFoundError(f"missing HaWoR npz: {path}")
    blob = np.load(path, allow_pickle=True)
    required = {
        "frame_idx",
        "R_c2w",
        "t_c2w",
        "left_vertices_world_m",
        "left_joints_world_m",
        "left_valid",
        "right_vertices_world_m",
        "right_joints_world_m",
        "right_valid",
    }
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"HaWoR npz missing required arrays {missing}: {path}")
    return {key: np.asarray(blob[key]) for key in blob.files}, str(path)


def camera_from_hawor(arrays: dict[str, np.ndarray]) -> dict[int, tuple[np.ndarray, str]]:
    if not arrays:
        return {}
    frame_idx = np.asarray(arrays["frame_idx"], dtype=int)
    r = np.asarray(arrays["R_c2w"], dtype=float)
    t = np.asarray(arrays["t_c2w"], dtype=float)
    if r.shape[0] != len(frame_idx) or t.shape[0] != len(frame_idx):
        raise RuntimeError("HaWoR camera arrays do not match frame_idx length")
    out: dict[int, tuple[np.ndarray, str]] = {}
    for idx, rr, tt in zip(frame_idx, r, t, strict=True):
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = np.asarray(rr, dtype=float)
        mat[:3, 3] = np.asarray(tt, dtype=float)
        if np.all(np.isfinite(mat)):
            out[int(idx)] = (mat, "hawor_npz_R_c2w_t_c2w")
    return out


def default_intrinsics_from_raw(frame: dict[str, Any], focal: float | None, source: str) -> tuple[list[float], str] | None:
    if focal is None or not np.isfinite(float(focal)) or float(focal) <= 0.0:
        return None
    width = int(frame.get("source_width") or frame.get("manifest_width") or 0)
    height = int(frame.get("source_height") or frame.get("manifest_height") or 0)
    if width <= 0 or height <= 0:
        return None
    return ([float(focal), float(focal), float(width) / 2.0, float(height) / 2.0], source)


def world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    r_c2w = T_world_camera[:3, :3]
    t_c2w = T_world_camera[:3, 3]
    return (np.asarray(points_world, dtype=float) - t_c2w[None, :]) @ r_c2w


def sample_vertices(vertices: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"invalid hand vertices shape {vertices.shape}")
    if len(vertices) <= count:
        ids = np.arange(len(vertices), dtype=np.int32)
    else:
        ids = np.linspace(0, len(vertices) - 1, count, dtype=np.int32)
    return vertices[ids], ids


def save_hawor_bridge(
    *,
    arrays: dict[str, np.ndarray],
    path: Path,
    poses: dict[int, tuple[np.ndarray, str]],
) -> tuple[dict[tuple[int, str], int], dict[str, Any]]:
    if not arrays:
        return {}, {"status": "not_written_no_hawor_npz"}
    frame_idx = np.asarray(arrays["frame_idx"], dtype=int)
    row_index: dict[tuple[int, str], int] = {}
    bridge_frame_idx: list[int] = []
    bridge_side: list[str] = []
    source_frame_index: list[int] = []
    vertices_w: list[np.ndarray] = []
    joints_w: list[np.ndarray] = []
    vertices_c: list[np.ndarray] = []
    joints_c: list[np.ndarray] = []
    for pos, idx_raw in enumerate(frame_idx.tolist()):
        idx = int(idx_raw)
        pose = poses.get(idx)
        if pose is None:
            continue
        T_world_camera = pose[0]
        for side in ("left", "right"):
            valid_key = f"{side}_valid"
            if valid_key not in arrays or int(np.asarray(arrays[valid_key])[pos]) == 0:
                continue
            vw = np.asarray(arrays[f"{side}_vertices_world_m"][pos], dtype=np.float32)
            jw = np.asarray(arrays[f"{side}_joints_world_m"][pos], dtype=np.float32)
            if vw.ndim != 2 or vw.shape[1] != 3 or jw.shape != (21, 3):
                continue
            row = len(bridge_frame_idx)
            row_index[(idx, side)] = row
            bridge_frame_idx.append(idx)
            bridge_side.append(side)
            source_frame_index.append(pos)
            vertices_w.append(vw)
            joints_w.append(jw)
            vertices_c.append(world_to_camera(vw, T_world_camera).astype(np.float32))
            joints_c.append(world_to_camera(jw, T_world_camera).astype(np.float32))
    if not bridge_frame_idx:
        return {}, {"status": "not_written_no_valid_hawor_hands"}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(bridge_frame_idx, dtype=np.int32),
        hand_side=np.asarray(bridge_side),
        source_frame_index=np.asarray(source_frame_index, dtype=np.int32),
        vertices_current_v18_world_from_hawor_projection_relift_m=np.stack(vertices_w, axis=0).astype(np.float32),
        joints_current_v18_world_from_hawor_projection_relift_m=np.stack(joints_w, axis=0).astype(np.float32),
        vertices_current_v18_camera_m=np.stack(vertices_c, axis=0).astype(np.float32),
        joints_current_v18_camera_m=np.stack(joints_c, axis=0).astype(np.float32),
        vertices_hawor_camera_m=np.stack(vertices_c, axis=0).astype(np.float32),
        v19_bridge_status=np.asarray(["direct_hawor_world_to_v19_base_no_cached_v18_bridge"]),
    )
    return row_index, {
        "status": "ok",
        "path": str(path),
        "rows": int(len(bridge_frame_idx)),
        "claim_scope": "V19 compatibility bridge from freshly exported HaWoR world vertices/joints; no cached V18 bridge root consumed",
    }


def parse_sam2_track_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        track_id, raw = value.split("=", 1)
        track_id = track_id.strip()
        if not track_id:
            raise RuntimeError(f"empty track id in --sam2-track {value!r}")
        return track_id, Path(raw).expanduser()
    path = Path(value).expanduser()
    if path.name == "sam2_track.json" and path.parent.name == "sam2":
        return path.parent.parent.name, path
    if path.name == "sam2_track.json":
        return path.parent.name, path
    raise RuntimeError(f"--sam2-track must be TRACK_ID=PATH or */sam2/sam2_track.json: {value}")


def load_sam2_tracks(values: list[str], root: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    pairs: list[tuple[str, Path]] = []
    for value in values:
        pairs.append(parse_sam2_track_arg(value))
    if root is not None:
        for path in sorted(root.glob("*/sam2/sam2_track.json")):
            pairs.append((path.parent.parent.name, path))
        for path in sorted(root.glob("*/sam2_track.json")):
            pairs.append((path.parent.name, path))
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for track_id, path in pairs:
        if not path.exists():
            raise FileNotFoundError(f"missing SAM2 track for {track_id}: {path}")
        payload = load_json(path)
        rows: dict[int, dict[str, Any]] = {}
        for key, row in payload.items():
            try:
                idx = int(key)
            except ValueError as exc:
                raise RuntimeError(f"SAM2 track {path} has non-frame key {key!r}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"SAM2 track {path} row {idx} is not an object")
            rows[idx] = dict(row)
        out[track_id] = rows
    return out


def load_object_plan(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"missing object plan: {path}")
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    objects = as_list(plan.get("objects") if isinstance(plan, dict) else None)
    out: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        track_id = obj.get("track_id")
        if isinstance(track_id, str) and track_id:
            out[track_id] = dict(obj)
    return out


def interval_active(obj: dict[str, Any], frame_idx: int) -> bool:
    intervals = as_list(obj.get("active_intervals"))
    if not intervals:
        return True
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        start = int(interval.get("start_frame", interval.get("start", -10**12)))
        end = int(interval.get("end_frame", interval.get("end", 10**12)))
        if start <= frame_idx <= end:
            return True
    return False


def localize_path(path: str | Path, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src, dst in ((remote_root, local_root), (local_root, remote_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(str(path))


def object_row(
    track_id: str,
    object_plan: dict[str, Any] | None,
    sam2_row: dict[str, Any] | None,
    frame_idx: int,
    *,
    remote_root: Path | None = None,
    local_root: Path | None = None,
) -> dict[str, Any] | None:
    plan = object_plan or {}
    active = interval_active(plan, frame_idx) if plan else sam2_row is not None
    visible = bool(sam2_row and sam2_row.get("visible") and sam2_row.get("mask_path"))
    if not active and not visible:
        return None
    oid = str(plan.get("object_id") or f"object:{track_id}")
    row = {
        "object_id": oid,
        "track_id": track_id,
        "label": str(plan.get("label") or track_id),
        "description": plan.get("description"),
        "status": "visible_mask_measurement" if visible else "planned_object_not_visible_or_untracked",
        "visible": bool(visible),
        "v19_physical_model": plan.get("physical_model"),
        "claim_scope": "object roster/mask measurement only; geometry and pose are produced by later V19 components",
    }
    if visible and sam2_row is not None:
        raw_mask_path = Path(str(sam2_row.get("mask_path")))
        try:
            mask_path = localize_path(raw_mask_path, remote_root, local_root)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"SAM2 mask path for {track_id} frame {frame_idx} does not exist locally or via remote/local root mapping: {raw_mask_path}") from exc
        row.update(
            {
                "mask_path": str(mask_path),
                "mask_path_original": str(raw_mask_path),
                "bbox_xyxy": sam2_row.get("bbox_xyxy"),
                "center_xy": sam2_row.get("center_xy"),
                "area_px": sam2_row.get("area_px"),
            }
        )
    return row


def build_hand_row(
    *,
    side: str,
    frame_idx: int,
    raw_frame: dict[str, Any],
    arrays: dict[str, np.ndarray],
    hawor_path: str | None,
    hawor_pos: int | None,
    bridge_path: Path | None,
    bridge_row_index: dict[tuple[int, str], int],
    T_world_camera: np.ndarray,
    intrinsics: list[float] | None,
) -> dict[str, Any]:
    base = {
        "hand_side": side,
        "visibility_state": "not_measured",
        "confidence": 0.0,
        "uncertainty": ["fresh_v19_base_no_valid_hawor_row"],
    }
    if not arrays or hawor_pos is None or hawor_path is None or f"{side}_valid" not in arrays:
        return base
    valid = int(np.asarray(arrays[f"{side}_valid"])[hawor_pos]) != 0
    detected = bool(int(np.asarray(arrays.get(f"{side}_detected_same_frame", np.zeros(len(arrays["frame_idx"]), dtype=np.uint8)))[hawor_pos])) if f"{side}_detected_same_frame" in arrays else False
    source_tag = "fresh_v19_hawor_world_export"
    if f"{side}_hybrid_source" in arrays:
        raw_source = np.asarray(arrays[f"{side}_hybrid_source"])[hawor_pos]
        source_tag = str(raw_source.item() if hasattr(raw_source, "item") else raw_source)
    if not valid:
        base.update(
            {
                "visibility_state": "hawor_invalid_or_not_visible",
                "same_frame_detection": detected,
                "uncertainty": ["hawor_export_invalid_for_frame_side"],
            }
        )
        return base
    joints_w = np.asarray(arrays[f"{side}_joints_world_m"][hawor_pos], dtype=float)
    verts_w = np.asarray(arrays[f"{side}_vertices_world_m"][hawor_pos], dtype=float)
    if joints_w.shape != (21, 3) or verts_w.ndim != 2 or verts_w.shape[1] != 3:
        raise RuntimeError(f"invalid HaWoR {side} geometry at frame {frame_idx}")
    joints_c = world_to_camera(joints_w, T_world_camera)
    verts_sample_w, sample_ids = sample_vertices(verts_w, 64)
    verts_sample_c = world_to_camera(verts_sample_w, T_world_camera)
    bridge_idx = bridge_row_index.get((frame_idx, side))
    if bridge_path is None or bridge_idx is None:
        raise RuntimeError(f"valid HaWoR hand lacks V19 bridge row for frame={frame_idx} side={side}")
    det_box = None
    if f"{side}_det_box_xyxyscore" in arrays:
        raw = np.asarray(arrays[f"{side}_det_box_xyxyscore"][hawor_pos], dtype=float).reshape(-1)
        if raw.size >= 4 and np.all(np.isfinite(raw[:4])):
            det_box = [float(v) for v in raw[:4].tolist()]
    metric = {
        "source": f"build_v19_base_annotations_from_{source_tag}",
        "case_frame_idx": int(frame_idx),
        "hand_side": side,
        "coordinate_status": "metric_world_from_hawor_camera_pose",
        "vertices_reference": {
            "bridge_npz": str(bridge_path),
            "bridge_vertices_world_array": "vertices_current_v18_world_from_hawor_projection_relift_m",
            "bridge_vertices_camera_array": "vertices_current_v18_camera_m",
            "bridge_raw_hawor_vertices_camera_array": "vertices_hawor_camera_m",
            "bridge_row_index": int(bridge_idx),
            "source_hawor_npz": hawor_path,
            "source_vertices_world_array": f"{side}_vertices_world_m",
            "source_joints_world_array": f"{side}_joints_world_m",
            "source_frame_index": int(hawor_pos),
            "shape_vertices": [int(verts_w.shape[0]), 3],
            "shape_joints": [21, 3],
            "v19_bridge_status": "direct_fresh_hawor_world_no_cached_v18_bridge",
        },
        "mano_params": {
            "parameterization": "hawor_axis_angle_world_export",
            "source_hawor_npz": hawor_path,
            "source_frame_index": int(hawor_pos),
            "side": side,
            "arrays": {
                "root_orient_axis_angle": f"{side}_root_orient_axis_angle",
                "hand_pose_axis_angle": f"{side}_hand_pose_axis_angle",
                "betas": f"{side}_betas",
                "trans_world_m": f"{side}_trans_world_m",
            },
            "root_orient_axis_angle": np.asarray(arrays.get(f"{side}_root_orient_axis_angle", np.zeros((len(arrays["frame_idx"]), 3)))[hawor_pos], dtype=float).tolist() if f"{side}_root_orient_axis_angle" in arrays else None,
            "hand_pose_axis_angle": np.asarray(arrays.get(f"{side}_hand_pose_axis_angle", np.zeros((len(arrays["frame_idx"]), 45)))[hawor_pos], dtype=float).tolist() if f"{side}_hand_pose_axis_angle" in arrays else None,
            "betas": np.asarray(arrays.get(f"{side}_betas", np.zeros((len(arrays["frame_idx"]), 10)))[hawor_pos], dtype=float).tolist() if f"{side}_betas" in arrays else None,
            "trans_world_m": np.asarray(arrays.get(f"{side}_trans_world_m", np.zeros((len(arrays["frame_idx"]), 3)))[hawor_pos], dtype=float).tolist() if f"{side}_trans_world_m" in arrays else None,
        },
        "joints_current_v18_world_m": joints_w.astype(float).tolist(),
        "joints_world_m": joints_w.astype(float).tolist(),
        "joints_current_v18_camera_m": joints_c.astype(float).tolist(),
        "vertices_world_sample_m": verts_sample_w.astype(float).tolist(),
        "vertices_camera_sample_m": verts_sample_c.astype(float).tolist(),
        "vertices_sample_indices": sample_ids.astype(int).tolist(),
        "current_v18_camera_intrinsics_fx_fy_cx_cy": intrinsics,
        "v19_camera_intrinsics_fx_fy_cx_cy": intrinsics,
        "same_frame_detection": detected,
        "support_state": source_tag,
    }
    base.update(
        {
            "visibility_state": "hybrid_valid_world_mano" if "wilor" in source_tag else "hawor_valid_world_mano",
            "bbox_xyxy": det_box,
            "same_frame_detection": detected,
            "hawor_same_frame_detection": detected,
            "hawor_candidate_present": True,
            "confidence": 0.65 if detected else 0.45,
            "uncertainty": [
                "fresh_hawor_world_candidate_not_final_interval_corrected_state"
            ] if "wilor" not in source_tag else [
                "wilor_visible_root_relative_geometry_on_hawor_metric_trajectory",
                "not_a_contact_or_nonpenetration_claim",
            ],
            "metric_mano_state": metric,
            "hand_geometry_source": source_tag,
        }
    )
    return base


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_frames, raw_payload = load_raw_manifest(args.raw_frame_manifest)
    raw_by_idx = {int(f["frame_idx"]): f for f in raw_frames}
    camera_poses, camera_intr, camera_path = load_camera_npz(args.camera_npz)
    depth_intr = load_depth_intrinsics(args.depth_npz)
    calibration_intr, calibration_summary = load_calibration_contract(args.calibration_contract, [int(f["frame_idx"]) for f in raw_frames])
    hawor_arrays, hawor_path = load_hawor_npz(args.hawor_npz)
    hawor_camera = camera_from_hawor(hawor_arrays)
    # Camera/world poses and HaWoR MANO vertices must live in the same world
    # frame. DROID/SLAM and HaWoR each define their own arbitrary world unless
    # an explicit alignment has been estimated. Default to HaWoR camera when
    # HaWoR MANO is present so hand surfaces and camera transforms stay in one
    # coordinate system; use --prefer-camera-npz only for an aligned camera NPZ.
    if hawor_camera and camera_poses and not args.prefer_camera_npz:
        poses = dict(camera_poses)
        poses.update(hawor_camera)
    else:
        poses = dict(hawor_camera)
        poses.update(camera_poses)
    if not poses:
        raise RuntimeError("V19 base annotations require a real camera/world pose source: provide --camera-npz or --hawor-npz")
    hawor_frame_pos = {int(v): i for i, v in enumerate(np.asarray(hawor_arrays.get("frame_idx", []), dtype=int).tolist())} if hawor_arrays else {}
    object_plan = load_object_plan(args.object_plan)
    sam2_tracks = load_sam2_tracks(args.sam2_track, args.sam2_output_root)
    for track_id in sam2_tracks:
        object_plan.setdefault(track_id, {"track_id": track_id, "description": f"SAM2 track {track_id}"})

    bridge_path = args.output_dir / "v19_mano_bridge_from_hawor_world.npz" if hawor_arrays else None
    bridge_rows, bridge_report = save_hawor_bridge(arrays=hawor_arrays, path=bridge_path or args.output_dir / "unused.npz", poses=poses)

    hawor_focal = None
    if hawor_arrays and "img_focal" in hawor_arrays:
        arr = np.asarray(hawor_arrays["img_focal"], dtype=float).reshape(-1)
        if arr.size:
            hawor_focal = float(arr[0])

    output_frames: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    missing_camera: list[int] = []
    for idx, raw in raw_by_idx.items():
        pose = poses.get(idx)
        if pose is None:
            missing_camera.append(idx)
            continue
        T_world_camera, camera_source = pose
        intr_pair = calibration_intr.get(idx) or depth_intr.get(idx) or camera_intr.get(idx) or default_intrinsics_from_raw(raw, hawor_focal, "hawor_img_focal_center_prior")
        intrinsics = intr_pair[0] if intr_pair is not None else None
        intr_source = intr_pair[1] if intr_pair is not None else None
        camera = {
            "T_world_camera_metric": T_world_camera.astype(float).tolist(),
            "T_world_camera": T_world_camera.astype(float).tolist(),
            "position_world_m": T_world_camera[:3, 3].astype(float).tolist(),
            "v19_camera_pose_source": camera_source,
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "intrinsics_source": intr_source,
        }
        hawor_pos = hawor_frame_pos.get(idx)
        hands = [
            build_hand_row(
                side=side,
                frame_idx=idx,
                raw_frame=raw,
                arrays=hawor_arrays,
                hawor_path=hawor_path,
                hawor_pos=hawor_pos,
                bridge_path=bridge_path if bridge_rows else None,
                bridge_row_index=bridge_rows,
                T_world_camera=T_world_camera,
                intrinsics=intrinsics,
            )
            for side in ("left", "right")
        ]
        for hand in hands:
            counts[f"hand::{hand.get('hand_side')}::{hand.get('visibility_state')}"] += 1
        objects = []
        for track_id, plan in sorted(object_plan.items()):
            sam2_row = sam2_tracks.get(track_id, {}).get(idx)
            row = object_row(
                track_id,
                plan,
                sam2_row,
                idx,
                remote_root=args.remote_root,
                local_root=args.local_root,
            )
            if row is not None:
                objects.append(row)
                counts[f"object::{track_id}::{row.get('status')}"] += 1
        frame = {
            "frame_idx": int(idx),
            "time_s": float(raw.get("time_s", 0.0)),
            "raw_frame_path": str(raw["raw_frame_path"]),
            "source_width": int(raw.get("source_width") or raw.get("manifest_width") or 0),
            "source_height": int(raw.get("source_height") or raw.get("manifest_height") or 0),
            "manifest_width": int(raw.get("manifest_width") or 0),
            "manifest_height": int(raw.get("manifest_height") or 0),
            "camera": camera,
            "hands": hands,
            "objects": objects,
            "contact_hypotheses": [],
            "frame_summary": {
                "v19_base_state": "fresh_measurement_backbone",
                "hand_count": int(len(hands)),
                "object_count": int(len(objects)),
            },
        }
        output_frames.append(frame)
    if missing_camera:
        preview = missing_camera[:10]
        raise RuntimeError(f"missing camera pose for {len(missing_camera)} raw frames; first missing frames: {preview}")
    if len(output_frames) != len(raw_frames):
        raise RuntimeError(f"base annotations frame count {len(output_frames)} != raw manifest count {len(raw_frames)}")
    output_frames.sort(key=lambda f: int(f["frame_idx"]))

    annotations = {
        "status": "ok",
        "method": "build_v19_base_annotations",
        "claim_scope": "fresh V19 one-frame-per-source-frame annotation backbone from run-root measurements; object pose/contact/occlusion are produced by later components",
        "case": args.case,
        "raw_video": raw_payload.get("video") if isinstance(raw_payload.get("video"), dict) else None,
        "input_video": raw_payload.get("input_video"),
        "raw_frame_count": int(len(output_frames)),
        "frames": output_frames,
        "v19_inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "camera_npz": camera_path,
            "depth_npz": str(args.depth_npz) if args.depth_npz else None,
            "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
            "hawor_npz": hawor_path,
            "object_plan": str(args.object_plan) if args.object_plan else None,
            "sam2_tracks": {track_id: "loaded" for track_id in sam2_tracks},
            "mano_bridge_npz": str(bridge_path) if bridge_rows and bridge_path is not None else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations_v19_base.json"
    write_json(annotations_path, annotations)
    physical_state = {
        "status": "ok",
        "method": "build_v19_base_annotations_state_sidecar",
        "case": args.case,
        "annotations": str(annotations_path),
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "camera_source_count": dict(Counter(frame["camera"]["v19_camera_pose_source"] for frame in output_frames)),
        "intrinsics_source_count": dict(Counter(frame["camera"].get("intrinsics_source") for frame in output_frames)),
        "calibration_contract": calibration_summary,
        "hand_state_source": "fresh_hawor_world_npz" if hawor_arrays else "none",
        "object_roster": sorted(object_plan),
        "mano_bridge": bridge_report,
        "claim_scope": "base physical measurement backbone only; final pose/contact/occlusion/interval correction state must be added by later V19 components",
    }
    state_path = args.output_dir / "v19_base_physical_state.json"
    write_json(state_path, physical_state)
    report = {
        "status": "ok",
        "method": "build_v19_base_annotations",
        "case": args.case,
        "outputs": {"annotations": str(annotations_path), "physical_state": str(state_path), "mano_bridge": bridge_report.get("path")},
        "frame_count": int(len(output_frames)),
        "camera_source_count": physical_state["camera_source_count"],
        "intrinsics_source_count": physical_state["intrinsics_source_count"],
        "calibration_contract": calibration_summary,
        "counts": dict(counts),
        "object_track_count": int(len(object_plan)),
        "sam2_track_count": int(len(sam2_tracks)),
        "mano_bridge": bridge_report,
    }
    write_json(args.output_dir / "v19_base_annotations_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-npz", type=Path, default=None, help="DROID or other camera trajectory NPZ with T_world_camera or R_c2w/t_c2w")
    parser.add_argument("--depth-npz", type=Path, default=None, help="Depth NPZ supplying per-frame intrinsics_fx_fy_cx_cy")
    parser.add_argument("--calibration-contract", type=Path, default=None, help="V19 camera calibration contract JSON. When supplied, its constant intrinsics override per-frame depth/model intrinsics.")
    parser.add_argument("--hawor-npz", type=Path, default=None, help="Fresh HaWoR world MANO export NPZ")
    parser.add_argument("--prefer-camera-npz", action="store_true", help="Prefer --camera-npz poses over HaWoR camera poses. Use only when the camera NPZ world frame is explicitly aligned to the HaWoR/MANO world frame.")
    parser.add_argument("--object-plan", type=Path, default=None, help="Agent-written object plan JSON")
    parser.add_argument("--sam2-track", action="append", default=[], help="Repeat TRACK_ID=PATH or pass */sam2/sam2_track.json")
    parser.add_argument("--sam2-output-root", type=Path, default=None, help="Root containing <track_id>/sam2/sam2_track.json outputs")
    parser.add_argument("--remote-root", type=Path, default=None, help="Remote path prefix to localize server-produced SAM2 mask paths")
    parser.add_argument("--local-root", type=Path, default=None, help="Local path prefix corresponding to --remote-root")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
