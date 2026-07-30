#!/usr/bin/env python3
"""Build a MANO-first foundation audit/state for V18.

This script recovers actual MANO surface candidates from raw hand-model outputs
(WiLoR and HaWoR where available), transforms camera-space surfaces into the V18
world frame, and writes an explicit validity gate. It is intentionally upstream
of contact/occlusion/nonpenetration: if this gate is false, downstream physical
claims involving hands remain diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HAND_SIDES = {"left", "right"}
EXPECTED_JOINTS = 21
EXPECTED_VERTICES = 778


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def arr(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if shape is not None and tuple(out.shape) != shape:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


def as_list(value: Any, n: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != n:
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def transform_points(T_world_camera: np.ndarray, pts_camera_local: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    pts_cam = pts_camera_local.astype(np.float32) + cam_t.astype(np.float32)[None, :]
    hom = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    return (hom @ T_world_camera.astype(np.float32).T)[:, :3]


def project_points(pts_camera_local: np.ndarray, cam_t: np.ndarray, intr: list[float]) -> np.ndarray:
    pts = pts_camera_local.astype(np.float64) + cam_t.astype(np.float64)[None, :]
    fx, fy, cx, cy = [float(v) for v in intr]
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    ok = z > 1e-6
    out[ok, 0] = fx * pts[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * pts[ok, 1] / z[ok] + cy
    return out


def percentile(vals: list[float], q: float) -> float | None:
    good = [float(v) for v in vals if math.isfinite(float(v))]
    if not good:
        return None
    return float(np.percentile(np.asarray(good, dtype=np.float64), q))


def frame_camera_index(full_ann: dict[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for frame in full_ann.get("frames", []):
        if not isinstance(frame, dict):
            continue
        idx = frame.get("frame_idx")
        cam = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        T = arr(cam.get("T_world_camera_metric"), (4, 4))
        if isinstance(idx, int) and T is not None:
            out[idx] = T.astype(np.float32)
    return out


def measurement_manifest(case: str, root: Path) -> dict[str, Any]:
    return load_json(root / case / "v17_measurement_manifest.json")


def wilor_raw_path(case: str, manifest_root: Path) -> Path | None:
    manifest = measurement_manifest(case, manifest_root)
    raw = manifest.get("wilor_raw")
    return Path(raw) if isinstance(raw, str) and raw else None


def hawor_source_paths(case: str, manifest_root: Path) -> list[Path]:
    manifest = measurement_manifest(case, manifest_root)
    rows = manifest.get("hawor_sources")
    out: list[Path] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("path"):
                out.append(Path(str(row["path"])))
    return out


def mano_param_arrays(params: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return source-native MANO parameter arrays flattened for storage.

    WiLoR stores rotations as matrices (`global_orient`: 1x3x3,
    `hand_pose`: 15x3x3). HaWoR stores axis-angle arrays. Both are real MANO
    parameterizations; rejecting matrix form would falsely classify available
    MANO state as absent.
    """
    orient_raw = params.get("global_orient") if "global_orient" in params else params.get("global_orient_axis_angle")
    pose_raw = params.get("hand_pose") if "hand_pose" in params else params.get("hand_pose_axis_angle")
    betas_raw = params.get("betas")
    orient = arr(orient_raw)
    pose = arr(pose_raw)
    betas = arr(betas_raw)
    if orient is not None:
        orient = orient.reshape(-1).astype(np.float32)
    if pose is not None:
        pose = pose.reshape(-1).astype(np.float32)
    if betas is not None:
        betas = betas.reshape(-1).astype(np.float32)
    return orient, pose, betas


def extract_wilor_case(case: str, full_ann: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw_path = wilor_raw_path(case, args.measurement_manifest_root)
    frame_count = int(full_ann.get("frame_count", 0))
    width = int(full_ann.get("raw_video", {}).get("width", 1920))
    height = int(full_ann.get("raw_video", {}).get("height", 1080))
    cameras = frame_camera_index(full_ann)
    result: dict[str, Any] = {
        "source_model": "WiLoR",
        "source_path": str(raw_path) if raw_path else None,
        "source_exists": bool(raw_path and raw_path.exists()),
        "rows_total": 0,
        "complete_surface_param_rows": 0,
        "complete_virtual_camera_candidate_rows": 0,
        "rows_by_side": {},
        "frames_by_side": {},
        "wilor_internal_projection_residual_px_median": None,
        "wilor_internal_projection_residual_px_p95": None,
        "npz_path": None,
        "blocking_reason_if_not_foundational": [],
    }
    if raw_path is None or not raw_path.exists():
        result["blocking_reason_if_not_foundational"].append("wilor_raw_source_missing")
        return result
    payload = load_json(raw_path)
    frames = payload.get("frames") if isinstance(payload, dict) else []
    if not isinstance(frames, list):
        result["blocking_reason_if_not_foundational"].append("wilor_raw_frames_missing")
        return result

    frame_idxs: list[int] = []
    side_codes: list[int] = []
    detector_scores: list[float] = []
    bboxes: list[list[float]] = []
    cam_ts: list[np.ndarray] = []
    intrinsics: list[list[float]] = []
    camera_transforms: list[np.ndarray] = []
    joints_world: list[np.ndarray] = []
    vertices_world: list[np.ndarray] = []
    global_orients: list[np.ndarray] = []
    hand_poses: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    residuals: list[float] = []
    by_side: Counter[str] = Counter()
    frames_by_side: dict[str, set[int]] = defaultdict(set)
    status_counts: Counter[str] = Counter()

    for raw_frame in frames:
        if not isinstance(raw_frame, dict):
            continue
        frame_idx = raw_frame.get("frame_idx")
        if not isinstance(frame_idx, int) or frame_idx < 0 or frame_idx >= frame_count:
            continue
        T = cameras.get(frame_idx)
        for raw_hand in raw_frame.get("raw_hands", []) if isinstance(raw_frame.get("raw_hands"), list) else []:
            if not isinstance(raw_hand, dict):
                continue
            side = raw_hand.get("side")
            if side not in HAND_SIDES:
                status_counts["bad_side"] += 1
                continue
            result["rows_total"] += 1
            joints = arr(raw_hand.get("joints3d_camera"), (EXPECTED_JOINTS, 3))
            vertices = arr(raw_hand.get("vertices_camera"), (EXPECTED_VERTICES, 3))
            cam_t = arr(raw_hand.get("cam_t"), (3,))
            bbox = as_list(raw_hand.get("bbox_xyxy"), 4)
            params = raw_hand.get("mano_params") if isinstance(raw_hand.get("mano_params"), dict) else {}
            orient, pose, beta = mano_param_arrays(params)
            if joints is None or vertices is None or cam_t is None or orient is None or pose is None or beta is None or bbox is None:
                status_counts["missing_required_mano_surface_or_params"] += 1
                continue
            result["complete_surface_param_rows"] += 1
            if T is None:
                status_counts["missing_v18_world_camera_transform"] += 1
                continue
            focal = raw_hand.get("focal_length")
            if isinstance(focal, list) and len(focal) >= 2:
                intr = [finite_float(focal[0]), finite_float(focal[1]), width / 2.0, height / 2.0]
            else:
                f = finite_float(focal, 2304.0)
                intr = [f, f, width / 2.0, height / 2.0]
            j_proj = project_points(joints, cam_t, intr)
            j2d = arr(raw_hand.get("joints2d"), (EXPECTED_JOINTS, 2))
            if j2d is not None and np.all(np.isfinite(j_proj)):
                per_joint = np.linalg.norm(j_proj - j2d.astype(np.float64), axis=1)
                residuals.append(float(np.median(per_joint)))
            frame_idxs.append(frame_idx)
            side_codes.append(0 if side == "left" else 1)
            detector_scores.append(finite_float(raw_hand.get("detector_score"), 0.0))
            bboxes.append(bbox)
            cam_ts.append(cam_t.astype(np.float32))
            intrinsics.append(intr)
            camera_transforms.append(T.astype(np.float32))
            joints_world.append(transform_points(T, joints, cam_t))
            vertices_world.append(transform_points(T, vertices, cam_t))
            global_orients.append(orient.astype(np.float32).reshape(-1))
            # Normalize variable-length hand pose/betas by preserving observed lengths; current WiLoR uses fixed lengths.
            hand_poses.append(pose.astype(np.float32).reshape(-1))
            betas.append(beta.astype(np.float32).reshape(-1))
            by_side[str(side)] += 1
            frames_by_side[str(side)].add(frame_idx)
            status_counts["world_surface_param_candidate"] += 1

    unique_frame_side_rows = sum(len(v) for v in frames_by_side.values())
    result["complete_virtual_camera_candidate_rows"] = len(frame_idxs)
    result["unique_virtual_camera_frame_side_rows"] = unique_frame_side_rows
    result["duplicate_virtual_camera_candidate_rows"] = len(frame_idxs) - unique_frame_side_rows
    result["rows_by_side"] = dict(by_side)
    result["frames_by_side"] = {k: {"count": len(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in sorted(frames_by_side.items())}
    result["status_counts"] = dict(status_counts)
    result["wilor_internal_projection_residual_px_median"] = percentile(residuals, 50.0)
    result["wilor_internal_projection_residual_px_p95"] = percentile(residuals, 95.0)
    result["source_sha256"] = sha256(raw_path) if args.hash_sources else None
    result["coordinate_status"] = "wilor_virtual_camera_surface_transformed_by_v18_camera_pose_not_metric_depth_aligned"
    result["metric_world_alignment_valid"] = False
    result["metric_world_alignment_blocker"] = "WiLoR cam_t/focal_length are virtual-camera scale; projection self-consistency does not prove metric depth or contact-scale world alignment"
    cam_t_z = [float(v[2]) for v in cam_ts]
    result["wilor_virtual_camera_cam_t_z_median"] = percentile(cam_t_z, 50.0)
    result["wilor_virtual_camera_cam_t_z_p95"] = percentile(cam_t_z, 95.0)
    if frame_idxs:
        # Pad pose arrays only if fixed; otherwise object arrays would make downstream use ambiguous.
        orient_lens = {len(x) for x in global_orients}
        pose_lens = {len(x) for x in hand_poses}
        beta_lens = {len(x) for x in betas}
        if len(orient_lens) != 1 or len(pose_lens) != 1 or len(beta_lens) != 1:
            raise RuntimeError(f"{case}: variable MANO orient/pose/beta lengths not supported: {orient_lens} {pose_lens} {beta_lens}")
        case_dir = args.output_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        legacy_npz = case_dir / "wilor_mano_world_candidates.npz"
        if legacy_npz.exists():
            legacy_npz.unlink()
        npz_path = case_dir / "wilor_mano_virtual_camera_candidates.npz"
        np.savez_compressed(
            npz_path,
            frame_idx=np.asarray(frame_idxs, dtype=np.int32),
            hand_side_code=np.asarray(side_codes, dtype=np.int8),
            detector_score=np.asarray(detector_scores, dtype=np.float32),
            bbox_xyxy=np.asarray(bboxes, dtype=np.float32),
            wilor_virtual_camera_intrinsics=np.asarray(intrinsics, dtype=np.float32),
            T_world_camera_metric=np.stack(camera_transforms, axis=0).astype(np.float32),
            cam_t=np.stack(cam_ts, axis=0).astype(np.float32),
            joints_v18_pose_transformed_from_wilor_virtual_camera=np.stack(joints_world, axis=0).astype(np.float32),
            vertices_v18_pose_transformed_from_wilor_virtual_camera=np.stack(vertices_world, axis=0).astype(np.float32),
            mano_global_orient=np.stack(global_orients, axis=0).astype(np.float32),
            mano_hand_pose=np.stack(hand_poses, axis=0).astype(np.float32),
            mano_betas=np.stack(betas, axis=0).astype(np.float32),
        )
        result["npz_path"] = str(npz_path)
        result["npz_arrays"] = {
            "frame_idx": [len(frame_idxs)],
            "hand_side_code": [len(frame_idxs)],
            "wilor_virtual_camera_intrinsics": [len(frame_idxs), 4],
            "T_world_camera_metric": [len(frame_idxs), 4, 4],
            "joints_v18_pose_transformed_from_wilor_virtual_camera": [len(frame_idxs), EXPECTED_JOINTS, 3],
            "vertices_v18_pose_transformed_from_wilor_virtual_camera": [len(frame_idxs), EXPECTED_VERTICES, 3],
            "mano_global_orient": [len(frame_idxs), int(next(iter(orient_lens)))],
            "mano_hand_pose": [len(frame_idxs), int(next(iter(pose_lens)))],
            "mano_betas": [len(frame_idxs), int(next(iter(beta_lens)))],
        }
    expected_rows = frame_count * 2
    result["timeline_expected_two_hand_rows"] = expected_rows
    result["virtual_camera_candidate_row_fraction_of_two_hand_timeline"] = result["complete_virtual_camera_candidate_rows"] / expected_rows if expected_rows else 0.0
    result["unique_virtual_camera_frame_side_fraction_of_two_hand_timeline"] = result["unique_virtual_camera_frame_side_rows"] / expected_rows if expected_rows else 0.0
    if result["unique_virtual_camera_frame_side_rows"] < expected_rows:
        result["blocking_reason_if_not_foundational"].append("wilor_missing_some_frame_side_rows_not_full_timeline")
    if result.get("metric_world_alignment_valid") is not True:
        result["blocking_reason_if_not_foundational"].append("wilor_virtual_camera_not_metric_world_aligned")
    if result["wilor_internal_projection_residual_px_median"] is None:
        result["blocking_reason_if_not_foundational"].append("wilor_projection_residual_unmeasured")
    elif result["wilor_internal_projection_residual_px_median"] > args.accept_projection_median_px:
        result["blocking_reason_if_not_foundational"].append("wilor_projection_residual_above_foundation_threshold")
    result["elapsed_s"] = time.perf_counter() - t0
    return result


def extract_hawor_case(case: str, args: argparse.Namespace, frame_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_model": "HaWoR",
        "source_paths": [],
        "source_exists_count": 0,
        "complete_world_surface_param_rows": 0,
        "measurement_available_complete_rows": 0,
        "motion_infill_complete_rows": 0,
        "frames_by_side": {},
        "blocking_reason_if_not_foundational": [],
    }
    paths = hawor_source_paths(case, args.measurement_manifest_root)
    result["source_paths"] = [str(p) for p in paths]
    frames_by_side: dict[str, set[int]] = defaultdict(set)
    status_counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            status_counts["source_missing"] += 1
            continue
        result["source_exists_count"] += 1
        payload = load_json(path)
        for frame in payload.get("frames", []) if isinstance(payload, dict) else []:
            if not isinstance(frame, dict):
                continue
            idx = frame.get("frame_idx")
            if not isinstance(idx, int) or idx < 0 or idx >= frame_count:
                continue
            for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
                if not isinstance(hand, dict):
                    continue
                side = hand.get("side")
                if side not in HAND_SIDES:
                    continue
                joints = arr(hand.get("joints3d_world_m"), (EXPECTED_JOINTS, 3))
                vertices = arr(hand.get("vertices_world_m"), (EXPECTED_VERTICES, 3))
                params = hand.get("mano_params") if isinstance(hand.get("mano_params"), dict) else {}
                orient, pose, beta = mano_param_arrays(params)
                if joints is not None and vertices is not None and orient is not None and pose is not None and beta is not None:
                    result["complete_world_surface_param_rows"] += 1
                    frames_by_side[str(side)].add(idx)
                    if hand.get("measurement_available") is True:
                        result["measurement_available_complete_rows"] += 1
                    else:
                        result["motion_infill_complete_rows"] += 1
                    status_counts["complete_world_surface_param_row"] += 1
    result["status_counts"] = dict(status_counts)
    result["frames_by_side"] = {k: {"count": len(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in sorted(frames_by_side.items())}
    expected_rows = frame_count * 2
    result["timeline_expected_two_hand_rows"] = expected_rows
    result["complete_world_row_fraction_of_two_hand_timeline"] = result["complete_world_surface_param_rows"] / expected_rows if expected_rows else 0.0
    if result["complete_world_surface_param_rows"] < expected_rows:
        result["blocking_reason_if_not_foundational"].append("hawor_not_full_timeline_complete_world_surface")
    if not paths:
        result["blocking_reason_if_not_foundational"].append("hawor_source_not_provisioned_or_not_run_for_case")
    return result


def max_gap(frames_by_side: dict[str, dict[str, Any]], frame_count: int) -> int | None:
    # Conservative gap from only count/min/max is not recoverable; return missing count lower bound instead.
    counts = [int(v.get("count", 0)) for v in frames_by_side.values() if isinstance(v, dict)]
    if not counts:
        return None
    return frame_count - min(counts)


def valid_surface_reference(ref: Any) -> bool:
    if not isinstance(ref, dict):
        return False
    npz_raw = ref.get("bridge_npz") or ref.get("npz")
    array = ref.get("bridge_vertices_world_array") or ref.get("array")
    row = ref.get("bridge_row_index") if "bridge_row_index" in ref else ref.get("row_index")
    if not (isinstance(npz_raw, str) and isinstance(array, str) and isinstance(row, int)):
        return False
    npz_path = Path(npz_raw)
    if not npz_path.exists():
        return False
    try:
        z = np.load(npz_path)
        if array not in z.files:
            return False
        shape = tuple(z[array].shape)
    except Exception:
        return False
    return len(shape) == 3 and 0 <= int(row) < shape[0] and shape[1:] == (EXPECTED_VERTICES, 3)


def valid_mano_param_contract(params: Any) -> bool:
    if not isinstance(params, dict):
        return False
    if arr(params.get("root_orient_axis_angle"), (3,)) is not None and arr(params.get("hand_pose_axis_angle"), (45,)) is not None and arr(params.get("betas"), (10,)) is not None and arr(params.get("trans_world_m"), (3,)) is not None:
        return True
    source_npz = params.get("source_hawor_npz")
    source_frame = params.get("source_frame_index")
    side = params.get("side")
    if not (isinstance(source_npz, str) and isinstance(source_frame, int) and side in HAND_SIDES and Path(source_npz).exists()):
        return False
    try:
        z = np.load(Path(source_npz), allow_pickle=True)
        required = {
            f"{side}_root_orient_axis_angle": (3,),
            f"{side}_hand_pose_axis_angle": (45,),
            f"{side}_betas": (10,),
            f"{side}_trans_world_m": (3,),
        }
        for name, trailing in required.items():
            if name not in z.files:
                return False
            shape = tuple(z[name].shape)
            if len(shape) != 2 or shape[1:] != trailing or not (0 <= source_frame < shape[0]):
                return False
        return True
    except Exception:
        return False


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    ann_path = args.v18_full_root / case / "annotations_v18_full.json"
    full_ann = load_json(ann_path)
    frame_count = int(full_ann.get("frame_count", 0))
    expected_two_hand_rows = frame_count * 2
    wilor = extract_wilor_case(case, full_ann, args)
    hawor = extract_hawor_case(case, args, frame_count)
    current_v18_rows = Counter()
    current_v18_sources = Counter()
    current_hawor_support_states = Counter()
    for frame in full_ann.get("frames", []) if isinstance(full_ann.get("frames"), list) else []:
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            current_v18_rows["hand_rows"] += 1
            mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else {}
            source = str(mano.get("source"))
            current_v18_sources[source] += 1
            support_state = hand.get("hawor_support_state")
            if isinstance(support_state, str):
                current_hawor_support_states[support_state] += 1
            if arr(mano.get("joints3d_camera"), (EXPECTED_JOINTS, 3)) is not None and arr(mano.get("cam_t"), (3,)) is not None:
                current_v18_rows["camera_joint_candidates"] += 1
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            if arr(mano.get("vertices_camera"), (EXPECTED_VERTICES, 3)) is not None or arr(mano.get("vertices_world_m"), (EXPECTED_VERTICES, 3)) is not None:
                current_v18_rows["surface_candidates_stored_in_v18_full"] += 1
            if valid_surface_reference(mano.get("surface_reference")) or valid_surface_reference(metric.get("vertices_reference")):
                current_v18_rows["surface_reference_rows_stored_in_v18_full"] += 1
            if isinstance(mano.get("mano_params"), dict):
                current_v18_rows["mano_params_stored_in_v18_full"] += 1
            surface_ok = valid_surface_reference(mano.get("surface_reference")) or valid_surface_reference(metric.get("vertices_reference"))
            params_ok = valid_mano_param_contract(mano.get("mano_params")) or valid_mano_param_contract(metric.get("mano_params"))
            if params_ok:
                current_v18_rows["reproducible_mano_param_rows_stored_in_v18_full"] += 1
            if str(source).startswith("HaWoR_metric_MANO") and surface_ok and params_ok:
                current_v18_rows["current_v18_hawor_surface_param_contract_rows"] += 1
    blockers: list[str] = []
    support_limitations: list[str] = []
    current_surface_rows = max(current_v18_rows.get("surface_candidates_stored_in_v18_full", 0), current_v18_rows.get("surface_reference_rows_stored_in_v18_full", 0))
    current_param_rows = max(current_v18_rows.get("mano_params_stored_in_v18_full", 0), current_v18_rows.get("reproducible_mano_param_rows_stored_in_v18_full", 0))
    if current_surface_rows < expected_two_hand_rows:
        blockers.append("current_v18_full_annotations_drop_mano_vertices")
    if current_param_rows < expected_two_hand_rows:
        blockers.append("current_v18_full_annotations_drop_mano_parameters")
    current_hawor_contract_rows = int(current_v18_rows.get("current_v18_hawor_surface_param_contract_rows", 0))
    hawor["current_v18_hawor_surface_param_contract_rows"] = current_hawor_contract_rows
    hawor["current_v18_hawor_support_state_counts"] = dict(current_hawor_support_states)
    hawor["current_v18_same_frame_detection_rows"] = int(current_hawor_support_states.get("observed_same_frame_detection", 0))
    hawor["current_v18_inferred_or_gap_rows"] = int(sum(v for k, v in current_hawor_support_states.items() if k != "observed_same_frame_detection"))
    hawor["current_v18_temporal_boundary_fill_rows"] = int(current_hawor_support_states.get("temporal_boundary_fill", 0))
    if current_hawor_contract_rows < expected_two_hand_rows and hawor.get("complete_world_surface_param_rows", 0) < expected_two_hand_rows:
        blockers.append("hawor_complete_world_surface_not_full_two_hand_timeline")
    if hawor.get("current_v18_inferred_or_gap_rows", 0) > 0:
        support_limitations.append("hawor_valid_rows_include_inferred_without_same_frame_detection_support")
    if hawor.get("current_v18_temporal_boundary_fill_rows", 0) > 0:
        support_limitations.append("hawor_timeline_contains_explicit_temporal_boundary_fill_rows")
    if hawor.get("source_exists_count", 0) == 0 and current_hawor_contract_rows == 0:
        blockers.append("hawor_missing_for_case")
    if wilor.get("wilor_internal_projection_residual_px_median") is not None and float(wilor["wilor_internal_projection_residual_px_median"]) > args.accept_projection_median_px:
        blockers.append("recovered_wilor_projection_residual_above_foundation_threshold")
    foundational_valid = not blockers and current_hawor_contract_rows == expected_two_hand_rows
    report = {
        "method": "build_v18_mano_foundation_state",
        "case": case,
        "claim_scope": "MANO_first_foundation_state_and_validity_gate_not_contact_occlusion_or_object_pose_closure",
        "source_annotation": str(ann_path),
        "frame_count": frame_count,
        "expected_two_hand_rows": expected_two_hand_rows,
        "current_v18_full_mano_storage": {
            "counts": dict(current_v18_rows),
            "sources": dict(current_v18_sources),
            "interpretation": "V18 full must preserve either full MANO surfaces or reproducible full-surface references plus MANO parameters for every hand row; this storage contract is necessary but not sufficient for foundation acceptance.",
        },
        "recovered_wilor_virtual_camera_mano_candidates": wilor,
        "hawor_world_mano_candidates": hawor,
        "foundational_mano_state_valid": foundational_valid,
        "v18_physical_pipeline_valid_without_further_hand_work": False,
        "physical_downstream_claim_scope": "diagnostic_only_until_foundational_mano_state_valid_true_and_consumed_by_contact_occlusion_nonpenetration",
        "blocking_reasons": blockers,
        "support_limitations": support_limitations,
        "support_qualified_mano_foundation_valid": foundational_valid,
        "observed_same_frame_physical_support_complete": bool(hawor.get("current_v18_same_frame_detection_rows", 0) == expected_two_hand_rows),
        "physical_claim_policy": "observed contact occlusion and nonpenetration claims require observed_same_frame_detection hand support; inferred and boundary-filled rows are renderable continuity only",
        "elapsed_s": time.perf_counter() - t0,
    }
    case_dir = args.output_root / case
    write_json(case_dir / "v18_mano_foundation_state_report.json", report)
    return report


def write_markdown(root: Path, reports: list[dict[str, Any]]) -> None:
    lines = [
        "# V18 MANO foundation audit",
        "",
        "This is a hand-state foundation artifact, not a downstream contact/geometry audit.",
        "A V18 physical pipeline is invalid unless it has metric, time-indexed MANO state with surface/parameters, world-frame semantics, coverage, provenance, and QC.",
        "",
    ]
    for r in reports:
        case = r["case"]
        wilor = r["recovered_wilor_virtual_camera_mano_candidates"]
        hawor = r["hawor_world_mano_candidates"]
        current = r["current_v18_full_mano_storage"]
        lines += [
            f"## {case}",
            "",
            f"- Foundational MANO valid: `{r['foundational_mano_state_valid']}`.",
            f"- Current V18 full hand rows: `{current['counts'].get('hand_rows', 0)}`; camera joint candidates: `{current['counts'].get('camera_joint_candidates', 0)}`; inline MANO surfaces: `{current['counts'].get('surface_candidates_stored_in_v18_full', 0)}`; surface references: `{current['counts'].get('surface_reference_rows_stored_in_v18_full', 0)}`; inline MANO params: `{current['counts'].get('mano_params_stored_in_v18_full', 0)}`; reproducible MANO param rows: `{current['counts'].get('reproducible_mano_param_rows_stored_in_v18_full', 0)}`.",
            f"- Current V18 HaWoR surface/parameter contract rows: `{hawor.get('current_v18_hawor_surface_param_contract_rows', 0)}/{r['expected_two_hand_rows']}`; support states: `{hawor.get('current_v18_hawor_support_state_counts', {})}`.",
            f"- Recovered WiLoR full virtual-camera MANO candidates: `{wilor.get('complete_virtual_camera_candidate_rows', 0)}` raw rows, `{wilor.get('unique_virtual_camera_frame_side_rows', 0)}/{r['expected_two_hand_rows']}` unique frame-side rows; side frames: `{wilor.get('frames_by_side')}`; internal projection residual px: `{wilor.get('wilor_internal_projection_residual_px_median')}`; metric-world alignment valid: `{wilor.get('metric_world_alignment_valid')}`; NPZ: `{wilor.get('npz_path')}`.",
            f"- Legacy HaWoR complete world MANO rows: `{hawor.get('complete_world_surface_param_rows', 0)}/{r['expected_two_hand_rows']}`; measurement rows: `{hawor.get('measurement_available_complete_rows')}`; motion-infill rows: `{hawor.get('motion_infill_complete_rows')}`; side frames: `{hawor.get('frames_by_side')}`.",
            f"- Support limitations: `{r.get('support_limitations', [])}`.",
            f"- Blocking reasons: `{r['blocking_reasons']}`.",
            "",
        ]
    lines += [
        "## Current commitment",
        "",
        "Recovered WiLoR virtual-camera surfaces and current HaWoR-backed final MANO contracts are real MANO evidence, but V18 is still not physically valid unless the HaWoR contract covers the full two-hand timeline with acceptable support/alignment semantics and downstream physical solvers consume that qualified foundation.",
    ]
    (root / "V18_MANO_FOUNDATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18-full-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--measurement-manifest-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/mano_foundation_audit"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--accept-projection-median-px", type=float, default=5.0)
    parser.add_argument("--hash-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_mano_foundation_state",
        "claim_scope": "MANO_first_foundation_state_and_validity_gate_not_full_v18_closure",
        "output_root": str(args.output_root),
        "cases": reports,
        "all_cases_foundational_mano_valid": all(r.get("foundational_mano_state_valid") is True for r in reports),
        "v18_physical_pipeline_valid_without_further_hand_work": False,
        "elapsed_s": time.perf_counter() - t0,
    }
    write_json(args.output_root / "v18_mano_foundation_audit_summary.json", summary)
    write_markdown(args.output_root, reports)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
