#!/usr/bin/env python3
"""Promote scale-preserving MANO mask/depth refits into V19 state.

The input refit annotations are diagnostic camera-frame outputs from
``refit_mano_articulation_mask_depth_v3.py``.  This script promotes only fits
that remain representable as a MANO pose plus one camera/world translation:
``vertices_source_camera_m ~= vertices_camera + trans_camera``.  Promoted fits
update a copied V19 bridge NPZ, a copied HaWoR-like source NPZ, and a copied V19
annotations file so downstream interval solvers/renderers consume the refit as
state rather than as a sidecar report.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def frame_map(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations must contain frames list")
    return {int(frame["frame_idx"]): frame for frame in frames if isinstance(frame, dict)}


def frame_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    mat = np.asarray(camera.get("T_world_camera_metric"), dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        mat = np.asarray(camera.get("T_world_camera"), dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks finite T_world_camera_metric")
    return mat, mat[:3, :3].astype(float), mat[:3, 3].astype(float)


def hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        if str(hand.get("hand_side") or hand.get("side") or metric.get("hand_side") or "") == side:
            return hand
    return None


def fitted_hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        if str(hand.get("source_v19_hand_side") or hand.get("side") or "") == side:
            return hand
    return None


def project_camera(points_camera: np.ndarray, intr: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=float)
    z = points[:, 2]
    out = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-6)
    fx, fy, cx, cy = intr.astype(float)
    out[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    out[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return out


def load_refit_frames(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return frame_map(load_json(path))


def load_qc_rows(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    rows: dict[int, dict[str, Any]] = {}
    for key in ("rows_preview", "all_rows_preview"):
        for row in payload.get(key, []) if isinstance(payload.get(key), list) else []:
            if isinstance(row, dict) and row.get("selected_for_annotation_stream", True):
                rows[int(row["frame_idx"])] = dict(row)
    return rows


def copy_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]).copy() for key in z.files}


def first_bridge_path(v19: dict[str, Any]) -> Path:
    for frame in v19.get("frames", []) if isinstance(v19.get("frames"), list) else []:
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            ref = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
            raw = ref.get("bridge_npz")
            if isinstance(raw, str):
                return Path(raw)
    raise RuntimeError("no bridge_npz reference found in V19 annotations")


def sample_vertices(vertices: np.ndarray, count: int = 64) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= count:
        ids = np.arange(len(vertices), dtype=np.int32)
    else:
        ids = np.linspace(0, len(vertices) - 1, count, dtype=np.int32)
    return vertices[ids], ids


def promote(args: argparse.Namespace) -> dict[str, Any]:
    v19 = load_json(args.v19_annotations)
    out = copy.deepcopy(v19)
    out_frames = frame_map(out)
    original_bridge = first_bridge_path(v19)
    bridge = copy_npz(original_bridge)
    source = copy_npz(args.source_hawor_npz)
    source_frame_indices = np.asarray(source.get("frame_idx"), dtype=int)
    if source_frame_indices.ndim != 1:
        raise RuntimeError("source_hawor_npz lacks 1D frame_idx")
    refit_frames = {
        "left": load_refit_frames(args.left_refit_annotations),
        "right": load_refit_frames(args.right_refit_annotations),
    }
    qc_rows = {
        "left": load_qc_rows(args.left_refit_qc),
        "right": load_qc_rows(args.right_refit_qc),
    }
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        frame = out_frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_v19_frame"})
            continue
        _T, r_c2w, t_c2w = frame_camera_pose(frame)
        for side in ("left", "right"):
            refit_frame = refit_frames[side].get(frame_idx)
            if refit_frame is None:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_refit_frame"})
                continue
            fitted = fitted_hand_by_side(refit_frame, side)
            target = hand_by_side(frame, side)
            if fitted is None or target is None:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_fitted_or_target_hand"})
                continue
            metric = target.get("metric_mano_state") if isinstance(target.get("metric_mano_state"), dict) else {}
            ref = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
            params0 = metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else {}
            try:
                row_index = int(ref["bridge_row_index"])
                source_index = int(params0.get("source_frame_index", ref.get("source_frame_index")))
                if source_index < 0 or source_index >= len(source_frame_indices):
                    raise RuntimeError(f"source_frame_index {source_index} out of range")
                vertices_local = np.asarray(fitted["vertices_camera"], dtype=float)
                joints_local = np.asarray(fitted["joints3d_camera"], dtype=float)
                vertices_camera = np.asarray(fitted["vertices_source_camera_m"], dtype=float)
                joints_camera = np.asarray(fitted["joints3d_source_camera_m"], dtype=float)
                if vertices_local.shape != vertices_camera.shape or vertices_camera.ndim != 2 or vertices_camera.shape[1] != 3:
                    raise RuntimeError("invalid fitted vertex shapes")
                if joints_local.shape != joints_camera.shape or joints_camera.shape != (21, 3):
                    raise RuntimeError("invalid fitted joint shapes")
                delta_vertices = vertices_camera - vertices_local
                trans_camera = np.median(delta_vertices, axis=0)
                translation_residual = np.linalg.norm(delta_vertices - trans_camera[None, :], axis=1)
                residual_p95 = float(np.percentile(translation_residual, 95.0))
                residual_max = float(np.max(translation_residual))
                refit_meta = fitted.get("v3_mano_articulation_mask_depth_refit") if isinstance(fitted.get("v3_mano_articulation_mask_depth_refit"), dict) else {}
                scale = float(refit_meta.get("scale", qc_rows[side].get(frame_idx, {}).get("scale", 1.0)))
                if abs(scale - 1.0) > float(args.max_scale_deviation):
                    raise RuntimeError(f"scale_not_promotable scale={scale:.6f}")
                if residual_p95 > float(args.max_translation_residual_p95_m) or residual_max > float(args.max_translation_residual_max_m):
                    raise RuntimeError(f"non_rigid_scale_or_translation_residual p95={residual_p95:.6g} max={residual_max:.6g}")
                root_camera = np.asarray(fitted["mano_params"]["global_orient"], dtype=float).reshape(3, 3)
                hand_pose_mats = np.asarray(fitted["mano_params"]["hand_pose"], dtype=float).reshape(15, 3, 3)
                betas = np.asarray(fitted["mano_params"]["betas"], dtype=float).reshape(10)
                root_world = r_c2w @ root_camera
                root_axis = Rotation.from_matrix(root_world).as_rotvec().astype(np.float32)
                hand_pose_axis = Rotation.from_matrix(hand_pose_mats.reshape(-1, 3, 3)).as_rotvec().reshape(45).astype(np.float32)
                trans_world = (trans_camera @ r_c2w.T + t_c2w).astype(np.float32)
                vertices_world = (vertices_camera @ r_c2w.T + t_c2w[None, :]).astype(np.float32)
                joints_world = (joints_camera @ r_c2w.T + t_c2w[None, :]).astype(np.float32)
                vertices_camera = vertices_camera.astype(np.float32)
                joints_camera = joints_camera.astype(np.float32)
                for key, value in (
                    (f"{side}_vertices_world_m", vertices_world),
                    (f"{side}_joints_world_m", joints_world),
                    (f"{side}_root_orient_axis_angle", root_axis),
                    (f"{side}_hand_pose_axis_angle", hand_pose_axis),
                    (f"{side}_betas", betas.astype(np.float32)),
                    (f"{side}_trans_world_m", trans_world),
                ):
                    if key not in source:
                        raise RuntimeError(f"source npz lacks {key}")
                    source[key][source_index] = value
                if f"{side}_valid" in source:
                    source[f"{side}_valid"][source_index] = 1
                bridge["vertices_current_v18_world_from_hawor_projection_relift_m"][row_index] = vertices_world
                bridge["joints_current_v18_world_from_hawor_projection_relift_m"][row_index] = joints_world
                bridge["vertices_current_v18_camera_m"][row_index] = vertices_camera
                bridge["joints_current_v18_camera_m"][row_index] = joints_camera
                if "vertices_hawor_camera_m" in bridge:
                    bridge["vertices_hawor_camera_m"][row_index] = vertices_camera
                sample, sample_ids = sample_vertices(vertices_world, 64)
                sample_cam = vertices_camera[sample_ids]
                intr = np.asarray(metric.get("v19_camera_intrinsics_fx_fy_cx_cy") or metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
                if intr.shape != (4,) or not np.isfinite(intr).all():
                    intr = np.asarray(frame.get("camera", {}).get("intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
                joints2d = project_camera(joints_camera, intr) if intr.shape == (4,) else np.full((21, 2), np.nan, dtype=float)
                metric.update(
                    {
                        "source": "apply_v19_mano_mask_depth_refit",
                        "coordinate_status": "metric_world_from_v19_mask_depth_refit_camera_pose",
                        "joints_current_v18_world_m": joints_world.astype(float).tolist(),
                        "joints_world_m": joints_world.astype(float).tolist(),
                        "joints_current_v18_camera_m": joints_camera.astype(float).tolist(),
                        "vertices_world_sample_m": sample.astype(float).tolist(),
                        "vertices_camera_sample_m": sample_cam.astype(float).tolist(),
                        "vertices_sample_indices": sample_ids.astype(int).tolist(),
                        "support_state": "v19_mask_depth_refit_candidate_uncertain",
                        "mask_depth_refit": {
                            "source_refit_annotations": str(args.left_refit_annotations if side == "left" else args.right_refit_annotations),
                            "source_refit_qc": str(args.left_refit_qc if side == "left" else args.right_refit_qc),
                            "measurement_available": bool(fitted.get("measurement_available", False)),
                            "filter_status": fitted.get("filter_status"),
                            "scale": scale,
                            "translation_residual_p95_m": residual_p95,
                            "translation_residual_max_m": residual_max,
                            "joints2d_after_refit": joints2d.astype(float).tolist(),
                            "refit_metrics": refit_meta,
                        },
                    }
                )
                reference = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
                reference.update(
                    {
                        "bridge_npz": str(args.output_bridge_npz),
                        "source_hawor_npz": str(args.output_source_npz),
                        "source_vertices_world_array": f"{side}_vertices_world_m",
                        "source_joints_world_array": f"{side}_joints_world_m",
                        "v19_bridge_status": "v19_mask_depth_refit_bridge_from_scale_preserving_sam2_depth_fit",
                    }
                )
                metric["vertices_reference"] = reference
                mano_params = metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else {}
                mano_params.update(
                    {
                        "parameterization": "v19_mask_depth_refit_axis_angle_world_export",
                        "source_hawor_npz": str(args.output_source_npz),
                        "source_frame_index": int(source_index),
                        "side": side,
                        "arrays": {
                            "root_orient_axis_angle": f"{side}_root_orient_axis_angle",
                            "hand_pose_axis_angle": f"{side}_hand_pose_axis_angle",
                            "betas": f"{side}_betas",
                            "trans_world_m": f"{side}_trans_world_m",
                        },
                        "root_orient_axis_angle": root_axis.astype(float).tolist(),
                        "hand_pose_axis_angle": hand_pose_axis.astype(float).tolist(),
                        "betas": betas.astype(float).tolist(),
                        "trans_world_m": trans_world.astype(float).tolist(),
                    }
                )
                metric["mano_params"] = mano_params
                target["metric_mano_state"] = metric
                target["hand_geometry_source"] = "v19_mask_depth_refit_candidate_uncertain"
                target["visibility_state"] = "v19_mask_depth_refit_candidate_uncertain"
                uncertainty = list(target.get("uncertainty", [])) if isinstance(target.get("uncertainty"), list) else []
                if "mask_depth_refit_uncertain_hand_owned_surface" not in uncertainty:
                    uncertainty.append("mask_depth_refit_uncertain_hand_owned_surface")
                target["uncertainty"] = uncertainty
                promoted.append(
                    {
                        "frame_idx": frame_idx,
                        "side": side,
                        "bridge_row_index": row_index,
                        "source_frame_index": source_index,
                        "measurement_available": bool(fitted.get("measurement_available", False)),
                        "filter_status": fitted.get("filter_status"),
                        "scale": scale,
                        "translation_residual_p95_m": residual_p95,
                        "translation_residual_max_m": residual_max,
                        "trans_camera_m": trans_camera.astype(float).tolist(),
                        "trans_world_m": trans_world.astype(float).tolist(),
                    }
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": str(exc)})
    args.output_bridge_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bridge_npz, **bridge)
    np.savez_compressed(args.output_source_npz, **source)
    write_json(args.output_annotations, out)
    promoted_arr = np.asarray([1.0 for _ in promoted], dtype=float)
    report = {
        "status": "ok",
        "method": "apply_v19_mano_mask_depth_refit",
        "claim_scope": "promotes only scale-preserving camera-frame MANO refits into V19 bridge/source annotations; downstream solver/render still decide physical usefulness",
        "source_v19_annotations": str(args.v19_annotations),
        "original_bridge_npz": str(original_bridge),
        "source_hawor_npz": str(args.source_hawor_npz),
        "output_annotations": str(args.output_annotations),
        "output_bridge_npz": str(args.output_bridge_npz),
        "output_source_npz": str(args.output_source_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "promoted_count": int(len(promoted)),
        "skipped_count": int(len(skipped)),
        "promoted_measurement_available_count": int(sum(1 for row in promoted if row.get("measurement_available"))),
        "parameters": {
            "max_scale_deviation": float(args.max_scale_deviation),
            "max_translation_residual_p95_m": float(args.max_translation_residual_p95_m),
            "max_translation_residual_max_m": float(args.max_translation_residual_max_m),
        },
        "promoted_preview": promoted[:160],
        "skipped_preview": skipped[:160],
    }
    write_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"promoted_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v19-annotations", type=Path, required=True)
    parser.add_argument("--source-hawor-npz", type=Path, required=True)
    parser.add_argument("--left-refit-annotations", type=Path)
    parser.add_argument("--right-refit-annotations", type=Path)
    parser.add_argument("--left-refit-qc", type=Path)
    parser.add_argument("--right-refit-qc", type=Path)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-bridge-npz", type=Path, required=True)
    parser.add_argument("--output-source-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--max-scale-deviation", type=float, default=0.020)
    parser.add_argument("--max-translation-residual-p95-m", type=float, default=2.0e-4)
    parser.add_argument("--max-translation-residual-max-m", type=float, default=8.0e-4)
    return parser.parse_args()


def main() -> None:
    promote(parse_args())


if __name__ == "__main__":
    main()
