#!/usr/bin/env python3
"""Prepare V19 MANO mask/depth refit inputs.

This is an adapter, not canonical state.  It converts V19 metric MANO state
(world-frame HaWoR export plus bridge vertices) into the legacy camera-frame
annotation contract consumed by ``refit_mano_articulation_mask_depth_v3.py``.
It also filters SAM2 hand masks into visible MANO-hand support by intersecting
model-produced hand masks with a dilated calibrated MANO projection and
subtracting object-owned pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HAND_EDGES = np.asarray(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    dtype=np.int32,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def read_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    if image.ndim == 3:
        image = image[..., 0]
    mask = image > 0
    if target_shape is not None and mask.shape != target_shape:
        height, width = target_shape
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    return int(image.shape[0]), int(image.shape[1])


def frame_target_shape(frame: dict[str, Any], frame_idx: int, tracks: list[dict[str, Any] | None], object_track: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    for track in tracks:
        row = track.get(str(frame_idx)) if isinstance(track, dict) else None
        if isinstance(row, dict) and row.get("mask_path"):
            path = localize_path(str(row["mask_path"]), args.remote_root, args.local_root)
            return image_shape(path)
    obj_row = object_track.get(str(frame_idx)) if isinstance(object_track, dict) else None
    if isinstance(obj_row, dict) and obj_row.get("mask_path"):
        path = localize_path(str(obj_row["mask_path"]), args.remote_root, args.local_root)
        return image_shape(path)
    raw = frame.get("raw_frame_path")
    if isinstance(raw, str) and raw:
        path = localize_path(raw, args.remote_root, args.local_root)
        return image_shape(path)
    return int(args.source_height), int(args.source_width)


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_center(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def frame_map(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations must contain frames list")
    out: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        idx = int(frame["frame_idx"])
        if idx in out:
            raise RuntimeError(f"duplicate frame_idx {idx}")
        out[idx] = frame
    return out


def frame_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    mat = np.asarray(camera.get("T_world_camera_metric"), dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        mat = np.asarray(camera.get("T_world_camera"), dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks finite T_world_camera_metric")
    return mat, mat[:3, :3].astype(float), mat[:3, 3].astype(float)


def intrinsics_for_hand(frame: dict[str, Any], hand: dict[str, Any]) -> np.ndarray:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    for key in ("v19_camera_intrinsics_fx_fy_cx_cy", "current_v18_camera_intrinsics_fx_fy_cx_cy"):
        arr = np.asarray(metric.get(key), dtype=float).reshape(-1)
        if arr.shape == (4,) and np.isfinite(arr).all() and arr[0] > 0 and arr[1] > 0:
            return arr.astype(float)
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    arr = np.asarray(camera.get("intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
    if arr.shape == (4,) and np.isfinite(arr).all() and arr[0] > 0 and arr[1] > 0:
        return arr.astype(float)
    raise RuntimeError(f"frame {frame.get('frame_idx')} hand lacks finite intrinsics")


def bridge_arrays(reference: dict[str, Any], cache: dict[Path, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_path = reference.get("bridge_npz")
    row_raw = reference.get("bridge_row_index")
    if not isinstance(raw_path, str) or row_raw is None:
        raise RuntimeError("hand metric state lacks bridge_npz/bridge_row_index")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path not in cache:
        with np.load(path, allow_pickle=True) as z:
            cache[path] = {key: np.asarray(z[key]) for key in z.files}
    z = cache[path]
    row = int(row_raw)
    vw_key = str(reference.get("bridge_vertices_world_array") or "vertices_current_v18_world_from_hawor_projection_relift_m")
    vc_key = str(reference.get("bridge_vertices_camera_array") or "vertices_current_v18_camera_m")
    jw_key = "joints_current_v18_world_from_hawor_projection_relift_m"
    jc_key = "joints_current_v18_camera_m"
    return (
        np.asarray(z[vw_key][row], dtype=float),
        np.asarray(z[jw_key][row], dtype=float),
        np.asarray(z[vc_key][row], dtype=float),
        np.asarray(z[jc_key][row], dtype=float),
    )


def project_camera(points_camera: np.ndarray, intr: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=float)
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-6)
    fx, fy, cx, cy = intr.astype(float)
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv


def projection_support_mask(
    vertices_camera: np.ndarray,
    joints_camera: np.ndarray,
    intr: np.ndarray,
    dilation_px: int,
    source_size: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    height, width = target_shape
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid source/target sizes source={source_size} target={target_shape}")
    mask = np.zeros((height, width), dtype=np.uint8)
    uv_vertices = project_camera(vertices_camera, intr)
    uv_joints = project_camera(joints_camera, intr)
    scale = np.asarray([float(width) / float(source_width), float(height) / float(source_height)], dtype=float)
    margin = float(max(width, height)) * 0.25
    pts = uv_vertices[np.isfinite(uv_vertices).all(axis=1)] * scale[None, :]
    pts = pts[(pts[:, 0] >= -margin) & (pts[:, 0] <= width + margin) & (pts[:, 1] >= -margin) & (pts[:, 1] <= height + margin)]
    if len(pts) >= 3:
        hull = cv2.convexHull(np.rint(pts).astype(np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 1)
    joints = uv_joints * scale[None, :]
    for a, b in HAND_EDGES:
        pa = joints[a]
        pb = joints[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(mask, tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int)), 1, 10, cv2.LINE_AA)
    for point in joints:
        if np.isfinite(point).all():
            cv2.circle(mask, tuple(np.rint(point).astype(int)), 8, 1, -1, cv2.LINE_AA)
    if int(dilation_px) > 0:
        radius = int(dilation_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def object_mask_for_frame(track: dict[str, Any], frame_idx: int, args: argparse.Namespace, target_shape: tuple[int, int]) -> np.ndarray | None:
    row = track.get(str(frame_idx))
    if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
        return None
    path = localize_path(str(row["mask_path"]), args.remote_root, args.local_root)
    mask = read_mask(path, target_shape)
    if int(args.object_mask_dilation_px_960) > 0:
        r = int(args.object_mask_dilation_px_960)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    return mask


def hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        if str(hand.get("hand_side") or hand.get("side") or "") == side:
            return hand
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        if str(metric.get("hand_side") or "") == side:
            return hand
    return None


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    frames = frame_map(annotations)
    left_track = load_json(args.left_hand_track) if args.left_hand_track is not None else None
    right_track = load_json(args.right_hand_track) if args.right_hand_track is not None else None
    object_track = load_json(args.object_track) if args.object_track is not None else {}
    bridge_cache: dict[Path, dict[str, np.ndarray]] = {}
    out_frames: list[dict[str, Any]] = []
    track_rows: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    diagnostics: list[dict[str, Any]] = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        frame = frames.get(frame_idx)
        if frame is None:
            diagnostics.append({"frame_idx": frame_idx, "state": "missing_annotation_frame"})
            continue
        T, r_c2w, t_c2w = frame_camera_pose(frame)
        source_width = int(frame.get("source_width") or args.source_width)
        source_height = int(frame.get("source_height") or args.source_height)
        target_shape = frame_target_shape(frame, frame_idx, [left_track, right_track], object_track, args)
        target_height, target_width = target_shape
        legacy_frame: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "time_s": frame.get("time_s"),
            "raw_frame_path": frame.get("raw_frame_path"),
            "camera": {"T_world_camera_metric": T.astype(float).tolist()},
            "object": {
                "source_image_size": [source_width, source_height],
                "mask_image_size": [target_width, target_height],
                "source_to_mask_scale_xy": [float(target_width) / float(source_width), float(target_height) / float(source_height)],
            },
            "hands": [],
        }
        object_mask = object_mask_for_frame(object_track, frame_idx, args, target_shape) if object_track else None
        for side, track in (("left", left_track), ("right", right_track)):
            hand = hand_by_side(frame, side)
            if hand is None:
                diagnostics.append({"frame_idx": frame_idx, "side": side, "state": "missing_hand"})
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            ref = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
            params = metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else {}
            try:
                vertices_world, joints_world, vertices_camera, joints_camera = bridge_arrays(ref, bridge_cache)
                intr = intrinsics_for_hand(frame, hand)
                root_world = np.asarray(params.get("root_orient_axis_angle"), dtype=float).reshape(3)
                pose_axis = np.asarray(params.get("hand_pose_axis_angle"), dtype=float).reshape(15, 3)
                betas = np.asarray(params.get("betas"), dtype=float).reshape(10)
                trans_world = np.asarray(params.get("trans_world_m"), dtype=float).reshape(3)
                if not np.isfinite(root_world).all() or not np.isfinite(pose_axis).all() or not np.isfinite(betas).all() or not np.isfinite(trans_world).all():
                    raise RuntimeError("non_finite_mano_params")
                root_world_mat = Rotation.from_rotvec(root_world).as_matrix()
                # Column-vector convention: X_world = R_c2w @ X_cam + t.  A root
                # rotation expressed in camera coordinates is R_cw @ R_world.
                root_camera_mat = r_c2w.T @ root_world_mat
                hand_pose_mats = Rotation.from_rotvec(pose_axis.reshape(-1, 3)).as_matrix().reshape(15, 3, 3)
                trans_camera = (trans_world - t_c2w) @ r_c2w
                local_vertices = vertices_camera - trans_camera[None, :]
                local_joints = joints_camera - trans_camera[None, :]
                joints2d = project_camera(joints_camera, intr)
                legacy_frame["hands"].append(
                    {
                        "side": side,
                        "track_id": f"v19_{side}_hand_sam2_mano_filtered",
                        "source_v19_hand_side": side,
                        "source_v19_frame_idx": int(frame_idx),
                        "source_v19_bridge_npz": ref.get("bridge_npz"),
                        "source_v19_bridge_row_index": ref.get("bridge_row_index"),
                        "source_intrinsics": intr.astype(float).tolist(),
                        "joints3d_camera": local_joints.astype(float).tolist(),
                        "vertices_camera": local_vertices.astype(float).tolist(),
                        "joints3d_source_camera_m": joints_camera.astype(float).tolist(),
                        "vertices_source_camera_m": vertices_camera.astype(float).tolist(),
                        "joints3d_world_m": joints_world.astype(float).tolist(),
                        "vertices_world_m": vertices_world.astype(float).tolist(),
                        "joints2d": joints2d.astype(float).tolist(),
                        "mano_params": {
                            "global_orient": root_camera_mat.reshape(1, 3, 3).astype(float).tolist(),
                            "hand_pose": hand_pose_mats.astype(float).tolist(),
                            "betas": betas.astype(float).tolist(),
                            "rotation_convention": "v19_world_root_converted_to_camera_rotation_matrix_pose2rot_false",
                            "source_world_root_orient_axis_angle": root_world.astype(float).tolist(),
                            "source_world_trans_m": trans_world.astype(float).tolist(),
                            "source_camera_trans_m": trans_camera.astype(float).tolist(),
                        },
                    }
                )
                if track is None:
                    diagnostics.append({"frame_idx": frame_idx, "side": side, "state": "missing_hand_track_arg"})
                    continue
                row = track.get(str(frame_idx))
                if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
                    diagnostics.append({"frame_idx": frame_idx, "side": side, "state": "missing_sam2_hand_mask"})
                    continue
                sam2_path = localize_path(str(row["mask_path"]), args.remote_root, args.local_root)
                sam2_mask = read_mask(sam2_path, target_shape)
                prior_mask = projection_support_mask(
                    vertices_camera,
                    joints_camera,
                    intr,
                    int(args.mano_projection_dilation_px_960),
                    (source_width, source_height),
                    target_shape,
                )
                filtered = sam2_mask & prior_mask
                object_overlap_px = 0
                object_subtracted_px = 0
                if object_mask is not None:
                    object_overlap_px = int(np.count_nonzero(filtered & object_mask))
                    filtered_before = int(np.count_nonzero(filtered))
                    filtered = filtered & ~object_mask
                    object_subtracted_px = filtered_before - int(np.count_nonzero(filtered))
                # Keep only the largest connected support to avoid small SAM2 islands
                # becoming depth targets.
                num, labels, stats, _cent = cv2.connectedComponentsWithStats(filtered.astype(np.uint8), 8)
                if num > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    filtered = labels == largest
                area = int(np.count_nonzero(filtered))
                mask_dir = args.output_dir / "mask_tracks" / side / "sam2_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{frame_idx:06d}.png"
                if area > 0:
                    cv2.imwrite(str(mask_path), (filtered.astype(np.uint8) * 255))
                visible = area >= int(args.min_filtered_area_px_960)
                if visible:
                    track_rows[side][str(frame_idx)] = {
                        "visible": True,
                        "mask_path": str(mask_path),
                        "bbox_xyxy": mask_bbox(filtered),
                        "center_xy": mask_center(filtered),
                        "area_px": float(area),
                        "source_sam2_mask_path": str(sam2_path),
                        "filter": "sam2_hand_intersect_dilated_mano_projection" + ("_minus_object_mask" if object_mask is not None else ""),
                    }
                else:
                    track_rows[side][str(frame_idx)] = {
                        "visible": False,
                        "mask_path": None,
                        "area_px": float(area),
                        "source_sam2_mask_path": str(sam2_path),
                        "filter": "filtered_mask_too_small_no_fallback_to_full_sam2",
                    }
                diagnostics.append(
                    {
                        "frame_idx": frame_idx,
                        "side": side,
                        "state": "ok" if visible else "filtered_mask_too_small",
                        "target_mask_shape_hw": [int(target_height), int(target_width)],
                        "source_size_wh": [int(source_width), int(source_height)],
                        "source_to_mask_scale_xy": [float(target_width) / float(source_width), float(target_height) / float(source_height)],
                        "sam2_area_px": int(np.count_nonzero(sam2_mask)),
                        "mano_projection_area_px": int(np.count_nonzero(prior_mask)),
                        "filtered_area_px": area,
                        "object_overlap_px": object_overlap_px,
                        "object_subtracted_px": object_subtracted_px,
                        "mask_path": str(mask_path) if visible else None,
                    }
                )
            except Exception as exc:
                diagnostics.append({"frame_idx": frame_idx, "side": side, "state": "error", "reason": str(exc)})
        out_frames.append(legacy_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_annotations = {
        "status": "ok",
        "method": "build_v19_mano_mask_depth_refit_inputs",
        "claim_scope": "temporary camera-frame adapter for MANO mask/depth refit; not canonical V19 state",
        "source_annotations": str(args.annotations),
        "frames": out_frames,
    }
    left_path = args.output_dir / "mask_tracks" / "left" / "sam2_track.json"
    right_path = args.output_dir / "mask_tracks" / "right" / "sam2_track.json"
    ann_path = args.output_dir / "legacy_mano_refit_input_annotations.json"
    write_json(ann_path, legacy_annotations)
    write_json(left_path, track_rows["left"])
    write_json(right_path, track_rows["right"])
    summary = {
        "status": "ok",
        "method": "build_v19_mano_mask_depth_refit_inputs",
        "annotations": str(ann_path),
        "mask_tracks": {"left": str(left_path), "right": str(right_path)},
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "parameters": {
            "mano_projection_dilation_px_960": int(args.mano_projection_dilation_px_960),
            "object_mask_dilation_px_960": int(args.object_mask_dilation_px_960),
            "min_filtered_area_px_960": int(args.min_filtered_area_px_960),
        },
        "counts": {
            "frames": len(out_frames),
            "left_visible_filtered_masks": sum(1 for r in track_rows["left"].values() if r.get("visible")),
            "right_visible_filtered_masks": sum(1 for r in track_rows["right"].values() if r.get("visible")),
        },
        "diagnostics": diagnostics,
    }
    write_json(args.output_dir / "v19_mano_mask_depth_refit_inputs_report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--left-hand-track", type=Path, required=True)
    parser.add_argument("--right-hand-track", type=Path, required=True)
    parser.add_argument("--object-track", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--mano-projection-dilation-px-960", type=int, default=28)
    parser.add_argument("--object-mask-dilation-px-960", type=int, default=3)
    parser.add_argument("--min-filtered-area-px-960", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
