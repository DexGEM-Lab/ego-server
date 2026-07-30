#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def source_intrinsics(
    intrinsic_vggt: np.ndarray,
    source_width: int,
    source_height: int,
    target_size: int,
) -> list[float]:
    if source_width >= source_height:
        new_width = int(target_size)
        new_height = round(source_height * (new_width / source_width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(source_width * (new_height / source_height) / 14) * 14
    if new_width <= 0 or new_height <= 0:
        raise RuntimeError("invalid VGGT preprocessing dimensions")
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    sx = new_width / float(source_width)
    sy = new_height / float(source_height)
    fx = float(intrinsic_vggt[0, 0] / sx)
    fy = float(intrinsic_vggt[1, 1] / sy)
    cx = float((intrinsic_vggt[0, 2] - pad_left) / sx)
    cy = float((intrinsic_vggt[1, 2] - pad_top) / sy)
    return [fx, fy, cx, cy]


def transform_points(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise RuntimeError(f"expected Nx3 camera points, got {points_camera.shape}")
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def validate_rotation(rotation: np.ndarray, name: str) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RuntimeError(f"{name} must be finite 3x3")
    det = float(np.linalg.det(rotation))
    orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    if abs(det - 1.0) > 1e-3 or orth_error > 1e-3:
        raise RuntimeError(f"{name} is not a valid rotation: det={det:.6f}, orth_error={orth_error:.6g}")


def load_vggt_poses(args: argparse.Namespace) -> dict[int, dict]:
    blob = np.load(args.vggt_archive)
    required = {
        "frame_idx",
        "extrinsic",
        "intrinsic",
        "camera_centers_aligned",
        "sim3_rotation",
        "sim3_scale",
        "sim3_translation",
    }
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{args.vggt_archive} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    extrinsic = blob["extrinsic"].astype(np.float64)
    intrinsic = blob["intrinsic"].astype(np.float64)
    centers_vggt = blob["camera_centers_vggt"].astype(np.float64) if "camera_centers_vggt" in blob.files else None
    centers = blob["camera_centers_aligned"].astype(np.float64)
    sim3_rotation = blob["sim3_rotation"].astype(np.float64)
    sim3_translation = blob["sim3_translation"].astype(np.float64)
    sim3_scale = float(blob["sim3_scale"][0])
    validate_rotation(sim3_rotation, "sim3_rotation")
    if extrinsic.shape[:2] != (len(frame_idx), 3) or extrinsic.shape[2] != 4:
        raise RuntimeError(f"invalid extrinsic shape {extrinsic.shape}")
    if intrinsic.shape != (len(frame_idx), 3, 3):
        raise RuntimeError(f"invalid intrinsic shape {intrinsic.shape}")
    if centers.shape != (len(frame_idx), 3):
        raise RuntimeError(f"invalid camera_centers_aligned shape {centers.shape}")
    if centers_vggt is not None and centers_vggt.shape != (len(frame_idx), 3):
        raise RuntimeError(f"invalid camera_centers_vggt shape {centers_vggt.shape}")

    center_map = {int(idx): centers[i] for i, idx in enumerate(frame_idx)}
    if args.pose_scale_mode == "sim3":
        scale = sim3_scale
        translation = sim3_translation
    elif args.pose_scale_mode == "custom_anchor":
        if centers_vggt is None:
            raise RuntimeError("custom_anchor mode requires camera_centers_vggt in VGGT archive")
        scale = float(args.custom_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"custom scale must be positive, got {scale}")
        anchor_idx = int(args.anchor_frame)
        if anchor_idx not in center_map:
            raise RuntimeError(f"anchor frame {anchor_idx} is absent from VGGT archive")
        annotations = load_json(args.annotations)
        frames = annotations.get("frames")
        if not isinstance(frames, list):
            raise RuntimeError("annotations must contain frames list")
        anchor_frames = [frame for frame in frames if int(frame["frame_idx"]) == anchor_idx]
        if len(anchor_frames) != 1:
            raise RuntimeError(f"annotations contain {len(anchor_frames)} rows for anchor frame {anchor_idx}")
        anchor_T = np.asarray(anchor_frames[0]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if anchor_T.shape != (4, 4) or not np.isfinite(anchor_T).all():
            raise RuntimeError(f"anchor frame {anchor_idx} has invalid camera transform")
        anchor_i = int(np.where(frame_idx == anchor_idx)[0][0])
        translation = anchor_T[:3, 3] - scale * (sim3_rotation @ centers_vggt[anchor_i])
    else:
        raise RuntimeError(f"unsupported pose scale mode: {args.pose_scale_mode}")

    out: dict[int, dict] = {}
    for i, idx in enumerate(frame_idx):
        R_world_to_camera_vggt = extrinsic[i, :3, :3]
        validate_rotation(R_world_to_camera_vggt, f"vggt extrinsic rotation frame {int(idx)}")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = sim3_rotation @ R_world_to_camera_vggt.T
        if args.pose_scale_mode == "sim3":
            T[:3, 3] = centers[i]
        else:
            assert centers_vggt is not None
            T[:3, 3] = scale * (sim3_rotation @ centers_vggt[i]) + translation
        validate_rotation(T[:3, :3], f"aligned camera-to-world rotation frame {int(idx)}")
        out[int(idx)] = {
            "T_world_camera_metric": T,
            "source_intrinsics": source_intrinsics(
                intrinsic[i],
                int(args.source_width),
                int(args.source_height),
                int(args.target_size),
            ),
            "vggt_padded_intrinsic": intrinsic[i],
            "pose_scale": float(scale),
        }
    if not out:
        raise RuntimeError("VGGT archive contains no poses")
    return out


def recompute_hand_world(frame: dict, T_world_camera: np.ndarray) -> int:
    changed = 0
    hands = frame.get("hands", [])
    if not isinstance(hands, list):
        raise RuntimeError(f"frame {frame.get('frame_idx')} hands must be a list")
    for hand in hands:
        if not isinstance(hand, dict):
            raise RuntimeError(f"frame {frame.get('frame_idx')} contains a non-object hand")
        if "joints3d_source_camera_m" not in hand:
            raise RuntimeError(f"frame {frame.get('frame_idx')} hand is missing joints3d_source_camera_m")
        joints = np.asarray(hand["joints3d_source_camera_m"], dtype=np.float64)
        hand["joints3d_world_m_before_vggt_pose_patch"] = hand.get("joints3d_world_m")
        hand["joints3d_world_m"] = transform_points(joints, T_world_camera).astype(float).tolist()
        vertex_keys = [
            ("vertices_source_camera_m", "vertices_world_m"),
            ("vertices_source_camera_m_sample", "vertices_world_m_sample"),
        ]
        updated_vertices = 0
        for source_key, world_key in vertex_keys:
            if source_key in hand:
                vertices = np.asarray(hand[source_key], dtype=np.float64)
                hand[f"{world_key}_before_vggt_pose_patch"] = hand.get(world_key)
                hand[world_key] = transform_points(vertices, T_world_camera).astype(float).tolist()
                updated_vertices += 1
        if updated_vertices == 0 and ("vertices_world_m" in hand or "vertices_world_m_sample" in hand):
            raise RuntimeError(f"frame {frame.get('frame_idx')} hand has world vertices but no source-camera vertices")
        hand["world_coordinate_status_before_vggt_pose_patch"] = hand.get("world_coordinate_status")
        hand["world_coordinate_status"] = "v3_vggt_full_se3_camera_pose_source_camera_mano_transformed"
        changed += 1
    return changed


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


def pair_speed(rows: list[dict], key: str, fps: float) -> dict:
    values = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        dt = (int(cur["frame_idx"]) - int(prev["frame_idx"])) / float(fps)
        if dt <= 0.0:
            raise RuntimeError("frame indices must increase")
        a = np.asarray(prev[key], dtype=np.float64)
        b = np.asarray(cur[key], dtype=np.float64)
        values.append(float(np.linalg.norm(b - a) / dt))
    return summarize(values)


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations must contain a nonempty frames list")
    vggt_poses = load_vggt_poses(args)

    rows = []
    changed_hands = 0
    for frame in frames:
        idx = int(frame["frame_idx"])
        if idx < int(args.frame_start) or idx > int(args.frame_end):
            continue
        if idx not in vggt_poses:
            raise RuntimeError(f"VGGT archive has no pose for frame {idx}")
        pose = vggt_poses[idx]
        new_T = pose["T_world_camera_metric"]
        camera = frame.get("camera")
        if not isinstance(camera, dict) or "T_world_camera_metric" not in camera:
            raise RuntimeError(f"frame {idx} missing camera.T_world_camera_metric")
        old_T = np.asarray(camera["T_world_camera_metric"], dtype=np.float64)
        if old_T.shape != (4, 4) or not np.isfinite(old_T).all():
            raise RuntimeError(f"frame {idx} old camera transform must be finite 4x4")
        validate_rotation(old_T[:3, :3], f"old annotation camera rotation frame {idx}")
        rotation_delta = Rotation.from_matrix(old_T[:3, :3].T @ new_T[:3, :3]).magnitude()
        translation_delta = float(np.linalg.norm(new_T[:3, 3] - old_T[:3, 3]))
        camera["T_world_camera_metric_before_vggt_pose_patch"] = old_T.astype(float).tolist()
        camera["position_world_m_before_vggt_pose_patch"] = camera.get("position_world_m")
        camera["T_world_camera_metric"] = new_T.astype(float).tolist()
        camera["position_world_m"] = new_T[:3, 3].astype(float).tolist()
        camera["vggt_source_intrinsics_fx_fy_cx_cy"] = [float(v) for v in pose["source_intrinsics"]]
        camera["vggt_padded_intrinsic_3x3"] = pose["vggt_padded_intrinsic"].astype(float).tolist()
        camera["vggt_pose_scale"] = float(pose["pose_scale"])
        camera["pose_source_status"] = f"v3_vggt_full_scene_{args.pose_scale_mode}_camera_pose"
        changed_hands += recompute_hand_world(frame, new_T)
        rows.append(
            {
                "frame_idx": idx,
                "old_center_world_m": old_T[:3, 3].astype(float).tolist(),
                "new_center_world_m": new_T[:3, 3].astype(float).tolist(),
                "translation_delta_m": translation_delta,
                "rotation_delta_rad": float(rotation_delta),
                "vggt_source_intrinsics_fx_fy_cx_cy": [float(v) for v in pose["source_intrinsics"]],
            }
        )

    if not rows:
        raise RuntimeError("no frames were patched")
    rows.sort(key=lambda row: int(row["frame_idx"]))
    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    report = {
        "status": "ok",
        "method": "patch_annotations_with_vggt_poses_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "changed_frames": int(len(rows)),
        "changed_hands": int(changed_hands),
        "pose_scale_mode": str(args.pose_scale_mode),
        "custom_scale": float(args.custom_scale) if args.pose_scale_mode == "custom_anchor" else None,
        "anchor_frame": int(args.anchor_frame),
        "translation_delta_m": summarize([row["translation_delta_m"] for row in rows]),
        "rotation_delta_rad": summarize([row["rotation_delta_rad"] for row in rows]),
        "old_center_speed_m_s": pair_speed(rows, "old_center_world_m", float(args.fps)),
        "new_center_speed_m_s": pair_speed(rows, "new_center_world_m", float(args.fps)),
        "source_intrinsics_fx": summarize([row["vggt_source_intrinsics_fx_fy_cx_cy"][0] for row in rows]),
        "source_intrinsics_fy": summarize([row["vggt_source_intrinsics_fx_fy_cx_cy"][1] for row in rows]),
        "rows": rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pose-scale-mode", choices=["sim3", "custom_anchor"], default="sim3")
    parser.add_argument("--custom-scale", type=float, default=1.0)
    parser.add_argument("--anchor-frame", type=int, default=880)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
