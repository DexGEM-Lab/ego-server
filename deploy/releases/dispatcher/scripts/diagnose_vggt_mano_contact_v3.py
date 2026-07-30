#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_intrinsics_focal_sweep_v3 import (
    hand_bone_scale_m,
    mask_distance_map,
    project_points,
    solve_source_camera_translation,
    source_local_vertices,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def vggt_affine_from_source(width: int, height: int, target_size: int) -> tuple[float, float, float]:
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    scale_x = float(new_width / width)
    scale_y = float(new_height / height)
    pad_top = float((target_size - new_height) // 2)
    return scale_x, scale_y, pad_top


def points_to_vggt_frame(points2d: np.ndarray, source_size: list[int], target_size: int) -> np.ndarray:
    width, height = int(source_size[0]), int(source_size[1])
    scale_x, scale_y, pad_top = vggt_affine_from_source(width, height, target_size)
    out = np.asarray(points2d, dtype=float).copy()
    out[:, 0] *= scale_x
    out[:, 1] = out[:, 1] * scale_y + pad_top
    return out


def resize_mask(path: Path, target_size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"could not read mask: {path}")
    height, width = mask.shape[:2]
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2
    out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    return out > 0


def vggt_frame_points(archive: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    offsets = archive["vertex_offsets"].astype(np.int64)
    points = archive["object_points_vggt"].astype(float)
    return points[int(offsets[index]) : int(offsets[index + 1])]


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    vggt = np.load(args.vggt_archive)
    frames = vggt["frame_idx"].astype(int)
    intrinsics = vggt["intrinsic"].astype(float)
    extrinsic = vggt["extrinsic"].astype(float)
    masks = {int(row["frame_idx"]): Path(row["mask"]) for row in load_json(args.dataset_manifest)["frames"]}
    rows = []
    for i, frame_idx in enumerate(frames.tolist()):
        frame = frame_by_idx.get(int(frame_idx))
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        mask_path = masks.get(int(frame_idx))
        if mask_path is None:
            raise RuntimeError(f"dataset manifest missing frame {frame_idx}")
        mask = resize_mask(mask_path, int(args.target_size))
        dist = mask_distance_map(mask)
        K = intrinsics[i]
        K4 = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=float)
        obj_points = vggt_frame_points(vggt, i)
        obj_camera = (obj_points @ extrinsic[i, :3, :3].T) + extrinsic[i, :3, 3][None, :]
        obj_z = obj_camera[:, 2]
        obj_z = obj_z[np.isfinite(obj_z) & (obj_z > 0)]
        if obj_z.size == 0:
            raise RuntimeError(f"VGGT object has no positive camera depth at frame {frame_idx}")
        object_depth = float(np.median(obj_z))
        for hand_i, hand in enumerate(frame.get("hands", [])):
            if not bool(hand.get("measurement_available", False)):
                continue
            local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
            local_vertices = source_local_vertices(hand)
            raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
            if local_joints.shape != (21, 3) or raw2d.shape != (21, 2):
                continue
            raw2d_vggt = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], int(args.target_size))
            trans = solve_source_camera_translation(local_joints, raw2d_vggt, K4)
            joints = local_joints + trans[None, :]
            vertices = local_vertices + trans[None, :]
            if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
                continue
            projected = project_points(joints, K4)
            reproj = np.linalg.norm(projected - raw2d_vggt, axis=1)
            uv = project_points(vertices, K4)
            valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0)
            x = np.clip(np.rint(uv[:, 0]).astype(int), 0, int(args.target_size) - 1)
            y = np.clip(np.rint(uv[:, 1]).astype(int), 0, int(args.target_size) - 1)
            near = valid & (dist[y, x] <= float(args.contact_distance_px))
            gap = vertices[near, 2] - object_depth if np.any(near) else np.zeros(0, dtype=float)
            score = float(hand.get("detector_score", np.nan))
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "hand_idx": int(hand_i),
                    "side": hand.get("side"),
                    "detector_score": score,
                    "median_joint_reprojection_px_vggt": float(np.median(reproj)),
                    "p95_joint_reprojection_px_vggt": float(np.percentile(reproj, 95)),
                    "cam_t_z_vggt": float(trans[2]),
                    "object_depth_vggt": object_depth,
                    "near_mask_vertices": int(np.count_nonzero(near)),
                    "near_mask_hand_minus_object_depth_median_vggt": float(np.median(gap)) if gap.size else None,
                    "near_mask_hand_minus_object_depth_p95_abs_vggt": float(np.percentile(np.abs(gap), 95)) if gap.size else None,
                    "hand_bone_scale_m": float(hand_bone_scale_m(joints)),
                    "detector_ok": bool(np.isfinite(score) and score >= float(args.min_detector_score)),
                }
            )
    measured = rows
    high = [row for row in rows if row["detector_ok"]]
    contact_rows = [row for row in high if int(row["near_mask_vertices"]) >= int(args.min_near_vertices)]
    report = {
        "status": "ok",
        "diagnostic_only": True,
        "method": "diagnose_vggt_mano_contact_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "rows": int(len(rows)),
        "high_score_rows": int(len(high)),
        "contact_rows": int(len(contact_rows)),
        "summary_high_score": {
            "joint_reprojection_px_vggt": summarize([row["median_joint_reprojection_px_vggt"] for row in high]),
            "cam_t_z_vggt": summarize([row["cam_t_z_vggt"] for row in high]),
            "object_depth_vggt": summarize([row["object_depth_vggt"] for row in high]),
            "near_mask_vertices": summarize([row["near_mask_vertices"] for row in high]),
            "contact_gap_vggt": summarize(
                [
                    row["near_mask_hand_minus_object_depth_median_vggt"]
                    for row in contact_rows
                    if row["near_mask_hand_minus_object_depth_median_vggt"] is not None
                ]
            ),
            "contact_gap_abs_p95_vggt": summarize(
                [
                    row["near_mask_hand_minus_object_depth_p95_abs_vggt"]
                    for row in contact_rows
                    if row["near_mask_hand_minus_object_depth_p95_abs_vggt"] is not None
                ]
            ),
            "hand_bone_scale_m": summarize([row["hand_bone_scale_m"] for row in high]),
        },
        "rows_detail": measured,
        "parameters": {
            "target_size": int(args.target_size),
            "contact_distance_px": float(args.contact_distance_px),
            "min_detector_score": float(args.min_detector_score),
            "min_near_vertices": int(args.min_near_vertices),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qc_vggt_mano_contact_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_detail"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
