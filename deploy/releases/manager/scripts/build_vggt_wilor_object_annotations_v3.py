#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fuse_v1_full_fidelity import (
    choose_hand_by_side,
    hand_metric_scale_from_raw,
    hand_vector,
    kalman_rts,
    normalize_hand_to_source_camera,
    project_points,
    vector_to_hand,
)
from patch_annotations_with_vggt_poses_v3 import transform_points
from run_v1_wilor_colmap import caption_for_frame, load_actions


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def load_vggt(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "T_world_camera_metric", "source_intrinsics_fx_fy_cx_cy", "vggt_to_meters", "anchor_frame"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    transforms = blob["T_world_camera_metric"].astype(np.float64)
    intrinsics = blob["source_intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if transforms.shape != (len(frame_idx), 4, 4) or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(f"invalid VGGT shapes: {transforms.shape}, {intrinsics.shape}")
    return {
        "frame_idx": frame_idx,
        "transforms": {int(idx): transforms[i] for i, idx in enumerate(frame_idx)},
        "intrinsics": {int(idx): intrinsics[i] for i, idx in enumerate(frame_idx)},
        "vggt_to_meters": float(blob["vggt_to_meters"][0]),
        "anchor_frame": int(blob["anchor_frame"][0]),
    }


def load_wilor(path: Path) -> list[dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    return sorted(frames, key=lambda row: int(row["frame_idx"]))


def load_mask_track(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    out = {}
    for key, row in payload.items():
        frame_idx = int(key)
        out[frame_idx] = row
    return out


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if info["fps"] <= 0 or info["width"] <= 0 or info["height"] <= 0 or info["frame_count"] <= 0:
        raise RuntimeError(f"invalid video info: {info}")
    return info


def transform_hand_to_world(hand: dict, T_world_camera: np.ndarray) -> None:
    joints = np.asarray(hand["joints3d_source_camera_m"], dtype=np.float64)
    hand["joints3d_world_m"] = transform_points(joints, T_world_camera).astype(float).tolist()
    if "vertices_source_camera_m" in hand:
        vertices = np.asarray(hand["vertices_source_camera_m"], dtype=np.float64)
        hand["vertices_world_m"] = transform_points(vertices, T_world_camera).astype(float).tolist()
    elif "vertices_source_camera_m_sample" in hand:
        vertices = np.asarray(hand["vertices_source_camera_m_sample"], dtype=np.float64)
        hand["vertices_world_m_sample"] = transform_points(vertices, T_world_camera).astype(float).tolist()
    else:
        raise RuntimeError("hand has no source-camera vertices")
    hand["world_coordinate_status"] = "v3_vggt_native_metric_depth_scaled_local_world"


def smooth_measured_hands(
    raw_frames: list[dict],
    vggt: dict,
    fps: float,
    wilor_to_meters: float,
) -> tuple[dict[int, list[dict]], dict]:
    normalized_by_frame = []
    rejected = {"left": 0, "right": 0, "unknown": 0}
    for frame in raw_frames:
        idx = int(frame["frame_idx"])
        intr = np.asarray(vggt["intrinsics"][idx], dtype=np.float64)
        normalized = []
        for raw_hand in frame.get("raw_hands", []):
            hand = normalize_hand_to_source_camera(raw_hand, intr, float(wilor_to_meters))
            if hand is None:
                side = str(raw_hand.get("side", "unknown"))
                rejected[side] = rejected.get(side, 0) + 1
            else:
                normalized.append(hand)
        normalized_by_frame.append({"frame_idx": idx, "raw_hands": normalized})

    chosen = [choose_hand_by_side(row["raw_hands"]) for row in normalized_by_frame]
    out = {int(row["frame_idx"]): [] for row in normalized_by_frame}
    stats: dict[str, object] = {"rejected_source_camera_solves": rejected}
    for side in ("left", "right"):
        templates = [row[side] for row in chosen if side in row]
        if not templates:
            stats[side] = {"measured_frames": 0, "predicted_frames": 0, "coverage": 0.0}
            continue
        template = templates[0]
        measurements = []
        confs = []
        best_by_frame = []
        for row in chosen:
            hand = row.get(side)
            best_by_frame.append(hand)
            measurements.append(hand_vector(hand) if hand is not None else None)
            confs.append(float(hand.get("detector_score", 0.0)) if hand is not None else 0.0)
        measured_indices = [i for i, measurement in enumerate(measurements) if measurement is not None]
        first_measured = measured_indices[0]
        last_measured = measured_indices[-1]
        dim = int(measurements[first_measured].shape[0])  # type: ignore[union-attr]
        meas_sigma = np.full(dim, 0.010, dtype=float)
        proc_pos = np.full(dim, 0.006, dtype=float)
        proc_vel = np.full(dim, 0.025, dtype=float)
        smoothed, statuses = kalman_rts(measurements, confs, fps, meas_sigma, proc_pos, proc_vel)
        measured_count = 0
        predicted_count = 0
        projection_medians = []
        for i, vec in enumerate(smoothed):
            if i < first_measured or i > last_measured:
                continue
            frame_idx = int(normalized_by_frame[i]["frame_idx"])
            source = best_by_frame[i] or template
            intr = np.asarray(vggt["intrinsics"][frame_idx], dtype=np.float64)
            hand = vector_to_hand(source, vec, statuses[i], confs[i], intr)
            hand["measurement_available"] = best_by_frame[i] is not None
            if hand["measurement_available"]:
                measured_count += 1
                raw = np.asarray(hand["joints2d_raw"], dtype=float)
                joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                err = np.linalg.norm(project_points(joints, intr) - raw, axis=1)
                hand["projection_residual_to_measurement_px"] = {
                    "median": float(np.median(err)),
                    "p95": float(np.percentile(err, 95)),
                }
                projection_medians.append(float(np.median(err)))
            else:
                predicted_count += 1
            transform_hand_to_world(hand, np.asarray(vggt["transforms"][frame_idx], dtype=np.float64))
            out[frame_idx].append(hand)
        stats[side] = {
            "measured_frames": measured_count,
            "predicted_frames": predicted_count,
            "coverage": measured_count / max(1, len(raw_frames)),
            "median_projection_residual_px": float(np.median(projection_medians)) if projection_medians else None,
        }
    return out, stats


def object_row(frame_idx: int, mask_track: dict[int, dict], source_size: tuple[int, int]) -> dict:
    row = mask_track.get(frame_idx)
    if row is None or not row.get("visible"):
        return {"label": "active_wild_rice_stem", "status": "unobserved_no_verified_mask"}
    mask_path = Path(str(row["mask_path"]))
    if not mask_path.exists():
        raise RuntimeError(f"missing object mask for frame {frame_idx}: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object mask {mask_path}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError(f"empty object mask {mask_path}")
    return {
        "label": "active_wild_rice_stem",
        "status": "measured_vlm_sam2_mask_unidepth_observed_surface",
        "track_id": "active_wild_rice_stem",
        "mask_path": str(mask_path),
        "mask_image_size": [int(mask.shape[1]), int(mask.shape[0])],
        "source_image_size": [int(source_size[0]), int(source_size[1])],
        "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        "center_xy": [float(np.mean(xs)), float(np.mean(ys))],
        "area_px": int(len(xs)),
        "mesh_status": "observed_surface_from_verified_mask_depth_not_complete_mesh",
    }


def run(args: argparse.Namespace) -> dict:
    info = video_info(args.video)
    vggt = load_vggt(args.vggt_archive)
    raw_frames = load_wilor(args.wilor_raw)
    frame_indices = [int(idx) for idx in vggt["frame_idx"].tolist()]
    raw_by_frame = {int(row["frame_idx"]): row for row in raw_frames}
    missing = [idx for idx in frame_indices if idx not in raw_by_frame]
    if missing:
        raise RuntimeError(f"WiLoR raw missing frames: {missing}")
    raw_window = [raw_by_frame[idx] for idx in frame_indices]
    actions = load_actions(args.actions_json)
    hand_scale = hand_metric_scale_from_raw(raw_window)
    hands_by_frame, hand_stats = smooth_measured_hands(raw_window, vggt, float(info["fps"]), float(hand_scale["wilor_local_to_meters"]))
    mask_track = load_mask_track(args.mask_track)
    frames = []
    for idx in frame_indices:
        T = np.asarray(vggt["transforms"][idx], dtype=np.float64)
        intr = np.asarray(vggt["intrinsics"][idx], dtype=np.float64)
        frame = {
            "frame_idx": int(idx),
            "time_s": float(idx / info["fps"]),
            "caption": caption_for_frame(actions, int(idx)),
            "camera": {
                "T_world_camera_metric": T.astype(float).tolist(),
                "position_world_m": T[:3, 3].astype(float).tolist(),
                "vggt_source_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                "pose_source_status": "v3_vggt_native_metric_depth_scaled_local_world",
                "vggt_to_meters": float(vggt["vggt_to_meters"]),
                "anchor_frame": int(vggt["anchor_frame"]),
            },
            "hands": hands_by_frame[int(idx)],
            "object": object_row(int(idx), mask_track, (int(info["width"]), int(info["height"]))),
        }
        if not frame["caption"]:
            frame["caption"] = "Peeling wild rice stem by hand"
        frames.append(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations_v3_vggt_wilor_object.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    measured_hands = [
        hand
        for frame in frames
        for hand in frame["hands"]
        if bool(hand.get("measurement_available", False))
    ]
    reproj = [
        float(hand["projection_residual_to_measurement_px"]["median"])
        for hand in measured_hands
        if "projection_residual_to_measurement_px" in hand
    ]
    object_measured = [frame for frame in frames if frame["object"].get("status") == "measured_vlm_sam2_mask_unidepth_observed_surface"]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "build_vggt_wilor_object_annotations_v3",
        "video": str(args.video),
        "vggt_archive": str(args.vggt_archive),
        "wilor_raw": str(args.wilor_raw),
        "mask_track": str(args.mask_track),
        "annotations": str(annotations_path),
        "frames": int(len(frames)),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "camera": {
            "scale_status": "VGGT native camera scaled by verified object-mask UniDepth/VGGT depth ratio",
            "vggt_to_meters": float(vggt["vggt_to_meters"]),
            "anchor_frame": int(vggt["anchor_frame"]),
        },
        "hands": {
            "hand_scale": hand_scale,
            "stream": hand_stats,
            "measured_hand_rows": int(len(measured_hands)),
            "measured_reprojection_median_px": summarize(reproj),
        },
        "object": {
            "measured_mask_frames": int(len(object_measured)),
            "measured_mask_frame_indices": [int(frame["frame_idx"]) for frame in object_measured],
            "status": "observed_surface_only_until dense masks and complete mesh are available",
        },
    }
    (args.output_dir / "qc_vggt_wilor_object_annotations_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--actions-json", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--wilor-raw", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
