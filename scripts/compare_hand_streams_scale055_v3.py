#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mmap
import re
from pathlib import Path

import cv2
import numpy as np

from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_contact_reliability_v3 import (
    camera_points_from_hand,
    condition_counts,
    depth_patch_iqr_ratio,
    hand_bone_scale_m,
    hand_tip_spread_m,
    resize_mask_to_depth,
    summarize_key,
)
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame_window(path: Path, frame_start: int, frame_end: int) -> dict[int, dict]:
    frames: dict[int, dict] = {}
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for match in re.finditer(rb'"frame_idx"\s*:\s*(\d+)', data):
                frame_idx = int(match.group(1))
                if frame_idx < frame_start:
                    continue
                if frame_idx > frame_end and frames:
                    break
                if not (frame_start <= frame_idx <= frame_end):
                    continue
                start = data.rfind(b"\n    {", 0, match.start())
                if start < 0:
                    start = data.rfind(b"{", 0, match.start())
                else:
                    start += 5
                if start < 0:
                    raise RuntimeError(f"could not locate frame object start for frame {frame_idx} in {path}")
                end = matching_brace(data, start)
                frame = json.loads(data[start : end + 1].decode("utf-8"))
                if int(frame["frame_idx"]) != frame_idx:
                    raise RuntimeError(f"parsed mismatched frame object for frame {frame_idx} in {path}")
                frames[frame_idx] = frame
    if not frames:
        raise RuntimeError(f"{path} has no frames in [{frame_start}, {frame_end}]")
    return frames


def matching_brace(data: mmap.mmap, start: int) -> int:
    if data[start : start + 1] != b"{":
        raise RuntimeError("matching_brace start is not an object")
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(data)):
        char = data[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                in_string = False
            continue
        if char == 34:
            in_string = True
        elif char == 123:
            depth += 1
        elif char == 125:
            depth -= 1
            if depth == 0:
                return pos
    raise RuntimeError("unterminated JSON object")


def load_depth_archive(path: Path) -> tuple[dict[int, int], np.ndarray]:
    blob = np.load(path)
    frame_idx = blob["frame_idx"].astype(int)
    if len(set(int(x) for x in frame_idx)) != len(frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    return {int(x): i for i, x in enumerate(frame_idx)}, np.asarray(blob["depth"], dtype=np.float32)


def load_mesh_archive(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    frame_idx = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(int)
    vertices = np.asarray(blob["vertices"], dtype=np.float32)
    out: dict[int, np.ndarray] = {}
    for i, frame in enumerate(frame_idx):
        frame_i = int(frame)
        if frame_i in out:
            raise RuntimeError(f"mesh archive has duplicate frame {frame_i}")
        out[frame_i] = vertices[int(offsets[i]) : int(offsets[i + 1])].astype(float)
    return out


def intrinsics_for(frame: dict, hand: dict, source: str, cli_intrinsics: list[float]) -> np.ndarray:
    if source == "hand":
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    elif source == "target-vggt":
        camera = frame.get("camera", {})
        intr = np.asarray(camera.get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=float)
    elif source == "cli":
        intr = np.asarray(cli_intrinsics, dtype=float)
    else:
        raise RuntimeError(f"unsupported intrinsics source: {source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid {source} intrinsics for frame {frame.get('frame_idx')}")
    return intr


def source_size_for(frame: dict) -> np.ndarray:
    obj = frame.get("object", {})
    size = np.asarray(obj.get("source_image_size", []), dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all():
        raise RuntimeError(f"invalid source image size for frame {frame.get('frame_idx')}")
    return size


def object_mask_for(frame: dict, depth: np.ndarray, remote_output_root: str, local_output_root: str) -> np.ndarray:
    obj = frame.get("object", {})
    if not obj.get("mask_path"):
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no object mask")
    mask_path = localize_path(str(obj["mask_path"]), remote_output_root, local_output_root)
    mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
    return resize_mask_to_depth(mask, depth)


def object_depth_for(frame: dict, object_vertices: np.ndarray) -> float:
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    homog = np.c_[object_vertices, np.ones(len(object_vertices), dtype=float)]
    camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if len(z) == 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} object mesh has no positive camera depth")
    return float(np.median(z))


def hand_joints(hand: dict) -> np.ndarray:
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    if joints.shape != (21, 3):
        joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    if joints.shape != (21, 3):
        raise RuntimeError("hand has no 21x3 camera joints")
    return joints


def hand_keypoints(hand: dict) -> np.ndarray:
    keypoints = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if keypoints.shape != (21, 2):
        keypoints = np.asarray(hand.get("joints2d", []), dtype=float)
    if keypoints.shape != (21, 2):
        raise RuntimeError("hand has no 21x2 2D keypoints")
    return keypoints


def hand_camera_vertices(hand: dict, target_frame: dict) -> np.ndarray:
    T_world_camera = np.asarray(target_frame["camera"]["T_world_camera_metric"], dtype=float)
    return camera_points_from_hand(hand, T_world_camera)


def row_for_hand(
    stream_name: str,
    frame_idx: int,
    target_frame: dict,
    hand_i: int,
    hand: dict,
    depth: np.ndarray,
    mask_depth: np.ndarray,
    object_depth: float,
    args: argparse.Namespace,
) -> dict:
    source_size = source_size_for(target_frame)
    joints = hand_joints(hand)
    keypoints = hand_keypoints(hand)
    vertices = hand_camera_vertices(hand, target_frame)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("hand vertices are not Nx3")
    if np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("hand vertices contain non-positive depth")

    hand_intr = intrinsics_for(target_frame, hand, "hand", args.intrinsics)
    target_intr = intrinsics_for(target_frame, hand, args.target_intrinsics_source, args.intrinsics)
    own_projected = project_points(joints, hand_intr)
    target_projected = project_points(joints, target_intr)
    own_reproj = np.linalg.norm(own_projected - keypoints, axis=1)
    target_reproj = np.linalg.norm(target_projected - keypoints, axis=1)

    metric_depth = sample_depth(depth, keypoints, source_size)
    valid_depth = np.isfinite(metric_depth) & (metric_depth > 0.0)
    target_good_depth = valid_depth & (target_reproj <= args.good_joint_reprojection_px)
    mano_minus_metric = joints[target_good_depth, 2] - metric_depth[target_good_depth]

    depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    depth_xy = keypoints * depth_scale[None, :]
    patch_ratios = np.asarray(
        [depth_patch_iqr_ratio(depth, xy, args.patch_radius) for xy in depth_xy[target_good_depth]],
        dtype=float,
    )
    stable_depth = patch_ratios[np.isfinite(patch_ratios)] <= args.max_depth_iqr_ratio

    dist = mask_distance_map(mask_depth)
    uv = project_points(vertices, target_intr)
    xy = uv * depth_scale[None, :]
    valid_uv = np.isfinite(xy).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = valid_uv & (dist[y, x] <= args.contact_distance_px)
    near_vertices = vertices[near]
    contact_gap = near_vertices[:, 2] - object_depth if len(near_vertices) else np.asarray([], dtype=float)

    score = float(hand.get("detector_score", np.nan))
    own_projection_ok = float(np.median(own_reproj)) <= args.max_good_median_reprojection_px
    target_projection_ok = float(np.median(target_reproj)) <= args.max_good_median_reprojection_px
    depth_ok = len(mano_minus_metric) >= args.min_good_depth_joints and abs(float(np.median(mano_minus_metric))) <= args.max_good_depth_bias_m
    patch_ok = len(stable_depth) >= args.min_good_depth_joints and float(np.mean(stable_depth)) >= args.min_stable_depth_fraction
    bone_scale = hand_bone_scale_m(joints)
    bone_scale_ok = args.min_bone_scale_m <= bone_scale <= args.max_bone_scale_m
    contact_ok = len(near_vertices) >= args.min_near_vertices and abs(float(np.median(contact_gap))) <= args.max_good_contact_gap_m
    measured = bool(hand.get("measurement_available", False))
    detector_ok = np.isfinite(score) and score >= args.min_detector_score
    reliable = bool(measured and detector_ok and target_projection_ok and depth_ok and patch_ok and bone_scale_ok and contact_ok)
    return {
        "stream": stream_name,
        "frame_idx": int(frame_idx),
        "hand_idx": int(hand_i),
        "side": hand.get("side"),
        "filter_status": hand.get("filter_status"),
        "measurement_available": measured,
        "detector_score": score,
        "own_intrinsics_median_joint_reprojection_px": float(np.median(own_reproj)),
        "target_intrinsics_median_joint_reprojection_px": float(np.median(target_reproj)),
        "p95_target_joint_reprojection_px": float(np.percentile(target_reproj, 95.0)),
        "median_joint_reprojection_px": float(np.median(target_reproj)),
        "good_depth_joints": int(np.count_nonzero(target_good_depth)),
        "mano_minus_metric_depth_median_m": None if len(mano_minus_metric) == 0 else float(np.median(mano_minus_metric)),
        "mano_minus_metric_depth_p95_abs_m": None if len(mano_minus_metric) == 0 else float(np.percentile(np.abs(mano_minus_metric), 95.0)),
        "stable_depth_fraction": None if len(stable_depth) == 0 else float(np.mean(stable_depth)),
        "hand_bone_scale_m": bone_scale,
        "hand_tip_spread_m": hand_tip_spread_m(joints),
        "near_mask_vertices": int(len(near_vertices)),
        "near_mask_hand_minus_object_depth_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "near_mask_hand_minus_object_depth_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
        "detector_ok": bool(detector_ok),
        "own_projection_ok": bool(own_projection_ok),
        "projection_ok": bool(target_projection_ok),
        "depth_ok": bool(depth_ok),
        "stable_depth_ok": bool(patch_ok),
        "bone_scale_ok": bool(bone_scale_ok),
        "contact_ok": bool(contact_ok),
        "reliable_for_contact": reliable,
    }


def summarize_rows(rows: list[dict]) -> dict:
    measured = [r for r in rows if r["measurement_available"]]
    measured_high = [r for r in measured if r["detector_ok"]]
    reliable = [r for r in rows if r["reliable_for_contact"]]
    return {
        "rows": len(rows),
        "measured_rows": len(measured),
        "measured_high_score_rows": len(measured_high),
        "reliable_contact_rows": len(reliable),
        "condition_counts_all": condition_counts(rows),
        "condition_counts_measured_high_score": condition_counts(measured_high),
        "summary_all": {
            "own_reprojection_px": summarize_key(rows, "own_intrinsics_median_joint_reprojection_px"),
            "target_reprojection_px": summarize_key(rows, "target_intrinsics_median_joint_reprojection_px"),
            "mano_minus_unidepth_m": summarize_key(rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(rows, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
            "near_mask_vertices": summarize(np.asarray([float(r["near_mask_vertices"]) for r in rows], dtype=float)),
        },
        "summary_measured_high_score": {
            "own_reprojection_px": summarize_key(measured_high, "own_intrinsics_median_joint_reprojection_px"),
            "target_reprojection_px": summarize_key(measured_high, "target_intrinsics_median_joint_reprojection_px"),
            "mano_minus_unidepth_m": summarize_key(measured_high, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(measured_high, "near_mask_hand_minus_object_depth_median_m"),
            "hand_bone_scale_m": summarize_key(measured_high, "hand_bone_scale_m"),
            "near_mask_vertices": summarize(np.asarray([float(r["near_mask_vertices"]) for r in measured_high], dtype=float)),
        },
    }


def run(args: argparse.Namespace) -> dict:
    target_frames = load_frame_window(args.target_annotations, args.frame_start, args.frame_end)
    frame_to_depth_i, depths = load_depth_archive(args.metric_depth_npz)
    mesh_frames = load_mesh_archive(args.object_mesh_npz)
    stream_reports = {}
    all_rows = []
    all_skipped = []
    for spec in args.hand_stream:
        if "=" not in spec:
            raise RuntimeError(f"hand stream must be name=path: {spec}")
        stream_name, stream_path_s = spec.split("=", 1)
        stream_path = Path(stream_path_s)
        hand_frames = load_frame_window(stream_path, args.frame_start, args.frame_end)
        rows: list[dict] = []
        skipped: list[dict] = []
        for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
            target_frame = target_frames.get(frame_idx)
            hand_frame = hand_frames.get(frame_idx)
            if target_frame is None or hand_frame is None:
                skipped.append({"stream": stream_name, "frame_idx": frame_idx, "reason": "missing_target_or_hand_frame"})
                continue
            try:
                depth = depth_frame(depths, frame_to_depth_i, frame_idx)
                mask_depth = object_mask_for(target_frame, depth, args.remote_output_root, args.local_output_root)
                object_vertices = mesh_frames[frame_idx]
                obj_depth = object_depth_for(target_frame, object_vertices)
            except Exception as exc:
                skipped.append({"stream": stream_name, "frame_idx": frame_idx, "reason": str(exc)})
                continue
            for hand_i, hand in enumerate(hand_frame.get("hands", [])):
                try:
                    row = row_for_hand(stream_name, frame_idx, target_frame, hand_i, hand, depth, mask_depth, obj_depth, args)
                    rows.append(row)
                    all_rows.append(row)
                except Exception as exc:
                    skipped.append(
                        {
                            "stream": stream_name,
                            "frame_idx": frame_idx,
                            "hand_idx": hand_i,
                            "side": hand.get("side"),
                            "reason": str(exc),
                        }
                    )
        stream_reports[stream_name] = {
            "annotations": str(stream_path),
            **summarize_rows(rows),
            "skipped_count": len(skipped),
            "skipped_reason_counts": reason_counts(skipped),
            "rows_preview": rows[:60],
            "skipped_preview": skipped[:60],
        }
        all_skipped.extend(skipped)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "target_annotations": str(args.target_annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "target_intrinsics_source": args.target_intrinsics_source,
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "max_good_median_reprojection_px": float(args.max_good_median_reprojection_px),
            "good_joint_reprojection_px": float(args.good_joint_reprojection_px),
            "max_good_depth_bias_m": float(args.max_good_depth_bias_m),
            "max_good_contact_gap_m": float(args.max_good_contact_gap_m),
            "min_good_depth_joints": int(args.min_good_depth_joints),
            "min_near_vertices": int(args.min_near_vertices),
        },
        "streams": stream_reports,
        "combined": summarize_rows(all_rows),
        "combined_skipped_count": len(all_skipped),
        "combined_skipped_reason_counts": reason_counts(all_skipped),
        "interpretation": (
            "Rows are evaluated against one fixed object/camera/depth contract: scale-0.55 VGGT object poses, "
            "target VGGT intrinsics, and UniDepth metric depth. Own-intrinsics reprojection is reported separately "
            "to distinguish a model output that fits its original camera contract from one that fits the target graph."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "streams"}, indent=2))
    return report


def reason_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--hand-stream", action="append", required=True, help="name=/path/to/annotations.json")
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-intrinsics-source", choices=["target-vggt", "cli"], default="target-vggt")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[2304.0, 2304.0, 960.0, 540.0])
    parser.add_argument("--remote-output-root", default="/mnt/user-home/yiwen/ego_annotation_remote/outputs")
    parser.add_argument("--local-output-root", default="/data2/ego_annotation_outputs")
    parser.add_argument("--min-detector-score", type=float, default=0.5)
    parser.add_argument("--max-good-median-reprojection-px", type=float, default=12.0)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--max-good-depth-bias-m", type=float, default=0.03)
    parser.add_argument("--max-good-contact-gap-m", type=float, default=0.03)
    parser.add_argument("--min-good-depth-joints", type=int, default=12)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--contact-distance-px", type=float, default=20.0)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.08)
    parser.add_argument("--min-stable-depth-fraction", type=float, default=0.75)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.12)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.24)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
