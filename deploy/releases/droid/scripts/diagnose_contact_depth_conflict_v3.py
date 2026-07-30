#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, project_world, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mesh_frame_vertices(mesh_npz: Path, frame_idx: int) -> np.ndarray:
    blob = np.load(mesh_npz)
    frames = blob["frame_idx"].astype(int)
    matches = np.flatnonzero(frames == int(frame_idx))
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(matches)} times in mesh archive")
    i = int(matches[0])
    v0, v1 = int(blob["vertex_offsets"][i]), int(blob["vertex_offsets"][i + 1])
    return blob["vertices"][v0:v1].astype(float)


def hand_vertices(frame: dict) -> np.ndarray:
    out = []
    for hand in frame.get("hands", []):
        if "vertices_world_m" in hand:
            out.append(np.asarray(hand["vertices_world_m"], dtype=float))
        elif "vertices_sample_world_m" in hand:
            out.append(np.asarray(hand["vertices_sample_world_m"], dtype=float))
        elif "joints3d_world_m" in hand:
            out.append(np.asarray(hand["joints3d_world_m"], dtype=float))
    if not out:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(out)


def summarize(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def run(args: argparse.Namespace) -> dict:
    frames = load_json(args.annotations)["frames"]
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    reports = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        if frame_idx not in by_idx:
            continue
        frame = by_idx[frame_idx]
        obj = frame.get("object", {})
        if not obj.get("mask_path"):
            continue
        try:
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
        except Exception as exc:
            reports.append({"frame_idx": frame_idx, "status": "skipped", "reason": str(exc)})
            continue
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        _, object_z = project_world(object_vertices, T, intrinsics)
        hands = hand_vertices(frame)
        hand_report = {"count": 0}
        near_report = {"count": 0}
        gap_report = None
        if len(hands):
            hand_uv, hand_z = project_world(hands, T, intrinsics)
            hand_report = summarize(hand_z[np.isfinite(hand_z) & (hand_z > 0.0)])
            distance = mask_distance_map(mask)
            scale = np.asarray(mask.shape[::-1], dtype=float) / np.asarray(obj["source_image_size"], dtype=float)
            xy = hand_uv * scale
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(hand_z) & (hand_z > 0.0)
            x = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
            y = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
            near = valid & (distance[y, x] <= args.contact_distance_px * float(scale.mean()))
            near_report = summarize(hand_z[near])
            if near.any():
                gap = hand_z[near] - float(np.median(object_z))
                gap_report = summarize(gap)
        reports.append(
            {
                "frame_idx": frame_idx,
                "status": "ok",
                "object_contact_ratio": float(obj.get("contact_ratio", 0.0)),
                "object_min_tip_dist_px": float(obj.get("min_tip_dist_px", math.inf)),
                "object_depth_z": summarize(object_z[np.isfinite(object_z) & (object_z > 0.0)]),
                "all_hand_depth_z": hand_report,
                "mask_near_hand_depth_z": near_report,
                "mask_near_hand_minus_object_depth_z": gap_report,
            }
        )
    ok = [row for row in reports if row.get("status") == "ok"]
    gaps = []
    for row in ok:
        gap = row.get("mask_near_hand_minus_object_depth_z")
        if gap and gap.get("count", 0):
            gaps.append(gap["median"])
    summary = {
        "status": "ok",
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames_reported": len(reports),
        "frames_with_near_mask_hands": int(len(gaps)),
        "near_mask_hand_object_depth_gap_median_m": float(np.median(np.asarray(gaps))) if gaps else None,
        "near_mask_hand_object_depth_gap_p95_m": float(np.percentile(np.asarray(gaps), 95.0)) if gaps else None,
        "frame_reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frame_reports"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
