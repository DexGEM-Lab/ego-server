#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_reprojection_depth_v3 import project_points


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_depth(path: Path) -> dict:
    blob = np.load(path)
    return {
        "frame_to_i": {int(idx): i for i, idx in enumerate(blob["frame_idx"].astype(int).tolist())},
        "depth": np.asarray(blob["depth"], dtype=np.float32),
        "source_size": np.asarray(blob["source_size"], dtype=float),
    }


def load_track(track_path: Path) -> dict[int, dict]:
    data = load_json(track_path)
    return {int(k): v for k, v in data.items() if isinstance(v, dict)}


def load_tracks(root: Path) -> list[dict]:
    tracks = []
    for track_path in sorted(root.glob("*/sam2/sam2_track.json")):
        qc_path = track_path.parent / "qc_sam2_image_points.json"
        qc = load_json(qc_path)
        tracks.append(
            {
                "track_id": str(qc["track_id"]),
                "track": load_track(track_path),
                "qc": qc,
            }
        )
    if not tracks:
        raise RuntimeError(f"no SAM2 tracks under {root}")
    return tracks


def localize(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    if remote_root is not None and local_root is not None:
        try:
            rel = raw.relative_to(remote_root)
        except ValueError:
            rel = None
        if rel is not None:
            candidate = local_root / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def mask_depth(depth: np.ndarray, mask_path: Path) -> tuple[float, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    values = depth[(mask > 0) & np.isfinite(depth) & (depth > 0.05)]
    if values.size == 0:
        raise RuntimeError(f"mask has no valid depth pixels: {mask_path}")
    return float(np.median(values)), int(values.size)


def hand_vertices(hand: dict) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_camera", "vertices_source_camera_m_sample", "vertices_camera_sample"):
        value = hand.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
    raise RuntimeError("hand has no usable vertices")


def hand_joints(hand: dict) -> np.ndarray:
    for key in ("joints3d_source_camera_m", "joints3d_camera"):
        value = hand.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=float)
            if arr.shape == (21, 3):
                return arr
    raise RuntimeError("hand has no usable joints")


def hand_intrinsics(hand: dict) -> np.ndarray:
    value = hand.get("source_intrinsics")
    if value is not None:
        arr = np.asarray(value, dtype=float)
        if arr.shape == (4,):
            return arr
    return np.asarray([2304.0, 2304.0, 960.0, 540.0], dtype=float)


def scaled_mask_distance(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def near_vertex_gaps(
    hand: dict,
    mask_path: Path,
    object_depth: float,
    depth_shape: tuple[int, int],
    source_size: np.ndarray,
    distance_px: float,
) -> tuple[np.ndarray, int]:
    vertices = hand_vertices(hand)
    intr = hand_intrinsics(hand)
    uv = project_points(vertices, intr)
    scale = np.asarray([depth_shape[1], depth_shape[0]], dtype=float) / source_size
    uv_depth = uv * scale[None, :]
    valid = np.isfinite(uv_depth).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    if not valid.any():
        return np.asarray([], dtype=float), 0
    dist = scaled_mask_distance(mask_path, depth_shape)
    x = np.clip(np.rint(uv_depth[:, 0]).astype(int), 0, depth_shape[1] - 1)
    y = np.clip(np.rint(uv_depth[:, 1]).astype(int), 0, depth_shape[0] - 1)
    near = valid & (dist[y, x] <= float(distance_px))
    gaps = vertices[near, 2] - float(object_depth)
    return gaps.astype(float), int(np.count_nonzero(near))


def keypoint_reprojection(hand: dict) -> float | None:
    raw2d = hand.get("joints2d_raw") or hand.get("joints2d")
    if raw2d is None:
        return None
    raw2d_arr = np.asarray(raw2d, dtype=float)
    if raw2d_arr.shape != (21, 2):
        return None
    joints = hand_joints(hand)
    uv = project_points(joints, hand_intrinsics(hand))
    err = np.linalg.norm(uv - raw2d_arr, axis=1)
    return float(np.median(err[np.isfinite(err)]))


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    depth_blob = load_depth(args.metric_depth_npz)
    tracks = load_tracks(args.sam2_root)
    rows = []
    for track in tracks:
        track_id = track["track_id"]
        for frame_idx, entry in sorted(track["track"].items()):
            if frame_idx < args.frame_start or frame_idx > args.frame_end:
                continue
            if not entry.get("visible") or not entry.get("mask_path"):
                continue
            if frame_idx not in depth_blob["frame_to_i"] or frame_idx not in frames:
                continue
            depth = depth_blob["depth"][depth_blob["frame_to_i"][frame_idx]]
            mask_path = localize(str(entry["mask_path"]), args.remote_output_root, args.local_output_root)
            object_depth, object_pixels = mask_depth(depth, mask_path)
            frame = frames[frame_idx]
            for hand_i, hand in enumerate(frame.get("hands", [])):
                if not hand.get("measurement_available", False):
                    continue
                score = float(hand.get("detector_score", np.nan))
                if not np.isfinite(score) or score < args.min_detector_score:
                    continue
                gaps, near_vertices = near_vertex_gaps(
                    hand,
                    mask_path,
                    object_depth,
                    depth.shape,
                    depth_blob["source_size"],
                    float(args.near_distance_px),
                )
                if near_vertices < int(args.min_near_vertices):
                    continue
                rows.append(
                    {
                        "frame_idx": int(frame_idx),
                        "track_id": track_id,
                        "hand_index": int(hand_i),
                        "side": str(hand.get("side", "unknown")),
                        "detector_score": score,
                        "object_depth_median_m": object_depth,
                        "object_depth_pixels": int(object_pixels),
                        "near_vertices": int(near_vertices),
                        "contact_gap_median_m": float(np.median(gaps)),
                        "contact_gap_p95_abs_m": float(np.percentile(np.abs(gaps), 95.0)),
                        "penetration_fraction_010m": float(np.mean(gaps < -0.010)),
                        "positive_gap_fraction_030m": float(np.mean(gaps > 0.030)),
                        "keypoint_reprojection_median_px": keypoint_reprojection(hand),
                    }
                )
    by_track = {}
    for track in tracks:
        tid = track["track_id"]
        track_rows = [row for row in rows if row["track_id"] == tid]
        by_track[tid] = {
            "rows": len(track_rows),
            "near_vertices": summarize(np.asarray([row["near_vertices"] for row in track_rows], dtype=float)),
            "contact_gap_median_m": summarize(np.asarray([row["contact_gap_median_m"] for row in track_rows], dtype=float)),
            "contact_gap_p95_abs_m": summarize(np.asarray([row["contact_gap_p95_abs_m"] for row in track_rows], dtype=float)),
            "penetration_fraction_010m": summarize(np.asarray([row["penetration_fraction_010m"] for row in track_rows], dtype=float)),
            "positive_gap_fraction_030m": summarize(np.asarray([row["positive_gap_fraction_030m"] for row in track_rows], dtype=float)),
            "keypoint_reprojection_median_px": summarize(
                np.asarray([row["keypoint_reprojection_median_px"] for row in track_rows if row["keypoint_reprojection_median_px"] is not None], dtype=float)
            ),
        }
    reliable = [
        row
        for row in rows
        if abs(row["contact_gap_median_m"]) <= args.accept_gap_m
        and row["contact_gap_p95_abs_m"] <= args.accept_p95_gap_m
        and row["penetration_fraction_010m"] <= args.accept_penetration_fraction
        and (row["keypoint_reprojection_median_px"] is None or row["keypoint_reprojection_median_px"] <= args.accept_reprojection_px)
    ]
    report = {
        "status": "diagnostic_fragment_contact_rows_found" if reliable else "diagnostic_no_reliable_fragment_contact_rows",
        "annotation_ready": False,
        "diagnostic_only": True,
        "sam2_root": str(args.sam2_root),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "rows": len(rows),
        "reliable_rows": len(reliable),
        "by_track": by_track,
        "rows_preview": rows[:240],
        "acceptance": {
            "accept_gap_m": float(args.accept_gap_m),
            "accept_p95_gap_m": float(args.accept_p95_gap_m),
            "accept_penetration_fraction": float(args.accept_penetration_fraction),
            "accept_reprojection_px": float(args.accept_reprojection_px),
        },
        "interpretation": (
            "This diagnostic treats SAM2/VLM masks as local observed contact-surface fragments. It tests measured MANO vertices "
            "against the median metric depth of each fragment. Passing rows are evidence for local contact only, not a complete object mesh."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--near-distance-px", type=float, default=18.0)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    parser.add_argument("--min-near-vertices", type=int, default=20)
    parser.add_argument("--min-detector-score", type=float, default=0.45)
    parser.add_argument("--accept-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-p95-gap-m", type=float, default=0.060)
    parser.add_argument("--accept-penetration-fraction", type=float, default=0.10)
    parser.add_argument("--accept-reprojection-px", type=float, default=18.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
