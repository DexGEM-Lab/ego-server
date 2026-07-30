#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m
from diagnose_hand_reprojection_depth_v3 import project_points


FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


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
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def load_depth_archive(path: Path, source_width: int, source_height: int) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    if depth.ndim != 3 or depth.shape[0] != len(frames):
        raise RuntimeError(f"{path} has invalid frame/depth shapes: {frames.shape}, {depth.shape}")
    if "source_size" in blob.files:
        source_size_raw = np.asarray(blob["source_size"], dtype=np.int64).reshape(-1)
        if source_size_raw.size != 2:
            raise RuntimeError(f"{path} source_size must have two values")
        source_size = (int(source_size_raw[0]), int(source_size_raw[1]))
    else:
        source_size = (int(source_width), int(source_height))
    source = {
        "frame_idx": frames,
        "depth": depth,
        "index": {int(frame_idx): i for i, frame_idx in enumerate(frames.tolist())},
        "source_size": source_size,
    }
    if "intrinsics_fx_fy_cx_cy" in blob.files:
        intrinsics = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        if intrinsics.shape != (len(frames), 4):
            raise RuntimeError(f"{path} intrinsics_fx_fy_cx_cy has invalid shape {intrinsics.shape}")
        source["intrinsics"] = intrinsics
    elif "focal_px" in blob.files:
        focal = np.asarray(blob["focal_px"], dtype=np.float64).reshape(-1)
        if focal.shape != (len(frames),):
            raise RuntimeError(f"{path} focal_px has invalid shape {focal.shape}")
        source["intrinsics"] = np.c_[
            focal,
            focal,
            np.full(len(frames), source_size[0] / 2.0, dtype=np.float64),
            np.full(len(frames), source_size[1] / 2.0, dtype=np.float64),
        ]
        source["intrinsics_note"] = "principal point set to image center because archive supplies focal_px only"
    else:
        raise RuntimeError(f"{path} must supply intrinsics_fx_fy_cx_cy or focal_px for hand-scale backprojection")
    return source


def parse_labeled_archives(entries: list[str], depthpro_archive: Path | None) -> dict[str, Path]:
    archives: dict[str, Path] = {}
    if depthpro_archive is not None:
        archives["depthpro"] = depthpro_archive
    for entry in entries:
        if "=" not in entry:
            raise RuntimeError(f"depth archive entry must be label=path, got {entry!r}")
        label, raw_path = entry.split("=", 1)
        label = label.strip()
        if not label or any((not ch.isalnum()) and ch != "_" for ch in label):
            raise RuntimeError(f"invalid depth archive label {label!r}; use letters, digits, and underscores")
        if label in {"manifest_depth", "vggt"}:
            raise RuntimeError(f"depth archive label {label!r} is reserved")
        if label in archives:
            raise RuntimeError(f"duplicate depth archive label {label!r}")
        archives[label] = Path(raw_path)
    return archives


def load_manifest_depth(path: Path, frame_start: int, frame_end: int, source_width: int, source_height: int) -> dict:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    selected = [row for row in frames if int(frame_start) <= int(row["frame_idx"]) <= int(frame_end)]
    selected.sort(key=lambda row: int(row["frame_idx"]))
    if not selected:
        raise RuntimeError(f"{path} has no frames in range {frame_start}:{frame_end}")
    frame_idx = []
    depths = []
    for row in selected:
        depth = cv2.imread(str(Path(row["depth"])), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"failed to read manifest depth: {row['depth']}")
        frame_idx.append(int(row["frame_idx"]))
        depths.append(depth.astype(np.float64) / 1000.0)
    return {
        "frame_idx": np.asarray(frame_idx, dtype=int),
        "depth": np.stack(depths, axis=0),
        "index": {int(frame): i for i, frame in enumerate(frame_idx)},
        "source_size": (int(source_width), int(source_height)),
    }


def source_intrinsics(
    intrinsic_vggt: np.ndarray,
    source_width: int,
    source_height: int,
    target_size: int,
) -> np.ndarray:
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
    return np.asarray(
        [
            float(intrinsic_vggt[0, 0] / sx),
            float(intrinsic_vggt[1, 1] / sy),
            float((intrinsic_vggt[0, 2] - pad_left) / sx),
            float((intrinsic_vggt[1, 2] - pad_top) / sy),
        ],
        dtype=np.float64,
    )


def vggt_affine(source_width: int, source_height: int, target_size: int) -> tuple[float, float, int, int]:
    if source_width >= source_height:
        new_width = int(target_size)
        new_height = round(source_height * (new_width / source_width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(source_width * (new_height / source_height) / 14) * 14
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    return new_width / float(source_width), new_height / float(source_height), pad_left, pad_top


def load_vggt_depth_source(path: Path, source_width: int, source_height: int, target_size: int) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsic"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    intrinsic = blob["intrinsic"].astype(np.float64)
    if depth.ndim != 3 or depth.shape[0] != len(frames) or intrinsic.shape[0] != len(frames):
        raise RuntimeError(f"{path} has invalid VGGT depth/intrinsic shapes")
    sx, sy, pad_left, pad_top = vggt_affine(source_width, source_height, target_size)
    intrinsics = [source_intrinsics(intrinsic[i], source_width, source_height, target_size) for i in range(len(frames))]
    return {
        "frame_idx": frames,
        "depth": depth,
        "index": {int(frame_idx): i for i, frame_idx in enumerate(frames.tolist())},
        "intrinsics": intrinsics,
        "source_to_depth_affine": (sx, sy, pad_left, pad_top),
        "source_size": (int(source_width), int(source_height)),
    }


def sample_depth(
    depth: np.ndarray,
    points_source: np.ndarray,
    affine: tuple[float, float, int, int] | None,
    source_size: tuple[int, int],
) -> np.ndarray:
    if affine is None:
        sx = depth.shape[1] / float(source_size[0])
        sy = depth.shape[0] / float(source_size[1])
        xy = points_source * np.asarray([sx, sy], dtype=np.float64)[None, :]
    else:
        sx, sy, pad_left, pad_top = affine
        xy = np.c_[points_source[:, 0] * sx + pad_left, points_source[:, 1] * sy + pad_top]
    valid = np.isfinite(xy).all(axis=1)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    out = np.full(len(points_source), np.nan, dtype=np.float64)
    out[valid] = depth[y[valid], x[valid]]
    return out


def hand_keypoints_2d(hand: dict, projected: np.ndarray) -> np.ndarray:
    raw = np.asarray(hand.get("joints2d_raw", []), dtype=np.float64)
    if raw.shape == (21, 2):
        return raw
    measured = np.asarray(hand.get("joints2d", []), dtype=np.float64)
    if measured.shape == (21, 2):
        return measured
    return projected


def backproject(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    return np.c_[(uv[:, 0] - cx) * depth / fx, (uv[:, 1] - cy) * depth / fy, depth]


def chain_scales(joints_ref: np.ndarray, joints_depth: np.ndarray) -> list[float]:
    scales = []
    for chain in FINGER_CHAINS:
        ref_length = 0.0
        depth_length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            ref_length += float(np.linalg.norm(joints_ref[b] - joints_ref[a]))
            depth_length += float(np.linalg.norm(joints_depth[b] - joints_depth[a]))
        if ref_length > 0.0 and np.isfinite(depth_length):
            scales.append(depth_length / ref_length)
    return scales


def depth_patch_iqr_ratio(
    depth: np.ndarray,
    xy_source: np.ndarray,
    affine: tuple[float, float, int, int] | None,
    source_size: tuple[int, int],
    radius: int,
) -> np.ndarray:
    if affine is None:
        sx = depth.shape[1] / float(source_size[0])
        sy = depth.shape[0] / float(source_size[1])
        xy = xy_source * np.asarray([sx, sy], dtype=np.float64)[None, :]
    else:
        sx, sy, pad_left, pad_top = affine
        xy = np.c_[xy_source[:, 0] * sx + pad_left, xy_source[:, 1] * sy + pad_top]
    ratios = []
    for x_raw, y_raw in xy:
        x = int(np.clip(round(float(x_raw)), 0, depth.shape[1] - 1))
        y = int(np.clip(round(float(y_raw)), 0, depth.shape[0] - 1))
        patch = depth[max(0, y - radius) : min(depth.shape[0], y + radius + 1), max(0, x - radius) : min(depth.shape[1], x + radius + 1)]
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if len(vals) == 0:
            ratios.append(float("nan"))
            continue
        med = float(np.median(vals))
        ratios.append(float((np.percentile(vals, 75.0) - np.percentile(vals, 25.0)) / max(1e-6, med)))
    return np.asarray(ratios, dtype=np.float64)


def depth_source_row(
    label: str,
    source: dict,
    frame_idx: int,
    uv: np.ndarray,
    projected_ref: np.ndarray,
    hand_intrinsics: np.ndarray,
    joints_ref: np.ndarray,
    args: argparse.Namespace,
) -> dict | None:
    src_i = source["index"].get(int(frame_idx))
    if src_i is None:
        return None
    depth = np.asarray(source["depth"][src_i], dtype=np.float64)
    affine = source.get("source_to_depth_affine")
    intrinsics = np.asarray(source.get("intrinsics", [hand_intrinsics])[src_i if "intrinsics" in source else 0], dtype=np.float64)
    source_size = tuple(int(v) for v in source["source_size"])
    sampled = sample_depth(depth, uv, affine, source_size)
    valid = np.isfinite(sampled) & (sampled > 0.0)
    if int(np.count_nonzero(valid)) < int(args.min_valid_joints):
        return {
            "depth_source": label,
            "valid_joints": int(np.count_nonzero(valid)),
            "status": "too_few_valid_joints",
        }
    joints_depth = backproject(uv, sampled, intrinsics)
    reproj_ref = np.linalg.norm(projected_ref - uv, axis=1)
    reproj_valid = valid & (reproj_ref <= float(args.max_reference_reprojection_px))
    if int(np.count_nonzero(reproj_valid)) < int(args.min_valid_joints):
        return {
            "depth_source": label,
            "valid_joints": int(np.count_nonzero(valid)),
            "reprojection_valid_joints": int(np.count_nonzero(reproj_valid)),
            "status": "too_few_reprojection_valid_joints",
        }
    scales = chain_scales(joints_ref, joints_depth)
    patch_ratio = depth_patch_iqr_ratio(depth, uv, affine, source_size, int(args.patch_radius_px))
    stable = patch_ratio[np.isfinite(patch_ratio)] <= float(args.max_depth_iqr_ratio)
    ref_bone_scale = hand_bone_scale_m(joints_ref)
    depth_bone_scale = hand_bone_scale_m(joints_depth)
    scale_ratio = depth_bone_scale / ref_bone_scale if ref_bone_scale > 0.0 else float("nan")
    return {
        "depth_source": label,
        "status": "ok",
        "valid_joints": int(np.count_nonzero(valid)),
        "reprojection_valid_joints": int(np.count_nonzero(reproj_valid)),
        "reference_bone_scale_m": float(ref_bone_scale),
        "depth_backprojected_bone_scale_m": float(depth_bone_scale),
        "depth_backprojected_over_reference_scale": float(scale_ratio),
        "chain_scale_median": float(np.median(scales)) if scales else float("nan"),
        "chain_scale_p95_abs_log": float(np.percentile(np.abs(np.log(np.asarray(scales, dtype=np.float64))), 95.0)) if scales else float("nan"),
        "depth_median_m": float(np.median(sampled[valid])),
        "depth_minus_reference_z_median_m": float(np.median(sampled[reproj_valid] - joints_ref[reproj_valid, 2])),
        "reference_reprojection_median_px": float(np.median(reproj_ref)),
        "stable_depth_fraction": float(np.mean(stable)) if len(stable) else float("nan"),
    }

def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations.get("frames", [])}
    if not frames:
        raise RuntimeError("annotations must contain frames")
    depth_archive_paths = parse_labeled_archives(args.depth_archive, args.depthpro_archive)
    depth_sources = {
        "manifest_depth": load_manifest_depth(
            args.manifest,
            int(args.frame_start),
            int(args.frame_end),
            int(args.source_width),
            int(args.source_height),
        ),
        "vggt": load_vggt_depth_source(args.vggt_archive, int(args.source_width), int(args.source_height), int(args.target_size)),
    }
    depth_sources.update(
        {
            label: load_depth_archive(path, int(args.source_width), int(args.source_height))
            for label, path in depth_archive_paths.items()
        }
    )
    rows = []
    skipped = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            score = float(hand.get("detector_score", np.nan))
            if not bool(hand.get("measurement_available", False)) or not np.isfinite(score) or score < float(args.min_detector_score):
                continue
            joints_ref = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=np.float64)
            if joints_ref.shape != (21, 3):
                local = np.asarray(hand.get("joints3d_camera", []), dtype=np.float64)
                cam_t = np.asarray(hand.get("cam_t", []), dtype=np.float64)
                if local.shape == (21, 3) and cam_t.shape == (3,):
                    joints_ref = local + cam_t[None, :]
            intrinsics = np.asarray(hand.get("source_intrinsics", []), dtype=np.float64)
            if joints_ref.shape != (21, 3) or intrinsics.shape != (4,):
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "reason": "invalid_hand_geometry"})
                continue
            if np.any(joints_ref[:, 2] <= 0.0):
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "reason": "non_positive_hand_depth"})
                continue
            projected_ref = project_points(joints_ref, intrinsics)
            uv = hand_keypoints_2d(hand, projected_ref)
            if uv.shape != (21, 2):
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "reason": "invalid_keypoints_2d"})
                continue
            for label, source in depth_sources.items():
                row = depth_source_row(label, source, frame_idx, uv, projected_ref, intrinsics, joints_ref, args)
                if row is None:
                    continue
                row.update(
                    {
                        "frame_idx": int(frame_idx),
                        "hand_idx": int(hand_i),
                        "side": hand.get("side"),
                        "detector_score": score,
                        "measurement_available": True,
                    }
                )
                rows.append(row)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_source = {}
    for source in sorted({row["depth_source"] for row in rows}):
        source_rows = [row for row in ok_rows if row["depth_source"] == source]
        by_source[source] = {
            "rows": int(len(source_rows)),
            "depth_backprojected_over_reference_scale": summarize(
                [row["depth_backprojected_over_reference_scale"] for row in source_rows]
            ),
            "depth_minus_reference_z_median_m": summarize([row["depth_minus_reference_z_median_m"] for row in source_rows]),
            "stable_depth_fraction": summarize([row["stable_depth_fraction"] for row in source_rows]),
            "reference_reprojection_median_px": summarize([row["reference_reprojection_median_px"] for row in source_rows]),
        }
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_depth_source_hand_scale_v3",
        "claim_tested": "candidate depth sources should backproject measured 2D hand keypoints to a 3D hand with MANO-consistent bone scale",
        "annotations": str(args.annotations),
        "manifest": str(args.manifest),
        "depth_sources": {
            "manifest_depth": "depth maps referenced by manifest rows",
            "vggt": str(args.vggt_archive),
            **{label: str(path) for label, path in depth_archive_paths.items()},
        },
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "rows": int(len(rows)),
        "ok_rows": int(len(ok_rows)),
        "by_source": by_source,
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "min_valid_joints": int(args.min_valid_joints),
            "max_reference_reprojection_px": float(args.max_reference_reprojection_px),
            "max_depth_iqr_ratio": float(args.max_depth_iqr_ratio),
        },
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:180],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depthpro-archive", type=Path)
    parser.add_argument("--depth-archive", action="append", default=[], help="Additional metric depth archive as label=path")
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-detector-score", type=float, default=0.5)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--max-reference-reprojection-px", type=float, default=20.0)
    parser.add_argument("--patch-radius-px", type=int, default=3)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
