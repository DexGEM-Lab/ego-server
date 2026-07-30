#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


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


def read_manifest(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def frame_annotations(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def robust_extent(points: np.ndarray) -> np.ndarray:
    return np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)


def mesh_extent(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle mesh")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return {
        "path": str(path),
        "extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        "robust_extent_m": robust_extent(vertices).astype(float).tolist(),
        "center_m": np.median(vertices, axis=0).astype(float).tolist(),
    }


def hand_rows(frame: dict) -> list[dict]:
    out = []
    for hand in frame.get("hands", []):
        if not isinstance(hand, dict):
            continue
        vertices = np.asarray(hand.get("vertices_source_camera_m", []), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            continue
        out.append(
            {
                "side": hand.get("side"),
                "score": float(hand.get("detector_score", 0.0)),
                "depth_median_m": float(np.median(vertices[:, 2])),
                "extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
                "robust_extent_m": robust_extent(vertices).astype(float).tolist(),
            }
        )
    return out


def vggt_frame_rows(vggt_archive: Path, frame_indices: list[int]) -> dict[int, dict]:
    blob = np.load(vggt_archive)
    required = {
        "frame_idx",
        "vertex_offsets",
        "object_points_vggt",
        "extrinsic",
        "intrinsic",
        "sim3_scale",
    }
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{vggt_archive} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    points = blob["object_points_vggt"].astype(np.float64)
    extrinsic = blob["extrinsic"].astype(np.float64)
    intrinsic = blob["intrinsic"].astype(np.float64)
    sim3_scale = float(blob["sim3_scale"][0])
    out = {}
    for frame_idx in frame_indices:
        hits = np.where(frames == int(frame_idx))[0]
        if len(hits) != 1:
            raise RuntimeError(f"VGGT archive has {len(hits)} rows for frame {frame_idx}")
        i = int(hits[0])
        start, end = int(offsets[i]), int(offsets[i + 1])
        pts = points[start:end]
        pts_cam = pts @ extrinsic[i, :3, :3].T + extrinsic[i, :3, 3][None, :]
        pts_cam = pts_cam[np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 0)]
        if len(pts_cam) == 0:
            raise RuntimeError(f"VGGT frame {frame_idx} has no positive camera points")
        out[int(frame_idx)] = {
            "vggt_unscaled_depth_median": float(np.median(pts_cam[:, 2])),
            "vggt_unscaled_robust_extent": robust_extent(pts_cam).astype(float).tolist(),
            "vggt_sim3_scaled_depth_median": float(np.median(pts_cam[:, 2]) * sim3_scale),
            "vggt_sim3_scaled_robust_extent": (robust_extent(pts_cam) * sim3_scale).astype(float).tolist(),
            "vggt_padded_fx_fy": [float(intrinsic[i, 0, 0]), float(intrinsic[i, 1, 1])],
            "sim3_scale": float(sim3_scale),
        }
    return out


def run(args: argparse.Namespace) -> dict:
    manifest = read_manifest(args.manifest)
    annotations = frame_annotations(args.annotations)
    frame_indices = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in manifest]
    if len(frame_indices) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_indices)} manifest frames selected")
    vggt = vggt_frame_rows(args.vggt_archive, frame_indices)
    rows = []
    for frame_idx in frame_indices:
        entry = manifest[frame_idx]
        frame = annotations.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise RuntimeError(f"failed to read mask/depth for frame {frame_idx}")
        x0, y0, x1, y1 = mask_bbox(mask)
        mask_width = x1 - x0
        mask_height = y1 - y0
        depth_m = depth.astype(np.float64) / 1000.0
        mask_valid_depth = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0)]
        if mask_valid_depth.size == 0:
            raise RuntimeError(f"frame {frame_idx} has no valid mask depth")
        intr = frame["camera"].get("vggt_source_intrinsics_fx_fy_cx_cy")
        if intr is None:
            intr = [float(args.default_vggt_fx), float(args.default_vggt_fy), mask.shape[1], mask.shape[0]]
        vfx, vfy = float(intr[0]), float(intr[1])
        droid_fx = float(args.droid_fx)
        droid_fy = float(args.droid_fy)
        depth_median = float(np.median(mask_valid_depth))
        vrow = vggt[frame_idx]
        vggt_depth = float(vrow["vggt_unscaled_depth_median"])
        row = {
            "frame_idx": int(frame_idx),
            "mask_width_px": int(mask_width),
            "mask_height_px": int(mask_height),
            "depth_anything_median_m": depth_median,
            "source_intrinsics_vggt_fx_fy": [vfx, vfy],
            "source_intrinsics_droid_fx_fy": [droid_fx, droid_fy],
            "width_from_depth_anything_droid_focal_m": float(mask_width * depth_median / droid_fx),
            "height_from_depth_anything_droid_focal_m": float(mask_height * depth_median / droid_fy),
            "width_from_vggt_depth_vggt_focal_m": float(mask_width * vggt_depth / vfx),
            "height_from_vggt_depth_vggt_focal_m": float(mask_height * vggt_depth / vfy),
            "width_from_vggt_sim3_depth_vggt_focal_m": float(mask_width * vrow["vggt_sim3_scaled_depth_median"] / vfx),
            "height_from_vggt_sim3_depth_vggt_focal_m": float(mask_height * vrow["vggt_sim3_scaled_depth_median"] / vfy),
            "vggt": vrow,
            "hands": hand_rows(frame),
        }
        rows.append(row)
    report = {
        "status": "ok",
        "method": "diagnose_object_scale_ownership_v3",
        "annotations": str(args.annotations),
        "manifest": str(args.manifest),
        "vggt_archive": str(args.vggt_archive),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(rows)),
        "width_from_depth_anything_droid_focal_m": summarize([r["width_from_depth_anything_droid_focal_m"] for r in rows]),
        "width_from_vggt_depth_vggt_focal_m": summarize([r["width_from_vggt_depth_vggt_focal_m"] for r in rows]),
        "width_from_vggt_sim3_depth_vggt_focal_m": summarize([r["width_from_vggt_sim3_depth_vggt_focal_m"] for r in rows]),
        "vggt_unscaled_depth_median": summarize([r["vggt"]["vggt_unscaled_depth_median"] for r in rows]),
        "vggt_sim3_scaled_depth_median": summarize([r["vggt"]["vggt_sim3_scaled_depth_median"] for r in rows]),
        "mesh_extents": [mesh_extent(path) for path in args.meshes],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--meshes", type=Path, nargs="+", default=[])
    parser.add_argument("--droid-fx", type=float, default=2304.0)
    parser.add_argument("--droid-fy", type=float, default=2304.0)
    parser.add_argument("--default-vggt-fx", type=float, default=1196.0)
    parser.add_argument("--default-vggt-fy", type=float, default=1175.0)
    parser.add_argument("--min-frames", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
