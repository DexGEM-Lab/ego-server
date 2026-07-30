#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def manifest_frames(path: Path) -> list[dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return frames


def manifest_entry(path: Path, frame_idx: int) -> dict:
    frames = manifest_frames(path)
    matches = [entry for entry in frames if int(entry["frame_idx"]) == int(frame_idx)]
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(matches)} times in {path}")
    return matches[0]


def load_intrinsics(dataset: Path) -> np.ndarray:
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"invalid intrinsics: {dataset / 'cam_K.txt'}")
    return K


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise RuntimeError("mask has no foreground component")
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def fill_depth(mask: np.ndarray, depth: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    valid = mask & np.isfinite(depth) & (depth > float(args.min_depth_m)) & (depth < float(args.max_depth_m))
    vals = depth[valid]
    if len(vals) < int(args.min_depth_pixels):
        raise RuntimeError(f"only {len(vals)} valid mask depth pixels")
    lo = float(np.quantile(vals, float(args.depth_low_quantile)))
    hi = float(np.quantile(vals, float(args.depth_high_quantile)))
    clipped = np.clip(depth, lo, hi).astype(np.float32)
    clipped[~valid] = np.nan
    known = np.isfinite(clipped) & mask
    unknown = mask & ~known
    filled = clipped.copy()
    if np.any(unknown):
        unknown_fraction = float(np.count_nonzero(unknown) / np.count_nonzero(mask))
        if not args.allow_depth_fill or unknown_fraction > float(args.max_filled_depth_fraction):
            raise RuntimeError(
                f"mask has {unknown_fraction:.6f} invalid depth fraction; "
                "rerun with --allow-depth-fill only if this is intended"
            )
        nearest = cv2.distanceTransformWithLabels(
            (~known).astype(np.uint8),
            cv2.DIST_L2,
            3,
            labelType=cv2.DIST_LABEL_PIXEL,
        )[1]
        coords = np.column_stack(np.nonzero(known))
        flat_labels = nearest[unknown] - 1
        flat_labels = np.clip(flat_labels, 0, len(coords) - 1)
        yyxx = coords[flat_labels]
        filled[unknown] = clipped[yyxx[:, 0], yyxx[:, 1]]
    k = int(args.depth_smooth_ksize)
    if k > 1:
        if k % 2 == 0:
            k += 1
        smooth = cv2.GaussianBlur(np.nan_to_num(filled, nan=float(np.median(vals))), (k, k), float(args.depth_smooth_sigma))
        filled[mask] = smooth[mask]
    return filled, {
        "raw_depth_median_m": float(np.median(vals)),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "valid_depth_pixels": int(len(vals)),
        "filled_depth_pixels": int(np.count_nonzero(unknown)),
    }


def build_mesh(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, stride: int, max_edge_m: float) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, mask.shape[0], int(stride), dtype=np.int32)
    xs = np.arange(0, mask.shape[1], int(stride), dtype=np.int32)
    mask_grid = mask[np.ix_(ys, xs)]
    depth_grid = depth[np.ix_(ys, xs)]
    grid_x, grid_y = np.meshgrid(xs, ys)
    index = np.full(mask_grid.shape, -1, dtype=np.int32)
    pts = []
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    for r, c in np.argwhere(mask_grid):
        z = float(depth_grid[r, c])
        if not np.isfinite(z) or z <= 0.0:
            continue
        x = float(grid_x[r, c])
        y = float(grid_y[r, c])
        index[r, c] = len(pts)
        pts.append(((x - cx) * z / fx, (y - cy) * z / fy, z))
    vertices = np.asarray(pts, dtype=np.float32)
    faces = []
    for r in range(index.shape[0] - 1):
        for c in range(index.shape[1] - 1):
            corners = (int(index[r, c]), int(index[r, c + 1]), int(index[r + 1, c]), int(index[r + 1, c + 1]))
            for tri in ((corners[0], corners[1], corners[2]), (corners[1], corners[3], corners[2])):
                if min(tri) < 0:
                    continue
                tri_pts = vertices[np.asarray(tri, dtype=np.int32)]
                edge = max(
                    float(np.linalg.norm(tri_pts[0] - tri_pts[1])),
                    float(np.linalg.norm(tri_pts[1] - tri_pts[2])),
                    float(np.linalg.norm(tri_pts[2] - tri_pts[0])),
                )
                if edge <= float(max_edge_m):
                    faces.append(tri)
    faces_arr = np.asarray(faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces_arr) == 0:
        raise RuntimeError("heightfield mesh is empty")
    used = np.unique(faces_arr.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used].astype(np.float32), remap[faces_arr].astype(np.int32)


def save_archive(
    path: Path,
    frame_indices: list[int],
    vertices_camera: list[np.ndarray],
    faces: list[np.ndarray],
    annotations: Path | None,
) -> tuple[Path, Path | None]:
    vertex_offsets = [0]
    face_offsets = [0]
    for frame_vertices, frame_faces in zip(vertices_camera, faces, strict=True):
        vertex_offsets.append(vertex_offsets[-1] + len(frame_vertices))
        face_offsets.append(face_offsets[-1] + len(frame_faces))
    archive_kwargs = {
        "frame_idx": np.asarray(frame_indices, dtype=np.int32),
        "vertex_offsets": np.asarray(vertex_offsets, dtype=np.int64),
        "face_offsets": np.asarray(face_offsets, dtype=np.int64),
        "faces": np.vstack(faces).astype(np.int32),
    }
    np.savez_compressed(path, vertices=np.vstack(vertices_camera).astype(np.float32), **archive_kwargs)
    if annotations is None:
        return path, None
    frame_map = {int(frame["frame_idx"]): frame for frame in load_json(annotations)["frames"]}
    world_vertices = []
    for frame_idx, frame_vertices in zip(frame_indices, vertices_camera, strict=True):
        if frame_idx not in frame_map:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        T = np.asarray(frame_map[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        homog = np.c_[frame_vertices.astype(np.float64), np.ones(len(frame_vertices), dtype=np.float64)]
        world_vertices.append((T @ homog.T).T[:, :3].astype(np.float32))
    world_path = path.with_name(path.stem + "_world.npz")
    np.savez_compressed(world_path, vertices=np.vstack(world_vertices).astype(np.float32), **archive_kwargs)
    return path, world_path


def selected_entries(args: argparse.Namespace) -> list[dict]:
    if args.frame_idx is not None:
        return [manifest_entry(args.manifest, int(args.frame_idx))]
    if args.frame_start is None or args.frame_end is None:
        raise RuntimeError("provide --frame-idx or both --frame-start and --frame-end")
    entries = [
        entry
        for entry in manifest_frames(args.manifest)
        if int(args.frame_start) <= int(entry["frame_idx"]) <= int(args.frame_end)
    ]
    expected = list(range(int(args.frame_start), int(args.frame_end) + 1))
    actual = [int(entry["frame_idx"]) for entry in entries]
    if actual != expected and not args.allow_sparse_frames:
        raise RuntimeError(f"manifest frames are not contiguous over requested range: expected {expected}, got {actual}")
    if not entries:
        raise RuntimeError("no manifest frames selected")
    return entries


def reconstruct_entry(args: argparse.Namespace, entry: dict, K: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, dict]:
    frame_idx = int(entry["frame_idx"])
    mask_img = cv2.imread(str(entry["mask"]), cv2.IMREAD_GRAYSCALE)
    depth_img = cv2.imread(str(entry["depth"]), cv2.IMREAD_UNCHANGED)
    if mask_img is None or depth_img is None:
        raise RuntimeError(f"failed to read mask/depth for frame {frame_idx}")
    mask = largest_component(mask_img > 0)
    if args.mask_erode_px > 0:
        kernel = np.ones((2 * args.mask_erode_px + 1, 2 * args.mask_erode_px + 1), dtype=np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    depth_filled, depth_report = fill_depth(mask, depth_img.astype(np.float32) / 1000.0, args)
    vertices, faces = build_mesh(mask, depth_filled, K, int(args.pixel_stride), float(args.max_triangle_edge_m))
    if len(vertices) < int(args.min_vertices) or len(faces) < int(args.min_faces):
        raise RuntimeError(f"frame {frame_idx} heightfield underconstrained: vertices={len(vertices)} faces={len(faces)}")
    row = {
        "frame_idx": int(frame_idx),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "camera_extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        "depth": depth_report,
    }
    return frame_idx, vertices, faces, row


def run(args: argparse.Namespace) -> dict:
    entries = selected_entries(args)
    K = load_intrinsics(args.dataset)
    frame_indices = []
    vertices_all = []
    faces_all = []
    rows = []
    for entry in entries:
        frame_idx, vertices, faces, row = reconstruct_entry(args, entry, K)
        frame_indices.append(frame_idx)
        vertices_all.append(vertices)
        faces_all.append(faces)
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(frame_indices) == 1:
        archive_name = f"heightfield_frame_{frame_indices[0]:06d}.npz"
    else:
        archive_name = f"heightfield_frames_{frame_indices[0]:06d}_{frame_indices[-1]:06d}.npz"
    archive, world_archive = save_archive(args.output_dir / archive_name, frame_indices, vertices_all, faces_all, args.annotations)
    ext = np.asarray([row["camera_extent_m"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "mask_depth_heightfield_completion_v3",
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "camera_archive": str(archive),
        "world_archive": str(world_archive) if world_archive is not None else None,
        "vertices": int(sum(len(v) for v in vertices_all)),
        "faces": int(sum(len(f) for f in faces_all)),
        "camera_extent_median_m": np.median(ext, axis=0).astype(float).tolist(),
        "camera_extent_p05_m": np.percentile(ext, 5.0, axis=0).astype(float).tolist(),
        "camera_extent_p95_m": np.percentile(ext, 95.0, axis=0).astype(float).tolist(),
        "rows": rows,
        "parameters": {
            "pixel_stride": int(args.pixel_stride),
            "depth_low_quantile": float(args.depth_low_quantile),
            "depth_high_quantile": float(args.depth_high_quantile),
            "depth_smooth_ksize": int(args.depth_smooth_ksize),
            "allow_depth_fill": bool(args.allow_depth_fill),
        },
    }
    if len(frame_indices) == 1:
        report_path = args.output_dir / f"qc_heightfield_frame_{frame_indices[0]:06d}.json"
    else:
        report_path = args.output_dir / f"qc_heightfield_frames_{frame_indices[0]:06d}_{frame_indices[-1]:06d}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--frame-idx", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=3)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.025)
    parser.add_argument("--depth-low-quantile", type=float, default=0.05)
    parser.add_argument("--depth-high-quantile", type=float, default=0.95)
    parser.add_argument("--depth-smooth-ksize", type=int, default=9)
    parser.add_argument("--depth-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=3.20)
    parser.add_argument("--min-depth-pixels", type=int, default=5000)
    parser.add_argument("--min-vertices", type=int, default=200)
    parser.add_argument("--min-faces", type=int, default=300)
    parser.add_argument("--mask-erode-px", type=int, default=0)
    parser.add_argument("--allow-depth-fill", action="store_true")
    parser.add_argument("--allow-sparse-frames", action="store_true")
    parser.add_argument("--max-filled-depth-fraction", type=float, default=0.002)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
