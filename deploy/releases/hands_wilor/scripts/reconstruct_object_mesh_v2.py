#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import trimesh
from tqdm import tqdm

from fuse_v1_full_fidelity import load_droid_reconstruction, load_json, source_camera_ray


MEASURED_OBJECT_STATUSES = {
    "measured_sam_kalman",
    "measured_plan_sam",
    "measured_plan_sam_vlm_verified",
    "measured_sam2_vlm_points",
}


def finite_matrix(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{name} has invalid shape or non-finite values: got {arr.shape}, expected {shape}")
    return arr


def localize_mask_path(mask_path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    path = Path(mask_path)
    if path.exists():
        return path
    if remote_root is not None and local_root is not None:
        try:
            rel = path.relative_to(remote_root)
        except ValueError:
            rel = None
        if rel is not None:
            candidate = local_root / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(mask_path)


def nearest_droid_depth_map(recon: dict, frame_idx: int, max_keyframe_gap: int) -> tuple[np.ndarray, int] | None:
    tstamps = recon["tstamps"]
    nearest = int(np.argmin(np.abs(tstamps - frame_idx)))
    source_idx = int(tstamps[nearest])
    if abs(source_idx - frame_idx) > max_keyframe_gap:
        return None
    disp = np.asarray(recon["disps"][nearest], dtype=float)
    if disp.ndim != 2:
        raise RuntimeError(f"DROID disparity map must be 2D, got {disp.shape}")
    return disp, source_idx


def load_metric_depth_archive(path: Path | None) -> dict | None:
    if path is None:
        return None
    blob = np.load(path)
    required = {"frame_idx", "depth", "depth_size", "source_size"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"metric depth archive missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(float)
    if depth.ndim != 3 or len(frames) != depth.shape[0]:
        raise RuntimeError(f"invalid metric depth archive shapes: frame_idx={frames.shape}, depth={depth.shape}")
    return {
        "frame_to_i": {int(frame_idx): i for i, frame_idx in enumerate(frames)},
        "depth": depth,
        "depth_size": tuple(int(v) for v in blob["depth_size"].tolist()),
        "source_size": tuple(int(v) for v in blob["source_size"].tolist()),
    }


def robust_patch_median(stack: np.ndarray, min_finite: int = 5) -> np.ndarray:
    finite = np.isfinite(stack) & (stack > 1e-4)
    clean = stack.astype(float).copy()
    clean[~finite] = np.nan
    out = np.nanmedian(clean, axis=1)
    out[finite.sum(axis=1) < min_finite] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def metric_depth_for_pixels(metric_depth: dict | None, frame_idx: int, pixels: np.ndarray, image_size: tuple[int, int]) -> np.ndarray | None:
    if metric_depth is None:
        return None
    table = metric_depth["frame_to_i"]
    if frame_idx not in table:
        return None
    depth = metric_depth["depth"][table[frame_idx]]
    width, height = image_size
    x = np.clip(np.rint(pixels[:, 0] / width * depth.shape[1]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1] / height * depth.shape[0]).astype(int), 0, depth.shape[0] - 1)
    samples = []
    for dy in (-1, 0, 1):
        yy = np.clip(y + dy, 0, depth.shape[0] - 1)
        for dx in (-1, 0, 1):
            xx = np.clip(x + dx, 0, depth.shape[1] - 1)
            samples.append(depth[yy, xx])
    return robust_patch_median(np.stack(samples, axis=1))


def median_depth_from_disparity(disp: np.ndarray, pixels: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    x = np.clip(np.rint(pixels[:, 0] / width * disp.shape[1]).astype(int), 0, disp.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1] / height * disp.shape[0]).astype(int), 0, disp.shape[0] - 1)
    samples = []
    for dy in (-1, 0, 1):
        yy = np.clip(y + dy, 0, disp.shape[0] - 1)
        for dx in (-1, 0, 1):
            xx = np.clip(x + dx, 0, disp.shape[1] - 1)
            samples.append(disp[yy, xx])
    disp_med = robust_patch_median(np.stack(samples, axis=1))
    depth = 1.0 / disp_med
    depth[~np.isfinite(depth)] = np.nan
    return depth


def lattice_from_mask(mask: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride <= 0:
        raise RuntimeError("--mask-stride must be positive")
    ys = np.arange(0, mask.shape[0], stride, dtype=int)
    xs = np.arange(0, mask.shape[1], stride, dtype=int)
    mask_grid = mask[np.ix_(ys, xs)] > 0
    return xs, ys, mask_grid


def build_faces(index_grid: np.ndarray, vertices: np.ndarray, max_edge_m: float) -> np.ndarray:
    faces = []
    rows, cols = index_grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = int(index_grid[r, c])
            b = int(index_grid[r, c + 1])
            d = int(index_grid[r + 1, c])
            e = int(index_grid[r + 1, c + 1])
            for tri in ((a, b, d), (b, e, d)):
                if min(tri) < 0:
                    continue
                pts = vertices[np.asarray(tri, dtype=int)]
                edge = max(
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[0]),
                )
                if edge <= max_edge_m:
                    faces.append(tri)
    return np.asarray(faces, dtype=np.int32)


def remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used].astype(np.float32), remap[faces].astype(np.int32)


def mesh_from_mask_depth(
    frame: dict,
    mask: np.ndarray,
    recon: dict,
    metric_depth: dict | None,
    intrinsics: np.ndarray,
    droid_to_meters: float,
    mask_stride: int,
    max_keyframe_gap: int,
    max_triangle_edge_m: float,
    min_vertices: int,
    min_faces: int,
    depth_source: str,
    enable_contact_depth_correction: bool,
    max_contact_depth_shift_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    obj = frame["object"]
    if "source_image_size" not in obj or "mask_image_size" not in obj:
        raise RuntimeError(f"object frame {frame['frame_idx']} lacks source_image_size or mask_image_size")
    source_size = tuple(int(x) for x in obj["source_image_size"])
    mask_size = tuple(int(x) for x in obj["mask_image_size"])
    if len(source_size) != 2 or len(mask_size) != 2 or min(source_size) <= 0 or min(mask_size) <= 0:
        raise RuntimeError(f"object frame {frame['frame_idx']} has invalid image size metadata")
    xs, ys, mask_grid = lattice_from_mask(mask, mask_stride)
    if int(mask_grid.sum()) < min_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {"reason": "mask_underconstrained"}
    grid_x, grid_y = np.meshgrid(xs, ys)
    mask_pixels = np.c_[grid_x[mask_grid], grid_y[mask_grid]].astype(float)
    source_pixels = mask_pixels * np.asarray([source_size[0] / mask_size[0], source_size[1] / mask_size[1]], dtype=float)
    depth_keyframe = None
    if depth_source == "droid":
        droid_depth = nearest_droid_depth_map(recon, int(frame["frame_idx"]), max_keyframe_gap)
        if droid_depth is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {"reason": "no_droid_depth"}
        disp, depth_keyframe = droid_depth
        relative_depth = median_depth_from_disparity(disp, source_pixels, source_size)
        depths = relative_depth * float(droid_to_meters)
    elif depth_source == "metric_depth":
        metric = metric_depth_for_pixels(metric_depth, int(frame["frame_idx"]), source_pixels, source_size)
        if metric is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {"reason": "no_metric_depth"}
        depths = metric
    else:
        raise RuntimeError(f"unsupported depth source: {depth_source}")
    depths, contact_report = contact_depth_shift(
        frame,
        source_pixels,
        depths,
        intrinsics,
        enable_contact_depth_correction,
        max_contact_depth_shift_m,
    )
    valid_depth = np.isfinite(depths) & (depths >= 0.20) & (depths <= 3.20)
    if int(valid_depth.sum()) < min_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {"reason": "depth_underconstrained"}
    T = finite_matrix(frame["camera"]["T_world_camera_metric"], (4, 4), "T_world_camera_metric")
    vertices = []
    index_grid = np.full(mask_grid.shape, -1, dtype=np.int32)
    valid_positions = np.argwhere(mask_grid)
    for local_i, (r, c) in enumerate(valid_positions):
        if not valid_depth[local_i]:
            continue
        ray = source_camera_ray(source_pixels[local_i], intrinsics)
        point = (T @ np.r_[ray * float(depths[local_i]), 1.0])[:3]
        if not np.isfinite(point).all():
            continue
        index_grid[int(r), int(c)] = len(vertices)
        vertices.append(point)
    if len(vertices) < min_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), {"reason": "world_vertex_underconstrained"}
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces = build_faces(index_grid, vertices_arr, max_triangle_edge_m)
    vertices_arr, faces = remove_unreferenced(vertices_arr, faces)
    if len(vertices_arr) < min_vertices or len(faces) < min_faces:
        return vertices_arr, faces, {"reason": "mesh_underconstrained", "depth_keyframe": int(depth_keyframe)}
    report = {
        "reason": "ok",
        "depth_source": depth_source,
        "depth_keyframe": int(depth_keyframe) if depth_keyframe is not None else None,
        "contact_depth_correction": contact_report,
        "vertices": int(len(vertices_arr)),
        "triangles": int(len(faces)),
        "depth_median_m": float(np.median(depths[valid_depth])),
        "extent_m": (vertices_arr.max(axis=0) - vertices_arr.min(axis=0)).astype(float).tolist(),
    }
    return vertices_arr, faces, report


def contact_depth_shift(
    frame: dict,
    source_pixels: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    enabled: bool,
    max_shift_m: float,
) -> tuple[np.ndarray, dict]:
    report = {"enabled": bool(enabled), "applied": False, "shift_m": 0.0, "support_vertices": 0}
    if not enabled:
        return depths, report
    if max_shift_m <= 0.0 or not math.isfinite(max_shift_m):
        raise RuntimeError("--max-contact-depth-shift-m must be positive and finite when correction is enabled")
    obj = frame.get("object", {})
    if float(obj.get("contact_ratio", 0.0)) <= 0.0 and float(obj.get("min_tip_dist_px", math.inf)) > 10.0:
        report["reason"] = "no_2d_contact"
        return depths, report
    hands = []
    for hand in frame.get("hands", []):
        if "joints2d" not in hand or "joints3d_camera" not in hand or "cam_t" not in hand:
            continue
        xy = np.asarray(hand["joints2d"], dtype=float)
        xyz = np.asarray(hand["joints3d_camera"], dtype=float)
        cam_t = np.asarray(hand["cam_t"], dtype=float)
        if xy.ndim == 2 and xy.shape[1] == 2 and xyz.ndim == 2 and xyz.shape[1] == 3 and cam_t.shape == (3,):
            hands.append((xy, xyz[:, 2] + float(cam_t[2])))
    if not hands:
        report["reason"] = "no_hand_depth"
        return depths, report
    pixels = np.vstack([xy for xy, _ in hands])
    hand_depths = np.concatenate([z for _, z in hands])
    finite = np.isfinite(hand_depths) & (hand_depths > 0.0)
    if not np.any(finite):
        report["reason"] = "no_finite_hand_depth"
        return depths, report
    pixels = pixels[finite]
    hand_depths = hand_depths[finite]
    diff = source_pixels[:, None, :] - pixels[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    close = dist.min(axis=1) < 40.0
    valid_depth = np.isfinite(depths) & (depths > 0.0)
    support = close & valid_depth
    if int(support.sum()) < 8:
        report["reason"] = "insufficient_contact_support"
        report["support_vertices"] = int(support.sum())
        return depths, report
    nearest = np.argmin(dist[support], axis=1)
    target = hand_depths[nearest]
    shift = float(np.median(target - depths[support]))
    if not math.isfinite(shift):
        report["reason"] = "nonfinite_shift"
        report["support_vertices"] = int(support.sum())
        return depths, report
    shift = float(np.clip(shift, -float(max_shift_m), float(max_shift_m)))
    corrected = depths.copy()
    corrected[valid_depth] = np.maximum(0.05, corrected[valid_depth] + shift)
    report.update({"applied": abs(shift) > 1e-9, "shift_m": shift, "support_vertices": int(support.sum())})
    return corrected, report


def hand_surface_points(frame: dict) -> np.ndarray:
    pts = []
    for hand in frame.get("hands", []):
        for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
            if key in hand:
                arr = np.asarray(hand[key], dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr):
                    pts.append(arr)
                break
    if not pts:
        return np.zeros((0, 3), dtype=float)
    points = np.vstack(pts)
    return points[np.isfinite(points).all(axis=1)]


def min_hand_mesh_distance(frame: dict, vertices: np.ndarray) -> float | None:
    hands = hand_surface_points(frame)
    if len(hands) == 0 or len(vertices) == 0:
        return None
    if len(hands) > 1600:
        ids = np.linspace(0, len(hands) - 1, 1600, dtype=int)
        hands = hands[ids]
    if len(vertices) > 2400:
        ids = np.linspace(0, len(vertices) - 1, 2400, dtype=int)
        vertices = vertices[ids]
    best = math.inf
    chunk = 256
    for start in range(0, len(hands), chunk):
        diff = hands[start : start + chunk, None, :] - vertices[None, :, :]
        best = min(best, float(np.sqrt(np.min(np.sum(diff * diff, axis=2)))))
    return best if math.isfinite(best) else None


def export_review_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def save_mesh_archive(
    output_path: Path,
    frame_indices: list[int],
    vertices_per_frame: list[np.ndarray],
    faces_per_frame: list[np.ndarray],
) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    for vertices, faces in zip(vertices_per_frame, faces_per_frame):
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    vertices = np.vstack(vertices_per_frame).astype(np.float32) if vertices_per_frame else np.zeros((0, 3), dtype=np.float32)
    faces = np.vstack(faces_per_frame).astype(np.int32) if faces_per_frame else np.zeros((0, 3), dtype=np.int32)
    np.savez_compressed(
        output_path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=vertices,
        faces=faces,
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    frames = load_json(args.annotations)["frames"]
    recon = load_droid_reconstruction(args.droid_reconstruction)
    metric_depth = load_metric_depth_archive(args.metric_depth_npz)
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    droid_to_meters = float(args.droid_to_meters)
    if not math.isfinite(droid_to_meters):
        raise RuntimeError("--droid-to-meters must be finite")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices: list[int] = []
    vertices_per_frame: list[np.ndarray] = []
    faces_per_frame: list[np.ndarray] = []
    reports = []
    reason_counts: Counter[str] = Counter()
    contact_distances = []

    for frame in tqdm(frames, desc="object_mesh_v2"):
        obj = frame.get("object", {})
        frame_idx = int(frame["frame_idx"])
        if obj.get("status") not in MEASURED_OBJECT_STATUSES or not obj.get("mask_path"):
            reason_counts["not_measured"] += 1
            reports.append({"frame_idx": frame_idx, "reason": "not_measured"})
            continue
        mask_file = localize_mask_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read object mask {mask_file}")
        vertices, faces, report = mesh_from_mask_depth(
            frame,
            mask,
            recon,
            metric_depth,
            intrinsics,
            droid_to_meters,
            args.mask_stride,
            args.max_keyframe_gap,
            args.max_triangle_edge_m,
            args.min_vertices,
            args.min_faces,
            args.depth_source,
            bool(args.enable_contact_depth_correction),
            float(args.max_contact_depth_shift_m),
        )
        reason = str(report["reason"])
        reason_counts[reason] += 1
        report["frame_idx"] = frame_idx
        report["label"] = obj.get("label")
        if reason == "ok":
            dist = min_hand_mesh_distance(frame, vertices)
            if dist is not None:
                report["min_hand_mesh_distance_m"] = float(dist)
                contact_distances.append(float(dist))
            frame_indices.append(frame_idx)
            vertices_per_frame.append(vertices)
            faces_per_frame.append(faces)
            if args.review_stride > 0 and len(frame_indices) % args.review_stride == 1:
                export_review_ply(args.output_dir / "review_meshes" / f"{frame_idx:06d}.ply", vertices, faces)
        reports.append(report)

    if not frame_indices:
        raise RuntimeError(f"no object mesh frames reconstructed; reasons={dict(reason_counts)}")

    archive_path = args.output_dir / "dynamic_object_meshes.npz"
    save_mesh_archive(archive_path, frame_indices, vertices_per_frame, faces_per_frame)
    all_vertices = np.vstack(vertices_per_frame)
    triangles = np.asarray([len(faces) for faces in faces_per_frame], dtype=float)
    verts = np.asarray([len(vertices) for vertices in vertices_per_frame], dtype=float)
    contact_arr = np.asarray(contact_distances, dtype=float)
    contact_frame_count = int(contact_arr.size)
    contact_corrections = [
        float(report.get("contact_depth_correction", {}).get("shift_m", 0.0))
        for report in reports
        if report.get("reason") == "ok" and report.get("contact_depth_correction", {}).get("applied")
    ]
    depth_source_counts = Counter(
        str(report.get("depth_source"))
        for report in reports
        if report.get("reason") == "ok" and report.get("depth_source") is not None
    )
    worst_distance = None
    worst_distance_frame = None
    for report in reports:
        value = report.get("min_hand_mesh_distance_m")
        if value is None:
            continue
        value = float(value)
        if worst_distance is None or value > worst_distance:
            worst_distance = value
            worst_distance_frame = int(report["frame_idx"])
    qc = {
        "status": "ok",
        "annotations": str(args.annotations),
        "droid_reconstruction": str(args.droid_reconstruction),
        "metric_depth_npz": str(args.metric_depth_npz) if args.metric_depth_npz is not None else None,
        "depth_source": str(args.depth_source),
        "depth_source_counts": dict(sorted(depth_source_counts.items())),
        "mask_stride_px": int(args.mask_stride),
        "contact_depth_correction_enabled": bool(args.enable_contact_depth_correction),
        "contact_depth_correction_applied_frames": int(len(contact_corrections)),
        "contact_depth_shift_median_m": float(np.median(np.asarray(contact_corrections))) if contact_corrections else None,
        "contact_depth_shift_max_abs_m": float(np.max(np.abs(np.asarray(contact_corrections)))) if contact_corrections else None,
        "droid_to_meters": droid_to_meters,
        "frames": len(frames),
        "mesh_frames": int(len(frame_indices)),
        "hand_mesh_distance_frame_count": contact_frame_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "mesh_archive": str(archive_path),
        "vertices_total": int(sum(len(v) for v in vertices_per_frame)),
        "triangles_total": int(sum(len(f) for f in faces_per_frame)),
        "vertices_per_frame_median": float(np.median(verts)),
        "triangles_per_frame_median": float(np.median(triangles)),
        "bounds_min": all_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": all_vertices.max(axis=0).astype(float).tolist(),
        "hand_mesh_distance_median_m": float(np.median(contact_arr)) if contact_arr.size else None,
        "hand_mesh_distance_p05_m": float(np.percentile(contact_arr, 5)) if contact_arr.size else None,
        "hand_mesh_distance_p95_m": float(np.percentile(contact_arr, 95)) if contact_arr.size else None,
        "hand_mesh_distance_max_m": worst_distance,
        "hand_mesh_distance_max_frame": worst_distance_frame,
        "elapsed_s": time.time() - started,
        "frame_reports": reports,
    }
    (args.output_dir / "qc_object_mesh_v2.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "frame_reports"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--droid-reconstruction", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--droid-to-meters", type=float, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--max-keyframe-gap", type=int, default=6)
    parser.add_argument("--depth-source", choices=("metric_depth", "droid"), default="metric_depth")
    parser.add_argument("--enable-contact-depth-correction", action="store_true")
    parser.add_argument("--max-contact-depth-shift-m", type=float, default=0.03)
    parser.add_argument("--mask-stride", type=int, default=8)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.10)
    parser.add_argument("--min-vertices", type=int, default=40)
    parser.add_argument("--min-faces", type=int, default=30)
    parser.add_argument("--review-stride", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
