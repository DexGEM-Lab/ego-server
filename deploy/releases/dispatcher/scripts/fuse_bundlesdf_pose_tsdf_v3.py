#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_intrinsics(dataset: Path, manifest: dict) -> tuple[float, float, float, float]:
    qc_path = dataset / "qc_bundlesdf_dataset_v3.json"
    if qc_path.exists():
        qc = load_json(qc_path)
        values = qc.get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return tuple(float(v) for v in values)
    values = manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return tuple(float(v) for v in values)
    cam_k = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if cam_k.shape != (3, 3):
        raise RuntimeError(f"{dataset / 'cam_K.txt'} must be a 3x3 matrix")
    return float(cam_k[0, 0]), float(cam_k[1, 1]), float(cam_k[0, 2]), float(cam_k[1, 2])


def entry_path(dataset: Path, subdir: str, index: int) -> Path:
    path = dataset / subdir / f"{index:06d}.png"
    if not path.exists():
        raise RuntimeError(f"missing {subdir} frame {index}: {path}")
    return path


def depth_window(depth_m: np.ndarray, mask: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict]:
    values = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0.05)]
    if values.size == 0:
        raise RuntimeError("masked depth has no valid samples")
    lo = float(np.quantile(values, low_q))
    hi = float(np.quantile(values, high_q))
    keep = mask & np.isfinite(depth_m) & (depth_m >= lo) & (depth_m <= hi)
    return keep, {
        "depth_samples": int(values.size),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "kept_depth_samples": int(np.count_nonzero(keep)),
    }


def integrate_frame(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    ob_in_cam: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    if rgb_bgr.shape[:2] != depth_mm.shape[:2] or mask.shape != depth_mm.shape[:2]:
        raise RuntimeError("RGB, depth, and mask shapes must match")
    if ob_in_cam.shape != (4, 4) or not np.isfinite(ob_in_cam).all():
        raise RuntimeError("BundleSDF ob_in_cam pose must be finite 4x4")

    depth_m = depth_mm.astype(np.float32) / 1000.0
    raw_mask = mask > 0
    if args.mask_erode_px > 0:
        kernel = np.ones((2 * args.mask_erode_px + 1, 2 * args.mask_erode_px + 1), dtype=np.uint8)
        raw_mask = cv2.erode(raw_mask.astype(np.uint8), kernel, iterations=1) > 0
    keep, stats = depth_window(depth_m, raw_mask, args.depth_low_quantile, args.depth_high_quantile)
    if np.count_nonzero(keep) < args.min_depth_pixels:
        raise RuntimeError(f"only {np.count_nonzero(keep)} masked depth pixels after depth filtering")

    filtered_depth = np.where(keep, depth_mm, 0).astype(np.uint16)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    color = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(filtered_depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000.0,
        depth_trunc=float(args.depth_trunc_m),
        convert_rgb_to_intensity=False,
    )
    volume.integrate(rgbd, intrinsic, ob_in_cam.astype(np.float64))
    return {
        "mask_pixels": int(np.count_nonzero(raw_mask)),
        **stats,
    }


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_component_faces: int) -> tuple[o3d.geometry.TriangleMesh, dict]:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("TSDF extracted an empty mesh")

    triangle_labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(triangle_labels, dtype=np.int32)
    counts_arr = np.asarray(counts, dtype=np.int64)
    areas_arr = np.asarray(areas, dtype=np.float64)
    keep_labels = np.flatnonzero(counts_arr >= int(min_component_faces))
    if len(keep_labels) == 0:
        keep_labels = np.asarray([int(np.argmax(counts_arr))], dtype=np.int64)
    remove = ~np.isin(labels, keep_labels)
    filtered = o3d.geometry.TriangleMesh(mesh)
    filtered.remove_triangles_by_mask(remove.tolist())
    filtered.remove_unreferenced_vertices()
    filtered.remove_duplicated_vertices()
    filtered.remove_duplicated_triangles()
    filtered.remove_degenerate_triangles()
    filtered.compute_vertex_normals()
    return filtered, {
        "raw_vertices": int(len(vertices)),
        "raw_faces": int(len(faces)),
        "components": int(len(counts_arr)),
        "component_faces": counts_arr.astype(int).tolist(),
        "component_area_m2": areas_arr.astype(float).tolist(),
        "kept_component_labels": keep_labels.astype(int).tolist(),
        "kept_vertices": int(len(filtered.vertices)),
        "kept_faces": int(len(filtered.triangles)),
    }


def mesh_extent(mesh: o3d.geometry.TriangleMesh) -> list[float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("mesh has no vertices")
    return (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist()


def run(args: argparse.Namespace) -> None:
    manifest_path = args.manifest or (args.dataset / "manifest.json")
    manifest = load_json(manifest_path)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{manifest_path} must contain a nonempty frames list")

    fx, fy, cx, cy = load_intrinsics(args.dataset, manifest)
    first_rgb = cv2.imread(str(entry_path(args.dataset, "rgb", int(entries[0]["index"]))), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError("failed to read first RGB frame")
    height, width = first_rgb.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_length_m),
        sdf_trunc=float(args.sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    rows = []
    for entry in entries:
        index = int(entry["index"])
        pose_index = int(entry.get("source_index", index))
        frame_idx = int(entry["frame_idx"])
        rgb = cv2.imread(str(entry_path(args.dataset, "rgb", index)), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(entry_path(args.dataset, "depth", index)), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(entry_path(args.dataset, "masks", index)), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise RuntimeError(f"failed to read RGB/depth/mask for dataset index {index}")
        pose_path = args.bundlesdf_output / "ob_in_cam" / f"{pose_index:06d}.txt"
        if not pose_path.exists():
            raise RuntimeError(f"missing BundleSDF pose: {pose_path}")
        ob_in_cam = np.loadtxt(pose_path).astype(np.float64)
        stats = integrate_frame(volume, intrinsic, rgb, depth, mask, ob_in_cam, args)
        rows.append({"index": index, "frame_idx": frame_idx, "pose": str(pose_path), **stats})

    raw_mesh = volume.extract_triangle_mesh()
    mesh, clean_stats = clean_mesh(raw_mesh, int(args.min_component_faces))
    extent = mesh_extent(mesh)
    max_extent = max(extent)
    if max_extent > args.max_extent_m:
        raise RuntimeError(f"TSDF mesh extent {extent} exceeds --max-extent-m {args.max_extent_m}")
    if max_extent < args.min_extent_m:
        raise RuntimeError(f"TSDF mesh extent {extent} is below --min-extent-m {args.min_extent_m}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "object_tsdf_mesh.obj"
    point_path = args.output_dir / "object_tsdf_point_cloud.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write mesh {mesh_path}")
    point_cloud = volume.extract_point_cloud()
    if not o3d.io.write_point_cloud(str(point_path), point_cloud, write_ascii=False):
        raise RuntimeError(f"failed to write point cloud {point_path}")

    report = {
        "status": "ok",
        "method": "bundlesdf_pose_masked_depth_tsdf_v3",
        "dataset": str(args.dataset),
        "manifest": str(manifest_path),
        "bundlesdf_output": str(args.bundlesdf_output),
        "mesh": str(mesh_path),
        "point_cloud": str(point_path),
        "frames": int(len(rows)),
        "intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
        "voxel_length_m": float(args.voxel_length_m),
        "sdf_trunc_m": float(args.sdf_trunc_m),
        "mask_erode_px": int(args.mask_erode_px),
        "depth_quantiles": [float(args.depth_low_quantile), float(args.depth_high_quantile)],
        "mesh_extent_m": extent,
        **clean_stats,
        "rows": rows,
    }
    (args.output_dir / "qc_bundlesdf_pose_tsdf_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bundlesdf-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--voxel-length-m", type=float, default=0.004)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.018)
    parser.add_argument("--depth-trunc-m", type=float, default=2.0)
    parser.add_argument("--depth-low-quantile", type=float, default=0.01)
    parser.add_argument("--depth-high-quantile", type=float, default=0.99)
    parser.add_argument("--mask-erode-px", type=int, default=0)
    parser.add_argument("--min-depth-pixels", type=int, default=2500)
    parser.add_argument("--min-component-faces", type=int, default=500)
    parser.add_argument("--min-extent-m", type=float, default=0.03)
    parser.add_argument("--max-extent-m", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
