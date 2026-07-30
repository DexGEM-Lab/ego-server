#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json, load_mesh_archive


def summarize(values: list[float] | np.ndarray) -> dict:
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


def summarize_by_mask_distance(depth_error: np.ndarray, mask_distance_px: np.ndarray) -> list[dict]:
    bins = [
        ("0_2px", 0.0, 2.0),
        ("2_5px", 2.0, 5.0),
        ("5_10px", 5.0, 10.0),
        ("10_20px", 10.0, 20.0),
        ("20_40px", 20.0, 40.0),
        ("40px_plus", 40.0, np.inf),
    ]
    rows = []
    for name, lo, hi in bins:
        if np.isinf(hi):
            take = mask_distance_px >= lo
        else:
            take = (mask_distance_px >= lo) & (mask_distance_px < hi)
        err = depth_error[take]
        rows.append(
            {
                "mask_distance_bin": name,
                "distance_px_low": float(lo),
                "distance_px_high": None if np.isinf(hi) else float(hi),
                "signed_m": summarize(err),
                "abs_m": summarize(np.abs(err)),
                "closer_than_depth_fraction_5mm": float(np.mean(err < -0.005)) if len(err) else None,
                "farther_than_depth_fraction_5mm": float(np.mean(err > 0.005)) if len(err) else None,
            }
        )
    return rows


def triangle_zbuffer(shape: tuple[int, int], uv: np.ndarray, z: np.ndarray, faces: np.ndarray, max_faces: int | None) -> np.ndarray:
    height, width = shape
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    valid_face = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid_face)
    if max_faces is not None and len(face_ids) > int(max_faces):
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, int(max_faces), dtype=np.int64)]
    order = np.argsort(z[faces[face_ids]].min(axis=1))[::-1]
    for face_id in face_ids[order]:
        poly_f = uv[faces[int(face_id)]]
        if np.any(poly_f[:, 0] < -width) or np.any(poly_f[:, 0] > 2 * width):
            continue
        if np.any(poly_f[:, 1] < -height) or np.any(poly_f[:, 1] > 2 * height):
            continue
        poly = np.round(poly_f).astype(np.int32)
        x0 = max(0, int(poly[:, 0].min()))
        y0 = max(0, int(poly[:, 1].min()))
        x1 = min(width, int(poly[:, 0].max()) + 1)
        y1 = min(height, int(poly[:, 1].max()) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local_poly = poly - np.asarray([x0, y0], dtype=np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_poly, 1, cv2.LINE_AA)
        tri = poly_f.astype(np.float64)
        tri_z = z[faces[int(face_id)]].astype(np.float64)
        denom = (
            (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
            + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
        )
        if abs(float(denom)) < 1e-9:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((tri[1, 1] - tri[2, 1]) * (px - tri[2, 0]) + (tri[2, 0] - tri[1, 0]) * (py - tri[2, 1])) / denom
        w1 = ((tri[2, 1] - tri[0, 1]) * (px - tri[2, 0]) + (tri[0, 0] - tri[2, 0]) * (py - tri[2, 1])) / denom
        w2 = 1.0 - w0 - w1
        bary_inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        face_depth = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
        region = zbuf[y0:y1, x0:x1]
        update = (mask > 0) & bary_inside & np.isfinite(face_depth) & (face_depth > 0.0) & (face_depth < region)
        region[update] = face_depth[update]
    return zbuf


def vertex_zbuffer(shape: tuple[int, int], uv: np.ndarray, z: np.ndarray, radius_px: int) -> np.ndarray:
    height, width = shape
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    valid = np.all(np.isfinite(uv), axis=1) & np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return zbuf
    xy = np.rint(uv[valid]).astype(np.int64)
    depth = z[valid].astype(np.float32)
    radius = int(radius_px)
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    for dx, dy in offsets:
        x = xy[:, 0] + dx
        y = xy[:, 1] + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.minimum.at(zbuf, (y[inside], x[inside]), depth[inside])
    return zbuf


def mesh_zbuffer(
    shape: tuple[int, int],
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    max_faces: int | None,
    vertex_radius_px: int,
    surface_mode: str = "triangles-plus-vertices",
) -> np.ndarray:
    zbuf = triangle_zbuffer(shape, uv, z, faces, max_faces)
    if surface_mode == "triangles":
        return zbuf
    if surface_mode != "triangles-plus-vertices":
        raise RuntimeError(f"unknown z-buffer surface mode: {surface_mode}")
    if int(vertex_radius_px) < 0:
        raise RuntimeError("vertex z-buffer radius must be non-negative")
    vertex_buf = vertex_zbuffer(shape, uv, z, int(vertex_radius_px))
    return np.minimum(zbuf, vertex_buf)


def draw_review(rgb: np.ndarray, object_mask: np.ndarray, silhouette: np.ndarray, zbuf: np.ndarray, row: dict) -> np.ndarray:
    image = rgb.copy()
    overlay = image.copy()
    overlay[object_mask] = (40, 170, 255)
    overlay[silhouette] = (70, 220, 80)
    overlay[object_mask & silhouette] = (255, 220, 60)
    cv2.addWeighted(overlay, 0.34, image, 0.66, 0, image)
    depth = zbuf[np.isfinite(zbuf)]
    if len(depth):
        norm = np.zeros_like(zbuf, dtype=np.uint8)
        lo, hi = np.percentile(depth, [5.0, 95.0])
        if hi > lo:
            norm[np.isfinite(zbuf)] = np.clip((zbuf[np.isfinite(zbuf)] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            depth_mask = np.isfinite(zbuf)
            image[depth_mask] = cv2.addWeighted(image, 0.72, color, 0.28, 0)[depth_mask]
    text = (
        f"frame {row['frame_idx']}  IoU {row['silhouette_mask_iou']:.3f}  "
        f"zbuf depth med {row.get('zbuffer_depth_abs_median_m', float('nan')):.3f}m"
    )
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def run(args: argparse.Namespace) -> dict:
    if args.max_faces is not None and int(args.max_faces) <= 0:
        args.max_faces = None
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    entries = manifest.get("frames")
    frames = annotations.get("frames")
    if not isinstance(entries, list) or not isinstance(frames, list):
        raise RuntimeError("manifest and annotations must contain frames lists")
    annotation_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    meshes = load_mesh_archive(args.mesh_archive)
    depth_archive = load_depth_archive(args.metric_depth_npz) if args.metric_depth_npz is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    rows = []
    writer = None
    selected_frame_set = None if args.frames is None else {int(frame) for frame in args.frames}
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if selected_frame_set is not None and frame_idx not in selected_frame_set:
            continue
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        if frame_idx not in meshes:
            raise RuntimeError(f"mesh archive lacks frame {frame_idx}")
        if frame_idx not in annotation_by_idx:
            raise RuntimeError(f"annotations lack frame {frame_idx}")
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise RuntimeError(f"failed to read RGB/mask for frame {frame_idx}")
        object_mask = mask > 0
        if depth_archive is None:
            depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"failed to read depth for frame {frame_idx}")
            depth_m = depth.astype(np.float64) / 1000.0
        else:
            if frame_idx not in depth_archive:
                raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
            depth_m = np.asarray(depth_archive[frame_idx], dtype=np.float64)
        if depth_m.shape != object_mask.shape:
            raise RuntimeError(f"depth shape {depth_m.shape} does not match mask shape {object_mask.shape}")
        vertices_world, faces = meshes[frame_idx]
        annotation = annotation_by_idx[frame_idx]
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        K = intrinsics_for_frame(args, entry, annotation)
        vertices_camera = camera_points(vertices_world, T_world_camera)
        z = vertices_camera[:, 2]
        uv = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
        positive = z > 0.0
        uv[positive, 0] = K[0, 0] * vertices_camera[positive, 0] / z[positive] + K[0, 2]
        uv[positive, 1] = K[1, 1] * vertices_camera[positive, 1] / z[positive] + K[1, 2]
        zbuf = mesh_zbuffer(
            object_mask.shape,
            uv,
            z,
            faces,
            args.max_faces,
            int(args.vertex_splat_radius_px),
            str(args.zbuffer_surface_mode),
        )
        silhouette = np.isfinite(zbuf)
        intersection = int(np.count_nonzero(silhouette & object_mask))
        union = int(np.count_nonzero(silhouette | object_mask))
        valid_depth = silhouette & object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
        depth_error = zbuf[valid_depth].astype(np.float64) - depth_m[valid_depth]
        distance_to_mask_edge = cv2.distanceTransform(object_mask.astype(np.uint8), cv2.DIST_L2, 3)
        depth_distance = distance_to_mask_edge[valid_depth].astype(np.float64)
        row = {
            "frame_idx": frame_idx,
            "silhouette_mask_iou": float(intersection / union) if union else 0.0,
            "silhouette_area_px": int(np.count_nonzero(silhouette)),
            "mask_area_px": int(np.count_nonzero(object_mask)),
            "zbuffer_depth_samples": int(len(depth_error)),
            "visible_silhouette_inside_mask_fraction": float(intersection / max(int(np.count_nonzero(silhouette)), 1)),
        }
        if len(depth_error):
            row.update(
                {
                    "zbuffer_depth_signed_m": summarize(depth_error),
                    "zbuffer_depth_median_m": float(np.median(depth_error)),
                    "zbuffer_depth_abs_median_m": float(np.median(np.abs(depth_error))),
                    "zbuffer_depth_abs_p95_m": float(np.percentile(np.abs(depth_error), 95.0)),
                    "zbuffer_closer_than_depth_fraction_5mm": float(np.mean(depth_error < -0.005)),
                    "zbuffer_farther_than_depth_fraction_5mm": float(np.mean(depth_error > 0.005)),
                    "zbuffer_depth_by_mask_distance": summarize_by_mask_distance(depth_error, depth_distance),
                }
            )
        rows.append(row)
        rendered = draw_review(rgb, object_mask, silhouette, zbuf, row)
        if args.render_width and rendered.shape[1] != int(args.render_width):
            height = int(round(int(args.render_width) * rendered.shape[0] / rendered.shape[1]))
            rendered = cv2.resize(rendered, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.output_dir / "mesh_zbuffer_projection_qc.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (rendered.shape[1], rendered.shape[0]),
            )
        writer.write(rendered)
        if frame_idx in set(args.still_frames):
            cv2.imwrite(str(still_dir / f"frame_{frame_idx:06d}.png"), rendered)
    if writer is not None:
        writer.release()
    if not rows:
        raise RuntimeError("no frames rendered")
    report = {
        "status": "ok",
        "method": "mesh_zbuffer_projection_qc_v3",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "intrinsics_source": str(args.intrinsics_source),
        "metric_depth_npz": str(args.metric_depth_npz) if args.metric_depth_npz is not None else None,
        "max_faces": None if args.max_faces is None else int(args.max_faces),
        "full_fidelity_zbuffer": bool(args.max_faces is None),
        "zbuffer_surface_mode": str(args.zbuffer_surface_mode),
        "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
        "frames": int(len(rows)),
        "silhouette_mask_iou": summarize([row["silhouette_mask_iou"] for row in rows]),
        "visible_silhouette_inside_mask_fraction": summarize([row["visible_silhouette_inside_mask_fraction"] for row in rows]),
        "zbuffer_depth_abs_median_m": summarize([row["zbuffer_depth_abs_median_m"] for row in rows if "zbuffer_depth_abs_median_m" in row]),
        "zbuffer_depth_abs_p95_m": summarize([row["zbuffer_depth_abs_p95_m"] for row in rows if "zbuffer_depth_abs_p95_m" in row]),
        "rows": rows,
        "video": str(args.output_dir / "mesh_zbuffer_projection_qc.mp4"),
        "stills_dir": str(still_dir),
    }
    (args.output_dir / "qc_mesh_zbuffer_projection_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--max-faces", type=int, default=60000)
    parser.add_argument("--zbuffer-surface-mode", choices=("triangles", "triangles-plus-vertices"), default="triangles-plus-vertices")
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
