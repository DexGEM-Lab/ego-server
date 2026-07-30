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


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        raise RuntimeError(f"mesh archive does not exist: {path}")
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"mesh archive missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError("mesh archive offsets do not match frame count")
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, source_idx in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        v = vertices[v0:v1]
        f = faces[f0:f1]
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError(f"empty mesh for frame {source_idx}")
        if f.min() < 0 or f.max() >= len(v):
            raise RuntimeError(f"face index out of range for frame {source_idx}")
        meshes[int(source_idx)] = (v, f)
    return meshes


def load_depth_archive(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    required = {"frame_idx", "depth"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    if depth.ndim != 3 or len(frame_idx) != depth.shape[0]:
        raise RuntimeError(f"{path} has invalid frame/depth shapes: {frame_idx.shape}, {depth.shape}")
    return {int(frame): depth[i] for i, frame in enumerate(frame_idx.tolist())}


def camera_points(world_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    T_camera_world = np.linalg.inv(T_world_camera)
    homog = np.c_[world_points, np.ones(len(world_points), dtype=np.float64)]
    return (T_camera_world @ homog.T).T[:, :3]


def project(camera_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = camera_xyz[:, 2]
    if np.any(z <= 0.0):
        raise RuntimeError("attempted to project points with non-positive camera depth")
    uv = np.empty((len(camera_xyz), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * camera_xyz[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * camera_xyz[:, 1] / z + K[1, 2]
    return uv


def render_silhouette(
    shape: tuple[int, int],
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    max_faces: int | None,
) -> np.ndarray:
    height, width = shape
    silhouette = np.zeros((height, width), dtype=np.uint8)
    valid_face = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid_face)
    if max_faces is not None and len(face_ids) > max_faces:
        step = max(1, len(face_ids) // max_faces)
        face_ids = face_ids[::step][:max_faces]
    order = np.argsort(z[faces[face_ids]].mean(axis=1))[::-1]
    for face_id in face_ids[order]:
        poly = uv[faces[int(face_id)]]
        if np.any(poly[:, 0] < -width) or np.any(poly[:, 0] > 2 * width):
            continue
        if np.any(poly[:, 1] < -height) or np.any(poly[:, 1] > 2 * height):
            continue
        cv2.fillConvexPoly(silhouette, np.round(poly).astype(np.int32), 255, cv2.LINE_AA)
    return silhouette > 0


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


def render_frame(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    mesh_silhouette: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    z: np.ndarray,
    metrics: dict,
    max_edges: int,
) -> np.ndarray:
    image = rgb.copy()
    mask_overlay = image.copy()
    mask_overlay[object_mask] = (40, 170, 255)
    mask_overlay[mesh_silhouette] = (80, 220, 80)
    both = object_mask & mesh_silhouette
    mask_overlay[both] = (255, 220, 60)
    cv2.addWeighted(mask_overlay, 0.34, image, 0.66, 0, image)

    valid_face = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid_face)
    if len(face_ids) > max_edges:
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, max_edges, dtype=int)]
    for face_id in face_ids:
        poly = np.round(uv[faces[int(face_id)]]).astype(np.int32)
        cv2.polylines(image, [poly], True, (0, 255, 0), 1, cv2.LINE_AA)

    text = (
        f"frame {metrics['frame_idx']}  IoU {metrics['silhouette_mask_iou']:.3f}  "
        f"depth med {metrics.get('vertex_depth_error_median_m', float('nan')):.3f}m"
    )
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def intrinsics_for_frame(args: argparse.Namespace, manifest: dict, annotation: dict) -> np.ndarray:
    if args.intrinsics_source == "manifest":
        vals = manifest.get("intrinsics_fx_fy_cx_cy")
        if vals is None:
            qc_path = args.manifest.parent / "qc_bundlesdf_dataset_v3.json"
            if not qc_path.exists():
                raise RuntimeError("manifest lacks intrinsics and dataset QC file is missing")
            vals = load_json(qc_path)["intrinsics_fx_fy_cx_cy"]
    elif args.intrinsics_source == "annotation-vggt":
        camera = annotation.get("camera")
        if not isinstance(camera, dict) or "vggt_source_intrinsics_fx_fy_cx_cy" not in camera:
            raise RuntimeError(f"frame {annotation.get('frame_idx')} missing camera.vggt_source_intrinsics_fx_fy_cx_cy")
        vals = camera["vggt_source_intrinsics_fx_fy_cx_cy"]
    else:
        raise RuntimeError(f"unsupported intrinsics source: {args.intrinsics_source}")
    fx, fy, cx, cy = [float(x) for x in vals]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    frames = annotations.get("frames")
    entries = manifest.get("frames")
    if not isinstance(frames, list) or not isinstance(entries, list):
        raise RuntimeError("annotations and manifest must contain frames lists")
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    meshes = load_mesh_archive(args.mesh_archive)
    depth_archive = load_depth_archive(args.metric_depth_npz) if args.metric_depth_npz is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    writer = None
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)

    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        annotation = frame_by_idx.get(frame_idx)
        if annotation is None:
            raise RuntimeError(f"missing annotation frame {frame_idx}")
        if frame_idx not in meshes:
            raise RuntimeError(f"missing mesh archive frame {frame_idx}")
        rgb_path = Path(entry["rgb"])
        mask_path = Path(entry["mask"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise RuntimeError(f"failed to read RGB/mask for frame {frame_idx}")
        object_mask = mask > 0
        if depth_archive is None:
            depth_path = Path(entry["depth"])
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"failed to read depth for frame {frame_idx}")
            depth_m = depth.astype(np.float64) / 1000.0
        else:
            if frame_idx not in depth_archive:
                raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
            depth_m = np.asarray(depth_archive[frame_idx], dtype=np.float64)
            if depth_m.shape != object_mask.shape:
                raise RuntimeError(f"metric depth frame {frame_idx} shape {depth_m.shape} does not match mask {object_mask.shape}")

        world_vertices, faces = meshes[frame_idx]
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        K = intrinsics_for_frame(args, manifest, annotation)
        cam_vertices = camera_points(world_vertices, T_world_camera)
        positive = cam_vertices[:, 2] > 0.0
        if np.count_nonzero(positive) < max(10, len(cam_vertices) // 20):
            raise RuntimeError(f"frame {frame_idx} has too few positive-depth mesh vertices")
        uv = np.full((len(cam_vertices), 2), np.nan, dtype=np.float64)
        uv[positive] = project(cam_vertices[positive], K)
        silhouette = render_silhouette(object_mask.shape, uv, cam_vertices[:, 2], faces, args.max_silhouette_faces)
        intersection = int(np.count_nonzero(silhouette & object_mask))
        union = int(np.count_nonzero(silhouette | object_mask))
        if union == 0:
            raise RuntimeError(f"frame {frame_idx} has empty mask/silhouette union")

        rounded = np.round(uv[positive]).astype(np.int32)
        in_bounds = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < rgb.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < rgb.shape[0])
        )
        rounded = rounded[in_bounds]
        depths = cam_vertices[positive, 2][in_bounds]
        if len(rounded) == 0:
            raise RuntimeError(f"frame {frame_idx} projects no vertices into the image")
        mask_hits = object_mask[rounded[:, 1], rounded[:, 0]]
        depth_hits = depth_m[rounded[:, 1], rounded[:, 0]]
        depth_valid = mask_hits & np.isfinite(depth_hits) & (depth_hits > 0.0)
        depth_errors = depths[depth_valid] - depth_hits[depth_valid]

        row = {
            "frame_idx": frame_idx,
            "silhouette_mask_iou": float(intersection / union),
            "silhouette_area_px": int(np.count_nonzero(silhouette)),
            "mask_area_px": int(np.count_nonzero(object_mask)),
            "projected_vertices_in_image": int(len(rounded)),
            "projected_vertices_inside_mask": int(np.count_nonzero(mask_hits)),
            "projected_vertex_mask_fraction": float(np.count_nonzero(mask_hits) / len(rounded)),
            "vertex_depth_samples": int(len(depth_errors)),
        }
        if len(depth_errors):
            row.update(
                {
                    "vertex_depth_error_median_m": float(np.median(depth_errors)),
                    "vertex_depth_error_abs_median_m": float(np.median(np.abs(depth_errors))),
                    "vertex_depth_error_abs_p95_m": float(np.percentile(np.abs(depth_errors), 95)),
                }
            )
        rows.append(row)

        rendered = render_frame(rgb, object_mask, silhouette, uv, faces, cam_vertices[:, 2], row, args.max_wire_faces)
        if args.render_width and rendered.shape[1] != args.render_width:
            height = int(round(args.render_width * rendered.shape[0] / rendered.shape[1]))
            rendered = cv2.resize(rendered, (args.render_width, height), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.output_dir / "bundlesdf_projection_qc.mp4"),
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

    summary = {
        "status": "ok",
        "method": "bundlesdf_mesh_projection_qc_v3",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "intrinsics_source": str(args.intrinsics_source),
        "metric_depth_npz": str(args.metric_depth_npz) if args.metric_depth_npz is not None else None,
        "frames": int(len(rows)),
        "silhouette_mask_iou": summarize([row["silhouette_mask_iou"] for row in rows]),
        "projected_vertex_mask_fraction": summarize([row["projected_vertex_mask_fraction"] for row in rows]),
        "vertex_depth_error_abs_median_m": summarize(
            [row["vertex_depth_error_abs_median_m"] for row in rows if "vertex_depth_error_abs_median_m" in row]
        ),
        "vertex_depth_error_abs_p95_m": summarize(
            [row["vertex_depth_error_abs_p95_m"] for row in rows if "vertex_depth_error_abs_p95_m" in row]
        ),
        "rows": rows,
        "video": str(args.output_dir / "bundlesdf_projection_qc.mp4"),
        "stills_dir": str(still_dir),
    }
    (args.output_dir / "qc_bundlesdf_projection_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--max-silhouette-faces", type=int, default=30000)
    parser.add_argument("--max-wire-faces", type=int, default=1200)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[858, 866, 878, 880])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
