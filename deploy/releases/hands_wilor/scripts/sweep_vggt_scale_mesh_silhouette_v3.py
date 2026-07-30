#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from align_mesh_prior_v3 import choose_alignment, nearest_distances, sample_mesh_surface
from mesh_vggt_scene_object_points_v3 import build_frame_mesh
from render_bundlesdf_mesh_qc_v3 import camera_points, project, render_frame, render_silhouette


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


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def annotations_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )


def robust_extent(vertices: np.ndarray) -> np.ndarray:
    return np.quantile(vertices, 0.95, axis=0) - np.quantile(vertices, 0.05, axis=0)


def source_intrinsics(
    intrinsic_vggt: np.ndarray,
    source_width: int,
    source_height: int,
    target_size: int,
) -> list[float]:
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
    fx = float(intrinsic_vggt[0, 0] / sx)
    fy = float(intrinsic_vggt[1, 1] / sy)
    cx = float((intrinsic_vggt[0, 2] - pad_left) / sx)
    cy = float((intrinsic_vggt[1, 2] - pad_top) / sy)
    return [fx, fy, cx, cy]


def K_from_fx_fy_cx_cy(vals: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in vals]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_vggt_archive(path: Path) -> dict:
    blob = np.load(path)
    required = {
        "frame_idx",
        "vertex_offsets",
        "object_points_aligned",
        "object_points_vggt",
        "object_colors",
        "extrinsic",
        "intrinsic",
        "camera_centers_vggt",
        "sim3_scale",
        "sim3_rotation",
        "sim3_translation",
    }
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    return {
        "frame_idx": frame_idx,
        "index": {int(frame): i for i, frame in enumerate(frame_idx.tolist())},
        "offsets": blob["vertex_offsets"].astype(np.int64),
        "points_aligned": blob["object_points_aligned"].astype(np.float32),
        "points_vggt": blob["object_points_vggt"].astype(np.float32),
        "colors": blob["object_colors"].astype(np.uint8),
        "extrinsic": blob["extrinsic"].astype(np.float64),
        "intrinsic": blob["intrinsic"].astype(np.float64),
        "camera_centers_vggt": blob["camera_centers_vggt"].astype(np.float64),
        "sim3_scale": float(blob["sim3_scale"][0]),
        "sim3_rotation": blob["sim3_rotation"].astype(np.float64),
        "sim3_translation": blob["sim3_translation"].astype(np.float64),
    }


def load_depth_archive(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    required = {"frame_idx", "depth"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    if depth.ndim != 3 or len(frames) != depth.shape[0]:
        raise RuntimeError(f"{path} has invalid frame/depth shapes: {frames.shape}, {depth.shape}")
    out: dict[int, np.ndarray] = {}
    for i, frame_idx in enumerate(frames.tolist()):
        frame_depth = depth[i]
        if not np.isfinite(frame_depth).all():
            raise RuntimeError(f"{path} frame {frame_idx} contains non-finite depth")
        out[int(frame_idx)] = frame_depth
    return out


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
        if label in {"manifest", "depth"}:
            raise RuntimeError(f"depth archive label {label!r} is reserved")
        if label in archives:
            raise RuntimeError(f"duplicate depth archive label {label!r}")
        archives[label] = Path(raw_path)
    return archives


def depth_abs_key(label: str) -> str:
    return "vertex_depth_error_abs_median_m" if label == "manifest" else f"vertex_{label}_depth_error_abs_median_m"


def make_camera_surface(vggt: dict, frame_idx: int, scale: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    i = vggt["index"].get(int(frame_idx))
    if i is None:
        raise RuntimeError(f"VGGT archive lacks frame {frame_idx}")
    start = int(vggt["offsets"][i])
    end = int(vggt["offsets"][i + 1])
    vertices_aligned, faces, row = build_frame_mesh(
        vggt["points_vggt"][start:end],
        vggt["points_aligned"][start:end],
        vggt["colors"][start:end],
        vggt["extrinsic"][i],
        vggt["intrinsic"][i],
        float(vggt["sim3_scale"]),
        vggt["sim3_rotation"],
        vggt["sim3_translation"],
        int(args.target_size),
        int(args.grid_px),
        float(args.max_triangle_edge_m),
    )
    vertices_vggt = (
        (vertices_aligned.astype(np.float64) - vggt["sim3_translation"][None, :])
        @ vggt["sim3_rotation"]
        / float(vggt["sim3_scale"])
    )
    vertices_camera = float(scale) * (
        vertices_vggt @ vggt["extrinsic"][i, :3, :3].T + vggt["extrinsic"][i, :3, 3][None, :]
    )
    if np.count_nonzero(vertices_camera[:, 2] > 0.0) < max(10, len(vertices_camera) // 2):
        raise RuntimeError(f"frame {frame_idx} VGGT camera surface has too few positive-depth vertices")
    row.update(
        {
            "frame_idx": int(frame_idx),
            "camera_depth_median_m": float(np.median(vertices_camera[:, 2])),
            "camera_robust_extent_m": robust_extent(vertices_camera).astype(float).tolist(),
        }
    )
    if "extent_world_m" in row:
        row["vggt_sim3_aligned_extent_m"] = row.pop("extent_world_m")
    return vertices_camera.astype(np.float64), faces.astype(np.int32), row


def transform_camera_to_world(vertices_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[vertices_camera, np.ones(len(vertices_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def projection_metrics(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    manifest_row: dict,
    depth_sources: dict[str, dict[int, np.ndarray]],
    max_faces: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = cv2.imread(str(Path(manifest_row["mask"])), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(str(Path(manifest_row["rgb"])), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(Path(manifest_row["depth"])), cv2.IMREAD_UNCHANGED)
    if mask is None or rgb is None or depth is None:
        raise RuntimeError(f"failed to read RGB/mask/depth for frame {manifest_row['frame_idx']}")
    object_mask = mask > 0
    depth_m = depth.astype(np.float64) / 1000.0
    cam = camera_points(vertices_world, T_world_camera)
    positive = cam[:, 2] > 0.0
    if np.count_nonzero(positive) < max(10, len(cam) // 20):
        raise RuntimeError(f"frame {manifest_row['frame_idx']} projects too few positive-depth vertices")
    uv = np.full((len(cam), 2), np.nan, dtype=np.float64)
    uv[positive] = project(cam[positive], K)
    silhouette = render_silhouette(object_mask.shape, uv, cam[:, 2], faces, max_faces)
    intersection = int(np.count_nonzero(silhouette & object_mask))
    union = int(np.count_nonzero(silhouette | object_mask))
    rounded = np.round(uv[positive]).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < object_mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < object_mask.shape[0])
    )
    rounded = rounded[in_bounds]
    depths = cam[positive, 2][in_bounds]
    mask_hits = object_mask[rounded[:, 1], rounded[:, 0]] if len(rounded) else np.zeros(0, dtype=bool)
    depth_hits = depth_m[rounded[:, 1], rounded[:, 0]] if len(rounded) else np.zeros(0, dtype=np.float64)
    mask_area = int(np.count_nonzero(object_mask))
    silhouette_area = int(np.count_nonzero(silhouette))
    metrics = {
        "frame_idx": int(manifest_row["frame_idx"]),
        "silhouette_mask_iou": float(intersection / union) if union else 0.0,
        "silhouette_area_px": silhouette_area,
        "mask_area_px": mask_area,
        "silhouette_to_mask_area_ratio": float(silhouette_area / mask_area) if mask_area else 0.0,
        "projected_vertices_in_image": int(len(rounded)),
        "projected_vertices_inside_mask": int(np.count_nonzero(mask_hits)),
        "projected_vertex_mask_fraction": float(np.mean(mask_hits)) if len(mask_hits) else 0.0,
    }
    add_depth_residual(metrics, "manifest", depths, mask_hits, depth_hits)
    for label, source in depth_sources.items():
        frame_depth = source.get(int(manifest_row["frame_idx"]))
        if frame_depth is None:
            raise RuntimeError(f"depth source {label} lacks frame {manifest_row['frame_idx']}")
        if frame_depth.shape != object_mask.shape:
            raise RuntimeError(
                f"depth source {label} frame {manifest_row['frame_idx']} shape {frame_depth.shape} "
                f"does not match mask shape {object_mask.shape}"
            )
        hits = frame_depth[rounded[:, 1], rounded[:, 0]] if len(rounded) else np.zeros(0, dtype=np.float64)
        add_depth_residual(metrics, label, depths, mask_hits, hits)
    return metrics, rgb, object_mask, silhouette, uv, cam[:, 2]


def add_depth_residual(
    metrics: dict,
    label: str,
    projected_depths: np.ndarray,
    mask_hits: np.ndarray,
    sampled_depths: np.ndarray,
) -> None:
    valid = mask_hits & np.isfinite(sampled_depths) & (sampled_depths > 0.0)
    errors = projected_depths[valid] - sampled_depths[valid]
    prefix = "vertex_depth_error" if label == "manifest" else f"vertex_{label}_depth_error"
    metrics[f"{prefix}_samples"] = int(len(errors))
    if label == "manifest":
        metrics["vertex_depth_samples"] = int(len(errors))
    if len(errors):
        metrics.update(
            {
                f"{prefix}_median_m": float(np.median(errors)),
                f"{prefix}_abs_median_m": float(np.median(np.abs(errors))),
                f"{prefix}_abs_p95_m": float(np.percentile(np.abs(errors), 95.0)),
            }
        )


def observed_surface_distances(
    observed_vertices_camera: np.ndarray,
    observed_faces: np.ndarray,
    T_world_camera: np.ndarray,
    static_prior_world_samples: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    observed_mesh = trimesh.Trimesh(vertices=observed_vertices_camera, faces=observed_faces, process=False)
    observed_samples_camera = sample_mesh_surface(
        observed_mesh,
        min(int(samples), max(1000, len(observed_faces) * 4)),
        int(seed),
    )
    observed_samples_world = transform_camera_to_world(observed_samples_camera, T_world_camera)
    distances = nearest_distances(observed_samples_world, static_prior_world_samples)
    return {
        "samples": int(len(distances)),
        "observed_to_static_prior_median_m": float(np.median(distances)),
        "observed_to_static_prior_p95_m": float(np.percentile(distances, 95.0)),
        "observed_to_static_prior_max_m": float(np.max(distances)),
    }


def write_review_still(
    path: Path,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    silhouette: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    z: np.ndarray,
    metrics: dict,
    max_edges: int,
    width: int,
) -> None:
    rendered = render_frame(rgb, object_mask, silhouette, uv, faces, z, metrics, max_edges)
    if width and rendered.shape[1] != width:
        height = int(round(width * rendered.shape[0] / rendered.shape[1]))
        rendered = cv2.resize(rendered, (width, height), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rendered)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_mesh = load_mesh(args.mesh_prior)
    prior_points = sample_mesh_surface(prior_mesh, int(args.samples), int(args.seed))
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    frame_indices = [idx for idx in sorted(manifest) if int(args.frame_start) <= idx <= int(args.frame_end)]
    if not frame_indices:
        raise RuntimeError("no manifest frames selected")
    if int(args.anchor_frame) not in frame_indices:
        raise RuntimeError("anchor frame must be inside the selected manifest frames")
    vggt = load_vggt_archive(args.vggt_archive)
    depth_archive_paths = parse_labeled_archives(args.depth_archive, args.depthpro_archive)
    depth_sources = {label: load_depth_archive(path) for label, path in depth_archive_paths.items()}
    scales = np.asarray(args.scales, dtype=np.float64)
    rows = []
    for scale in scales:
        if not np.isfinite(scale) or float(scale) <= 0.0:
            raise RuntimeError(f"invalid scale {scale}")
        scale_value = float(scale)
        scale_dir = args.output_dir / f"scale_{scale_value:.4f}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        poses = camera_poses_for_scale(vggt, annotations, scale_value, int(args.anchor_frame), frame_indices, args)
        anchor_vertices_camera, anchor_faces, anchor_surface_row = make_camera_surface(
            vggt,
            int(args.anchor_frame),
            scale_value,
            args,
        )
        anchor_mesh = trimesh.Trimesh(vertices=anchor_vertices_camera, faces=anchor_faces, process=False)
        anchor_points = sample_mesh_surface(
            anchor_mesh,
            min(int(args.samples), max(1000, len(anchor_faces) * 4)),
            int(args.seed) + int(round(scale_value * 1000.0)),
        )
        sim, align_report = choose_alignment(prior_points, anchor_points)
        aligned_vertices_camera = sim.apply(np.asarray(prior_mesh.vertices, dtype=np.float64))
        aligned_prior_mesh_camera = trimesh.Trimesh(
            vertices=aligned_vertices_camera,
            faces=np.asarray(prior_mesh.faces, dtype=np.int32),
            process=False,
        )
        aligned_prior_mesh_camera.export(scale_dir / "aligned_prior_anchor_camera.obj")
        anchor_T = poses[int(args.anchor_frame)]["T_world_camera_metric"]
        static_vertices_world = transform_camera_to_world(aligned_vertices_camera, anchor_T)
        static_prior_world_samples = transform_camera_to_world(
            sample_mesh_surface(aligned_prior_mesh_camera, int(args.samples), int(args.seed) + 19),
            anchor_T,
        )

        frame_rows = []
        for frame_idx in frame_indices:
            pose = poses[frame_idx]
            K = K_from_fx_fy_cx_cy(pose["source_intrinsics"])
            metrics, rgb, object_mask, silhouette, uv, z = projection_metrics(
                static_vertices_world,
                np.asarray(prior_mesh.faces, dtype=np.int32),
                pose["T_world_camera_metric"],
                K,
                manifest[frame_idx],
                depth_sources,
                int(args.max_silhouette_faces),
            )
            observed_vertices_camera, observed_faces, observed_surface_row = make_camera_surface(
                vggt,
                frame_idx,
                scale_value,
                args,
            )
            surface_distance = observed_surface_distances(
                observed_vertices_camera,
                observed_faces,
                pose["T_world_camera_metric"],
                static_prior_world_samples,
                min(int(args.samples), int(args.surface_samples)),
                int(args.seed) + frame_idx,
            )
            metrics.update(surface_distance)
            metrics["source_intrinsics_fx_fy_cx_cy"] = [float(v) for v in pose["source_intrinsics"]]
            metrics["observed_surface"] = observed_surface_row
            frame_rows.append(metrics)
            write_review_still(
                scale_dir / "stills" / f"frame_{frame_idx:06d}.png",
                rgb,
                object_mask,
                silhouette,
                uv,
                np.asarray(prior_mesh.faces, dtype=np.int32),
                z,
                metrics,
                int(args.max_wire_faces),
                int(args.render_width),
            )
        ious = [row["silhouette_mask_iou"] for row in frame_rows]
        vertex_mask = [row["projected_vertex_mask_fraction"] for row in frame_rows]
        patch_median = [row["observed_to_static_prior_median_m"] for row in frame_rows]
        patch_p95 = [row["observed_to_static_prior_p95_m"] for row in frame_rows]
        depth_abs = [row["vertex_depth_error_abs_median_m"] for row in frame_rows if "vertex_depth_error_abs_median_m" in row]
        source_depth_abs = {}
        for label in depth_sources:
            key = depth_abs_key(label)
            values = [frame_row[key] for frame_row in frame_rows if key in frame_row]
            source_depth_abs[label] = float(np.median(values)) if values else None
        row = {
            "scale": scale_value,
            "anchor_frame": int(args.anchor_frame),
            "aligned_mesh_camera": str(scale_dir / "aligned_prior_anchor_camera.obj"),
            "alignment": align_report["selected"],
            "anchor_surface": anchor_surface_row,
            "aligned_camera_extent_m": (aligned_vertices_camera.max(axis=0) - aligned_vertices_camera.min(axis=0)).astype(float).tolist(),
            "aligned_camera_robust_extent_m": robust_extent(aligned_vertices_camera).astype(float).tolist(),
            "observed_anchor_camera_robust_extent_m": robust_extent(anchor_vertices_camera).astype(float).tolist(),
            "median_silhouette_mask_iou": float(np.median(ious)),
            "min_silhouette_mask_iou": float(np.min(ious)),
            "median_projected_vertex_mask_fraction": float(np.median(vertex_mask)),
            "median_observed_to_static_prior_median_m": float(np.median(patch_median)),
            "median_observed_to_static_prior_p95_m": float(np.median(patch_p95)),
            "median_vertex_depth_error_abs_median_m": float(np.median(depth_abs)) if depth_abs else None,
            "median_depth_source_error_abs_median_m": source_depth_abs,
            "frames": frame_rows,
        }
        for label, value in source_depth_abs.items():
            row[f"median_{depth_abs_key(label)}"] = value
        rows.append(row)

    best_by_iou = max(rows, key=lambda row: row["median_silhouette_mask_iou"])
    best_by_patch = min(rows, key=lambda row: row["median_observed_to_static_prior_median_m"])
    depth_rows = [row for row in rows if row["median_vertex_depth_error_abs_median_m"] is not None]
    best_by_manifest_depth = min(depth_rows, key=lambda row: row["median_vertex_depth_error_abs_median_m"]) if depth_rows else None
    best_by_depth_source = {}
    depth_source_summaries = {}
    for label in depth_sources:
        key = depth_abs_key(label)
        source_rows = [row for row in rows if row.get(f"median_{key}") is not None]
        depth_source_summaries[label] = summarize([row[f"median_{key}"] for row in source_rows])
        best_by_depth_source[label] = (
            {
                out_key: value
                for out_key, value in min(source_rows, key=lambda row: row[f"median_{key}"]).items()
                if out_key not in {"frames"}
            }
            if source_rows
            else None
        )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "sweep_vggt_scale_static_mesh_silhouette_v3",
        "claim_tested": "one complete TRELLIS mesh aligned to the anchor visible VGGT surface remains static in world coordinates and projects plausibly across the selected frames",
        "mesh_prior": str(args.mesh_prior),
        "vggt_archive": str(args.vggt_archive),
        "depth_sources": {
            "manifest": "Depth map referenced by manifest rows",
            **{label: str(path) for label, path in depth_archive_paths.items()},
        },
        "annotations": str(args.annotations),
        "manifest": str(args.manifest),
        "frames": frame_indices,
        "anchor_frame": int(args.anchor_frame),
        "scales": [float(v) for v in scales],
        "silhouette_mask_iou": summarize([row["median_silhouette_mask_iou"] for row in rows]),
        "projected_vertex_mask_fraction": summarize([row["median_projected_vertex_mask_fraction"] for row in rows]),
        "observed_to_static_prior_median_m": summarize([row["median_observed_to_static_prior_median_m"] for row in rows]),
        "observed_to_static_prior_p95_m": summarize([row["median_observed_to_static_prior_p95_m"] for row in rows]),
        "vertex_depth_error_abs_median_m": summarize(
            [row["median_vertex_depth_error_abs_median_m"] for row in rows if row["median_vertex_depth_error_abs_median_m"] is not None]
        ),
        "depth_source_error_abs_median_m": depth_source_summaries,
        "best_by_iou": {
            key: value
            for key, value in best_by_iou.items()
            if key not in {"frames"}
        },
        "best_by_patch": {
            key: value
            for key, value in best_by_patch.items()
            if key not in {"frames"}
        },
        "best_by_manifest_depth": (
            {
                key: value
                for key, value in best_by_manifest_depth.items()
                if key not in {"frames"}
            }
            if best_by_manifest_depth is not None
            else None
        ),
        "best_by_depth_source": best_by_depth_source,
        "best_by_depthpro_depth": best_by_depth_source.get("depthpro"),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows"}}, indent=2))
    return report


def camera_poses_for_scale(
    vggt: dict,
    annotations: dict[int, dict],
    scale: float,
    anchor_frame: int,
    frame_indices: list[int],
    args: argparse.Namespace,
) -> dict[int, dict]:
    anchor_i = vggt["index"].get(int(anchor_frame))
    if anchor_i is None:
        raise RuntimeError(f"VGGT archive lacks anchor frame {anchor_frame}")
    anchor_annotation = annotations.get(int(anchor_frame))
    if anchor_annotation is None:
        raise RuntimeError(f"annotations lack anchor frame {anchor_frame}")
    anchor_T = np.asarray(anchor_annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
    if anchor_T.shape != (4, 4) or not np.isfinite(anchor_T).all():
        raise RuntimeError(f"invalid anchor camera transform for frame {anchor_frame}")
    translation = anchor_T[:3, 3] - float(scale) * (vggt["sim3_rotation"] @ vggt["camera_centers_vggt"][anchor_i])
    out: dict[int, dict] = {}
    for frame_idx in frame_indices:
        i = vggt["index"].get(int(frame_idx))
        if i is None:
            raise RuntimeError(f"VGGT archive lacks frame {frame_idx}")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = vggt["sim3_rotation"] @ vggt["extrinsic"][i, :3, :3].T
        T[:3, 3] = float(scale) * (vggt["sim3_rotation"] @ vggt["camera_centers_vggt"][i]) + translation
        out[int(frame_idx)] = {
            "T_world_camera_metric": T,
            "source_intrinsics": source_intrinsics(
                vggt["intrinsic"][i],
                int(args.source_width),
                int(args.source_height),
                int(args.target_size),
            ),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--depthpro-archive", type=Path)
    parser.add_argument("--depth-archive", action="append", default=[], help="Additional metric depth archive as label=path")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--surface-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=881)
    parser.add_argument("--max-silhouette-faces", type=int, default=30000)
    parser.add_argument("--max-wire-faces", type=int, default=1600)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--grid-px", type=int, default=5)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
