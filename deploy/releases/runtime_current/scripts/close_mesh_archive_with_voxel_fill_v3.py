#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure
import trimesh

from render_bundlesdf_mesh_qc_v3 import load_json, load_mesh_archive


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def topology(mesh: trimesh.Trimesh) -> dict:
    if len(mesh.faces) == 0:
        raise RuntimeError("mesh has no faces")
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "euler_number": int(mesh.euler_number),
        "components": int(len(mesh.split(only_watertight=False))),
        "area_m2": float(mesh.area),
        "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
        "extent_m": np.asarray(mesh.extents, dtype=np.float64).astype(float).tolist(),
    }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def voxel_grid_to_mesh(vox: trimesh.voxel.VoxelGrid) -> trimesh.Trimesh:
    closed = vox.marching_cubes
    closed.vertices = transform_points(np.asarray(closed.vertices, dtype=np.float64), np.asarray(vox.transform, dtype=np.float64))
    return closed


def sdf_level_set_mesh(
    vox: trimesh.voxel.VoxelGrid,
    pitch_m: float,
    pad_voxels: int,
    smooth_sigma_voxels: float,
) -> trimesh.Trimesh:
    occ = np.asarray(vox.matrix, dtype=bool)
    pad = int(pad_voxels)
    if pad < 2:
        raise RuntimeError("SDF level-set extraction needs at least two pad voxels")
    occ_pad = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ_pad, sampling=[float(pitch_m)] * 3)
    inside = distance_transform_edt(occ_pad, sampling=[float(pitch_m)] * 3)
    sdf = outside - inside
    if float(smooth_sigma_voxels) > 0.0:
        sdf = gaussian_filter(sdf, sigma=float(smooth_sigma_voxels), mode="nearest")
    vertices, faces, normals, _ = measure.marching_cubes(
        volume=sdf.astype(np.float32),
        level=0.0,
        spacing=(1.0, 1.0, 1.0),
        allow_degenerate=False,
    )
    transform = np.asarray(vox.transform, dtype=np.float64).copy()
    transform[:3, 3] -= float(pitch_m) * pad
    mesh = trimesh.Trimesh(
        vertices=transform_points(np.asarray(vertices, dtype=np.float64), transform),
        faces=np.asarray(faces, dtype=np.int32),
        vertex_normals=np.asarray(normals, dtype=np.float64),
        process=False,
    )
    return mesh


def close_camera_mesh(
    mesh_camera: trimesh.Trimesh,
    pitch_m: float,
    frame_idx: int,
    max_discarded_component_area_fraction: float,
    surface_mode: str,
    sdf_pad_voxels: int,
    sdf_smooth_sigma_voxels: float,
) -> tuple[trimesh.Trimesh, dict]:
    if pitch_m <= 0.0:
        raise RuntimeError(f"invalid pitch {pitch_m}")
    vox = mesh_camera.voxelized(float(pitch_m)).fill()
    if int(np.count_nonzero(vox.matrix)) == 0:
        raise RuntimeError("voxel fill produced no occupied cells")
    if surface_mode == "occupancy":
        closed = voxel_grid_to_mesh(vox)
    elif surface_mode == "sdf":
        closed = sdf_level_set_mesh(vox, float(pitch_m), int(sdf_pad_voxels), float(sdf_smooth_sigma_voxels))
    else:
        raise RuntimeError(f"unknown surface mode {surface_mode}")
    closed = trimesh.Trimesh(vertices=np.asarray(closed.vertices), faces=np.asarray(closed.faces), process=True)
    closed.update_faces(closed.nondegenerate_faces())
    closed.update_faces(closed.unique_faces())
    closed.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(closed)
    raw_topology = topology(closed)
    component_report = []
    components = closed.split(only_watertight=False)
    if len(components) > 1:
        component_areas = np.asarray([float(component.area) for component in components], dtype=np.float64)
        keep = int(np.argmax(component_areas))
        total_area = float(component_areas.sum())
        discarded_area_fraction = float((total_area - component_areas[keep]) / total_area) if total_area > 0.0 else 1.0
        component_report = [
            {
                "component": int(i),
                "area_m2": float(component.area),
                "faces": int(len(component.faces)),
                "vertices": int(len(component.vertices)),
                "watertight": bool(component.is_watertight),
            }
            for i, component in enumerate(components)
        ]
        if discarded_area_fraction > float(max_discarded_component_area_fraction):
            raise RuntimeError(
                f"frame {frame_idx} voxel closure has non-negligible disconnected components: "
                f"discarded_area_fraction={discarded_area_fraction:.6f}, components={component_report}"
            )
        closed = trimesh.Trimesh(
            vertices=np.asarray(components[keep].vertices),
            faces=np.asarray(components[keep].faces),
            process=True,
        )
        trimesh.repair.fix_normals(closed)
    topo = topology(closed)
    topo["raw_before_component_filter"] = raw_topology
    topo["component_filter"] = {
        "max_discarded_component_area_fraction": float(max_discarded_component_area_fraction),
        "component_count_before_filter": int(len(components)),
        "components_before_filter": component_report,
    }
    topo["voxel_pitch_m"] = float(pitch_m)
    topo["surface_mode"] = str(surface_mode)
    topo["sdf_pad_voxels"] = int(sdf_pad_voxels) if surface_mode == "sdf" else None
    topo["sdf_smooth_sigma_voxels"] = float(sdf_smooth_sigma_voxels) if surface_mode == "sdf" else None
    topo["voxel_shape"] = [int(v) for v in vox.matrix.shape]
    topo["voxel_occupied"] = int(np.count_nonzero(vox.matrix))
    topo["voxel_transform"] = np.asarray(vox.transform, dtype=np.float64).astype(float).tolist()
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"frame {frame_idx} closed voxel mesh is not topologically closed: {topo}")
    return closed, topo


def annotations_by_frame(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations must contain frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def save_archive(path: Path, frame_ids: list[int], meshes_world: list[trimesh.Trimesh]) -> None:
    vertices = []
    faces = []
    vertex_offsets = [0]
    face_offsets = [0]
    vertex_cursor = 0
    for mesh in meshes_world:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int32)
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError("empty mesh during archive save")
        vertices.append(v)
        faces.append(f)
        vertex_cursor += int(len(v))
        vertex_offsets.append(vertex_cursor)
        face_offsets.append(face_offsets[-1] + int(len(f)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_ids, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
    )


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    annotations = annotations_by_frame(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted(meshes)
    if args.frame_start is not None:
        frame_ids = [frame for frame in frame_ids if frame >= int(args.frame_start)]
    if args.frame_end is not None:
        frame_ids = [frame for frame in frame_ids if frame <= int(args.frame_end)]
    if not frame_ids:
        raise RuntimeError("no mesh frames selected")
    rows = []
    closed_world = []
    for frame_idx in frame_ids:
        if frame_idx not in annotations:
            raise RuntimeError(f"annotations lack frame {frame_idx}")
        vertices_world, faces = meshes[frame_idx]
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        T_camera_world = np.linalg.inv(T_world_camera)
        vertices_camera = transform_points(np.asarray(vertices_world, dtype=np.float64), T_camera_world)
        source = trimesh.Trimesh(vertices=vertices_camera, faces=np.asarray(faces, dtype=np.int32), process=True)
        source_topology = topology(source)
        closed_camera, closed_topology = close_camera_mesh(
            source,
            float(args.pitch_m),
            int(frame_idx),
            float(args.max_discarded_component_area_fraction),
            str(args.surface_mode),
            int(args.sdf_pad_voxels),
            float(args.sdf_smooth_sigma_voxels),
        )
        closed_vertices_world = transform_points(np.asarray(closed_camera.vertices, dtype=np.float64), T_world_camera)
        closed_mesh_world = trimesh.Trimesh(
            vertices=closed_vertices_world,
            faces=np.asarray(closed_camera.faces, dtype=np.int32),
            process=False,
        )
        closed_world.append(closed_mesh_world)
        if int(frame_idx) in set(args.export_frames):
            closed_camera.export(args.output_dir / f"closed_voxel_frame_{frame_idx:06d}_camera.obj")
            closed_mesh_world.export(args.output_dir / f"closed_voxel_frame_{frame_idx:06d}_world.obj")
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "source_topology": source_topology,
                "closed_topology": topology(closed_mesh_world),
                "closed_camera_topology": closed_topology,
            }
        )
    archive_path = args.output_dir / "voxel_closed_meshes_world.npz"
    save_archive(archive_path, frame_ids, closed_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "close_mesh_archive_with_voxel_fill_v3",
        "claim_tested": "an observed fused mesh archive can be closed by exporting the same voxel fill used for SDF contact",
        "mesh_archive": str(args.mesh_archive),
        "annotations": str(args.annotations),
        "output_archive": str(archive_path),
        "frames": frame_ids,
        "pitch_m": float(args.pitch_m),
        "surface_mode": str(args.surface_mode),
        "sdf_pad_voxels": int(args.sdf_pad_voxels),
        "sdf_smooth_sigma_voxels": float(args.sdf_smooth_sigma_voxels),
        "max_discarded_component_area_fraction": float(args.max_discarded_component_area_fraction),
        "source_boundary_edges": summarize([row["source_topology"]["boundary_edges"] for row in rows]),
        "closed_boundary_edges": summarize([row["closed_topology"]["boundary_edges"] for row in rows]),
        "closed_nonmanifold_edges": summarize([row["closed_topology"]["nonmanifold_edges"] for row in rows]),
        "closed_vertices": summarize([row["closed_topology"]["vertices"] for row in rows]),
        "closed_faces": summarize([row["closed_topology"]["faces"] for row in rows]),
        "rows": rows,
    }
    save_json(args.output_dir / "qc_voxel_closed_mesh_archive_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--pitch-m", type=float, default=0.0015)
    parser.add_argument("--surface-mode", choices=["occupancy", "sdf"], default="occupancy")
    parser.add_argument("--sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--sdf-smooth-sigma-voxels", type=float, default=0.75)
    parser.add_argument("--max-discarded-component-area-fraction", type=float, default=0.002)
    parser.add_argument("--export-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
