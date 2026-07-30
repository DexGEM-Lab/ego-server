#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"{path} did not load as a triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"{path} contains no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"{path} contains no triangular faces")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError(f"{path} face indices are invalid")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def edge_count_report(mesh: trimesh.Trimesh) -> dict:
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64).reshape(-1, 2), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
    }


def boundary_degree_report(mesh: trimesh.Trimesh) -> dict:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    edges = boundary_edges(faces)
    vertices, counts = np.unique(edges.reshape(-1), return_counts=True)
    hist = {int(degree): int(np.count_nonzero(counts == degree)) for degree in sorted(set(counts.tolist()))}
    irregular = vertices[counts != 2]
    return {
        "boundary_edges": int(len(edges)),
        "boundary_vertices": int(len(vertices)),
        "boundary_degree_histogram": hist,
        "irregular_boundary_vertices": irregular.astype(int).tolist(),
    }


def trim_irregular_boundary_vertices(mesh: trimesh.Trimesh, max_iterations: int = 8) -> tuple[trimesh.Trimesh, list[dict]]:
    reports = []
    current = mesh
    for iteration in range(int(max_iterations)):
        report = boundary_degree_report(current)
        report["iteration"] = int(iteration)
        reports.append(report)
        irregular = np.asarray(report["irregular_boundary_vertices"], dtype=np.int64)
        if len(irregular) == 0:
            return current, reports
        faces = np.asarray(current.faces, dtype=np.int32)
        remove = np.isin(faces, irregular).any(axis=1)
        removed = int(np.count_nonzero(remove))
        report["removed_faces"] = removed
        if removed == 0 or removed == len(faces):
            raise RuntimeError(f"cannot trim irregular boundary vertices: {report}")
        trimmed = trimesh.Trimesh(vertices=np.asarray(current.vertices), faces=faces[~remove], process=False)
        trimmed.remove_unreferenced_vertices()
        current = trimmed
    final_report = boundary_degree_report(current)
    final_report["iteration"] = int(max_iterations)
    reports.append(final_report)
    if final_report["irregular_boundary_vertices"]:
        raise RuntimeError(f"irregular boundary vertices remain after trimming: {final_report}")
    return current, reports


def clean_open_surface(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    before = edge_count_report(mesh)
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        raise RuntimeError("mesh has no connected components")
    main = max(components, key=lambda comp: int(len(comp.faces)))
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(main.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(main.faces, dtype=np.int32)),
    )
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh.remove_unreferenced_vertices()
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("open-surface cleanup removed all geometry")
    cleaned = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    after = edge_count_report(cleaned)
    if after["nonmanifold_edges"] != 0:
        raise RuntimeError(f"open-surface cleanup did not remove non-manifold edges: {after}")
    trimmed, trim_report = trim_irregular_boundary_vertices(cleaned)
    after_trim = edge_count_report(trimmed)
    if after_trim["nonmanifold_edges"] != 0:
        raise RuntimeError(f"boundary trimming introduced non-manifold edges: {after_trim}")
    return trimmed, {"before": before, "after_open3d": after, "boundary_trim": trim_report, "after": after_trim}


def annotations_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def pca_sheet_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.astype(np.float64)
    if np.linalg.det(axes) < 0.0:
        axes[-1] *= -1.0
    ratios = singular / max(float(singular[0]), 1e-12)
    return center, axes, ratios


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64)
    sorted_edges = np.sort(edges, axis=1)
    unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        raise RuntimeError("input mesh has no boundary; sheet solidification expects an open surface")
    if np.any(counts > 2):
        raise RuntimeError("input mesh has non-manifold edges; sheet solidification would hide topology errors")
    return boundary.astype(np.int32)


def camera_depth_axis(annotations: Path | None, anchor_frame: int | None) -> np.ndarray | None:
    if annotations is None or anchor_frame is None:
        return None
    frame_map = annotations_by_frame(annotations)
    if int(anchor_frame) not in frame_map:
        raise RuntimeError(f"annotations missing anchor frame {anchor_frame}")
    transform = np.asarray(frame_map[int(anchor_frame)]["camera"]["T_world_camera_metric"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"anchor frame {anchor_frame} has invalid camera transform")
    axis = transform[:3, 2]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("anchor camera depth axis is invalid")
    return axis / norm


def solidify_sheet(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    thickness_m: float,
) -> tuple[trimesh.Trimesh, dict]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    boundary = boundary_edges(faces)
    back_vertices = vertices + normal[None, :] * float(thickness_m)
    n = len(vertices)
    front_faces = faces.copy()
    back_faces = (faces[:, [2, 1, 0]] + n).astype(np.int32)
    side_faces = []
    for a, b in boundary:
        side_faces.append((int(a), int(b), int(b) + n))
        side_faces.append((int(a), int(b) + n, int(a) + n))
    faces_out = np.vstack((front_faces, back_faces, np.asarray(side_faces, dtype=np.int32)))
    vertices_out = np.vstack((vertices, back_vertices)).astype(np.float64)
    out = trimesh.Trimesh(vertices=vertices_out, faces=faces_out, process=False)
    out.fix_normals()
    edge_counts = np.unique(np.sort(out.edges.reshape(-1, 2), axis=1), axis=0, return_counts=True)[1]
    report = {
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "boundary_edges": int(len(boundary)),
        "output_vertices": int(len(out.vertices)),
        "output_faces": int(len(out.faces)),
        "nonmanifold_edges_after": int(np.count_nonzero(edge_counts > 2)),
        "boundary_edges_after": int(np.count_nonzero(edge_counts == 1)),
        "watertight": bool(out.is_watertight),
        "winding_consistent": bool(out.is_winding_consistent),
        "euler_number": int(out.euler_number),
    }
    if report["boundary_edges_after"] != 0 or report["nonmanifold_edges_after"] != 0:
        raise RuntimeError(f"solidified mesh topology is invalid: {report}")
    return out, report


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ homog.T).T[:, :3]


def save_pose_archive(path: Path, mesh: trimesh.Trimesh, poses_json: Path, frame_start: int, frame_end: int) -> dict:
    payload = load_json(poses_json)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    frame_indices = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame_idx in range(int(frame_start), int(frame_end) + 1):
        key = str(frame_idx)
        if key not in payload:
            continue
        row = payload[key]
        transform = np.asarray(row.get("T_frame_world_from_anchor"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"pose row {key} lacks finite T_frame_world_from_anchor")
        frame_vertices = transform_points(vertices, transform)
        frame_indices.append(frame_idx)
        vertices_all.append(frame_vertices.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(frame_vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    if not frame_indices:
        raise RuntimeError("no frame poses selected")
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )
    return {
        "archive": str(path),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
    }


def load_mesh_archive(path: Path) -> list[tuple[int, trimesh.Trimesh]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError(f"{path} has invalid offset lengths")
    out = []
    for i, idx in enumerate(frame_idx.tolist()):
        v = vertices[vertex_offsets[i] : vertex_offsets[i + 1]]
        f = faces[face_offsets[i] : face_offsets[i + 1]]
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError(f"{path} frame {idx} is empty")
        out.append((int(idx), trimesh.Trimesh(vertices=v, faces=f, process=False)))
    return out


def save_mesh_archive(path: Path, frame_meshes: list[tuple[int, trimesh.Trimesh]]) -> dict:
    frame_indices = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame_idx, mesh in frame_meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        frame_indices.append(int(frame_idx))
        vertices_all.append(vertices)
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    if not frame_indices:
        raise RuntimeError("no frame meshes selected")
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )
    return {
        "archive": str(path),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
    }


def sheet_thickness(vertices: np.ndarray, normal: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    normal_coordinates = (vertices - np.median(vertices, axis=0)[None, :]) @ normal
    normal_extent = float(np.quantile(normal_coordinates, 0.95) - np.quantile(normal_coordinates, 0.05))
    thickness = float(args.thickness_m) if args.thickness_m is not None else float(
        np.clip(normal_extent * float(args.thickness_scale), float(args.min_thickness_m), float(args.max_thickness_m))
    )
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise RuntimeError(f"invalid closure thickness {thickness}")
    return thickness, normal_extent


def solidify_mesh(mesh: trimesh.Trimesh, args: argparse.Namespace, depth_axis: np.ndarray | None = None) -> tuple[trimesh.Trimesh, dict]:
    cleaned, cleanup_report = clean_open_surface(mesh)
    vertices = np.asarray(cleaned.vertices, dtype=np.float64)
    _center, axes, ratios = pca_sheet_frame(vertices)
    if float(ratios[2]) > float(args.max_sheet_planarity_ratio):
        raise RuntimeError(
            f"mesh is not sheet-like enough for measured-surface solidification: "
            f"smallest PCA ratio {float(ratios[2]):.6f} > {float(args.max_sheet_planarity_ratio):.6f}"
        )
    normal = axes[2].copy()
    if depth_axis is not None and float(np.dot(normal, depth_axis)) < 0.0:
        normal *= -1.0
    thickness, normal_extent = sheet_thickness(vertices, normal, args)
    solid, topology = solidify_sheet(cleaned, normal, thickness)
    ext = np.asarray(solid.vertices, dtype=np.float64).max(axis=0) - np.asarray(solid.vertices, dtype=np.float64).min(axis=0)
    report = {
        "pca_ratios": ratios.astype(float).tolist(),
        "normal": normal.astype(float).tolist(),
        "normal_extent_5_95_m": normal_extent,
        "thickness_m": thickness,
        "extent_m": ext.astype(float).tolist(),
        "topology": topology,
        "input_cleanup": cleanup_report,
    }
    return solid, report


def run_archive(args: argparse.Namespace) -> dict:
    frame_meshes = load_mesh_archive(args.input_archive)
    annotations = annotations_by_frame(args.annotations) if args.annotations is not None else {}
    solids = []
    rows = []
    for frame_idx, mesh in frame_meshes:
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        depth_axis = None
        if args.annotations is not None:
            if frame_idx not in annotations:
                raise RuntimeError(f"annotations missing frame {frame_idx}")
            transform = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise RuntimeError(f"frame {frame_idx} has invalid camera transform")
            axis = transform[:3, 2]
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(f"frame {frame_idx} has invalid camera depth axis")
            depth_axis = axis / norm
        solid, row = solidify_mesh(mesh, args, depth_axis)
        row["frame_idx"] = int(frame_idx)
        rows.append(row)
        solids.append((int(frame_idx), solid))
    if len(solids) < int(args.min_frames):
        raise RuntimeError(f"only {len(solids)} solidified frames selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_report = save_mesh_archive(args.output_dir / "solidified_sheet_object_meshes_world.npz", solids)
    thicknesses = np.asarray([row["thickness_m"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "solidify_sheet_mesh_archive_v3",
        "claim_tested": "per-frame measured open surfaces that are geometrically sheet-like can be closed by data-derived small-thickness shells without category-specific primitives",
        "input_archive": str(args.input_archive),
        "annotations": str(args.annotations) if args.annotations is not None else None,
        "mesh_archive": archive_report["archive"],
        "frames": archive_report["frames"],
        "first_frame": archive_report["first_frame"],
        "last_frame": archive_report["last_frame"],
        "thickness_median_m": float(np.median(thicknesses)),
        "thickness_p05_m": float(np.percentile(thicknesses, 5.0)),
        "thickness_p95_m": float(np.percentile(thicknesses, 95.0)),
        "rows": rows,
        "parameters": {
            "max_sheet_planarity_ratio": float(args.max_sheet_planarity_ratio),
            "thickness_m": float(args.thickness_m) if args.thickness_m is not None else None,
            "thickness_scale": float(args.thickness_scale),
            "min_thickness_m": float(args.min_thickness_m),
            "max_thickness_m": float(args.max_thickness_m),
        },
    }
    (args.output_dir / "qc_solidify_sheet_mesh_archive_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def run(args: argparse.Namespace) -> dict:
    if args.input_archive is not None:
        return run_archive(args)
    if args.canonical_mesh is None or args.poses_json is None:
        raise RuntimeError("--canonical-mesh and --poses-json are required without --input-archive")
    raw_mesh = load_mesh(args.canonical_mesh)
    solid, solid_report = solidify_mesh(raw_mesh, args, camera_depth_axis(args.annotations, args.anchor_frame))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = args.output_dir / "solidified_sheet_canonical_mesh.obj"
    solid.export(canonical_path)
    archive_report = save_pose_archive(
        args.output_dir / "solidified_sheet_object_meshes_world.npz",
        solid,
        args.poses_json,
        int(args.frame_start),
        int(args.frame_end),
    )
    ext = np.asarray(solid.vertices, dtype=np.float64).max(axis=0) - np.asarray(solid.vertices, dtype=np.float64).min(axis=0)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "solidify_sheet_mesh_archive_v3",
        "claim_tested": "a measured open surface that is geometrically sheet-like can be closed by a data-derived small-thickness shell without category-specific primitives",
        "canonical_mesh": str(args.canonical_mesh),
        "poses_json": str(args.poses_json),
        "canonical_solid_mesh": str(canonical_path),
        "mesh_archive": archive_report["archive"],
        "frames": archive_report["frames"],
        "first_frame": archive_report["first_frame"],
        "last_frame": archive_report["last_frame"],
        "pca_ratios": solid_report["pca_ratios"],
        "normal": solid_report["normal"],
        "normal_extent_5_95_m": solid_report["normal_extent_5_95_m"],
        "thickness_m": solid_report["thickness_m"],
        "extent_m": ext.astype(float).tolist(),
        "topology": solid_report["topology"],
        "input_cleanup": solid_report["input_cleanup"],
        "parameters": {
            "max_sheet_planarity_ratio": float(args.max_sheet_planarity_ratio),
            "thickness_m": float(args.thickness_m) if args.thickness_m is not None else None,
            "thickness_scale": float(args.thickness_scale),
            "min_thickness_m": float(args.min_thickness_m),
            "max_thickness_m": float(args.max_thickness_m),
        },
    }
    (args.output_dir / "qc_solidify_sheet_mesh_archive_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-mesh", type=Path)
    parser.add_argument("--poses-json", type=Path)
    parser.add_argument("--input-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-sheet-planarity-ratio", type=float, default=0.12)
    parser.add_argument("--thickness-m", type=float)
    parser.add_argument("--thickness-scale", type=float, default=0.50)
    parser.add_argument("--min-thickness-m", type=float, default=0.008)
    parser.add_argument("--max-thickness-m", type=float, default=0.035)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
