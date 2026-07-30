#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_triangle_nonpenetration_evidence"

PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def load_final_frames(case: str, args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], str]:
    ann_path = args.full_pipeline_root / case / "annotations_v18_full.json"
    ann = load_json(ann_path)
    frames = {int(frame["frame_idx"]): frame for frame in ann.get("frames", []) if isinstance(frame, dict) and isinstance(frame.get("frame_idx"), int)}
    return frames, str(ann_path)


def strict_nonpenetration_eligibility(schema_row: dict[str, Any] | None) -> tuple[bool, str, list[str]]:
    if not isinstance(schema_row, dict):
        return False, "physical_schema_missing", ["missing_physical_state_schema_row"]
    physical = str(schema_row.get("model_physical_state_type") or "unknown")
    blockers: list[str] = []
    if physical != "rigid":
        blockers.append(f"physical_state_{physical}_not_strict_rigid")
    if schema_row.get("requires_part_or_relative_motion_model") is True:
        blockers.append("requires_part_or_relative_motion_model")
    if schema_row.get("secondary_deformable_or_surface_component") is True:
        blockers.append("secondary_deformable_or_surface_component")
    if schema_row.get("surface_change_without_pose_state") is True:
        blockers.append("surface_change_without_pose_model")
    return not blockers, "strict_rigid_nonpenetration_eligible" if not blockers else "strict_rigid_nonpenetration_not_eligible", blockers


def load_physical_schema_index(case: str, args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str | None]:
    report_path = args.physical_state_schema_root / case / "v18_physical_state_schema_report.json"
    if not report_path.exists():
        return {}, None
    report = load_json(report_path)
    out: dict[str, dict[str, Any]] = {}
    for row in report.get("object_rows", []) if isinstance(report, dict) else []:
        if isinstance(row, dict) and isinstance(row.get("object_id"), str):
            out[str(row["object_id"])] = row
    return out, str(report_path)


def load_depth_fused_mesh_index(case: str, args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str | None]:
    report_path = args.depth_fused_root / case / "v18_depth_fused_reconstruction_report.json"
    if not report_path.exists():
        return {}, None
    report = load_json(report_path)
    out: dict[str, dict[str, Any]] = {}
    for row in report.get("object_rows", []) if isinstance(report, dict) else []:
        if not isinstance(row, dict):
            continue
        object_id = str(row.get("object_id"))
        mesh = row.get("mesh_reconstruction") if isinstance(row.get("mesh_reconstruction"), dict) else {}
        poisson = mesh.get("poisson_mesh_path")
        hull = mesh.get("convex_hull_mesh_path")
        fused = mesh.get("fused_point_cloud_path")
        if isinstance(poisson, str) and poisson:
            out[object_id] = {
                "poisson_mesh_path": poisson,
                "convex_hull_mesh_path": hull,
                "fused_point_cloud_path": fused,
                "canonical_coordinate_source": row.get("canonical_coordinate_source"),
                "source_frame_count": row.get("source_frame_count"),
                "source_point_count": row.get("source_point_count"),
                "sampled_point_count": row.get("sampled_point_count"),
                "mesh_status": mesh.get("status"),
                "poisson_vertices": mesh.get("poisson_vertices"),
                "poisson_faces": mesh.get("poisson_faces"),
            }
    return out, str(report_path)


def _read_binary_scalar(data: bytes, offset: int, typ: str) -> tuple[float | int, int]:
    fmt = PLY_SCALAR_FORMATS[typ]
    size = struct.calcsize("<" + fmt)
    return struct.unpack_from("<" + fmt, data, offset)[0], offset + size


def load_ply_mesh(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not path.exists():
        return None, None, "ply_mesh_missing"
    data = path.read_bytes()
    marker = b"end_header\n"
    end = data.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = data.find(marker)
    if end < 0:
        return None, None, "ply_end_header_missing"
    header_bytes = data[: end + len(marker)]
    body = data[end + len(marker) :]
    lines = header_bytes.decode("ascii", errors="replace").splitlines()
    fmt = None
    vertex_count = 0
    face_count = 0
    section = None
    vertex_props: list[tuple[str, str]] = []
    face_list_types: tuple[str, str] | None = None
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            section = parts[1]
            if section == "vertex":
                vertex_count = int(parts[2])
            elif section == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and section == "vertex" and len(parts) >= 3 and parts[1] != "list":
            vertex_props.append((parts[2], parts[1]))
        elif parts[0] == "property" and section == "face" and len(parts) >= 5 and parts[1] == "list":
            face_list_types = (parts[2], parts[3])
    if fmt not in {"ascii", "binary_little_endian"}:
        return None, None, f"unsupported_ply_format_{fmt}"
    if vertex_count <= 0 or face_count <= 0 or len(vertex_props) < 3 or face_list_types is None:
        return None, None, "invalid_ply_header_counts_or_properties"
    prop_names = [name for name, _typ in vertex_props]
    try:
        x_i, y_i, z_i = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    except ValueError:
        return None, None, "ply_missing_xyz_properties"

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace").splitlines()
        if len(text) < vertex_count + face_count:
            return None, None, "ascii_ply_body_too_short"
        verts = []
        for i in range(vertex_count):
            vals = text[i].split()
            if len(vals) < len(vertex_props):
                return None, None, "ascii_ply_vertex_row_short"
            verts.append([float(vals[x_i]), float(vals[y_i]), float(vals[z_i])])
        faces: list[list[int]] = []
        for i in range(face_count):
            vals = text[vertex_count + i].split()
            if not vals:
                continue
            n = int(vals[0])
            ids = [int(v) for v in vals[1 : 1 + n]]
            if n == 3:
                faces.append(ids)
            elif n > 3:
                for j in range(1, n - 1):
                    faces.append([ids[0], ids[j], ids[j + 1]])
        vertices = np.asarray(verts, dtype=np.float64)
        face_arr = np.asarray(faces, dtype=np.int64)
    else:
        offset = 0
        verts = np.empty((vertex_count, 3), dtype=np.float64)
        for i in range(vertex_count):
            xyz = [0.0, 0.0, 0.0]
            for j, (_name, typ) in enumerate(vertex_props):
                value, offset = _read_binary_scalar(body, offset, typ)
                if j == x_i:
                    xyz[0] = float(value)
                elif j == y_i:
                    xyz[1] = float(value)
                elif j == z_i:
                    xyz[2] = float(value)
            verts[i] = xyz
        count_type, index_type = face_list_types
        faces = []
        for _ in range(face_count):
            n_raw, offset = _read_binary_scalar(body, offset, count_type)
            n = int(n_raw)
            ids = []
            for _j in range(n):
                idx_raw, offset = _read_binary_scalar(body, offset, index_type)
                ids.append(int(idx_raw))
            if n == 3:
                faces.append(ids)
            elif n > 3:
                for j in range(1, n - 1):
                    faces.append([ids[0], ids[j], ids[j + 1]])
        vertices = verts
        face_arr = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or face_arr.ndim != 2 or face_arr.shape[1] != 3:
        return None, None, "invalid_ply_mesh_arrays"
    if not np.isfinite(vertices).all() or len(vertices) == 0 or len(face_arr) == 0:
        return None, None, "nonfinite_or_empty_ply_mesh"
    valid = np.all((face_arr >= 0) & (face_arr < len(vertices)), axis=1)
    face_arr = face_arr[valid]
    if len(face_arr) == 0:
        return None, None, "ply_faces_out_of_range"
    return vertices, face_arr, None


def rotation_matrix_from_object_pose(obj: dict[str, Any]) -> np.ndarray | None:
    pose = obj.get("object_se3_observation") if isinstance(obj.get("object_se3_observation"), dict) else {}
    R = pose.get("rotation_world_from_object_matrix")
    if isinstance(R, list):
        arr = np.asarray(R, dtype=np.float64)
        if arr.shape == (3, 3) and np.isfinite(arr).all():
            return arr
    return None


def translation_from_object_pose(obj: dict[str, Any]) -> np.ndarray | None:
    pose = obj.get("object_se3_observation") if isinstance(obj.get("object_se3_observation"), dict) else {}
    t = pose.get("translation_world_m")
    if isinstance(t, list) and len(t) == 3:
        arr = np.asarray(t, dtype=np.float64)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr
    return None


def object_by_id(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and str(obj.get("object_id")) == object_id:
            return obj
    return None


def frame_depth_fused_mesh(
    frame: dict[str, Any] | None,
    object_id: str,
    mesh_index: dict[str, dict[str, Any]],
    mesh_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None, str | None]],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict[str, Any]]:
    if frame is None:
        return None, None, "missing_final_frame", {}
    obj = object_by_id(frame, object_id)
    if obj is None:
        return None, None, "missing_final_object_row", {}
    mesh_meta = mesh_index.get(object_id)
    if mesh_meta is None:
        return None, None, "missing_depth_fused_mesh_for_object", {}
    preferred_path = mesh_meta.get("convex_hull_mesh_path") or mesh_meta.get("poisson_mesh_path")
    mesh_path = Path(str(preferred_path))
    cache_key = str(mesh_path)
    if cache_key not in mesh_cache:
        mesh_cache[cache_key] = load_ply_mesh(mesh_path)
    canonical_vertices, faces, blocker = mesh_cache[cache_key]
    if blocker or canonical_vertices is None or faces is None:
        return None, None, blocker or "invalid_depth_fused_mesh", mesh_meta
    R = rotation_matrix_from_object_pose(obj)
    t = translation_from_object_pose(obj)
    if R is None or t is None:
        return None, None, "missing_final_object_se3_for_depth_fused_mesh", mesh_meta
    vertices_world = canonical_vertices @ R.T + t[None, :]
    return vertices_world, faces, None, mesh_meta


def final_hand_points(
    frame: dict[str, Any] | None,
    hand_side: str,
    args: argparse.Namespace,
    ref_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[np.ndarray | None, str | None, str | None, str | None]:
    if frame is None:
        return None, "missing_final_frame", None, None
    hand_row = None
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if isinstance(hand, dict) and str(hand.get("hand_side")) == hand_side:
            hand_row = hand
            break
    if hand_row is None:
        return None, "missing_final_hand_side", None, None
    support_state = str(hand_row.get("hawor_support_state") or "missing_hawor_support")
    if args.require_observed_hawor_support and support_state != "observed_same_frame_detection":
        return None, "hand_not_observed_hawor_support_for_nonpenetration_claim", None, support_state
    metric = hand_row.get("metric_mano_state") if isinstance(hand_row.get("metric_mano_state"), dict) else {}
    mano = hand_row.get("mano_candidate") if isinstance(hand_row.get("mano_candidate"), dict) else {}
    ref = mano.get("surface_reference") if isinstance(mano.get("surface_reference"), dict) else metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else None
    if not isinstance(ref, dict):
        sample = metric.get("vertices_world_sample_m")
        if isinstance(sample, list) and sample:
            pts = np.asarray(sample, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and np.isfinite(pts).all():
                return pts, None, "final_metric_mano_state_vertices_world_sample_m", support_state
        return None, "missing_final_hawor_surface_reference", None, support_state
    npz_raw = ref.get("bridge_npz") or ref.get("npz")
    arr_name = ref.get("bridge_vertices_world_array") or ref.get("array")
    row_idx = ref.get("bridge_row_index") if "bridge_row_index" in ref else ref.get("row_index")
    if not (isinstance(npz_raw, str) and isinstance(arr_name, str) and isinstance(row_idx, int)):
        return None, "invalid_final_hawor_surface_reference", None, support_state
    key = (npz_raw, arr_name, int(row_idx))
    if key not in ref_cache:
        z = np.load(Path(npz_raw), allow_pickle=True)
        if arr_name not in z.files or not (0 <= int(row_idx) < np.asarray(z[arr_name]).shape[0]):
            return None, "surface_reference_row_out_of_range", None, support_state
        ref_cache[key] = np.asarray(z[arr_name][int(row_idx)], dtype=np.float64)
    pts = ref_cache[key]
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        return None, "invalid_surface_reference_points", None, support_state
    if len(pts) > args.max_hand_points:
        step = max(1, int(math.ceil(len(pts) / args.max_hand_points)))
        pts = pts[::step]
    return pts, None, "HaWoR_metric_MANO_full_surface_reference_current_V18_world", support_state


def mesh_edge_diagnostics(faces: np.ndarray) -> dict[str, Any]:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return {
        "unique_edge_count": int(len(unique)),
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "mesh_watertight_by_edges": bool(boundary == 0 and nonmanifold == 0),
    }


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-10
    tri = tri[valid]
    centroids = centroids[valid]
    normals = normals[valid] / norms[valid, None]
    mesh_center = vertices.mean(axis=0)
    outward = np.sum(normals * (centroids - mesh_center), axis=1)
    normals[outward < 0.0] *= -1.0
    face_ids = np.nonzero(valid)[0]
    return tri, centroids, normals, face_ids


def closest_points_on_triangles(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    v = vb / denom_safe
    w = vc / denom_safe
    out = a + ab * v[:, None] + ac * w[:, None]
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    out[mask_a] = a[mask_a]
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    out[mask_b] = b[mask_b]
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    vab = d1 / np.where(np.abs(d1 - d3) > 1e-12, d1 - d3, 1.0)
    out[mask_ab] = a[mask_ab] + ab[mask_ab] * vab[mask_ab, None]
    mask_c = (d6 >= 0.0) & (d5 <= d6)
    out[mask_c] = c[mask_c]
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    wac = d2 / np.where(np.abs(d2 - d6) > 1e-12, d2 - d6, 1.0)
    out[mask_ac] = a[mask_ac] + ac[mask_ac] * wac[mask_ac, None]
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    wbc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-12, (d4 - d3) + (d5 - d6), 1.0)
    out[mask_bc] = b[mask_bc] + (c[mask_bc] - b[mask_bc]) * wbc[mask_bc, None]
    return out


def prepare_triangle_geometry(vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    tri, centroids, normals, face_ids = face_geometry(vertices, faces)
    if len(tri) == 0:
        return {"blocker": "invalid_triangle_normals"}
    diagnostics = mesh_edge_diagnostics(faces)
    k = min(args.nearest_triangle_candidates, len(tri))
    return {
        **diagnostics,
        "tri": tri,
        "centroids": centroids,
        "normals": normals,
        "face_ids": face_ids,
        "tree": cKDTree(centroids),
        "nearest_triangle_candidate_count": int(k),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
    }


def triangle_signed_stats_prepared(points: np.ndarray, prepared: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if prepared.get("blocker"):
        return {"blocker": prepared.get("blocker")}
    if len(points) > args.max_query_hand_points:
        step = max(1, int(math.ceil(len(points) / args.max_query_hand_points)))
        points = points[::step]
    tri = prepared["tri"]
    normals = prepared["normals"]
    face_ids = prepared["face_ids"]
    tree = prepared["tree"]
    k = int(prepared["nearest_triangle_candidate_count"])
    _, raw_indices = tree.query(points, k=k)
    indices = np.atleast_2d(raw_indices)
    if indices.shape[0] != len(points):
        indices = indices.T
    signed: list[float] = []
    unsigned: list[float] = []
    nearest_face_ids: list[int] = []
    for i, point in enumerate(points):
        cand = np.asarray(indices[i], dtype=np.int64)
        cand_tri = tri[cand]
        closest = closest_points_on_triangles(point, cand_tri)
        vec = point[None, :] - closest
        d2 = np.einsum("ij,ij->i", vec, vec)
        best_local = int(np.argmin(d2))
        best_idx = int(cand[best_local])
        normal = normals[best_idx]
        s = float(np.dot(vec[best_local], normal))
        signed.append(s)
        unsigned.append(float(math.sqrt(max(0.0, float(d2[best_local])))))
        nearest_face_ids.append(int(face_ids[best_idx]))
    signed_arr = np.asarray(signed, dtype=np.float64)
    unsigned_arr = np.asarray(unsigned, dtype=np.float64)
    negative = signed_arr < 0.0
    penetration = signed_arr < -args.penetration_tolerance_m
    return {
        "unique_edge_count": int(prepared["unique_edge_count"]),
        "boundary_edge_count": int(prepared["boundary_edge_count"]),
        "nonmanifold_edge_count": int(prepared["nonmanifold_edge_count"]),
        "mesh_watertight_by_edges": bool(prepared["mesh_watertight_by_edges"]),
        "sampled_hand_points": int(len(points)),
        "mesh_vertex_count": int(prepared["mesh_vertex_count"]),
        "mesh_face_count": int(prepared["mesh_face_count"]),
        "nearest_triangle_candidate_count": int(k),
        "min_triangle_unsigned_distance_m": float(np.min(unsigned_arr)),
        "median_triangle_unsigned_distance_m": float(np.median(unsigned_arr)),
        "min_local_triangle_signed_distance_m": float(np.min(signed_arr)),
        "median_local_triangle_signed_distance_m": float(np.median(signed_arr)),
        "negative_triangle_signed_distance_count": int(np.sum(negative)),
        "negative_triangle_signed_distance_fraction": float(np.mean(negative)),
        "local_triangle_penetration_detected": bool(np.any(penetration)),
        "local_triangle_signed_distance_semantics": "closest_point_on_depth_fused_completion_mesh_triangles_with_centroid_oriented_normals_not_ground_truth_sdf",
        "nearest_face_ids_sample": nearest_face_ids[: min(8, len(nearest_face_ids))],
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    contact_path = args.contact_ownership_root / case / "v18_contact_ownership_graph_report.json"
    contact = load_json(contact_path)
    final_frames, final_ann_path = load_final_frames(case, args)
    mesh_index, depth_report_path = load_depth_fused_mesh_index(case, args)
    physical_schema, physical_schema_path = load_physical_schema_index(case, args)
    mesh_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None, str | None]] = {}
    hand_cache: dict[tuple[int, str], tuple[np.ndarray | None, str | None, str | None, str | None]] = {}
    ref_cache: dict[tuple[str, str, int], np.ndarray] = {}
    prepared_mesh_cache: dict[tuple[int, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    blockers: Counter[str] = Counter()
    for raw in contact.get("rows", []):
        if not isinstance(raw, dict):
            continue
        frame_idx = raw.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        hand_side = str(raw.get("hand_side"))
        object_id = str(raw.get("object_id"))
        final_frame = final_frames.get(frame_idx)
        schema_row = physical_schema.get(object_id)
        strict_eligible, eligibility_state, eligibility_blockers = strict_nonpenetration_eligibility(schema_row)
        vertices, faces, mesh_blocker, mesh_meta = frame_depth_fused_mesh(final_frame, object_id, mesh_index, mesh_cache) if strict_eligible else (None, None, "object_not_strict_rigid_nonpenetration_eligible", mesh_index.get(object_id, {}))
        hand_key = (frame_idx, hand_side)
        if hand_key not in hand_cache:
            hand_cache[hand_key] = final_hand_points(final_frame, hand_side, args, ref_cache)
        points, hand_blocker, hand_source, hand_support_state = hand_cache[hand_key]
        row = {
            "frame_idx": frame_idx,
            "hand_side": hand_side,
            "object_id": object_id,
            "source_contact_owner_claim": raw.get("contact_owner_claim"),
            "source_accepted_contact_owner": bool(raw.get("accepted_contact_owner") is True),
            "source_min_unsigned_distance_m": raw.get("min_hand_surface_to_v16_object_mesh_m"),
            "source_contact_graph_v16_mesh_match": raw.get("v16_mesh_match"),
            "triangle_nonpenetration_claim": "not_evaluated",
            "triangle_nonpenetration_complete": False,
            "hand_support_state": hand_support_state,
            "require_observed_hawor_support": bool(args.require_observed_hawor_support),
            "object_mesh_backend": "depth_fused_convex_hull_visible_completion_candidate" if mesh_meta and mesh_meta.get("convex_hull_mesh_path") else "depth_fused_poisson_visible_completion_candidate" if mesh_meta else None,
            "object_mesh_path": (mesh_meta.get("convex_hull_mesh_path") or mesh_meta.get("poisson_mesh_path")) if mesh_meta else None,
            "object_mesh_status": mesh_meta.get("mesh_status") if mesh_meta else None,
            "object_mesh_canonical_coordinate_source": mesh_meta.get("canonical_coordinate_source") if mesh_meta else None,
            "object_physical_state_type": schema_row.get("model_physical_state_type") if isinstance(schema_row, dict) else None,
            "object_requires_part_or_relative_motion_model": bool(schema_row.get("requires_part_or_relative_motion_model")) if isinstance(schema_row, dict) else None,
            "object_secondary_deformable_or_surface_component": bool(schema_row.get("secondary_deformable_or_surface_component")) if isinstance(schema_row, dict) else None,
            "strict_nonpenetration_eligibility": eligibility_state,
            "strict_nonpenetration_eligibility_blockers": eligibility_blockers,
        }
        if mesh_blocker or hand_blocker or vertices is None or faces is None or points is None:
            blocker = mesh_blocker or hand_blocker or "missing_geometry"
            blockers[str(blocker)] += 1
            claim = "not_evaluated_object_not_strict_rigid_nonpenetration_eligible" if blocker == "object_not_strict_rigid_nonpenetration_eligible" else "blocked"
            row.update({"blocker": blocker, "triangle_nonpenetration_claim": claim})
            rows.append(row)
            continue
        prepared_key = (frame_idx, object_id)
        if prepared_key not in prepared_mesh_cache:
            prepared_mesh_cache[prepared_key] = prepare_triangle_geometry(vertices, faces, args)
        stats = triangle_signed_stats_prepared(points, prepared_mesh_cache[prepared_key], args)
        if stats.get("blocker"):
            blocker = str(stats["blocker"])
            blockers[blocker] += 1
            row.update({"blocker": blocker, "triangle_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        penetration = bool(stats.get("local_triangle_penetration_detected") is True)
        watertight = bool(stats.get("mesh_watertight_by_edges") is True)
        row.update(
            {
                **stats,
                "hand_geometry_source": hand_source,
                "penetration_tolerance_m": args.penetration_tolerance_m,
                "triangle_nonpenetration_claim": "depth_fused_mesh_triangle_penetration_evidence" if penetration else "depth_fused_mesh_triangle_no_penetration_beyond_tolerance_evidence",
                "triangle_nonpenetration_complete": False,
                "triangle_nonpenetration_scope": "depth_fused_visible_point_completion_mesh_against_support_gated_hawor_mano_vertices_not_complete_object_ground_truth_sdf",
                "watertight_candidate_mesh_available": watertight,
            }
        )
        rows.append(row)
    evaluated = sum(1 for r in rows if r.get("triangle_nonpenetration_claim") in {"depth_fused_mesh_triangle_penetration_evidence", "depth_fused_mesh_triangle_no_penetration_beyond_tolerance_evidence"})
    penetration_rows = sum(1 for r in rows if r.get("local_triangle_penetration_detected") is True)
    watertight_rows = sum(1 for r in rows if r.get("mesh_watertight_by_edges") is True)
    support_blocked_rows = sum(1 for r in rows if r.get("blocker") == "hand_not_observed_hawor_support_for_nonpenetration_claim")
    out = {
        "method": "build_v18_triangle_nonpenetration_evidence",
        "status": STATUS,
        "claim": "Computes closest-point-to-triangle signed evidence from support-gated HaWoR MANO hand vertices to depth-fused object completion mesh candidates, preferring watertight convex hulls when Poisson meshes are open. Watertight edge diagnostics are reported; because object meshes are depth-fused candidates from visible evidence, this is still not a complete ground-truth SDF proof.",
        "case": case,
        "sources": {
            "contact_ownership_graph": str(contact_path),
            "v18_full_annotations": final_ann_path,
            "depth_fused_reconstruction_report": depth_report_path,
            "physical_state_schema_report": physical_schema_path,
        },
        "source_contact_rows": len(contact.get("rows", [])) if isinstance(contact.get("rows"), list) else None,
        "accepted_contact_rows": int(contact.get("contact_ownership_accepted_rows", 0)),
        "triangle_rows": len(rows),
        "evaluated_triangle_rows": evaluated,
        "support_blocked_rows": support_blocked_rows,
        "local_triangle_penetration_detected_rows": penetration_rows,
        "mesh_watertight_rows": watertight_rows,
        "depth_fused_mesh_object_count": len(mesh_index),
        "blocker_counts": dict(sorted(blockers.items())),
        "parameters": {
            "max_hand_points": args.max_hand_points,
            "max_query_hand_points": args.max_query_hand_points,
            "nearest_triangle_candidates": args.nearest_triangle_candidates,
            "penetration_tolerance_m": args.penetration_tolerance_m,
            "require_observed_hawor_support": args.require_observed_hawor_support,
        },
        "rows": rows,
        "triangle_nonpenetration_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_triangle_nonpenetration_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_triangle_nonpenetration_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": r["case"],
                "triangle_rows": r["triangle_rows"],
                "evaluated_triangle_rows": r["evaluated_triangle_rows"],
                "support_blocked_rows": r["support_blocked_rows"],
                "local_triangle_penetration_detected_rows": r["local_triangle_penetration_detected_rows"],
                "mesh_watertight_rows": r["mesh_watertight_rows"],
                "triangle_nonpenetration_complete": r["triangle_nonpenetration_complete"],
            }
            for r in reports
        ],
        "claim_scope": "support_gated_hawor_to_depth_fused_completion_mesh_triangle_evidence_not_complete_sdf",
    }
    write_json(args.output_root / "v18_triangle_nonpenetration_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-ownership-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--depth-fused-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-query-hand-points", type=int, default=128)
    parser.add_argument("--nearest-triangle-candidates", type=int, default=32)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    parser.add_argument("--require-observed-hawor-support", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
