#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from archive_aligned_mesh_prior_v7 import load_triangle_mesh, write_mesh_archive
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize
from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json
from render_mesh_zbuffer_qc_v3 import mesh_zbuffer


def rows_by_frame(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain nonempty rows")
    out: dict[int, dict] = {}
    for row in rows:
        idx = int(row["frame_idx"])
        if idx in out:
            raise RuntimeError(f"{path} has duplicate frame {idx}")
        sim3 = row.get("sim3")
        if not isinstance(sim3, dict):
            raise RuntimeError(f"{path} row {idx} lacks sim3")
        out[idx] = row
    return out


def manifest_by_frame(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    return {int(frame["frame_idx"]): frame for frame in frames}


def annotations_by_frame(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    return {int(frame["frame_idx"]): frame for frame in frames}


def aligned_from_prior(points_prior: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rotation = np.asarray(sim3["rotation"], dtype=np.float64)
    translation = np.asarray(sim3["translation_m"], dtype=np.float64)
    if scale <= 0.0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("invalid Sim3 row")
    points = scale * (np.asarray(points_prior, dtype=np.float64) @ rotation.T) + translation[None, :]
    if not np.isfinite(points).all():
        raise RuntimeError("Sim3 produced non-finite points")
    return points


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("cannot sample empty mesh")
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not bool(valid.any()):
        raise RuntimeError("mesh has no positive-area faces")
    tri = tri[valid]
    areas = areas[valid]
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(tri), size=int(count), replace=True, p=areas / areas.sum())
    chosen = tri[ids]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])


def connected_components(face_count: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(int(face_count), dtype=np.int64)
    rank = np.zeros(int(face_count), dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in np.asarray(edges, dtype=np.int64):
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(int(face_count))], dtype=np.int64)
    unique = {root: i for i, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([unique[int(root)] for root in roots], dtype=np.int64)


def face_adjacency_labels(faces: np.ndarray) -> np.ndarray:
    edge_owner: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        verts = [int(face[0]), int(face[1]), int(face[2])]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            key = (a, b) if a < b else (b, a)
            prev = edge_owner.get(key)
            if prev is None:
                edge_owner[key] = int(face_id)
            else:
                edges.append((prev, int(face_id)))
    if not edges:
        return np.arange(len(faces), dtype=np.int64)
    return connected_components(len(faces), np.asarray(edges, dtype=np.int64))


def component_keep_mask(faces: np.ndarray, face_keep: np.ndarray, min_faces: int) -> tuple[np.ndarray, list[dict]]:
    labels = face_adjacency_labels(faces)
    keep = np.zeros(len(faces), dtype=bool)
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == int(label))
        kept = idx[face_keep[idx]]
        accept = len(kept) >= int(min_faces)
        if accept:
            keep[kept] = True
        rows.append(
            {
                "component": int(label),
                "faces_total": int(len(idx)),
                "faces_after_filter": int(len(kept)),
                "kept": bool(accept),
            }
        )
    return keep, rows


def face_id_zbuffer(shape: tuple[int, int], uv: np.ndarray, z: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    facebuf = np.full((height, width), -1, dtype=np.int32)
    faces = np.asarray(faces, dtype=np.int64)
    valid_face = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid_face)
    order = np.argsort(z[faces[face_ids]].min(axis=1))[::-1]
    for local_face_id in face_ids[order]:
        poly_f = uv[faces[int(local_face_id)]]
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
        tri_z = z[faces[int(local_face_id)]].astype(np.float64)
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
        if not bool(update.any()):
            continue
        region[update] = face_depth[update].astype(np.float32)
        facebuf[y0:y1, x0:x1][update] = int(local_face_id)
    return zbuf, facebuf


def append_faces(observed_vertices: np.ndarray, observed_faces: np.ndarray, prior_vertices: np.ndarray, prior_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(prior_faces) == 0:
        return np.asarray(observed_vertices, dtype=np.float64), np.asarray(observed_faces, dtype=np.int32)
    prior_faces = np.asarray(prior_faces, dtype=np.int32)
    used_vertices, compact_faces = np.unique(prior_faces.reshape(-1), return_inverse=True)
    compact_faces = compact_faces.reshape(prior_faces.shape).astype(np.int32)
    compact_prior_vertices = np.asarray(prior_vertices, dtype=np.float64)[used_vertices]
    vertices = np.vstack([np.asarray(observed_vertices, dtype=np.float64), compact_prior_vertices])
    faces = np.vstack([np.asarray(observed_faces, dtype=np.int32), compact_faces + len(observed_vertices)])
    return vertices, faces


def simplify_prior_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray, dict]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if int(max_faces) <= 0 or len(faces) <= int(max_faces):
        return vertices, faces, {
            "simplified": False,
            "input_vertices": int(len(vertices)),
            "input_faces": int(len(faces)),
            "output_vertices": int(len(vertices)),
            "output_faces": int(len(faces)),
        }
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    out_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(out_vertices) == 0 or len(out_faces) == 0:
        raise RuntimeError("mesh simplification produced empty prior")
    return out_vertices, out_faces, {
        "simplified": True,
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "output_vertices": int(len(out_vertices)),
        "output_faces": int(len(out_faces)),
    }


def projection_filter(
    candidate_vertices: np.ndarray,
    prior_faces: np.ndarray,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    manifest_row: dict,
    annotation_row: dict,
    depth_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    rgb = cv2.imread(str(Path(manifest_row["rgb"])), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(Path(manifest_row["mask"])), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        raise RuntimeError(f"failed to read RGB or mask for frame {manifest_row.get('frame_idx')}")
    object_mask = mask > 0
    height, width = object_mask.shape
    T_world_camera = np.asarray(annotation_row["camera"]["T_world_camera_metric"], dtype=np.float64)
    K = intrinsics_for_frame(args, manifest_row, annotation_row)

    observed_camera = camera_points(observed_vertices, T_world_camera)
    observed_positive = observed_camera[:, 2] > 0.0
    observed_uv = np.full((len(observed_vertices), 2), np.nan, dtype=np.float64)
    observed_uv[observed_positive, 0] = K[0, 0] * observed_camera[observed_positive, 0] / observed_camera[observed_positive, 2] + K[0, 2]
    observed_uv[observed_positive, 1] = K[1, 1] * observed_camera[observed_positive, 1] / observed_camera[observed_positive, 2] + K[1, 2]
    observed_zbuf = mesh_zbuffer(
        object_mask.shape,
        observed_uv,
        observed_camera[:, 2],
        np.asarray(observed_faces, dtype=np.int32),
        max_faces=None,
        vertex_radius_px=0,
    )

    prior_camera = camera_points(candidate_vertices, T_world_camera)
    z = prior_camera[:, 2]
    uv = np.full((len(candidate_vertices), 2), np.nan, dtype=np.float64)
    positive = z > 0.0
    uv[positive, 0] = K[0, 0] * prior_camera[positive, 0] / z[positive] + K[0, 2]
    uv[positive, 1] = K[1, 1] * prior_camera[positive, 1] / z[positive] + K[1, 2]

    centroids = candidate_vertices[prior_faces].mean(axis=1)
    centroid_camera = camera_points(centroids, T_world_camera)
    centroid_z = centroid_camera[:, 2]
    centroid_uv = np.full((len(centroids), 2), np.nan, dtype=np.float64)
    cpos = centroid_z > 0.0
    centroid_uv[cpos, 0] = K[0, 0] * centroid_camera[cpos, 0] / centroid_z[cpos] + K[0, 2]
    centroid_uv[cpos, 1] = K[1, 1] * centroid_camera[cpos, 1] / centroid_z[cpos] + K[1, 2]

    face_uv = uv[prior_faces]
    projected = np.isfinite(face_uv).all(axis=(1, 2)) & np.isfinite(centroid_uv).all(axis=1) & np.isfinite(centroid_z) & (centroid_z > 0.0)
    vertex_xy = np.rint(uv).astype(np.int64)
    vertex_inside_image = (
        np.isfinite(uv).all(axis=1)
        & (z > 0.0)
        & (vertex_xy[:, 0] >= 0)
        & (vertex_xy[:, 0] < width)
        & (vertex_xy[:, 1] >= 0)
        & (vertex_xy[:, 1] < height)
    )
    vertex_inside_mask = np.zeros(len(candidate_vertices), dtype=bool)
    vertex_ids = np.flatnonzero(vertex_inside_image)
    vertex_inside_mask[vertex_ids] = object_mask[vertex_xy[vertex_ids, 1], vertex_xy[vertex_ids, 0]]
    vertex_valid_depth = np.zeros(len(candidate_vertices), dtype=bool)
    vertex_depth_delta = np.full(len(candidate_vertices), np.nan, dtype=np.float64)
    vdepth_ids = np.flatnonzero(vertex_inside_mask)
    if len(vdepth_ids):
        vd = depth_m[vertex_xy[vdepth_ids, 1], vertex_xy[vdepth_ids, 0]].astype(np.float64)
        vv = np.isfinite(vd) & (vd > 0.0)
        vertex_valid_depth[vdepth_ids[vv]] = True
        vertex_depth_delta[vdepth_ids[vv]] = z[vdepth_ids[vv]] - vd[vv]

    centroid_xy = np.rint(centroid_uv).astype(np.int64)
    inside_image = (
        projected
        & (centroid_xy[:, 0] >= 0)
        & (centroid_xy[:, 0] < width)
        & (centroid_xy[:, 1] >= 0)
        & (centroid_xy[:, 1] < height)
    )
    inside_mask = np.zeros(len(prior_faces), dtype=bool)
    valid_ids = np.flatnonzero(inside_image)
    inside_mask[valid_ids] = object_mask[centroid_xy[valid_ids, 1], centroid_xy[valid_ids, 0]]

    valid_depth = np.zeros(len(prior_faces), dtype=bool)
    depth_delta = np.full(len(prior_faces), np.nan, dtype=np.float64)
    depth_ids = np.flatnonzero(inside_mask)
    if len(depth_ids):
        d = depth_m[centroid_xy[depth_ids, 1], centroid_xy[depth_ids, 0]].astype(np.float64)
        valid = np.isfinite(d) & (d > 0.0)
        valid_depth[depth_ids[valid]] = True
        depth_delta[depth_ids[valid]] = centroid_z[depth_ids[valid]] - d[valid]
    observed_zbuf_valid = np.zeros(len(prior_faces), dtype=bool)
    observed_zbuf_delta = np.full(len(prior_faces), np.nan, dtype=np.float64)
    if len(valid_ids):
        oz = observed_zbuf[centroid_xy[valid_ids, 1], centroid_xy[valid_ids, 0]].astype(np.float64)
        ov = np.isfinite(oz) & (oz > 0.0)
        observed_zbuf_valid[valid_ids[ov]] = True
        observed_zbuf_delta[valid_ids[ov]] = centroid_z[valid_ids[ov]] - oz[ov]

    observed_tree = cKDTree(observed_vertices)
    observed_dist = observed_tree.query(centroids, workers=-1)[0]
    hidden_enough = observed_dist >= float(args.min_hidden_distance_m)
    face_vertex_inside_mask = vertex_inside_mask[prior_faces]
    face_vertex_valid_depth = vertex_valid_depth[prior_faces]
    face_vertex_depth_delta = vertex_depth_delta[prior_faces]
    face_vertex_zbuf_valid = np.zeros(face_vertex_valid_depth.shape, dtype=bool)
    face_vertex_zbuf_delta = np.full(face_vertex_depth_delta.shape, np.nan, dtype=np.float64)
    for corner in range(3):
        vids = prior_faces[:, corner]
        vxy = vertex_xy[vids]
        vimage = vertex_inside_image[vids]
        if not bool(vimage.any()):
            continue
        oz = observed_zbuf[vxy[vimage, 1], vxy[vimage, 0]].astype(np.float64)
        ov = np.isfinite(oz) & (oz > 0.0)
        local = np.flatnonzero(vimage)
        valid_local = local[ov]
        face_vertex_zbuf_valid[valid_local, corner] = True
        face_vertex_zbuf_delta[valid_local, corner] = z[vids[valid_local]] - oz[ov]
    vertex_depth_violation = face_vertex_inside_mask & face_vertex_valid_depth & (np.abs(face_vertex_depth_delta) > float(args.max_visible_depth_abs_m))
    vertex_front_violation = face_vertex_inside_mask & face_vertex_valid_depth & (face_vertex_depth_delta < -float(args.max_front_free_space_m))
    any_vertex_depth_violation = np.any(vertex_depth_violation, axis=1)
    any_vertex_front_violation = np.any(vertex_front_violation, axis=1)
    zbuf_front_violation = observed_zbuf_valid & (observed_zbuf_delta < float(args.min_hidden_behind_observed_m))
    vertex_zbuf_front_violation = face_vertex_zbuf_valid & (face_vertex_zbuf_delta < float(args.min_hidden_behind_observed_m))
    any_vertex_zbuf_front_violation = np.any(vertex_zbuf_front_violation, axis=1)
    all_vertices_not_visible_mask = ~np.any(face_vertex_inside_mask, axis=1)
    violates_known_visible = inside_mask & valid_depth & (np.abs(depth_delta) > float(args.max_visible_depth_abs_m))
    in_free_space = inside_mask & valid_depth & (depth_delta < -float(args.max_front_free_space_m))
    face_keep = (
        hidden_enough
        & ~violates_known_visible
        & ~in_free_space
        & ~any_vertex_depth_violation
        & ~any_vertex_front_violation
        & ~zbuf_front_violation
        & ~any_vertex_zbuf_front_violation
    )
    if not bool(args.keep_inside_image_hidden_faces):
        face_keep &= (~inside_image | ~inside_mask) & all_vertices_not_visible_mask
    pre_raster_keep, pre_raster_components = component_keep_mask(prior_faces, face_keep, int(args.min_component_faces))
    active = pre_raster_keep.copy()
    raster_rows = []
    observed_visible = np.isfinite(observed_zbuf)
    for iteration in range(int(args.max_raster_filter_iters)):
        active_ids = np.flatnonzero(active)
        if len(active_ids) == 0:
            raster_rows.append(
                {
                    "iteration": int(iteration),
                    "active_faces": 0,
                    "visible_pixels": 0,
                    "conflict_pixels": 0,
                    "conflict_faces": 0,
                }
            )
            break
        active_faces = prior_faces[active_ids]
        prior_zbuf, prior_facebuf = face_id_zbuffer(object_mask.shape, uv, z, active_faces)
        prior_visible = np.isfinite(prior_zbuf)
        visible_outside_mask = prior_visible & ~object_mask
        visible_on_observed = prior_visible & object_mask & observed_visible
        visible_without_observed = prior_visible & object_mask & ~observed_visible
        conflict_pixels = visible_outside_mask | (visible_on_observed & (prior_zbuf < observed_zbuf + float(args.min_hidden_behind_observed_m)))
        if not bool(args.allow_visible_mask_fill):
            conflict_pixels |= visible_without_observed
        else:
            valid_fill_depth = np.isfinite(depth_m) & (depth_m > 0.0)
            fill_depth_delta = prior_zbuf.astype(np.float64) - depth_m.astype(np.float64)
            conflict_pixels |= visible_without_observed & (~valid_fill_depth | (np.abs(fill_depth_delta) > float(args.max_visible_depth_abs_m)))
        conflict_face_ids = np.unique(prior_facebuf[conflict_pixels])
        conflict_face_ids = conflict_face_ids[conflict_face_ids >= 0]
        conflict_global_ids = active_ids[conflict_face_ids.astype(np.int64)]
        raster_rows.append(
            {
                "iteration": int(iteration),
                "active_faces": int(len(active_ids)),
                "visible_pixels": int(np.count_nonzero(prior_visible)),
                "conflict_pixels": int(np.count_nonzero(conflict_pixels)),
                "conflict_faces": int(len(conflict_global_ids)),
            }
        )
        if len(conflict_global_ids) == 0:
            break
        active[conflict_global_ids] = False
    post_raster_face_keep = active
    raster_visible_pixels = raster_rows[-1]["visible_pixels"] if raster_rows else 0
    raster_conflict_pixels = raster_rows[-1]["conflict_pixels"] if raster_rows else 0
    raster_conflict_faces = int(sum(row["conflict_faces"] for row in raster_rows))
    component_keep, components = component_keep_mask(prior_faces, post_raster_face_keep, int(args.min_component_faces))
    row = {
        "candidate_prior_faces": int(len(prior_faces)),
        "inside_image_faces": int(np.count_nonzero(inside_image)),
        "inside_mask_faces": int(np.count_nonzero(inside_mask)),
        "any_vertex_inside_mask_faces": int(np.count_nonzero(np.any(face_vertex_inside_mask, axis=1))),
        "valid_depth_faces": int(np.count_nonzero(valid_depth)),
        "hidden_enough_faces": int(np.count_nonzero(hidden_enough)),
        "visible_depth_violation_faces": int(np.count_nonzero(violates_known_visible)),
        "front_free_space_violation_faces": int(np.count_nonzero(in_free_space)),
        "vertex_depth_violation_faces": int(np.count_nonzero(any_vertex_depth_violation)),
        "vertex_front_free_space_violation_faces": int(np.count_nonzero(any_vertex_front_violation)),
        "observed_zbuffer_valid_faces": int(np.count_nonzero(observed_zbuf_valid)),
        "observed_zbuffer_front_violation_faces": int(np.count_nonzero(zbuf_front_violation)),
        "vertex_observed_zbuffer_front_violation_faces": int(np.count_nonzero(any_vertex_zbuf_front_violation)),
        "pre_raster_kept_prior_faces": int(np.count_nonzero(pre_raster_keep)),
        "raster_prior_visible_pixels": raster_visible_pixels,
        "raster_conflict_pixels": raster_conflict_pixels,
        "raster_conflict_faces": raster_conflict_faces,
        "raster_filter_iterations": raster_rows,
        "kept_prior_faces": int(np.count_nonzero(component_keep)),
        "observed_distance_m": summarize(observed_dist.astype(np.float64)),
        "visible_depth_delta_m": summarize(depth_delta[np.isfinite(depth_delta)].astype(np.float64)),
        "visible_vertex_depth_delta_m": summarize(vertex_depth_delta[np.isfinite(vertex_depth_delta)].astype(np.float64)),
        "observed_zbuffer_delta_m": summarize(observed_zbuf_delta[np.isfinite(observed_zbuf_delta)].astype(np.float64)),
        "vertex_observed_zbuffer_delta_m": summarize(face_vertex_zbuf_delta[np.isfinite(face_vertex_zbuf_delta)].astype(np.float64)),
        "pre_raster_components": pre_raster_components,
        "components": components,
    }
    return component_keep, row


def run(args: argparse.Namespace) -> dict:
    prior = load_triangle_mesh(args.mesh_prior)
    observed = load_mesh_archive(args.observed_mesh_archive)
    align_rows = rows_by_frame(args.alignment_report)
    manifest = manifest_by_frame(args.manifest)
    annotations = annotations_by_frame(args.annotations)
    depths = load_depth_archive(args.metric_depth_npz)
    prior_vertices_raw = np.asarray(prior.vertices, dtype=np.float64)
    prior_faces_raw = np.asarray(prior.faces, dtype=np.int32)
    prior_vertices, prior_faces, simplification_report = simplify_prior_mesh(
        prior_vertices_raw,
        prior_faces_raw,
        int(args.max_prior_faces),
    )
    frames = [
        idx
        for idx in range(int(args.frame_start), int(args.frame_end) + 1)
        if idx in observed and idx in align_rows and idx in manifest and idx in annotations and idx in depths
    ]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames have all required inputs")

    rows = []
    archive_rows = []
    for frame_idx in frames:
        observed_vertices, observed_faces = observed[frame_idx]
        candidate_vertices = aligned_from_prior(prior_vertices, align_rows[frame_idx]["sim3"])
        face_keep, row = projection_filter(
            candidate_vertices,
            prior_faces,
            observed_vertices,
            observed_faces,
            manifest[frame_idx],
            annotations[frame_idx],
            np.asarray(depths[frame_idx], dtype=np.float64),
            args,
        )
        kept_prior_faces = prior_faces[face_keep]
        vertices, faces = append_faces(observed_vertices, observed_faces, candidate_vertices, kept_prior_faces)
        archive_rows.append((frame_idx, vertices, faces))
        row.update(
            {
                "frame_idx": int(frame_idx),
                "observed_vertices": int(len(observed_vertices)),
                "observed_faces": int(len(observed_faces)),
                "archive_vertices": int(len(vertices)),
                "archive_faces": int(len(faces)),
            }
        )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observed_plus_hidden_prior_meshes_world.npz"
    write_mesh_archive(archive, archive_rows)
    kept = np.asarray([row["kept_prior_faces"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "append_v9_hidden_prior_faces_to_observed_mesh",
        "claim_tested": "accepted measured visible object meshes remain exact while hidden generated-prior faces are appended where object mask, metric depth, camera free space, and measured surface proximity support them",
        "mesh_prior": str(args.mesh_prior),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "alignment_report": str(args.alignment_report),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "mesh_archive": str(archive),
        "output_mesh_archive": str(archive),
        "frame_count": int(len(frames)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "kept_prior_faces": summarize(kept),
        "prior_simplification": simplification_report,
        "parameters": {
            "max_prior_faces": int(args.max_prior_faces),
            "min_hidden_distance_m": float(args.min_hidden_distance_m),
            "max_visible_depth_abs_m": float(args.max_visible_depth_abs_m),
            "max_front_free_space_m": float(args.max_front_free_space_m),
            "min_hidden_behind_observed_m": float(args.min_hidden_behind_observed_m),
            "max_raster_filter_iters": int(args.max_raster_filter_iters),
            "keep_inside_image_hidden_faces": bool(args.keep_inside_image_hidden_faces),
            "min_component_faces": int(args.min_component_faces),
        },
        "rows": rows,
    }
    out = args.output_dir / "qc_append_v9_hidden_prior_faces_to_observed_mesh.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--intrinsics-source", choices=("annotation-vggt", "manifest"), default="annotation-vggt")
    parser.add_argument("--max-prior-faces", type=int, default=60000)
    parser.add_argument("--min-hidden-distance-m", type=float, default=0.004)
    parser.add_argument("--max-visible-depth-abs-m", type=float, default=0.008)
    parser.add_argument("--max-front-free-space-m", type=float, default=0.004)
    parser.add_argument("--min-hidden-behind-observed-m", type=float, default=0.003)
    parser.add_argument("--max-raster-filter-iters", type=int, default=12)
    parser.add_argument("--allow-visible-mask-fill", action="store_true")
    parser.add_argument("--keep-inside-image-hidden-faces", action="store_true")
    parser.add_argument("--min-component-faces", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
