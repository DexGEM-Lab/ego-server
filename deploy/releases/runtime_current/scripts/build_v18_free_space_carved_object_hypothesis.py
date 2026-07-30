#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a free-space-carved compact-rigid object hypothesis.

This is a physical repair attempt for compact-rigid completions whose hidden
volume is repeatedly contradicted by metric depth. It aggregates per-canonical-
vertex depth provenance over fitted pose frames and removes faces whose vertices
are consistently in observed free space without comparable observed support.

The output is not accepted complete geometry. It is an observed/free-space
repaired object hypothesis that can be rendered and used to remeasure MANO
constraints without letting repeatedly invalid volume dominate H_{t,h}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources, project_points  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import (  # noqa: E402
    VERTEX_BEHIND_OBSERVED,
    VERTEX_FREE_SPACE_CONFLICT,
    VERTEX_OBSERVED_SUPPORTED,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    load_mesh,
    numeric_summary,
    pose_map,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-margin-m", type=float, default=0.015)
    parser.add_argument("--free-space-margin-m", type=float, default=0.025)
    parser.add_argument("--free-ratio-remove", type=float, default=0.25)
    parser.add_argument("--min-free-observations-remove", type=int, default=3)
    parser.add_argument("--support-protection-ratio", type=float, default=1.25)
    parser.add_argument("--min-component-faces", type=int, default=100)
    return parser.parse_args()


def classify_frame_vertices(
    *,
    frame: dict[str, Any],
    vertices_object: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    depth_row: dict[str, Any] | None,
    support_margin_m: float,
    free_space_margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(vertices_object)
    classes = np.zeros((n,), dtype=np.uint8)
    finite_mask = np.zeros((n,), dtype=bool)
    if depth_row is None:
        return classes, finite_mask
    r_obj, t_obj = pose
    vertices_world = vertices_object @ r_obj.T + t_obj[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    vertices_camera = (vertices_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row["depth"], dtype=np.float32)
    height, width = depth.shape
    u, v, valid = project_points(vertices_camera, np.asarray(depth_row["intrinsics"], dtype=float), width, height)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return classes, finite_mask
    z_mesh = vertices_camera[idx, 2]
    z_obs = depth[v[idx], u[idx]].astype(float)
    finite = np.isfinite(z_obs) & (z_obs > 0.0)
    finite_idx = idx[finite]
    finite_mask[finite_idx] = True
    residual = z_mesh[finite] - z_obs[finite]
    classes[finite_idx[np.abs(residual) <= float(support_margin_m)]] = VERTEX_OBSERVED_SUPPORTED
    classes[finite_idx[residual < -float(free_space_margin_m)]] = VERTEX_FREE_SPACE_CONFLICT
    classes[finite_idx[residual > float(support_margin_m)]] = VERTEX_BEHIND_OBSERVED
    return classes, finite_mask


def reindex_mesh(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept_faces_old = faces[face_mask]
    if kept_faces_old.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    kept_vertices_old = np.unique(kept_faces_old.reshape(-1))
    remap = -np.ones((len(vertices),), dtype=np.int64)
    remap[kept_vertices_old] = np.arange(len(kept_vertices_old), dtype=np.int64)
    new_faces = remap[kept_faces_old]
    return vertices[kept_vertices_old], new_faces.astype(np.int64), kept_vertices_old.astype(np.int64)


def connected_component_summary(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        comps = []
    for i, comp in enumerate(comps):
        bounds = np.asarray(comp.bounds, dtype=float) if len(comp.vertices) else np.zeros((2, 3), dtype=float)
        extent = bounds[1] - bounds[0] if bounds.shape == (2, 3) else np.zeros((3,), dtype=float)
        out.append(
            {
                "component_index": int(i),
                "vertex_count": int(len(comp.vertices)),
                "face_count": int(len(comp.faces)),
                "extent_m": extent.astype(float).tolist(),
            }
        )
    out.sort(key=lambda row: int(row["face_count"]), reverse=True)
    return out


def main() -> None:
    args = parse_args()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    depth_by_frame = load_depth_sources(args.depth_npz)
    mesh = load_mesh(args.completed_mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(vertices)

    projected = np.zeros((n,), dtype=np.int32)
    finite = np.zeros((n,), dtype=np.int32)
    support = np.zeros((n,), dtype=np.int32)
    free = np.zeros((n,), dtype=np.int32)
    behind = np.zeros((n,), dtype=np.int32)

    frame_rows: list[dict[str, Any]] = []
    frames_by_idx = {int(frame["frame_idx"]): frame for frame in as_list(annotations.get("frames")) if isinstance(frame, dict) and frame.get("frame_idx") is not None}
    for frame_idx, pose in sorted(poses.items()):
        frame = frames_by_idx.get(int(frame_idx))
        if frame is None:
            continue
        classes, finite_mask = classify_frame_vertices(
            frame=frame,
            vertices_object=vertices,
            pose=pose,
            depth_row=depth_by_frame.get(int(frame_idx)),
            support_margin_m=float(args.support_margin_m),
            free_space_margin_m=float(args.free_space_margin_m),
        )
        projected += finite_mask.astype(np.int32)
        finite += finite_mask.astype(np.int32)
        support += ((classes == VERTEX_OBSERVED_SUPPORTED) & finite_mask).astype(np.int32)
        free += ((classes == VERTEX_FREE_SPACE_CONFLICT) & finite_mask).astype(np.int32)
        behind += ((classes == VERTEX_BEHIND_OBSERVED) & finite_mask).astype(np.int32)
        finite_count = int(np.count_nonzero(finite_mask))
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "finite_vertex_count": finite_count,
                "observed_supported_vertex_count": int(np.count_nonzero((classes == VERTEX_OBSERVED_SUPPORTED) & finite_mask)),
                "free_space_conflict_vertex_count": int(np.count_nonzero((classes == VERTEX_FREE_SPACE_CONFLICT) & finite_mask)),
                "behind_observed_vertex_count": int(np.count_nonzero((classes == VERTEX_BEHIND_OBSERVED) & finite_mask)),
            }
        )

    finite_float = np.maximum(finite.astype(float), 1.0)
    free_ratio = free.astype(float) / finite_float
    support_ratio = support.astype(float) / finite_float
    behind_ratio = behind.astype(float) / finite_float
    support_protect = support.astype(float) >= float(args.support_protection_ratio) * np.maximum(free.astype(float), 1.0)
    remove_vertex = (
        (free >= int(args.min_free_observations_remove))
        & (free_ratio >= float(args.free_ratio_remove))
        & (~support_protect)
    )
    keep_vertex = ~remove_vertex
    face_keep = np.all(keep_vertex[faces], axis=1)

    # Drop tiny disconnected face islands caused by carving, but keep the main non-free hypotheses.
    carved_vertices, carved_faces, old_vertex_ids = reindex_mesh(vertices, faces, face_keep)
    carved_mesh = trimesh.Trimesh(vertices=carved_vertices, faces=carved_faces, process=False)
    components = connected_component_summary(carved_mesh)
    if components and int(args.min_component_faces) > 0:
        try:
            split = carved_mesh.split(only_watertight=False)
            kept_components = [comp for comp in split if len(comp.faces) >= int(args.min_component_faces)]
            if kept_components:
                carved_mesh = trimesh.util.concatenate(kept_components)
                carved_vertices = np.asarray(carved_mesh.vertices, dtype=float)
                carved_faces = np.asarray(carved_mesh.faces, dtype=np.int64)
        except Exception:
            pass

    out_dir = args.output_dir / str(args.case) / str(args.object_id).replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "free_space_carved_object_mesh.ply"
    carved_mesh.export(mesh_path)

    removed_free_ratio = free_ratio[remove_vertex]
    kept_free_ratio = free_ratio[~remove_vertex]
    original_bounds = np.asarray(mesh.bounds, dtype=float)
    carved_bounds = np.asarray(carved_mesh.bounds, dtype=float) if len(carved_mesh.vertices) else np.zeros((2, 3), dtype=float)
    report = {
        "method": "build_v18_free_space_carved_object_hypothesis",
        "status": "ok",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Free-space carving of a compact-rigid completion using repeated metric-depth contradictions. The carved mesh is "
            "an uncertain observed/free-space repaired hypothesis, not accepted complete object geometry or full nonpenetration proof."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(path) for path in args.depth_npz],
        },
        "parameters": {
            "support_margin_m": float(args.support_margin_m),
            "free_space_margin_m": float(args.free_space_margin_m),
            "free_ratio_remove": float(args.free_ratio_remove),
            "min_free_observations_remove": int(args.min_free_observations_remove),
            "support_protection_ratio": float(args.support_protection_ratio),
            "min_component_faces": int(args.min_component_faces),
            "remove_rule": "remove canonical vertices repeatedly observed in free space unless observed support outweighs free evidence",
        },
        "outputs": {"carved_mesh": str(mesh_path)},
        "summary": {
            "pose_frame_count": int(len(poses)),
            "evaluated_pose_frame_count": int(len(frame_rows)),
            "original_vertex_count": int(len(vertices)),
            "original_face_count": int(len(faces)),
            "removed_vertex_count": int(np.count_nonzero(remove_vertex)),
            "kept_vertex_count_before_component_filter": int(np.count_nonzero(keep_vertex)),
            "kept_face_count_before_component_filter": int(np.count_nonzero(face_keep)),
            "carved_vertex_count": int(len(carved_mesh.vertices)),
            "carved_face_count": int(len(carved_mesh.faces)),
            "original_extent_m": (original_bounds[1] - original_bounds[0]).astype(float).tolist(),
            "carved_extent_m": (carved_bounds[1] - carved_bounds[0]).astype(float).tolist() if carved_bounds.shape == (2, 3) else [0.0, 0.0, 0.0],
            "removed_vertex_free_ratio": numeric_summary(removed_free_ratio),
            "kept_vertex_free_ratio": numeric_summary(kept_free_ratio),
            "free_ratio_all_vertices": numeric_summary(free_ratio),
            "support_ratio_all_vertices": numeric_summary(support_ratio),
            "behind_ratio_all_vertices": numeric_summary(behind_ratio),
        },
        "component_summary_before_filter": components[:20],
        "frame_rows": frame_rows,
        "physical_conclusion": (
            "If the carved mesh visibly reduces free-space volume and reduces MANO observed-surface blockers, the prior completion was "
            "overbroad. If it destroys observed support or leaves MANO blockers unchanged, the current pose/depth/hand model remains the blocker."
        ),
    }
    report_path = out_dir / "v18_free_space_carved_object_hypothesis_report.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "mesh": str(mesh_path), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
