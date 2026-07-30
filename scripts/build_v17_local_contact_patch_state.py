#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from build_v17_contact_measurements import point_mesh_distances, summarize
from build_v17_contact_depth_object_repair import frames_by_index, object_masks_by_frame
from optimize_object_factor_graph_v3 import mask_distance_map, project_world, resize_bool_mask
from run_v16_full_pipeline import load_metric_depth, mesh_from_mask_depth, save_mesh_archive


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def slug(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    if not out:
        raise RuntimeError(f"cannot build slug from {value!r}")
    return out


def source_size_from_hand(hand: dict[str, Any]) -> tuple[int, int]:
    intr = np.asarray(hand.get("source_intrinsics"), dtype=np.float64)
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError("repair hand has no valid source_intrinsics")
    return int(round(max(1.0, 2.0 * float(intr[2])))), int(round(max(1.0, 2.0 * float(intr[3]))))


def selected_hand(
    frame: dict[str, Any],
    required_hand_side: str | None,
    required_repair_candidate_id: str | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for hand_i, hand in enumerate(frame.get("hands", [])):
        if not isinstance(hand, dict):
            continue
        if required_hand_side is not None and hand.get("side") != required_hand_side:
            continue
        if required_repair_candidate_id is not None and hand.get("v17_repair_candidate_id") != required_repair_candidate_id:
            continue
        vertices = np.asarray(hand.get("vertices_world_m"), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            continue
        row = dict(hand)
        row["hand_index"] = hand_i
        row["vertices_world"] = vertices
        candidates.append(row)
    if len(candidates) != 1:
        raise RuntimeError(
            f"frame {frame.get('frame_idx')} expected one selected hand, found {len(candidates)}: "
            f"side={required_hand_side!r} repair_candidate={required_repair_candidate_id!r}"
        )
    return candidates[0]


def finite_residual_median(hand: dict[str, Any]) -> float | None:
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return None
    value = residual.get("median")
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def local_patch_mask(
    frame: dict[str, Any],
    hand: dict[str, Any],
    object_mask_path: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_size = source_size_from_hand(hand)
    object_mask = resize_bool_mask(object_mask_path, source_size)
    distance = mask_distance_map(object_mask)
    vertices = np.asarray(hand["vertices_world"], dtype=np.float64)
    intr = np.asarray(hand["source_intrinsics"], dtype=np.float64)
    T_wc = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    if T_wc.shape != (4, 4) or not np.isfinite(T_wc).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid camera pose")
    uv, z = project_world(vertices, T_wc, intr)
    xy = np.rint(uv).astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (z > 1e-5)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < distance.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < distance.shape[0])
    )
    d = np.full(len(vertices), np.nan, dtype=np.float64)
    d[inside] = distance[xy[inside, 1], xy[inside, 0]]
    near = np.isfinite(d) & (d <= float(args.image_near_px))
    near_count = int(np.count_nonzero(near))
    if near_count < int(args.min_near_hand_vertices):
        raise RuntimeError(f"frame {frame.get('frame_idx')} has only {near_count} near hand vertices")
    near_xy = xy[near]
    contact_center_xy = np.median(near_xy.astype(np.float64), axis=0)
    yy, xx = np.ogrid[: object_mask.shape[0], : object_mask.shape[1]]
    radius2 = float(args.patch_radius_px) ** 2
    patch = ((xx - float(contact_center_xy[0])) ** 2 + (yy - float(contact_center_xy[1])) ** 2 <= radius2) & object_mask
    patch_pixels = int(np.count_nonzero(patch))
    if patch_pixels < int(args.min_patch_pixels):
        raise RuntimeError(f"frame {frame.get('frame_idx')} local patch has only {patch_pixels} pixels")
    T_cw = np.linalg.inv(T_wc)
    vertices_camera = (np.c_[vertices, np.ones(len(vertices))] @ T_cw.T)[:, :3]
    near_depth = vertices_camera[near, 2]
    return patch, {
        "projected_hand_vertices": int(np.count_nonzero(inside)),
        "hand_vertices_near_object_mask": near_count,
        "hand_depth_anchor_m": float(np.median(near_depth)),
        "hand_depth_p05_m": float(np.percentile(near_depth, 5.0)),
        "hand_depth_p95_m": float(np.percentile(near_depth, 95.0)),
        "image_mask_distance_px": summarize(d[np.isfinite(d)]),
        "contact_center_xy": [float(contact_center_xy[0]), float(contact_center_xy[1])],
        "patch_radius_px": float(args.patch_radius_px),
        "patch_pixels": patch_pixels,
        "object_mask_pixels": int(np.count_nonzero(object_mask)),
        "patch_object_mask_fraction": float(patch_pixels / max(1, int(np.count_nonzero(object_mask)))),
    }


def frame_with_patch(frame: dict[str, Any], object_id: str, patch_mask_path: Path, depth_anchor_m: float) -> dict[str, Any]:
    out = dict(frame)
    obj = dict(frame.get("object", {}))
    obj["label"] = object_id
    obj["mask_path"] = str(patch_mask_path)
    obj["depth_m"] = float(depth_anchor_m)
    obj["v17_object_state"] = "local_deformable_contact_patch"
    out["object"] = obj
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.output_dir / "patch_masks"
    patch_dir.mkdir(parents=True, exist_ok=True)
    frames = frames_by_index(load_json(args.annotations))
    masks = object_masks_by_frame(args.sam2_object_mask_measurements, str(args.object_id))
    depth = load_metric_depth(args.metric_depth_npz)
    frame_indices = [int(part) for raw in args.frame_indices for part in raw.split(",") if part]
    if not frame_indices:
        raise RuntimeError("at least one frame index is required")
    patch_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    mesh_frames: list[int] = []
    mesh_vertices: list[np.ndarray] = []
    mesh_faces: list[np.ndarray] = []
    for frame_idx in frame_indices:
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        object_row = masks.get(frame_idx)
        if object_row is None:
            raise RuntimeError(f"{args.object_id} has no visible object mask measurement at frame {frame_idx}")
        hand = selected_hand(frame, args.required_hand_side, args.required_repair_candidate_id)
        object_mask_path = Path(str(object_row["mask_path"]))
        patch_mask, evidence = local_patch_mask(frame, hand, object_mask_path, args)
        patch_path = patch_dir / f"{frame_idx:06d}_{slug(args.object_id)}_{slug(str(hand.get('side') or 'hand'))}.png"
        cv2.imwrite(str(patch_path), (patch_mask.astype(np.uint8) * 255))
        vertices, faces, mesh_row = mesh_from_mask_depth(
            frame_with_patch(frame, str(args.object_id), patch_path, float(evidence["hand_depth_anchor_m"])),
            depth,
            mask_stride=int(args.mask_stride),
            mask_erode_px=int(args.mask_erode_px),
            max_triangle_edge_m=float(args.max_triangle_edge_m),
            min_vertices=int(args.min_vertices),
            min_faces=int(args.min_faces),
            min_depth_m=float(args.min_depth_m),
            max_depth_m=float(args.max_depth_m),
            depth_low_quantile=float(args.depth_low_quantile),
            depth_high_quantile=float(args.depth_high_quantile),
        )
        if mesh_row["status"] != "measured_mesh_from_mask_metric_depth":
            raise RuntimeError(f"frame {frame_idx} local contact patch mesh failed: {mesh_row}")
        mesh_frames.append(frame_idx)
        mesh_vertices.append(vertices)
        mesh_faces.append(faces)
        mesh_dist = point_mesh_distances(
            np.asarray(hand["vertices_world"], dtype=np.float64),
            vertices,
            int(args.max_hand_vertices_3d),
            int(args.max_patch_vertices_3d),
        )
        hand_reprojection_median = finite_residual_median(hand)
        hand_measurement_valid = bool(
            hand.get("measurement_available") is not False
            and hand_reprojection_median is not None
            and hand_reprojection_median <= float(args.max_hand_reprojection_px)
        )
        image_candidate = evidence["image_mask_distance_px"].get("min") is not None and float(evidence["image_mask_distance_px"]["min"]) <= float(args.image_near_px)
        metric_candidate = mesh_dist.size > 0 and float(np.min(mesh_dist)) <= float(args.metric_near_m)
        contact_state = "candidate_contact_image_and_metric" if image_candidate and metric_candidate else "candidate_contact_patch_unresolved"
        if contact_state.startswith("candidate_contact") and not hand_measurement_valid:
            contact_state = "contact_evidence_requires_hand_repair"
        patch_id = f"local_contact_patch:{frame_idx}:{slug(str(args.object_id))}:{hand.get('side')}"
        annotation_ready = contact_state == "candidate_contact_image_and_metric"
        patch_rows.append(
            {
                "measurement_id": patch_id,
                "frame_idx": frame_idx,
                "entity_type": "object",
                "entity_id": str(args.object_id),
                "measurement_type": "local_deformable_contact_patch_state",
                "source_model": "v17_hand_object_mask_local_patch_depth_state",
                "coordinate_frame": "v16_world_metric",
                "status": "accepted_local_contact_patch_state" if annotation_ready else "unresolved_local_contact_patch_state",
                "annotation_ready": annotation_ready,
                "contact_measurement_id": f"contact_patch:v17:{frame_idx}:{hand.get('side')}:{hand['hand_index']}",
                "contact_state_measurement": contact_state,
                "patch_mask_path": str(patch_path),
                "source_object_mask_path": str(object_mask_path),
                "mesh_archive": str(args.output_dir / "local_contact_patch_meshes_world.npz"),
                "hand_side": hand.get("side"),
                "hand_index": int(hand["hand_index"]),
                "hand_repair_candidate_id": hand.get("v17_repair_candidate_id"),
                "mesh_vertices": int(len(vertices)),
                "mesh_faces": int(len(faces)),
                "world_extent_m": mesh_row.get("world_extent_m"),
                "hand_object_mesh_distance_m": summarize(mesh_dist),
                **evidence,
            }
        )
        contact_rows.append(
            {
                "measurement_id": f"contact_patch:v17:{frame_idx}:{hand.get('side')}:{hand['hand_index']}",
                "frame_idx": frame_idx,
                "entity_type": "contact",
                "entity_id": f"contact:{hand.get('side')}:object",
                "measurement_type": "hand_object_contact_evidence",
                "source_model": "v17_local_deformable_contact_patch_state",
                "coordinate_frame": "source_image_pixels_and_v16_world",
                "confidence": None,
                "hand_side": hand.get("side"),
                "hand_index": int(hand["hand_index"]),
                "object_label": str(args.object_id),
                "contact_state_measurement": contact_state,
                "image_near_px": float(args.image_near_px),
                "metric_near_m": float(args.metric_near_m),
                "projected_hand_vertices": evidence["projected_hand_vertices"],
                "hand_vertices_near_object_mask": evidence["hand_vertices_near_object_mask"],
                "image_mask_distance_px": evidence["image_mask_distance_px"],
                "hand_object_mesh_distance_m": summarize(mesh_dist),
                "hand_detector_score": hand.get("detector_score"),
                "hand_measurement_available": hand.get("measurement_available"),
                "hand_projection_residual_to_measurement_px": hand.get("projection_residual_to_measurement_px"),
                "hand_measurement_valid_for_contact": hand_measurement_valid,
                "max_hand_reprojection_px": float(args.max_hand_reprojection_px),
                "local_patch_state_id": patch_id,
                "local_patch_mask_path": str(patch_path),
                "failure_reason": None if contact_state == "candidate_contact_image_and_metric" else "local_patch_contact_not_metric_supported",
            }
        )
    mesh_archive = args.output_dir / "local_contact_patch_meshes_world.npz"
    save_mesh_archive(mesh_archive, mesh_frames, mesh_vertices, mesh_faces)
    write_json(args.output_dir / "local_contact_patch_states.json", patch_rows)
    write_json(args.output_dir / "local_contact_patch_contact_measurements.json", contact_rows)
    summary = {
        "status": "ok",
        "method": "build_v17_local_contact_patch_state",
        "claim": "local deformable object contact patch state for mask-supported hand contact without shifting whole-object depth",
        "object_id": str(args.object_id),
        "frame_indices": frame_indices,
        "patch_state_count": len(patch_rows),
        "contact_measurement_count": len(contact_rows),
        "mesh_archive": str(mesh_archive),
        "patch_states": str(args.output_dir / "local_contact_patch_states.json"),
        "contact_measurements": str(args.output_dir / "local_contact_patch_contact_measurements.json"),
    }
    write_json(args.output_dir / "local_contact_patch_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sam2-object-mask-measurements", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", nargs="+", required=True)
    parser.add_argument("--required-hand-side")
    parser.add_argument("--required-repair-candidate-id")
    parser.add_argument("--image-near-px", type=float, default=12.0)
    parser.add_argument("--metric-near-m", type=float, default=0.02)
    parser.add_argument("--patch-radius-px", type=float, default=64.0)
    parser.add_argument("--min-near-hand-vertices", type=int, default=40)
    parser.add_argument("--min-patch-pixels", type=int, default=500)
    parser.add_argument("--mask-stride", type=int, default=2)
    parser.add_argument("--mask-erode-px", type=int, default=0)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.06)
    parser.add_argument("--min-vertices", type=int, default=40)
    parser.add_argument("--min-faces", type=int, default=40)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=3.20)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--max-hand-reprojection-px", type=float, default=45.0)
    parser.add_argument("--max-hand-vertices-3d", type=int, default=180)
    parser.add_argument("--max-patch-vertices-3d", type=int, default=2200)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
