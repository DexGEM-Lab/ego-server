#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from optimize_object_factor_graph_v3 import mask_distance_map, project_world, resize_bool_mask
from run_v16_full_pipeline import load_mesh_archive


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compact_intrinsics(frame: dict[str, Any]) -> np.ndarray | None:
    for hand in frame.get("hands", []):
        vals = hand.get("source_intrinsics")
        if vals is None:
            continue
        intr = np.asarray(vals, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    obj = frame.get("object", {})
    qc = obj.get("mesh_qc", {}) if isinstance(obj, dict) else {}
    vals = qc.get("source_intrinsics")
    if vals is not None:
        intr = np.asarray(vals, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    return None


def hand_vertices(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for hand_i, hand in enumerate(frame.get("hands", [])):
        verts = None
        for key in ("vertices_world_m", "vertices3d_world_m"):
            if key in hand:
                candidate = np.asarray(hand[key], dtype=np.float64)
                if candidate.ndim == 2 and candidate.shape[1] == 3 and np.isfinite(candidate).all():
                    verts = candidate
                    break
        if verts is None:
            continue
        rows.append(
            {
                "hand_index": int(hand_i),
                "side": hand.get("side") or f"hand_{hand_i}",
                "vertices_world": verts,
                "detector_score": hand.get("detector_score"),
                "measurement_available": hand.get("measurement_available"),
                "projection_residual_to_measurement_px": hand.get("projection_residual_to_measurement_px"),
            }
        )
    return rows


def sample_rows(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    take = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return points[take]


def point_mesh_distances(points: np.ndarray, vertices: np.ndarray, max_points: int, max_vertices: int) -> np.ndarray:
    if len(points) == 0 or len(vertices) == 0:
        return np.zeros(0, dtype=np.float64)
    points = sample_rows(points, max_points)
    vertices = sample_rows(vertices, max_vertices)
    diff = points[:, None, :] - vertices[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def distance_at_points(distance: np.ndarray, uv: np.ndarray) -> np.ndarray:
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        return np.zeros(0, dtype=np.float32)
    xy = np.rint(uv[valid]).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < distance.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < distance.shape[0])
    if not inside.any():
        return np.zeros(0, dtype=np.float32)
    xy = xy[inside]
    return distance[xy[:, 1], xy[:, 0]]


def object_mask_path(frame: dict[str, Any]) -> Path | None:
    obj = frame.get("object", {})
    if not isinstance(obj, dict):
        return None
    mask = obj.get("mask_path")
    if isinstance(mask, str) and mask:
        path = Path(mask)
        if path.exists():
            return path
    return None


def summarize(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    vals = values.astype(float)
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def contact_measurement(
    frame: dict[str, Any],
    hand: dict[str, Any],
    object_vertices: np.ndarray | None,
    mask: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    idx = int(frame["frame_idx"])
    intr = compact_intrinsics(frame)
    T_wc = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    verts = hand["vertices_world"]
    sampled = sample_rows(verts, int(args.max_hand_vertices_image))
    image_dist = np.zeros(0, dtype=np.float32)
    projected_vertices = 0
    near_mask = None
    if intr is not None and mask is not None:
        uv, z = project_world(sampled, T_wc, intr)
        projected_vertices = int(np.isfinite(uv).all(axis=1).sum())
        image_dist = distance_at_points(mask_distance_map(mask), uv[z > 1e-5])
        if image_dist.size:
            near_mask = int(np.sum(image_dist <= float(args.image_near_px)))
    mesh_dist = np.zeros(0, dtype=np.float64)
    if object_vertices is not None:
        mesh_dist = point_mesh_distances(verts, object_vertices, int(args.max_hand_vertices_3d), int(args.max_object_vertices_3d))
    image_candidate = bool(image_dist.size and float(np.min(image_dist)) <= float(args.image_near_px))
    metric_candidate = bool(mesh_dist.size and float(np.min(mesh_dist)) <= float(args.metric_near_m))
    hand_residual = hand.get("projection_residual_to_measurement_px")
    hand_reprojection_median = None
    if isinstance(hand_residual, dict) and hand_residual.get("median") is not None:
        hand_reprojection_median = float(hand_residual["median"])
    hand_measurement_valid = bool(
        hand.get("measurement_available") is not False
        and hand_reprojection_median is not None
        and hand_reprojection_median <= float(args.max_hand_reprojection_px)
    )
    if image_candidate and metric_candidate:
        contact_state = "candidate_contact_image_and_metric"
    elif image_candidate:
        contact_state = "candidate_contact_image_only"
    elif metric_candidate:
        contact_state = "candidate_contact_metric_only"
    else:
        contact_state = "no_contact_evidence"
    if contact_state.startswith("candidate_contact") and not hand_measurement_valid:
        contact_state = "contact_evidence_requires_hand_repair"
    return {
        "measurement_id": f"{args.measurement_id_prefix}:{idx}:{hand['side']}:{hand['hand_index']}",
        "frame_idx": idx,
        "entity_type": "contact",
        "entity_id": f"contact:{hand['side']}:object",
        "measurement_type": "hand_object_contact_evidence",
        "source_model": "v17_mask_projection_plus_mesh_proximity",
        "coordinate_frame": "source_image_pixels_and_v16_world",
        "confidence": None,
        "hand_side": hand["side"],
        "hand_index": hand["hand_index"],
        "object_label": frame.get("object", {}).get("label"),
        "contact_state_measurement": contact_state,
        "image_near_px": float(args.image_near_px),
        "metric_near_m": float(args.metric_near_m),
        "projected_hand_vertices": projected_vertices,
        "hand_vertices_near_object_mask": near_mask,
        "image_mask_distance_px": summarize(image_dist),
        "hand_object_mesh_distance_m": summarize(mesh_dist),
        "hand_detector_score": hand.get("detector_score"),
        "hand_measurement_available": hand.get("measurement_available"),
        "hand_projection_residual_to_measurement_px": hand.get("projection_residual_to_measurement_px"),
        "hand_measurement_valid_for_contact": hand_measurement_valid,
        "max_hand_reprojection_px": float(args.max_hand_reprojection_px),
        "failure_reason": None if image_dist.size or mesh_dist.size else "missing_image_and_metric_contact_evidence",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    meshes = load_mesh_archive(args.object_mesh_archive) if args.object_mesh_archive else {}
    frame_indices = sorted(frames) if not args.frame_indices else sorted({int(part) for raw in args.frame_indices for part in raw.split(",") if part})
    measurements = []
    by_state: dict[str, int] = {}
    for idx in frame_indices:
        frame = frames.get(idx)
        if frame is None:
            continue
        mask_path = object_mask_path(frame)
        mask = resize_bool_mask(mask_path, (int(args.mask_width), int(args.mask_height))) if mask_path else None
        object_vertices = meshes.get(idx, (None, None))[0] if idx in meshes else None
        for hand in hand_vertices(frame):
            row = contact_measurement(frame, hand, object_vertices, mask, args)
            measurements.append(row)
            state = row["contact_state_measurement"]
            by_state[state] = by_state.get(state, 0) + 1
    report = {
        "status": "ok",
        "method": "build_v17_contact_measurements",
        "annotations": str(args.annotations),
        "object_mesh_archive": str(args.object_mesh_archive) if args.object_mesh_archive else None,
        "frames_considered": len(frame_indices),
        "measurement_count": len(measurements),
        "contact_state_counts": by_state,
        "thresholds": {"image_near_px": float(args.image_near_px), "metric_near_m": float(args.metric_near_m)},
        "output_measurements": str(args.output_json),
    }
    write_json(args.output_json, measurements)
    write_json(args.output_summary, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--frame-indices", nargs="*")
    parser.add_argument("--mask-width", type=int, default=1920)
    parser.add_argument("--mask-height", type=int, default=1080)
    parser.add_argument("--image-near-px", type=float, default=12.0)
    parser.add_argument("--metric-near-m", type=float, default=0.02)
    parser.add_argument("--max-hand-reprojection-px", type=float, default=45.0)
    parser.add_argument("--max-hand-vertices-image", type=int, default=778)
    parser.add_argument("--max-hand-vertices-3d", type=int, default=180)
    parser.add_argument("--max-object-vertices-3d", type=int, default=2200)
    parser.add_argument("--measurement-id-prefix", default="contact:v17")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
