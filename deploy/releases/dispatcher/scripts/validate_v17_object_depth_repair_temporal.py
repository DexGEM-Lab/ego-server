#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

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


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compact_intrinsics(frame: dict[str, Any]) -> np.ndarray:
    for hand in frame.get("hands", []):
        raw = hand.get("source_intrinsics")
        if raw is None:
            continue
        intr = np.asarray(raw, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    obj = frame.get("object", {})
    qc = obj.get("mesh_qc", {}) if isinstance(obj, dict) else {}
    raw = qc.get("source_intrinsics")
    if raw is not None:
        intr = np.asarray(raw, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    raise RuntimeError(f"frame {frame.get('frame_idx')} has no valid source intrinsics")


def source_size(frame: dict[str, Any]) -> tuple[int, int]:
    intr = compact_intrinsics(frame)
    source_w = int(round(float(frame.get("source_width", 0) or 0)))
    source_h = int(round(float(frame.get("source_height", 0) or 0)))
    if source_w > 0 and source_h > 0:
        return source_w, source_h
    return int(round(2.0 * float(intr[2]))), int(round(2.0 * float(intr[3])))


def depth_values_for_mask(frame_idx: int, mask_path: Path, depth: dict[str, Any]) -> np.ndarray:
    depth_i = depth["frame_to_i"].get(int(frame_idx))
    if depth_i is None:
        raise RuntimeError(f"depth archive has no frame {frame_idx}")
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    if mask.shape != depth_m.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    values = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)]
    if values.size == 0:
        raise RuntimeError(f"frame {frame_idx} mask has no positive finite depth values")
    return values


def frame_with_object(frame: dict[str, Any], mask_path: Path, depth_anchor_m: float | None) -> dict[str, Any]:
    out = dict(frame)
    obj = dict(frame.get("object", {}))
    obj["mask_path"] = str(mask_path)
    if depth_anchor_m is None:
        obj.pop("depth_m", None)
    else:
        obj["depth_m"] = float(depth_anchor_m)
    out["object"] = obj
    return out


def mesh_for_hypothesis(
    frame: dict[str, Any],
    mask_path: Path,
    depth: dict[str, Any],
    depth_anchor_m: float | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices, faces, row = mesh_from_mask_depth(
        frame_with_object(frame, mask_path, depth_anchor_m),
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
    if row["status"] != "measured_mesh_from_mask_metric_depth":
        raise RuntimeError(f"hypothesis mesh failed for frame {frame.get('frame_idx')}: {row}")
    return vertices, faces, row


def projected_vertex_mask(
    uv: np.ndarray,
    z: np.ndarray,
    shape: tuple[int, int],
    dilate_px: int,
) -> tuple[np.ndarray, int]:
    h, w = shape
    xy = np.rint(uv).astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (z > 1e-5)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[xy[inside, 1], xy[inside, 0]] = 1
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)
    return canvas > 0, int(np.count_nonzero(inside))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        raise RuntimeError("cannot compare two empty masks")
    return float(inter / union)


def projection_row(
    hypothesis_name: str,
    vertices: np.ndarray,
    base_frame: dict[str, Any],
    base_mask_path: Path,
    target_frame: dict[str, Any],
    target_mask_path: Path,
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask = resize_bool_mask(target_mask_path, source_size(target_frame))
    base_mask = resize_bool_mask(base_mask_path, source_size(target_frame))
    intr = compact_intrinsics(target_frame)
    T_wc = np.asarray(target_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    if T_wc.shape != (4, 4) or not np.isfinite(T_wc).all():
        raise RuntimeError(f"frame {target_frame.get('frame_idx')} has invalid camera pose")
    uv, z = project_world(vertices, T_wc, intr)
    xy = np.rint(uv).astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (z > 1e-5)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    dist = np.full(len(vertices), np.nan, dtype=np.float64)
    distance = mask_distance_map(mask)
    dist[inside] = distance[xy[inside, 1], xy[inside, 0]]
    finite_dist = dist[np.isfinite(dist)]
    near = finite_dist <= float(args.image_near_px)
    proj_mask, projected_vertices = projected_vertex_mask(uv, z, mask.shape, int(args.projection_dilate_px))
    overlap = proj_mask & mask
    mask_pixels = int(np.count_nonzero(mask))
    proj_pixels = int(np.count_nonzero(proj_mask))

    depth_i = depth["frame_to_i"].get(int(target_frame["frame_idx"]))
    if depth_i is None:
        raise RuntimeError(f"depth archive has no frame {target_frame['frame_idx']}")
    raw_depth = depth["depth"][int(depth_i)].astype(np.float64)
    if raw_depth.shape != mask.shape:
        raw_depth = cv2.resize(raw_depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth_sample_ok = inside & mask[xy[:, 1].clip(0, mask.shape[0] - 1), xy[:, 0].clip(0, mask.shape[1] - 1)]
    sampled_raw = raw_depth[xy[depth_sample_ok, 1], xy[depth_sample_ok, 0]]
    sampled_z = z[depth_sample_ok]
    valid_depth = np.isfinite(sampled_raw) & (sampled_raw > 0.0) & np.isfinite(sampled_z)
    depth_abs_residual = np.abs(sampled_z[valid_depth] - sampled_raw[valid_depth])

    base_center = np.asarray(base_frame["camera"]["T_world_camera_metric"], dtype=np.float64)[:3, 3]
    target_center = T_wc[:3, 3]
    return {
        "hypothesis": hypothesis_name,
        "target_frame_idx": int(target_frame["frame_idx"]),
        "frame_offset": int(target_frame["frame_idx"]) - int(base_frame["frame_idx"]),
        "base_target_mask_iou": mask_iou(base_mask, mask),
        "camera_translation_m": float(np.linalg.norm(target_center - base_center)),
        "projected_vertices": projected_vertices,
        "vertex_mask_distance_px": summarize(finite_dist),
        "vertex_near_mask_fraction": float(np.mean(near)) if finite_dist.size else None,
        "projected_pixel_count": proj_pixels,
        "projected_pixel_inside_mask_fraction": float(np.count_nonzero(overlap) / proj_pixels) if proj_pixels else None,
        "object_mask_coverage_fraction": float(np.count_nonzero(overlap) / mask_pixels) if mask_pixels else None,
        "raw_depth_abs_residual_m": summarize(depth_abs_residual),
    }


def aggregate_projection(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    comparable_rows = [
        row
        for row in rows
        if float(row["base_target_mask_iou"]) >= float(args.min_base_target_mask_iou)
        and float(row["camera_translation_m"]) >= float(args.min_camera_translation_m)
    ]
    medians = np.asarray(
        [
            row["vertex_mask_distance_px"]["median"]
            for row in comparable_rows
            if row["vertex_mask_distance_px"]["median"] is not None
        ],
        dtype=np.float64,
    )
    near = np.asarray(
        [row["vertex_near_mask_fraction"] for row in comparable_rows if row["vertex_near_mask_fraction"] is not None]
    )
    inside = np.asarray(
        [
            row["projected_pixel_inside_mask_fraction"]
            for row in comparable_rows
            if row["projected_pixel_inside_mask_fraction"] is not None
        ],
        dtype=np.float64,
    )
    coverage = np.asarray(
        [
            row["object_mask_coverage_fraction"]
            for row in comparable_rows
            if row["object_mask_coverage_fraction"] is not None
        ],
        dtype=np.float64,
    )
    depth_residual = np.asarray(
        [
            row["raw_depth_abs_residual_m"]["median"]
            for row in comparable_rows
            if row["raw_depth_abs_residual_m"]["median"] is not None
        ],
        dtype=np.float64,
    )
    camera_motion = np.asarray([row["camera_translation_m"] for row in rows], dtype=np.float64)
    comparable_motion = np.asarray([row["camera_translation_m"] for row in comparable_rows], dtype=np.float64)
    mask_iou_values = np.asarray([row["base_target_mask_iou"] for row in rows], dtype=np.float64)
    return {
        "projection_frame_count": len(rows),
        "comparable_projection_frame_count": len(comparable_rows),
        "min_base_target_mask_iou": float(args.min_base_target_mask_iou),
        "median_base_target_mask_iou": float(np.median(mask_iou_values)) if mask_iou_values.size else None,
        "median_vertex_mask_distance_px": float(np.median(medians)) if medians.size else None,
        "median_vertex_near_mask_fraction": float(np.median(near)) if near.size else None,
        "median_projected_pixel_inside_mask_fraction": float(np.median(inside)) if inside.size else None,
        "median_object_mask_coverage_fraction": float(np.median(coverage)) if coverage.size else None,
        "median_raw_depth_abs_residual_m": float(np.median(depth_residual)) if depth_residual.size else None,
        "median_camera_translation_m": float(np.median(camera_motion)) if camera_motion.size else None,
        "max_camera_translation_m": float(np.max(camera_motion)) if camera_motion.size else None,
        "median_comparable_camera_translation_m": float(np.median(comparable_motion)) if comparable_motion.size else None,
        "max_comparable_camera_translation_m": float(np.max(comparable_motion)) if comparable_motion.size else None,
    }


def validation_status(hypotheses: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    contact = hypotheses.get("contact_depth_candidate")
    raw = hypotheses.get("raw_depth_median")
    if contact is None or raw is None:
        raise RuntimeError("validation requires contact_depth_candidate and raw_depth_median hypotheses")
    contact_agg = contact["aggregate_projection"]
    raw_agg = raw["aggregate_projection"]
    if int(contact_agg["comparable_projection_frame_count"]) < int(args.min_temporal_frames):
        return {
            "status": "insufficient_comparable_temporal_frames",
            "reason": "too_few_neighbor_masks_with_similar_object_support_and_camera_motion",
            "comparable_projection_frame_count": int(contact_agg["comparable_projection_frame_count"]),
            "min_temporal_frames": int(args.min_temporal_frames),
            "min_base_target_mask_iou": float(args.min_base_target_mask_iou),
        }
    if float(contact_agg["max_comparable_camera_translation_m"] or 0.0) < float(args.min_camera_translation_m):
        return {"status": "insufficient_temporal_parallax", "reason": "camera_motion_too_small"}
    contact_distance = contact_agg["median_vertex_mask_distance_px"]
    raw_distance = raw_agg["median_vertex_mask_distance_px"]
    contact_inside = contact_agg["median_projected_pixel_inside_mask_fraction"]
    if contact_distance is None or raw_distance is None or contact_inside is None:
        return {"status": "invalid_temporal_measurement", "reason": "missing_projection_metric"}
    contact_good = (
        float(contact_distance) <= float(args.max_contact_median_mask_distance_px)
        and float(contact_inside) >= float(args.min_projected_inside_fraction)
    )
    improves_raw = float(contact_distance) + float(args.min_mask_distance_improvement_px) <= float(raw_distance)
    status = "accepted_temporal_mask_support" if contact_good and improves_raw else "rejected_temporal_mask_support"
    return {
        "status": status,
        "contact_median_vertex_mask_distance_px": float(contact_distance),
        "raw_median_vertex_mask_distance_px": float(raw_distance),
        "contact_median_projected_inside_fraction": float(contact_inside),
        "min_mask_distance_improvement_px": float(args.min_mask_distance_improvement_px),
        "max_contact_median_mask_distance_px": float(args.max_contact_median_mask_distance_px),
        "min_projected_inside_fraction": float(args.min_projected_inside_fraction),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = frames_by_index(load_json(args.annotations))
    object_masks = object_masks_by_frame(args.sam2_object_mask_measurements, args.object_id)
    candidates = load_json(args.object_depth_repair_candidates)
    if not isinstance(candidates, list):
        raise RuntimeError(f"{args.object_depth_repair_candidates} must contain a JSON list")
    depth = load_metric_depth(args.metric_depth_npz)
    reports = []
    saved_frames: list[int] = []
    saved_vertices: list[np.ndarray] = []
    saved_faces: list[np.ndarray] = []
    for candidate_i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"candidate row {candidate_i} is not a JSON object")
        frame_idx = int(candidate["frame_idx"])
        base_frame = annotations.get(frame_idx)
        if base_frame is None:
            raise RuntimeError(f"annotations missing candidate frame {frame_idx}")
        base_mask_row = object_masks.get(frame_idx)
        if base_mask_row is None:
            raise RuntimeError(f"{args.object_id} has no visible mask at candidate frame {frame_idx}")
        base_mask_path = Path(str(base_mask_row["mask_path"]))
        raw_values = depth_values_for_mask(frame_idx, base_mask_path, depth)
        hypotheses: dict[str, dict[str, Any]] = {
            "raw_depth_median": {"depth_anchor_m": None, "source": "masked_unidepth_median"},
            "contact_depth_candidate": {
                "depth_anchor_m": float(candidate["hand_depth_anchor_m"]),
                "source": "repaired_hand_contact_depth",
            },
        }
        obj = base_frame.get("object", {})
        obj_depth = obj.get("depth_m") if isinstance(obj, dict) else None
        if isinstance(obj_depth, int | float) and np.isfinite(float(obj_depth)) and float(obj_depth) > 0.0:
            hypotheses["annotation_object_depth"] = {
                "depth_anchor_m": float(obj_depth),
                "source": "annotation_object_depth",
            }
        for name, hypothesis in hypotheses.items():
            vertices, faces, mesh_row = mesh_for_hypothesis(
                base_frame,
                base_mask_path,
                depth,
                hypothesis["depth_anchor_m"],
                args,
            )
            projection_rows: list[dict[str, Any]] = []
            for target_idx in range(frame_idx - int(args.window_radius), frame_idx + int(args.window_radius) + 1):
                if target_idx == frame_idx:
                    continue
                target_frame = annotations.get(target_idx)
                target_mask = object_masks.get(target_idx)
                if target_frame is None or target_mask is None:
                    continue
                projection_rows.append(
                    projection_row(
                        name,
                        vertices,
                        base_frame,
                        base_mask_path,
                        target_frame,
                        Path(str(target_mask["mask_path"])),
                        depth,
                        args,
                    )
                )
            if name == "contact_depth_candidate":
                saved_frames.append(frame_idx)
                saved_vertices.append(vertices)
                saved_faces.append(faces)
            hypothesis.update(
                {
                    "mesh_qc": mesh_row,
                    "projection_rows": projection_rows,
                    "aggregate_projection": aggregate_projection(projection_rows, args),
                }
            )
        reports.append(
            {
                "measurement_id": f"object_depth_repair_temporal_validation:{frame_idx}:{args.object_id}",
                "frame_idx": frame_idx,
                "entity_type": "object",
                "entity_id": args.object_id,
                "measurement_type": "object_depth_repair_temporal_validation",
                "source_model": "v17_temporal_mask_projection_depth_hypothesis_test",
                "coordinate_frame": "source_image_pixels_and_v16_world",
                "candidate_measurement_id": candidate.get("measurement_id"),
                "candidate_validation_status": candidate.get("validation_status"),
                "base_raw_depth_m": summarize(raw_values),
                "hypotheses": hypotheses,
                "validation": validation_status(hypotheses, args),
            }
        )
    mesh_archive = args.output_dir / "validated_contact_depth_candidate_meshes_world.npz"
    save_mesh_archive(mesh_archive, saved_frames, saved_vertices, saved_faces)
    for report in reports:
        report["validated_contact_depth_mesh_archive"] = str(mesh_archive)
    report_payload = {
        "status": "ok",
        "method": "validate_v17_object_depth_repair_temporal",
        "annotations": str(args.annotations),
        "sam2_object_mask_measurements": str(args.sam2_object_mask_measurements),
        "object_depth_repair_candidates": str(args.object_depth_repair_candidates),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_id": args.object_id,
        "validation_count": len(reports),
        "rows": reports,
    }
    write_json(args.output_dir / "object_depth_repair_temporal_validation.json", reports)
    write_json(args.output_dir / "object_depth_repair_temporal_validation_summary.json", report_payload)
    print(json.dumps({k: v for k, v in report_payload.items() if k != "rows"}, indent=2))
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sam2-object-mask-measurements", type=Path, required=True)
    parser.add_argument("--object-depth-repair-candidates", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-radius", type=int, default=6)
    parser.add_argument("--image-near-px", type=float, default=12.0)
    parser.add_argument("--projection-dilate-px", type=int, default=4)
    parser.add_argument("--min-temporal-frames", type=int, default=6)
    parser.add_argument("--min-camera-translation-m", type=float, default=0.01)
    parser.add_argument("--min-base-target-mask-iou", type=float, default=0.35)
    parser.add_argument("--max-contact-median-mask-distance-px", type=float, default=20.0)
    parser.add_argument("--min-projected-inside-fraction", type=float, default=0.25)
    parser.add_argument("--min-mask-distance-improvement-px", type=float, default=1.0)
    parser.add_argument("--mask-stride", type=int, default=2)
    parser.add_argument("--mask-erode-px", type=int, default=0)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=3.20)
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--min-faces", type=int, default=100)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
