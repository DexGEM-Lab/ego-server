#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from build_v17_contact_measurements import mask_distance_map, project_world, resize_bool_mask
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


def required_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} field {field} must be a JSON integer, got {value!r}")
    return value


def frames_by_index(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations payload has no frames list")
    out: dict[int, dict[str, Any]] = {}
    for row_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"annotation frame row {row_i} is not a JSON object")
        out[required_int(frame.get("frame_idx"), "frame_idx", f"annotation frame row {row_i}")] = frame
    return out


def object_masks_by_frame(path: Path, object_id: str) -> dict[int, dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain a JSON list")
    out: dict[int, dict[str, Any]] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} row {row_i} is not a JSON object")
        if row.get("entity_id") != object_id or row.get("visible") is not True:
            continue
        mask_path = row.get("mask_path")
        if not isinstance(mask_path, str) or not mask_path:
            raise RuntimeError(f"{path} row {row_i} has no mask_path for visible {object_id}")
        out[required_int(row.get("frame_idx"), "frame_idx", f"{path} row {row_i}")] = row
    return out


def source_size_from_hand(hand: dict[str, Any]) -> tuple[int, int]:
    raw = hand.get("source_intrinsics")
    intr = np.asarray(raw, dtype=np.float64)
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError("repair hand has no valid source_intrinsics")
    return int(round(max(1.0, 2.0 * float(intr[2])))), int(round(max(1.0, 2.0 * float(intr[3]))))


def hand_contact_anchor(
    frame: dict[str, Any],
    mask_path: Path,
    image_near_px: float,
    min_near_vertices: int,
    required_hand_side: str | None,
    required_repair_candidate_id: str | None,
) -> dict[str, Any]:
    hands = frame.get("hands")
    if not isinstance(hands, list) or not hands:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no repaired hands")
    candidates: list[dict[str, Any]] = []
    for hand_i, hand in enumerate(hands):
        if not isinstance(hand, dict):
            raise RuntimeError(f"frame {frame.get('frame_idx')} hand row {hand_i} is not a JSON object")
        if required_hand_side is not None and hand.get("side") != required_hand_side:
            continue
        if required_repair_candidate_id is not None and hand.get("v17_repair_candidate_id") != required_repair_candidate_id:
            continue
        vertices = np.asarray(hand.get("vertices_world_m"), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            continue
        intr = np.asarray(hand.get("source_intrinsics"), dtype=np.float64)
        if intr.shape != (4,) or not np.isfinite(intr).all():
            continue
        source_size = source_size_from_hand(hand)
        mask = resize_bool_mask(mask_path, source_size)
        distance = mask_distance_map(mask)
        T_wc = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
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
        near = np.isfinite(d) & (d <= float(image_near_px))
        near_count = int(np.count_nonzero(near))
        if near_count < int(min_near_vertices):
            continue
        T_cw = np.linalg.inv(T_wc)
        vertices_camera = (np.c_[vertices, np.ones(len(vertices))] @ T_cw.T)[:, :3]
        near_depth = vertices_camera[near, 2]
        candidates.append(
            {
                "hand_index": hand_i,
                "hand_side": hand.get("side"),
                "hand_repair_candidate_id": hand.get("v17_repair_candidate_id"),
                "near_vertices": near_count,
                "image_distance_px_median": float(np.nanmedian(d[np.isfinite(d)])),
                "image_distance_px_min": float(np.nanmin(d)),
                "hand_depth_anchor_m": float(np.median(near_depth)),
                "hand_depth_p05_m": float(np.percentile(near_depth, 5.0)),
                "hand_depth_p95_m": float(np.percentile(near_depth, 95.0)),
            }
        )
    if not candidates:
        constraints = {
            "required_hand_side": required_hand_side,
            "required_repair_candidate_id": required_repair_candidate_id,
        }
        raise RuntimeError(
            f"frame {frame.get('frame_idx')} has no matching hand vertices within {image_near_px}px of {mask_path}: {constraints}"
        )
    return sorted(candidates, key=lambda row: (-int(row["near_vertices"]), float(row["image_distance_px_median"])))[0]


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = frames_by_index(load_json(args.annotations))
    object_masks = object_masks_by_frame(args.sam2_object_mask_measurements, args.object_id)
    depth = load_metric_depth(args.metric_depth_npz)
    frame_indices = sorted({int(part) for raw in args.frame_indices for part in raw.split(",") if part})
    frame_outputs: list[tuple[int, np.ndarray, np.ndarray]] = []
    annotation_frames: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for frame_idx in frame_indices:
        frame = annotations.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        object_row = object_masks.get(frame_idx)
        if object_row is None:
            raise RuntimeError(f"{args.object_id} has no visible object mask measurement at frame {frame_idx}")
        mask_path = Path(str(object_row["mask_path"]))
        if cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) is None:
            raise RuntimeError(f"failed to read object mask {mask_path}")
        anchor = hand_contact_anchor(
            frame,
            mask_path,
            float(args.image_near_px),
            int(args.min_near_vertices),
            args.required_hand_side,
            args.required_repair_candidate_id,
        )
        repair_frame = dict(frame)
        repair_frame["object"] = dict(frame.get("object", {}))
        repair_frame["object"]["label"] = args.object_id
        repair_frame["object"]["mask_path"] = str(mask_path)
        repair_frame["object"]["depth_m"] = float(anchor["hand_depth_anchor_m"])
        repair_frame["object"]["v17_object_depth_repair_state"] = "contact_depth_repair_candidate"
        vertices, faces, mesh_row = mesh_from_mask_depth(
            repair_frame,
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
            raise RuntimeError(f"frame {frame_idx} repair mesh failed: {mesh_row}")
        annotation_frames.append(repair_frame)
        frame_outputs.append((frame_idx, vertices, faces))
        rows.append(
            {
                "measurement_id": f"object_depth_repair_candidate:{frame_idx}:{slug(args.object_id)}",
                "frame_idx": frame_idx,
                "entity_type": "object",
                "entity_id": args.object_id,
                "measurement_type": "object_depth_repair_candidate",
                "source_model": "sam2_relative_depth_contact_anchor",
                "coordinate_frame": "v16_world_metric",
                "mask_path": str(mask_path),
                "anchor_source": "repaired_hand_mask_contact_depth_median",
                "validation_status": "requires_temporal_validation",
                **anchor,
                "raw_depth_median_m": mesh_row["depth_median_m"],
                "depth_scale_to_contact_anchor": float(anchor["hand_depth_anchor_m"]) / float(mesh_row["depth_median_m"]),
                "vertices": mesh_row["vertices"],
                "faces": mesh_row["faces"],
                "world_extent_m": mesh_row["world_extent_m"],
            }
        )
    archive = args.output_dir / "object_depth_repair_meshes_world.npz"
    save_mesh_archive(
        archive,
        [row[0] for row in frame_outputs],
        [row[1] for row in frame_outputs],
        [row[2] for row in frame_outputs],
    )
    for row in rows:
        row["mesh_archive"] = str(archive)
    repair_annotations = args.output_dir / "object_depth_repair_annotations.json"
    write_json(
        repair_annotations,
        {
            "status": "ok",
            "method": "build_v17_contact_depth_object_repair",
            "object_id": args.object_id,
            "mesh_archive": str(archive),
            "frames": annotation_frames,
        },
    )
    report = {
        "status": "ok",
        "method": "build_v17_contact_depth_object_repair",
        "annotations": str(args.annotations),
        "sam2_object_mask_measurements": str(args.sam2_object_mask_measurements),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_id": args.object_id,
        "mesh_archive": str(archive),
        "repair_annotations": str(repair_annotations),
        "candidate_count": len(rows),
        "rows": rows,
    }
    write_json(args.output_dir / "object_depth_repair_candidates.json", rows)
    write_json(args.output_dir / "object_depth_repair_summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sam2-object-mask-measurements", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--frame-indices", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-near-px", type=float, default=12.0)
    parser.add_argument("--min-near-vertices", type=int, default=25)
    parser.add_argument("--mask-stride", type=int, default=2)
    parser.add_argument("--mask-erode-px", type=int, default=0)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=3.20)
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--min-faces", type=int, default=100)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.06)
    parser.add_argument("--required-hand-side", choices=("left", "right"))
    parser.add_argument("--required-repair-candidate-id")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
