#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    visible = load_json(args.visible_geometry_annotations)
    state = load_json(args.v21_state)
    object_plan = load_json(args.object_plan)
    frame_rows = []
    depths = []
    areas = []
    vertex_counts = []
    bbox_widths = []
    bbox_heights = []
    for frame in visible.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if not isinstance(obj, dict):
                continue
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            if geom.get("status") != "visible_surface_from_sam2_mask_metric_depth":
                continue
            bbox = obj.get("bbox_xyxy") if isinstance(obj.get("bbox_xyxy"), list) else None
            bbox_wh = None
            if bbox and len(bbox) == 4:
                bbox_wh = [float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])]
                bbox_widths.append(bbox_wh[0])
                bbox_heights.append(bbox_wh[1])
            depth = obj.get("depth_m")
            if depth is not None:
                depths.append(float(depth))
            area = obj.get("area_px")
            if area is not None:
                areas.append(float(area))
            vertex_count = int(geom.get("vertex_count", 0))
            vertex_counts.append(float(vertex_count))
            frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "object_id": obj.get("object_id"),
                    "track_id": obj.get("track_id"),
                    "mask_path": obj.get("mask_path"),
                    "bbox_xyxy": bbox,
                    "bbox_wh_px": bbox_wh,
                    "depth_m": depth,
                    "vertex_count": vertex_count,
                    "visible_geometry_status": geom.get("status"),
                    "intrinsics_fx_fy_cx_cy": geom.get("intrinsics_fx_fy_cx_cy"),
                }
            )
    if not frame_rows:
        raise ContractError("visible_geometry_has_no_metric_surface_rows")
    packet = {
        "schema": "v21_geometry_condition_packet.v0",
        "status": "ok",
        "method": "build_v21_geometry_condition_packet",
        "case_id": state.get("case_id"),
        "v21_state": str(args.v21_state),
        "object_plan": str(args.object_plan),
        "visible_geometry_annotations": str(args.visible_geometry_annotations),
        "target_object_plan": object_plan,
        "support_summary": {
            "visible_metric_frame_count": int(len(frame_rows)),
            "depth_m": summarize(depths),
            "area_px": summarize(areas),
            "vertex_count": summarize(vertex_counts),
            "bbox_width_px": summarize(bbox_widths),
            "bbox_height_px": summarize(bbox_heights),
        },
        "mesh_candidate_policy": {
            "accepted_mesh_available": False,
            "required_next_mechanism": "run_or_import_completed_object_mesh_candidate_then_validate_against_visible_surface_depth_silhouette_free_space",
            "forbidden_substitutes": ["centroid", "sphere", "bbox", "visible_surface_scatter_as_complete_mesh"],
        },
        "conditioning_frames": frame_rows[:: max(1, len(frame_rows) // int(args.max_conditioning_rows))][: int(args.max_conditioning_rows)],
        "claim_scope": "Condition packet for object mesh generation/selection. It summarizes accepted visible masks and DepthPro surfels; it is not object mesh reconstruction or object pose.",
    }
    write_json(args.output_packet, packet)
    print(json.dumps(packet, indent=2, ensure_ascii=False)[:5000])
    return packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V21 object geometry condition packet from visible metric surfaces.")
    parser.add_argument("--v21-state", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--visible-geometry-annotations", type=Path, required=True)
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--max-conditioning-rows", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
