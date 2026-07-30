#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v20_common import ContractError, load_json, numeric_summary, write_json


def hand_vertices(hand: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else hand
    for key in ("vertices_current_v18_world_m", "vertices_world", "optimized_vertices_world_sample_m", "vertices_camera"):
        arr = np.asarray(metric.get(key) if metric.get(key) is not None else [], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0 and np.isfinite(arr).all():
            return arr, key
    return None, None


def object_points(obj: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    for key in ("world_vertices_sample_m", "camera_vertices_sample_m"):
        arr = np.asarray(geom.get(key) if geom.get(key) is not None else [], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0 and np.isfinite(arr).all():
            return arr, f"visible_geometry_candidate.{key}"
    return None, None


def contact_mode(hand: dict[str, Any], obj: dict[str, Any], distance_m: float, args: argparse.Namespace) -> str:
    for source in (hand, obj):
        raw = source.get("contact_state") or source.get("contact_mode") or source.get("physical_contact_mode")
        if isinstance(raw, str) and raw:
            return raw
    if distance_m <= float(args.contact_band_m):
        return "supported_near_noncontact"
    if distance_m <= float(args.unresolved_band_m):
        return "depth_occluded_contact_possible"
    return "no_rendered_contact_point"


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    rows = []
    skipped = []
    for frame in annotations.get("frames", []) if isinstance(annotations, dict) else []:
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        for hand in hands:
            h_vertices, h_source = hand_vertices(hand)
            if h_vertices is None:
                skipped.append({"frame_idx": frame_idx, "reason": "hand_has_no_surface_vertices"})
                continue
            hand_side = str(hand.get("side") or hand.get("hand_side") or "unknown")
            for obj in objects:
                object_id = str(obj.get("object_id") or obj.get("track_id") or "object:unknown")
                o_points, o_source = object_points(obj)
                if o_points is None:
                    skipped.append({"frame_idx": frame_idx, "object_id": object_id, "reason": "object_has_no_visible_surface_points"})
                    continue
                diff = h_vertices[:, None, :] - o_points[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                indices = np.argmin(dist2, axis=1)
                distances = np.sqrt(dist2[np.arange(len(h_vertices)), indices])
                best_i = int(np.argmin(distances))
                distance = float(distances[best_i])
                mode = contact_mode(hand, obj, distance, args)
                if mode == "no_rendered_contact_point" and not args.render_unresolved:
                    continue
                h_point = h_vertices[best_i]
                o_point = o_points[int(indices[best_i])]
                render_point = 0.5 * (h_point + o_point)
                rows.append({
                    "frame_idx": frame_idx,
                    "hand_side": hand_side,
                    "object_id": object_id,
                    "render_point_world_m": render_point.astype(float).tolist(),
                    "hand_surface_point_world_m": h_point.astype(float).tolist(),
                    "object_surface_point_world_m": o_point.astype(float).tolist(),
                    "surface_distance_m": distance,
                    "render_point_source": "center_of_local_hand_object_interface_from_existing_surfaces",
                    "input_contact_mode": mode,
                    "source_surfaces": [h_source, o_source],
                    "uncertainty_radius_m": max(float(args.min_uncertainty_radius_m), distance),
                    "evidence_created": False,
                })
    payload = {
        "schema": "v20_contact_point_render_rows.v0",
        "claim_scope": "Render-only contact/near-contact markers from existing hand/object surfaces. These rows must not enter contact evidence, ownership, or solver factors.",
        "evidence_created": False,
        "row_count": len(rows),
        "distance_summary_m": numeric_summary([row["surface_distance_m"] for row in rows]),
        "rows": rows,
        "skipped_preview": skipped[:100],
    }
    write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V20 render-only contact point rows from existing non-GT hand/object surface state.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=10**9)
    parser.add_argument("--contact-band-m", type=float, default=0.012)
    parser.add_argument("--unresolved-band-m", type=float, default=0.040)
    parser.add_argument("--min-uncertainty-radius-m", type=float, default=0.006)
    parser.add_argument("--render-unresolved", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args())["row_count"])
