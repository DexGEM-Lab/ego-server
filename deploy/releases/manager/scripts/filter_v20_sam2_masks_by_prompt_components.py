#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, load_json, load_mask, write_json


def object_rows(plan_path: Path) -> list[dict[str, Any]]:
    payload = load_json(plan_path)
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_objects")
    return [row for row in rows if isinstance(row, dict)]


def prompt_points(row: dict[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    prompts = row.get("point_prompts") if isinstance(row.get("point_prompts"), list) else []
    for prompt in prompts:
        frame_idx = int(prompt.get("frame_idx", 0))
        positives = prompt.get("positive_points") if isinstance(prompt.get("positive_points"), list) else []
        pts = []
        for point in positives:
            if isinstance(point, dict) and "x" in point and "y" in point:
                pts.append([float(point["x"]), float(point["y"])])
        if pts:
            out[frame_idx] = np.asarray(pts, dtype=float)
    return out


def nearest_prompt(frame_idx: int, prompts: dict[int, np.ndarray]) -> np.ndarray | None:
    if not prompts:
        return None
    key = min(prompts, key=lambda k: abs(k - frame_idx))
    return prompts[key]


def component_filter(mask: np.ndarray, positive_xy: np.ndarray | None, max_area_frac: float, min_area_px: int, dilate_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    h, w = mask.shape[:2]
    max_area = max_area_frac * h * w
    keep = np.zeros(mask.shape, dtype=bool)
    components = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area:
            components.append({"label": label, "area": area, "kept": False, "reason": "area_filter"})
            continue
        comp = labels == label
        keep_component = False
        reason = "not_near_prompt"
        if positive_xy is None:
            keep_component = True
            reason = "no_prompt_keep_area_valid"
        else:
            for x, y in positive_xy:
                xi = int(round(x)); yi = int(round(y))
                if 0 <= xi < w and 0 <= yi < h and comp[yi, xi]:
                    keep_component = True
                    reason = "contains_positive_point"
                    break
            if not keep_component:
                cx, cy = centroids[label]
                dist = float(np.min(np.linalg.norm(positive_xy - np.asarray([cx, cy]), axis=1)))
                if dist <= max(60.0, 0.12 * max(h, w)):
                    keep_component = True
                    reason = f"near_positive_centroid_{dist:.1f}px"
        if keep_component:
            keep |= comp
        components.append({"label": label, "area": area, "kept": bool(keep_component), "reason": reason})
    if dilate_px > 0 and np.any(keep):
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return keep, {"component_count": int(n - 1), "kept_area": int(np.count_nonzero(keep)), "components_preview": components[:20]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for obj in object_rows(args.object_plan):
        object_id = str(obj.get("object_id") or obj.get("target_object_id") or obj.get("model_object_id"))
        track_id = str(obj.get("track_id") or object_id).replace(":", "_")
        src_track = args.sam2_root / track_id / "sam2" / "sam2_track.json"
        if not src_track.exists():
            raise ContractError(f"missing_sam2_track: {src_track}")
        track = load_json(src_track)
        prompts = prompt_points(obj)
        out_track = {}
        reports = []
        for key, value in track.items():
            if not isinstance(value, dict):
                continue
            frame_idx = int(key)
            mask_path = value.get("mask_path")
            if not mask_path:
                out_track[key] = value
                continue
            mask = load_mask(Path(mask_path))
            keep, report = component_filter(mask, nearest_prompt(frame_idx, prompts), float(args.max_area_frac), int(args.min_area_px), int(args.dilate_px))
            out_dir = args.output_root / track_id / "sam2" / "sam2_masks_filtered"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_mask = out_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(out_mask), keep.astype(np.uint8) * 255)
            new_value = dict(value)
            new_value["mask_path"] = str(out_mask)
            new_value["visible"] = bool(np.any(keep))
            new_value["filter_report"] = report
            out_track[key] = new_value
            reports.append({"frame_idx": frame_idx, **report})
        out_track_path = args.output_root / track_id / "sam2" / "sam2_track.json"
        write_json(out_track_path, out_track)
        rows.append({
            "object_id": object_id,
            "track_id": track_id,
            "source_track": str(src_track),
            "output_track": str(out_track_path),
            "frames": len(out_track),
            "visible_frames": sum(1 for v in out_track.values() if isinstance(v, dict) and v.get("visible")),
            "median_kept_area": float(np.median([r["kept_area"] for r in reports])) if reports else None,
        })
    summary = {"status": "ok", "method": "filter_v20_sam2_masks_by_prompt_components", "tracks": rows}
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SAM2 masks by connected components near object-plan positive prompt points.")
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--max-area-frac", type=float, default=0.18)
    parser.add_argument("--min-area-px", type=int, default=50)
    parser.add_argument("--dilate-px", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
