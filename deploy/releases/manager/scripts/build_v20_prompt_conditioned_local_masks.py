#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, safe_id, write_json


GC_BG = cv2.GC_BGD
GC_FG = cv2.GC_FGD
GC_PR_BG = cv2.GC_PR_BGD
GC_PR_FG = cv2.GC_PR_FGD


def point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict) and "x" in point and "y" in point:
        return float(point["x"]), float(point["y"])
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise ContractError(f"invalid_point: {point}")


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def bbox_from_points(points: list[tuple[float, float]], margin_x: float, margin_y: float, width: int, height: int) -> list[int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return clamp_bbox([min(xs) - margin_x, min(ys) - margin_y, max(xs) + margin_x, max(ys) + margin_y], width, height)


def object_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_objects")
    return [row for row in rows if isinstance(row, dict)]


def prompt_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = row.get("point_prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ContractError(f"object_plan_row_missing_point_prompts: {row.get('object_id')}")
    by_frame = {int(p["frame_idx"]): dict(p) for p in prompts if isinstance(p, dict) and "frame_idx" in p}
    roi_rows = row.get("local_mask_rois")
    if isinstance(roi_rows, list):
        for roi in roi_rows:
            if not isinstance(roi, dict) or "frame_idx" not in roi or "bbox_xyxy" not in roi:
                continue
            frame_idx = int(roi["frame_idx"])
            if frame_idx in by_frame:
                by_frame[frame_idx]["bbox_xyxy"] = roi["bbox_xyxy"]
                by_frame[frame_idx]["local_mask_roi_source"] = roi.get("source", "object_plan_local_mask_roi")
    out = sorted(by_frame.values(), key=lambda p: int(p["frame_idx"]))
    if not out:
        raise ContractError(f"object_plan_row_has_no_valid_prompts: {row.get('object_id')}")
    return out


def normalized_points(points: list[tuple[float, float]], bbox: list[int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    return [((x - x1) / width, (y - y1) / height) for x, y in points]


def denormalize_points(points: list[tuple[float, float]], bbox: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    return [(x1 + nx * width, y1 + ny * height) for nx, ny in points]


def build_anchors(row: dict[str, Any], image_width: int, image_height: int, margin_x: float, margin_y: float) -> list[dict[str, Any]]:
    anchors = []
    for prompt in prompt_rows(row):
        positives = [point_xy(point) for point in prompt.get("positive_points", [])]
        negatives = [point_xy(point) for point in prompt.get("negative_points", [])]
        if not positives:
            continue
        if "bbox_xyxy" in prompt:
            bbox = clamp_bbox(prompt["bbox_xyxy"], image_width, image_height)
        else:
            bbox = bbox_from_points(positives, margin_x, margin_y, image_width, image_height)
        center = [float(np.mean([p[0] for p in positives])), float(np.mean([p[1] for p in positives]))]
        anchors.append({
            "frame_idx": int(prompt["frame_idx"]),
            "bbox_xyxy": bbox,
            "positive_norm": normalized_points(positives, bbox),
            "negative_norm": normalized_points(negatives, bbox),
            "center_norm": normalized_points([tuple(center)], bbox)[0],
            "target_visible": bool(prompt.get("target_visible", True)),
            "prompt_source": prompt.get("prompt_source", "object_plan_prompt"),
            "local_mask_roi_source": prompt.get("local_mask_roi_source", "derived_from_prompt_points"),
        })
    if not anchors:
        raise ContractError(f"object_plan_row_has_no_positive_prompt_anchors: {row.get('object_id')}")
    return sorted(anchors, key=lambda a: int(a["frame_idx"]))


def interpolate_list(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [(1.0 - alpha) * float(x) + alpha * float(y) for x, y in zip(a, b)]


def interpolate_points(a: list[tuple[float, float]], b: list[tuple[float, float]], alpha: float) -> list[tuple[float, float]]:
    count = min(len(a), len(b))
    if count == 0:
        return list(a or b)
    return [((1.0 - alpha) * a[i][0] + alpha * b[i][0], (1.0 - alpha) * a[i][1] + alpha * b[i][1]) for i in range(count)]


def interpolate_anchor(anchors: list[dict[str, Any]], frame_idx: int) -> dict[str, Any]:
    if frame_idx <= anchors[0]["frame_idx"]:
        base = dict(anchors[0])
        base["anchor_mode"] = "nearest_before_first"
        return base
    if frame_idx >= anchors[-1]["frame_idx"]:
        base = dict(anchors[-1])
        base["anchor_mode"] = "nearest_after_last"
        return base
    for left, right in zip(anchors[:-1], anchors[1:]):
        if left["frame_idx"] <= frame_idx <= right["frame_idx"]:
            span = max(1, int(right["frame_idx"]) - int(left["frame_idx"]))
            alpha = (frame_idx - int(left["frame_idx"])) / span
            bbox = interpolate_list(left["bbox_xyxy"], right["bbox_xyxy"], alpha)
            return {
                "frame_idx": frame_idx,
                "bbox_xyxy": bbox,
                "positive_norm": interpolate_points(left["positive_norm"], right["positive_norm"], alpha),
                "negative_norm": interpolate_points(left["negative_norm"], right["negative_norm"], alpha),
                "center_norm": interpolate_points([left["center_norm"]], [right["center_norm"]], alpha)[0],
                "target_visible": bool(left.get("target_visible", True) or right.get("target_visible", True)),
                "prompt_source": "interpolated_object_plan_prompts",
                "local_mask_roi_source": "interpolated_object_plan_local_mask_rois",
                "anchor_mode": "interpolated_between_prompts",
            }
    raise ContractError(f"could_not_interpolate_anchor_for_frame: {frame_idx}")


def active_intervals(row: dict[str, Any], frame_count: int) -> list[tuple[int, int]]:
    intervals = row.get("active_intervals")
    if isinstance(intervals, list) and intervals:
        out = []
        for item in intervals:
            if not isinstance(item, dict):
                continue
            start = int(item.get("start_frame", 0))
            end = int(item.get("end_frame", frame_count - 1))
            out.append((max(0, start), min(frame_count - 1, end)))
        return out or [(0, frame_count - 1)]
    return [(0, frame_count - 1)]


def is_active(frame_idx: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame_idx <= end for start, end in intervals)


def draw_seed_disks(gc_mask: np.ndarray, points: list[tuple[float, float]], value: int, radius: int) -> None:
    for x, y in points:
        cv2.circle(gc_mask, (int(round(x)), int(round(y))), int(radius), int(value), -1)


def component_touching_points(mask: np.ndarray, points: list[tuple[float, float]], max_dist_px: float) -> np.ndarray:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(bool)
    keep = np.zeros(num, dtype=bool)
    for label in range(1, num):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0] and labels[yi, xi] == label:
                keep[label] = True
                break
            cx, cy = centroids[label]
            if math.hypot(float(cx) - x, float(cy) - y) <= max_dist_px:
                keep[label] = True
                break
    if not keep.any():
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1 + int(np.argmax(areas))] = True
    return keep[labels]


def fit_mask(image: np.ndarray, anchor: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    bbox = clamp_bbox(anchor["bbox_xyxy"], width, height)
    x1, y1, x2, y2 = bbox
    roi_width = max(1, x2 - x1 + 1)
    roi_height = max(1, y2 - y1 + 1)
    full_mask = np.zeros((height, width), dtype=np.uint8)
    if roi_width < 8 or roi_height < 8:
        return full_mask.astype(bool), {"status": "roi_too_small", "bbox_xyxy": bbox}
    roi = image[y1 : y2 + 1, x1 : x2 + 1]
    gc_mask = np.full((roi_height, roi_width), GC_BG, dtype=np.uint8)
    inner = [max(1, int(0.08 * roi_width)), max(1, int(0.08 * roi_height)), max(2, int(0.84 * roi_width)), max(2, int(0.84 * roi_height))]
    gc_mask[inner[1] : inner[1] + inner[3], inner[0] : inner[0] + inner[2]] = GC_PR_FG
    positive_global = denormalize_points(anchor.get("positive_norm", []), bbox)
    negative_global = denormalize_points(anchor.get("negative_norm", []), bbox)
    positive_local = [(x - x1, y - y1) for x, y in positive_global]
    negative_local = [(x - x1, y - y1) for x, y in negative_global]
    draw_seed_disks(gc_mask, positive_local, GC_FG, args.fg_seed_radius)
    draw_seed_disks(gc_mask, negative_local, GC_BG, args.bg_seed_radius)
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, int(args.grabcut_iterations), cv2.GC_INIT_WITH_MASK)
        local = np.logical_or(gc_mask == GC_FG, gc_mask == GC_PR_FG)
        local = component_touching_points(local, positive_local, args.max_component_seed_distance_px)
    except cv2.error as exc:
        local = np.zeros((roi_height, roi_width), dtype=bool)
        report = {"status": "grabcut_failed", "error": str(exc), "bbox_xyxy": bbox}
        return local, report
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_u8 = cv2.morphologyEx(local.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    local_u8 = cv2.morphologyEx(local_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    area = int(local_u8.sum())
    full_mask[y1 : y2 + 1, x1 : x2 + 1] = local_u8 * 255
    return full_mask.astype(bool), {
        "status": "ok",
        "bbox_xyxy": bbox,
        "roi_area_px": int(roi_width * roi_height),
        "local_area_px": area,
        "positive_points": [[float(x), float(y)] for x, y in positive_global],
        "negative_points": [[float(x), float(y)] for x, y in negative_global],
        "anchor_mode": anchor.get("anchor_mode"),
    }


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def center_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.raw_frame_manifest)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("raw_frame_manifest_has_no_frames")
    first = cv2.imread(str(frames[0]["rgb"]), cv2.IMREAD_COLOR)
    if first is None:
        raise ContractError(f"could_not_read_first_frame: {frames[0]['rgb']}")
    image_height, image_width = first.shape[:2]
    payload = load_json(args.object_plan)
    ensure_no_gt_in_prediction(payload, "object_plan")
    rows = object_rows(payload)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_tracks = []
    for row in rows:
        object_id = str(row.get("object_id") or row.get("target_object_id") or row.get("model_object_id") or "")
        if not object_id:
            raise ContractError("object_plan_row_missing_object_id")
        track_id = safe_id(str(row.get("track_id") or object_id))
        anchors = build_anchors(row, image_width, image_height, args.roi_margin_x, args.roi_margin_y)
        intervals = active_intervals(row, len(frames))
        track_dir = args.output_root / track_id / "sam2"
        mask_dir = track_dir / "sam2_masks_filtered"
        mask_dir.mkdir(parents=True, exist_ok=True)
        track: dict[str, Any] = {}
        visible_count = 0
        areas = []
        for frame in frames:
            frame_idx = int(frame.get("frame_idx", frame.get("index", 0)))
            if not is_active(frame_idx, intervals):
                track[str(frame_idx)] = {"visible": False, "area_px": 0, "failure_reason": "outside_active_interval"}
                continue
            image = cv2.imread(str(frame["rgb"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError(f"could_not_read_frame: {frame['rgb']}")
            if image.shape[:2] != (image_height, image_width):
                raise ContractError("raw_manifest_frame_size_changed_unexpectedly")
            anchor = interpolate_anchor(anchors, frame_idx)
            mask, fit_report = fit_mask(image, anchor, args)
            area = int(mask.sum())
            max_area = int(args.max_area_frac * image_width * image_height)
            if area < int(args.min_area_px):
                visible = False
                reason = "local_mask_area_below_minimum"
            elif area > max_area:
                visible = False
                reason = "local_mask_area_above_cap"
            else:
                visible = True
                reason = None
            mask_path = None
            if visible:
                mask_path = mask_dir / f"{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                visible_count += 1
                areas.append(area)
            row_out = {
                "visible": visible,
                "area_px": float(area if visible else 0),
                "mask_path": str(mask_path) if mask_path else None,
                "bbox_xyxy": bbox_from_mask(mask) if visible else None,
                "center_xy": center_from_mask(mask) if visible else None,
                "source": "prompt_conditioned_local_grabcut_mask",
                "confidence": "weak_prompt_conditioned_measurement",
                "failure_reason": reason,
                "fit_report": fit_report,
                "claim_scope": "Local prompt-conditioned mask from object-plan points/ROI; weak segmentation evidence, not complete object geometry or pose.",
            }
            track[str(frame_idx)] = row_out
        track_path = track_dir / "sam2_track.json"
        write_json(track_path, track)
        summary_tracks.append({
            "object_id": object_id,
            "track_id": track_id,
            "output_track": str(track_path),
            "frames": len(frames),
            "visible_frames": visible_count,
            "median_area_px": float(np.median(areas)) if areas else None,
            "anchors": [{"frame_idx": a["frame_idx"], "bbox_xyxy": a["bbox_xyxy"]} for a in anchors],
        })
    summary = {
        "schema": "v20_prompt_conditioned_local_masks_report.v0",
        "status": "ok",
        "method": "build_v20_prompt_conditioned_local_masks",
        "object_plan": str(args.object_plan),
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "output_root": str(args.output_root),
        "tracks": summary_tracks,
        "claim_scope": "Fallback segmentation measurement used when SAM2 propagation contaminates background. It is prompt/ROI-conditioned weak evidence only.",
    }
    ensure_no_gt_in_prediction(summary, "prompt_conditioned_local_masks_report")
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt-conditioned local mask tracks for V20 infer when SAM2 propagation contaminates background.")
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--roi-margin-x", type=float, default=80.0)
    parser.add_argument("--roi-margin-y", type=float, default=80.0)
    parser.add_argument("--fg-seed-radius", type=int, default=8)
    parser.add_argument("--bg-seed-radius", type=int, default=10)
    parser.add_argument("--grabcut-iterations", type=int, default=4)
    parser.add_argument("--min-area-px", type=int, default=40)
    parser.add_argument("--max-area-frac", type=float, default=0.08)
    parser.add_argument("--max-component-seed-distance-px", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
