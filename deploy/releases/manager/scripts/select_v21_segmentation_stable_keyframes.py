#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
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


def resolve_path(run_root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    candidates = [path]
    text = str(raw)
    if text.startswith("outputs/"):
        candidates.append(Path("output") / Path(text).relative_to("outputs"))
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        candidates.append(Path("output") / Path(text[len(historical_prefix) :]))
    if not path.is_absolute():
        candidates.append(run_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def require_path(run_root: Path, raw: str | Path | None, label: str) -> Path:
    path = resolve_path(run_root, raw)
    if path is None or not path.exists():
        raise ContractError(f"missing_{label}: {raw}")
    return path


def object_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
    rows = root.get("objects") if isinstance(root, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_objects")
    return [row for row in rows if isinstance(row, dict)]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def cv(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    mean = statistics.fmean(vals)
    if abs(mean) < 1.0e-9:
        return float("inf")
    return float(statistics.pstdev(vals) / abs(mean))


def median(values: list[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else default


def load_manifest(path: Path | None) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {}
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return {}, payload
    by_idx: dict[int, dict[str, Any]] = {}
    for ordinal, row in enumerate(frames):
        if not isinstance(row, dict):
            continue
        frame_idx = int(row.get("frame_idx", row.get("index", ordinal)))
        by_idx[frame_idx] = row
    return by_idx, payload


def source_dimensions(source_manifest: dict[str, Any], raw_manifest: dict[str, Any], track: dict[int, dict[str, Any]]) -> tuple[int, int]:
    for payload in (source_manifest, raw_manifest):
        video = payload.get("video") if isinstance(payload, dict) else None
        if isinstance(video, dict):
            width = int(video.get("width") or video.get("source_width") or 0)
            height = int(video.get("height") or video.get("source_height") or 0)
            if width > 0 and height > 0:
                return width, height
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if isinstance(frames, list) and frames:
            row = next((item for item in frames if isinstance(item, dict)), None)
            if row:
                width = int(row.get("source_width") or row.get("manifest_width") or 0)
                height = int(row.get("source_height") or row.get("manifest_height") or 0)
                if width > 0 and height > 0:
                    return width, height
    for row in track.values():
        bbox = row.get("bbox_xyxy") if isinstance(row, dict) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            return int(math.ceil(max(float(bbox[2]), 1.0))), int(math.ceil(max(float(bbox[3]), 1.0)))
    raise ContractError("cannot_infer_source_dimensions")


def load_track(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        out[int(key)] = value
    if not out:
        raise ContractError(f"sam2_track_has_no_frame_rows: {path}")
    return out


def load_hand_index(path: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    if path is None or not path.exists():
        return {"by_frame": {}, "by_local": {}}
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return {"by_frame": {}, "by_local": {}}
    by_frame: dict[int, dict[str, Any]] = {}
    by_local: dict[int, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict):
            continue
        if row.get("frame_idx") is not None:
            by_frame[int(row["frame_idx"])] = row
        if row.get("local_frame_idx") is not None:
            by_local[int(row["local_frame_idx"])] = row
    return {"by_frame": by_frame, "by_local": by_local}


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_diag(bbox: list[float]) -> float:
    return math.hypot(max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1])))


def bbox_distance(a: list[float], b: list[float]) -> float:
    dx = max(float(b[0]) - float(a[2]), float(a[0]) - float(b[2]), 0.0)
    dy = max(float(b[1]) - float(a[3]), float(a[1]) - float(b[3]), 0.0)
    return float(math.hypot(dx, dy))


def bbox_intersection(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def mask_component_stats(mask_path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if mask_path is None or not mask_path.exists():
        return {
            "mask_readable": False,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "mask_area_px": 0,
            "mask_width": 0,
            "mask_height": 0,
            "largest_component_label": None,
        }
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            "mask_readable": False,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "mask_area_px": 0,
            "mask_width": 0,
            "mask_height": 0,
            "largest_component_label": None,
        }
    mask = (image > 0).astype(np.uint8)
    total = int(mask.sum())
    if total <= 0:
        return {
            "mask_readable": True,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "mask_area_px": 0,
            "mask_width": int(mask.shape[1]),
            "mask_height": int(mask.shape[0]),
            "largest_component_label": None,
        }
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    min_area = max(int(args.min_component_area_px), int(round(float(args.component_min_fraction) * float(total))))
    kept = [(label, area) for label, area in enumerate(component_areas, start=1) if area >= min_area]
    largest_label = max(range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA])) if count > 1 else None
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA]) if largest_label is not None else 0
    return {
        "mask_readable": True,
        "component_count": int(len(kept)),
        "largest_component_fraction": float(largest_area / max(1, total)),
        "mask_area_px": int(total),
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
        "largest_component_label": int(largest_label) if largest_label is not None else None,
    }


def hand_candidates_for_frame(
    frame_idx: int,
    raw_rows: dict[int, dict[str, Any]],
    hand_index: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[int] = [int(frame_idx)]
    raw_row = raw_rows.get(int(frame_idx))
    if raw_row and raw_row.get("source_frame_idx") is not None:
        candidates.append(int(raw_row["source_frame_idx"]))
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        for table_name in ("by_local", "by_frame"):
            table = hand_index.get(table_name, {})
            row = table.get(candidate)
            if row is not None and id(row) not in seen:
                out.append(row)
                seen.add(id(row))
    return out


def resolve_mask_path(run_root: Path, row: dict[str, Any], args: argparse.Namespace) -> Path | None:
    raw = row.get("mask_path")
    path = resolve_path(run_root, raw)
    if path is not None and path.exists():
        return path
    if args.mask_root and raw:
        root = resolve_path(run_root, args.mask_root)
        if root is not None:
            candidate = root / Path(str(raw)).name
            if candidate.exists():
                return candidate
    return path


def build_frame_metrics(
    run_root: Path,
    track: dict[int, dict[str, Any]],
    raw_rows: dict[int, dict[str, Any]],
    hand_index: dict[str, dict[int, dict[str, Any]]],
    source_size: tuple[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for frame_idx in range(min(track), max(track) + 1):
        row = track.get(frame_idx, {})
        visible = bool(row.get("visible")) and isinstance(row.get("bbox_xyxy"), list)
        source_frame_idx = raw_rows.get(frame_idx, {}).get("source_frame_idx", frame_idx)
        metric: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "source_frame_idx": int(source_frame_idx),
            "visible": bool(visible),
            "interaction_class": "no_interaction",
            "class_reason": "object_not_visible" if not visible else "unclassified",
        }
        if not visible:
            metrics.append(metric)
            continue
        bbox = [float(v) for v in row["bbox_xyxy"]]
        area = safe_float(row.get("area_px"), 0.0)
        box_area = bbox_area(bbox)
        mask_path = resolve_mask_path(run_root, row, args)
        component = mask_component_stats(mask_path, args)
        hands = hand_candidates_for_frame(frame_idx, raw_rows, hand_index)
        hand_distances: list[float] = []
        hand_overlaps: list[float] = []
        for hand_frame in hands:
            for hand in hand_frame.get("hands", []) if isinstance(hand_frame.get("hands"), list) else []:
                hb = hand.get("bbox_xyxy") if isinstance(hand, dict) else None
                if not isinstance(hb, list) or len(hb) != 4:
                    continue
                hand_bbox = [float(v) for v in hb]
                hand_distances.append(bbox_distance(bbox, hand_bbox))
                hand_overlaps.append(bbox_intersection(bbox, hand_bbox) / max(1.0, box_area))
        metric.update(
            {
                "bbox_xyxy": bbox,
                "bbox_width_px": float(max(0.0, bbox[2] - bbox[0])),
                "bbox_height_px": float(max(0.0, bbox[3] - bbox[1])),
                "bbox_area_px": float(box_area),
                "bbox_diag_px": float(bbox_diag(bbox)),
                "center_xy": row.get("center_xy"),
                "mask_area_px": float(area),
                "mask_pixel_count_inside_bbox": float(area),
                "fill_ratio": float(area / max(1.0, box_area)),
                "aspect_ratio": float((bbox[2] - bbox[0]) / max(1.0, bbox[3] - bbox[1])),
                "mask_path": str(mask_path) if mask_path else None,
                "hand_count": int(sum(len(hf.get("hands", [])) for hf in hands)),
                "min_hand_object_distance_px": min(hand_distances) if hand_distances else None,
                "max_hand_object_overlap_fraction": max(hand_overlaps) if hand_overlaps else 0.0,
                "source_width": int(source_size[0]),
                "source_height": int(source_size[1]),
            }
        )
        metric.update(component)
        metrics.append(metric)
    return metrics


def local_median(metrics: list[dict[str, Any]], center_pos: int, key: str, radius: int, side: str) -> float | None:
    if side == "prev":
        lo, hi = max(0, center_pos - radius), center_pos
    else:
        lo, hi = center_pos + 1, min(len(metrics), center_pos + 1 + radius)
    vals: list[float] = []
    for metric in metrics[lo:hi]:
        value = metric.get(key)
        if value is None:
            continue
        value_f = safe_float(value)
        if math.isfinite(value_f):
            vals.append(value_f)
    return median(vals, None) if vals else None


def classify_interactions(metrics: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for pos, metric in enumerate(metrics):
        if not metric.get("visible"):
            metric["interaction_class"] = "no_interaction"
            metric["class_reason"] = "object_not_visible"
            continue
        diag = safe_float(metric.get("bbox_diag_px"), 0.0)
        near = max(float(args.near_px), float(args.near_bbox_diag_fraction) * diag)
        far = max(float(args.far_px), float(args.far_near_multiplier) * near)
        distance = metric.get("min_hand_object_distance_px")
        overlap = safe_float(metric.get("max_hand_object_overlap_fraction"), 0.0)
        if distance is not None:
            distance_f = safe_float(distance)
        else:
            distance_f = float("nan")
        if overlap >= float(args.manipulation_overlap_fraction) or (math.isfinite(distance_f) and distance_f <= near):
            metric["interaction_class"] = "manipulating_object"
            metric["class_reason"] = "hand_object_overlap_or_near"
            continue
        prev_dist = local_median(metrics, pos, "min_hand_object_distance_px", int(args.approach_trend_window), "prev")
        next_dist = local_median(metrics, pos, "min_hand_object_distance_px", int(args.approach_trend_window), "next")
        trend_px = (prev_dist - next_dist) if prev_dist is not None and next_dist is not None else 0.0
        if math.isfinite(distance_f) and (distance_f <= far or trend_px >= float(args.approach_trend_px)):
            metric["interaction_class"] = "approaching_object"
            metric["class_reason"] = "hand_distance_decreasing_or_within_far_threshold"
            metric["hand_distance_trend_px"] = float(trend_px)
            continue
        prev_area = local_median(metrics, pos, "mask_area_px", int(args.approach_trend_window), "prev")
        next_area = local_median(metrics, pos, "mask_area_px", int(args.approach_trend_window), "next")
        area_growth = 0.0
        if prev_area is not None and next_area is not None and prev_area > 1.0:
            area_growth = float((next_area - prev_area) / prev_area)
        if area_growth >= float(args.approach_area_growth_fraction):
            metric["interaction_class"] = "approaching_object"
            metric["class_reason"] = "object_mask_area_growing_without_hand_contact"
            metric["area_growth_fraction"] = area_growth
            continue
        metric["interaction_class"] = "no_interaction"
        metric["class_reason"] = "hand_far_and_mask_area_not_approaching"


def smooth_interaction_classes(metrics: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not metrics:
        return
    radius = max(0, int(args.class_smoothing_window) // 2)
    original = [str(row.get("interaction_class", "no_interaction")) for row in metrics]
    priority = {"manipulating_object": 3, "approaching_object": 2, "no_interaction": 1}
    smoothed: list[str] = []
    for pos in range(len(metrics)):
        lo = max(0, pos - radius)
        hi = min(len(metrics), pos + radius + 1)
        counts = Counter(original[lo:hi])
        chosen = max(counts.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]
        smoothed.append(chosen)
    for _ in range(4):
        changed = False
        segments = segments_from_classes(smoothed)
        for seg_idx, (start, end, cls) in enumerate(segments):
            length = end - start + 1
            if length >= int(args.min_class_segment_frames):
                continue
            left = segments[seg_idx - 1] if seg_idx > 0 else None
            right = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None
            choices = [item for item in [left, right] if item is not None]
            if not choices:
                continue
            replacement = max(choices, key=lambda item: (item[1] - item[0] + 1, priority.get(item[2], 0)))[2]
            for pos in range(start, end + 1):
                smoothed[pos] = replacement
            changed = True
        if not changed:
            break
    for metric, old, new in zip(metrics, original, smoothed):
        metric["raw_interaction_class"] = old
        metric["interaction_class"] = new
        if old != new:
            metric["class_reason"] = f"temporal_segment_smoothing_from_{old}"


def segments_from_classes(classes: list[str]) -> list[tuple[int, int, str]]:
    if not classes:
        return []
    out: list[tuple[int, int, str]] = []
    start_pos = 0
    current = classes[0]
    for pos, cls in enumerate(classes[1:], start=1):
        if cls != current:
            out.append((start_pos, pos - 1, current))
            start_pos = pos
            current = cls
    out.append((start_pos, len(classes) - 1, current))
    return out


def contiguous_segments(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classes = [str(row.get("interaction_class", "no_interaction")) for row in metrics]
    return [
        {"start_pos": start, "end_pos": end, "interaction_class": cls}
        for start, end, cls in segments_from_classes(classes)
    ]


def evaluate_segment(metrics: list[dict[str, Any]], start_pos: int, end_pos: int, interaction_class: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = metrics[start_pos : end_pos + 1]
    visible_rows = [row for row in rows if row.get("visible")]
    length = int(end_pos - start_pos + 1)
    visible_fraction = float(len(visible_rows) / max(1, length))
    frame_start = int(rows[0]["frame_idx"])
    frame_end = int(rows[-1]["frame_idx"])
    blockers: list[str] = []
    uncertainty_flags: list[str] = []
    if length < int(args.min_segment_frames):
        blockers.append("segment_too_short")
    if visible_fraction < float(args.min_visible_fraction):
        blockers.append("visible_fraction_below_threshold")
    if not visible_rows:
        blockers.append("no_visible_mask_rows")
    values = {
        "mask_area_px": [safe_float(row.get("mask_area_px")) for row in visible_rows],
        "bbox_width_px": [safe_float(row.get("bbox_width_px")) for row in visible_rows],
        "bbox_height_px": [safe_float(row.get("bbox_height_px")) for row in visible_rows],
        "fill_ratio": [safe_float(row.get("fill_ratio")) for row in visible_rows],
        "aspect_ratio": [safe_float(row.get("aspect_ratio")) for row in visible_rows],
        "largest_component_fraction": [safe_float(row.get("largest_component_fraction")) for row in visible_rows],
    }
    component_counts = [int(row.get("component_count", 0)) for row in visible_rows]
    count_mode = None
    count_mode_fraction = 0.0
    if component_counts:
        counter = Counter(component_counts)
        count_mode, count_mode_n = counter.most_common(1)[0]
        count_mode_fraction = float(count_mode_n / max(1, len(component_counts)))
    stats = {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "mid_frame": int(round((frame_start + frame_end) / 2.0)),
        "source_frame_start": int(rows[0].get("source_frame_idx", frame_start)),
        "source_frame_end": int(rows[-1].get("source_frame_idx", frame_end)),
        "interaction_class": interaction_class,
        "length_frames": length,
        "visible_frames": int(len(visible_rows)),
        "visible_fraction": visible_fraction,
        "mask_area_median_px": median(values["mask_area_px"], 0.0),
        "mask_area_cv": cv(values["mask_area_px"]),
        "bbox_width_cv": cv(values["bbox_width_px"]),
        "bbox_height_cv": cv(values["bbox_height_px"]),
        "fill_ratio_cv": cv(values["fill_ratio"]),
        "aspect_ratio_cv": cv(values["aspect_ratio"]),
        "object_count_mode": count_mode,
        "object_count_mode_fraction": count_mode_fraction,
        "largest_component_fraction_median": median(values["largest_component_fraction"], 0.0),
    }
    if stats["mask_area_median_px"] < float(args.min_mask_area_px):
        blockers.append("mask_area_below_threshold")
    if stats["mask_area_cv"] > float(args.max_mask_area_cv):
        blockers.append("mask_area_unstable")
    if max(stats["bbox_width_cv"], stats["bbox_height_cv"]) > float(args.max_bbox_dim_cv):
        blockers.append("bbox_shape_unstable")
    if stats["fill_ratio_cv"] > float(args.max_fill_ratio_cv):
        blockers.append("mask_fill_shape_unstable")
    if stats["aspect_ratio_cv"] > float(args.max_aspect_ratio_cv):
        blockers.append("bbox_aspect_unstable")
    if count_mode is None or int(count_mode) <= 0 or count_mode_fraction < float(args.min_object_count_mode_fraction):
        blockers.append("object_count_not_stable")
    if stats["largest_component_fraction_median"] < float(args.min_largest_component_fraction):
        if (
            bool(args.allow_stable_multicomponent_masks)
            and count_mode is not None
            and int(count_mode) > 1
            and count_mode_fraction >= float(args.min_object_count_mode_fraction)
        ):
            uncertainty_flags.append("stable_multicomponent_mask_largest_component_not_dominant")
        else:
            blockers.append("largest_component_not_dominant")
    stability_score = float(
        stats["mask_area_cv"]
        + stats["bbox_width_cv"]
        + stats["bbox_height_cv"]
        + stats["fill_ratio_cv"]
        + stats["aspect_ratio_cv"]
        + max(0.0, 1.0 - stats["object_count_mode_fraction"])
    )
    stats["stable"] = not blockers
    stats["stability_score"] = stability_score
    stats["blockers"] = blockers
    stats["uncertainty_flags"] = uncertainty_flags
    return stats


def stable_candidates_for_class_segment(metrics: list[dict[str, Any]], segment: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    start = int(segment["start_pos"])
    end = int(segment["end_pos"])
    cls = str(segment["interaction_class"])
    whole = evaluate_segment(metrics, start, end, cls, args)
    candidates = [whole]
    if whole["stable"]:
        return candidates
    window = int(args.stable_window_frames)
    stride = int(args.stable_window_stride)
    if window <= 0 or stride <= 0 or end - start + 1 < int(args.min_segment_frames):
        return candidates
    pos = start
    while pos <= end:
        win_end = min(end, pos + window - 1)
        if win_end - pos + 1 >= int(args.min_segment_frames):
            candidates.append(evaluate_segment(metrics, pos, win_end, cls, args))
        if win_end == end:
            break
        pos += stride
    return candidates


def select_non_overlapping_stable(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    stable = [row for row in candidates if row.get("stable")]
    stable.sort(key=lambda row: (float(row.get("stability_score", 999.0)), -int(row.get("length_frames", 0))))
    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for row in stable:
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        if any(not (end < a or start > b) for a, b in occupied):
            continue
        selected.append(row)
        occupied.append((start, end))
        if len(selected) >= int(args.max_keyframes):
            break
    selected.sort(key=lambda row: int(row["frame_start"]))
    return selected


def update_object_plan(
    plan: dict[str, Any],
    track_id: str,
    selected: list[dict[str, Any]],
    track: dict[int, dict[str, Any]],
    run_root: Path,
    source_size: tuple[int, int],
    report_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    out["method"] = "segmentation_stable_interaction_segment_midpoint_keyframes"
    out["producer"] = "Pi/Codex segmentation-stable keyframe selector from SAM2 object masks and RTMLib hand proximity"
    out["reviewed_source_frames"] = [int(row["mid_frame"]) for row in selected]
    source_inputs = out.setdefault("source_inputs", {})
    if isinstance(source_inputs, dict):
        source_inputs["segmentation_stable_keyframe_selection"] = str(report_path)
        source_inputs["sam2_track_for_keyframe_selection"] = str(args.sam2_track)
        if args.rtmlib_hand2d:
            source_inputs["rtmlib_hand2d_for_interaction_segments"] = str(args.rtmlib_hand2d)
    out["keyframe_selection_policy"] = {
        "method": "interaction_class_segments_then_segmentation_stability_midpoints",
        "segment_classes": ["approaching_object", "manipulating_object", "no_interaction"],
        "selected_midpoint_frames": [int(row["mid_frame"]) for row in selected],
        "source_report": str(report_path),
    }
    rows = object_rows(out)
    target = None
    for row in rows:
        if str(row.get("track_id") or "") == track_id:
            target = row
            break
    if target is None:
        raise ContractError(f"track_id_not_found_in_object_plan: {track_id}")
    rois: list[dict[str, Any]] = []
    detector_keyframes: list[dict[str, Any]] = []
    for segment in selected:
        frame_idx = int(segment["mid_frame"])
        if frame_idx not in track or not track[frame_idx].get("visible"):
            raise ContractError(f"selected_midpoint_missing_visible_track_row: {frame_idx}")
        roi = {
            "frame_idx": frame_idx,
            "bbox_xyxy": [float(v) for v in track[frame_idx]["bbox_xyxy"]],
            "source": "segmentation_stable_segment_midpoint_track_bbox_for_owlv2_selection",
            "interaction_class": segment["interaction_class"],
        }
        rois.append(roi)
        detector_keyframes.append(
            {
                "frame_idx": frame_idx,
                "target_visible": True,
                "keyframe_source": "segmentation_stable_segment_midpoint",
                "coordinate_semantics": "source_rgb_pixel_coordinates",
                "interaction_class": segment["interaction_class"],
                "stable_segment": {
                    "frame_start": int(segment["frame_start"]),
                    "frame_end": int(segment["frame_end"]),
                    "stability_score": float(segment["stability_score"]),
                },
                "agent_keyframe_roi": roi,
            }
        )
    target.pop("point_prompts", None)
    target.pop("point_prompt_coordinate_semantics", None)
    target.pop("point_prompt_source_width", None)
    target.pop("point_prompt_source_height", None)
    target.pop("prompt_source", None)
    target["local_mask_rois"] = rois
    target["owlv2_detector_keyframes"] = detector_keyframes
    target["stable_keyframe_segments"] = selected
    target["keyframe_source_width"] = int(source_size[0])
    target["keyframe_source_height"] = int(source_size[1])
    target["measurement_policy"] = "Use stable-segmentation midpoint frames as OWLv2 detector keyframes; SAM2 is seeded only by approved OWLv2 bbox prompts."
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    sam2_track_path = require_path(run_root, args.sam2_track, "sam2_track")
    raw_path = resolve_path(run_root, args.raw_frame_manifest) if args.raw_frame_manifest else run_root / "input/raw_frame_manifest/manifest.json"
    source_path = resolve_path(run_root, args.source_frame_manifest) if args.source_frame_manifest else run_root / "input/source_frame_manifest/manifest.json"
    raw_rows, raw_manifest = load_manifest(raw_path)
    _, source_manifest = load_manifest(source_path)
    track = load_track(sam2_track_path)
    hand_path = resolve_path(run_root, args.rtmlib_hand2d) if args.rtmlib_hand2d else None
    hand_index = load_hand_index(hand_path)
    source_size = source_dimensions(source_manifest, raw_manifest, track)
    metrics = build_frame_metrics(run_root, track, raw_rows, hand_index, source_size, args)
    classify_interactions(metrics, args)
    smooth_interaction_classes(metrics, args)
    class_segments = contiguous_segments(metrics)
    candidates: list[dict[str, Any]] = []
    for segment in class_segments:
        candidates.extend(stable_candidates_for_class_segment(metrics, segment, args))
    selected = select_non_overlapping_stable(candidates, args)
    if not selected:
        report = {
            "schema": "v21_segmentation_stable_keyframes.v0",
            "status": "failed",
            "failure_reason": "no_stable_segmentation_segments",
            "run_root": str(run_root),
            "track_id": args.track_id,
            "sam2_track": str(sam2_track_path),
            "rtmlib_hand2d": str(hand_path) if hand_path else None,
            "mask_root_override": str(args.mask_root) if args.mask_root else None,
            "source_size": list(source_size),
            "thresholds": thresholds_payload(args),
            "class_segments": class_segments,
            "candidate_segments": candidates,
            "claim_scope": "Failed selection preserves evidence. No start/mid/end fallback was used.",
        }
        write_json(args.output, report)
        raise ContractError(f"no_stable_segmentation_segments: wrote {args.output}")
    selected_keyframes = [
        {
            "frame_idx": int(row["mid_frame"]),
            "interaction_class": row["interaction_class"],
            "segment_frame_start": int(row["frame_start"]),
            "segment_frame_end": int(row["frame_end"]),
            "stability_score": float(row["stability_score"]),
        }
        for row in selected
    ]
    report = {
        "schema": "v21_segmentation_stable_keyframes.v0",
        "status": "ok",
        "method": "segmentation_stable_interaction_segment_midpoint_keyframes",
        "run_root": str(run_root),
        "track_id": args.track_id,
        "sam2_track": str(sam2_track_path),
        "raw_frame_manifest": str(raw_path) if raw_path else None,
        "source_frame_manifest": str(source_path) if source_path else None,
        "rtmlib_hand2d": str(hand_path) if hand_path else None,
        "mask_root_override": str(args.mask_root) if args.mask_root else None,
        "source_size": list(source_size),
        "segment_classes": ["approaching_object", "manipulating_object", "no_interaction"],
        "thresholds": thresholds_payload(args),
        "class_segments": class_segments,
        "candidate_segments": candidates,
        "selected_segments": selected,
        "selected_keyframes": selected_keyframes,
        "frames_csv": ",".join(str(row["frame_idx"]) for row in selected_keyframes),
        "claim_scope": "Keyframes are midpoints of segmentation-stable interaction-class segments. They are prompt/detector keyframes only, not masks, geometry, object pose, or contact evidence.",
    }
    write_json(args.output, report)
    if args.object_plan and args.object_plan_output:
        plan_path = require_path(run_root, args.object_plan, "object_plan")
        plan = load_json(plan_path)
        updated = update_object_plan(plan, args.track_id, selected, track, run_root, source_size, Path(args.output), args)
        write_json(args.object_plan_output, updated)
        report["object_plan_output"] = str(args.object_plan_output)
        write_json(args.output, report)
    print(json.dumps({"status": "ok", "selected_keyframes": selected_keyframes, "output": str(args.output)}, indent=2))
    return report


def thresholds_payload(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "min_segment_frames",
        "stable_window_frames",
        "stable_window_stride",
        "min_visible_fraction",
        "min_mask_area_px",
        "max_mask_area_cv",
        "max_bbox_dim_cv",
        "max_fill_ratio_cv",
        "max_aspect_ratio_cv",
        "min_object_count_mode_fraction",
        "min_largest_component_fraction",
        "min_component_area_px",
        "component_min_fraction",
        "near_px",
        "near_bbox_diag_fraction",
        "far_px",
        "far_near_multiplier",
        "manipulation_overlap_fraction",
        "approach_trend_window",
        "approach_trend_px",
        "approach_area_growth_fraction",
        "max_keyframes",
        "class_smoothing_window",
        "min_class_segment_frames",
        "allow_stable_multicomponent_masks",
    ]
    return {key: getattr(args, key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select V21 keyframes from segmentation-stable interaction segments.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--sam2-track", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-frame-manifest", type=Path)
    parser.add_argument("--source-frame-manifest", type=Path)
    parser.add_argument("--rtmlib-hand2d", type=Path)
    parser.add_argument("--mask-root", type=Path, help="Optional replacement directory for mask_path basenames when a preserved track JSON points to moved masks.")
    parser.add_argument("--object-plan", type=Path)
    parser.add_argument("--object-plan-output", type=Path)
    parser.add_argument("--min-segment-frames", type=int, default=45)
    parser.add_argument("--stable-window-frames", type=int, default=120)
    parser.add_argument("--stable-window-stride", type=int, default=30)
    parser.add_argument("--min-visible-fraction", type=float, default=0.92)
    parser.add_argument("--min-mask-area-px", type=float, default=800.0)
    parser.add_argument("--max-mask-area-cv", type=float, default=0.18)
    parser.add_argument("--max-bbox-dim-cv", type=float, default=0.18)
    parser.add_argument("--max-fill-ratio-cv", type=float, default=0.12)
    parser.add_argument("--max-aspect-ratio-cv", type=float, default=0.12)
    parser.add_argument("--min-object-count-mode-fraction", type=float, default=0.95)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.95)
    parser.add_argument("--min-component-area-px", type=int, default=12)
    parser.add_argument("--component-min-fraction", type=float, default=0.01)
    parser.add_argument("--near-px", type=float, default=35.0)
    parser.add_argument("--near-bbox-diag-fraction", type=float, default=0.22)
    parser.add_argument("--far-px", type=float, default=120.0)
    parser.add_argument("--far-near-multiplier", type=float, default=3.0)
    parser.add_argument("--manipulation-overlap-fraction", type=float, default=0.03)
    parser.add_argument("--approach-trend-window", type=int, default=12)
    parser.add_argument("--approach-trend-px", type=float, default=25.0)
    parser.add_argument("--approach-area-growth-fraction", type=float, default=0.08)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--class-smoothing-window", type=int, default=15)
    parser.add_argument("--min-class-segment-frames", type=int, default=15)
    parser.add_argument("--allow-stable-multicomponent-masks", action="store_true", help="Allow a segment to seed prompts when the mask is consistently multi-component; records an uncertainty flag instead of treating largest-component dominance as a hard blocker.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
