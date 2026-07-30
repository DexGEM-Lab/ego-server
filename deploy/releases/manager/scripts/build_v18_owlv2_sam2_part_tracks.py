#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore[import-not-found]


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_owlv2_sam2_part_tracks"
CLAIM = (
    "This artifact runs the V18 baseline object/part perception path for part-required objects: "
    "VLM physical notes -> OWLv2 keyframe part detections -> SAM2 prompted video tracking. "
    "Accepted tracks are semantic temporal mask evidence only; they are not geometry, pose, contact, or final readiness."
)

PHYSICAL_PART_TERMS = {
    "lid",
    "hinge",
    "hinged",
    "handle",
    "lever",
    "rim",
    "opening",
    "body",
    "edge",
    "flange",
    "panel",
    "top",
    "knob",
    "button",
    "spout",
    "cap",
}
STOPWORDS = {
    "the",
    "and",
    "with",
    "without",
    "only",
    "mainly",
    "itself",
    "does",
    "not",
    "change",
    "changes",
    "shape",
    "position",
    "relative",
    "rigid",
    "fixed",
    "visible",
    "manipulated",
    "part",
    "parts",
    "object",
    "whole",
    "some",
    "frames",
    "easier",
    "than",
    "isolating",
    "segmenting",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": percentile(values, 50.0),
        "p05": percentile(values, 5.0),
        "p95": percentile(values, 95.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "unnamed"


def normalize_label(text_label: str) -> str:
    text = text_label.lower().strip()
    for prefix in ("a close-up photo of the ", "a photo of a ", "a photo of an ", "a photo of the "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return slug(text)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def center_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def box_area(box: list[float] | list[int] | None) -> float:
    if box is None or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(a: list[float] | list[int] | None, b: list[float] | list[int] | None) -> float:
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def load_mask(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write mask {path}")


def crop_box_from_mask(mask: np.ndarray, image_w: int, image_h: int, pad_fraction: float) -> list[int]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return [0, 0, image_w, image_h]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return [
        max(0, int(round(x1 - bw * pad_fraction))),
        max(0, int(round(y1 - bh * pad_fraction))),
        min(image_w, int(round(x2 + bw * pad_fraction))),
        min(image_h, int(round(y2 + bh * pad_fraction))),
    ]


def quantile_select(rows: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: require_int(row.get("frame_idx"), "frame_idx"))
    if len(rows) <= max_count:
        return rows
    if max_count <= 1:
        return [rows[len(rows) // 2]]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for i in range(max_count):
        idx = int(round(i * (len(rows) - 1) / float(max_count - 1)))
        while idx in used and idx + 1 < len(rows):
            idx += 1
        used.add(idx)
        selected.append(rows[idx])
    return selected


def object_ids_requiring_parts(schema_rows: dict[str, dict[str, Any]]) -> set[str]:
    return {
        object_id
        for object_id, row in schema_rows.items()
        if row.get("requires_part_or_relative_motion_model") is True
    }


def annotation_object_rows(annotation: dict[str, Any], object_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        raw_frame_path = require_str(frame.get("raw_frame_path"), "raw_frame_path")
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            obj = require_dict(raw_obj, "frame object")
            object_id = str(obj.get("object_id"))
            if object_id not in object_ids:
                continue
            if obj.get("visibility_state") != "visible" or obj.get("renderable_mask") is not True:
                continue
            mask_path = obj.get("mask_path")
            if not isinstance(mask_path, str) or not Path(mask_path).exists():
                continue
            rows[object_id].append(
                {
                    "frame_idx": frame_idx,
                    "raw_frame_path": raw_frame_path,
                    "object_id": object_id,
                    "track_id": obj.get("track_id"),
                    "name": obj.get("name"),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "geometry_scope": obj.get("geometry_scope"),
                    "mask_path": mask_path,
                }
            )
    return rows


def object_mask_index(annotation: dict[str, Any], object_ids: set[str]) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            obj = require_dict(raw_obj, "frame object")
            object_id = str(obj.get("object_id"))
            if object_id not in object_ids:
                continue
            mask_path = obj.get("mask_path")
            if obj.get("renderable_mask") is True and isinstance(mask_path, str) and Path(mask_path).exists():
                out[(frame_idx, object_id)] = mask_path
    return out


def raw_video_dir(annotation: dict[str, Any]) -> Path:
    first = require_dict(require_list(annotation.get("frames"), "annotation frames")[0], "first frame")
    path = Path(require_str(first.get("raw_frame_path"), "raw_frame_path"))
    return path.parent


def schema_by_object(case: str, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = args.physical_state_schema_root / case / "v18_physical_state_schema_report.json"
    report = require_dict(load_json(path), f"{case} physical schema")
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("object_rows"), "schema object rows"):
        row = require_dict(raw, "schema row")
        out[require_str(row.get("object_id"), "object_id")] = row
    return out


def prompt_terms_for_object(schema_row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    text_parts = [str(schema_row.get("physical_notes") or "")]
    for key in ("part_or_relative_motion_evidence_terms", "primary_articulation_evidence_terms"):
        value = schema_row.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
    text = " ".join(text_parts).lower()
    words = re.findall(r"[a-z][a-z_/-]*", text)
    terms: list[str] = []
    for word in words:
        word = word.replace("/", " ").replace("_", "-").strip("- ")
        for piece in word.split():
            piece = piece.strip("- ")
            if not piece or piece in STOPWORDS:
                continue
            if piece in PHYSICAL_PART_TERMS:
                terms.append("hinge" if piece == "hinged" else piece)
    for item in args.extra_generic_part_terms:
        if item in text or item in {"part"}:
            terms.append(str(item))
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped[: int(args.max_prompt_terms_per_object)]


def text_prompts(terms: list[str]) -> list[str]:
    return [f"a photo of a {term}" for term in terms]


def run_owlv2_on_frame(
    processor: Any,
    model: Any,
    row: dict[str, Any],
    prompts: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image = Image.open(require_str(row.get("raw_frame_path"), "raw_frame_path")).convert("RGB")
    image_arr = np.asarray(image)
    object_mask = load_mask(Path(require_str(row.get("mask_path"), "mask_path")), image_arr.shape[:2])
    crop = crop_box_from_mask(object_mask, image.width, image.height, float(args.crop_pad_fraction))
    crop_img = image.crop((crop[0], crop[1], crop[2], crop[3]))
    inputs = processor(text=[prompts], images=crop_img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=float(args.owlv2_threshold),
        target_sizes=[(crop_img.height, crop_img.width)],
        text_labels=[prompts],
    )[0]
    detections: list[dict[str, Any]] = []
    crop_mask = object_mask[crop[1] : crop[3], crop[0] : crop[2]]
    object_area = int(crop_mask.sum())
    for box_tensor, score_tensor, text_label in zip(results["boxes"], results["scores"], results["text_labels"]):
        local_box = [float(v) for v in box_tensor.detach().cpu().tolist()]
        x1 = max(0, min(int(round(local_box[0])), crop_img.width))
        y1 = max(0, min(int(round(local_box[1])), crop_img.height))
        x2 = max(0, min(int(round(local_box[2])), crop_img.width))
        y2 = max(0, min(int(round(local_box[3])), crop_img.height))
        if x2 <= x1 or y2 <= y1:
            continue
        box_mask = np.zeros(crop_mask.shape, dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        box_area_px = int(box_mask.sum())
        overlap = int(np.logical_and(box_mask, crop_mask).sum())
        containment = overlap / float(box_area_px) if box_area_px else 0.0
        object_coverage = overlap / float(object_area) if object_area else 0.0
        state = "candidate_owlv2_part_box"
        if containment < float(args.min_box_containment):
            state = "rejected_box_not_contained_in_object"
        elif object_coverage < float(args.min_box_object_coverage):
            state = "rejected_box_too_small"
        elif object_coverage > float(args.max_box_object_coverage):
            state = "rejected_box_whole_object_like"
        full_box = [x1 + crop[0], y1 + crop[1], x2 + crop[0], y2 + crop[1]]
        detections.append(
            {
                "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
                "object_id": require_str(row.get("object_id"), "object_id"),
                "raw_frame_path": require_str(row.get("raw_frame_path"), "raw_frame_path"),
                "object_mask_path": require_str(row.get("mask_path"), "mask_path"),
                "text_label": str(text_label),
                "part_label": normalize_label(str(text_label)),
                "owlv2_score": float(score_tensor.detach().cpu().item()),
                "bbox_xyxy": full_box,
                "crop_xyxy": crop,
                "box_containment_in_object": float(containment),
                "box_object_coverage": float(object_coverage),
                "detection_state": state,
            }
        )
    return detections


def best_detections_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_label_frame: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("detection_state") != "candidate_owlv2_part_box":
            continue
        key = (require_str(row.get("part_label"), "part_label"), require_int(row.get("frame_idx"), "frame_idx"))
        current = by_label_frame.get(key)
        if current is None or finite_float(row.get("owlv2_score"), "score") > finite_float(current.get("owlv2_score"), "score"):
            by_label_frame[key] = row
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (label, _frame), row in by_label_frame.items():
        out[label].append(row)
    return {label: sorted(items, key=lambda row: require_int(row.get("frame_idx"), "frame_idx")) for label, items in out.items()}


def import_sam2(args: argparse.Namespace) -> Any:
    sam2_path = Path(args.sam2_repo)
    if str(sam2_path) not in sys.path:
        sys.path.insert(0, str(sam2_path))
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

    return build_sam2_video_predictor(args.sam2_model_cfg, str(args.sam2_checkpoint), device=args.device, vos_optimized=bool(args.vos_optimized))


def add_sam2_prompts(predictor: Any, state: Any, track_prompts: list[dict[str, Any]], obj_id_by_track: dict[str, int]) -> None:
    for track in track_prompts:
        obj_id = obj_id_by_track[require_str(track.get("track_key"), "track_key")]
        for prompt in require_list(track.get("prompt_detections"), "prompt detections"):
            box = np.asarray(prompt["bbox_xyxy"], dtype=np.float32)
            predictor.add_new_points_or_box(
                state,
                frame_idx=require_int(prompt.get("frame_idx"), "prompt frame_idx"),
                obj_id=obj_id,
                box=box,
            )


def mask_fraction_in_box(mask: np.ndarray, box: list[int] | list[float]) -> float:
    x1 = max(0, min(int(round(float(box[0]))), mask.shape[1]))
    y1 = max(0, min(int(round(float(box[1]))), mask.shape[0]))
    x2 = max(0, min(int(round(float(box[2]))), mask.shape[1]))
    y2 = max(0, min(int(round(float(box[3]))), mask.shape[0]))
    area = int(mask.sum())
    if area <= 0 or x2 <= x1 or y2 <= y1:
        return 0.0
    return int(mask[y1:y2, x1:x2].sum()) / float(area)


def propagate_tracks(
    predictor: Any,
    state: Any,
    track_prompts: list[dict[str, Any]],
    obj_id_by_track: dict[str, int],
    object_masks: dict[tuple[int, str], str],
    args: argparse.Namespace,
) -> dict[str, dict[int, np.ndarray]]:
    masks_by_track: dict[str, dict[int, np.ndarray]] = {track["track_key"]: {} for track in track_prompts}
    track_by_obj_id = {obj_id: key for key, obj_id in obj_id_by_track.items()}
    passes = [(False, None), (True, None)]
    for reverse, start in passes:
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
            if mask_logits is None:
                continue
            for i, obj_id in enumerate(obj_ids):
                key = track_by_obj_id.get(int(obj_id))
                if key is None:
                    continue
                mask = (mask_logits[i].detach().cpu().numpy() > 0.0)
                if mask.ndim == 3:
                    mask = mask[0]
                masks_by_track[key][int(frame_idx)] = mask.astype(bool)
    return masks_by_track


def materialize_track(
    case: str,
    track: dict[str, Any],
    frame_masks: dict[int, np.ndarray],
    object_masks: dict[tuple[int, str], str],
    output_root: Path,
    accepted: bool,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    object_id = require_str(track.get("object_id"), "track object_id")
    label = require_str(track.get("part_label"), "part_label")
    object_slug = slug(object_id.replace("object:", ""))
    track_label = f"owlv2_sam2_{object_slug}_{label}"
    root = output_root / case / ("accepted_tracks" if accepted else "rejected_tracks") / track_label / "sam2"
    mask_dir = root / "sam2_masks"
    payload: dict[str, Any] = {}
    containments: list[float] = []
    coverages: list[float] = []
    areas: list[float] = []
    prompt_box_ious: list[float] = []
    prompt_mask_in_box: list[float] = []
    prompt_by_frame = {require_int(row.get("frame_idx"), "prompt frame_idx"): row for row in require_list(track.get("prompt_detections"), "prompt detections")}
    for frame_idx in sorted(frame_masks):
        mask = frame_masks[frame_idx]
        object_mask_path = object_masks.get((frame_idx, object_id))
        visible = False
        row: dict[str, Any]
        if object_mask_path is not None:
            object_mask = load_mask(Path(object_mask_path), mask.shape)
            inter = int(np.logical_and(mask, object_mask).sum())
            area = int(mask.sum())
            object_area = int(object_mask.sum())
            containment = inter / float(area) if area > 0 else 0.0
            coverage = inter / float(object_area) if object_area > 0 else 0.0
            visible = (
                area >= int(args.min_saved_mask_area_px)
                and containment >= float(args.min_track_frame_containment)
                and coverage >= float(args.min_track_frame_object_coverage)
                and coverage <= float(args.max_track_frame_object_coverage)
            )
            bbox = bbox_from_mask(mask)
            center = center_from_mask(mask)
            if visible:
                containments.append(float(containment))
                coverages.append(float(coverage))
                areas.append(float(area))
                mask_path = mask_dir / f"{frame_idx:06d}.png"
                save_mask(mask_path, mask)
            else:
                mask_path = None
            prompt = prompt_by_frame.get(frame_idx)
            if prompt is not None:
                prompt_box = prompt["bbox_xyxy"]
                prompt_box_ious.append(box_iou(bbox, prompt_box))
                prompt_mask_in_box.append(mask_fraction_in_box(mask, prompt_box))
            row = {
                "visible": visible,
                "bbox_xyxy": bbox,
                "center_xy": center,
                "area_px": float(area),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "object_mask_path": object_mask_path,
                "part_containment_in_object": float(containment),
                "object_coverage_by_part": float(coverage),
            }
        else:
            row = {"visible": False, "bbox_xyxy": None, "center_xy": None, "area_px": 0.0, "mask_path": None, "object_mask_path": None}
        payload[str(frame_idx)] = row
    track_path = root / "sam2_track.json"
    write_json(track_path, payload)
    visible_frames = [int(k) for k, row in payload.items() if require_dict(row, "track frame").get("visible") is True]
    report = {
        "track_label": track_label,
        "object_id": object_id,
        "part_label": label,
        "accepted_as_semantic_temporal_part_track": accepted,
        "track_path": str(track_path),
        "prompt_detection_count": len(require_list(track.get("prompt_detections"), "prompt detections")),
        "prompt_frames": [require_int(row.get("frame_idx"), "prompt frame_idx") for row in require_list(track.get("prompt_detections"), "prompt detections")],
        "visible_frame_count": len(visible_frames),
        "visible_frame_min": min(visible_frames) if visible_frames else None,
        "visible_frame_max": max(visible_frames) if visible_frames else None,
        "part_containment_in_object": stats(containments),
        "object_coverage_by_part": stats(coverages),
        "mask_area_px": stats(areas),
        "prompt_box_iou_with_mask_bbox": stats(prompt_box_ious),
        "prompt_mask_fraction_inside_box": stats(prompt_mask_in_box),
        "prompt_detections": track.get("prompt_detections"),
    }
    write_json(root.parent / "v18_owlv2_sam2_part_track_report.json", report)
    return report, track_path


def track_acceptance(track: dict[str, Any], frame_masks: dict[int, np.ndarray], object_masks: dict[tuple[int, str], str], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, Any]]:
    object_id = require_str(track.get("object_id"), "object_id")
    prompt_detections = require_list(track.get("prompt_detections"), "prompt detections")
    containments: list[float] = []
    coverages: list[float] = []
    areas: list[float] = []
    prompt_box_ious: list[float] = []
    prompt_mask_in_box: list[float] = []
    prompt_by_frame = {require_int(row.get("frame_idx"), "frame_idx"): row for row in prompt_detections}
    visible_count = 0
    for frame_idx, mask in frame_masks.items():
        object_mask_path = object_masks.get((frame_idx, object_id))
        if object_mask_path is None:
            continue
        object_mask = load_mask(Path(object_mask_path), mask.shape)
        area = int(mask.sum())
        object_area = int(object_mask.sum())
        if area <= 0 or object_area <= 0:
            continue
        inter = int(np.logical_and(mask, object_mask).sum())
        containment = inter / float(area)
        coverage = inter / float(object_area)
        if (
            area >= int(args.min_saved_mask_area_px)
            and containment >= float(args.min_track_frame_containment)
            and coverage >= float(args.min_track_frame_object_coverage)
            and coverage <= float(args.max_track_frame_object_coverage)
        ):
            visible_count += 1
            containments.append(float(containment))
            coverages.append(float(coverage))
            areas.append(float(area))
        prompt = prompt_by_frame.get(frame_idx)
        if prompt is not None:
            prompt_box_ious.append(box_iou(bbox_from_mask(mask), prompt["bbox_xyxy"]))
            prompt_mask_in_box.append(mask_fraction_in_box(mask, prompt["bbox_xyxy"]))
    blockers: list[str] = []
    if len(prompt_detections) < int(args.min_prompt_detections_for_acceptance):
        blockers.append("too_few_owlv2_keyframe_detections")
    if visible_count < int(args.min_visible_track_frames):
        blockers.append("too_few_sam2_visible_track_frames")
    if percentile(containments, 50.0) is None or float(percentile(containments, 50.0) or 0.0) < float(args.min_track_median_containment):
        blockers.append("low_median_track_containment_in_object")
    if percentile(coverages, 50.0) is None or float(percentile(coverages, 50.0) or 0.0) > float(args.max_track_median_object_coverage):
        blockers.append("track_is_whole_object_like")
    if prompt_box_ious and float(percentile(prompt_box_ious, 50.0) or 0.0) < float(args.min_prompt_box_iou_median):
        blockers.append("sam2_mask_bbox_disagrees_with_owlv2_prompt_box")
    if prompt_mask_in_box and float(percentile(prompt_mask_in_box, 50.0) or 0.0) < float(args.min_prompt_mask_in_box_median):
        blockers.append("sam2_mask_not_supported_by_owlv2_prompt_box")
    metrics = {
        "visible_count": visible_count,
        "part_containment_in_object": stats(containments),
        "object_coverage_by_part": stats(coverages),
        "mask_area_px": stats(areas),
        "prompt_box_iou_with_mask_bbox": stats(prompt_box_ious),
        "prompt_mask_fraction_inside_box": stats(prompt_mask_in_box),
    }
    return not blockers, blockers, metrics


def case_report(case: str, processor: Any, owlv2_model: Any, sam2_predictor: Any, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    schema = schema_by_object(case, args)
    object_ids = object_ids_requiring_parts(schema)
    rows_by_object = annotation_object_rows(annotation, object_ids)
    object_masks = object_mask_index(annotation, object_ids)
    object_records: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    track_prompts: list[dict[str, Any]] = []
    for object_id in sorted(object_ids):
        schema_row = schema.get(object_id, {})
        terms = prompt_terms_for_object(schema_row, args)
        prompts = text_prompts(terms)
        available = rows_by_object.get(object_id, [])
        keyframes = quantile_select(available, int(args.max_keyframes_per_object))
        object_detections: list[dict[str, Any]] = []
        for row in keyframes:
            if prompts:
                object_detections.extend(run_owlv2_on_frame(processor, owlv2_model, row, prompts, args))
        detection_rows.extend(object_detections)
        by_label = best_detections_by_label(object_detections)
        for label, detections in by_label.items():
            track_key = f"{slug(object_id)}::{label}"
            track_prompts.append(
                {
                    "track_key": track_key,
                    "object_id": object_id,
                    "part_label": label,
                    "prompt_detections": detections,
                    "prompt_source": {
                        "source": "v18_physical_state_schema_model_notes_plus_generic_physical_part_lexicon",
                        "terms": terms,
                        "text_prompts": prompts,
                        "physical_notes": schema_row.get("physical_notes"),
                        "part_or_relative_motion_evidence_terms": schema_row.get("part_or_relative_motion_evidence_terms"),
                    },
                }
            )
        object_records.append(
            {
                "object_id": object_id,
                "available_visible_object_mask_frame_count": len(available),
                "selected_keyframe_count": len(keyframes),
                "prompt_terms": terms,
                "text_prompts": prompts,
                "owlv2_detection_count": len(object_detections),
                "owlv2_candidate_detection_count": sum(1 for row in object_detections if row.get("detection_state") == "candidate_owlv2_part_box"),
                "candidate_part_labels": sorted(by_label),
            }
        )
    if not track_prompts:
        report = {
            "method": "build_v18_owlv2_sam2_part_tracks",
            "status": STATUS,
            "claim": CLAIM,
            "case": case,
            "sources": {"physical_state_schema": str(args.physical_state_schema_root / case / "v18_physical_state_schema_report.json"), "annotation_state": str(annotation_path)},
            "object_records": object_records,
            "detection_rows": detection_rows,
            "track_records": [],
            "accepted_track_count": 0,
            "rejected_track_count": 0,
            "mask_evidence_created_count": 0,
            "part_pose_ready_count": 0,
            "object_pose_requirement_met_count": 0,
            "default_path_uses_bundlesdf_or_nerf": False,
            **FALSE_READY,
        }
        write_json(args.output_root / case / "v18_owlv2_sam2_part_tracks_report.json", report)
        return report

    video_dir = raw_video_dir(annotation)
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        state = sam2_predictor.init_state(str(video_dir), offload_video_to_cpu=bool(args.offload_video_to_cpu), offload_state_to_cpu=bool(args.offload_state_to_cpu))
        obj_id_by_track = {require_str(track.get("track_key"), "track_key"): i + 1 for i, track in enumerate(track_prompts)}
        add_sam2_prompts(sam2_predictor, state, track_prompts, obj_id_by_track)
        masks_by_track = propagate_tracks(sam2_predictor, state, track_prompts, obj_id_by_track, object_masks, args)
    track_records: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for track in track_prompts:
        track_key = require_str(track.get("track_key"), "track_key")
        accepted, blockers, metrics = track_acceptance(track, masks_by_track.get(track_key, {}), object_masks, args)
        state = "accepted_semantic_temporal_part_track" if accepted else "rejected_semantic_temporal_track_qc"
        state_counts[state] += 1
        materialized, _track_path = materialize_track(case, track, masks_by_track.get(track_key, {}), object_masks, args.output_root, accepted, args)
        track_records.append(
            {
                **materialized,
                "track_key": track_key,
                "track_state": state,
                "track_blockers": blockers,
                "acceptance_metrics": metrics,
            }
        )
    report = {
        "method": "build_v18_owlv2_sam2_part_tracks",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"annotation_state": str(annotation_path), "physical_state_schema": str(args.physical_state_schema_root / case / "v18_physical_state_schema_report.json")},
        "owlv2_backend": {"model": str(args.owlv2_model), "threshold": args.owlv2_threshold},
        "sam2_backend": {"repo": str(args.sam2_repo), "model_cfg": str(args.sam2_model_cfg), "checkpoint": str(args.sam2_checkpoint), "device": args.device},
        "object_records": object_records,
        "detection_state_counts": dict(sorted(Counter(str(row.get("detection_state")) for row in detection_rows).items())),
        "detection_rows": detection_rows,
        "track_state_counts": dict(sorted(state_counts.items())),
        "track_records": track_records,
        "accepted_track_count": sum(1 for row in track_records if row.get("accepted_as_semantic_temporal_part_track") is True),
        "rejected_track_count": sum(1 for row in track_records if row.get("accepted_as_semantic_temporal_part_track") is not True),
        "mask_evidence_created_count": sum(1 for row in track_records if row.get("accepted_as_semantic_temporal_part_track") is True),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_owlv2_sam2_part_tracks_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not args.sam2_checkpoint.exists():
        raise RuntimeError(f"SAM2 checkpoint missing: {args.sam2_checkpoint}")
    if not Path(args.owlv2_model).exists():
        raise RuntimeError(f"OWLv2 local model path missing: {args.owlv2_model}")
    processor = Owlv2Processor.from_pretrained(str(args.owlv2_model), local_files_only=True)
    owlv2_model: Any = Owlv2ForObjectDetection.from_pretrained(str(args.owlv2_model), local_files_only=True)
    owlv2_model = owlv2_model.to(args.device)
    owlv2_model.eval()
    sam2_predictor = import_sam2(args)
    reports = [case_report(case, processor, owlv2_model, sam2_predictor, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    detection_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(require_dict(report.get("track_state_counts", {}), "track state counts"))
        detection_counts.update(require_dict(report.get("detection_state_counts", {}), "detection state counts"))
    summary = {
        "method": "build_v18_owlv2_sam2_part_tracks",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "track_state_counts": dict(sorted(state_counts.items())),
        "detection_state_counts": dict(sorted(detection_counts.items())),
        "accepted_track_count": sum(require_int(report.get("accepted_track_count"), "accepted track count") for report in reports),
        "rejected_track_count": sum(require_int(report.get("rejected_track_count"), "rejected track count") for report in reports),
        "mask_evidence_created_count": sum(require_int(report.get("mask_evidence_created_count"), "mask evidence count") for report in reports),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "accepted_track_roots_by_case": {
            str(report["case"]): str(args.output_root / str(report["case"]) / "accepted_tracks") for report in reports
        },
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_owlv2_sam2_part_tracks_report.json"),
                "accepted_track_count": report["accepted_track_count"],
                "rejected_track_count": report["rejected_track_count"],
                "mask_evidence_created_count": report["mask_evidence_created_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_owlv2_sam2_part_tracks_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_owlv2_sam2_part_tracks"))
    parser.add_argument("--owlv2-model", type=Path, default=Path("/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d"))
    parser.add_argument("--sam2-repo", type=Path, default=Path("third_party/sam2"))
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-keyframes-per-object", type=int, default=5)
    parser.add_argument("--max-prompt-terms-per-object", type=int, default=6)
    parser.add_argument("--extra-generic-part-terms", nargs="+", default=[])
    parser.add_argument("--owlv2-threshold", type=float, default=0.03)
    parser.add_argument("--crop-pad-fraction", type=float, default=0.25)
    parser.add_argument("--min-box-containment", type=float, default=0.35)
    parser.add_argument("--min-box-object-coverage", type=float, default=0.01)
    parser.add_argument("--max-box-object-coverage", type=float, default=0.90)
    parser.add_argument("--min-prompt-detections-for-acceptance", type=int, default=2)
    parser.add_argument("--min-visible-track-frames", type=int, default=8)
    parser.add_argument("--min-saved-mask-area-px", type=int, default=50)
    parser.add_argument("--min-track-frame-containment", type=float, default=0.50)
    parser.add_argument("--min-track-frame-object-coverage", type=float, default=0.005)
    parser.add_argument("--max-track-frame-object-coverage", type=float, default=0.90)
    parser.add_argument("--min-track-median-containment", type=float, default=0.60)
    parser.add_argument("--max-track-median-object-coverage", type=float, default=0.85)
    parser.add_argument("--min-prompt-box-iou-median", type=float, default=0.05)
    parser.add_argument("--min-prompt-mask-in-box-median", type=float, default=0.35)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--vos-optimized", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
