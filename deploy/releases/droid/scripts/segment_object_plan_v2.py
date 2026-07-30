#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fuse_v1_full_fidelity import (
    DEFAULT_CLIP,
    RenderSpec,
    hand_association_geometry,
    load_json,
    load_owl_detector,
    load_sam,
    mask_contact_score,
    open_video,
    read_video_frame,
)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1 + 1.0), max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def plan_objects(plan_path: Path) -> list[dict]:
    blob = load_json(plan_path)
    plan = blob["plan"] if "plan" in blob else blob
    objects = plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"object plan has no objects: {plan_path}")
    return objects


def owl_boxes_for_prompts(processor, model, frame: np.ndarray, threshold: float, prompts: list[str]) -> list[dict]:
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    text = [[str(prompt) for prompt in prompts]]
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]],
        text_labels=text,
    )[0]
    boxes = []
    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        boxes.append({"box": box.detach().cpu().numpy().astype(float).tolist(), "score": float(score), "label": str(label), "source": "owlv2_plan_prompt"})
    return boxes


def mask_metrics(mask: np.ndarray, geom: dict, prev_mask: np.ndarray | None) -> dict:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("empty mask cannot be scored")
    center = np.asarray([xs.mean(), ys.mean()], dtype=float)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    contact_ratio, min_tip_dist = mask_contact_score(mask, geom["tips"])
    temporal_iou = 0.0 if prev_mask is None else bbox_iou(bbox, mask_bbox(prev_mask))
    return {
        "center_xy": [float(center[0]), float(center[1])],
        "bbox_xyxy": bbox,
        "area_px": int(mask.sum()),
        "contact_ratio": float(contact_ratio),
        "min_tip_dist_px": float(min_tip_dist),
        "temporal_iou": float(temporal_iou),
    }


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def sam_mask_candidates(sam, frame: np.ndarray, boxes: list[dict], geom: dict, prev_mask: np.ndarray | None) -> list[tuple[float, np.ndarray, dict]]:
    if not boxes:
        return []
    sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scored = []
    for box_info in boxes[:16]:
        box = np.asarray(box_info["box"], dtype=float)
        masks, scores, _ = sam.predict(box=box, multimask_output=True)
        for mask, sam_score in zip(masks, scores):
            mask = mask.astype(bool)
            if int(mask.sum()) < 80:
                continue
            metrics = mask_metrics(mask, geom, prev_mask)
            area = float(metrics["area_px"])
            box_area = max(1.0, (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0))
            area_ratio = min(area / box_area, box_area / max(area, 1.0))
            hand_term = 1.0 / (1.0 + min(metrics["min_tip_dist_px"], 400.0) / 80.0)
            temporal_term = float(metrics["temporal_iou"])
            score = 0.55 * float(sam_score) + 0.35 * float(box_info["score"]) + 0.25 * area_ratio + 0.25 * hand_term + 0.20 * temporal_term
            candidate_metrics = dict(metrics)
            candidate_metrics.update(
                {
                    "score": float(score),
                    "sam_score": float(sam_score),
                    "owl_score": float(box_info["score"]),
                    "owl_label": box_info["label"],
                    "proposal_source": box_info["source"],
                }
            )
            scored.append((float(score), mask, candidate_metrics))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def select_sam_mask(sam, frame: np.ndarray, boxes: list[dict], geom: dict, prev_mask: np.ndarray | None) -> tuple[np.ndarray, dict] | None:
    scored = sam_mask_candidates(sam, frame, boxes, geom, prev_mask)
    if not scored:
        return None
    _, mask, metrics = scored[0]
    return mask, metrics


def intervals_for_object(obj: dict, frame_start: int, frame_end: int) -> list[tuple[int, int]]:
    intervals = []
    for raw in obj.get("active_intervals", []):
        start = max(int(raw["start_frame"]), frame_start)
        end = min(int(raw["end_frame"]), frame_end)
        if start <= end:
            intervals.append((start, end))
    return intervals


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    frames = annotations["frames"]
    available = {int(frame["frame_idx"]): i for i, frame in enumerate(frames)}
    frame_start = min(available) if args.frame_start is None else int(args.frame_start)
    frame_end = max(available) if args.frame_end is None else int(args.frame_end)
    objects = plan_objects(args.object_plan)
    if args.object_index >= len(objects):
        raise RuntimeError(f"--object-index {args.object_index} out of range for {len(objects)} planned objects")
    obj_plan = objects[args.object_index]
    prompts = [str(prompt) for prompt in obj_plan["open_vocabulary_prompts"]]
    intervals = intervals_for_object(obj_plan, frame_start, frame_end)
    if not intervals:
        raise RuntimeError("selected planned object has no active interval in requested frame range")

    cap, info = open_video(args.clip)
    render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_meas: list[dict | None] = [None] * len(frames)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_processor, owl_model = load_owl_detector(device)
    sam = load_sam(args.sam_checkpoint, device)
    prev_mask = None
    processed = 0
    detected = 0
    try:
        source_indices = []
        for start, end in intervals:
            for idx in range(start, end + 1, max(1, int(args.object_stride))):
                if idx in available:
                    source_indices.append(idx)
        for source_idx in tqdm(sorted(set(source_indices)), desc="object_plan_sam"):
            local_idx = available[source_idx]
            frame_ann = frames[local_idx]
            frame = read_video_frame(cap, source_idx)
            boxes = owl_boxes_for_prompts(owl_processor, owl_model, frame, args.owl_threshold, prompts)
            selected = select_sam_mask(sam, frame, boxes, hand_association_geometry(frame_ann), prev_mask)
            processed += 1
            if selected is None:
                continue
            mask, metrics = selected
            mask_path = args.output_dir / "object_masks" / f"{source_idx:06d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_small = cv2.resize(mask.astype(np.uint8) * 255, (render.width, render.height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_path), mask_small)
            prev_mask = mask
            object_meas[local_idx] = {
                **metrics,
                "status": "measured_plan_sam",
                "track_id": obj_plan["track_id"],
                "label": obj_plan["description"],
                "prompts": prompts,
                "mask_path": str(mask_path),
                "mask_image_size": [int(render.width), int(render.height)],
                "source_image_size": [int(frame.shape[1]), int(frame.shape[0])],
            }
            detected += 1
    finally:
        cap.release()
        del owl_model, sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, measurement in enumerate(object_meas):
        if measurement is not None:
            frames[i]["object"] = measurement
        else:
            source_idx = int(frames[i]["frame_idx"])
            active = any(start <= source_idx <= end for start, end in intervals)
            frames[i]["object"] = {
                "status": "unobserved_no_plan_mask" if active else "outside_plan_interval",
                "track_id": obj_plan["track_id"] if active else None,
                "label": obj_plan["description"] if active else None,
            }

    annotations_path = args.output_dir / "annotations_plan_masks.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "backend": "VLM object plan + OWLv2 plan prompts + SAM masks",
        "object_plan": str(args.object_plan),
        "object_index": int(args.object_index),
        "track_id": obj_plan["track_id"],
        "description": obj_plan["description"],
        "prompts": prompts,
        "intervals": [{"start_frame": start, "end_frame": end} for start, end in intervals],
        "processed_frames": processed,
        "detected_frames": detected,
        "detection_rate_on_processed": detected / max(1, processed),
        "annotations": str(annotations_path),
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "qc_plan_masks.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--object-stride", type=int, default=1)
    parser.add_argument("--owl-threshold", type=float, default=0.03)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
