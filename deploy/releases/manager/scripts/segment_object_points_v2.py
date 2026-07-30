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
    load_sam,
    mask_contact_score,
    open_video,
    read_video_frame,
)


def prompt_by_frame(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"point prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def scale_points(points: list[dict], from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    scale = np.asarray([to_size[0] / from_size[0], to_size[1] / from_size[1]], dtype=np.float32)
    xy = np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    return xy * scale[None, :]


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_metrics(mask: np.ndarray, frame_ann: dict, prompt: dict, score: float) -> dict:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("empty mask")
    center = [float(xs.mean()), float(ys.mean())]
    geom = hand_association_geometry(frame_ann)
    contact_ratio, min_tip_dist = mask_contact_score(mask, geom["tips"])
    return {
        "center_xy": center,
        "bbox_xyxy": mask_bbox(mask),
        "area_px": int(mask.sum()),
        "contact_ratio": float(contact_ratio),
        "min_tip_dist_px": float(min_tip_dist),
        "score": float(score),
        "sam_score": float(score),
        "proposal_source": "vlm_points_sam",
        "vlm_visual_evidence": prompt.get("visual_evidence", ""),
        "vlm_confidence": float(prompt.get("confidence", 0.0)),
        "positive_point_count": len(prompt.get("positive_points", [])),
        "negative_point_count": len(prompt.get("negative_points", [])),
    }


def draw_prompt_review(frame: np.ndarray, prompt: dict, prompt_size: tuple[int, int], mask: np.ndarray | None, output_path: Path) -> None:
    tile = cv2.resize(frame, prompt_size, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask_small = cv2.resize(mask.astype(np.uint8), prompt_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = tile.copy()
        overlay[mask_small] = (0.35 * overlay[mask_small] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
        tile = overlay
    if len(prompt.get("bbox_xyxy", [])) >= 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in prompt["bbox_xyxy"][:4]]
        cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for point in prompt.get("positive_points", []):
        cv2.circle(tile, (int(round(float(point["x"]))), int(round(float(point["y"])))), 8, (0, 255, 0), -1)
    for point in prompt.get("negative_points", []):
        cv2.circle(tile, (int(round(float(point["x"]))), int(round(float(point["y"])))), 8, (0, 0, 255), -1)
    label = f"frame {prompt['frame_idx']}"
    cv2.putText(tile, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), tile):
        raise RuntimeError(f"failed to write review image {output_path}")


def select_mask(sam, frame: np.ndarray, prompt: dict, prompt_size: tuple[int, int]) -> tuple[np.ndarray, float] | None:
    positives = prompt.get("positive_points", [])
    if not bool(prompt.get("target_visible")) or not positives:
        return None
    source_size = (frame.shape[1], frame.shape[0])
    pos = scale_points(positives, prompt_size, source_size)
    neg = scale_points(prompt.get("negative_points", []), prompt_size, source_size)
    point_coords = np.vstack([pos, neg]).astype(np.float32)
    point_labels = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    if len(point_coords) == 0:
        return None
    box = None
    if len(prompt.get("bbox_xyxy", [])) >= 4:
        box_arr = np.asarray(prompt["bbox_xyxy"][:4], dtype=np.float32)
        box_scale = np.asarray([source_size[0] / prompt_size[0], source_size[1] / prompt_size[1], source_size[0] / prompt_size[0], source_size[1] / prompt_size[1]], dtype=np.float32)
        box = box_arr * box_scale
    sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    masks, scores, _ = sam.predict(point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=True)
    candidates = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(bool)
        area = int(mask.sum())
        if area < 80:
            continue
        positives_inside = 0
        for x, y in pos:
            xi = int(np.clip(round(float(x)), 0, mask.shape[1] - 1))
            yi = int(np.clip(round(float(y)), 0, mask.shape[0] - 1))
            positives_inside += int(mask[yi, xi])
        negatives_inside = 0
        for x, y in neg:
            xi = int(np.clip(round(float(x)), 0, mask.shape[1] - 1))
            yi = int(np.clip(round(float(y)), 0, mask.shape[0] - 1))
            negatives_inside += int(mask[yi, xi])
        point_score = positives_inside - negatives_inside
        candidates.append((float(score) + 0.25 * float(point_score), mask, float(score)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    frames = annotations["frames"]
    available = {int(frame["frame_idx"]): i for i, frame in enumerate(frames)}
    prompts, prompt_payload = prompt_by_frame(args.point_prompts)
    prompt_size = (
        int(prompt_payload["prompt_image_width"]),
        int(round(int(prompt_payload["prompt_image_width"]) * 9 / 16)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(args.clip)
    render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = load_sam(args.sam_checkpoint, device)
    detected = 0
    processed = 0
    try:
        for source_idx in tqdm(sorted(set(prompts) & set(available)), desc="vlm_points_sam"):
            if args.frame_start is not None and source_idx < int(args.frame_start):
                continue
            if args.frame_end is not None and source_idx > int(args.frame_end):
                continue
            prompt = prompts[source_idx]
            local_idx = available[source_idx]
            frame_ann = frames[local_idx]
            frame = read_video_frame(cap, source_idx)
            processed += 1
            selected = select_mask(sam, frame, prompt, prompt_size)
            if selected is None:
                draw_prompt_review(frame, prompt, prompt_size, None, args.output_dir / "review_stills" / f"{source_idx:06d}.jpg")
                frame_ann["object"] = {
                    "status": "unobserved_vlm_points",
                    "track_id": prompt_payload["track_id"],
                    "label": prompt_payload["description"],
                    "vlm_point_prompt": prompt,
                }
                continue
            mask, score = selected
            metrics = mask_metrics(mask, frame_ann, prompt, score)
            mask_path = args.output_dir / "object_masks" / f"{source_idx:06d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_small = cv2.resize(mask.astype(np.uint8) * 255, (render.width, render.height), interpolation=cv2.INTER_NEAREST)
            if not cv2.imwrite(str(mask_path), mask_small):
                raise RuntimeError(f"failed to write mask {mask_path}")
            draw_prompt_review(frame, prompt, prompt_size, mask, args.output_dir / "review_stills" / f"{source_idx:06d}.jpg")
            frame_ann["object"] = {
                **metrics,
                "status": "measured_vlm_points_sam",
                "track_id": prompt_payload["track_id"],
                "label": prompt_payload["description"],
                "mask_path": str(mask_path),
                "mask_image_size": [int(render.width), int(render.height)],
                "source_image_size": [int(frame.shape[1]), int(frame.shape[0])],
                "vlm_point_prompt": prompt,
            }
            detected += 1
    finally:
        cap.release()
        del sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for idx, prompt in prompts.items():
        if idx not in available:
            continue
        frame = frames[available[idx]]
        if "object" not in frame:
            frame["object"] = {
                "status": "unobserved_vlm_points",
                "track_id": prompt_payload["track_id"],
                "label": prompt_payload["description"],
                "vlm_point_prompt": prompt,
            }
    annotations_path = args.output_dir / "annotations_vlm_points_sam.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "backend": "VLM point prompts + SAM masks",
        "clip": str(args.clip),
        "source_annotations": str(args.annotations),
        "point_prompts": str(args.point_prompts),
        "track_id": prompt_payload["track_id"],
        "processed_frames": processed,
        "detected_frames": detected,
        "detection_rate_on_processed": detected / max(1, processed),
        "annotations": str(annotations_path),
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "qc_vlm_points_sam.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
