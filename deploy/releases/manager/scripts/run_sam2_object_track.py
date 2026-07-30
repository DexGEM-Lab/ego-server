#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fuse_v1_full_fidelity import DEFAULT_CLIP, DEFAULT_MANO_RIGHT, draw_object_overlay, load_json, open_video, put_caption, read_video_frame


SAM2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def selected_frame_indices(frames: list[dict], start: int | None, end: int | None) -> list[int]:
    out = []
    for i, frame in enumerate(frames):
        source_idx = int(frame["frame_idx"])
        if start is not None and source_idx < start:
            continue
        if end is not None and source_idx > end:
            continue
        out.append(i)
    if not out:
        raise RuntimeError("selected source range contains no annotation frames")
    return out


def prompt_local_index(frames: list[dict], selected: list[int]) -> int:
    for local_idx, frame_idx in enumerate(selected):
        obj = frames[frame_idx].get("object", {})
        if obj.get("bbox_xyxy") is not None and obj.get("center_xy") is not None:
            return local_idx
    raise RuntimeError("selected source range contains no object bbox prompt")


def extract_frames(clip: Path, frames: list[dict], selected: list[int], output_dir: Path, image_width: int) -> Path:
    frame_dir = output_dir / "sam2_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(clip)
    scale = image_width / float(info.width)
    image_height = int(round(info.height * scale))
    for local_idx, frame_idx in enumerate(selected):
        source_idx = int(frames[frame_idx]["frame_idx"])
        frame = read_video_frame(cap, source_idx)
        resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
        path = frame_dir / f"{local_idx:06d}.jpg"
        if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise RuntimeError(f"failed to write {path}")
    cap.release()
    return frame_dir


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)
    return box, float(xs.size), center


def iou_box(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else None


def run_sam2(args: argparse.Namespace, frames: list[dict], selected: list[int], frame_dir: Path, scale: float) -> dict[int, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 video predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=device, vos_optimized=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        prompt_local = prompt_local_index(frames, selected)
        prompt_frame = frames[selected[prompt_local]]
        box = np.asarray(prompt_frame["object"]["bbox_xyxy"], dtype=np.float32) * scale
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_local,
            obj_id=1,
            box=box,
        )
        mask_dir = args.output_dir / "sam2_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, dict] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if 1 not in [int(v) for v in out_obj_ids]:
                continue
            obj_pos = [int(v) for v in out_obj_ids].index(1)
            mask = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0).astype(np.uint8)
            box_small, area_small, center_small = mask_box(mask)
            source_frame_idx = selected[int(out_frame_idx)]
            source_idx = int(frames[source_frame_idx]["frame_idx"])
            if box_small is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            inv = 1.0 / scale
            box_source = [float(v * inv) for v in box_small]
            center_source = (center_small * inv).astype(float).tolist()
            v2_box = frames[source_frame_idx]["object"].get("bbox_xyxy")
            results[source_idx] = {
                "visible": True,
                "bbox_xyxy": box_source,
                "center_xy": center_source,
                "area_px": float(area_small * inv * inv),
                "iou_with_v2_bbox": iou_box(box_source, v2_box),
                "mask_path": str(mask_path),
            }
    return results


def render_comparison(args: argparse.Namespace, frames: list[dict], selected: list[int], sam2: dict[int, dict], scale: float) -> Path:
    cap, info = open_video(args.clip)
    out_path = args.output_dir / "sam2_overlay_compare.mp4"
    render_height = int(round(args.render_width * info.height / info.width))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, render_height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer {out_path}")
    sx = args.render_width / float(info.width)
    sy = render_height / float(info.height)
    for frame_idx in selected:
        ann = frames[frame_idx]
        source_idx = int(ann["frame_idx"])
        image = read_video_frame(cap, source_idx)
        image = cv2.resize(image, (args.render_width, render_height), interpolation=cv2.INTER_AREA)
        draw_object_overlay(image, ann, sx, sy)
        result = sam2.get(source_idx, {})
        if result.get("visible"):
            mask_path = result.get("mask_path")
            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read {mask_path}")
                mask = cv2.resize(mask, (args.render_width, render_height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(image)
                tint[:, :, 0] = 255
                tint[:, :, 2] = 255
                image[mask] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask]
            box = np.asarray(result["bbox_xyxy"], dtype=float) * np.asarray([sx, sy, sx, sy])
            center = np.asarray(result["center_xy"], dtype=float) * np.asarray([sx, sy])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.drawMarker(image, tuple(center.astype(int)), (255, 0, 255), cv2.MARKER_DIAMOND, 16, 2)
        put_caption(image, "SAM2 magenta vs v2 red", source_idx)
        writer.write(image)
    writer.release()
    cap.release()
    return out_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.annotations)
    frames = payload["frames"]
    selected = selected_frame_indices(frames, args.frame_start, args.frame_end)
    prompt_local = prompt_local_index(frames, selected)
    frame_dir = extract_frames(args.clip, frames, selected, args.output_dir, args.sam2_image_width)
    cap, info = open_video(args.clip)
    cap.release()
    scale = args.sam2_image_width / float(info.width)
    sam2 = run_sam2(args, frames, selected, frame_dir, scale)
    compare_path = render_comparison(args, frames, selected, sam2, scale)
    ious = [v["iou_with_v2_bbox"] for v in sam2.values() if v.get("iou_with_v2_bbox") is not None]
    visible = sum(1 for v in sam2.values() if v.get("visible"))
    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "source_annotations": str(args.annotations),
        "frames_selected": len(selected),
        "source_frame_start": int(frames[selected[0]]["frame_idx"]),
        "source_frame_end": int(frames[selected[-1]]["frame_idx"]),
        "prompt_local_frame": prompt_local,
        "prompt_source_frame": int(frames[selected[prompt_local]]["frame_idx"]),
        "v2_bbox_prompt_frames": sum(1 for frame_idx in selected if frames[frame_idx].get("object", {}).get("bbox_xyxy") is not None),
        "sam2_visible_frames": visible,
        "median_iou_with_v2_bbox": float(np.median(ious)) if ious else None,
        "p05_iou_with_v2_bbox": float(np.percentile(ious, 5)) if ious else None,
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "elapsed_s": time.time() - started,
        "outputs": {
            "sam2_track": str(args.output_dir / "sam2_track.json"),
            "sam2_masks": str(args.output_dir / "sam2_masks"),
            "comparison_video": str(compare_path),
        },
    }
    (args.output_dir / "sam2_track.json").write_text(json.dumps(sam2, indent=2), encoding="utf-8")
    (args.output_dir / "qc_sam2_track.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-image-width", type=int, default=768)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
