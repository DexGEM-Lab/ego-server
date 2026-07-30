#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, meta


def run(args: argparse.Namespace) -> dict[str, Any]:
    ann = load_json(args.annotations)
    source_video = Path(str(ann["source_video"]))
    cap, meta = open_video(source_video)
    render_width = int(args.render_width)
    render_height = int(round(render_width * meta["height"] / meta["width"]))
    if render_height % 2:
        render_height += 1
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]), (render_width, render_height))
    if not writer.isOpened():
        cap.release()
        raise ContractError(f"could_not_open_writer: {args.output_video}")
    draw_masks = 0
    draw_boxes = 0
    expected = len(ann.get("frames", []))
    try:
        for frame_row in ann.get("frames", []):
            frame_idx = int(frame_row["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"could_not_read_video_frame: {frame_idx}")
            frame = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            caption_parts = [f"V21 segmentation baseline f{frame_idx}"]
            for obj in frame_row.get("objects", []):
                if not isinstance(obj, dict) or not obj.get("mask_path"):
                    continue
                mask = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = cv2.resize(mask, (render_width, render_height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(frame)
                tint[:, :, 1] = 220
                tint[:, :, 2] = 80
                frame[mask] = cv2.addWeighted(frame, 0.52, tint, 0.48, 0.0)[mask]
                draw_masks += 1
                bbox = obj.get("bbox_xyxy")
                if isinstance(bbox, list) and len(bbox) == 4:
                    sx = render_width / float(meta["width"])
                    sy = render_height / float(meta["height"])
                    x0, y0, x1, y1 = [int(round(float(v) * (sx if i % 2 == 0 else sy))) for i, v in enumerate(bbox)]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
                    draw_boxes += 1
                caption_parts.append(str(obj.get("track_id", "object")))
            text = " | ".join(caption_parts)
            cv2.rectangle(frame, (0, render_height - 42), (render_width, render_height), (0, 0, 0), -1)
            cv2.putText(frame, text[:120], (14, render_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    cap2 = cv2.VideoCapture(str(args.output_video))
    out_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) if cap2.isOpened() else 0
    cap2.release()
    summary = {
        "schema": "v21_segmentation_overlay_render_summary.v0",
        "status": "ok" if out_frames == expected else "frame_count_mismatch",
        "method": "render_v21_segmentation_overlay",
        "annotations": str(args.annotations),
        "output_video": str(args.output_video),
        "expected_frame_count": int(expected),
        "overlay_frame_count": int(out_frames),
        "frame_count_match": bool(out_frames == expected),
        "draw_counts": {"masks": int(draw_masks), "boxes": int(draw_boxes)},
        "claim_scope": "V21 segmentation overlay render. This visibly displays accepted SAM2 mask evidence only; it does not render metric MANO, object mesh pose, contact, occlusion, or nonpenetration.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render V21 accepted segmentation mask overlay from state/annotations boundary.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
