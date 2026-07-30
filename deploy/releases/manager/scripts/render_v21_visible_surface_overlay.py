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


def project(points: np.ndarray, intr: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intr]
    z = np.clip(points[:, 2], 1e-6, None)
    return np.c_[fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy]


def run(args: argparse.Namespace) -> dict[str, Any]:
    ann = load_json(args.visible_geometry_annotations)
    raw_video = ann.get("raw_video") if isinstance(ann.get("raw_video"), dict) else {}
    fps = float(raw_video.get("fps", args.fps))
    src_w = int(raw_video.get("width", 0) or ann.get("frames", [{}])[0].get("source_width", 0))
    src_h = int(raw_video.get("height", 0) or ann.get("frames", [{}])[0].get("source_height", 0))
    if src_w <= 0 or src_h <= 0:
        raise ContractError("could_not_determine_source_resolution")
    render_w = int(args.render_width)
    render_h = int(round(render_w * src_h / src_w))
    if render_h % 2:
        render_h += 1
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    if not writer.isOpened():
        raise ContractError(f"could_not_open_writer: {args.output_video}")
    surfel_frames = 0
    surfel_points = 0
    mask_frames = 0
    frames = ann.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("visible_geometry_annotations_have_no_frames")
    sx = render_w / float(src_w)
    sy = render_h / float(src_h)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", 0))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        image = cv2.resize(image, (render_w, render_h), interpolation=cv2.INTER_AREA)
        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            mask_path = obj.get("mask_path")
            if mask_path:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST) > 0
                    tint = np.zeros_like(image)
                    tint[:, :, 1] = 130
                    tint[:, :, 2] = 255
                    image[mask] = cv2.addWeighted(image, 0.68, tint, 0.32, 0.0)[mask]
                    mask_frames += 1
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            pts = np.asarray(geom.get("camera_vertices_sample_m") or geom.get("world_vertices_sample_m") or [], dtype=float)
            intr = geom.get("intrinsics_fx_fy_cx_cy")
            if pts.ndim == 2 and pts.shape[1] == 3 and isinstance(intr, list) and len(intr) == 4 and len(pts) > 0:
                xy = project(pts, intr)
                valid = np.isfinite(xy).all(axis=1) & np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
                xy = xy[valid]
                if len(xy):
                    surfel_frames += 1
                    surfel_points += int(len(xy))
                    for x, y in xy:
                        px = int(round(float(x) * sx))
                        py = int(round(float(y) * sy))
                        if 0 <= px < render_w and 0 <= py < render_h:
                            cv2.circle(image, (px, py), int(args.point_radius), (0, 255, 255), -1, cv2.LINE_AA)
                bbox = obj.get("bbox_xyxy")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x0, y0, x1, y1 = [int(round(float(v) * (sx if i % 2 == 0 else sy))) for i, v in enumerate(bbox)]
                    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
        text = f"V21 visible surface f{frame_idx}: mask+DepthPro surfels, not mesh pose"
        cv2.rectangle(image, (0, render_h - 42), (render_w, render_h), (0, 0, 0), -1)
        cv2.putText(image, text[:120], (14, render_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    cap = cv2.VideoCapture(str(args.output_video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    summary = {
        "schema": "v21_visible_surface_overlay_render_summary.v0",
        "status": "ok" if frame_count == len(frames) else "frame_count_mismatch",
        "method": "render_v21_visible_surface_overlay",
        "visible_geometry_annotations": str(args.visible_geometry_annotations),
        "output_video": str(args.output_video),
        "expected_frame_count": int(len(frames)),
        "overlay_frame_count": int(frame_count),
        "frame_count_match": bool(frame_count == len(frames)),
        "draw_counts": {"mask_frames": int(mask_frames), "surfel_frames": int(surfel_frames), "surfel_points": int(surfel_points)},
        "claim_scope": "V21 visible-surface overlay render. Yellow points are mask+DepthPro surfel measurements; this is not completed object mesh, rigid pose, or contact.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render V21 visible surface surfels projected over RGB.")
    parser.add_argument("--visible-geometry-annotations", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
