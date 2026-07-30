#!/usr/bin/env python3
"""Compose the primary V22 overlay video.

The primary view is two panes:
- left: image-space hand skeleton overlay driven by the metric hand state;
- right: 3D world-coordinate head/camera + hand visualization.

Semantic rows from D9b are rendered as a bottom subtitle band on every frame.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ffprobe_count(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def build_frame_index(rows: list[dict[str, Any]], frame_count: int) -> list[list[dict[str, Any]]]:
    index: list[list[dict[str, Any]]] = [[] for _ in range(frame_count)]
    for row in rows:
        try:
            start = int(row.get("start_frame"))
            end = int(row.get("end_frame"))
        except (TypeError, ValueError):
            continue
        start = max(0, min(frame_count, start))
        end = max(0, min(frame_count, end))
        if end <= start:
            continue
        for frame_idx in range(start, end):
            index[frame_idx].append(row)
    return index


def resize_to_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x0 = max(0, (rw - width) // 2)
    y0 = max(0, (rh - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width]


def wrap_chars(text: str, max_chars: int, max_lines: int) -> list[str]:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:max_chars]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def hand_payload(row: dict[str, Any], side: str) -> dict[str, Any]:
    per_hand = row.get("per_hand") or row.get("hands") or row.get("hand_annotations")
    if isinstance(per_hand, dict):
        value = per_hand.get(side)
        return value if isinstance(value, dict) else {}
    if isinstance(per_hand, list):
        for item in per_hand:
            if isinstance(item, dict) and str(item.get("side")) == side:
                return item
    return {}


def format_hand(row: dict[str, Any], side: str) -> str:
    hand = hand_payload(row, side)
    prefix = "L" if side == "left" else "R"
    if not hand:
        return f"{prefix}: no semantic hand row"
    action = str(hand.get("action_state") or hand.get("action") or "unknown")
    obj = str(hand.get("object") or hand.get("object_name") or "unknown object")
    rigidity = str(hand.get("object_rigidity") or hand.get("rigidity") or "unknown rigidity")
    assembly = str(hand.get("object_assembly") or hand.get("is_assembly") or "unknown assembly")
    contact = str(hand.get("contact_location") or hand.get("contact") or "contact unresolved")
    conf = hand.get("confidence")
    conf_text = f" conf={float(conf):.2f}" if isinstance(conf, (int, float)) else ""
    return f"{prefix}: {action} | obj={obj} | rigid={rigidity} | assembly={assembly} | contact={contact}{conf_text}"


def draw_label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.rectangle(canvas, (x, y), (x + 520, y + 34), (0, 0, 0), -1)
    cv2.addWeighted(canvas, 0.92, canvas, 0.08, 0.0, dst=canvas)
    cv2.putText(canvas, text, (x + 10, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)


def draw_subtitle_band(canvas: np.ndarray, row: dict[str, Any] | None, frame_idx: int, frame_count: int, row_count: int, band_h: int) -> None:
    h, w = canvas.shape[:2]
    y0 = max(0, h - band_h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.64, canvas, 0.36, 0.0, dst=canvas)
    if row is None:
        header = f"frame {frame_idx:06d}/{frame_count} | D9b semantic rows {row_count} | no active row"
        lines = ["L: no action row", "R: no action row", "object/contact/rigidity unavailable for this frame"]
    else:
        header = f"frame {frame_idx:06d}/{frame_count} | {row.get('clip_id')} | frames {row.get('start_frame')}-{row.get('end_frame')} | source={row.get('source')}"
        caption = str(row.get("caption") or "")
        left = format_hand(row, "left")
        right = format_hand(row, "right")
        provenance = str(row.get("provenance") or row.get("grounding_status") or "")
        lines = [caption, left, right, provenance]
    cv2.putText(canvas, header[:180], (16, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 225, 255), 1, cv2.LINE_AA)
    y = y0 + 54
    for raw in lines:
        for line in wrap_chars(raw, max(58, w // 32), 2):
            cv2.putText(canvas, line[:190], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 250, 250), 1, cv2.LINE_AA)
            y += 24
            if y > h - 12:
                return


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    raw_manifest = load_json(args.raw_frame_manifest or (run_root / "input" / "raw_frame_manifest" / "manifest.json"))
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    if not frames:
        raise RuntimeError("raw frame manifest contains no frames")
    frame_count = len(frames)
    fps = float(raw_manifest.get("fps") or (raw_manifest.get("video") or {}).get("fps") or 30.0)
    caption_stage_path = args.captioning_stage or (run_root / "state" / "semantic_clips" / "v22_captioning_stage.json")
    caption_stage = load_json(caption_stage_path) if caption_stage_path.exists() else {"semantic_rows": [], "status": "missing"}
    rows = caption_stage.get("semantic_rows") if isinstance(caption_stage.get("semantic_rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    frame_index = build_frame_index(rows, frame_count)

    hand_overlay = args.hand_overlay or (run_root / "renders" / "v22_hybrid_hand_overlay.mp4")
    if not hand_overlay.exists():
        hand_overlay = run_root / "renders" / "v22_wilor_hand_overlay.mp4"
    world_video = args.world_video or (run_root / "renders" / "v22_world_head_hand_3d.mp4")
    if not hand_overlay.exists():
        raise RuntimeError(f"missing hand overlay for primary composition: {hand_overlay}")
    if not world_video.exists():
        raise RuntimeError(f"missing 3D world video for primary composition: {world_video}")

    output = args.output or (run_root / "renders" / "v22_overlay.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    width = int(args.width)
    height = int(args.height)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    pane_w = width // 2
    subtitle_h = int(args.subtitle_height)
    if subtitle_h % 2:
        subtitle_h += 1
    pane_h = height - subtitle_h
    if pane_h <= 0:
        raise RuntimeError(f"invalid primary overlay layout: height={height}, subtitle_height={subtitle_h}")

    hand_cap = cv2.VideoCapture(str(hand_overlay))
    world_cap = cv2.VideoCapture(str(world_video))
    if not hand_cap.isOpened():
        raise RuntimeError(f"failed to open hand overlay: {hand_overlay}")
    if not world_cap.isOpened():
        raise RuntimeError(f"failed to open world video: {world_video}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output video writer: {output}")
    active_frames = 0
    try:
        for i, meta in enumerate(frames):
            ok_l, left = hand_cap.read()
            ok_r, right = world_cap.read()
            if not ok_l or left is None:
                raise RuntimeError(f"hand overlay ended before frame {i}: {hand_overlay}")
            if not ok_r or right is None:
                raise RuntimeError(f"3D world video ended before frame {i}: {world_video}")
            left_pane = resize_to_cover(left, pane_w, pane_h)
            right_pane = resize_to_cover(right, pane_w, pane_h)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:pane_h, :pane_w] = left_pane
            canvas[:pane_h, pane_w:] = right_pane
            draw_label(canvas, "image overlay: projected hand skeleton", 10, 10)
            draw_label(canvas, "3D world: side-rear view looking forward", pane_w + 10, 10)
            frame_idx = int(meta.get("frame_idx", i))
            active = frame_index[i]
            row = active[0] if active else None
            if row is not None:
                active_frames += 1
            draw_subtitle_band(canvas, row, frame_idx, frame_count, len(rows), subtitle_h)
            writer.write(canvas)
    finally:
        writer.release()
        hand_cap.release()
        world_cap.release()

    report = {
        "schema": "v22_primary_overlay_render.v0",
        "status": "ok" if rows else str(caption_stage.get("status") or "missing_semantic_rows"),
        "method": "render_v22_primary_overlay",
        "claim_scope": "Primary display overlay composed from the metric hand skeleton overlay and the 3D side-rear world head/hand render. Subtitle text is driven by D9b semantic rows; semantic contact/object fields are visual-language annotations unless backed by object geometry.",
        "run_root": str(run_root),
        "hand_overlay": str(hand_overlay),
        "world_video": str(world_video),
        "captioning_stage": str(caption_stage_path),
        "output_video": str(output),
        "frame_count": frame_count,
        "video_frame_count": ffprobe_count(output),
        "fps": fps,
        "render_size": [width, height],
        "semantic_row_count": len(rows),
        "active_subtitle_frame_count": active_frames,
        "timeline_coverage_fraction": active_frames / frame_count if frame_count else None,
    }
    write_json(args.report_json or (output.parent / "v22_primary_overlay_report.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hand-overlay", type=Path, default=None)
    parser.add_argument("--world-video", type=Path, default=None)
    parser.add_argument("--captioning-stage", type=Path, default=None)
    parser.add_argument("--raw-frame-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--subtitle-height", type=int, default=180)
    return parser.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
