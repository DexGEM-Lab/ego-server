#!/usr/bin/env python3
"""Render a full-length semantic subtitle video from V22 D9b semantic rows.

The video is driven by source-backed temporal segments emitted by
run_v22_captioning_stage.py. When no source rows exist, the renderer still writes
an inspectable full-timeline no-source video, but the report status remains
source_absent_no_caption_rows and strict clean-completion must reject it for a
semantic-subtitle deliverable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ffprobe_count(path: Path) -> int | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:max_chars]
    if current:
        lines.append(current)
    return lines[:3]


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
    return f"{prefix}: {action} | obj={obj} | rigid={rigidity} | assembly={assembly} | contact={contact}"


def row_lines(row: dict[str, Any]) -> list[str]:
    lines = [str(row.get("caption") or row.get("action") or "")]
    lines.append(format_hand(row, "left"))
    lines.append(format_hand(row, "right"))
    provenance = str(row.get("provenance") or row.get("grounding_status") or "")
    if provenance:
        lines.append(provenance)
    return [line for line in lines if line]


def draw_subtitle(frame, lines: list[str], header: str, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    box_h = 34 + 28 * max(1, len(lines))
    y0 = max(0, h - box_h - 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0.0, dst=frame)
    cv2.putText(frame, header[:120], (16, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 230, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(frame, line[:160], (16, y0 + 54 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    raw_manifest = load_json(args.raw_frame_manifest or (run_root / "input" / "raw_frame_manifest" / "manifest.json"))
    caption_stage_path = args.captioning_stage or (run_root / "state" / "semantic_clips" / "v22_captioning_stage.json")
    caption_stage = load_json(caption_stage_path)
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    if not frames:
        raise RuntimeError("raw frame manifest contains no frames")
    frame_count = len(frames)
    fps = float(raw_manifest.get("fps") or (raw_manifest.get("video") or {}).get("fps") or 30.0)
    rows = caption_stage.get("semantic_rows") if isinstance(caption_stage.get("semantic_rows"), list) else []
    frame_index = build_frame_index(rows, frame_count)
    first_img = cv2.imread(str(frames[0].get("rgb") or frames[0].get("raw_frame_path")))
    if first_img is None:
        raise RuntimeError("failed to read first raw frame")
    height, width = first_img.shape[:2]
    output = args.output or (run_root / "renders" / "v22_semantic_subtitle.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    active_frames = 0
    try:
        for i, meta in enumerate(frames):
            image_path = Path(str(meta.get("rgb") or meta.get("raw_frame_path")))
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"failed to read frame: {image_path}")
            active = frame_index[i]
            if active:
                active_frames += 1
                row = active[0]
                header = f"D9b semantic segment {row.get('clip_id')}  frames {row.get('start_frame')}-{row.get('end_frame')}"
                lines = []
                for line in row_lines(row):
                    lines.extend(wrap_text(line, max(32, width // 22)))
                draw_subtitle(img, lines or ["(empty semantic row)"], header, (255, 255, 255))
            else:
                if rows:
                    draw_subtitle(img, ["no active semantic segment for this frame"], "D9b timeline gap", (180, 210, 255))
                else:
                    draw_subtitle(img, ["semantic/caption source absent; no source-backed subtitle rows"], "D9b no-source state", (165, 190, 255))
            cv2.putText(img, f"frame {i:06d}/{frame_count} | semantic rows {len(rows)}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
    report_status = "ok" if rows else str(caption_stage.get("status") or "source_absent_no_caption_rows")
    report = {
        "schema": "v22_semantic_subtitle_render.v0",
        "status": report_status,
        "method": "render_v22_semantic_subtitle_video",
        "claim_scope": "Full-timeline subtitle video driven by D9b temporal semantic rows. Object/contact/rigidity wording is semantic annotation unless separately backed by object geometry. If no rows exist, the output is a no-source visualization and not a completed semantic subtitle deliverable.",
        "run_root": str(run_root),
        "captioning_stage": str(caption_stage_path),
        "output_video": str(output),
        "frame_count": frame_count,
        "video_frame_count": ffprobe_count(output),
        "fps": fps,
        "semantic_row_count": len(rows),
        "active_subtitle_frame_count": active_frames,
        "timeline_coverage_fraction": active_frames / frame_count if frame_count else None,
    }
    write_json(args.report_json or (output.parent / "v22_semantic_subtitle_report.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, default=None)
    parser.add_argument("--captioning-stage", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
