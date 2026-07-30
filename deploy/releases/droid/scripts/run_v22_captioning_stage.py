#!/usr/bin/env python3
"""Build D9b semantic clip/caption rows for a V22 annotation run.

This stage is source-driven: it consumes existing task/action/caption JSON when
provided. Without a caption source it writes an explicit no-source artifact
instead of inventing captions from filenames or visual guesses.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def extract_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("actions"), list):
        return [row for row in payload["actions"] if isinstance(row, dict)]
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    actions: list[dict[str, Any]] = []
    for task in tasks:
        if isinstance(task, dict) and isinstance(task.get("actions"), list):
            actions.extend(row for row in task["actions"] if isinstance(row, dict))
    return actions


def frame_count_and_fps(raw_manifest: dict[str, Any]) -> tuple[int, float]:
    frame_count = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    fps = finite_positive(raw_manifest.get("fps"))
    video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
    if fps is None:
        fps = finite_positive(video.get("fps"))
    if fps is None:
        raise RuntimeError("raw frame manifest lacks fps")
    return frame_count, fps


def build_captions(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    raw_manifest_path = args.raw_frame_manifest or (run_root / "input" / "raw_frame_manifest" / "manifest.json")
    output = args.output or (run_root / "state" / "semantic_clips" / "v22_captioning_stage.json")
    raw_manifest = load_json(raw_manifest_path)
    frame_count, fps = frame_count_and_fps(raw_manifest)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    source = args.actions_json or args.captions_json
    status = "source_absent_no_caption_rows"
    source_kind = None
    if source is not None:
        source_kind = "actions_json" if args.actions_json else "captions_json"
        payload = load_json(source)
        actions = extract_actions(payload)
        status = "ok"
        segment_frames = max(1, int(round(float(args.max_clip_s) * fps)))
        for action_idx, action in enumerate(actions):
            start_raw = action.get("start_frame", action.get("start", action.get("frame_start", -1)))
            end_raw = action.get("end_frame", action.get("end", action.get("frame_end", -1)))
            try:
                start_frame = int(start_raw)
                end_frame = int(end_raw)
            except (TypeError, ValueError):
                continue
            caption = str(action.get("description") or action.get("caption") or action.get("action") or "").strip()
            if not caption or end_frame <= start_frame:
                continue
            start_frame = max(0, min(frame_count, start_frame))
            end_frame = max(0, min(frame_count, end_frame))
            if end_frame <= start_frame:
                continue
            clip_start = start_frame
            part = 0
            while clip_start < end_frame:
                clip_end = min(end_frame, clip_start + segment_frames)
                row = {
                    "clip_id": f"action_{action_idx:04d}_{part:03d}",
                    "start_frame": int(clip_start),
                    "end_frame": int(clip_end),
                    "start_s": float(clip_start / fps),
                    "end_s": float(clip_end / fps),
                    "duration_s": float((clip_end - clip_start) / fps),
                    "caption": caption,
                    "confidence": action.get("confidence"),
                    "source": f"{source_kind}:{source}",
                    "grounding_status": "aligned_from_existing_task_action_frames",
                    "evidence_frames": [int(clip_start), int(max(clip_start, clip_end - 1))],
                }
                rows.append(row)
                events.append({"event": "semantic_clip_from_caption_source", **row})
                clip_start = clip_end
                part += 1
        if not rows:
            status = "source_loaded_no_valid_caption_rows"
    payload_out = {
        "schema": "v22_captioning_stage.v0",
        "status": status,
        "method": "adapt_existing_action_or_caption_json_to_semantic_clips",
        "run_root": str(run_root),
        "raw_frame_manifest": str(raw_manifest_path),
        "source": str(source) if source else None,
        "source_kind": source_kind,
        "frame_count": frame_count,
        "fps": fps,
        "semantic_rows": rows,
        "caption_events": events,
        "summary": {
            "semantic_clip_count": len(rows),
            "caption_event_count": len(events),
            "timeline_coverage_fraction": sum(row["end_frame"] - row["start_frame"] for row in rows) / frame_count if frame_count > 0 else None,
        },
        "claim_scope": "D9b captioning stage. Captions are source-backed only; no visual caption is hallucinated when source captions are absent.",
        "elapsed_s": float(time.time() - started),
    }
    write_json(output, payload_out)
    print(json.dumps({"status": status, "semantic_clip_count": len(rows), "output": str(output)}, indent=2))
    return payload_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, default=None)
    parser.add_argument("--actions-json", type=Path, default=None)
    parser.add_argument("--captions-json", type=Path, default=None)
    parser.add_argument("--max-clip-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_captions(parse_args())
