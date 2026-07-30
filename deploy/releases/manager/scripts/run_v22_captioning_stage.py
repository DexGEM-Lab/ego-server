#!/usr/bin/env python3
"""Build D9b semantic clip/caption rows for a V22 annotation run.

Priority order:
1. supplied task/action/caption JSON sidecars;
2. a source-backed semantic review file under `state/semantic_clips/`;
3. explicit no-source artifact.

This stage only normalizes a scripted or supplied semantic source into timeline
rows; it does not infer object geometry or physical contact from filenames.
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


def extract_review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("semantic_rows", "segments", "clips", "annotations"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def frame_count_and_fps(raw_manifest: dict[str, Any]) -> tuple[int, float]:
    frame_count = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    fps = finite_positive(raw_manifest.get("fps"))
    video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
    if fps is None:
        fps = finite_positive(video.get("fps"))
    if fps is None:
        raise RuntimeError("raw frame manifest lacks fps")
    return frame_count, fps


def parse_frame_interval(row: dict[str, Any], frame_count: int, fps: float) -> tuple[int, int] | None:
    start_raw = row.get("start_frame", row.get("start", row.get("frame_start")))
    end_raw = row.get("end_frame", row.get("end", row.get("frame_end")))
    if start_raw is None and row.get("start_s") is not None:
        start_raw = int(round(float(row.get("start_s")) * fps))
    if end_raw is None and row.get("end_s") is not None:
        end_raw = int(round(float(row.get("end_s")) * fps))
    try:
        start_frame = int(start_raw)
        end_frame = int(end_raw)
    except (TypeError, ValueError):
        return None
    start_frame = max(0, min(frame_count, start_frame))
    end_frame = max(0, min(frame_count, end_frame))
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame


def row_caption(row: dict[str, Any]) -> str:
    caption = str(row.get("caption") or row.get("description") or row.get("action") or "").strip()
    if caption:
        return caption
    per_hand = row.get("per_hand") or row.get("hands") or {}
    if isinstance(per_hand, dict):
        parts = []
        for side in ("left", "right"):
            hand = per_hand.get(side)
            if isinstance(hand, dict):
                action = str(hand.get("action_state") or hand.get("action") or "unknown")
                obj = str(hand.get("object") or "object")
                parts.append(f"{side} hand {action} {obj}")
        if parts:
            return "; ".join(parts)
    return ""


def normalize_rows(raw_rows: list[dict[str, Any]], *, frame_count: int, fps: float, max_clip_s: float, source_label: str, event_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    segment_frames = max(1, int(round(float(max_clip_s) * fps)))
    for action_idx, raw in enumerate(raw_rows):
        interval = parse_frame_interval(raw, frame_count, fps)
        if interval is None:
            continue
        start_frame, end_frame = interval
        caption = row_caption(raw)
        if not caption:
            continue
        clip_start = start_frame
        part = 0
        while clip_start < end_frame:
            clip_end = min(end_frame, clip_start + segment_frames)
            row = dict(raw)
            row.update(
                {
                    "clip_id": str(raw.get("clip_id") or f"semantic_{action_idx:04d}_{part:03d}"),
                    "start_frame": int(clip_start),
                    "end_frame": int(clip_end),
                    "start_s": float(clip_start / fps),
                    "end_s": float(clip_end / fps),
                    "duration_s": float((clip_end - clip_start) / fps),
                    "caption": caption,
                    "source": str(raw.get("source") or source_label),
                    "grounding_status": str(raw.get("grounding_status") or "visual_semantic_review"),
                    "evidence_frames": raw.get("evidence_frames") if isinstance(raw.get("evidence_frames"), list) else [int(clip_start), int(max(clip_start, clip_end - 1))],
                }
            )
            if "provenance" not in row:
                row["provenance"] = "semantic fields are source-backed video-understanding estimates; object/contact words are not object-pose or nonpenetration proof"
            rows.append(row)
            events.append({"event": event_name, **row})
            clip_start = clip_end
            part += 1
    return rows, events


def copy_semantic_review_into_run_root(source: Path, default_path: Path) -> Path:
    source = source.expanduser().resolve()
    default_path = default_path.resolve()
    if source == default_path:
        return default_path
    payload = load_json(source)
    write_json(default_path, payload)
    return default_path


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
    source_kind = None
    method = "adapt_existing_action_or_caption_json_to_semantic_clips"
    status = "source_absent_no_caption_rows"
    semantic_dir = run_root / "state" / "semantic_clips"
    default_semantic_review_json = semantic_dir / "v22_semantic_review.json"
    semantic_review_json: Path | None
    if args.semantic_review_json:
        source_path = args.semantic_review_json.expanduser().resolve()
        if source_path.is_relative_to(run_root):
            semantic_review_json = source_path
        else:
            semantic_review_json = copy_semantic_review_into_run_root(source_path, default_semantic_review_json)
    else:
        candidates = [
            semantic_dir / "v22_cosmos_semantic_review.json",
            default_semantic_review_json,
        ]
        semantic_review_json = next((path for path in candidates if path.exists()), None)

    if source is not None:
        source_kind = "actions_json" if args.actions_json else "captions_json"
        payload = load_json(source)
        raw_rows = extract_actions(payload)
        rows, events = normalize_rows(raw_rows, frame_count=frame_count, fps=fps, max_clip_s=args.max_clip_s, source_label=f"{source_kind}:{source}", event_name="semantic_clip_from_caption_source")
        status = "ok" if rows else "source_loaded_no_valid_caption_rows"
    elif semantic_review_json is not None and semantic_review_json.exists():
        source_kind = "semantic_review_json"
        payload = load_json(semantic_review_json)
        raw_rows = extract_review_rows(payload)
        rows, events = normalize_rows(raw_rows, frame_count=frame_count, fps=fps, max_clip_s=args.max_clip_s, source_label=f"semantic_review:{semantic_review_json}", event_name="semantic_clip_from_semantic_review")
        method = "normalize_semantic_review_to_timeline_rows"
        status = "ok" if rows else "semantic_review_no_valid_rows"
        source = semantic_review_json

    payload_out = {
        "schema": "v22_captioning_stage.v1",
        "status": status,
        "method": method,
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
        "claim_scope": "D9b captioning stage. Rows come from supplied semantic sources or scripted Cosmos video-understanding review. Object/contact/rigidity wording is semantic annotation unless separately backed by object geometry.",
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
    parser.add_argument("--semantic-review-json", type=Path, default=None)
    parser.add_argument("--max-clip-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_captions(parse_args())
