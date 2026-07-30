#!/usr/bin/env python3
"""Create a scripted D9b semantic caption source with Cosmos video understanding.

This stage follows the remote Cosmos gallery/boundary method, but it performs no
agent launch and no retry sleep. Each model call is a single OpenAI-compatible
HTTP request; an invalid or failed response is a stage error.
"""
from __future__ import annotations

import argparse
import csv
import functools
import http.server
import json
import math
import re
import socketserver
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINTS = {
    "cosmos2": "http://127.0.0.1:8000/v1",
    "cosmos3": "http://127.0.0.1:8001/v1",
}

HAND_KEYS = ("left_hand", "right_hand")
HAND_DEFAULT = {
    "in_frame": "unknown",
    "contact": "unknown",
    "object": "unknown",
    "rigidity": "unknown",
    "assembly": "unknown",
    "contact_location": "unknown",
}

COARSE_GALLERY_PROMPT = """Return ONLY one minified JSON object. Do not use markdown fences.

You are given an image gallery sampled from one egocentric manipulation video at __SAMPLE_FPS__ fps. Each image is tagged with image_index, original-video frame_idx, and original-video time_sec. Analyze every image independently at its timestamp, but keep object names and labels consistent across the gallery.

For EACH image, answer these fields for BOTH left_hand and right_hand:
1. in_frame: yes|no|unknown
2. contact: yes|no|unknown
3. object: short consistent noun or none|unknown
4. rigidity: rigid|flexible|mixed|unknown
5. assembly: yes|no|unknown
6. contact_location: short consistent phrase or none|unknown
Also include understanding: <=18 words describing the image at that timestamp.

Return this exact schema:
{"items":[{"image_index":integer,"frame_idx":integer,"time_sec":number,"left_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"right_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"understanding":"<=18 words"}]}.

Critical requirements:
- Return exactly one item per image listed in IMAGE_GRID.
- Preserve image_index/frame_idx/time_sec from IMAGE_GRID.
- Do not summarize the gallery into segments.
- If a hand is not visible or not touching anything, use object=none and contact_location=none.
- Use consistent object names across all images; avoid changing synonyms for the same object.

IMAGE_GRID(format: image_index@frame_idx@time_sec):
__IMAGE_GRID__

Stop immediately after the JSON object."""

SINGLE_IMAGE_PROMPT = """Return ONLY one minified JSON object. Do not use markdown fences.

Analyze this single egocentric image sampled from a video. The image tag is image_index=__IMAGE_INDEX__, original frame_idx=__FRAME_IDX__, original time_sec=__TIME_SEC__.

For BOTH left_hand and right_hand answer:
1. in_frame: yes|no|unknown
2. contact: yes|no|unknown
3. object: short consistent noun or none|unknown
4. rigidity: rigid|flexible|mixed|unknown
5. assembly: yes|no|unknown
6. contact_location: short consistent phrase or none|unknown
Also include understanding: <=18 words.

Return this exact schema:
{"items":[{"image_index":__IMAGE_INDEX__,"frame_idx":__FRAME_IDX__,"time_sec":__TIME_SEC__,"left_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"right_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"understanding":"<=18 words"}]}.

Stop immediately after the JSON object."""

BOUNDARY_PROMPT = """Return ONLY one minified JSON object. Do not use markdown fences.

You are given the ORIGINAL full video. A sampled image-gallery pass found that the hand-object annotation changed somewhere between two sampled images. Inspect the original video and localize the first frame/time where the NEXT annotation becomes true.

Focus range:
- previous_sample_image_index=__PREV_INDEX__
- previous_sample_frame_idx=__PREV_FRAME__
- previous_sample_time_sec=__PREV_TIME__
- next_sample_image_index=__NEXT_INDEX__
- next_sample_frame_idx=__NEXT_FRAME__
- next_sample_time_sec=__NEXT_TIME__
- source_fps=__SOURCE_FPS__
- source_frame_count=__SOURCE_FRAME_COUNT__

Previous annotation at previous sample:
__PREV_LABEL__

Next annotation at next sample:
__NEXT_LABEL__

Return the earliest original frame inside the focus range where the state matches the NEXT annotation rather than the PREVIOUS annotation. If exact localization is visually ambiguous, return your best estimate inside the range and set confidence accordingly.

Return this exact schema:
{"change":{"change_index":__CHANGE_INDEX__,"change_frame_idx":integer,"change_time_sec":number,"confidence":"high|medium|low","evidence":"<=20 words"}}.

Stop immediately after the JSON object."""


def run_cmd(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def ffprobe(path: Path) -> dict[str, Any]:
    data = json.loads(run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,duration,nb_frames,width,height",
        "-of", "json", str(path),
    ]).stdout)
    stream = data["streams"][0]

    def parse_rate(value: str | None) -> float | None:
        if not value or value == "0/0":
            return None
        num, den = value.split("/", 1)
        return float(num) / float(den)

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    if not fps or fps <= 0:
        raise RuntimeError(f"ffprobe did not return a valid fps for {path}")
    duration = float(stream.get("duration") or 0.0)
    frame_count = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else int(round(duration * fps))
    return {
        "fps": float(fps),
        "duration_sec": float(duration),
        "frame_count": int(frame_count),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def sample_grid(meta: dict[str, Any], sample_fps: float) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise RuntimeError("sample_fps must be positive")
    points: list[dict[str, Any]] = []
    index = 0
    t = 0.0
    duration = float(meta["duration_sec"])
    fps = float(meta["fps"])
    frame_count = int(meta["frame_count"])
    while t < duration - 1e-9:
        frame_idx = min(frame_count - 1, int(round(t * fps)))
        points.append({"image_index": index, "frame_idx": frame_idx, "time_sec": round(frame_idx / fps, 6)})
        index += 1
        t = index / sample_fps
    if not points and frame_count > 0:
        points.append({"image_index": 0, "frame_idx": 0, "time_sec": 0.0})
    return points


def extract_frame(video: Path, out: Path, frame_idx: int, width: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{int(frame_idx)}),scale={int(width)}:-2"
    run_cmd(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", vf, "-frames:v", "1", "-q:v", "2", str(out)], timeout=900)


def extract_gallery(video: Path, media_dir: Path, grid: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    frames_dir = media_dir / "gallery_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gallery: list[dict[str, Any]] = []
    for point in grid:
        image = frames_dir / f"img_{point['image_index']:04d}_frame_{point['frame_idx']:06d}_t_{float(point['time_sec']):.3f}.jpg"
        extract_frame(video, image, int(point["frame_idx"]), width)
        row = dict(point)
        row["path"] = str(image)
        gallery.append(row)
    return gallery


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return


def start_server(directory: Path) -> tuple[socketserver.ThreadingTCPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def http_text(url: str, timeout: int) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def get_model(api_base: str, timeout: int) -> str:
    data = json.loads(http_text(f"{api_base}/models", timeout))
    return str(data["data"][0]["id"])


def parse_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return {"items": payload}
        if isinstance(payload, dict):
            return payload
    return None


def post_json_once(api_base: str, model: str, content: list[dict[str, Any]], raw_path: Path, max_tokens: int, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 0.8,
    }
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except Exception as exc:
        record = {"http_status": 0, "client_wall_sec": time.perf_counter() - started, "error": repr(exc)}
        raw_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"cosmos_request_failed:{raw_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    content_text = ((choice.get("message") or {}).get("content") or "") if isinstance(choice, dict) else ""
    record = {
        "http_status": status,
        "client_wall_sec": time.perf_counter() - started,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "usage": data.get("usage") if isinstance(data, dict) else None,
        "content": content_text,
        "parsed": parse_json_object(content_text),
        "raw_response": data if data is not None else text,
    }
    raw_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if status != 200 or not record["parsed"]:
        raise RuntimeError(f"cosmos_invalid_response:{raw_path}: http={status} finish={record.get('finish_reason')}")
    return record


def norm_yn(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return "yes" if int(value) == 1 else "no"
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "visible", "present"}:
        return "yes"
    if text in {"no", "n", "false", "not visible", "absent", "none"}:
        return "no"
    return "unknown"


def norm_choice(value: Any, allowed: set[str]) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else "unknown"


def clean_text(value: Any, default: str, limit: int) -> str:
    text = str(value if value is not None else default).strip().lower()
    return (text or default)[:limit]


def normalize_hand(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raw = {}
    contact = norm_yn(raw.get("contact"))
    obj = clean_text(raw.get("object"), "unknown", 80)
    loc = clean_text(raw.get("contact_location"), "unknown", 120)
    if contact == "no":
        obj = "none" if obj == "unknown" else obj
        loc = "none" if loc == "unknown" else loc
    return {
        "in_frame": norm_yn(raw.get("in_frame")),
        "contact": contact,
        "object": obj,
        "rigidity": norm_choice(raw.get("rigidity"), {"rigid", "flexible", "mixed", "unknown"}),
        "assembly": norm_yn(raw.get("assembly")),
        "contact_location": loc,
    }


def image_grid_text(gallery: list[dict[str, Any]]) -> str:
    return "; ".join(f"{row['image_index']}@{row['frame_idx']}@{float(row['time_sec']):.6f}" for row in gallery)


def image_url_content(image: Path, media_dir: Path, port: int) -> dict[str, Any]:
    rel = image.relative_to(media_dir).as_posix()
    return {"type": "image_url", "image_url": {"url": f"http://127.0.0.1:{port}/{urllib.parse.quote(rel)}"}}


def gallery_content(gallery: list[dict[str, Any]], media_dir: Path, port: int, prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for row in gallery:
        content.append({"type": "text", "text": f"Image image_index={row['image_index']} frame_idx={row['frame_idx']} time_sec={float(row['time_sec']):.6f}"})
        content.append(image_url_content(Path(row["path"]), media_dir, port))
    content.append({"type": "text", "text": prompt})
    return content


def single_image_content(row: dict[str, Any], media_dir: Path, port: int) -> list[dict[str, Any]]:
    prompt = (SINGLE_IMAGE_PROMPT
        .replace("__IMAGE_INDEX__", str(row["image_index"]))
        .replace("__FRAME_IDX__", str(row["frame_idx"]))
        .replace("__TIME_SEC__", f"{float(row['time_sec']):.6f}"))
    return [image_url_content(Path(row["path"]), media_dir, port), {"type": "text", "text": prompt}]


def normalize_gallery_items(parsed: dict[str, Any] | None, gallery: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    raw = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        raw = []
    by_index: dict[int, dict[str, Any]] = {}
    grid_by_index = {int(row["image_index"]): row for row in gallery}
    grid_by_frame = {int(row["frame_idx"]): row for row in gallery}
    valid = set(grid_by_index)
    for item in raw:
        if not isinstance(item, dict):
            continue
        image_index = item.get("image_index", item.get("index"))
        frame_idx = item.get("frame_idx", item.get("frame"))
        try:
            image_index = int(image_index)
        except Exception:
            image_index = None
        try:
            frame_idx = int(frame_idx)
        except Exception:
            frame_idx = None
        if image_index is None and frame_idx in grid_by_frame:
            image_index = int(grid_by_frame[int(frame_idx)]["image_index"])
        if image_index not in valid and len(gallery) == 1:
            image_index = int(gallery[0]["image_index"])
        if image_index not in valid:
            continue
        grid = grid_by_index[int(image_index)]
        row: dict[str, Any] = {
            "image_index": int(image_index),
            "frame_idx": int(grid["frame_idx"]),
            "time_sec": float(grid["time_sec"]),
            "understanding": str(item.get("understanding", item.get("summary", "")))[:240],
        }
        for hand_key in HAND_KEYS:
            hand = normalize_hand(item.get(hand_key))
            for key, value in hand.items():
                row[f"{hand_key}_{key}"] = value
        by_index[int(image_index)] = row
    ordered = [int(row["image_index"]) for row in gallery]
    rows = [by_index[i] for i in ordered if i in by_index]
    missing = [i for i in ordered if i not in by_index]
    return rows, missing


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(f"{hand}_{key}", "unknown") for hand in HAND_KEYS for key in HAND_DEFAULT)


def detect_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for prev, nxt in zip(rows, rows[1:]):
        if row_signature(prev) == row_signature(nxt):
            continue
        changes.append({
            "change_index": len(changes) + 1,
            "prev": prev,
            "next": nxt,
            "prev_image_index": int(prev["image_index"]),
            "next_image_index": int(nxt["image_index"]),
            "prev_frame_idx": int(prev["frame_idx"]),
            "next_frame_idx": int(nxt["frame_idx"]),
            "prev_time_sec": float(prev["time_sec"]),
            "next_time_sec": float(nxt["time_sec"]),
        })
    return changes


def compact_label(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_index": row.get("image_index"),
        "frame_idx": row.get("frame_idx"),
        "time_sec": row.get("time_sec"),
        "left_hand": {key: row.get(f"left_hand_{key}", "unknown") for key in HAND_DEFAULT},
        "right_hand": {key: row.get(f"right_hand_{key}", "unknown") for key in HAND_DEFAULT},
    }


def boundary_content(video: Path, media_dir: Path, port: int, prompt: str) -> list[dict[str, Any]]:
    rel = video.relative_to(media_dir).as_posix()
    url = f"http://127.0.0.1:{port}/{urllib.parse.quote(rel)}"
    return [{"type": "video_url", "video_url": {"url": url}}, {"type": "text", "text": prompt}]


def parse_boundary(parsed: dict[str, Any] | None, change: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    raw = parsed.get("change") if isinstance(parsed, dict) else None
    if not isinstance(raw, dict):
        raw = parsed if isinstance(parsed, dict) else {}
    lo = int(change["prev_frame_idx"])
    hi = int(change["next_frame_idx"])
    try:
        frame = int(round(float(raw.get("change_frame_idx"))))
    except Exception:
        frame = int(round((lo + hi) / 2))
    frame = max(lo, min(hi, frame))
    try:
        t = float(raw.get("change_time_sec"))
    except Exception:
        t = frame / float(meta["fps"])
    t = max(float(change["prev_time_sec"]), min(float(change["next_time_sec"]), t))
    confidence = str(raw.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "change_index": int(change["change_index"]),
        "prev_image_index": int(change["prev_image_index"]),
        "next_image_index": int(change["next_image_index"]),
        "range_frame_start": lo,
        "range_frame_end": hi,
        "range_time_start": float(change["prev_time_sec"]),
        "range_time_end": float(change["next_time_sec"]),
        "change_frame_idx": int(frame),
        "change_time_sec": round(float(t), 6),
        "confidence": confidence,
        "evidence": str(raw.get("evidence", ""))[:240],
        "previous_label": compact_label(change["prev"]),
        "next_label": compact_label(change["next"]),
    }


def label_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_hand": {key: row.get(f"left_hand_{key}", "unknown") for key in HAND_DEFAULT},
        "right_hand": {key: row.get(f"right_hand_{key}", "unknown") for key in HAND_DEFAULT},
        "understanding": row.get("understanding", ""),
    }


def segment_signature(segment: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(segment[hand].get(key, "unknown") for hand in HAND_KEYS for key in HAND_DEFAULT)


def merge_adjacent(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in segments:
        if segment["end_sec"] <= segment["start_sec"]:
            continue
        if out and segment_signature(out[-1]) == segment_signature(segment):
            out[-1]["end_sec"] = segment["end_sec"]
            if segment.get("understanding") and segment["understanding"] not in out[-1].get("understanding", ""):
                out[-1]["understanding"] = (out[-1].get("understanding", "") + "; " + segment["understanding"])[:240]
        else:
            out.append(dict(segment))
    return out


def build_segments(rows: list[dict[str, Any]], refinements: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_next = {int(row["next_image_index"]): row for row in refinements}
    segments: list[dict[str, Any]] = []
    current = rows[0]
    start = 0.0
    for index in range(1, len(rows)):
        if row_signature(rows[index - 1]) == row_signature(rows[index]):
            continue
        refinement = by_next.get(int(rows[index]["image_index"]))
        boundary = float(refinement["change_time_sec"]) if refinement else float(rows[index]["time_sec"])
        boundary = max(start, min(float(duration), boundary))
        segments.append({"start_sec": start, "end_sec": boundary, **label_from_row(current), "source": "cosmos_gallery_until_boundary"})
        start = boundary
        current = rows[index]
    segments.append({"start_sec": start, "end_sec": float(duration), **label_from_row(current), "source": "cosmos_gallery_after_last_boundary"})
    return merge_adjacent(segments)


def hand_action(hand: dict[str, Any]) -> str:
    if hand.get("in_frame") == "no":
        return "no_action"
    if hand.get("contact") == "yes":
        return "contacting_or_operating_object"
    if hand.get("in_frame") == "yes":
        return "visible_no_contact"
    return "unknown"


def caption_from_segment(segment: dict[str, Any]) -> str:
    text = str(segment.get("understanding") or "").strip()
    if text:
        return text
    left = segment["left_hand"]
    right = segment["right_hand"]
    return f"Left hand {hand_action(left)} {left.get('object', 'unknown')}; right hand {hand_action(right)} {right.get('object', 'unknown')}."


def semantic_rows_from_segments(segments: list[dict[str, Any]], fps: float, frame_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start_frame = max(0, min(frame_count, int(round(float(segment["start_sec"]) * fps))))
        end_frame = max(start_frame + 1, min(frame_count, int(round(float(segment["end_sec"]) * fps))))
        row = {
            "clip_id": f"cosmos_segment_{index:04d}",
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_s": float(start_frame / fps),
            "end_s": float(end_frame / fps),
            "duration_s": float((end_frame - start_frame) / fps),
            "caption": caption_from_segment(segment),
            "evidence_frames": [int(start_frame), int(max(start_frame, end_frame - 1))],
            "grounding_status": "cosmos_gallery_boundary_video_understanding",
            "provenance": "Cosmos scripted video-understanding output from sampled gallery frames and boundary refinement; object/contact words are semantic annotations, not object-pose or nonpenetration proof.",
            "source": "cosmos_gallery_boundary",
            "per_hand": {},
        }
        for side, hand_key in (("left", "left_hand"), ("right", "right_hand")):
            hand = segment[hand_key]
            row["per_hand"][side] = {
                "action_state": hand_action(hand),
                "in_frame": hand.get("in_frame", "unknown"),
                "contact": hand.get("contact", "unknown"),
                "object": hand.get("object", "unknown"),
                "object_rigidity": hand.get("rigidity", "unknown"),
                "object_assembly": hand.get("assembly", "unknown"),
                "contact_location": hand.get("contact_location", "unknown"),
            }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (dict, list)):
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise RuntimeError(f"missing video: {video}")
    output = args.output or (args.run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json")
    work_dir = args.work_dir or (args.run_root / "state" / "semantic_clips" / "cosmos_captioning_work")
    media_dir = work_dir / "media"
    raw_dir = work_dir / "raw"
    output_dir = work_dir / "outputs"
    for path in (media_dir, raw_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    api_base = args.api_base or ENDPOINTS[args.model_kind]
    model_id = args.model_id or get_model(api_base, args.http_timeout_s)
    meta = ffprobe(video)
    grid = sample_grid(meta, args.sample_fps)
    gallery = extract_gallery(video, media_dir, grid, args.gallery_width)
    native_link = media_dir / f"{video.stem}_native{video.suffix}"
    if native_link.exists() or native_link.is_symlink():
        native_link.unlink()
    native_link.symlink_to(video)

    server, port = start_server(media_dir)
    try:
        by_index: dict[int, dict[str, Any]] = {}
        raw_gallery_paths: list[str] = []
        chunks = [gallery[i:i + args.gallery_chunk_size] for i in range(0, len(gallery), args.gallery_chunk_size)]
        for chunk_index, chunk in enumerate(chunks, 1):
            prompt = (COARSE_GALLERY_PROMPT
                .replace("__SAMPLE_FPS__", f"{args.sample_fps:.3f}")
                .replace("__IMAGE_GRID__", image_grid_text(chunk)))
            raw_path = raw_dir / f"coarse_gallery_chunk_{chunk_index:02d}_response.json"
            record = post_json_once(api_base, model_id, gallery_content(chunk, media_dir, port, prompt), raw_path, args.gallery_max_tokens, args.http_timeout_s)
            raw_gallery_paths.append(str(raw_path))
            rows, _ = normalize_gallery_items(record.get("parsed"), chunk)
            for row in rows:
                by_index[int(row["image_index"])] = row
        missing = [int(row["image_index"]) for row in gallery if int(row["image_index"]) not in by_index]
        missing_before_individual_fallback = list(missing)
        if missing and args.fallback_individual:
            for image_index in list(missing):
                item = gallery[image_index]
                raw_path = raw_dir / f"coarse_image_{image_index:04d}_response.json"
                record = post_json_once(api_base, model_id, single_image_content(item, media_dir, port), raw_path, args.single_image_max_tokens, args.http_timeout_s)
                rows, _ = normalize_gallery_items(record.get("parsed"), [item])
                if rows:
                    by_index[image_index] = rows[0]
        coarse_rows = [by_index[int(row["image_index"])] for row in gallery if int(row["image_index"]) in by_index]
        missing = [int(row["image_index"]) for row in gallery if int(row["image_index"]) not in by_index]
        if missing:
            raise RuntimeError(f"cosmos_missing_gallery_items:{missing}")
        changes = detect_changes(coarse_rows)
        refinements: list[dict[str, Any]] = []
        for change in changes:
            prompt = (BOUNDARY_PROMPT
                .replace("__PREV_INDEX__", str(change["prev_image_index"]))
                .replace("__PREV_FRAME__", str(change["prev_frame_idx"]))
                .replace("__PREV_TIME__", f"{float(change['prev_time_sec']):.6f}")
                .replace("__NEXT_INDEX__", str(change["next_image_index"]))
                .replace("__NEXT_FRAME__", str(change["next_frame_idx"]))
                .replace("__NEXT_TIME__", f"{float(change['next_time_sec']):.6f}")
                .replace("__SOURCE_FPS__", f"{float(meta['fps']):.6f}")
                .replace("__SOURCE_FRAME_COUNT__", str(meta["frame_count"]))
                .replace("__PREV_LABEL__", json.dumps(compact_label(change["prev"]), ensure_ascii=False))
                .replace("__NEXT_LABEL__", json.dumps(compact_label(change["next"]), ensure_ascii=False))
                .replace("__CHANGE_INDEX__", str(change["change_index"])))
            raw_path = raw_dir / f"boundary_change_{change['change_index']:02d}_response.json"
            record = post_json_once(api_base, model_id, boundary_content(native_link, media_dir, port, prompt), raw_path, args.boundary_max_tokens, args.http_timeout_s)
            refinement = parse_boundary(record.get("parsed"), change, meta)
            refinement["raw_response_path"] = str(raw_path)
            refinement["finish_reason"] = record.get("finish_reason")
            refinements.append(refinement)
    finally:
        server.shutdown()

    segments = build_segments(coarse_rows, refinements, float(meta["duration_sec"]))
    semantic_rows = semantic_rows_from_segments(segments, float(meta["fps"]), int(meta["frame_count"]))
    if not semantic_rows:
        raise RuntimeError("cosmos produced no semantic rows")

    write_json(output_dir / "gallery_manifest.json", gallery)
    write_json(output_dir / "coarse_image_annotations.json", coarse_rows)
    write_json(output_dir / "coarse_changes.json", changes)
    write_json(output_dir / "boundary_refinements.json", refinements)
    write_json(output_dir / "final_segments.json", segments)
    write_csv(output_dir / "coarse_image_labels.csv", coarse_rows)

    payload = {
        "schema": "v22_cosmos_semantic_review.v1",
        "status": "ok",
        "method": "cosmos_gallery_boundary_video_understanding",
        "case_id": args.case_id,
        "source_video": str(video),
        "video_metadata": meta,
        "cosmos": {
            "model_kind": args.model_kind,
            "api_base": api_base,
            "model_id": model_id,
            "sample_fps": args.sample_fps,
            "gallery_chunk_size": args.gallery_chunk_size,
            "raw_response_dir": str(raw_dir),
        },
        "gallery": {
            "expected_images": len(gallery),
            "returned_images": len(coarse_rows),
            "missing_image_indices": missing,
            "missing_before_individual_fallback": missing_before_individual_fallback,
            "individual_fallback_enabled": bool(args.fallback_individual),
            "individual_fallback_used": bool(missing_before_individual_fallback and args.fallback_individual),
            "raw_response_paths": raw_gallery_paths,
        },
        "coarse_rows": coarse_rows,
        "boundary_refinements": refinements,
        "segments": segments,
        "semantic_rows": semantic_rows,
        "summary": {
            "semantic_clip_count": len(semantic_rows),
            "detected_changes": len(changes),
            "boundary_refinements": len(refinements),
            "timeline_coverage_fraction": sum(row["end_frame"] - row["start_frame"] for row in semantic_rows) / int(meta["frame_count"]),
        },
        "claim_scope": "Scripted D9b Cosmos video-understanding source. Object/contact/rigidity wording is semantic annotation unless separately backed by object geometry.",
        "elapsed_s": float(time.perf_counter() - started),
    }
    write_json(output, payload)
    print(json.dumps({"status": "ok", "output": str(output), "semantic_rows": len(semantic_rows), "elapsed_s": payload["elapsed_s"]}, indent=2, ensure_ascii=False))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--model-kind", choices=sorted(ENDPOINTS), default="cosmos3")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--gallery-width", type=int, default=960)
    parser.add_argument("--gallery-chunk-size", type=int, default=8)
    parser.add_argument("--gallery-max-tokens", type=int, default=12000)
    parser.add_argument("--single-image-max-tokens", type=int, default=1200)
    parser.add_argument("--boundary-max-tokens", type=int, default=1000)
    parser.add_argument("--http-timeout-s", type=int, default=1800)
    parser.add_argument("--fallback-individual", action="store_true", default=False, help="Opt in to single-image Cosmos calls when a gallery chunk omits items; disabled by default so missing gallery items fail the stage.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
