#!/usr/bin/env python3
"""Run the D9b semantic annotation agent for one V22 video.

This is the only runtime annotation-agent stage in the single-video MVP. It
creates a source-backed semantic review JSON at exactly 2 fps. The downstream
captioning stage normalizes this review into full-timeline subtitle rows.

The stage is intentionally narrow: it does not modify hand/camera/depth state,
does not inspect GT/evaluator targets, and does not change the four GPU-heavy
model request contracts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """Return ONLY one minified JSON object. Do not use markdown fences.

You are annotating 2fps analysis frames sampled from an egocentric manipulation video. Each tile comes from the ORIGINAL native-fps video at the listed frame_idx/time_sec. Use the sampled visual evidence and temporal ordering for reasoning.

Critical output-frequency requirement:
- Analyze and report at exactly 2 fps, meaning one output item every 0.5 seconds.
- The first output item is the first frame / time 0.000s.
- The next output item is the next 2fps analysis point / time 0.500s.
- Continue in order: 0.000s, 0.500s, 1.000s, 1.500s, ... until the last listed point.
- Use the explicit ANALYSIS_GRID below as the source of truth. Return exactly one item for every grid point. Do not add extra points and do not omit points.
- Each item describes the visual state at that sampled instant, not an interval summary.
- Use original-video frame_idx and original-video time_sec exactly as provided for that analysis point.

For BOTH left_hand and right_hand answer these fields:
1. in_frame: yes|no|unknown
2. contact: yes|no|unknown
3. object: short noun or none|unknown
4. rigidity: rigid|flexible|mixed|unknown
5. assembly: yes|no|unknown
6. contact_location: short phrase or none|unknown

Return this exact JSON schema:
{"points":[{"analysis_index":integer,"frame_idx":integer,"time_sec":number,"left_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"right_hand":{"in_frame":"yes|no|unknown","contact":"yes|no|unknown","object":"short noun|none|unknown","rigidity":"rigid|flexible|mixed|unknown","assembly":"yes|no|unknown","contact_location":"short phrase|none|unknown"},"understanding":"<=20 words describing the state at this analysis point"}]}.

Video metadata:
- duration_sec=__DURATION_SEC__
- source_fps=__SOURCE_FPS__
- source_frame_count=__SOURCE_FRAME_COUNT__
- analysis_fps=2.0
- expected_point_count=__EXPECTED_POINT_COUNT__

ANALYSIS_GRID(format: analysis_index@frame_idx@time_sec):
__ANALYSIS_GRID__

The attached contact sheet images show the listed analysis grid points sampled from the original video at the listed native frame indices. The overlay text in each tile is analysis_index@frame_idx@time_sec. Use the temporal ordering across sheets.

Stop immediately after the JSON object."""

VALID_YES_NO_UNKNOWN = {"yes", "no", "unknown"}
VALID_RIGIDITY = {"rigid", "flexible", "mixed", "unknown"}
VALID_ASSEMBLY = {"yes", "no", "unknown"}

CV2_HELPER_CODE = r'''
from __future__ import annotations
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

payload = json.loads(__import__('sys').stdin.read())
mode = payload['mode']
video = Path(payload['video'])
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f'failed to open video: {video}')
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
if fps <= 0:
    fps = 30.0
if frame_count <= 0:
    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1
    cap.release()
    cap = cv2.VideoCapture(str(video))
duration = float(frame_count / fps) if fps > 0 else 0.0
if mode == 'metadata':
    print(json.dumps({'frame_count': frame_count, 'fps': fps, 'duration': duration}))
    raise SystemExit(0)
work_dir = Path(payload['work_dir'])
points = payload['points']
frames_dir = work_dir / 'analysis_frames'
sheets_dir = work_dir / 'contact_sheets'
if frames_dir.exists():
    shutil.rmtree(frames_dir)
if sheets_dir.exists():
    shutil.rmtree(sheets_dir)
frames_dir.mkdir(parents=True, exist_ok=True)
sheets_dir.mkdir(parents=True, exist_ok=True)
frame_paths = []
for point in points:
    idx = int(point['analysis_index'])
    frame_idx = int(point['frame_idx'])
    time_sec = float(point['time_sec'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, image = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ok, image = cap.read()
    if not ok:
        raise SystemExit(f'failed to read frame {frame_idx}')
    h, w = image.shape[:2]
    new_w = 640
    new_h = max(1, int(round(h * (new_w / max(1, w)))))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    label = f"{idx}@{frame_idx}@{time_sec:.3f}s"
    cv2.rectangle(image, (6, 6), (6 + 18 * len(label), 40), (0, 0, 0), -1)
    cv2.putText(image, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    out = frames_dir / f"analysis_{idx:04d}_frame_{frame_idx:06d}.jpg"
    if not cv2.imwrite(str(out), image):
        raise SystemExit(f'failed to write {out}')
    frame_paths.append(str(out))
cap.release()
cols = int(payload.get('cols') or 4)
rows = int(payload.get('rows') or 3)
per_sheet = cols * rows
sheets = []
for sheet_idx, start in enumerate(range(0, len(frame_paths), per_sheet)):
    group = frame_paths[start:start + per_sheet]
    imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in group]
    if any(img is None for img in imgs):
        raise SystemExit('failed to read generated analysis frame')
    tile_h = max(img.shape[0] for img in imgs)
    tile_w = max(img.shape[1] for img in imgs)
    canvas = np.zeros((rows * tile_h + (rows + 1) * 8, cols * tile_w + (cols + 1) * 8, 3), dtype=np.uint8)
    for local_idx, img in enumerate(imgs):
        r = local_idx // cols
        c = local_idx % cols
        y = 8 + r * (tile_h + 8)
        x = 8 + c * (tile_w + 8)
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    out = sheets_dir / f"semantic_sheet_{sheet_idx:03d}.jpg"
    if not cv2.imwrite(str(out), canvas):
        raise SystemExit(f'failed to write {out}')
    sheets.append(str(out))
print(json.dumps({'frame_count': frame_count, 'fps': fps, 'duration': duration, 'sheets': sheets, 'frames': frame_paths}))
'''


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


def run_cv2_helper(payload: dict[str, Any], *, cv2_python: str) -> dict[str, Any]:
    proc = subprocess.run([cv2_python, "-c", CV2_HELPER_CODE], input=json.dumps(payload), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"cv2_helper_failed rc={proc.returncode}: {proc.stderr[-2000:]} {proc.stdout[-1000:]}")
    result = json.loads(proc.stdout)
    if not isinstance(result, dict):
        raise RuntimeError("cv2 helper returned non-object JSON")
    return result


def read_video_metadata(video: Path, *, ffprobe: str, cv2_python: str) -> tuple[int, float, float, str]:
    if shutil.which(ffprobe):
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(video),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0:
            payload = json.loads(proc.stdout)
            streams = payload.get("streams") if isinstance(payload, dict) else None
            if streams:
                stream = streams[0]
                frame_count = int(stream.get("nb_read_frames") or 0)
                rate = str(stream.get("avg_frame_rate") or "")
                fps = None
                if "/" in rate:
                    num, den = rate.split("/", 1)
                    den_f = finite_positive(den)
                    num_f = finite_positive(num)
                    if num_f is not None and den_f is not None:
                        fps = num_f / den_f
                if fps is None:
                    fps = finite_positive(stream.get("fps"))
                duration = finite_positive(stream.get("duration"))
                if frame_count <= 0 and fps and duration:
                    frame_count = int(round(fps * duration))
                if fps is None and duration and frame_count > 0:
                    fps = frame_count / duration
                if duration is None and fps and frame_count > 0:
                    duration = frame_count / fps
                if frame_count > 0 and fps and duration:
                    return frame_count, float(fps), float(duration), "ffprobe"
    payload = run_cv2_helper({"mode": "metadata", "video": str(video)}, cv2_python=cv2_python)
    frame_count = int(payload["frame_count"])
    fps = float(payload["fps"])
    duration = float(payload["duration"])
    if frame_count <= 0 or fps <= 0 or duration <= 0:
        raise RuntimeError(f"invalid cv2 metadata: {payload}")
    return frame_count, fps, duration, "opencv_helper"


def analysis_grid(frame_count: int, fps: float, analysis_fps: float = 2.0) -> list[dict[str, Any]]:
    step = fps / analysis_fps
    points: list[dict[str, Any]] = []
    index = 0
    while True:
        frame_idx = int(round(index * step))
        if frame_idx >= frame_count:
            break
        points.append({"analysis_index": index, "frame_idx": frame_idx, "time_sec": float(frame_idx / fps)})
        index += 1
    if not points:
        points.append({"analysis_index": 0, "frame_idx": 0, "time_sec": 0.0})
    return points


def grid_text(points: list[dict[str, Any]]) -> str:
    return "\n".join(f"{p['analysis_index']}@{p['frame_idx']}@{p['time_sec']:.3f}" for p in points)


def prompt_for(points: list[dict[str, Any]], *, frame_count: int, fps: float, duration: float) -> str:
    return (
        PROMPT_TEMPLATE.replace("__DURATION_SEC__", f"{duration:.6f}")
        .replace("__SOURCE_FPS__", f"{fps:.6f}")
        .replace("__SOURCE_FRAME_COUNT__", str(frame_count))
        .replace("__EXPECTED_POINT_COUNT__", str(len(points)))
        .replace("__ANALYSIS_GRID__", grid_text(points))
    )


def run_checked(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command_failed rc={proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
    return proc


def extract_frame(video: Path, output: Path, *, time_sec: float, label: str, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    drawtext = f"drawtext=text='{label}':x=12:y=12:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.65"
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{time_sec:.6f}", "-i", str(video), "-frames:v", "1"]
    cmd = base + ["-vf", f"scale=640:-1,{drawtext}", str(output)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0 and output.exists() and output.stat().st_size > 0:
        return
    fallback = base + ["-vf", "scale=640:-1", str(output)]
    run_checked(fallback)


def make_contact_sheets(frames: list[Path], sheets_dir: Path, *, ffmpeg: str, cols: int = 4, rows: int = 3) -> list[Path]:
    sheets_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows
    sheets: list[Path] = []
    for sheet_idx, start in enumerate(range(0, len(frames), per_sheet)):
        group = frames[start : start + per_sheet]
        group_dir = sheets_dir / f"sheet_{sheet_idx:03d}_frames"
        if group_dir.exists():
            shutil.rmtree(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(group):
            shutil.copy2(src, group_dir / f"tile_{i:03d}.jpg")
        output = sheets_dir / f"semantic_sheet_{sheet_idx:03d}.jpg"
        pattern = group_dir / "tile_%03d.jpg"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(pattern),
            "-frames:v",
            "1",
            "-vf",
            f"tile={cols}x{rows}:padding=8:margin=8:color=black",
            str(output),
        ]
        run_checked(cmd)
        sheets.append(output)
    return sheets


def prepare_visual_inputs_with_cv2(video: Path, work_dir: Path, points: list[dict[str, Any]], *, cv2_python: str) -> list[Path]:
    payload = run_cv2_helper(
        {"mode": "extract", "video": str(video), "work_dir": str(work_dir), "points": points, "cols": 4, "rows": 3},
        cv2_python=cv2_python,
    )
    sheets = payload.get("sheets") if isinstance(payload.get("sheets"), list) else []
    if not sheets:
        raise RuntimeError("opencv helper produced no contact sheets")
    return [Path(str(path)) for path in sheets]


def prepare_visual_inputs(video: Path, work_dir: Path, points: list[dict[str, Any]], *, ffmpeg: str, cv2_python: str) -> tuple[list[Path], str]:
    if shutil.which(ffmpeg):
        try:
            frames_dir = work_dir / "analysis_frames"
            sheets_dir = work_dir / "contact_sheets"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            if sheets_dir.exists():
                shutil.rmtree(sheets_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_paths: list[Path] = []
            for point in points:
                label = f"{point['analysis_index']}@{point['frame_idx']}@{point['time_sec']:.3f}s"
                frame_path = frames_dir / f"analysis_{point['analysis_index']:04d}_frame_{point['frame_idx']:06d}.jpg"
                extract_frame(video, frame_path, time_sec=float(point["time_sec"]), label=label, ffmpeg=ffmpeg)
                frame_paths.append(frame_path)
            return make_contact_sheets(frame_paths, sheets_dir, ffmpeg=ffmpeg), "ffmpeg"
        except Exception:
            if not cv2_python:
                raise
    return prepare_visual_inputs_with_cv2(video, work_dir, points, cv2_python=cv2_python), "opencv_helper"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return load_json_text(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"semantic agent did not return a JSON object: {stripped[-1000:]}")
    return load_json_text(stripped[start : end + 1])


def load_json_text(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError("semantic agent JSON root must be an object")
    return payload


def normalized_choice(value: Any, allowed: set[str], default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    return text if text in allowed else default


def short_text(value: Any, default: str = "unknown", max_len: int = 64) -> str:
    text = str(value or default).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return (text or default)[:max_len]


def normalize_hand(value: Any) -> dict[str, str]:
    hand = value if isinstance(value, dict) else {}
    return {
        "in_frame": normalized_choice(hand.get("in_frame"), VALID_YES_NO_UNKNOWN),
        "contact": normalized_choice(hand.get("contact"), VALID_YES_NO_UNKNOWN),
        "object": short_text(hand.get("object"), "unknown"),
        "rigidity": normalized_choice(hand.get("rigidity"), VALID_RIGIDITY),
        "assembly": normalized_choice(hand.get("assembly"), VALID_ASSEMBLY),
        "contact_location": short_text(hand.get("contact_location"), "unknown"),
    }


def action_state(hand: dict[str, str]) -> str:
    if hand["in_frame"] == "no":
        return "no_action"
    if hand["contact"] == "yes":
        return "contacting_or_operating_object"
    if hand["contact"] == "no":
        obj = hand.get("object", "unknown")
        if obj not in {"none", "unknown"}:
            return "approaching_or_near_object"
        return "visible_no_contact"
    return "unknown"


def validate_points(payload: dict[str, Any], grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise RuntimeError("semantic agent JSON lacks points[]")
    if len(raw_points) != len(grid):
        raise RuntimeError(f"semantic agent returned {len(raw_points)} points; expected {len(grid)}")
    out: list[dict[str, Any]] = []
    for expected, raw in zip(grid, raw_points, strict=True):
        if not isinstance(raw, dict):
            raise RuntimeError("semantic agent point is not an object")
        try:
            analysis_index = int(raw.get("analysis_index"))
            frame_idx = int(raw.get("frame_idx"))
            time_sec = float(raw.get("time_sec"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid semantic point index/frame/time: {raw}") from exc
        if analysis_index != int(expected["analysis_index"]) or frame_idx != int(expected["frame_idx"]) or abs(time_sec - float(expected["time_sec"])) > 1e-3:
            raise RuntimeError(f"semantic point grid mismatch: got {raw}; expected {expected}")
        left = normalize_hand(raw.get("left_hand"))
        right = normalize_hand(raw.get("right_hand"))
        out.append(
            {
                "analysis_index": analysis_index,
                "frame_idx": frame_idx,
                "time_sec": float(time_sec),
                "left_hand": left,
                "right_hand": right,
                "understanding": short_text(raw.get("understanding"), "unknown", max_len=160),
            }
        )
    return out


def point_to_semantic_row(point: dict[str, Any], next_frame: int, fps: float) -> dict[str, Any]:
    start_frame = int(point["frame_idx"])
    end_frame = max(start_frame + 1, int(next_frame))
    left = point["left_hand"]
    right = point["right_hand"]
    left_obj = left.get("object", "unknown")
    right_obj = right.get("object", "unknown")
    caption = short_text(point.get("understanding"), "unknown", max_len=160)
    if not caption or caption == "unknown":
        caption = f"left {left['contact']} contact {left_obj}; right {right['contact']} contact {right_obj}"
    return {
        "clip_id": f"semantic_point_{point['analysis_index']:04d}",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_s": float(start_frame / fps),
        "end_s": float(end_frame / fps),
        "duration_s": float((end_frame - start_frame) / fps),
        "caption": caption,
        "analysis_index": int(point["analysis_index"]),
        "evidence_frames": [start_frame],
        "grounding_status": "semantic_agent_2fps_visual_review",
        "provenance": "D9b semantic agent visual estimate at 2fps; object/contact words are not object-pose or nonpenetration proof",
        "per_hand": {
            "left": {
                "action_state": action_state(left),
                "in_frame": left["in_frame"],
                "contact": left["contact"],
                "object": left_obj,
                "object_rigidity": left["rigidity"],
                "object_assembly": left["assembly"],
                "contact_location": left["contact_location"],
            },
            "right": {
                "action_state": action_state(right),
                "in_frame": right["in_frame"],
                "contact": right["contact"],
                "object": right_obj,
                "object_rigidity": right["rigidity"],
                "object_assembly": right["assembly"],
                "contact_location": right["contact_location"],
            },
        },
    }


def points_to_rows(points: list[dict[str, Any]], *, frame_count: int, fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        next_frame = int(points[i + 1]["frame_idx"]) if i + 1 < len(points) else frame_count
        rows.append(point_to_semantic_row(point, next_frame, fps))
    return rows


def run_pi_agent(prompt: str, images: list[Path], *, pi_binary: str, model: str, provider: str | None, thinking: str, timeout: int, session_dir: Path) -> tuple[str, str, list[str]]:
    cmd = [
        pi_binary,
        "-p",
        "--no-context-files",
        "--no-skills",
        "--no-extensions",
        "--no-prompt-templates",
        "--no-tools",
        "--no-session",
        "--thinking",
        thinking,
        "--model",
        model,
    ]
    if provider:
        cmd.extend(["--provider", provider])
    for image in images:
        cmd.append("@" + str(image))
    cmd.append(prompt)
    env = os.environ.copy()
    session_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"semantic_agent_failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
    return proc.stdout, proc.stderr, cmd


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    video = args.video.expanduser().resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"semantic annotation video missing: {video}")
    output = args.output.expanduser().resolve()
    work_dir = (args.work_dir or output.parent / "semantic_agent_work").expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    frame_count, fps, duration, metadata_backend = read_video_metadata(video, ffprobe=args.ffprobe, cv2_python=args.cv2_python)
    points_grid = analysis_grid(frame_count, fps, analysis_fps=2.0)
    prompt = prompt_for(points_grid, frame_count=frame_count, fps=fps, duration=duration)
    prompt_path = work_dir / "semantic_agent_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    sheets, extraction_backend = prepare_visual_inputs(video, work_dir, points_grid, ffmpeg=args.ffmpeg, cv2_python=args.cv2_python)
    stdout, stderr, cmd = run_pi_agent(
        prompt,
        sheets,
        pi_binary=args.pi_binary,
        model=args.model,
        provider=args.provider,
        thinking=args.thinking,
        timeout=args.timeout,
        session_dir=work_dir / "pi_session",
    )
    (work_dir / "semantic_agent_stdout.txt").write_text(stdout, encoding="utf-8")
    (work_dir / "semantic_agent_stderr.txt").write_text(stderr, encoding="utf-8")
    agent_payload = extract_json_object(stdout)
    points = validate_points(agent_payload, points_grid)
    semantic_rows = points_to_rows(points, frame_count=frame_count, fps=fps)

    result = {
        "schema": "v22_semantic_agent_review.v1",
        "status": "ok",
        "method": "pi_visual_semantic_annotation_2fps",
        "case_id": args.case_id,
        "source_video": str(video),
        "video_metadata": {
            "duration_sec": duration,
            "source_fps": fps,
            "source_frame_count": frame_count,
            "analysis_fps": 2.0,
            "expected_point_count": len(points_grid),
            "metadata_backend": metadata_backend,
        },
        "analysis_grid": points_grid,
        "points": points,
        "semantic_rows": semantic_rows,
        "agent": {
            "pi_binary": args.pi_binary,
            "model": args.model,
            "provider": args.provider,
            "thinking": args.thinking,
            "image_sheets": [str(path) for path in sheets],
            "extraction_backend": extraction_backend,
            "prompt": str(prompt_path),
            "stdout": str(work_dir / "semantic_agent_stdout.txt"),
            "stderr": str(work_dir / "semantic_agent_stderr.txt"),
            "command": cmd[:1] + ["<pi semantic args redacted; see prompt and sheets>"],
        },
        "claim_scope": "D9b semantic video understanding at 2fps. It writes subtitle/action/contact language only; it does not modify D2-D8 physical state or the four GPU-heavy request contracts.",
        "elapsed_s": float(time.time() - started),
    }
    write_json(output, result)
    print(json.dumps({"status": "ok", "output": str(output), "points": len(points), "semantic_rows": len(semantic_rows)}, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--pi-binary", default=os.environ.get("ANNOTATION_PI_BINARY", "pi"))
    parser.add_argument("--model", default=os.environ.get("ANNOTATION_SEMANTIC_AGENT_MODEL", "openai/gpt-5.5"))
    parser.add_argument("--provider", default=os.environ.get("ANNOTATION_SEMANTIC_AGENT_PROVIDER"))
    parser.add_argument("--thinking", default=os.environ.get("ANNOTATION_SEMANTIC_AGENT_THINKING", "xhigh"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("ANNOTATION_SEMANTIC_AGENT_TIMEOUT_S", "900")))
    parser.add_argument("--cv2-python", default=os.environ.get("ANNOTATION_METADATA_PYTHON", sys.executable))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE", "ffprobe"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
