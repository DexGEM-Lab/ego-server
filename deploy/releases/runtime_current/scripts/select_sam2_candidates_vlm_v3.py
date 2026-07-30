#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from build_object_plan_vlm import load_env_file
from segment_object_plan_v2 import plan_objects


SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_idx": {"type": "integer"},
                    "accepted": {"type": "boolean"},
                    "candidate": {"type": "integer"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["frame_idx", "accepted", "candidate", "confidence", "reason"],
            },
        }
    },
    "required": ["selections"],
}


def image_data_url(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def load_reports(path: Path) -> list[dict]:
    qc = json.loads(path.read_text(encoding="utf-8"))
    reports = qc.get("reports")
    if not isinstance(reports, list) or not reports:
        raise RuntimeError(f"QC has no reports list: {path}")
    return reports


def build_sheet(candidate_dir: Path, frames: list[int], output_path: Path, tile_width: int) -> None:
    tiles = []
    for frame_idx in frames:
        for candidate in range(3):
            path = candidate_dir / f"{frame_idx:06d}_candidate_{candidate}.jpg"
            if not path.exists():
                raise RuntimeError(f"missing candidate review image: {path}")
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"failed to read {path}")
            height = int(round(image.shape[0] * tile_width / image.shape[1]))
            tile = cv2.resize(image, (tile_width, height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (tile_width, 28), (0, 0, 0), -1)
            cv2.putText(
                tile,
                f"frame {frame_idx} candidate {candidate}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
    if not tiles:
        raise RuntimeError("no candidate tiles")
    tile_h = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < tile_h:
            pad = np.full((tile_h - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    rows = []
    for start in range(0, len(padded), 3):
        row_tiles = padded[start : start + 3]
        if len(row_tiles) < 3:
            blank = np.full_like(padded[0], 255)
            row_tiles.extend([blank] * (3 - len(row_tiles)))
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"failed to write {output_path}")


def call_vlm(args: argparse.Namespace, object_plan: dict, frames: list[int], sheet_path: Path) -> list[dict]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    base_url = args.base_url
    if args.base_url_env:
        base_url = os.environ.get(args.base_url_env, "")
        if not base_url:
            raise RuntimeError(f"{args.base_url_env} is not set")
    base_url = str(base_url).strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"invalid Responses API base URL: {base_url!r}")
    content = [
        {
            "type": "input_text",
            "text": (
                "Select SAM2 mask candidates for an egocentric manipulation object. "
                "Each row has three candidate masks for the same source frame. The yellow overlay is the candidate mask. "
                "Choose one candidate only if it primarily covers the target physical object instance and avoids hands, supports, tools, containers, background, and nearby similar objects. "
                "The selected mask must represent one contiguous physical object instance, not a union of several nearby instances or disconnected visual artifacts. "
                "Reject the frame when every candidate is a fragment, includes multiple object instances, switches to a nearby object, or leaks onto hands/background. "
                "Set candidate to -1 when accepted is false.\n\n"
                f"Target object plan:\n{json.dumps(object_plan, ensure_ascii=True)}\n\n"
                f"Requested frames: {frames}"
            ),
        },
        {"type": "input_image", "image_url": image_data_url(sheet_path), "detail": args.detail},
    ]
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sam2_candidate_selections",
                "strict": True,
                "schema": SELECTION_SCHEMA,
            }
        },
    }
    with httpx.Client(timeout=float(args.timeout_s)) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API failed {response.status_code}: {response.text[:1000]}")
    body = response.json()
    output_text = body.get("output_text")
    if output_text is None:
        output_text = "\n".join(
            part.get("text", "")
            for item in body.get("output", [])
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    if not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    parsed = json.loads(output_text)
    selections = parsed.get("selections")
    if not isinstance(selections, list):
        raise RuntimeError(f"VLM returned no selections list: {parsed}")
    by_frame = {int(row["frame_idx"]): row for row in selections}
    missing = [frame for frame in frames if frame not in by_frame]
    if missing:
        raise RuntimeError(f"VLM omitted frames: {missing}")
    out = []
    for frame in frames:
        row = dict(by_frame[frame])
        row["frame_idx"] = int(frame)
        row["candidate"] = int(row["candidate"])
        if bool(row["accepted"]) and row["candidate"] not in {0, 1, 2}:
            raise RuntimeError(f"accepted frame {frame} has invalid candidate {row['candidate']}")
        if not bool(row["accepted"]) and row["candidate"] != -1:
            raise RuntimeError(f"rejected frame {frame} must use candidate -1")
        out.append(row)
    return out


def frame_batches(frames: list[int], batch_size: int) -> list[list[int]]:
    if batch_size < 1:
        raise RuntimeError("--batch-size must be positive")
    return [frames[start : start + batch_size] for start in range(0, len(frames), batch_size)]


def write_selected_masks(args: argparse.Namespace, selections: list[dict], reports: list[dict]) -> dict:
    report_by_frame = {int(row["frame_idx"]): row for row in reports}
    mask_dir = args.output_dir / "selected_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    track = {}
    for row in selections:
        frame_idx = int(row["frame_idx"])
        if not row["accepted"]:
            track[str(frame_idx)] = {"visible": False, "selection": row}
            continue
        candidate = int(row["candidate"])
        source = args.sam2_dir / "sam2_candidate_masks" / f"{frame_idx:06d}_candidate_{candidate}.png"
        if not source.exists():
            raise RuntimeError(f"missing selected mask for frame {frame_idx} candidate {candidate}: {source}")
        target = mask_dir / f"{frame_idx:06d}.png"
        shutil.copy2(source, target)
        report = report_by_frame[frame_idx]
        selected = next((c for c in report.get("candidates", []) if int(c["candidate"]) == candidate), {})
        track[str(frame_idx)] = {
            "visible": True,
            "mask_path": str(target),
            "candidate": candidate,
            "selection": row,
            "candidate_report": selected,
        }
    track_path = args.output_dir / "sam2_vlm_selected_track.json"
    track_path.write_text(json.dumps(track, indent=2), encoding="utf-8")
    return {"track": str(track_path), "masks": str(mask_dir)}


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    objects = plan_objects(args.object_plan)
    if args.object_index >= len(objects):
        raise RuntimeError(f"--object-index {args.object_index} out of range for {len(objects)} planned objects")
    object_plan = objects[args.object_index]
    reports = load_reports(args.sam2_dir / "qc_sam2_image_points.json")
    frames = sorted(int(row["frame_idx"]) for row in reports)
    if args.frames:
        wanted = {int(part) for raw in args.frames for part in raw.split(",") if part.strip()}
        frames = [frame for frame in frames if frame in wanted]
    if not frames:
        raise RuntimeError("no frames to select")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    sheet_paths = []
    for batch_i, batch_frames in enumerate(frame_batches(frames, int(args.batch_size))):
        if len(frames) == len(batch_frames):
            sheet_path = args.output_dir / "sam2_candidate_selection_sheet.jpg"
        else:
            sheet_path = args.output_dir / f"sam2_candidate_selection_sheet_batch_{batch_i:03d}.jpg"
        build_sheet(args.sam2_dir / "sam2_candidate_review", batch_frames, sheet_path, int(args.tile_width))
        sheet_paths.append(str(sheet_path))
        selections.extend(call_vlm(args, object_plan, batch_frames, sheet_path))
    outputs = write_selected_masks(args, selections, reports)
    qc = {
        "status": "ok",
        "backend": "VLM SAM2 candidate selection",
        "model": args.model,
        "sam2_dir": str(args.sam2_dir),
        "object_plan": str(args.object_plan),
        "object_index": int(args.object_index),
        "track_id": object_plan["track_id"],
        "frames": frames,
        "accepted_frames": [int(row["frame_idx"]) for row in selections if row["accepted"]],
        "rejected_frames": [int(row["frame_idx"]) for row in selections if not row["accepted"]],
        "selections": selections,
        "selection_sheets": sheet_paths,
        "outputs": outputs,
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "qc_sam2_candidate_selection_vlm.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "selections"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-dir", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--frames", nargs="*")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if int(args.batch_size) == 0:
        args.batch_size = 10**9
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
