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

from build_object_plan_vlm import load_env_file
from select_sam2_candidates_vlm_v3 import SELECTION_SCHEMA, build_sheet, load_reports


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def response_text(body: dict) -> str:
    output_text = body.get("output_text")
    if output_text is not None:
        return str(output_text)
    return "\n".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def call_vlm(args: argparse.Namespace, frames: list[int], sheet_path: Path) -> list[dict]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    base_url = str(os.environ.get(args.base_url_env, "") if args.base_url_env else args.base_url).strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"invalid Responses API base URL: {base_url!r}")
    content = [
        {
            "type": "input_text",
            "text": (
                "Select SAM2 mask candidates for one visual track in egocentric manipulation video. "
                "Each row has three candidate masks for the same source frame. The yellow overlay is the candidate mask. "
                "Accept one candidate only when it primarily covers the visible pixels of the target track and excludes confusing adjacent surfaces. "
                "Reject the frame when every candidate is a fragment, merges with cloth/sleeve/object/background, switches target, or covers multiple physical surfaces. "
                "Set candidate to -1 when accepted is false.\n\n"
                f"Target track id: {args.track_id}\n"
                f"Target visual description: {args.track_description}\n"
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
                "name": "sam2_visual_track_candidate_selections",
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
    text = response_text(response.json())
    if not text:
        raise RuntimeError("Responses API returned no output text")
    selections = json.loads(text).get("selections")
    if not isinstance(selections, list):
        raise RuntimeError(f"VLM returned no selections list: {text[:1000]}")
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
    track: dict[str, dict] = {}
    for row in selections:
        frame_idx = int(row["frame_idx"])
        if not bool(row["accepted"]):
            track[str(frame_idx)] = {"visible": False, "selection": row}
            continue
        candidate = int(row["candidate"])
        source = args.sam2_dir / "sam2_candidate_masks" / f"{frame_idx:06d}_candidate_{candidate}.png"
        if not source.exists():
            raise RuntimeError(f"missing selected candidate mask: {source}")
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


def render_selected_video(args: argparse.Namespace, track_path: Path) -> str | None:
    if args.video is None:
        return None
    track = json.loads(track_path.read_text(encoding="utf-8"))
    frames = sorted(int(frame) for frame in track)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    writer = None
    video_path = args.output_dir / "selected_visual_track_masks.mp4"
    try:
        for frame_idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {frame_idx}")
            row = track[str(frame_idx)]
            if row.get("visible") and row.get("mask_path"):
                mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read mask {row['mask_path']}")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                tint = image.copy()
                tint[:, :] = (50, 210, 230)
                image[mask] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask]
            cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
            status = "accepted" if row.get("visible") else "rejected"
            cv2.putText(image, f"{frame_idx} {args.track_id} {status}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            if args.render_width and image.shape[1] != int(args.render_width):
                height = int(round(int(args.render_width) * image.shape[0] / image.shape[1]))
                image = cv2.resize(image, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.output_fps), (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open writer {video_path}")
            writer.write(image)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return str(video_path)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    reports = load_reports(args.sam2_dir / "qc_sam2_image_points.json")
    frames = sorted(int(row["frame_idx"]) for row in reports)
    if args.frames:
        wanted = {int(part) for raw in args.frames for part in raw.split(",") if part.strip()}
        frames = [frame for frame in frames if frame in wanted]
    if not frames:
        raise RuntimeError("no frames to select")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    sheets = []
    for batch_i, batch in enumerate(frame_batches(frames, int(args.batch_size))):
        sheet_path = args.output_dir / f"sam2_visual_track_candidate_sheet_batch_{batch_i:03d}.jpg"
        build_sheet(args.sam2_dir / "sam2_candidate_review", batch, sheet_path, int(args.tile_width))
        sheets.append(str(sheet_path))
        selections.extend(call_vlm(args, batch, sheet_path))
    outputs = write_selected_masks(args, selections, reports)
    review_video = render_selected_video(args, Path(outputs["track"]))
    qc = {
        "status": "ok",
        "backend": "VLM SAM2 visual-track candidate selection",
        "model": args.model,
        "sam2_dir": str(args.sam2_dir),
        "track_id": args.track_id,
        "track_description": args.track_description,
        "frames": frames,
        "accepted_frames": [int(row["frame_idx"]) for row in selections if row["accepted"]],
        "rejected_frames": [int(row["frame_idx"]) for row in selections if not row["accepted"]],
        "selection_sheets": sheets,
        "outputs": outputs,
        "review_video": review_video,
        "elapsed_s": time.time() - started,
        "selections": selections,
    }
    (args.output_dir / "qc_sam2_visual_track_candidate_selection_vlm.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "selections"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--track-description", required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", nargs="*")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--output-fps", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
