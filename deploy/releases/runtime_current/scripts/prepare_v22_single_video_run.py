#!/usr/bin/env python3
"""Prepare a V22-compatible run root from one ordinary video file."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2


class ContractError(RuntimeError):
    pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {path}")
    meta = {
        "path": str(path),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if meta["fps"] <= 0 or meta["width"] <= 0 or meta["height"] <= 0 or meta["frame_count"] <= 0:
        raise ContractError(f"invalid_video_metadata: {path} {meta}")
    meta["duration_s"] = float(meta["frame_count"] / meta["fps"])
    return meta


def even_height(width: int, source_width: int, source_height: int) -> int:
    height = int(round(float(width) * float(source_height) / float(source_width)))
    return max(2, height + height % 2)


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def ensure_run_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            children = list(path.iterdir())
            if children and not all(child.name == "logs" and child.is_dir() for child in children):
                raise ContractError(f"run_root_exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def copy_window(source: Path, output: Path, start_s: float | None, end_s: float | None) -> dict[str, Any]:
    meta = video_metadata(source)
    start_frame = 0 if start_s is None else max(0, int(round(start_s * float(meta["fps"]))))
    end_frame = int(meta["frame_count"]) if end_s is None or end_s <= 0 else min(int(meta["frame_count"]), int(round(end_s * float(meta["fps"]))))
    if end_frame <= start_frame:
        raise ContractError(f"invalid_time_window: start_s={start_s} end_s={end_s}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    size = (int(meta["width"]), int(meta["height"]))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]), size)
    if not writer.isOpened():
        cap.release()
        raise ContractError(f"could_not_open_writer: {output}")
    written = 0
    try:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: {source} frame={frame_idx}")
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    return {"source_video": str(source), "clip_video": str(output), "start_frame": start_frame, "end_frame_exclusive": end_frame, "frame_count": written, "source_metadata": meta, "clip_metadata": video_metadata(output)}


def extract_frames(video: Path, run_root: Path, render_width: int | None, source_start_frame: int) -> dict[str, Any]:
    meta = video_metadata(video)
    resolved_render_width = int(render_width) if render_width is not None and int(render_width) > 0 else int(meta["width"])
    render_height = even_height(resolved_render_width, int(meta["width"]), int(meta["height"]))
    rgb_dir = run_root / "input" / "raw_frame_manifest" / "rgb"
    source_rgb_dir = run_root / "input" / "source_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    source_rgb_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {video}")
    frames: list[dict[str, Any]] = []
    try:
        for frame_idx in range(int(meta["frame_count"])):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: {video} frame={frame_idx}")
            resized = cv2.resize(frame, (resolved_render_width, render_height), interpolation=cv2.INTER_AREA)
            out_path = rgb_dir / f"{frame_idx:06d}.jpg"
            source_out_path = source_rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise ContractError(f"could_not_write_frame: {out_path}")
            shutil.copy2(out_path, source_out_path)
            frames.append({"index": int(frame_idx), "frame_idx": int(frame_idx), "source_frame_idx": int(source_start_frame + frame_idx), "time_s": float(frame_idx / meta["fps"]), "source_time_s": float((source_start_frame + frame_idx) / meta["fps"]), "rgb": str(out_path), "raw_frame_path": str(out_path), "source_width": int(meta["width"]), "source_height": int(meta["height"]), "manifest_width": int(resolved_render_width), "manifest_height": int(render_height), "source_video": str(video), "coordinate_semantics": "raw_manifest_pixel_coordinates"})
    finally:
        cap.release()
    manifest = {"schema": "v22_raw_frame_manifest.v0", "status": "ok", "method": "prepare_v22_single_video_run", "clip": str(video), "video": meta, "fps": float(meta["fps"]), "frame_count": len(frames), "render_width": int(resolved_render_width), "render_height": int(render_height), "frames": frames, "claim_scope": "Decoded input timeline only; not a physical annotation measurement."}
    raw_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    source_path = run_root / "input" / "source_frame_manifest" / "manifest.json"
    write_json(raw_path, manifest)
    source_manifest = dict(manifest)
    source_manifest["schema"] = "v22_source_frame_manifest.v0"
    source_manifest["frames"] = [{**row, "rgb": str(source_rgb_dir / f"{row['frame_idx']:06d}.jpg"), "raw_frame_path": str(source_rgb_dir / f"{row['frame_idx']:06d}.jpg")} for row in frames]
    write_json(source_path, source_manifest)
    return {"raw_frame_manifest": str(raw_path), "source_frame_manifest": str(source_path), "frame_count": len(frames), "video": meta, "render_width": int(resolved_render_width), "render_height": render_height}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    input_video = args.input_video.resolve()
    if not input_video.exists():
        raise ContractError(f"missing_input_video: {input_video}")
    ensure_run_root(run_root, args.overwrite)
    clip_path = run_root / "input" / "clips" / f"{args.case_id}.mp4"
    window = copy_window(input_video, clip_path, args.start_s, args.end_s)
    frame_manifest = extract_frames(clip_path, run_root, args.render_width, int(window["start_frame"]))
    input_manifest = {"schema": "v22_input_manifest.v0", "mode": "v22_infer", "case_id": args.case_id, "run_root": str(run_root), "input_kind": "single_video", "input_modality": "single_rgb_video", "primary_video": str(clip_path), "original_video": str(input_video), "source_fingerprint": file_fingerprint(input_video), "clip_window": window, "raw_frame_manifest": frame_manifest["raw_frame_manifest"], "source_frame_manifest": frame_manifest["source_frame_manifest"], "claim_scope": "V22 input/run bootstrap only. No depth, hand, segmentation, object geometry, pose, or renderable annotation has run."}
    write_json(run_root / "input" / "input_manifest.json", input_manifest)
    initial_state = {"schema": "v22_physical_state.v0", "status": "input_timeline_bootstrap_complete_physical_measurements_pending", "mode": "v22_infer", "case_id": args.case_id, "run_root": str(run_root), "input": str(input_video), "timeline": {"frame_count": int(frame_manifest["frame_count"]), "fps": float(frame_manifest["video"]["fps"]), "duration_s": float(frame_manifest["video"]["duration_s"]), "resolution": [int(frame_manifest["video"]["width"]), int(frame_manifest["video"]["height"])], "raw_frame_manifest": frame_manifest["raw_frame_manifest"], "source_frame_manifest": frame_manifest["source_frame_manifest"]}, "camera_depth": {"state": "unmeasured", "required_for_metric_claims": True}, "hands": [], "objects": [], "contacts": [], "occlusions": [], "nonpenetration": [], "renderer_boundary": "renders consume V22 state files only"}
    write_json(run_root / "state" / "v22_physical_state.json", initial_state)
    write_json(run_root / "state" / "v22_uncertainty_state.json", {"schema": "v22_uncertainty_state.v0", "case_id": args.case_id, "states": []})
    write_json(run_root / "state" / "v22_observation_bundle.json", {"schema": "v22_observation_bundle.v0", "case_id": args.case_id, "observations": []})
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    summary = {"status": "ok", "method": "prepare_v22_single_video_run", "case_id": args.case_id, "run_root": str(run_root), "input_manifest": str(run_root / "input" / "input_manifest.json"), **frame_manifest, "elapsed_s": float(time.time() - started)}
    write_json(run_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
