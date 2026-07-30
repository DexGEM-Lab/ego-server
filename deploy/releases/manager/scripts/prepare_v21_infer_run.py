#!/usr/bin/env python3
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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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


def stable_file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for _ in range(16):
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_first_16mb": digest.hexdigest(),
    }


def ensure_run_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ContractError(f"run_root_exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise ContractError(f"could_not_open_writer: {path}")
    return writer


def clip_single_video(
    source: Path,
    output: Path,
    start_frame: int,
    frame_count: int,
    crop: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    meta = video_metadata(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if crop is None:
        out_size = (int(meta["width"]), int(meta["height"]))
    else:
        x0, y0, x1, y1 = crop
        out_size = (int(x1 - x0), int(y1 - y0))
    writer = open_writer(output, float(meta["fps"]), out_size)
    written = 0
    try:
        for local_idx in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: {source} frame={start_frame + local_idx}")
            if crop is not None:
                x0, y0, x1, y1 = crop
                frame = frame[y0:y1, x0:x1]
            if frame.shape[1] != out_size[0] or frame.shape[0] != out_size[1]:
                raise ContractError(f"unexpected_frame_size: {frame.shape} expected={out_size}")
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    out_meta = video_metadata(output)
    return {"source": str(source), "output": str(output), "start_frame": start_frame, "frame_count": written, "video": out_meta}


def extract_manifest_frames(clip: Path, run_root: Path, render_width: int, source_start_frame: int = 0) -> dict[str, Any]:
    meta = video_metadata(clip)
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {clip}")
    render_height = int(round(render_width * meta["height"] / meta["width"]))
    if render_height % 2:
        render_height += 1
    rgb_dir = run_root / "input" / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    try:
        for frame_idx in range(int(meta["frame_count"])):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: {clip} frame={frame_idx}")
            resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            frame_path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise ContractError(f"could_not_write_frame: {frame_path}")
            frames.append(
                {
                    "index": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "source_frame_idx": int(source_start_frame + frame_idx),
                    "time_s": float(frame_idx / meta["fps"]),
                    "source_time_s": float((source_start_frame + frame_idx) / meta["fps"]),
                    "rgb": str(frame_path),
                    "source_width": int(meta["width"]),
                    "source_height": int(meta["height"]),
                    "manifest_width": int(render_width),
                    "manifest_height": int(render_height),
                }
            )
    finally:
        cap.release()
    manifest = {
        "schema": "v21_raw_frame_manifest.v0",
        "status": "ok",
        "method": "prepare_v21_infer_run.extract_manifest_frames",
        "clip": str(clip),
        "video": meta,
        "render_width": int(render_width),
        "render_height": int(render_height),
        "frames": frames,
    }
    manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    write_json(manifest_path, manifest)
    return {"path": manifest_path, "frame_count": len(frames), "video": meta, "render_width": render_width, "render_height": render_height}


def living_room_sources(root: Path) -> list[dict[str, Any]]:
    videos = sorted(root.glob("observation.images.camera*/chunk-000/file-000.mp4"))
    if not videos:
        videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    if not videos:
        raise ContractError(f"no_videos_found: {root}")
    sources = []
    for path in videos:
        camera_id = path.parts[-3] if len(path.parts) >= 3 else path.stem
        sources.append({"camera_id": camera_id, "path": str(path), "metadata": video_metadata(path)})
    return sources


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ensure_run_root(args.run_root, args.overwrite)
    clip_dir = args.run_root / "input" / "clips"
    source_records: list[dict[str, Any]] = []
    multiview_sources = None
    stereo_right = None
    source_span = None

    if args.input_kind == "pico_side_by_side":
        if args.input_video is None:
            raise ContractError("--input-video is required for pico_side_by_side")
        source = args.input_video
        meta = video_metadata(source)
        start_frame = int(round(float(args.start_s) * float(meta["fps"])))
        end_frame = int(round(float(args.end_s) * float(meta["fps"])))
        frame_count = max(1, end_frame - start_frame)
        if int(meta["width"]) % 2 != 0:
            raise ContractError(f"side_by_side_width_not_even: {meta['width']}")
        half_width = int(meta["width"]) // 2
        left_clip = clip_dir / "pico_left_100s_120s.mp4"
        right_clip = clip_dir / "pico_right_100s_120s.mp4"
        source_records.append(clip_single_video(source, left_clip, start_frame, frame_count, (0, 0, half_width, int(meta["height"]))))
        right_record = clip_single_video(source, right_clip, start_frame, frame_count, (half_width, 0, int(meta["width"]), int(meta["height"])))
        source_records.append(right_record)
        primary_video = left_clip
        stereo_right = right_clip
        input_modality = "side_by_side_stereo_pair_uncalibrated"
        source_span = {"start_s": float(args.start_s), "end_s": float(args.end_s), "start_frame": start_frame, "end_frame_exclusive": end_frame}
        raw_manifest = extract_manifest_frames(primary_video, args.run_root, int(args.render_width), start_frame)
    elif args.input_kind == "living_room_multiview":
        if args.input_root is None:
            raise ContractError("--input-root is required for living_room_multiview")
        multiview_sources = living_room_sources(args.input_root)
        by_camera = {row["camera_id"]: Path(row["path"]) for row in multiview_sources}
        primary_video = by_camera.get(args.primary_camera) or by_camera.get(f"observation.images.{args.primary_camera}")
        if primary_video is None:
            primary_video = Path(multiview_sources[0]["path"])
        right_video = by_camera.get(args.stereo_camera) or by_camera.get(f"observation.images.{args.stereo_camera}")
        if right_video is not None:
            stereo_right = right_video
        input_modality = "synchronized_multiview_uncalibrated_with_primary_view"
        raw_manifest = extract_manifest_frames(primary_video, args.run_root, int(args.render_width), 0)
    else:
        raise ContractError(f"unsupported_input_kind: {args.input_kind}")

    input_manifest = {
        "schema": "v21_input_manifest.v0",
        "mode": "v21_infer",
        "case_id": args.case_id,
        "run_root": str(args.run_root),
        "input_kind": args.input_kind,
        "input_modality": input_modality,
        "primary_video": str(primary_video),
        "primary_video_metadata": video_metadata(primary_video),
        "stereo_right_video": str(stereo_right) if stereo_right else None,
        "stereo_right_metadata": video_metadata(stereo_right) if stereo_right else None,
        "multiview_sources": multiview_sources,
        "source_span": source_span,
        "source_fingerprint": stable_file_fingerprint(args.input_video) if args.input_video else None,
        "raw_frame_manifest": str(raw_manifest["path"]),
        "raw_frame_manifest_summary": raw_manifest,
        "claim_scope": "V21 input/run bootstrap only. It decodes the requested timeline and modality; no physical depth, segmentation, MANO, object geometry, pose graph, or renderable annotation has run.",
    }
    write_json(args.run_root / "input" / "input_manifest.json", input_manifest)
    state = {
        "schema": "v21_physical_state.v0",
        "status": "input_timeline_bootstrap_complete_physical_measurements_pending",
        "case_id": args.case_id,
        "run_root": str(args.run_root),
        "timeline": {
            "frame_count": int(raw_manifest["frame_count"]),
            "fps": float(raw_manifest["video"]["fps"]),
            "duration_s": float(raw_manifest["video"]["duration_s"]),
            "resolution": [int(raw_manifest["video"]["width"]), int(raw_manifest["video"]["height"])],
            "raw_frame_manifest": str(raw_manifest["path"]),
        },
        "camera_depth": {"state": "unmeasured", "required_for_metric_claims": True},
        "segmentation": {"state": "unmeasured", "required_for_object_geometry": True},
        "hands": {"state": "unmeasured", "metric_mano_required_for_contact": True},
        "objects": {"state": "unmeasured", "mesh_pose_required_for_object_pose_claims": True},
        "renderer_boundary": "V21 renders must consume state/annotations_v21_renderable.json assembled from selected physical mechanism outputs.",
    }
    write_json(args.run_root / "state" / "v21_physical_state.json", state)
    summary = {
        "status": "ok",
        "method": "prepare_v21_infer_run",
        "case_id": args.case_id,
        "run_root": str(args.run_root),
        "input_manifest": str(args.run_root / "input" / "input_manifest.json"),
        "raw_frame_manifest": str(raw_manifest["path"]),
        "frame_count": int(raw_manifest["frame_count"]),
        "elapsed_s": float(time.time() - started),
        "next_required_physical_stage": "build_v21_depth_modality_report_then_run_depth_camera_candidates",
    }
    write_json(args.run_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a V21 infer run root from raw video inputs without claiming physical annotation.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("pico_side_by_side", "living_room_multiview"), required=True)
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=0.0)
    parser.add_argument("--primary-camera", default="observation.images.camera2")
    parser.add_argument("--stereo-camera", default="observation.images.camera1")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
