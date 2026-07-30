#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


SAM2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)


COLORS_BGR = [
    (60, 80, 255),
    (80, 230, 80),
    (255, 150, 60),
    (60, 220, 255),
    (255, 80, 220),
    (180, 120, 255),
    (120, 255, 220),
]


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class Track:
    obj_id: int
    track_id: str
    description: str
    prompt_path: Path
    prompts: dict[int, dict]
    active_intervals: tuple[tuple[int, int], ...]
    payload: dict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_idx}")
    return frame


def put_caption(frame: np.ndarray, caption: str, frame_idx: int) -> None:
    text = f"{frame_idx:04d}  {caption}"
    cv2.rectangle(frame, (0, frame.shape[0] - 34), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def prompt_rows(payload: dict) -> dict[int, dict]:
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError("point prompt payload has no point_prompts list")
    return {int(row["frame_idx"]): row for row in rows}


def active_intervals(payload: dict) -> tuple[tuple[int, int], ...]:
    intervals = []
    object_plan = payload.get("object_plan_payload") or payload.get("object_plan_record") or payload.get("target_object_plan")
    if not isinstance(object_plan, dict):
        object_plan = {}
    for row in object_plan.get("active_intervals") or []:
        if not isinstance(row, dict):
            continue
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if end < start:
            raise RuntimeError(f"invalid active interval for {payload.get('track_id')}: {row}")
        intervals.append((start, end))
    if intervals:
        return tuple(intervals)
    prompt_indices = sorted(int(row["frame_idx"]) for row in payload.get("point_prompts", []))
    if not prompt_indices:
        raise RuntimeError(f"track {payload.get('track_id')} has no prompt frames")
    return ((prompt_indices[0], prompt_indices[-1]),)


def track_active(track: Track, frame_idx: int) -> bool:
    return any(start <= frame_idx <= end for start, end in track.active_intervals)


def load_tracks(point_root: Path) -> list[Track]:
    files = sorted(point_root.glob("*/object_point_prompts_vlm.json"))
    if not files:
        raise RuntimeError(f"no object_point_prompts_vlm.json files under {point_root}")
    tracks = []
    seen = set()
    for i, path in enumerate(files, start=1):
        payload = load_json(path)
        track_id = str(payload["track_id"])
        if track_id in seen:
            raise RuntimeError(f"duplicate track_id: {track_id}")
        seen.add(track_id)
        tracks.append(
            Track(
                obj_id=i,
                track_id=track_id,
                description=str(payload["description"]),
                prompt_path=path,
                prompts=prompt_rows(payload),
                active_intervals=active_intervals(payload),
                payload=payload,
            )
        )
    return tracks


def selected_frames(frame_start: int, frame_end: int) -> list[dict]:
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    return [{"frame_idx": idx} for idx in range(frame_start, frame_end + 1)]


def extract_frames(clip: Path, frames: list[dict], output_dir: Path, image_width: int) -> Path:
    frame_dir = output_dir / "sam2_shared_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(clip)
    image_height = int(round(info.height * image_width / info.width))
    try:
        for local_idx, frame in enumerate(frames):
            source_idx = int(frame["frame_idx"])
            image = read_video_frame(cap, source_idx)
            resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            path = frame_dir / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write {path}")
    finally:
        cap.release()
    return frame_dir


def infer_prompt_size(payload: dict, source_size: tuple[int, int]) -> tuple[int, int]:
    """Return the coordinate size in which prompt points are expressed.

    Runtime agents may provide source-video pixel points while also storing a
    smaller prompt-image width for review images. SAM2 must scale from the actual
    point coordinate frame; otherwise source-frame keyboard points outside a
    review-sized frame are pushed to the image edge and produce false masks.
    """
    frame = str(payload.get("point_coordinate_frame") or payload.get("coordinate_frame") or "")
    match = re.search(r"source_video_pixels[_:](\d+)x(\d+)", frame)
    if match:
        return int(match.group(1)), int(match.group(2))
    if frame in {"source_video_pixels", "source_pixels", "raw_video_pixels"}:
        return int(source_size[0]), int(source_size[1])
    if "source_video_pixels" in frame and not match:
        return int(source_size[0]), int(source_size[1])
    if payload.get("prompt_image_width") is None:
        raise RuntimeError("prompt payload missing prompt_image_width and point_coordinate_frame is not source pixels")
    width = int(payload["prompt_image_width"])
    height = int(payload.get("prompt_image_height") or round(width * source_size[1] / source_size[0]))
    return width, height


def validate_points_in_frame(points: list[dict], prompt_size: tuple[int, int], context: str) -> None:
    for point in points:
        x = float(point.get("x"))
        y = float(point.get("y"))
        if not (0.0 <= x < float(prompt_size[0]) and 0.0 <= y < float(prompt_size[1])):
            raise RuntimeError(f"{context} point ({x:.1f},{y:.1f}) outside declared coordinate frame {prompt_size}")


def scaled_points(points: list[dict], prompt_size: tuple[int, int], video_size: tuple[int, int], context: str) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    validate_points_in_frame(points, prompt_size, context)
    scale = np.asarray([video_size[0] / prompt_size[0], video_size[1] / prompt_size[1]], dtype=np.float32)
    return np.asarray([[float(point["x"]) * scale[0], float(point["y"]) * scale[1]] for point in points], dtype=np.float32)


def scaled_box(box_xyxy: Any, prompt_size: tuple[int, int], video_size: tuple[int, int], context: str) -> np.ndarray | None:
    if box_xyxy is None:
        return None
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        raise RuntimeError(f"{context} box_xyxy must be a 4-number list")
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if not (0.0 <= x1 < x2 <= float(prompt_size[0]) and 0.0 <= y1 < y2 <= float(prompt_size[1])):
        raise RuntimeError(f"{context} box {box_xyxy} outside declared coordinate frame {prompt_size}")
    scale = np.asarray([video_size[0] / prompt_size[0], video_size[1] / prompt_size[1]], dtype=np.float32)
    return np.asarray([x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1]], dtype=np.float32)


def prompt_box_payload(prompt: dict) -> Any:
    return prompt.get("box_xyxy") or prompt.get("object_box_xyxy") or prompt.get("visible_object_box_xyxy")


def prompt_has_spatial_constraint(prompt: dict) -> bool:
    return bool(prompt.get("positive_points")) or prompt_box_payload(prompt) is not None


def prompt_points(
    track: Track,
    source_idx: int,
    tracks: list[Track],
    prompt_sizes: dict[str, tuple[int, int]],
    video_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompt = track.prompts[source_idx]
    positives = prompt.get("positive_points", [])
    negatives = prompt.get("negative_points", [])
    own_prompt_size = prompt_sizes[track.track_id]
    pos = scaled_points(positives, own_prompt_size, video_size, f"{track.track_id} frame {source_idx} positive")
    own_neg = scaled_points(negatives, own_prompt_size, video_size, f"{track.track_id} frame {source_idx} negative")
    competing_pos = []
    for other in tracks:
        if other.track_id == track.track_id:
            continue
        other_prompt = other.prompts.get(source_idx)
        if other_prompt and other_prompt.get("target_visible") and other_prompt.get("positive_points"):
            competing_pos.append(
                scaled_points(
                    other_prompt["positive_points"],
                    prompt_sizes[other.track_id],
                    video_size,
                    f"{other.track_id} frame {source_idx} competing positive",
                )
            )
    extra_neg = np.vstack(competing_pos).astype(np.float32) if competing_pos else np.zeros((0, 2), dtype=np.float32)
    neg = np.vstack([own_neg, extra_neg]).astype(np.float32) if len(own_neg) or len(extra_neg) else np.zeros((0, 2), dtype=np.float32)
    points = np.vstack([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    return points, labels, pos


def positive_prompt_box(
    positive_points: np.ndarray,
    video_size: tuple[int, int],
    *,
    pad_ratio: float,
    min_pad_px: float,
) -> np.ndarray | None:
    """Build a generic SAM2 box from model/agent-produced positive surface clicks.

    The box is not category-specific. It constrains SAM2 to the local surface
    supported by the positive clicks while negative clicks still suppress hands,
    table, and competing objects. Without this, sparse points can let the video
    predictor choose a broad connected support region and contaminate geometry.
    """
    if positive_points.ndim != 2 or positive_points.shape[0] < 2 or positive_points.shape[1] != 2:
        return None
    xy_min = positive_points.min(axis=0)
    xy_max = positive_points.max(axis=0)
    diag = float(np.linalg.norm(xy_max - xy_min))
    pad = max(float(min_pad_px), float(pad_ratio) * diag)
    x1 = max(0.0, float(xy_min[0] - pad))
    y1 = max(0.0, float(xy_min[1] - pad))
    x2 = min(float(video_size[0] - 1), float(xy_max[0] + pad))
    y2 = min(float(video_size[1] - 1), float(xy_max[1] + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)
    return box, float(xs.size), center


def point_hits(mask: np.ndarray, points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return int(mask[y, x].sum())


def prompt_contract(mask: np.ndarray, points: np.ndarray, labels: np.ndarray) -> dict:
    pos = points[labels == 1]
    neg = points[labels == 0]
    return {
        "positive_hits": point_hits(mask, pos),
        "positive_points": int(len(pos)),
        "negative_hits": point_hits(mask, neg),
        "negative_points": int(len(neg)),
        "satisfies_prompt_contract": bool(point_hits(mask, pos) == len(pos) and point_hits(mask, neg) == 0),
    }


def add_prompt_frames(
    predictor,
    state,
    tracks: list[Track],
    frames: list[dict],
    prompt_frames: list[int],
    prompt_sizes: dict[str, tuple[int, int]],
    video_size: tuple[int, int],
    *,
    use_positive_prompt_box: bool,
    prompt_box_pad_ratio: float,
    prompt_box_min_pad_px: float,
) -> list[dict]:
    selected = [int(frame["frame_idx"]) for frame in frames]
    local_by_source = {source_idx: local for local, source_idx in enumerate(selected)}
    reports = []
    for source_idx in prompt_frames:
        if source_idx not in local_by_source:
            continue
        for track in tracks:
            prompt = track.prompts.get(source_idx)
            if not prompt or not prompt.get("target_visible") or not prompt_has_spatial_constraint(prompt):
                continue
            prompt_box = scaled_box(
                prompt_box_payload(prompt),
                prompt_sizes[track.track_id],
                video_size,
                f"{track.track_id} frame {source_idx}",
            )
            points, labels, positive_points = prompt_points(track, source_idx, tracks, prompt_sizes, video_size)
            box = prompt_box
            if box is None and use_positive_prompt_box:
                box = positive_prompt_box(
                    positive_points,
                    video_size,
                    pad_ratio=float(prompt_box_pad_ratio),
                    min_pad_px=float(prompt_box_min_pad_px),
                )
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_by_source[source_idx],
                obj_id=track.obj_id,
                points=None if len(points) == 0 else points,
                labels=None if len(labels) == 0 else labels,
                box=box,
            )
            ids = [int(v) for v in out_obj_ids]
            report = {
                "frame_idx": int(source_idx),
                "track_id": track.track_id,
                "obj_id": int(track.obj_id),
                "sam2_out_frame_idx": int(out_frame_idx),
                "object_ids_after_prompt": ids,
                "point_coordinate_frame": str(track.payload.get("point_coordinate_frame") or track.payload.get("coordinate_frame") or ""),
                "prompt_coordinate_size": list(prompt_sizes[track.track_id]),
                "sam2_video_size": list(video_size),
                "prompt_box_source": "prompt_json_box_xyxy" if prompt_box is not None else ("positive_points_derived_box" if box is not None else None),
                "positive_prompt_box_enabled": bool(use_positive_prompt_box),
                "positive_prompt_box_xyxy": None if box is None else [float(v) for v in box.tolist()],
                "positive_prompt_box_pad_ratio": float(prompt_box_pad_ratio),
                "positive_prompt_box_min_pad_px": float(prompt_box_min_pad_px),
            }
            if track.obj_id in ids:
                obj_pos = ids.index(track.obj_id)
                mask = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0)
                report.update(prompt_contract(mask, points, labels))
                report["area_px"] = int(mask.sum())
            reports.append(report)
            print(
                json.dumps(
                    {
                        "event": "prompt_added",
                        "frame_idx": int(source_idx),
                        "track_id": track.track_id,
                        "obj_id": int(track.obj_id),
                        "area_px": int(report.get("area_px", 0)),
                        "prompt_contract": bool(report.get("satisfies_prompt_contract", False)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return reports


def write_track_results(
    args: argparse.Namespace,
    tracks: list[Track],
    frames: list[dict],
    propagated: dict[int, dict[int, np.ndarray]],
    prompt_reports: list[dict],
    prompt_frames: list[int],
    video_size: tuple[int, int],
    source_size: tuple[int, int],
) -> dict[str, dict[int, dict]]:
    sx = source_size[0] / float(video_size[0])
    sy = source_size[1] / float(video_size[1])
    all_results: dict[str, dict[int, dict]] = {}
    for track in tracks:
        out_dir = args.output_root / track.track_id / "sam2"
        mask_dir = out_dir / "sam2_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, dict] = {}
        for frame in frames:
            source_idx = int(frame["frame_idx"])
            if not track_active(track, source_idx):
                results[source_idx] = {"visible": False, "area_px": 0.0, "failure_reason": "outside_vlm_active_interval"}
                continue
            mask = propagated.get(source_idx, {}).get(track.obj_id)
            if mask is None or int(mask.sum()) == 0:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            box_small, area_small, center_small = mask_box(mask)
            if box_small is None or center_small is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            results[source_idx] = {
                "visible": True,
                "bbox_xyxy": [float(box_small[0] * sx), float(box_small[1] * sy), float(box_small[2] * sx), float(box_small[3] * sy)],
                "center_xy": [float(center_small[0] * sx), float(center_small[1] * sy)],
                "area_px": float(area_small * sx * sy),
                "mask_path": str(mask_path),
            }
        visible = sum(1 for row in results.values() if row.get("visible"))
        track_reports = [row for row in prompt_reports if row["track_id"] == track.track_id]
        qc = {
            "status": "ok",
            "backend": "SAM2 multi-object propagation from spatial prompts",
            "clip": str(args.clip),
            "point_prompts": str(track.prompt_path),
            "track_id": track.track_id,
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "frames": len(frames),
            "active_intervals": [[int(start), int(end)] for start, end in track.active_intervals],
            "prompt_frames": prompt_frames,
            "visible_frames": visible,
            "checkpoint": str(args.checkpoint),
            "model_cfg": args.model_cfg,
            "non_overlap_masks": True,
            "prompt_contract_reports": track_reports,
            "outputs": {
                "sam2_track": str(out_dir / "sam2_track.json"),
                "sam2_masks": str(mask_dir),
            },
        }
        (out_dir / "sam2_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "qc_sam2_vlm_points_track.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        all_results[track.track_id] = results
    return all_results


def render_combined(args: argparse.Namespace, tracks: list[Track], frames: list[dict], all_results: dict[str, dict[int, dict]]) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_root / "sam2_multiobject_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    try:
        for frame in frames:
            source_idx = int(frame["frame_idx"])
            image = read_video_frame(cap, source_idx)
            image = cv2.resize(image, (args.render_width, height), interpolation=cv2.INTER_AREA)
            for i, track in enumerate(tracks):
                result = all_results[track.track_id].get(source_idx, {})
                if not result.get("visible") or not result.get("mask_path"):
                    continue
                mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read {result['mask_path']}")
                mask = cv2.resize(mask, (args.render_width, height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(image)
                tint[:, :] = np.asarray(COLORS_BGR[i % len(COLORS_BGR)], dtype=np.uint8)
                image[mask] = cv2.addWeighted(image, 0.58, tint, 0.42, 0.0)[mask]
            y = 18
            for i, track in enumerate(tracks):
                color = COLORS_BGR[i % len(COLORS_BGR)]
                cv2.rectangle(image, (8, y - 10), (22, y + 4), color, -1)
                cv2.putText(image, track.track_id[:38], (30, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
            put_caption(image, "SAM2 multi-surface non-overlap", source_idx)
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(args.point_root)
    frames = selected_frames(int(args.frame_start), int(args.frame_end))
    cap, info = open_video(args.clip)
    cap.release()
    video_height = int(round(info.height * int(args.sam2_image_width) / info.width))
    video_size = (int(args.sam2_image_width), video_height)
    source_size = (int(info.width), int(info.height))
    prompt_sizes = {track.track_id: infer_prompt_size(track.payload, source_size) for track in tracks}
    for track in tracks:
        for prompt in track.prompts.values():
            if prompt.get("target_visible"):
                validate_points_in_frame(
                    list(prompt.get("positive_points", [])) + list(prompt.get("negative_points", [])),
                    prompt_sizes[track.track_id],
                    f"{track.track_id} frame {prompt.get('frame_idx')} prompt",
                )
    frame_dir = extract_frames(args.clip, frames, args.output_root, int(args.sam2_image_width))
    selected = {int(frame["frame_idx"]) for frame in frames}
    prompt_frames = sorted(
        {
            frame_idx
            for track in tracks
            for frame_idx, prompt in track.prompts.items()
            if frame_idx in selected and prompt.get("target_visible") and prompt_has_spatial_constraint(prompt)
        }
    )
    if not prompt_frames:
        raise RuntimeError("no visible prompt frames inside selected range")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 video predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=device, vos_optimized=False)
    predictor.non_overlap_masks = True
    predictor.add_all_frames_to_correct_as_cond = True
    propagated: dict[int, dict[int, np.ndarray]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        prompt_reports = add_prompt_frames(
            predictor,
            state,
            tracks,
            frames,
            prompt_frames,
            prompt_sizes,
            video_size,
            use_positive_prompt_box=bool(args.use_positive_prompt_box),
            prompt_box_pad_ratio=float(args.prompt_box_pad_ratio),
            prompt_box_min_pad_px=float(args.prompt_box_min_pad_px),
        )
        selected_frames_list = [int(frame["frame_idx"]) for frame in frames]
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            source_idx = selected_frames_list[int(out_frame_idx)]
            ids = [int(v) for v in out_obj_ids]
            per_frame: dict[int, np.ndarray] = {}
            for obj_pos, obj_id in enumerate(ids):
                per_frame[obj_id] = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0).astype(np.uint8)
            propagated[source_idx] = per_frame
            if int(out_frame_idx) % 10 == 0 or int(out_frame_idx) == len(selected_frames_list) - 1:
                print(json.dumps({"event": "propagated", "local_frame": int(out_frame_idx), "source_frame": int(source_idx), "object_ids": ids}, sort_keys=True), flush=True)
    all_results = write_track_results(args, tracks, frames, propagated, prompt_reports, prompt_frames, video_size, (info.width, info.height))
    overlay = render_combined(args, tracks, frames, all_results)
    summary = {
        "status": "ok",
        "backend": "SAM2 multi-object propagation from spatial prompts",
        "clip": str(args.clip),
        "point_root": str(args.point_root),
        "output_root": str(args.output_root),
        "track_ids": [track.track_id for track in tracks],
        "active_intervals_by_track": {
            track.track_id: [[int(start), int(end)] for start, end in track.active_intervals] for track in tracks
        },
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "prompt_frames": prompt_frames,
        "visible_frames_by_track": {
            track.track_id: int(sum(1 for row in all_results[track.track_id].values() if row.get("visible")))
            for track in tracks
        },
        "prompt_contract_satisfied": int(sum(1 for row in prompt_reports if row.get("satisfies_prompt_contract"))),
        "prompt_contract_reports": len(prompt_reports),
        "prompt_coordinate_sizes_by_track": {track_id: list(size) for track_id, size in prompt_sizes.items()},
        "source_video_size": list(source_size),
        "sam2_video_size": list(video_size),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "non_overlap_masks": True,
        "positive_prompt_box_enabled": bool(args.use_positive_prompt_box),
        "prompt_box_pad_ratio": float(args.prompt_box_pad_ratio),
        "prompt_box_min_pad_px": float(args.prompt_box_min_pad_px),
        "overlay": str(overlay),
        "elapsed_s": time.time() - started,
    }
    (args.output_root / "qc_sam2_multiobject_points.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "track_ids"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument(
        "--use-positive-prompt-box",
        action="store_true",
        help="Constrain each SAM2 conditioning frame with a box derived from positive prompt points, while still applying positive/negative clicks.",
    )
    parser.add_argument("--prompt-box-pad-ratio", type=float, default=0.18)
    parser.add_argument("--prompt-box-min-pad-px", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
