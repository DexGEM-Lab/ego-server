#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SAM2_VLM_POINTS_STATUS = "measured_sam2_vlm_points"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_geometry(mask_path: Path, source_size: tuple[int, int]) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    mask_bool = mask > 0
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        raise RuntimeError(f"empty visible SAM2 mask {mask_path}")
    mask_height, mask_width = mask_bool.shape
    sx = source_size[0] / float(mask_width)
    sy = source_size[1] / float(mask_height)
    return {
        "center_xy": [float((xs.mean() + 0.5) * sx), float((ys.mean() + 0.5) * sy)],
        "bbox_xyxy": [
            float(xs.min() * sx),
            float(ys.min() * sy),
            float((xs.max() + 1) * sx),
            float((ys.max() + 1) * sy),
        ],
        "area_px": float(mask_bool.sum() * sx * sy),
        "mask_image_size": [int(mask_width), int(mask_height)],
    }


def prompt_rows(point_prompt_payload: dict) -> dict[int, dict]:
    rows = point_prompt_payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError("point prompt payload has no point_prompts list")
    return {int(row["frame_idx"]): row for row in rows}


def localize_mask_path(mask_path: str, remote_output_root: Path | None, local_output_root: Path | None) -> Path:
    path = Path(mask_path)
    if path.exists():
        return path
    if remote_output_root is not None and local_output_root is not None:
        try:
            rel = path.relative_to(remote_output_root)
        except ValueError:
            rel = None
        if rel is not None:
            candidate = local_output_root / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(mask_path)


def object_record(
    frame_idx: int,
    sam2_result: dict,
    prompt_payload: dict,
    prompt_by_frame: dict[int, dict],
    source_size: tuple[int, int],
    remote_output_root: Path | None,
    local_output_root: Path | None,
) -> dict:
    mask_path = localize_mask_path(str(sam2_result["mask_path"]), remote_output_root, local_output_root)
    prompt_row = prompt_by_frame.get(frame_idx, {})
    geometry = mask_geometry(mask_path, source_size)
    score = float(prompt_row.get("confidence", 1.0))
    if not np.isfinite(score):
        raise RuntimeError(f"non-finite prompt score for frame {frame_idx}")
    return {
        **geometry,
        "score": score,
        "sam_score": None,
        "proposal_source": "sam2_vlm_points",
        "status": SAM2_VLM_POINTS_STATUS,
        "track_id": str(prompt_payload["track_id"]),
        "label": str(prompt_payload["description"]),
        "prompts": [str(prompt_payload["description"])],
        "mask_path": str(mask_path),
        "source_image_size": [int(source_size[0]), int(source_size[1])],
        "point_prompt_visible": bool(prompt_row.get("target_visible", True)),
        "point_prompt_confidence": score,
        "point_prompt_visual_evidence": str(prompt_row.get("visual_evidence", "")),
        "point_prompt_object_plan": str(prompt_payload.get("object_plan")),
    }


def resolve_source_size(args: argparse.Namespace, prompt_payload: dict, annotations: dict) -> tuple[int, int]:
    if args.source_width is not None and args.source_height is not None:
        return int(args.source_width), int(args.source_height)
    video = prompt_payload.get("video") or {}
    if "width" in video and "height" in video:
        return int(video["width"]), int(video["height"])
    for frame in annotations.get("frames", []):
        obj = frame.get("object") or {}
        if obj.get("source_image_size"):
            width, height = obj["source_image_size"]
            return int(width), int(height)
    raise RuntimeError("cannot infer source image size; pass --source-width and --source-height")


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.base_annotations)
    track = load_json(args.sam2_track)
    prompt_payload = load_json(args.point_prompts)
    by_prompt = prompt_rows(prompt_payload)
    source_size = resolve_source_size(args, prompt_payload, annotations)
    frame_start = int(args.frame_start)
    frame_end = int(args.frame_end)
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    converted = 0
    invisible = 0
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < frame_start or frame_idx > frame_end:
            frame["object"] = {
                "status": "outside_sam2_vlm_points_window",
                "track_id": str(prompt_payload["track_id"]),
                "label": str(prompt_payload["description"]),
            }
            continue
        result = track.get(str(frame_idx)) or track.get(frame_idx)
        if not isinstance(result, dict) or not result.get("visible"):
            frame["object"] = {
                "status": "not_measured_sam2_vlm_points",
                "track_id": str(prompt_payload["track_id"]),
                "label": str(prompt_payload["description"]),
            }
            invisible += 1
            continue
        if not result.get("mask_path"):
            raise RuntimeError(f"visible SAM2 result lacks mask_path for frame {frame_idx}")
        frame["object"] = object_record(
            frame_idx,
            result,
            prompt_payload,
            by_prompt,
            source_size,
            args.remote_output_root,
            args.local_output_root,
        )
        converted += 1
    if converted == 0:
        raise RuntimeError("SAM2 track produced no visible annotation rows")
    adapter = {
        "status": "ok",
        "base_annotations": str(args.base_annotations),
        "sam2_track": str(args.sam2_track),
        "point_prompts": str(args.point_prompts),
        "track_id": str(prompt_payload["track_id"]),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "converted_visible_frames": converted,
        "not_measured_frames": invisible,
        "measured_status": SAM2_VLM_POINTS_STATUS,
    }
    annotations["sam2_vlm_points_adapter"] = adapter
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(json.dumps(adapter, indent=2))
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--sam2-track", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--source-height", type=int)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
