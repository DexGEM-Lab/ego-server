#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prompt_rows(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"reference prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def rtmlib_frames(path: Path) -> tuple[dict[int, list[dict]], dict]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"RTMLib JSON has no frames list: {path}")
    return {int(frame["frame_idx"]): list(frame.get("hands", [])) for frame in frames}, payload


def scale_box(box: list[float], source_size: tuple[int, int], prompt_size: tuple[int, int], margin_ratio: float) -> list[float]:
    arr = np.asarray(box, dtype=float)
    if arr.shape != (4,) or not np.isfinite(arr).all() or arr[2] <= arr[0] or arr[3] <= arr[1]:
        raise RuntimeError(f"invalid detector box: {box}")
    w = float(arr[2] - arr[0])
    h = float(arr[3] - arr[1])
    margin = float(margin_ratio) * max(w, h)
    arr += np.asarray([-margin, -margin, margin, margin], dtype=float)
    sx = prompt_size[0] / float(source_size[0])
    sy = prompt_size[1] / float(source_size[1])
    scaled = arr * np.asarray([sx, sy, sx, sy], dtype=float)
    scaled[[0, 2]] = np.clip(scaled[[0, 2]], 0.0, prompt_size[0] - 1.0)
    scaled[[1, 3]] = np.clip(scaled[[1, 3]], 0.0, prompt_size[1] - 1.0)
    if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
        raise RuntimeError(f"detector box collapsed after scaling: {box}")
    return [float(v) for v in scaled]


def point_xy(point: dict) -> np.ndarray:
    return np.asarray([float(point["x"]), float(point["y"])], dtype=float)


def inside_box(point: dict, box: list[float]) -> bool:
    x, y = point_xy(point)
    return bool(box[0] <= x <= box[2] and box[1] <= y <= box[3])


def select_hand(hands: list[dict], ref: dict, source_size: tuple[int, int], prompt_size: tuple[int, int], args: argparse.Namespace) -> tuple[dict, dict]:
    positives = list(ref.get("positive_points", []))
    if not positives:
        raise RuntimeError(f"frame {ref.get('frame_idx')} has no positive reference points")
    scored = []
    pos_xy = np.stack([point_xy(point) for point in positives]).astype(float)
    for hand in hands:
        scores = np.asarray(hand.get("scores", []), dtype=float)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            continue
        mean_score = float(np.mean(finite_scores))
        if mean_score < float(args.min_mean_score):
            continue
        box = scale_box(hand.get("bbox_xyxy", []), source_size, prompt_size, float(args.box_margin_ratio))
        inside = int(sum(inside_box(point, box) for point in positives))
        center = np.asarray([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=float)
        mean_distance = float(np.mean(np.linalg.norm(pos_xy - center[None, :], axis=1)))
        hand_idx = int(hand.get("hand_idx", len(scored)))
        scored.append((inside, mean_score, -mean_distance, -hand_idx, hand, box))
    if not scored:
        raise RuntimeError(f"frame {ref.get('frame_idx')} has no RTMLib hand satisfying detector contract")
    scored.sort(reverse=True, key=lambda row: row[:4])
    inside, mean_score, neg_distance, neg_hand_idx, hand, box = scored[0]
    required = max(1, int(math.ceil(float(args.min_positive_inside_fraction) * len(positives))))
    if inside < required:
        raise RuntimeError(
            f"frame {ref.get('frame_idx')} selected RTMLib hand covers {inside}/{len(positives)} positive points, required {required}"
        )
    return hand, {
        "selected_rtmlib_hand_idx": int(-neg_hand_idx),
        "selected_rtmlib_mean_score": float(mean_score),
        "reference_positive_points_inside_detector_box": int(inside),
        "reference_positive_points": int(len(positives)),
        "mean_positive_distance_to_detector_center_px": float(-neg_distance),
        "detector_bbox_xyxy_prompt": box,
    }


def run(args: argparse.Namespace) -> dict:
    ref_by_frame, ref_payload = prompt_rows(args.reference_prompts)
    rtm_by_frame, rtm_payload = rtmlib_frames(args.rtmlib_json)
    video = rtm_payload.get("video") or ref_payload.get("video") or {}
    source_width = video.get("width", args.source_width)
    source_height = video.get("height", args.source_height)
    if source_width is None or source_height is None:
        raise RuntimeError("source image size is missing from RTMLib/reference payloads; pass --source-width and --source-height")
    source_size = (int(source_width), int(source_height))
    if source_size[0] <= 0 or source_size[1] <= 0:
        raise RuntimeError("source video size is missing")
    prompt_width_value = ref_payload.get("prompt_image_width", args.prompt_image_width)
    if prompt_width_value is None:
        raise RuntimeError("prompt image width is missing from reference payload; pass --prompt-image-width")
    prompt_width = int(prompt_width_value)
    prompt_size = (prompt_width, int(round(prompt_width * source_size[1] / source_size[0])))
    output_rows = []
    diagnostics = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in ref_by_frame:
            raise RuntimeError(f"reference prompt missing frame {frame_idx}")
        if frame_idx not in rtm_by_frame:
            raise RuntimeError(f"RTMLib detections missing frame {frame_idx}")
        ref = ref_by_frame[frame_idx]
        hand, diag = select_hand(rtm_by_frame[frame_idx], ref, source_size, prompt_size, args)
        positives = [point for point in ref.get("positive_points", []) if inside_box(point, diag["detector_bbox_xyxy_prompt"])]
        if len(positives) < int(args.min_positive_points):
            raise RuntimeError(f"frame {frame_idx} has only {len(positives)} positive points inside selected detector box")
        row = {
            "frame_idx": int(frame_idx),
            "target_visible": True,
            "confidence": float(hand.get("mean_score", diag["selected_rtmlib_mean_score"])),
            "bbox_xyxy": diag["detector_bbox_xyxy_prompt"],
            "positive_points": positives,
            "negative_points": list(ref.get("negative_points", [])),
            "visual_evidence": "RTMLib detector box fused with reference VLM hand points.",
        }
        output_rows.append(row)
        diagnostics.append({"frame_idx": int(frame_idx), **diag})
    payload = {
        "status": "ok",
        "backend": "RTMLib detector boxes fused with VLM hand point prompts for SAM2",
        "rtmlib_json": str(args.rtmlib_json),
        "reference_prompts": str(args.reference_prompts),
        "track_id": str(args.track_id or ref_payload.get("track_id", "rtmlib_hand")),
        "target_track_id": str(args.track_id or ref_payload.get("target_track_id", ref_payload.get("track_id", "rtmlib_hand"))),
        "description": str(ref_payload.get("description", "")),
        "prompt_image_width": int(prompt_width),
        "frames_prompted": int(len(output_rows)),
        "visible_frames": int(len(output_rows)),
        "point_prompts": output_rows,
        "diagnostics": diagnostics,
    }
    save_json(args.output_json, payload)
    print(json.dumps({k: payload[k] for k in ("status", "frames_prompted", "visible_frames", "track_id")}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtmlib-json", type=Path, required=True)
    parser.add_argument("--reference-prompts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--track-id", default="")
    parser.add_argument("--prompt-image-width", type=int)
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--source-height", type=int)
    parser.add_argument("--box-margin-ratio", type=float, default=0.04)
    parser.add_argument("--min-mean-score", type=float, default=0.25)
    parser.add_argument("--min-positive-inside-fraction", type=float, default=0.60)
    parser.add_argument("--min-positive-points", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
