#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from build_object_plan_vlm import load_env_file
from fuse_v1_full_fidelity import DEFAULT_CLIP, load_json, open_video, read_video_frame


VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_idx": {"type": "integer"},
                    "mask_correct_for_track": {"type": "boolean"},
                    "target_visible": {"type": "boolean"},
                    "dominant_mask_object": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "frame_idx",
                    "mask_correct_for_track",
                    "target_visible",
                    "dominant_mask_object",
                    "visual_evidence",
                    "confidence",
                ],
            },
        }
    },
    "required": ["verdicts"],
}


def image_data_url(image: np.ndarray, quality: int = 86) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode review image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def localize_mask_path(mask_path: str, remote_root: str | None, local_root: str | None) -> Path:
    if remote_root and local_root and mask_path.startswith(remote_root):
        return Path(local_root + mask_path[len(remote_root) :])
    return Path(mask_path)


def plan_objects(plan_path: Path) -> list[dict]:
    blob = load_json(plan_path)
    plan = blob["plan"] if "plan" in blob else blob
    objects = plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"object plan has no objects: {plan_path}")
    return objects


def measured_indices(frames: list[dict], track_id: str, frame_start: int, frame_end: int, stride: int) -> list[int]:
    out = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        if idx < frame_start or idx > frame_end or idx % stride:
            continue
        obj = frame.get("object") or {}
        if obj.get("track_id") == track_id and str(obj.get("status", "")).startswith("measured"):
            out.append(idx)
    return out


def build_sheet(
    cap,
    frames: list[dict],
    index_by_frame: dict[int, int],
    indices: list[int],
    object_plan: dict,
    args: argparse.Namespace,
) -> np.ndarray:
    tile_w = int(args.tile_width)
    tile_h = int(round(tile_w * 9 / 16))
    tiles = []
    for idx in indices:
        frame_ann = frames[index_by_frame[idx]]
        frame = read_video_frame(cap, idx)
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        obj = frame_ann.get("object") or {}
        mask = None
        mask_path = obj.get("mask_path")
        if mask_path:
            local = localize_mask_path(mask_path, args.remote_output_root, args.local_output_root)
            raw = cv2.imread(str(local), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise RuntimeError(f"mask image missing: {local}")
            mask = raw > 0
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mask = cv2.resize(mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST) > 0
        if mask is not None:
            overlay = tile.copy()
            overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
            tile = overlay
        if obj.get("bbox_xyxy"):
            sx = tile_w / frame.shape[1]
            sy = tile_h / frame.shape[0]
            x1, y1, x2, y2 = [float(v) for v in obj["bbox_xyxy"]]
            cv2.rectangle(tile, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 255, 255), 2)
        label = f"frame {idx} target {object_plan['track_id']}"
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cols = min(len(tiles), int(args.sheet_cols))
    rows = int(np.ceil(len(tiles) / max(1, cols)))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = (28, 28, 28)
    for i, tile in enumerate(tiles):
        row = i // cols
        col = i % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return sheet


def call_verifier(args: argparse.Namespace, object_plan: dict, all_objects: list[dict], indices: list[int], sheet: np.ndarray) -> list[dict]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    object_brief = [
        {
            "track_id": obj["track_id"],
            "description": obj["description"],
            "prompts": obj.get("open_vocabulary_prompts", []),
        }
        for obj in all_objects
    ]
    content = [
        {
            "type": "input_text",
            "text": (
                "You are verifying egocentric object masks. The magenta overlay is the proposed mask. "
                "For each tile, decide whether the magenta mask primarily covers the target object track. "
                "If the target object is hidden or only a tiny ambiguous fragment is visible, set target_visible false. "
                "If the magenta mask covers a different planned object, put that track_id in dominant_mask_object. "
                "Use 'background_or_unknown' when it is neither a planned object nor the target. "
                "Return one verdict for each requested frame index.\n\n"
                f"Target object:\n{json.dumps(object_plan, ensure_ascii=True)}\n\n"
                f"All planned objects:\n{json.dumps(object_brief, ensure_ascii=True)}\n\n"
                f"Requested frame indices: {indices}"
            ),
        },
        {"type": "input_image", "image_url": image_data_url(sheet), "detail": args.detail},
    ]
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_object_mask_verdicts",
                "strict": True,
                "schema": VERDICT_SCHEMA,
            }
        },
    }
    with httpx.Client(timeout=float(args.timeout_s)) as client:
        response = client.post(
            f"{args.base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API failed {response.status_code}: {response.text[:1000]}")
    body = response.json()
    output_text = body.get("output_text")
    if output_text is None:
        texts = []
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        output_text = "\n".join(texts)
    if not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    parsed = json.loads(output_text)
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list):
        raise RuntimeError(f"verifier returned invalid verdicts: {parsed}")
    by_frame = {int(v["frame_idx"]): v for v in verdicts}
    missing = [idx for idx in indices if idx not in by_frame]
    if missing:
        raise RuntimeError(f"verifier omitted frame verdicts: {missing}")
    return [by_frame[idx] for idx in indices]


def apply_verdicts(frames: list[dict], index_by_frame: dict[int, int], verdicts: list[dict]) -> dict:
    counts = {"accepted": 0, "rejected_wrong_object": 0, "rejected_not_visible": 0}
    for verdict in verdicts:
        idx = int(verdict["frame_idx"])
        frame = frames[index_by_frame[idx]]
        obj = frame.get("object") or {}
        obj["vlm_verification"] = verdict
        if bool(verdict["mask_correct_for_track"]) and bool(verdict["target_visible"]):
            obj["status"] = "measured_plan_sam_vlm_verified"
            counts["accepted"] += 1
        elif not bool(verdict["target_visible"]):
            obj["status"] = "unobserved_vlm_target_not_visible"
            counts["rejected_not_visible"] += 1
        else:
            obj["status"] = "rejected_vlm_wrong_object"
            counts["rejected_wrong_object"] += 1
        frame["object"] = obj
    return counts


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    frames = annotations["frames"]
    index_by_frame = {int(frame["frame_idx"]): i for i, frame in enumerate(frames)}
    all_objects = plan_objects(args.object_plan)
    if args.object_index >= len(all_objects):
        raise RuntimeError(f"--object-index {args.object_index} out of range for {len(all_objects)} planned objects")
    object_plan = all_objects[args.object_index]
    frame_start = min(index_by_frame) if args.frame_start is None else int(args.frame_start)
    frame_end = max(index_by_frame) if args.frame_end is None else int(args.frame_end)
    indices = measured_indices(frames, object_plan["track_id"], frame_start, frame_end, max(1, int(args.frame_stride)))
    if args.max_frames is not None:
        indices = indices[: int(args.max_frames)]
    if not indices:
        raise RuntimeError("no measured masks selected for verification")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, _ = open_video(args.clip)
    all_verdicts: list[dict] = []
    batch_counts = []
    try:
        for batch_id, start in enumerate(range(0, len(indices), max(1, int(args.batch_size)))):
            batch = indices[start : start + int(args.batch_size)]
            sheet = build_sheet(cap, frames, index_by_frame, batch, object_plan, args)
            sheet_path = args.output_dir / "review_sheets" / f"batch_{batch_id:04d}_{batch[0]:06d}_{batch[-1]:06d}.jpg"
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sheet_path), sheet)
            verdicts = call_verifier(args, object_plan, all_objects, batch, sheet)
            all_verdicts.extend(verdicts)
            batch_counts.append({"batch": batch_id, "frames": batch, "sheet": str(sheet_path)})
    finally:
        cap.release()

    counts = apply_verdicts(frames, index_by_frame, all_verdicts)
    annotations_path = args.output_dir / "annotations_plan_masks_vlm_verified.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    verdict_path = args.output_dir / "vlm_mask_verdicts.json"
    verdict_path.write_text(json.dumps({"verdicts": all_verdicts, "batches": batch_counts}, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "backend": "VLM mask verification over plan-driven SAM masks",
        "model": args.model,
        "object_plan": str(args.object_plan),
        "object_index": int(args.object_index),
        "track_id": object_plan["track_id"],
        "verified_frames": len(all_verdicts),
        **counts,
        "annotations": str(annotations_path),
        "verdicts": str(verdict_path),
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "qc_vlm_mask_verification.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--tile-width", type=int, default=420)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--remote-output-root")
    parser.add_argument("--local-output-root")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
