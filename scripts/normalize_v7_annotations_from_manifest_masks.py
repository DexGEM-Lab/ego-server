#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def frame_rows(payload: dict, path: Path) -> list[dict]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} lacks nonempty frames list")
    return frames


def manifest_mask_by_frame(path: Path) -> dict[int, str]:
    rows = frame_rows(load_json(path), path)
    out = {}
    for row in rows:
        frame_idx = int(row["frame_idx"])
        raw = row.get("mask") or row.get("source_mask")
        if isinstance(raw, str) and raw:
            mask_path = Path(raw)
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            out[frame_idx] = str(mask_path)
    return out


def mask_size(path: Path) -> list[int]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    height, width = mask.shape[:2]
    return [int(width), int(height)]


def normalize_frame(frame: dict, mask_path: str) -> dict:
    path = Path(mask_path)
    if not path.exists():
        raise RuntimeError(f"manifest mask path does not exist: {path}")
    out = dict(frame)
    obj = out.get("object")
    if not isinstance(obj, dict):
        obj = {}
    else:
        obj = dict(obj)
    obj["mask_path"] = str(path)
    obj["mask_image_size"] = mask_size(path)
    obj.setdefault("source_image_size", obj["mask_image_size"])
    if not obj.get("status"):
        obj["status"] = "manifest_mask_normalized"
    out["object"] = obj
    return out


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = frame_rows(annotations, args.annotations)
    masks = manifest_mask_by_frame(args.manifest)
    normalized = []
    normalized_frames = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if args.frame_start <= frame_idx <= args.frame_end:
            mask = masks.get(frame_idx)
            if mask is None:
                raise RuntimeError(f"manifest lacks mask for frame {frame_idx}")
            normalized.append(normalize_frame(frame, mask))
            normalized_frames.append(frame_idx)
        else:
            normalized.append(frame)
    if normalized_frames != list(range(args.frame_start, args.frame_end + 1)):
        raise RuntimeError(
            f"annotations did not contain every requested frame: "
            f"{normalized_frames} != {list(range(args.frame_start, args.frame_end + 1))}"
        )
    output = dict(annotations)
    output["frames"] = normalized
    output.setdefault("v7_annotation_normalization", {})
    output["v7_annotation_normalization"] = {
        "method": "normalize_v7_annotations_from_manifest_masks",
        "source_annotations": str(args.annotations),
        "manifest": str(args.manifest),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "normalized_frames": normalized_frames,
        "claim": "object masks are copied from the model-produced manifest rows into frame.object for downstream contact diagnostics",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "normalize_v7_annotations_from_manifest_masks",
        "output_json": str(args.output_json),
        "normalized_frames": normalized_frames,
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
