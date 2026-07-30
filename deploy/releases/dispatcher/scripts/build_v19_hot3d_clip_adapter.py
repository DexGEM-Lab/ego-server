#!/usr/bin/env python3
"""Adapt a HOT3D-Clips WebDataset tar into a V19 benchmark input root.

The adapter reads one public HOT3D-Clips tar, extracts a chosen egocentric image
stream into a V19-style raw-frame manifest and MP4, and writes HOT3D metadata to
``evaluation/hot3d_gt`` for later scoring.  Ground truth is never written into
pipeline state or used as perception input.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_json(raw: bytes, name: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to decode JSON member {name}") from exc


def decode_image(raw: bytes, name: str) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode image member {name}")
    return image


def member_key_field(name: str) -> tuple[str, str] | None:
    base = Path(name).name
    if "." not in base:
        return None
    key, field = base.split(".", 1)
    if not key.isdigit():
        return None
    return key, field


def read_tar(tar_path: Path) -> tuple[dict[str, dict[str, bytes]], dict[str, Any] | None]:
    groups: dict[str, dict[str, bytes]] = {}
    hand_shapes: dict[str, Any] | None = None
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if Path(member.name).name == "__hand_shapes.json__":
                f = tar.extractfile(member)
                if f is not None:
                    payload = safe_json(f.read(), member.name)
                    if isinstance(payload, dict):
                        hand_shapes = payload
                continue
            parsed = member_key_field(member.name)
            if parsed is None:
                continue
            key, field = parsed
            f = tar.extractfile(member)
            if f is None:
                continue
            groups.setdefault(key, {})[field] = f.read()
    if not groups:
        raise RuntimeError(f"no numeric WebDataset samples found in {tar_path}")
    return groups, hand_shapes


def summarize_json_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.get("json", {}).items():
            counts[key] = counts.get(key, 0) + (1 if value is not None else 0)
    return {"json_field_counts": counts}


def build(args: argparse.Namespace) -> dict[str, Any]:
    groups, hand_shapes = read_tar(args.tar)
    keys = sorted(groups, key=lambda x: int(x))
    image_field = args.image_field
    available_image_fields = sorted({field for g in groups.values() for field in g if field.lower().endswith((".jpg", ".jpeg", ".png"))})
    if image_field not in available_image_fields:
        raise RuntimeError(f"image field {image_field!r} not found; available image fields: {available_image_fields}")

    raw_dir = args.output_root / "input" / "hot3d_raw_frame_manifest" / "rgb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.output_root / "evaluation" / "hot3d_gt"
    eval_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    video_path = input_dir / f"{args.clip_id}_{image_field.replace('.', '_')}.mp4"
    rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    writer: cv2.VideoWriter | None = None
    width = height = None

    try:
        for out_idx, key in enumerate(keys):
            sample = groups[key]
            image = decode_image(sample[image_field], f"{key}.{image_field}")
            h, w = image.shape[:2]
            if writer is None:
                width, height = int(w), int(h)
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open writer {video_path}")
            if (w, h) != (width, height):
                raise RuntimeError(f"image size changed at key {key}: {(w, h)} vs {(width, height)}")
            frame_path = raw_dir / f"{out_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f"failed to write {frame_path}")
            writer.write(image)
            json_payloads: dict[str, Any] = {}
            for field, raw in sample.items():
                if field.lower().endswith(".json"):
                    json_payloads[field] = safe_json(raw, f"{key}.{field}")
            info = json_payloads.get("info.json") if isinstance(json_payloads.get("info.json"), dict) else {}
            timestamp_ns = info.get("ref_timestamp_ns")
            rows.append(
                {
                    "frame_idx": int(out_idx),
                    "source_key": str(key),
                    "time_s": float(out_idx / float(args.fps)),
                    "rgb": str(frame_path.relative_to(args.output_root)),
                    "raw_frame_path": str(frame_path),
                    "source_width": int(width),
                    "source_height": int(height),
                    "manifest_width": int(width),
                    "manifest_height": int(height),
                    "hot3d_ref_timestamp_ns": int(timestamp_ns) if timestamp_ns is not None else None,
                    "hot3d_image_field": image_field,
                }
            )
            gt_rows.append({"frame_idx": int(out_idx), "source_key": str(key), "json": json_payloads})
    finally:
        if writer is not None:
            writer.release()

    if width is None or height is None:
        raise RuntimeError("no images written")

    manifest = {
        "schema": "v19_hot3d_clip_raw_frame_manifest_v1",
        "source_dataset": "bop-benchmark/hot3d HOT3D-Clips WebDataset",
        "source_tar": str(args.tar),
        "clip_id": args.clip_id,
        "split": args.split,
        "image_field": image_field,
        "fps": float(args.fps),
        "frame_count": len(rows),
        "width": int(width),
        "height": int(height),
        "video": str(video_path),
        "frames": rows,
        "causal_boundary": "HOT3D ground truth is evaluation-only and must not be consumed by V19 perception/state construction.",
    }
    gt = {
        "schema": "v19_hot3d_clip_ground_truth_sidecar_v1",
        "source_dataset": "bop-benchmark/hot3d HOT3D-Clips WebDataset",
        "source_tar": str(args.tar),
        "clip_id": args.clip_id,
        "split": args.split,
        "image_field": image_field,
        "hand_shapes": hand_shapes,
        "frames": gt_rows,
        **summarize_json_fields(gt_rows),
    }
    write_json(args.output_root / "input" / "raw_frame_manifest" / "manifest.json", manifest)
    write_json(eval_dir / "hot3d_clip_gt_sidecar.json", gt)
    if hand_shapes is not None:
        write_json(eval_dir / "hot3d_hand_shapes.json", hand_shapes)
    report = {
        "status": "ok",
        "method": "build_v19_hot3d_clip_adapter",
        "claim_scope": "benchmark input/GT adapter only; no V19 predictions or scores produced",
        "output_root": str(args.output_root),
        "input_video": str(video_path),
        "raw_frame_manifest": str(args.output_root / "input" / "raw_frame_manifest" / "manifest.json"),
        "ground_truth_sidecar": str(eval_dir / "hot3d_clip_gt_sidecar.json"),
        "frame_count": len(rows),
        "width": int(width),
        "height": int(height),
        "available_image_fields": available_image_fields,
        "json_field_counts": gt["json_field_counts"],
        "hand_shapes_present": hand_shapes is not None,
        "causal_boundary": manifest["causal_boundary"],
    }
    write_json(args.output_root / "evaluation" / "hot3d_adapter_report.json", report)
    print(json.dumps(report, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-id", default="clip-001849")
    parser.add_argument("--split", default="train_aria")
    parser.add_argument("--image-field", default="image_214-1.jpg")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
