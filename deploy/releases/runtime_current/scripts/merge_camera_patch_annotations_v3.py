#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def run(args: argparse.Namespace) -> dict:
    base = load_json(args.annotations)
    patch = load_json(args.camera_patch)
    frames = base.get("frames")
    patch_frames = patch.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("base annotations must contain nonempty frames")
    if not isinstance(patch_frames, list) or not patch_frames:
        raise RuntimeError("camera patch must contain nonempty frames")
    patch_by_idx = {int(frame["frame_idx"]): frame for frame in patch_frames}
    changed = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        if idx not in patch_by_idx:
            continue
        camera = frame.setdefault("camera", {})
        old = camera.get("T_world_camera_metric")
        camera["T_world_camera_metric_before_patch"] = old
        camera["T_world_camera_metric"] = patch_by_idx[idx]["camera"]["T_world_camera_metric"]
        changed.append(idx)
    if not changed:
        raise RuntimeError("camera patch changed no frames")
    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations.write_text(json.dumps(base, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "merge_camera_patch_annotations_v3",
        "annotations": str(args.annotations),
        "camera_patch": str(args.camera_patch),
        "output_annotations": str(args.output_annotations),
        "changed_frames": changed,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--camera-patch", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
