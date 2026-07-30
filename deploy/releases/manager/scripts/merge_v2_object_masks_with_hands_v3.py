#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    hand_annotations = load_json(args.hand_annotations)
    object_annotations = load_json(args.object_annotations)
    hand_frames = {int(frame["frame_idx"]): frame for frame in hand_annotations["frames"]}
    object_frames = {int(frame["frame_idx"]): frame for frame in object_annotations["frames"]}
    merged = []
    used_objects = 0
    for frame_idx in sorted(hand_frames):
        frame = dict(hand_frames[frame_idx])
        obj_frame = object_frames.get(frame_idx)
        if obj_frame is None:
            frame["object"] = {"label": "not_visible", "status": "not_visible"}
        else:
            frame["object"] = obj_frame.get("object", {"label": "not_visible", "status": "not_visible"})
            if frame["object"].get("mask_path"):
                used_objects += 1
        merged.append(frame)
    if not merged:
        raise RuntimeError("no merged frames")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"frames": merged}, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "hand_annotations": str(args.hand_annotations),
        "object_annotations": str(args.object_annotations),
        "output_json": str(args.output_json),
        "frames": len(merged),
        "object_mask_frames": int(used_objects),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-annotations", type=Path, required=True)
    parser.add_argument("--object-annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
