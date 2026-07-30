from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


VIDEO_KEYS = ("overlay", "reconstruction_3d", "side_by_side")


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def video_shape(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, fps, width, height


def expected_frames(qc: dict, annotation_frames: int) -> int:
    value = qc.get("processed_frames")
    if isinstance(value, int):
        return value
    if annotation_frames:
        return annotation_frames
    raise RuntimeError("QC file does not expose a full-timeline frame count")


def check_qc(path: Path) -> dict:
    qc = load(path)
    outputs = qc.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError(f"{path}: missing outputs object")

    annotation_value = outputs.get("annotations", qc.get("output_annotations"))
    if annotation_value is None:
        annotations = None
        actual_annotation_frames = 0
    else:
        annotations = Path(annotation_value)
        if not annotations.exists():
            raise FileNotFoundError(annotations)
        actual_annotation_frames = len(load(annotations).get("frames", []))
    timeline_frames = expected_frames(qc, actual_annotation_frames)
    if annotations is not None and actual_annotation_frames != timeline_frames:
        raise RuntimeError(
            f"{annotations}: annotation frame count {actual_annotation_frames} != {timeline_frames}"
        )

    render = qc.get("render", {})
    expected_render = (
        int(render["width"]),
        int(render["height"]),
        float(render["fps"]),
    )
    videos = {}
    for key in VIDEO_KEYS:
        video = Path(outputs[key])
        if not video.exists():
            raise FileNotFoundError(video)
        frames, fps, width, height = video_shape(video)
        expected_width = expected_render[0] * (2 if key == "side_by_side" else 1)
        expected_height = expected_render[1]
        if frames != timeline_frames:
            raise RuntimeError(f"{video}: frame count {frames} != {timeline_frames}")
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(f"{video}: shape {(width, height)} != {(expected_width, expected_height)}")
        if abs(fps - expected_render[2]) > 0.02:
            raise RuntimeError(f"{video}: fps {fps} != {expected_render[2]}")
        videos[key] = {"frames": frames, "fps": fps, "width": width, "height": height}

    return {
        "qc": str(path),
        "annotations": str(annotations) if annotations is not None else None,
        "frames": timeline_frames,
        "videos": videos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("qc", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    reports = [check_qc(path) for path in parse_args().qc]
    print(json.dumps({"status": "ok", "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
