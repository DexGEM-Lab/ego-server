#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"non-finite numeric value: {value!r}")
    return out


def finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{name} must be a list of length {length}")
    return [finite_float(v) for v in value]


def finite_matrix4(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError(f"{name} must be a 4x4 list")
    return [finite_vector(row, 4, name) for row in value]


def finite_points2d(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 21:
        raise RuntimeError(f"{name} must contain 21 2D keypoints")
    return [finite_vector(row, 2, name) for row in value]


def hand_intrinsics(hand: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = hand.get("source_intrinsics")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    return (finite_float(raw[0]), finite_float(raw[1]), finite_float(raw[2]), finite_float(raw[3]))


def frame_hands(frame: dict[str, Any]) -> list[Any]:
    hands = frame.get("hands", [])
    if hands is None:
        return []
    if not isinstance(hands, list):
        raise RuntimeError(f"frame {frame.get('frame_idx')} hands must be a list")
    return hands


def unique_source_intrinsics(frames: list[dict[str, Any]]) -> list[float]:
    values = {
        intr
        for frame in frames
        for hand in frame_hands(frame)
        if isinstance(hand, dict)
        for intr in [hand_intrinsics(hand)]
        if intr is not None
    }
    if not values:
        raise RuntimeError("no hand source_intrinsics found in input annotations")
    if len(values) != 1:
        preview = sorted(values)[:5]
        raise RuntimeError(f"multiple source_intrinsics values found: {preview}")
    return [float(v) for v in next(iter(values))]


def compact_hand(hand: dict[str, Any], clip_intrinsics: list[float]) -> dict[str, Any] | None:
    if not bool(hand.get("measurement_available", False)):
        return None
    side = str(hand.get("side", "")).lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"measured hand has invalid side: {hand.get('side')!r}")
    joints2d_raw = finite_points2d(hand.get("joints2d_raw"), f"{side} joints2d_raw")
    intrinsics = hand_intrinsics(hand)
    if intrinsics is None:
        raise RuntimeError(f"measured {side} hand has no source_intrinsics")
    bbox = hand.get("bbox_xyxy")
    compact: dict[str, Any] = {
        "side": side,
        "measurement_available": True,
        "detector_score": finite_float(hand.get("detector_score", 0.0)),
        "joints2d_raw": joints2d_raw,
        "source_intrinsics": [float(v) for v in intrinsics],
    }
    if bbox is not None:
        compact["bbox_xyxy"] = finite_vector(bbox, 4, f"{side} bbox_xyxy")
    backend = hand.get("backend")
    if backend is not None:
        compact["source_backend"] = str(backend)
    filter_status = hand.get("filter_status")
    if filter_status is not None:
        compact["source_filter_status"] = str(filter_status)
    return compact


def compact_frame(frame: dict[str, Any], clip_intrinsics: list[float]) -> dict[str, Any]:
    idx = int(frame["frame_idx"])
    camera = frame.get("camera")
    if not isinstance(camera, dict):
        raise RuntimeError(f"frame {idx} has no camera record")
    compact_camera: dict[str, Any] = {
        "T_world_camera_metric": finite_matrix4(
            camera.get("T_world_camera_metric"), f"frame {idx} T_world_camera_metric"
        ),
        "vggt_source_intrinsics_fx_fy_cx_cy": clip_intrinsics,
    }
    if camera.get("position_world_m") is not None:
        compact_camera["position_world_m"] = finite_vector(
            camera.get("position_world_m"), 3, f"frame {idx} position_world_m"
        )
    hands = [
        compact
        for hand in frame_hands(frame)
        if isinstance(hand, dict)
        for compact in [compact_hand(hand, clip_intrinsics)]
        if compact is not None
    ]
    out: dict[str, Any] = {
        "frame_idx": idx,
        "time_s": finite_float(frame.get("time_s", idx / 30.0)),
        "camera": compact_camera,
        "hands": hands,
    }
    if frame.get("caption") is not None:
        out["caption"] = str(frame.get("caption"))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.annotations)
    frames = source.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{args.annotations} has no non-empty frames list")
    frame_ids = [int(frame["frame_idx"]) for frame in frames]
    if frame_ids != list(range(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise RuntimeError("input frames must be contiguous by frame_idx")
    clip_intrinsics = unique_source_intrinsics(frames)
    compact_frames = [compact_frame(frame, clip_intrinsics) for frame in frames]
    measured_hand_rows = sum(len(frame["hands"]) for frame in compact_frames)
    frames_with_hands = sum(1 for frame in compact_frames if frame["hands"])
    output = {
        "frames": compact_frames,
        "metadata": {
            "method": "build_v17_hawor_adapter_input",
            "source_annotations": str(args.annotations),
            "purpose": "minimal full-video annotation input for adapt_hawor_camera_local_v3",
            "source_intrinsics_fx_fy_cx_cy": clip_intrinsics,
        },
    }
    report = {
        "status": "ok",
        "method": "build_v17_hawor_adapter_input",
        "source_annotations": str(args.annotations),
        "output_json": str(args.output_json),
        "frame_count": len(compact_frames),
        "frame_start": frame_ids[0],
        "frame_end": frame_ids[-1],
        "source_intrinsics_fx_fy_cx_cy": clip_intrinsics,
        "measured_hand_rows": measured_hand_rows,
        "frames_with_measured_hands": frames_with_hands,
        "dropped_unavailable_or_nonhand_rows": sum(len(frame_hands(frame)) for frame in frames) - measured_hand_rows,
        "contract": {
            "preserved": [
                "frame_idx",
                "time_s",
                "camera.T_world_camera_metric",
                "camera.vggt_source_intrinsics_fx_fy_cx_cy",
                "measured hand side",
                "measured hand detector_score",
                "measured hand bbox_xyxy",
                "measured hand joints2d_raw",
                "measured hand source_intrinsics",
            ],
            "omitted": [
                "MANO vertices",
                "MANO joints3d",
                "object mesh state",
                "render-only fields",
            ],
        },
    }
    write_json(args.output_json, output)
    write_json(args.output_qc, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
