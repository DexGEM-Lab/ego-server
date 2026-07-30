#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def frame_map(payload: dict) -> dict[int, dict]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("hand stream has no frames")
    out = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate frame {frame_idx}")
        out[frame_idx] = frame
    return out


def source_label(path: Path, hand: dict) -> str:
    backend = str(hand.get("backend", "unknown"))
    track_source = str(hand.get("track_source", path.stem))
    return f"{backend}:{track_source}"


def run(args: argparse.Namespace) -> dict:
    base = load_json(args.base_annotations)
    base_frames = frame_map(base)
    streams = [(path, frame_map(load_json(path))) for path in args.hand_streams]
    output_frames = []
    rows = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in base_frames:
            raise RuntimeError(f"base annotations missing frame {frame_idx}")
        out_frame = copy.deepcopy(base_frames[frame_idx])
        merged = []
        for path, stream in streams:
            frame = stream.get(frame_idx)
            if frame is None:
                raise RuntimeError(f"{path} missing frame {frame_idx}")
            for hand in frame.get("hands", []):
                hand_out = copy.deepcopy(hand)
                hand_out["v7_candidate_stream"] = {
                    "source_annotations": str(path),
                    "source_label": source_label(path, hand_out),
                }
                merged.append(hand_out)
        out_frame["hands"] = merged
        output_frames.append(out_frame)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "candidate_hands": int(len(merged)),
                "measured_candidates_before_refit": int(sum(1 for hand in merged if bool(hand.get("measurement_available")))),
            }
        )
    report = {
        "status": "ok",
        "method": "merge_hand_candidate_streams_v7",
        "base_annotations": str(args.base_annotations),
        "hand_streams": [str(path) for path in args.hand_streams],
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(output_frames)),
        "candidate_hands": int(sum(row["candidate_hands"] for row in rows)),
        "rows": rows,
    }
    save_json(args.output_annotations, {"frames": output_frames})
    save_json(args.output_qc, report)
    print(json.dumps({k: report[k] for k in ("status", "frames", "candidate_hands")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--hand-streams", type=Path, nargs="+", required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
