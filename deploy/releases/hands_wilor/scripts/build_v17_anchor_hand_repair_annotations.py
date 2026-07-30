#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_json_int(row: dict[str, Any], field: str, context: str) -> int:
    if field not in row:
        raise RuntimeError(f"{context} missing integer field {field}")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} field {field} must be a JSON integer, got {value!r}")
    return value


def require_json_str(row: dict[str, Any], field: str, context: str) -> str:
    if field not in row:
        raise RuntimeError(f"{context} missing string field {field}")
    value = row[field]
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} field {field} must be a non-empty string, got {value!r}")
    return value


def require_finite_number(row: dict[str, Any], field: str, context: str) -> float:
    if field not in row:
        raise RuntimeError(f"{context} missing numeric field {field}")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{context} field {field} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{context} field {field} must be finite, got {value!r}")
    return out


def source_hand(
    annotation_cache: dict[Path, dict[str, Any]],
    source_annotation: Path,
    frame_idx: int,
    detector_hand_idx: int,
    side: str,
) -> dict[str, Any]:
    payload = annotation_cache.setdefault(source_annotation, load_json(source_annotation))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{source_annotation} has no frames list")
    matches: list[dict[str, Any]] = []
    for frame_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        source_frame_idx = require_json_int(frame, "frame_idx", f"{source_annotation} frame row {frame_i}")
        if source_frame_idx != frame_idx:
            continue
        hands = frame.get("hands")
        if not isinstance(hands, list):
            raise RuntimeError(f"{source_annotation} frame {frame_idx} has no hands list")
        for hand_i, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            if hand.get("backend") != "HaMeR" or hand.get("side") != side:
                continue
            rtmlib = hand.get("rtmlib_measurement")
            if not isinstance(rtmlib, dict):
                raise RuntimeError(f"{source_annotation} frame {frame_idx} hand {hand_i} has no RTMLib measurement")
            hand_idx = require_json_int(
                rtmlib,
                "hand_idx",
                f"{source_annotation} frame {frame_idx} hand {hand_i} RTMLib measurement",
            )
            if hand_idx == detector_hand_idx:
                matches.append(hand)
    if len(matches) != 1:
        raise RuntimeError(
            f"{source_annotation} frame {frame_idx} detector {detector_hand_idx} side {side} matched {len(matches)} hands"
        )
    return copy.deepcopy(matches[0])


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = load_json(args.base_annotations)
    selected = load_json(args.selected_candidates)
    candidates = selected.get("selected_candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"{args.selected_candidates} has no selected_candidates list")
    base_rows = base.get("frames")
    if not isinstance(base_rows, list):
        raise RuntimeError(f"{args.base_annotations} has no frames list")
    base_frames: dict[int, dict[str, Any]] = {}
    for frame_i, frame in enumerate(base_rows):
        if not isinstance(frame, dict):
            raise RuntimeError(f"{args.base_annotations} frame row {frame_i} is not a JSON object")
        base_frames[require_json_int(frame, "frame_idx", f"{args.base_annotations} frame row {frame_i}")] = frame
    annotation_cache: dict[Path, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row_i, raw in enumerate(candidates):
        if not isinstance(raw, dict):
            raise RuntimeError(f"{args.selected_candidates} selected candidate row {row_i} is not a JSON object")
        frame_idx = require_json_int(raw, "frame_idx", f"{args.selected_candidates} selected candidate row {row_i}")
        grouped.setdefault(frame_idx, []).append(raw)

    output_frames: list[dict[str, Any]] = []
    for frame_idx in sorted(grouped):
        base_frame = base_frames.get(frame_idx)
        if base_frame is None:
            raise RuntimeError(f"base annotations missing frame {frame_idx}")
        frame = {
            "frame_idx": frame_idx,
            "time_s": base_frame.get("time_s"),
            "camera": base_frame["camera"],
            "caption": base_frame.get("caption", ""),
            "object": base_frame.get("object", {}),
            "hands": [],
        }
        for candidate in sorted(
            grouped[frame_idx],
            key=lambda row: require_json_int(
                row,
                "detector_hand_idx",
                f"{args.selected_candidates} selected candidate frame {frame_idx}",
            ),
        ):
            detector_hand_idx = require_json_int(
                candidate,
                "detector_hand_idx",
                f"{args.selected_candidates} selected candidate frame {frame_idx}",
            )
            side = require_json_str(candidate, "side", f"{args.selected_candidates} selected candidate frame {frame_idx}")
            source_annotation = Path(
                require_json_str(
                    candidate,
                    "source_annotation",
                    f"{args.selected_candidates} selected candidate frame {frame_idx}",
                )
            )
            hand = source_hand(
                annotation_cache,
                source_annotation,
                frame_idx,
                detector_hand_idx,
                side,
            )
            hand["v17_repair_state"] = "selected_hamer_anchor_repair_candidate"
            hand["v17_repair_candidate_id"] = require_json_str(
                candidate,
                "candidate_id",
                f"{args.selected_candidates} selected candidate frame {frame_idx}",
            )
            hand["v17_source_measurement_id"] = require_json_str(
                candidate,
                "measurement_id",
                f"{args.selected_candidates} selected candidate frame {frame_idx}",
            )
            hand["v17_selection_status"] = require_json_str(
                candidate,
                "selection_status",
                f"{args.selected_candidates} selected candidate frame {frame_idx}",
            )
            hand["projection_residual_to_measurement_px"] = {
                "median": require_finite_number(
                    candidate,
                    "projection_residual_px_median",
                    f"{args.selected_candidates} selected candidate frame {frame_idx}",
                ),
                "p95": require_finite_number(
                    candidate,
                    "projection_residual_px_p95",
                    f"{args.selected_candidates} selected candidate frame {frame_idx}",
                ),
            }
            hand["v17_repair_contact_validation"] = "pending"
            frame["hands"].append(hand)
        output_frames.append(frame)

    payload = {
        "status": "ok",
        "method": "build_v17_anchor_hand_repair_annotations",
        "base_annotations": str(args.base_annotations),
        "selected_candidates": str(args.selected_candidates),
        "frames": output_frames,
    }
    write_json(args.output_annotations, payload)
    report = {
        "status": "ok",
        "method": "build_v17_anchor_hand_repair_annotations",
        "output_annotations": str(args.output_annotations),
        "frame_count": len(output_frames),
        "hand_count": sum(len(frame["hands"]) for frame in output_frames),
        "frames": [int(frame["frame_idx"]) for frame in output_frames],
    }
    write_json(args.output_summary, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
