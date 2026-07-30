#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def as_finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{field} must be finite, got {value!r}")
    return out


def as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be an integer, got {value!r}") from exc


def row_source(row: dict[str, Any]) -> str:
    source = row.get("source_summary") or row.get("source_annotation")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"HaMeR row has no source path: {row.get('measurement_id')}")
    return source


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        as_finite_float(row.get("projection_residual_px_median"), "projection_residual_px_median"),
        as_finite_float(row.get("projection_residual_px_p95"), "projection_residual_px_p95"),
        -as_finite_float(row.get("confidence"), "confidence"),
    )


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "measurement_id": row.get("measurement_id"),
        "frame_idx": row.get("frame_idx"),
        "entity_id": row.get("entity_id"),
        "side": row.get("side"),
        "detector_hand_idx": row.get("detector_hand_idx"),
        "confidence": row.get("confidence"),
        "bbox_xyxy": row.get("bbox_xyxy"),
        "projection_residual_px_median": row.get("projection_residual_px_median"),
        "projection_residual_px_p95": row.get("projection_residual_px_p95"),
        "median_depth_m": row.get("median_depth_m"),
        "hand_bone_scale_m": row.get("hand_bone_scale_m"),
        "source_summary": row.get("source_summary"),
        "source_annotation": row.get("source_annotation"),
    }


def group_choice(
    group: list[dict[str, Any]],
    selected_side: str | None,
) -> dict[str, Any]:
    candidates = [row for row in group if selected_side is None or row.get("side") == selected_side]
    if not candidates:
        raise RuntimeError(f"group has no hypothesis for side {selected_side!r}")
    return sorted(candidates, key=sort_key)[0]


def side_priors(paths: list[Path]) -> dict[int, str]:
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a JSON list")
        for row_i, raw in enumerate(rows):
            if not isinstance(raw, dict):
                raise RuntimeError(f"{path} row {row_i} is not a JSON object")
            if raw.get("measurement_available") is False:
                continue
            if raw.get("failure_reason") is not None:
                continue
            side = raw.get("side")
            if side not in ("left", "right"):
                entity_id = raw.get("entity_id")
                if entity_id == "hand:left":
                    side = "left"
                elif entity_id == "hand:right":
                    side = "right"
            if side not in ("left", "right"):
                continue
            frame_idx = as_int(raw.get("frame_idx"), f"{path} row {row_i} frame_idx")
            counts[frame_idx][str(side)] += 1
    out: dict[int, str] = {}
    for frame_idx, by_side in counts.items():
        left = by_side.get("left", 0)
        right = by_side.get("right", 0)
        if left == right:
            continue
        out[frame_idx] = "left" if left > right else "right"
    return out


def frame_choices(
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    selected_side: str | None,
) -> tuple[list[dict[str, Any]], str]:
    if selected_side is not None and len(groups) == 1:
        group = next(iter(groups.values()))
        return [group_choice(group, selected_side)], "selected_independent_side_prior"
    if len(groups) == 2:
        keys = sorted(groups)
        available_sides = [{str(row.get("side")) for row in groups[key]} for key in keys]
        if all({"left", "right"}.issubset(sides) for sides in available_sides):
            assignments = [("left", "right"), ("right", "left")]
            ranked: list[tuple[tuple[float, float, float], list[dict[str, Any]]]] = []
            for assignment in assignments:
                chosen = [group_choice(groups[key], side) for key, side in zip(keys, assignment)]
                score = (
                    sum(sort_key(row)[0] for row in chosen),
                    sum(sort_key(row)[1] for row in chosen),
                    sum(sort_key(row)[2] for row in chosen),
                )
                ranked.append((score, chosen))
            return sorted(ranked, key=lambda item: item[0])[0][1], "selected_frame_unique_side_assignment"
    return [group_choice(group, None) for _, group in sorted(groups.items())], "selected_min_projection_residual"


def select_candidates(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_json(args.hamer_measurements)
    if not isinstance(rows, list):
        raise RuntimeError(f"{args.hamer_measurements} must contain a JSON list")
    wanted_frames = {int(frame) for frame in args.frame_indices} if args.frame_indices else None
    priors = side_priors(args.side_prior_measurements)
    groups_by_frame: dict[int, dict[tuple[int, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for raw in rows:
        if not isinstance(raw, dict):
            raise RuntimeError("HaMeR measurements must be JSON objects")
        if not bool(raw.get("measurement_available")):
            continue
        frame_idx = as_int(raw.get("frame_idx"), "frame_idx")
        if wanted_frames is not None and frame_idx not in wanted_frames:
            continue
        source = row_source(raw)
        if args.source_substring not in source:
            continue
        detector_hand_idx = as_int(raw.get("detector_hand_idx"), "detector_hand_idx")
        sort_key(raw)
        groups_by_frame[frame_idx][(detector_hand_idx, source)].append(raw)

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for frame_idx, frame_groups in sorted(groups_by_frame.items()):
        winners, selection_status = frame_choices(frame_groups, priors.get(frame_idx))
        winner_ids = {id(row) for row in winners}
        for group_key, group in sorted(frame_groups.items()):
            detector_hand_idx, source = group_key
            group_winners = [row for row in winners if id(row) in {id(candidate) for candidate in group}]
            if len(group_winners) != 1:
                raise RuntimeError(f"frame {frame_idx} group {group_key} selected {len(group_winners)} winners")
            winner = dict(compact_row(group_winners[0]))
            winner.update(
                {
                    "candidate_id": f"v17_hamer_repair:{frame_idx}:{detector_hand_idx}:{len(selected)}",
                    "selection_status": selection_status,
                    "selection_group": {
                        "frame_idx": frame_idx,
                        "detector_hand_idx": detector_hand_idx,
                        "source": source,
                        "candidate_count": len(group),
                    },
                }
            )
            selected.append(winner)
            for row in sorted(group, key=sort_key):
                if id(row) in winner_ids:
                    continue
                loser = dict(compact_row(row))
                loser.update(
                    {
                        "selection_status": "rejected_by_frame_side_assignment_or_residual",
                        "selected_measurement_id": winner["measurement_id"],
                        "selection_group": winner["selection_group"],
                    }
                )
                rejected.append(loser)

    by_frame: dict[int, int] = defaultdict(int)
    for row in selected:
        by_frame[as_int(row["frame_idx"], "frame_idx")] += 1

    report = {
        "status": "ok",
        "method": "select_v17_hamer_hand_repair_candidates",
        "hamer_measurements": str(args.hamer_measurements),
        "source_substring": args.source_substring,
        "side_prior_measurements": [str(path) for path in args.side_prior_measurements],
        "side_priors": {str(idx): side for idx, side in sorted(priors.items())},
        "frame_indices": sorted(wanted_frames) if wanted_frames is not None else None,
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected_count_by_frame": {str(idx): count for idx, count in sorted(by_frame.items())},
        "selected_candidates": selected,
        "rejected_candidates": rejected,
    }
    write_json(args.output_json, report)
    print(json.dumps({k: report[k] for k in ("status", "method", "selected_count", "rejected_count", "selected_count_by_frame")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hamer-measurements", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="*")
    parser.add_argument("--source-substring", default="hamer_vlm_box_summary")
    parser.add_argument("--side-prior-measurements", type=Path, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    select_candidates(parse_args())


if __name__ == "__main__":
    main()
