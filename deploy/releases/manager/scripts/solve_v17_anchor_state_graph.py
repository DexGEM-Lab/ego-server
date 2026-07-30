#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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


def finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{field} must be finite, got {value!r}")
    return out


def json_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be a JSON integer, got {value!r}")
    return value


def source_path(row: dict[str, Any]) -> str:
    source = row.get("source_summary") or row.get("source_annotation")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"measurement {row.get('measurement_id')} has no source path")
    return source


def compact_hamer(row: dict[str, Any]) -> dict[str, Any]:
    median = finite_float(row.get("projection_residual_px_median"), "projection_residual_px_median")
    p95 = finite_float(row.get("projection_residual_px_p95"), "projection_residual_px_p95")
    confidence = finite_float(row.get("confidence"), "confidence")
    return {
        "measurement_id": row.get("measurement_id"),
        "frame_idx": json_int(row.get("frame_idx"), "frame_idx"),
        "entity_id": row.get("entity_id"),
        "side": row.get("side"),
        "detector_hand_idx": json_int(row.get("detector_hand_idx"), "detector_hand_idx"),
        "confidence": confidence,
        "bbox_xyxy": row.get("bbox_xyxy"),
        "projection_residual_px_median": median,
        "projection_residual_px_p95": p95,
        "median_depth_m": row.get("median_depth_m"),
        "hand_bone_scale_m": row.get("hand_bone_scale_m"),
        "source_summary": row.get("source_summary"),
        "source_annotation": row.get("source_annotation"),
    }


def load_hamer_candidates(path: Path, source_substring: str, frame_indices: set[int]) -> dict[int, dict[tuple[int, str], list[dict[str, Any]]]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain a JSON list")
    groups: dict[int, dict[tuple[int, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row_i, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise RuntimeError(f"{path} row {row_i} is not a JSON object")
        if raw.get("measurement_available") is not True:
            continue
        idx = json_int(raw.get("frame_idx"), f"{path} row {row_i} frame_idx")
        if idx not in frame_indices:
            continue
        source = source_path(raw)
        if source_substring not in source:
            continue
        row = compact_hamer(raw)
        side = row.get("side")
        if side not in ("left", "right"):
            raise RuntimeError(f"{path} row {row_i} has invalid side {side!r}")
        groups[idx][(int(row["detector_hand_idx"]), source)].append(row)
    return groups


def side_evidence(paths: list[Path], frame_indices: set[int]) -> dict[int, dict[str, dict[str, Any]]]:
    evidence: dict[int, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "sources": []}))
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a JSON list")
        for row_i, raw in enumerate(rows):
            if not isinstance(raw, dict):
                raise RuntimeError(f"{path} row {row_i} is not a JSON object")
            idx = raw.get("frame_idx")
            if isinstance(idx, bool) or not isinstance(idx, int) or idx not in frame_indices:
                continue
            if raw.get("measurement_available") is False or raw.get("failure_reason") is not None:
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
            source_model = raw.get("source_model") or path.name
            bucket = evidence[idx][str(side)]
            bucket["count"] += 1
            bucket["sources"].append(str(source_model))
    return evidence


def candidate_cost(candidate: dict[str, Any], side_counts: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    median = finite_float(candidate["projection_residual_px_median"], "projection_residual_px_median")
    p95 = finite_float(candidate["projection_residual_px_p95"], "projection_residual_px_p95")
    confidence = finite_float(candidate["confidence"], "confidence")
    side = str(candidate["side"])
    opposing_side = "right" if side == "left" else "left"
    own_votes = int(side_counts.get(side, {}).get("count", 0))
    opposing_votes = int(side_counts.get(opposing_side, {}).get("count", 0))
    terms = {
        "reprojection_median_px": float(args.w_median * median),
        "reprojection_tail_px": float(args.w_p95 * max(0.0, p95 - median)),
        "low_confidence": float(args.w_confidence * max(0.0, 1.0 - confidence)),
        "independent_side_disagreement": float(args.w_side_vote * max(0, opposing_votes - own_votes)),
    }
    return {
        "candidate": candidate,
        "side_vote_counts": {"own": own_votes, "opposing": opposing_votes},
        "factor_terms": terms,
        "total_cost": float(sum(terms.values())),
    }


def enumerate_assignments(group_scores: list[list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    assignments = []
    for combo in itertools.product(*group_scores):
        sides = [str(row["candidate"]["side"]) for row in combo]
        duplicate_penalty = 0.0
        if len(sides) > 1 and len(set(sides)) < len(sides):
            duplicate_penalty = float(args.w_duplicate_side)
        total = float(sum(float(row["total_cost"]) for row in combo) + duplicate_penalty)
        assignments.append({"choices": list(combo), "duplicate_side_penalty": duplicate_penalty, "total_cost": total})
    return sorted(assignments, key=lambda row: float(row["total_cost"]))


def solve(args: argparse.Namespace) -> dict[str, Any]:
    frame_indices = {int(frame) for frame in args.frame_indices}
    groups_by_frame = load_hamer_candidates(args.hamer_measurements, args.source_substring, frame_indices)
    side_by_frame = side_evidence(args.side_prior_measurements, frame_indices)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    frame_reports: list[dict[str, Any]] = []
    for frame_idx in sorted(frame_indices):
        frame_groups = groups_by_frame.get(frame_idx)
        if not frame_groups:
            raise RuntimeError(f"no HaMeR candidates for frame {frame_idx}")
        scored_groups = []
        for group_key, group in sorted(frame_groups.items()):
            scored = [candidate_cost(row, side_by_frame.get(frame_idx, {}), args) for row in group]
            scored_groups.append(sorted(scored, key=lambda row: float(row["total_cost"])))
        ranked = enumerate_assignments(scored_groups, args)
        winner = ranked[0]
        winner_ids = {id(choice["candidate"]) for choice in winner["choices"]}
        frame_reports.append(
            {
                "frame_idx": frame_idx,
                "side_evidence": side_by_frame.get(frame_idx, {}),
                "assignment_count": len(ranked),
                "selected_total_cost": winner["total_cost"],
                "selected_duplicate_side_penalty": winner["duplicate_side_penalty"],
                "selected_measurement_ids": [choice["candidate"].get("measurement_id") for choice in winner["choices"]],
                "second_best_total_cost": ranked[1]["total_cost"] if len(ranked) > 1 else None,
            }
        )
        for group_i, group in enumerate(scored_groups):
            group_winner = [choice for choice in winner["choices"] if choice in group]
            if len(group_winner) != 1:
                raise RuntimeError(f"frame {frame_idx} group {group_i} selected {len(group_winner)} winners")
            choice = group_winner[0]
            row = dict(choice["candidate"])
            row.update(
                {
                    "candidate_id": f"v17_anchor_graph:{frame_idx}:{row['detector_hand_idx']}:{len(selected)}",
                    "selection_status": "selected_anchor_factor_graph",
                    "selection_factors": choice["factor_terms"],
                    "selection_total_cost": choice["total_cost"],
                    "selection_side_vote_counts": choice["side_vote_counts"],
                    "selection_group": {
                        "frame_idx": frame_idx,
                        "detector_hand_idx": row["detector_hand_idx"],
                        "source": source_path(row),
                        "candidate_count": len(group),
                    },
                }
            )
            selected.append(row)
            for loser in group:
                if id(loser["candidate"]) in winner_ids:
                    continue
                raw = dict(loser["candidate"])
                raw.update(
                    {
                        "selection_status": "rejected_anchor_factor_graph",
                        "selection_factors": loser["factor_terms"],
                        "selection_total_cost": loser["total_cost"],
                        "selected_measurement_id": row["measurement_id"],
                        "selection_group": row["selection_group"],
                    }
                )
                rejected.append(raw)
    selected_by_frame: dict[int, int] = defaultdict(int)
    for row in selected:
        selected_by_frame[int(row["frame_idx"])] += 1
    report = {
        "status": "ok",
        "method": "solve_v17_anchor_state_graph",
        "hamer_measurements": str(args.hamer_measurements),
        "side_prior_measurements": [str(path) for path in args.side_prior_measurements],
        "source_substring": args.source_substring,
        "frame_indices": sorted(frame_indices),
        "objective_weights": {
            "w_median": float(args.w_median),
            "w_p95": float(args.w_p95),
            "w_confidence": float(args.w_confidence),
            "w_side_vote": float(args.w_side_vote),
            "w_duplicate_side": float(args.w_duplicate_side),
        },
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected_count_by_frame": {str(idx): count for idx, count in sorted(selected_by_frame.items())},
        "frame_reports": frame_reports,
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
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--source-substring", default="hamer_vlm_box_summary")
    parser.add_argument("--side-prior-measurements", type=Path, nargs="*", default=[])
    parser.add_argument("--w-median", type=float, default=1.0)
    parser.add_argument("--w-p95", type=float, default=0.15)
    parser.add_argument("--w-confidence", type=float, default=10.0)
    parser.add_argument("--w-side-vote", type=float, default=12.0)
    parser.add_argument("--w-duplicate-side", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
