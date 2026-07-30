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


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rows_by_frame(paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a JSON list")
        for row_i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"{path} row {row_i} is not a JSON object")
            idx = row.get("frame_idx")
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise RuntimeError(f"{path} row {row_i} has invalid frame_idx {idx!r}")
            row = dict(row)
            row["source_contact_measurements"] = str(path)
            out[idx].append(row)
    return out


def anchor_specs(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    anchors = payload.get("anchors")
    if not isinstance(anchors, list):
        raise RuntimeError(f"{path} has no anchors list")
    out: dict[int, dict[str, Any]] = {}
    for row_i, row in enumerate(anchors):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} anchor row {row_i} is not a JSON object")
        idx = row.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} anchor row {row_i} has invalid frame_idx {idx!r}")
        out[idx] = row
    return out

def temporal_validation_by_frame(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a JSON list")
        for row_i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"{path} row {row_i} is not a JSON object")
            idx = row.get("frame_idx")
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise RuntimeError(f"{path} row {row_i} has invalid frame_idx {idx!r}")
            if idx in out:
                raise RuntimeError(f"duplicate temporal validation for frame {idx}")
            row = dict(row)
            row["source_temporal_validation"] = str(path)
            out[idx] = row
    return out


def contact_cost(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    state = str(row.get("contact_state_measurement"))
    image = row.get("image_mask_distance_px")
    metric = row.get("hand_object_mesh_distance_m")
    image_min = finite(image.get("min") if isinstance(image, dict) else None)
    metric_min = finite(metric.get("min") if isinstance(metric, dict) else None)
    hand_valid = bool(row.get("hand_measurement_valid_for_contact"))
    terms = {
        "image_gap": float(args.w_image_gap * max(0.0, (image_min if image_min is not None else 1e6) - float(args.image_near_px))),
        "metric_gap": float(args.w_metric_gap * max(0.0, (metric_min if metric_min is not None else 1e6) - float(args.metric_near_m))),
        "invalid_hand": 0.0 if hand_valid else float(args.w_invalid_hand),
        "state_missing_modality": 0.0 if state == "candidate_contact_image_and_metric" else float(args.w_missing_modality),
    }
    return {"row": row, "factor_terms": terms, "total_cost": float(sum(terms.values()))}


def needs_temporal_validation(row: dict[str, Any], args: argparse.Namespace) -> bool:
    source = str(row.get("source_contact_measurements") or "")
    return any(term in source for term in args.temporal_validation_required_source_substring)


def solve_frame(
    frame_idx: int,
    anchor: dict[str, Any],
    rows: list[dict[str, Any]],
    temporal_validation: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = str(anchor.get("expected_contact"))
    scored = sorted([contact_cost(row, args) for row in rows], key=lambda item: float(item["total_cost"]))
    validation = temporal_validation.get(frame_idx)
    temporal_status = None
    if validation is not None:
        raw = validation.get("validation")
        temporal_status = raw.get("status") if isinstance(raw, dict) else None
    if expected in ("contact", "contact_or_near_contact"):
        if not scored:
            status = "unresolved_missing_contact_measurement"
            selected = None
        else:
            selected = None
            for candidate in scored:
                candidate_state = str(candidate["row"].get("contact_state_measurement"))
                temporal_required = needs_temporal_validation(candidate["row"], args)
                if temporal_required and temporal_status != "accepted_temporal_mask_support":
                    continue
                if (
                    candidate_state == "candidate_contact_image_and_metric"
                    and float(candidate["total_cost"]) <= float(args.max_accept_cost)
                ):
                    selected = candidate
                    break
            if selected is not None:
                status = "accepted_contact"
            else:
                selected = scored[0]
                selected_state = str(selected["row"].get("contact_state_measurement"))
                temporal_required = needs_temporal_validation(selected["row"], args)
                if temporal_required and temporal_status is None:
                    status = "unresolved_missing_temporal_object_validation"
                elif temporal_required and temporal_status != "accepted_temporal_mask_support":
                    status = "unresolved_temporal_object_contact_conflict"
                elif selected_state.startswith("candidate_contact"):
                    status = "unresolved_contact_geometry"
                else:
                    status = "rejected_contact_absent"
    else:
        selected = scored[0] if scored else None
        status = "accepted_no_contact" if selected is None or str(selected["row"].get("contact_state_measurement")) == "no_contact_evidence" else "unresolved_unexpected_contact"
    return {
        "frame_idx": frame_idx,
        "expected_contact": expected,
        "status": status,
        "temporal_validation_status": temporal_status,
        "selected_measurement_id": selected["row"].get("measurement_id") if selected else None,
        "selected_contact_state_measurement": selected["row"].get("contact_state_measurement") if selected else None,
        "selected_requires_temporal_validation": needs_temporal_validation(selected["row"], args) if selected else None,
        "selected_total_cost": selected["total_cost"] if selected else None,
        "selected_factor_terms": selected["factor_terms"] if selected else None,
        "candidate_count": len(scored),
        "candidates": scored,
    }


def solve(args: argparse.Namespace) -> dict[str, Any]:
    anchors = anchor_specs(args.anchor_qc)
    contact_by_frame = rows_by_frame(args.contact_measurements)
    temporal = temporal_validation_by_frame(args.object_depth_temporal_validation)
    wanted = sorted({int(frame) for frame in args.frame_indices}) if args.frame_indices else sorted(anchors)
    states = [solve_frame(idx, anchors[idx], contact_by_frame.get(idx, []), temporal, args) for idx in wanted]
    counts: dict[str, int] = defaultdict(int)
    for row in states:
        counts[str(row["status"])] += 1
    report = {
        "status": "ok",
        "method": "solve_v17_anchor_contact_state_graph",
        "anchor_qc": str(args.anchor_qc),
        "contact_measurements": [str(path) for path in args.contact_measurements],
        "object_depth_temporal_validation": [str(path) for path in args.object_depth_temporal_validation],
        "frame_indices": wanted,
        "temporal_validation_required_source_substring": list(args.temporal_validation_required_source_substring),
        "state_counts": dict(sorted(counts.items())),
        "states": states,
    }
    write_json(args.output_json, report)
    print(json.dumps({"status": report["status"], "method": report["method"], "state_counts": report["state_counts"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-qc", type=Path, required=True)
    parser.add_argument("--contact-measurements", type=Path, nargs="+", required=True)
    parser.add_argument("--object-depth-temporal-validation", type=Path, nargs="*", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="*")
    parser.add_argument("--image-near-px", type=float, default=12.0)
    parser.add_argument("--metric-near-m", type=float, default=0.02)
    parser.add_argument("--w-image-gap", type=float, default=1.0)
    parser.add_argument("--w-metric-gap", type=float, default=100.0)
    parser.add_argument("--w-invalid-hand", type=float, default=100.0)
    parser.add_argument("--w-missing-modality", type=float, default=25.0)
    parser.add_argument("--max-accept-cost", type=float, default=8.0)
    parser.add_argument("--temporal-validation-required-source-substring", nargs="*", default=["object_depth"])
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
