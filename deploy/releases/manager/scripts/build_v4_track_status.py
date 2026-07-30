#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object) -> float | None:
    if value is None:
        return None
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def row_by_frame(rows: list[dict]) -> dict[int, dict]:
    return {int(row["frame_idx"]): row for row in rows}


def verdict_by_frame(prune_report: dict) -> dict[int, dict]:
    return {int(row["frame_idx"]): row for row in prune_report.get("rows", [])}


def zbuffer_status(row: dict | None, args: argparse.Namespace) -> tuple[str, list[str]]:
    if row is None:
        return "missing", ["no_zbuffer_row"]
    iou = finite(row.get("silhouette_mask_iou"))
    p95 = finite(row.get("zbuffer_depth_abs_p95_m"))
    inside = finite(row.get("visible_silhouette_inside_mask_fraction"))
    reasons = []
    if iou is None or iou < float(args.min_silhouette_iou):
        reasons.append("low_silhouette_iou")
    if p95 is None or p95 > float(args.max_zbuffer_p95_m):
        reasons.append("high_zbuffer_p95")
    if inside is None or inside < float(args.min_visible_inside_fraction):
        reasons.append("low_visible_inside_fraction")
    return ("ok" if not reasons else "ambiguous"), reasons


def contact_frames(contact_report: dict) -> set[int]:
    return {
        int(row["frame_idx"])
        for row in contact_report.get("rows_detail", [])
        if bool(row.get("reliable_for_contact", False))
        or bool(row.get("geometry_backed_temporal_contact", False))
    }


def build(args: argparse.Namespace) -> dict:
    vlm = load_json(args.vlm_selection_qc)
    prune = load_json(args.prune_qc)
    zbuffer = load_json(args.zbuffer_qc)
    contact = load_json(args.contact_qc)
    selected_sdf = load_json(args.selected_contact_sdf_qc)
    full_sdf = load_json(args.full_hand_sdf_qc)

    vlm_frames = set(map(int, vlm.get("frames", [])))
    vlm_accepted = set(map(int, vlm.get("accepted_frames", [])))
    prune_rows = verdict_by_frame(prune)
    z_rows = row_by_frame(zbuffer.get("rows", []))
    contact_frame_set = contact_frames(contact)

    rows = []
    counts: dict[str, int] = {}
    for frame_idx in sorted(vlm_frames.union(z_rows).union(prune_rows)):
        prune_row = prune_rows.get(frame_idx)
        verdict = (prune_row or {}).get("verdict", {})
        model_correct = bool(verdict.get("mask_correct_for_track", frame_idx in vlm_accepted))
        z_status, z_reasons = zbuffer_status(z_rows.get(frame_idx), args)
        status_reasons = []
        if frame_idx not in vlm_accepted:
            status_reasons.append("vlm_no_selected_mask")
        if not model_correct:
            status_reasons.append("semantic_reject")
        status_reasons.extend(z_reasons)
        if "semantic_reject" in status_reasons:
            status = "semantic_reject"
        elif "vlm_no_selected_mask" in status_reasons or z_status == "missing":
            status = "needs_completion"
        elif z_status == "ambiguous":
            status = "ambiguous_measured"
        else:
            status = "measured"
        counts[status] = counts.get(status, 0) + 1
        z_row = z_rows.get(frame_idx, {})
        rows.append(
            {
                "frame_idx": frame_idx,
                "status": status,
                "completion_required": status in {"semantic_reject", "needs_completion"},
                "reasons": status_reasons,
                "vlm_selected": frame_idx in vlm_accepted,
                "vlm_mask_correct_for_track": model_correct,
                "vlm_confidence": verdict.get("confidence"),
                "dominant_mask_object": verdict.get("dominant_mask_object"),
                "prune_kept": None if prune_row is None else bool(prune_row.get("kept", False)),
                "prune_reasons": [] if prune_row is None else list(prune_row.get("prune_reasons", [])),
                "has_reliable_contact": frame_idx in contact_frame_set,
                "silhouette_mask_iou": z_row.get("silhouette_mask_iou"),
                "visible_silhouette_inside_mask_fraction": z_row.get("visible_silhouette_inside_mask_fraction"),
                "zbuffer_depth_abs_median_m": z_row.get("zbuffer_depth_abs_median_m"),
                "zbuffer_depth_abs_p95_m": z_row.get("zbuffer_depth_abs_p95_m"),
            }
        )

    report = {
        "status": "ok",
        "method": "build_v4_track_status",
        "track_id": vlm.get("track_id"),
        "object_status_semantics": {
            "measured": "model-selected mask is semantically correct and measured geometry passes residual thresholds",
            "ambiguous_measured": "model-selected mask is semantically correct but at least one measured residual is weak",
            "semantic_reject": "model-selected mask is not the intended object track; the target track needs completion or a new model hypothesis",
            "needs_completion": "no selected model mask or no measured geometry residual row is available",
        },
        "thresholds": {
            "min_silhouette_iou": float(args.min_silhouette_iou),
            "max_zbuffer_p95_m": float(args.max_zbuffer_p95_m),
            "min_visible_inside_fraction": float(args.min_visible_inside_fraction),
        },
        "counts": counts,
        "frames": rows,
        "inputs": {
            "vlm_selection_qc": str(args.vlm_selection_qc),
            "prune_qc": str(args.prune_qc),
            "zbuffer_qc": str(args.zbuffer_qc),
            "contact_qc": str(args.contact_qc),
            "selected_contact_sdf_qc": str(args.selected_contact_sdf_qc),
            "full_hand_sdf_qc": str(args.full_hand_sdf_qc),
        },
        "sdf_summaries": {
            "selected_contact": selected_sdf.get("summary", {}),
            "full_hand": full_sdf.get("summary", {}),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-selection-qc", type=Path, required=True)
    parser.add_argument("--prune-qc", type=Path, required=True)
    parser.add_argument("--zbuffer-qc", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path, required=True)
    parser.add_argument("--selected-contact-sdf-qc", type=Path, required=True)
    parser.add_argument("--full-hand-sdf-qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-silhouette-iou", type=float, default=0.90)
    parser.add_argument("--max-zbuffer-p95-m", type=float, default=0.010)
    parser.add_argument("--min-visible-inside-fraction", type=float, default=0.97)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
