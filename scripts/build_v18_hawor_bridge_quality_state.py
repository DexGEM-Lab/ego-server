#!/usr/bin/env python3
"""Classify V18 HaWoR bridge candidates without accepting them.

The input bridge is HaWoR-only. Current V18 visible hand candidates are used only
as an image-space residual/reference signal; they are not a substitute for HaWoR.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SIDES = ("left", "right")
INT_TO_SIDE = {0: "left", 1: "right"}
HAND_JOINTS = 21


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_camera, dtype=np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64)
    if points.shape != (HAND_JOINTS, 3) or intr.shape != (4,):
        return None
    if np.any(points[:, 2] <= 1e-6):
        return None
    fx, fy, cx, cy = intr
    uv = np.empty((HAND_JOINTS, 2), dtype=np.float64)
    uv[:, 0] = fx * points[:, 0] / points[:, 2] + cx
    uv[:, 1] = fy * points[:, 1] / points[:, 2] + cy
    return uv if np.isfinite(uv).all() else None


def image_inside_fraction(points2d: np.ndarray, width: float = 1920.0, height: float = 1080.0) -> float:
    inside = (points2d[:, 0] >= 0.0) & (points2d[:, 0] < width) & (points2d[:, 1] >= 0.0) & (points2d[:, 1] < height)
    return float(np.mean(inside))


def bbox_contains_fraction(points2d: np.ndarray, bbox: Any) -> float | None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    vals = np.asarray([finite_float(v) for v in bbox], dtype=np.float64)
    if not np.isfinite(vals).all():
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    inside = (points2d[:, 0] >= x0) & (points2d[:, 0] <= x1) & (points2d[:, 1] >= y0) & (points2d[:, 1] <= y1)
    return float(np.mean(inside))


def current_hands_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        side = str(hand.get("hand_side") or hand.get("side") or "").lower()
        if side in SIDES:
            out[side] = hand
    return out


def current_reference_projection(hand: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else None
    if mano is None:
        return None, None
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics")
    if not (isinstance(joints, list) and len(joints) == HAND_JOINTS and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return None, str(mano.get("source")) if mano.get("source") is not None else None
    pts = np.asarray(joints, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)[None, :]
    return project(pts, np.asarray(intr, dtype=np.float64)), str(mano.get("source")) if mano.get("source") is not None else None


def classify_row(
    residual_median: float | None,
    current_visibility: str | None,
    hawor_inside: float | None,
    reference_inside: float | None,
    bbox_inside: float | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if residual_median is None:
        return "no_current_projection_reference_candidate", ["current_visible_hand_projection_reference_missing"]
    if hawor_inside is not None and hawor_inside < 0.5:
        blockers.append("hawor_projection_mostly_out_of_frame")
    if reference_inside is not None and reference_inside < 0.5:
        blockers.append("current_reference_projection_mostly_out_of_frame")
    if current_visibility and current_visibility != "visible":
        blockers.append(f"current_visibility_{current_visibility}")
    if residual_median <= 50.0 and (hawor_inside is None or hawor_inside >= 0.8) and (reference_inside is None or reference_inside >= 0.8):
        if current_visibility == "visible":
            return "projection_supported_visible_hawor_bridge_candidate", blockers
        return "projection_supported_nonvisible_hawor_bridge_candidate", blockers
    if residual_median <= 100.0 and (hawor_inside is None or hawor_inside >= 0.5):
        blockers.append("moderate_projection_residual")
        return "moderate_residual_uncertain_hawor_bridge_candidate", blockers
    if residual_median > 500.0 and hawor_inside is not None and hawor_inside < 0.2:
        blockers.append("large_residual_with_hawor_out_of_frame")
        return "residual_tail_hawor_out_of_frame_or_visibility_conflict", blockers
    if residual_median > 200.0:
        blockers.append("large_projection_residual")
        if hawor_inside is not None and reference_inside is not None and hawor_inside >= 0.8 and reference_inside >= 0.8:
            blockers.append("both_projections_in_frame_but_disagree")
            return "large_residual_in_frame_conflict_candidate", blockers
        return "large_residual_uncertain_bridge_candidate", blockers
    blockers.append("residual_above_supported_threshold")
    return "unsupported_residual_uncertain_hawor_bridge_candidate", blockers


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann_path = args.source_root / case / "annotations_v18_full.json"
    ann = load_json(ann_path)
    frames = ann.get("frames") if isinstance(ann.get("frames"), list) else []
    frame_count = int(ann.get("frame_count") or len(frames))
    bridge_report_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_state_report.json"
    out_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_state.json"
    base = {
        "method": "build_v18_hawor_bridge_quality_state",
        "case": case,
        "source_annotations": str(ann_path),
        "bridge_report": str(bridge_report_path),
        "expected_frame_count": frame_count,
        "expected_frame_side_rows": frame_count * 2,
        "claim_scope": "HaWoR_bridge_quality_candidate_state_no_WiLoR_substitution_no_foundation_acceptance",
        "accepted_v18_hawor_foundation": False,
        "v18_physical_hand_state_valid_from_quality": False,
        "contact_occlusion_nonpenetration_recomputed": False,
    }
    if not bridge_report_path.exists():
        report = {
            **base,
            "status": "blocked_missing_hawor_bridge_report",
            "quality_rows": [],
            "quality_counts": {},
            "blocking_reasons": ["hawor_bridge_report_missing"],
            "elapsed_s": time.perf_counter() - start,
        }
        write_json(out_path, report)
        return report
    bridge_report = load_json(bridge_report_path)
    npz_path = bridge_report.get("bridge_candidate_npz")
    if not isinstance(npz_path, str) or not Path(npz_path).exists():
        report = {
            **base,
            "status": "blocked_no_hawor_bridge_candidates_for_case",
            "bridge_candidate_npz": npz_path,
            "quality_rows": [],
            "quality_counts": {},
            "blocking_reasons": bridge_report.get("blocking_reasons", ["hawor_bridge_candidate_npz_missing"]),
            "elapsed_s": time.perf_counter() - start,
        }
        write_json(out_path, report)
        return report
    z = np.load(npz_path)
    by_frame = {int(frame.get("frame_idx", -1)): frame for frame in frames if isinstance(frame, dict)}
    intr_hawor = np.asarray([2304.0, 2304.0, 960.0, 540.0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    supported_residuals: list[float] = []
    reference_sources: Counter[str] = Counter()
    for row_idx, (frame_idx_raw, side_raw, joints_camera) in enumerate(zip(np.asarray(z["frame_idx"], dtype=np.int32), np.asarray(z["side"], dtype=np.int8), np.asarray(z["joints_hawor_camera_m"], dtype=np.float64))):
        frame_idx = int(frame_idx_raw)
        side = INT_TO_SIDE.get(int(side_raw), "unknown")
        frame = by_frame.get(frame_idx, {})
        current = current_hands_by_side(frame).get(side, {}) if isinstance(frame, dict) else {}
        hawor_projection = project(joints_camera, intr_hawor)
        current_projection, ref_source = current_reference_projection(current) if current else (None, None)
        residual_median: float | None = None
        residual_p95: float | None = None
        h_inside: float | None = None
        r_inside: float | None = None
        bbox_inside: float | None = None
        if hawor_projection is not None:
            h_inside = image_inside_fraction(hawor_projection)
            if current:
                bbox_inside = bbox_contains_fraction(hawor_projection, current.get("bbox_xyxy"))
        if hawor_projection is not None and current_projection is not None:
            diff = np.linalg.norm(hawor_projection - current_projection, axis=1)
            residual_median = float(np.median(diff))
            residual_p95 = float(np.percentile(diff, 95.0))
            r_inside = image_inside_fraction(current_projection)
            residuals.append(residual_median)
            if ref_source:
                reference_sources[str(ref_source)] += 1
        visibility = str(current.get("visibility_state")) if current.get("visibility_state") is not None else None
        quality, blockers = classify_row(residual_median, visibility, h_inside, r_inside, bbox_inside)
        if quality.startswith("projection_supported") and residual_median is not None:
            supported_residuals.append(residual_median)
        rows.append({
            "frame_idx": frame_idx,
            "side": side,
            "quality_state": quality,
            "quality_blockers": blockers,
            "current_visibility_state": visibility,
            "reference_projection_source_family": "current_v18_visible_hand_candidate_projection_used_only_for_residual_measurement_not_requirement_substitution",
            "reference_projection_source_backend": ref_source,
            "projection_residual_px_median": residual_median,
            "projection_residual_px_p95": residual_p95,
            "hawor_projected_inside_image_fraction": h_inside,
            "reference_projected_inside_image_fraction": r_inside,
            "hawor_projected_inside_current_bbox_fraction": bbox_inside,
        })
    counts = Counter(row["quality_state"] for row in rows)
    accepted_like = [key for key in counts if "accepted" in key]
    blocking_reasons = [
        "quality_state_is_candidate_only_not_foundation_acceptance",
        "task5_hawor_absent_blocks_all_cases_requirement" if case == "trash_1050" else "case_hawor_absent_or_bridge_missing",
        "contact_occlusion_nonpenetration_not_recomputed_from_quality_state",
    ]
    if counts.get("residual_tail_hawor_out_of_frame_or_visibility_conflict", 0) > 0:
        blocking_reasons.append("residual_tail_requires_visibility_reference_localization_before_downstream_use")
    report = {
        **base,
        "status": "hawor_bridge_quality_candidate_state_built_not_accepted",
        "bridge_candidate_npz": npz_path,
        "bridge_candidate_rows": int(len(rows)),
        "missing_bridge_rows": int(frame_count * 2 - len(rows)),
        "quality_counts": dict(sorted(counts.items())),
        "accepted_like_quality_state_names": accepted_like,
        "reference_projection_rows": int(len(residuals)),
        "reference_projection_source_backend_counts": dict(sorted(reference_sources.items())),
        "projection_residual_px_median_per_row": summarize(residuals),
        "supported_candidate_projection_residual_px_median_per_row": summarize(supported_residuals),
        "quality_rows": rows,
        "blocking_reasons": blocking_reasons,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(out_path, report)
    return report


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V18 HaWoR bridge quality state",
        "",
        "This artifact classifies HaWoR bridge candidates. It is not foundation acceptance and does not use current V18/WiLoR candidates as a substitute for HaWoR; current candidates are residual references only.",
        "",
        f"Status: `{summary['status']}`",
        f"All cases quality accepted as foundation: `{summary['all_cases_quality_foundation_accepted']}`",
        f"V18 physical hand state valid from quality: `{summary['v18_physical_hand_state_valid_from_quality']}`",
        "",
    ]
    for case in summary["cases"]:
        lines += [
            f"## {case['case']}",
            "",
            f"Status: `{case['status']}`",
            f"Bridge rows: `{case.get('bridge_candidate_rows')}/{case.get('expected_frame_side_rows')}`",
            f"Quality counts: `{case.get('quality_counts')}`",
            f"Reference residual summary: `{case.get('projection_residual_px_median_per_row')}`",
            f"Supported candidate residual summary: `{case.get('supported_candidate_projection_residual_px_median_per_row')}`",
            f"Blocking reasons: `{case.get('blocking_reasons')}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_hawor_bridge_quality_state",
        "status": "candidate_quality_state_built_not_foundation_accepted",
        "claim_scope": "HaWoR_bridge_quality_state_no_model_substitution_no_full_V18_closure",
        "output_root": str(args.output_root),
        "all_cases_quality_foundation_accepted": False,
        "v18_physical_hand_state_valid_from_quality": False,
        "cases": cases,
        "blocking_reasons": sorted({reason for case in cases for reason in case.get("blocking_reasons", []) if isinstance(reason, str)}),
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_bridge_state"
    write_json(out_dir / "v18_hawor_bridge_quality_state_summary.json", summary)
    write_markdown(out_dir / "V18_HAWOR_BRIDGE_QUALITY_STATE.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
