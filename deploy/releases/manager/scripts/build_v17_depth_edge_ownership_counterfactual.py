#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import ndimage

from build_v17_hand_depth_repair_residual_owner_state import owner_valid_mask, row_samples
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    depth_archive,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)


STATUS = "v17_depth_edge_ownership_counterfactual_qc"
CLAIM = (
    "This artifact tests whether UniDepth depth-discontinuity ownership explains the persistent "
    "hand-depth residual rows after the pose-enabled full-residual graph. It recomputes the same "
    "metric-depth acceptance predicate on the current solved hand state while excluding UniDepth "
    "pixels inside hand-independent depth-discontinuity bands. It changes no hand state and is a "
    "measurement-ownership counterfactual, not solver closure."
)

PRESERVED = "legacy_compatible_interior_compatible"
BROKEN = "legacy_compatible_interior_lost"
FLIPPED = "legacy_residual_interior_compatible"
UNOBSERVED = "legacy_residual_interior_unobserved"
STILL_INCOMPATIBLE = "legacy_residual_interior_incompatible"
NOT_MEASURED = "row_not_measured"


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(finite_number(value, key))
    return summarize(values)


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "method": require_str(payload.get("method"), "source method"),
        "status": require_str(payload.get("status"), "source status"),
    }


def depth_edge_band(
    depth_m: np.ndarray,
    *,
    edge_window_px: int,
    edge_depth_range_m: float,
    edge_band_dilation_px: int,
) -> np.ndarray:
    """Hand-independent depth-discontinuity band mask from UniDepth only."""
    if depth_m.ndim != 2:
        raise RuntimeError("depth frame must be 2D")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    footprint = np.ones((edge_window_px, edge_window_px), dtype=bool)
    high = np.where(valid, depth_m, -np.inf)
    low = np.where(valid, depth_m, np.inf)
    local_max = ndimage.grey_dilation(high, footprint=footprint, mode="nearest")
    local_min = ndimage.grey_erosion(low, footprint=footprint, mode="nearest")
    local_range = local_max - local_min
    edge = (~valid) | ~np.isfinite(local_range) | (local_range > edge_depth_range_m)
    if edge_band_dilation_px > 0:
        edge = ndimage.binary_dilation(edge, iterations=edge_band_dilation_px)
    return cast(np.ndarray, edge)


def applied_rows(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("rows"), f"{label} rows"):
        row = require_dict(raw, f"{label} row")
        if row.get("relinearized_delta_applied") is not True:
            continue
        graph_id = require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            f"{label} hand depth repair graph id",
        )
        if graph_id in out:
            raise RuntimeError(f"{label} duplicate applied row id: {graph_id}")
        out[graph_id] = row
    expected = require_int(report.get("relinearized_variable_rows"), f"{label} variable rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} applied rows {len(out)} do not match reported variables {expected}")
    return out


def studied_row_ids(report: dict[str, Any], label: str) -> set[str]:
    out: set[str] = set()
    for raw in require_list(report.get("rows"), f"{label} rows"):
        row = require_dict(raw, f"{label} row")
        if row.get("studied_depth_owner_row") is True:
            out.add(
                require_str(
                    row.get("source_hand_depth_repair_graph_variable_id"),
                    f"{label} source graph id",
                )
            )
    expected = require_int(report.get("studied_depth_owner_rows"), f"{label} studied rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} studied row ids {len(out)} do not match reported {expected}")
    return out


def interior_state(
    *,
    interior_count: int,
    median_gap: float | None,
    p95_abs_gap: float | None,
    residual_ok: bool,
    args: argparse.Namespace,
) -> str:
    if interior_count < int(args.min_depth_pixels) or median_gap is None or p95_abs_gap is None:
        return "interior_unobserved"
    signal = bool(
        abs(median_gap) <= float(args.max_median_abs_depth_gap_m)
        and p95_abs_gap <= float(args.max_p95_abs_depth_gap_m)
    )
    if signal and residual_ok:
        return "interior_metric_depth_compatible"
    if signal:
        return "interior_depth_match_projection_residual_untrusted"
    if median_gap > float(args.max_median_abs_depth_gap_m):
        return "interior_hand_behind_metric_depth"
    if median_gap < -float(args.max_median_abs_depth_gap_m):
        return "interior_hand_in_front_of_metric_depth"
    return "interior_depth_tail_incompatible"


def transition_state(legacy_compatible: bool, interior: str) -> str:
    interior_compatible = bool(interior == "interior_metric_depth_compatible")
    if legacy_compatible and interior_compatible:
        return PRESERVED
    if legacy_compatible:
        return BROKEN
    if interior_compatible:
        return FLIPPED
    if interior == "interior_unobserved":
        return UNOBSERVED
    return STILL_INCOMPATIBLE


def row_counterfactual(
    *,
    graph_id: str,
    row: dict[str, Any],
    edge: np.ndarray,
    studied: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    common = {
        "case": require_str(row.get("case"), "case"),
        "source_hand_depth_repair_graph_variable_id": graph_id,
        "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(row.get("hand_side"), "hand_side"),
        "hand_index": require_int(row.get("hand_index"), "hand_index"),
        "studied_depth_owner_row": studied,
        "legacy_metric_depth_compatible": bool(row.get("metric_depth_compatible") is True),
        "legacy_owner_depth_state": require_str(row.get("owner_depth_state"), "owner depth state"),
        **FALSE_READY,
    }
    if row.get("partitions") is None:
        return {
            **common,
            "interior_state": NOT_MEASURED,
            "transition_state": NOT_MEASURED,
            "owner_valid_pixels": 0,
            "edge_owner_pixels": 0,
            "interior_valid_pixels": 0,
            "edge_owner_fraction": None,
            "interior_median_gap_m": None,
            "interior_p95_abs_gap_m": None,
            "legacy_owner_median_gap_m": None,
        }
    samples = row_samples(row)
    valid = owner_valid_mask(row, samples, args)
    x = cast(np.ndarray, samples["x"]).astype(np.int32)
    y = cast(np.ndarray, samples["y"]).astype(np.int32)
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    depth_shape = cast(tuple[int, int], samples["depth_shape"])
    if edge.shape != depth_shape:
        raise RuntimeError(f"edge mask shape {edge.shape} disagrees with row depth shape {depth_shape}")
    on_edge = edge[y, x]
    interior = valid & ~on_edge
    gap = hand_z - metric_z
    valid_count = int(np.count_nonzero(valid))
    interior_count = int(np.count_nonzero(interior))
    edge_count = int(np.count_nonzero(valid & on_edge))
    median_gap: float | None = None
    p95_abs_gap: float | None = None
    if interior_count > 0:
        interior_gap = gap[interior]
        median_gap = float(np.median(interior_gap))
        p95_abs_gap = float(np.percentile(np.abs(interior_gap), 95.0))
    projection = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    residual_ok = bool(projection.get("residual_ok") is True)
    state = interior_state(
        interior_count=interior_count,
        median_gap=median_gap,
        p95_abs_gap=p95_abs_gap,
        residual_ok=residual_ok,
        args=args,
    )
    legacy_gap = row.get("owner_median_gap_m")
    return {
        **common,
        "interior_state": state,
        "transition_state": transition_state(common["legacy_metric_depth_compatible"] is True, state),
        "owner_valid_pixels": valid_count,
        "edge_owner_pixels": edge_count,
        "interior_valid_pixels": interior_count,
        "edge_owner_fraction": None if valid_count == 0 else float(edge_count / valid_count),
        "interior_median_gap_m": median_gap,
        "interior_p95_abs_gap_m": p95_abs_gap,
        "interior_gap_m": summarize(gap[interior].astype(float).tolist()),
        "legacy_owner_median_gap_m": None
        if legacy_gap is None
        else finite_number(legacy_gap, "legacy owner median gap"),
    }


def report_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    studied = [row for row in rows if row.get("studied_depth_owner_row") is True]
    legacy_compatible = [row for row in rows if row.get("legacy_metric_depth_compatible") is True]
    still = [row for row in rows if row.get("transition_state") == STILL_INCOMPATIBLE]
    return {
        "variable_rows": len(rows),
        "legacy_compatible_rows": len(legacy_compatible),
        "transition_state_counts": state_counts(rows, "transition_state"),
        "interior_state_counts": state_counts(rows, "interior_state"),
        "preserved_rows": sum(1 for row in rows if row.get("transition_state") == PRESERVED),
        "broken_prior_acceptance_rows": sum(1 for row in rows if row.get("transition_state") == BROKEN),
        "flipped_to_interior_compatible_rows": sum(
            1 for row in rows if row.get("transition_state") == FLIPPED
        ),
        "interior_unobserved_rows": sum(1 for row in rows if row.get("transition_state") == UNOBSERVED),
        "interior_still_incompatible_rows": len(still),
        "interior_still_incompatible_state_counts": state_counts(still, "interior_state"),
        "edge_owner_fraction": numeric_summary(rows, "edge_owner_fraction"),
        "interior_median_gap_m": numeric_summary(rows, "interior_median_gap_m"),
        "studied_depth_owner_rows": len(studied),
        "studied_transition_state_counts": state_counts(studied, "transition_state"),
        "studied_flipped_to_interior_compatible_rows": sum(
            1 for row in studied if row.get("transition_state") == FLIPPED
        ),
        "studied_interior_unobserved_rows": sum(
            1 for row in studied if row.get("transition_state") == UNOBSERVED
        ),
        "studied_interior_still_incompatible_rows": sum(
            1 for row in studied if row.get("transition_state") == STILL_INCOMPATIBLE
        ),
        "studied_interior_median_gap_m": numeric_summary(studied, "interior_median_gap_m"),
        "studied_edge_owner_fraction": numeric_summary(studied, "edge_owner_fraction"),
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    pose_graph_path = existing_path(
        args.pose_full_residual_graph_root
        / case
        / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
        f"{case} pose full-residual graph",
    )
    depth_owner_path = existing_path(
        args.depth_owner_diagnostic_root / case / "v17_full_residual_depth_owner_diagnostic.json",
        f"{case} depth-owner diagnostic",
    )
    pose_graph = require_dict(load_json(pose_graph_path), f"{case} pose full-residual graph")
    depth_owner = require_dict(load_json(depth_owner_path), f"{case} depth-owner diagnostic")
    if require_int(pose_graph.get("frame_count"), f"{case} pose frame_count") != require_int(
        depth_owner.get("frame_count"),
        f"{case} depth-owner frame_count",
    ):
        raise RuntimeError(f"{case} frame_count mismatch between pose graph and depth-owner diagnostic")
    rows_by_id = applied_rows(pose_graph, f"{case} pose graph")
    studied = studied_row_ids(depth_owner, f"{case} depth-owner diagnostic")
    unknown_studied = studied.difference(rows_by_id)
    if unknown_studied:
        raise RuntimeError(f"{case} studied rows missing from pose graph: {sorted(unknown_studied)[:3]}")
    depth_path = existing_path(
        Path(require_str(pose_graph.get("metric_depth_npz"), "metric_depth_npz")),
        f"{case} metric depth archive",
    )
    depth = depth_archive(depth_path)
    edge_cache: dict[int, np.ndarray] = {}

    def edge_for_frame(frame_idx: int) -> np.ndarray:
        cached = edge_cache.get(frame_idx)
        if cached is not None:
            return cached
        depth_i = depth["frame_to_i"].get(frame_idx)
        if depth_i is None:
            raise RuntimeError(f"{case} frame {frame_idx} missing from metric depth archive")
        edge = depth_edge_band(
            np.asarray(depth["depth"][depth_i], dtype=np.float64),
            edge_window_px=int(args.edge_window_px),
            edge_depth_range_m=float(args.edge_depth_range_m),
            edge_band_dilation_px=int(args.edge_band_dilation_px),
        )
        edge_cache[frame_idx] = edge
        return edge

    rows = [
        row_counterfactual(
            graph_id=graph_id,
            row=rows_by_id[graph_id],
            edge=edge_for_frame(require_int(rows_by_id[graph_id].get("frame_idx"), "frame_idx")),
            studied=graph_id in studied,
            args=args,
        )
        for graph_id in sorted(rows_by_id)
    ]
    report = {
        "method": "build_v17_depth_edge_ownership_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "pose_full_residual_graph": source_summary(pose_graph_path, pose_graph),
            "depth_owner_diagnostic": source_summary(depth_owner_path, depth_owner),
            "metric_depth_npz": str(depth_path),
        },
        "frame_count": require_int(pose_graph.get("frame_count"), f"{case} frame_count"),
        "parameters": {
            "edge_window_px": int(args.edge_window_px),
            "edge_depth_range_m": float(args.edge_depth_range_m),
            "edge_band_dilation_px": int(args.edge_band_dilation_px),
            "min_depth_pixels": int(args.min_depth_pixels),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
        },
        **report_counts(rows),
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_depth_edge_ownership_counterfactual.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [
        require_dict(raw, "case row")
        for case in cases
        for raw in require_list(case.get("rows"), "case rows")
    ]
    summary = {
        "method": "build_v17_depth_edge_ownership_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame_count") for case in cases),
        **report_counts(rows),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame_count"),
                "variable_rows": require_int(case.get("variable_rows"), "case variable rows"),
                "flipped_to_interior_compatible_rows": require_int(
                    case.get("flipped_to_interior_compatible_rows"),
                    "case flipped rows",
                ),
                "broken_prior_acceptance_rows": require_int(
                    case.get("broken_prior_acceptance_rows"),
                    "case broken rows",
                ),
                "studied_flipped_to_interior_compatible_rows": require_int(
                    case.get("studied_flipped_to_interior_compatible_rows"),
                    "case studied flipped rows",
                ),
                "studied_interior_still_incompatible_rows": require_int(
                    case.get("studied_interior_still_incompatible_rows"),
                    "case studied still rows",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_depth_edge_ownership_counterfactual_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-full-residual-graph-root",
        type=Path,
        default=Path(
            "/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose"
        ),
    )
    parser.add_argument(
        "--depth-owner-diagnostic-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_edge_ownership_counterfactual"),
    )
    parser.add_argument("--edge-window-px", type=int, default=7)
    parser.add_argument("--edge-depth-range-m", type=float, default=0.10)
    parser.add_argument("--edge-band-dilation-px", type=int, default=6)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
