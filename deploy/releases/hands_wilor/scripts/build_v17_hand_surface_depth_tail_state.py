#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)


STATUS = "v17_hand_surface_depth_tail_state_qc"
CLAIM = (
    "This artifact materializes the residual hand-surface depth tail after the intrinsics and per-row "
    "scale counterfactuals. Rows that still fail depth compatibility after the per-row scalar oracle need "
    "local MANO surface, pose, occlusion, or depth-observation variables; a single hand-depth scalar cannot "
    "explain their visible depth signal."
)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def source_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if payload is not None:
        out["status"] = payload.get("status")
        out["method"] = payload.get("method")
    return out


def optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return require_str(value, label)


def optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_from_summary(summary: dict[str, Any], key: str, label: str) -> float:
    value = optional_finite(summary.get(key))
    if value is None:
        raise RuntimeError(f"{label}.{key} must be finite")
    return value


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value = optional_finite(row.get(key))
        if value is not None:
            values.append(value)
    return summarize(values)


def owner_partition(mode: dict[str, Any], owner_label: str | None) -> dict[str, Any] | None:
    if owner_label is None:
        return None
    partitions = require_dict(mode.get("partitions"), "per-row scale partitions")
    return require_dict(partitions.get(owner_label), f"per-row scale owner partition {owner_label}")


def tail_pattern(gap: dict[str, Any], abs_gap: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    median = finite_from_summary(gap, "median", "owner gap")
    p05 = finite_from_summary(gap, "p05", "owner gap")
    p95 = finite_from_summary(gap, "p95", "owner gap")
    abs_p95 = finite_from_summary(abs_gap, "p95", "owner abs gap")
    negative_tail = p05 < -float(args.max_p95_abs_depth_gap_m)
    positive_tail = p95 > float(args.max_p95_abs_depth_gap_m)
    median_offset = abs(median) > float(args.max_median_abs_depth_gap_m)
    if negative_tail and positive_tail:
        pattern = "bidirectional_surface_depth_tail"
    elif negative_tail:
        pattern = "hand_surface_in_front_tail"
    elif positive_tail:
        pattern = "hand_surface_behind_tail"
    elif median_offset:
        pattern = "median_depth_offset_after_scalar_scale"
    elif abs_p95 > float(args.max_p95_abs_depth_gap_m):
        pattern = "abs_depth_tail_without_signed_quantile"
    else:
        pattern = "no_scalar_depth_tail"
    return {
        "tail_pattern": pattern,
        "negative_tail_exceeds_threshold": negative_tail,
        "positive_tail_exceeds_threshold": positive_tail,
        "median_offset_exceeds_threshold": median_offset,
        "signed_gap_p05_m": p05,
        "signed_gap_p95_m": p95,
        "signed_gap_median_m": median,
        "abs_gap_p95_m": abs_p95,
    }


def row_scale_spread(row: dict[str, Any]) -> float | None:
    candidate = require_dict(row.get("row_scale_candidate"), "row scale candidate")
    if candidate.get("available") is not True:
        return None
    ratio = require_dict(candidate.get("sample_ratio_summary"), "row scale sample-ratio summary")
    p05 = optional_finite(ratio.get("p05"))
    p95 = optional_finite(ratio.get("p95"))
    if p05 is None or p95 is None:
        return None
    return p95 - p05


def row_tail_state(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mode = require_dict(row.get("per_row_scale_oracle"), "per_row_scale_oracle")
    state = require_str(mode.get("state"), "per_row_scale_oracle state")
    owner_label = optional_str(mode.get("owner_sample_partition"), "per_row_scale_oracle owner partition")
    owner = owner_partition(mode, owner_label)
    ratio_spread = row_scale_spread(row)
    base = {
        "case": require_str(row.get("case"), "case"),
        "hand_surface_depth_tail_variable_id": require_str(
            row.get("hand_scale_counterfactual_variable_id"),
            "hand scale variable id",
        ).replace("hand_metric_depth:", "hand_surface_depth_tail:", 1),
        "source_hand_scale_counterfactual_variable_id": require_str(
            row.get("hand_scale_counterfactual_variable_id"),
            "hand scale variable id",
        ),
        "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(row.get("hand_side"), "hand_side"),
        "hand_index": require_int(row.get("hand_index"), "hand_index"),
        "base_available": bool(row.get("base_available") is True),
        "per_row_scale": mode.get("scale"),
        "scaled_wrist_to_middle_tip_m": mode.get("scaled_wrist_to_middle_tip_m"),
        "owner_sample_partition": owner_label,
        "owner_depth_state": mode.get("owner_depth_state"),
        "row_scale_ratio_spread_p95_minus_p05": ratio_spread,
        **FALSE_READY,
    }
    if state == "metric_depth_compatible_under_per_row_scale_oracle":
        return {
            **base,
            "tail_state": "scalar_depth_compatible",
            "tail_factor_candidate": False,
            "projection_untrusted": False,
            "unobserved": False,
            "tail_pattern": "no_scalar_depth_tail",
            "owner_gap_summary": require_dict(owner, "owner partition").get("hand_minus_unidepth_depth_m")
            if owner is not None
            else None,
            "owner_abs_gap_summary": require_dict(owner, "owner partition").get("abs_hand_minus_unidepth_depth_m")
            if owner is not None
            else None,
        }
    if state == "unobserved_under_per_row_scale_oracle":
        return {
            **base,
            "tail_state": "unobserved_after_scalar_scale",
            "tail_factor_candidate": False,
            "projection_untrusted": False,
            "unobserved": True,
            "tail_pattern": "unobserved_after_scalar_scale",
            "owner_gap_summary": None,
            "owner_abs_gap_summary": None,
        }
    if owner is None:
        raise RuntimeError(f"{base['source_hand_scale_counterfactual_variable_id']} has measured state without owner partition")
    gap = require_dict(owner.get("hand_minus_unidepth_depth_m"), "owner hand-minus-UniDepth gap")
    abs_gap = require_dict(owner.get("abs_hand_minus_unidepth_depth_m"), "owner abs hand-minus-UniDepth gap")
    pattern = tail_pattern(gap, abs_gap, args)
    if state == "depth_repair_candidate_under_per_row_scale_oracle":
        return {
            **base,
            "tail_state": f"scalar_depth_tail_incompatible:{owner_label}",
            "tail_factor_candidate": True,
            "projection_untrusted": False,
            "unobserved": False,
            **pattern,
            "owner_gap_summary": gap,
            "owner_abs_gap_summary": abs_gap,
            "owner_distance_to_active_object_mask_px": owner.get("distance_to_active_object_mask_px"),
        }
    if state == "metric_depth_measured_projection_untrusted_under_per_row_scale_oracle":
        return {
            **base,
            "tail_state": f"projection_untrusted_after_scalar_scale:{owner_label}",
            "tail_factor_candidate": False,
            "projection_untrusted": True,
            "unobserved": False,
            **pattern,
            "owner_gap_summary": gap,
            "owner_abs_gap_summary": abs_gap,
            "owner_distance_to_active_object_mask_px": owner.get("distance_to_active_object_mask_px"),
        }
    raise RuntimeError(f"unsupported per-row scale oracle state: {state}")


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    scale_path = existing_path(
        args.hand_scale_depth_counterfactual_root / case / "v17_hand_scale_depth_counterfactual.json",
        f"{case} hand scale-depth counterfactual",
    )
    scale = require_dict(load_json(scale_path), f"{case} hand scale-depth counterfactual")
    rows = [row_tail_state(require_dict(raw, "hand scale row"), args) for raw in require_list(scale.get("rows"), "scale rows")]
    if len(rows) != require_int(scale.get("hand_scale_counterfactual_variable_count"), f"{case} scale variable count"):
        raise RuntimeError(f"{case} surface-tail rows disagree with scale counterfactual")
    scale_modes = require_dict(scale.get("mode_summaries"), f"{case} scale mode summaries")
    scale_oracle = require_dict(scale_modes.get("per_row_scale_oracle"), f"{case} per-row scale oracle")
    accepted_rows = bool_count(rows, "tail_factor_candidate") + sum(
        1 for row in rows if row.get("tail_state") == "scalar_depth_compatible"
    )
    if accepted_rows > len(rows):
        raise RuntimeError(f"{case} impossible surface-tail count")
    if bool_count(rows, "tail_factor_candidate") != require_int(
        scale_oracle.get("depth_repair_factor_candidate_rows"),
        f"{case} per-row scale repair rows",
    ):
        raise RuntimeError(f"{case} surface-tail candidates disagree with scale counterfactual")
    scalar_compatible_rows = sum(1 for row in rows if row.get("tail_state") == "scalar_depth_compatible")
    if scalar_compatible_rows != require_int(
        scale_oracle.get("metric_hand_state_accepted_rows"),
        f"{case} per-row scale accepted rows",
    ):
        raise RuntimeError(f"{case} surface-tail accepted rows disagree with scale counterfactual")
    tail_rows = [row for row in rows if row.get("tail_factor_candidate") is True]
    report = {
        "method": "build_v17_hand_surface_depth_tail_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "hand_scale_depth_counterfactual": source_summary(scale_path, scale),
        },
        "frame_count": require_int(scale.get("frame_count"), f"{case} frame_count"),
        "hand_surface_depth_tail_variable_count": len(rows),
        "scalar_depth_compatible_rows": scalar_compatible_rows,
        "scalar_depth_tail_factor_candidate_rows": len(tail_rows),
        "projection_untrusted_after_scalar_scale_rows": bool_count(rows, "projection_untrusted"),
        "unobserved_after_scalar_scale_rows": bool_count(rows, "unobserved"),
        "tail_state_counts": state_counts(rows, "tail_state"),
        "tail_owner_partition_counts": state_counts(
            [row for row in rows if row.get("owner_sample_partition") is not None],
            "owner_sample_partition",
        ),
        "tail_pattern_counts": state_counts(rows, "tail_pattern"),
        "tail_candidate_pattern_counts": state_counts(tail_rows, "tail_pattern"),
        "tail_candidate_owner_partition_counts": state_counts(tail_rows, "owner_sample_partition"),
        "tail_candidate_abs_gap_p95_m": numeric_summary(tail_rows, "abs_gap_p95_m"),
        "tail_candidate_signed_gap_p05_m": numeric_summary(tail_rows, "signed_gap_p05_m"),
        "tail_candidate_signed_gap_p95_m": numeric_summary(tail_rows, "signed_gap_p95_m"),
        "tail_candidate_row_scale_ratio_spread_p95_minus_p05": numeric_summary(
            tail_rows,
            "row_scale_ratio_spread_p95_minus_p05",
        ),
        "all_rows_row_scale_ratio_spread_p95_minus_p05": numeric_summary(
            rows,
            "row_scale_ratio_spread_p95_minus_p05",
        ),
        "problem_semantics": {
            "scalar_depth_compatible": "the per-row scale oracle satisfies median and p95 depth thresholds",
            "scalar_depth_tail_factor_candidate": "the per-row scale oracle leaves p95 depth tails, so local surface, pose, occlusion, or depth-observation variables are required",
            "projection_untrusted_after_scalar_scale": "the depth signal may be compatible or incompatible, but the 2D hand projection residual fails the factor-readiness predicate",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_surface_depth_tail_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_scale_depth_counterfactual_root / "v17_hand_scale_depth_counterfactual_summary.json",
        "hand scale-depth counterfactual summary",
    )
    summary = require_dict(load_json(summary_path), "hand scale-depth counterfactual summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary case {i}").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_surface_depth_tail_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_scale_depth_counterfactual_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_surface_depth_tail_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_surface_depth_tail_variable_count": require_int(
                    report.get("hand_surface_depth_tail_variable_count"),
                    "tail variable count",
                ),
                "scalar_depth_compatible_rows": require_int(
                    report.get("scalar_depth_compatible_rows"),
                    "scalar compatible rows",
                ),
                "scalar_depth_tail_factor_candidate_rows": require_int(
                    report.get("scalar_depth_tail_factor_candidate_rows"),
                    "tail candidate rows",
                ),
                "projection_untrusted_after_scalar_scale_rows": require_int(
                    report.get("projection_untrusted_after_scalar_scale_rows"),
                    "projection-untrusted rows",
                ),
                "unobserved_after_scalar_scale_rows": require_int(
                    report.get("unobserved_after_scalar_scale_rows"),
                    "unobserved rows",
                ),
                "tail_state_counts": require_dict(report.get("tail_state_counts"), "tail state counts"),
                "tail_pattern_counts": require_dict(report.get("tail_pattern_counts"), "tail pattern counts"),
                "tail_candidate_pattern_counts": require_dict(
                    report.get("tail_candidate_pattern_counts"),
                    "tail candidate pattern counts",
                ),
                "tail_candidate_owner_partition_counts": require_dict(
                    report.get("tail_candidate_owner_partition_counts"),
                    "tail candidate owner counts",
                ),
                "tail_candidate_abs_gap_p95_m": require_dict(
                    report.get("tail_candidate_abs_gap_p95_m"),
                    "tail candidate abs p95",
                ),
                "tail_candidate_row_scale_ratio_spread_p95_minus_p05": require_dict(
                    report.get("tail_candidate_row_scale_ratio_spread_p95_minus_p05"),
                    "tail candidate ratio spread",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_surface_depth_tail_variable_count": sum(
            require_int(report.get("hand_surface_depth_tail_variable_count"), "tail variable count")
            for report in reports
        ),
        "scalar_depth_compatible_rows": sum(
            require_int(report.get("scalar_depth_compatible_rows"), "scalar compatible rows")
            for report in reports
        ),
        "scalar_depth_tail_factor_candidate_rows": sum(
            require_int(report.get("scalar_depth_tail_factor_candidate_rows"), "tail candidate rows")
            for report in reports
        ),
        "projection_untrusted_after_scalar_scale_rows": sum(
            require_int(report.get("projection_untrusted_after_scalar_scale_rows"), "projection-untrusted rows")
            for report in reports
        ),
        "unobserved_after_scalar_scale_rows": sum(
            require_int(report.get("unobserved_after_scalar_scale_rows"), "unobserved rows")
            for report in reports
        ),
        "tail_state_counts": dict(
            sorted(
                sum((Counter(require_dict(report.get("tail_state_counts"), "tail states")) for report in reports), Counter()).items()
            )
        ),
        "tail_pattern_counts": dict(
            sorted(
                sum(
                    (Counter(require_dict(report.get("tail_pattern_counts"), "tail patterns")) for report in reports),
                    Counter(),
                ).items()
            )
        ),
        "tail_candidate_pattern_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("tail_candidate_pattern_counts"), "tail candidate patterns"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_surface_depth_tail_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-scale-depth-counterfactual-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state"),
    )
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
