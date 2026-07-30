#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    annotation_frames,
    depth_archive,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import (
    existing_path,
    recompute_row_samples,
    source_summary,
)
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index


STATUS = "v17_hand_tail_depth_observation_state_qc"
CLAIM = (
    "This artifact tests whether supported residual hand-surface depth tails have nearby UniDepth samples "
    "compatible with the scaled MANO surface. Nearby compatible depth supports local MANO surface/projection "
    "repair; absent nearby compatible depth supports a depth-observation or occlusion variable."
)


def optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def dilation_kernel(radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise RuntimeError("radius_px must be non-negative")
    size = 2 * radius_px + 1
    yy, xx = np.ogrid[:size, :size]
    rr = (xx - radius_px) ** 2 + (yy - radius_px) ** 2
    return (rr <= radius_px * radius_px).astype(np.uint8)


def local_depth_compatibility(
    *,
    x: np.ndarray,
    y: np.ndarray,
    hand_z: np.ndarray,
    metric_z: np.ndarray,
    selected: np.ndarray,
    depth_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_i = np.flatnonzero(selected)
    if selected_i.size == 0:
        return {
            "sample_count": 0,
            "local_search_radius_px": int(args.local_depth_search_radius_px),
            "direct_compatible_fraction": None,
            "nearby_compatible_fraction": None,
            "nearest_compatible_pixel_distance_px": summarize([]),
            "target_tail_depth_abs_gap_m": summarize([]),
        }
    depth_h, depth_w = depth_shape
    direct_abs = np.abs(hand_z[selected_i] - metric_z[selected_i])
    direct = direct_abs <= float(args.compatible_depth_abs_m)
    radius = int(args.local_depth_search_radius_px)
    finite_depth = (
        np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    compatible_seed = finite_depth & (np.abs(hand_z - metric_z) <= float(args.compatible_depth_abs_m))
    mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
    sx = x[compatible_seed].astype(np.int32)
    sy = y[compatible_seed].astype(np.int32)
    inside = (sx >= 0) & (sx < depth_w) & (sy >= 0) & (sy < depth_h)
    mask[sy[inside], sx[inside]] = 255
    kernel = dilation_kernel(radius)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    tx = x[selected_i].astype(np.int32)
    ty = y[selected_i].astype(np.int32)
    in_bounds = (tx >= 0) & (tx < depth_w) & (ty >= 0) & (ty < depth_h)
    nearby = np.zeros(selected_i.size, dtype=bool)
    nearest_distance = np.full(selected_i.size, np.nan, dtype=np.float64)
    nearby[in_bounds] = dilated[ty[in_bounds], tx[in_bounds]] > 0
    nearest_distance[in_bounds] = distance[ty[in_bounds], tx[in_bounds]]
    return {
        "sample_count": int(selected_i.size),
        "local_search_radius_px": radius,
        "direct_compatible_fraction": float(np.mean(direct)),
        "nearby_compatible_fraction": float(np.mean(nearby)),
        "nearest_compatible_pixel_distance_px": summarize(nearest_distance[np.isfinite(nearest_distance)].astype(float).tolist()),
        "target_tail_depth_abs_gap_m": summarize(direct_abs.astype(float).tolist()),
    }


def depth_observation_state(row: dict[str, Any], summary: dict[str, Any]) -> str:
    if row.get("tail_factor_candidate") is not True:
        return "not_tail_factor_candidate"
    support_state = require_str(row.get("independent_support_state"), "independent support state")
    if support_state == "tail_pixels_unsupported_by_independent_model_boxes":
        return "unsupported_projection_tail"
    fraction = optional_float(summary.get("nearby_compatible_fraction"))
    if fraction is None:
        return "unobserved_depth_search"
    if fraction >= 0.75:
        return "supported_tail_has_nearby_compatible_depth"
    if fraction >= 0.25:
        return "supported_tail_has_partial_nearby_compatible_depth"
    return "supported_tail_lacks_nearby_compatible_depth"


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "tail_state": existing_path(
            args.hand_surface_depth_tail_state_root / case / "v17_hand_surface_depth_tail_state.json",
            f"{case} hand surface-depth tail state report",
        ),
        "tail_support": existing_path(
            args.hand_tail_support_state_root / case / "v17_hand_tail_support_state.json",
            f"{case} hand tail support state report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    depth = depth_archive(existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive"))
    support_rows = [
        require_dict(raw, "support row")
        for raw in require_list(payloads["tail_support"].get("rows"), f"{case} support rows")
    ]
    surface_rows = [
        require_dict(raw, "surface tail row")
        for raw in require_list(payloads["tail_state"].get("rows"), f"{case} surface tail rows")
    ]
    if len(support_rows) != len(surface_rows):
        raise RuntimeError(f"{case} support and surface-tail row counts disagree")
    surface_by_id = {
        require_str(row.get("hand_surface_depth_tail_variable_id"), "surface tail variable id"): row
        for row in surface_rows
    }
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    for support_row in support_rows:
        variable_id = require_str(support_row.get("source_hand_surface_depth_tail_variable_id"), "source tail variable id")
        tail_row = require_dict(surface_by_id.get(variable_id), f"surface tail row {variable_id}")
        frame_idx = require_int(support_row.get("frame_idx"), "support frame_idx")
        side = require_str(support_row.get("hand_side"), "support hand_side")
        hand_i = require_int(support_row.get("hand_index"), "support hand_index")
        frame = frames.get(frame_idx)
        hand = hand_index.get((frame_idx, side, hand_i))
        base = {
            "case": case,
            "hand_tail_depth_observation_variable_id": variable_id.replace(
                "hand_surface_depth_tail:",
                "hand_tail_depth_observation:",
                1,
            ),
            "source_hand_tail_support_variable_id": require_str(
                support_row.get("hand_tail_support_variable_id"),
                "support variable id",
            ),
            "source_hand_surface_depth_tail_variable_id": variable_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_i,
            "tail_factor_candidate": bool(support_row.get("tail_factor_candidate") is True),
            "tail_pattern": support_row.get("tail_pattern"),
            "selected_support_state": support_row.get("selected_support_state"),
            "independent_support_state": support_row.get("independent_support_state"),
            "owner_sample_partition": support_row.get("owner_sample_partition"),
            **FALSE_READY,
        }
        if frame is None or hand is None:
            rows.append(
                {
                    **base,
                    "depth_observation_state": "missing_annotation_hand",
                    "missing_observation_inputs": ["annotation_hand"],
                }
            )
            continue
        samples = recompute_row_samples(
            case=case,
            tail_row=tail_row,
            frame=frame,
            hand=hand,
            depth=depth,
            mask_cache=mask_cache,
            args=args,
        )
        if samples is None:
            rows.append(
                {
                    **base,
                    "depth_observation_state": "unobserved_depth_search",
                    "missing_observation_inputs": ["recomputed_tail_pixels"],
                }
            )
            continue
        valid = samples["valid"]
        gap = samples["gap"]
        abs_tail = valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))
        summary = local_depth_compatibility(
            x=samples["x"],
            y=samples["y"],
            hand_z=samples["hand_z"],
            metric_z=samples["metric_z"],
            selected=abs_tail,
            depth_shape=samples["depth_shape"],
            args=args,
        )
        rows.append(
            {
                **base,
                "depth_observation_state": depth_observation_state(base, summary),
                "abs_tail_sample_count": int(np.count_nonzero(abs_tail)),
                "local_depth_compatibility": summary,
                "missing_observation_inputs": [],
            }
        )
    tail_rows = [row for row in rows if row.get("tail_factor_candidate") is True]
    supported_tail_rows = [
        row
        for row in tail_rows
        if row.get("independent_support_state") != "tail_pixels_unsupported_by_independent_model_boxes"
    ]
    report = {
        "method": "build_v17_hand_tail_depth_observation_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads.get(name)) for name, path in paths.items()},
        "frame_count": require_int(payloads["tail_support"].get("frame_count"), f"{case} support frame_count"),
        "hand_tail_depth_observation_variable_count": len(rows),
        "tail_factor_candidate_rows": len(tail_rows),
        "independent_supported_tail_candidate_rows": len(supported_tail_rows),
        "independent_unsupported_tail_candidate_rows": len(tail_rows) - len(supported_tail_rows),
        "depth_observation_state_counts": state_counts(rows, "depth_observation_state"),
        "tail_depth_observation_state_counts": state_counts(tail_rows, "depth_observation_state"),
        "supported_tail_depth_observation_state_counts": state_counts(supported_tail_rows, "depth_observation_state"),
        "tail_abs_sample_count": sum(require_int(row.get("abs_tail_sample_count", 0), "abs tail samples") for row in tail_rows),
        "problem_semantics": {
            "supported_tail_has_nearby_compatible_depth": "residual tail pixels have hand-compatible UniDepth samples within the local image search radius",
            "supported_tail_lacks_nearby_compatible_depth": "residual tail pixels have independent hand support but no nearby hand-compatible UniDepth samples",
            "unsupported_projection_tail": "residual tail pixels lack independent hand support and remain a projection or detector-support problem",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_tail_depth_observation_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_tail_support_state_root / "v17_hand_tail_support_state_summary.json",
        "hand tail support summary",
    )
    summary = require_dict(load_json(summary_path), "hand tail support summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary case {i}").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_tail_depth_observation_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_tail_support_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_tail_depth_observation_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_tail_depth_observation_variable_count": require_int(
                    report.get("hand_tail_depth_observation_variable_count"),
                    "depth observation variable count",
                ),
                "tail_factor_candidate_rows": require_int(
                    report.get("tail_factor_candidate_rows"),
                    "tail candidate rows",
                ),
                "independent_supported_tail_candidate_rows": require_int(
                    report.get("independent_supported_tail_candidate_rows"),
                    "supported tail candidate rows",
                ),
                "independent_unsupported_tail_candidate_rows": require_int(
                    report.get("independent_unsupported_tail_candidate_rows"),
                    "unsupported tail candidate rows",
                ),
                "tail_depth_observation_state_counts": require_dict(
                    report.get("tail_depth_observation_state_counts"),
                    "tail depth observation state counts",
                ),
                "supported_tail_depth_observation_state_counts": require_dict(
                    report.get("supported_tail_depth_observation_state_counts"),
                    "supported tail depth observation state counts",
                ),
                "tail_abs_sample_count": require_int(report.get("tail_abs_sample_count"), "tail abs sample count"),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_tail_depth_observation_variable_count": sum(
            require_int(report.get("hand_tail_depth_observation_variable_count"), "depth observation variable count")
            for report in reports
        ),
        "tail_factor_candidate_rows": sum(
            require_int(report.get("tail_factor_candidate_rows"), "tail candidate rows") for report in reports
        ),
        "independent_supported_tail_candidate_rows": sum(
            require_int(report.get("independent_supported_tail_candidate_rows"), "supported tail rows")
            for report in reports
        ),
        "independent_unsupported_tail_candidate_rows": sum(
            require_int(report.get("independent_unsupported_tail_candidate_rows"), "unsupported tail rows")
            for report in reports
        ),
        "tail_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("tail_depth_observation_state_counts"), "tail state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "supported_tail_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("supported_tail_depth_observation_state_counts"),
                                "supported tail state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "tail_abs_sample_count": sum(
            require_int(report.get("tail_abs_sample_count"), "tail abs sample count") for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_tail_depth_observation_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--hand-surface-depth-tail-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state"),
    )
    parser.add_argument(
        "--hand-tail-support-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_support_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state"),
    )
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--local-depth-search-radius-px", type=int, default=8)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
