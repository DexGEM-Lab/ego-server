#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_hand_depth_factor_problem_qc"
CLAIM = (
    "This artifact converts current V17 MANO hand-depth evidence into a hand-centric factor problem. "
    "It identifies the inherited source-camera translation solve, the 2D reprojection factors, and the UniDepth "
    "front-surface depth factors that a real joint solver must own. It is a problem materialization, not an optimizer."
)
FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def side_key(hand: dict[str, Any], hand_i: int) -> str:
    side = hand.get("side")
    if isinstance(side, str) and side:
        return side
    return f"hand_{hand_i}"


def annotation_hands(path: Path) -> tuple[int, dict[tuple[int, str, int], dict[str, Any]]]:
    payload = require_dict(load_json(path), f"{path} annotations")
    frames = require_list(payload.get("frames"), f"{path} frames")
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row_i, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"{path} frames[{row_i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"{path} frames[{row_i}].frame_idx")
        for hand_i, raw_hand in enumerate(require_list(frame.get("hands", []), f"{path} frame {frame_idx} hands")):
            hand = require_dict(raw_hand, f"{path} frame {frame_idx} hands[{hand_i}]")
            key = (frame_idx, side_key(hand, hand_i), hand_i)
            if key in out:
                raise RuntimeError(f"duplicate hand key {key} in {path}")
            out[key] = hand
    return len(frames), out


def vector3_z(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return float(arr[2])


def partition(row: dict[str, Any], name: str) -> dict[str, Any]:
    parts = require_dict(row.get("sample_partitions"), "sample_partitions")
    raw = parts.get(name)
    if raw is None:
        return {
            "sample_partition": name,
            "measured": False,
            "state": "unobserved_hand_metric_depth",
            "metric_depth_signal_compatible": False,
            "metric_depth_compatible": False,
        }
    return require_dict(raw, f"sample_partitions.{name}")


def partition_measured(row: dict[str, Any], name: str) -> bool:
    return partition(row, name).get("measured") is True


def residual_ok(row: dict[str, Any]) -> bool:
    residual = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    return bool(residual.get("residual_ok") is True)


def annotation_source_state(hand: dict[str, Any] | None) -> dict[str, Any]:
    if hand is None:
        return {
            "annotation_hand_present": False,
            "source_camera_solve_status": "missing_annotation_hand",
            "backend": None,
            "track_source": None,
            "cam_t_z_m": None,
            "source_solve_median_depth_m": None,
            "source_solve_median_reprojection_error_px": None,
            "source_solve_p95_reprojection_error_px": None,
            "sparse_graph_hand_ray_shift_m": None,
        }
    solve = hand.get("source_camera_solve")
    solve_dict = solve if isinstance(solve, dict) else {}
    graph = hand.get("v17_full_timeline_factor_graph")
    graph_dict = graph if isinstance(graph, dict) else {}
    status = solve_dict.get("status")
    return {
        "annotation_hand_present": True,
        "source_camera_solve_status": status if isinstance(status, str) and status else "missing_source_camera_solve",
        "backend": hand.get("backend"),
        "track_source": hand.get("track_source"),
        "cam_t_z_m": vector3_z(hand.get("cam_t")),
        "source_solve_median_depth_m": finite_float(solve_dict.get("median_depth_m")),
        "source_solve_median_reprojection_error_px": finite_float(solve_dict.get("median_reprojection_error_px")),
        "source_solve_p95_reprojection_error_px": finite_float(solve_dict.get("p95_reprojection_error_px")),
        "sparse_graph_hand_ray_shift_m": finite_float(graph_dict.get("hand_ray_shift_m")),
    }


def build_hand_factor_row(
    row: dict[str, Any],
    annotation_by_key: dict[tuple[int, str, int], dict[str, Any]],
) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "hand row frame_idx")
    side = require_str(row.get("hand_side"), "hand row side")
    hand_index = require_int(row.get("hand_index"), "hand row hand_index")
    key = (frame_idx, side, hand_index)
    source_state = annotation_source_state(annotation_by_key.get(key))
    all_part = partition(row, "all_projected_hand_pixels")
    near_part = partition(row, "near_active_object_masks")
    far_part = partition(row, "far_from_active_object_masks")
    measured = bool(all_part.get("measured") is True)
    projection_ok = residual_ok(row)
    metric_depth_compatible = bool(row.get("metric_depth_compatible") is True)
    depth_signal_compatible = bool(all_part.get("metric_depth_signal_compatible") is True)
    depth_repair_candidate = bool(measured and projection_ok and not metric_depth_compatible)
    accepted = bool(measured and projection_ok and metric_depth_compatible)
    if accepted:
        state = "metric_hand_state_accepted"
    elif depth_repair_candidate:
        state = "depth_repair_factor_candidate"
    elif measured and not projection_ok:
        state = "metric_depth_measured_projection_untrusted"
    elif projection_ok and not measured:
        state = "projection_only_depth_unobserved"
    else:
        state = "unresolved_hand_depth_factor"
    return {
        "case": row.get("case"),
        "hand_depth_variable_id": require_str(row.get("hand_metric_depth_variable_id"), "hand variable id"),
        "frame_idx": frame_idx,
        "hand_side": side,
        "hand_index": hand_index,
        "factor_problem_state": state,
        "current_hand_metric_depth_state": require_str(row.get("hand_metric_depth_state"), "hand metric-depth state"),
        "current_metric_depth_compatible": metric_depth_compatible,
        "current_depth_signal_compatible": depth_signal_compatible,
        "projection_residual_ok": projection_ok,
        "metric_depth_measured": measured,
        "depth_repair_candidate": depth_repair_candidate,
        "metric_hand_state_accepted": accepted,
        "source_state": source_state,
        "factor_blocks": [
            {
                "factor_block": "inherited_source_camera_translation",
                "variable": f"source_camera_translation[{frame_idx:06d},{side},{hand_index}]",
                "measurement_status": source_state["source_camera_solve_status"],
                "cam_t_z_m": source_state["cam_t_z_m"],
                "source_solve_median_depth_m": source_state["source_solve_median_depth_m"],
                "factor_ready": source_state["annotation_hand_present"],
                "semantics": "monocular MANO local geometry plus 2D keypoints solved a camera translation; this is a measurement prior, not accepted metric depth.",
            },
            {
                "factor_block": "hand_2d_reprojection",
                "variable": f"mano_pose_translation_projection[{frame_idx:06d},{side},{hand_index}]",
                "projection_residual_to_measurement_px": row.get("projection_residual_to_measurement_px"),
                "factor_ready": projection_ok,
                "semantics": "2D support is live when residual_ok is true; it cannot determine metric hand depth by itself.",
            },
            {
                "factor_block": "unidepth_front_surface",
                "variable": f"hand_metric_depth[{frame_idx:06d},{side},{hand_index}]",
                "all_projected_hand_pixels": all_part,
                "near_active_object_masks": near_part,
                "far_from_active_object_masks": far_part,
                "factor_ready": measured,
                "metric_depth_compatible": metric_depth_compatible,
                "semantics": "front-most projected MANO surface depth must agree with UniDepth where the hand state is metric.",
            },
            {
                "factor_block": "temporal_hand_depth_smoothness",
                "variable": f"hand_depth_motion[{side}]",
                "factor_ready": False,
                "semantics": "future optimizer must couple adjacent hand-depth variables; this artifact only materializes per-row evidence.",
            },
        ],
        "required_solver_variables": [
            f"delta_source_camera_translation[{frame_idx:06d},{side},{hand_index}]",
            f"mano_pose_or_local_surface_state[{frame_idx:06d},{side},{hand_index}]",
            f"hand_depth_observation_switch[{frame_idx:06d},{side},{hand_index}]",
        ],
        **FALSE_READY,
    }


def rows_by_state(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def source_state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        state = require_dict(row.get("source_state"), "source_state")
        counts[require_str(state.get("source_camera_solve_status"), "source_camera_solve_status")] += 1
    return dict(sorted(counts.items()))


def source_numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        state = require_dict(row.get("source_state"), "source_state")
        value = finite_float(state.get(key))
        if value is not None:
            values.append(value)
    return summarize(values)


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
        "sparse_graph": existing_path(
            args.sparse_graph_root / case / "v17_full_timeline_factor_graph_report.json",
            f"{case} sparse graph report",
        ),
        "sparse_graph_annotations": existing_path(
            args.sparse_graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} sparse graph annotations",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items() if name != "sparse_graph_annotations"}
    frame_count, annotations = annotation_hands(paths["sparse_graph_annotations"])
    hand_metric = payloads["hand_metric_depth_state"]
    sparse = payloads["sparse_graph"]
    if frame_count != require_int(hand_metric.get("frame_count"), f"{case} hand metric frame_count"):
        raise RuntimeError(f"{case} annotation frame count disagrees with hand metric-depth state")
    if frame_count != require_int(sparse.get("frame_count"), f"{case} sparse graph frame_count"):
        raise RuntimeError(f"{case} annotation frame count disagrees with sparse graph")
    metric_rows = [
        require_dict(raw, f"{case} hand metric-depth rows[{i}]")
        for i, raw in enumerate(require_list(hand_metric.get("rows"), f"{case} hand metric-depth rows"))
    ]
    factor_rows = [build_hand_factor_row(row, annotations) for row in metric_rows]
    measured_rows = [row for row in factor_rows if row["metric_depth_measured"] is True]
    projection_ok_rows = [row for row in factor_rows if row["projection_residual_ok"] is True]
    depth_repair_rows = [row for row in factor_rows if row["depth_repair_candidate"] is True]
    accepted_rows = [row for row in factor_rows if row["metric_hand_state_accepted"] is True]
    missing_annotation_rows = [
        row
        for row in factor_rows
        if require_dict(row.get("source_state"), "source_state").get("annotation_hand_present") is not True
    ]
    if len(factor_rows) != require_int(hand_metric.get("hand_metric_depth_variable_count"), f"{case} hand variable count"):
        raise RuntimeError(f"{case} hand factor rows disagree with hand metric-depth variable count")
    if len(measured_rows) != require_int(hand_metric.get("measured_hand_depth_rows"), f"{case} measured rows"):
        raise RuntimeError(f"{case} hand factor measured rows disagree with hand metric-depth report")
    if len(projection_ok_rows) != require_int(
        hand_metric.get("projection_residual_ok_hand_rows"),
        f"{case} projection residual ok rows",
    ):
        raise RuntimeError(f"{case} projection residual-ok rows disagree with hand metric-depth report")
    report = {
        "method": "build_v17_hand_depth_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "hand_metric_depth_state": source_summary(paths["hand_metric_depth_state"], hand_metric),
            "sparse_graph": source_summary(paths["sparse_graph"], sparse),
            "sparse_graph_annotations": {
                "path": str(paths["sparse_graph_annotations"]),
                "exists": paths["sparse_graph_annotations"].exists(),
                "hand_rows_indexed": len(annotations),
            },
        },
        "frame_count": frame_count,
        "hand_depth_variable_count": len(factor_rows),
        "metric_depth_factor_rows": len(measured_rows),
        "projection_factor_ready_rows": len(projection_ok_rows),
        "depth_repair_factor_candidate_rows": len(depth_repair_rows),
        "metric_hand_state_accepted_rows": len(accepted_rows),
        "missing_annotation_hand_rows": len(missing_annotation_rows),
        "factor_problem_state_counts": rows_by_state(factor_rows, "factor_problem_state"),
        "current_hand_metric_depth_state_counts": rows_by_state(factor_rows, "current_hand_metric_depth_state"),
        "source_camera_solve_status_counts": source_state_counts(factor_rows),
        "source_solve_median_depth_m": source_numeric_summary(factor_rows, "source_solve_median_depth_m"),
        "source_cam_t_z_m": source_numeric_summary(factor_rows, "cam_t_z_m"),
        "sparse_graph_hand_ray_shift_m": source_numeric_summary(factor_rows, "sparse_graph_hand_ray_shift_m"),
        "hand_metric_depth_partition_summaries": require_dict(
            hand_metric.get("partition_summaries"),
            f"{case} hand metric-depth partition summaries",
        ),
        "sparse_graph_correction_summary": sparse.get("correction_summary"),
        "problem_semantics": {
            "current_source_camera_owner": "least-squares source-camera translation from MANO-family local geometry and 2D keypoints",
            "falsifying_measurement": "UniDepth front-surface depth at projected MANO pixels",
            "source_of_depth_error": (
                "The current hand rows have live 2D projection factors but their source-camera translations were not "
                "solved with UniDepth. The sparse graph hand-ray correction is millimeter-scale and cannot explain "
                "a near-meter MANO-minus-UniDepth gap."
            ),
            "required_solver_step": (
                "Promote hand source-camera translation, local MANO pose or local surface state, and hand-depth "
                "observation switches into the joint graph before activating physical hand-object contact factors."
            ),
        },
        "factor_rows": factor_rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_depth_factor_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_metric_depth_state_root / "v17_hand_metric_depth_state_summary.json",
        "hand metric-depth state summary",
    )
    summary = require_dict(load_json(summary_path), "hand metric-depth state summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    state_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(require_dict(report.get("factor_problem_state_counts"), "factor_problem_state_counts"))
        source_counts.update(require_dict(report.get("source_camera_solve_status_counts"), "source counts"))
    payload = {
        "method": "build_v17_hand_depth_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_metric_depth_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "problem_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_depth_factor_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_depth_variable_count": require_int(
                    report.get("hand_depth_variable_count"),
                    "hand_depth_variable_count",
                ),
                "metric_depth_factor_rows": require_int(
                    report.get("metric_depth_factor_rows"),
                    "metric_depth_factor_rows",
                ),
                "projection_factor_ready_rows": require_int(
                    report.get("projection_factor_ready_rows"),
                    "projection_factor_ready_rows",
                ),
                "depth_repair_factor_candidate_rows": require_int(
                    report.get("depth_repair_factor_candidate_rows"),
                    "depth_repair_factor_candidate_rows",
                ),
                "metric_hand_state_accepted_rows": require_int(
                    report.get("metric_hand_state_accepted_rows"),
                    "metric_hand_state_accepted_rows",
                ),
                "factor_problem_state_counts": require_dict(
                    report.get("factor_problem_state_counts"),
                    "factor_problem_state_counts",
                ),
                "source_camera_solve_status_counts": require_dict(
                    report.get("source_camera_solve_status_counts"),
                    "source_camera_solve_status_counts",
                ),
                "source_solve_median_depth_m": require_dict(
                    report.get("source_solve_median_depth_m"),
                    "source_solve_median_depth_m",
                ),
                "sparse_graph_hand_ray_shift_m": require_dict(
                    report.get("sparse_graph_hand_ray_shift_m"),
                    "sparse_graph_hand_ray_shift_m",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_depth_variable_count": sum(
            require_int(report.get("hand_depth_variable_count"), "hand_depth_variable_count")
            for report in reports
        ),
        "metric_depth_factor_rows": sum(
            require_int(report.get("metric_depth_factor_rows"), "metric_depth_factor_rows")
            for report in reports
        ),
        "projection_factor_ready_rows": sum(
            require_int(report.get("projection_factor_ready_rows"), "projection_factor_ready_rows")
            for report in reports
        ),
        "depth_repair_factor_candidate_rows": sum(
            require_int(report.get("depth_repair_factor_candidate_rows"), "depth_repair_factor_candidate_rows")
            for report in reports
        ),
        "metric_hand_state_accepted_rows": sum(
            require_int(report.get("metric_hand_state_accepted_rows"), "metric_hand_state_accepted_rows")
            for report in reports
        ),
        "factor_problem_state_counts": dict(sorted(state_counts.items())),
        "source_camera_solve_status_counts": dict(sorted(source_counts.items())),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_depth_factor_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-metric-depth-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument(
        "--sparse-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_factor_problem"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
