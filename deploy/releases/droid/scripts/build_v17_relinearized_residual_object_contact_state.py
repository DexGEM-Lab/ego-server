#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUS = "v17_relinearized_residual_object_contact_state_qc"
CLAIM = (
    "This artifact tests whether the hand-depth residual rows left after the relinearized "
    "hand surface-observation graph can be owned by current object/contact evidence. It "
    "joins each residual hand row to active-object mask proximity, multi-object visible "
    "surface distances, pairwise hand-object image/depth checks, and contact-owner variables. "
    "It is a diagnostic, not a solver."
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
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be dict")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be list")
    return value


def require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be nonempty string")
    return value


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be finite number")
    return float(value)


def optional_finite_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, name)


def existing_path(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": payload.get("status"),
        "method": payload.get("method"),
        "frame_count": payload.get("frame_count"),
        "annotation_ready": bool(payload.get("annotation_ready") is True),
        "v3_solver_complete": bool(payload.get("v3_solver_complete") is True),
    }


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for i, row in enumerate(rows) if require_bool(row.get(key), f"{key} row {i}"))


def numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0}

    def pct(q: float) -> float:
        if len(finite) == 1:
            return finite[0]
        pos = (len(finite) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return finite[lo]
        frac = pos - lo
        return finite[lo] * (1.0 - frac) + finite[hi] * frac

    return {
        "count": len(finite),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "min": finite[0],
        "max": finite[-1],
    }


def require_finite_values(values: Any, name: str) -> list[float]:
    out: list[float] = []
    for i, value in enumerate(require_list(values, name)):
        if value is None:
            raise TypeError(f"{name}[{i}] must be finite number")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric")
        if not math.isfinite(float(value)):
            raise TypeError(f"{name}[{i}] must be finite")
        out.append(float(value))
    return out


def optional_finite_values(values: Any, name: str) -> list[float]:
    out: list[float] = []
    for i, value in enumerate(require_list(values, name)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric or null")
        if not math.isfinite(float(value)):
            raise TypeError(f"{name}[{i}] must be finite or null")
        out.append(float(value))
    return out


def require_same_length(context: str, **sample_arrays: list[Any]) -> None:
    lengths = {name: len(values) for name, values in sample_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise RuntimeError(f"{context} sample field length mismatch: {lengths}")


def indexed_rows(report: dict[str, Any], row_key: str) -> dict[tuple[int, str], list[dict[str, Any]]]:
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in require_list(report.get(row_key), row_key):
        row = require_dict(raw, row_key)
        frame_idx = require_int(row.get("frame_idx"), "frame_idx")
        hand_side = require_str(row.get("hand_side"), "hand_side")
        out[(frame_idx, hand_side)].append(row)
    return out


def contact_owner_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(report.get("problem_variables"), "contact problem variables"):
        row = require_dict(raw, "contact owner variable")
        key = (
            require_int(row.get("frame_idx"), "contact owner frame_idx"),
            require_str(row.get("hand_side"), "contact owner hand_side"),
        )
        if key in out:
            raise RuntimeError(f"duplicate contact owner variable for {key}")
        out[key] = row
    return out


def count_true(values: Any, name: str) -> int:
    count = 0
    for i, value in enumerate(require_list(values, name)):
        if not isinstance(value, bool):
            raise TypeError(f"{name}[{i}] must be bool")
        count += int(value)
    return count


def object_proximity_state(near_count: int, far_count: int, sample_count: int) -> str:
    if near_count > 0 and far_count > 0:
        return "mixed_active_object_mask_proximity"
    if near_count > 0:
        return "near_active_object_mask_proximity"
    if far_count > 0:
        return "far_from_active_object_mask_proximity"
    if sample_count > 0:
        return "no_active_object_mask_partition"
    return "no_residual_depth_samples"


def min_optional(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else min(finite)


def evidence_state(
    *,
    proximity_state: str,
    visible_surface_candidate_rows: int,
    pairwise_image_candidate_rows: int,
    pairwise_metric_depth_compatible_rows: int,
    contact_owner_factor_ready: bool,
    contact_owner_geometry_supported: int,
) -> str:
    if contact_owner_factor_ready:
        return "contact_owner_factor_ready"
    if pairwise_metric_depth_compatible_rows > 0:
        return "pairwise_metric_depth_contact_supported_without_owner_factor"
    if pairwise_image_candidate_rows > 0:
        return "image_contact_metric_depth_contradiction"
    if visible_surface_candidate_rows > 0:
        return "visible_surface_near_without_contact_owner_factor"
    if contact_owner_geometry_supported > 0:
        return "geometry_supported_contact_owner_without_ready_factor"
    if proximity_state in {
        "near_active_object_mask_proximity",
        "mixed_active_object_mask_proximity",
    }:
        return "active_object_mask_near_without_metric_contact"
    if proximity_state == "far_from_active_object_mask_proximity":
        return "far_field_hand_depth_residual_not_object_contact"
    return "no_object_contact_evidence_for_residual"


def row_summary(
    row: dict[str, Any],
    *,
    contact_rows: list[dict[str, Any]],
    pairwise_depth_rows: list[dict[str, Any]],
    contact_owner: dict[str, Any] | None,
) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "hand row frame_idx")
    hand_side = require_str(row.get("hand_side"), "hand row hand_side")
    near_values = require_list(row.get("near"), "near")
    far_values = require_list(row.get("far"), "far")
    hand_z_values = require_finite_values(row.get("hand_z"), "hand_z")
    object_distance_raw = require_list(row.get("object_distance_px"), "object_distance_px")
    require_same_length(
        f"{frame_idx}:{hand_side}",
        near=near_values,
        far=far_values,
        hand_z=hand_z_values,
        object_distance_px=object_distance_raw,
    )
    near_count = count_true(near_values, "near")
    far_count = count_true(far_values, "far")
    sample_count = len(hand_z_values)
    proximity = object_proximity_state(near_count, far_count, sample_count)
    measured_contact_rows = [
        item
        for item in contact_rows
        if item.get("contact_mode_state") == "measured_distance_evidence"
    ]
    visible_candidates = [
        item
        for item in measured_contact_rows
        if item.get("visible_surface_distance_candidate") is True
    ]
    contact_ready = [item for item in contact_rows if item.get("contact_factor_ready") is True]
    pairwise_image = [
        item for item in pairwise_depth_rows if item.get("pair_contact_image_candidate") is True
    ]
    pairwise_metric = [
        item
        for item in pairwise_depth_rows
        if item.get("metric_depth_compatible_candidate") is True
    ]
    pairwise_ready = [
        item
        for item in pairwise_depth_rows
        if item.get("physical_contact_factor_ready") is True
    ]
    contact_owner_factor_ready = bool(
        contact_owner is not None and contact_owner.get("contact_owner_factor_ready") is True
    )
    geometry_supported = (
        0
        if contact_owner is None
        else require_int(
            contact_owner.get("geometrically_supported_candidate_count"),
            "geometrically supported candidate count",
        )
    )
    visible_min = min_optional(
        [
            finite_number(item.get("min_symmetric_distance_m"), "min symmetric distance")
            for item in measured_contact_rows
            if item.get("min_symmetric_distance_m") is not None
        ]
    )
    object_distance_values = optional_finite_values(object_distance_raw, "object_distance_px")
    object_distance_sample_count = len(object_distance_raw)
    pairwise_abs_depth_min = min_optional(
        [
            finite_number(
                require_dict(item.get("abs_hand_minus_object_depth_m"), "abs depth").get("median"),
                "abs hand-object median",
            )
            for item in pairwise_depth_rows
            if item.get("abs_hand_minus_object_depth_m") is not None
        ]
    )
    state = evidence_state(
        proximity_state=proximity,
        visible_surface_candidate_rows=len(visible_candidates),
        pairwise_image_candidate_rows=len(pairwise_image),
        pairwise_metric_depth_compatible_rows=len(pairwise_metric),
        contact_owner_factor_ready=contact_owner_factor_ready,
        contact_owner_geometry_supported=geometry_supported,
    )
    return {
        "case": require_str(row.get("case"), "case"),
        "relinearized_residual_object_contact_state_id": require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            "hand graph id",
        ).replace("hand_depth_repair_graph:", "relinearized_residual_object_contact:", 1),
        "source_hand_depth_repair_graph_variable_id": require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            "hand graph id",
        ),
        "frame_idx": frame_idx,
        "hand_side": hand_side,
        "hand_index": require_int(row.get("hand_index"), "hand_index"),
        "relinearized_delta_applied": require_bool(
            row.get("relinearized_delta_applied"),
            "relinearized_delta_applied",
        ),
        "relinearized_reprojection_state": require_str(
            row.get("relinearized_reprojection_state"),
            "relinearized state",
        ),
        "owner_sample_partition": require_str(row.get("owner_sample_partition"), "owner sample partition"),
        "owner_depth_state": require_str(row.get("owner_depth_state"), "owner depth state"),
        "owner_median_gap_m": optional_finite_number(row.get("owner_median_gap_m"), "owner median gap"),
        "selected_residual_sample_count": sample_count,
        "near_active_object_sample_count": near_count,
        "far_from_active_object_sample_count": far_count,
        "active_object_proximity_state": proximity,
        "object_distance_px": numeric_summary(object_distance_values),
        "object_distance_valid_sample_count": len(object_distance_values),
        "object_distance_invalid_sample_count": object_distance_sample_count - len(object_distance_values),
        "multi_object_contact_measured_rows": len(measured_contact_rows),
        "multi_object_visible_surface_candidate_rows": len(visible_candidates),
        "multi_object_contact_factor_ready_rows": len(contact_ready),
        "multi_object_min_visible_surface_distance_m": visible_min,
        "pairwise_contact_depth_rows": len(pairwise_depth_rows),
        "pairwise_image_contact_candidate_rows": len(pairwise_image),
        "pairwise_metric_depth_compatible_candidate_rows": len(pairwise_metric),
        "pairwise_physical_contact_factor_ready_rows": len(pairwise_ready),
        "pairwise_depth_gap_state_counts": dict(
            sorted(
                Counter(
                    require_str(item.get("depth_gap_state"), "depth gap state")
                    for item in pairwise_depth_rows
                    if item.get("depth_gap_state") is not None
                ).items()
            )
        ),
        "pairwise_abs_hand_minus_object_depth_median_min_m": pairwise_abs_depth_min,
        "contact_owner_variable_available": bool(contact_owner is not None),
        "contact_owner_variable_state": "no_contact_owner_variable"
        if contact_owner is None
        else require_str(contact_owner.get("owner_variable_state"), "owner variable state"),
        "contact_owner_supported_candidate_count": 0
        if contact_owner is None
        else require_int(contact_owner.get("supported_candidate_count"), "supported candidate count"),
        "contact_owner_geometrically_supported_candidate_count": geometry_supported,
        "contact_owner_factor_ready": contact_owner_factor_ready,
        "residual_object_contact_evidence_state": state,
        "object_contact_closure_supported": bool(contact_owner_factor_ready),
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "relinearized_hand_surface_observation_graph": existing_path(
            args.relinearized_hand_surface_observation_graph_root
            / case
            / "v17_relinearized_hand_surface_observation_graph.json",
            f"{case} relinearized graph",
        ),
        "multi_object_contact_evidence": existing_path(
            args.multi_object_contact_evidence_root
            / case
            / "v17_multi_object_contact_evidence_report.json",
            f"{case} multi-object contact evidence",
        ),
        "pairwise_contact_depth_gap": existing_path(
            args.pairwise_contact_depth_gap_root / case / "v17_pairwise_contact_depth_gap.json",
            f"{case} pairwise contact depth gap",
        ),
        "contact_ownership_problem": existing_path(
            args.contact_ownership_problem_root / case / "v17_contact_ownership_problem.json",
            f"{case} contact ownership problem",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frame_count = require_int(
        payloads["relinearized_hand_surface_observation_graph"].get("frame_count"),
        f"{case} frame_count",
    )
    for name in [
        "multi_object_contact_evidence",
        "pairwise_contact_depth_gap",
        "contact_ownership_problem",
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count mismatch for {name}")
    contact_by_frame_side = indexed_rows(payloads["multi_object_contact_evidence"], "rows")
    pairwise_by_frame_side = indexed_rows(payloads["pairwise_contact_depth_gap"], "rows")
    owner_by_frame_side = contact_owner_index(payloads["contact_ownership_problem"])
    residual_rows = []
    for raw_row in require_list(
        payloads["relinearized_hand_surface_observation_graph"].get("rows"),
        f"{case} relinearized rows",
    ):
        row = require_dict(raw_row, "relinearized row")
        if require_bool(row.get("depth_repair_factor_candidate"), "depth_repair_factor_candidate"):
            residual_rows.append(row)
    rows = []
    for row in residual_rows:
        key = (
            require_int(row.get("frame_idx"), "hand row frame_idx"),
            require_str(row.get("hand_side"), "hand row hand_side"),
        )
        rows.append(
            row_summary(
                row,
                contact_rows=contact_by_frame_side.get(key, []),
                pairwise_depth_rows=pairwise_by_frame_side.get(key, []),
                contact_owner=owner_by_frame_side.get(key),
            )
        )
    report = {
        "method": "build_v17_relinearized_residual_object_contact_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "relinearized_hand_residual_rows": len(rows),
        "applied_relinearized_residual_rows": bool_count(rows, "relinearized_delta_applied"),
        "nonapplied_relinearized_residual_rows": len(rows) - bool_count(rows, "relinearized_delta_applied"),
        "near_active_object_residual_rows": sum(
            1 for row in rows if row["near_active_object_sample_count"] > 0
        ),
        "far_from_active_object_residual_rows": sum(
            1 for row in rows if row["far_from_active_object_sample_count"] > 0
        ),
        "active_object_proximity_state_counts": state_counts(rows, "active_object_proximity_state"),
        "residual_object_contact_evidence_state_counts": state_counts(
            rows,
            "residual_object_contact_evidence_state",
        ),
        "rows_with_pairwise_image_contact_candidate": sum(
            1 for row in rows if row["pairwise_image_contact_candidate_rows"] > 0
        ),
        "rows_with_pairwise_metric_depth_compatible_candidate": sum(
            1 for row in rows if row["pairwise_metric_depth_compatible_candidate_rows"] > 0
        ),
        "rows_with_multi_object_visible_surface_candidate": sum(
            1 for row in rows if row["multi_object_visible_surface_candidate_rows"] > 0
        ),
        "rows_with_contact_owner_variable": bool_count(rows, "contact_owner_variable_available"),
        "rows_with_contact_owner_factor_ready": bool_count(rows, "contact_owner_factor_ready"),
        "rows_with_object_contact_closure_supported": bool_count(rows, "object_contact_closure_supported"),
        "object_distance_valid_sample_count": sum(
            require_int(row.get("object_distance_valid_sample_count"), "object distance valid samples")
            for row in rows
        ),
        "object_distance_invalid_sample_count": sum(
            require_int(row.get("object_distance_invalid_sample_count"), "object distance invalid samples")
            for row in rows
        ),
        "rows_with_invalid_object_distance_samples": sum(
            1
            for row in rows
            if require_int(
                row.get("object_distance_invalid_sample_count"),
                "object distance invalid samples",
            )
            > 0
        ),
        "multi_object_min_visible_surface_distance_m": numeric_summary(
            [
                finite_number(row["multi_object_min_visible_surface_distance_m"], "visible min distance")
                for row in rows
                if row["multi_object_min_visible_surface_distance_m"] is not None
            ]
        ),
        "pairwise_abs_hand_minus_object_depth_median_min_m": numeric_summary(
            [
                finite_number(row["pairwise_abs_hand_minus_object_depth_median_min_m"], "pairwise abs gap")
                for row in rows
                if row["pairwise_abs_hand_minus_object_depth_median_min_m"] is not None
            ]
        ),
        "object_contact_closure_supported": bool_count(rows, "object_contact_closure_supported") > 0,
        "problem_semantics": {
            "object_contact_closure_supported": (
                "current contact-owner evidence supplies a geometry-backed object-contact factor for "
                "the same frame and hand side as a relinearized hand-depth residual"
            ),
            "image_contact_metric_depth_contradiction": (
                "image-plane hand/object support exists, but pairwise object depth is incompatible with hand depth"
            ),
            "active_object_mask_near_without_metric_contact": (
                "the hand residual samples are near an active object mask, but current object geometry/depth "
                "does not support a contact owner"
            ),
            "far_field_hand_depth_residual_not_object_contact": (
                "the hand residual samples are far from active object masks under the current object plans"
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_relinearized_residual_object_contact_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [row for case in cases for row in require_list(case.get("rows"), "case rows")]
    summary = {
        "method": "build_v17_relinearized_residual_object_contact_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame_count") for case in cases),
        "relinearized_hand_residual_rows": len(rows),
        "applied_relinearized_residual_rows": bool_count(rows, "relinearized_delta_applied"),
        "nonapplied_relinearized_residual_rows": len(rows) - bool_count(rows, "relinearized_delta_applied"),
        "near_active_object_residual_rows": sum(
            1 for row in rows if require_int(row.get("near_active_object_sample_count"), "near count") > 0
        ),
        "far_from_active_object_residual_rows": sum(
            1 for row in rows if require_int(row.get("far_from_active_object_sample_count"), "far count") > 0
        ),
        "active_object_proximity_state_counts": state_counts(rows, "active_object_proximity_state"),
        "residual_object_contact_evidence_state_counts": state_counts(
            rows,
            "residual_object_contact_evidence_state",
        ),
        "rows_with_pairwise_image_contact_candidate": sum(
            1
            for row in rows
            if require_int(row.get("pairwise_image_contact_candidate_rows"), "pairwise image rows") > 0
        ),
        "rows_with_pairwise_metric_depth_compatible_candidate": sum(
            1
            for row in rows
            if require_int(
                row.get("pairwise_metric_depth_compatible_candidate_rows"),
                "pairwise metric rows",
            )
            > 0
        ),
        "rows_with_multi_object_visible_surface_candidate": sum(
            1
            for row in rows
            if require_int(
                row.get("multi_object_visible_surface_candidate_rows"),
                "visible candidate rows",
            )
            > 0
        ),
        "rows_with_contact_owner_variable": bool_count(rows, "contact_owner_variable_available"),
        "rows_with_contact_owner_factor_ready": bool_count(rows, "contact_owner_factor_ready"),
        "rows_with_object_contact_closure_supported": bool_count(rows, "object_contact_closure_supported"),
        "object_distance_valid_sample_count": sum(
            require_int(row.get("object_distance_valid_sample_count"), "object distance valid samples")
            for row in rows
        ),
        "object_distance_invalid_sample_count": sum(
            require_int(row.get("object_distance_invalid_sample_count"), "object distance invalid samples")
            for row in rows
        ),
        "rows_with_invalid_object_distance_samples": sum(
            1
            for row in rows
            if require_int(
                row.get("object_distance_invalid_sample_count"),
                "object distance invalid samples",
            )
            > 0
        ),
        "multi_object_min_visible_surface_distance_m": numeric_summary(
            [
                finite_number(row["multi_object_min_visible_surface_distance_m"], "visible min distance")
                for row in rows
                if row["multi_object_min_visible_surface_distance_m"] is not None
            ]
        ),
        "pairwise_abs_hand_minus_object_depth_median_min_m": numeric_summary(
            [
                finite_number(row["pairwise_abs_hand_minus_object_depth_median_min_m"], "pairwise abs gap")
                for row in rows
                if row["pairwise_abs_hand_minus_object_depth_median_min_m"] is not None
            ]
        ),
        "object_contact_closure_supported": bool_count(rows, "object_contact_closure_supported") > 0,
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame_count"),
                "relinearized_hand_residual_rows": require_int(
                    case.get("relinearized_hand_residual_rows"),
                    "case residual rows",
                ),
                "applied_relinearized_residual_rows": require_int(
                    case.get("applied_relinearized_residual_rows"),
                    "case applied residual rows",
                ),
                "near_active_object_residual_rows": require_int(
                    case.get("near_active_object_residual_rows"),
                    "case near residual rows",
                ),
                "rows_with_pairwise_image_contact_candidate": require_int(
                    case.get("rows_with_pairwise_image_contact_candidate"),
                    "case image contact rows",
                ),
                "rows_with_pairwise_metric_depth_compatible_candidate": require_int(
                    case.get("rows_with_pairwise_metric_depth_compatible_candidate"),
                    "case metric contact rows",
                ),
                "rows_with_contact_owner_factor_ready": require_int(
                    case.get("rows_with_contact_owner_factor_ready"),
                    "case contact owner ready rows",
                ),
                "object_distance_invalid_sample_count": require_int(
                    case.get("object_distance_invalid_sample_count"),
                    "case invalid object distance samples",
                ),
                "rows_with_invalid_object_distance_samples": require_int(
                    case.get("rows_with_invalid_object_distance_samples"),
                    "case rows with invalid object distance samples",
                ),
                "residual_object_contact_evidence_state_counts": require_dict(
                    case.get("residual_object_contact_evidence_state_counts"),
                    "case evidence state counts",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_relinearized_residual_object_contact_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relinearized-hand-surface-observation-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--pairwise-contact-depth-gap-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument(
        "--contact-ownership-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_ownership_problem"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_residual_object_contact_state"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
