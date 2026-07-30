#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
from build_v17_hand_intrinsics_depth_counterfactual import (
    annotation_hand_index,
    local_hand_geometry,
    project,
    scale_depth_intrinsics,
    solve_translation,
    source_intrinsics,
    source_size_from_intrinsics,
)
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
from build_v17_hand_tail_support_state import existing_path, source_summary


STATUS = "v17_mano_articulation_factor_input_qc"
CLAIM = (
    "This artifact materializes the factor inputs needed before a MANO articulation solve can consume "
    "V17 local projection residuals. It reconstructs the repaired front-most MANO surface for each "
    "parameter-owned local projection candidate, assigns residual pixels to nearby compatible-depth "
    "seed pixels with the same search rule as the local projection problem, and records the residual "
    "and seed MANO surface vertex ids. It does not optimize MANO pose or update accepted annotation state."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def front_surface_vertex_samples(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    depth_h, depth_w = depth_shape
    uv, valid_z = project(points_camera, intrinsics)
    scale = np.asarray(
        [
            float(depth_w) / float(projection_source_size[0]),
            float(depth_h) / float(projection_source_size[1]),
        ],
        dtype=np.float64,
    )
    xy = uv * scale[None, :]
    valid = (
        valid_z
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < depth_w)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < depth_h)
    )
    if not np.any(valid):
        return None
    valid_ids = np.flatnonzero(valid)
    x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, depth_w - 1)
    y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, depth_h - 1)
    hand_z = points_camera[valid_ids, 2].astype(np.float64)
    lin = y.astype(np.int64) * int(depth_w) + x.astype(np.int64)
    order = np.lexsort((hand_z, lin))
    sorted_lin = lin[order]
    keep = np.r_[True, sorted_lin[1:] != sorted_lin[:-1]]
    front = order[keep]
    return {
        "x": x[front],
        "y": y[front],
        "hand_z": hand_z[front],
        "vertex_id": valid_ids[front].astype(np.int32),
    }


def repaired_surface_vertex_ids(
    *,
    graph_row: dict[str, Any],
    hand: dict[str, Any],
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    frame_idx = require_int(graph_row.get("frame_idx"), "graph row frame_idx")
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        raise RuntimeError(f"missing depth frame for {frame_idx}")
    hand_intrinsics = source_intrinsics(hand)
    if hand_intrinsics is None:
        raise RuntimeError(f"missing hand source intrinsics for frame {frame_idx}")
    geometry = local_hand_geometry(hand)
    if geometry is None:
        raise RuntimeError(f"missing local hand geometry for frame {frame_idx}")
    local_joints, local_vertices, keypoints2d = geometry
    depth_m = np.asarray(depth["depth"][int(depth_i)], dtype=np.float64)
    depth_intrinsics = np.asarray(depth["intrinsics"][int(depth_i)], dtype=np.float64)
    projection_source_size = source_size_from_intrinsics(hand_intrinsics)
    candidate_intrinsics = scale_depth_intrinsics(
        depth_intrinsics,
        cast(tuple[int, int], depth["source_size"]),
        projection_source_size,
    )
    translation = solve_translation(local_joints, keypoints2d, candidate_intrinsics)
    source_joints = local_joints + translation[None, :]
    source_vertices = local_vertices + translation[None, :]
    center = np.median(source_joints, axis=0)
    if not np.all(np.isfinite(center)) or float(center[2]) <= 1e-6:
        raise RuntimeError(f"nonpositive source hand center depth for frame {frame_idx}")
    center_ray = center / float(center[2])
    solved_scale = finite_float(graph_row.get("solved_scale"), "solved_scale")
    hand_ray_shift_m = finite_float(graph_row.get("hand_ray_shift_m"), "hand_ray_shift_m")
    corrected_vertices = solved_scale * source_vertices + hand_ray_shift_m * center_ray[None, :]
    depth_shape_raw = graph_row.get("depth_shape")
    if not isinstance(depth_shape_raw, list) or len(depth_shape_raw) != 2:
        raise RuntimeError("graph row depth_shape must be a two-item list")
    depth_shape = (int(depth_shape_raw[0]), int(depth_shape_raw[1]))
    if depth_shape != (int(depth_m.shape[0]), int(depth_m.shape[1])):
        raise RuntimeError(f"graph row depth shape disagrees with metric depth frame {frame_idx}")
    samples = front_surface_vertex_samples(
        corrected_vertices,
        candidate_intrinsics,
        projection_source_size,
        depth_shape,
    )
    if samples is None:
        raise RuntimeError(f"corrected MANO surface projects outside depth frame {frame_idx}")
    graph_x = np.asarray(graph_row.get("x"), dtype=np.int32)
    graph_y = np.asarray(graph_row.get("y"), dtype=np.int32)
    graph_z = np.asarray(graph_row.get("hand_z"), dtype=np.float64)
    if not (
        len(graph_x)
        == len(graph_y)
        == len(graph_z)
        == len(cast(np.ndarray, samples["x"]))
        == len(cast(np.ndarray, samples["y"]))
        == len(cast(np.ndarray, samples["hand_z"]))
    ):
        raise RuntimeError(f"front-surface sample count mismatch for frame {frame_idx}")
    if not np.array_equal(graph_x, cast(np.ndarray, samples["x"]).astype(np.int32)):
        raise RuntimeError(f"front-surface x pixel mismatch for frame {frame_idx}")
    if not np.array_equal(graph_y, cast(np.ndarray, samples["y"]).astype(np.int32)):
        raise RuntimeError(f"front-surface y pixel mismatch for frame {frame_idx}")
    max_depth_error = float(
        np.max(np.abs(graph_z - cast(np.ndarray, samples["hand_z"]).astype(np.float64)))
    )
    if max_depth_error > float(args.max_surface_depth_reconstruction_error_m):
        raise RuntimeError(
            f"front-surface depth mismatch for frame {frame_idx}: {max_depth_error:.9f} m"
        )
    return cast(np.ndarray, samples["vertex_id"]).astype(np.int32)


def assignment_pairs(
    graph_row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    vertex_id: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x = cast(np.ndarray, samples["x"]).astype(np.int32)
    y = cast(np.ndarray, samples["y"]).astype(np.int32)
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    if len(vertex_id) != len(x):
        raise RuntimeError("surface vertex id count disagrees with repaired graph samples")
    gap = hand_z - metric_z
    all_valid = (
        np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    compatible_seed = all_valid & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    selected_i = np.flatnonzero(selected)
    seed_i = np.flatnonzero(compatible_seed)
    out: dict[str, Any] = {
        "residual_sample_count": int(selected_i.size),
        "compatible_seed_sample_count": int(seed_i.size),
        "local_projection_search_radius_px": float(args.local_projection_search_radius_px),
        "compatible_depth_abs_m": float(args.compatible_depth_abs_m),
        "assigned_residual_sample_count": 0,
        "unassigned_residual_sample_count": int(selected_i.size),
        "nearby_compatible_assignment_fraction": 0.0 if selected_i.size else None,
        "assigned_pixel_shift_px": summarize([]),
        "assigned_source_residual_abs_gap_m": summarize([]),
        "assigned_target_seed_abs_gap_m": summarize([]),
        "assigned_hand_depth_delta_to_seed_m": summarize([]),
        "assigned_metric_depth_delta_to_seed_m": summarize([]),
        "factor_pair_arrays": {
            "residual_sample_index": [],
            "seed_sample_index": [],
            "residual_vertex_id": [],
            "seed_vertex_id": [],
            "residual_x": [],
            "residual_y": [],
            "seed_x": [],
            "seed_y": [],
            "pixel_shift_px": [],
            "residual_hand_depth_m": [],
            "seed_hand_depth_m": [],
            "residual_metric_depth_m": [],
            "seed_metric_depth_m": [],
        },
    }
    if selected_i.size == 0 or seed_i.size == 0:
        return out
    selected_xy = np.stack([x[selected_i].astype(np.float64), y[selected_i].astype(np.float64)], axis=1)
    seed_xy = np.stack([x[seed_i].astype(np.float64), y[seed_i].astype(np.float64)], axis=1)
    dist, nearest = cKDTree(seed_xy).query(selected_xy, k=1)
    dist = np.asarray(dist, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    assigned = np.isfinite(dist) & (dist <= float(args.local_projection_search_radius_px))
    assigned_count = int(np.count_nonzero(assigned))
    out["assigned_residual_sample_count"] = assigned_count
    out["unassigned_residual_sample_count"] = int(selected_i.size) - assigned_count
    out["nearby_compatible_assignment_fraction"] = float(assigned_count / int(selected_i.size))
    if assigned_count == 0:
        return out
    selected_match_i = selected_i[assigned]
    seed_match_i = seed_i[nearest[assigned]]
    assigned_dist = dist[assigned].astype(np.float64)
    out["assigned_pixel_shift_px"] = summarize(assigned_dist.astype(float).tolist())
    out["assigned_source_residual_abs_gap_m"] = summarize(
        np.abs(gap[selected_match_i]).astype(float).tolist()
    )
    out["assigned_target_seed_abs_gap_m"] = summarize(np.abs(gap[seed_match_i]).astype(float).tolist())
    out["assigned_hand_depth_delta_to_seed_m"] = summarize(
        (hand_z[seed_match_i] - hand_z[selected_match_i]).astype(float).tolist()
    )
    out["assigned_metric_depth_delta_to_seed_m"] = summarize(
        (metric_z[seed_match_i] - metric_z[selected_match_i]).astype(float).tolist()
    )
    out["factor_pair_arrays"] = {
        "residual_sample_index": selected_match_i.astype(int).tolist(),
        "seed_sample_index": seed_match_i.astype(int).tolist(),
        "residual_vertex_id": vertex_id[selected_match_i].astype(int).tolist(),
        "seed_vertex_id": vertex_id[seed_match_i].astype(int).tolist(),
        "residual_x": x[selected_match_i].astype(int).tolist(),
        "residual_y": y[selected_match_i].astype(int).tolist(),
        "seed_x": x[seed_match_i].astype(int).tolist(),
        "seed_y": y[seed_match_i].astype(int).tolist(),
        "pixel_shift_px": assigned_dist.astype(float).tolist(),
        "residual_hand_depth_m": hand_z[selected_match_i].astype(float).tolist(),
        "seed_hand_depth_m": hand_z[seed_match_i].astype(float).tolist(),
        "residual_metric_depth_m": metric_z[selected_match_i].astype(float).tolist(),
        "seed_metric_depth_m": metric_z[seed_match_i].astype(float).tolist(),
    }
    return out


def assert_assignment_matches_source(case: str, row_id: str, pairs: dict[str, Any], local_row: dict[str, Any]) -> None:
    assignment = require_dict(local_row.get("assignment"), f"{case} local assignment {row_id}")
    for key in [
        "residual_sample_count",
        "compatible_seed_sample_count",
        "assigned_residual_sample_count",
        "unassigned_residual_sample_count",
    ]:
        if require_int(pairs.get(key), f"{row_id} reconstructed {key}") != require_int(
            assignment.get(key),
            f"{row_id} source {key}",
        ):
            raise RuntimeError(f"{case} {row_id} reconstructed assignment disagrees on {key}")
    reconstructed_fraction = pairs.get("nearby_compatible_assignment_fraction")
    source_fraction = assignment.get("nearby_compatible_assignment_fraction")
    if reconstructed_fraction is None or source_fraction is None:
        if reconstructed_fraction != source_fraction:
            raise RuntimeError(f"{case} {row_id} reconstructed assignment fraction disagrees")
        return
    if abs(float(reconstructed_fraction) - float(source_fraction)) > 1e-12:
        raise RuntimeError(f"{case} {row_id} reconstructed assignment fraction disagrees")


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
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph report",
        ),
        "hand_local_projection_repair_problem": existing_path(
            args.hand_local_projection_repair_problem_root
            / case
            / "v17_hand_local_projection_repair_problem.json",
            f"{case} hand local projection repair problem",
        ),
        "mano_parameter_ownership_state": existing_path(
            args.mano_parameter_ownership_state_root / case / "v17_mano_parameter_ownership_state.json",
            f"{case} MANO parameter ownership state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hands = annotation_hand_index(frames)
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth npz",
    )
    depth = depth_archive(depth_path)
    repair_report = payloads["hand_depth_repair_graph"]
    local_report = payloads["hand_local_projection_repair_problem"]
    ownership_report = payloads["mano_parameter_ownership_state"]
    frame_count = len(frames)
    for name, report in [
        ("hand depth repair graph", repair_report),
        ("hand local projection repair problem", local_report),
        ("MANO parameter ownership state", ownership_report),
    ]:
        if frame_count != require_int(report.get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame count disagrees with {name}")
    graph_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "graph variable id"): row
        for row in [require_dict(raw, "repair graph row") for raw in require_list(repair_report.get("rows"), "rows")]
    }
    local_by_graph_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "local source graph id"): row
        for row in [
            require_dict(raw, "local projection row")
            for raw in require_list(local_report.get("rows"), "local projection rows")
        ]
    }
    ownership_rows = [
        require_dict(raw, "MANO ownership row")
        for raw in require_list(ownership_report.get("rows"), f"{case} ownership rows")
    ]
    candidate_rows = [
        row for row in ownership_rows if row.get("local_projection_articulation_factor_candidate") is True
    ]
    rows: list[dict[str, Any]] = []
    for owner_row in candidate_rows:
        graph_id = require_str(
            owner_row.get("source_hand_depth_repair_graph_variable_id"),
            "ownership source graph id",
        )
        graph_row = require_dict(graph_by_id.get(graph_id), f"{case} graph row {graph_id}")
        local_row = require_dict(local_by_graph_id.get(graph_id), f"{case} local row {graph_id}")
        frame_idx = require_int(owner_row.get("frame_idx"), "ownership frame_idx")
        side = require_str(owner_row.get("hand_side"), "ownership hand_side")
        hand_index = require_int(owner_row.get("hand_index"), "ownership hand_index")
        hand = require_dict(hands.get((frame_idx, side, hand_index)), f"{case} annotation hand {graph_id}")
        samples = row_samples(graph_row)
        selected = selected_residual(graph_row, samples, args)
        vertex_id = repaired_surface_vertex_ids(graph_row=graph_row, hand=hand, depth=depth, args=args)
        pairs = assignment_pairs(graph_row, samples, selected, vertex_id, args)
        assert_assignment_matches_source(case, graph_id, pairs, local_row)
        materialized = bool(
            require_int(pairs.get("assigned_residual_sample_count"), "assigned residual samples") > 0
        )
        rows.append(
            {
                "case": case,
                "mano_articulation_factor_input_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "mano_articulation_factor_input:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_local_projection_repair_variable_id": owner_row.get(
                    "source_hand_local_projection_repair_variable_id"
                ),
                "source_mano_parameter_ownership_variable_id": owner_row.get(
                    "mano_parameter_ownership_variable_id"
                ),
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_index": hand_index,
                "surface_correspondence_state": "articulation_factor_input_materialized"
                if materialized
                else "articulation_factor_input_unassigned",
                "articulation_factor_input_materialized": materialized,
                "repair_owner_sample_partition": graph_row.get("owner_sample_partition"),
                "repair_owner_depth_state": graph_row.get("owner_depth_state"),
                "local_projection_repair_state": owner_row.get("local_projection_repair_state"),
                "assignment": pairs,
                **FALSE_READY,
            }
        )
    materialized_rows = [
        row for row in rows if row.get("articulation_factor_input_materialized") is True
    ]
    report = {
        "method": "build_v17_mano_articulation_factor_input",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "local_projection_articulation_factor_candidate_rows": len(candidate_rows),
        "mano_articulation_factor_input_candidate_rows": len(rows),
        "mano_articulation_factor_input_materialized_rows": len(materialized_rows),
        "surface_correspondence_state_counts": state_counts(rows, "surface_correspondence_state"),
        "assigned_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"), "assigned")
            for row in rows
        ),
        "residual_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("residual_sample_count"), "residual")
            for row in rows
        ),
        "compatible_seed_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("compatible_seed_sample_count"), "seed")
            for row in rows
        ),
        "assigned_pixel_shift_px": summarize(
            [
                float(require_dict(row.get("assignment"), "assignment")["assigned_pixel_shift_px"]["median"])
                for row in rows
                if require_int(
                    require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                    "assigned",
                )
                > 0
            ]
        ),
        "source_mano_parameter_ownership_comparison": {
            "local_projection_articulation_factor_candidate_rows": ownership_report.get(
                "local_projection_articulation_factor_candidate_rows"
            ),
            "residual_mano_parameter_owned_rows": ownership_report.get("residual_mano_parameter_owned_rows"),
            "residual_mano_parameter_ownership_state_counts": ownership_report.get(
                "residual_mano_parameter_ownership_state_counts"
            ),
        },
        "problem_semantics": {
            "surface_correspondence": "front-most repaired MANO vertex id at each residual or seed depth pixel",
            "factor_pair": "a residual hand-surface pixel paired with a nearby same-hand compatible-depth seed pixel",
            "solver_implication": "materialized pairs can define local projection/articulation residuals, but MANO pose has not been optimized",
        },
        "parameters": {
            "local_projection_search_radius_px": float(args.local_projection_search_radius_px),
            "compatible_depth_abs_m": float(args.compatible_depth_abs_m),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
            "max_surface_depth_reconstruction_error_m": float(
                args.max_surface_depth_reconstruction_error_m
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_mano_articulation_factor_input.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.mano_parameter_ownership_state_root / "v17_mano_parameter_ownership_state_summary.json",
        "MANO parameter ownership summary",
    )
    ownership_summary = require_dict(load_json(summary_path), "MANO parameter ownership summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(ownership_summary.get("cases"), "summary cases"))
    ]
    rows = [
        require_dict(row, "articulation factor row")
        for report in reports
        for row in require_list(report.get("rows"), "articulation factor rows")
    ]
    payload = {
        "method": "build_v17_mano_articulation_factor_input",
        "status": STATUS,
        "claim": CLAIM,
        "source_mano_parameter_ownership_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_mano_articulation_factor_input.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "local_projection_articulation_factor_candidate_rows": require_int(
                    report.get("local_projection_articulation_factor_candidate_rows"),
                    "local projection articulation candidates",
                ),
                "mano_articulation_factor_input_candidate_rows": require_int(
                    report.get("mano_articulation_factor_input_candidate_rows"),
                    "factor input candidate rows",
                ),
                "mano_articulation_factor_input_materialized_rows": require_int(
                    report.get("mano_articulation_factor_input_materialized_rows"),
                    "materialized factor input rows",
                ),
                "assigned_factor_sample_count": require_int(
                    report.get("assigned_factor_sample_count"),
                    "assigned factor sample count",
                ),
                "residual_factor_sample_count": require_int(
                    report.get("residual_factor_sample_count"),
                    "residual factor sample count",
                ),
                "surface_correspondence_state_counts": require_dict(
                    report.get("surface_correspondence_state_counts"),
                    "surface correspondence state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "local_projection_articulation_factor_candidate_rows": sum(
            require_int(
                report.get("local_projection_articulation_factor_candidate_rows"),
                "local projection articulation candidates",
            )
            for report in reports
        ),
        "mano_articulation_factor_input_candidate_rows": len(rows),
        "mano_articulation_factor_input_materialized_rows": bool_count(
            rows,
            "articulation_factor_input_materialized",
        ),
        "assigned_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"), "assigned")
            for row in rows
        ),
        "residual_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("residual_sample_count"), "residual")
            for row in rows
        ),
        "compatible_seed_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("compatible_seed_sample_count"), "seed")
            for row in rows
        ),
        "surface_correspondence_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("surface_correspondence_state_counts"), "state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "assigned_pixel_shift_px": summarize(
            [
                float(require_dict(row.get("assignment"), "assignment")["assigned_pixel_shift_px"]["median"])
                for row in rows
                if require_int(
                    require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                    "assigned",
                )
                > 0
            ]
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_mano_articulation_factor_input_summary.json", payload)
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
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-local-projection-repair-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem"),
    )
    parser.add_argument(
        "--mano-parameter-ownership-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_articulation_factor_input"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--max-surface-depth-reconstruction-error-m", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
