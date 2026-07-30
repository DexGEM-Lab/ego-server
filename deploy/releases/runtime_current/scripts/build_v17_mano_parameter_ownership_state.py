#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from build_v17_hand_tail_support_state import existing_path, source_summary


STATUS = "v17_mano_parameter_ownership_state_qc"
CLAIM = (
    "This artifact tests whether saved MANO parameters own the local hand geometry used by V17 "
    "hand-depth residuals. It replays each residual row through the WiLoR MANO wrapper, applies the "
    "hand-side convention, and aligns the wrapper output to the stored V17 local hand vertices. Rows "
    "that reproduce the stored vertices and joints can support future MANO articulation variables; "
    "rows that fail this ownership test require a parameter-geometry contract repair before any "
    "articulation optimizer can consume local projection factors."
)


def patch_legacy_mano_loader() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if name not in np.__dict__:
            setattr(np, name, value)


def load_wilor_mano_class(wilor_root: Path) -> Any:
    path = wilor_root / "wilor" / "models" / "mano_wrapper.py"
    if not path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO wrapper: {path}")
    spec = importlib.util.spec_from_file_location("wilor_mano_wrapper_for_v17_ownership", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WiLoR MANO wrapper spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def nested_float(row: dict[str, Any], key: str) -> float | None:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value for row in rows if (value := nested_float(row, key)) is not None]
    return summarize(values)


def annotation_index(annotations: dict[str, Any]) -> dict[tuple[int, str, int], dict[str, Any]]:
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for raw_frame in require_list(annotations.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "annotation frame_idx")
        for hand_i, raw_hand in enumerate(require_list(frame.get("hands", []), "annotation hands")):
            hand = require_dict(raw_hand, "annotation hand")
            side = require_str(hand.get("side"), "hand side")
            key = (frame_idx, side, hand_i)
            if key in out:
                raise RuntimeError(f"duplicate hand annotation key {key}")
            out[key] = hand
    return out


def array2d(value: Any, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise RuntimeError(f"{name} must be an Nx{width} array")
    if np.any(~np.isfinite(arr)):
        raise RuntimeError(f"{name} contains nonfinite values")
    return arr


def mano_param_arrays(hand: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = require_dict(hand.get("mano_params"), "mano_params")
    global_orient = np.asarray(params.get("global_orient"), dtype=np.float64)
    hand_pose = np.asarray(params.get("hand_pose"), dtype=np.float64)
    betas = np.asarray(params.get("betas"), dtype=np.float64)
    if global_orient.shape != (1, 3, 3):
        raise RuntimeError("global_orient must be 1x3x3 rotation matrix")
    if hand_pose.shape != (15, 3, 3):
        raise RuntimeError("hand_pose must be 15x3x3 rotation matrices")
    if betas.shape != (10,):
        raise RuntimeError("betas must contain 10 shape coefficients")
    return global_orient, hand_pose, betas


def side_sign(side: str) -> float:
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise RuntimeError(f"unsupported hand side {side}")


def apply_side_sign(points: np.ndarray, side: str) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., 0] = side_sign(side) * out[..., 0]
    return out


def similarity_from_to(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape or source.shape[1] != 3:
        raise RuntimeError("similarity alignment expects matching Nx3 arrays")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    tgt = target - target_center
    covariance = src.T @ tgt / float(len(source))
    u, s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    variance = float(np.sum(src * src) / float(len(source)))
    if variance <= 0.0 or not np.isfinite(variance):
        raise RuntimeError("degenerate MANO wrapper geometry for similarity alignment")
    scale = float(np.sum(s) / variance)
    translation = target_center - scale * (source_center @ rotation.T)
    aligned = scale * (source @ rotation.T) + translation[None, :]
    error = np.linalg.norm(aligned - target, axis=1)
    return scale, rotation, translation, error


def replay_mano_local_geometry(model: Any, hand: dict[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    global_orient, hand_pose, betas = mano_param_arrays(hand)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient[None], dtype=torch.float32),
            hand_pose=torch.tensor(hand_pose[None], dtype=torch.float32),
            betas=torch.tensor(betas[None], dtype=torch.float32),
            return_verts=True,
            pose2rot=False,
        )
    vertices = apply_side_sign(out.vertices[0].detach().cpu().numpy(), side)
    joints = apply_side_sign(out.joints[0].detach().cpu().numpy(), side)
    return vertices, joints


def ownership_metrics(
    model: Any,
    hand: dict[str, Any],
    side: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_vertices = array2d(hand.get("vertices_camera"), 3, "vertices_camera")
    target_joints = array2d(hand.get("joints3d_camera"), 3, "joints3d_camera")
    source_vertices, source_joints = replay_mano_local_geometry(model, hand, side)
    if source_vertices.shape != target_vertices.shape:
        raise RuntimeError("MANO wrapper vertices do not match stored local vertices")
    if source_joints.shape != target_joints.shape:
        raise RuntimeError("MANO wrapper joints do not match stored local joints")
    scale, rotation, translation, vertex_error = similarity_from_to(source_vertices, target_vertices)
    aligned_joints = scale * (source_joints @ rotation.T) + translation[None, :]
    joint_error = np.linalg.norm(aligned_joints - target_joints, axis=1)
    vertex_median = float(np.median(vertex_error))
    vertex_p95 = float(np.percentile(vertex_error, 95.0))
    joint_median = float(np.median(joint_error))
    joint_p95 = float(np.percentile(joint_error, 95.0))
    owned = bool(
        vertex_median <= float(args.max_vertex_median_error_m)
        and vertex_p95 <= float(args.max_vertex_p95_error_m)
        and joint_median <= float(args.max_joint_median_error_m)
        and joint_p95 <= float(args.max_joint_p95_error_m)
    )
    return {
        "mano_parameter_geometry_owned": owned,
        "wilor_similarity_scale": float(scale),
        "wilor_similarity_rotation_det": float(np.linalg.det(rotation)),
        "wilor_similarity_translation_m": translation.astype(float).tolist(),
        "vertex_alignment_error_m": summarize(vertex_error.astype(float).tolist()),
        "joint_alignment_error_m": summarize(joint_error.astype(float).tolist()),
        "thresholds": {
            "max_vertex_median_error_m": float(args.max_vertex_median_error_m),
            "max_vertex_p95_error_m": float(args.max_vertex_p95_error_m),
            "max_joint_median_error_m": float(args.max_joint_median_error_m),
            "max_joint_p95_error_m": float(args.max_joint_p95_error_m),
        },
    }


def row_state(base: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    if base.get("repair_residual_factor_candidate") is not True:
        return "not_repair_residual_factor_candidate"
    if metrics is None:
        return "mano_parameter_owner_missing"
    if metrics.get("mano_parameter_geometry_owned") is True:
        return "mano_parameters_own_local_geometry"
    return "mano_parameter_geometry_mismatch"


def case_problem(case: str, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "hand_local_projection_repair_problem": existing_path(
            args.hand_local_projection_repair_problem_root
            / case
            / "v17_hand_local_projection_repair_problem.json",
            f"{case} hand local projection repair problem",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    hands = annotation_index(payloads["annotations"])
    local_report = payloads["hand_local_projection_repair_problem"]
    rows: list[dict[str, Any]] = []
    for raw in require_list(local_report.get("rows"), f"{case} local projection rows"):
        local_row = require_dict(raw, "local projection row")
        frame_idx = require_int(local_row.get("frame_idx"), "local projection frame_idx")
        side = require_str(local_row.get("hand_side"), "local projection hand_side")
        hand_index = require_int(local_row.get("hand_index"), "local projection hand_index")
        local_state = require_str(
            local_row.get("local_projection_repair_state"),
            "local projection repair state",
        )
        base = {
            "case": case,
            "mano_parameter_ownership_variable_id": require_str(
                local_row.get("hand_local_projection_repair_variable_id"),
                "local projection variable id",
            ).replace("hand_local_projection_repair:", "mano_parameter_ownership:", 1),
            "source_hand_local_projection_repair_variable_id": require_str(
                local_row.get("hand_local_projection_repair_variable_id"),
                "local projection variable id",
            ),
            "source_hand_depth_repair_graph_variable_id": local_row.get(
                "source_hand_depth_repair_graph_variable_id"
            ),
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "repair_residual_factor_candidate": bool(
                local_row.get("repair_residual_factor_candidate") is True
            ),
            "local_projection_repair_state": local_state,
            "local_projection_repair_factor_candidate": bool(
                local_row.get("local_projection_repair_factor_candidate") is True
            ),
            **FALSE_READY,
        }
        if base["repair_residual_factor_candidate"] is not True:
            rows.append(
                {
                    **base,
                    "mano_parameter_ownership_state": "not_repair_residual_factor_candidate",
                    "mano_parameter_geometry_owned": False,
                    "local_projection_articulation_factor_candidate": False,
                    "mixed_projection_articulation_observation_candidate": False,
                    "ownership_metrics": None,
                    "missing_mano_parameter_inputs": [],
                }
            )
            continue
        hand = hands.get((frame_idx, side, hand_index))
        if hand is None:
            rows.append(
                {
                    **base,
                    "mano_parameter_ownership_state": "mano_parameter_owner_missing",
                    "mano_parameter_geometry_owned": False,
                    "local_projection_articulation_factor_candidate": False,
                    "mixed_projection_articulation_observation_candidate": False,
                    "ownership_metrics": None,
                    "missing_mano_parameter_inputs": ["annotation_hand"],
                }
            )
            continue
        metrics = ownership_metrics(model, hand, side, args)
        state = row_state(base, metrics)
        owned = bool(metrics.get("mano_parameter_geometry_owned") is True)
        rows.append(
            {
                **base,
                "mano_parameter_ownership_state": state,
                "mano_parameter_geometry_owned": owned,
                "local_projection_articulation_factor_candidate": bool(
                    owned and local_state == "local_projection_repair_factor_candidate"
                ),
                "mixed_projection_articulation_observation_candidate": bool(
                    owned and local_state == "mixed_projection_depth_observation_owner"
                ),
                "ownership_metrics": metrics,
                "missing_mano_parameter_inputs": [],
            }
        )
    residual_rows = [row for row in rows if row.get("repair_residual_factor_candidate") is True]
    owned_rows = [row for row in residual_rows if row.get("mano_parameter_geometry_owned") is True]
    report = {
        "method": "build_v17_mano_parameter_ownership_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(local_report.get("frame_count"), f"{case} frame_count"),
        "mano_parameter_ownership_variable_count": len(rows),
        "repair_residual_factor_candidate_rows": len(residual_rows),
        "residual_mano_parameter_owned_rows": len(owned_rows),
        "local_projection_repair_factor_candidate_rows": bool_count(
            residual_rows,
            "local_projection_repair_factor_candidate",
        ),
        "local_projection_articulation_factor_candidate_rows": bool_count(
            residual_rows,
            "local_projection_articulation_factor_candidate",
        ),
        "mixed_projection_articulation_observation_candidate_rows": bool_count(
            residual_rows,
            "mixed_projection_articulation_observation_candidate",
        ),
        "mano_parameter_ownership_state_counts": state_counts(rows, "mano_parameter_ownership_state"),
        "residual_mano_parameter_ownership_state_counts": state_counts(
            residual_rows,
            "mano_parameter_ownership_state",
        ),
        "owned_alignment_error_summary": {
            "vertex_median_error_m": numeric_summary(owned_rows, "ownership_metrics.vertex_alignment_error_m.median"),
            "vertex_p95_error_m": numeric_summary(owned_rows, "ownership_metrics.vertex_alignment_error_m.p95"),
            "joint_median_error_m": numeric_summary(owned_rows, "ownership_metrics.joint_alignment_error_m.median"),
            "joint_p95_error_m": numeric_summary(owned_rows, "ownership_metrics.joint_alignment_error_m.p95"),
            "wilor_similarity_scale": numeric_summary(owned_rows, "ownership_metrics.wilor_similarity_scale"),
        },
        "source_local_projection_comparison": {
            "repair_residual_factor_candidate_rows": local_report.get(
                "repair_residual_factor_candidate_rows"
            ),
            "local_projection_repair_factor_candidate_rows": local_report.get(
                "local_projection_repair_factor_candidate_rows"
            ),
            "partial_projection_depth_mixed_owner_rows": local_report.get(
                "partial_projection_depth_mixed_owner_rows"
            ),
            "depth_observation_or_occlusion_owner_rows": local_report.get(
                "depth_observation_or_occlusion_owner_rows"
            ),
            "projection_support_unresolved_rows": local_report.get("projection_support_unresolved_rows"),
        },
        "problem_semantics": {
            "mano_parameters_own_local_geometry": "saved MANO parameters reproduce stored V17 local vertices and joints after WiLoR side convention and per-row similarity alignment",
            "mano_parameter_geometry_mismatch": "saved MANO parameters cannot own the local geometry used by the residual without a parameter-geometry repair",
            "local_projection_articulation_factor_candidate": "local projection factor has a MANO parameter owner but no articulation solve has been run",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_mano_parameter_ownership_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    patch_legacy_mano_loader()
    mano_model_path = args.wilor_mano_right
    if mano_model_path is None:
        mano_model_path = args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_model_path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO_RIGHT model: {mano_model_path}")
    mano_cls = load_wilor_mano_class(args.wilor_root)
    with contextlib.redirect_stdout(sys.stderr):
        model = mano_cls(
            model_path=str(mano_model_path),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False,
            batch_size=1,
        ).to(torch.device("cpu"))
    summary_path = existing_path(
        args.hand_local_projection_repair_problem_root
        / "v17_hand_local_projection_repair_problem_summary.json",
        "hand local projection repair problem summary",
    )
    local_summary = require_dict(load_json(summary_path), "hand local projection repair problem summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), model, args)
        for i, raw in enumerate(require_list(local_summary.get("cases"), "summary cases"))
    ]
    residual_rows = [
        row
        for report in reports
        for row in require_list(report.get("rows"), "mano ownership rows")
        if require_dict(row, "mano ownership row").get("repair_residual_factor_candidate") is True
    ]
    owned_rows = [
        require_dict(row, "mano ownership row")
        for row in residual_rows
        if require_dict(row, "mano ownership row").get("mano_parameter_geometry_owned") is True
    ]
    payload = {
        "method": "build_v17_mano_parameter_ownership_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_local_projection_repair_problem_summary": str(summary_path),
        "wilor_root": str(args.wilor_root),
        "wilor_mano_right": str(mano_model_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_mano_parameter_ownership_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "mano_parameter_ownership_variable_count": require_int(
                    report.get("mano_parameter_ownership_variable_count"),
                    "mano ownership variable count",
                ),
                "repair_residual_factor_candidate_rows": require_int(
                    report.get("repair_residual_factor_candidate_rows"),
                    "repair residual rows",
                ),
                "residual_mano_parameter_owned_rows": require_int(
                    report.get("residual_mano_parameter_owned_rows"),
                    "owned residual rows",
                ),
                "local_projection_repair_factor_candidate_rows": require_int(
                    report.get("local_projection_repair_factor_candidate_rows"),
                    "local projection rows",
                ),
                "local_projection_articulation_factor_candidate_rows": require_int(
                    report.get("local_projection_articulation_factor_candidate_rows"),
                    "local articulation rows",
                ),
                "mixed_projection_articulation_observation_candidate_rows": require_int(
                    report.get("mixed_projection_articulation_observation_candidate_rows"),
                    "mixed articulation rows",
                ),
                "residual_mano_parameter_ownership_state_counts": require_dict(
                    report.get("residual_mano_parameter_ownership_state_counts"),
                    "residual ownership state counts",
                ),
                "owned_alignment_error_summary": require_dict(
                    report.get("owned_alignment_error_summary"),
                    "owned alignment summary",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "mano_parameter_ownership_variable_count": sum(
            require_int(report.get("mano_parameter_ownership_variable_count"), "variable count")
            for report in reports
        ),
        "repair_residual_factor_candidate_rows": len(residual_rows),
        "residual_mano_parameter_owned_rows": len(owned_rows),
        "local_projection_repair_factor_candidate_rows": sum(
            require_int(report.get("local_projection_repair_factor_candidate_rows"), "local projection rows")
            for report in reports
        ),
        "local_projection_articulation_factor_candidate_rows": sum(
            require_int(
                report.get("local_projection_articulation_factor_candidate_rows"),
                "local articulation rows",
            )
            for report in reports
        ),
        "mixed_projection_articulation_observation_candidate_rows": sum(
            require_int(
                report.get("mixed_projection_articulation_observation_candidate_rows"),
                "mixed articulation rows",
            )
            for report in reports
        ),
        "residual_mano_parameter_ownership_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("residual_mano_parameter_ownership_state_counts"),
                                "state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "owned_alignment_error_summary": {
            "vertex_median_error_m": numeric_summary(owned_rows, "ownership_metrics.vertex_alignment_error_m.median"),
            "vertex_p95_error_m": numeric_summary(owned_rows, "ownership_metrics.vertex_alignment_error_m.p95"),
            "joint_median_error_m": numeric_summary(owned_rows, "ownership_metrics.joint_alignment_error_m.median"),
            "joint_p95_error_m": numeric_summary(owned_rows, "ownership_metrics.joint_alignment_error_m.p95"),
            "wilor_similarity_scale": numeric_summary(owned_rows, "ownership_metrics.wilor_similarity_scale"),
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_mano_parameter_ownership_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-local-projection-repair-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem"),
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--max-vertex-median-error-m", type=float, default=0.010)
    parser.add_argument("--max-vertex-p95-error-m", type=float, default=0.030)
    parser.add_argument("--max-joint-median-error-m", type=float, default=0.010)
    parser.add_argument("--max-joint-p95-error-m", type=float, default=0.030)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
