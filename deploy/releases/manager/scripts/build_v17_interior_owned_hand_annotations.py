#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apply_v17_hand_far_field_temporal_refit import finite_float
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
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
from solve_v17_interior_owned_full_residual_hand_graph import pose_delta_array
from solve_v17_mano_articulation_local import (
    corrected_replayed_state,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
)
from solve_v17_relinearized_hand_surface_observation_graph import replay_vertices


STATUS = "v17_interior_owned_hand_annotations_qc"
CLAIM = (
    "This artifact bakes the interior-owned full-residual hand graph solution into full-timeline "
    "annotations: for every solved hand variable it replays MANO with the solved pose delta, applies "
    "the case-global scale and solved camera-ray shift, and rewrites the hand's source-camera and "
    "world coordinates. Hands without a solved variable keep their prior state and are labeled. It is "
    "an annotation-state baking layer for rendering and downstream contact reasoning, not solver "
    "closure, and it does not change V17 readiness."
)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def camera_transform(frame: dict[str, Any], frame_idx: int) -> np.ndarray:
    camera = require_dict(frame.get("camera"), f"frame {frame_idx} camera")
    transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"frame {frame_idx} T_world_camera_metric must be a finite 4x4")
    return transform


def to_world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def case_problem(case: str, model: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    annotations_path = existing_path(
        args.graph_root / case / "annotations_v17_full_timeline_graph.json",
        f"{case} graph annotations",
    )
    payload = require_dict(load_json(annotations_path), f"{case} annotations payload")
    frames = annotation_frames(payload)
    hands = annotation_hand_index(frames)
    repair_path = existing_path(
        args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
        f"{case} hand depth repair graph",
    )
    repair = require_dict(load_json(repair_path), f"{case} hand depth repair graph")
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    interior_path = existing_path(
        args.interior_owned_graph_root / case / "v17_interior_owned_full_residual_hand_graph.json",
        f"{case} interior-owned hand graph",
    )
    interior = require_dict(load_json(interior_path), f"{case} interior-owned hand graph")
    visible_surface_path = existing_path(
        args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
        f"{case} visible-surface report",
    )
    visible_surface = require_dict(load_json(visible_surface_path), f"{case} visible-surface report")
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        f"{case} metric depth archive",
    )
    depth = depth_archive(depth_path)
    pose_graph_path = existing_path(
        args.pose_full_residual_graph_root
        / case
        / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
        f"{case} pose full-residual graph",
    )
    pose_graph = require_dict(load_json(pose_graph_path), f"{case} pose full-residual graph")
    pose_rows_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "pose graph id"): row
        for row in [
            require_dict(raw, "pose row") for raw in require_list(pose_graph.get("rows"), "pose rows")
        ]
    }
    baked = 0
    baked_interior_compatible = 0
    skipped_without_variable = 0
    span_values: list[float] = []
    rows_audit: list[dict[str, Any]] = []
    for raw in require_list(interior.get("rows"), f"{case} interior rows"):
        row = require_dict(raw, "interior row")
        graph_id = require_str(row.get("source_hand_depth_repair_graph_variable_id"), "interior graph id")
        frame_idx = require_int(row.get("frame_idx"), "interior frame_idx")
        side = require_str(row.get("hand_side"), "interior hand_side")
        hand_i = require_int(row.get("hand_index"), "interior hand_index")
        final_shift = finite_float(row.get("interior_total_hand_ray_shift_m"), "interior total shift")
        pose_row = require_dict(pose_rows_by_id.get(graph_id), f"{case} pose row {graph_id}")
        pose_delta_np = pose_delta_array(pose_row)
        hand = require_dict(hands.get((frame_idx, side, hand_i)), f"{case} annotation hand {graph_id}")
        repair_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
        state = corrected_replayed_state(
            model=model,
            hand=hand,
            graph_row={**repair_row, "hand_ray_shift_m": 0.0},
            depth=depth,
            device=device,
        )
        graph_scale = finite_float(repair_row.get("solved_scale"), f"{graph_id} solved_scale")
        if abs(graph_scale - scale) > 1e-9:
            raise RuntimeError(f"{graph_id} repair row scale {graph_scale} disagrees with case-global {scale}")
        pose_delta = torch.tensor(pose_delta_np[None], dtype=torch.float32, device=device)
        ray_delta = torch.tensor(float(final_shift), dtype=torch.float32, device=device)
        with torch.no_grad():
            corrected_v, corrected_j, _, _ = replay_vertices(
                model=model,
                state=state,
                pose_delta=pose_delta,
                ray_delta=ray_delta,
            )
        corrected_vertices = corrected_v[0].detach().cpu().numpy().astype(np.float64)
        corrected_joints = corrected_j[0].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(corrected_vertices).all() or not np.isfinite(corrected_joints).all():
            raise RuntimeError(f"{graph_id} corrected hand state is not finite")
        frame = require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}")
        transform = camera_transform(frame, frame_idx)
        solver_intrinsics = state["intrinsics"].detach().cpu().numpy().astype(np.float64).reshape(-1)
        if solver_intrinsics.shape != (4,) or not np.isfinite(solver_intrinsics).all():
            raise RuntimeError(f"{graph_id} solver intrinsics must be a finite 4-vector")
        prior_intrinsics = hand.get("source_intrinsics")
        hand["source_intrinsics"] = solver_intrinsics.astype(float).tolist()
        hand["v16_source_intrinsics"] = prior_intrinsics
        hand["vertices_source_camera_m"] = corrected_vertices.astype(float).tolist()
        hand["joints3d_source_camera_m"] = corrected_joints.astype(float).tolist()
        hand["vertices_world_m"] = to_world(corrected_vertices, transform).astype(float).tolist()
        hand["joints3d_world_m"] = to_world(corrected_joints, transform).astype(float).tolist()
        hand["v17_hand_state_source"] = {
            "status": "baked_from_v17_interior_owned_full_residual_hand_graph",
            "interior_owned_variable_id": require_str(
                row.get("interior_owned_variable_id"),
                "interior variable id",
            ),
            "case_global_scale": float(scale),
            "total_hand_ray_shift_m": float(final_shift),
            "pose_delta_abs_max_rad": float(np.max(np.abs(pose_delta_np))),
            "interior_state": require_str(row.get("interior_state"), "interior state"),
            "interior_metric_depth_compatible": bool(row.get("interior_metric_depth_compatible") is True),
            "camera_model": "unidepth_scaled_source_intrinsics",
        }
        span = float(
            np.linalg.norm(
                corrected_joints[12] - corrected_joints[0]
            )
        )
        span_values.append(span)
        baked += 1
        if row.get("interior_metric_depth_compatible") is True:
            baked_interior_compatible += 1
        rows_audit.append(
            {
                "graph_id": graph_id,
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_index": hand_i,
                "interior_state": hand["v17_hand_state_source"]["interior_state"],
                "total_hand_ray_shift_m": float(final_shift),
                "wrist_to_middle_tip_m": span,
            }
        )
    for (frame_idx, side, hand_i), hand in hands.items():
        if "v17_hand_state_source" not in hand:
            hand["v17_hand_state_source"] = {
                "status": "kept_prior_state_no_interior_owned_variable",
            }
            skipped_without_variable += 1
    output_dir = args.output_root / case
    annotations_out = output_dir / "annotations_v17_interior_owned_hands.json"
    payload["v17_hand_state_baking"] = {
        "method": "build_v17_interior_owned_hand_annotations",
        "status": STATUS,
        "claim": CLAIM,
        "baked_hand_rows": baked,
        "kept_prior_hand_rows": skipped_without_variable,
    }
    write_json(annotations_out, payload)
    report = {
        "method": "build_v17_interior_owned_hand_annotations",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "annotations": str(annotations_path),
            "interior_owned_graph": str(interior_path),
            "pose_full_residual_graph": str(pose_graph_path),
            "hand_depth_repair_graph": str(repair_path),
        },
        "annotations_output": str(annotations_out),
        "frame_count": len(frames),
        "total_hand_rows": len(hands),
        "baked_hand_rows": baked,
        "baked_interior_compatible_rows": baked_interior_compatible,
        "kept_prior_hand_rows": skipped_without_variable,
        "expected_interior_variable_rows": require_int(
            interior.get("interior_owned_variable_rows"),
            "interior variable rows",
        ),
        "wrist_to_middle_tip_m": summarize(span_values),
        "rows": rows_audit,
        **FALSE_READY,
    }
    if baked != require_int(interior.get("interior_owned_variable_rows"), "interior variable rows"):
        raise RuntimeError(
            f"{case} baked {baked} hands but interior graph has "
            f"{interior.get('interior_owned_variable_rows')} variables"
        )
    write_json(output_dir / "v17_interior_owned_hand_annotations_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    patch_legacy_mano_loader()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
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
        ).to(device)
    reports = [case_problem(case, model, args, device) for case in args.cases]
    summary = {
        "method": "build_v17_interior_owned_hand_annotations",
        "status": STATUS,
        "claim": CLAIM,
        "device": str(device),
        "case_count": len(reports),
        "baked_hand_rows": sum(require_int(r.get("baked_hand_rows"), "baked rows") for r in reports),
        "baked_interior_compatible_rows": sum(
            require_int(r.get("baked_interior_compatible_rows"), "baked compatible rows") for r in reports
        ),
        "kept_prior_hand_rows": sum(
            require_int(r.get("kept_prior_hand_rows"), "kept rows") for r in reports
        ),
        "cases": [
            {
                "case": require_str(r.get("case"), "case"),
                "frame_count": require_int(r.get("frame_count"), "frame_count"),
                "annotations_output": require_str(r.get("annotations_output"), "annotations output"),
                "baked_hand_rows": r["baked_hand_rows"],
                "kept_prior_hand_rows": r["kept_prior_hand_rows"],
                "wrist_to_middle_tip_m": r["wrist_to_middle_tip_m"],
                **FALSE_READY,
            }
            for r in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_interior_owned_hand_annotations_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--interior-owned-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"),
    )
    parser.add_argument(
        "--pose-full-residual-graph-root",
        type=Path,
        default=Path(
            "/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose"
        ),
    )
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_interior_owned_hand_annotations"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
