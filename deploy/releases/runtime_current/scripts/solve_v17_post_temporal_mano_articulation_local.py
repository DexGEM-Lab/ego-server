#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

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
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary
from solve_v17_mano_articulation_local import (
    bool_count,
    finite_number,
    load_wilor_mano_class,
    numeric_summary,
    patch_legacy_mano_loader,
    solve_row,
    state_counts,
)


STATUS = "v17_post_temporal_mano_articulation_local_solve_qc"
CLAIM = (
    "This artifact tests local MANO articulation after the owner-weighted temporal hand-depth refit. "
    "It consumes current post-temporal MANO correspondence factors, keeps the owner-weighted total "
    "camera-ray hand shift fixed, optimizes only per-row MANO pose deltas, and reports whether those "
    "deltas reduce the remaining local or mixed surface residual. It does not write corrected "
    "annotations and does not complete the V3 joint solver."
)


def close_enough(a: Any, b: Any, label: str) -> None:
    left = finite_number(a, f"{label} left")
    right = finite_number(b, f"{label} right")
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"{label} mismatch: {left} vs {right}")


def case_problem(case: str, model: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
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
            f"{case} hand depth repair graph",
        ),
        "hand_temporal_owner_weighted_refit": existing_path(
            args.hand_temporal_owner_weighted_refit_root
            / case
            / "v17_hand_temporal_owner_weighted_refit.json",
            f"{case} hand temporal owner-weighted refit",
        ),
        "post_temporal_mano_factor_input": existing_path(
            args.post_temporal_mano_factor_input_root
            / case
            / "v17_post_temporal_mano_factor_input.json",
            f"{case} post-temporal MANO factor input",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hands = annotation_hand_index(frames)
    frame_count = len(frames)
    for name in [
        "visible_surface",
        "hand_depth_repair_graph",
        "hand_temporal_owner_weighted_refit",
        "post_temporal_mano_factor_input",
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame count disagrees with {name}")
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth npz",
    )
    depth = depth_archive(depth_path)
    graph_report = payloads["hand_depth_repair_graph"]
    owner_weighted_report = payloads["hand_temporal_owner_weighted_refit"]
    factor_report = payloads["post_temporal_mano_factor_input"]
    graph_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "graph id"): row
        for row in [
            require_dict(raw, "graph row")
            for raw in require_list(graph_report.get("rows"), f"{case} graph rows")
        ]
    }
    owner_weighted_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "owner-weighted graph id"): row
        for row in [
            require_dict(raw, "owner-weighted row")
            for raw in require_list(owner_weighted_report.get("rows"), f"{case} owner-weighted rows")
        ]
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(factor_report.get("rows"), f"{case} post-temporal factor input rows"):
        factor_row = require_dict(raw, "post-temporal factor input row")
        if factor_row.get("post_temporal_mano_factor_input_materialized") is not True:
            raise RuntimeError(f"{case} post-temporal MANO solve received an unmaterialized factor row")
        graph_id = require_str(
            factor_row.get("source_hand_depth_repair_graph_variable_id"),
            "factor source graph id",
        )
        graph_row = require_dict(graph_by_id.get(graph_id), f"{case} graph row {graph_id}")
        owner_weighted_row = require_dict(
            owner_weighted_by_id.get(graph_id),
            f"{case} owner-weighted row {graph_id}",
        )
        if factor_row.get("source_owner_weighted_reprojection_state") != owner_weighted_row.get(
            "owner_weighted_reprojection_state"
        ):
            raise RuntimeError(f"{case} owner-weighted state disagrees for {graph_id}")
        close_enough(
            factor_row.get("owner_weighted_total_hand_ray_shift_m"),
            owner_weighted_row.get("owner_weighted_total_hand_ray_shift_m"),
            f"{case} owner-weighted total shift {graph_id}",
        )
        frame_idx = require_int(factor_row.get("frame_idx"), "factor frame_idx")
        side = require_str(factor_row.get("hand_side"), "factor hand_side")
        hand_index = require_int(factor_row.get("hand_index"), "factor hand_index")
        hand = require_dict(hands.get((frame_idx, side, hand_index)), f"{case} hand {graph_id}")
        shifted_graph_row = {
            **graph_row,
            "hand_ray_shift_m": factor_row.get("owner_weighted_total_hand_ray_shift_m"),
        }
        fit = solve_row(
            model=model,
            hand=hand,
            graph_row=shifted_graph_row,
            factor_row=factor_row,
            depth=depth,
            args=args,
            device=device,
        )
        rows.append(
            {
                "case": case,
                "post_temporal_mano_articulation_local_solve_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "post_temporal_mano_articulation_local_solve:",
                    1,
                ),
                "source_post_temporal_mano_factor_input_id": factor_row.get(
                    "post_temporal_mano_factor_input_id"
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_temporal_owner_weighted_refit_variable_id": factor_row.get(
                    "source_hand_temporal_owner_weighted_refit_variable_id"
                ),
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_index": hand_index,
                "source_owner_weighted_reprojection_state": factor_row.get(
                    "source_owner_weighted_reprojection_state"
                ),
                "post_temporal_mano_local_surface_factor_row": bool(
                    factor_row.get("post_temporal_mano_local_surface_factor_row") is True
                ),
                "post_temporal_mano_mixed_surface_depth_factor_row": bool(
                    factor_row.get("post_temporal_mano_mixed_surface_depth_factor_row") is True
                ),
                "hand_depth_repair_graph_shift_m": graph_row.get("hand_ray_shift_m"),
                "owner_weighted_delta_shift_m": factor_row.get("owner_weighted_delta_shift_m"),
                "owner_weighted_total_hand_ray_shift_m": factor_row.get(
                    "owner_weighted_total_hand_ray_shift_m"
                ),
                "post_temporal_mano_articulation_solve_state": fit["local_articulation_solve_state"],
                **fit,
                **FALSE_READY,
            }
        )
    source_candidate_rows = require_int(
        factor_report.get("post_temporal_mano_factor_input_candidate_rows"),
        f"{case} post-temporal candidate rows",
    )
    if len(rows) != source_candidate_rows:
        raise RuntimeError(f"{case} post-temporal MANO solve row count disagrees with factor input")
    solved_rows = [row for row in rows if row.get("local_articulation_depth_improved") is True]
    threshold_rows = [row for row in rows if row.get("local_articulation_depth_threshold_met") is True]
    trusted_rows = [row for row in rows if row.get("local_articulation_projection_trusted") is True]
    clamp_hit_rows = [
        row
        for row in rows
        if float(row.get("pose_delta_abs_max_rad", 0.0)) >= float(args.max_pose_delta_rad) - 1.0e-5
    ]
    report = {
        "method": "solve_v17_post_temporal_mano_articulation_local",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "post_temporal_mano_articulation_solve_candidate_rows": len(rows),
        "post_temporal_mano_local_surface_solve_rows": bool_count(
            rows,
            "post_temporal_mano_local_surface_factor_row",
        ),
        "post_temporal_mano_mixed_surface_depth_solve_rows": bool_count(
            rows,
            "post_temporal_mano_mixed_surface_depth_factor_row",
        ),
        "post_temporal_mano_articulation_depth_improved_rows": len(solved_rows),
        "post_temporal_mano_articulation_depth_threshold_met_rows": len(threshold_rows),
        "post_temporal_mano_articulation_projection_trusted_rows": len(trusted_rows),
        "post_temporal_mano_articulation_pose_delta_clamp_hit_rows": len(clamp_hit_rows),
        "post_temporal_mano_articulation_solve_state_counts": state_counts(
            rows,
            "post_temporal_mano_articulation_solve_state",
        ),
        "source_owner_weighted_reprojection_state_counts": state_counts(
            rows,
            "source_owner_weighted_reprojection_state",
        ),
        "before_depth_abs_median_m": numeric_summary(rows, "before.depth_abs_median_m"),
        "after_depth_abs_median_m": numeric_summary(rows, "after.depth_abs_median_m"),
        "depth_abs_median_improvement_m": numeric_summary(rows, "depth_abs_median_improvement_m"),
        "after_joint_reprojection_median_px": numeric_summary(rows, "after.joint_reprojection_median_px"),
        "after_joint_reprojection_p95_px": numeric_summary(rows, "after.joint_reprojection_p95_px"),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        "source_post_temporal_mano_factor_input_comparison": {
            "post_temporal_mano_factor_input_candidate_rows": factor_report.get(
                "post_temporal_mano_factor_input_candidate_rows"
            ),
            "post_temporal_mano_factor_input_materialized_rows": factor_report.get(
                "post_temporal_mano_factor_input_materialized_rows"
            ),
            "post_temporal_mano_local_surface_factor_rows": factor_report.get(
                "post_temporal_mano_local_surface_factor_rows"
            ),
            "post_temporal_mano_mixed_surface_depth_factor_rows": factor_report.get(
                "post_temporal_mano_mixed_surface_depth_factor_rows"
            ),
            "assigned_factor_sample_count": factor_report.get("assigned_factor_sample_count"),
        },
        "problem_semantics": {
            "optimized_variables": "per-row MANO hand_pose rotation deltas only",
            "fixed_state": "saved MANO shape, global orientation, owner-weighted total hand ray shift, graph scale, and source-camera translation",
            "local_factor": "post-temporal residual surface vertices are pulled toward compatible-depth seed pixels from the same current hand surface",
            "solver_scope": "local diagnostic solve; no full-timeline temporal coupling, no depth-observation variables, and no annotation update",
        },
        "parameters": {
            "device": str(device),
            "iters": int(args.iters),
            "lr": float(args.lr),
            "max_pairs_per_row": int(args.max_pairs_per_row),
            "sigma_depth_m": float(args.sigma_depth_m),
            "sigma_projection_px": float(args.sigma_projection_px),
            "sigma_joint_px": float(args.sigma_joint_px),
            "sigma_pose_delta_rad": float(args.sigma_pose_delta_rad),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "accept_depth_median_m": float(args.accept_depth_median_m),
            "accept_depth_p95_m": float(args.accept_depth_p95_m),
            "accept_joint_median_px": float(args.accept_joint_median_px),
            "accept_joint_p95_px": float(args.accept_joint_p95_px),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_post_temporal_mano_articulation_local_solve.json", report)
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
    summary_path = existing_path(
        args.post_temporal_mano_factor_input_root / "v17_post_temporal_mano_factor_input_summary.json",
        "post-temporal MANO factor input summary",
    )
    factor_summary = require_dict(load_json(summary_path), "post-temporal MANO factor input summary")
    reports = [
        case_problem(require_str(raw, "summary case"), model, args, device)
        for raw in require_list(factor_summary.get("cases"), "summary cases")
    ]
    rows = [
        require_dict(row, "post-temporal MANO articulation row")
        for report in reports
        for row in require_list(report.get("rows"), "post-temporal MANO articulation rows")
    ]
    payload = {
        "method": "solve_v17_post_temporal_mano_articulation_local",
        "status": STATUS,
        "claim": CLAIM,
        "source_post_temporal_mano_factor_input_summary": str(summary_path),
        "wilor_root": str(args.wilor_root),
        "wilor_mano_right": str(mano_model_path),
        "device": str(device),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_post_temporal_mano_articulation_local_solve.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "post_temporal_mano_articulation_solve_candidate_rows": require_int(
                    report.get("post_temporal_mano_articulation_solve_candidate_rows"),
                    "candidate rows",
                ),
                "post_temporal_mano_articulation_depth_improved_rows": require_int(
                    report.get("post_temporal_mano_articulation_depth_improved_rows"),
                    "improved rows",
                ),
                "post_temporal_mano_articulation_depth_threshold_met_rows": require_int(
                    report.get("post_temporal_mano_articulation_depth_threshold_met_rows"),
                    "threshold rows",
                ),
                "post_temporal_mano_articulation_projection_trusted_rows": require_int(
                    report.get("post_temporal_mano_articulation_projection_trusted_rows"),
                    "projection trusted rows",
                ),
                "post_temporal_mano_articulation_pose_delta_clamp_hit_rows": require_int(
                    report.get("post_temporal_mano_articulation_pose_delta_clamp_hit_rows"),
                    "pose clamp hit rows",
                ),
                "post_temporal_mano_articulation_solve_state_counts": require_dict(
                    report.get("post_temporal_mano_articulation_solve_state_counts"),
                    "state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "post_temporal_mano_articulation_solve_candidate_rows": len(rows),
        "post_temporal_mano_local_surface_solve_rows": bool_count(
            rows,
            "post_temporal_mano_local_surface_factor_row",
        ),
        "post_temporal_mano_mixed_surface_depth_solve_rows": bool_count(
            rows,
            "post_temporal_mano_mixed_surface_depth_factor_row",
        ),
        "post_temporal_mano_articulation_depth_improved_rows": bool_count(
            rows,
            "local_articulation_depth_improved",
        ),
        "post_temporal_mano_articulation_depth_threshold_met_rows": bool_count(
            rows,
            "local_articulation_depth_threshold_met",
        ),
        "post_temporal_mano_articulation_projection_trusted_rows": bool_count(
            rows,
            "local_articulation_projection_trusted",
        ),
        "post_temporal_mano_articulation_pose_delta_clamp_hit_rows": sum(
            1
            for row in rows
            if float(row.get("pose_delta_abs_max_rad", 0.0)) >= float(args.max_pose_delta_rad) - 1.0e-5
        ),
        "post_temporal_mano_articulation_solve_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_mano_articulation_solve_state_counts"),
                                "state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "source_owner_weighted_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("source_owner_weighted_reprojection_state_counts"),
                                "source state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "before_depth_abs_median_m": numeric_summary(rows, "before.depth_abs_median_m"),
        "after_depth_abs_median_m": numeric_summary(rows, "after.depth_abs_median_m"),
        "depth_abs_median_improvement_m": numeric_summary(rows, "depth_abs_median_improvement_m"),
        "after_joint_reprojection_median_px": numeric_summary(rows, "after.joint_reprojection_median_px"),
        "after_joint_reprojection_p95_px": numeric_summary(rows, "after.joint_reprojection_p95_px"),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_post_temporal_mano_articulation_local_solve_summary.json", payload)
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
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--post-temporal-mano-factor-input-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_mano_articulation_local_solve"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max-pairs-per-row", type=int, default=96)
    parser.add_argument("--sigma-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-projection-px", type=float, default=6.0)
    parser.add_argument("--sigma-joint-px", type=float, default=18.0)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.18)
    parser.add_argument("--sigma-span-m", type=float, default=0.02)
    parser.add_argument("--w-depth", type=float, default=2.0)
    parser.add_argument("--w-projection", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.6)
    parser.add_argument("--w-pose", type=float, default=0.25)
    parser.add_argument("--w-span", type=float, default=0.25)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.35)
    parser.add_argument("--min-span-m", type=float, default=0.10)
    parser.add_argument("--max-span-m", type=float, default=0.22)
    parser.add_argument("--accept-depth-median-m", type=float, default=0.030)
    parser.add_argument("--accept-depth-p95-m", type=float, default=0.080)
    parser.add_argument("--accept-joint-median-px", type=float, default=45.0)
    parser.add_argument("--accept-joint-p95-px", type=float, default=95.0)
    parser.add_argument("--min-depth-median-improvement-m", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
