#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Probe whether local MANO assets reproduce saved HaWoR left-hand surfaces.

This is a mechanism/provenance test for V18 left-hand eligibility. A left-hand
optimizer is only physically valid if the MANO model and side convention replay
saved HaWoR vertices/joints in the same metric world frame with near-zero error.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    as_list,
    load_json,
    rotvec_to_matrix,
    source_npz_for_hand,
    write_json,
)

DEFAULT_CASE_ANNOTATIONS = {
    "task5_tomato_960": Path(
        "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/task5_tomato_960/annotations_v18_full.json"
    ),
    "trash_1050": Path(
        "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/trash_1050/annotations_v18_full.json"
    ),
}
DEFAULT_RIGHT_MODEL = Path("third_party/WiLoR/mano_data/MANO_RIGHT.pkl")
DEFAULT_LEFT_MODEL = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl")
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_mano_left_replay_probe_v1/v18_mano_left_replay_conventions_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task5-annotations", type=Path, default=DEFAULT_CASE_ANNOTATIONS["task5_tomato_960"])
    parser.add_argument("--trash-annotations", type=Path, default=DEFAULT_CASE_ANNOTATIONS["trash_1050"])
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--mano-left", type=Path, default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames-per-case-side", type=int, default=0, help="0 means all available frames")
    parser.add_argument("--near-zero-median-m", type=float, default=1.0e-5)
    parser.add_argument("--near-zero-p95-m", type=float, default=5.0e-5)
    return parser.parse_args()


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
        if not hasattr(np, name):
            setattr(np, name, value)


def load_wilor_mano_class(wilor_root: Path):
    path = wilor_root / "wilor" / "models" / "mano_wrapper.py"
    if not path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO wrapper: {path}")
    spec = importlib.util.spec_from_file_location("wilor_mano_wrapper_v18_left_replay_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WiLoR MANO wrapper spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO


def load_source_arrays(cache: dict[Path, dict[str, np.ndarray]], path: Path) -> dict[str, np.ndarray]:
    if path not in cache:
        with np.load(path, allow_pickle=True) as z:
            cache[path] = {key: np.asarray(z[key]) for key in z.files}
    return cache[path]


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def collect_rows(case: str, annotation_path: Path, source_cache: dict[Path, dict[str, np.ndarray]], max_rows: int) -> dict[str, list[dict[str, Any]]]:
    payload = load_json(annotation_path)
    out: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for frame in as_list(payload.get("frames")):
        frame_idx = int(frame.get("frame_idx", len(out["left"])))
        for hand in as_list(frame.get("hands")):
            side = str(hand.get("hand_side", "")).lower()
            if side not in out:
                continue
            if max_rows and len(out[side]) >= max_rows:
                continue
            source_info = source_npz_for_hand(hand)
            if source_info is None:
                continue
            source_path, source_frame = source_info
            source = load_source_arrays(source_cache, source_path)
            required = [
                f"{side}_vertices_world_m",
                f"{side}_joints_world_m",
                f"{side}_root_orient_axis_angle",
                f"{side}_hand_pose_axis_angle",
                f"{side}_betas",
                f"{side}_trans_world_m",
            ]
            if any(key not in source for key in required):
                continue
            row = {
                "case": case,
                "frame_idx": frame_idx,
                "hand_side": side,
                "source_hawor_npz": str(source_path),
                "source_frame_index": int(source_frame),
                "root_orient_axis_angle": np.asarray(source[f"{side}_root_orient_axis_angle"][source_frame], dtype=np.float32),
                "hand_pose_axis_angle": np.asarray(source[f"{side}_hand_pose_axis_angle"][source_frame], dtype=np.float32),
                "betas": np.asarray(source[f"{side}_betas"][source_frame], dtype=np.float32),
                "trans_world_m": np.asarray(source[f"{side}_trans_world_m"][source_frame], dtype=np.float32),
                "vertices_world_m": np.asarray(source[f"{side}_vertices_world_m"][source_frame], dtype=np.float32),
                "joints_world_m": np.asarray(source[f"{side}_joints_world_m"][source_frame], dtype=np.float32),
            }
            out[side].append(row)
    return out


def evaluate_convention(model: Any, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "status": "no_rows"}
    root = torch.tensor(np.stack([r["root_orient_axis_angle"] for r in rows]).reshape(len(rows), 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r["hand_pose_axis_angle"] for r in rows]).reshape(len(rows), 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r["betas"] for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r["trans_world_m"] for r in rows]), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(
            global_orient=rotvec_to_matrix(root),
            hand_pose=rotvec_to_matrix(pose),
            betas=betas,
            transl=trans,
            return_verts=True,
            pose2rot=False,
        )
    pred_vertices = out.vertices.detach().cpu().numpy().astype(np.float64)
    pred_joints = out.joints.detach().cpu().numpy().astype(np.float64)
    target_vertices = np.stack([r["vertices_world_m"] for r in rows]).astype(np.float64)
    target_joints = np.stack([r["joints_world_m"] for r in rows]).astype(np.float64)
    vertex_err = np.linalg.norm(pred_vertices - target_vertices, axis=2)
    joint_err = np.linalg.norm(pred_joints - target_joints, axis=2)
    frame_vertex_median = np.median(vertex_err, axis=1)
    frame_vertex_p95 = np.percentile(vertex_err, 95, axis=1)
    frame_vertex_max = np.max(vertex_err, axis=1)
    frame_joint_median = np.median(joint_err, axis=1)
    frame_joint_p95 = np.percentile(joint_err, 95, axis=1)
    frame_joint_max = np.max(joint_err, axis=1)
    worst_order = np.argsort(frame_vertex_median + frame_joint_median)[::-1]
    worst_frames = []
    for idx in worst_order[:10]:
        row = rows[int(idx)]
        worst_frames.append(
            {
                "case": row["case"],
                "frame_idx": int(row["frame_idx"]),
                "hand_side": row["hand_side"],
                "source_frame_index": int(row["source_frame_index"]),
                "vertex_median_m": float(frame_vertex_median[idx]),
                "vertex_p95_m": float(frame_vertex_p95[idx]),
                "vertex_max_m": float(frame_vertex_max[idx]),
                "joint_median_m": float(frame_joint_median[idx]),
                "joint_p95_m": float(frame_joint_p95[idx]),
                "joint_max_m": float(frame_joint_max[idx]),
            }
        )
    return {
        "row_count": int(len(rows)),
        "vertex_error_m": {
            "all_vertices": numeric_summary(vertex_err.reshape(-1)),
            "frame_median": numeric_summary(frame_vertex_median),
            "frame_p95": numeric_summary(frame_vertex_p95),
            "frame_max": numeric_summary(frame_vertex_max),
        },
        "joint_error_m": {
            "all_joints": numeric_summary(joint_err.reshape(-1)),
            "frame_median": numeric_summary(frame_joint_median),
            "frame_p95": numeric_summary(frame_joint_p95),
            "frame_max": numeric_summary(frame_joint_max),
        },
        "worst_frames_by_median_error": worst_frames,
    }


def main() -> None:
    args = parse_args()
    patch_legacy_mano_loader()
    annotation_paths = {
        "task5_tomato_960": args.task5_annotations,
        "trash_1050": args.trash_annotations,
    }
    for case, path in annotation_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing annotations for {case}: {path}")
    if not args.mano_right.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT: {args.mano_right}")
    if not args.mano_left.exists():
        raise FileNotFoundError(f"missing MANO_LEFT: {args.mano_left}")
    source_cache: dict[Path, dict[str, np.ndarray]] = {}
    rows_by_case_side: dict[str, dict[str, list[dict[str, Any]]]] = {}
    merged_by_side: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    max_rows = int(args.max_frames_per_case_side)
    for case, path in annotation_paths.items():
        rows = collect_rows(case, path, source_cache, max_rows=max_rows)
        rows_by_case_side[case] = rows
        for side in ("left", "right"):
            merged_by_side[side].extend(rows[side])
    mano_cls = load_wilor_mano_class(args.wilor_root)
    device = torch.device(args.device)
    conventions = {
        "MANO_RIGHT_pkl_is_rhand_true": (args.mano_right, True, False),
        "MANO_RIGHT_pkl_is_rhand_false": (args.mano_right, False, False),
        "MANO_LEFT_pkl_is_rhand_false": (args.mano_left, False, False),
        "MANO_LEFT_pkl_is_rhand_true": (args.mano_left, True, False),
        "MANO_LEFT_pkl_is_rhand_false_hawor_shapedirs_x_fix": (args.mano_left, False, True),
    }
    results: dict[str, Any] = {}
    for name, (model_path, is_rhand, hawor_left_shapedirs_x_fix) in conventions.items():
        model = mano_cls(model_path=str(model_path), is_rhand=is_rhand, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        if hawor_left_shapedirs_x_fix:
            with torch.no_grad():
                model.shapedirs[:, 0, :] *= -1
        model.eval()
        side_results = {}
        for side in ("left", "right"):
            side_results[side] = evaluate_convention(model, merged_by_side[side], device)
        results[name] = {
            "model_path": str(model_path),
            "is_rhand": bool(is_rhand),
            "hawor_left_shapedirs_x_fix": bool(hawor_left_shapedirs_x_fix),
            "sides": side_results,
        }
    best_by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        ranked = []
        for name, result in results.items():
            side_result = result["sides"][side]
            frame_median = side_result.get("vertex_error_m", {}).get("frame_median", {}).get("max", float("inf"))
            joint_median = side_result.get("joint_error_m", {}).get("frame_median", {}).get("max", float("inf"))
            frame_p95 = side_result.get("vertex_error_m", {}).get("frame_p95", {}).get("max", float("inf"))
            ranked.append((float(frame_median), float(joint_median), float(frame_p95), name))
        ranked.sort()
        best_name = ranked[0][3]
        best = results[best_name]["sides"][side]
        best_by_side[side] = {
            "best_convention": best_name,
            "ranking_by_max_frame_vertex_median_then_joint_median": [
                {
                    "convention": name,
                    "max_frame_vertex_median_m": vm,
                    "max_frame_joint_median_m": jm,
                    "max_frame_vertex_p95_m": vp95,
                }
                for vm, jm, vp95, name in ranked
            ],
            "near_zero_replay": bool(
                best.get("vertex_error_m", {}).get("frame_median", {}).get("max", float("inf")) <= float(args.near_zero_median_m)
                and best.get("joint_error_m", {}).get("frame_median", {}).get("max", float("inf")) <= float(args.near_zero_median_m)
                and best.get("vertex_error_m", {}).get("frame_p95", {}).get("max", float("inf")) <= float(args.near_zero_p95_m)
                and best.get("joint_error_m", {}).get("frame_p95", {}).get("max", float("inf")) <= float(args.near_zero_p95_m)
            ),
        }
    report = {
        "method": "probe_v18_mano_left_replay_conventions",
        "status": "ok",
        "claim_scope": (
            "Tests whether local MANO model assets and side conventions reproduce saved HaWoR raw vertices/joints. "
            "A left-hand optimizer is eligible only if left replay is near-zero; plausible mirrored geometry is rejected."
        ),
        "inputs": {
            "annotations": {case: str(path) for case, path in annotation_paths.items()},
            "mano_right": str(args.mano_right),
            "mano_left": str(args.mano_left),
            "wilor_root": str(args.wilor_root),
            "max_frames_per_case_side": int(args.max_frames_per_case_side),
            "near_zero_median_m": float(args.near_zero_median_m),
            "near_zero_p95_m": float(args.near_zero_p95_m),
        },
        "case_side_row_counts": {
            case: {side: int(len(rows_by_case_side[case][side])) for side in ("left", "right")}
            for case in rows_by_case_side
        },
        "source_npz_paths": sorted(str(path) for path in source_cache),
        "best_by_side": best_by_side,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "best_by_side": best_by_side, "case_side_row_counts": report["case_side_row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
