#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, load_json, numeric_summary, project_points, write_json


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(axis_angle.astype(np.float64).reshape(3, 1))
    return matrix.astype(np.float64)


def fail_if_oracle_state(payload: Any) -> None:
    hits: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                text = str(key).lower()
                if "oracle" in text or text.startswith("gt") or "ground_truth" in text:
                    hits.append(f"{path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")
        elif isinstance(value, str):
            text = value.lower()
            if "oracle" in text or "state copied from gt" in text or "gt-driven" in text:
                hits.append(path)

    visit(payload, "state")
    if hits:
        raise ContractError(f"v20_gt_evaluation_contract_failed: prediction_state_contains_gt_or_oracle_markers {hits[:20]}")


def frame_list_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    annotations_path = None
    render_inputs = state.get("render_inputs") if isinstance(state.get("render_inputs"), dict) else {}
    if isinstance(render_inputs.get("annotations"), str):
        annotations_path = Path(render_inputs["annotations"])
    if annotations_path is None and isinstance(state.get("annotations_v20_renderable"), str):
        annotations_path = Path(state["annotations_v20_renderable"])
    if annotations_path is not None and annotations_path.exists():
        payload = load_json(annotations_path)
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if isinstance(frames, list):
            return [row for row in frames if isinstance(row, dict)]
    frames = state.get("frames")
    if isinstance(frames, list):
        return [row for row in frames if isinstance(row, dict)]
    hands = state.get("hands")
    objects = state.get("objects")
    selected = state.get("timeline", {}).get("selected_frames") if isinstance(state.get("timeline"), dict) else None
    if isinstance(hands, list) or isinstance(objects, list):
        count = max(len(hands) if isinstance(hands, list) else 0, len(objects) if isinstance(objects, list) else 0)
        out = []
        for i in range(count):
            frame_idx = selected[i] if isinstance(selected, list) and i < len(selected) else i
            out.append({
                "frame_idx": frame_idx,
                "hands": hands[i] if isinstance(hands, list) and i < len(hands) else [],
                "objects": objects[i] if isinstance(objects, list) and i < len(objects) else [],
            })
        return out
    raise ContractError("v20_gt_evaluation_contract_failed: no_prediction_frames_in_state_or_annotations")


def prediction_hand_joints(hand: dict[str, Any]) -> np.ndarray | None:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else hand
    for key in ("joint_3d_camera_m", "joints_3d_camera_m", "joints_current_v18_camera_m", "joints3d_camera"):
        arr = np.asarray(metric.get(key) if metric.get(key) is not None else [], dtype=float)
        if arr.shape == (21, 3) and np.isfinite(arr).all():
            return arr
    return None


def prediction_object_pose(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if "T_camera_object_3x4" in obj:
        T = np.asarray(obj["T_camera_object_3x4"], dtype=float)
        if T.shape == (3, 4) and np.isfinite(T).all():
            return T[:, :3], T[:, 3]
    if "R_camera_object" in obj and "t_camera_object_m" in obj:
        R = np.asarray(obj["R_camera_object"], dtype=float)
        t = np.asarray(obj["t_camera_object_m"], dtype=float).reshape(-1)
        if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
            return R, t
    pose = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
    R = np.asarray(pose.get("rotation_camera_from_object_matrix") or pose.get("rotation_world_from_canonical_matrix") or [], dtype=float)
    t = np.asarray(pose.get("translation_camera_m") or pose.get("translation_world_m") or [], dtype=float).reshape(-1)
    if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
        return R, t
    return None




def dexycb_gt_pose_index(obj: dict[str, Any], dataset_manifest: dict[str, Any]) -> int | None:
    object_id = str(obj.get("object_id", ""))
    label = obj.get("dataset_label_id_public_roster") or obj.get("dataset_label_id")
    rows = dataset_manifest.get("public_object_model_roster") or dataset_manifest.get("objects")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            same_object = str(row.get("object_id")) == object_id
            same_label = label is not None and row.get("dataset_label_id") is not None and int(row.get("dataset_label_id")) == int(label)
            if same_object or same_label:
                object_index = row.get("object_index")
                if object_index is not None:
                    try:
                        return int(object_index)
                    except (TypeError, ValueError):
                        return None
    return None


def load_gt_by_frame(reference_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    dataset = reference_manifest["dataset"]
    out: dict[int, dict[str, Any]] = {}
    if dataset == "dexycb":
        for row in reference_manifest.get("frames", []):
            frame_idx = int(row["frame_index"])
            ref = row["label_npz"]
            labels = np.load(ref["path"], allow_pickle=False)
            out[frame_idx] = {key: labels[key] for key in ref["keys"] if key in labels.files}
    elif dataset == "ho3d":
        for row in reference_manifest.get("frames", []):
            frame_idx = int(row["frame_index"])
            ref = row["meta_pkl"]
            with Path(ref["path"]).open("rb") as handle:
                meta = pickle.load(handle)
            out[frame_idx] = meta
    else:
        raise ContractError(f"unsupported_gt_dataset: {dataset}")
    return out


def rotation_error_rad(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_pred @ R_gt.T
    trace = float(np.trace(R))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.state)
    uncertainty = load_json(args.uncertainty_state) if args.uncertainty_state else None
    reference_manifest = load_json(args.reference_manifest)
    dataset_manifest = load_json(args.dataset_manifest)
    fail_if_oracle_state(state)
    frames = frame_list_from_state(state)
    pred_by_frame = {int(frame.get("frame_idx", frame.get("frame_index", i))): frame for i, frame in enumerate(frames)}
    gt_by_frame = load_gt_by_frame(reference_manifest)
    hand_errors = []
    object_translation_errors = []
    object_rotation_errors = []
    unsupported = []
    per_frame = []
    dataset = reference_manifest["dataset"]
    for frame_idx, gt in sorted(gt_by_frame.items()):
        pred = pred_by_frame.get(frame_idx)
        row_report: dict[str, Any] = {"frame_idx": frame_idx}
        if pred is None:
            row_report["status"] = "missing_prediction_frame"
            per_frame.append(row_report)
            continue
        pred_hands = pred.get("hands") if isinstance(pred.get("hands"), list) else []
        pred_objects = pred.get("objects") if isinstance(pred.get("objects"), list) else []
        if dataset == "dexycb":
            gt_joints = np.asarray(gt.get("joint_3d") if gt.get("joint_3d") is not None else [], dtype=float)
            if gt_joints.ndim == 3 and pred_hands:
                for i, hand in enumerate(pred_hands[: gt_joints.shape[0]]):
                    pred_joints = prediction_hand_joints(hand)
                    if pred_joints is not None:
                        err = np.linalg.norm(pred_joints - gt_joints[i], axis=1)
                        hand_errors.extend(err.astype(float).tolist())
            gt_poses = np.asarray(gt.get("pose_y") if gt.get("pose_y") is not None else [], dtype=float)
            if gt_poses.ndim == 3 and pred_objects:
                for obj in pred_objects:
                    gt_index = dexycb_gt_pose_index(obj, dataset_manifest)
                    if gt_index is None or gt_index < 0 or gt_index >= gt_poses.shape[0]:
                        continue
                    pose = prediction_object_pose(obj)
                    if pose is not None:
                        R_pred, t_pred = pose
                        T_gt = gt_poses[gt_index]
                        object_translation_errors.append(float(np.linalg.norm(t_pred - T_gt[:, 3])))
                        object_rotation_errors.append(rotation_error_rad(R_pred, T_gt[:, :3]))
        else:
            gt_joints = np.asarray(gt.get("handJoints3D") if gt.get("handJoints3D") is not None else [], dtype=float)
            if gt_joints.shape == (21, 3) and pred_hands:
                pred_joints = prediction_hand_joints(pred_hands[0])
                if pred_joints is not None:
                    err = np.linalg.norm(pred_joints - gt_joints, axis=1)
                    hand_errors.extend(err.astype(float).tolist())
            if pred_objects and gt.get("objRot") is not None and gt.get("objTrans") is not None:
                pose = prediction_object_pose(pred_objects[0])
                if pose is not None:
                    R_pred, t_pred = pose
                    R_gt = rodrigues(np.asarray(gt["objRot"], dtype=float))
                    t_gt = np.asarray(gt["objTrans"], dtype=float).reshape(3)
                    object_translation_errors.append(float(np.linalg.norm(t_pred - t_gt)))
                    object_rotation_errors.append(rotation_error_rad(R_pred, R_gt))
        per_frame.append(row_report)
    if not hand_errors:
        unsupported.append("hand_joint_camera_m_missing_or_semantics_unmatched")
    if not object_translation_errors:
        unsupported.append("object_pose_camera_missing_or_semantics_unmatched")
    metrics = {
        "schema": "v20_gt_metrics.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_without_reference_labels",
        "dataset": dataset,
        "sample_id": reference_manifest.get("sample_id"),
        "frame_count_gt": len(gt_by_frame),
        "frame_count_prediction": len(pred_by_frame),
        "evaluated_physical_variable_families": {
            "hand_joint_camera_m": bool(hand_errors),
            "object_pose_camera": bool(object_translation_errors),
        },
        "unsupported_physical_variable_families": unsupported,
        "hand_joint_camera_m_error": numeric_summary(hand_errors),
        "object_translation_camera_m_error": numeric_summary(object_translation_errors),
        "object_rotation_camera_rad_error": numeric_summary(object_rotation_errors),
        "evaluation_reference_policy": "Reference labels consumed only by this evaluator after prediction state exists; evaluator did not edit prediction state.",
    }
    alignment = {
        "schema": "v20_gt_alignment.v0",
        "dataset": dataset,
        "prediction_state": str(args.state),
        "dataset_manifest": str(args.dataset_manifest),
        "reference_manifest": str(args.reference_manifest),
        "coordinate_convention": dataset_manifest.get("coordinate_convention"),
        "hand_joint_alignment": "camera-frame if prediction state exposes camera-frame joints; otherwise unsupported",
        "object_pose_alignment": "camera-frame object pose matched by prediction object_id/dataset_label_id to reference pose slot; public roster order is not treated as a prediction target list",
        "world_alignment": "not evaluated unless prediction and dataset share a documented world frame",
    }
    failures = {
        "schema": "v20_failure_clusters.v0",
        "clusters": [],
        "unsupported": unsupported,
    }
    if "hand_joint_camera_m_missing_or_semantics_unmatched" in unsupported:
        failures["clusters"].append({"mechanism": "hand_state_or_state_adapter_missing", "evidence": "no comparable camera-frame hand joints in prediction state"})
    if "object_pose_camera_missing_or_semantics_unmatched" in unsupported:
        failures["clusters"].append({"mechanism": "object_pose_or_state_adapter_missing", "evidence": "no comparable camera-frame object pose in prediction state"})
    report_lines = [
        "# V20 Benchmark GT Evaluation",
        "",
        "Observation: evaluator consumed GT only after loading a prediction state and rejecting oracle/GT markers in that state.",
        f"Hand joint samples: {metrics['hand_joint_camera_m_error']['count']}",
        f"Object pose samples: {metrics['object_translation_camera_m_error']['count']}",
        "GT did not modify prediction state, render inputs, candidates, or observation bundle.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gt_metrics.json", metrics)
    write_json(args.output_dir / "gt_alignment.json", alignment)
    write_json(args.output_dir / "failure_clusters.json", failures)
    (args.output_dir / "evaluation_agent_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"metrics": metrics, "alignment": alignment, "failure_clusters": failures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a completed V20 benchmark prediction against reference labels.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--uncertainty-state", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = evaluate(parse_args())
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
