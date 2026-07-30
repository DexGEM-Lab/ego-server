#!/usr/bin/env python3
"""Run D11 evaluator/readiness stage over frozen V22 prediction artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRIC_IDS = [
    "head_camera_ate_translation_m",
    "head_camera_rpe_translation_m",
    "head_camera_rotation_deg",
    "head_camera_scale_error_ratio",
    "hand_wrist_root_error_m",
    "hand_all_joint_mpjpe_m",
    "hand_root_relative_mpjpe_m",
    "hand_mpvpe_surface_m",
    "hand_reprojection_error_px",
    "visibility_state_accuracy",
    "temporal_wrist_jitter_m_per_frame",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "frames", "head_camera", "hand_states"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise RuntimeError(f"cannot load row list from {path}")


def vector3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def norm3(vec: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def translation(row: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("t_world_camera_m", "translation_m", "position_m", "wrist_t_world_m", "wrist_t_camera_m", "root_t_camera_m"):
        val = vector3(row.get(key))
        if val is not None:
            return val
    return None


def quat(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = row.get("q_world_camera_xyzw") or row.get("q_xyzw") or row.get("quaternion_xyzw")
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z"), value.get("w")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        q = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    n = math.sqrt(sum(x * x for x in q))
    if n <= 0 or not math.isfinite(n):
        return None
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_angle_deg(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def points(value: Any) -> list[tuple[float, float, float]] | None:
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        v = vector3(item)
        if v is None:
            return None
        out.append(v)
    return out if out else None


def mean_point_error(pred: list[tuple[float, float, float]], gt: list[tuple[float, float, float]]) -> float | None:
    n = min(len(pred), len(gt))
    if n <= 0:
        return None
    return sum(norm3(sub3(pred[i], gt[i])) for i in range(n)) / n


def root_relative(vals: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    root = vals[0]
    return [sub3(v, root) for v in vals]


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {"count": int(arr.size), "p50": float(np.median(arr)), "p95": float(np.percentile(arr, 95)), "rmse": float(math.sqrt(float(np.mean(arr * arr))))}


PREDICTION_DIAGNOSTIC_METRICS = {"hand_reprojection_error_px", "temporal_wrist_jitter_m_per_frame", "temporal_root_rotation_jitter_deg_per_frame"}


def metric_rows(observations: dict[str, list[float]], diagnostic_observations: dict[str, list[float]] | None = None) -> list[dict[str, Any]]:
    diagnostic_observations = diagnostic_observations or {}
    rows = []
    for metric_id in METRIC_IDS:
        vals = observations.get(metric_id, [])
        diag_vals = diagnostic_observations.get(metric_id, [])
        if vals:
            rows.append({"metric_id": metric_id, "status": "measured", "summary": summarize(vals), "measurement_role": "gt_backed_evaluator_metric"})
        elif diag_vals:
            rows.append({"metric_id": metric_id, "status": "prediction_diagnostic", "summary": summarize(diag_vals), "measurement_role": "prediction_only_diagnostic_not_gt_accuracy"})
        else:
            rows.append({"metric_id": metric_id, "status": "unmeasured", "summary": summarize([])})
    return rows


def read_camera_rows(run_root: Path) -> list[dict[str, Any]]:
    dense = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.json"
    if not dense.exists():
        return []
    payload = load_json(dense)
    rows = []
    for row in payload.get("frames") if isinstance(payload, dict) and isinstance(payload.get("frames"), list) else []:
        if not isinstance(row, dict):
            continue
        pose = row.get("pose_world_camera_xyzw")
        rows.append({"frame_idx": int(row.get("frame_idx", len(rows))), "t_world_camera_m": pose[:3] if isinstance(pose, list) and len(pose) >= 7 else None, "q_world_camera_xyzw": pose[3:7] if isinstance(pose, list) and len(pose) >= 7 else None})
    return rows


def read_hand_rows(run_root: Path) -> list[dict[str, Any]]:
    npz = run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz"
    if not npz.exists():
        return []
    blob = np.load(npz, allow_pickle=True)
    if "frame_idx" not in blob.files:
        return []
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    rows = []
    for pos, idx in enumerate(frame_idx.tolist()):
        for side in ("left", "right"):
            joints = np.asarray(blob[f"{side}_joints_world_m"])[pos].astype(float).tolist() if f"{side}_joints_world_m" in blob.files else None
            row = {"frame_idx": int(idx), "side": side, "joints_m": joints, "wrist_t_world_m": joints[0] if isinstance(joints, list) and joints else None}
            med_key = f"{side}_wilor_fit_reprojection_median_px"
            if med_key in blob.files:
                val = float(np.asarray(blob[med_key])[pos])
                if math.isfinite(val):
                    row["reprojection_error_px"] = val
            valid_key = f"{side}_valid"
            row["visibility"] = "visible" if valid_key in blob.files and int(np.asarray(blob[valid_key])[pos]) else "unresolved"
            rows.append(row)
    return rows


def by_frame(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in rows:
        try:
            out[int(row.get("frame_idx", row.get("frame")))] = row
        except (TypeError, ValueError):
            continue
    return out


def by_frame_side(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out = {}
    for row in rows:
        try:
            idx = int(row.get("frame_idx", row.get("frame")))
        except (TypeError, ValueError):
            continue
        out[(idx, str(row.get("side") or row.get("hand") or "unknown"))] = row
    return out


def evaluate_head(pred_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    pred = by_frame(pred_rows)
    gt = by_frame(gt_rows)
    frames = sorted(set(pred) & set(gt))
    out: dict[str, list[float]] = defaultdict(list)
    for idx in frames:
        pt = translation(pred[idx])
        gt_t = translation(gt[idx])
        if pt is not None and gt_t is not None:
            out["head_camera_ate_translation_m"].append(norm3(sub3(pt, gt_t)))
        pq = quat(pred[idx])
        gq = quat(gt[idx])
        if pq is not None and gq is not None:
            out["head_camera_rotation_deg"].append(quat_angle_deg(pq, gq))
    for a, b in zip(frames, frames[1:]):
        pa, pb = translation(pred[a]), translation(pred[b])
        ga, gb = translation(gt[a]), translation(gt[b])
        if pa is None or pb is None or ga is None or gb is None:
            continue
        pd = sub3(pb, pa)
        gd = sub3(gb, ga)
        out["head_camera_rpe_translation_m"].append(norm3(sub3(pd, gd)))
        gl = norm3(gd)
        if gl > 1.0e-9:
            out["head_camera_scale_error_ratio"].append(norm3(pd) / gl)
    return dict(out)


def evaluate_hands(pred_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]] | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    pred = by_frame_side(pred_rows)
    gt = by_frame_side(gt_rows or [])
    for key, pred_row in pred.items():
        reproj = pred_row.get("reprojection_error_px")
        try:
            reproj_f = float(reproj)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(reproj_f):
                out["hand_reprojection_error_px"].append(reproj_f)
        gt_row = gt.get(key)
        if gt_row is None:
            continue
        pw, gw = translation(pred_row), translation(gt_row)
        if pw is not None and gw is not None:
            out["hand_wrist_root_error_m"].append(norm3(sub3(pw, gw)))
        pj = points(pred_row.get("joints_m") or pred_row.get("joints_world_m") or pred_row.get("joints_camera_m"))
        gj = points(gt_row.get("joints_m") or gt_row.get("joints_world_m") or gt_row.get("joints_camera_m"))
        if pj is not None and gj is not None:
            err = mean_point_error(pj, gj)
            if err is not None:
                out["hand_all_joint_mpjpe_m"].append(err)
            rr = mean_point_error(root_relative(pj), root_relative(gj))
            if rr is not None:
                out["hand_root_relative_mpjpe_m"].append(rr)
        if "visibility" in pred_row and "visibility" in gt_row:
            out["visibility_state_accuracy"].append(1.0 if str(pred_row["visibility"]) == str(gt_row["visibility"]) else 0.0)
    grouped: dict[str, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
    for (idx, side), row in pred.items():
        wrist = translation(row)
        if wrist is not None:
            grouped[side].append((idx, wrist))
    for vals in grouped.values():
        vals.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(vals, vals[1:]):
            out["temporal_wrist_jitter_m_per_frame"].append(norm3(sub3(b, a)))
    return dict(out)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    output = args.output or (run_root / "evaluation" / "v22_evaluator_stage.json")
    head_pred = read_camera_rows(run_root)
    hand_pred = read_hand_rows(run_root)
    head_gt = load_rows(args.head_gt) if args.head_gt else None
    hand_gt = load_rows(args.hand_gt) if args.hand_gt else None
    observations: dict[str, list[float]] = defaultdict(list)
    diagnostic_observations: dict[str, list[float]] = defaultdict(list)
    if head_gt is not None:
        for key, vals in evaluate_head(head_pred, head_gt).items():
            observations[key].extend(vals)
    if hand_pred:
        for key, vals in evaluate_hands(hand_pred, hand_gt).items():
            target = diagnostic_observations if key in PREDICTION_DIAGNOSTIC_METRICS else observations
            target[key].extend(vals)
    rows = metric_rows(dict(observations), dict(diagnostic_observations))
    has_any_gt = head_gt is not None or hand_gt is not None
    measured = sum(1 for row in rows if row["status"] == "measured")
    diagnostic_count = sum(1 for row in rows if row["status"] == "prediction_diagnostic")
    status = "ok" if has_any_gt and measured else "no_gt_unmeasured" if not has_any_gt else "gt_available_no_metrics_measured"
    payload = {
        "schema": "v22_evaluator_stage.v0",
        "status": status,
        "method": "evaluate_frozen_v22_prediction_rows",
        "run_root": str(run_root),
        "prediction_rows": {"head_camera": len(head_pred), "hand_states": len(hand_pred)},
        "gt_inputs": {"head_gt": str(args.head_gt) if args.head_gt else None, "hand_gt": str(args.hand_gt) if args.hand_gt else None},
        "metric_observations": dict(observations),
        "diagnostic_observations": dict(diagnostic_observations),
        "validation_metrics": rows,
        "summary": {"metrics_measured": measured, "prediction_diagnostics": diagnostic_count, "metrics_total": len(rows), "gt_available": has_any_gt},
        "claim_scope": "D11 evaluator stage over frozen prediction artifacts. Without GT inputs, correctness metrics remain unmeasured but evaluator readiness is recorded.",
        "elapsed_s": float(time.time() - started),
    }
    write_json(output, payload)
    print(json.dumps({"status": status, "summary": payload["summary"], "output": str(output)}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--head-gt", type=Path, default=None)
    parser.add_argument("--hand-gt", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
