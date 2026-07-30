#!/usr/bin/env python3
"""Evaluate V19 camera trajectory consistency against HOT3D camera GT.

The V19 world frame and HOT3D world frame need not match.  This evaluator fits
one constant transform from the V19 world frame to the HOT3D world frame and
measures whether that single transform explains the whole camera trajectory.  A
large time-varying residual means camera/head-pose drift is a systematic source
of downstream object/hand misregistration; a small residual points the remaining
object/hand gap back to object pose, object shape, hand state, or contact truth.

This is evaluation-only.  It consumes GT after prediction state is frozen and
must not write prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _as_np4(m: Any) -> np.ndarray:
    arr = np.asarray(m, dtype=float)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError(f"expected finite 4x4 transform, got {arr.shape}")
    return arr


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def quat_average_wxyz(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        raise ValueError("no quaternions")
    ref = quats[0]
    A = np.zeros((4, 4), dtype=float)
    for q in quats:
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        if float(np.dot(q, ref)) < 0:
            q = -q
        A += np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, int(np.argmax(vals))]
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(list(t), dtype=float)
    return T


def transform_from_hot3d(x: dict[str, Any]) -> np.ndarray:
    return make_T(quat_wxyz_to_R(x["quaternion_wxyz"]), x["translation_xyz"])


def transform_angle_rad(T: np.ndarray) -> float:
    R = np.asarray(T[:3, :3], dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.acos(c)


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    vals.sort()
    if not vals:
        return {"count": 0}

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]

    return {
        "count": len(vals),
        "min": vals[0],
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": vals[-1],
    }


def load_render_state_annotation_path(render_state_path: Path) -> Path:
    state = json.loads(render_state_path.read_text())
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    value = inputs.get("annotations") or (state.get("annotation_backbone") or {}).get("path")
    if not value:
        raise RuntimeError(f"render state {render_state_path} lacks annotation path")
    return Path(value)


def load_v19_cameras(annotation_path: Path) -> dict[int, np.ndarray]:
    d = json.loads(annotation_path.read_text())
    cams: dict[int, np.ndarray] = {}
    for pos, fr in enumerate(d.get("frames", [])):
        if not isinstance(fr, dict):
            continue
        cam = fr.get("camera") if isinstance(fr.get("camera"), dict) else {}
        T = cam.get("T_world_camera_metric") or cam.get("T_world_camera")
        if T is None:
            continue
        cams[int(fr.get("frame_idx", pos))] = _as_np4(T)
    return cams


def load_hot3d_cameras(sidecar_path: Path, stream_id: str) -> dict[int, np.ndarray]:
    d = json.loads(sidecar_path.read_text())
    cams: dict[int, np.ndarray] = {}
    for fr in d.get("frames", []):
        if not isinstance(fr, dict):
            continue
        idx = int(fr["frame_idx"])
        j = fr.get("json") if isinstance(fr.get("json"), dict) else {}
        cam_block = (j.get("cameras.json") or {}).get(stream_id)
        if not cam_block:
            continue
        cams[idx] = transform_from_hot3d(cam_block["T_world_from_camera"])
    return cams


def load_object_residual_by_frame(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    out: dict[int, float] = {}
    for row in data.get("rows", []) if isinstance(data, dict) else []:
        if not isinstance(row, dict):
            continue
        val = row.get("constant_transform_residual_translation_m")
        if val is None:
            continue
        out[int(row["frame_idx"])] = float(val)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render-state", type=Path, required=True)
    p.add_argument("--hot3d-gt", type=Path, required=True)
    p.add_argument("--stream-id", default="214-1")
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--object-alignment-report", type=Path, default=None, help="Optional object residual report for framewise camera/object residual correlation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = load_render_state_annotation_path(args.render_state)
    v19_cams = load_v19_cameras(annotation_path)
    gt_cams = load_hot3d_cameras(args.hot3d_gt, args.stream_id)
    common = sorted(set(v19_cams) & set(gt_cams))
    if not common:
        raise SystemExit("no common camera frames")

    Xs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx in common:
        # If camera trajectories differ only by world-frame convention,
        # T_gt_world_cam = X * T_v19_world_cam with one constant X.
        X = gt_cams[idx] @ np.linalg.inv(v19_cams[idx])
        Xs.append(X)
        rows.append({"frame_idx": int(idx), "relative_world_transform_translation_m": X[:3, 3].tolist(), "relative_world_transform_rotation_rad": transform_angle_rad(X)})

    q_mean = quat_average_wxyz([R_to_quat_wxyz(X[:3, :3]) for X in Xs])
    R_mean = quat_wxyz_to_R(q_mean)
    t_mean = np.mean([X[:3, 3] for X in Xs], axis=0)
    X_fit = make_T(R_mean, t_mean)

    rel_t = [float(np.linalg.norm(X[:3, 3] - t_mean)) for X in Xs]
    rel_rot = [transform_angle_rad(np.linalg.inv(X_fit) @ X) for X in Xs]
    residual_t: list[float] = []
    residual_rot: list[float] = []
    for row, idx in zip(rows, common):
        T_pred_gt_world_cam = X_fit @ v19_cams[idx]
        T_res = np.linalg.inv(gt_cams[idx]) @ T_pred_gt_world_cam
        rt = float(np.linalg.norm(T_res[:3, 3]))
        rr = transform_angle_rad(T_res)
        residual_t.append(rt)
        residual_rot.append(rr)
        row["constant_world_transform_camera_residual_translation_m"] = rt
        row["constant_world_transform_camera_residual_rotation_rad"] = rr
        row["constant_world_transform_camera_residual_rotation_deg"] = math.degrees(rr)

    obj_by_frame = load_object_residual_by_frame(args.object_alignment_report)
    corr = None
    paired_camera: list[float] = []
    paired_object: list[float] = []
    if obj_by_frame:
        cam_by_frame = {idx: residual_t[i] for i, idx in enumerate(common)}
        for idx in sorted(set(cam_by_frame) & set(obj_by_frame)):
            paired_camera.append(cam_by_frame[idx])
            paired_object.append(obj_by_frame[idx])
        corr = pearson(paired_camera, paired_object)

    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_camera_trajectory_alignment",
        "claim_scope": "camera trajectory consistency after fitting one fixed transform from V19 world to HOT3D world; evaluation-only, no prediction mutation",
        "inputs": {
            "render_state": str(args.render_state),
            "annotations": str(annotation_path),
            "hot3d_gt": str(args.hot3d_gt),
            "stream_id": args.stream_id,
            "object_alignment_report": str(args.object_alignment_report) if args.object_alignment_report else None,
        },
        "row_count": len(common),
        "frame_min": int(min(common)),
        "frame_max": int(max(common)),
        "fitted_constant_transform_v19_world_to_hot3d_world": {
            "rotation_quaternion_wxyz": q_mean.tolist(),
            "translation_xyz_m": t_mean.tolist(),
        },
        "relative_transform_variability": {
            "translation_deviation_from_mean_m": summarize(rel_t),
            "rotation_deviation_from_mean_rad": summarize(rel_rot),
            "rotation_deviation_from_mean_deg": summarize([math.degrees(v) for v in rel_rot]),
        },
        "constant_transform_camera_residual": {
            "translation_m": summarize(residual_t),
            "rotation_rad": summarize(residual_rot),
            "rotation_deg": summarize([math.degrees(v) for v in residual_rot]),
        },
        "camera_object_residual_coupling": {
            "paired_frame_count": len(paired_camera),
            "pearson_camera_translation_vs_object_translation_residual": corr,
            "camera_translation_residual_m": summarize(paired_camera),
            "object_translation_residual_m": summarize(paired_object),
        },
        "interpretation_notes": [
            "Small camera residual relative to object residual means object-pose/shape dominates the evaluated object mismatch.",
            "Large camera residual with positive camera-object residual coupling means camera/head-pose drift may be upstream of object-pose inconsistency.",
            "This evaluator does not score hand pose, contact, occlusion, object shape, or nonpenetration.",
        ],
        "rows": rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "row_count", "frame_min", "frame_max", "constant_transform_camera_residual", "camera_object_residual_coupling"]}, indent=2))


if __name__ == "__main__":
    main()
