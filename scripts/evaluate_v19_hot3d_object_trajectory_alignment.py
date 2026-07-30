#!/usr/bin/env python3
"""Evaluate V19 object-pose trajectory consistency against HOT3D object GT.

This evaluator does not require the predicted object canonical mesh frame to match the
HOT3D/BOP object frame.  It converts both trajectories to camera coordinates and asks
whether a single fixed transform between the two object frames explains the entire
trajectory.  If the relative transform varies over time, the object pose/camera path is
systematically wrong; if it is stable, remaining hand-object failures are not explained
by temporal object-pose drift alone.
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
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {arr.shape}")
    return arr


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
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


def load_v19_rows(render_state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = render_state.get("object_pose_trajectory", {}).get("pose_rows") or []
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict) or "frame_idx" not in r:
            continue
        R = np.asarray(r.get("rotation_world_from_completed_canonical_matrix"), dtype=float)
        t = np.asarray(r.get("translation_world_m"), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            continue
        out[int(r["frame_idx"])] = r
    return out


def v19_T_world_obj(row: dict[str, Any]) -> np.ndarray:
    return make_T(row["rotation_world_from_completed_canonical_matrix"], row["translation_world_m"])


def load_v19_cameras(annotation_path: Path) -> dict[int, np.ndarray]:
    d = json.loads(annotation_path.read_text())
    cams: dict[int, np.ndarray] = {}
    for fr in d.get("frames", []):
        cam = fr.get("camera") or {}
        T = cam.get("T_world_camera_metric") or cam.get("T_world_camera")
        if T is None:
            continue
        cams[int(fr["frame_idx"])] = _as_np4(T)
    return cams


def load_hot3d_camera_object(sidecar_path: Path, object_bop_id: str, stream_id: str) -> dict[int, dict[str, Any]]:
    d = json.loads(sidecar_path.read_text())
    out: dict[int, dict[str, Any]] = {}
    for fr in d.get("frames", []):
        idx = int(fr["frame_idx"])
        j = fr.get("json") or {}
        cams = j.get("cameras.json") or {}
        objs = j.get("objects.json") or {}
        cam = cams.get(stream_id)
        entries = objs.get(str(object_bop_id)) or []
        if not cam or not entries:
            continue
        obj = entries[0]
        T_world_cam = transform_from_hot3d(cam["T_world_from_camera"])
        T_world_obj = transform_from_hot3d(obj["T_world_from_object"])
        out[idx] = {
            "T_world_camera": T_world_cam,
            "T_world_object": T_world_obj,
            "object_name": obj.get("object_name"),
            "visibility_modeled": (obj.get("visibilities_modeled") or {}).get(stream_id),
            "box_amodal": (obj.get("boxes_amodal") or {}).get(stream_id),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-state", required=True, type=Path)
    ap.add_argument("--hot3d-gt", required=True, type=Path)
    ap.add_argument("--object-bop-id", required=True)
    ap.add_argument("--stream-id", default="214-1")
    ap.add_argument("--output-report", required=True, type=Path)
    ap.add_argument("--min-visibility", type=float, default=0.0)
    ap.add_argument("--direct-only", action="store_true", help="Evaluate only direct visible object-pose observations, excluding temporal holds.")
    args = ap.parse_args()

    render_state = json.loads(args.render_state.read_text())
    ann_path = Path(render_state["inputs"]["annotations"])
    v19_rows = load_v19_rows(render_state)
    v19_cams = load_v19_cameras(ann_path)
    gt_rows = load_hot3d_camera_object(args.hot3d_gt, str(args.object_bop_id), args.stream_id)

    rows: list[dict[str, Any]] = []
    Xs: list[np.ndarray] = []
    for idx in sorted(set(v19_rows) & set(v19_cams) & set(gt_rows)):
        vrow = v19_rows[idx]
        if args.direct_only and not (vrow.get("temporal_pose_graph") or {}).get("direct_visible_measurement", True):
            continue
        gt = gt_rows[idx]
        vis = gt.get("visibility_modeled")
        if vis is not None and float(vis) < args.min_visibility:
            continue
        T_v19_cam_obj = np.linalg.inv(v19_cams[idx]) @ v19_T_world_obj(vrow)
        T_gt_cam_obj = np.linalg.inv(gt["T_world_camera"]) @ gt["T_world_object"]
        X = np.linalg.inv(T_v19_cam_obj) @ T_gt_cam_obj
        Xs.append(X)
        rows.append(
            {
                "frame_idx": idx,
                "v19_pose_status": vrow.get("status"),
                "v19_pose_measurement_status": vrow.get("pose_measurement_status"),
                "v19_direct_visible_measurement": (vrow.get("temporal_pose_graph") or {}).get("direct_visible_measurement"),
                "gt_visibility_modeled": vis,
                "gt_box_amodal": gt.get("box_amodal"),
                "relative_transform_translation_m": X[:3, 3].tolist(),
                "relative_transform_rotation_angle_rad": transform_angle_rad(X),
            }
        )

    if not rows:
        raise SystemExit("no common evaluable frames")

    # Fit one constant object-frame transform X using quaternion averaging + mean translation.
    q_mean = quat_average_wxyz([R_to_quat_wxyz(X[:3, :3]) for X in Xs])
    R_mean = quat_wxyz_to_R(q_mean)
    t_mean = np.mean([X[:3, 3] for X in Xs], axis=0)
    X_fit = make_T(R_mean, t_mean)

    rel_t = [float(np.linalg.norm(X[:3, 3] - t_mean)) for X in Xs]
    rel_rot = [transform_angle_rad(np.linalg.inv(X_fit) @ X) for X in Xs]
    residual_t: list[float] = []
    residual_rot: list[float] = []
    camera_depth_delta: list[float] = []
    for row, X in zip(rows, Xs):
        idx = int(row["frame_idx"])
        T_v19_cam_obj = np.linalg.inv(v19_cams[idx]) @ v19_T_world_obj(v19_rows[idx])
        T_gt_cam_obj = np.linalg.inv(gt_rows[idx]["T_world_camera"]) @ gt_rows[idx]["T_world_object"]
        T_res = np.linalg.inv(T_gt_cam_obj) @ (T_v19_cam_obj @ X_fit)
        residual_t.append(float(np.linalg.norm(T_res[:3, 3])))
        residual_rot.append(transform_angle_rad(T_res))
        pred_origin_cam = (T_v19_cam_obj @ X_fit)[:3, 3]
        gt_origin_cam = T_gt_cam_obj[:3, 3]
        camera_depth_delta.append(float(pred_origin_cam[2] - gt_origin_cam[2]))
        row["constant_transform_residual_translation_m"] = residual_t[-1]
        row["constant_transform_residual_rotation_rad"] = residual_rot[-1]
        row["aligned_object_origin_depth_delta_m"] = camera_depth_delta[-1]

    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_object_trajectory_alignment",
        "claim_scope": "camera-coordinate object-pose trajectory consistency after fitting one fixed transform between predicted and HOT3D object canonical frames; does not score object shape/contact/nonpenetration",
        "inputs": {
            "render_state": str(args.render_state),
            "annotations": str(ann_path),
            "hot3d_gt": str(args.hot3d_gt),
            "object_bop_id": str(args.object_bop_id),
            "stream_id": args.stream_id,
            "min_visibility": args.min_visibility,
            "direct_only": args.direct_only,
        },
        "row_count": len(rows),
        "frame_min": min(r["frame_idx"] for r in rows),
        "frame_max": max(r["frame_idx"] for r in rows),
        "hot3d_object_name": next((gt_rows[int(r["frame_idx"])].get("object_name") for r in rows), None),
        "fitted_constant_transform_pred_object_to_hot3d_object": {
            "rotation_quaternion_wxyz": q_mean.tolist(),
            "translation_xyz_m": t_mean.tolist(),
        },
        "relative_transform_variability": {
            "translation_deviation_from_mean_m": summarize(rel_t),
            "rotation_deviation_from_mean_rad": summarize(rel_rot),
            "rotation_deviation_from_mean_deg": summarize([math.degrees(v) for v in rel_rot]),
        },
        "constant_transform_residual": {
            "translation_m": summarize(residual_t),
            "rotation_rad": summarize(residual_rot),
            "rotation_deg": summarize([math.degrees(v) for v in residual_rot]),
            "aligned_object_origin_depth_delta_m": summarize(camera_depth_delta),
        },
        "interpretation_notes": [
            "A stable relative transform means the predicted and GT object trajectories are temporally consistent up to object-frame convention.",
            "This cannot certify object shape, contact, or signed nonpenetration because the HOT3D CAD model is not used here.",
            "Large residuals after one fixed transform indicate object pose/camera trajectory inconsistency, not ordinary per-frame contact measurement noise.",
        ],
        "rows": rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "row_count", "frame_min", "frame_max", "hot3d_object_name", "relative_transform_variability", "constant_transform_residual"]}, indent=2))


if __name__ == "__main__":
    main()
