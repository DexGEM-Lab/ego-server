"""Dependency-light evaluator scaffold for annotation metric observations."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "frames", "head_camera", "hand_states"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    raise ValueError(f"cannot load row list from {path}")


def vector3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in out):
        return None
    return out


def vector_norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def translation(row: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("t_world_camera_m", "t_camera_world_m", "translation_m", "position_m", "wrist_t_camera_m", "root_t_camera_m"):
        val = vector3(row.get(key))
        if val is not None:
            return val
    return None


def quaternion_xyzw(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = row.get("q_world_camera_xyzw") or row.get("q_xyzw") or row.get("quaternion_xyzw")
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z"), value.get("w")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        q = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(x * x for x in q))
    if norm <= 0 or not math.isfinite(norm):
        return None
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def quat_angle_deg(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def points3(value: Any) -> list[tuple[float, float, float]] | None:
    if not isinstance(value, list):
        return None
    out: list[tuple[float, float, float]] = []
    for item in value:
        vec = vector3(item)
        if vec is None:
            return None
        out.append(vec)
    return out if out else None


def mean_point_error(pred: list[tuple[float, float, float]], gt: list[tuple[float, float, float]]) -> float | None:
    n = min(len(pred), len(gt))
    if n <= 0:
        return None
    return sum(vector_norm(sub(pred[i], gt[i])) for i in range(n)) / n


def root_relative(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    root = points[0]
    return [sub(p, root) for p in points]


def by_frame(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = row.get("frame_idx")
        if idx is None:
            idx = row.get("frame")
        try:
            out[int(idx)] = row
        except (TypeError, ValueError):
            continue
    return out


def by_frame_side(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        try:
            idx = int(row.get("frame_idx", row.get("frame")))
        except (TypeError, ValueError):
            continue
        side = str(row.get("side") or row.get("hand") or "unknown")
        out[(idx, side)] = row
    return out


def evaluate_head_camera(pred_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    pred = by_frame(pred_rows)
    gt = by_frame(gt_rows)
    frames = sorted(set(pred) & set(gt))
    out: dict[str, list[float]] = defaultdict(list)
    for idx in frames:
        pt = translation(pred[idx])
        gt_t = translation(gt[idx])
        if pt is not None and gt_t is not None:
            out["head_camera_ate_translation_m"].append(vector_norm(sub(pt, gt_t)))
        pq = quaternion_xyzw(pred[idx])
        gq = quaternion_xyzw(gt[idx])
        if pq is not None and gq is not None:
            out["head_camera_rotation_deg"].append(quat_angle_deg(pq, gq))
    for a, b in zip(frames, frames[1:]):
        pa = translation(pred[a])
        pb = translation(pred[b])
        ga = translation(gt[a])
        gb = translation(gt[b])
        if pa is None or pb is None or ga is None or gb is None:
            continue
        pred_delta = sub(pb, pa)
        gt_delta = sub(gb, ga)
        out["head_camera_rpe_translation_m"].append(vector_norm(sub(pred_delta, gt_delta)))
        gt_len = vector_norm(gt_delta)
        pred_len = vector_norm(pred_delta)
        if gt_len > 1.0e-9:
            out["head_camera_scale_error_ratio"].append(pred_len / gt_len)
    return dict(out)


def evaluate_hands(pred_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]] | None = None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    pred = by_frame_side(pred_rows)
    gt = by_frame_side(gt_rows or [])
    for key, pred_row in pred.items():
        gt_row = gt.get(key)
        if gt_row is not None:
            pw = translation(pred_row)
            gw = translation(gt_row)
            if pw is not None and gw is not None:
                out["hand_wrist_root_error_m"].append(vector_norm(sub(pw, gw)))
            pj = points3(pred_row.get("joints_camera_m") or pred_row.get("joints_m"))
            gj = points3(gt_row.get("joints_camera_m") or gt_row.get("joints_m"))
            if pj is not None and gj is not None:
                err = mean_point_error(pj, gj)
                if err is not None:
                    out["hand_all_joint_mpjpe_m"].append(err)
                rr = mean_point_error(root_relative(pj), root_relative(gj))
                if rr is not None:
                    out["hand_root_relative_mpjpe_m"].append(rr)
            pv = points3(pred_row.get("vertices_camera_m") or pred_row.get("vertices_m"))
            gv = points3(gt_row.get("vertices_camera_m") or gt_row.get("vertices_m"))
            if pv is not None and gv is not None:
                err = mean_point_error(pv, gv)
                if err is not None:
                    out["hand_mpvpe_surface_m"].append(err)
            if "visibility" in pred_row and "visibility" in gt_row:
                out["visibility_state_accuracy"].append(1.0 if str(pred_row["visibility"]) == str(gt_row["visibility"]) else 0.0)
        reproj = pred_row.get("reprojection_error_px")
        try:
            reproj_f = float(reproj)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(reproj_f):
                out["hand_reprojection_error_px"].append(reproj_f)

    grouped: dict[str, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
    for (idx, side), row in pred.items():
        wrist = translation(row)
        if wrist is not None:
            grouped[side].append((idx, wrist))
    for rows in grouped.values():
        rows.sort(key=lambda x: x[0])
        for (_, prev), (_, cur) in zip(rows, rows[1:]):
            out["temporal_wrist_jitter_m_per_frame"].append(vector_norm(sub(cur, prev)))
    return dict(out)


def merge_observations(*parts: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(list)
    for part in parts:
        for key, values in part.items():
            merged[key].extend(values)
    return dict(merged)
