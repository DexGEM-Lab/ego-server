#!/usr/bin/env python3
"""Decompose V19 interval MANO changes against a HaWoR HOT3D baseline.

This is an evaluator/autoresearch tool, not a prediction-stage script. It consumes
HOT3D GT only after the runtime prediction boundary is frozen. The diagnostic
answers a causal question left open by the scalar evaluator: did the interval
state improve/worsen MANO accuracy by changing wrist/root translation, or by
changing wrist-relative articulation?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_v19_hot3d_hawor_mano3d import (  # noqa: E402
    HAWOR_MANO_TO_OPENPOSE_MAPPING,
    load_hand_shape,
    load_hawor_prediction,
    load_interval_state_prediction,
    load_json,
    load_smplx_mano,
    per_joint_errors,
    replay_hot3d_mano,
    se3_from_hot3d_dict,
    summarize,
    world_to_camera,
    write_json,
)


COLOR_BASE = (85, 85, 85)
COLOR_INTERVAL = (0, 88, 220)
COLOR_BASE_ROOT = (180, 120, 0)
COLOR_INTERVAL_ROOT = (0, 160, 220)
COLOR_DISP = (170, 70, 180)
COLOR_AWAY = (0, 120, 0)
COLOR_ZERO = (190, 190, 190)
COLOR_TEXT = (30, 30, 30)


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def median(values: list[float]) -> float | None:
    vals = finite(values)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "baseline_wrist_error_m",
        "interval_wrist_error_m",
        "delta_wrist_error_m",
        "baseline_mpjpe_m",
        "interval_mpjpe_m",
        "delta_mpjpe_m",
        "baseline_root_aligned_mpjpe_m",
        "interval_root_aligned_mpjpe_m",
        "delta_root_aligned_mpjpe_m",
        "mpjpe_delta_minus_root_aligned_delta_m",
        "wrist_correction_norm_m",
        "wrist_correction_projection_along_baseline_error_m",
    ]
    out: dict[str, Any] = {key: summarize([float(r[key]) for r in records if r.get(key) is not None]) for key in keys}
    out["count"] = len(records)
    out["interval_wrist_improved_fraction"] = float(
        np.mean([r["delta_wrist_error_m"] < 0 for r in records]) if records else 0.0
    )
    out["interval_mpjpe_improved_fraction"] = float(
        np.mean([r["delta_mpjpe_m"] < 0 for r in records]) if records else 0.0
    )
    out["wrist_correction_away_fraction"] = float(
        np.mean([r["wrist_correction_projection_along_baseline_error_m"] > 0 for r in records]) if records else 0.0
    )
    return out


def group_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        value = row.get(key)
        if value is None:
            value = "<none>"
        groups.setdefault(str(value), []).append(row)
    return {name: metric_summary(rows) for name, rows in sorted(groups.items(), key=lambda item: item[0])}


def paired_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    gt = load_json(args.hot3d_gt)
    frames_list = gt.get("frames")
    if not isinstance(frames_list, list) or not frames_list:
        raise RuntimeError("HOT3D GT sidecar has no frames")
    gt_frames = {int(row["frame_idx"]): row for row in frames_list if isinstance(row, dict)}
    base_ns = SimpleNamespace(
        hawor_npz=args.hawor_npz,
        interval_state=None,
        remote_root=args.remote_root,
        local_root=args.local_root,
    )
    interval_ns = SimpleNamespace(
        hawor_npz=None,
        interval_state=args.interval_state,
        remote_root=args.remote_root,
        local_root=args.local_root,
    )
    baseline = load_hawor_prediction(base_ns, frames_list)
    interval = load_interval_state_prediction(interval_ns, frames_list)
    base_frame_idx = np.asarray(baseline["frame_idx"], dtype=int)
    interval_frame_idx = np.asarray(interval["frame_idx"], dtype=int)
    if base_frame_idx.shape != interval_frame_idx.shape or not np.array_equal(base_frame_idx, interval_frame_idx):
        raise RuntimeError("baseline and interval predictions have different frame_idx arrays")

    beta, hand_shape_source = load_hand_shape(gt, args.hot3d_gt)
    layers = load_smplx_mano(args)
    device = torch.device(args.device)
    for layer in layers.values():
        layer.to(device)
        layer.eval()

    rows: list[dict[str, Any]] = []
    mapping = np.asarray(HAWOR_MANO_TO_OPENPOSE_MAPPING, dtype=int)
    for local_i, frame in enumerate(base_frame_idx.tolist()):
        gt_row = gt_frames.get(int(frame))
        if not isinstance(gt_row, dict):
            continue
        hot3d_cam = gt_row.get("json", {}).get("cameras.json", {}).get(args.stream_id)
        if not isinstance(hot3d_cam, dict):
            continue
        R_hot3d_c2w, t_hot3d_c2w = se3_from_hot3d_dict(hot3d_cam["T_world_from_camera"])
        R_base_c2w = np.asarray(baseline["R_c2w"][local_i], dtype=np.float64)
        t_base_c2w = np.asarray(baseline["t_c2w"][local_i], dtype=np.float64)
        R_interval_c2w = np.asarray(interval["R_c2w"][local_i], dtype=np.float64)
        t_interval_c2w = np.asarray(interval["t_c2w"][local_i], dtype=np.float64)
        for side in ("left", "right"):
            hand = gt_row.get("json", {}).get("hands.json", {}).get(side)
            if not isinstance(hand, dict) or "mano_pose" not in hand:
                continue
            if not bool(baseline["sides"][side]["valid"][local_i]):
                continue
            if not bool(interval["sides"][side]["valid"][local_i]):
                continue
            mano = hand["mano_pose"]
            theta = np.asarray(mano["thetas"], dtype=np.float32)
            wrist_xform = np.asarray(mano["wrist_xform"], dtype=np.float32)
            _, gt_joints_raw_w, _ = replay_hot3d_mano(layers[side], beta, theta, wrist_xform, device)
            gt_joints_cam = world_to_camera(gt_joints_raw_w, R_hot3d_c2w, t_hot3d_c2w)[mapping]
            base_joints_w = np.asarray(baseline["sides"][side]["joints_world_m"][local_i], dtype=np.float64)
            interval_joints_w = np.asarray(interval["sides"][side]["joints_world_m"][local_i], dtype=np.float64)
            base_joints_cam = world_to_camera(base_joints_w, R_base_c2w, t_base_c2w)
            interval_joints_cam = world_to_camera(interval_joints_w, R_interval_c2w, t_interval_c2w)
            b_wrist, b_mpjpe, b_med, b_root, b_root_med, b_root_p95 = per_joint_errors(base_joints_cam, gt_joints_cam)
            i_wrist, i_mpjpe, i_med, i_root, i_root_med, i_root_p95 = per_joint_errors(interval_joints_cam, gt_joints_cam)
            wrist_correction = interval_joints_cam[0] - base_joints_cam[0]
            baseline_error = base_joints_cam[0] - gt_joints_cam[0]
            baseline_error_norm = float(np.linalg.norm(baseline_error))
            if baseline_error_norm > 1.0e-9:
                projection = float(np.dot(wrist_correction, baseline_error) / baseline_error_norm)
            else:
                projection = 0.0
            row_meta = interval["sides"][side].get("row_meta", [None] * len(base_frame_idx))[local_i]
            if not isinstance(row_meta, dict):
                row_meta = {}
            rows.append(
                {
                    "frame_idx": int(frame),
                    "side": side,
                    "hand_shape_source": hand_shape_source,
                    "baseline_wrist_error_m": float(b_wrist),
                    "interval_wrist_error_m": float(i_wrist),
                    "delta_wrist_error_m": float(i_wrist - b_wrist),
                    "baseline_mpjpe_m": float(b_mpjpe),
                    "interval_mpjpe_m": float(i_mpjpe),
                    "delta_mpjpe_m": float(i_mpjpe - b_mpjpe),
                    "baseline_joint_median_error_m": float(b_med),
                    "interval_joint_median_error_m": float(i_med),
                    "baseline_root_aligned_mpjpe_m": float(b_root),
                    "interval_root_aligned_mpjpe_m": float(i_root),
                    "delta_root_aligned_mpjpe_m": float(i_root - b_root),
                    "baseline_root_aligned_median_error_m": float(b_root_med),
                    "interval_root_aligned_median_error_m": float(i_root_med),
                    "baseline_root_aligned_p95_error_m": float(b_root_p95),
                    "interval_root_aligned_p95_error_m": float(i_root_p95),
                    "mpjpe_delta_minus_root_aligned_delta_m": float((i_mpjpe - b_mpjpe) - (i_root - b_root)),
                    "wrist_correction_norm_m": float(np.linalg.norm(wrist_correction)),
                    "wrist_correction_projection_along_baseline_error_m": projection,
                    "wrist_correction_direction": "away_from_gt" if projection > 0 else "toward_gt_or_orthogonal",
                    "temporal_mano_state": row_meta.get("temporal_mano_state"),
                    "contact_patch_state_optimized": row_meta.get("contact_patch_state_optimized"),
                    "visible_surface_depth_order_selected_vertex_count": row_meta.get(
                        "visible_surface_depth_order_selected_vertex_count"
                    ),
                    "optimized_vertices_world_sample_count": row_meta.get("optimized_vertices_world_sample_count"),
                    "source_frame_index": row_meta.get("source_frame_index"),
                }
            )
    return rows


def safe_range(series: list[list[float]], include_zero: bool = False) -> tuple[float, float]:
    vals: list[float] = []
    for s in series:
        vals.extend(finite(s))
    if include_zero:
        vals.append(0.0)
    if not vals:
        return 0.0, 1.0
    lo = float(min(vals))
    hi = float(max(vals))
    if hi <= lo:
        pad = max(abs(hi) * 0.1, 1.0e-3)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def draw_panel(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[int], list[float], tuple[int, int, int]]],
    y_label: str,
    include_zero: bool = False,
) -> None:
    x0, y0, w, h = box
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    all_frames = [f for _, frames, _, _ in series for f in frames]
    if not all_frames:
        return
    fmin, fmax = min(all_frames), max(all_frames)
    ymin, ymax = safe_range([vals for _, _, vals, _ in series], include_zero=include_zero)
    plot_x0, plot_y0 = x0 + 70, y0 + 45
    plot_w, plot_h = w - 95, h - 85
    cv2.line(canvas, (plot_x0, plot_y0 + plot_h), (plot_x0 + plot_w, plot_y0 + plot_h), (130, 130, 130), 1)
    cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y0 + plot_h), (130, 130, 130), 1)
    if include_zero and ymin <= 0.0 <= ymax:
        zy = int(round(plot_y0 + plot_h - (0.0 - ymin) / (ymax - ymin) * plot_h))
        cv2.line(canvas, (plot_x0, zy), (plot_x0 + plot_w, zy), COLOR_ZERO, 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (plot_x0 - 28, zy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (x0 + 8, y0 + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ymax:.3f}", (plot_x0 - 62, plot_y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ymin:.3f}", (plot_x0 - 62, plot_y0 + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    legend_x = x0 + w - 420
    legend_y = y0 + 58
    for li, (name, frames, values, color) in enumerate(series):
        cv2.line(canvas, (legend_x, legend_y + li * 22), (legend_x + 24, legend_y + li * 22), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, name, (legend_x + 32, legend_y + li * 22 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1, cv2.LINE_AA)
        pts: list[tuple[int, int]] = []
        for frame, value in zip(frames, values):
            if not np.isfinite(value):
                if len(pts) >= 2:
                    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
                pts = []
                continue
            x = int(round(plot_x0 + (frame - fmin) / max(1, fmax - fmin) * plot_w))
            y = int(round(plot_y0 + plot_h - (float(value) - ymin) / (ymax - ymin) * plot_h))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def side_series(records: list[dict[str, Any]], side: str, key: str) -> tuple[list[int], list[float]]:
    rows = sorted([r for r in records if r["side"] == side], key=lambda r: int(r["frame_idx"]))
    return [int(r["frame_idx"]) for r in rows], [float(r[key]) for r in rows]


def draw_review(path: Path, report: dict[str, Any]) -> None:
    records = report["records"]
    canvas = np.full((1500, 1800, 3), 255, dtype=np.uint8)
    title = "V19 P18 interval MANO vs HaWoR baseline on HOT3D clip-001850"
    cv2.putText(canvas, title, (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Positive correction projection means the interval wrist moved farther along the baseline->away-from-GT error direction.",
        (34, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    for side, y in [("right", 120), ("left", 790)]:
        f, base_mpjpe = side_series(records, side, "baseline_mpjpe_m")
        _, int_mpjpe = side_series(records, side, "interval_mpjpe_m")
        _, base_root = side_series(records, side, "baseline_root_aligned_mpjpe_m")
        _, int_root = side_series(records, side, "interval_root_aligned_mpjpe_m")
        draw_panel(
            canvas,
            (34, y, 840, 290),
            f"{side} hand: MPJPE vs wrist-subtracted MPJPE",
            [
                ("baseline MPJPE", f, base_mpjpe, COLOR_BASE),
                ("interval MPJPE", f, int_mpjpe, COLOR_INTERVAL),
                ("baseline root-aligned", f, base_root, COLOR_BASE_ROOT),
                ("interval root-aligned", f, int_root, COLOR_INTERVAL_ROOT),
            ],
            "meters",
        )
        _, disp = side_series(records, side, "wrist_correction_norm_m")
        _, away = side_series(records, side, "wrist_correction_projection_along_baseline_error_m")
        draw_panel(
            canvas,
            (920, y, 840, 290),
            f"{side} hand: interval wrist displacement and direction",
            [
                ("|interval wrist - baseline wrist|", f, disp, COLOR_DISP),
                ("projection along baseline error", f, away, COLOR_AWAY),
            ],
            "meters",
            include_zero=True,
        )
    summary = report["summary"]
    right = summary["by_side"]["right"]
    left = summary["by_side"]["left"]
    lines = [
        "Mechanism summary:",
        f"right mean MPJPE delta = {right['delta_mpjpe_m']['mean']:.4f} m, p90 = {right['delta_mpjpe_m']['p90']:.4f} m; right root-aligned mean delta = {right['delta_root_aligned_mpjpe_m']['mean']:.4f} m",
        f"right median wrist displacement = {right['wrist_correction_norm_m']['median']:.4f} m; right correction-away fraction = {right['wrist_correction_away_fraction']:.2f}; wrist improved fraction = {right['interval_wrist_improved_fraction']:.2f}",
        f"left mean MPJPE delta = {left['delta_mpjpe_m']['mean']:.4f} m; left correction-away fraction = {left['wrist_correction_away_fraction']:.2f}; many unchanged rows make the overall per-row median near zero.",
        "Supported implication: protect/gate wrist-root translation before claiming MANO accuracy; contact/object terms may remain uncertainty evidence.",
    ]
    y = 1430 - 28 * (len(lines) - 1)
    for line in lines:
        cv2.putText(canvas, line, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2 if line.endswith(":") else 1, cv2.LINE_AA)
        y += 28
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records = paired_rows(args)
    if not records:
        raise RuntimeError("no paired baseline/interval/GT rows")
    summary = {
        "overall": metric_summary(records),
        "by_side": {side: metric_summary([r for r in records if r["side"] == side]) for side in ("left", "right")},
        "by_temporal_mano_state": group_summary(records, "temporal_mano_state"),
        "by_contact_patch_state_optimized": group_summary(records, "contact_patch_state_optimized"),
        "by_wrist_correction_direction": group_summary(records, "wrist_correction_direction"),
    }
    top_worsened = sorted(records, key=lambda r: r["delta_mpjpe_m"], reverse=True)[:12]
    top_improved = sorted(records, key=lambda r: r["delta_mpjpe_m"])[:12]
    return {
        "status": "ok",
        "claim_scope": "Evaluator-only decomposition of frozen V19 P18 interval MANO joints versus runtime HaWoR baseline on HOT3D camera-coordinate 21-joint MANO. Not contact/object/occlusion/nonpenetration scoring.",
        "hot3d_gt": str(args.hot3d_gt),
        "hawor_npz": str(args.hawor_npz),
        "interval_state": str(args.interval_state),
        "row_count": len(records),
        "summary": summary,
        "top_worsened_by_delta_mpjpe": top_worsened,
        "top_improved_by_delta_mpjpe": top_improved,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot3d-gt", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--interval-state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--mano-left", type=Path, default=Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl"))
    parser.add_argument("--mano-right", type=Path, default=Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_RIGHT.pkl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--remote-root", type=Path, default=None)
    parser.add_argument("--local-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(args.output_json, report)
    draw_review(args.output_review, report)
    compact = {
        "status": report["status"],
        "row_count": report["row_count"],
        "overall": report["summary"]["overall"],
        "by_side": report["summary"]["by_side"],
        "output_json": str(args.output_json),
        "output_review": str(args.output_review),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
