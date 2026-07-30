#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def load_rows(path: Path, min_vertices: int) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for frame in report.get("frame_reports", []):
        if frame.get("status") != "ok":
            continue
        hand = frame.get("mask_near_hand_depth_z") or {}
        obj = frame.get("object_depth_z") or {}
        n = int(hand.get("count", 0))
        if n < min_vertices or int(obj.get("count", 0)) <= 0:
            continue
        hand_z = float(hand["median"])
        obj_z = float(obj["median"])
        if hand_z <= 0 or obj_z <= 0:
            continue
        rows.append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "hand_depth_m": hand_z,
                "object_depth_m": obj_z,
                "near_vertices": n,
                "raw_gap_m": hand_z - obj_z,
            }
        )
    if not rows:
        raise RuntimeError(f"no usable rows in {path}")
    return rows


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def residual(params: np.ndarray, rows: list[dict], args: argparse.Namespace) -> np.ndarray:
    n = len(rows)
    hand_log_scale = params[:n]
    object_log_scale = params[n : 2 * n]
    hand_shift = params[2 * n : 3 * n]
    object_shift = params[3 * n : 4 * n]
    residuals = []
    for i, row in enumerate(rows):
        h = row["hand_depth_m"] * np.exp(hand_log_scale[i]) + hand_shift[i]
        o = row["object_depth_m"] * np.exp(object_log_scale[i]) + object_shift[i]
        contact_sigma = args.sigma_contact_m / max(1.0, np.sqrt(row["near_vertices"] / args.vertex_count_reference))
        residuals.append(np.asarray([(h - o) / contact_sigma], dtype=float))
        residuals.append(np.asarray([hand_log_scale[i] / args.sigma_hand_log_scale], dtype=float))
        residuals.append(np.asarray([object_log_scale[i] / args.sigma_object_log_scale], dtype=float))
        residuals.append(np.asarray([hand_shift[i] / args.sigma_hand_shift_m], dtype=float))
        residuals.append(np.asarray([object_shift[i] / args.sigma_object_shift_m], dtype=float))
    for i in range(1, n):
        residuals.append(np.asarray([(hand_log_scale[i] - hand_log_scale[i - 1]) / args.sigma_hand_log_scale_step], dtype=float))
        residuals.append(np.asarray([(object_log_scale[i] - object_log_scale[i - 1]) / args.sigma_object_log_scale_step], dtype=float))
        residuals.append(np.asarray([(hand_shift[i] - hand_shift[i - 1]) / args.sigma_hand_shift_step_m], dtype=float))
        residuals.append(np.asarray([(object_shift[i] - object_shift[i - 1]) / args.sigma_object_shift_step_m], dtype=float))
    return np.concatenate(residuals)


def row_metrics(params: np.ndarray, rows: list[dict]) -> list[dict]:
    n = len(rows)
    hand_log_scale = params[:n]
    object_log_scale = params[n : 2 * n]
    hand_shift = params[2 * n : 3 * n]
    object_shift = params[3 * n : 4 * n]
    out = []
    for i, row in enumerate(rows):
        h = row["hand_depth_m"] * np.exp(hand_log_scale[i]) + hand_shift[i]
        o = row["object_depth_m"] * np.exp(object_log_scale[i]) + object_shift[i]
        out.append(
            {
                **row,
                "hand_log_scale": float(hand_log_scale[i]),
                "object_log_scale": float(object_log_scale[i]),
                "hand_depth_scale": float(np.exp(hand_log_scale[i])),
                "object_depth_scale": float(np.exp(object_log_scale[i])),
                "hand_shift_m": float(hand_shift[i]),
                "object_shift_m": float(object_shift[i]),
                "corrected_hand_depth_m": float(h),
                "corrected_object_depth_m": float(o),
                "corrected_gap_m": float(h - o),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict:
    rows = load_rows(args.contact_depth_report, args.min_near_vertices)
    x0 = np.zeros(4 * len(rows), dtype=float)
    before = residual(x0, rows, args)
    result = least_squares(
        lambda x: residual(x, rows, args),
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
    )
    before_rows = row_metrics(x0, rows)
    after_rows = row_metrics(result.x, rows)
    raw_gap = np.asarray([row["raw_gap_m"] for row in rows], dtype=float)
    corrected_gap = np.asarray([row["corrected_gap_m"] for row in after_rows], dtype=float)
    hand_scale = np.asarray([row["hand_depth_scale"] for row in after_rows], dtype=float)
    object_scale = np.asarray([row["object_depth_scale"] for row in after_rows], dtype=float)
    hand_shift = np.asarray([row["hand_shift_m"] for row in after_rows], dtype=float)
    object_shift = np.asarray([row["object_shift_m"] for row in after_rows], dtype=float)
    report = {
        "status": "diagnostic_only",
        "annotation_ready": False,
        "identifiability": "underdetermined_per_frame_depth_variables",
        "contact_depth_report": str(args.contact_depth_report),
        "rows": len(rows),
        "variables": int(len(result.x)),
        "contact_equations": int(len(rows)),
        "free_depth_variables_per_frame": 4,
        "degrees_of_freedom_note": (
            "Each frame has hand scale, object scale, hand shift, and object shift for one contact-depth equation. "
            "Priors regularize a diagnostic tradeoff; corrected gaps do not identify the true faulty subsystem."
        ),
        "min_near_vertices": int(args.min_near_vertices),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(residual(result.x, rows, args) ** 2))),
        "raw_gap_m": summarize(raw_gap),
        "corrected_gap_m": summarize(corrected_gap),
        "hand_depth_scale": summarize(hand_scale),
        "object_depth_scale": summarize(object_scale),
        "hand_shift_m": summarize(hand_shift),
        "object_shift_m": summarize(object_shift),
        "priors": {
            "sigma_contact_m": float(args.sigma_contact_m),
            "sigma_hand_log_scale": float(args.sigma_hand_log_scale),
            "sigma_object_log_scale": float(args.sigma_object_log_scale),
            "sigma_hand_shift_m": float(args.sigma_hand_shift_m),
            "sigma_object_shift_m": float(args.sigma_object_shift_m),
        },
        "interpretation": (
            "This diagnostic solves only the 1D depth conflict for near-mask hand vertices. "
            "Large inferred hand/object scale or shift changes mean the full v3 factor graph "
            "must correct the corresponding upstream geometry rather than silently forcing contact."
        ),
        "rows_before_preview": before_rows[:30],
        "rows_after_preview": after_rows[:30],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_before_preview", "rows_after_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-depth-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--vertex-count-reference", type=float, default=240.0)
    parser.add_argument("--sigma-contact-m", type=float, default=0.010)
    parser.add_argument("--sigma-hand-log-scale", type=float, default=0.12)
    parser.add_argument("--sigma-object-log-scale", type=float, default=0.20)
    parser.add_argument("--sigma-hand-shift-m", type=float, default=0.080)
    parser.add_argument("--sigma-object-shift-m", type=float, default=0.120)
    parser.add_argument("--sigma-hand-log-scale-step", type=float, default=0.030)
    parser.add_argument("--sigma-object-log-scale-step", type=float, default=0.050)
    parser.add_argument("--sigma-hand-shift-step-m", type=float, default=0.030)
    parser.add_argument("--sigma-object-shift-step-m", type=float, default=0.050)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
