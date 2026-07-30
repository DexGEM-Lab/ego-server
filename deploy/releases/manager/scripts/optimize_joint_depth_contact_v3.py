#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from optimize_contact_depth_scale_v3 import load_rows, summarize


def unpack(params: np.ndarray, n: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    hand_log_scale = float(params[0])
    object_log_scale = float(params[1])
    hand_shift = np.asarray(params[2 : 2 + n], dtype=float)
    object_shift = np.asarray(params[2 + n : 2 + 2 * n], dtype=float)
    return hand_log_scale, object_log_scale, hand_shift, object_shift


def residual(params: np.ndarray, rows: list[dict], args: argparse.Namespace) -> np.ndarray:
    n = len(rows)
    hand_log_scale, object_log_scale, hand_shift, object_shift = unpack(params, n)
    residuals: list[np.ndarray] = []
    residuals.append(np.asarray([hand_log_scale / args.sigma_hand_log_scale], dtype=float))
    residuals.append(np.asarray([object_log_scale / args.sigma_object_log_scale], dtype=float))
    for i, row in enumerate(rows):
        h = row["hand_depth_m"] * np.exp(hand_log_scale) + hand_shift[i]
        o = row["object_depth_m"] * np.exp(object_log_scale) + object_shift[i]
        contact_sigma = args.sigma_contact_m / max(1.0, np.sqrt(row["near_vertices"] / args.vertex_count_reference))
        residuals.append(np.asarray([(h - o) / contact_sigma], dtype=float))
        residuals.append(np.asarray([hand_shift[i] / args.sigma_hand_shift_m], dtype=float))
        residuals.append(np.asarray([object_shift[i] / args.sigma_object_shift_m], dtype=float))
    for i in range(1, n):
        residuals.append(np.asarray([(hand_shift[i] - hand_shift[i - 1]) / args.sigma_hand_shift_step_m], dtype=float))
        residuals.append(np.asarray([(object_shift[i] - object_shift[i - 1]) / args.sigma_object_shift_step_m], dtype=float))
    for i in range(1, n - 1):
        residuals.append(
            np.asarray([(hand_shift[i + 1] - 2.0 * hand_shift[i] + hand_shift[i - 1]) / args.sigma_hand_shift_accel_m], dtype=float)
        )
        residuals.append(
            np.asarray(
                [(object_shift[i + 1] - 2.0 * object_shift[i] + object_shift[i - 1]) / args.sigma_object_shift_accel_m],
                dtype=float,
            )
        )
    return np.concatenate(residuals)


def row_metrics(params: np.ndarray, rows: list[dict]) -> list[dict]:
    n = len(rows)
    hand_log_scale, object_log_scale, hand_shift, object_shift = unpack(params, n)
    out = []
    for i, row in enumerate(rows):
        h = row["hand_depth_m"] * np.exp(hand_log_scale) + hand_shift[i]
        o = row["object_depth_m"] * np.exp(object_log_scale) + object_shift[i]
        out.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "raw_gap_m": float(row["raw_gap_m"]),
                "corrected_gap_m": float(h - o),
                "hand_depth_m": float(row["hand_depth_m"]),
                "object_depth_m": float(row["object_depth_m"]),
                "hand_depth_corrected_m": float(h),
                "object_depth_corrected_m": float(o),
                "hand_shift_m": float(hand_shift[i]),
                "object_shift_m": float(object_shift[i]),
                "near_vertices": int(row["near_vertices"]),
            }
        )
    return out


def residual_blocks(params: np.ndarray, rows: list[dict], args: argparse.Namespace) -> dict:
    n = len(rows)
    hand_log_scale, object_log_scale, hand_shift, object_shift = unpack(params, n)
    contact = []
    for i, row in enumerate(rows):
        h = row["hand_depth_m"] * np.exp(hand_log_scale) + hand_shift[i]
        o = row["object_depth_m"] * np.exp(object_log_scale) + object_shift[i]
        contact_sigma = args.sigma_contact_m / max(1.0, np.sqrt(row["near_vertices"] / args.vertex_count_reference))
        contact.append((h - o) / contact_sigma)
    blocks = {
        "contact": np.asarray(contact, dtype=float),
        "scale_prior": np.asarray(
            [hand_log_scale / args.sigma_hand_log_scale, object_log_scale / args.sigma_object_log_scale],
            dtype=float,
        ),
        "shift_prior": np.concatenate([hand_shift / args.sigma_hand_shift_m, object_shift / args.sigma_object_shift_m]),
    }
    if n > 1:
        blocks["shift_step"] = np.concatenate(
            [
                np.diff(hand_shift) / args.sigma_hand_shift_step_m,
                np.diff(object_shift) / args.sigma_object_shift_step_m,
            ]
        )
    else:
        blocks["shift_step"] = np.zeros(0, dtype=float)
    if n > 2:
        blocks["shift_accel"] = np.concatenate(
            [
                (hand_shift[2:] - 2.0 * hand_shift[1:-1] + hand_shift[:-2]) / args.sigma_hand_shift_accel_m,
                (object_shift[2:] - 2.0 * object_shift[1:-1] + object_shift[:-2]) / args.sigma_object_shift_accel_m,
            ]
        )
    else:
        blocks["shift_accel"] = np.zeros(0, dtype=float)
    return {
        name: {
            "count": int(values.size),
            "rms": None if values.size == 0 else float(np.sqrt(np.mean(values * values))),
            "max_abs": None if values.size == 0 else float(np.max(np.abs(values))),
        }
        for name, values in blocks.items()
    }


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    rows = load_rows(args.contact_depth_report, args.min_near_vertices)
    x0 = np.zeros(2 + 2 * len(rows), dtype=float)
    lower = np.concatenate(
        [
            np.asarray([np.log(args.min_hand_depth_scale), np.log(args.min_object_depth_scale)], dtype=float),
            np.full(len(rows), -args.max_abs_hand_shift_m, dtype=float),
            np.full(len(rows), -args.max_abs_object_shift_m, dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.asarray([np.log(args.max_hand_depth_scale), np.log(args.max_object_depth_scale)], dtype=float),
            np.full(len(rows), args.max_abs_hand_shift_m, dtype=float),
            np.full(len(rows), args.max_abs_object_shift_m, dtype=float),
        ]
    )
    before = residual(x0, rows, args)
    result = least_squares(
        lambda x: residual(x, rows, args),
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
    )
    before_rows = row_metrics(x0, rows)
    after_rows = row_metrics(result.x, rows)
    hand_log_scale, object_log_scale, hand_shift, object_shift = unpack(result.x, len(rows))
    raw_gap = np.asarray([row["raw_gap_m"] for row in rows], dtype=float)
    corrected_gap = np.asarray([row["corrected_gap_m"] for row in after_rows], dtype=float)
    scale_at_bound = (
        abs(np.exp(hand_log_scale) - args.min_hand_depth_scale) <= args.bound_tolerance
        or abs(np.exp(hand_log_scale) - args.max_hand_depth_scale) <= args.bound_tolerance
        or abs(np.exp(object_log_scale) - args.min_object_depth_scale) <= args.bound_tolerance
        or abs(np.exp(object_log_scale) - args.max_object_depth_scale) <= args.bound_tolerance
    )
    shift_at_bound = (
        np.max(np.abs(hand_shift)) >= args.max_abs_hand_shift_m - args.bound_tolerance
        or np.max(np.abs(object_shift)) >= args.max_abs_object_shift_m - args.bound_tolerance
    )
    corrected_p95 = float(np.percentile(np.abs(corrected_gap), 95))
    corrected_median_abs = float(np.median(np.abs(corrected_gap)))
    if corrected_p95 > args.contact_solved_p95_m:
        status = "diagnostic_contact_depth_conflict_remains"
    elif scale_at_bound or shift_at_bound:
        status = "diagnostic_contact_fit_requires_bound_saturation"
    else:
        status = "diagnostic_contact_depth_fit_with_bounded_low_dim_variables"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "contact_depth_report": str(args.contact_depth_report),
        "rows": len(rows),
        "variables": int(result.x.size),
        "contact_equations": int(len(rows)),
        "model": "shared_hand_scale_shared_object_scale_smooth_per_frame_depth_shifts",
        "identifiability": (
            "Lower-dimensional than per-frame scale fitting, but still a depth-only diagnostic. "
            "It does not identify whether MANO depth, object depth, or camera scale is the true fault."
        ),
        "min_near_vertices": int(args.min_near_vertices),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(residual(result.x, rows, args) ** 2))),
        "residual_blocks_before": residual_blocks(x0, rows, args),
        "residual_blocks_after": residual_blocks(result.x, rows, args),
        "raw_gap_m": summarize(raw_gap),
        "corrected_gap_m": summarize(corrected_gap),
        "corrected_gap_abs_median_m": corrected_median_abs,
        "corrected_gap_abs_p95_m": corrected_p95,
        "contact_solved_p95_m": float(args.contact_solved_p95_m),
        "hand_depth_scale": float(np.exp(hand_log_scale)),
        "object_depth_scale": float(np.exp(object_log_scale)),
        "scale_at_bound": bool(scale_at_bound),
        "hand_shift_m": summarize(hand_shift),
        "object_shift_m": summarize(object_shift),
        "shift_at_bound": bool(shift_at_bound),
        "bounds": {
            "hand_depth_scale": [float(args.min_hand_depth_scale), float(args.max_hand_depth_scale)],
            "object_depth_scale": [float(args.min_object_depth_scale), float(args.max_object_depth_scale)],
            "max_abs_hand_shift_m": float(args.max_abs_hand_shift_m),
            "max_abs_object_shift_m": float(args.max_abs_object_shift_m),
            "bound_tolerance": float(args.bound_tolerance),
        },
        "priors": {
            "sigma_contact_m": float(args.sigma_contact_m),
            "sigma_hand_log_scale": float(args.sigma_hand_log_scale),
            "sigma_object_log_scale": float(args.sigma_object_log_scale),
            "sigma_hand_shift_m": float(args.sigma_hand_shift_m),
            "sigma_object_shift_m": float(args.sigma_object_shift_m),
            "sigma_hand_shift_step_m": float(args.sigma_hand_shift_step_m),
            "sigma_object_shift_step_m": float(args.sigma_object_shift_step_m),
            "sigma_hand_shift_accel_m": float(args.sigma_hand_shift_accel_m),
            "sigma_object_shift_accel_m": float(args.sigma_object_shift_accel_m),
        },
        "interpretation": (
            "If this diagnostic leaves a large corrected gap, the contact conflict cannot be explained by "
            "shared depth scale and smooth depth shifts. If it removes the gap only through large shifts or "
            "scales, the full v3 solver must expose that correction as a state variable with vision and "
            "reprojection evidence."
        ),
        "rows_before_preview": before_rows[:40],
        "rows_after_preview": after_rows[:40],
        "elapsed_s": time.time() - started,
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
    parser.add_argument("--sigma-hand-log-scale", type=float, default=0.080)
    parser.add_argument("--sigma-object-log-scale", type=float, default=0.120)
    parser.add_argument("--sigma-hand-shift-m", type=float, default=0.080)
    parser.add_argument("--sigma-object-shift-m", type=float, default=0.120)
    parser.add_argument("--sigma-hand-shift-step-m", type=float, default=0.020)
    parser.add_argument("--sigma-object-shift-step-m", type=float, default=0.030)
    parser.add_argument("--sigma-hand-shift-accel-m", type=float, default=0.015)
    parser.add_argument("--sigma-object-shift-accel-m", type=float, default=0.025)
    parser.add_argument("--min-hand-depth-scale", type=float, default=0.75)
    parser.add_argument("--max-hand-depth-scale", type=float, default=1.25)
    parser.add_argument("--min-object-depth-scale", type=float, default=0.60)
    parser.add_argument("--max-object-depth-scale", type=float, default=1.80)
    parser.add_argument("--max-abs-hand-shift-m", type=float, default=0.120)
    parser.add_argument("--max-abs-object-shift-m", type=float, default=0.180)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--contact-solved-p95-m", type=float, default=0.010)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
