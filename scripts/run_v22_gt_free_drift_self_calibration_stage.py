#!/usr/bin/env python3
"""Estimate GT-free hand drift/self-calibration hypotheses from frozen V22 outputs.

The stage consumes the current hybrid metric hand candidate, canonical K, and
WiLoR same-frame 2D evidence. It estimates smooth per-frame/side image-plane
bias terms b_uv(t) from projection residuals. The output is a calibration
hypothesis and uncertainty report; it does not claim fixed-gauge accuracy and
it does not mutate the hand state.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_intrinsics(path: Path) -> np.ndarray:
    payload = load_json(path)
    values = payload.get("intrinsics_fx_fy_cx_cy")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (4,) or not np.isfinite(arr).all() or arr[0] <= 0 or arr[1] <= 0:
        raise RuntimeError(f"invalid intrinsics_fx_fy_cx_cy in {path}")
    return arr


def project(points_cam: np.ndarray, intr: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(pts).all(axis=1) & (z > 1.0e-8)
    out[good, 0] = pts[good, 0] / z[good] * intr[0] + intr[2]
    out[good, 1] = pts[good, 1] / z[good] * intr[1] + intr[3]
    return out


def world_to_camera(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - np.asarray(t_c2w, dtype=np.float64)[None, :]) @ np.asarray(r_c2w, dtype=np.float64)


def smooth(values: list[float | None], radius: int) -> list[float | None]:
    out: list[float | None] = []
    n = len(values)
    for i in range(n):
        vals = [values[j] for j in range(max(0, i - radius), min(n, i + radius + 1))]
        finite = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        out.append(float(np.median(finite)) if finite else None)
    return out


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def wilor_by_frame_side(wilor_raw: dict[str, Any], min_score: float) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    frames = wilor_raw.get("frames") if isinstance(wilor_raw.get("frames"), list) else []
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        frame_idx = int(frame["frame_idx"])
        hands = frame.get("raw_hands") if isinstance(frame.get("raw_hands"), list) else []
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("side") or "unknown")
            score = finite_float(hand.get("detector_score"))
            if score is None or score < min_score:
                continue
            key = (frame_idx, side)
            prev = out.get(key)
            prev_score = finite_float(prev.get("detector_score")) if prev else None
            if prev is None or prev_score is None or score > prev_score:
                out[key] = hand
    return out


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    calibration = args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    wilor_path = args.wilor_raw or (run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json")
    hybrid_npz = args.hybrid_npz or (run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz")
    output = args.output or (run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json")
    intr = load_intrinsics(calibration)
    wilor_raw = load_json(wilor_path)
    evidence = wilor_by_frame_side(wilor_raw, float(args.min_wilor_score))
    if not hybrid_npz.exists():
        payload = {
            "schema": "v22_gt_free_drift_self_calibration.v0",
            "status": "no_hybrid_state",
            "run_root": str(run_root),
            "calibration_contract": str(calibration),
            "wilor_raw": str(wilor_path),
            "hybrid_npz": str(hybrid_npz),
            "claim_scope": "D8 stage executed but cannot estimate drift without hybrid metric hand state.",
            "rows": [],
            "summary": {"residual_px": {"count": 0}},
            "elapsed_s": float(time.time() - started),
        }
        write_json(output, payload)
        print(json.dumps(payload, indent=2))
        return payload

    blob = np.load(hybrid_npz, allow_pickle=True)
    required = {"frame_idx", "R_c2w", "t_c2w"}
    missing = sorted(required - set(blob.files))
    if missing:
        raise RuntimeError(f"hybrid NPZ lacks required arrays for D8: {missing}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    rows_by_side: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    residual_norms: list[float] = []
    for pos, idx in enumerate(frame_idx.tolist()):
        r_c2w = np.asarray(blob["R_c2w"])[pos]
        t_c2w = np.asarray(blob["t_c2w"])[pos]
        for side in ("left", "right"):
            joints_key = f"{side}_joints_world_m"
            if joints_key not in blob.files:
                continue
            hand = evidence.get((int(idx), side))
            target = np.asarray((hand or {}).get("joints2d") or (hand or {}).get("joints2d_raw"), dtype=np.float64)
            joints_world = np.asarray(blob[joints_key])[pos]
            valid = target.shape == (21, 2) and joints_world.shape == (21, 3)
            row: dict[str, Any] = {
                "frame_idx": int(idx),
                "side": side,
                "evidence": "wilor_same_frame_2d" if valid else "no_same_frame_2d",
                "accepted_correction": False,
            }
            if valid:
                joints_cam = world_to_camera(joints_world, r_c2w, t_c2w)
                uv = project(joints_cam, intr)
                finite = np.isfinite(uv).all(axis=1) & np.isfinite(target).all(axis=1)
                if np.count_nonzero(finite) >= int(args.min_joint_count):
                    residual = target[finite] - uv[finite]
                    norm = np.linalg.norm(residual, axis=1)
                    bias = np.median(residual, axis=0)
                    row.update(
                        {
                            "finite_joint_count": int(np.count_nonzero(finite)),
                            "median_residual_uv_px": bias.astype(float).tolist(),
                            "median_residual_norm_px": float(np.median(norm)),
                            "p90_residual_norm_px": float(np.percentile(norm, 90)),
                            "max_residual_norm_px": float(np.max(norm)),
                        }
                    )
                    residual_norms.append(float(np.median(norm)))
                else:
                    row["evidence"] = "insufficient_projectable_joints"
            rows_by_side[side].append(row)

    rows: list[dict[str, Any]] = []
    for side, side_rows in rows_by_side.items():
        xs = [((row.get("median_residual_uv_px") or [None, None])[0] if isinstance(row.get("median_residual_uv_px"), list) else None) for row in side_rows]
        ys = [((row.get("median_residual_uv_px") or [None, None])[1] if isinstance(row.get("median_residual_uv_px"), list) else None) for row in side_rows]
        sm_x = smooth(xs, int(args.smooth_radius))
        sm_y = smooth(ys, int(args.smooth_radius))
        for row, bx, by in zip(side_rows, sm_x, sm_y):
            row["smooth_bias_uv_px"] = [bx, by] if bx is not None and by is not None else None
            row["correction_family"] = "per_side_smooth_image_plane_bias_b_uv_t"
            if row.get("median_residual_norm_px") is not None and bx is not None and by is not None:
                row["accepted_correction"] = float(row["median_residual_norm_px"]) > float(args.accept_residual_px)
            rows.append(row)

    residual_summary = summarize(residual_norms)
    status = "ok" if residual_summary["count"] else "no_same_frame_support"
    payload = {
        "schema": "v22_gt_free_drift_self_calibration.v0",
        "status": status,
        "method": "project_hybrid_hand_state_to_wilor_2d_and_smooth_bias",
        "run_root": str(run_root),
        "calibration_contract": str(calibration),
        "wilor_raw": str(wilor_path),
        "hybrid_npz": str(hybrid_npz),
        "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
        "parameters": {
            "min_wilor_score": float(args.min_wilor_score),
            "min_joint_count": int(args.min_joint_count),
            "smooth_radius": int(args.smooth_radius),
            "accept_residual_px": float(args.accept_residual_px),
        },
        "summary": {
            "residual_px": residual_summary,
            "rows": len(rows),
            "rows_with_same_frame_support": sum(1 for row in rows if row.get("median_residual_norm_px") is not None),
            "accepted_correction_rows": sum(1 for row in rows if row.get("accepted_correction")),
        },
        "rows": rows,
        "claim_scope": "GT-free D8 drift/self-calibration hypothesis. It estimates smooth image-plane bias from cross-source residuals and does not certify fixed-gauge 3D accuracy.",
        "elapsed_s": float(time.time() - started),
    }
    write_json(output, payload)
    print(json.dumps({k: payload[k] for k in ("status", "method", "summary", "claim_scope")}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--wilor-raw", type=Path, default=None)
    parser.add_argument("--hybrid-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-wilor-score", type=float, default=0.30)
    parser.add_argument("--min-joint-count", type=int, default=8)
    parser.add_argument("--smooth-radius", type=int, default=5)
    parser.add_argument("--accept-residual-px", type=float, default=8.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    estimate(parse_args())
