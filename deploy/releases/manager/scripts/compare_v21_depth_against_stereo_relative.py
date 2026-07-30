#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ContractError(RuntimeError):
    pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {"count": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)), "max": float(np.max(arr))}


def rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 10 or b.size < 10:
        return None
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    if np.std(ar) <= 1e-9 or np.std(br) <= 1e-9:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def run(args: argparse.Namespace) -> dict[str, Any]:
    depth_blob = np.load(args.depthpro_npz)
    stereo_blob = np.load(args.stereo_npz, allow_pickle=True)
    depth_frames = depth_blob["frame_idx"].astype(int)
    stereo_frames = stereo_blob["frame_idx"].astype(int)
    depth = depth_blob["depth"].astype(np.float32)
    rel = stereo_blob["relative_inverse_depth"].astype(np.float32)
    depth_map = {int(f): i for i, f in enumerate(depth_frames)}
    stereo_map = {int(f): i for i, f in enumerate(stereo_frames)}
    common = sorted(set(depth_map).intersection(stereo_map))
    if not common:
        raise ContractError("no_common_frames")
    rows = []
    corrs = []
    med_depths = []
    med_rel = []
    for frame_idx in common[:: max(1, int(args.frame_stride))]:
        d = depth[depth_map[frame_idx]]
        r = rel[stereo_map[frame_idx]]
        if r.shape != d.shape:
            r = cv2.resize(r, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(d) & np.isfinite(r) & (d > 0) & (r > 0)
        if np.count_nonzero(valid) < int(args.min_valid_pixels):
            rows.append({"frame_idx": frame_idx, "status": "too_few_overlap_pixels", "overlap_pixels": int(np.count_nonzero(valid))})
            continue
        vals_d = d[valid]
        vals_inv = 1.0 / np.maximum(vals_d, 1e-6)
        vals_r = r[valid]
        if vals_d.size > int(args.max_samples_per_frame):
            idx = np.linspace(0, vals_d.size - 1, int(args.max_samples_per_frame), dtype=int)
            vals_inv = vals_inv[idx]
            vals_r = vals_r[idx]
            vals_d = vals_d[idx]
        corr = rank_corr(vals_inv.astype(float), vals_r.astype(float))
        if corr is not None:
            corrs.append(corr)
        med_depths.append(float(np.median(vals_d)))
        med_rel.append(float(np.median(vals_r)))
        rows.append({"frame_idx": frame_idx, "status": "ok", "overlap_pixels": int(np.count_nonzero(valid)), "rank_corr_inverse_depth_vs_stereo_relative": corr, "depthpro_depth_median_m": float(np.median(vals_d)), "stereo_relative_median": float(np.median(vals_r))})
    report = {
        "schema": "v21_depthpro_vs_stereo_relative_report.v0",
        "status": "ok",
        "method": "compare_v21_depth_against_stereo_relative",
        "depthpro_npz": str(args.depthpro_npz),
        "stereo_npz": str(args.stereo_npz),
        "evaluated_rows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "rank_corr_inverse_depth_vs_stereo_relative": summarize(corrs),
        "depthpro_depth_median_m": summarize(med_depths),
        "stereo_relative_median": summarize(med_rel),
        "rows": rows,
        "interpretation": "Positive rank correlation means uncalibrated stereo relative inverse-depth broadly agrees with DepthPro ordering. This does not make stereo metric or primary.",
    }
    write_json(args.output_report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DepthPro metric depth ordering with uncalibrated stereo relative inverse-depth.")
    parser.add_argument("--depthpro-npz", type=Path, required=True)
    parser.add_argument("--stereo-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=10000)
    parser.add_argument("--max-samples-per-frame", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
