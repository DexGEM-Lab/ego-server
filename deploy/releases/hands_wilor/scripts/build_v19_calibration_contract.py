#!/usr/bin/env python3
"""Build a V19 run-root camera calibration contract.

The raw egocentric task shards used by V19 do not currently provide camera
calibration.  UniDepth can estimate an intrinsics matrix for each frame, but a
physical camera should have a constant intrinsics model unless the video has
explicit digital zoom/crop/stabilization metadata.  This component turns a noisy
per-frame intrinsics measurement stream into an explicit video-level calibration
hypothesis that downstream hand/object lifting can consume uniformly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def raw_frames(path: Path) -> tuple[list[int], tuple[int, int], dict[str, Any]]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"raw frame manifest has no frames: {path}")
    ids: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for row in frames:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("frame_idx", row.get("index", -1)))
        if idx < 0:
            raise RuntimeError(f"raw manifest row lacks frame_idx/index: {row}")
        ids.append(idx)
        w = int(row.get("source_width") or row.get("width") or row.get("manifest_width") or 0)
        h = int(row.get("source_height") or row.get("height") or row.get("manifest_height") or 0)
        if w > 0:
            widths.append(w)
        if h > 0:
            heights.append(h)
    if not ids:
        raise RuntimeError(f"raw frame manifest has no valid frame ids: {path}")
    if not widths or not heights:
        video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
        w = int(video.get("width") or 0)
        h = int(video.get("height") or 0)
    else:
        w = int(np.median(widths))
        h = int(np.median(heights))
    if w <= 0 or h <= 0:
        raise RuntimeError(f"cannot determine source image size from {path}")
    return sorted(ids), (w, h), payload


def load_unidepth_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(path)
    blob = np.load(path, allow_pickle=True)
    required = {"frame_idx", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing UniDepth intrinsics keys: {missing}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    intr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    if intr.shape != (len(frame_idx), 4):
        raise RuntimeError(f"{path} intrinsics shape mismatch: {intr.shape}")
    source_size = np.asarray(blob["source_size"], dtype=int) if "source_size" in blob.files else None
    if not np.isfinite(intr).all():
        raise RuntimeError(f"{path} contains non-finite intrinsics")
    return frame_idx, intr, source_size


def summarize(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    q = np.percentile(arr, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    med = float(q[5])
    mad = float(np.median(np.abs(arr - med)))
    return {
        "count": int(arr.size),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p10": float(q[3]),
        "p25": float(q[4]),
        "median": med,
        "p75": float(q[6]),
        "p90": float(q[7]),
        "p95": float(q[8]),
        "p99": float(q[9]),
        "max": float(q[10]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mad": mad,
        "relative_mad_fraction": float(mad / med) if med else 0.0,
        "relative_p05_p95_fraction": float((q[8] - q[2]) / med) if med else 0.0,
    }


def fov_degrees(width: int, height: int, fx: float, fy: float) -> dict[str, float]:
    return {
        "horizontal": float(2.0 * math.degrees(math.atan(width / (2.0 * fx)))) if fx > 0 else float("nan"),
        "vertical": float(2.0 * math.degrees(math.atan(height / (2.0 * fy)))) if fy > 0 else float("nan"),
    }


def select_rows(
    frame_idx: np.ndarray,
    intr: np.ndarray,
    *,
    frame_start: int | None,
    frame_end: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(frame_idx), dtype=bool)
    if frame_start is not None:
        mask &= frame_idx >= int(frame_start)
    if frame_end is not None:
        mask &= frame_idx <= int(frame_end)
    if not np.any(mask):
        raise RuntimeError(f"no UniDepth intrinsics rows in requested range {frame_start}:{frame_end}")
    return frame_idx[mask], intr[mask]


def aggregate_intrinsics(intr: np.ndarray, method: str, trim_low: float, trim_high: float) -> tuple[np.ndarray, dict[str, Any]]:
    fx, fy, cx, cy = intr.T
    focal = np.sqrt(np.maximum(1e-9, fx * fy))
    if method == "median":
        k = np.asarray([np.median(fx), np.median(fy), np.median(cx), np.median(cy)], dtype=np.float64)
        used = np.ones(len(intr), dtype=bool)
    elif method == "trimmed_mean":
        if not (0.0 <= trim_low < trim_high <= 1.0):
            raise RuntimeError(f"invalid trim quantiles: {trim_low}, {trim_high}")
        lo, hi = np.quantile(focal, [trim_low, trim_high])
        used = (focal >= lo) & (focal <= hi)
        if not np.any(used):
            raise RuntimeError("trimmed aggregation removed every row")
        k = np.asarray([np.mean(fx[used]), np.mean(fy[used]), np.mean(cx[used]), np.mean(cy[used])], dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported aggregation method: {method}")
    return k, {
        "method": method,
        "trim_low_quantile": float(trim_low) if method == "trimmed_mean" else None,
        "trim_high_quantile": float(trim_high) if method == "trimmed_mean" else None,
        "used_frame_count": int(np.count_nonzero(used)),
        "input_frame_count": int(len(intr)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    frame_ids, (width, height), raw_payload = raw_frames(args.raw_frame_manifest)
    uni_frame_idx, uni_intr, uni_source_size = load_unidepth_intrinsics(args.unidepth_npz)
    selected_frame_idx, selected_intr = select_rows(
        uni_frame_idx,
        uni_intr,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    k, aggregation = aggregate_intrinsics(
        selected_intr,
        method=args.aggregation,
        trim_low=float(args.trim_low),
        trim_high=float(args.trim_high),
    )
    if args.square_focal:
        f = float(math.sqrt(float(k[0]) * float(k[1])))
        k[0] = f
        k[1] = f
    if args.center_principal_point:
        k[2] = float(width) / 2.0
        k[3] = float(height) / 2.0
    focal_selected = np.sqrt(np.maximum(1e-9, selected_intr[:, 0] * selected_intr[:, 1]))
    focal_all = np.sqrt(np.maximum(1e-9, uni_intr[:, 0] * uni_intr[:, 1]))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = out_dir / "v19_camera_calibration_contract.json"
    intr_npz_path = out_dir / "v19_camera_calibration_intrinsics.npz"
    repeated = np.repeat(k[None, :], len(frame_ids), axis=0).astype(np.float32)
    np.savez_compressed(
        intr_npz_path,
        frame_idx=np.asarray(frame_ids, dtype=np.int32),
        intrinsics_fx_fy_cx_cy=repeated,
        constant_intrinsics_fx_fy_cx_cy=k.astype(np.float32),
        source_size=np.asarray([width, height], dtype=np.int32),
        calibration_contract_json=np.asarray([str(contract_path)]),
        intrinsics_source=np.asarray(["v19_calibration_contract_unidepth_robust_video_constant"]),
    )
    report = {
        "status": "ok",
        "method": "build_v19_calibration_contract",
        "case": args.case,
        "claim_scope": "constant pinhole intrinsics hypothesis for this V19 video/run; derived from robust aggregation of UniDepth per-frame intrinsics because dataset calibration is absent in the local shard",
        "inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "unidepth_npz": str(args.unidepth_npz),
        },
        "outputs": {
            "contract_json": str(contract_path),
            "intrinsics_npz": str(intr_npz_path),
        },
        "raw_video": raw_payload.get("video") if isinstance(raw_payload.get("video"), dict) else None,
        "source_size": {"width": int(width), "height": int(height)},
        "unidepth_source_size": uni_source_size.astype(int).tolist() if uni_source_size is not None else None,
        "intrinsics_model": "constant_pinhole_fx_fy_cx_cy",
        "intrinsics_fx_fy_cx_cy": [float(x) for x in k.tolist()],
        "focal_geom_px": float(math.sqrt(float(k[0]) * float(k[1]))),
        "fov_degrees": fov_degrees(width, height, float(k[0]), float(k[1])),
        "intrinsics_source": "v19_calibration_contract_unidepth_robust_video_constant",
        "aggregation": aggregation,
        "options": {
            "frame_start": int(args.frame_start) if args.frame_start is not None else None,
            "frame_end": int(args.frame_end) if args.frame_end is not None else None,
            "square_focal": bool(args.square_focal),
            "center_principal_point": bool(args.center_principal_point),
        },
        "diagnostics": {
            "selected_frame_count": int(len(selected_frame_idx)),
            "all_unidepth_frame_count": int(len(uni_frame_idx)),
            "all_unidepth_stats": {
                "fx": summarize(uni_intr[:, 0]),
                "fy": summarize(uni_intr[:, 1]),
                "cx": summarize(uni_intr[:, 2]),
                "cy": summarize(uni_intr[:, 3]),
                "focal_geom": summarize(focal_all),
            },
            "selected_stats": {
                "fx": summarize(selected_intr[:, 0]),
                "fy": summarize(selected_intr[:, 1]),
                "cx": summarize(selected_intr[:, 2]),
                "cy": summarize(selected_intr[:, 3]),
                "focal_geom": summarize(focal_selected),
            },
        },
    }
    med = float(np.median(focal_selected))
    delta = np.abs(focal_selected - med)
    order = np.argsort(delta)[::-1][: min(20, len(delta))]
    report["diagnostics"]["largest_selected_focal_deviations"] = [
        {
            "frame_idx": int(selected_frame_idx[i]),
            "fx": float(selected_intr[i, 0]),
            "fy": float(selected_intr[i, 1]),
            "cx": float(selected_intr[i, 2]),
            "cy": float(selected_intr[i, 3]),
            "focal_geom_px": float(focal_selected[i]),
            "delta_from_selected_median_px": float(focal_selected[i] - med),
        }
        for i in order
    ]
    write_json(contract_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--unidepth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=None, help="First frame whose UniDepth intrinsics may contribute to aggregation")
    parser.add_argument("--frame-end", type=int, default=None, help="Last frame whose UniDepth intrinsics may contribute to aggregation")
    parser.add_argument("--aggregation", choices=["median", "trimmed_mean"], default="median")
    parser.add_argument("--trim-low", type=float, default=0.10)
    parser.add_argument("--trim-high", type=float, default=0.90)
    parser.add_argument("--square-focal", action="store_true", help="Replace fx/fy with sqrt(fx*fy) after robust aggregation")
    parser.add_argument("--center-principal-point", action="store_true", help="Replace cx/cy with source image center after robust aggregation")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
