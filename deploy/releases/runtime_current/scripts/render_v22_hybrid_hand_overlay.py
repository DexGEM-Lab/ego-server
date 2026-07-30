#!/usr/bin/env python3
"""Render a V22 overlay from the hybrid metric hand NPZ.

This renderer is a QC projection of the current numeric hand state. It consumes
the canonical calibration contract and fails if K is unavailable; it does not use
WiLoR 2D points as the rendered state.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS = {"left": (70, 230, 80), "right": (40, 210, 255)}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def load_k(path: Path) -> list[float]:
    contract = load_json(path)
    values = contract.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise RuntimeError(f"calibration contract lacks intrinsics_fx_fy_cx_cy: {path}")
    parsed = [finite_positive(v) for v in values]
    if any(v is None for v in parsed):
        raise RuntimeError(f"calibration contract contains invalid K: {values}")
    return [float(v) for v in parsed if v is not None]


def scale_k(k: list[float], frame: dict[str, Any]) -> tuple[float, float, float, float]:
    source_w = finite_positive(frame.get("source_width")) or finite_positive(frame.get("width"))
    source_h = finite_positive(frame.get("source_height")) or finite_positive(frame.get("height"))
    render_w = finite_positive(frame.get("manifest_width")) or source_w
    render_h = finite_positive(frame.get("manifest_height")) or source_h
    if source_w is None or source_h is None or render_w is None or render_h is None:
        raise RuntimeError("raw frame manifest row lacks source/render dimensions for K scaling")
    sx = render_w / source_w
    sy = render_h / source_h
    return float(k[0] * sx), float(k[1] * sy), float(k[2] * sx), float(k[3] * sy)


def world_to_camera(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (points_world.astype(np.float64) - t_c2w.astype(np.float64)[None, :]) @ r_c2w.astype(np.float64)


def project(points_cam: np.ndarray, k_scaled: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = k_scaled
    z = points_cam[:, 2]
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    uv[valid, 0] = points_cam[valid, 0] / z[valid] * fx + cx
    uv[valid, 1] = points_cam[valid, 1] / z[valid] * fy + cy
    return uv


def draw_hand(image: np.ndarray, uv: np.ndarray, side: str, degraded: bool) -> int:
    color = COLORS.get(side, (255, 255, 255))
    if degraded:
        color = tuple(max(0, int(c * 0.65)) for c in color)
    h, w = image.shape[:2]
    drawn = 0
    for a, b in HAND_EDGES:
        pa = uv[a]
        pb = uv[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
            xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
            if -50 <= xa <= w + 50 and -50 <= xb <= w + 50 and -50 <= ya <= h + 50 and -50 <= yb <= h + 50:
                cv2.line(image, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
                drawn += 1
    for p in uv:
        if np.isfinite(p).all():
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if -20 <= x <= w + 20 and -20 <= y <= h + 20:
                cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
                drawn += 1
    return drawn


def encode_video(frame_dir: Path, output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}", "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output),
    ]
    subprocess.run(cmd, check=True)


def ffprobe_count(path: Path) -> int | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    raw_manifest = load_json(args.raw_frame_manifest or (run_root / "input" / "raw_frame_manifest" / "manifest.json"))
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    if not frames:
        raise RuntimeError("raw frame manifest contains no frames")
    k = load_k(args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"))
    hybrid_npz = args.hybrid_npz or (run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz")
    blob = np.load(hybrid_npz, allow_pickle=True)
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    frame_to_pos = {int(v): i for i, v in enumerate(frame_idx.tolist())}
    fps = finite_positive(raw_manifest.get("fps")) or finite_positive((raw_manifest.get("video") or {}).get("fps")) or 30.0
    frame_dir = args.frame_dir or (run_root / "renders" / "hybrid_overlay_frames")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (run_root / "renders" / "v22_hybrid_hand_overlay.mp4")
    counts = {"frames": 0, "frames_with_projected_hands": 0, "hands_projected": 0, "degraded_hands_projected": 0}
    for frame in frames:
        idx = int(frame.get("frame_idx", frame.get("index", counts["frames"])))
        image_path = Path(str(frame.get("rgb") or frame.get("raw_frame_path")))
        if not image_path.exists():
            raise FileNotFoundError(f"raw frame image missing: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read frame image: {image_path}")
        pos = frame_to_pos.get(idx)
        frame_hands = 0
        if pos is not None:
            k_scaled = scale_k(k, frame)
            r_c2w = np.asarray(blob["R_c2w"])[pos]
            t_c2w = np.asarray(blob["t_c2w"])[pos]
            for side in ("left", "right"):
                valid_key = f"{side}_valid"
                if valid_key in blob.files and int(np.asarray(blob[valid_key])[pos]) == 0:
                    continue
                joints_w = np.asarray(blob[f"{side}_joints_world_m"])[pos].astype(np.float64)
                joints_c = world_to_camera(joints_w, r_c2w, t_c2w)
                uv = project(joints_c, k_scaled)
                med_key = f"{side}_wilor_fit_reprojection_median_px"
                med = float(np.asarray(blob[med_key])[pos]) if med_key in blob.files and np.isfinite(np.asarray(blob[med_key])[pos]) else None
                degraded = med is not None and med > float(args.degraded_residual_px)
                drawn = draw_hand(image, uv, side, degraded)
                if drawn > 0:
                    frame_hands += 1
                    counts["hands_projected"] += 1
                    if degraded:
                        counts["degraded_hands_projected"] += 1
        label = f"V22 hybrid hand candidate | frame {idx:06d} | projected hands {frame_hands}/2"
        cv2.putText(image, label, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        if counts["degraded_hands_projected"]:
            cv2.putText(image, "degraded rows dimmed: large WiLoR-vs-HaWoR residual", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frame_dir / f"{idx:06d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        counts["frames"] += 1
        if frame_hands:
            counts["frames_with_projected_hands"] += 1
    encode_video(frame_dir, output, float(fps))
    report = {
        "status": "ok",
        "method": "render_v22_hybrid_hand_overlay",
        "claim_scope": "QC render of hybrid metric hand candidate rows under canonical K; visualizes current hand state but does not certify 5mm accuracy.",
        "run_root": str(run_root),
        "hybrid_npz": str(hybrid_npz),
        "calibration_contract": str(args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")),
        "output_video": str(output),
        "frame_count": len(frames),
        "video_frame_count": ffprobe_count(output),
        "draw_counts": counts,
    }
    write_json(args.report_json or (output.parent / "v22_hybrid_hand_overlay_report.json"), report)
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hybrid-npz", type=Path, default=None)
    parser.add_argument("--raw-frame-manifest", type=Path, default=None)
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--degraded-residual-px", type=float, default=50.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
