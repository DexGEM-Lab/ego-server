#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
SIDE_NAMES = {0: "left", 1: "right"}
BASE_BGR = {"left": (60, 230, 60), "right": (30, 170, 255)}
INFERRED_BGR = {"left": (80, 130, 80), "right": (40, 100, 150)}
BOUNDARY_BGR = {"left": (255, 0, 255), "right": (255, 0, 255)}


def project(points: np.ndarray, intr: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intr.astype(float)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = fx * points[:, 0] / points[:, 2] + cx
    uv[:, 1] = fy * points[:, 1] / points[:, 2] + cy
    return uv


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.55, thickness: int = 1) -> None:
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def inside_fraction(uv: np.ndarray, width: int, height: int) -> float:
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return float(np.mean(inside)) if uv.size else 0.0


def row_state(support_npz: np.lib.npyio.NpzFile, side: str, frame: int) -> tuple[str, bool, bool]:
    detected = bool(np.asarray(support_npz[f"{side}_detected_same_frame"])[frame]) if f"{side}_detected_same_frame" in support_npz.files else False
    boundary = bool(np.asarray(support_npz[f"{side}_temporal_boundary_filled"])[frame]) if f"{side}_temporal_boundary_filled" in support_npz.files else False
    if boundary:
        return "boundary-fill", detected, boundary
    if detected:
        return "observed", detected, boundary
    return "inferred", detected, boundary


def draw_hand(img: np.ndarray, uv: np.ndarray, side: str, state: str, label_extra: str) -> None:
    if state == "observed":
        color = BASE_BGR[side]
        thickness = 3
    elif state == "boundary-fill":
        color = BOUNDARY_BGR[side]
        thickness = 2
    else:
        color = INFERRED_BGR[side]
        thickness = 1
    pts = np.round(uv).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    radius = 4 if state == "observed" else 3
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    wrist = pts[0]
    x = int(np.clip(wrist[0] + 8, 0, img.shape[1] - 260))
    y = int(np.clip(wrist[1] + 20, 24, img.shape[0] - 8))
    draw_text(img, f"HaWoR {side} {state}{label_extra}", (x, y), color, scale=0.5, thickness=1)


def load_bridge_rows(bridge_npz: Path) -> dict[int, dict[str, np.ndarray]]:
    b = np.load(bridge_npz)
    rows: dict[int, dict[str, np.ndarray]] = {}
    for i, f in enumerate(np.asarray(b["frame_idx"], dtype=np.int32)):
        side = SIDE_NAMES.get(int(np.asarray(b["side"])[i]), str(int(np.asarray(b["side"])[i])))
        rows.setdefault(int(f), {})[side] = np.asarray(b["joints_hawor_camera_m"][i], dtype=np.float64)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    rows = load_bridge_rows(args.bridge_npz)
    support = np.load(args.support_npz, allow_pickle=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    focal = float(np.asarray(support["img_focal"]).reshape(-1)[0]) if "img_focal" in support.files else 2304.0
    intr = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer {args.output_video}")
    counts = {"observed": 0, "inferred": 0, "boundary-fill": 0, "missing_bridge_row": 0, "nonpositive_depth": 0}
    inside_values: dict[str, list[float]] = {"observed": [], "inferred": [], "boundary-fill": []}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rows = rows.get(frame_idx, {})
        for side in ("left", "right"):
            joints = frame_rows.get(side)
            if joints is None:
                counts["missing_bridge_row"] += 1
                continue
            state, detected, boundary = row_state(support, side, frame_idx)
            if np.any(joints[:, 2] <= 1e-6) or not np.isfinite(joints).all():
                counts["nonpositive_depth"] += 1
                continue
            uv = project(joints, intr)
            inside = inside_fraction(uv, width, height)
            inside_values[state].append(inside)
            counts[state] += 1
            extra = f" in={inside:.2f}"
            draw_hand(frame, uv, side, state, extra)
        draw_text(frame, f"{args.case} frame {frame_idx}/{frame_count-1} observed=solid inferred=dim boundary=magenta", (12, 28), (255, 255, 255), scale=0.65, thickness=1)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    summary = {}
    for state, vals in inside_values.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[state] = {"count": int(arr.size), "mean_inside": float(arr.mean()) if arr.size else None, "inside_lt_0_5": int(np.count_nonzero(arr < 0.5)) if arr.size else 0}
    report = {
        "method": "render_v18_hawor_support_overlay",
        "case": args.case,
        "video": str(args.video),
        "bridge_npz": str(args.bridge_npz),
        "support_npz": str(args.support_npz),
        "output_video": str(args.output_video),
        "frames_written": frame_idx,
        "video_frame_count": frame_count,
        "fps": fps,
        "size": [width, height],
        "counts": counts,
        "inside_summary": summary,
        "claim_scope": "support_aware_HaWoR_overlay_render_no_foundation_acceptance_no_downstream_physics",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bridge-npz", type=Path, required=True)
    parser.add_argument("--support-npz", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
