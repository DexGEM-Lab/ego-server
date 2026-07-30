#!/usr/bin/env python3
"""Build SAM2 hand point prompts from V19 HaWoR MANO projections.

This script does not create a hand state.  It converts a metric MANO candidate
into image-space prompts so SAM2 can produce an independent hand-mask/depth
measurement for downstream hand-owned-surface refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# MANO/OpenPose-style joints spread over wrist, MCPs, and fingertips.  These are
# used only as prompt seeds; SAM2/image/depth remain the hand-surface evidence.
PROMPT_JOINT_ORDER = (0, 5, 9, 13, 17, 4, 8, 12, 16, 20, 6, 10, 14, 18)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_size_from_manifest(path: Path) -> tuple[int, int]:
    payload = load_json(path)
    rows = payload.get("frames") or payload.get("frame_rows") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"raw frame manifest has no frame rows: {path}")
    row = rows[0]
    width = int(row.get("source_width") or row.get("width") or payload.get("source_width") or 0)
    height = int(row.get("source_height") or row.get("height") or payload.get("source_height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"raw frame manifest has invalid source size: {path}")
    return width, height


def calibration_intrinsics(path: Path | None, hawor: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    if path is not None:
        payload = load_json(path)
        intr = payload.get("intrinsics_fx_fy_cx_cy") or payload.get("K_fx_fy_cx_cy") or payload.get("intrinsics")
        if intr is None and "K" in payload:
            K = np.asarray(payload["K"], dtype=float)
            if K.shape == (3, 3):
                intr = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        arr = np.asarray(intr, dtype=float)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)) or np.any(arr[:2] <= 0):
            raise RuntimeError(f"invalid calibration intrinsics in {path}: {intr!r}")
        return arr, f"calibration_contract:{path}"
    focal = np.asarray(hawor["img_focal"], dtype=float).reshape(-1)
    if focal.size < 1 or not np.isfinite(focal[0]) or focal[0] <= 0:
        raise RuntimeError("HaWoR NPZ has no finite img_focal and no calibration contract was supplied")
    # Source center is filled by caller because it depends on manifest size.
    return np.asarray([float(focal[0]), float(focal[0]), np.nan, np.nan], dtype=float), "hawor_img_focal_center_from_manifest"


def world_to_camera(points_world: np.ndarray, R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (points_world - t_c2w[None, :]) @ R_c2w


def project_camera(points_camera: np.ndarray, intr: np.ndarray) -> np.ndarray:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=float)
    good = np.isfinite(points_camera).all(axis=1) & (z > 1.0e-6)
    uv[good, 0] = intr[0] * points_camera[good, 0] / z[good] + intr[2]
    uv[good, 1] = intr[1] * points_camera[good, 1] / z[good] + intr[3]
    return uv


def valid_box(box: np.ndarray, source_size: tuple[int, int]) -> bool:
    if box.shape[0] < 4 or not np.all(np.isfinite(box[:4])):
        return False
    w, h = source_size
    return bool(box[2] > box[0] and box[3] > box[1] and box[2] >= 0 and box[3] >= 0 and box[0] <= w and box[1] <= h)


def bbox_from_points(points: np.ndarray, source_size: tuple[int, int], margin_px: float) -> np.ndarray:
    w, h = source_size
    good = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
    if int(np.count_nonzero(good)) < 2:
        raise RuntimeError("too few projected points inside image for prompt bbox")
    pts = points[good]
    box = np.asarray([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=float)
    box += np.asarray([-margin_px, -margin_px, margin_px, margin_px], dtype=float)
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, w - 1.0)
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, h - 1.0)
    return box


def scale_point(pt: np.ndarray, prompt_size: tuple[int, int], source_size: tuple[int, int]) -> dict[str, float]:
    sx = prompt_size[0] / float(source_size[0])
    sy = prompt_size[1] / float(source_size[1])
    return {"x": float(np.clip(pt[0] * sx, 0.0, prompt_size[0] - 1.0)), "y": float(np.clip(pt[1] * sy, 0.0, prompt_size[1] - 1.0))}


def prompt_points_from_projection(
    joints_uv: np.ndarray,
    vertices_uv: np.ndarray,
    box: np.ndarray,
    source_size: tuple[int, int],
    prompt_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    w, h = source_size
    margin = float(args.positive_box_margin_px)
    expanded = box + np.asarray([-margin, -margin, margin, margin], dtype=float)
    expanded[[0, 2]] = np.clip(expanded[[0, 2]], 0.0, w - 1.0)
    expanded[[1, 3]] = np.clip(expanded[[1, 3]], 0.0, h - 1.0)

    pos_xy: list[np.ndarray] = []
    for idx in PROMPT_JOINT_ORDER:
        if idx >= len(joints_uv):
            continue
        pt = joints_uv[idx]
        if not np.isfinite(pt).all():
            continue
        if not (0 <= pt[0] <= w - 1 and 0 <= pt[1] <= h - 1):
            continue
        if expanded[0] <= pt[0] <= expanded[2] and expanded[1] <= pt[1] <= expanded[3]:
            pos_xy.append(pt.astype(float))
        if len(pos_xy) >= int(args.max_positive_points):
            break

    if len(pos_xy) < int(args.min_positive_points):
        candidates = vertices_uv[np.isfinite(vertices_uv).all(axis=1)]
        inside = candidates[
            (candidates[:, 0] >= expanded[0]) & (candidates[:, 0] <= expanded[2]) &
            (candidates[:, 1] >= expanded[1]) & (candidates[:, 1] <= expanded[3])
        ]
        if len(inside):
            # Spread extra prompts across the projected surface by sorted linear index.
            order = np.linspace(0, len(inside) - 1, num=min(int(args.max_positive_points) - len(pos_xy), len(inside)), dtype=int)
            for pt in inside[order]:
                pos_xy.append(pt.astype(float))
                if len(pos_xy) >= int(args.max_positive_points):
                    break

    if len(pos_xy) < int(args.min_positive_points):
        raise RuntimeError(f"only {len(pos_xy)} positive hand prompt points; required {args.min_positive_points}")

    # Negative points outside the hand box but near enough to suppress hand/scene leakage.
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    gap = max(float(args.negative_margin_px), 1.0)
    neg_xy = [
        np.asarray([box[0] - gap, cy], dtype=float),
        np.asarray([box[2] + gap, cy], dtype=float),
        np.asarray([cx, box[1] - gap], dtype=float),
        np.asarray([cx, box[3] + gap], dtype=float),
    ]
    neg_xy = [np.asarray([np.clip(pt[0], 0.0, w - 1.0), np.clip(pt[1], 0.0, h - 1.0)], dtype=float) for pt in neg_xy]

    positives = [scale_point(pt, prompt_size, source_size) for pt in pos_xy]
    negatives = [scale_point(pt, prompt_size, source_size) for pt in neg_xy]
    diag = {
        "positive_points": len(positives),
        "negative_points": len(negatives),
        "source_bbox_xyxy": [float(v) for v in box],
        "expanded_positive_bbox_xyxy": [float(v) for v in expanded],
    }
    return positives, negatives, diag


def frame_indices(args: argparse.Namespace) -> list[int]:
    explicit = list(args.prompt_frames or [])
    if explicit:
        frames = sorted({int(x) for x in explicit})
    else:
        stride = max(1, int(args.prompt_stride))
        frames = list(range(int(args.frame_start), int(args.frame_end) + 1, stride))
        if frames[-1] != int(args.frame_end):
            frames.append(int(args.frame_end))
    return [idx for idx in frames if int(args.frame_start) <= idx <= int(args.frame_end)]


def build_side_payload(side: str, hawor: np.lib.npyio.NpzFile, idx_by_frame: dict[int, int], intr: np.ndarray, intr_source: str, source_size: tuple[int, int], prompt_size: tuple[int, int], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    diagnostics = []
    frame_idx_arr = np.asarray(hawor["frame_idx"], dtype=int)
    R_all = np.asarray(hawor["R_c2w"], dtype=float)
    t_all = np.asarray(hawor["t_c2w"], dtype=float)
    joints_all = np.asarray(hawor[f"{side}_joints_world_m"], dtype=float)
    verts_all = np.asarray(hawor[f"{side}_vertices_world_m"], dtype=float)
    valid_all = np.asarray(hawor[f"{side}_valid"], dtype=bool)
    det_all = np.asarray(hawor.get(f"{side}_det_box_xyxyscore"), dtype=float) if f"{side}_det_box_xyxyscore" in hawor.files else None
    detected_all = np.asarray(hawor.get(f"{side}_detected_same_frame"), dtype=bool) if f"{side}_detected_same_frame" in hawor.files else np.zeros_like(valid_all, dtype=bool)

    for frame_idx in frame_indices(args):
        if frame_idx not in idx_by_frame:
            diagnostics.append({"frame_idx": frame_idx, "target_visible": False, "reason": "missing_hawor_frame"})
            continue
        i = idx_by_frame[frame_idx]
        if not bool(valid_all[i]):
            diagnostics.append({"frame_idx": frame_idx, "target_visible": False, "reason": "invalid_hawor_hand"})
            continue
        R = R_all[i]
        t = t_all[i]
        joints_cam = world_to_camera(joints_all[i], R, t)
        verts_cam = world_to_camera(verts_all[i], R, t)
        joints_uv = project_camera(joints_cam, intr)
        verts_uv = project_camera(verts_cam[:: max(1, int(args.vertex_prompt_stride))], intr)
        if det_all is not None and bool(detected_all[i]) and valid_box(det_all[i, :4], source_size):
            box = det_all[i, :4].astype(float)
            source = "hawor_same_frame_detector_box"
        else:
            box = bbox_from_points(np.vstack([joints_uv, verts_uv]), source_size, float(args.projected_bbox_margin_px))
            source = "projected_mano_bbox"
        positives, negatives, diag = prompt_points_from_projection(joints_uv, verts_uv, box, source_size, prompt_size, args)
        row = {
            "frame_idx": int(frame_idx),
            "target_visible": True,
            "confidence": float(det_all[i, 4]) if det_all is not None and det_all.shape[1] >= 5 and np.isfinite(det_all[i, 4]) else 1.0,
            "bbox_xyxy": [float(v * prompt_size[0] / source_size[0]) if j in (0, 2) else float(v * prompt_size[1] / source_size[1]) for j, v in enumerate(box)],
            "positive_points": positives,
            "negative_points": negatives,
            "visual_evidence": "HaWoR projected MANO/detector evidence used only as SAM2 hand-mask prompt seeds.",
        }
        rows.append(row)
        diagnostics.append({
            "frame_idx": int(frame_idx),
            "target_visible": True,
            "bbox_source": source,
            "same_frame_detection": bool(detected_all[i]),
            "camera_z_median_m": float(np.median(joints_cam[:, 2])),
            **diag,
        })
    if len(rows) < int(args.min_prompt_frames):
        raise RuntimeError(f"{side} hand produced {len(rows)} prompt frames; required {args.min_prompt_frames}; diagnostics={diagnostics[:20]}")
    return {
        "status": "ok",
        "backend": "V19 HaWoR MANO projection to SAM2 hand prompts",
        "track_id": f"hand_{side}_hawor_prompt",
        "target_track_id": f"hand_{side}_hawor_prompt",
        "description": f"{side} hand mask prompts from calibrated HaWoR projection; prompts are not accepted hand state.",
        "prompt_image_width": int(prompt_size[0]),
        "prompt_image_height": int(prompt_size[1]),
        "source_image_width": int(source_size[0]),
        "source_image_height": int(source_size[1]),
        "object_plan_payload": {"active_intervals": [{"start_frame": int(args.frame_start), "end_frame": int(args.frame_end)}]},
        "hand_side": side,
        "hawor_npz": str(args.hawor_npz),
        "intrinsics_fx_fy_cx_cy": [float(v) for v in intr],
        "intrinsics_source": intr_source,
        "point_prompts": rows,
        "diagnostics": diagnostics,
        "claim_scope": "Prompt generation for SAM2 hand-mask measurement only; does not constitute hand pose/contact acceptance.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    hawor = np.load(args.hawor_npz, allow_pickle=True)
    source_size = source_size_from_manifest(args.raw_frame_manifest)
    prompt_width = int(args.prompt_image_width or round(source_size[0] / 2))
    prompt_size = (prompt_width, int(round(prompt_width * source_size[1] / source_size[0])))
    intr, intr_source = calibration_intrinsics(args.calibration_contract, hawor)
    if not np.isfinite(intr[2:]).all():
        intr[2] = 0.5 * source_size[0]
        intr[3] = 0.5 * source_size[1]
    frames = np.asarray(hawor["frame_idx"], dtype=int)
    idx_by_frame = {int(v): i for i, v in enumerate(frames)}
    sides = list(args.sides)
    summary: dict[str, Any] = {
        "status": "ok",
        "hawor_npz": str(args.hawor_npz),
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "calibration_contract": None if args.calibration_contract is None else str(args.calibration_contract),
        "output_root": str(args.output_root),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "sides": sides,
        "tracks": {},
    }
    for side in sides:
        payload = build_side_payload(side, hawor, idx_by_frame, intr, intr_source, source_size, prompt_size, args)
        out_path = args.output_root / payload["track_id"] / "object_point_prompts_vlm.json"
        save_json(out_path, payload)
        summary["tracks"][side] = {
            "track_id": payload["track_id"],
            "prompt_path": str(out_path),
            "prompt_frames": len(payload["point_prompts"]),
            "diagnostic_visible_frames": int(sum(1 for row in payload["diagnostics"] if row.get("target_visible"))),
        }
    save_json(args.output_root / "v19_hawor_hand_sam2_prompt_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--calibration-contract", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--prompt-frames", type=int, nargs="*", default=None)
    parser.add_argument("--prompt-stride", type=int, default=8)
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--prompt-image-width", type=int, default=None)
    parser.add_argument("--min-prompt-frames", type=int, default=2)
    parser.add_argument("--min-positive-points", type=int, default=4)
    parser.add_argument("--max-positive-points", type=int, default=10)
    parser.add_argument("--vertex-prompt-stride", type=int, default=12)
    parser.add_argument("--projected-bbox-margin-px", type=float, default=25.0)
    parser.add_argument("--positive-box-margin-px", type=float, default=18.0)
    parser.add_argument("--negative-margin-px", type=float, default=18.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
