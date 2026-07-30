#!/usr/bin/env python3
"""Build a HaWoR-compatible WiLoR hand NPZ with root depth from V19 depth images.

This is an experimental visible-frame hand-root solver.  It does not use object
contact.  For each WiLoR hand detection, it:

1. keeps WiLoR's MANO-local joints/vertices/articulation;
2. projects WiLoR vertices with WiLoR's own canonical camera to select visible
   hand-support pixels;
3. reads V19/UniDepth metric depth at those pixels;
4. estimates a camera-frame translation z from depth - local_vertex_z;
5. estimates x/y from WiLoR 2D joints under the target V19 intrinsics;
6. maps the resulting camera-frame hand to the supplied trajectory from the
   HaWoR-compatible NPZ schema.

The output is a candidate hand stream only.  HOT3D/visual evaluation must decide
whether the depth source is metrically reliable enough for downstream contact.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> tuple[np.ndarray, str]:
    data = load_json(path)
    intr = np.asarray(data.get("intrinsics_fx_fy_cx_cy"), dtype=np.float64).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0 or intr[1] <= 0:
        raise RuntimeError(f"invalid intrinsics_fx_fy_cx_cy in {path}")
    return intr, str(data.get("intrinsics_source") or data.get("method") or "calibration_contract")


def project(points_cam: np.ndarray, intr: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(pts).all(axis=1) & (z > 1.0e-8)
    uv[good, 0] = pts[good, 0] / z[good] * intr[0] + intr[2]
    uv[good, 1] = pts[good, 1] / z[good] * intr[1] + intr[3]
    return uv


def camera_to_world(points_cam: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return np.asarray(points_cam, dtype=np.float64) @ np.asarray(r_c2w, dtype=np.float64).T + np.asarray(t_c2w, dtype=np.float64)[None, :]


def read_depth_m(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        depth = img.astype(np.float32) / 1000.0
    else:
        depth = img.astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.02) | (depth > 10.0)] = np.nan
    return depth


def choose_wilor_hand(frame: dict[str, Any], side: str, min_score: float) -> dict[str, Any] | None:
    hands = frame.get("raw_hands") if isinstance(frame.get("raw_hands"), list) else []
    cand = [h for h in hands if isinstance(h, dict) and str(h.get("side")) == side and float(h.get("detector_score", -math.inf)) >= min_score]
    if not cand:
        return None
    cand.sort(key=lambda h: float(h.get("detector_score", -math.inf)), reverse=True)
    return cand[0]


def matrix_to_rotvec(mat_like: Any) -> list[float] | None:
    arr = np.asarray(mat_like, dtype=np.float64)
    if arr.shape == (1, 3, 3):
        arr = arr[0]
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        return None
    return Rotation.from_matrix(arr).as_rotvec().astype(float).tolist()


def hand_pose_to_rotvecs(raw: Any) -> list[float] | None:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (15, 3, 3) or not np.isfinite(arr).all():
        return None
    return Rotation.from_matrix(arr).as_rotvec().reshape(-1).astype(float).tolist()


def sample_depth_at_uv(depth: np.ndarray, uv: np.ndarray, radius: int) -> np.ndarray:
    h, w = depth.shape[:2]
    vals: list[float] = []
    for u, v in np.asarray(uv, dtype=np.float64):
        if not np.isfinite([u, v]).all():
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        patch = depth[max(0, y - radius): min(h, y + radius + 1), max(0, x - radius): min(w, x + radius + 1)]
        pv = patch[np.isfinite(patch)]
        if pv.size:
            vals.append(float(np.median(pv)))
    return np.asarray(vals, dtype=np.float64)


def estimate_translation_from_depth(
    *,
    depth: np.ndarray,
    local_joints: np.ndarray,
    local_vertices: np.ndarray,
    wilor_joints2d: np.ndarray,
    wilor_cam_t: np.ndarray,
    wilor_focal: float,
    target_intr: np.ndarray,
    depth_quantile: float,
    sample_radius: int,
    support: str,
    image_center: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    local_joints = np.asarray(local_joints, dtype=np.float64)
    local_vertices = np.asarray(local_vertices, dtype=np.float64)
    wilor_cam_t = np.asarray(wilor_cam_t, dtype=np.float64).reshape(3)
    cx0, cy0 = image_center
    if support == "vertices":
        raw_pts = local_vertices + wilor_cam_t[None, :]
        raw_uv = project(raw_pts, np.asarray([wilor_focal, wilor_focal, cx0, cy0], dtype=np.float64))
        dvals = sample_depth_at_uv(depth, raw_uv, sample_radius)
        zloc = []
        h, w = depth.shape[:2]
        for (u, v), zv in zip(raw_uv, local_vertices[:, 2], strict=False):
            x = int(round(float(u))) if np.isfinite(u) else -1
            y = int(round(float(v))) if np.isfinite(v) else -1
            if 0 <= x < w and 0 <= y < h:
                patch = depth[max(0, y - sample_radius): min(h, y + sample_radius + 1), max(0, x - sample_radius): min(w, x + sample_radius + 1)]
                if np.isfinite(patch).any():
                    zloc.append(float(zv))
        zloc_arr = np.asarray(zloc, dtype=np.float64)
    elif support == "joints":
        dvals = sample_depth_at_uv(depth, wilor_joints2d, sample_radius)
        zloc = []
        h, w = depth.shape[:2]
        for (u, v), zj in zip(wilor_joints2d, local_joints[:, 2], strict=False):
            x = int(round(float(u))) if np.isfinite(u) else -1
            y = int(round(float(v))) if np.isfinite(v) else -1
            if 0 <= x < w and 0 <= y < h:
                patch = depth[max(0, y - sample_radius): min(h, y + sample_radius + 1), max(0, x - sample_radius): min(w, x + sample_radius + 1)]
                if np.isfinite(patch).any():
                    zloc.append(float(zj))
        zloc_arr = np.asarray(zloc, dtype=np.float64)
    else:
        raise RuntimeError(f"unknown depth support {support}")
    n = min(len(dvals), len(zloc_arr))
    if n < 12:
        raise RuntimeError(f"too few depth support samples: {n}")
    z_candidates = dvals[:n] - zloc_arr[:n]
    z_candidates = z_candidates[np.isfinite(z_candidates) & (z_candidates > 0.03) & (z_candidates < 3.0)]
    if z_candidates.size < 12:
        raise RuntimeError(f"too few finite z candidates: {z_candidates.size}")
    tz = float(np.quantile(z_candidates, depth_quantile))
    # Given tz, each 2D joint gives one tx and one ty consistent with target K.
    uv = np.asarray(wilor_joints2d, dtype=np.float64)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(local_joints).all(axis=1) & ((local_joints[:, 2] + tz) > 0.02)
    if np.count_nonzero(valid) < 8:
        raise RuntimeError("too few finite 2D joints for x/y translation")
    z = local_joints[valid, 2] + tz
    tx_i = (uv[valid, 0] - target_intr[2]) * z / target_intr[0] - local_joints[valid, 0]
    ty_i = (uv[valid, 1] - target_intr[3]) * z / target_intr[1] - local_joints[valid, 1]
    tx = float(np.median(tx_i))
    ty = float(np.median(ty_i))
    t = np.asarray([tx, ty, tz], dtype=np.float64)
    pred_uv = project(local_joints + t[None, :], target_intr)
    err = np.linalg.norm(pred_uv[valid] - uv[valid], axis=1)
    z_res = z_candidates - tz
    return t, {
        "status": "ok",
        "support": support,
        "depth_quantile": float(depth_quantile),
        "sample_radius_px": int(sample_radius),
        "depth_support_count": int(z_candidates.size),
        "translation_camera_m": t.astype(float).tolist(),
        "depth_z_candidate_m": {
            "q10": float(np.quantile(z_candidates, 0.10)),
            "q20": float(np.quantile(z_candidates, 0.20)),
            "median": float(np.median(z_candidates)),
            "q80": float(np.quantile(z_candidates, 0.80)),
            "q90": float(np.quantile(z_candidates, 0.90)),
        },
        "depth_residual_to_selected_z_m": {
            "median_abs": float(np.median(np.abs(z_res))),
            "p90_abs": float(np.quantile(np.abs(z_res), 0.90)),
        },
        "reprojection_error_px": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "p90": float(np.quantile(err, 0.90)),
            "max": float(np.max(err)),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    hawor_blob = np.load(args.trajectory_npz, allow_pickle=True)
    hawor = {key: np.asarray(hawor_blob[key]) for key in hawor_blob.files}
    required = ["frame_idx", "R_c2w", "t_c2w", "left_vertices_world_m", "right_vertices_world_m", "left_joints_world_m", "right_joints_world_m"]
    missing = [k for k in required if k not in hawor]
    if missing:
        raise RuntimeError(f"trajectory NPZ missing keys {missing}: {args.trajectory_npz}")
    wilor = load_json(args.wilor_raw)
    wilor_by_frame = {int(row.get("frame_idx")): row for row in wilor.get("frames", []) if isinstance(row, dict) and row.get("frame_idx") is not None}
    intr, intr_source = load_calibration(args.calibration_contract)
    out = {key: np.array(value, copy=True) for key, value in hawor.items()}
    frame_idx = np.asarray(hawor["frame_idx"], dtype=np.int32)
    n = len(frame_idx)
    provenance = {"left": [], "right": []}
    reproj_median = {"left": np.full(n, np.nan, np.float32), "right": np.full(n, np.nan, np.float32)}
    depth_count = {"left": np.zeros(n, np.int32), "right": np.zeros(n, np.int32)}
    fitted_t = {"left": np.full((n, 3), np.nan, np.float32), "right": np.full((n, 3), np.nan, np.float32)}
    row_reports: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    depth_cache: dict[int, np.ndarray] = {}
    cx0 = float((wilor.get("video") or {}).get("width") or 1408) / 2.0
    cy0 = float((wilor.get("video") or {}).get("height") or 1408) / 2.0

    for pos, idx_raw in enumerate(frame_idx.tolist()):
        idx = int(idx_raw)
        wframe = wilor_by_frame.get(idx)
        r_c2w = np.asarray(hawor["R_c2w"][pos], dtype=np.float64)
        t_c2w = np.asarray(hawor["t_c2w"][pos], dtype=np.float64)
        if idx not in depth_cache:
            depth_cache[idx] = read_depth_m(args.depth_dir / f"{idx:06d}.png")
        depth = depth_cache[idx]
        for side in ("left", "right"):
            selected = choose_wilor_hand(wframe or {}, side, args.min_wilor_score) if wframe is not None else None
            if selected is None:
                provenance[side].append("trajectory_fallback_no_wilor_detection")
                counts[f"{side}_fallback_no_wilor"] += 1
                continue
            try:
                local_joints = np.asarray(selected["joints3d_camera"], dtype=np.float64)
                local_vertices = np.asarray(selected["vertices_camera"], dtype=np.float64)
                joints2d = np.asarray(selected["joints2d"], dtype=np.float64)
                cam_t = np.asarray(selected["cam_t"], dtype=np.float64)
                source_focal = float(selected.get("focal_length") or 0.0)
                t_cam, fit = estimate_translation_from_depth(
                    depth=depth,
                    local_joints=local_joints,
                    local_vertices=local_vertices,
                    wilor_joints2d=joints2d,
                    wilor_cam_t=cam_t,
                    wilor_focal=source_focal,
                    target_intr=intr,
                    depth_quantile=args.depth_quantile,
                    sample_radius=args.depth_sample_radius_px,
                    support=args.depth_support,
                    image_center=(cx0, cy0),
                )
                joints_cam = local_joints + t_cam[None, :]
                verts_cam = local_vertices + t_cam[None, :]
                if not (np.isfinite(joints_cam).all() and np.isfinite(verts_cam).all() and np.all(joints_cam[:, 2] > 0.02)):
                    raise RuntimeError("non-finite or non-positive-depth fitted hand")
                joints_world = camera_to_world(joints_cam, r_c2w, t_c2w).astype(np.float32)
                verts_world = camera_to_world(verts_cam, r_c2w, t_c2w).astype(np.float32)
                out[f"{side}_joints_world_m"][pos] = joints_world
                out[f"{side}_vertices_world_m"][pos] = verts_world
                out[f"{side}_valid"][pos] = np.uint8(1)
                if f"{side}_detected_same_frame" in out:
                    out[f"{side}_detected_same_frame"][pos] = np.uint8(1)
                if f"{side}_det_box_xyxyscore" in out:
                    box = np.asarray(selected.get("bbox_xyxy") or [np.nan] * 4, dtype=np.float32).reshape(-1)
                    score = float(selected.get("detector_score", np.nan))
                    if box.shape[0] >= 4:
                        out[f"{side}_det_box_xyxyscore"][pos] = np.asarray([box[0], box[1], box[2], box[3], score], dtype=np.float32)
                if f"{side}_trans_world_m" in out:
                    out[f"{side}_trans_world_m"][pos] = camera_to_world(t_cam[None, :], r_c2w, t_c2w)[0].astype(np.float32)
                params = selected.get("mano_params") if isinstance(selected.get("mano_params"), dict) else {}
                root_vec = matrix_to_rotvec(params.get("global_orient"))
                pose_vec = hand_pose_to_rotvecs(params.get("hand_pose"))
                betas = np.asarray(params.get("betas"), dtype=np.float32).reshape(-1) if params.get("betas") is not None else None
                if root_vec is not None and f"{side}_root_orient_axis_angle" in out:
                    out[f"{side}_root_orient_axis_angle"][pos] = np.asarray(root_vec, dtype=np.float32)
                if pose_vec is not None and f"{side}_hand_pose_axis_angle" in out:
                    out[f"{side}_hand_pose_axis_angle"][pos] = np.asarray(pose_vec, dtype=np.float32)
                if betas is not None and betas.shape[0] >= 10 and f"{side}_betas" in out:
                    out[f"{side}_betas"][pos] = betas[:10].astype(np.float32)
                source_label = f"wilor_visible_depth_root_{args.depth_support}_q{args.depth_quantile:.2f}_v19_depth"
                provenance[side].append(source_label)
                reproj_median[side][pos] = np.float32(fit["reprojection_error_px"]["median"])
                depth_count[side][pos] = int(fit["depth_support_count"])
                fitted_t[side][pos] = t_cam.astype(np.float32)
                counts[f"{side}_wilor_depth_root"] += 1
                if len(row_reports) < args.report_row_limit:
                    row_reports.append({
                        "frame_idx": idx,
                        "side": side,
                        "source": source_label,
                        "wilor_score": float(selected.get("detector_score", np.nan)),
                        "wilor_bbox_xyxy": selected.get("bbox_xyxy"),
                        "fit": fit,
                    })
            except Exception as exc:
                provenance[side].append(f"trajectory_fallback_depth_root_rejected:{type(exc).__name__}:{str(exc)[:160]}")
                counts[f"{side}_fallback_rejected"] += 1
                if len(row_reports) < args.report_row_limit:
                    row_reports.append({"frame_idx": idx, "side": side, "source": "fallback", "reason": f"{type(exc).__name__}: {exc}"})
    for side in ("left", "right"):
        out[f"{side}_hybrid_source"] = np.asarray(provenance[side])
        out[f"{side}_wilor_depth_root_reprojection_median_px"] = reproj_median[side]
        out[f"{side}_wilor_depth_root_support_count"] = depth_count[side]
        out[f"{side}_wilor_depth_root_translation_camera_m"] = fitted_t[side]
    policy = {
        "method": "build_v19_wilor_depth_root_hand_npz",
        "visible_source": "WiLoR MANO plus V19 depth-derived root translation",
        "trajectory_source_npz": str(args.trajectory_npz),
        "depth_dir": str(args.depth_dir),
        "depth_support": args.depth_support,
        "depth_quantile": float(args.depth_quantile),
        "depth_sample_radius_px": int(args.depth_sample_radius_px),
        "calibration_contract": str(args.calibration_contract),
        "intrinsics_source": intr_source,
        "claim_scope": "candidate hand root-depth stream only; not contact/nonpenetration/object evidence",
    }
    out["hybrid_policy"] = np.asarray([json.dumps(policy, sort_keys=True)])
    out["hybrid_created_unix_s"] = np.asarray([time.time()], dtype=np.float64)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    report = {
        "status": "ok",
        "method": "build_v19_wilor_depth_root_hand_npz",
        "inputs": {
            "wilor_raw": str(args.wilor_raw),
            "trajectory_npz": str(args.trajectory_npz),
            "depth_dir": str(args.depth_dir),
            "calibration_contract": str(args.calibration_contract),
        },
        "outputs": {"hybrid_npz": str(args.output_npz)},
        "policy": policy,
        "counts": dict(counts),
        "frame_count": int(n),
        "row_reports_preview": row_reports,
        "elapsed_s": float(time.time() - started),
    }
    if args.report_json:
        write_json(args.report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wilor-raw", type=Path, required=True)
    p.add_argument("--trajectory-npz", type=Path, required=True, help="HaWoR-compatible NPZ supplying frame_idx/R_c2w/t_c2w and fallback schema")
    p.add_argument("--depth-dir", type=Path, required=True, help="Directory containing per-frame uint16 depth PNGs in millimetres")
    p.add_argument("--calibration-contract", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument("--min-wilor-score", type=float, default=0.30)
    p.add_argument("--depth-support", choices=("vertices", "joints"), default="vertices")
    p.add_argument("--depth-quantile", type=float, default=0.20)
    p.add_argument("--depth-sample-radius-px", type=int, default=2)
    p.add_argument("--report-row-limit", type=int, default=24)
    return p.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2)[:16000])


if __name__ == "__main__":
    main()
