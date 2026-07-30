#!/usr/bin/env python3
"""Build a HaWoR-compatible hybrid hand NPZ from WiLoR visible rows plus HaWoR fallback.

The output intentionally preserves the existing V19/HaWoR camera trajectory and
NPZ schema so downstream V19 base-annotation, contact, and render scripts can
consume it without special cases.  For each frame/side, WiLoR is used only when a
same-frame detection is present and its MANO-local geometry can be translated
under the target V19 intrinsics to reproduce WiLoR's own 2D joints.  Otherwise
the original HaWoR row is kept as the infill/fallback hypothesis.

This script does not claim contact or nonpenetration.  It creates a hand-source
candidate with explicit provenance arrays and a JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: Path | None) -> tuple[np.ndarray | None, str | None]:
    if path is None:
        return None, None
    data = load_json(path)
    intr = np.asarray(data.get("intrinsics_fx_fy_cx_cy"), dtype=np.float64).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0 or intr[1] <= 0:
        raise RuntimeError(f"invalid intrinsics_fx_fy_cx_cy in {path}")
    source = str(data.get("intrinsics_source") or data.get("method") or "calibration_contract")
    return intr, source


def project(points_cam: np.ndarray, intr: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(pts).all(axis=1) & (z > 1.0e-8)
    out[good, 0] = pts[good, 0] / z[good] * intr[0] + intr[2]
    out[good, 1] = pts[good, 1] / z[good] * intr[1] + intr[3]
    return out


def camera_to_world(points_cam: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return np.asarray(points_cam, dtype=np.float64) @ np.asarray(r_c2w, dtype=np.float64).T + np.asarray(t_c2w, dtype=np.float64)[None, :]


def world_to_camera(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - np.asarray(t_c2w, dtype=np.float64)[None, :]) @ np.asarray(r_c2w, dtype=np.float64)


def reprojection_report(points_cam: np.ndarray, target_uv: np.ndarray, intr: np.ndarray) -> dict[str, Any]:
    target_uv = np.asarray(target_uv, dtype=np.float64)
    uv = project(np.asarray(points_cam, dtype=np.float64), intr)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(target_uv).all(axis=1)
    if not np.any(valid):
        return {"status": "no_finite_projected_joints", "finite_joint_count": 0}
    err = np.linalg.norm(uv[valid] - target_uv[valid], axis=1)
    return {
        "status": "ok",
        "finite_joint_count": int(np.count_nonzero(valid)),
        "reprojection_error_px": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "max": float(np.max(err)),
        },
    }


def fit_translation_for_intrinsics(
    local_joints: np.ndarray,
    target_uv: np.ndarray,
    intr: np.ndarray,
    initial_cam_t: np.ndarray,
    source_focal: float,
    *,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    local_joints = np.asarray(local_joints, dtype=np.float64)
    target_uv = np.asarray(target_uv, dtype=np.float64)
    initial_cam_t = np.asarray(initial_cam_t, dtype=np.float64).reshape(3)
    if local_joints.shape != (21, 3) or target_uv.shape != (21, 2):
        raise RuntimeError(f"invalid WiLoR joint shapes {local_joints.shape}, {target_uv.shape}")
    valid = np.isfinite(local_joints).all(axis=1) & np.isfinite(target_uv).all(axis=1)
    if np.count_nonzero(valid) < 8:
        raise RuntimeError("too few finite WiLoR joints for translation fit")
    focal_scale = float(intr[0]) / float(source_focal) if source_focal > 0 else 1.0
    # WiLoR/HaMeR-style cameras often use a large canonical focal.  Preserve
    # local MANO geometry, but initialize depth by focal normalization so the
    # target camera projects to the same image scale.
    init = initial_cam_t.copy()
    init[2] = max(0.05, float(initial_cam_t[2]) * focal_scale)
    # Recenter x/y using the median observed pixel under the target depth.  This
    # absorbs principal-point differences between WiLoR's image center and V19 K.
    median_uv = np.nanmedian(target_uv[valid], axis=0)
    init[0] = (median_uv[0] - intr[2]) * init[2] / intr[0] - float(np.nanmedian(local_joints[valid, 0]))
    init[1] = (median_uv[1] - intr[3]) * init[2] / intr[1] - float(np.nanmedian(local_joints[valid, 1]))

    def residual(x: np.ndarray) -> np.ndarray:
        pts = local_joints[valid] + x[None, :]
        uv = project(pts, intr)
        return (uv - target_uv[valid]).reshape(-1)

    lower = np.asarray([-2.0, -2.0, 0.04], dtype=np.float64)
    upper = np.asarray([2.0, 2.0, 5.0], dtype=np.float64)
    init = np.minimum(np.maximum(init, lower + 1.0e-6), upper - 1.0e-6)
    res = least_squares(residual, init, bounds=(lower, upper), loss="soft_l1", f_scale=8.0, max_nfev=max_nfev)
    pts = local_joints + res.x[None, :]
    uv = project(pts, intr)
    err = np.linalg.norm(uv[valid] - target_uv[valid], axis=1)
    report = {
        "status": "ok" if res.success else "optimizer_not_successful",
        "success": bool(res.success),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "initial_translation_camera_m": init.astype(float).tolist(),
        "fitted_translation_camera_m": res.x.astype(float).tolist(),
        "source_focal_px": float(source_focal),
        "target_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
        "finite_joint_count": int(np.count_nonzero(valid)),
        "reprojection_error_px": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "max": float(np.max(err)),
        },
    }
    return res.x.astype(np.float64), report


def choose_wilor_hand(frame: dict[str, Any], side: str, min_score: float) -> dict[str, Any] | None:
    hands = frame.get("raw_hands") if isinstance(frame.get("raw_hands"), list) else []
    candidates = [h for h in hands if isinstance(h, dict) and str(h.get("side")) == side and float(h.get("detector_score", -math.inf)) >= min_score]
    if not candidates:
        return None
    candidates.sort(key=lambda h: float(h.get("detector_score", -math.inf)), reverse=True)
    return candidates[0]


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


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    hawor_blob = np.load(args.hawor_npz, allow_pickle=True)
    hawor = {key: np.asarray(hawor_blob[key]) for key in hawor_blob.files}
    for key in ("frame_idx", "R_c2w", "t_c2w", "left_vertices_world_m", "right_vertices_world_m", "left_joints_world_m", "right_joints_world_m"):
        if key not in hawor:
            raise RuntimeError(f"HaWoR NPZ lacks required key {key}: {args.hawor_npz}")
    wilor = load_json(args.wilor_raw)
    frames = wilor.get("frames") if isinstance(wilor.get("frames"), list) else []
    wilor_by_frame = {int(row.get("frame_idx")): row for row in frames if isinstance(row, dict) and "frame_idx" in row}
    intr, intr_source = load_calibration(args.calibration_contract)
    if intr is None:
        if not args.allow_heuristic_intrinsics:
            raise RuntimeError("missing canonical calibration contract; pass --calibration-contract or explicitly opt into --allow-heuristic-intrinsics for diagnostics")
        width = int((wilor.get("video") or {}).get("width") or 0)
        height = int((wilor.get("video") or {}).get("height") or 0)
        focal = float(np.asarray(hawor.get("img_focal", [0.0]), dtype=float).reshape(-1)[0])
        if width <= 0 or height <= 0 or focal <= 0:
            raise RuntimeError("cannot infer diagnostic heuristic intrinsics; pass --calibration-contract")
        intr = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
        intr_source = "diagnostic_heuristic_hawor_focal_image_center_not_canonical"

    out: dict[str, np.ndarray] = {key: np.array(value, copy=True) for key, value in hawor.items()}
    frame_idx = np.asarray(hawor["frame_idx"], dtype=np.int32)
    n = len(frame_idx)
    provenance: dict[str, list[str]] = {"left": [], "right": []}
    reproj_median: dict[str, np.ndarray] = {"left": np.full(n, np.nan, np.float32), "right": np.full(n, np.nan, np.float32)}
    reproj_p90: dict[str, np.ndarray] = {"left": np.full(n, np.nan, np.float32), "right": np.full(n, np.nan, np.float32)}
    fitted_cam_t: dict[str, np.ndarray] = {"left": np.full((n, 3), np.nan, np.float32), "right": np.full((n, 3), np.nan, np.float32)}
    row_reports: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for pos, idx_raw in enumerate(frame_idx.tolist()):
        idx = int(idx_raw)
        wframe = wilor_by_frame.get(idx)
        r_c2w = np.asarray(hawor["R_c2w"][pos], dtype=np.float64)
        t_c2w = np.asarray(hawor["t_c2w"][pos], dtype=np.float64)
        for side in ("left", "right"):
            selected = choose_wilor_hand(wframe or {}, side, float(args.min_wilor_score)) if wframe is not None else None
            if selected is None:
                provenance[side].append("hawor_infill_or_fallback_no_wilor_visible_detection")
                counts[f"{side}_hawor_fallback_no_wilor"] += 1
                continue
            try:
                local_joints = np.asarray(selected["joints3d_camera"], dtype=np.float64)
                local_vertices = np.asarray(selected["vertices_camera"], dtype=np.float64)
                target_uv = np.asarray(selected.get("joints2d") or selected.get("joints2d_raw"), dtype=np.float64)
                cam_t_raw = np.asarray(selected["cam_t"], dtype=np.float64)
                source_focal = float(selected.get("focal_length") or 0.0)
                if local_vertices.ndim != 2 or local_vertices.shape[1] != 3 or local_joints.shape != (21, 3):
                    raise RuntimeError(f"invalid WiLoR geometry shapes vertices={local_vertices.shape} joints={local_joints.shape}")
                if args.translation_policy == "wilor_metricfit":
                    fitted_t, fit_report = fit_translation_for_intrinsics(
                        local_joints,
                        target_uv,
                        intr,
                        cam_t_raw,
                        source_focal,
                        max_nfev=int(args.max_fit_nfev),
                    )
                    med = float(fit_report["reprojection_error_px"]["median"])
                    p90 = float(fit_report["reprojection_error_px"]["p90"])
                    if not np.isfinite(med) or med > float(args.max_fit_median_px):
                        raise RuntimeError(f"WiLoR translation fit reprojection median too high: {med:.3f}px")
                    verts_cam = local_vertices + fitted_t[None, :]
                    joints_cam = local_joints + fitted_t[None, :]
                    if not (np.isfinite(verts_cam).all() and np.isfinite(joints_cam).all() and np.all(joints_cam[:, 2] > 0.02)):
                        raise RuntimeError("non-finite or non-positive-depth WiLoR fitted camera geometry")
                    verts_world = camera_to_world(verts_cam, r_c2w, t_c2w).astype(np.float32)
                    joints_world = camera_to_world(joints_cam, r_c2w, t_c2w).astype(np.float32)
                    source_label = "wilor_visible_translation_refit_v19_intrinsics"
                elif args.translation_policy == "hawor_wrist_aligned":
                    hawor_joints = np.asarray(hawor[f"{side}_joints_world_m"][pos], dtype=np.float64)
                    if hawor_joints.shape != (21, 3) or not np.isfinite(hawor_joints[0]).all():
                        raise RuntimeError("missing finite HaWoR wrist for shared trajectory alignment")
                    root_world = hawor_joints[0]
                    root_cam = world_to_camera(root_world[None, :], r_c2w, t_c2w)[0]
                    rel_joints_cam = local_joints - local_joints[0][None, :]
                    rel_vertices_cam = local_vertices - local_joints[0][None, :]
                    joints_cam = root_cam[None, :] + rel_joints_cam
                    verts_cam = root_cam[None, :] + rel_vertices_cam
                    if not (np.isfinite(verts_cam).all() and np.isfinite(joints_cam).all() and np.all(joints_cam[:, 2] > 0.02)):
                        raise RuntimeError("non-finite or non-positive-depth WiLoR-on-HaWoR camera geometry")
                    joints_world = (root_world[None, :] + rel_joints_cam @ r_c2w.T).astype(np.float32)
                    verts_world = (root_world[None, :] + rel_vertices_cam @ r_c2w.T).astype(np.float32)
                    fitted_t = (root_cam - local_joints[0]).astype(np.float64)
                    fit_report = reprojection_report(joints_cam, target_uv, intr)
                    fit_report.update({
                        "status": "ok",
                        "translation_policy": "hawor_wrist_aligned",
                        "haWoR_wrist_world_m": root_world.astype(float).tolist(),
                        "wilor_local_wrist_camera_m": local_joints[0].astype(float).tolist(),
                        "fitted_translation_camera_m": fitted_t.astype(float).tolist(),
                        "target_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                    })
                    med = float(fit_report["reprojection_error_px"]["median"])
                    p90 = float(fit_report["reprojection_error_px"]["p90"])
                    source_label = "wilor_root_relative_on_hawor_metric_wrist_trajectory"
                else:
                    raise RuntimeError(f"unknown translation policy {args.translation_policy}")
                out[f"{side}_vertices_world_m"][pos] = verts_world
                out[f"{side}_joints_world_m"][pos] = joints_world
                out[f"{side}_valid"][pos] = np.uint8(1)
                if f"{side}_detected_same_frame" in out:
                    out[f"{side}_detected_same_frame"][pos] = np.uint8(1)
                if f"{side}_det_box_xyxyscore" in out:
                    box = np.asarray(selected.get("bbox_xyxy") or [np.nan, np.nan, np.nan, np.nan], dtype=np.float32).reshape(-1)
                    score = float(selected.get("detector_score", np.nan))
                    if box.shape[0] >= 4:
                        out[f"{side}_det_box_xyxyscore"][pos] = np.asarray([box[0], box[1], box[2], box[3], score], dtype=np.float32)
                if f"{side}_track_id" in out:
                    out[f"{side}_track_id"][pos] = f"wilor_visible_{side}"
                if f"{side}_trans_world_m" in out:
                    out[f"{side}_trans_world_m"][pos] = camera_to_world(fitted_t[None, :], r_c2w, t_c2w)[0].astype(np.float32)
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
                provenance[side].append(source_label)
                reproj_median[side][pos] = np.float32(med)
                reproj_p90[side][pos] = np.float32(p90)
                fitted_cam_t[side][pos] = fitted_t.astype(np.float32)
                counts[f"{side}_wilor_visible"] += 1
                if len(row_reports) < int(args.report_row_limit):
                    row_reports.append({
                        "frame_idx": idx,
                        "side": side,
                        "source": source_label,
                        "wilor_score": float(selected.get("detector_score", np.nan)),
                        "wilor_bbox_xyxy": selected.get("bbox_xyxy"),
                        "fit": fit_report,
                        "camera_depth_m_summary": {
                            "joints_median_z": float(np.median(joints_cam[:, 2])),
                            "vertices_median_z": float(np.median(verts_cam[:, 2])),
                        },
                    })
            except Exception as exc:
                provenance[side].append(f"hawor_infill_or_fallback_wilor_rejected:{type(exc).__name__}:{str(exc)[:160]}")
                counts[f"{side}_hawor_fallback_wilor_rejected"] += 1
                if len(row_reports) < int(args.report_row_limit):
                    row_reports.append({"frame_idx": idx, "side": side, "source": "hawor_fallback", "reason": f"{type(exc).__name__}: {exc}"})

    for side in ("left", "right"):
        out[f"{side}_hybrid_source"] = np.asarray(provenance[side])
        out[f"{side}_wilor_fit_reprojection_median_px"] = reproj_median[side]
        out[f"{side}_wilor_fit_reprojection_p90_px"] = reproj_p90[side]
        out[f"{side}_wilor_fitted_translation_camera_m"] = fitted_cam_t[side]
    out["hybrid_policy"] = np.asarray([json.dumps({
        "method": "build_v19_wilor_hawor_hybrid_hand_npz",
        "visible_source": "WiLoR same-frame detections using the requested translation policy",
        "translation_policy": args.translation_policy,
        "fallback_source": "original HaWoR/infiller NPZ rows",
        "min_wilor_score": float(args.min_wilor_score),
        "max_fit_median_px": float(args.max_fit_median_px),
        "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
        "intrinsics_source": intr_source,
        "claim_scope": "hybrid hand candidate only; no contact or nonpenetration claim",
    }, sort_keys=True)])
    out["hybrid_created_unix_s"] = np.asarray([time.time()], dtype=np.float64)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    report = {
        "status": "ok",
        "method": "build_v19_wilor_hawor_hybrid_hand_npz",
        "inputs": {
            "wilor_raw": str(args.wilor_raw),
            "hawor_npz": str(args.hawor_npz),
            "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
        },
        "outputs": {"hybrid_npz": str(args.output_npz)},
        "counts": dict(counts),
        "frame_count": int(n),
        "intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
        "intrinsics_source": intr_source,
        "row_reports_preview": row_reports,
        "translation_policy": args.translation_policy,
        "claim_scope": "WiLoR visible-frame MANO candidate with HaWoR/infiller fallback for missing/unsupported rows; downstream render/contact must validate physical usefulness.",
        "elapsed_s": float(time.time() - started),
    }
    if args.report_json is not None:
        write_json(args.report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wilor-raw", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--allow-heuristic-intrinsics", action="store_true", help="Diagnostic-only opt-in when no canonical calibration contract exists; output is not a resolved camera contract.")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--min-wilor-score", type=float, default=0.30)
    parser.add_argument("--max-fit-median-px", type=float, default=4.0)
    parser.add_argument("--max-fit-nfev", type=int, default=100)
    parser.add_argument("--translation-policy", choices=["wilor_metricfit", "hawor_wrist_aligned"], default="wilor_metricfit", help="Metric semantics for accepted WiLoR visible rows. wilor_metricfit preserves WiLoR local geometry and refits translation under V19 intrinsics. hawor_wrist_aligned uses HaWoR/SLAM wrist trajectory as metric translation and WiLoR root-relative geometry as visible articulation.")
    parser.add_argument("--report-row-limit", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2)[:12000])


if __name__ == "__main__":
    main()
