#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_hand_contact_reliability_v3 import depth_patch_iqr_ratio, hand_bone_scale_m
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_contact_depth_scale_v3 import summarize
from refit_mano_pose_contact_v3 import apply_side_sign, load_wilor_mano_class, patch_legacy_mano_loader


@dataclass(frozen=True)
class HypothesisInput:
    frame_idx: int
    hand_idx: int
    stored_side: str
    candidate_side: str
    track_id: str | None
    detector_score: float
    measurement_available: bool
    raw2d: np.ndarray
    intrinsics: np.ndarray
    metric_depth: np.ndarray
    depth_valid: np.ndarray
    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    source_joints_stored: np.ndarray
    source_vertices_stored: np.ndarray
    stored_reprojection_median_px: float | None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def transform(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    scale = math.exp(float(params[6]))
    return scale * (points @ rotation_matrix(params[:3]).T) + params[3:6][None, :]


def robust_residual(x: np.ndarray, sigma: float) -> np.ndarray:
    return x / float(sigma)


def stable_depth_mask(depth: np.ndarray, keypoints: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    ratios = np.asarray([depth_patch_iqr_ratio(depth, xy * scale, args.patch_radius) for xy in keypoints], dtype=float)
    return np.isfinite(ratios) & (ratios <= float(args.max_depth_iqr_ratio))


def source_size_for(frame: dict) -> np.ndarray:
    obj = frame.get("object", {})
    size = np.asarray(obj.get("source_image_size", []), dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid source image size")
    return size


def frame_intrinsics(frame: dict, hand: dict, args: argparse.Namespace) -> np.ndarray:
    if args.intrinsics_source == "annotation-vggt":
        intr = np.asarray(frame.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=float)
    elif args.intrinsics_source == "hand":
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    elif args.intrinsics_source == "cli":
        intr = np.asarray(args.intrinsics, dtype=float)
    else:
        raise RuntimeError(f"unsupported intrinsics source {args.intrinsics_source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid intrinsics for frame {frame.get('frame_idx')}")
    return intr


def local_vertices_key(hand: dict) -> str:
    if "vertices_camera" in hand:
        return "vertices_camera"
    if "vertices_camera_sample" in hand:
        return "vertices_camera_sample"
    raise RuntimeError("hand has no local MANO vertices key")


def source_vertices_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    if "vertices_source_camera_m_sample" in hand:
        return "vertices_source_camera_m_sample"
    raise RuntimeError("hand has no source MANO vertices key")


def source_to_world(points: np.ndarray, frame: dict) -> np.ndarray:
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T @ homog.T).T[:, :3]


def initial_params(joints: np.ndarray, raw2d: np.ndarray, intr: np.ndarray, metric_depth: np.ndarray, depth_valid: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    base_bone = hand_bone_scale_m(joints)
    if not np.isfinite(base_bone) or base_bone <= 0.0:
        raise RuntimeError("invalid MANO bone scale")
    scale = float(np.clip(float(args.hand_bone_prior_m) / base_bone, args.min_scale, args.max_scale))
    z = float(np.median(metric_depth[depth_valid]))
    if not np.isfinite(z) or z <= args.min_depth_m:
        raise RuntimeError("invalid metric depth for side hypothesis")
    center_px = np.median(raw2d[depth_valid], axis=0)
    fx, fy, cx, cy = intr
    center = np.asarray([(center_px[0] - cx) * z / fx, (center_px[1] - cy) * z / fy, z], dtype=float)
    local_center = np.median(scale * joints, axis=0)
    return np.r_[np.zeros(3, dtype=float), center - local_center, math.log(scale)]


def residual(params: np.ndarray, joints: np.ndarray, raw2d: np.ndarray, intr: np.ndarray, metric_depth: np.ndarray, depth_valid: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    source_joints = transform(joints, params)
    z_violation = np.clip(float(args.min_depth_m) - source_joints[:, 2], 0.0, None)
    safe_joints = source_joints.copy()
    safe_joints[:, 2] = np.clip(safe_joints[:, 2], args.min_depth_m, None)
    reproj = (project_points(safe_joints, intr) - raw2d).reshape(-1)
    scale = math.exp(float(params[6]))
    bone = scale * hand_bone_scale_m(joints)
    return np.concatenate(
        [
            robust_residual(np.asarray([float(np.max(z_violation))]), args.sigma_min_depth_m),
            robust_residual(reproj, args.sigma_reprojection_px),
            robust_residual(safe_joints[depth_valid, 2] - metric_depth[depth_valid], args.sigma_metric_depth_m),
            robust_residual(np.asarray([bone - float(args.hand_bone_prior_m)]), args.sigma_bone_prior_m),
            robust_residual(params[:3], args.sigma_rotation_prior_rad),
            robust_residual(np.asarray([params[6]]), args.sigma_log_scale_prior),
        ]
    )


def solve_candidate(joints: np.ndarray, vertices: np.ndarray, item: HypothesisInput, args: argparse.Namespace) -> dict:
    init = initial_params(joints, item.raw2d, item.intrinsics, item.metric_depth, item.depth_valid, args)
    lo = np.asarray(
        [
            -args.max_rotation_rad,
            -args.max_rotation_rad,
            -args.max_rotation_rad,
            -2.0,
            -2.0,
            args.min_depth_m,
            math.log(args.min_scale),
        ],
        dtype=float,
    )
    hi = np.asarray(
        [
            args.max_rotation_rad,
            args.max_rotation_rad,
            args.max_rotation_rad,
            2.0,
            2.0,
            args.max_depth_m,
            math.log(args.max_scale),
        ],
        dtype=float,
    )
    result = least_squares(
        residual,
        init,
        args=(joints, item.raw2d, item.intrinsics, item.metric_depth, item.depth_valid, args),
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
    )
    fitted_joints = transform(joints, result.x)
    fitted_vertices = transform(vertices, result.x)
    fitted_local_joints = fitted_joints - result.x[3:6][None, :]
    fitted_local_vertices = fitted_vertices - result.x[3:6][None, :]
    projected = project_points(fitted_joints, item.intrinsics)
    reproj = np.linalg.norm(projected - item.raw2d, axis=1)
    depth_err = fitted_joints[item.depth_valid, 2] - item.metric_depth[item.depth_valid]
    bone = math.exp(float(result.x[6])) * hand_bone_scale_m(joints)
    chirality = float(np.linalg.det(np.stack([fitted_joints[5] - fitted_joints[0], fitted_joints[17] - fitted_joints[0], fitted_joints[12] - fitted_joints[0]], axis=1)))
    cross2d = float(np.cross(np.r_[item.raw2d[5] - item.raw2d[0], 0.0], np.r_[item.raw2d[17] - item.raw2d[0], 0.0])[2])
    stored_gap = np.linalg.norm(fitted_joints - item.source_joints_stored, axis=1)
    vertex_gap = np.linalg.norm(fitted_vertices - item.source_vertices_stored, axis=1)
    save_geometry = bool(args.keep_geometry or args.output_annotations is not None)
    return {
        "candidate_side": item.candidate_side,
        "stored_side": item.stored_side,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "median_reprojection_px": float(np.median(reproj)),
        "p95_reprojection_px": float(np.percentile(reproj, 95.0)),
        "depth_joints": int(np.count_nonzero(item.depth_valid)),
        "mano_minus_metric_depth_median_m": float(np.median(depth_err)),
        "mano_minus_metric_depth_p95_abs_m": float(np.percentile(np.abs(depth_err), 95.0)),
        "hand_bone_m": float(bone),
        "scale": float(math.exp(float(result.x[6]))),
        "rotation_norm_rad": float(np.linalg.norm(result.x[:3])),
        "translation_m": result.x[3:6].astype(float).tolist(),
        "chirality_det_m3": chirality,
        "chirality_det_abs_e6": float(abs(chirality) * 1e6),
        "raw2d_palm_cross": cross2d,
        "chirality_sign_agrees_with_raw2d_cross": bool(np.sign(chirality) == np.sign(cross2d)) if chirality != 0.0 and cross2d != 0.0 else False,
        "joint_gap_to_stored_source_median_m": float(np.median(stored_gap)),
        "joint_gap_to_stored_source_p95_m": float(np.percentile(stored_gap, 95.0)),
        "vertex_gap_to_stored_source_median_m": float(np.median(vertex_gap)),
        "fitted_joints2d": projected.astype(float).tolist() if save_geometry else None,
        "fitted_local_joints_camera_m": fitted_local_joints.astype(float).tolist() if save_geometry else None,
        "fitted_local_vertices_camera_m": fitted_local_vertices.astype(float).tolist() if save_geometry else None,
        "fitted_joints_source_camera_m": fitted_joints.astype(float).tolist() if save_geometry else None,
        "fitted_vertices_source_camera_m": fitted_vertices.astype(float).tolist() if save_geometry else None,
    }


def canonical_geometry(model, item: HypothesisInput) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if item.candidate_side == "right" else -1.0
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(item.global_orient[None], dtype=torch.float32),
            hand_pose=torch.tensor(item.hand_pose[None], dtype=torch.float32),
            betas=torch.tensor(item.betas[None], dtype=torch.float32),
            return_verts=True,
            pose2rot=False,
        )
        vertices = apply_side_sign(out.vertices, sign)[0].cpu().numpy()
        joints = apply_side_sign(out.joints, sign)[0].cpu().numpy()
    return joints, vertices


def iter_inputs(data: dict, args: argparse.Namespace) -> tuple[list[HypothesisInput], list[dict]]:
    depth_blob = np.load(args.metric_depth_npz)
    frame_to_depth = {int(idx): i for i, idx in enumerate(depth_blob["frame_idx"].astype(int).tolist())}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    items: list[HypothesisInput] = []
    skipped: list[dict] = []
    for frame in data["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        if args.only_track_id is not None:
            frame_tracks = [hand.get("track_id") for hand in frame.get("hands", [])]
            if args.only_track_id not in frame_tracks:
                continue
        try:
            depth = depth_frame(depths, frame_to_depth, frame_idx)
            source_size = source_size_for(frame)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_idx, hand in enumerate(frame.get("hands", [])):
            try:
                track_id = hand.get("track_id")
                if args.only_track_id is not None and track_id != args.only_track_id:
                    continue
                if not bool(hand.get("measurement_available", False)):
                    raise RuntimeError("not_measured")
                score = float(hand.get("detector_score", np.nan))
                if not np.isfinite(score) or score < float(args.min_detector_score):
                    raise RuntimeError("low_detector_score")
                raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
                if raw2d.shape != (21, 2):
                    raise RuntimeError("invalid raw2d")
                intr = frame_intrinsics(frame, hand, args)
                metric_depth = sample_depth(depth, raw2d, source_size)
                valid = np.isfinite(metric_depth) & (metric_depth > float(args.min_depth_m))
                stable = stable_depth_mask(depth, raw2d, source_size, args)
                depth_valid = valid & stable
                if int(np.count_nonzero(depth_valid)) < int(args.min_depth_keypoints):
                    depth_valid = valid
                if int(np.count_nonzero(depth_valid)) < int(args.min_depth_keypoints):
                    raise RuntimeError("too_few_metric_depth_keypoints")
                params = hand["mano_params"]
                global_orient = np.asarray(params["global_orient"], dtype=float)
                hand_pose = np.asarray(params["hand_pose"], dtype=float)
                betas = np.asarray(params["betas"], dtype=float)
                source_joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
                source_vertices = np.asarray(hand.get("vertices_source_camera_m", []), dtype=float)
                if global_orient.shape != (1, 3, 3) or hand_pose.shape != (15, 3, 3) or betas.shape != (10,):
                    raise RuntimeError("invalid WiLoR MANO parameters")
                if source_joints.shape != (21, 3) or source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
                    raise RuntimeError("missing stored source geometry")
                stored_residual = hand.get("projection_residual_to_measurement_px", {})
                stored_median = stored_residual.get("median") if isinstance(stored_residual, dict) else None
                for candidate_side in ("left", "right"):
                    items.append(
                        HypothesisInput(
                            frame_idx=frame_idx,
                            hand_idx=hand_idx,
                            stored_side=str(hand.get("side")),
                            candidate_side=candidate_side,
                            track_id=track_id,
                            detector_score=score,
                            measurement_available=True,
                            raw2d=raw2d,
                            intrinsics=intr,
                            metric_depth=metric_depth,
                            depth_valid=depth_valid.astype(bool),
                            global_orient=global_orient,
                            hand_pose=hand_pose,
                            betas=betas,
                            source_joints_stored=source_joints,
                            source_vertices_stored=source_vertices,
                            stored_reprojection_median_px=None if stored_median is None else float(stored_median),
                        )
                    )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_idx, "side": hand.get("side"), "reason": str(exc)})
    if not items:
        raise RuntimeError(f"no side hypotheses; skipped={skipped[:20]}")
    return items, skipped


def summarize_rows(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return summarize(np.asarray(values, dtype=float))


def best_rows_by_score(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    grouped: dict[tuple[int, int, str | None], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["frame_idx"]), int(row["hand_idx"]), row.get("track_id")), []).append(row)
    out: list[dict] = []
    for key, candidates in grouped.items():
        best = min(
            candidates,
            key=lambda r: (
                float(r["median_reprojection_px"]) / args.score_reprojection_px
                + abs(float(r["mano_minus_metric_depth_median_m"])) / args.score_depth_m
                + abs(float(r["hand_bone_m"]) - float(args.hand_bone_prior_m)) / args.score_bone_m
                + float(r["rotation_norm_rad"]) / args.score_rotation_rad
            ),
        )
        out.append(best)
    out.sort(key=lambda r: (int(r["frame_idx"]), int(r["hand_idx"])))
    return out


def pick_by_score(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    out = []
    for best in best_rows_by_score(rows, args):
        out.append(
            {
                "frame_idx": int(best["frame_idx"]),
                "hand_idx": int(best["hand_idx"]),
                "track_id": best.get("track_id"),
                "stored_side": best["stored_side"],
                "selected_side": best["candidate_side"],
                "selected_equals_stored": bool(best["candidate_side"] == best["stored_side"]),
                "selected_median_reprojection_px": best["median_reprojection_px"],
                "selected_depth_bias_m": best["mano_minus_metric_depth_median_m"],
                "selected_chirality_sign_agrees_with_raw2d_cross": best["chirality_sign_agrees_with_raw2d_cross"],
            }
        )
    return out


def apply_selected_geometry(data: dict, chosen: list[dict], args: argparse.Namespace) -> dict:
    out = copy.deepcopy(data)
    by_key = {(int(row["frame_idx"]), int(row["hand_idx"])): row for row in chosen}
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        for hand_idx, hand in enumerate(frame.get("hands", [])):
            row = by_key.get((frame_idx, hand_idx))
            if row is None:
                continue
            local_joints = np.asarray(row["fitted_local_joints_camera_m"], dtype=float)
            local_vertices = np.asarray(row["fitted_local_vertices_camera_m"], dtype=float)
            source_joints = np.asarray(row["fitted_joints_source_camera_m"], dtype=float)
            source_vertices = np.asarray(row["fitted_vertices_source_camera_m"], dtype=float)
            projected = np.asarray(row["fitted_joints2d"], dtype=float)
            raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
            err = np.linalg.norm(projected - raw2d, axis=1)
            hand["side"] = str(row["candidate_side"])
            hand["joints3d_camera"] = local_joints.astype(float).tolist()
            hand[local_vertices_key(hand)] = local_vertices.astype(float).tolist()
            hand["cam_t"] = np.asarray(row["translation_m"], dtype=float).astype(float).tolist()
            hand["joints3d_source_camera_m"] = source_joints.astype(float).tolist()
            hand[source_vertices_key(hand)] = source_vertices.astype(float).tolist()
            hand["joints2d"] = projected.astype(float).tolist()
            hand["joints3d_world_m"] = source_to_world(source_joints, frame).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(source_vertices, frame).astype(float).tolist()
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(err)),
                "p95": float(np.percentile(err, 95.0)),
            }
            hand["v3_mano_side_metric_refit"] = {
                "status": "applied",
                "stored_side": str(row["stored_side"]),
                "selected_side": str(row["candidate_side"]),
                "median_reprojection_px": float(row["median_reprojection_px"]),
                "mano_minus_metric_depth_median_m": float(row["mano_minus_metric_depth_median_m"]),
                "hand_bone_m": float(row["hand_bone_m"]),
                "scale": float(row["scale"]),
                "rotation_norm_rad": float(row["rotation_norm_rad"]),
                "source": "wilor_mano_rotation_matrices_side_hypothesis_metric_fit",
            }
            hand["world_coordinate_status"] = "v3_mano_side_metric_refit_source_camera_mano_transformed_by_existing_camera_pose"
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_mano_side_metric_refit"
    return out


def strip_geometry(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("fitted_")}


def run(args: argparse.Namespace) -> dict:
    patch_legacy_mano_loader()
    data = load_json(args.annotations)
    items, skipped = iter_inputs(data, args)
    mano_cls = load_wilor_mano_class(args.wilor_root)
    mano_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO_RIGHT model: {mano_path}")
    model = mano_cls(model_path=str(mano_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to("cpu")
    rows: list[dict] = []
    for item in items:
        joints, vertices = canonical_geometry(model, item)
        row = solve_candidate(joints, vertices, item, args)
        row.update(
            {
                "frame_idx": int(item.frame_idx),
                "hand_idx": int(item.hand_idx),
                "track_id": item.track_id,
                "detector_score": float(item.detector_score),
                "stored_reprojection_median_px": item.stored_reprojection_median_px,
            }
        )
        rows.append(row)
    rows.sort(key=lambda r: (int(r["frame_idx"]), int(r["hand_idx"]), str(r["candidate_side"])))
    chosen = best_rows_by_score(rows, args)
    if args.output_annotations is not None:
        output = apply_selected_geometry(data, chosen, args)
        save_json(args.output_annotations, output)
    detail_rows = rows if args.keep_geometry else [strip_geometry(row) for row in rows]
    selected = pick_by_score(rows, args)
    report = {
        "status": "diagnostic_mano_side_hypotheses_v3",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "output_annotations": None if args.output_annotations is None else str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "intrinsics_source": args.intrinsics_source,
        "only_track_id": args.only_track_id,
        "rows": int(len(rows)),
        "summary": {
            "median_reprojection_px": summarize_rows(rows, "median_reprojection_px"),
            "mano_minus_metric_depth_median_m": summarize_rows(rows, "mano_minus_metric_depth_median_m"),
            "hand_bone_m": summarize_rows(rows, "hand_bone_m"),
            "rotation_norm_rad": summarize_rows(rows, "rotation_norm_rad"),
        },
        "selected_by_score": selected,
        "rows_detail": detail_rows,
        "skipped_preview": skipped[:120],
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "min_depth_keypoints": int(args.min_depth_keypoints),
            "hand_bone_prior_m": float(args.hand_bone_prior_m),
            "sigma_reprojection_px": float(args.sigma_reprojection_px),
            "sigma_metric_depth_m": float(args.sigma_metric_depth_m),
        },
    }
    save_json(args.output_json, report)
    compact = {k: v for k, v in report.items() if k not in {"rows_detail", "skipped_preview"}}
    print(json.dumps(compact, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--only-track-id")
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "hand", "cli"], default="annotation-vggt")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[1200.0, 1175.0, 960.0, 540.0])
    parser.add_argument("--min-detector-score", type=float, default=0.30)
    parser.add_argument("--min-depth-keypoints", type=int, default=8)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--min-scale", type=float, default=0.60)
    parser.add_argument("--max-scale", type=float, default=1.60)
    parser.add_argument("--max-rotation-rad", type=float, default=1.20)
    parser.add_argument("--max-nfev", type=int, default=220)
    parser.add_argument("--hand-bone-prior-m", type=float, default=0.165)
    parser.add_argument("--sigma-reprojection-px", type=float, default=10.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.025)
    parser.add_argument("--sigma-min-depth-m", type=float, default=0.020)
    parser.add_argument("--sigma-bone-prior-m", type=float, default=0.030)
    parser.add_argument("--sigma-rotation-prior-rad", type=float, default=0.60)
    parser.add_argument("--sigma-log-scale-prior", type=float, default=0.35)
    parser.add_argument("--score-reprojection-px", type=float, default=10.0)
    parser.add_argument("--score-depth-m", type=float, default=0.025)
    parser.add_argument("--score-bone-m", type=float, default=0.030)
    parser.add_argument("--score-rotation-rad", type=float, default=0.60)
    parser.add_argument("--keep-geometry", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
