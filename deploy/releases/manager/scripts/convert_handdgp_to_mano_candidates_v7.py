#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_reprojection_depth_v3 import project_points
from refit_mano_pose_contact_v3 import (
    apply_side_sign,
    hand_span_torch,
    load_wilor_mano_class,
    robust_l1,
    rotvec_to_matrix,
    side_sign,
)
from run_hamer_rtmlib_hand_stream_v3 import hand_bone_scale_m, sample_vertices, solve_metric_hand


def patch_legacy_mano_loader() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if not hasattr(np, name):
            setattr(np, name, value)


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_to_world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (transform @ homog.T).T[:, :3]


def fit_mano_to_handdgp(model, hand: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    device = torch.device(args.device)
    model = model.to(device)
    side = str(hand.get("side", "")).lower()
    sign = side_sign(side)
    target_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
    target_vertices = np.asarray(hand.get("vertices_camera", []), dtype=float)
    if target_joints.shape != (21, 3) or target_vertices.ndim != 2 or target_vertices.shape[1] != 3:
        raise RuntimeError("HandDGP hand must contain local 21 joints and local vertices")
    if not np.isfinite(target_joints).all() or not np.isfinite(target_vertices).all():
        raise RuntimeError("HandDGP local geometry contains non-finite values")

    target_joint_t = torch.tensor(target_joints, dtype=torch.float32, device=device)
    target_vertex_t = torch.tensor(target_vertices, dtype=torch.float32, device=device)
    orient_delta = torch.zeros((1, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    pose_delta = torch.zeros((1, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device, requires_grad=True)
    trans = torch.zeros((1, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    log_scale = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([orient_delta, pose_delta, betas, trans, log_scale], lr=float(args.lr))
    use_vertex_loss = int(target_vertices.shape[0]) == int(args.expected_mano_vertices)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(int(args.iters)):
        opt.zero_grad(set_to_none=True)
        global_orient = rotvec_to_matrix(orient_delta)
        hand_pose = rotvec_to_matrix(pose_delta)
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas, return_verts=True, pose2rot=False)
        canonical_vertices = apply_side_sign(out.vertices, sign)
        canonical_joints = apply_side_sign(out.joints, sign)
        scale = torch.exp(log_scale).reshape(1, 1, 1)
        local_vertices = scale * canonical_vertices + trans
        local_joints = scale * canonical_joints + trans
        joint_loss = robust_l1((local_joints[0] - target_joint_t) / float(args.sigma_joint_m)).mean()
        if use_vertex_loss:
            vertex_loss = robust_l1((local_vertices[0] - target_vertex_t) / float(args.sigma_vertex_m)).mean()
        else:
            vertex_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        span = hand_span_torch(local_joints)
        span_loss = (
            robust_l1(torch.relu(float(args.min_span_m) - span) / float(args.sigma_span_m))
            + robust_l1(torch.relu(span - float(args.max_span_m)) / float(args.sigma_span_m))
        )
        loss = (
            float(args.w_joint) * joint_loss
            + float(args.w_vertex) * vertex_loss
            + float(args.w_pose) * robust_l1(pose_delta / float(args.sigma_pose_rad)).mean()
            + float(args.w_orient) * robust_l1(orient_delta / float(args.sigma_orient_rad)).mean()
            + float(args.w_beta) * robust_l1(betas / float(args.sigma_beta)).mean()
            + float(args.w_scale) * robust_l1(log_scale / float(args.sigma_log_scale)).mean()
            + float(args.w_span) * span_loss
        )
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "global_orient": global_orient.detach().clone(),
                "hand_pose": hand_pose.detach().clone(),
                "betas": betas.detach().clone(),
                "local_vertices": local_vertices.detach().clone(),
                "local_joints": local_joints.detach().clone(),
                "scale": torch.exp(log_scale.detach()).clone(),
                "pose_delta": pose_delta.detach().clone(),
                "orient_delta": orient_delta.detach().clone(),
            }
        loss.backward()
        opt.step()
        with torch.no_grad():
            pose_delta.clamp_(-float(args.max_pose_rad), float(args.max_pose_rad))
            orient_delta.clamp_(-float(args.max_orient_rad), float(args.max_orient_rad))
            betas.clamp_(-float(args.max_beta), float(args.max_beta))
            log_scale.clamp_(np.log(float(args.min_scale)), np.log(float(args.max_scale)))
    if best_state is None:
        raise RuntimeError("MANO inverse fit produced no state")

    local_joints = best_state["local_joints"][0].cpu().numpy()
    local_vertices = best_state["local_vertices"][0].cpu().numpy()
    joint_err = np.linalg.norm(local_joints - target_joints, axis=1)
    if use_vertex_loss:
        vertex_err = np.linalg.norm(local_vertices - target_vertices, axis=1)
    else:
        vertex_err = np.asarray([], dtype=float)
    return (
        {
            "global_orient": best_state["global_orient"].cpu().numpy(),
            "hand_pose": best_state["hand_pose"].cpu().numpy(),
            "betas": best_state["betas"][0].cpu().numpy(),
            "local_vertices": local_vertices,
            "local_joints": local_joints,
        },
        {
            "inverse_fit_loss": best_loss,
            "target_vertex_count": int(target_vertices.shape[0]),
            "used_vertex_correspondence": bool(use_vertex_loss),
            "joint_fit_median_m": float(np.median(joint_err)),
            "joint_fit_p95_m": float(np.percentile(joint_err, 95.0)),
            "vertex_fit_median_m": None if len(vertex_err) == 0 else float(np.median(vertex_err)),
            "vertex_fit_p95_m": None if len(vertex_err) == 0 else float(np.percentile(vertex_err, 95.0)),
            "scale": float(best_state["scale"]),
            "hand_bone_m": float(hand_bone_scale_m(local_joints)),
            "pose_delta_abs_max_rad": float(torch.max(torch.abs(best_state["pose_delta"])).cpu()),
            "orient_delta_abs_max_rad": float(torch.max(torch.abs(best_state["orient_delta"])).cpu()),
        },
    )


def convert_hand(frame: dict, hand: dict, fit: dict, inverse_metrics: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if intr.shape != (4,) or raw2d.shape != (21, 2):
        raise RuntimeError("HandDGP candidate lacks source intrinsics or 2D measurement")
    local_joints = np.asarray(fit["local_joints"], dtype=float)
    local_vertices = np.asarray(fit["local_vertices"], dtype=float)
    translation, source_joints, source_vertices, solve_metrics = solve_metric_hand(local_joints, local_vertices, raw2d, intr)
    sampled_vertices, surface_status = sample_vertices(local_vertices, int(args.max_vertices_per_hand))
    sampled_source_vertices = sampled_vertices + translation[None, :]
    projected = project_points(source_joints, intr)
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    transform = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    measurement_available = bool(
        float(args.min_depth_m) <= float(solve_metrics["median_depth_m"]) <= float(args.max_depth_m)
        and float(solve_metrics["median_reprojection_error_px"]) <= float(args.max_initial_reprojection_px)
        and float(args.min_hand_bone_m) <= float(solve_metrics["hand_bone_scale_m"]) <= float(args.max_hand_bone_m)
        and float(inverse_metrics["joint_fit_median_m"]) <= float(args.max_inverse_joint_median_m)
        and float(inverse_metrics["joint_fit_p95_m"]) <= float(args.max_inverse_joint_p95_m)
    )
    out = copy.deepcopy(hand)
    out["backend"] = "HandDGP-MANO"
    out["track_id"] = str(hand.get("track_id") or args.track_id or "")
    out["track_source"] = "handdgp_inverse_mano_from_maskbox_candidate"
    out["cam_t"] = translation.astype(float).tolist()
    out["source_intrinsics"] = intr.astype(float).tolist()
    out["joints3d_camera"] = local_joints.astype(float).tolist()
    out["vertices_camera"] = local_vertices.astype(float).tolist()
    out["joints3d_source_camera_m"] = source_joints.astype(float).tolist()
    out["vertices_source_camera_m"] = source_vertices.astype(float).tolist()
    out["vertices_source_camera_m_sample"] = sampled_source_vertices.astype(float).tolist()
    out["joints2d"] = projected.astype(float).tolist()
    out["projection_residual_to_measurement_px"] = {
        "median": float(np.median(reproj)),
        "p95": float(np.percentile(reproj, 95.0)),
    }
    out["joints3d_world_m"] = source_to_world(source_joints, transform).astype(float).tolist()
    out["vertices_world_m"] = source_to_world(source_vertices, transform).astype(float).tolist()
    out["mano_params"] = {
        "global_orient": np.asarray(fit["global_orient"], dtype=float).reshape(1, 3, 3).tolist(),
        "hand_pose": np.asarray(fit["hand_pose"], dtype=float).reshape(15, 3, 3).tolist(),
        "betas": np.asarray(fit["betas"], dtype=float).reshape(10).tolist(),
        "rotation_convention": "wilor_rotation_matrix_pose2rot_false_with_side_x_sign",
    }
    out["mano_surface_status"] = surface_status
    out["mano_vertex_count"] = int(len(local_vertices))
    out["measurement_available"] = measurement_available
    out["filter_status"] = "handdgp_inverse_mano_measured" if measurement_available else "handdgp_inverse_mano_rejected_initial_metric_qc"
    out["handdgp_inverse_mano"] = {
        "status": "fit_mano_rotation_matrices_and_betas_to_handdgp_local_geometry",
        "inverse_fit": inverse_metrics,
        "source_camera_solve": solve_metrics,
        "source_backend": hand.get("backend"),
        "source_handdgp": hand.get("handdgp"),
    }
    out["world_coordinate_status"] = "handdgp_inverse_mano_source_camera_transformed_by_existing_camera_pose"
    return out, {
        "frame_idx": int(frame["frame_idx"]),
        "side": out.get("side"),
        "measurement_available": bool(measurement_available),
        "detector_score": float(out.get("detector_score", 0.0)),
        "median_reprojection_error_px": solve_metrics["median_reprojection_error_px"],
        "hand_bone_scale_m": solve_metrics["hand_bone_scale_m"],
        **inverse_metrics,
    }


def summarize_key(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return summarize(np.asarray(values, dtype=float))


def run(args: argparse.Namespace) -> dict:
    patch_legacy_mano_loader()
    annotations = load_json(args.annotations)
    mano_cls = load_wilor_mano_class(args.wilor_root)
    mano_path = args.mano_right if args.mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_path.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT model: {mano_path}")
    model = mano_cls(model_path=str(mano_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1)
    output = copy.deepcopy(annotations)
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame in output.get("frames", []):
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        converted = []
        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                side = str(hand.get("side", "")).lower()
                if args.side != "any" and side != args.side:
                    continue
                fit, inverse_metrics = fit_mano_to_handdgp(model, hand, args)
                converted_hand, row = convert_hand(frame, hand, fit, inverse_metrics, args)
                converted.append(converted_hand)
                rows.append({"hand_idx": int(hand_i), **row})
            except Exception as exc:
                skipped.append({"frame_idx": int(frame_idx), "hand_idx": int(hand_i), "reason": str(exc)})
        frame["hands"] = converted
    save_json(args.output_annotations, output)
    measured = [row for row in rows if bool(row.get("measurement_available"))]
    report = {
        "status": "ok" if len(measured) >= int(args.min_measured_hands) else "insufficient_measured_hands",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "convert_handdgp_to_mano_candidates_v7",
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "candidate_hands": int(len(rows)),
        "measured_candidate_hands": int(len(measured)),
        "summary": {
            "joint_fit_median_m": summarize_key(rows, "joint_fit_median_m"),
            "joint_fit_p95_m": summarize_key(rows, "joint_fit_p95_m"),
            "median_reprojection_error_px": summarize_key(rows, "median_reprojection_error_px"),
            "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
            "scale": summarize_key(rows, "scale"),
        },
        "thresholds": {
            "max_inverse_joint_median_m": float(args.max_inverse_joint_median_m),
            "max_inverse_joint_p95_m": float(args.max_inverse_joint_p95_m),
            "max_initial_reprojection_px": float(args.max_initial_reprojection_px),
            "min_hand_bone_m": float(args.min_hand_bone_m),
            "max_hand_bone_m": float(args.max_hand_bone_m),
        },
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:180],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: report[k] for k in ("status", "candidate_hands", "measured_candidate_hands")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--mano-right", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--track-id", default="")
    parser.add_argument("--side", choices=["left", "right", "any"], default="any")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-mano-vertices", type=int, default=778)
    parser.add_argument("--max-vertices-per-hand", type=int, default=778)
    parser.add_argument("--min-measured-hands", type=int, default=1)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.50)
    parser.add_argument("--max-initial-reprojection-px", type=float, default=35.0)
    parser.add_argument("--min-hand-bone-m", type=float, default=0.115)
    parser.add_argument("--max-hand-bone-m", type=float, default=0.205)
    parser.add_argument("--max-inverse-joint-median-m", type=float, default=0.018)
    parser.add_argument("--max-inverse-joint-p95-m", type=float, default=0.035)
    parser.add_argument("--sigma-joint-m", type=float, default=0.012)
    parser.add_argument("--sigma-vertex-m", type=float, default=0.012)
    parser.add_argument("--sigma-pose-rad", type=float, default=0.45)
    parser.add_argument("--sigma-orient-rad", type=float, default=0.70)
    parser.add_argument("--sigma-beta", type=float, default=2.5)
    parser.add_argument("--sigma-log-scale", type=float, default=0.30)
    parser.add_argument("--sigma-span-m", type=float, default=0.020)
    parser.add_argument("--min-span-m", type=float, default=0.100)
    parser.add_argument("--max-span-m", type=float, default=0.230)
    parser.add_argument("--min-scale", type=float, default=0.65)
    parser.add_argument("--max-scale", type=float, default=1.60)
    parser.add_argument("--max-pose-rad", type=float, default=1.25)
    parser.add_argument("--max-orient-rad", type=float, default=3.15)
    parser.add_argument("--max-beta", type=float, default=4.0)
    parser.add_argument("--w-joint", type=float, default=1.0)
    parser.add_argument("--w-vertex", type=float, default=0.45)
    parser.add_argument("--w-pose", type=float, default=0.08)
    parser.add_argument("--w-orient", type=float, default=0.03)
    parser.add_argument("--w-beta", type=float, default=0.03)
    parser.add_argument("--w-scale", type=float, default=0.02)
    parser.add_argument("--w-span", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.045)
    parser.add_argument("--iters", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
