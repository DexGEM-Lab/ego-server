#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_contact_depth_scale_v3 import summarize
from optimize_hand_translation_contact_v3 import mesh_vertices_by_frame, object_camera_depth, source_to_world
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


TIP_JOINT_IDS = np.asarray([4, 8, 12, 16, 20], dtype=int)


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


@dataclass(frozen=True)
class FitInput:
    frame_idx: int
    hand_index: int
    side: str
    score: float
    raw2d: torch.Tensor
    intrinsics: torch.Tensor
    metric_depth: torch.Tensor
    depth_valid: torch.Tensor
    base_global_orient: torch.Tensor
    base_hand_pose: torch.Tensor
    betas: torch.Tensor
    base_cam_t: torch.Tensor
    base_local_joints: np.ndarray
    base_local_vertices: np.ndarray
    base_joints_source: np.ndarray
    object_depth_m: float
    contact_vertex_ids: torch.Tensor
    T_world_camera: np.ndarray


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project_torch(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-4)
    fx, fy, cx, cy = intrinsics
    return torch.stack([fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy], dim=-1)


def robust_l1(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x * x + 1e-6)


def hand_span_torch(joints: torch.Tensor) -> torch.Tensor:
    tips = joints[:, TIP_JOINT_IDS, :][0]
    dist = torch.linalg.norm(tips[:, None, :] - tips[None, :, :], dim=-1)
    return dist.max()


def load_wilor_mano_class(wilor_root: Path):
    path = wilor_root / "wilor" / "models" / "mano_wrapper.py"
    if not path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO wrapper: {path}")
    spec = importlib.util.spec_from_file_location("wilor_mano_wrapper_for_v3_refit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WiLoR MANO wrapper spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO


def side_sign(side: str) -> float:
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise RuntimeError(f"unsupported hand side {side}")


def skew(rotvec: torch.Tensor) -> torch.Tensor:
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(skew(rotvec))


def apply_side_sign(points: torch.Tensor, sign: float) -> torch.Tensor:
    out = points.clone()
    out[..., 0] = float(sign) * out[..., 0]
    return out


def similarity_from_to(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape or source.shape[1] != 3:
        raise RuntimeError("similarity alignment expects matching Nx3 arrays")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    tgt = target - target_center
    covariance = src.T @ tgt / len(source)
    u, s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    variance = float(np.sum(src * src) / len(source))
    if variance <= 0.0 or not np.isfinite(variance):
        raise RuntimeError("degenerate MANO canonical vertices for similarity alignment")
    scale = float(np.sum(s) / variance)
    translation = target_center - scale * (source_center @ rotation.T)
    aligned = scale * (source @ rotation.T) + translation[None, :]
    error = np.linalg.norm(aligned - target, axis=1)
    return scale, rotation, translation, error


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def contact_vertex_ids(vertices_source: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray, depth: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    depth_mask = resize_mask_to_depth(mask, depth)
    dist = mask_distance_map(depth_mask)
    depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    uv = project_points(vertices_source, intrinsics)
    xy = uv * depth_scale[None, :]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(vertices_source).all(axis=1) & (vertices_source[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = np.flatnonzero(valid & (dist[y, x] <= args.contact_distance_px))
    if len(near) <= args.max_contact_vertices:
        return near.astype(int)
    order = np.argsort(np.abs(vertices_source[near, 2] - np.median(vertices_source[near, 2])))[: args.max_contact_vertices]
    return near[order].astype(int)


def build_inputs(args: argparse.Namespace) -> tuple[dict, list[FitInput], list[dict]]:
    data = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    object_vertices = mesh_vertices_by_frame(args.object_mesh_npz)
    out: list[FitInput] = []
    skipped: list[dict] = []
    for frame in data["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        obj = frame.get("object", {})
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            object_depth = object_camera_depth(object_vertices[frame_idx], T_world_camera)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            score = float(hand.get("detector_score", np.nan))
            if not hand.get("measurement_available", False) or not np.isfinite(score) or score < args.min_detector_score:
                continue
            try:
                raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
                intr = np.asarray(hand["source_intrinsics"], dtype=float)
                base_local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                base_local_vertices = np.asarray(hand["vertices_camera"], dtype=float)
                base_joints_source = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                base_vertices_source = np.asarray(hand["vertices_source_camera_m"], dtype=float)
                if base_local_joints.shape != (21, 3) or base_local_vertices.ndim != 2 or base_local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid local MANO geometry")
                projected = project_points(base_joints_source, intr)
                reproj = np.linalg.norm(projected - raw2d, axis=1)
                metric_depth = sample_depth(depth, raw2d, source_size)
                depth_valid = np.isfinite(metric_depth) & (metric_depth > 0.0) & (reproj <= args.good_joint_reprojection_px)
                if np.count_nonzero(depth_valid) < args.min_depth_joints:
                    depth_valid = np.zeros(21, dtype=bool)
                contact_ids = contact_vertex_ids(base_vertices_source, intr, mask, depth, source_size, args)
                params = hand["mano_params"]
                global_orient = np.asarray(params["global_orient"], dtype=float)
                hand_pose = np.asarray(params["hand_pose"], dtype=float)
                betas = np.asarray(params["betas"], dtype=float)
                if global_orient.shape != (1, 3, 3) or hand_pose.shape != (15, 3, 3) or betas.shape != (10,):
                    raise RuntimeError("WiLoR MANO parameters must be rotation matrices plus 10 betas")
                out.append(
                    FitInput(
                        frame_idx=frame_idx,
                        hand_index=hand_i,
                        side=str(hand.get("side")),
                        score=score,
                        raw2d=torch.tensor(raw2d, dtype=torch.float32),
                        intrinsics=torch.tensor(intr, dtype=torch.float32),
                        metric_depth=torch.tensor(np.nan_to_num(metric_depth, nan=0.0), dtype=torch.float32),
                        depth_valid=torch.tensor(depth_valid, dtype=torch.bool),
                        base_global_orient=torch.tensor(global_orient[None], dtype=torch.float32),
                        base_hand_pose=torch.tensor(hand_pose[None], dtype=torch.float32),
                        betas=torch.tensor(betas[None], dtype=torch.float32),
                        base_cam_t=torch.tensor(np.asarray(hand["cam_t"], dtype=float)[None], dtype=torch.float32),
                        base_local_joints=base_local_joints,
                        base_local_vertices=base_local_vertices,
                        base_joints_source=base_joints_source,
                        object_depth_m=float(object_depth),
                        contact_vertex_ids=torch.tensor(contact_ids, dtype=torch.long),
                        T_world_camera=T_world_camera,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
    if not out:
        raise RuntimeError("no measured hands for MANO pose refit")
    return data, out, skipped


def fit_one(model, item: FitInput, args: argparse.Namespace) -> tuple[dict, dict]:
    device = torch.device("cpu")
    model = model.to(device)
    sign = side_sign(item.side)
    with torch.no_grad():
        base_out = model(
            global_orient=item.base_global_orient.to(device),
            hand_pose=item.base_hand_pose.to(device),
            betas=item.betas.to(device),
            return_verts=True,
            pose2rot=False,
        )
        base_vertices = apply_side_sign(base_out.vertices, sign)[0].cpu().numpy()
        base_joints = apply_side_sign(base_out.joints, sign)[0].cpu().numpy()
    local_scale, local_rotation, local_translation, local_vertex_error = similarity_from_to(base_vertices, item.base_local_vertices)
    aligned_base_joints = local_scale * (base_joints @ local_rotation.T) + local_translation[None, :]
    joint_error = np.linalg.norm(aligned_base_joints - item.base_local_joints, axis=1)
    if float(np.median(local_vertex_error)) > args.max_zero_state_vertex_error_m:
        raise RuntimeError(
            f"WiLoR MANO zero-state vertex mismatch {float(np.median(local_vertex_error)):.6f}m for frame {item.frame_idx}"
        )
    if float(np.median(joint_error)) > args.max_zero_state_joint_error_m:
        raise RuntimeError(
            f"WiLoR MANO zero-state joint mismatch {float(np.median(joint_error)):.6f}m for frame {item.frame_idx}"
        )
    local_rotation_t = torch.tensor(local_rotation, dtype=torch.float32, device=device)
    local_translation_t = torch.tensor(local_translation, dtype=torch.float32, device=device).reshape(1, 1, 3)
    local_scale_t = torch.tensor(float(local_scale), dtype=torch.float32, device=device)

    pose_delta = torch.zeros((1, 15, 3), dtype=torch.float32, requires_grad=True)
    orient_delta = torch.zeros((1, 1, 3), dtype=torch.float32, requires_grad=True)
    trans_delta = torch.zeros_like(item.base_cam_t, requires_grad=True)
    log_scale = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([pose_delta, orient_delta, trans_delta, log_scale], lr=args.lr)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        global_orient = rotvec_to_matrix(orient_delta) @ item.base_global_orient.to(device)
        hand_pose = rotvec_to_matrix(pose_delta) @ item.base_hand_pose.to(device)
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=item.betas.to(device), return_verts=True, pose2rot=False)
        canonical_vertices = apply_side_sign(out.vertices, sign)
        canonical_joints = apply_side_sign(out.joints, sign)
        local_vertices = local_scale_t * torch.matmul(canonical_vertices, local_rotation_t.T) + local_translation_t
        local_joints = local_scale_t * torch.matmul(canonical_joints, local_rotation_t.T) + local_translation_t
        scale = torch.exp(log_scale).reshape(1, 1, 1)
        vertices = scale * local_vertices + item.base_cam_t[:, None, :].to(device) + trans_delta[:, None, :]
        joints = scale * local_joints + item.base_cam_t[:, None, :].to(device) + trans_delta[:, None, :]
        uv = project_torch(joints[0], item.intrinsics)
        reproj = robust_l1((uv - item.raw2d) / args.sigma_reprojection_px).mean()
        depth_loss = torch.tensor(0.0)
        if bool(item.depth_valid.any()):
            depth_loss = robust_l1((joints[0, item.depth_valid, 2] - item.metric_depth[item.depth_valid]) / args.sigma_metric_depth_m).mean()
        contact_loss = torch.tensor(0.0)
        if int(item.contact_vertex_ids.numel()) >= args.min_near_vertices:
            gap = torch.median(vertices[0, item.contact_vertex_ids, 2] - float(item.object_depth_m))
            contact_loss = robust_l1(gap / args.sigma_contact_depth_m)
        span = hand_span_torch(joints)
        span_loss = robust_l1(torch.relu(args.min_span_m - span) / args.sigma_span_m) + robust_l1(torch.relu(span - args.max_span_m) / args.sigma_span_m)
        pose_loss = robust_l1(pose_delta / args.sigma_pose_delta_rad).mean()
        orient_loss = robust_l1(orient_delta / args.sigma_orient_delta_rad).mean()
        trans_loss = robust_l1(trans_delta / args.sigma_translation_m).mean()
        scale_loss = robust_l1(log_scale / args.sigma_log_scale).mean()
        loss = (
            args.w_reprojection * reproj
            + args.w_depth * depth_loss
            + args.w_contact * contact_loss
            + args.w_span * span_loss
            + args.w_pose * pose_loss
            + args.w_orient * orient_loss
            + args.w_translation * trans_loss
            + args.w_scale * scale_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(np.log(args.min_scale), np.log(args.max_scale))
            pose_delta.clamp_(-args.max_pose_delta_rad, args.max_pose_delta_rad)
            orient_delta.clamp_(-args.max_orient_delta_rad, args.max_orient_delta_rad)
            trans_delta.clamp_(-args.max_translation_m, args.max_translation_m)
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_state = {
                    "global_orient": global_orient.detach().clone(),
                    "hand_pose": hand_pose.detach().clone(),
                    "local_vertices": local_vertices.detach().clone(),
                    "local_joints": local_joints.detach().clone(),
                    "vertices": vertices.detach().clone(),
                    "joints": joints.detach().clone(),
                    "scale": torch.exp(log_scale.detach()).clone(),
                    "trans_delta": trans_delta.detach().clone(),
                    "pose_delta": pose_delta.detach().clone(),
                    "orient_delta": orient_delta.detach().clone(),
                    "loss": loss.detach().clone(),
                    "reproj": reproj.detach().clone(),
                    "depth": depth_loss.detach().clone(),
                    "contact": contact_loss.detach().clone(),
                    "span_loss": span_loss.detach().clone(),
                }
    if best_state is None:
        raise RuntimeError("MANO optimization produced no state")
    vertices_np = best_state["vertices"][0].cpu().numpy()
    joints_np = best_state["joints"][0].cpu().numpy()
    uv_np = project_points(joints_np, item.intrinsics.cpu().numpy())
    reproj_np = np.linalg.norm(uv_np - item.raw2d.cpu().numpy(), axis=1)
    depth_mask = item.depth_valid.cpu().numpy()
    depth_err = joints_np[depth_mask, 2] - item.metric_depth.cpu().numpy()[depth_mask]
    contact_ids = item.contact_vertex_ids.cpu().numpy()
    contact_gap = vertices_np[contact_ids, 2] - item.object_depth_m if len(contact_ids) else np.asarray([], dtype=float)
    fit = {
        "vertices": vertices_np,
        "joints": joints_np,
        "joints2d": uv_np,
        "global_orient": best_state["global_orient"].cpu().numpy(),
        "hand_pose": best_state["hand_pose"].cpu().numpy(),
        "local_vertices": best_state["local_vertices"][0].cpu().numpy(),
        "local_joints": best_state["local_joints"][0].cpu().numpy(),
        "scale": float(best_state["scale"]),
        "trans_delta": best_state["trans_delta"][0].cpu().numpy(),
    }
    metrics = {
        "frame_idx": int(item.frame_idx),
        "side": item.side,
        "loss": best_loss,
        "joint_reprojection_px_median": float(np.median(reproj_np)),
        "joint_reprojection_px_p95": float(np.percentile(reproj_np, 95.0)),
        "depth_joints": int(np.count_nonzero(depth_mask)),
        "mano_minus_metric_depth_median_m": None if len(depth_err) == 0 else float(np.median(depth_err)),
        "mano_minus_metric_depth_p95_abs_m": None if len(depth_err) == 0 else float(np.percentile(np.abs(depth_err), 95.0)),
        "near_mask_vertices": int(len(contact_ids)),
        "contact_gap_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "contact_gap_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
        "hand_span_m": float(hand_span_torch(torch.tensor(joints_np[None], dtype=torch.float32))),
        "scale": fit["scale"],
        "translation_delta_norm_m": float(np.linalg.norm(fit["trans_delta"])),
        "pose_delta_abs_max_rad": float(torch.max(torch.abs(best_state["pose_delta"]))),
        "orient_delta_abs_max_rad": float(torch.max(torch.abs(best_state["orient_delta"]))),
        "zero_state_vertex_error_median_m": float(np.median(local_vertex_error)),
        "zero_state_vertex_error_p95_m": float(np.percentile(local_vertex_error, 95.0)),
        "zero_state_joint_error_median_m": float(np.median(joint_error)),
    }
    return fit, metrics


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return summarize(np.asarray(values, dtype=float))


def apply_fits(data: dict, fits: dict[tuple[int, str, int], dict], args: argparse.Namespace) -> dict:
    out = copy.deepcopy(data)
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, str(hand.get("side")), hand_i)
            fit = fits.get(key)
            if fit is None:
                continue
            hand["vertices_camera"] = fit["local_vertices"].astype(float).tolist()
            hand["joints3d_camera"] = fit["local_joints"].astype(float).tolist()
            hand["vertices_source_camera_m"] = fit["vertices"].astype(float).tolist()
            hand["joints3d_source_camera_m"] = fit["joints"].astype(float).tolist()
            hand["joints2d"] = fit["joints2d"].astype(float).tolist()
            err = np.linalg.norm(fit["joints2d"] - np.asarray(hand["joints2d_raw"], dtype=float), axis=1)
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(err)),
                "p95": float(np.percentile(err, 95.0)),
            }
            hand["joints3d_world_m"] = source_to_world(fit["joints"], T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(fit["vertices"], T_world_camera).astype(float).tolist()
            hand["mano_params"]["global_orient"] = fit["global_orient"].reshape(1, 3, 3).astype(float).tolist()
            hand["mano_params"]["hand_pose"] = fit["hand_pose"].reshape(15, 3, 3).astype(float).tolist()
            hand["mano_params"]["rotation_convention"] = "wilor_rotation_matrix_pose2rot_false_with_side_x_sign"
            hand["v3_mano_pose_contact_refit"] = {
                "scale": float(fit["scale"]),
                "translation_delta_m": fit["trans_delta"].astype(float).tolist(),
                "source_mano_convention": "wilor_right_mano_wrapper_pose2rot_false",
            }
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_mano_pose_contact_refit"
            hand["world_coordinate_status"] = "v3_mano_pose_contact_refit_source_camera_mano_transformed_by_existing_camera_pose"
    return out


def run(args: argparse.Namespace) -> dict:
    patch_legacy_mano_loader()
    data, items, skipped = build_inputs(args)
    mano_cls = load_wilor_mano_class(args.wilor_root)
    mano_model_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_model_path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO_RIGHT model: {mano_model_path}")
    model = mano_cls(model_path=str(mano_model_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1)
    fits: dict[tuple[int, str, int], dict] = {}
    rows: list[dict] = []
    for item in items:
        fit, metrics = fit_one(model, item, args)
        fits[(item.frame_idx, item.side, item.hand_index)] = fit
        rows.append(metrics)
    output = apply_fits(data, fits, args)
    save_json(args.output_annotations, output)
    report = {
        "status": "diagnostic_mano_pose_contact_refit",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "fit_rows": int(len(rows)),
        "summary": {
            "joint_reprojection_px": summarize_key(rows, "joint_reprojection_px_median"),
            "mano_minus_metric_depth_m": summarize_key(rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(rows, "contact_gap_median_m"),
            "hand_span_m": summarize_key(rows, "hand_span_m"),
            "scale": summarize_key(rows, "scale"),
            "translation_delta_norm_m": summarize_key(rows, "translation_delta_norm_m"),
            "pose_delta_abs_max_rad": summarize_key(rows, "pose_delta_abs_max_rad"),
            "zero_state_vertex_error_m": summarize_key(rows, "zero_state_vertex_error_median_m"),
            "zero_state_joint_error_m": summarize_key(rows, "zero_state_joint_error_median_m"),
        },
        "thresholds": {
            "min_span_m": float(args.min_span_m),
            "max_span_m": float(args.max_span_m),
            "min_near_vertices": int(args.min_near_vertices),
            "max_zero_state_vertex_error_m": float(args.max_zero_state_vertex_error_m),
            "max_zero_state_joint_error_m": float(args.max_zero_state_joint_error_m),
        },
        "rows_preview": rows[:120],
        "skipped_preview": skipped[:120],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--min-depth-joints", type=int, default=8)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--max-contact-vertices", type=int, default=180)
    parser.add_argument("--sigma-reprojection-px", type=float, default=18.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-contact-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-span-m", type=float, default=0.020)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.35)
    parser.add_argument("--sigma-orient-delta-rad", type=float, default=0.25)
    parser.add_argument("--sigma-translation-m", type=float, default=0.080)
    parser.add_argument("--sigma-log-scale", type=float, default=0.20)
    parser.add_argument("--min-span-m", type=float, default=0.110)
    parser.add_argument("--max-span-m", type=float, default=0.210)
    parser.add_argument("--min-scale", type=float, default=0.80)
    parser.add_argument("--max-scale", type=float, default=1.30)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.80)
    parser.add_argument("--max-orient-delta-rad", type=float, default=0.60)
    parser.add_argument("--max-translation-m", type=float, default=0.35)
    parser.add_argument("--max-zero-state-vertex-error-m", type=float, default=0.025)
    parser.add_argument("--max-zero-state-joint-error-m", type=float, default=0.025)
    parser.add_argument("--w-reprojection", type=float, default=1.0)
    parser.add_argument("--w-depth", type=float, default=0.8)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-span", type=float, default=1.0)
    parser.add_argument("--w-pose", type=float, default=0.25)
    parser.add_argument("--w-orient", type=float, default=0.25)
    parser.add_argument("--w-translation", type=float, default=0.25)
    parser.add_argument("--w-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--iters", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
