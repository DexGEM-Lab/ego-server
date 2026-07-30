#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from align_mesh_prior_v3 import Sim3, load_observed_frame


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rodrigues(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    x, y, z = axis
    K = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx].astype(float)


def observed_mesh_points(mesh_npz: Path, frame_idx: int, max_points: int, seed: int) -> np.ndarray:
    vertices, _ = load_observed_frame(mesh_npz, frame_idx)
    return sample_points(vertices, max_points, seed)


def load_initial_sim3(path: Path) -> Sim3:
    report = load_json(path)
    sim = report["sim3"]
    return Sim3(
        scale=float(sim["scale"]),
        rotation=np.asarray(sim["rotation"], dtype=float),
        translation=np.asarray(sim["translation"], dtype=float),
    )


def active_frames(annotations: dict, start: int, end: int) -> list[dict]:
    frames = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        if start <= idx <= end and frame.get("object", {}).get("status") in {"measured_plan_sam", "measured_plan_sam_vlm_verified", "measured_sam_kalman"}:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no active measured frames in requested window")
    return frames


def hand_points(frame: dict, max_points: int) -> np.ndarray:
    pts = []
    for hand in frame.get("hands", []):
        if "joints3d_world_m" in hand:
            pts.append(np.asarray(hand["joints3d_world_m"], dtype=float))
        for key in ("vertices_world_m", "vertices_sample_world_m", "vertices3d_world_m"):
            if key in hand:
                vertices = np.asarray(hand[key], dtype=float)
                if len(vertices):
                    pts.append(sample_points(vertices, max_points, int(frame["frame_idx"]) + len(pts)))
                break
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def transform_vertices(base_vertices: np.ndarray, base_sim: Sim3, params: np.ndarray) -> np.ndarray:
    scale = base_sim.scale * float(np.exp(params[0]))
    rotation = rodrigues(params[1:4]) @ base_sim.rotation
    translation = base_sim.translation + params[4:7]
    return scale * (base_vertices @ rotation.T) + translation


def residuals(
    params: np.ndarray,
    base_vertices: np.ndarray,
    base_sim: Sim3,
    observed_by_frame: dict[int, np.ndarray],
    hands_by_frame: dict[int, np.ndarray],
    prior_points: np.ndarray,
    sigma_obs: float,
    sigma_pen: float,
    sigma_contact: float,
    contact_distance_m: float,
    prior_sigma: float,
) -> np.ndarray:
    transformed_vertices = transform_vertices(base_vertices, base_sim, params)
    tree = cKDTree(transformed_vertices)
    res = []
    for frame_idx, observed in observed_by_frame.items():
        d_obs, _ = tree.query(observed, k=1)
        res.append(np.clip(d_obs / sigma_obs, 0.0, 8.0))
        hands = hands_by_frame.get(frame_idx)
        if hands is None or len(hands) == 0:
            continue
        d_hand, _ = tree.query(hands, k=1)
        # Keep residual length fixed across evaluations; inactive proximity terms contribute zero.
        res.append(np.maximum(0.0, contact_distance_m - d_hand) / sigma_pen)
        res.append(np.asarray([np.min(d_hand) / sigma_contact], dtype=float))
    prior_transformed = transform_vertices(prior_points, base_sim, params)
    base_transformed = base_sim.apply(prior_points)
    res.append(((prior_transformed - base_transformed).reshape(-1)) / prior_sigma)
    res.append(params / np.asarray([0.15, 0.35, 0.35, 0.35, 0.080, 0.080, 0.080], dtype=float))
    return np.concatenate([r.reshape(-1) for r in res])


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = active_frames(annotations, args.frame_start, args.frame_end)
    prior_mesh = trimesh.load(args.mesh_prior, force="mesh", process=False)
    if not isinstance(prior_mesh, trimesh.Trimesh) or len(prior_mesh.vertices) == 0:
        raise RuntimeError(f"invalid mesh prior: {args.mesh_prior}")
    base_vertices = np.asarray(prior_mesh.vertices, dtype=float)
    base_sim = load_initial_sim3(args.initial_alignment_qc)
    observed_by_frame = {}
    hands_by_frame = {}
    used_frames = []
    skipped_frames = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        try:
            observed_by_frame[idx] = observed_mesh_points(args.observed_mesh_npz, idx, args.max_observed_points, args.seed + idx)
        except Exception as exc:
            skipped_frames.append({"frame_idx": idx, "reason": str(exc)})
            continue
        hands_by_frame[idx] = hand_points(frame, args.max_hand_points)
        used_frames.append(idx)
    if len(used_frames) < 2:
        raise RuntimeError("too few frames with observed meshes for window optimization")
    prior_points = sample_points(base_vertices, args.max_prior_points, args.seed)
    initial = np.zeros(7, dtype=float)
    before = residuals(
        initial,
        base_vertices,
        base_sim,
        observed_by_frame,
        hands_by_frame,
        prior_points,
        args.sigma_obs,
        args.sigma_pen,
        args.sigma_contact,
        args.contact_distance_m,
        args.prior_sigma,
    )
    result = least_squares(
        lambda x: residuals(
            x,
            base_vertices,
            base_sim,
            observed_by_frame,
            hands_by_frame,
            prior_points,
            args.sigma_obs,
            args.sigma_pen,
            args.sigma_contact,
            args.contact_distance_m,
            args.prior_sigma,
        ),
        initial,
        max_nfev=args.max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )
    aligned_vertices = transform_vertices(base_vertices, base_sim, result.x)
    out_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=np.asarray(prior_mesh.faces), process=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / f"window_prior_{args.frame_start:06d}_{args.frame_end:06d}.obj"
    out_mesh.export(mesh_path)
    after = residuals(
        result.x,
        base_vertices,
        base_sim,
        observed_by_frame,
        hands_by_frame,
        prior_points,
        args.sigma_obs,
        args.sigma_pen,
        args.sigma_contact,
        args.contact_distance_m,
        args.prior_sigma,
    )
    frame_metrics = {}
    tree = cKDTree(aligned_vertices)
    for idx, observed in observed_by_frame.items():
        d_obs, _ = tree.query(observed, k=1)
        hands = hands_by_frame.get(idx, np.zeros((0, 3), dtype=float))
        if len(hands):
            d_hand, _ = tree.query(hands, k=1)
            hand_min = float(np.min(d_hand))
            hand_med = float(np.median(d_hand))
        else:
            hand_min = None
            hand_med = None
        frame_metrics[str(idx)] = {
            "observed_to_prior_median_m": float(np.median(d_obs)),
            "observed_to_prior_p95_m": float(np.percentile(d_obs, 95.0)),
            "hand_to_prior_min_m": hand_min,
            "hand_to_prior_median_m": hand_med,
        }
    status = "optimizer_converged" if result.success else "optimizer_failed"
    report = {
        "status": status,
        "interpretation": "prototype_metrics_only",
        "mesh_prior": str(args.mesh_prior),
        "initial_alignment_qc": str(args.initial_alignment_qc),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "annotations": str(args.annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "used_frames": used_frames,
        "skipped_frames": skipped_frames,
        "variables": 7,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        "params": result.x.astype(float).tolist(),
        "mesh_out": str(mesh_path),
        "frame_metrics": frame_metrics,
    }
    (args.output_dir / "qc_optimize_mesh_prior_window_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--initial-alignment-qc", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-observed-points", type=int, default=800)
    parser.add_argument("--max-hand-points", type=int, default=160)
    parser.add_argument("--max-prior-points", type=int, default=600)
    parser.add_argument("--sigma-obs", type=float, default=0.030)
    parser.add_argument("--sigma-pen", type=float, default=0.020)
    parser.add_argument("--sigma-contact", type=float, default=0.015)
    parser.add_argument("--contact-distance-m", type=float, default=0.010)
    parser.add_argument("--prior-sigma", type=float, default=0.060)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
