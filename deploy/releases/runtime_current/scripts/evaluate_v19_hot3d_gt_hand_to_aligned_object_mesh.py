#!/usr/bin/env python3
"""Measure HOT3D GT hand distance to a V19 mesh aligned onto HOT3D object poses.

This is an evaluator/attribution tool, not a prediction-stage correction.  It
answers a causal question left open by V19 Workbench item 6: is a large V19
hand/object source gap evidence of a bad MANO/object correction, or is the clip
mostly not in physical contact?

HOT3D object CAD is not always available in the local benchmark copy.  This
script uses the reconstructed V19 object mesh as a common shape proxy, and places
that mesh on the HOT3D object trajectory via the constant object-frame transform
already fitted by ``evaluate_v19_hot3d_object_trajectory_alignment.py``.

Convention: that evaluator computes X ~= inv(T_v19_cam_obj) @ T_hot3d_cam_obj,
so X maps HOT3D object-frame coordinates into the V19 completed-canonical frame.
Therefore a V19 mesh point p_v19 is placed in the HOT3D camera frame by
T_hot3d_cam_obj @ inv(X) @ p_v19.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from evaluate_v19_hot3d_hawor_mano3d import (  # type: ignore
    load_hand_shape,
    load_smplx_mano,
    replay_hot3d_mano,
    se3_from_hot3d_dict,
    world_to_camera,
)
from evaluate_v19_hot3d_object_trajectory_alignment import (  # type: ignore
    make_T,
    quat_wxyz_to_R,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid mesh {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, int(count), seed=np.random.default_rng(seed))
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise RuntimeError("invalid sampled mesh")
    return pts


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=float)], axis=1)
    return (np.asarray(T, dtype=float) @ pts_h.T).T[:, :3]


def summarize(vals: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def load_alignment_transform(alignment_report: Path) -> np.ndarray:
    data = load_json(alignment_report)
    block = data.get("fitted_constant_transform_pred_object_to_hot3d_object")
    if not isinstance(block, dict):
        raise RuntimeError(f"alignment report {alignment_report} lacks fitted transform block")
    # Name is historical.  Per evaluator code, this is X ~= inv(T_v19_cam_obj) @ T_hot3d_cam_obj,
    # mapping HOT3D object coordinates into the V19 completed-canonical coordinates.
    q = block.get("rotation_quaternion_wxyz")
    t = block.get("translation_xyz_m")
    if not (isinstance(q, list) and len(q) == 4 and isinstance(t, list) and len(t) == 3):
        raise RuntimeError("invalid fitted transform fields")
    return make_T(quat_wxyz_to_R(q), t)


def hot3d_frames(sidecar: Path, object_bop_id: str, stream_id: str) -> dict[int, dict[str, Any]]:
    gt = load_json(sidecar)
    out: dict[int, dict[str, Any]] = {}
    for fr in gt.get("frames", []) if isinstance(gt.get("frames"), list) else []:
        if not isinstance(fr, dict):
            continue
        idx = int(fr.get("frame_idx"))
        j = fr.get("json") if isinstance(fr.get("json"), dict) else {}
        cams = j.get("cameras.json") if isinstance(j.get("cameras.json"), dict) else {}
        objs = j.get("objects.json") if isinstance(j.get("objects.json"), dict) else {}
        hands = j.get("hands.json") if isinstance(j.get("hands.json"), dict) else {}
        cam = cams.get(stream_id)
        entries = objs.get(str(object_bop_id)) or []
        if not isinstance(cam, dict) or not entries:
            continue
        obj = entries[0]
        if not isinstance(obj, dict):
            continue
        out[idx] = {"camera": cam, "object": obj, "hands": hands}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt = load_json(args.hot3d_gt)
    beta, hand_shape_source = load_hand_shape(gt, args.hot3d_gt)
    layers = load_smplx_mano(args)
    device = torch.device(args.device)
    for layer in layers.values():
        layer.to(device)
        layer.eval()

    mesh = load_mesh(args.completed_mesh)
    mesh_pts_v19 = sample_mesh(mesh, int(args.mesh_sample_count), int(args.seed))
    X_hot3d_to_v19 = load_alignment_transform(args.object_alignment_report)
    X_v19_to_hot3d = np.linalg.inv(X_hot3d_to_v19)
    frames = hot3d_frames(args.hot3d_gt, str(args.object_bop_id), str(args.stream_id))
    if not frames:
        raise RuntimeError("no HOT3D frames with object/camera")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    hand_cache: dict[tuple[int, str], np.ndarray] = {}
    sigma = math.sqrt(args.hand_sigma_m**2 + args.object_sigma_m**2 + args.depth_order_sigma_m**2)

    for idx in sorted(frames):
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        rec = frames[idx]
        obj = rec["object"]
        cam = rec["camera"]
        vis = (obj.get("visibilities_modeled") or {}).get(args.stream_id) if isinstance(obj.get("visibilities_modeled"), dict) else None
        if vis is not None and float(vis) < float(args.min_visibility):
            continue
        R_world_cam, t_world_cam = se3_from_hot3d_dict(cam["T_world_from_camera"])
        R_world_obj, t_world_obj = se3_from_hot3d_dict(obj["T_world_from_object"])
        T_world_cam = make_T(R_world_cam, t_world_cam)
        T_world_obj = make_T(R_world_obj, t_world_obj)
        T_cam_obj = np.linalg.inv(T_world_cam) @ T_world_obj
        mesh_cam = transform_points(T_cam_obj @ X_v19_to_hot3d, mesh_pts_v19)
        tree = cKDTree(mesh_cam)
        for side in ("left", "right"):
            hand = rec.get("hands", {}).get(side) if isinstance(rec.get("hands"), dict) else None
            if not isinstance(hand, dict) or "mano_pose" not in hand:
                skipped.append({"frame_idx": idx, "side": side, "reason": "missing_gt_hand"})
                continue
            key = (idx, side)
            if key not in hand_cache:
                mano = hand["mano_pose"]
                gt_verts_w, _, _ = replay_hot3d_mano(
                    layers[side],
                    beta,
                    np.asarray(mano["thetas"], dtype=np.float32),
                    np.asarray(mano["wrist_xform"], dtype=np.float32),
                    device,
                )
                R_cam, t_cam = T_world_cam[:3, :3], T_world_cam[:3, 3]
                hand_cache[key] = world_to_camera(gt_verts_w.astype(np.float64), R_cam, t_cam)
            hand_cam = hand_cache[key]
            d, _ = tree.query(hand_cam, k=1, workers=-1)
            rows.append(
                {
                    "frame_idx": idx,
                    "side": side,
                    "object_visibility_modeled": vis,
                    "hand_visibility_modeled": (hand.get("visibilities_modeled") or {}).get(args.stream_id) if isinstance(hand.get("visibilities_modeled"), dict) else None,
                    "gt_hand_vertex_count": int(hand_cam.shape[0]),
                    "mesh_sample_count": int(mesh_cam.shape[0]),
                    "gt_hand_to_aligned_v19_mesh_min_m": float(np.min(d)),
                    "gt_hand_to_aligned_v19_mesh_p01_m": float(np.percentile(d, 1.0)),
                    "gt_hand_to_aligned_v19_mesh_p05_m": float(np.percentile(d, 5.0)),
                    "gt_hand_to_aligned_v19_mesh_p10_m": float(np.percentile(d, 10.0)),
                    "gt_hand_to_aligned_v19_mesh_median_m": float(np.median(d)),
                    "contact_compatibility_min_vertex": math.exp(-0.5 * (float(np.min(d)) / sigma) ** 2),
                    "contact_compatibility_p05_vertex": math.exp(-0.5 * (float(np.percentile(d, 5.0)) / sigma) ** 2),
                }
            )
    if not rows:
        raise RuntimeError(f"no evaluable hand/object rows; skipped={skipped[:10]}")

    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sr = [r for r in rows if r["side"] == side]
        by_side[side] = {
            "row_count": len(sr),
            "min_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_min_m"] for r in sr),
            "p05_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_p05_m"] for r in sr),
            "p10_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_p10_m"] for r in sr),
            "median_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_median_m"] for r in sr),
            "rows_min_below_1sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= sigma for r in sr)),
            "rows_min_below_2sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= 2 * sigma for r in sr)),
            "rows_min_below_3sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= 3 * sigma for r in sr)),
            "rows_p05_below_3sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_p05_m"] <= 3 * sigma for r in sr)),
        }

    report = {
        "status": "ok",
        "method": "evaluate_v19_hot3d_gt_hand_to_aligned_object_mesh",
        "claim_scope": "Evaluation-side contact-truth attribution using HOT3D GT MANO/object poses and the V19 reconstructed mesh aligned into HOT3D object frame; does not modify prediction state and cannot replace HOT3D CAD contact if CAD is later available.",
        "inputs": {
            "hot3d_gt": str(args.hot3d_gt),
            "completed_mesh": str(args.completed_mesh),
            "object_alignment_report": str(args.object_alignment_report),
            "object_bop_id": str(args.object_bop_id),
            "stream_id": str(args.stream_id),
            "mesh_sample_count": int(args.mesh_sample_count),
            "hand_shape_source": hand_shape_source,
        },
        "transform_convention": {
            "X_hot3d_to_v19_from_alignment_report": X_hot3d_to_v19.astype(float).tolist(),
            "use_for_mesh": "mesh_cam = inv(T_world_camera) @ T_world_object @ inv(X_hot3d_to_v19) @ mesh_v19",
        },
        "compatibility_model": {
            "combined_sigma_m": sigma,
            "hand_sigma_m": float(args.hand_sigma_m),
            "object_sigma_m": float(args.object_sigma_m),
            "depth_order_sigma_m": float(args.depth_order_sigma_m),
        },
        "row_count": len(rows),
        "summary": {
            "min_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_min_m"] for r in rows),
            "p01_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_p01_m"] for r in rows),
            "p05_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_p05_m"] for r in rows),
            "p10_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_p10_m"] for r in rows),
            "median_distance_m": summarize(r["gt_hand_to_aligned_v19_mesh_median_m"] for r in rows),
            "rows_min_below_1sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= sigma for r in rows)),
            "rows_min_below_2sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= 2 * sigma for r in rows)),
            "rows_min_below_3sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_min_m"] <= 3 * sigma for r in rows)),
            "rows_p05_below_3sigma": int(sum(r["gt_hand_to_aligned_v19_mesh_p05_m"] <= 3 * sigma for r in rows)),
        },
        "by_side": by_side,
        "interpretation_notes": [
            "Small min distances on HOT3D GT imply at least one hand surface point is near the object mesh; large V19 selected-source gaps would then implicate target selection/object pose/hand choice rather than true non-contact.",
            "Large GT min and p05 distances imply the interval is mostly non-contact under the aligned V19 shape proxy, so forcing contact closure would be a false correction target.",
            "Because the mesh is V19-reconstructed rather than HOT3D CAD, this is a contact-truth attribution proxy, not signed nonpenetration or official contact ground truth.",
        ],
        "rows": rows,
        "skipped_preview": skipped[:40],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "row_count": len(rows), "summary": report["summary"], "by_side": by_side}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hot3d-gt", type=Path, required=True)
    ap.add_argument("--completed-mesh", type=Path, required=True)
    ap.add_argument("--object-alignment-report", type=Path, required=True)
    ap.add_argument("--object-bop-id", required=True)
    ap.add_argument("--stream-id", default="214-1")
    ap.add_argument("--mano-left", type=Path, required=True)
    ap.add_argument("--mano-right", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mesh-sample-count", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=4217)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--hand-sigma-m", type=float, default=0.027)
    ap.add_argument("--object-sigma-m", type=float, default=0.010)
    ap.add_argument("--depth-order-sigma-m", type=float, default=0.010)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
