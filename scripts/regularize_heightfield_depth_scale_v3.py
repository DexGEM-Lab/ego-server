#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy import sparse


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_archive(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError("archive offsets do not match frame count")
    return {
        "frame_idx": frame_idx,
        "vertex_offsets": vertex_offsets,
        "face_offsets": face_offsets,
        "vertices": vertices,
        "faces": faces,
    }


def frame_annotations(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def frame_vertices(archive: dict, i: int) -> np.ndarray:
    v0 = int(archive["vertex_offsets"][i])
    v1 = int(archive["vertex_offsets"][i + 1])
    vertices = archive["vertices"][v0:v1]
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise RuntimeError(f"invalid vertices for archive index {i}")
    return vertices


def robust_extent(points: np.ndarray, q: float) -> np.ndarray:
    lo = np.quantile(points, float(q), axis=0)
    hi = np.quantile(points, 1.0 - float(q), axis=0)
    extent = hi - lo
    if not np.isfinite(extent).all() or np.any(extent <= 1e-7):
        raise RuntimeError("degenerate robust extent")
    return extent


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def reference_extent(extents_xy: np.ndarray, frame_idx: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.reference_extent_xy_m is not None:
        ref = np.asarray(args.reference_extent_xy_m, dtype=np.float64)
    elif args.reference == "median":
        ref = np.median(extents_xy, axis=0)
    elif args.reference == "anchor":
        if args.anchor_frame is None:
            raise RuntimeError("--anchor-frame is required with --reference anchor")
        matches = np.flatnonzero(frame_idx == int(args.anchor_frame))
        if len(matches) != 1:
            raise RuntimeError(f"anchor frame {args.anchor_frame} appears {len(matches)} times")
        ref = extents_xy[int(matches[0])]
    else:
        raise RuntimeError(f"unsupported reference: {args.reference}")
    if ref.shape != (2,) or not np.isfinite(ref).all() or np.any(ref <= 0.0):
        raise RuntimeError("invalid reference XY extent")
    return ref


def residuals(log_scale: np.ndarray, extents_xy: np.ndarray, ref_xy: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    residual = []
    predicted = np.log(extents_xy) + log_scale[:, None]
    target = np.log(ref_xy)[None, :]
    residual.append(((predicted - target) / float(args.sigma_extent_log)).reshape(-1))
    residual.append(log_scale / float(args.sigma_scale_prior_log))
    if len(log_scale) > 1:
        residual.append(np.diff(log_scale) / float(args.sigma_scale_step_log))
    if len(log_scale) > 2:
        residual.append(np.diff(log_scale, n=2) / float(args.sigma_scale_accel_log))
    return np.concatenate([part.reshape(-1) for part in residual])


def sparsity(n: int) -> sparse.csr_matrix:
    entries = []
    row = 0
    for i in range(n):
        entries.extend([(row, i), (row + 1, i)])
        row += 2
    for i in range(n):
        entries.append((row, i))
        row += 1
    for i in range(n - 1):
        entries.extend([(row, i), (row, i + 1)])
        row += 1
    for i in range(n - 2):
        entries.extend([(row, i), (row, i + 1), (row, i + 2)])
        row += 1
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, n))


def save_archives(
    output_dir: Path,
    archive: dict,
    scaled_vertices_per_frame: list[np.ndarray],
    annotations_path: Path | None,
    scale: np.ndarray,
) -> tuple[Path, Path | None]:
    frame_idx = archive["frame_idx"].astype(np.int32)
    vertex_offsets = archive["vertex_offsets"].astype(np.int64)
    face_offsets = archive["face_offsets"].astype(np.int64)
    faces = archive["faces"].astype(np.int32)
    camera_path = output_dir / "depth_scaled_heightfield_meshes_camera.npz"
    archive_kwargs = {
        "frame_idx": frame_idx,
        "vertex_offsets": vertex_offsets,
        "face_offsets": face_offsets,
        "faces": faces,
        "depth_scale": scale.astype(np.float32),
    }
    np.savez_compressed(camera_path, vertices=np.vstack(scaled_vertices_per_frame).astype(np.float32), **archive_kwargs)
    world_path = None
    if annotations_path is not None:
        annotations = frame_annotations(annotations_path)
        world_vertices = []
        for idx, vertices in zip(frame_idx.astype(int), scaled_vertices_per_frame, strict=True):
            if idx not in annotations:
                raise RuntimeError(f"annotations missing frame {idx}")
            T = np.asarray(annotations[idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
            homog = np.c_[vertices, np.ones(len(vertices), dtype=np.float64)]
            world_vertices.append((T @ homog.T).T[:, :3].astype(np.float32))
        world_path = output_dir / "depth_scaled_heightfield_meshes_world.npz"
        np.savez_compressed(world_path, vertices=np.vstack(world_vertices).astype(np.float32), **archive_kwargs)
    return camera_path, world_path


def run(args: argparse.Namespace) -> dict:
    archive = load_archive(args.camera_mesh_archive)
    frame_idx = archive["frame_idx"].astype(int)
    raw_vertices = [frame_vertices(archive, i) for i in range(len(frame_idx))]
    extents = np.asarray([robust_extent(vertices, float(args.robust_quantile)) for vertices in raw_vertices], dtype=np.float64)
    ref_xy = reference_extent(extents[:, :2], frame_idx, args)
    analytic = np.mean(np.log(ref_xy[None, :]) - np.log(extents[:, :2]), axis=1)
    result = least_squares(
        lambda x: residuals(x, extents[:, :2], ref_xy, args),
        analytic,
        jac_sparsity=sparsity(len(frame_idx)),
        loss="linear",
        x_scale="jac",
        max_nfev=int(args.max_nfev),
    )
    scale = np.exp(result.x)
    if np.any(scale < float(args.min_scale)) or np.any(scale > float(args.max_scale)):
        raise RuntimeError(
            f"depth scale outside [{args.min_scale}, {args.max_scale}]: "
            f"{float(scale.min()):.4f} to {float(scale.max()):.4f}"
        )
    scaled_vertices = [vertices * float(s) for vertices, s in zip(raw_vertices, scale, strict=True)]
    scaled_extents = np.asarray([robust_extent(vertices, float(args.robust_quantile)) for vertices in scaled_vertices], dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_path, world_path = save_archives(args.output_dir, archive, scaled_vertices, args.annotations, scale)
    rows = []
    for idx, raw, scaled, raw_extent, scaled_extent, s in zip(frame_idx, raw_vertices, scaled_vertices, extents, scaled_extents, scale, strict=True):
        rows.append(
            {
                "frame_idx": int(idx),
                "depth_scale": float(s),
                "raw_center_camera_m": np.median(raw, axis=0).astype(float).tolist(),
                "scaled_center_camera_m": np.median(scaled, axis=0).astype(float).tolist(),
                "raw_robust_extent_camera_m": raw_extent.astype(float).tolist(),
                "scaled_robust_extent_camera_m": scaled_extent.astype(float).tolist(),
            }
        )
    report = {
        "status": "ok" if result.success else "optimizer_incomplete",
        "annotation_ready": False,
        "method": "heightfield_depth_scale_regularization_v3",
        "camera_mesh_archive": str(args.camera_mesh_archive),
        "annotations": str(args.annotations) if args.annotations is not None else None,
        "camera_archive": str(camera_path),
        "world_archive": str(world_path) if world_path is not None else None,
        "frames": int(len(frame_idx)),
        "first_frame": int(frame_idx[0]),
        "last_frame": int(frame_idx[-1]),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "reference_xy_extent_m": ref_xy.astype(float).tolist(),
        "depth_scale": summarize(scale),
        "raw_robust_extent_camera_median_m": np.median(extents, axis=0).astype(float).tolist(),
        "scaled_robust_extent_camera_median_m": np.median(scaled_extents, axis=0).astype(float).tolist(),
        "scaled_xy_extent_ratio_to_reference": {
            "x": summarize(scaled_extents[:, 0] / ref_xy[0]),
            "y": summarize(scaled_extents[:, 1] / ref_xy[1]),
        },
        "residual_rms_before": float(np.sqrt(np.mean(residuals(np.zeros(len(frame_idx)), extents[:, :2], ref_xy, args) ** 2))),
        "residual_rms_after": float(np.sqrt(np.mean(residuals(result.x, extents[:, :2], ref_xy, args) ** 2))),
        "parameters": {
            "reference": args.reference,
            "anchor_frame": int(args.anchor_frame) if args.anchor_frame is not None else None,
            "robust_quantile": float(args.robust_quantile),
            "sigma_extent_log": float(args.sigma_extent_log),
            "sigma_scale_prior_log": float(args.sigma_scale_prior_log),
            "sigma_scale_step_log": float(args.sigma_scale_step_log),
            "sigma_scale_accel_log": float(args.sigma_scale_accel_log),
        },
        "rows": rows,
    }
    (args.output_dir / "qc_heightfield_depth_scale_regularization_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", choices=("median", "anchor"), default="median")
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument("--reference-extent-xy-m", type=float, nargs=2)
    parser.add_argument("--robust-quantile", type=float, default=0.05)
    parser.add_argument("--sigma-extent-log", type=float, default=0.055)
    parser.add_argument("--sigma-scale-prior-log", type=float, default=0.38)
    parser.add_argument("--sigma-scale-step-log", type=float, default=0.16)
    parser.add_argument("--sigma-scale-accel-log", type=float, default=0.10)
    parser.add_argument("--min-scale", type=float, default=0.55)
    parser.add_argument("--max-scale", type=float, default=1.80)
    parser.add_argument("--max-nfev", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
