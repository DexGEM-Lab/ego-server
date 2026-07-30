#!/usr/bin/env python3
"""Adapt one DROID D4 reconstruction to the HaWoR SLAM file contract.

The adapter deliberately consumes the raw trajectory/disparity products emitted
by ``run_droid_full_frame.py``. It does not import or instantiate DROID. The
only model work performed here is the existing HaWoR Metric3D scale estimate;
HaWoR's hand tracking, infilling, and MANO export remain downstream consumers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "v22_droid_to_hawor_adapter.v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def resolve_artifact(manifest_path: Path, payload: Any, label: str) -> Path:
    if not isinstance(payload, dict) or not payload.get("path"):
        raise RuntimeError(f"shared DROID manifest lacks {label} artifact path: {manifest_path}")
    path = Path(str(payload["path"])).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"shared DROID {label} artifact missing: {path}")
    expected_hash = payload.get("sha256")
    if expected_hash:
        actual_hash = sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                f"shared DROID {label} artifact hash mismatch: {path}; "
                f"expected={expected_hash} actual={actual_hash}"
            )
    return path


def load_reconstruction(path: Path) -> dict[str, np.ndarray | str]:
    """Load the D4 reconstruction without requiring torch for NPZ test fixtures."""
    if path.suffix.lower() == ".npz":
        blob = np.load(path, allow_pickle=True)
        out: dict[str, np.ndarray | str] = {key: np.asarray(blob[key]) for key in blob.files}
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised on runtime host
            raise RuntimeError(f"loading DROID reconstruction requires torch: {path}") from exc
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch without weights_only
            raw = torch.load(path, map_location="cpu")
        if not isinstance(raw, dict):
            raise RuntimeError(f"DROID reconstruction is not a mapping: {path}")
        out = {}
        for key, value in raw.items():
            if isinstance(value, str):
                out[key] = value
            elif hasattr(value, "detach"):
                out[key] = value.detach().cpu().numpy()
            else:
                out[key] = np.asarray(value)
    required = {"tstamps", "disps"}
    missing = sorted(required.difference(out))
    if missing:
        raise RuntimeError(f"DROID reconstruction lacks required fields {missing}: {path}")
    return out


def load_shared_geometry(manifest_path: Path, expected_frames: int | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "v22_shared_droid_geometry.v1":
        raise RuntimeError(f"unsupported shared DROID manifest schema: {manifest.get('schema')}")
    if manifest.get("backend") != "droid":
        raise RuntimeError(f"shared geometry is not a DROID artifact: {manifest.get('backend')}")
    if manifest.get("status") != "ok":
        raise RuntimeError(f"shared DROID manifest is not successful: {manifest.get('status')}")
    invocation = manifest.get("droid_invocation")
    if not isinstance(invocation, dict) or int(invocation.get("instance_count") or 0) != 1 or int(invocation.get("terminate_call_count") or 0) != 1:
        raise RuntimeError(f"shared geometry does not prove one DROID instance/termination: {invocation}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"shared DROID manifest lacks artifacts: {manifest_path}")
    dense_path = resolve_artifact(manifest_path, artifacts.get("dense_trajectory"), "dense trajectory")
    reconstruction_path = resolve_artifact(manifest_path, artifacts.get("keyframe_reconstruction"), "keyframe reconstruction")
    dynamic_mask_path: Path | None = None
    if artifacts.get("dynamic_mask") is not None:
        dynamic_mask_path = resolve_artifact(manifest_path, artifacts.get("dynamic_mask"), "dynamic mask")

    dense_blob = np.load(dense_path, allow_pickle=True)
    for key in ("frame_idx", "pose_world_camera_xyzw"):
        if key not in dense_blob.files:
            raise RuntimeError(f"DROID dense trajectory lacks {key}: {dense_path}")
    frame_idx = np.asarray(dense_blob["frame_idx"], dtype=np.int64).reshape(-1)
    traj = np.asarray(dense_blob["pose_world_camera_xyzw"], dtype=np.float32)
    if traj.shape != (len(frame_idx), 7):
        raise RuntimeError(f"DROID dense trajectory shape mismatch: frame_idx={frame_idx.shape} traj={traj.shape}")
    if len(frame_idx) == 0 or not np.isfinite(traj).all():
        raise RuntimeError("DROID dense trajectory is empty or non-finite")
    if not np.array_equal(frame_idx, np.arange(len(frame_idx), dtype=np.int64)):
        raise RuntimeError("DROID dense frame_idx must be contiguous from zero for HaWoR")
    if expected_frames is not None and len(frame_idx) != int(expected_frames):
        raise RuntimeError(f"DROID dense frame count {len(frame_idx)} != expected {expected_frames}")
    if manifest.get("processed_frames") is not None and int(manifest["processed_frames"]) != len(frame_idx):
        raise RuntimeError("shared DROID manifest processed_frames disagrees with dense trajectory")
    if expected_frames is not None and manifest.get("full_source_timeline") is not True:
        raise RuntimeError("shared DROID manifest does not declare full source timeline coverage")

    reconstruction = load_reconstruction(reconstruction_path)
    tstamp = np.asarray(reconstruction["tstamps"], dtype=np.float32).reshape(-1)
    disps = np.asarray(reconstruction["disps"], dtype=np.float32)
    if tstamp.ndim != 1 or disps.ndim != 3 or len(tstamp) != disps.shape[0]:
        raise RuntimeError(f"DROID keyframe fields have incompatible shapes: tstamp={tstamp.shape} disps={disps.shape}")
    tstamp_int = np.rint(tstamp).astype(np.int64)
    if not np.allclose(tstamp, tstamp_int) or np.any(tstamp_int < 0) or np.any(tstamp_int >= len(frame_idx)):
        raise RuntimeError("DROID keyframe timestamps are not valid source-frame indices")
    if len(np.unique(tstamp_int)) != len(tstamp_int):
        raise RuntimeError("DROID keyframe timestamps contain duplicates")
    if not np.isfinite(disps).all() or np.any(disps <= 0):
        raise RuntimeError("DROID keyframe disparities must be finite and positive")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "dense_path": dense_path,
        "reconstruction_path": reconstruction_path,
        "frame_idx": frame_idx.astype(np.int32),
        "traj": traj,
        "tstamp": tstamp_int.astype(np.int32),
        "disps": disps,
        "depth_level": str(reconstruction.get("depth_level", "unknown")),
        "dynamic_mask_path": dynamic_mask_path,
    }


def droid_internal_size(imgfiles: list[str] | tuple[str, ...]) -> tuple[int, int]:
    if not imgfiles:
        raise RuntimeError("cannot infer DROID image size from empty image list")
    image = cv2.imread(str(imgfiles[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read HaWoR image: {imgfiles[0]}")
    h0, w0 = image.shape[:2]
    h1 = int(h0 * math.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * math.sqrt((384 * 512) / (h0 * w0)))
    h1 -= h1 % 8
    w1 -= w1 % 8
    return h1, w1


def import_scale_helpers(hawor_root: Path):
    metric_root = hawor_root / "thirdparty" / "Metric3D"
    if str(metric_root) not in sys.path:
        sys.path.insert(0, str(metric_root))
    pipeline_root = hawor_root / "lib" / "pipeline"
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))
    from est_scale import est_scale_hybrid  # type: ignore
    from metric import Metric3D  # type: ignore

    return Metric3D, est_scale_hybrid


def estimate_metric_scale(
    geometry: dict[str, Any],
    *,
    imgfiles: list[str],
    masks: np.ndarray,
    calib: np.ndarray,
    hawor_root: Path,
    metric_checkpoint: Path | None = None,
    near_threshold: float = 0.4,
    far_threshold: float = 0.7,
) -> tuple[float, dict[str, Any]]:
    if masks.ndim < 3 or masks.shape[0] < len(geometry["frame_idx"]):
        raise RuntimeError(f"HaWoR masks do not cover the DROID timeline: {masks.shape}")
    tstamp = np.asarray(geometry["tstamp"], dtype=np.int32)
    disps = np.asarray(geometry["disps"], dtype=np.float32)
    if np.any(tstamp >= len(masks)):
        raise RuntimeError("DROID keyframe timestamp exceeds HaWoR mask timeline")
    h, w = droid_internal_size(imgfiles)
    metric_checkpoint = metric_checkpoint or (hawor_root / "thirdparty" / "Metric3D" / "weights" / "metric_depth_vit_large_800k.pth")
    metric_checkpoint = metric_checkpoint.expanduser().resolve()
    if not metric_checkpoint.is_file():
        raise FileNotFoundError(f"Metric3D scale checkpoint missing: {metric_checkpoint}")
    Metric3D, est_scale_hybrid = import_scale_helpers(hawor_root)
    metric = Metric3D(str(metric_checkpoint))
    scales: list[float] = []
    per_keyframe: list[dict[str, Any]] = []
    thresholds = [(float(near_threshold), float(far_threshold)), (0.3, 0.8), (0.2, 1.0), (0.0, 2.0)]
    for pos, frame in enumerate(tstamp.tolist()):
        pred_depth = np.asarray(metric(imgfiles[int(frame)], calib), dtype=np.float32)
        pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        slam_depth = 1.0 / np.maximum(disps[pos], 1.0e-8)
        mask = np.asarray(masks[int(frame)]).astype(np.float32)
        scale = float("nan")
        used_threshold: tuple[float, float] | None = None
        error: str | None = None
        for near, far in thresholds:
            try:
                candidate = float(est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=mask, near_thresh=near, far_thresh=far))
            except Exception as exc:  # pragma: no cover - runtime model/optimizer failures
                error = f"{type(exc).__name__}: {exc}"
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                scale = candidate
                used_threshold = (near, far)
                break
        row = {
            "keyframe_position": int(pos),
            "frame_idx": int(frame),
            "scale": scale if math.isfinite(scale) else None,
            "threshold": list(used_threshold) if used_threshold else None,
            "error": error,
        }
        per_keyframe.append(row)
        if math.isfinite(scale) and scale > 0.0:
            scales.append(scale)
    del metric
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - runtime host has torch
        pass
    if not scales:
        raise RuntimeError("Metric3D/DROID disparity scale estimation produced no finite positive scales")
    median_scale = float(np.median(np.asarray(scales, dtype=np.float64)))
    scale_values = np.asarray(scales, dtype=np.float64)
    report = {
        "status": "ok",
        "scale": median_scale,
        "keyframes": int(len(tstamp)),
        "finite_keyframe_scales": int(len(scales)),
        "near_threshold": float(near_threshold),
        "far_threshold": float(far_threshold),
        "metric_checkpoint": str(metric_checkpoint),
        "metric_checkpoint_sha256": sha256_file(metric_checkpoint),
        "internal_depth_size_hw": [int(h), int(w)],
        "scale_statistics": {
            "min": float(np.min(scale_values)),
            "p10": float(np.percentile(scale_values, 10)),
            "median": float(np.median(scale_values)),
            "p90": float(np.percentile(scale_values, 90)),
            "max": float(np.max(scale_values)),
        },
        "per_keyframe": per_keyframe,
        "per_keyframe_preview": per_keyframe[:16],
    }
    return median_scale, report


def adapt_droid_to_hawor(
    manifest_path: Path,
    *,
    output_path: Path,
    imgfiles: list[str],
    masks_path: Path,
    img_focal: float,
    img_center: tuple[float, float],
    hawor_root: Path,
    expected_frames: int | None = None,
    expected_clip_sha256: str | None = None,
    metric_checkpoint: Path | None = None,
    scale: float | None = None,
) -> dict[str, Any]:
    geometry = load_shared_geometry(manifest_path, expected_frames=expected_frames)
    manifest_clip_sha256 = geometry["manifest"].get("clip_sha256")
    if expected_clip_sha256 and manifest_clip_sha256 != expected_clip_sha256:
        raise RuntimeError(
            "shared DROID clip fingerprint does not match the HaWoR input video: "
            f"shared={manifest_clip_sha256} hawor={expected_clip_sha256}"
        )
    masks_path = masks_path.expanduser().resolve()
    if not masks_path.is_file():
        raise FileNotFoundError(f"HaWoR mask artifact missing for DROID adapter: {masks_path}")
    shared_dynamic_mask_path = geometry.get("dynamic_mask_path")
    if shared_dynamic_mask_path is not None:
        if masks_path != shared_dynamic_mask_path:
            raise RuntimeError(
                "HaWoR mask artifact does not match the mask-bound shared DROID manifest: "
                f"shared={shared_dynamic_mask_path} hawor={masks_path}"
            )
        shared_mask_hash = sha256_file(shared_dynamic_mask_path)
    else:
        shared_mask_hash = None
    masks = np.load(masks_path, allow_pickle=True)
    if not isinstance(masks, np.ndarray):
        masks = np.asarray(masks)
    scale_report: dict[str, Any]
    if scale is None:
        scale, scale_report = estimate_metric_scale(
            geometry,
            imgfiles=imgfiles,
            masks=masks,
            calib=np.asarray([img_focal, img_focal, img_center[0], img_center[1]], dtype=np.float32),
            hawor_root=hawor_root,
            metric_checkpoint=metric_checkpoint,
        )
    else:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"adapter scale must be finite and positive: {scale}")
        scale_report = {"status": "provided", "scale": scale}
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        tstamp=np.asarray(geometry["tstamp"], dtype=np.int32),
        disps=np.asarray(geometry["disps"], dtype=np.float32),
        traj=np.asarray(geometry["traj"], dtype=np.float32),
        img_focal=np.asarray(float(img_focal), dtype=np.float64),
        img_center=np.asarray(img_center, dtype=np.float64),
        scale=np.asarray(float(scale), dtype=np.float64),
        shared_droid_manifest=np.asarray([str(geometry["manifest_path"])]),
        shared_droid_manifest_sha256=np.asarray([sha256_file(geometry["manifest_path"])]),
        shared_droid_dense_sha256=np.asarray([sha256_file(geometry["dense_path"])]),
        shared_droid_reconstruction_sha256=np.asarray([sha256_file(geometry["reconstruction_path"])]),
        droid_adapter_schema=np.asarray([SCHEMA]),
        droid_invocation_count=np.asarray([1], dtype=np.int32),
        legacy_hawor_droid_executed=np.asarray([False], dtype=np.bool_),
        droid_mask_policy=np.asarray(["shared D4 DROID and HaWoR adapter consumed the same hash-bound HaWoR dynamic mask" if shared_dynamic_mask_path is not None else "shared D4 DROID was unmasked; HaWoR masks used for metric scale only"]),
    )
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "shared_droid_manifest": str(geometry["manifest_path"]),
        "shared_droid_manifest_sha256": sha256_file(geometry["manifest_path"]),
        "shared_droid_dense_trajectory": str(geometry["dense_path"]),
        "shared_droid_reconstruction": str(geometry["reconstruction_path"]),
        "video_frame_count": int(len(geometry["frame_idx"])),
        "droid_keyframe_count": int(len(geometry["tstamp"])),
        "trajectory_shape": list(np.asarray(geometry["traj"]).shape),
        "disparity_shape": list(np.asarray(geometry["disps"]).shape),
        "scale": float(scale),
        "scale_report": scale_report,
        "pose_contract": "traj is copied from the raw DROID terminate vector consumed by HaWoR load_slam_cam; no D4 T_world_camera inversion is applied",
        "legacy_hawor_droid_executed": False,
        "droid_invocation_count": 1,
        "mask_policy": "shared D4 DROID and HaWoR adapter consumed the same hash-bound HaWoR dynamic mask" if shared_dynamic_mask_path is not None else "D4 shared run was unmasked; HaWoR masks are applied to Metric3D scale estimation only",
        "masks_path": str(masks_path),
        "masks_sha256": sha256_file(masks_path),
        "shared_droid_dynamic_mask": str(shared_dynamic_mask_path) if shared_dynamic_mask_path is not None else None,
        "shared_droid_dynamic_mask_sha256": shared_mask_hash,
    }
    report_path = output_path.with_name("hawor_slam_adapter_report.json")
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--img-focal", type=float, required=True)
    parser.add_argument("--img-center", type=float, nargs=2, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing zero-padded extracted HaWoR JPG frames; required when scale is not supplied.")
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--expected-clip-sha256", default=None)
    parser.add_argument("--metric-checkpoint", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    imgfiles = sorted(str(path) for path in args.image_dir.glob("*.jpg")) if args.image_dir is not None else []
    if args.scale is None and not imgfiles:
        raise RuntimeError("--image-dir with extracted JPG frames is required when --scale is not supplied")
    report = adapt_droid_to_hawor(
        args.shared_manifest,
        output_path=args.output,
        imgfiles=imgfiles,
        masks_path=args.masks,
        img_focal=args.img_focal,
        img_center=(float(args.img_center[0]), float(args.img_center[1])),
        hawor_root=args.hawor_root,
        expected_frames=args.expected_frames,
        expected_clip_sha256=args.expected_clip_sha256,
        metric_checkpoint=args.metric_checkpoint,
        scale=args.scale,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
