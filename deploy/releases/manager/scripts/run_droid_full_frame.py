#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime DROID environment supplies torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
from scipy.spatial.transform import Rotation
from tqdm import tqdm


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)
DEFAULT_DROID_ROOT = Path("third_party/DROID-SLAM")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class DROIDConfig:
    droid_root: str
    weights: str
    focal_scale: float
    fx: float
    fy: float
    cx: float
    cy: float
    buffer: int
    filter_thresh: float
    warmup: int
    keyframe_thresh: float
    frontend_thresh: float
    backend_thresh: float
    internal_width: int
    internal_height: int
    target_area: int


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def camera_prior(info: VideoInfo, focal_scale: float) -> np.ndarray:
    focal = focal_scale * max(info.width, info.height)
    return np.asarray([focal, focal, 0.5 * info.width, 0.5 * info.height], dtype=np.float32)


def droid_resize(
    frame: np.ndarray,
    intrinsics: np.ndarray,
    target_area: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    h0, w0 = frame.shape[:2]
    h1 = int(h0 * np.sqrt(target_area / (h0 * w0)))
    w1 = int(w0 * np.sqrt(target_area / (h0 * w0)))
    frame = cv2.resize(frame, (w1, h1), interpolation=cv2.INTER_AREA)
    frame = frame[: h1 - h1 % 8, : w1 - w1 % 8]

    scaled = torch.as_tensor(intrinsics.copy(), dtype=torch.float32)
    scaled[0::2] *= frame.shape[1] / w0
    scaled[1::2] *= frame.shape[0] / h0

    image = torch.as_tensor(frame).permute(2, 0, 1)[None]
    return image, scaled, (frame.shape[1], frame.shape[0])


def video_stream(
    clip: Path,
    intrinsics: np.ndarray,
    max_frames: int | None,
    target_area: int,
) -> tuple[VideoInfo, tuple[int, int], object]:
    cap, info = open_video(clip)
    internal_size: tuple[int, int] | None = None

    def gen():
        nonlocal internal_size
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            image, scaled, size = droid_resize(frame, intrinsics, target_area)
            internal_size = size
            yield frame_idx, image, scaled
            frame_idx += 1
        cap.release()

    return info, internal_size or (0, 0), gen()


def import_droid(droid_root: Path):
    root = droid_root.resolve()
    sys.path.insert(0, str(root / "droid_slam"))
    from droid import Droid  # type: ignore

    return Droid


def pose_vec_xyzw_to_matrix(vec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(vec[3:7]).as_matrix()
    T[:3, 3] = vec[:3]
    return T


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dynamic_masks(path: Path, video: VideoInfo) -> tuple[Path, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DROID dynamic mask artifact missing: {resolved}")
    masks = np.load(resolved, mmap_mode="r", allow_pickle=False)
    expected_shape = (int(video.frame_count), int(video.height), int(video.width))
    if not isinstance(masks, np.ndarray) or tuple(masks.shape) != expected_shape:
        actual_shape = tuple(masks.shape) if isinstance(masks, np.ndarray) else None
        raise RuntimeError(
            f"DROID dynamic masks must match the full video timeline {expected_shape}; got {actual_shape}"
        )
    if masks.dtype.kind not in {"b", "u", "i", "f"}:
        raise RuntimeError(f"DROID dynamic masks have unsupported dtype: {masks.dtype}")
    return resolved, masks


def apply_dynamic_mask(image: torch.Tensor, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply HaWoR's masked-DROID image and stride-8 confidence-mask contract."""
    mask_array = np.array(mask, copy=True)
    if mask_array.ndim != 2:
        raise RuntimeError(f"DROID dynamic mask frame must be 2D; got {mask_array.shape}")
    if not np.isfinite(mask_array).all():
        raise RuntimeError("DROID dynamic mask frame contains non-finite values")
    mask_min = float(mask_array.min()) if mask_array.size else 0.0
    mask_max = float(mask_array.max()) if mask_array.size else 0.0
    if mask_min < 0.0 or mask_max > 1.0:
        raise RuntimeError(f"DROID dynamic mask values must lie in [0, 1]; got [{mask_min}, {mask_max}]")
    if torch is None or F is None:
        raise RuntimeError("applying DROID dynamic masks requires torch")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32)[None, None]
    image_mask = F.interpolate(
        mask_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    confidence_mask = F.interpolate(
        mask_tensor,
        size=(height // 8, width // 8),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    masked_image = image * (image_mask < 0.5)
    coverage = float(np.count_nonzero(mask_array >= 0.5) / max(mask_array.size, 1))
    return masked_image, confidence_mask, coverage


def save_reconstruction(
    droid,
    out_dir: Path,
    use_upsampled_depth: bool,
    dense_traj: np.ndarray | None = None,
) -> dict:
    video = droid.video2 if hasattr(droid, "video2") else droid.video
    t = int(video.counter.value)
    disps_low = video.disps[:t].detach().cpu()
    disps_up = video.disps_up[:t].detach().cpu()
    depth_level = "upsampled" if use_upsampled_depth else "network_stride_8"
    saved_disps = disps_up if use_upsampled_depth else disps_low
    blob = {
        "tstamps": video.tstamp[:t].detach().cpu(),
        "images": video.images[:t].detach().cpu(),
        "disps": saved_disps,
        "disps_low": disps_low,
        "depth_level": depth_level,
        "poses": video.poses[:t].detach().cpu(),
        "intrinsics": video.intrinsics[:t].detach().cpu(),
    }
    if dense_traj is not None:
        dense_traj = np.asarray(dense_traj, dtype=np.float32)
        blob["dense_tstamps"] = torch.arange(dense_traj.shape[0], dtype=torch.int32)
        blob["dense_pose_world_camera_xyzw"] = torch.from_numpy(dense_traj.copy())
    torch.save(blob, out_dir / "droid_keyframe_reconstruction.pth")

    pose_internal = blob["poses"].numpy()
    keyframes = []
    for i, tstamp in enumerate(blob["tstamps"].numpy().astype(int).tolist()):
        cam_from_world = pose_vec_xyzw_to_matrix(pose_internal[i])
        world_from_cam = np.linalg.inv(cam_from_world)
        keyframes.append(
            {
                "keyframe_index": i,
                "source_frame_idx": int(tstamp),
                "T_world_camera": world_from_cam.tolist(),
                "pose_internal_cam_from_world_xyzw": pose_internal[i].astype(float).tolist(),
            }
        )

    (out_dir / "droid_keyframes.json").write_text(
        json.dumps({"keyframes": keyframes}, indent=2), encoding="utf-8"
    )
    return {
        "keyframe_count": t,
        "depth_level": depth_level,
        "keyframe_path": str(out_dir / "droid_keyframes.json"),
        "reconstruction_path": str(out_dir / "droid_keyframe_reconstruction.pth"),
        "keyframe_tstamp_key": "tstamps",
        "keyframe_disparity_key": "disps",
        "keyframe_intrinsics_key": "intrinsics",
        "dense_trajectory_key": "dense_pose_world_camera_xyzw" if dense_traj is not None else None,
    }


def write_shared_geometry_manifest(
    out_dir: Path,
    *,
    clip: Path,
    clip_sha256: str | None,
    video: VideoInfo,
    processed: int,
    droid_config: DROIDConfig,
    droid_qc_path: Path,
    trajectory_path: Path,
    reconstruction_path: Path,
    keyframes_path: Path,
    dynamic_mask: dict[str, Any],
) -> dict[str, Any]:
    required = [trajectory_path, reconstruction_path, keyframes_path, droid_qc_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot write shared DROID manifest; missing artifacts: {missing}")
    artifacts = {
        "dense_trajectory": {
            "path": str(trajectory_path),
            "sha256": sha256_file(trajectory_path),
            "frame_idx_key": "frame_idx",
            "trajectory_key": "pose_world_camera_xyzw",
            "matrix_key": "T_world_camera",
            "fps_key": "fps",
            "trajectory_for_hawor": "pose_world_camera_xyzw",
        },
        "keyframe_reconstruction": {
            "path": str(reconstruction_path),
            "sha256": sha256_file(reconstruction_path),
            "timestamp_key": "tstamps",
            "disparity_key": "disps",
            "low_resolution_disparity_key": "disps_low",
            "intrinsics_key": "intrinsics",
            "depth_level_key": "depth_level",
        },
        "keyframes": {
            "path": str(keyframes_path),
            "sha256": sha256_file(keyframes_path),
        },
        "droid_qc": {
            "path": str(droid_qc_path),
            "sha256": sha256_file(droid_qc_path),
        },
    }
    if dynamic_mask.get("status") == "applied":
        artifacts["dynamic_mask"] = {
            "path": str(dynamic_mask["path"]),
            "sha256": str(dynamic_mask["sha256"]),
            "shape": list(dynamic_mask["shape"]),
            "dtype": str(dynamic_mask["dtype"]),
            "source": str(dynamic_mask.get("source", "hawor_model_masks")),
        }
    manifest: dict[str, Any] = {
        "schema": "v22_shared_droid_geometry.v1",
        "status": "ok",
        "backend": "droid",
        "clip": str(clip),
        "clip_sha256": clip_sha256,
        "video": asdict(video),
        "processed_frames": int(processed),
        "full_source_timeline": bool(processed == video.frame_count),
        "droid_invocation": {
            "class": "droid.Droid",
            "instance_count": 1,
            "track_call_count": int(processed),
            "terminate_call_count": 1,
            "droid_root": droid_config.droid_root,
            "weights": droid_config.weights,
            "weights_sha256": sha256_file(Path(droid_config.weights)) if Path(droid_config.weights).is_file() else None,
        },
        "droid_config": asdict(droid_config),
        "pose_contract": {
            "raw_terminate_vector": "[tx, ty, tz, qx, qy, qz, qw]",
            "hawor_consumption_key": "dense_trajectory.pose_world_camera_xyzw",
            "d4_matrix_key": "dense_trajectory.T_world_camera",
            "conversion_rule": "HaWoR adapter consumes raw terminate vector; it must not reconstruct traj from the D4 matrix without an explicit inverse/convention test.",
        },
        "scale_contract": {
            "droid_scale": "arbitrary_video_gauge",
            "metric_scale": "computed by HaWoR adapter from DROID disparity and Metric3D; not part of DROID inference",
        },
        "dynamic_mask": dynamic_mask,
        "artifacts": artifacts,
        "consumers": ["D4_camera_trajectory", "HaWoR_SLAM_adapter", "D7_hybrid_fusion_via_HaWoR_world_npz"],
    }
    path = out_dir / "droid_shared_geometry.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def make_droid_args(args: argparse.Namespace, image_size: list[int]) -> argparse.Namespace:
    return argparse.Namespace(
        weights=str(args.weights),
        buffer=args.buffer,
        image_size=image_size,
        disable_vis=True,
        beta=args.beta,
        filter_thresh=args.filter_thresh,
        warmup=args.warmup,
        keyframe_thresh=args.keyframe_thresh,
        frontend_thresh=args.frontend_thresh,
        frontend_window=args.frontend_window,
        frontend_radius=args.frontend_radius,
        frontend_nms=args.frontend_nms,
        backend_thresh=args.backend_thresh,
        backend_radius=args.backend_radius,
        backend_nms=args.backend_nms,
        upsample=args.upsample,
        asynchronous=False,
        frontend_device="cuda",
        backend_device="cuda",
        stereo=False,
    )


def run(args: argparse.Namespace) -> dict:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, info = open_video(args.clip)
    cap.release()
    intrinsics = camera_prior(info, args.focal_scale)
    dynamic_mask_path: Path | None = None
    dynamic_masks: np.ndarray | None = None
    dynamic_mask_sha256: str | None = None
    mask_coverage: list[float] = []
    if args.dynamic_mask_npy is not None:
        dynamic_mask_path, dynamic_masks = load_dynamic_masks(args.dynamic_mask_npy, info)
        dynamic_mask_sha256 = sha256_file(dynamic_mask_path)
        if args.dynamic_mask_sha256 is not None and dynamic_mask_sha256 != args.dynamic_mask_sha256:
            raise RuntimeError(
                "DROID dynamic mask hash mismatch: "
                f"expected={args.dynamic_mask_sha256} actual={dynamic_mask_sha256} path={dynamic_mask_path}"
            )

    if torch is None:
        raise RuntimeError("running DROID requires torch in the selected runner environment")
    torch.multiprocessing.set_start_method("spawn", force=True)
    Droid = import_droid(args.droid_root)

    droid = None
    processed = 0
    first_internal_size = (0, 0)
    started = time.time()

    stream_info, _, stream = video_stream(args.clip, intrinsics, args.max_frames, args.droid_area)
    for tstamp, image, scaled_intrinsics in tqdm(stream, desc="droid_full_frame"):
        if droid is None:
            first_internal_size = (int(image.shape[3]), int(image.shape[2]))
            droid_args = make_droid_args(args, [int(image.shape[2]), int(image.shape[3])])
            droid = Droid(droid_args)
        confidence_mask = None
        if dynamic_masks is not None:
            image, confidence_mask, coverage = apply_dynamic_mask(image, dynamic_masks[int(tstamp)])
            mask_coverage.append(coverage)
        droid.track(tstamp, image, intrinsics=scaled_intrinsics, mask=confidence_mask)
        processed += 1

    if droid is None:
        raise RuntimeError("DROID received no frames")

    _, _, fill_stream = video_stream(args.clip, intrinsics, args.max_frames, args.droid_area)
    traj = droid.terminate(fill_stream)
    if traj.shape[0] != processed:
        raise RuntimeError(f"DROID dense trajectory has {traj.shape[0]} poses for {processed} frames")
    if not np.isfinite(traj).all():
        raise RuntimeError("DROID dense trajectory contains non-finite values")

    matrices = np.stack([pose_vec_xyzw_to_matrix(row) for row in traj], axis=0)
    translations = matrices[:, :3, 3]
    step = np.linalg.norm(np.diff(translations, axis=0), axis=1) if len(translations) > 1 else np.zeros(0)

    frame_indices = np.arange(processed, dtype=np.int32)
    np.savez_compressed(
        out_dir / "droid_dense_trajectory.npz",
        frame_idx=frame_indices,
        pose_world_camera_xyzw=traj.astype(np.float32),
        T_world_camera=matrices.astype(np.float32),
        intrinsics_source=intrinsics.astype(np.float32),
        fps=np.asarray([info.fps], dtype=np.float32),
    )
    dense_json = [
        {
            "frame_idx": int(i),
            "pose_world_camera_xyzw": traj[i].astype(float).tolist(),
            "T_world_camera": matrices[i].astype(float).tolist(),
        }
        for i in range(processed)
    ]
    (out_dir / "droid_dense_trajectory.json").write_text(
        json.dumps({"frames": dense_json}, indent=2), encoding="utf-8"
    )

    recon_qc = save_reconstruction(droid, out_dir, bool(args.upsample), dense_traj=traj)
    cfg = DROIDConfig(
        droid_root=str(args.droid_root),
        weights=str(args.weights),
        focal_scale=float(args.focal_scale),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        buffer=int(args.buffer),
        filter_thresh=float(args.filter_thresh),
        warmup=int(args.warmup),
        keyframe_thresh=float(args.keyframe_thresh),
        frontend_thresh=float(args.frontend_thresh),
        backend_thresh=float(args.backend_thresh),
        internal_width=int(first_internal_size[0]),
        internal_height=int(first_internal_size[1]),
        target_area=int(args.droid_area),
    )
    clip_sha256 = sha256_file(args.clip) if args.clip.is_file() else None
    dynamic_mask_info: dict[str, Any]
    if dynamic_mask_path is not None and dynamic_masks is not None and dynamic_mask_sha256 is not None:
        coverage_array = np.asarray(mask_coverage, dtype=np.float64)
        dynamic_mask_info = {
            "status": "applied",
            "source": "hawor_model_masks",
            "path": str(dynamic_mask_path),
            "sha256": dynamic_mask_sha256,
            "shape": [int(value) for value in dynamic_masks.shape],
            "dtype": str(dynamic_masks.dtype),
            "processed_mask_frames": int(len(mask_coverage)),
            "coverage_fraction": {
                "mean": float(np.mean(coverage_array)),
                "median": float(np.median(coverage_array)),
                "p95": float(np.percentile(coverage_array, 95)),
                "max": float(np.max(coverage_array)),
            },
            "image_policy": "bilinear-antialiased mask resized to DROID input; pixels with mask >= 0.5 are zeroed",
            "confidence_policy": "bilinear-antialiased mask resized to stride 8 and passed to Droid.track(mask=...)",
            "terminate_stream_policy": "unmasked source frames, matching HaWoR legacy masked_droid_slam trajectory filling",
        }
    else:
        dynamic_mask_info = {
            "status": "not_provided",
            "image_policy": "full-frame RGB",
            "confidence_policy": "Droid.track(mask=None)",
            "terminate_stream_policy": "unmasked source frames",
        }
    droid_qc_path = out_dir / "droid_qc.json"
    dense_npz_path = out_dir / "droid_dense_trajectory.npz"
    keyframe_reconstruction_path = out_dir / "droid_keyframe_reconstruction.pth"
    keyframes_path = out_dir / "droid_keyframes.json"
    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "clip_sha256": clip_sha256,
        "video": asdict(stream_info),
        "processed_frames": int(processed),
        "dense_trajectory_frames": int(traj.shape[0]),
        "full_source_timeline": bool(args.max_frames is None and processed == info.frame_count),
        "pose_convention": "T_world_camera, pose vector [tx, ty, tz, qx, qy, qz, qw]",
        "calibration_source": "dataset has no calibration file; pinhole prior uses focal_scale * max(width, height)",
        "droid": asdict(cfg),
        "droid_invocation": {"class": "droid.Droid", "instance_count": 1, "track_call_count": int(processed), "terminate_call_count": 1},
        "dynamic_mask": dynamic_mask_info,
        "keyframes": recon_qc,
        "trajectory_path_length": float(step.sum()),
        "median_step": float(np.median(step)) if step.size else 0.0,
        "p95_step": float(np.percentile(step, 95)) if step.size else 0.0,
        "elapsed_s": float(time.time() - started),
        "outputs": {
            "dense_npz": str(dense_npz_path),
            "dense_json": str(out_dir / "droid_dense_trajectory.json"),
            "keyframe_reconstruction": str(keyframe_reconstruction_path),
            "keyframes": str(keyframes_path),
            "shared_geometry_manifest": str(out_dir / "droid_shared_geometry.json"),
        },
        "shared_geometry": {
            "manifest_path": str(out_dir / "droid_shared_geometry.json"),
            "backend": "droid",
            "consumer_contract": "HaWoR adapter consumes artifacts listed by this manifest; no second DROID call is allowed.",
        },
    }
    droid_qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    shared_manifest = write_shared_geometry_manifest(
        out_dir,
        clip=args.clip,
        clip_sha256=clip_sha256,
        video=stream_info,
        processed=processed,
        droid_config=cfg,
        droid_qc_path=droid_qc_path,
        trajectory_path=dense_npz_path,
        reconstruction_path=keyframe_reconstruction_path,
        keyframes_path=keyframes_path,
        dynamic_mask=dynamic_mask_info,
    )
    if shared_manifest["manifest_path"] != qc["shared_geometry"]["manifest_path"]:
        raise RuntimeError("shared DROID manifest path diverged from the D4 QC contract")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--droid-root", type=Path, default=DEFAULT_DROID_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_DROID_ROOT / "droid.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/droid"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dynamic-mask-npy", type=Path, default=None, help="Optional full-resolution [T,H,W] HaWoR model mask artifact used by the one canonical DROID run.")
    parser.add_argument("--dynamic-mask-sha256", default=None, help="Expected hash from the HaWoR preparation report; mismatch fails closed before DROID starts.")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--droid-area", type=int, default=384 * 512)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter-thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe-thresh", type=float, default=4.0)
    parser.add_argument("--frontend-thresh", type=float, default=16.0)
    parser.add_argument("--frontend-window", type=int, default=25)
    parser.add_argument("--frontend-radius", type=int, default=2)
    parser.add_argument("--frontend-nms", type=int, default=1)
    parser.add_argument("--backend-thresh", type=float, default=22.0)
    parser.add_argument("--backend-radius", type=int, default=2)
    parser.add_argument("--backend-nms", type=int, default=3)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
