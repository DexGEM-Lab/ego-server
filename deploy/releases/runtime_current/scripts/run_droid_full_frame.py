#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
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


def save_reconstruction(droid, out_dir: Path, use_upsampled_depth: bool) -> dict:
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
    }


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
        droid.track(tstamp, image, intrinsics=scaled_intrinsics)
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

    recon_qc = save_reconstruction(droid, out_dir, bool(args.upsample))
    cfg = DROIDConfig(
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
    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "video": asdict(stream_info),
        "processed_frames": int(processed),
        "dense_trajectory_frames": int(traj.shape[0]),
        "full_source_timeline": bool(args.max_frames is None and processed == info.frame_count),
        "pose_convention": "T_world_camera, pose vector [tx, ty, tz, qx, qy, qz, qw]",
        "calibration_source": "dataset has no calibration file; pinhole prior uses focal_scale * max(width, height)",
        "droid": asdict(cfg),
        "keyframes": recon_qc,
        "trajectory_path_length": float(step.sum()),
        "median_step": float(np.median(step)) if step.size else 0.0,
        "p95_step": float(np.percentile(step, 95)) if step.size else 0.0,
        "elapsed_s": float(time.time() - started),
        "outputs": {
            "dense_npz": str(out_dir / "droid_dense_trajectory.npz"),
            "dense_json": str(out_dir / "droid_dense_trajectory.json"),
            "keyframe_reconstruction": str(out_dir / "droid_keyframe_reconstruction.pth"),
            "keyframes": str(out_dir / "droid_keyframes.json"),
        },
    }
    (out_dir / "droid_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--droid-root", type=Path, default=DEFAULT_DROID_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_DROID_ROOT / "droid.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/droid"))
    parser.add_argument("--max-frames", type=int)
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
