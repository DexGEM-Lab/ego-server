#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)
DEFAULT_WILOR_ROOT = Path("third_party/WiLoR")
DEFAULT_MANO_RIGHT = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_RIGHT.pkl")

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LEFT_COLOR = (0, 205, 0)
RIGHT_COLOR = (0, 135, 255)
OBJECT_COLOR = (25, 35, 230)


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class FrameSample:
    frame_idx: int
    time_s: float
    name: str


@dataclass(frozen=True)
class RenderInfo:
    width: int
    height: int


def patch_legacy_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    raw_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return raw_load(*args, **kwargs)

    torch.load = torch_load_compat  # type: ignore[assignment]


def ensure_wilor_assets(wilor_root: Path, mano_right: Path) -> None:
    required = [
        wilor_root / "wilor" / "models" / "wilor.py",
        wilor_root / "pretrained_models" / "wilor_final.ckpt",
        wilor_root / "pretrained_models" / "detector.pt",
        wilor_root / "pretrained_models" / "model_config.yaml",
        wilor_root / "mano_data" / "mano_mean_params.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing WiLoR assets: " + ", ".join(missing))
    target = wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not target.exists():
        if not mano_right.exists():
            raise FileNotFoundError(f"missing MANO_RIGHT source: {mano_right}")
        shutil.copy2(mano_right, target)


def load_actions(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    tasks = data.get("tasks") or []
    if not tasks:
        raise ValueError(f"no task records in {json_path}")
    return tasks[0].get("actions") or []


def caption_for_frame(actions: list[dict], frame_idx: int) -> str:
    for action in actions:
        start = int(action.get("start_frame", -1))
        end = int(action.get("end_frame", -1))
        if start <= frame_idx < end:
            return str(action.get("description") or action.get("action") or "")
    return ""


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def frame_indices(info: ClipInfo, sample_fps: float) -> list[int]:
    stride = max(1, round(info.fps / sample_fps))
    return list(range(0, info.frame_count, stride))


def extract_sampled_frames(clip: Path, frame_dir: Path, sample_fps: float, render_width: int) -> tuple[ClipInfo, list[FrameSample]]:
    cap, info = open_video(clip)
    wanted = set(frame_indices(info, sample_fps))
    render_height = int(round(render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in wanted:
            continue
        resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
        name = f"frame_{idx:06d}.jpg"
        if not cv2.imwrite(str(frame_dir / name), resized):
            raise RuntimeError(f"failed to write sampled frame: {name}")
        samples.append(FrameSample(frame_idx=idx, time_s=idx / info.fps, name=name))
    cap.release()
    if not samples:
        raise RuntimeError("no sampled frames extracted")
    return info, samples


def select_keyframes(samples: list[FrameSample], output_fps: float, camera_keyframe_fps: float) -> list[FrameSample]:
    if camera_keyframe_fps <= 0:
        raise ValueError("camera keyframe fps must be positive")
    step = max(1, round(output_fps / camera_keyframe_fps))
    return [sample for i, sample in enumerate(samples) if i % step == 0]


def copy_keyframes(dense_dir: Path, keyframe_dir: Path, keyframes: list[FrameSample]) -> None:
    if keyframe_dir.exists():
        shutil.rmtree(keyframe_dir)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    for sample in keyframes:
        src = dense_dir / sample.name
        dst = keyframe_dir / sample.name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)


def run_colmap(frame_dir: Path, output_dir: Path, reuse: bool) -> dict:
    import pycolmap

    sfm_dir = output_dir / "sfm_colmap_keyframes"
    sparse = sfm_dir / "sparse"
    best = sparse / "best"
    if reuse and best.exists():
        return read_colmap(best)
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sparse.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    pycolmap.extract_features(database, frame_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    maps = pycolmap.incremental_mapping(database, frame_dir, sparse)
    if not maps:
        raise RuntimeError("pycolmap returned no reconstruction")
    recon = max(maps.values(), key=lambda r: r.num_reg_images())
    best.mkdir(parents=True, exist_ok=True)
    recon.write(best)
    return colmap_to_json(recon, best)


def read_colmap(model_dir: Path) -> dict:
    return colmap_to_json(pycolmap.Reconstruction(model_dir), model_dir)


def frame_number(name: str) -> int:
    stem = Path(name).stem
    return int(stem.split("_")[-1])


def colmap_to_json(recon, model_dir: Path) -> dict:
    frames = []
    for image in recon.images.values():
        if not image.has_pose:
            continue
        cam_from_world = np.eye(4)
        cam_from_world[:3, :4] = np.asarray(image.cam_from_world().matrix(), dtype=float)
        world_from_cam = np.linalg.inv(cam_from_world)
        frames.append(
            {
                "name": image.name,
                "frame_idx": frame_number(image.name),
                "T_world_camera": world_from_cam.tolist(),
            }
        )
    if not frames:
        raise RuntimeError("pycolmap reconstruction has no registered camera poses")
    frames.sort(key=lambda x: x["frame_idx"])
    return {
        "backend": "pycolmap_sfm_keyframes",
        "status": "ok",
        "registered_images": int(recon.num_reg_images()),
        "num_images": int(len(recon.images)),
        "num_points3d": int(recon.num_points3D()),
        "frames": frames,
        "model_dir": str(model_dir),
    }


def smooth_positions(poses: list[dict], window: int) -> None:
    if window <= 1:
        return
    valid = np.asarray([pose["camera_status"] != "unavailable" for pose in poses], dtype=bool)
    if not valid.any():
        return
    positions = np.zeros((len(poses), 3), dtype=float)
    for i, pose in enumerate(poses):
        if valid[i]:
            positions[i] = np.asarray(pose["T_world_camera"], dtype=float)[:3, 3]
    smoothed = positions.copy()
    half = window // 2
    for i in range(len(poses)):
        lo = max(0, i - half)
        hi = min(len(poses), i + half + 1)
        m = valid[lo:hi]
        if m.any() and valid[i]:
            smoothed[i] = positions[lo:hi][m].mean(axis=0)
    for pose, xyz in zip(poses, smoothed):
        if pose["camera_status"] != "unavailable":
            T = np.asarray(pose["T_world_camera"], dtype=float)
            T[:3, 3] = xyz
            pose["T_world_camera"] = T.tolist()


def interpolate_camera_poses(colmap: dict, samples: list[FrameSample], smooth_window: int, edge_prediction_frames: int) -> tuple[list[dict], dict]:
    keyframes = sorted(colmap["frames"], key=lambda item: item["frame_idx"])
    key_idxs = np.asarray([item["frame_idx"] for item in keyframes], dtype=float)
    key_T = [np.asarray(item["T_world_camera"], dtype=float) for item in keyframes]
    key_pos = np.asarray([T[:3, 3] for T in key_T], dtype=float)
    rotations = Rotation.from_matrix(np.asarray([T[:3, :3] for T in key_T], dtype=float))
    slerp = Slerp(key_idxs, rotations)
    exact = {int(item["frame_idx"]): item for item in keyframes}
    poses: list[dict] = []
    for sample in samples:
        idx = float(sample.frame_idx)
        if sample.frame_idx in exact:
            T = np.asarray(exact[sample.frame_idx]["T_world_camera"], dtype=float)
            poses.append(
                {
                    "backend": colmap["backend"],
                    "camera_status": "registered_keyframe",
                    "name": sample.name,
                    "frame_idx": sample.frame_idx,
                    "T_world_camera": T.tolist(),
                    "source_keyframes": [sample.frame_idx],
                    "scale_status": "relative",
                }
            )
            continue
        if idx < key_idxs[0] or idx > key_idxs[-1]:
            if idx < key_idxs[0]:
                nearest_i = 0
                edge_gap = int(key_idxs[0] - idx)
            else:
                nearest_i = len(key_idxs) - 1
                edge_gap = int(idx - key_idxs[-1])
            if edge_gap <= edge_prediction_frames:
                poses.append(
                    {
                        "backend": colmap["backend"],
                        "camera_status": "predicted_edge",
                        "name": sample.name,
                        "frame_idx": sample.frame_idx,
                        "T_world_camera": key_T[nearest_i].tolist(),
                        "source_keyframes": [int(key_idxs[nearest_i])],
                        "reason": "outside registered keyframe span, nearest-pose prediction",
                        "scale_status": "relative",
                    }
                )
                continue
            poses.append(
                {
                    "backend": colmap["backend"],
                    "camera_status": "unavailable",
                    "name": sample.name,
                    "frame_idx": sample.frame_idx,
                    "reason": "outside registered keyframe span",
                    "scale_status": "relative",
                }
            )
            continue
        right = int(np.searchsorted(key_idxs, idx, side="right"))
        left = right - 1
        denom = key_idxs[right] - key_idxs[left]
        alpha = float((idx - key_idxs[left]) / denom) if denom else 0.0
        T = np.eye(4)
        T[:3, :3] = slerp([idx]).as_matrix()[0]
        T[:3, 3] = (1.0 - alpha) * key_pos[left] + alpha * key_pos[right]
        poses.append(
            {
                "backend": colmap["backend"],
                "camera_status": "interpolated",
                "name": sample.name,
                "frame_idx": sample.frame_idx,
                "T_world_camera": T.tolist(),
                "source_keyframes": [int(key_idxs[left]), int(key_idxs[right])],
                "scale_status": "relative",
            }
        )
    smooth_positions(poses, smooth_window)
    statuses = [pose["camera_status"] for pose in poses]
    qc = {
        "dense_frames": len(samples),
        "registered_keyframes": statuses.count("registered_keyframe"),
        "interpolated_frames": statuses.count("interpolated"),
        "predicted_edge_frames": statuses.count("predicted_edge"),
        "unavailable_frames": statuses.count("unavailable"),
        "dense_pose_rate": (len(samples) - statuses.count("unavailable")) / max(1, len(samples)),
        "position_smoothing_window": smooth_window,
        "edge_prediction_source_frame_limit": edge_prediction_frames,
    }
    return poses, qc


def load_wilor_backend(wilor_root: Path):
    patch_legacy_imports()
    sys.path.insert(0, str(wilor_root.resolve()))
    from ultralytics import YOLO
    from wilor.models import load_wilor

    cwd = Path.cwd()
    os.chdir(wilor_root)
    try:
        model, cfg = load_wilor("./pretrained_models/wilor_final.ckpt", "./pretrained_models/model_config.yaml")
        detector = YOLO("./pretrained_models/detector.pt")
    finally:
        os.chdir(cwd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = detector.to(device)
    return model, cfg, detector, device


def project_full_image(points: np.ndarray, cam_t: np.ndarray, focal: float, img_size: np.ndarray) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(img_size[0]) / 2.0
    K[1, 2] = float(img_size[1]) / 2.0
    pts = points + cam_t
    pts = pts / pts[..., [-1]]
    return (K @ pts.T).T[:, :2]


def mano_params_for_sample(out: dict, n: int) -> dict:
    params = {}
    for key, value in out["pred_mano_params"].items():
        params[key] = value[n].detach().cpu().numpy().astype(float).tolist()
    return params


def run_wilor_on_frame(model, cfg, detector, device, frame: np.ndarray, rescale_factor: float, batch_size: int) -> list[dict]:
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    detections = detector(frame, conf=0.3, verbose=False)[0]
    boxes: list[list[float]] = []
    scores: list[float] = []
    is_right: list[float] = []
    for det in detections:
        arr = det.boxes.data.cpu().detach().squeeze().numpy()
        if arr.ndim == 0 or arr.size < 6:
            continue
        boxes.append(arr[:4].astype(float).tolist())
        scores.append(float(arr[4]))
        is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
    if not boxes:
        return []
    boxes_np = np.asarray(boxes, dtype=np.float32)
    right_np = np.asarray(is_right, dtype=np.float32)
    dataset = ViTDetDataset(cfg, frame, boxes_np, right_np, rescale_factor=rescale_factor, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[dict] = []
    det_offset = 0
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        focal = float(scaled_focal_length.detach().cpu().numpy())
        for n in range(batch["img"].shape[0]):
            det_idx = det_offset + n
            hand_side = "right" if float(batch["right"][n].detach().cpu().numpy()) >= 0.5 else "left"
            verts = out["pred_vertices"][n].detach().cpu().numpy().astype(float)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
            side_sign = 1.0 if hand_side == "right" else -1.0
            verts[:, 0] = side_sign * verts[:, 0]
            joints[:, 0] = side_sign * joints[:, 0]
            joints2d = project_full_image(joints, cam_t[n], focal, img_size[n].detach().cpu().numpy())
            predictions.append(
                {
                    "backend": "WiLoR",
                    "side": hand_side,
                    "detector_score": scores[det_idx],
                    "bbox_xyxy": boxes_np[det_idx].astype(float).tolist(),
                    "cam_t": cam_t[n].astype(float).tolist(),
                    "focal_length": focal,
                    "joints3d_camera": joints.tolist(),
                    "joints2d_raw": joints2d.astype(float).tolist(),
                    "joints2d": joints2d.astype(float).tolist(),
                    "mano_params": mano_params_for_sample(out, n),
                    "vertices_camera": verts.tolist(),
                    "vertices_camera_sample": verts[::10].tolist(),
                    "filter_status": "measured_raw",
                }
            )
        det_offset += batch["img"].shape[0]
    return predictions


def choose_hand_by_side(hands: list[dict]) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for hand in hands:
        side = hand["side"]
        current = chosen.get(side)
        if current is None or hand.get("detector_score", 0.0) > current.get("detector_score", 0.0):
            chosen[side] = hand
    return chosen


def clone_predicted_hand(template: dict, status: str) -> dict:
    keep = {
        "backend",
        "side",
        "bbox_xyxy",
        "cam_t",
        "focal_length",
        "joints3d_camera",
        "joints2d",
        "mano_params",
        "vertices_camera_sample",
    }
    hand = {key: template[key] for key in keep if key in template}
    hand["detector_score"] = 0.0
    hand["filter_status"] = status
    hand["source_filter"] = "temporal_interpolation"
    return hand


def lerp_hand(a: dict, b: dict, alpha: float, status: str) -> dict:
    hand = clone_predicted_hand(a, status)
    for key in ["bbox_xyxy", "cam_t", "joints3d_camera", "joints2d", "vertices_camera_sample"]:
        if key in a and key in b:
            av = np.asarray(a[key], dtype=float)
            bv = np.asarray(b[key], dtype=float)
            hand[key] = ((1.0 - alpha) * av + alpha * bv).astype(float).tolist()
    return hand


def ema_hand(raw: dict, prev: dict | None, alpha: float) -> dict:
    if prev is None:
        hand = dict(raw)
        hand["joints2d"] = raw["joints2d_raw"]
        hand["filter_status"] = "measured_smoothed"
        return hand
    hand = dict(raw)
    for key in ["bbox_xyxy", "cam_t", "joints3d_camera", "joints2d", "vertices_camera_sample"]:
        raw_key = "joints2d_raw" if key == "joints2d" else key
        if raw_key in raw and key in prev:
            rv = np.asarray(raw[raw_key], dtype=float)
            pv = np.asarray(prev[key], dtype=float)
            hand[key] = (alpha * rv + (1.0 - alpha) * pv).astype(float).tolist()
    hand["filter_status"] = "measured_smoothed"
    return hand


def filter_hands(annotations: list[dict], max_gap_frames: int, alpha: float) -> dict:
    raw_by_frame = [choose_hand_by_side(ann["raw_hands"]) for ann in annotations]
    output_by_side: dict[str, list[dict | None]] = {"left": [None] * len(annotations), "right": [None] * len(annotations)}
    stats = {}
    for side in ["left", "right"]:
        measured = [i for i, hands in enumerate(raw_by_frame) if side in hands]
        prev: dict | None = None
        for i in measured:
            filtered = ema_hand(raw_by_frame[i][side], prev, alpha)
            output_by_side[side][i] = filtered
            prev = filtered
        for a, b in zip(measured[:-1], measured[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap_frames:
                left = output_by_side[side][a]
                right = output_by_side[side][b]
                if left is None or right is None:
                    continue
                for i in range(a + 1, b):
                    output_by_side[side][i] = lerp_hand(left, right, (i - a) / (b - a), "interpolated_occlusion")
        for i in range(len(annotations)):
            if output_by_side[side][i] is None:
                prev_i = max([m for m in measured if m < i], default=None)
                if prev_i is not None and i - prev_i <= max_gap_frames and output_by_side[side][prev_i] is not None:
                    output_by_side[side][i] = clone_predicted_hand(output_by_side[side][prev_i], "predicted_lost")
        statuses = [hand["filter_status"] for hand in output_by_side[side] if hand is not None]
        stats[side] = {
            "measured_raw_frames": len(measured),
            "output_frames": len(statuses),
            "measured_smoothed": statuses.count("measured_smoothed"),
            "interpolated_occlusion": statuses.count("interpolated_occlusion"),
            "predicted_lost": statuses.count("predicted_lost"),
        }
    for i, ann in enumerate(annotations):
        hands = [hand for side in ["left", "right"] if (hand := output_by_side[side][i]) is not None]
        ann["hands"] = hands
    return stats


def red_tomato_candidates(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 70, 45], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array([168, 70, 45], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 120.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / max(1.0, float(w * h))
        perim = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area / max(1e-6, perim * perim)
        if fill < 0.26 or circularity < 0.16:
            continue
        candidates.append(
            {
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                "center_xy": [float(x + w / 2.0), float(y + h / 2.0)],
                "area": area,
                "fill": float(fill),
                "circularity": float(circularity),
            }
        )
    return candidates


def hand_overlap_penalty(candidate: dict, hands: list[dict]) -> float:
    cx1, cy1, cx2, cy2 = candidate["bbox_xyxy"]
    c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
    penalty = 0.0
    for hand in hands:
        hx1, hy1, hx2, hy2 = hand["bbox_xyxy"]
        ix1, iy1 = max(cx1, hx1), max(cy1, hy1)
        ix2, iy2 = min(cx2, hx2), min(cy2, hy2)
        if ix2 > ix1 and iy2 > iy1:
            penalty += ((ix2 - ix1) * (iy2 - iy1)) / c_area
    return float(penalty)


def hand_proximity_bonus(candidate: dict, hands: list[dict]) -> float:
    if not hands:
        return 0.0
    center = np.asarray(candidate["center_xy"], dtype=float)
    distances = []
    for hand in hands:
        x1, y1, x2, y2 = np.asarray(hand["bbox_xyxy"], dtype=float)
        dx = max(x1 - center[0], 0.0, center[0] - x2)
        dy = max(y1 - center[1], 0.0, center[1] - y2)
        distances.append(float(np.hypot(dx, dy)))
    return float(0.7 * np.exp(-min(distances) / 140.0))


def choose_object_candidate(candidates: list[dict], prediction: np.ndarray | None, hands: list[dict]) -> dict | None:
    if not candidates:
        return None
    scored = []
    for cand in candidates:
        center = np.asarray(cand["center_xy"], dtype=float)
        if prediction is None:
            distance_penalty = 0.0
            tracking_penalty = 0.0
        else:
            distance_penalty = 0.00002 * float(np.sum((center - prediction) ** 2))
            tracking_penalty = min(0.50, distance_penalty)
        vertical_prior = 0.0035 * max(0.0, center[1] - 390.0) + 0.0040 * max(0.0, 135.0 - center[1])
        overlap = hand_overlap_penalty(cand, hands)
        proximity = hand_proximity_bonus(cand, hands)
        occlusion_penalty = 1.5 * max(0.0, overlap - 0.45)
        score = (
            1.25 * cand["fill"]
            + 1.00 * cand["circularity"]
            + 0.24 * np.log1p(cand["area"])
            + proximity
            - tracking_penalty
            - 0.8 * overlap
            - occlusion_penalty
            - vertical_prior
        )
        scored.append((score, cand))
    return max(scored, key=lambda item: item[0])[1]


def track_object(annotations: list[dict], frame_dir: Path, max_gap_frames: int, alpha: float) -> dict:
    track: list[dict] = []
    prev_center: np.ndarray | None = None
    prev_bbox: np.ndarray | None = None
    velocity = np.zeros(2, dtype=float)
    lost = max_gap_frames + 1
    for ann in annotations:
        frame = cv2.imread(str(frame_dir / ann["image"]))
        if frame is None:
            raise RuntimeError(f"failed to read sampled frame {ann['image']} for object tracking")
        wants_tomato = "tomato" in ann["caption"].lower()
        candidates = red_tomato_candidates(frame) if wants_tomato else []
        prediction = None if prev_center is None else prev_center + velocity
        chosen = choose_object_candidate(candidates, prediction, ann["hands"])
        if chosen is not None:
            center = np.asarray(chosen["center_xy"], dtype=float)
            bbox = np.asarray(chosen["bbox_xyxy"], dtype=float)
            if prev_center is None:
                smooth_center = center
                smooth_bbox = bbox
            else:
                smooth_center = alpha * center + (1.0 - alpha) * prediction
                smooth_bbox = alpha * bbox + (1.0 - alpha) * prev_bbox
                velocity = 0.7 * velocity + 0.3 * (smooth_center - prev_center)
            prev_center = smooth_center
            prev_bbox = smooth_bbox
            lost = 0
            overlap = hand_overlap_penalty(chosen, ann["hands"])
            status = "detected_occluded" if overlap > 0.45 else "detected_smoothed"
            obj = {
                "backend": "caption_conditioned_red_component_tracker",
                "label": "tomato",
                "status": status,
                "bbox_xyxy": smooth_bbox.astype(float).tolist(),
                "center_xy": smooth_center.astype(float).tolist(),
                "raw_bbox_xyxy": chosen["bbox_xyxy"],
                "raw_center_xy": chosen["center_xy"],
                "candidate_count": len(candidates),
                "hand_overlap": overlap,
                "pose_status": "not_estimated_2d_track_only",
            }
        elif wants_tomato and prev_center is not None and lost < max_gap_frames:
            prev_center = prev_center + velocity
            if prev_bbox is not None:
                prev_bbox = prev_bbox + np.r_[velocity, velocity]
            lost += 1
            obj = {
                "backend": "caption_conditioned_red_component_tracker",
                "label": "tomato",
                "status": "predicted_lost",
                "bbox_xyxy": prev_bbox.astype(float).tolist() if prev_bbox is not None else None,
                "center_xy": prev_center.astype(float).tolist(),
                "candidate_count": len(candidates),
                "pose_status": "not_estimated_2d_track_only",
            }
        else:
            obj = {
                "backend": "caption_conditioned_red_component_tracker",
                "label": "tomato",
                "status": "not_visible" if not wants_tomato else "lost",
                "candidate_count": len(candidates),
                "pose_status": "not_estimated_2d_track_only",
            }
            if not wants_tomato:
                prev_center = None
                prev_bbox = None
                velocity[:] = 0.0
                lost = max_gap_frames + 1
        ann["object"] = obj
        track.append(obj)
    statuses = [obj["status"] for obj in track]
    return {
        "backend": "caption_conditioned_red_component_tracker",
        "tracked_label": "tomato",
        "tracked_frames": statuses.count("detected_smoothed"),
        "occluded_tracked_frames": statuses.count("detected_occluded"),
        "predicted_frames": statuses.count("predicted_lost"),
        "lost_frames": statuses.count("lost"),
        "not_visible_frames": statuses.count("not_visible"),
        "pose_status": "not_estimated_2d_track_only",
    }


def hand_depth_scale_from_colmap(output_dir: Path, annotations: list[dict], max_samples: int = 120) -> dict:
    model_dir = output_dir / "sfm_colmap_keyframes" / "sparse" / "best"
    if not model_dir.exists():
        return {"status": "unavailable", "reason": "missing pycolmap model"}
    recon = pycolmap.Reconstruction(model_dir)
    images = {image.name: image for image in recon.images.values()}
    ratios: list[float] = []
    for ann in annotations:
        if len(ratios) >= max_samples:
            break
        image = images.get(ann["image"])
        if image is None or not image.has_pose:
            continue
        cam_from_world = np.eye(4)
        cam_from_world[:3, :4] = np.asarray(image.cam_from_world().matrix(), dtype=float)
        obs = []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            world = np.r_[recon.points3D[point2d.point3D_id].xyz, 1.0]
            cam = cam_from_world @ world
            if cam[2] > 0:
                obs.append((float(point2d.xy[0]), float(point2d.xy[1]), float(cam[2])))
        if not obs:
            continue
        obs_np = np.asarray(obs, dtype=float)
        for hand in ann["hands"]:
            x1, y1, x2, y2 = np.asarray(hand["bbox_xyxy"], dtype=float)
            pad = 40.0
            mask = (
                (obs_np[:, 0] >= x1 - pad)
                & (obs_np[:, 0] <= x2 + pad)
                & (obs_np[:, 1] >= y1 - pad)
                & (obs_np[:, 1] <= y2 + pad)
            )
            if mask.sum() < 8:
                continue
            colmap_depth = float(np.median(obs_np[mask, 2]))
            joints = np.asarray(hand["joints3d_camera"], dtype=float) + np.asarray(hand["cam_t"], dtype=float)
            wilor_depth = float(np.median(joints[:, 2]))
            if colmap_depth > 0 and wilor_depth > 0:
                ratios.append(colmap_depth / wilor_depth)
    if not ratios:
        return {"status": "unavailable", "reason": "no nearby sparse points around hands"}
    arr = np.asarray(ratios, dtype=float)
    return {
        "status": "estimated_for_visualization",
        "median_colmap_depth_per_wilor_depth": float(np.median(arr)),
        "sample_count": int(arr.size),
        "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "contract": "relative-scale visualization only; this is not metric calibration",
    }


def attach_relative_world_hands(annotations: list[dict], depth_scale_qc: dict) -> None:
    if depth_scale_qc.get("status") != "estimated_for_visualization":
        for ann in annotations:
            for hand in ann["hands"]:
                hand["world_coordinate_status"] = "unavailable_no_depth_scale"
        return
    scale = float(depth_scale_qc["median_colmap_depth_per_wilor_depth"])
    for ann in annotations:
        camera = ann["camera"]
        if "T_world_camera" not in camera:
            for hand in ann["hands"]:
                hand["world_coordinate_status"] = "unavailable_no_camera_pose"
            continue
        T = np.asarray(camera["T_world_camera"], dtype=float)
        for hand in ann["hands"]:
            cam_t = np.asarray(hand["cam_t"], dtype=float)
            joints_cam = scale * (np.asarray(hand["joints3d_camera"], dtype=float) + cam_t)
            joints_h = np.c_[joints_cam, np.ones(len(joints_cam))]
            hand["joints3d_world_relative"] = (T @ joints_h.T).T[:, :3].astype(float).tolist()
            if "vertices_camera_sample" in hand:
                verts_cam = scale * (np.asarray(hand["vertices_camera_sample"], dtype=float) + cam_t)
                verts_h = np.c_[verts_cam, np.ones(len(verts_cam))]
                hand["vertices_world_relative_sample"] = (T @ verts_h.T).T[:, :3].astype(float).tolist()
            hand["world_coordinate_status"] = "relative_depth_scaled"
            hand["world_scale"] = scale


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def draw_caption(frame: np.ndarray, text: str) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    words = text.split()
    lines: list[str] = []
    line = ""
    max_width = frame.shape[1] - 24
    for word in words:
        trial = word if not line else f"{line} {word}"
        if cv2.getTextSize(trial, font, scale, thickness)[0][0] <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
        if len(lines) == 2:
            break
    if line and len(lines) < 2:
        lines.append(line)
    y0 = frame.shape[0] - 24 * len(lines) - 10
    cv2.rectangle(frame, (0, y0 - 8), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, row in enumerate(lines):
        cv2.putText(frame, row, (12, y0 + 17 + i * 24), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, hands: list[dict]) -> None:
    for hand in hands:
        pts = np.asarray(hand["joints2d"], dtype=float)
        color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
        x1, y1, x2, y2 = np.asarray(hand["bbox_xyxy"], dtype=int)
        status = hand.get("filter_status", "measured")
        thickness = 1 if status != "measured_smoothed" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thickness + 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p.astype(int)), 4, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{hand['side']} {status}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_object(frame: np.ndarray, obj: dict) -> None:
    if obj.get("bbox_xyxy") is None or obj["status"] in {"not_visible", "lost"}:
        return
    x1, y1, x2, y2 = np.asarray(obj["bbox_xyxy"], dtype=int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_COLOR, 2)
    cx, cy = np.asarray(obj["center_xy"], dtype=int)
    cv2.drawMarker(frame, (cx, cy), OBJECT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    cv2.putText(frame, f"tomato {obj['status']}", (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, OBJECT_COLOR, 1, cv2.LINE_AA)


def path_panel_xy(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    xy = points[:, [0, 2]]
    center = xy.mean(axis=0)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    scale = 0.60 * min(width, height) / span
    origin = np.array([width * 0.5, height * 0.48])
    return origin + (xy - center) * np.array([scale, -scale])


def camera_centers(camera_poses: list[dict]) -> np.ndarray:
    centers = []
    for pose in camera_poses:
        if "T_world_camera" in pose:
            centers.append(np.asarray(pose["T_world_camera"], dtype=float)[:3, 3])
    return np.asarray(centers, dtype=float).reshape(-1, 3)


def panel_project(points: np.ndarray, width: int, height: int, center: np.ndarray, span: float, origin_y: float = 0.48) -> np.ndarray:
    xy = points[:, [0, 2]]
    scale = 0.60 * min(width, height) / max(span, 1e-6)
    origin = np.array([width * 0.5, height * origin_y])
    return origin + (xy - center) * np.array([scale, -scale])


def draw_local_hand_panel(panel: np.ndarray, ann: dict, origin: np.ndarray) -> None:
    cv2.putText(panel, "camera-local MANO", tuple((origin + np.array([-145, -92])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (65, 65, 65), 1, cv2.LINE_AA)
    for hand in ann["hands"]:
        joints = np.asarray(hand["joints3d_camera"], dtype=float)
        local = joints - joints[0]
        pts = origin + local[:, [0, 1]] * np.array([720.0, -720.0])
        color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
        for a, b in HAND_EDGES:
            cv2.line(panel, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)


def draw_world_hand_anchors(panel: np.ndarray, ann: dict, width: int, height: int, center: np.ndarray, span: float) -> None:
    for hand in ann["hands"]:
        joints = hand.get("joints3d_world_relative")
        if joints is None:
            continue
        joints_np = np.asarray(joints, dtype=float)
        wrist = panel_project(joints_np[[0]], width, height, center, span)[0]
        wrist[1] = max(wrist[1], 92.0)
        color = LEFT_COLOR if hand["side"] == "left" else RIGHT_COLOR
        cv2.circle(panel, tuple(wrist.astype(int)), 4, color, -1, cv2.LINE_AA)
        cv2.line(panel, tuple(wrist.astype(int)), tuple((wrist + np.array([10, 12])).astype(int)), color, 1, cv2.LINE_AA)
        cv2.putText(panel, hand["side"], tuple((wrist + np.array([13, 18])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)


def render_world_panel(render: RenderInfo, ann: dict, all_centers: np.ndarray, camera_pose: dict) -> np.ndarray:
    panel = np.full((render.height, render.width, 3), 244, dtype=np.uint8)
    cv2.putText(panel, "Head camera SfM", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)
    text = (
        f"frame {ann['frame_idx']} | camera {camera_pose['camera_status']} | "
        f"hands {len(ann['hands'])} | object {ann['object']['status']}"
    )
    cv2.putText(panel, text[:118], (22, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    if all_centers.size:
        path_xy = all_centers[:, [0, 2]]
        path_center = path_xy.mean(axis=0)
        path_span = max(float(np.ptp(path_xy[:, 0])), float(np.ptp(path_xy[:, 1])), 1e-6)
    else:
        path_center = np.zeros(2, dtype=float)
        path_span = 1.0
    pts = path_panel_xy(all_centers, render.width, render.height)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(panel, tuple(a.astype(int)), tuple(b.astype(int)), (145, 145, 145), 2, cv2.LINE_AA)
    for p in pts[:: max(1, len(pts) // 36)]:
        cv2.circle(panel, tuple(p.astype(int)), 2, (110, 110, 110), -1, cv2.LINE_AA)
    if "T_world_camera" in camera_pose and all_centers.size:
        current_center = np.asarray(camera_pose["T_world_camera"], dtype=float)[:3, 3]
        current = path_panel_xy(np.vstack([all_centers, current_center]), render.width, render.height)[-1]
        cv2.circle(panel, tuple(current.astype(int)), 8, (30, 30, 220), -1, cv2.LINE_AA)
        cv2.putText(panel, "camera", tuple((current + np.array([12, -8])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 220), 2, cv2.LINE_AA)
    draw_world_hand_anchors(panel, ann, render.width, render.height, path_center, path_span)
    draw_local_hand_panel(panel, ann, np.array([render.width * 0.50, render.height * 0.79]))
    cv2.putText(
        panel,
        "SfM scale is relative; MANO shown camera-local",
        (22, render.height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_videos(output_dir: Path, annotations: list[dict], samples: list[FrameSample], frame_dir: Path, render: RenderInfo, fps: float) -> None:
    all_centers = camera_centers([ann["camera"] for ann in annotations])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(output_dir / "overlay.mp4"), fourcc, fps, (render.width, render.height))
    world_writer = cv2.VideoWriter(str(output_dir / "reconstruction_3d.mp4"), fourcc, fps, (render.width, render.height))
    side_writer = cv2.VideoWriter(str(output_dir / "side_by_side.mp4"), fourcc, fps, (render.width * 2, render.height))
    if not overlay_writer.isOpened() or not world_writer.isOpened() or not side_writer.isOpened():
        raise RuntimeError("failed to open video writers")
    by_name = {sample.name: ann for sample, ann in zip(samples, annotations)}
    try:
        for sample in samples:
            frame = cv2.imread(str(frame_dir / sample.name))
            if frame is None:
                raise RuntimeError(f"failed to read {sample.name}")
            ann = by_name[sample.name]
            draw_hands(frame, ann["hands"])
            draw_object(frame, ann["object"])
            cam_status = ann["camera"]["camera_status"]
            cv2.putText(
                frame,
                f"frame {ann['frame_idx']} | WiLoR {len(ann['hands'])} hands | camera {cam_status}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_caption(frame, ann["caption"])
            panel = render_world_panel(render, ann, all_centers, ann["camera"])
            overlay_writer.write(frame)
            world_writer.write(panel)
            side_writer.write(np.concatenate([frame, panel], axis=1))
    finally:
        overlay_writer.release()
        world_writer.release()
        side_writer.release()


def build_annotations(samples: list[FrameSample], actions: list[dict], cameras: list[dict]) -> list[dict]:
    return [
        {
            "frame_idx": sample.frame_idx,
            "time_s": sample.time_s,
            "image": sample.name,
            "caption": caption_for_frame(actions, sample.frame_idx),
            "camera": camera,
            "raw_hands": [],
            "hands": [],
            "object": {"status": "not_run"},
        }
        for sample, camera in zip(samples, cameras)
    ]


def run(args: argparse.Namespace) -> None:
    started = time.time()
    clip = Path(args.clip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = output_dir / "sampled_frames"
    keyframe_dir = output_dir / "camera_keyframes"
    actions = load_actions(clip.with_suffix(".json"))
    info, samples = extract_sampled_frames(clip, dense_dir, args.output_fps, args.render_width)
    render_height = int(round(args.render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    render = RenderInfo(args.render_width, render_height)
    keyframes = select_keyframes(samples, args.output_fps, args.camera_keyframe_fps)
    copy_keyframes(dense_dir, keyframe_dir, keyframes)
    ensure_wilor_assets(Path(args.wilor_root), Path(args.mano_right))
    colmap = run_colmap(keyframe_dir, output_dir, args.reuse_colmap)
    cameras, camera_qc = interpolate_camera_poses(colmap, samples, args.camera_smooth_window, args.camera_edge_prediction_frames)
    annotations = build_annotations(samples, actions, cameras)
    model, cfg, detector, device = load_wilor_backend(Path(args.wilor_root))
    detection_counts: list[float] = []
    for i, ann in enumerate(annotations):
        frame = cv2.imread(str(dense_dir / ann["image"]))
        if frame is None:
            raise RuntimeError(f"failed to read sampled frame {ann['image']}")
        hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.hand_rescale_factor, args.hand_batch_size)
        ann["raw_hands"] = hands
        detection_counts.append(float(len(hands)))
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"wilor_processed={i + 1}/{len(samples)} elapsed_s={time.time() - started:.1f}", flush=True)
    hand_filter_qc = filter_hands(annotations, args.hand_max_gap_frames, args.hand_filter_alpha)
    object_qc = track_object(annotations, dense_dir, args.object_max_gap_frames, args.object_filter_alpha)
    depth_scale_qc = hand_depth_scale_from_colmap(output_dir, annotations)
    attach_relative_world_hands(annotations, depth_scale_qc)
    pose_rate = camera_qc["dense_pose_rate"]
    raw_hand_frame_rate = sum(1 for n in detection_counts if n > 0) / max(1, len(detection_counts))
    filtered_hand_frame_rate = sum(1 for ann in annotations if ann["hands"]) / max(1, len(annotations))
    object_accounted = object_qc["tracked_frames"] + object_qc["occluded_tracked_frames"] + object_qc["predicted_frames"]
    tracked_object_rate = object_accounted / max(1, object_accounted + object_qc["lost_frames"])
    if pose_rate < args.min_camera_pose_rate:
        raise RuntimeError(f"dense camera pose rate {pose_rate:.3f} below required {args.min_camera_pose_rate:.3f}")
    if filtered_hand_frame_rate < args.min_hand_frame_rate:
        raise RuntimeError(f"filtered hand-frame rate {filtered_hand_frame_rate:.3f} below required {args.min_hand_frame_rate:.3f}")
    render_videos(output_dir, annotations, samples, dense_dir, render, args.output_fps)
    qc = {
        "quality_decision": "v1_candidate_needs_visual_qc",
        "clip": str(clip),
        "source_fps": info.fps,
        "output_fps": args.output_fps,
        "processed_frames": len(samples),
        "render_size": [render.width, render.height],
        "camera_backend": colmap["backend"],
        "camera_keyframe_fps": args.camera_keyframe_fps,
        "camera_keyframes": len(keyframes),
        "camera_registered_keyframes": colmap["registered_images"],
        "camera_num_points3d": colmap["num_points3d"],
        "camera_dense_qc": camera_qc,
        "camera_scale_status": "relative",
        "hand_backend": "WiLoR",
        "hand_raw_frames_with_detection": int(sum(1 for n in detection_counts if n > 0)),
        "hand_raw_frame_rate": raw_hand_frame_rate,
        "hand_filtered_frame_rate": filtered_hand_frame_rate,
        "hands_per_frame_raw": summarize(detection_counts),
        "hand_filter_qc": hand_filter_qc,
        "object_qc": object_qc,
        "object_tracked_rate_inside_tomato_segments": tracked_object_rate,
        "hand_colmap_depth_scale_qc": depth_scale_qc,
        "deliverables": {
            "overlay": str(output_dir / "overlay.mp4"),
            "reconstruction_3d": str(output_dir / "reconstruction_3d.mp4"),
            "side_by_side": str(output_dir / "side_by_side.mp4"),
            "annotations": str(output_dir / "annotations.json"),
            "qc": str(output_dir / "qc.json"),
            "sfm_model": str(output_dir / "sfm_colmap_keyframes" / "sparse" / "best"),
        },
        "backend_attempts": {
            "wilor": {"status": "ran", "checkpoint": str(Path(args.wilor_root) / "pretrained_models" / "wilor_final.ckpt")},
            "pycolmap": {"status": "ran", "model_dir": str(output_dir / "sfm_colmap_keyframes" / "sparse" / "best")},
            "dpvo": {
                "status": "build_failed",
                "reason": "CUDA extension compilation failed against current Torch/CUDA dispatch API: AT_DISPATCH_FLOATING_TYPES_AND_HALF used tensor.type() in dpvo/altcorr/correlation_kernel.cu.",
            },
            "yoloworld_object_detector": {
                "status": "not_used",
                "reason": "Ultralytics attempted an implicit CLIP auto-install outside the uv environment; v1 uses the local caption-conditioned tracker instead.",
            },
        },
        "known_limits": [
            "Camera pose is pycolmap offline SfM on keyframes with dense interpolation, not online SLAM.",
            "COLMAP and WiLoR remain relative-scale systems; the render does not claim metric hand-world fusion.",
            "Object output is 2D tomato detection/tracking only; 6D object pose needs a real object-pose backend and shape or CAD prior.",
            "Temporal filtering marks measured, interpolated, and predicted frames in JSON and overlay.",
        ],
    }
    (output_dir / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--output-dir", default="outputs/examples/tomato_v1_wilor_colmap_dense")
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--camera-keyframe-fps", type=float, default=1.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--wilor-root", default=str(DEFAULT_WILOR_ROOT))
    parser.add_argument("--mano-right", default=str(DEFAULT_MANO_RIGHT))
    parser.add_argument("--hand-rescale-factor", type=float, default=2.0)
    parser.add_argument("--hand-batch-size", type=int, default=8)
    parser.add_argument("--hand-max-gap-frames", type=int, default=12)
    parser.add_argument("--hand-filter-alpha", type=float, default=0.65)
    parser.add_argument("--object-max-gap-frames", type=int, default=10)
    parser.add_argument("--object-filter-alpha", type=float, default=0.60)
    parser.add_argument("--camera-smooth-window", type=int, default=7)
    parser.add_argument("--camera-edge-prediction-frames", type=int, default=30)
    parser.add_argument("--min-camera-pose-rate", type=float, default=0.95)
    parser.add_argument("--min-hand-frame-rate", type=float, default=0.75)
    parser.add_argument("--reuse-colmap", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
