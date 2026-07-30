#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
from tqdm import tqdm

DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)
DEFAULT_WILOR_ROOT = Path("third_party/WiLoR")
DEFAULT_MANO_RIGHT = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_RIGHT.pkl")


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


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


def explicit_hand_sides(actions: list[dict]) -> set[str] | None:
    sides: set[str] = set()
    for action in actions:
        raw = action.get("hand_sides")
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise RuntimeError(f"hand_sides must be a list when present: {action}")
        for item in raw:
            side = str(item).lower()
            if side not in {"left", "right"}:
                raise RuntimeError(f"unknown hand side in hand_sides: {item}")
            sides.add(side)
    return sides or None


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_wilor_assets(args.wilor_root, args.mano_right)
    model, cfg, detector, device = load_wilor_backend(args.wilor_root)

    json_path = args.actions_json if args.actions_json is not None else args.clip.with_suffix(".json")
    actions = load_actions(json_path)
    allowed_sides = explicit_hand_sides(actions)
    cap, info = open_video(args.clip)
    frame_start = 0 if args.frame_start is None else int(args.frame_start)
    frame_end = info.frame_count - 1 if args.frame_end is None else int(args.frame_end)
    if frame_start < 0 or frame_end < frame_start or frame_end >= info.frame_count:
        raise RuntimeError(f"invalid frame window {frame_start}:{frame_end} for {info.frame_count} frames")
    if args.max_frames is not None:
        frame_end = min(frame_end, frame_start + int(args.max_frames) - 1)
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start):
        raise RuntimeError(f"failed to seek to frame {frame_start}")

    frames = []
    started = time.time()
    detected_frames = 0
    detected_hands = 0
    filtered_hands = 0
    frame_idx = frame_start
    pbar = tqdm(total=frame_end - frame_start + 1, desc="wilor_full_frame")
    try:
        while frame_idx <= frame_end:
            ok, frame = cap.read()
            if not ok:
                break
            hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.rescale_factor, args.batch_size)
            if allowed_sides is not None:
                before = len(hands)
                hands = [hand for hand in hands if str(hand.get("side", "")).lower() in allowed_sides]
                filtered_hands += before - len(hands)
            if hands:
                detected_frames += 1
                detected_hands += len(hands)
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "time_s": frame_idx / info.fps,
                    "caption": caption_for_frame(actions, frame_idx),
                    "raw_hands": hands,
                }
            )
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        del model, detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not frames:
        raise RuntimeError("WiLoR received no frames")

    raw_path = args.output_dir / "wilor_raw.json"
    raw_path.write_text(json.dumps({"video": info.__dict__, "frames": frames}, indent=2), encoding="utf-8")

    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "video": info.__dict__,
        "processed_frames": len(frames),
        "source_frame_range": [int(frames[0]["frame_idx"]), int(frames[-1]["frame_idx"])],
        "full_source_timeline": bool(
            args.frame_start is None and args.frame_end is None and args.max_frames is None and len(frames) == info.frame_count
        ),
        "frames_with_hands": detected_frames,
        "hand_detection_rate": detected_frames / max(1, len(frames)),
        "detected_hands": detected_hands,
        "mean_hands_per_frame": detected_hands / max(1, len(frames)),
        "explicit_hand_sides": sorted(allowed_sides) if allowed_sides is not None else None,
        "filtered_hands_by_explicit_side": filtered_hands,
        "elapsed_s": time.time() - started,
        "raw_path": str(raw_path),
    }
    (args.output_dir / "wilor_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/wilor"))
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--actions-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
