#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pycolmap
import torch


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)
DEFAULT_MANO_ROOT = Path("/data/dex_home/yiwen/mano_assets/mano")
DEFAULT_MANOTORCH_ROOT = Path("/data/dex_home/yiwen/manotorch")

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class RenderInfo:
    width: int
    height: int


def patch_legacy_mano_imports() -> None:
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


def load_mano_layers(mano_root: Path, manotorch_root: Path) -> dict[str, object]:
    if not (mano_root / "models" / "MANO_RIGHT.pkl").exists():
        raise FileNotFoundError(f"missing MANO_RIGHT.pkl under {mano_root / 'models'}")
    if not (mano_root / "models" / "MANO_LEFT.pkl").exists():
        raise FileNotFoundError(f"missing MANO_LEFT.pkl under {mano_root / 'models'}")
    if not (manotorch_root / "manotorch" / "manolayer.py").exists():
        raise FileNotFoundError(f"missing manotorch source under {manotorch_root}")
    patch_legacy_mano_imports()
    sys.path.insert(0, str(manotorch_root))
    from manotorch.manolayer import ManoLayer

    return {
        "Left": ManoLayer(rot_mode="axisang", use_pca=False, side="left", center_idx=None, mano_assets_root=str(mano_root), flat_hand_mean=False),
        "Right": ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None, mano_assets_root=str(mano_root), flat_hand_mean=False),
    }


def load_actions(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    tasks = data.get("tasks") or []
    if not tasks:
        raise ValueError(f"no tasks in {json_path}")
    return tasks[0].get("actions") or []


def caption_for_frame(actions: list[dict], frame_idx: int) -> str:
    for action in actions:
        if int(action.get("start_frame", -1)) <= frame_idx < int(action.get("end_frame", -1)):
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


def extract_sampled_frames(clip: Path, frame_dir: Path, sample_fps: float, render_width: int) -> tuple[ClipInfo, list[dict]]:
    cap, info = open_video(clip)
    stride = max(1, round(info.fps / sample_fps))
    render_height = int(round(render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    frames = []
    idx = -1
    frame_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        name = f"frame_{idx:06d}.jpg"
        resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frame_dir / name), resized)
        frames.append({"frame_idx": idx, "time_s": idx / info.fps, "name": name})
    cap.release()
    return info, frames


def run_colmap(frame_dir: Path, output_dir: Path) -> dict:
    sfm_dir = output_dir / "sfm"
    database = sfm_dir / "database.db"
    sparse = sfm_dir / "sparse"
    if (sparse / "best").exists():
        return read_colmap_reconstruction(sparse / "best")
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sparse.mkdir(parents=True, exist_ok=True)
    try:
        pycolmap.extract_features(database, frame_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.cpu)
        pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
        maps = pycolmap.incremental_mapping(database, frame_dir, sparse)
    except Exception as exc:
        raise RuntimeError("pycolmap reconstruction failed") from exc
    if not maps:
        raise RuntimeError("pycolmap returned no reconstruction")
    recon = max(maps.values(), key=lambda r: r.num_reg_images())
    best_dir = sparse / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    recon.write(best_dir)
    return colmap_status_from_reconstruction(recon, best_dir)


def read_colmap_reconstruction(path: Path) -> dict:
    try:
        recon = pycolmap.Reconstruction(path)
    except Exception as exc:
        raise RuntimeError(f"failed to read pycolmap reconstruction: {path}") from exc
    return colmap_status_from_reconstruction(recon, path)


def colmap_status_from_reconstruction(recon, sparse_model: Path) -> dict:
    frame_entries = []
    for image_id, image in recon.images.items():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world()
        mat3x4 = np.asarray(cam_from_world.matrix(), dtype=float)
        mat = np.eye(4, dtype=float)
        mat[:3, :4] = mat3x4
        world_from_cam = np.linalg.inv(mat)
        frame_entries.append({"name": image.name, "T_world_camera": world_from_cam.tolist()})
    if not frame_entries:
        raise RuntimeError(f"pycolmap reconstruction has no registered camera poses: {sparse_model}")
    return {
        "status": "ok",
        "registered_images": int(recon.num_reg_images()),
        "num_points3d": int(recon.num_points3D()),
        "registration_rate": float(recon.num_reg_images() / max(1, len(list(recon.images.keys())))),
        "frames": sorted(frame_entries, key=lambda x: x["name"]),
        "sparse_model": str(sparse_model),
    }


def colmap_centers_by_name(colmap_status: dict) -> dict[str, np.ndarray]:
    centers = {}
    for frame in colmap_status.get("frames", []):
        T = np.asarray(frame["T_world_camera"], dtype=float)
        centers[frame["name"]] = T[[0, 2], 3]
    return centers


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "status": "empty"}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def fit_mano_to_2d(layer, target_uv: np.ndarray, width: int, height: int, iterations: int) -> dict:
    device = torch.device("cpu")
    target = torch.tensor(target_uv, dtype=torch.float32, device=device)
    pose = torch.zeros(1, 48, dtype=torch.float32, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, dtype=torch.float32, device=device, requires_grad=True)
    rot_z = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    trans = torch.zeros(1, 2, dtype=torch.float32, device=device, requires_grad=True)
    log_scale = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([pose, betas, rot_z, trans, log_scale], lr=0.035)
    center = torch.tensor([width / 2, height / 2], dtype=torch.float32, device=device)
    norm = torch.tensor([width, height], dtype=torch.float32, device=device)
    best_repro = None
    best_payload = None
    for _ in range(iterations):
        opt.zero_grad()
        out = layer(pose, betas)
        joints = out.joints[0, :, :2]
        theta = rot_z[0]
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        pred = (joints @ rot.T) * torch.exp(log_scale)[0] * min(width, height) + center + trans[0] * norm
        repro = torch.mean(torch.linalg.norm(pred - target, dim=1))
        pose_reg = 0.002 * torch.mean(pose[:, 3:] ** 2)
        beta_reg = 0.01 * torch.mean(betas**2)
        loss = repro + pose_reg + beta_reg
        repro_value = repro.detach()
        if best_repro is None or float(repro_value.cpu()) < float(best_repro.cpu()):
            best_repro = repro_value
            best_payload = {
                "joints": out.joints.detach().cpu().numpy()[0].astype(float),
                "verts": out.verts.detach().cpu().numpy()[0].astype(float),
                "joints2d": pred.detach().cpu().numpy().astype(float),
                "pose": pose.detach().cpu().numpy()[0].astype(float),
                "betas": betas.detach().cpu().numpy()[0].astype(float),
            }
        loss.backward()
        opt.step()
    assert best_repro is not None and best_payload is not None
    return {
        "status": "fit_2d",
        "mean_reprojection_px": float(best_repro.cpu()),
        "joints3d": best_payload["joints"].tolist(),
        "joints2d": best_payload["joints2d"].tolist(),
        "vertices_sample": best_payload["verts"][::20].tolist(),
        "pose": best_payload["pose"].tolist(),
        "betas": best_payload["betas"].tolist(),
    }


def detect_tomato_mask(frame: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 55, 45), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 55, 45), (179, 255, 255))
    mask = cv2.medianBlur(cv2.bitwise_or(mask1, mask2), 5)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        x, y, w, h = [int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        comps.append({"area": area, "bbox": [x, y, w, h], "centroid": [float(centroids[i][0]), float(centroids[i][1])]})
    comps.sort(key=lambda c: c["area"], reverse=True)
    return {
        "status": "mask_track" if comps else "not_visible",
        "components": comps[:3],
        "total_components": len(comps),
        "total_area_px": int(sum(c["area"] for c in comps)),
    }


def detect_semantic_object(frame: np.ndarray, caption: str) -> dict:
    target = "tomato" if "tomato" in caption.lower() else ""
    if not target:
        return {"status": "not_requested", "target": "", "reason": "caption_has_no_tomato"}
    result = detect_tomato_mask(frame)
    result["target"] = target
    return result


def wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if cv2.getTextSize(trial, font, scale, thickness)[0][0] <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def draw_hand(frame: np.ndarray, points: list[dict], label: str) -> None:
    color = (80, 220, 80) if label == "Left" else (80, 180, 255)
    for a, b in HAND_EDGES:
        pa, pb = points[a], points[b]
        cv2.line(frame, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), color, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, color, 1, cv2.LINE_AA)


def draw_mano_projection(frame: np.ndarray, hand: dict) -> None:
    mano = hand.get("mano") or {}
    joints2d = mano.get("joints2d")
    if not joints2d:
        return
    color = (20, 210, 40) if hand["label"] == "Left" else (20, 130, 255)
    pts = np.asarray(joints2d, dtype=float)
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p.astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(p.astype(int)), 5, color, 1, cv2.LINE_AA)


def draw_caption(frame: np.ndarray, text: str) -> None:
    if not text:
        return
    lines = wrap_text(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2, frame.shape[1] - 24, 2)
    line_h = 24
    y0 = frame.shape[0] - line_h * len(lines) - 10
    cv2.rectangle(frame, (0, y0 - 8), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, y0 + i * line_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def path_to_panel(points_xy: np.ndarray, current_xy: np.ndarray | None, origin: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray | None]:
    if points_xy.size == 0:
        return points_xy.reshape(0, 2), None
    values = points_xy if current_xy is None else np.vstack([points_xy, current_xy.reshape(1, 2)])
    center = np.mean(values, axis=0)
    span = float(np.max(np.ptp(values, axis=0)))
    scale = 1.0 if span <= 1e-9 else radius / span
    panel_path = origin + (points_xy - center) * np.array([scale, -scale])
    panel_current = None if current_xy is None else origin + (current_xy - center) * np.array([scale, -scale])
    return panel_path, panel_current


def render_3d_panel(render: RenderInfo, frame_idx: int, camera_path_xy: np.ndarray, current_camera_xy: np.ndarray | None, hands: list[dict], caption: str) -> np.ndarray:
    panel = np.full((render.height, render.width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Real-backend summary", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    camera_status = "registered" if current_camera_xy is not None else "not registered"
    for i, line in enumerate(wrap_text(f"frame {frame_idx} | COLMAP {camera_status}, arbitrary scale | MANO 2D fit | {caption}", cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2, render.width - 48, 2)):
        cv2.putText(panel, line, (24, 70 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (65, 65, 65), 2, cv2.LINE_AA)
    origin = np.array([render.width * 0.5, render.height * 0.56])
    path_pts, current_pt = path_to_panel(camera_path_xy, current_camera_xy, origin, min(render.width, render.height) * 0.28)
    for a, b in zip(path_pts[:-1], path_pts[1:]):
        cv2.line(panel, tuple(a.astype(int)), tuple(b.astype(int)), (150, 150, 150), 2, cv2.LINE_AA)
    for p in path_pts[:: max(1, len(path_pts) // 25)]:
        cv2.circle(panel, tuple(p.astype(int)), 2, (120, 120, 120), -1, cv2.LINE_AA)
    if current_pt is not None:
        cv2.circle(panel, tuple(current_pt.astype(int)), 8, (40, 40, 210), -1, cv2.LINE_AA)
        cv2.putText(panel, "camera", tuple((current_pt + [12, -10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 210), 2, cv2.LINE_AA)
    for hand in hands:
        color = (70, 160, 70) if hand["label"] == "Left" else (40, 130, 230)
        if "mano" in hand and hand["mano"].get("joints3d"):
            joints = np.asarray(hand["mano"]["joints3d"], dtype=float)
            pts = origin + joints[:, :2] * 160 + np.array([0, 90])
            for a, b in HAND_EDGES:
                cv2.line(panel, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)
    return panel


def run(args: argparse.Namespace) -> None:
    started = time.time()
    clip = Path(args.clip)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / "sampled_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    actions = load_actions(clip.with_suffix(".json"))
    info, sampled = extract_sampled_frames(clip, frame_dir, args.output_fps, args.render_width)
    render_height = int(round(args.render_width * info.height / info.width))
    render = RenderInfo(args.render_width, render_height)
    if render.height % 2:
        render.height += 1
    colmap_status = run_colmap(frame_dir, out)
    mano_layers = load_mano_layers(Path(args.mano_root), Path(args.manotorch_root))

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.45, min_tracking_confidence=0.45)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(out / "overlay.mp4"), fourcc, args.output_fps, (render.width, render.height))
    recon_writer = cv2.VideoWriter(str(out / "reconstruction_3d.mp4"), fourcc, args.output_fps, (render.width, render.height))
    side_writer = cv2.VideoWriter(str(out / "side_by_side.mp4"), fourcc, args.output_fps, (render.width * 2, render.height))
    if not overlay_writer.isOpened() or not recon_writer.isOpened() or not side_writer.isOpened():
        raise RuntimeError("failed to open video writers")

    annotations = []
    detected = 0
    mano_fit_count = 0
    mano_reprojection_px: list[float] = []
    object_visible = 0
    object_component_counts: list[int] = []
    object_total_area_px: list[int] = []
    colmap_by_name = {f["name"]: f for f in colmap_status.get("frames", [])}
    camera_centers = colmap_centers_by_name(colmap_status)
    camera_path_xy = np.asarray([camera_centers[name] for name in sorted(camera_centers)], dtype=float).reshape(-1, 2)
    for i, sample in enumerate(sampled):
        frame = cv2.imread(str(frame_dir / sample["name"]))
        if frame is None:
            raise RuntimeError(f"failed to read sampled frame {sample['name']}")
        result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = caption_for_frame(actions, sample["frame_idx"])
        object_track = detect_semantic_object(frame, caption)
        if object_track["status"] == "mask_track":
            object_visible += 1
            object_component_counts.append(int(object_track["total_components"]))
            object_total_area_px.append(int(object_track["total_area_px"]))
            for comp in object_track["components"]:
                x, y, w, h = comp["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 80, 240), 2)
        hands = []
        if result.multi_hand_landmarks:
            detected += 1
            handedness = result.multi_handedness or []
            for h_idx, lm in enumerate(result.multi_hand_landmarks):
                label = "Unknown"
                score = 0.0
                if h_idx < len(handedness):
                    cls = handedness[h_idx].classification[0]
                    label = cls.label
                    score = float(cls.score)
                points = [{"x": float(p.x * render.width), "y": float(p.y * render.height), "z": float(p.z)} for p in lm.landmark]
                hand = {"label": label, "score": score, "points": points}
                if label in mano_layers:
                    uv = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
                    hand["mano"] = fit_mano_to_2d(mano_layers[label], uv, render.width, render.height, args.mano_iters)
                    mano_reprojection_px.append(float(hand["mano"]["mean_reprojection_px"]))
                    mano_fit_count += 1
                hands.append(hand)
                if "mano" in hand:
                    draw_mano_projection(frame, hand)
                else:
                    draw_hand(frame, points, label)
        colmap_frame = colmap_by_name.get(sample["name"])
        cv2.putText(
            frame,
            f"frame {sample['frame_idx']} | hands {len(hands)} | MANO fits {sum('mano' in h for h in hands)} | COLMAP {'yes' if colmap_frame else 'no'}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        draw_caption(frame, caption)
        current_camera_xy = camera_centers.get(sample["name"])
        panel = render_3d_panel(render, sample["frame_idx"], camera_path_xy, current_camera_xy, hands, caption)
        overlay_writer.write(frame)
        recon_writer.write(panel)
        side_writer.write(np.concatenate([frame, panel], axis=1))
        annotations.append(
            {
                "frame_idx": sample["frame_idx"],
                "time_s": sample["time_s"],
                "caption": caption,
                "camera": colmap_frame or {"status": "not_registered"},
                "hands": hands,
                "object": object_track,
            }
        )
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"processed={i + 1}/{len(sampled)} elapsed_s={time.time() - started:.1f}", flush=True)

    detector.close()
    overlay_writer.release()
    recon_writer.release()
    side_writer.release()
    colmap_reg_rate = float(colmap_status["registered_images"] / max(1, len(sampled)))
    if colmap_reg_rate < args.min_colmap_reg_rate:
        raise RuntimeError(f"COLMAP registered {colmap_reg_rate:.3f} of sampled frames; required {args.min_colmap_reg_rate:.3f}")
    if mano_fit_count < args.min_mano_fit_count:
        raise RuntimeError(f"MANO fit count {mano_fit_count} is below required {args.min_mano_fit_count}")
    if args.require_object_mask and object_visible == 0:
        raise RuntimeError("required object mask evidence, but no object mask frames were produced")
    qc = {
        "clip": str(clip),
        "fps": info.fps,
        "sample_fps": args.output_fps,
        "processed_frames": len(sampled),
        "colmap_registration_rate": colmap_reg_rate,
        "hand_detection_frames": detected,
        "hand_detection_rate": detected / max(1, len(sampled)),
        "mano_fit_count": mano_fit_count,
        "mano_reprojection_px": summarize_values(mano_reprojection_px),
        "object_mask_visible_frames": object_visible,
        "object_total_components": summarize_values([float(v) for v in object_component_counts]),
        "object_total_area_px": summarize_values([float(v) for v in object_total_area_px]),
        "colmap": colmap_status,
        "deliverables": {
            "overlay": str(out / "overlay.mp4"),
            "reconstruction_3d": str(out / "reconstruction_3d.mp4"),
            "side_by_side": str(out / "side_by_side.mp4"),
            "annotations": str(out / "annotations.json"),
            "qc": str(out / "qc.json"),
            "sampled_frames": str(frame_dir),
        },
        "quality_decision": "real_partial",
        "quality_notes": [
            "MANO is fitted per detected hand using 2D landmarks and manotorch; this is a real differentiable MANO fit but uses landmark supervision, not HaMeR/WiLoR RGB regression.",
            "COLMAP is attempted on sampled frames; registered camera frames are used where available.",
            "Object module is a tomato color-mask track, not 6D object pose; reviewed frames show false positives on red packaging and skin-like regions.",
            "No physical refinement is implemented yet beyond MANO 2D fitting regularization.",
        ],
    }
    (out / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    (out / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--output-dir", default="outputs/examples/tomato_v1_real")
    parser.add_argument("--output-fps", type=float, default=1.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--mano-root", default=str(DEFAULT_MANO_ROOT))
    parser.add_argument("--manotorch-root", default=str(DEFAULT_MANOTORCH_ROOT))
    parser.add_argument("--mano-iters", type=int, default=40)
    parser.add_argument("--min-colmap-reg-rate", type=float, default=0.5)
    parser.add_argument("--min-mano-fit-count", type=int, default=1)
    parser.add_argument("--require-object-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
