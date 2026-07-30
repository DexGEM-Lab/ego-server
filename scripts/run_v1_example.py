#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
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
    scale_x: float
    scale_y: float


def load_mano_template(mano_root: Path, manotorch_root: Path) -> dict:
    right = mano_root / "models" / "MANO_RIGHT.pkl"
    left = mano_root / "models" / "MANO_LEFT.pkl"
    if not right.exists() or not left.exists():
        raise FileNotFoundError(f"expected MANO_LEFT.pkl and MANO_RIGHT.pkl under {mano_root / 'models'}")
    if not (manotorch_root / "manotorch" / "manolayer.py").exists():
        raise FileNotFoundError(f"manotorch source not found under {manotorch_root}")

    import inspect
    import sys

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    numpy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in numpy_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    sys.path.insert(0, str(manotorch_root))
    from manotorch.manolayer import ManoLayer

    templates = {}
    for side in ("left", "right"):
        layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            side=side,
            center_idx=None,
            mano_assets_root=str(mano_root),
            flat_hand_mean=False,
        )
        pose = torch.zeros(1, 48)
        shape = torch.zeros(1, 10)
        with torch.no_grad():
            out = layer(pose, shape)
        templates[side] = {
            "verts": out.verts[0].cpu().numpy().astype(float),
            "joints": out.joints[0].cpu().numpy().astype(float),
            "faces": layer.th_faces.cpu().numpy().astype(int),
        }
    return templates


def write_template_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for f in faces:
        a, b, c = f + 1
        lines.append(f"f {a} {b} {c}\n")
    path.write_text("".join(lines), encoding="utf-8")


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
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: fps={fps} size={width}x{height} frames={frame_count}")
    return cap, ClipInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def draw_caption(frame: np.ndarray, text: str) -> None:
    if not text:
        return
    pad = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 2
    lines = wrap_text(text, font, scale, thickness, frame.shape[1] - 2 * pad, max_lines=2)
    line_h = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 9
    y0 = frame.shape[0] - line_h * len(lines) - pad
    cv2.rectangle(frame, (0, y0 - pad), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, y0 + i * line_h + line_h - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        width = cv2.getTextSize(trial, font, scale, thickness)[0][0]
        if width <= max_width:
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


def draw_hand(frame: np.ndarray, pts: list[dict], label: str, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    color = (80, 220, 80) if label == "Left" else (80, 180, 255)
    for a, b in HAND_EDGES:
        pa, pb = pts[a], pts[b]
        cv2.line(
            frame,
            (int(pa["x"] * scale_x), int(pa["y"] * scale_y)),
            (int(pb["x"] * scale_x), int(pb["y"] * scale_y)),
            color,
            3,
            cv2.LINE_AA,
        )
    for p in pts:
        cv2.circle(frame, (int(p["x"] * scale_x), int(p["y"] * scale_y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(p["x"] * scale_x), int(p["y"] * scale_y)), 5, color, 1, cv2.LINE_AA)
    wrist = pts[0]
    cv2.putText(
        frame,
        label,
        (int(wrist["x"] * scale_x) + 8, int(wrist["y"] * scale_y) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def estimate_relative_camera_path(prev_gray: np.ndarray | None, gray: np.ndarray, state: dict, mode: str) -> tuple[np.ndarray | None, dict]:
    if mode == "static":
        state["positions"].append(state["positions"][-1].copy())
        return gray, state
    if prev_gray is None:
        return gray, state
    orb = state["orb"]
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is None or des2 is None or len(kp1) < 16 or len(kp2) < 16:
        state["lost"] += 1
        state["positions"].append(state["positions"][-1].copy())
        return gray, state

    matches = state["matcher"].knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 16:
        state["lost"] += 1
        state["positions"].append(state["positions"][-1].copy())
        return gray, state

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    focal = state["focal"]
    pp = state["pp"]
    e_mat, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    inliers = int(mask.sum()) if mask is not None else 0
    if e_mat is None or inliers < 12:
        state["lost"] += 1
        state["positions"].append(state["positions"][-1].copy())
        return gray, state

    _, r, t, _ = cv2.recoverPose(e_mat, pts1, pts2, focal=focal, pp=pp)
    state["rotation"] = state["rotation"] @ r
    step = state["rotation"] @ t.reshape(3)
    norm = float(np.linalg.norm(step))
    if norm > 0:
        step = step / norm
    state["positions"].append(state["positions"][-1] + step * 0.03)
    state["vo_inliers"].append(inliers)
    state["vo_matches"].append(len(good))
    return gray, state


def render_3d_panel(
    frame_idx: int,
    info: RenderInfo,
    camera_positions: list[np.ndarray],
    hands_world: list[dict],
    caption: str,
) -> np.ndarray:
    panel = np.full((info.height, info.width, 3), 245, dtype=np.uint8)
    origin = np.array([info.width * 0.5, info.height * 0.56], dtype=float)
    scale = min(info.width, info.height) * 0.34

    def project(point: np.ndarray) -> tuple[int, int]:
        x = origin[0] + (point[0] - point[2] * 0.35) * scale
        y = origin[1] - (point[1] + point[2] * 0.22) * scale
        return int(np.clip(x, 0, info.width - 1)), int(np.clip(y, 0, info.height - 1))

    cv2.putText(panel, "Relative 3D proxy", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    for i, line in enumerate(wrap_text(f"frame {frame_idx} | scale relative | {caption}", cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2, info.width - 48, 2)):
        cv2.putText(panel, line, (24, 74 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.line(panel, (80, int(origin[1])), (info.width - 80, int(origin[1])), (210, 210, 210), 1)
    cv2.line(panel, (int(origin[0]), 110), (int(origin[0]), info.height - 80), (210, 210, 210), 1)

    pos = np.asarray(camera_positions, dtype=float)
    if len(pos) > 1:
        pts = [project(p) for p in pos]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(panel, a, b, (150, 90, 40), 3, cv2.LINE_AA)
    cv2.circle(panel, project(pos[-1]), 10, (40, 40, 210), -1, cv2.LINE_AA)
    cv2.putText(panel, "camera", tuple(np.array(project(pos[-1])) + np.array([12, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 210), 2, cv2.LINE_AA)

    for hand in hands_world:
        pts = np.asarray([[p["wx"], p["wy"], p["wz"]] for p in hand["points"]], dtype=float)
        color = (70, 160, 70) if hand["label"] == "Left" else (40, 130, 230)
        for a, b in HAND_EDGES:
            cv2.line(panel, project(pts[a]), project(pts[b]), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(panel, project(p), 3, color, -1, cv2.LINE_AA)
    return panel


def landmarks_to_points(landmarks, world_landmarks, width: int, height: int, camera_pos: np.ndarray) -> list[dict]:
    points = []
    for idx, lm in enumerate(landmarks.landmark):
        wl = world_landmarks.landmark[idx] if world_landmarks else None
        wx = float(camera_pos[0] + (wl.x if wl else (lm.x - 0.5) * 0.3))
        wy = float(camera_pos[1] + (wl.y if wl else (lm.y - 0.5) * 0.3))
        wz = float(camera_pos[2] + (wl.z if wl else lm.z * 0.3) - 0.45)
        points.append(
            {
                "x": float(lm.x * width),
                "y": float(lm.y * height),
                "z": float(lm.z),
                "wx": wx,
                "wy": wy,
                "wz": wz,
            }
        )
    return points


def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    clip = Path(args.clip)
    json_path = clip.with_suffix(".json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mano_root = Path(args.mano_root)
    manotorch_root = Path(args.manotorch_root)
    mano_templates = load_mano_template(mano_root, manotorch_root)
    write_template_obj(output_dir / "mano_right_template.obj", mano_templates["right"]["verts"], mano_templates["right"]["faces"])
    write_template_obj(output_dir / "mano_left_template.obj", mano_templates["left"]["verts"], mano_templates["left"]["faces"])
    actions = load_actions(json_path)
    cap, info = open_video(clip)
    stride = max(1, round(info.fps / args.output_fps))
    out_fps = info.fps / stride
    render_width = min(info.width, int(args.render_width))
    render_height = int(round(render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    render = RenderInfo(
        width=render_width,
        height=render_height,
        scale_x=render_width / info.width,
        scale_y=render_height / info.height,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(output_dir / "overlay.mp4"), fourcc, out_fps, (render.width, render.height))
    recon_writer = cv2.VideoWriter(str(output_dir / "reconstruction_3d.mp4"), fourcc, out_fps, (render.width, render.height))
    side_writer = cv2.VideoWriter(str(output_dir / "side_by_side.mp4"), fourcc, out_fps, (render.width * 2, render.height))
    if not overlay_writer.isOpened() or not recon_writer.isOpened() or not side_writer.isOpened():
        raise RuntimeError("failed to open one or more video writers")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.45, min_tracking_confidence=0.45)

    vo_state = {
        "orb": cv2.ORB_create(nfeatures=450),
        "matcher": cv2.BFMatcher(cv2.NORM_HAMMING),
        "focal": info.width * 0.8,
        "pp": (info.width / 2.0, info.height / 2.0),
        "rotation": np.eye(3),
        "positions": [np.zeros(3)],
        "lost": 0,
        "vo_matches": [],
        "vo_inliers": [],
    }

    annotations = []
    detection_frames = 0
    processed = 0
    prev_gray = None
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        if args.max_frames and processed >= args.max_frames:
            break

        small = cv2.resize(frame, (640, int(640 * info.height / info.width)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev_gray, vo_state = estimate_relative_camera_path(prev_gray, gray, vo_state, args.camera_proxy)
        camera_pos = vo_state["positions"][-1]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        caption = caption_for_frame(actions, frame_idx)
        overlay = cv2.resize(frame, (render.width, render.height), interpolation=cv2.INTER_AREA)
        hands_out = []

        if result.multi_hand_landmarks:
            detection_frames += 1
            handedness = result.multi_handedness or []
            world = result.multi_hand_world_landmarks or [None] * len(result.multi_hand_landmarks)
            for i, lm in enumerate(result.multi_hand_landmarks):
                label = "Unknown"
                score = 0.0
                if i < len(handedness):
                    cls = handedness[i].classification[0]
                    label = cls.label
                    score = float(cls.score)
                pts = landmarks_to_points(lm, world[i] if i < len(world) else None, info.width, info.height, camera_pos)
                draw_hand(overlay, pts, label, render.scale_x, render.scale_y)
                hands_out.append({"label": label, "score": score, "points": pts})

        status = "proposal_keypoints"
        cv2.putText(
            overlay,
            f"v1 {status} | frame {frame_idx} | hands {len(hands_out)} | scale relative",
            (14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        draw_caption(overlay, caption)
        panel = render_3d_panel(frame_idx, render, vo_state["positions"], hands_out, caption)

        overlay_writer.write(overlay)
        recon_writer.write(panel)
        side_writer.write(np.concatenate([overlay, panel], axis=1))

        annotations.append(
            {
                "frame_idx": frame_idx,
                "time_s": frame_idx / info.fps,
                "caption": caption,
                "camera": {
                    "position": [float(x) for x in camera_pos],
                    "scale_status": "relative",
                    "source": f"{args.camera_proxy}_proxy",
                },
                "hands": hands_out,
                "mano": {
                    "status": "template_available_pose_not_estimated",
                    "asset_root": str(mano_root),
                    "reason": "MANO layer is initialized, but this runner does not regress per-frame MANO pose from RGB.",
                },
            }
        )
        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            elapsed = time.time() - started_at
            print(f"processed={processed} frame_idx={frame_idx} elapsed_s={elapsed:.1f}", flush=True)

    cap.release()
    hands.close()
    overlay_writer.release()
    recon_writer.release()
    side_writer.release()

    if processed == 0:
        raise RuntimeError("no frames processed")

    detection_rate = detection_frames / processed
    qc = {
        "clip": str(clip),
        "json": str(json_path),
        "fps": info.fps,
        "output_fps": out_fps,
        "resolution": [info.width, info.height],
        "render_resolution": [render.width, render.height],
        "input_frame_count": info.frame_count,
        "processed_frames": processed,
        "hand_detection_frames": detection_frames,
        "hand_detection_rate": detection_rate,
        "camera_pose_status": f"{args.camera_proxy}_proxy",
        "camera_pose_warning": "Camera path is a placeholder visualization proxy, not production SLAM.",
        "mano_status": "template_available_pose_not_estimated",
        "mano_assets": {
            "root": str(mano_root),
            "right_vertices": int(mano_templates["right"]["verts"].shape[0]),
            "right_faces": int(mano_templates["right"]["faces"].shape[0]),
            "left_vertices": int(mano_templates["left"]["verts"].shape[0]),
            "left_faces": int(mano_templates["left"]["faces"].shape[0]),
            "right_template_obj": str(output_dir / "mano_right_template.obj"),
            "left_template_obj": str(output_dir / "mano_left_template.obj"),
        },
        "object_status": "not_run",
        "vo_lost_steps": vo_state["lost"],
        "vo_match_median": float(np.median(vo_state["vo_matches"])) if vo_state["vo_matches"] else 0.0,
        "vo_inlier_median": float(np.median(vo_state["vo_inliers"])) if vo_state["vo_inliers"] else 0.0,
        "deliverables": {
            "overlay": str(output_dir / "overlay.mp4"),
            "reconstruction_3d": str(output_dir / "reconstruction_3d.mp4"),
            "side_by_side": str(output_dir / "side_by_side.mp4"),
            "annotations": str(output_dir / "annotations.json"),
            "qc": str(output_dir / "qc.json"),
        },
        "quality_decision": "partial" if detection_rate < 0.6 else "proposal_pass",
        "quality_notes": [
            "MediaPipe keypoints are proposal annotations only; MANO mesh is required for final hand deliverable.",
            "MANO assets and manotorch layer are present; per-frame MANO pose regression remains a HaMeR/WiLoR integration task.",
            "Camera trajectory is relative-scale VO proxy; production v1 must replace it with MASt3R-SLAM/DROID-SLAM/DPVO.",
            "Caption source is existing action JSON.",
        ],
    }
    (output_dir / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--output-dir", default="outputs/examples/tomato_v1")
    parser.add_argument("--output-fps", type=float, default=3.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--camera-proxy", choices=["static", "orb"], default="static")
    parser.add_argument("--mano-root", default=str(DEFAULT_MANO_ROOT))
    parser.add_argument("--manotorch-root", default=str(DEFAULT_MANOTORCH_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
