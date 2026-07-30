#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import einops
import numpy as np
import torch


LIGHT_BLUE = (32, 208, 234)
LIGHT_RED = (250, 120, 126)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(float)
    z = np.clip(points[:, 2], 1e-6, None)
    return np.c_[fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy]


def source_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def frames_by_index(annotations: dict) -> dict[int, dict]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations must contain frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def observed_hand(frame: dict, side: str) -> dict | None:
    hands = [
        hand
        for hand in frame.get("hands", [])
        if str(hand.get("side", "")).lower() == side and bool(hand.get("measurement_available", False))
    ]
    if not hands:
        return None
    return max(hands, key=lambda hand: float(hand.get("detector_score", 0.0)))


def intrinsics_for_frame(frame: dict, explicit_intrinsics: np.ndarray | None) -> np.ndarray:
    if explicit_intrinsics is not None:
        return explicit_intrinsics.astype(float)
    for hand in frame.get("hands", []):
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
        if intr.shape == (4,):
            return intr
    raise RuntimeError("no intrinsics found; pass --intrinsics fx fy cx cy")


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = min(max(x0, 0.0), float(width - 1))
    x1 = min(max(x1, 0.0), float(width - 1))
    y0 = min(max(y0, 0.0), float(height - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid bbox after clipping: {box}")
    return [x0, y0, x1, y1]


def make_bbox_json(frames: list[dict], width: int, height: int) -> dict:
    boxes = {"right": {}, "left": {}}
    missing: list[dict] = []
    for local_i, frame in enumerate(frames):
        for side in ("right", "left"):
            hand = observed_hand(frame, side)
            if hand is None or "bbox_xyxy" not in hand:
                missing.append({"frame_idx": int(frame["frame_idx"]), "side": side})
                continue
            boxes[side][str(local_i)] = clip_box(hand["bbox_xyxy"], width, height)
    if missing:
        raise RuntimeError(f"OmniHands paired-hand export needs measured boxes for both hands; missing={missing[:20]}")
    return boxes


def write_clip(video: Path, frame_indices: list[int], output_video: Path) -> tuple[float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create video: {output_video}")
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_idx}")
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return fps, (width, height)


def load_omnihands(root: Path, checkpoint: Path, config: Path, gpu: int):
    sys.path.insert(0, str(root))
    from hands_4d.models import load_from_ckpt

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_from_ckpt(str(checkpoint), str(config))
    model = model.to(device).eval()
    return model, model_cfg, device


def gen_sequences(frame_count: int, seq_len: int, gap: int) -> np.ndarray:
    sequences = []
    for frame_idx in range(frame_count):
        seq = []
        start = frame_idx - (seq_len // 2) * gap
        for offset in range(seq_len):
            cur = start + offset * gap
            seq.append(min(max(cur, 0), frame_count - 1))
        sequences.append(seq)
    return np.asarray(sequences, dtype=int)


def run_omnihands_model(
    root: Path,
    model,
    model_cfg,
    device,
    clip_video: Path,
    bbox_json: Path,
    batch_size: int,
    token_batch_size: int,
) -> dict:
    sys.path.insert(0, str(root))
    from hands_4d.datasets.vitdet_dataset import HandToken_Sequence, ViTDetInterDataset_Batch, ViTDetInterDataset_Sequence
    from hands_4d.utils import recursive_to

    bboxes = load_json(bbox_json)
    cap = cv2.VideoCapture(str(clip_video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open clip video: {clip_video}")
    imgs = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            imgs.append(torch.as_tensor(frame).cpu())
    finally:
        cap.release()
    if not imgs:
        raise RuntimeError("OmniHands clip has no frames")

    frame_count = len(imgs)
    sequences = gen_sequences(frame_count, int(model_cfg.MODEL.SEQ_LEN), 10)
    dataset_single = ViTDetInterDataset_Batch(model_cfg, imgs, bboxes, rescale_factor=2.0)
    dataset_seq = ViTDetInterDataset_Sequence(model_cfg, imgs, bboxes, sequences, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset_single, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    hand_tokens = []
    bbox_right = []
    bbox_left = []
    with torch.no_grad():
        for batch in dataloader:
            batch = recursive_to(batch, device)
            tokens = model.inference_token_forward(batch)
            tokens = einops.rearrange(tokens, "(b s) c -> b s c", s=2)
            hand_tokens.append(tokens.detach().cpu())
            bbox_right.append(batch["bbox_right"].detach().cpu().numpy())
            bbox_left.append(batch["bbox_left"].detach().cpu().numpy())
    hand_tokens_tensor = torch.cat(hand_tokens, dim=0)
    token_dataset = HandToken_Sequence(hand_tokens_tensor, sequences)
    token_loader = torch.utils.data.DataLoader(token_dataset, batch_size=token_batch_size, shuffle=False, num_workers=2, drop_last=False)
    seq_loader = torch.utils.data.DataLoader(dataset_seq, batch_size=token_batch_size, shuffle=False, num_workers=2, drop_last=False)

    verts_right = []
    verts_left = []
    joints_right = []
    joints_left = []
    cams_right = []
    cams_left = []
    joints2d_right = []
    joints2d_left = []
    with torch.no_grad():
        for tokens, batch in zip(token_loader, seq_loader):
            batch = recursive_to(batch, device)
            tokens = tokens.to(device)
            tokens = einops.rearrange(tokens, "b t s c -> (b t s) c")
            output = model.inference_temp_forward(tokens, batch)
            verts_right.append(output["verts3d_world_right"].detach().cpu().numpy())
            verts_left.append(output["verts3d_world_left"].detach().cpu().numpy())
            joints_right.append(output["joints3d_world_right"].detach().cpu().numpy())
            joints_left.append(output["joints3d_world_left"].detach().cpu().numpy())
            cams_right.append(output["cam_aligned_right"].detach().cpu().numpy())
            cams_left.append(output["cam_aligned_left"].detach().cpu().numpy())
            joints2d_right.append(output["joints2d_world_right"].detach().cpu().numpy())
            joints2d_left.append(output["joints2d_world_left"].detach().cpu().numpy())

    return {
        "verts_right": np.concatenate(verts_right, axis=0),
        "verts_left": np.concatenate(verts_left, axis=0),
        "joints_right": np.concatenate(joints_right, axis=0),
        "joints_left": np.concatenate(joints_left, axis=0),
        "cams_right": np.concatenate(cams_right, axis=0),
        "cams_left": np.concatenate(cams_left, axis=0),
        "joints2d_right": np.concatenate(joints2d_right, axis=0),
        "joints2d_left": np.concatenate(joints2d_left, axis=0),
        "bbox_right": np.concatenate(bbox_right, axis=0),
        "bbox_left": np.concatenate(bbox_left, axis=0),
    }


def hand_from_output(
    *,
    side: str,
    local_i: int,
    frame: dict,
    vertices_model: np.ndarray,
    joints_model: np.ndarray,
    cam_t: np.ndarray,
    model_joints2d: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[dict, dict]:
    vertices_camera = vertices_model + cam_t[None, :]
    joints_camera = joints_model + cam_t[None, :]
    if np.any(vertices_camera[:, 2] <= 0.0) or np.any(joints_camera[:, 2] <= 0.0):
        raise RuntimeError(f"{side} OmniHands has non-positive camera depth")
    projected = project(joints_camera, intrinsics)
    observed = observed_hand(frame, side)
    if observed is None:
        raise RuntimeError(f"missing observed {side} hand")
    raw2d = np.asarray(observed.get("joints2d_raw", []), dtype=float)
    if raw2d.shape != (21, 2):
        raw2d = np.asarray(model_joints2d, dtype=float)
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    vertices_world = source_to_world(vertices_camera, T_world_camera)
    joints_world = source_to_world(joints_camera, T_world_camera)
    hand = {
        "backend": "OmniHands",
        "side": side,
        "measurement_available": True,
        "detector_score": float(observed.get("detector_score", 0.0)),
        "filter_status": "omnihands_camera_aligned_diagnostic",
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "bbox_xyxy": [float(v) for v in observed["bbox_xyxy"]],
        "cam_t": cam_t.astype(float).tolist(),
        "joints3d_camera": joints_model.astype(float).tolist(),
        "vertices_camera": vertices_model.astype(float).tolist(),
        "joints3d_source_camera_m": joints_camera.astype(float).tolist(),
        "vertices_source_camera_m": vertices_camera.astype(float).tolist(),
        "joints3d_world_m": joints_world.astype(float).tolist(),
        "vertices_world_m": vertices_world.astype(float).tolist(),
        "joints2d": projected.astype(float).tolist(),
        "joints2d_raw": raw2d.astype(float).tolist(),
        "omnihands_joints2d_world": np.asarray(model_joints2d, dtype=float).tolist(),
        "projection_residual_to_measurement_px": {
            "median": float(np.median(reproj)),
            "p95": float(np.percentile(reproj, 95.0)),
        },
        "world_coordinate_status": "omnihands_camera_aligned_transformed_by_existing_annotation_camera_pose",
        "mano_surface_status": "full_vertices",
        "mano_vertex_count": int(len(vertices_camera)),
    }
    row = {
        "frame_idx": int(frame["frame_idx"]),
        "local_frame_idx": int(local_i),
        "side": side,
        "detector_score": float(hand["detector_score"]),
        "joint_reprojection_px_median": float(np.median(reproj)),
        "joint_reprojection_px_p95": float(np.percentile(reproj, 95.0)),
        "median_camera_depth_m": float(np.median(joints_camera[:, 2])),
        "vertex_count": int(len(vertices_camera)),
    }
    return hand, row


def render_overlay(video: Path, frame_indices: list[int], output: dict, output_video: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create overlay video: {output_video}")
    colors = {"left": LIGHT_RED, "right": LIGHT_BLUE}
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_idx}")
            out_frame = frames_by_index(output)[frame_idx]
            draw = bgr.copy()
            for hand in out_frame.get("hands", []):
                pts = np.asarray(hand.get("joints2d", []), dtype=float)
                if pts.shape != (21, 2):
                    continue
                color = colors.get(str(hand.get("side")), (255, 255, 255))
                for x, y in pts:
                    cv2.circle(draw, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
            cv2.putText(draw, "OmniHands diagnostic", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(draw)
    finally:
        writer.release()
        cap.release()


def summarize(rows: list[dict], key: str) -> dict:
    vals = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=float)
    if len(vals) == 0:
        return {"count": 0}
    return {
        "count": int(len(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
        "max": float(np.max(vals)),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frame_map = frames_by_index(annotations)
    frame_indices = list(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))
    frames = []
    for frame_idx in frame_indices:
        frame = frame_map.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"missing annotation frame {frame_idx}")
        frames.append(frame)
    explicit_intrinsics = None if args.intrinsics is None else np.asarray(args.intrinsics, dtype=float)
    if explicit_intrinsics is not None and explicit_intrinsics.shape != (4,):
        raise RuntimeError("--intrinsics must contain fx fy cx cy")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_video = args.output_dir / f"omnihands_input_{args.frame_start}_{args.frame_end}.mp4"
    fps, (width, height) = write_clip(args.video, frame_indices, clip_video)
    bbox_json = args.output_dir / "bbox.json"
    save_json(bbox_json, make_bbox_json(frames, width, height))

    model, model_cfg, device = load_omnihands(args.omnihands_root, args.checkpoint, args.config, args.gpu)
    raw = run_omnihands_model(
        args.omnihands_root,
        model,
        model_cfg,
        device,
        clip_video,
        bbox_json,
        args.batch_size,
        args.token_batch_size,
    )

    output = copy.deepcopy(annotations)
    output_by_idx = frames_by_index(output)
    rows: list[dict] = []
    raw_by_frame: dict[str, dict] = {}
    for local_i, frame_idx in enumerate(frame_indices):
        frame = frame_map[frame_idx]
        out_frame = output_by_idx[frame_idx]
        intr = intrinsics_for_frame(frame, explicit_intrinsics)
        out_frame["hands"] = []
        frame_raw: dict[str, dict] = {}
        for side in ("right", "left"):
            prefix = "right" if side == "right" else "left"
            try:
                hand, row = hand_from_output(
                    side=side,
                    local_i=local_i,
                    frame=frame,
                    vertices_model=np.asarray(raw[f"verts_{prefix}"][local_i], dtype=float),
                    joints_model=np.asarray(raw[f"joints_{prefix}"][local_i], dtype=float),
                    cam_t=np.asarray(raw[f"cams_{prefix}"][local_i], dtype=float),
                    model_joints2d=np.asarray(raw[f"joints2d_{prefix}"][local_i], dtype=float),
                    intrinsics=intr,
                )
            except Exception as exc:
                raise RuntimeError(f"frame {frame_idx} {side}: {exc}") from exc
            out_frame["hands"].append(hand)
            rows.append(row)
            frame_raw[side] = {
                "vertices_source_camera_m": hand["vertices_source_camera_m"],
                "joints3d_source_camera_m": hand["joints3d_source_camera_m"],
                "cam_t": hand["cam_t"],
                "joints2d": hand["joints2d"],
            }
        raw_by_frame[str(frame_idx)] = frame_raw

    output_annotations = args.output_dir / "annotations_omnihands.json"
    output_npz = args.output_dir / "omnihands_raw.npz"
    output_qc = args.output_dir / "qc_omnihands.json"
    save_json(output_annotations, output)
    np.savez_compressed(
        output_npz,
        frame_idx=np.asarray(frame_indices, dtype=int),
        raw_by_frame_json=np.asarray(json.dumps(raw_by_frame), dtype=object),
        verts_right=raw["verts_right"],
        verts_left=raw["verts_left"],
        joints_right=raw["joints_right"],
        joints_left=raw["joints_left"],
        cams_right=raw["cams_right"],
        cams_left=raw["cams_left"],
        joints2d_right=raw["joints2d_right"],
        joints2d_left=raw["joints2d_left"],
        fps=np.asarray(fps, dtype=float),
        image_size=np.asarray([width, height], dtype=int),
    )
    overlay_video = args.output_dir / "omnihands_overlay.mp4"
    render_overlay(args.video, frame_indices, output, overlay_video)

    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "video": str(args.video),
        "annotations": str(args.annotations),
        "omnihands_root": str(args.omnihands_root),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "output_annotations": str(output_annotations),
        "output_npz": str(output_npz),
        "overlay_video": str(overlay_video),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "frames": int(len(frame_indices)),
        "hand_rows": int(len(rows)),
        "video_info": {"fps": fps, "width": width, "height": height},
        "summary": {
            "joint_reprojection_px": summarize(rows, "joint_reprojection_px_median"),
            "median_camera_depth_m": summarize(rows, "median_camera_depth_m"),
        },
        "rows_preview": rows[:180],
    }
    save_json(output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--omnihands-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--intrinsics", nargs=4, type=float)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-batch-size", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
