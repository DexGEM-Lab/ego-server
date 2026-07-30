#!/usr/bin/env python3
"""Generate overlay video for an atomic algorithm result.

Examples:
  python scripts/generate_algorithm_overlay.py --type depth --run-root ... --output-dir ... --data-path depth.npz
  python scripts/generate_algorithm_overlay.py --type segmentation --run-root ... --output-dir ... --data-path mask_dir
  python scripts/generate_algorithm_overlay.py --type detection --run-root ... --output-dir ... --data-path bbox_proposals.json
  python scripts/generate_algorithm_overlay.py --type hand --run-root ... --output-dir ... --data-path hands.json
  python scripts/generate_algorithm_overlay.py --type report --run-root ... --output-dir ... --data-path report.json
  python scripts/generate_algorithm_overlay.py --type heightfield --run-root ... --output-dir ... --data-path reconstruction_dir
  python scripts/generate_algorithm_overlay.py --type mesh_candidate --run-root ... --output-dir ... --data-path mesh_completion_report.json
  python scripts/generate_algorithm_overlay.py --type camera --run-root ... --output-dir ... --data-path depthpro_as_droid.npz
  python scripts/generate_algorithm_overlay.py --type prompts --run-root ... --output-dir ... --data-path object_point_prompts_v21.json
  python scripts/generate_algorithm_overlay.py --type visible_geometry --run-root ... --output-dir ... --data-path annotations_v19_visible_geometry.json
  python scripts/generate_algorithm_overlay.py --type pose_mesh --run-root ... --output-dir ... --data-path v19_rigid_object_pose_graph_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def load_manifest(run_root: Path) -> dict[str, Any]:
    return load_json(run_root / "input/raw_frame_manifest/manifest.json")


def get_source_frames(run_root: Path, manifest: dict[str, Any]) -> list[tuple[int, str]]:
    """Return list of (frame_idx, source_image_path)."""
    repo = Path.cwd()
    frames: list[tuple[int, str]] = []
    for fm in manifest.get("frames", []):
        if not isinstance(fm, dict):
            continue
        fidx = int(fm.get("frame_idx", fm.get("index", 0)))
        candidates = [
            run_root / f"input/source_frame_manifest/rgb/{fidx:06d}.jpg",
            run_root / f"input/raw_frame_manifest/rgb/{fidx:06d}.jpg",
        ]
        raw_rgb = fm.get("rgb") or fm.get("raw_frame_path")
        if raw_rgb:
            raw_path = Path(str(raw_rgb))
            candidates.append(raw_path if raw_path.is_absolute() else repo / raw_path)
            candidates.append(run_root / raw_path)
        src_path = next((path for path in candidates if path.exists()), None)
        if src_path is not None:
            frames.append((fidx, str(src_path)))
    if not frames:
        raise ContractError(f"no_source_frames_found: {run_root}")
    return frames


def encode_frames(frame_dir: Path, out_path: Path, fps: float) -> Path:
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def draw_text_block(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 24), scale: float = 0.55) -> None:
    if not lines:
        return
    x, y = origin
    line_h = int(23 * scale / 0.55)
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines]
    box_w = min(max(widths, default=0) + 18, img.shape[1] - x - 4)
    box_h = line_h * len(lines) + 12
    cv2.rectangle(img, (x - 6, max(0, y - line_h)), (x - 6 + box_w, y - line_h + box_h), (0, 0, 0), -1)
    cv2.addWeighted(img, 0.86, img, 0.14, 0, dst=img)
    for i, line in enumerate(lines):
        cv2.putText(img, line[:110], (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_timeline(img: np.ndarray, frame_idx: int, frame_count: int, active: bool = False, color: tuple[int, int, int] = (0, 220, 255)) -> None:
    h, w = img.shape[:2]
    x0, x1 = 20, w - 20
    y = h - 22
    cv2.line(img, (x0, y), (x1, y), (80, 80, 80), 4)
    if frame_count > 1:
        x = int(x0 + (x1 - x0) * np.clip(frame_idx / float(frame_count - 1), 0.0, 1.0))
    else:
        x = x0
    cv2.circle(img, (x, y), 7 if active else 5, color if active else (220, 220, 220), -1)


def normalize_depth_for_color(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0.1)]
    if len(valid) > 0:
        d_min, d_max = np.percentile(valid, [2, 98])
        d_norm = np.clip((depth - d_min) / max(float(d_max - d_min), 0.01), 0, 1)
    else:
        d_norm = np.zeros_like(depth, dtype=np.float32)
    color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[~np.isfinite(depth) | (depth < 0.1)] = 0
    return color


def render_depth_overlay(frames: list[tuple[int, str]], depth_data_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    depth_data = np.load(depth_data_path)
    if "depth" in depth_data:
        depth = depth_data["depth"]
    elif "depths" in depth_data:
        depth = depth_data["depths"]
    elif "relative_inverse_depth" in depth_data:
        rel = np.asarray(depth_data["relative_inverse_depth"], dtype=np.float32)
        depth = np.zeros_like(rel, dtype=np.float32)
        valid = np.isfinite(rel) & (rel > 0)
        depth[valid] = 1.0 / np.maximum(rel[valid], 1e-6)
    else:
        raise ContractError(f"depth_npz_missing_depth_array: {depth_data_path}")
    fidxs = depth_data["frame_idx"] if "frame_idx" in depth_data else np.arange(depth.shape[0])
    fidx_to_i = {int(f): i for i, f in enumerate(fidxs)}
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        if fidx in fidx_to_i:
            d = np.asarray(depth[fidx_to_i[fidx]], dtype=np.float32)
            d_resized = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) if d.shape[:2] != img.shape[:2] else d
            d_color = normalize_depth_for_color(d_resized)
            overlay = cv2.addWeighted(img, 0.4, d_color, 0.6, 0)
        else:
            overlay = img
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_depth_overlay_frames_written")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def resolve_mask_dir(mask_path: Path) -> Path:
    if mask_path.is_dir():
        return mask_path
    report = load_json(mask_path)
    method = str(report.get("method", ""))
    if "groundingdino" in method.lower() or "grounding_dino" in method.lower():
        raise ContractError(f"groundingdino_seeded_segmentation_overlay_disabled_for_v21: {mask_path}")
    candidates = []
    for key in ["mask_dir", "masks", "output_mask_dir"]:
        if report.get(key):
            candidates.append(Path(str(report[key])))
    candidates.extend([mask_path.parent / "mask", mask_path.parent / "masks", mask_path.parent / "sam2_masks"])
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise ContractError(f"segmentation_report_has_no_resolvable_mask_dir: {mask_path}")


def render_segmentation_overlay(frames: list[tuple[int, str]], mask_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    mask_dir = resolve_mask_dir(mask_path)
    mask_files = {int(Path(f).stem): f for f in sorted(glob.glob(str(mask_dir / "*.png")))}
    if not mask_files:
        raise ContractError(f"mask_dir_has_no_png_masks: {mask_dir}")
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    mask_frames_drawn = 0
    no_mask_frames = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        if fidx in mask_files:
            mask = cv2.imread(mask_files[fidx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                blended = img
            else:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay = img.copy()
                overlay[mask > 127] = [0, 0, 255]
                blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
                contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
                cv2.putText(blended, f"mask: {int((mask > 127).sum())}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                mask_frames_drawn += 1
        else:
            blended = img
            no_mask_frames += 1
            cv2.putText(blended, "no mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_segmentation_overlay_frames_written")
    qc = {
        "schema": "segmentation_overlay_qc.v1",
        "overlay_type": "mask_on_source_rgb",
        "data_path": str(mask_path),
        "mask_dir": str(mask_dir),
        "frame_count": int(written),
        "mask_frames_drawn": int(mask_frames_drawn),
        "no_mask_frames": int(no_mask_frames),
        "claim_scope": "Draws active segmentation masks on source RGB frames. This visualizes mask evidence only; it is not geometry, pose, contact, or visual acceptance.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def render_detection_overlay(frames: list[tuple[int, str]], detections_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    detections = load_json(detections_path)
    method = str(detections.get("method", "bbox_proposals"))
    if "groundingdino" in method.lower() or "grounding_dino" in method.lower():
        raise ContractError(f"groundingdino_detection_overlay_disabled_for_v21: {detections_path}")
    det_by_frame = {int(f["frame_idx"]): f.get("detections", []) for f in detections.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        for det in det_by_frame.get(fidx, []):
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox_xyxy", det.get("bbox", []))
            score = float(det.get("score", det.get("owlv2_score", 0.0)) or 0.0)
            label = str(det.get("label", det.get("text_label", "bbox")))
            if isinstance(bbox, list) and len(bbox) == 4 and score > 0.0:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {score:.2f}", (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_detection_overlay_frames_written")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def normalize_hand_frames(hand_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    hand_by_frame: dict[int, list[dict[str, Any]]] = {}
    frames_data = hand_data.get("frames", [])
    if isinstance(frames_data, list) and frames_data:
        for frame in frames_data:
            if not isinstance(frame, dict) or frame.get("frame_idx") is None:
                continue
            frame_idx = int(frame.get("local_frame_idx", frame["frame_idx"]))
            hand_by_frame[frame_idx] = list(frame.get("hands", frame.get("raw_hands", [])) or [])
    top_hands = hand_data.get("hands", [])
    if isinstance(top_hands, list) and top_hands:
        for hand in top_hands:
            if not isinstance(hand, dict) or hand.get("frame_idx") is None:
                continue
            hand_by_frame.setdefault(int(hand["frame_idx"]), []).append(hand)
    return hand_by_frame


def infer_run_root_from_hand_path(hand_data_path: Path) -> Path | None:
    for parent in hand_data_path.parents:
        if (parent / "measurements" / "hand_candidates").exists():
            return parent
    return None


def hand_overlay_coord_scale(hand_data_path: Path, hand_data: dict[str, Any]) -> tuple[float, float, str]:
    """Return scale from stored hand coordinates to source RGB overlay coordinates."""
    path_text = str(hand_data_path)
    needs_manifest_scale = (
        "/wilor_v21/" in path_text
        or "/wilor_v21_metric/" in path_text
        or str(hand_data.get("method")) == "v21_active_mano_optimizer"
    )
    if not needs_manifest_scale:
        return 1.0, 1.0, "source_rgb_coordinates"
    run_root = infer_run_root_from_hand_path(hand_data_path)
    if run_root is None:
        return 1.0, 1.0, "unknown_run_root_no_scaling"
    manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    if not manifest_path.exists():
        return 1.0, 1.0, "raw_manifest_missing_no_scaling"
    try:
        first = load_json(manifest_path).get("frames", [])[0]
        src_w = float(first.get("source_width") or first.get("manifest_width") or 0.0)
        src_h = float(first.get("source_height") or first.get("manifest_height") or 0.0)
        man_w = float(first.get("manifest_width") or src_w)
        man_h = float(first.get("manifest_height") or src_h)
    except Exception:
        return 1.0, 1.0, "raw_manifest_invalid_no_scaling"
    if src_w <= 0 or src_h <= 0 or man_w <= 0 or man_h <= 0:
        return 1.0, 1.0, "raw_manifest_bad_size_no_scaling"
    return src_w / man_w, src_h / man_h, "manifest_to_source_rgb_coordinates"


def scale_bbox_xyxy(bbox: Any, scale_xy: tuple[float, float]) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    sx, sy = scale_xy
    return [float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy]


def project_mano_joints(points: np.ndarray, intr: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intr[:4]]
    z = np.clip(points[:, 2], 1.0e-4, None)
    return np.column_stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy])


def augment_active_mano_render_joints(hand_data_path: Path, hand_data: dict[str, Any]) -> dict[str, Any]:
    if str(hand_data.get("method")) != "v21_active_mano_optimizer":
        return hand_data
    hands = hand_data.get("hands")
    if not isinstance(hands, list) or not hands:
        return hand_data
    run_root = infer_run_root_from_hand_path(hand_data_path)
    metric_path = run_root / "measurements" / "hand_candidates" / "wilor_v21_metric" / "wilor_metric_hands.json" if run_root else None
    if metric_path is None or not metric_path.exists():
        return hand_data
    metric = load_json(metric_path)
    metric_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in metric.get("frames", []) if isinstance(metric.get("frames"), list) else []:
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        for idx, row in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
            if isinstance(row, dict):
                metric_by_key[(int(frame["frame_idx"]), int(idx))] = row
    try:
        import os
        import sys
        import torch
        from scipy.spatial.transform import Rotation
        wilor_dir = Path(os.environ.get("WILOR_DIR", "/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model"))
        sys.path.insert(0, str(wilor_dir))
        from wilor.models.mano_wrapper import MANO
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MANO(model_path=str(wilor_dir / "mano_data"), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device).eval()
    except Exception as exc:
        hand_data["render_joint_generation_error"] = str(exc)
        return hand_data
    augmented = 0
    for row in hands:
        if not isinstance(row, dict):
            continue
        fidx = int(row.get("frame_idx", -1))
        hand_idx = int(row.get("hand_idx", 0))
        metric_row = metric_by_key.get((fidx, hand_idx))
        intr = metric_row.get("intrinsics_manifest") if isinstance(metric_row, dict) else None
        if not isinstance(intr, list) or len(intr) < 4:
            continue
        try:
            go = Rotation.from_rotvec(np.asarray(row["global_orient_rotvec"], dtype=float)).as_matrix().reshape(1, 1, 3, 3)
            hp = np.asarray(row["hand_pose_rotvec"], dtype=float).reshape(15, 3)
            hp_mat = np.asarray([Rotation.from_rotvec(v).as_matrix() for v in hp], dtype=np.float32).reshape(1, 15, 3, 3)
            betas = np.asarray(row.get("betas") or hand_data.get("betas_track"), dtype=np.float32).reshape(1, 10)
            scale = float(row.get("scale", hand_data.get("scale_track", 1.0)))
            trans = np.asarray(row["translation_m"], dtype=np.float32).reshape(3)
            with torch.no_grad():
                out = model(
                    global_orient=torch.tensor(go, dtype=torch.float32, device=device),
                    hand_pose=torch.tensor(hp_mat, dtype=torch.float32, device=device),
                    betas=torch.tensor(betas, dtype=torch.float32, device=device),
                    return_verts=True,
                    pose2rot=False,
                )
            joints = out.joints[0].detach().cpu().numpy().astype(np.float32) * scale + trans[None, :]
            row["optimized_joints3d_camera_m"] = joints.astype(float).tolist()
            row["optimized_joints2d"] = project_mano_joints(joints, intr).astype(float).tolist()
            if isinstance(metric_row, dict) and isinstance(metric_row.get("bbox_xyxy"), list):
                row.setdefault("bbox_xyxy", metric_row["bbox_xyxy"])
            augmented += 1
        except Exception as exc:
            row["render_joint_generation_error"] = str(exc)
    hand_data["render_joint_generation"] = {
        "method": "gpu_mano_forward_from_v21_active_mano_parameters",
        "device": str(device),
        "augmented_rows": int(augmented),
        "source_metric_hands": str(metric_path),
    }
    return hand_data


def valid_points2d(value: Any, img_shape: tuple[int, int, int], bbox: Any = None, scale_xy: tuple[float, float] = (1.0, 1.0)) -> np.ndarray | None:
    try:
        pts = np.asarray(value, dtype=float)
    except Exception:
        return None
    if pts.ndim != 2 or pts.shape[0] < 21 or pts.shape[1] < 2 or not np.isfinite(pts[:, :2]).all():
        return None
    pts = pts[:21, :2].copy()
    sx, sy = scale_xy
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    h, w = img_shape[:2]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if int(inside.sum()) < 12:
        return None
    scaled_bbox = scale_bbox_xyxy(bbox, scale_xy)
    if scaled_bbox is not None:
        x1, y1, x2, y2 = scaled_bbox
        bx = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)
        pc = np.nanmedian(pts, axis=0)
        diag = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
        # Reject points that are in-frame but clearly from a different coordinate convention.
        if float(np.linalg.norm(pc - bx)) > diag * 1.75:
            return None
    return pts


def project_hand_joints3d(hand: dict[str, Any], img_shape: tuple[int, int, int], scale_xy: tuple[float, float] = (1.0, 1.0)) -> np.ndarray | None:
    joints = hand.get("joints3d_source_camera_m") or hand.get("joints3d_camera_metric") or hand.get("joints3d_camera")
    if not isinstance(joints, list) or len(joints) != 21:
        return None
    try:
        pts = np.asarray(joints, dtype=float)
    except Exception:
        return None
    if pts.shape != (21, 3) or not np.isfinite(pts).all():
        return None
    if "joints3d_source_camera_m" not in hand:
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric") or hand.get("cam_t")
        if isinstance(cam_t, list) and len(cam_t) == 3:
            pts = pts + np.asarray(cam_t, dtype=float)[None, :]
    intr = hand.get("source_intrinsics") or hand.get("intrinsics_manifest")
    if isinstance(intr, list) and len(intr) >= 4:
        fx, fy, cx, cy = [float(v) for v in intr[:4]]
    else:
        h, w = img_shape[:2]
        fx = fy = max(w, h) * 1.2
        cx = w / 2.0
        cy = h / 2.0
    z = pts[:, 2]
    ok = z > 1.0e-4
    if int(ok.sum()) < 12:
        return None
    out = np.full((21, 2), np.nan, dtype=float)
    out[ok, 0] = fx * pts[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * pts[ok, 1] / z[ok] + cy
    return valid_points2d(out, img_shape, hand.get("bbox_xyxy"), scale_xy)


def hand_points2d(hand: dict[str, Any], img_shape: tuple[int, int, int], scale_xy: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray | None, str]:
    bbox = hand.get("bbox_xyxy")
    for key in ["keypoints", "keypoints2d", "joints2d_raw", "joints2d", "optimized_joints2d"]:
        pts = valid_points2d(hand.get(key), img_shape, bbox, scale_xy)
        if pts is not None:
            return pts, key
    pts = project_hand_joints3d(hand, img_shape, scale_xy)
    if pts is not None:
        return pts, "projected_3d_joints"
    return None, "none"


def draw_hand_skeleton(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    for a, b in HAND_EDGES:
        pa, pb = pts[a], pts[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(img, (int(round(pa[0])), int(round(pa[1]))), (int(round(pb[0])), int(round(pb[1]))), color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        if np.isfinite(pt).all():
            radius = 5 if i == 0 else 4
            cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)


def render_hand_overlay(frames: list[tuple[int, str]], hand_data_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    hand_data = augment_active_mano_render_joints(hand_data_path, load_json(hand_data_path))
    scale_x, scale_y, coord_policy = hand_overlay_coord_scale(hand_data_path, hand_data)
    scale_xy = (scale_x, scale_y)
    hand_by_frame = normalize_hand_frames(hand_data)
    optimizer = hand_data.get("optimizer", {}) if isinstance(hand_data.get("optimizer"), dict) else {}
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skeleton_rows = 0
    bbox_only_rows = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        hands = hand_by_frame.get(fidx, [])
        sources: list[str] = []
        for h in hands:
            if not isinstance(h, dict):
                continue
            side = str(h.get("side", h.get("hand_side", "right"))).lower()
            color = (0, 220, 255) if side == "right" else (255, 80, 80)
            bbox = h.get("bbox_xyxy") or h.get("wilor_bbox_xyxy")
            scaled_bbox = scale_bbox_xyxy(bbox, scale_xy)
            if scaled_bbox is not None:
                x1, y1, x2, y2 = [int(round(float(v))) for v in scaled_bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            pts, source = hand_points2d(h, img.shape, scale_xy)
            if pts is not None:
                draw_hand_skeleton(img, pts, color)
                skeleton_rows += 1
                sources.append(source)
            else:
                bbox_only_rows += 1
            score = float(h.get("detector_score", h.get("score", h.get("mean_score", 0.0))) or 0.0)
            label_xy = (int(scaled_bbox[0]) if scaled_bbox is not None else 10, max(int(scaled_bbox[1]) - 10 if scaled_bbox is not None else 30, 15))
            cv2.putText(img, f"{side[:1].upper()} {score:.2f}", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        lines = [f"Frame {fidx}: {len(hands)} hand rows; skeletons={len(sources)}"]
        if sources:
            lines.append("joints: " + ",".join(sorted(set(sources)))[:80])
        if bbox_only_rows and not sources:
            lines.append("bbox only: no renderable 2D/3D joints in this atom row")
        if optimizer:
            lines.append(f"optimizer success={optimizer.get('success')} nfev={optimizer.get('nfev')} rms={optimizer.get('residual_rms')}")
        draw_text_block(img, lines, (10, img.shape[0] - 58 if optimizer or bbox_only_rows else img.shape[0] - 32), 0.5)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_hand_overlay_frames_written")
    qc = {
        "schema": "hand_skeleton_overlay_qc.v1",
        "overlay_type": "hand_skeleton",
        "data_path": str(hand_data_path),
        "skeleton_rows_drawn": int(skeleton_rows),
        "bbox_only_rows": int(bbox_only_rows),
        "coordinate_policy": coord_policy,
        "coordinate_scale_xy": [float(scale_x), float(scale_y)],
        "claim_scope": "Draws 21-point hand skeletons when the atomic hand output contains renderable 2D joints or projectable 3D joints. Bboxes are retained only as context and for rows with no joint/surface data. WiLoR-family outputs are scaled from raw-manifest coordinates to source RGB coordinates before drawing.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def resolve_heightfield_depth_dir(data_path: Path) -> Path:
    candidates = []
    if data_path.is_dir():
        candidates.extend([data_path / "depth_full", data_path / "depth", data_path.parent / "dataset" / "depth_full", data_path.parent / "dataset" / "depth"])
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate
    raise ContractError(f"heightfield_depth_png_dir_not_found: {data_path}")


def resolve_heightfield_qc_report(data_path: Path) -> Path | None:
    candidates: list[Path] = []
    if data_path.is_file() and data_path.suffix.lower() == ".json":
        candidates.append(data_path)
    if data_path.is_dir():
        candidates.extend(sorted(data_path.glob("qc_heightfield*.json")))
        candidates.extend(sorted(data_path.glob("qc_*.json")))
        candidates.extend(sorted(data_path.glob("*.json")))
    candidates.extend([data_path.parent / "qc.json", data_path.parent / "report.json"])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".json":
            return candidate
    return None


def render_heightfield_overlay(frames: list[tuple[int, str]], data_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    try:
        depth_dir = resolve_heightfield_depth_dir(data_path)
    except ContractError:
        qc_report = resolve_heightfield_qc_report(data_path)
        if qc_report is not None:
            return render_report_overlay(frames, qc_report, output_dir, fps)
        raise
    depth_files = {int(p.stem): p for p in sorted(depth_dir.glob("*.png"))}
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        depth_path = depth_files.get(fidx)
        if depth_path is not None:
            d_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d_raw is not None:
                d = np.asarray(d_raw, dtype=np.float32)
                if d.max(initial=0) > 100:
                    d = d / 1000.0
                if d.shape[:2] != img.shape[:2]:
                    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                d_color = normalize_depth_for_color(d)
                img = cv2.addWeighted(img, 0.45, d_color, 0.55, 0)
                draw_text_block(img, [f"heightfield observed depth frame {fidx}", f"valid px: {int((d > 0.1).sum())}"], (10, 28), 0.52)
        else:
            draw_text_block(img, [f"heightfield observed: no depth png for frame {fidx}"], (10, 28), 0.52)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_heightfield_overlay_frames_written")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
            elif line.startswith("f "):
                idxs: list[int] = []
                for token in line.strip().split()[1:4]:
                    try:
                        idxs.append(int(token.split("/")[0]) - 1)
                    except ValueError:
                        idxs = []
                        break
                if len(idxs) == 3:
                    faces.append(idxs)
    if not verts:
        raise ContractError(f"obj_has_no_vertices: {path}")
    verts_arr = np.asarray(verts, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return verts_arr, faces_arr


def scale_intrinsics_for_image(intr: np.ndarray, from_hw: tuple[int, int] | None, to_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(intr, dtype=np.float64).copy()
    if from_hw is None or from_hw == to_hw:
        return out
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    sx = float(to_w) / max(1.0, float(from_w))
    sy = float(to_h) / max(1.0, float(from_h))
    out[0] *= sx
    out[2] *= sx
    out[1] *= sy
    out[3] *= sy
    return out


def project_points(points: np.ndarray, intr: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    out = np.full((len(points), 2), np.nan, dtype=np.float32)
    ok = z > 1.0e-4
    out[ok, 0] = float(intr[0]) * points[ok, 0] / z[ok] + float(intr[2])
    out[ok, 1] = float(intr[1]) * points[ok, 1] / z[ok] + float(intr[3])
    return out


def draw_projected_mesh(img: np.ndarray, pts2d: np.ndarray, z: np.ndarray, faces: np.ndarray, color: tuple[int, int, int] = (0, 230, 255)) -> int:
    h, w = img.shape[:2]
    finite = np.isfinite(pts2d).all(axis=1) & (z > 1.0e-4)
    inside = finite & (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
    if int(inside.sum()) == 0:
        return 0
    overlay = img.copy()
    pts_i = np.round(pts2d).astype(np.int32)
    face_stride = max(1, len(faces) // 1800) if len(faces) else 1
    for tri in faces[::face_stride]:
        if tri.max(initial=-1) >= len(pts_i) or tri.min(initial=0) < 0:
            continue
        if bool(inside[tri].all()):
            poly = pts_i[tri].reshape((-1, 1, 2))
            cv2.polylines(overlay, [poly], True, color, 1, cv2.LINE_AA)
    point_idx = np.flatnonzero(inside)
    point_stride = max(1, len(point_idx) // 6000)
    for idx in point_idx[::point_stride]:
        x, y = pts_i[idx]
        cv2.circle(overlay, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)
    if len(point_idx) >= 3:
        hull = cv2.convexHull(pts_i[point_idx].reshape((-1, 1, 2)))
        cv2.polylines(overlay, [hull], True, (0, 255, 80), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, dst=img)
    return int(inside.sum())


def render_prompt_overlay(frames: list[tuple[int, str]], prompt_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    prompts = load_json(prompt_path)
    rows = prompts.get("point_prompts")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"prompt_file_has_no_point_prompts: {prompt_path}")
    by_frame = {int(row["frame_idx"]): row for row in rows if isinstance(row, dict) and row.get("frame_idx") is not None}
    prompt_width = float(prompts.get("prompt_image_width") or 0.0)
    if prompt_width < 0.0:
        raise ContractError(f"invalid_prompt_image_width: {prompt_width}")
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    positive_count = 0
    negative_count = 0
    prompt_frames_drawn = 0
    scale_samples: list[tuple[float, float]] = []

    def scaled_xy(pt: dict[str, Any], img: np.ndarray) -> tuple[int, int]:
        sx = float(img.shape[1]) / prompt_width if prompt_width > 0.0 else 1.0
        sy = sx
        scale_samples.append((sx, sy))
        return int(round(float(pt["x"]) * sx)), int(round(float(pt["y"]) * sy))

    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        row = by_frame.get(int(fidx))
        active = row is not None and bool(row.get("target_visible", True))
        if row is not None:
            for pt in row.get("negative_points", []) if isinstance(row.get("negative_points"), list) else []:
                if isinstance(pt, dict) and pt.get("x") is not None and pt.get("y") is not None:
                    x, y = scaled_xy(pt, img)
                    cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
                    negative_count += 1
            for pt in row.get("positive_points", []) if isinstance(row.get("positive_points"), list) else []:
                if isinstance(pt, dict) and pt.get("x") is not None and pt.get("y") is not None:
                    x, y = scaled_xy(pt, img)
                    cv2.circle(img, (x, y), 7, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(img, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
                    positive_count += 1
            prompt_frames_drawn += 1
        draw_timeline(img, written, len(frames), active=active, color=(0, 255, 0))
        lines = [
            f"object point prompts: {prompts.get('track_id', prompt_path.parent.name)}",
            f"frame {fidx}: {'prompt frame' if row is not None else 'no prompt on this frame'}",
        ]
        if row is not None:
            lines.append(f"positive={len(row.get('positive_points', []) or [])} negative={len(row.get('negative_points', []) or [])}")
            lines.append(str(row.get("prompt_source", "prompt_source_unknown"))[:90])
        draw_text_block(img, lines, (10, 28), 0.52)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_prompt_overlay_frames_written")
    sx, sy = scale_samples[0] if scale_samples else (1.0, 1.0)
    qc = {
        "schema": "prompt_points_overlay_qc.v1",
        "overlay_type": "object_point_prompts_on_source_rgb",
        "data_path": str(prompt_path),
        "prompt_frame_count": int(len(by_frame)),
        "prompt_frames_drawn": int(prompt_frames_drawn),
        "positive_points_drawn": int(positive_count),
        "negative_points_drawn": int(negative_count),
        "prompt_image_width": None if prompt_width == 0.0 else float(prompt_width),
        "coordinate_policy": "prompt_image_width_scaled_to_source_rgb" if prompt_width > 0.0 and abs(sx - 1.0) > 1.0e-6 else "source_rgb_coordinates",
        "coordinate_scale_xy": [float(sx), float(sy)],
        "claim_scope": "Draws the agent/VLM-style positive and negative point prompts that seed inherited V19 SAM2 propagation. Points select the target instance; they are not masks, geometry, or pose.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def visible_geometry_annotations_path(data_path: Path) -> Path:
    if data_path.name == "annotations_v19_visible_geometry.json":
        return data_path
    report = load_json(data_path)
    raw = report.get("outputs", {}).get("annotations") if isinstance(report.get("outputs"), dict) else None
    if not raw:
        raw = report.get("annotations")
    if not raw:
        raise ContractError(f"visible_geometry_report_missing_annotations: {data_path}")
    path = Path(str(raw))
    return path if path.is_absolute() else Path.cwd() / path


def frame_object(frame: dict[str, Any], object_id: str | None = None) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if not isinstance(obj, dict):
            continue
        if object_id is None or obj.get("object_id") == object_id or obj.get("track_id") == object_id:
            return obj
    return None


def world_to_camera_points(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    T = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or [], dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return points_world
    Tcw = np.linalg.inv(T)
    pts_h = np.column_stack([points_world, np.ones(len(points_world), dtype=float)])
    return (pts_h @ Tcw.T)[:, :3]


def render_visible_geometry_overlay(frames: list[tuple[int, str]], data_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    annotations_path = visible_geometry_annotations_path(data_path)
    annotations = load_json(annotations_path)
    frames_by_idx = {int(frame.get("frame_idx")): frame for frame in annotations.get("frames", []) if isinstance(frame, dict) and frame.get("frame_idx") is not None}
    adapter = annotations.get("v19_visible_geometry_adapter", {}) if isinstance(annotations.get("v19_visible_geometry_adapter"), dict) else {}
    object_id = adapter.get("object_id")
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    visible_frames = 0
    projected_frames = 0
    projected_points_total = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        frame = frames_by_idx.get(int(fidx), {})
        obj = frame_object(frame, object_id)
        projected = 0
        vertex_count = 0
        if obj is not None:
            bbox = obj.get("bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 2)
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            pts = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
            intr = np.asarray(geom.get("intrinsics_fx_fy_cx_cy") or [], dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0 and intr.shape == (4,):
                visible_frames += 1
                vertex_count = int(len(pts))
                pts_cam = world_to_camera_points(pts, frame)
                pts2d = project_points(pts_cam, intr)
                projected = draw_projected_mesh(img, pts2d, pts_cam[:, 2], np.zeros((0, 3), dtype=np.int32), color=(255, 190, 0))
                if projected:
                    projected_frames += 1
                    projected_points_total += projected
        draw_timeline(img, written, len(frames), active=projected > 0, color=(255, 190, 0))
        draw_text_block(
            img,
            [
                f"visible geometry surfels: {object_id or 'object'}",
                f"frame {fidx}: vertices={vertex_count} projected={projected}",
                "SAM2 mask + metric depth backprojection; not final pose",
            ],
            (10, 28),
            0.52,
        )
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_visible_geometry_overlay_frames_written")
    qc = {
        "schema": "visible_geometry_overlay_qc.v1",
        "overlay_type": "visible_surfels_on_rgb",
        "data_path": str(data_path),
        "annotations_path": str(annotations_path),
        "visible_geometry_frames": int(visible_frames),
        "projected_frame_count": int(projected_frames),
        "projected_points_total": int(projected_points_total),
        "claim_scope": "Projects V19 visible surfel measurements from SAM2 masks plus metric depth onto source RGB. This is visible-surface evidence and centroid initialization, not completed object geometry or accepted object pose.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def resolve_pose_mesh(report: dict[str, Any], report_path: Path) -> Path:
    inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
    raw = inputs.get("completed_mesh")
    if not raw and inputs.get("completion_report"):
        completion_path = Path(str(inputs["completion_report"]))
        if not completion_path.is_absolute():
            completion_path = Path.cwd() / completion_path
        completion = load_json(completion_path)
        raw = completion.get("outputs", {}).get("completed_mesh_labeled") if isinstance(completion.get("outputs"), dict) else None
    if not raw:
        raise ContractError(f"pose_report_missing_completed_mesh: {report_path}")
    mesh_path = Path(str(raw))
    return mesh_path if mesh_path.is_absolute() else Path.cwd() / mesh_path


def render_pose_mesh_overlay(frames: list[tuple[int, str]], report_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    report = load_json(report_path)
    mesh_path = resolve_pose_mesh(report, report_path)
    verts, faces = read_obj_mesh(mesh_path)
    pose_rows = report.get("pose_rows") if isinstance(report.get("pose_rows"), list) else []
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows if isinstance(row, dict) and row.get("frame_idx") is not None}
    run_root = next((parent for parent in report_path.parents if (parent / "input" / "source_frame_manifest").exists()), None)
    if run_root is None:
        run_root = next((parent for parent in report_path.parents if (parent / "input" / "raw_frame_manifest").exists()), None)
    depth_source = (run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz") if run_root else None
    if depth_source is None or not depth_source.exists():
        raise ContractError(f"pose_overlay_depth_intrinsics_missing_for: {report_path}")
    depth_npz = np.load(str(depth_source))
    intrinsics = np.asarray(depth_npz["intrinsics_fx_fy_cx_cy"], dtype=float)
    depth_shape_hw = tuple(int(v) for v in depth_npz["depth"].shape[1:3]) if "depth" in depth_npz.files else None
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    method = str(report.get("method", report_path.parent.name))
    status = str(report.get("status", "unknown"))
    title = output_dir.name
    written = 0
    projected_frames = 0
    projected_points_total = 0
    pose_rows_with_projection = 0
    status_counts: dict[str, int] = {}
    for row in pose_rows:
        if isinstance(row, dict):
            status_counts[str(row.get("status", "unknown"))] = status_counts.get(str(row.get("status", "unknown")), 0) + 1
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        row = pose_by_frame.get(int(fidx))
        projected = 0
        row_status = "missing_pose_row"
        if row is not None:
            row_status = str(row.get("status", "unknown"))
            R = np.asarray(row.get("rotation_world_from_completed_canonical_matrix"), dtype=float)
            t = np.asarray(row.get("translation_world_m"), dtype=float)
            if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all() and 0 <= int(fidx) < len(intrinsics):
                pts_cam = verts @ R.T + t[None, :]
                intr = scale_intrinsics_for_image(intrinsics[int(fidx)], depth_shape_hw, img.shape[:2])
                pts2d = project_points(pts_cam, intr)
                projected = draw_projected_mesh(img, pts2d, pts_cam[:, 2], faces, color=(0, 230, 255) if title != "adopted_object_pose" else (0, 255, 80))
                pose_rows_with_projection += 1
                if projected:
                    projected_frames += 1
                    projected_points_total += projected
        draw_timeline(img, written, len(frames), active=projected > 0, color=(0, 255, 80) if title == "adopted_object_pose" else (0, 230, 255))
        lines = [
            f"{title}: projected completed mesh",
            f"{method[:70]}",
            f"report_status={status} row_status={row_status}",
            f"frame {fidx}: projected_points={projected}",
        ]
        if title == "adopted_object_pose":
            lines.append("adopted final pose source: v19_rigid_pose_graph")
        draw_text_block(img, lines, (10, 28), 0.5)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_pose_mesh_overlay_frames_written")
    residual_summary = report.get("final_observed_to_mesh_median_summary_m") if isinstance(report.get("final_observed_to_mesh_median_summary_m"), dict) else None
    residual_median = residual_summary.get("median") if residual_summary else None
    large_deviation = isinstance(residual_median, (int, float)) and float(residual_median) > 0.05
    qc = {
        "schema": "pose_mesh_projection_overlay_qc.v1",
        "overlay_type": "pose_mesh_projection_on_rgb",
        "data_path": str(report_path),
        "mesh_path": str(mesh_path),
        "method": method,
        "status": status,
        "pose_row_count": int(len(pose_rows)),
        "pose_rows_with_projection_fields": int(pose_rows_with_projection),
        "projected_frame_count": int(projected_frames),
        "projected_points_total": int(projected_points_total),
        "pose_row_status_counts": status_counts,
        "final_observed_to_mesh_median_summary_m": residual_summary,
        "large_deviation_detected": bool(large_deviation),
        "tuning_required": bool(large_deviation),
        "large_deviation_reason": "final observed-to-mesh median residual exceeds 0.05 m" if large_deviation else None,
        "claim_scope": "Projects the completed object mesh onto source RGB using this atom's pose rows. This visualizes pose evidence/adoption; it does not by itself prove contact, occlusion ownership, or nonpenetration.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def load_pose_rows(run_root: Path, object_id: str) -> dict[int, dict[str, Any]]:
    path = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v19_pose_graph" / "v19_rigid_object_pose_graph_report.json"
    if not path.exists():
        return {}
    report = load_json(path)
    return {int(row["frame_idx"]): row for row in report.get("pose_rows", []) if isinstance(row, dict) and row.get("frame_idx") is not None}


def resolve_mesh_candidate(data_path: Path) -> tuple[dict[str, Any], Path]:
    report = load_json(data_path)
    summary_path = data_path.parent / "mesh_candidate_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else report
    mesh_raw = summary.get("mesh_path") or report.get("outputs", {}).get("completed_mesh_labeled")
    if not mesh_raw:
        raise ContractError(f"mesh_candidate_report_missing_mesh_path: {data_path}")
    mesh_path = Path(str(mesh_raw))
    if not mesh_path.is_absolute():
        mesh_path = Path.cwd() / mesh_path
    if not mesh_path.exists():
        raise ContractError(f"mesh_candidate_obj_missing: {mesh_path}")
    return summary, mesh_path


def render_mesh_candidate_overlay(frames: list[tuple[int, str]], data_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    summary, mesh_path = resolve_mesh_candidate(data_path)
    verts, faces = read_obj_mesh(mesh_path)
    run_root = Path(str(summary.get("run_root") or data_path.parents[4]))
    if not run_root.is_absolute():
        run_root = Path.cwd() / run_root
    object_id = str(summary.get("object_id") or mesh_path.parent.name)
    depth_source = Path(str(summary.get("depth_source") or run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"))
    if not depth_source.is_absolute():
        depth_source = Path.cwd() / depth_source
    depth_data = np.load(str(depth_source))
    intrinsics = np.asarray(depth_data["intrinsics_fx_fy_cx_cy"], dtype=float)
    depth_shape_hw = tuple(int(v) for v in depth_data["depth"].shape[1:3]) if "depth" in depth_data.files else None
    pose_by_frame = load_pose_rows(run_root, object_id)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(frames)
    anchor = int(summary.get("anchor_frame", -1) or -1)
    lines = [
        f"mesh candidate projected on RGB: {object_id}",
        f"vertices={len(verts)} faces={len(faces)}",
        f"pose source={'v19_rigid_pose_graph' if pose_by_frame else 'anchor_camera_only'}",
    ]
    written = 0
    projected_frames = 0
    projected_points_total = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        pose = pose_by_frame.get(int(fidx))
        projection_source = None
        if pose is not None:
            R = np.asarray(pose.get("rotation_world_from_completed_canonical_matrix"), dtype=float)
            t = np.asarray(pose.get("translation_world_m"), dtype=float)
            if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
                pts_cam = verts @ R.T + t[None, :]
                projection_source = "pose_graph"
            else:
                pts_cam = verts
        elif int(fidx) == anchor:
            pts_cam = verts
            projection_source = "anchor_camera"
        else:
            pts_cam = verts
        if projection_source is not None and 0 <= int(fidx) < len(intrinsics):
            intr = scale_intrinsics_for_image(intrinsics[int(fidx)], depth_shape_hw, img.shape[:2])
            pts2d = project_points(pts_cam, intr)
            projected = draw_projected_mesh(img, pts2d, pts_cam[:, 2], faces)
            if projected:
                projected_frames += 1
                projected_points_total += projected
            active = int(fidx) == anchor
            draw_timeline(img, written, frame_count, active=active, color=(0, 255, 0) if active else (0, 220, 255))
            draw_text_block(img, lines + [f"frame {fidx}: projected_points={projected} source={projection_source}", f"anchor frame {anchor}"], (10, 28), 0.52)
        else:
            draw_timeline(img, written, frame_count, active=False)
            draw_text_block(img, lines + [f"frame {fidx}: no pose row; mesh not projected"], (10, 28), 0.52)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_mesh_candidate_overlay_frames_written")
    qc = {
        "schema": "mesh_projection_overlay_qc.v1",
        "overlay_type": "projected_mesh_on_rgb",
        "data_path": str(data_path),
        "mesh_path": str(mesh_path),
        "pose_graph_rows": int(len(pose_by_frame)),
        "projected_frame_count": int(projected_frames),
        "projected_points_total": int(projected_points_total),
        "claim_scope": "Projects the mesh candidate onto source RGB using V19 rigid pose graph rows when available, otherwise only the anchor camera frame. This visualizes the current mesh/pose evidence; it does not by itself accept object pose, contact, or nonpenetration.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def extract_report_frames(report: dict[str, Any]) -> set[int]:
    frames: set[int] = set()
    for key in ["used_frames", "graph_frames", "candidate_frames"]:
        vals = report.get(key)
        if isinstance(vals, list):
            for v in vals:
                try:
                    frames.add(int(v))
                except Exception:
                    pass
    for key in ["rows", "contact_rows"]:
        vals = report.get(key)
        if isinstance(vals, list):
            for row in vals:
                if isinstance(row, dict) and row.get("frame_idx") is not None:
                    frames.add(int(row["frame_idx"]))
    return frames


def report_metric_by_frame(report: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    graph_frames = report.get("graph_frames")
    pose_rows = report.get("pose_rows")
    if isinstance(graph_frames, list) and isinstance(pose_rows, list):
        for fidx, row in zip(graph_frames, pose_rows):
            if not isinstance(row, dict):
                continue
            metric = row.get("observed_to_mesh_final", row.get("mesh_to_observed_final", {}))
            if isinstance(metric, dict) and metric.get("median_m") is not None:
                out[int(fidx)] = float(metric["median_m"])
    for key in ["frame_metrics_after", "frame_metrics_before"]:
        frame_metrics = report.get(key)
        if isinstance(frame_metrics, dict):
            for fidx_raw, metrics in frame_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_key in ["observed_to_prior_median_m", "observed_to_mesh_median_m", "silhouette_outside_median_px"]:
                    val = metrics.get(metric_key)
                    if isinstance(val, (int, float)):
                        out[int(fidx_raw)] = float(val)
                        break
                    if isinstance(val, dict) and val.get("median") is not None:
                        out[int(fidx_raw)] = float(val["median"])
                        break
    return out


def resolve_report_path(report: dict[str, Any], raw: str | Path | None, report_path: Path | None = None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    candidates: list[Path] = []
    candidates.append(path)
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        if report_path is not None:
            candidates.append(report_path.parent / path)
    run_root_raw = report.get("run_root")
    if run_root_raw:
        run_root = Path(str(run_root_raw))
        if not run_root.is_absolute():
            run_root = Path.cwd() / run_root
        if not path.is_absolute():
            candidates.append(run_root / path)
    text = str(raw)
    if text.startswith("outputs/"):
        replacement = Path("output") / Path(text).relative_to("outputs")
        candidates.extend([replacement, Path.cwd() / replacement])
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        replacement = Path("output") / Path(text[len(historical_prefix) :])
        candidates.extend([replacement, Path.cwd() / replacement])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_keyframe_track(report: dict[str, Any], report_path: Path) -> dict[int, dict[str, Any]]:
    track_path = resolve_report_path(report, report.get("sam2_track"), report_path)
    if track_path is None:
        return {}
    try:
        payload = load_json(track_path)
    except Exception:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        try:
            out[int(key)] = value
        except Exception:
            continue
    return out


def selected_keyframe_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected = report.get("selected_keyframes") if isinstance(report.get("selected_keyframes"), list) else []
    out: dict[int, dict[str, Any]] = {}
    for row in selected:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        out[int(row["frame_idx"])] = row
    return out


def draw_keyframe_spatial_evidence(
    img: np.ndarray,
    report: dict[str, Any],
    report_path: Path,
    track_rows: dict[int, dict[str, Any]],
    keyframe_rows: dict[int, dict[str, Any]],
    frame_idx: int,
) -> bool:
    if str(report.get("schema", "")) != "v21_segmentation_stable_keyframes.v0":
        return False
    if int(frame_idx) not in keyframe_rows:
        return False
    row = track_rows.get(int(frame_idx), {})
    if not row.get("visible"):
        return False
    mask_path = resolve_report_path(report, row.get("mask_path"), report_path)
    mask_drawn = False
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            support = mask > 127
            if bool(support.any()):
                overlay = img.copy()
                overlay[support] = (0, 220, 255)
                cv2.addWeighted(overlay, 0.52, img, 0.48, 0, dst=img)
                contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
                mask_drawn = True
    bbox = row.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(img.shape[1] - 1, x1))
        x2 = max(0, min(img.shape[1] - 1, x2))
        y1 = max(0, min(img.shape[0] - 1, y1))
        y2 = max(0, min(img.shape[0] - 1, y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 3)
            cv2.putText(img, "selected keyframe bbox", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 80), 2, cv2.LINE_AA)
            mask_drawn = True
    return mask_drawn


def keyframe_segment_lines(report: dict[str, Any], frame_idx: int) -> list[str]:
    if str(report.get("schema", "")) != "v21_segmentation_stable_keyframes.v0":
        return []
    segments = report.get("class_segments")
    if not isinstance(segments, list) or not segments:
        return []
    current = None
    current_i = -1
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        start = int(segment.get("frame_start", segment.get("start_pos", -1)))
        end = int(segment.get("frame_end", segment.get("end_pos", -1)))
        if start <= int(frame_idx) <= end:
            current = segment
            current_i = idx
            break
    if current is None:
        return [f"segment: unresolved for frame {frame_idx}"]
    start = int(current.get("frame_start", current.get("start_pos", -1)))
    end = int(current.get("frame_end", current.get("end_pos", -1)))
    cls = str(current.get("interaction_class", "unknown"))
    selected = report.get("selected_keyframes") if isinstance(report.get("selected_keyframes"), list) else []
    keyframe = next((row for row in selected if isinstance(row, dict) and int(row.get("frame_idx", -999999)) == int(frame_idx)), None)
    prefix = "KEYFRAME" if keyframe is not None else "segment"
    lines = [f"{prefix} {current_i + 1}/{len(segments)}: {cls} frames {start}-{end}"]
    if keyframe is not None:
        score = keyframe.get("stability_score")
        if score is not None:
            lines.append(f"selected midpoint stability={float(score):.3f}")
    return lines


def render_report_overlay(frames: list[tuple[int, str]], report_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    report = load_json(report_path)
    active_frames = extract_report_frames(report)
    metric = report_metric_by_frame(report)
    row_count = len(report.get("rows", []) or []) if isinstance(report.get("rows", []), list) else 0
    contact_count = len(report.get("contact_rows", []) or []) if isinstance(report.get("contact_rows", []), list) else 0
    method = str(report.get("method", report_path.parent.name))
    status = str(report.get("status", "unknown"))
    annotation_ready = report.get("annotation_ready")
    accepted = report.get("contact_ownership_accepted_rows", report.get("accepted_occlusion_owner_rows", report.get("evaluated_signed_rows", report.get("evaluated_triangle_rows"))))
    keyframe_track_rows = load_keyframe_track(report, report_path)
    keyframe_rows = selected_keyframe_rows(report)
    keyframe_evidence_frames_drawn = 0
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(frames)
    written = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        active = fidx in active_frames
        draw_timeline(img, written, frame_count, active=active)
        lines = [
            f"{method[:70]}",
            f"status={status} annotation_ready={annotation_ready}",
            f"rows={row_count} contact_rows={contact_count} accepted/evaluated={accepted}",
        ]
        if fidx in metric:
            lines.append(f"frame metric={metric[fidx]:.4f}")
        elif active:
            lines.append("frame participates; no projectable spatial row in report")
        elif not active_frames:
            lines.append("report contains no per-frame rows to project")
        segment_lines = keyframe_segment_lines(report, fidx)
        if draw_keyframe_spatial_evidence(img, report, report_path, keyframe_track_rows, keyframe_rows, fidx):
            keyframe_evidence_frames_drawn += 1
            lines.append("selected keyframe evidence: SAM2 mask + bbox")
        if segment_lines:
            lines.extend(segment_lines)
        draw_text_block(img, lines, (10, 28), 0.5)
        if segment_lines:
            draw_text_block(img, segment_lines, (10, max(28, img.shape[0] - 56)), 0.62)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_report_overlay_frames_written")
    qc = {
        "schema": "report_overlay_qc.v1",
        "overlay_type": "report_timeline_on_source_rgb",
        "data_path": str(report_path),
        "method": method,
        "status": status,
        "annotation_ready": annotation_ready,
        "frame_count": int(written),
        "active_frame_count": int(len(active_frames)),
        "row_count": int(row_count),
        "contact_row_count": int(contact_count),
        "segment_subtitles": str(report.get("schema", "")) == "v21_segmentation_stable_keyframes.v0",
        "keyframe_spatial_evidence": str(report.get("schema", "")) == "v21_segmentation_stable_keyframes.v0",
        "selected_keyframe_count": int(len(keyframe_rows)),
        "keyframe_evidence_frames_drawn": int(keyframe_evidence_frames_drawn),
        "claim_scope": "Draws report/status/timeline evidence over source RGB. This visualizes a consumed information artifact; it does not by itself validate physical correctness.",
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def render_camera_overlay(frames: list[tuple[int, str]], camera_path: Path, output_dir: Path, fps: float = 25.0) -> Path:
    data = np.load(camera_path, allow_pickle=True)
    if "poses" not in data:
        return render_depth_overlay(frames, camera_path, output_dir, fps)
    poses = np.asarray(data["poses"], dtype=np.float32)
    fidxs = data["frame_idx"] if "frame_idx" in data else np.arange(len(poses))
    fidx_to_i = {int(f): i for i, f in enumerate(fidxs)}
    trans = poses[:, :3, 3] if poses.ndim == 3 and poses.shape[1:] == (4, 4) else np.zeros((len(poses), 3), dtype=np.float32)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fidx, img_path in frames:
        img = cv2.imread(img_path)
        if img is None:
            continue
        i = fidx_to_i.get(fidx, min(written, len(trans) - 1))
        h, w = img.shape[:2]
        x0, y0, iw, ih = w - 300, 24, 270, 190
        cv2.rectangle(img, (x0, y0), (x0 + iw, y0 + ih), (0, 0, 0), -1)
        pts = trans[:, [0, 2]]
        span = np.nanmax(np.ptp(pts, axis=0)) if len(pts) else 1.0
        if not np.isfinite(span) or span < 1e-6:
            span = 1.0
        center = np.nanmean(pts, axis=0) if len(pts) else np.zeros(2)
        norm = (pts - center) / span
        px = (x0 + iw / 2 + norm[:, 0] * iw * 0.42).astype(int)
        py = (y0 + ih / 2 - norm[:, 1] * ih * 0.42).astype(int)
        for a, b in zip(range(max(0, i)), range(1, max(1, i + 1))):
            cv2.line(img, (int(px[a]), int(py[a])), (int(px[b]), int(py[b])), (80, 220, 255), 1)
        if 0 <= i < len(px):
            cv2.circle(img, (int(px[i]), int(py[i])), 5, (0, 255, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + iw, y0 + ih), (200, 200, 200), 1)
        draw_text_block(img, ["camera/depth trajectory", f"frame={fidx} pose_idx={i}", f"translation={trans[i].round(3).tolist() if len(trans) else []}"], (10, 28), 0.5)
        cv2.imwrite(str(frame_dir / f"{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    if written == 0:
        raise ContractError("no_camera_overlay_frames_written")
    return encode_frames(frame_dir, output_dir / "overlay.mp4", fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["depth", "segmentation", "detection", "hand", "heightfield", "mesh_candidate", "report", "camera", "prompts", "visible_geometry", "pose_mesh"])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-path", required=True, help="Path to algorithm output (npz/json/mask dir)")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run_root)
    frames = get_source_frames(run_root, manifest)
    data_path = Path(args.data_path)
    if args.type == "depth":
        out = render_depth_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "segmentation":
        out = render_segmentation_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "detection":
        out = render_detection_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "hand":
        out = render_hand_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "heightfield":
        out = render_heightfield_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "mesh_candidate":
        out = render_mesh_candidate_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "report":
        out = render_report_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "camera":
        out = render_camera_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "prompts":
        out = render_prompt_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "visible_geometry":
        out = render_visible_geometry_overlay(frames, data_path, output_dir, args.fps)
    elif args.type == "pose_mesh":
        out = render_pose_mesh_overlay(frames, data_path, output_dir, args.fps)
    print(json.dumps({"status": "ok", "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
