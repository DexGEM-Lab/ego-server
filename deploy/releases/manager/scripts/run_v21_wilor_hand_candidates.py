#!/usr/bin/env python3
"""V21 WiLoR hand candidate runner.

Runs WiLoR on all frames from a V21 raw-frame manifest to produce raw 3D MANO
hand candidate evidence. This is candidate evidence only, not accepted MANO state.

The output feeds the V21 hand diagnosis path (Section 5 of the V21 design):
  - detector box quality and side mapping
  - crop/resize convention (handled by WiLoR internally)
  - metric scale alignment against DepthPro depth
  - temporal consistency
  - visibility/occlusion state

Compute target: declared by --compute-target and recorded in QC metadata.

Output:
  measurements/hand_candidates/wilor_v21/wilor_raw_hands.json
  measurements/hand_candidates/wilor_v21/wilor_qc.json
  measurements/hand_candidates/wilor_v21/wilor_v21_diagnosis.json
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from tqdm import tqdm

WILOR_DIR = Path(os.environ.get("V22_WILOR_DIR", "/home/zjh/ego-annation-checkpoints/wilor_model"))
DEFAULT_MANO_RIGHT = Path(os.environ.get("V22_MANO_RIGHT", "/home/zjh/ego-annation-checkpoints/wilor_model/mano_data/MANO_RIGHT.pkl"))


def patch_legacy_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    raw_load = torch.load
    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return raw_load(*args, **kwargs)
    torch.load = torch_load_compat


def load_wilor_backend(wilor_root: Path):
    patch_legacy_imports()
    sys.path.insert(0, str(wilor_root.resolve()))
    from ultralytics import YOLO
    from wilor.models import load_wilor

    cwd = Path.cwd()
    os.chdir(wilor_root)
    try:
        model, cfg = load_wilor(
            str(wilor_root / "pretrained_models" / "wilor_final.ckpt"),
            str(wilor_root / "pretrained_models" / "model_config.yaml"),
        )
        detector = YOLO(str(wilor_root / "pretrained_models" / "detector.pt"))
    finally:
        os.chdir(cwd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = detector.to(device)
    return model, cfg, detector, device


def project_full_image(points, cam_t, focal, img_size):
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(img_size[0]) / 2.0
    K[1, 2] = float(img_size[1]) / 2.0
    pts = points + cam_t
    z = np.clip(pts[:, 2:3], 1e-6, None)
    pts = pts / z
    return (K @ pts.T).T[:, :2]


def run_wilor_on_frame(model, cfg, detector, device, frame, rescale_factor=2.0, batch_size=8, conf_thresh=0.3):
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    detections = detector(frame, conf=conf_thresh, verbose=False)[0]
    boxes, scores, is_right = [], [], []
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
    predictions = []
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
            mano_params = {}
            for key, value in out["pred_mano_params"].items():
                mano_params[key] = value[n].detach().cpu().numpy().astype(float).tolist()
            predictions.append({
                "backend": "WiLoR",
                "side": hand_side,
                "detector_score": scores[det_idx],
                "bbox_xyxy": boxes_np[det_idx].astype(float).tolist(),
                "cam_t": cam_t[n].astype(float).tolist(),
                "focal_length": focal,
                "joints3d_camera": joints.tolist(),
                "joints2d": joints2d.astype(float).tolist(),
                "mano_params": mano_params,
                "vertices_camera": verts.tolist(),
                "vertices_camera_sample": verts[::10].tolist(),
                "filter_status": "measured_raw",
            })
        det_offset += batch["img"].shape[0]
    return predictions


def load_depth_archive(npz_path):
    data = np.load(str(npz_path))
    # Try common keys
    for key in ["depth_meters", "depth", "depth_metric"]:
        if key in data:
            return data[key]
    # Fallback: first array
    return data[data.files[0]]


def depth_at_points(depth_map, points2d, img_h, img_w):
    """Sample depth at projected 2D points."""
    px = np.clip(points2d[:, 0].astype(int), 0, img_w - 1)
    py = np.clip(points2d[:, 1].astype(int), 0, img_h - 1)
    return depth_map[py, px]


def diagnose_hand_candidates(raw_frames, depth_archive_path, manifest):
    """Diagnose raw WiLoR candidates against DepthPro depth for metric alignment.

    Checks:
    - detection rate and score distribution
    - side consistency (left/right switching)
    - metric scale: WiLoR cam_t z vs DepthPro depth at hand centroid
    - temporal consistency: cam_t jump detection
    """
    total_frames = len(raw_frames)
    frames_with_hands = sum(1 for f in raw_frames if f["raw_hands"])
    total_hands = sum(len(f["raw_hands"]) for f in raw_frames)

    # Side consistency
    side_sequence = []
    for f in raw_frames:
        if f["raw_hands"]:
            sides = sorted(set(h["side"] for h in f["raw_hands"]))
            side_sequence.append(sides)

    # Detection scores
    all_scores = [h["detector_score"] for f in raw_frames for h in f["raw_hands"]]
    score_stats = {}
    if all_scores:
        arr = np.array(all_scores)
        score_stats = {
            "count": len(arr),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    # Depth alignment diagnosis (sample frames)
    depth = None
    depth_residuals = []
    if depth_archive_path and Path(depth_archive_path).exists():
        try:
            depth = load_depth_archive(depth_archive_path)
            depth_h, depth_w = depth.shape[:2]
            img_h = manifest["timeline"]["resolution"][1] if "timeline" in manifest else depth_h
            img_w = manifest["timeline"]["resolution"][0] if "timeline" in manifest else depth_w
            for f in raw_frames:
                for h in f["raw_hands"]:
                    joints2d = np.array(h["joints2d"])
                    cam_t_z = h["cam_t"][2]
                    # Scale joints2d to depth resolution if needed
                    pts2d = joints2d.copy()
                    if depth_w != img_w:
                        pts2d[:, 0] *= depth_w / img_w
                    if depth_h != img_h:
                        pts2d[:, 1] *= depth_h / img_h
                    hand_depth = depth_at_points(depth, pts2d, depth_h, depth_w)
                    valid = hand_depth[hand_depth > 0.1]
                    if len(valid) > 5:
                        median_depth = float(np.median(valid))
                        residual = abs(cam_t_z - median_depth)
                        depth_residuals.append(residual)
        except Exception as e:
            print(f"  depth diagnosis warning: {e}", file=sys.stderr)

    depth_alignment = {}
    if depth_residuals:
        arr = np.array(depth_residuals)
        depth_alignment = {
            "samples": len(arr),
            "median_residual_m": float(np.median(arr)),
            "p95_residual_m": float(np.percentile(arr, 95)),
            "interpretation": "wilor_cam_t_z_vs_depthpro_hand_region_depth; large residual means WiLoR local scale needs metric refit",
        }

    # Temporal consistency: cam_t jumps
    temporal_jumps = []
    prev_cam_t = None
    for f in raw_frames:
        for h in f["raw_hands"]:
            cam_t = np.array(h["cam_t"])
            if prev_cam_t is not None:
                jump = float(np.linalg.norm(cam_t - prev_cam_t))
                temporal_jumps.append(jump)
            prev_cam_t = cam_t
    temporal_stats = {}
    if temporal_jumps:
        arr = np.array(temporal_jumps)
        temporal_stats = {
            "median_frame_jump_m": float(np.median(arr)),
            "p95_frame_jump_m": float(np.percentile(arr, 95)),
        }

    diagnosis = {
        "schema": "v21_hand_candidate_diagnosis.v0",
        "status": "ok",
        "backend": "WiLoR",
        "total_frames": total_frames,
        "frames_with_hands": frames_with_hands,
        "detection_rate": frames_with_hands / max(1, total_frames),
        "total_hands": total_hands,
        "mean_hands_per_frame": total_hands / max(1, total_frames),
        "detection_score_stats": score_stats,
        "depth_alignment": depth_alignment,
        "temporal_consistency": temporal_stats,
        "metric_scale_status": "wilor_local_scale_not_metric; requires_depth_refit_before_acceptance" if depth_residuals else "depth_archive_not_available_for_diagnosis",
        "candidate_state": "raw_evidence_not_accepted_mano_state",
        "next_required_step": "active_shape_pose_scale_optimization_against_depth_and_keypoints",
    }
    return diagnosis


def run(args):
    run_root = Path(args.run_root)
    output_dir = run_root / "measurements" / "hand_candidates" / "wilor_v21"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames_meta = manifest["frames"]
    if args.max_frames is not None:
        frames_meta = frames_meta[:args.max_frames]
    total_frames = len(frames_meta)
    repo_root = Path(args.repo_root)

    print(f"Loading WiLoR backend from {WILOR_DIR}...", flush=True)
    model, cfg, detector, device = load_wilor_backend(WILOR_DIR)
    print(f"WiLoR loaded on {device}", flush=True)

    # Load depth archive for diagnosis
    depth_archive = None
    depth_report_path = run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json"
    if depth_report_path.exists():
        depth_report = json.loads(depth_report_path.read_text())
        depth_archive = depth_report.get("primary_depth_archive")
    if depth_archive is None:
        npz = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
        if npz.exists():
            depth_archive = str(npz)

    raw_frames = []
    detected_frames = 0
    detected_hands = 0
    started = time.time()

    pbar = tqdm(total=total_frames, desc="WiLoR V21 hand candidates")
    for fm in frames_meta:
        frame_idx = fm["frame_idx"]
        rgb_path = repo_root / fm["rgb"]
        img = cv2.imread(str(rgb_path))
        if img is None:
            raw_frames.append({"frame_idx": frame_idx, "time_s": fm["time_s"], "raw_hands": []})
            pbar.update(1)
            continue
        hands = run_wilor_on_frame(model, cfg, detector, device, img,
                                   rescale_factor=args.rescale_factor,
                                   batch_size=args.batch_size,
                                   conf_thresh=args.conf_thresh)
        if hands:
            detected_frames += 1
            detected_hands += len(hands)
        raw_frames.append({
            "frame_idx": frame_idx,
            "time_s": fm["time_s"],
            "raw_hands": hands,
        })
        pbar.update(1)
    pbar.close()

    del model, detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save raw output
    raw_path = output_dir / "wilor_raw_hands.json"
    raw_payload = {
        "schema": "v21_wilor_hand_candidates.v0",
        "backend": "WiLoR",
        "wilor_root": str(WILOR_DIR),
        "frame_count": total_frames,
        "frames": raw_frames,
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    # Diagnosis
    diagnosis = diagnose_hand_candidates(raw_frames, depth_archive, manifest)
    elapsed = time.time() - started

    qc = {
        "status": "ok",
        "run_root": str(run_root),
        "processed_frames": total_frames,
        "frames_with_hands": detected_frames,
        "detection_rate": detected_frames / max(1, total_frames),
        "detected_hands": detected_hands,
        "mean_hands_per_frame": detected_hands / max(1, total_frames),
        "elapsed_s": elapsed,
        "compute_target": args.compute_target,
        "raw_path": str(raw_path),
        "diagnosis_path": str(output_dir / "wilor_v21_diagnosis.json"),
    }
    (output_dir / "wilor_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    (output_dir / "wilor_v21_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")

    print(json.dumps(qc, indent=2))
    print("\n=== Diagnosis ===")
    print(json.dumps(diagnosis, indent=2))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="V21 run root directory")
    ap.add_argument("--repo-root", default="/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    ap.add_argument("--rescale-factor", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames for testing")
    ap.add_argument("--compute-target", default=os.environ.get("V21_COMPUTE_TARGET", "declared_gpu_compute_target"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
