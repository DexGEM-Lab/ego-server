#!/usr/bin/env python3
"""V21 benchmark MANO hand injection.

Runs WiLoR hand detection on benchmark dataset frames (DexYCB, HO3D),
applies metric scale refit, and injects camera-frame metric joints into
the benchmark annotation JSON. This transitions hand_joint_camera_m_error
from "unsupported" to evaluated.

This is the mechanism that connects V21's hand estimation to the
existing V20 benchmark evaluation harness.

Output:
  Updates <benchmark_run>/state/annotations_v20_renderable.json
  with hands containing metric_mano_state.joints3d_camera_metric
"""
from __future__ import annotations
import argparse, json, os, sys, inspect, gc, time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch

WILOR_DIR = Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model")


def patch_legacy_imports():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, val in {"bool":bool,"int":int,"float":float,"complex":complex,"object":object,"unicode":str,"str":str}.items():
        if not hasattr(np, name): setattr(np, name, val)
    raw_load = torch.load
    def compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return raw_load(*a, **kw)
    torch.load = compat


def load_wilor():
    patch_legacy_imports()
    sys.path.insert(0, str(WILOR_DIR))
    os.chdir(WILOR_DIR)
    from ultralytics import YOLO
    from wilor.models import load_wilor
    model, cfg = load_wilor(
        str(WILOR_DIR / "pretrained_models/wilor_final.ckpt"),
        str(WILOR_DIR / "pretrained_models/model_config.yaml"))
    detector = YOLO(str(WILOR_DIR / "pretrained_models/detector.pt"))
    os.chdir("/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    device = torch.device("cuda")
    model = model.to(device).eval()
    detector = detector.to(device)
    return model, cfg, detector, device


def run_wilor_frame(model, cfg, detector, device, frame_img, rescale_factor=2.0, batch_size=8, conf_thresh=0.3):
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    detections = detector(frame_img, conf=conf_thresh, verbose=False)[0]
    boxes, scores, is_right = [], [], []
    for det in detections:
        arr = det.boxes.data.cpu().detach().squeeze().numpy()
        if arr.ndim == 0 or arr.size < 6: continue
        boxes.append(arr[:4].astype(float).tolist())
        scores.append(float(arr[4]))
        is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
    if not boxes:
        return []
    boxes_np = np.asarray(boxes, dtype=np.float32)
    right_np = np.asarray(is_right, dtype=np.float32)
    dataset = ViTDetDataset(cfg, frame_img, boxes_np, right_np, rescale_factor=rescale_factor, fp16=False)
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
        scaled_focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal).detach().cpu().numpy()
        focal = float(scaled_focal.detach().cpu().numpy())
        for n in range(batch["img"].shape[0]):
            det_idx = det_offset + n
            hand_side = "right" if float(batch["right"][n].detach().cpu().numpy()) >= 0.5 else "left"
            verts = out["pred_vertices"][n].detach().cpu().numpy().astype(float)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
            side_sign = 1.0 if hand_side == "right" else -1.0
            verts[:, 0] = side_sign * verts[:, 0]
            joints[:, 0] = side_sign * joints[:, 0]
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
                "mano_params": mano_params,
            })
        det_offset += batch["img"].shape[0]
    return predictions


def metric_scale_joints(joints3d_local, target_bone_m=0.171):
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    lengths = []
    for c in chains:
        l = sum(np.linalg.norm(joints3d_local[c[i+1]] - joints3d_local[c[i]]) for i in range(len(c)-1))
        lengths.append(l)
    median_bone = float(np.median(lengths))
    scale = target_bone_m / median_bone if median_bone > 0.01 else 1.0
    return joints3d_local * scale, scale


def run(args):
    benchmark_root = Path(args.benchmark_root)
    
    print("Loading WiLoR...", flush=True)
    model, cfg, detector, device = load_wilor()
    print("WiLoR loaded!", flush=True)
    
    for case in args.cases:
        case_dir = benchmark_root / case
        if not case_dir.exists():
            print(f"  SKIP {case}: not found", flush=True)
            continue
        
        # Load manifest
        manifest_path = case_dir / "input/raw_frame_manifest/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        frames_meta = manifest["frames"]
        print(f"\n=== {case}: {len(frames_meta)} frames ===", flush=True)
        
        # Run WiLoR on each frame
        hand_data_by_frame = {}
        total_hands = 0
        for fm in frames_meta:
            fidx = fm["frame_idx"]
            rgb_path = fm.get("rgb", "")
            if not rgb_path or not Path(rgb_path).exists():
                print(f"  Frame {fidx}: RGB not found", flush=True)
                continue
            img = cv2.imread(rgb_path)
            hands = run_wilor_frame(model, cfg, detector, device, img)
            
            # Apply metric scale
            img_w = fm.get("source_width", img.shape[1])
            img_h = fm.get("source_height", img.shape[0])
            
            for h in hands:
                joints_local = np.array(h["joints3d_camera"], dtype=float)
                joints_scaled, scale = metric_scale_joints(joints_local)
                cam_t = np.array(h["cam_t"], dtype=float)
                # Camera-frame joints (metric)
                joints_camera_m = joints_scaled + cam_t
                
                h["metric_mano_state"] = {
                    "joints3d_camera_metric": joints_camera_m.tolist(),
                    "joints_3d_camera_m": joints_camera_m.tolist(),  # evaluator key
                    "cam_t_metric": cam_t.tolist(),
                    "scale": scale,
                    "source": "WiLoR_v21_metric_refit",
                }
                total_hands += 1
            
            hand_data_by_frame[fidx] = hands
        
        print(f"  Detected {total_hands} hands across {len(hand_data_by_frame)} frames", flush=True)
        
        # Inject into annotation
        ann_path = case_dir / "state/annotations_v20_renderable.json"
        ann = json.loads(ann_path.read_text())
        
        for frame in ann.get("frames", []):
            fidx = frame["frame_idx"]
            if fidx in hand_data_by_frame:
                frame["hands"] = hand_data_by_frame[fidx]
        
        ann_path.write_text(json.dumps(ann, indent=2))
        print(f"  Injected hands into {ann_path}", flush=True)
        
        # Also update physical state
        state_path = case_dir / "state/v20_physical_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            # Update hand fields
            state.setdefault("hand_shape", {})
            state["hand_shape"]["status"] = "wilor_metric_mano_injected"
            state["hand_shape"]["hand_frame_count"] = len(hand_data_by_frame)
            state_path.write_text(json.dumps(state, indent=2))
    
    del model, detector
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone! Hand joints injected into benchmark annotations.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    args = ap.parse_args()
    run(args)
