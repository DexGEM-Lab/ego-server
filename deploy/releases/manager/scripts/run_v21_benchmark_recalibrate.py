#!/usr/bin/env python3
"""Recalibrate benchmark MANO joints using dataset camera intrinsics."""
import json, os, sys, inspect, numpy as np
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
WILOR_DIR = Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model")
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for n, v in {"bool":bool,"int":int,"float":float,"complex":complex,"object":object,"unicode":str,"str":str}.items():
    if not hasattr(np, n): setattr(np, n, v)

import torch, cv2
sys.path.insert(0, str(WILOR_DIR))
os.chdir(WILOR_DIR)
from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full

print("Loading WiLoR...", flush=True)
model, cfg = load_wilor(str(WILOR_DIR/"pretrained_models/wilor_final.ckpt"),
                        str(WILOR_DIR/"pretrained_models/model_config.yaml"))
detector = YOLO(str(WILOR_DIR/"pretrained_models/detector.pt"))
device = torch.device("cuda")
model = model.to(device).eval()
detector = detector.to(device)
os.chdir("/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
print("WiLoR loaded!", flush=True)

def solve_cam_t(joints_local, joints2d, K):
    fx, fy, cx, cy = K
    qx = (joints2d[:, 0] - cx) / fx
    qy = (joints2d[:, 1] - cy) / fy
    rows, rhs = [], []
    for (x, y, z), u, v in zip(joints_local, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.array(rows, float), np.array(rhs, float), rcond=None)
    return trans

for case in ["dexycb_20200813_151041_932122062010", "ho3d_train_MC1"]:
    run_dir = Path(f"outputs/v20_benchmark_two_datasets_20260626/{case}")
    manifest = json.loads((run_dir / "input/raw_frame_manifest/manifest.json").read_text())
    dm = json.loads((run_dir / "input/dataset_manifest.json").read_text())
    K = np.array(dm["camera_intrinsics"])
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    print(f"\n=== {case} (fx={fx:.1f}, cx={cx:.1f}) ===", flush=True)
    
    hand_data = {}
    for fi, fm in enumerate(manifest["frames"]):
        fidx = fm["frame_idx"]
        img = cv2.imread(fm["rgb"])
        if img is None: continue
        
        dets = detector(img, conf=0.3, verbose=False)[0]
        boxes, scores, is_right = [], [], []
        for det in dets:
            arr = det.boxes.data.cpu().detach().squeeze().numpy()
            if arr.ndim == 0 or arr.size < 6: continue
            boxes.append(arr[:4].astype(float).tolist())
            scores.append(float(arr[4]))
            is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
        if not boxes: continue
        
        boxes_np = np.asarray(boxes, dtype=np.float32)
        ds = ViTDetDataset(cfg, img, boxes_np, np.asarray(is_right, dtype=np.float32), rescale_factor=2.0, fp16=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        
        hands_list = []
        det_off = 0
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            scaled_focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
            cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, scaled_focal).detach().cpu().numpy()
            
            for n in range(batch["img"].shape[0]):
                side = "right" if float(batch["right"][n].cpu().numpy()) >= 0.5 else "left"
                joints = out["pred_keypoints_3d"][n].cpu().numpy().astype(float)
                sign = 1.0 if side == "right" else -1.0
                joints[:, 0] *= sign
                
                # Bone-length metric scale
                chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
                lengths = [sum(np.linalg.norm(joints[c[i+1]]-joints[c[i]]) for i in range(len(c)-1)) for c in chains]
                scale = 0.171 / max(np.median(lengths), 0.01)
                joints_m = joints * scale
                
                # WiLoR's 2D projection
                focal_w = float(scaled_focal.cpu().numpy())
                cam_t_w = cam_t[n]
                jc = joints_m + cam_t_w
                z = np.clip(jc[:, 2], 0.01, None)
                j2d = np.column_stack([focal_w * jc[:,0]/z + img.shape[1]/2, focal_w * jc[:,1]/z + img.shape[0]/2])
                
                # Solve cam_t using dataset intrinsics
                cam_t_ds = solve_cam_t(joints_m, j2d, [fx, fy, cx, cy])
                joints_cam_ds = joints_m + cam_t_ds
                
                hands_list.append({
                    "hand_side": side,
                    "bbox_xyxy": boxes_np[det_off+n].tolist(),
                    "metric_mano_state": {
                        "joints3d_camera_metric": joints_cam_ds.tolist(),
                        "joints_3d_camera_m": joints_cam_ds.tolist(),
                        "cam_t_metric": cam_t_ds.tolist(),
                        "scale": scale,
                        "source": "WiLoR_v21_dataset_intrinsics_recalibrated",
                    },
                })
            det_off += batch["img"].shape[0]
        
        if hands_list:
            hand_data[fidx] = hands_list
        
        if fi % 20 == 0:
            print(f"  Frame {fidx}: {len(hands_list)} hands", flush=True)
    
    total = sum(len(h) for h in hand_data.values())
    print(f"  Total: {total} hands across {len(hand_data)} frames", flush=True)
    
    # Inject
    ann_path = run_dir / "state/annotations_v20_renderable.json"
    ann = json.loads(ann_path.read_text())
    for frame in ann.get("frames", []):
        fidx = frame["frame_idx"]
        if fidx in hand_data:
            frame["hands"] = hand_data[fidx]
    ann_path.write_text(json.dumps(ann, indent=2))
    print(f"  Injected into annotation", flush=True)

print("\nDone!", flush=True)
