#!/usr/bin/env python3
"""V21 active MANO optimizer.

Performs active betas/scale/pose optimization as required by V21 Section 5.
This is the V21 equivalent of optimize_contact_aware_mano_graph_v8.py,
designed to work directly with WiLoR candidates and V21 measurement format.

Optimization variables (per-frame + track-level):
- Per-frame: global_orient (3 rotvec), hand_pose (15×3 rotvec), translation (3)
- Track-level: betas (10), log_scale (1)

Loss terms:
1. 2D keypoint reprojection (WiLoR joints2d)
2. Metric depth alignment (DepthPro depth at hand region)
3. Bone-length prior (target hand span)
4. MANO shape prior (betas regularization)
5. Temporal smoothness (pose + translation)

Output:
  measurements/hand_candidates/v21_active_mano/optimized_mano_state.json
"""
from __future__ import annotations
import argparse, json, sys, os, inspect, time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from scipy.optimize import least_squares

WILOR_DIR = Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model")

def patch_legacy_imports():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, val in {"bool":bool,"int":int,"float":float,"complex":complex,"object":object,"unicode":str,"str":str}.items():
        if not hasattr(np, name): setattr(np, name, val)
    raw = torch.load
    def compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return raw(*a, **kw)
    torch.load = compat


def project(pts_3d, intrinsics):
    fx, fy, cx, cy = intrinsics
    z = np.clip(pts_3d[:, 2], 0.01, None)
    return np.column_stack([fx * pts_3d[:, 0] / z + cx, fy * pts_3d[:, 1] / z + cy])


def hand_bone_scale(joints):
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    lengths = []
    for c in chains:
        l = sum(np.linalg.norm(joints[c[i+1]] - joints[c[i]]) for i in range(len(c)-1))
        lengths.append(l)
    return float(np.median(lengths))


def load_json(p): return json.loads(Path(p).read_text())


def run(args):
    run_root = Path(args.run_root).resolve()
    patch_legacy_imports()
    
    sys.path.insert(0, str(WILOR_DIR))
    os.chdir(WILOR_DIR)
    from wilor.models.mano_wrapper import MANO
    model = MANO(
        model_path=str(WILOR_DIR / "mano_data"),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to("cuda").eval()
    os.chdir("/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")  # back to repo root
    
    # Load hand candidates
    hands_data = load_json(run_root / "measurements/hand_candidates/wilor_v21_metric/wilor_metric_hands.json")
    
    # Load depth
    depth_npz = np.load(str(run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"))
    depth_maps = depth_npz["depth"]
    depth_intrinsics = depth_npz["intrinsics_fx_fy_cx_cy"]
    
    # Collect frames with hands
    frame_data = []
    for f in hands_data["frames"]:
        if not f["hands"]:
            continue
        for hi, hand in enumerate(f["hands"]):
            frame_data.append((f["frame_idx"], hi, hand))
    
    # Limit for testing
    if args.max_frames:
        frame_data = frame_data[:args.max_frames]
    
    print(f"Optimizing {len(frame_data)} hand observations...", flush=True)
    
    # Extract initial parameters
    # Per-frame: global_orient(3) + hand_pose(45) + translation(3) = 51
    # Track-level: betas(10) + log_scale(1) = 11
    n = len(frame_data)
    n_per_frame = 51
    n_track = 11
    
    def unpack(x):
        track = x[:n_track]
        betas = track[:10]
        log_scale = track[10:11]
        frames = x[n_track:].reshape(n, n_per_frame)
        global_orients = frames[:, :3]
        hand_poses = frames[:, 3:48]
        translations = frames[:, 48:51]
        return betas, log_scale, global_orients, hand_poses, translations
    
    # Initialize from WiLoR
    x0 = np.zeros(n_track + n * n_per_frame, dtype=np.float64)
    
    # Track-level init
    all_betas = []
    for _, _, hand in frame_data:
        mp = hand.get("mano_params", {})
        b = np.array(mp.get("betas", [0]*10)).flatten()[:10]
        if len(b) == 10:
            all_betas.append(b)
    if all_betas:
        x0[:10] = np.median(all_betas, axis=0)
    x0[10] = np.log(1.04)  # initial scale from bone-length refit
    
    # Per-frame init
    for i, (fidx, hi, hand) in enumerate(frame_data):
        mp = hand.get("mano_params", {})
        go = np.array(mp.get("global_orient", [[1,0,0],[0,1,0],[0,0,1]])).reshape(1, 3, 3)[0]
        from scipy.spatial.transform import Rotation
        gov = Rotation.from_matrix(go).as_rotvec()
        hp = np.array(mp.get("hand_pose", [np.eye(3)]*15)).reshape(15, 3, 3)
        hpv = np.array([Rotation.from_matrix(r).as_rotvec() for r in hp]).flatten()
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric", [0,0,1.5])
        
        off = n_track + i * n_per_frame
        x0[off:off+3] = gov
        x0[off+3:off+48] = hpv
        x0[off+48:off+51] = cam_t
    
    print("Initial parameters set", flush=True)
    
    # Precompute targets
    targets = []
    for fidx, hi, hand in frame_data:
        joints2d = np.array(hand["joints2d"], dtype=np.float64)
        joints3d_local = np.array(hand["joints3d_camera_metric"], dtype=float)
        intr = hand.get("intrinsics_manifest")
        # Get depth at hand region
        d_intr = depth_intrinsics[fidx]
        src_w = depth_maps.shape[2]
        manifest_w = intr[4] if len(intr) > 4 else 960
        scale_x = src_w / manifest_w
        
        # Sample depth at 2D joint positions
        j2d_src = joints2d.copy()
        if len(j2d_src) > 0 and abs(scale_x - 1.0) > 0.01:
            j2d_src[:, 0] *= scale_x
            j2d_src[:, 1] *= (depth_maps.shape[1] / (intr[5] if len(intr) > 5 else 720))
        
        px = np.clip(j2d_src[:, 0].astype(int), 0, depth_maps.shape[2]-1)
        py = np.clip(j2d_src[:, 1].astype(int), 0, depth_maps.shape[1]-1)
        hand_depths = depth_maps[fidx, py, px]
        valid_d = hand_depths[hand_depths > 0.1]
        median_depth = float(np.median(valid_d)) if len(valid_d) > 5 else 1.5
        
        intr_4 = [intr[0], intr[1], intr[2], intr[3]] if len(intr) >= 4 else [1000, 1000, 480, 360]
        targets.append({
            "joints2d": joints2d,
            "joints3d_local": joints3d_local,
            "intrinsics": intr_4,
            "depth_median": median_depth,
        })
    
    def residuals(x):
        betas, log_scale, global_orients, hand_poses, translations = unpack(x)
        scale = np.exp(log_scale[0])
        res = []
        
        # Track-level: shape prior
        res.append(betas / 2.0)  # regularize betas
        # Scale prior (target ~1.04)
        res.append(np.array([(log_scale[0] - np.log(1.04)) / 0.1]))
        
        for i in range(n):
            t = targets[i]
            go = global_orients[i]
            hp = hand_poses[i].reshape(15, 3)
            trans = translations[i]
            
            # Convert rotvecs to rotation matrices for MANO
            from scipy.spatial.transform import Rotation
            go_mat = Rotation.from_rotvec(go).as_matrix().reshape(1, 1, 3, 3)
            hp_mat = np.array([Rotation.from_rotvec(r).as_matrix() for r in hp]).reshape(1, 15, 3, 3)
            
            # Run MANO forward (batch 1)
            with torch.no_grad():
                out = model(
                    global_orient=torch.tensor(go_mat, dtype=torch.float32, device="cuda"),
                    hand_pose=torch.tensor(hp_mat, dtype=torch.float32, device="cuda"),
                    betas=torch.tensor(betas[None], dtype=torch.float32, device="cuda"),
                    return_verts=True,
                    pose2rot=False,
                )
                joints = out.joints[0]
            
            joints_local = joints.cpu().numpy() * scale
            joints_cam = joints_local + trans
            
            # 2D reprojection
            j2d_pred = project(joints_cam, t["intrinsics"])
            j2d_res = (j2d_pred - t["joints2d"]).flatten() / 5.0  # 5px sigma
            res.append(j2d_res[:42])  # 21 joints × 2
            
            # Depth alignment
            depth_res = np.array([(trans[2] - t["depth_median"]) / 0.05])  # 5cm sigma
            res.append(depth_res)
            
            # Bone-length prior
            bone = hand_bone_scale(joints_local)
            bone_res = np.array([(bone - 0.171) / 0.02])  # 2cm sigma
            res.append(bone_res)
        
        # Temporal smoothness
        for i in range(1, n):
            # Only smooth consecutive frames from same track
            prev_fidx = frame_data[i-1][0]
            curr_fidx = frame_data[i][0]
            if curr_fidx - prev_fidx <= 3:  # within 3 frames
                trans_diff = (translations[i] - translations[i-1]) / 0.02  # 2cm sigma
                res.append(trans_diff)
                pose_diff = (hand_poses[i] - hand_poses[i-1]) / 0.3  # 0.3 rad sigma
                res.append(pose_diff)
        
        return np.concatenate([r.flatten() for r in res]).astype(np.float64)
    
    # Run optimization
    print("Starting optimization...", flush=True)
    started = time.time()
    
    # Use scipy least_squares
    result = least_squares(
        residuals, x0, max_nfev=int(args.iters),
        method="trf", loss="soft_l1", f_scale=1.0,
        verbose=2 if args.verbose else 0,
    )
    
    elapsed = time.time() - started
    print(f"Optimization done in {elapsed:.1f}s, nfev={result.nfev}", flush=True)
    
    # Build output
    betas, log_scale, global_orients, hand_poses, translations = unpack(result.x)
    scale = float(np.exp(log_scale[0]))
    
    optimized_hands = []
    for i, (fidx, hi, hand) in enumerate(frame_data):
        optimized_hands.append({
            "frame_idx": fidx,
            "hand_idx": hi,
            "side": hand["side"],
            "global_orient_rotvec": global_orients[i].tolist(),
            "hand_pose_rotvec": hand_poses[i].tolist(),
            "translation_m": translations[i].tolist(),
            "betas": betas.tolist(),
            "scale": scale,
            "depth_at_hand_m": targets[i]["depth_median"],
        })
    
    output_dir = run_root / "measurements/hand_candidates/v21_active_mano"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "schema": "v21_active_mano_optimized.v0",
        "method": "v21_active_mano_optimizer",
        "betas_track": betas.tolist(),
        "scale_track": scale,
        "total_observations": n,
        "optimizer": {
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "residual_rms": float(np.sqrt(np.mean(result.fun**2))),
        },
        "elapsed_s": elapsed,
        "hands": optimized_hands,
    }
    (output_dir / "optimized_mano_state.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ["method", "total_observations", "optimizer", "scale_track"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)
