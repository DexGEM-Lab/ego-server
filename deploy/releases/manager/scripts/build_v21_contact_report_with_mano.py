#!/usr/bin/env python3
"""V21 → V18 contact report adapter with full MANO vertices.

Generates the contact report format that optimize_contact_patch_object_pose_graph_v3.py
expects, including:
- Full 778-vertex MANO hands (via MANO forward from WiLoR params)
- Contact patch vertex IDs (hand vertices nearest to object mesh)
- rows_detail with reliable_for_contact flags

Also patches the V18 annotation to include full 778-vertex
vertices_source_camera_m for each hand.

Output:
  measurements/contact_occlusion_nonpenetration/v18_contact_report.json
  state/annotations_v18_full_mano.json (patched with 778-vertex hands)
"""
from __future__ import annotations
import argparse, json, os, sys, inspect
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

WILOR_DIR = Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model")

def patch_legacy_imports():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for n, v in {"bool":bool,"int":int,"float":float,"complex":complex,"object":object,"unicode":str,"str":str}.items():
        if not hasattr(np, n): setattr(np, n, v)
    raw = torch.load
    def compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return raw(*a, **kw)
    torch.load = compat


def load_json(p): return json.loads(Path(p).read_text())


def run_mano_forward(global_orient_mat, hand_pose_mats, betas, side, model):
    """Run MANO forward pass to get 778 vertices and 21 joints."""
    go = torch.tensor(global_orient_mat.reshape(1, 1, 3, 3), dtype=torch.float32, device="cuda")
    hp = torch.tensor(hand_pose_mats.reshape(1, 15, 3, 3), dtype=torch.float32, device="cuda")
    bt = torch.tensor(betas.reshape(1, 10), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        out = model(global_orient=go, hand_pose=hp, betas=bt, return_verts=True, pose2rot=False)
    
    vertices = out.vertices[0].cpu().numpy().astype(np.float64)
    joints = out.joints[0].cpu().numpy().astype(np.float64)
    
    # Apply side sign
    sign = 1.0 if side == "right" else -1.0
    vertices[:, 0] *= sign
    joints[:, 0] *= sign
    
    return vertices, joints


def run(args):
    run_root = Path(args.run_root).resolve()
    obj = args.object_id
    repo_root = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master"
    
    patch_legacy_imports()
    sys.path.insert(0, str(WILOR_DIR))
    os.chdir(WILOR_DIR)
    from wilor.models.mano_wrapper import MANO
    mano_model = MANO(
        model_path=str(WILOR_DIR / "mano_data"),
        is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1,
    ).to("cuda").eval()
    os.chdir(repo_root)
    print("MANO model loaded", flush=True)
    
    # Load annotation
    ann_path = run_root / "state/annotations_v18_compatible.json"
    ann = load_json(ann_path)
    
    # Load mesh for contact distance
    import trimesh
    mesh = trimesh.load(str(run_root / f"measurements/object_geometry/v21_mesh_candidate/{obj}/mesh_candidate.obj"), process=False)
    mesh_verts = np.asarray(mesh.vertices, dtype=np.float64)
    
    # Load pose graph for object pose
    pose_path = run_root / f"measurements/object_geometry_mesh_pose/{obj}/v19_pose_graph/v19_rigid_object_pose_graph_report.json"
    pose_graph = load_json(pose_path) if pose_path.exists() else {"pose_rows": []}
    pose_by_frame = {r["frame_idx"]: r for r in pose_graph.get("pose_rows", [])}
    
    # Load robust pose as fallback
    robust_path = run_root / f"measurements/object_geometry_mesh_pose/{obj}/v21_robust_pose.json"
    robust = load_json(robust_path) if robust_path.exists() else {"pose_rows": []}
    robust_by_frame = {r["frame_idx"]: r for r in robust.get("pose_rows", [])}
    
    contact_rows_detail = []
    frames_processed = 0
    
    for frame in ann["frames"]:
        fidx = frame["frame_idx"]
        hands = frame.get("hands", [])
        objects = frame.get("objects", [])
        if not hands or not objects:
            continue
        
        # Object pose
        pose = pose_by_frame.get(fidx, robust_by_frame.get(fidx, {}))
        R = np.array(pose.get("rotation_world_from_completed_canonical_matrix", 
                              pose.get("rotation_matrix", np.eye(3).tolist())))
        t = np.array(pose.get("translation_world_m",
                              pose.get("translation_m", [0, 0, 1.5])))
        obj_verts_world = (R @ mesh_verts.T).T + t
        obj_tree = cKDTree(obj_verts_world)
        
        for hand_idx, hand in enumerate(hands):
            mp = hand.get("mano_params", {})
            if not mp:
                continue
            
            side = hand.get("side", hand.get("hand_side", "right"))
            cam_t = np.array(hand.get("cam_t", [0, 0, 1.5]))
            
            # Run MANO forward
            go_mat = np.array(mp.get("global_orient", [np.eye(3)]))
            hp_mat = np.array(mp.get("hand_pose", [np.eye(3)] * 15))
            betas = np.array(mp.get("betas", [0] * 10)).flatten()[:10]
            
            try:
                vertices_local, joints_local = run_mano_forward(go_mat, hp_mat, betas, side, mano_model)
            except Exception as e:
                print(f"  Frame {fidx} hand {hand_idx}: MANO forward failed: {e}", flush=True)
                continue
            
            # Scale and translate to camera space
            vertices_cam = vertices_local + cam_t
            joints_cam = joints_local + cam_t
            
            # Update hand in annotation
            hand["vertices_source_camera_m"] = vertices_cam.tolist()
            hand["joints3d_source_camera_m"] = joints_cam.tolist()
            hand["vertices_camera"] = vertices_local.tolist()
            hand["joints3d_camera"] = joints_local.tolist()
            hand["measurement_available"] = True
            hand["hand_idx"] = hand_idx
            
            # Find contact patch: nearest MANO vertices to object mesh
            dists, _ = obj_tree.query(vertices_cam, k=1)
            contact_thresh = 0.03  # 3cm contact threshold
            contact_ids = np.where(dists < contact_thresh)[0]
            
            if len(contact_ids) >= 5:  # at least 5 vertices in contact
                # Select best patch (nearest vertices)
                sorted_ids = contact_ids[np.argsort(dists[contact_ids])][:50]  # top 50
                reliable = True
            else:
                # Use anatomical fallback (fingertips: vertices 745, 317, 444, 556, 673 in MANO)
                sorted_ids = np.array([745, 317, 444, 556, 673, 320, 341, 431, 555, 660])
                reliable = False
            
            contact_rows_detail.append({
                "frame_idx": fidx,
                "hand_idx": hand_idx,
                "track_id": obj,
                "reliable_for_contact": reliable,
                "geometry_backed_temporal_contact": reliable,
                "selected_patch_source": "best_patch" if reliable else "anatomical_patch",
                "best_patch_vertex_ids": sorted_ids.tolist() if reliable else [],
                "anatomical_patch_vertex_ids": [] if reliable else sorted_ids.tolist(),
                "distance_median_m": float(np.median(dists[sorted_ids])),
                "contact_count": int(len(contact_ids)),
            })
        
        frames_processed += 1
        if frames_processed % 50 == 0:
            print(f"  Processed {frames_processed} frames, {len(contact_rows_detail)} contact rows", flush=True)
    
    # Save patched annotation
    patched_ann_path = run_root / "state/annotations_v18_full_mano.json"
    patched_ann_path.write_text(json.dumps(ann, indent=2))
    print(f"Saved full-MANO annotation: {patched_ann_path}", flush=True)
    
    # Save contact report
    reliable_count = sum(1 for r in contact_rows_detail if r["reliable_for_contact"])
    report = {
        "method": "v21_contact_report_with_full_mano",
        "status": "ok",
        "total_rows": len(contact_rows_detail),
        "reliable_contact_rows": reliable_count,
        "rows_detail": contact_rows_detail,
    }
    contact_path = run_root / "measurements/contact_occlusion_nonpenetration/v18_contact_report_full.json"
    contact_path.write_text(json.dumps(report, indent=2))
    print(f"Saved contact report: {contact_path} ({reliable_count} reliable contacts)", flush=True)
    
    return {"status": "ok", "total_rows": len(contact_rows_detail), "reliable": reliable_count}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
