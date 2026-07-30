#!/usr/bin/env python3
"""V21 dual-pane renderer: skeleton overlay + world 3D point cloud.

Pane 1 (left): Original video + thin MANO skeleton + semi-transparent MANO vertex point cloud
Pane 2 (right): World coordinate 3D point cloud mesh (hand vertices + object mesh at pose)
Side-by-side video output.

This matches the V18 deliverable format.
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

SIDE_COLORS_RGB = {"right": (100,255,0), "left": (100,100,255)}


def project_pts(pts_3d, K):
    z = np.clip(pts_3d[:, 2], 0.01, None)
    return np.column_stack([K[0]*pts_3d[:,0]/z + K[2], K[1]*pts_3d[:,1]/z + K[3]])


def render_overlay_pane(img, hands_data, obj_verts2d):
    """Pane 1: raw video + thin skeleton + semi-transparent MANO vertices."""
    # Work with PIL for semi-transparent drawing
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, 'RGBA')
    
    # Draw object mesh vertices as semi-transparent dots
    if obj_verts2d is not None and len(obj_verts2d) > 0:
        for pt in obj_verts2d:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < pil.width and 0 <= py < pil.height:
                draw.ellipse((px-2, py-2, px+2, py+2), fill=(255, 165, 0, 80))
    
    for hand in hands_data:
        side = hand.get("side", hand.get("hand_side", "right"))
        color_rgb = SIDE_COLORS_RGB.get(side, (255, 255, 0))
        color_rgba = (*color_rgb, 255)
        color_transparent = (*color_rgb, 60)
        
        # Get camera-space vertices and joints
        verts = hand.get("vertices_source_camera_m", [])
        joints = hand.get("joints3d_source_camera_m", hand.get("joints3d_source_camera_m", []))
        
        cam_t = hand.get("cam_t", hand.get("mano_candidate", {}).get("cam_t", [0,0,1]))
        K = hand.get("source_intrinsics", hand.get("mano_candidate", {}).get("source_intrinsics", [1000,1000,img.shape[1]/2,img.shape[0]/2]))
        if len(K) > 4:
            K = K[:4]
        
        # Project MANO vertices to 2D
        if verts:
            v3d = np.array(verts, dtype=float)
            v2d = project_pts(v3d, K).astype(int)
            # Draw as semi-transparent point cloud
            for pt in v2d:
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < pil.width and 0 <= py < pil.height:
                    draw.ellipse((px-1, py-1, px+1, py+1), fill=color_transparent)
        
        # Draw thin skeleton
        if joints and len(joints) >= 21:
            j3d = np.array(joints, dtype=float)
            j2d = project_pts(j3d, K).astype(int)
            for a, b in HAND_EDGES:
                if a < len(j2d) and b < len(j2d):
                    pa = tuple(j2d[a])
                    pb = tuple(j2d[b])
                    draw.line((pa[0], pa[1], pb[0], pb[1]), fill=color_rgba, width=1)
            for pt in j2d:
                draw.ellipse((pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2), fill=color_rgba)
    
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render_world_pane(canvas_w, canvas_h, hands_data, obj_verts_world):
    """Pane 2: world coordinate 3D point cloud (bird's-eye view)."""
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    
    # Title
    cv2.putText(canvas, "World 3D View (XZ plane, depth)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    
    cx = canvas_w // 2
    cy_bottom = canvas_h - 60
    scale = 80  # pixels per meter
    
    # Draw camera
    cv2.circle(canvas, (cx, cy_bottom), 8, (255, 255, 0), -1)
    cv2.putText(canvas, "CAM", (cx + 12, cy_bottom + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Draw depth grid lines
    for d in range(0, 6):
        y = cy_bottom - int(d * scale)
        if 0 < y < canvas_h:
            cv2.line(canvas, (cx - 300, y), (cx + 300, y), (40, 40, 40), 1)
            cv2.putText(canvas, f"{d}m", (cx - 330, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    # Draw object point cloud in world
    if obj_verts_world is not None and len(obj_verts_world) > 0:
        for v in obj_verts_world:
            wx = cx + int(v[0] * scale)
            wy = cy_bottom - int(v[2] * scale)
            if 0 <= wx < canvas_w and 0 <= wy < canvas_h:
                cv2.circle(canvas, (wx, wy), 1, (0, 165, 255), -1)
    
    # Draw hand point cloud in world
    for hand in hands_data:
        side = hand.get("side", hand.get("hand_side", "right"))
        color_bgr = (0, 255, 100) if side == "right" else (255, 100, 100)
        
        verts = hand.get("vertices_source_camera_m", [])
        if verts:
            for v in verts:
                wx = cx + int(float(v[0]) * scale)
                wy = cy_bottom - int(float(v[2]) * scale)
                if 0 <= wx < canvas_w and 0 <= wy < canvas_h:
                    cv2.circle(canvas, (wx, wy), 1, color_bgr, -1)
    
    return canvas


def run(args):
    run_root = Path(args.run_root).resolve()
    obj = args.object_id
    repo_root = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master"
    
    # Load data
    manifest = json.loads((run_root / "input/raw_frame_manifest/manifest.json").read_text())
    
    ann_path = run_root / "state/annotations_v18_full_mano.json"
    ann = json.loads(ann_path.read_text())
    frames_by_idx = {f["frame_idx"]: f for f in ann["frames"]}
    
    # Depth for intrinsics
    depth_data = np.load(str(run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"))
    per_frame_intrinsics = depth_data["intrinsics_fx_fy_cx_cy"]
    
    # Object mesh + pose
    mesh_path = run_root / f"measurements/object_geometry/v21_mesh_candidate/{obj}/mesh_candidate.obj"
    if mesh_path.exists() and trimesh:
        mesh = trimesh.load(str(mesh_path), process=False)
        mesh_samples, _ = trimesh.sample.sample_surface(mesh, 500)
        mesh_samples = np.asarray(mesh_samples, dtype=float)
    else:
        mesh_samples = None
    
    pose_path = run_root / f"measurements/object_geometry_mesh_pose/{obj}/v19_pose_graph/v19_rigid_object_pose_graph_report.json"
    pose_by_frame = {}
    if pose_path.exists():
        pg = json.loads(pose_path.read_text())
        for r in pg.get("pose_rows", []):
            pose_by_frame[r["frame_idx"]] = r
    
    robust_path = run_root / f"measurements/object_geometry_mesh_pose/{obj}/v21_robust_pose.json"
    if robust_path.exists():
        rp = json.loads(robust_path.read_text())
        for r in rp.get("pose_rows", []):
            if r["frame_idx"] not in pose_by_frame:
                pose_by_frame[r["frame_idx"]] = r
    
    img_w = manifest["frames"][0]["manifest_width"]
    img_h = manifest["frames"][0]["manifest_height"]
    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]
    fps = manifest.get("fps", 25.0)
    
    # Output
    output_dir = run_root / "renders"
    output_dir.mkdir(exist_ok=True)
    overlay_path = output_dir / "v21_overlay.mp4"
    world_path = output_dir / "v21_world.mp4"
    sidebyside_path = output_dir / "v21_side_by_side.mp4"
    
    frame_dir = run_root / "logs/dual_pane_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    for fm in tqdm(manifest["frames"], desc="Dual-pane render"):
        fidx = fm["frame_idx"]
        rgb_path = str(Path(repo_root) / fm["rgb"])
        img = cv2.imread(rgb_path)
        if img is None:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        frame_ann = frames_by_idx.get(fidx, {})
        hands = frame_ann.get("hands", [])
        
        # Object pose
        pose = pose_by_frame.get(fidx, {})
        R = np.array(pose.get("rotation_world_from_completed_canonical_matrix",
                              pose.get("rotation_matrix", np.eye(3).tolist())))
        t = np.array(pose.get("translation_world_m",
                              pose.get("translation_m", [0,0,1.5])))
        
        # Object mesh in world
        obj_verts_world = None
        obj_verts2d = None
        if mesh_samples is not None:
            obj_verts_world = (R @ mesh_samples.T).T + t
            intr = per_frame_intrinsics[fidx]
            scale_f = img_w / src_w
            K = [intr[0]*scale_f, intr[1]*scale_f, intr[2]*scale_f, intr[3]*scale_f]
            obj_verts2d = project_pts(obj_verts_world, K)
        
        # Pane 1: overlay
        overlay = render_overlay_pane(img, hands, obj_verts2d)
        
        # Pane 2: world
        world = render_world_pane(img_w, img_h, hands, obj_verts_world)
        
        # Side by side
        sbs = np.hstack([overlay, world])
        cv2.imwrite(str(frame_dir / f"{fidx:06d}.jpg"), sbs, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Encode videos
    for name, path in [("side_by_side", sidebyside_path), ("overlay", overlay_path), ("world", world_path)]:
        if name == "side_by_side":
            input_pattern = str(frame_dir / "%06d.jpg")
        elif name == "overlay":
            # Re-encode overlay only from left half
            overlay_dir = run_root / f"logs/{name}_frames"
            overlay_dir.mkdir(exist_ok=True)
            for f in sorted(frame_dir.glob("*.jpg")):
                sbs_img = cv2.imread(str(f))
                left = sbs_img[:, :img_w]
                cv2.imwrite(str(overlay_dir / f.name), left)
            input_pattern = str(overlay_dir / "%06d.jpg")
        else:
            world_dir = run_root / f"logs/{name}_frames"
            world_dir.mkdir(exist_ok=True)
            for f in sorted(frame_dir.glob("*.jpg")):
                sbs_img = cv2.imread(str(f))
                right = sbs_img[:, img_w:]
                cv2.imwrite(str(world_dir / f.name), right)
            input_pattern = str(world_dir / "%06d.jpg")
        
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-i", input_pattern,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
               "-r", str(fps), str(path)]
        subprocess.run(cmd, check=True)
    
    summary = {
        "status": "ok",
        "method": "render_v21_dual_pane",
        "overlay": str(overlay_path),
        "world": str(world_path),
        "side_by_side": str(sidebyside_path),
        "frame_count": len(manifest["frames"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    run(args)
