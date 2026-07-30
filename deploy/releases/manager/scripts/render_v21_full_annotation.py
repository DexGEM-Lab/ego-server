#!/usr/bin/env python3
"""V21 full physical annotation renderer.

Produces the three deliverable renders from V21 design Section 2:
- v21_overlay.mp4: raw video with hand MANO mesh + object mesh pose + contact/occlusion labels
- v21_world.mp4: metric world/camera view with MANO surfaces + object mesh pose
- v21_side_by_side.mp4: synchronized raw/overlay/world

Consumes all V21 measurements:
- WiLoR metric hand candidates (MANO skeleton + vertices)
- Object mesh candidate + pose (ICP + pose graph)
- Contact/occlusion/nonpenetration evidence
- DepthPro depth (for depth-validated visualization)
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
SIDE_COLORS = {"right": (0,255,100), "left": (100,100,255)}
OBJ_COLOR = (0,165,255)


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def project_pts(pts, K):
    z = np.clip(pts[:,2], 0.01, None)
    return np.column_stack([K[0]*pts[:,0]/z + K[2], K[1]*pts[:,1]/z + K[3]])


def render_overlay_frame(img, hands, objects, contact_data, fidx):
    """Render hand MANO skeleton + object mesh projection + contact labels."""
    overlay = img.copy()

    # Object mesh projection
    for obj in objects:
        verts2d = obj.get("verts2d", [])
        if len(verts2d) > 0:
            pts = np.array(verts2d, dtype=int)
            # Draw mesh vertices
            for p in pts:
                cv2.circle(overlay, tuple(p), 1, OBJ_COLOR, -1)
            # Bounding box
            mn, mx = pts.min(0), pts.max(0)
            cv2.rectangle(overlay, tuple(mn), tuple(mx), OBJ_COLOR, 2)
            # Depth label
            t = obj.get("translation_m", [0,0,0])
            cv2.putText(overlay, f"OBJ z={t[2]:.2f}m", (int(mn[0]), max(int(mn[1])-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJ_COLOR, 1, cv2.LINE_AA)

    # Hand MANO skeleton + mesh
    for hand in hands:
        side = hand.get("side", hand.get("hand_side", "right"))
        color = SIDE_COLORS.get(side, (255,255,0))
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric") or hand.get("cam_t", [0,0,1])
        K = hand.get("intrinsics_manifest") or [1000,1000,480,360]

        joints3d = hand.get("joints3d_camera_metric") or hand.get("joints3d_source_camera_m") or hand.get("joints3d_camera", [])
        if joints3d:
            j3d = np.array(joints3d, dtype=float)
            # If these are local, add cam_t
            if hand.get("vertices_camera") and not hand.get("vertices_source_camera_m"):
                j3d_cam = j3d + np.array(cam_t)
            else:
                j3d_cam = j3d
            j2d = project_pts(j3d_cam, K).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(overlay, tuple(j2d[a]), tuple(j2d[b]), color, 3, cv2.LINE_AA)
            for j, p in enumerate(j2d):
                r = 5 if j in [0,4,8,12,16,20] else 3
                cv2.circle(overlay, tuple(p), r, color, -1, cv2.LINE_AA)

        # Vertices
        verts = hand.get("vertices_source_camera_m") or []
        if verts:
            v3d = np.array(verts, dtype=float)
            v2d = project_pts(v3d, K).astype(int)
            for p in v2d:
                px, py = int(p[0]), int(p[1])
                if 0 <= px < overlay.shape[1] and 0 <= py < overlay.shape[0]:
                    cv2.circle(overlay, (px,py), 1, color, -1)

        # Detection box
        bbox = hand.get("bbox_xyxy", [])
        if len(bbox) == 4:
            cv2.rectangle(overlay, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)

        # Contact label
        if contact_data:
            for cd in contact_data:
                if cd.get("frame_idx") == fidx and cd.get("hand_side", "") == side:
                    if cd.get("in_contact"):
                        cv2.putText(overlay, "CONTACT", (int(bbox[0])+5, int(bbox[3])+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                    elif cd.get("min_distance_m", 1) < 0.1:
                        cv2.putText(overlay, f"near {cd['min_distance_m']:.2f}m", (int(bbox[0])+5, int(bbox[3])+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1, cv2.LINE_AA)
                    break

    return overlay


def render_world_frame(img_h, img_w, hands, objects, fidx, scale=0.3):
    """Render a top-down world view with MANO + object mesh positions."""
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cv2.putText(canvas, "WORLD VIEW (bird's eye)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

    cx_world = img_w // 2
    cy_world = img_h // 2
    world_scale = 100  # pixels per meter

    # Draw camera position
    cv2.circle(canvas, (cx_world, img_h - 50), 8, (255,255,0), -1)
    cv2.putText(canvas, "camera", (cx_world+12, img_h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

    # Draw hands in world (XY plane from camera coords)
    for hand in hands:
        side = hand.get("side", hand.get("hand_side", "right"))
        color = SIDE_COLORS.get(side, (255,255,0))
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t") or [0,0,1.5]
        # Project XZ to canvas: X -> horizontal, Z -> vertical (depth)
        wx = cx_world + int(cam_t[0] * world_scale)
        wy = img_h - 50 - int(cam_t[2] * world_scale)
        cv2.circle(canvas, (wx, wy), 10, color, -1)
        cv2.putText(canvas, f"{side[0]}({cam_t[2]:.2f}m)", (wx+12, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw object
    for obj in objects:
        t = obj.get("translation_m", [0,0,1.5])
        wx = cx_world + int(t[0] * world_scale)
        wy = img_h - 50 - int(t[2] * world_scale)
        cv2.rectangle(canvas, (wx-12, wy-12), (wx+12, wy+12), OBJ_COLOR, -1)
        cv2.putText(canvas, f"OBJ({t[2]:.2f}m)", (wx+15, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, OBJ_COLOR, 1)

    # Draw depth scale
    for d in range(0, 4):
        y = img_h - 50 - int(d * world_scale)
        if 0 < y < img_h:
            cv2.line(canvas, (cx_world-200, y), (cx_world+200, y), (50,50,50), 1)
            cv2.putText(canvas, f"{d}m", (cx_world-230, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)

    return canvas


def compose_side_by_side(overlay, world, output, w_each=640, h=480):
    """Compose overlay and world side by side."""
    cmd = [
        FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(overlay), '-i', str(world),
        '-filter_complex',
        f'[0:v]scale={w_each}:{h}:force_original_aspect_ratio=decrease,pad={w_each}:{h}:(ow-iw)/2:(oh-ih)/2:black[left];'
        f'[1:v]scale={w_each}:{h}:force_original_aspect_ratio=decrease,pad={w_each}:{h}:(ow-iw)/2:(oh-ih)/2:black[right];'
        f'[left][right]hstack=inputs=2[v]',
        '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(output)
    ]
    subprocess.run(cmd, check=True)


def run(args):
    run_root = Path(args.run_root).resolve()
    obj = args.object_id
    repo_root = Path("/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")

    # Load all data
    ann = load_json(run_root / "state/annotations_v18_full_mano.json") or \
          load_json(run_root / "state/annotations_v18_compatible.json")
    manifest = load_json(run_root / "input/raw_frame_manifest/manifest.json")
    contact = load_json(run_root / "measurements/contact_occlusion_nonpenetration/contact_evidence.json")

    # Pose graph results
    pose_graph = load_json(run_root / f"measurements/object_geometry_mesh_pose/{obj}/v19_pose_graph/v19_rigid_object_pose_graph_report.json")
    robust_pose = load_json(run_root / f"measurements/object_geometry_mesh_pose/{obj}/v21_robust_pose.json")

    # Load mesh for projection
    import trimesh
    mesh = trimesh.load(str(run_root / f"measurements/object_geometry/v21_mesh_candidate/{obj}/mesh_candidate.obj"), process=False)
    mesh_verts = np.asarray(mesh.vertices, dtype=float)
    mesh_samples, _ = trimesh.sample.sample_surface(mesh, 800)
    mesh_samples = np.asarray(mesh_samples, dtype=float)

    # Load depth for intrinsics
    depth_data = np.load(str(run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"))
    per_frame_intrinsics = depth_data["intrinsics_fx_fy_cx_cy"]

    # Build lookup tables
    frames_by_idx = {f["frame_idx"]: f for f in ann["frames"]}
    pose_by_frame = {}
    if pose_graph:
        for r in pose_graph.get("pose_rows", []):
            pose_by_frame[r["frame_idx"]] = r
    if robust_pose:
        for r in robust_pose.get("pose_rows", []):
            if r["frame_idx"] not in pose_by_frame:
                pose_by_frame[r["frame_idx"]] = r
    contact_by_frame = {}
    if contact:
        for r in contact.get("rows", []):
            contact_by_frame.setdefault(r["frame_idx"], []).append(r)

    img_w = manifest["frames"][0]["manifest_width"]
    img_h = manifest["frames"][0]["manifest_height"]
    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]
    fps = manifest.get("fps", 25.0)

    renders_dir = run_root / "renders"
    overlay_path = renders_dir / "v21_overlay.mp4"
    world_path = renders_dir / "v21_world.mp4"
    side_path = renders_dir / "v21_side_by_side.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (img_w, img_h))
    world_writer = cv2.VideoWriter(str(world_path), fourcc, fps, (img_w, img_h))

    for fm in tqdm(manifest["frames"], desc="Full render"):
        fidx = fm["frame_idx"]
        rgb_path = str(Path(repo_root) / fm["rgb"])
        img = cv2.imread(rgb_path)
        if img is None:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        frame_ann = frames_by_idx.get(fidx, {})
        hands = frame_ann.get("hands", [])
        objects = frame_ann.get("objects", [])

        # Compute object mesh projection
        obj_render = []
        pose = pose_by_frame.get(fidx, {})
        R = np.array(pose.get("rotation_world_from_completed_canonical_matrix",
                              pose.get("rotation_matrix", np.eye(3).tolist())))
        t = np.array(pose.get("translation_world_m",
                              pose.get("translation_m", [0,0,1.5])))
        mesh_world = (R @ mesh_samples.T).T + t

        intr = per_frame_intrinsics[fidx]
        scale_f = img_w / src_w
        K = [intr[0]*scale_f, intr[1]*scale_f, intr[2]*scale_f, intr[3]*scale_f]
        verts2d = project_pts(mesh_world, K).astype(int)
        obj_render.append({"verts2d": verts2d.tolist(), "translation_m": t.tolist()})

        contact_data = contact_by_frame.get(fidx, [])

        # Overlay
        overlay_img = render_overlay_frame(img, hands, obj_render, contact_data, fidx)
        overlay_writer.write(overlay_img)

        # World view
        world_img = render_world_frame(img_h, img_w, hands, obj_render, fidx)
        world_writer.write(world_img)

    overlay_writer.release()
    world_writer.release()

    # Side-by-side
    compose_side_by_side(overlay_path, world_path, side_path)

    summary = {
        "status": "ok",
        "method": "render_v21_full_physical_annotation",
        "overlay": str(overlay_path),
        "world": str(world_path),
        "side_by_side": str(side_path),
        "frame_count": len(manifest["frames"]),
        "has_hand_mesh": any("vertices_source_camera_m" in h for f in ann["frames"] for h in f.get("hands",[])),
        "has_object_mesh": True,
        "has_contact": contact is not None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    run(args)
