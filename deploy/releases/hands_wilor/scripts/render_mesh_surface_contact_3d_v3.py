#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_camera(points_world: np.ndarray, t_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_world, np.ones(len(points_world), dtype=float)]
    return (np.linalg.inv(t_world_camera) @ homog.T).T[:, :3]


def hand_vertices_camera(hand: dict, t_world_camera: np.ndarray) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        arr = np.asarray(hand.get(key, []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    for key in ("vertices_world_m", "vertices_world_m_sample"):
        arr = np.asarray(hand.get(key, []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return to_camera(arr, t_world_camera)
    raise RuntimeError("hand has no MANO vertices in camera or world coordinates")


def hand_joints_camera(hand: dict, t_world_camera: np.ndarray) -> np.ndarray:
    arr = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    if arr.shape == (21, 3):
        return arr
    arr = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
    if arr.shape == (21, 3):
        return to_camera(arr, t_world_camera)
    raise RuntimeError("hand has no 21x3 joints in camera or world coordinates")


def reliable_rows(contact: dict) -> list[dict]:
    rows = [row for row in contact.get("rows_detail", []) if bool(row.get("reliable_for_contact", False))]
    return sorted(rows, key=lambda row: (int(row["frame_idx"]), int(row["hand_idx"])))


def equal_axes(ax, points: np.ndarray, radius_m: float) -> None:
    center = np.median(points, axis=0)
    ax.set_xlim(center[0] - radius_m, center[0] + radius_m)
    ax.set_ylim(center[1] - radius_m, center[1] + radius_m)
    ax.set_zlim(center[2] - radius_m, center[2] + radius_m)
    ax.set_box_aspect((1, 1, 1))


def draw_camera_frustum(ax, origin: np.ndarray, scale: float, label: str = "head camera") -> None:
    near = scale
    width = 0.65 * scale
    height = 0.40 * scale
    origin = np.asarray(origin, dtype=float)
    corners = np.asarray(
        [
            [-width, -height, near],
            [width, -height, near],
            [width, height, near],
            [-width, height, near],
        ],
        dtype=float,
    ) + origin[None, :]
    for corner in corners:
        ax.plot([origin[0], corner[0]], [origin[1], corner[1]], [origin[2], corner[2]], color="#222222", lw=1.8)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#222222", lw=2.0)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="#111111", s=42, depthshade=False)
    ax.text(origin[0], origin[1] - 0.08 * scale, origin[2] - 0.06 * scale, label, color="#111111", fontsize=9)


def simplify_mesh_for_display(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= max_faces:
        return vertices, faces
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    out_vertices = np.asarray(mesh.vertices, dtype=float)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    if out_vertices.ndim != 2 or out_vertices.shape[1] != 3 or out_faces.ndim != 2 or out_faces.shape[1] != 3:
        raise RuntimeError("display mesh simplification produced invalid geometry")
    if len(out_faces) == 0:
        raise RuntimeError("display mesh simplification produced no faces")
    return out_vertices, out_faces


def draw_mesh(ax, vertices: np.ndarray, faces: np.ndarray) -> None:
    tris = vertices[faces]
    collection = Poly3DCollection(tris, facecolors="#d95b59", edgecolors="#9d2f2f", linewidths=0.04, alpha=0.86)
    ax.add_collection3d(collection)


def draw_hand(ax, joints: np.ndarray, contact_vertices: np.ndarray) -> None:
    for a, b in HAND_EDGES:
        pts = joints[[a, b]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#18a64a", lw=3.0)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="#35d165", s=16, depthshade=False)
    ax.scatter(
        contact_vertices[:, 0],
        contact_vertices[:, 1],
        contact_vertices[:, 2],
        color="#ffd400",
        edgecolors="#111111",
        s=54,
        depthshade=False,
    )


def set_clean_axis(ax) -> None:
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=67, azim=-123)


def annotate_render(image: np.ndarray, row: dict, depth_m: float) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (255, 255, 255), -1)
    title = (
        f"frame {int(row['frame_idx'])}   {row['side']} hand mesh-surface contact   "
        f"surface p95 {row['best_patch_distance_p95_m'] * 1000.0:.1f} mm   "
        f"signed p95 {row['best_patch_signed_gap_p95_abs_m'] * 1000.0:.1f} mm"
    )
    cv2.putText(out, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    y = out.shape[0] - 38
    cv2.line(out, (34, y), (164, y), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(out, "0.10 m", (52, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)
    note = f"local 3D contact view in camera coordinates; head camera is {depth_m:.2f} m along the viewing direction"
    cv2.putText(out, note, (204, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
    return out


def render_frame_3d(frame_idx: int, ann: dict, mesh: tuple[np.ndarray, np.ndarray], row: dict, args: argparse.Namespace) -> np.ndarray:
    vertices_world, faces = mesh
    t_world_camera = np.asarray(ann["camera"]["T_world_camera_metric"], dtype=float)
    object_camera = to_camera(vertices_world, t_world_camera)
    object_camera_display, faces_display = simplify_mesh_for_display(object_camera, faces, int(args.max_faces))
    hand = ann["hands"][int(row["hand_idx"])]
    joints = hand_joints_camera(hand, t_world_camera)
    vertices = hand_vertices_camera(hand, t_world_camera)
    ids = np.asarray(row["best_patch_vertex_ids"], dtype=int)
    contact_vertices = vertices[ids]

    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_mesh(ax, object_camera_display, faces_display)
    draw_hand(ax, joints, contact_vertices)
    object_center = np.median(object_camera, axis=0)
    camera_marker = object_center + np.asarray([0.0, -0.16 * float(args.view_radius_m), -0.52 * float(args.view_radius_m)])
    draw_camera_frustum(ax, camera_marker, float(args.frustum_scale_m), "head camera")
    visible = np.vstack([object_camera, joints, contact_vertices, camera_marker[None, :]])
    equal_axes(ax, visible, float(args.view_radius_m))
    set_clean_axis(ax)
    title = (
        f"frame {frame_idx}  {row['side']} hand contact  "
        f"surface p95 {row['best_patch_distance_p95_m'] * 1000.0:.1f} mm  "
        f"signed p95 {row['best_patch_signed_gap_p95_abs_m'] * 1000.0:.1f} mm"
    )
    ax.set_title(title, fontsize=0, pad=0)
    fig.tight_layout(pad=0.0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return annotate_render(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), row, float(np.median(object_camera[:, 2])))


def read_overlay(overlay_dir: Path, frame_idx: int) -> np.ndarray:
    path = overlay_dir / "stills" / f"frame_{frame_idx:06d}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"missing overlay still {path}")
    return image


def side_by_side(overlay: np.ndarray, view3d: np.ndarray, caption: str, width: int) -> np.ndarray:
    half = width // 2
    height = int(round(half * 9 / 16))
    left = cv2.resize(overlay, (half, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(view3d, (width - half, height), interpolation=cv2.INTER_AREA)
    canvas = np.hstack([left, right])
    bar = np.zeros((54, width, 3), dtype=np.uint8)
    cv2.putText(bar, caption, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def run(args: argparse.Namespace) -> dict:
    annotations = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    meshes = load_mesh_archive(args.object_mesh_npz)
    contact = load_json(args.contact_report)
    rows = reliable_rows(contact)
    if not rows:
        raise RuntimeError("contact report has no reliable temporal contact rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills_3d"
    side_dir = args.output_dir / "stills_side_by_side"
    still_dir.mkdir(exist_ok=True)
    side_dir.mkdir(exist_ok=True)
    video_path = args.output_dir / "mesh_surface_contact_3d.mp4"
    side_video_path = args.output_dir / "mesh_surface_contact_side_by_side.mp4"
    writer_3d = None
    writer_side = None
    rendered = []
    try:
        for row in rows:
            frame_idx = int(row["frame_idx"])
            if frame_idx not in annotations:
                raise RuntimeError(f"annotations missing frame {frame_idx}")
            if frame_idx not in meshes:
                raise RuntimeError(f"mesh archive missing frame {frame_idx}")
            ann = annotations[frame_idx]
            view3d = render_frame_3d(frame_idx, ann, meshes[frame_idx], row, args)
            still_path = still_dir / f"frame_{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(still_path), view3d, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write {still_path}")
            if writer_3d is None:
                writer_3d = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (view3d.shape[1], view3d.shape[0]))
                if not writer_3d.isOpened():
                    raise RuntimeError(f"failed to open {video_path}")
            writer_3d.write(view3d)
            overlay = read_overlay(args.overlay_dir, frame_idx)
            caption = str(ann.get("caption", "")).strip()
            if not caption:
                raise RuntimeError(f"frame {frame_idx} has no semantic caption")
            paired = side_by_side(overlay, view3d, caption, int(args.side_by_side_width))
            side_path = side_dir / f"frame_{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(side_path), paired, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write {side_path}")
            if writer_side is None:
                writer_side = cv2.VideoWriter(
                    str(side_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (paired.shape[1], paired.shape[0]),
                )
                if not writer_side.isOpened():
                    raise RuntimeError(f"failed to open {side_video_path}")
            writer_side.write(paired)
            rendered.append({"frame_idx": frame_idx, "view3d": str(still_path), "side_by_side": str(side_path)})
    finally:
        if writer_3d is not None:
            writer_3d.release()
        if writer_side is not None:
            writer_side.release()
    report = {
        "status": "ok",
        "method": "render_mesh_surface_contact_3d_v3",
        "frames": [int(row["frame_idx"]) for row in rows],
        "video_3d": str(video_path),
        "video_side_by_side": str(side_video_path),
        "stills_3d": str(still_dir),
        "stills_side_by_side": str(side_dir),
        "rendered": rendered,
        "annotations": str(args.annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "contact_report": str(args.contact_report),
        "overlay_dir": str(args.overlay_dir),
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--view-radius-m", type=float, default=0.18)
    parser.add_argument("--frustum-scale-m", type=float, default=0.045)
    parser.add_argument("--max-faces", type=int, default=1200)
    parser.add_argument("--side-by-side-width", type=int, default=1920)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
