#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Render observed-surface MANO constraint separation.

Colors the compact-rigid object by depth provenance per frame:
  green   = object vertices supported by observed metric depth;
  magenta = object vertices that lie in free space in front of observed depth;
  gray    = hidden/behind/out-of-frame/unvalidated vertices.

It overlays current MANO (orange) and articulated candidate MANO (cyan, when
available) plus the observed-surface MANO state. This is a visual diagnostic of
which object geometry is physically eligible to constrain H_{t,h}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import (  # noqa: E402
    VERTEX_BEHIND_OBSERVED,
    VERTEX_FREE_SPACE_CONFLICT,
    VERTEX_OBSERVED_SUPPORTED,
    classify_object_vertices_against_depth,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    load_mesh,
    pose_map,
)
from render_v18_compact_rigid_tomato_temporal_mano_attempt import (  # noqa: E402
    HAND_EDGES,
    apply_temporal_hypothesis,
    first_intrinsics,
    project_camera_points,
    world_points_to_camera,
)

DEFAULT_OUTPUT_ROOT = Path("/data2/ego_annotation_outputs/v18_task5_observed_surface_mano_constraint_render_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-label", default="observed/quarantined tomato")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True)
    parser.add_argument("--temporal-mano-state", type=Path, required=True)
    parser.add_argument("--observed-surface-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mesh-projection-stride", type=int, default=8)
    parser.add_argument("--world-mesh-stride", type=int, default=8)
    parser.add_argument("--support-margin-m", type=float, default=0.015)
    parser.add_argument("--free-space-margin-m", type=float, default=0.025)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_temporal(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    data = load_json(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in data.get("per_frame_states", []) if isinstance(data, dict) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def load_observed_state(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    data = load_json(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in data.get("per_frame_states", []) if isinstance(data, dict) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out, data


def frame_camera_transform(frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    transform = np.asarray(camera.get("T_world_camera_metric", np.eye(4)), dtype=float)
    if transform.shape != (4, 4):
        return np.eye(4)
    return transform


def draw_skeleton_camera(
    image: np.ndarray,
    joints_camera: np.ndarray,
    intr: tuple[float, float, float, float],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = image.shape[:2]
    u, v, valid = project_camera_points(joints_camera, intr, w, h)
    for a, b in HAND_EDGES:
        if valid[a] and valid[b]:
            cv2.line(image, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), color, thickness)


def world_bounds_for_frame(frame: dict[str, Any], object_world: np.ndarray | None, padding: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    if object_world is not None and len(object_world):
        chunks.append(object_world[:: max(1, len(object_world) // 600)])
    for hand in as_list(frame.get("hands")):
        if not isinstance(hand, dict):
            continue
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        joints = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
        if joints.shape == (21, 3):
            chunks.append(joints)
    if not chunks:
        return np.asarray([-1, -1, -1], dtype=float), np.asarray([1, 1, 1], dtype=float)
    pts = np.vstack(chunks)
    return pts.min(axis=0) - padding, pts.max(axis=0) + padding


def world_to_screen(point: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    extent = np.maximum(max_xyz - min_xyz, 1.0e-6)
    x = int(round((point[0] - min_xyz[0]) / extent[0] * width))
    y = int(round(height - (point[2] - min_xyz[2]) / extent[2] * height))
    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def draw_world_skeleton(image: np.ndarray, joints_world: np.ndarray, min_xyz: np.ndarray, max_xyz: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    h, w = image.shape[:2]
    for a, b in HAND_EDGES:
        pa = world_to_screen(joints_world[a], min_xyz, max_xyz, w, h)
        pb = world_to_screen(joints_world[b], min_xyz, max_xyz, w, h)
        if pa is not None and pb is not None:
            cv2.line(image, pa, pb, color, thickness)


def class_color(cls: int) -> tuple[int, int, int]:
    if cls == VERTEX_OBSERVED_SUPPORTED:
        return (40, 255, 80)
    if cls == VERTEX_FREE_SPACE_CONFLICT:
        return (255, 0, 255)
    if cls == VERTEX_BEHIND_OBSERVED:
        return (150, 150, 150)
    return (80, 80, 80)


def state_color(state: str) -> tuple[int, int, int]:
    if "blocked" in state:
        return (0, 0, 255)
    if "compatible" in state or "clear" in state:
        return (0, 220, 255)
    if "hidden" in state:
        return (255, 0, 255)
    return (180, 180, 180)


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
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
            str(output_path),
        ],
        check=True,
    )


def render(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    mesh = load_mesh(args.completed_mesh)
    object_vertices = np.asarray(mesh.vertices, dtype=float)
    depth_by_frame = load_depth_sources(args.depth_npz)
    temporal = load_temporal(args.temporal_mano_state)
    observed, observed_payload = load_observed_state(args.observed_surface_state)
    frames = as_list(annotations.get("frames"))
    if args.max_frames is not None:
        frames = frames[: int(args.max_frames)]
    fps = float((annotations.get("raw_video") or {}).get("fps", 30.0))

    case_dir = args.output_root / str(args.case)
    overlay_dir = case_dir / "overlay_frames"
    world_dir = case_dir / "world_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    world_dir.mkdir(parents=True, exist_ok=True)

    for frame_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", frame_i))
        raw_path = Path(frame.get("raw_frame_path", ""))
        overlay = cv2.imread(str(raw_path)) if raw_path.exists() else None
        if overlay is None:
            overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
        h, w = overlay.shape[:2]
        T_world_camera = frame_camera_transform(frame)
        intr = first_intrinsics(frame, w, h)
        pose = poses.get(frame_idx)
        classes = None
        object_world = None
        if pose is not None:
            classes, object_depth_row = classify_object_vertices_against_depth(
                frame=frame,
                vertices_object=object_vertices,
                pose=pose,
                depth_row=depth_by_frame.get(frame_idx),
                support_margin_m=float(args.support_margin_m),
                free_space_margin_m=float(args.free_space_margin_m),
            )
            rot, trans = pose
            object_world = object_vertices @ rot.T + trans[None, :]
            object_camera = (object_world - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]
            uu, vv, valid = project_camera_points(object_camera, intr, w, h)
            stride = max(1, int(args.mesh_projection_stride))
            for cls in [VERTEX_BEHIND_OBSERVED, VERTEX_FREE_SPACE_CONFLICT, VERTEX_OBSERVED_SUPPORTED]:
                idx = np.where((classes == cls) & valid)[0][::stride]
                color = class_color(cls)
                for x, y in zip(uu[idx], vv[idx]):
                    cv2.circle(overlay, (int(x), int(y)), 1, color, -1)
            counts = object_depth_row.get("vertex_class_counts", {})
            object_text = (
                f"{args.object_label} observed={counts.get('observed_supported', 0)} "
                f"free={counts.get('free_space_conflict', 0)} hidden/behind={counts.get('behind_observed', 0)}"
            )
        else:
            object_text = f"{args.object_label} pose missing"
        cv2.putText(overlay, object_text[:150], (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(overlay, "green=observed depth surface  magenta=free-space conflict  gray=hidden/behind", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        for hand_idx, hand in enumerate(as_list(frame.get("hands"))):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            row = observed.get((frame_idx, side), {})
            temporal_row = temporal.get((frame_idx, side))
            obs_state = str(row.get("observed_surface_mano_state", "not_measured"))
            text_color = state_color(obs_state)
            label_y = 88 + hand_idx * 122
            cand_measure = row.get("candidate_full_778_measurement") if isinstance(row.get("candidate_full_778_measurement"), dict) else {}
            cand_obs = cand_measure.get("observed_supported_strict_penetration_m") if isinstance(cand_measure, dict) else {}
            cand_full = cand_measure.get("max_penetration_m") if isinstance(cand_measure, dict) else None
            cand_obs_max = cand_obs.get("max") if isinstance(cand_obs, dict) else None
            cv2.putText(overlay, f"{side} observed-surface MANO: {obs_state[:62]}", (12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 2)
            cv2.putText(overlay, f"candidate full max={cand_full if cand_full is not None else '?'} obs-supported max={cand_obs_max if cand_obs_max is not None else '?'}", (12, label_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
            joints_cam = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
            intr_hand = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if joints_cam.shape == (21, 3) and isinstance(intr_hand, list) and len(intr_hand) == 4:
                intr_tuple = tuple(float(x) for x in intr_hand)  # type: ignore[assignment]
                draw_skeleton_camera(overlay, joints_cam, intr_tuple, (0, 140, 255), 6)
                draw_skeleton_camera(overlay, joints_cam, intr_tuple, (0, 210, 255), 3)
                if temporal_row is not None:
                    joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
                    if joints_world.shape == (21, 3):
                        candidate_world = apply_temporal_hypothesis(joints_world, temporal_row)
                        if candidate_world is not None:
                            candidate_cam = world_points_to_camera(candidate_world, T_world_camera)
                            draw_skeleton_camera(overlay, candidate_cam, intr_tuple, (255, 255, 0), 2)
                    candidate_vertices = np.asarray(temporal_row.get("optimized_vertices_world_sample_m") or [], dtype=float)
                    if candidate_vertices.ndim == 2 and candidate_vertices.shape[1] == 3 and len(candidate_vertices):
                        candidate_cam = world_points_to_camera(candidate_vertices, T_world_camera)
                        cu, cv, cvalid = project_camera_points(candidate_cam, intr_tuple, w, h)
                        for x, y in zip(cu[cvalid], cv[cvalid]):
                            cv2.circle(overlay, (int(x), int(y)), 1, (255, 255, 0), -1)
        cv2.imwrite(str(overlay_dir / f"{frame_idx:06d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

        world = np.zeros((720, 1280, 3), dtype=np.uint8)
        min_xyz, max_xyz = world_bounds_for_frame(frame, object_world)
        if object_world is not None and classes is not None:
            for cls in [VERTEX_BEHIND_OBSERVED, VERTEX_FREE_SPACE_CONFLICT, VERTEX_OBSERVED_SUPPORTED]:
                idx = np.where(classes == cls)[0][:: max(1, int(args.world_mesh_stride))]
                color = class_color(cls)
                for vertex in object_world[idx]:
                    pt = world_to_screen(vertex, min_xyz, max_xyz, 1280, 720)
                    if pt is not None:
                        cv2.circle(world, pt, 1, color, -1)
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints_world = np.asarray(metric.get("joints_current_v18_world_m") or [], dtype=float)
            if joints_world.shape == (21, 3):
                draw_world_skeleton(world, joints_world, min_xyz, max_xyz, (0, 210, 255), 3)
                temporal_row = temporal.get((frame_idx, side))
                if temporal_row is not None:
                    candidate_world = apply_temporal_hypothesis(joints_world, temporal_row)
                    if candidate_world is not None:
                        draw_world_skeleton(world, candidate_world, min_xyz, max_xyz, (255, 255, 0), 2)
                    candidate_vertices = np.asarray(temporal_row.get("optimized_vertices_world_sample_m") or [], dtype=float)
                    if candidate_vertices.ndim == 2 and candidate_vertices.shape[1] == 3 and len(candidate_vertices):
                        for vertex in candidate_vertices:
                            pt = world_to_screen(vertex, min_xyz, max_xyz, 1280, 720)
                            if pt is not None:
                                cv2.circle(world, pt, 1, (255, 255, 0), -1)
        cv2.putText(world, f"observed-surface world frame {frame_idx:04d}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(world_dir / f"{frame_idx:06d}.jpg"), world, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if frame_idx % 120 == 0:
            print(f"rendered frame {frame_idx}/{len(frames)}", flush=True)

    overlay_video = case_dir / "v18_overlay_observed_surface_mano_constraints.mp4"
    world_video = case_dir / "v18_world_observed_surface_mano_constraints.mp4"
    side_video = case_dir / "v18_side_by_side_observed_surface_mano_constraints.mp4"
    encode_video(overlay_dir, overlay_video, fps)
    encode_video(world_dir, world_video, fps)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(overlay_video),
            "-i",
            str(world_video),
            "-filter_complex",
            "[0:v]scale=960:540[left];[1:v]scale=960:540[right];[left][right]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(side_video),
        ],
        check=True,
    )
    summary = {
        "case": str(args.case),
        "output_root": str(case_dir),
        "overlay_video": str(overlay_video),
        "world_video": str(world_video),
        "side_by_side_video": str(side_video),
        "frame_count": int(len(frames)),
        "fps": fps,
        "observed_surface_state_summary": observed_payload.get("summary") if isinstance(observed_payload, dict) else None,
        "runtime_s": time.time() - started,
    }
    (case_dir / "render_v18_observed_surface_mano_constraint_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
