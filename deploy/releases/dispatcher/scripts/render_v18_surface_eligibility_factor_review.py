#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Render representative surface-eligibility factor states over raw frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources, project_points  # noqa: E402
from build_v18_surface_eligibility_factor import (  # noqa: E402
    STATE_FREE,
    STATE_HAND_QUARANTINED,
    STATE_HIDDEN,
    STATE_OBSERVED,
    STATE_OUTSIDE,
    STATE_UNRESOLVED,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    load_mesh,
    pose_map,
    write_json,
)

COLORS = {
    STATE_OBSERVED: (0, 220, 0),
    STATE_HAND_QUARANTINED: (0, 220, 255),
    STATE_FREE: (0, 0, 255),
    STATE_HIDDEN: (220, 0, 220),
    STATE_OUTSIDE: (128, 128, 128),
    STATE_UNRESOLVED: (180, 180, 0),
}
DRAW_ORDER = [STATE_HIDDEN, STATE_FREE, STATE_UNRESOLVED, STATE_HAND_QUARANTINED, STATE_OBSERVED]


def frames_by_id(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(f["frame_idx"]): f for f in as_list(annotations.get("frames")) if isinstance(f, dict) and f.get("frame_idx") is not None}


def project_face_centers(frame: dict[str, Any], vertices: np.ndarray, faces: np.ndarray, pose: tuple[np.ndarray, np.ndarray], intrinsics: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    centers_obj = vertices[faces].mean(axis=1)
    r_obj, t_obj = pose
    world = centers_obj @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    r_c2w, t_c2w = frame_camera_pose(frame)
    cam = (world - t_c2w[None, :]) @ r_c2w
    h, w = shape
    u, v, valid = project_points(cam, intrinsics, w, h)
    return np.column_stack([u, v]), valid


def render(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(args.surface_eligibility_factor_report)
    inputs = report.get("inputs", {}) if isinstance(report, dict) else {}
    annotations_path = args.annotations or Path(str(inputs.get("annotations")))
    pose_path = args.pose_report or Path(str(inputs.get("pose_report")))
    mesh_path = args.completed_mesh or Path(str(inputs.get("completed_mesh")))
    annotations = load_json(annotations_path)
    poses = pose_map(load_json(pose_path))
    frames = frames_by_id(annotations)
    mesh = load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    depth_rows = load_depth_sources([args.depth_npz])
    rows = [r for r in as_list(report.get("factor_rows")) if isinstance(r, dict)]
    row_map = {(int(r["frame_idx"]), str(r["hand_side"])): r for r in rows if r.get("frame_idx") is not None and r.get("hand_side") is not None}
    frame_ids = args.frames or sorted({int(r["frame_idx"]) for r in rows})[: min(6, len(rows))]
    sides = args.sides or sorted({str(r["hand_side"]) for r in rows})
    out_root = args.output_root / str(report.get("case", "case"))
    out_root.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    for frame_idx in frame_ids:
        frame = frames.get(int(frame_idx))
        pose = poses.get(int(frame_idx))
        if frame is None or pose is None:
            continue
        raw = frame.get("raw_frame_path")
        img = cv2.imread(str(raw), cv2.IMREAD_COLOR) if raw else None
        if img is None:
            continue
        depth_row = depth_rows.get(int(frame_idx))
        if depth_row is None:
            continue
        uv, valid = project_face_centers(frame, vertices, faces, pose, np.asarray(depth_row["intrinsics"], dtype=float), img.shape[:2])
        for side in sides:
            row = row_map.get((int(frame_idx), side))
            if row is None:
                continue
            npz_path = Path(str(row.get("face_state_npz_path")))
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                masks = {state: np.asarray(data[state], dtype=bool) if state in data else np.zeros(len(faces), dtype=bool) for state in COLORS}
            canvas = img.copy()
            for state in DRAW_ORDER:
                ids = np.where(masks[state] & valid)[0]
                if args.max_points_per_state > 0 and ids.size > args.max_points_per_state:
                    step = int(np.ceil(ids.size / args.max_points_per_state))
                    ids = ids[::step]
                color = COLORS[state]
                pts = uv[ids].astype(int)
                for u, v in pts:
                    cv2.circle(canvas, (int(u), int(v)), int(args.radius_px), color, -1, lineType=cv2.LINE_AA)
            legend_y = 24
            cv2.putText(canvas, f"surface eligibility frame {frame_idx} {side}", (16, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            legend_y += 28
            for state in [STATE_OBSERVED, STATE_HAND_QUARANTINED, STATE_FREE, STATE_HIDDEN, STATE_UNRESOLVED, STATE_OUTSIDE]:
                color = COLORS[state]
                count = int(row.get("counts", {}).get(state, int(np.count_nonzero(masks[state]))))
                cv2.putText(canvas, f"{state}: {count}", (16, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
                legend_y += 22
            out_path = out_root / "review_frames" / f"{int(frame_idx):06d}_{side}_surface_eligibility.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), canvas)
            rendered.append({"frame_idx": int(frame_idx), "hand_side": side, "path": str(out_path)})
    manifest = {"method": "render_v18_surface_eligibility_factor_review", "surface_eligibility_factor_report": str(args.surface_eligibility_factor_report), "rendered": rendered}
    out_manifest = out_root / "v18_surface_eligibility_review_manifest.json"
    write_json(out_manifest, manifest)
    print(json.dumps({"manifest": str(out_manifest), "rendered_count": len(rendered)}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface-eligibility-factor-report", type=Path, required=True)
    p.add_argument("--depth-npz", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--pose-report", type=Path, default=None)
    p.add_argument("--completed-mesh", type=Path, default=None)
    p.add_argument("--frames", type=int, nargs="*", default=None)
    p.add_argument("--sides", nargs="*", default=None)
    p.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_surface_eligibility_factor_review_v1"))
    p.add_argument("--max-points-per-state", type=int, default=4500)
    p.add_argument("--radius-px", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    render(parse_args())
