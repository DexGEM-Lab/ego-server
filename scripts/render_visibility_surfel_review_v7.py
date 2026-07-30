#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_json


def load_surfel_npz(path: Path) -> dict[str, np.ndarray]:
    blob = np.load(path)
    required = {"node_id", "frame_idx", "track_id", "measured_position_m", "solved_position_m"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    return {key: blob[key] for key in required}


def project_points(points_world: np.ndarray, annotation: dict, args: argparse.Namespace, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
    camera_xyz = camera_points(points_world, T_world_camera)
    z = camera_xyz[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    positive = z > 0.0
    if np.any(positive):
        K = intrinsics_for_frame(args, entry, annotation)
        uv[positive, 0] = K[0, 0] * camera_xyz[positive, 0] / z[positive] + K[0, 2]
        uv[positive, 1] = K[1, 1] * camera_xyz[positive, 1] / z[positive] + K[1, 2]
    return uv, z


def draw_points(image: np.ndarray, uv: np.ndarray, z: np.ndarray, color: tuple[int, int, int], radius: int) -> int:
    height, width = image.shape[:2]
    drawn = 0
    for (x, y), depth in zip(uv, z, strict=True):
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(depth) or depth <= 0.0:
            continue
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if 0 <= ix < width and 0 <= iy < height:
            cv2.circle(image, (ix, iy), int(radius), color, -1, cv2.LINE_AA)
            drawn += 1
    return drawn


def draw_review(rgb: np.ndarray, mask: np.ndarray, measured_uv: np.ndarray, measured_z: np.ndarray, solved_uv: np.ndarray, solved_z: np.ndarray, frame_idx: int, row: dict, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    image = rgb.copy()
    overlay = image.copy()
    overlay[mask] = (40, 170, 255)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)
    measured_count = draw_points(image, measured_uv, measured_z, (0, 230, 255), int(args.point_radius_px))
    solved_count = draw_points(image, solved_uv, solved_z, (40, 255, 60), int(args.point_radius_px))
    for a, b in zip(measured_uv, solved_uv, strict=True):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        pa = tuple(np.rint(a).astype(int).tolist())
        pb = tuple(np.rint(b).astype(int).tolist())
        if pa == pb:
            continue
        cv2.line(image, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
    text = f"frame {frame_idx}  surfels {len(row['node_ids'])}  measured yellow  solved green"
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image, {"measured_points_drawn": int(measured_count), "solved_points_drawn": int(solved_count)}


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    entries = manifest.get("frames")
    frames = annotations.get("frames")
    if not isinstance(entries, list) or not isinstance(frames, list):
        raise RuntimeError("manifest and annotations must contain frames lists")
    entry_by_frame = {int(entry["frame_idx"]): entry for entry in entries}
    annotation_by_frame = {int(frame["frame_idx"]): frame for frame in frames}
    surfels = load_surfel_npz(args.surfel_npz)
    frame_idx = surfels["frame_idx"].astype(int)
    measured = surfels["measured_position_m"].astype(np.float64)
    solved = surfels["solved_position_m"].astype(np.float64)
    node_id = surfels["node_id"].astype(int)
    frames_to_render = sorted({int(idx) for idx in frame_idx if int(args.frame_start) <= int(idx) <= int(args.frame_end)})
    if not frames_to_render:
        raise RuntimeError("no surfel frames selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    writer = None
    rows = []
    for idx in frames_to_render:
        if idx not in entry_by_frame or idx not in annotation_by_frame:
            raise RuntimeError(f"missing manifest or annotation frame {idx}")
        entry = entry_by_frame[idx]
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise RuntimeError(f"failed to read RGB/mask for frame {idx}")
        take = frame_idx == idx
        measured_uv, measured_z = project_points(measured[take], annotation_by_frame[idx], args, entry)
        solved_uv, solved_z = project_points(solved[take], annotation_by_frame[idx], args, entry)
        disp = np.linalg.norm(solved[take] - measured[take], axis=1)
        row = {
            "frame_idx": int(idx),
            "node_ids": node_id[take].astype(int).tolist(),
            "node_count": int(np.count_nonzero(take)),
            "correction_displacement_median_m": float(np.median(disp)) if len(disp) else None,
            "correction_displacement_p95_m": float(np.percentile(disp, 95.0)) if len(disp) else None,
        }
        rendered, draw_row = draw_review(rgb, mask > 0, measured_uv, measured_z, solved_uv, solved_z, idx, row, args)
        row.update(draw_row)
        if args.render_width and rendered.shape[1] != int(args.render_width):
            height = int(round(int(args.render_width) * rendered.shape[0] / rendered.shape[1]))
            rendered = cv2.resize(rendered, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.output_dir / "visibility_surfel_review_v7.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (rendered.shape[1], rendered.shape[0]),
            )
        writer.write(rendered)
        if idx in set(args.still_frames):
            cv2.imwrite(str(still_dir / f"frame_{idx:06d}.png"), rendered)
        rows.append(row)
    if writer is not None:
        writer.release()
    report = {
        "status": "ok",
        "method": "render_visibility_surfel_review_v7",
        "surfel_npz": str(args.surfel_npz),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "frames": int(len(rows)),
        "video": str(args.output_dir / "visibility_surfel_review_v7.mp4"),
        "stills_dir": str(still_dir),
        "rows": rows,
    }
    (args.output_dir / "review_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surfel-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="annotation-vggt")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--point-radius-px", type=int, default=3)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
