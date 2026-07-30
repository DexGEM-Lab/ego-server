#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
import torch


STATUS = "v17_object_material_track_measurement"
CLAIM = (
    "CoTracker point tracks over one V17 object RGB/mask/depth dataset provide candidate material "
    "correspondences. This is tracking evidence only; object geometry, object pose, and V3 solver "
    "closure remain false until a rigid/deformable motion test accepts the correspondences."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def summarize(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def annotations_by_frame(path: Path) -> dict[int, dict[str, Any]]:
    payload = require_dict(load_json(path), f"{path}")
    frames = require_list(payload.get("frames"), f"{path}.frames")
    out: dict[int, dict[str, Any]] = {}
    for i, raw in enumerate(frames):
        row = require_dict(raw, f"{path}.frames[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"{path}.frames[{i}].frame_idx")
        out[frame_idx] = row
    if len(out) != len(frames):
        raise RuntimeError(f"annotation file has duplicate frame ids: {path}")
    return out


def selected_manifest_frames(manifest_path: Path, first: int, last: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = require_dict(load_json(manifest_path), f"{manifest_path}")
    frames = require_list(manifest.get("frames"), f"{manifest_path}.frames")
    selected: list[dict[str, Any]] = []
    for i, raw in enumerate(frames):
        row = require_dict(raw, f"{manifest_path}.frames[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"{manifest_path}.frames[{i}].frame_idx")
        if first <= frame_idx <= last:
            selected.append(row)
    actual = [require_int(row.get("frame_idx"), "selected frame_idx") for row in selected]
    expected = list(range(first, last + 1))
    if actual != expected:
        raise RuntimeError(f"selected object dataset window is not dense: expected {expected}, got {actual}")
    return manifest, selected


def read_frames(entries: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: list[list[float]] = []
    for entry in entries:
        frame_idx = require_int(entry.get("frame_idx"), "entry.frame_idx")
        image_path = Path(require_str(entry.get("rgb"), f"frame {frame_idx}.rgb"))
        mask_path = Path(require_str(entry.get("mask"), f"frame {frame_idx}.mask"))
        depth_path = Path(require_str(entry.get("depth"), f"frame {frame_idx}.depth"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read RGB frame: {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask frame: {mask_path}")
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f"failed to read depth frame: {depth_path}")
        if depth_mm.ndim != 2:
            raise RuntimeError(f"depth frame is not single-channel: {depth_path}")
        if mask.shape[:2] != image.shape[:2] or depth_mm.shape[:2] != image.shape[:2]:
            raise RuntimeError(
                f"image/mask/depth shape mismatch for frame {frame_idx}: "
                f"{image.shape[:2]} {mask.shape[:2]} {depth_mm.shape[:2]}"
            )
        raw_intrinsics = require_list(entry.get("intrinsics_fx_fy_cx_cy"), f"frame {frame_idx}.intrinsics_fx_fy_cx_cy")
        if len(raw_intrinsics) != 4:
            raise RuntimeError(f"frame {frame_idx} intrinsics must have four values")
        intr = [finite_float(value, f"frame {frame_idx} intrinsics[{j}]") for j, value in enumerate(raw_intrinsics)]
        if intr[0] <= 0.0 or intr[1] <= 0.0:
            raise RuntimeError(f"frame {frame_idx} has nonpositive focal length")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks.append(mask > 0)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m[(depth_m <= 0.0) | ~np.isfinite(depth_m)] = np.nan
        depths.append(depth_m)
        intrinsics.append(intr)
    return (
        np.stack(images, axis=0),
        np.stack(masks, axis=0),
        np.stack(depths, axis=0),
        np.asarray(intrinsics, dtype=np.float64),
    )


def sample_query_points(mask: np.ndarray, grid_step_px: int, max_points: int) -> np.ndarray:
    if grid_step_px <= 0:
        raise RuntimeError("grid_step_px must be positive")
    if max_points <= 0:
        raise RuntimeError("max_points must be positive")
    ys, xs = np.mgrid[0 : mask.shape[0] : grid_step_px, 0 : mask.shape[1] : grid_step_px]
    candidates = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    keep = mask[candidates[:, 1].astype(np.int64), candidates[:, 0].astype(np.int64)]
    candidates = candidates[keep]
    if candidates.size == 0:
        raise RuntimeError("no query points lie inside the query mask")
    if len(candidates) <= max_points:
        return candidates
    chosen = [int(np.argmin(candidates[:, 0] + candidates[:, 1]))]
    dist2 = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
    while len(chosen) < max_points:
        idx = int(np.argmax(dist2))
        chosen.append(idx)
        dist2 = np.minimum(dist2, np.sum((candidates - candidates[idx]) ** 2, axis=1))
    return candidates[np.asarray(chosen, dtype=np.int64)]


def sample_depth_nearest(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (0 <= x) & (x < depth.shape[1]) & (0 <= y) & (y < depth.shape[0])
    out = np.full((xy.shape[0],), np.nan, dtype=np.float32)
    out[valid] = depth[y[valid], x[valid]]
    out[(out <= 0.0) | ~np.isfinite(out)] = np.nan
    return out


def mask_hit(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (0 <= x) & (x < mask.shape[1]) & (0 <= y) & (y < mask.shape[0])
    out = np.zeros((xy.shape[0],), dtype=bool)
    out[valid] = mask[y[valid], x[valid]]
    return out


def world_points(xy: np.ndarray, z: np.ndarray, intrinsics: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    camera = require_dict(frame.get("camera"), "annotation frame.camera")
    fx, fy, cx, cy = [float(v) for v in intrinsics.tolist()]
    cam = np.full((len(xy), 3), np.nan, dtype=np.float64)
    valid = np.isfinite(z)
    cam[valid, 0] = (xy[valid, 0] - cx) / fx * z[valid]
    cam[valid, 1] = (xy[valid, 1] - cy) / fy * z[valid]
    cam[valid, 2] = z[valid]
    T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise RuntimeError("annotation camera pose must be a finite 4x4 T_world_camera_metric")
    world = np.full_like(cam, np.nan)
    world[valid] = cam[valid] @ T[:3, :3].T + T[:3, 3]
    return world


def draw_tracks(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    tracks: np.ndarray,
    accepted: np.ndarray,
    frame_i: int,
    frame_idx: int,
    object_label: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 60, 255), 2, cv2.LINE_AA)
    palette = np.array(
        [
            [255, 80, 60],
            [60, 180, 255],
            [80, 230, 110],
            [210, 110, 255],
            [255, 210, 70],
            [70, 255, 220],
        ],
        dtype=np.uint8,
    )
    for track_i in range(tracks.shape[1]):
        if not accepted[frame_i, track_i]:
            continue
        color = tuple(int(v) for v in palette[track_i % len(palette)].tolist())
        pts = []
        for t in range(frame_i + 1):
            if accepted[t, track_i]:
                pts.append(tuple(np.rint(tracks[t, track_i]).astype(int).tolist()))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 3, color, -1, cv2.LINE_AA)
    label = f"V17 material-track QC {object_label} frame {frame_idx}"
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, entries = selected_manifest_frames(args.manifest, int(args.frame_start), int(args.frame_end))
    annotations = annotations_by_frame(args.annotations)
    frame_idx = [require_int(entry.get("frame_idx"), "selected frame_idx") for entry in entries]
    missing_annotations = [idx for idx in frame_idx if idx not in annotations]
    if missing_annotations:
        raise RuntimeError(f"annotations missing selected frames: {missing_annotations[:10]}")
    images, masks, depths, intrinsics = read_frames(entries)
    query_i = int(args.query_frame_index)
    if query_i < 0 or query_i >= len(frame_idx):
        raise RuntimeError("query frame index outside selected window")
    query_points = sample_query_points(masks[query_i], int(args.grid_step_px), int(args.max_points))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("V17 object material tracking requires CUDA")
    video = torch.from_numpy(images).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.zeros((1, len(query_points), 3), dtype=torch.float32, device=device)
    queries[0, :, 0] = float(query_i)
    queries[0, :, 1:] = torch.from_numpy(query_points).to(device)
    model: Any = torch.hub.load(
        str(args.torchhub_repo),
        str(args.torchhub_model),
        source=str(args.torchhub_source),
        trust_repo=True,
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=bool(args.backward_tracking))
    tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)

    if tracks.shape[:2] != visibility.shape:
        raise RuntimeError("CoTracker track and visibility shapes disagree")
    if tracks.shape[0] != len(frame_idx) or tracks.shape[1] != len(query_points):
        raise RuntimeError("CoTracker output shape does not match selected window/query points")
    mask_hits = np.stack([mask_hit(mask, tracks[i]) for i, mask in enumerate(masks)], axis=0)
    sampled_depth = np.stack([sample_depth_nearest(depths[i], tracks[i]) for i in range(len(frame_idx))], axis=0)
    depth_valid = np.isfinite(sampled_depth)
    accepted = visibility & mask_hits & depth_valid
    world = np.stack(
        [world_points(tracks[i], sampled_depth[i], intrinsics[i], annotations[frame]) for i, frame in enumerate(frame_idx)],
        axis=0,
    )
    valid_frames_per_track = accepted.sum(axis=0)
    all_frame_tracks = valid_frames_per_track == len(frame_idx)
    consecutive = accepted[:-1] & accepted[1:]
    world_step_m = np.linalg.norm(world[1:] - world[:-1], axis=2)[consecutive]
    flow_px = np.linalg.norm(tracks[1:] - tracks[:-1], axis=2)[visibility[:-1] & visibility[1:]]

    rows: list[dict[str, Any]] = []
    for i, raw_frame in enumerate(frame_idx):
        visible = visibility[i]
        visible_mask = visible & mask_hits[i]
        rows.append(
            {
                "frame_idx": int(raw_frame),
                "visible_fraction": float(np.mean(visible)),
                "mask_hit_fraction_of_visible": float(np.mean(mask_hits[i][visible])) if np.any(visible) else 0.0,
                "depth_valid_fraction_of_visible_mask": float(np.mean(depth_valid[i][visible_mask])) if np.any(visible_mask) else 0.0,
                "accepted_track_count": int(np.count_nonzero(accepted[i])),
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_id = require_str(manifest.get("object_id"), "manifest.object_id")
    track_id = str(manifest.get("track_id") or object_id.removeprefix("object:"))
    npz_path = args.output_dir / "v17_object_material_tracks.npz"
    np.savez_compressed(
        npz_path,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        tracks_xy=tracks,
        visibility=visibility,
        mask_hits=mask_hits,
        depth_m=sampled_depth,
        accepted=accepted,
        query_xy=query_points,
        world_xyz=world,
        intrinsics_fx_fy_cx_cy=intrinsics.astype(np.float32),
        annotation_ready=np.asarray(False),
        object_geometry_complete=np.asarray(False),
        object_pose_requirement_met=np.asarray(False),
        v3_solver_complete=np.asarray(False),
    )
    h, w = images.shape[1:3]
    video_path = args.output_dir / "qc_v17_object_material_tracks.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.output_fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {video_path}")
    stills_dir = args.output_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    still_set = {int(frame) for frame in args.still_frames}
    for i, raw_frame in enumerate(frame_idx):
        rendered = draw_tracks(images[i], masks[i], tracks, accepted, i, raw_frame, track_id)
        writer.write(rendered)
        if raw_frame in still_set:
            still_path = stills_dir / f"frame_{raw_frame:06d}.jpg"
            if not cv2.imwrite(str(still_path), rendered):
                raise RuntimeError(f"failed to write still: {still_path}")
    writer.release()

    report = {
        "status": STATUS,
        "claim": CLAIM,
        "method": "run_v17_object_material_tracks",
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "v3_solver_complete": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "diagnostic_only": True,
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "object_id": object_id,
        "track_id": track_id,
        "frames": [int(frame) for frame in frame_idx],
        "frame_count": len(frame_idx),
        "query_frame_index": query_i,
        "query_frame_idx": int(frame_idx[query_i]),
        "query_points": int(len(query_points)),
        "all_frame_accepted_tracks": int(np.count_nonzero(all_frame_tracks)),
        "valid_frames_per_track": summarize(valid_frames_per_track),
        "world_step_m": summarize(world_step_m),
        "flow_px": summarize(flow_px),
        "rows": rows,
        "outputs": {
            "tracks_npz": str(npz_path),
            "qc_overlay_video": str(video_path),
            "stills_dir": str(stills_dir),
        },
        "parameters": {
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "grid_step_px": int(args.grid_step_px),
            "max_points": int(args.max_points),
            "torchhub_repo": str(args.torchhub_repo),
            "torchhub_model": str(args.torchhub_model),
            "torchhub_source": str(args.torchhub_source),
            "backward_tracking": bool(args.backward_tracking),
            "output_fps": float(args.output_fps),
            "device": device,
        },
    }
    report_path = args.output_dir / "v17_object_material_track_report.json"
    write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--query-frame-index", type=int, default=0)
    parser.add_argument("--grid-step-px", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--output-fps", type=float, default=6.0)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[])
    parser.add_argument("--torchhub-repo", default="facebookresearch/co-tracker")
    parser.add_argument("--torchhub-model", default="cotracker3_offline")
    parser.add_argument("--torchhub-source", default="github")
    parser.add_argument("--backward-tracking", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
