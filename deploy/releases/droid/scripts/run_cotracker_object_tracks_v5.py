#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def summarize(values: np.ndarray) -> dict:
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


def frame_entries(manifest_path: Path, first: int, last: int) -> list[dict]:
    manifest = load_json(manifest_path)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"manifest has no frames list: {manifest_path}")
    selected = [row for row in frames if first <= int(row["frame_idx"]) <= last]
    actual = [int(row["frame_idx"]) for row in selected]
    expected = list(range(first, last + 1))
    if actual != expected:
        raise RuntimeError(f"manifest is not dense: expected {expected}, got {actual}")
    return selected


def load_video_and_masks(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    images = []
    masks = []
    for entry in entries:
        image = cv2.imread(str(entry["rgb"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read RGB frame: {entry['rgb']}")
        mask = cv2.imread(str(entry["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask frame: {entry['mask']}")
        if mask.shape[:2] != image.shape[:2]:
            raise RuntimeError(f"mask/image shape mismatch for frame {entry['frame_idx']}: {mask.shape} vs {image.shape}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks.append(mask > 0)
    return np.stack(images, axis=0), np.stack(masks, axis=0)


def sample_query_points(mask: np.ndarray, grid_step_px: int, max_points: int) -> np.ndarray:
    if grid_step_px <= 0:
        raise RuntimeError("grid_step_px must be positive")
    ys, xs = np.mgrid[0 : mask.shape[0] : grid_step_px, 0 : mask.shape[1] : grid_step_px]
    candidates = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    keep = mask[candidates[:, 1].astype(int), candidates[:, 0].astype(int)]
    candidates = candidates[keep]
    if candidates.size == 0:
        raise RuntimeError("no query points lie inside the query mask")
    if len(candidates) <= max_points:
        return candidates
    # Deterministic farthest-point subsampling in image space keeps coverage explicit.
    chosen = [int(np.argmin(candidates[:, 0] + candidates[:, 1]))]
    dist2 = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
    while len(chosen) < max_points:
        idx = int(np.argmax(dist2))
        chosen.append(idx)
        dist2 = np.minimum(dist2, np.sum((candidates - candidates[idx]) ** 2, axis=1))
    return candidates[np.asarray(chosen, dtype=np.int64)]


def annotations_by_frame(path: Path) -> dict[int, dict]:
    data = load_json(path)
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"annotations has no frames list: {path}")
    return {int(row["frame_idx"]): row for row in frames}


def depth_archive(path: Path) -> tuple[np.ndarray, dict[int, int]]:
    data = np.load(path, allow_pickle=True)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    depths = np.asarray(data["depth"], dtype=np.float32)
    frame_to_i = {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())}
    if len(frame_to_i) != len(frame_idx):
        raise RuntimeError(f"depth archive has duplicate frame ids: {path}")
    return depths, frame_to_i


def sample_depth_nearest(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (0 <= x) & (x < depth.shape[1]) & (0 <= y) & (y < depth.shape[0])
    out = np.full((xy.shape[0],), np.nan, dtype=np.float32)
    out[valid] = depth[y[valid], x[valid]]
    out[(out <= 0) | ~np.isfinite(out)] = np.nan
    return out


def mask_hit(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (0 <= x) & (x < mask.shape[1]) & (0 <= y) & (y < mask.shape[0])
    out = np.zeros((xy.shape[0],), dtype=bool)
    out[valid] = mask[y[valid], x[valid]]
    return out


def world_points(xy: np.ndarray, z: np.ndarray, frame: dict) -> np.ndarray:
    camera = frame["camera"]
    fx, fy, cx, cy = [float(v) for v in camera["vggt_source_intrinsics_fx_fy_cx_cy"]]
    cam = np.full((len(xy), 3), np.nan, dtype=np.float64)
    valid = np.isfinite(z)
    cam[valid, 0] = (xy[valid, 0] - cx) / fx * z[valid]
    cam[valid, 1] = (xy[valid, 1] - cy) / fy * z[valid]
    cam[valid, 2] = z[valid]
    T = np.asarray(camera["T_world_camera_metric"], dtype=np.float64)
    world = np.full_like(cam, np.nan)
    world[valid] = cam[valid] @ T[:3, :3].T + T[:3, 3]
    return world


def draw_tracks(image_rgb: np.ndarray, mask: np.ndarray, tracks: np.ndarray, visible: np.ndarray, frame_i: int) -> np.ndarray:
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
        if not visible[frame_i, track_i]:
            continue
        color = tuple(int(v) for v in palette[track_i % len(palette)].tolist())
        pts = []
        for t in range(frame_i + 1):
            if visible[t, track_i]:
                pts.append(tuple(np.rint(tracks[t, track_i]).astype(int).tolist()))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 3, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"CoTracker object tracks frame {frame_i}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"CoTracker object tracks frame {frame_i}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run(args: argparse.Namespace) -> dict:
    entries = frame_entries(args.manifest, int(args.frame_start), int(args.frame_end))
    frame_ids = [int(row["frame_idx"]) for row in entries]
    images, masks = load_video_and_masks(entries)
    if images.shape[0] <= int(args.query_frame_index):
        raise RuntimeError("query frame index outside video")
    query_points = sample_query_points(masks[int(args.query_frame_index)], int(args.grid_step_px), int(args.max_points))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CoTracker run requires CUDA")
    video = torch.from_numpy(images).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.zeros((1, len(query_points), 3), dtype=torch.float32, device=device)
    queries[0, :, 0] = float(args.query_frame_index)
    queries[0, :, 1:] = torch.from_numpy(query_points).to(device)
    model = torch.hub.load(
        str(args.torchhub_repo),
        str(args.torchhub_model),
        source=str(args.torchhub_source),
        trust_repo=True,
    ).to(device)
    model.eval()
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=bool(args.backward_tracking))
    tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)

    annotations = annotations_by_frame(args.annotations)
    depths, frame_to_depth_i = depth_archive(args.metric_depth_npz)
    if any(frame not in annotations for frame in frame_ids):
        raise RuntimeError("annotations missing frames for CoTracker QC")
    if any(frame not in frame_to_depth_i for frame in frame_ids):
        raise RuntimeError("depth archive missing frames for CoTracker QC")

    mask_hits = np.stack([mask_hit(mask, tracks[i]) for i, mask in enumerate(masks)], axis=0)
    depths_sampled = np.stack(
        [sample_depth_nearest(depths[frame_to_depth_i[frame]], tracks[i]) for i, frame in enumerate(frame_ids)],
        axis=0,
    )
    depth_valid = np.isfinite(depths_sampled)
    accepted = visibility & mask_hits & depth_valid
    world = np.stack([world_points(tracks[i], depths_sampled[i], annotations[frame]) for i, frame in enumerate(frame_ids)], axis=0)
    consecutive = accepted[:-1] & accepted[1:]
    step_m = np.linalg.norm(world[1:] - world[:-1], axis=2)
    step_m = step_m[consecutive]
    flow_px = np.linalg.norm(tracks[1:] - tracks[:-1], axis=2)
    flow_px = flow_px[visibility[:-1] & visibility[1:]]
    valid_frames_per_track = accepted.sum(axis=0)
    all_frame_tracks = valid_frames_per_track == len(frame_ids)

    rows = []
    for i, frame in enumerate(frame_ids):
        rows.append(
            {
                "frame_idx": int(frame),
                "visible_fraction": float(np.mean(visibility[i])),
                "mask_hit_fraction_of_visible": float(np.mean(mask_hits[i][visibility[i]])) if np.any(visibility[i]) else 0.0,
                "depth_valid_fraction_of_visible_mask": float(np.mean(depth_valid[i][visibility[i] & mask_hits[i]]))
                if np.any(visibility[i] & mask_hits[i])
                else 0.0,
                "accepted_track_count": int(np.count_nonzero(accepted[i])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "cotracker_object_tracks_v5.npz",
        frame_idx=np.asarray(frame_ids, dtype=np.int32),
        tracks_xy=tracks,
        visibility=visibility,
        mask_hits=mask_hits,
        accepted=accepted,
        query_xy=query_points,
        world_xyz=world,
    )
    h, w = images.shape[1:3]
    writer = cv2.VideoWriter(
        str(args.output_dir / "cotracker_tracks_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.output_fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open CoTracker overlay writer")
    stills_dir = args.output_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frame_ids):
        rendered = draw_tracks(images[i], masks[i], tracks, accepted, i)
        writer.write(rendered)
        if frame in set(args.still_frames):
            cv2.imwrite(str(stills_dir / f"frame_{frame:06d}.jpg"), rendered)
    writer.release()

    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "run_cotracker_object_tracks_v5",
        "claim_tested": "learned point tracking on the repaired object mask provides candidate material correspondences before any mesh regularization is applied",
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frames": frame_ids,
        "query_frame_index": int(args.query_frame_index),
        "query_frame_idx": int(frame_ids[int(args.query_frame_index)]),
        "query_points": int(len(query_points)),
        "all_frame_accepted_tracks": int(np.count_nonzero(all_frame_tracks)),
        "valid_frames_per_track": summarize(valid_frames_per_track),
        "world_step_m": summarize(step_m),
        "flow_px": summarize(flow_px),
        "rows": rows,
        "outputs": {
            "tracks_npz": str(args.output_dir / "cotracker_object_tracks_v5.npz"),
            "overlay_video": str(args.output_dir / "cotracker_tracks_overlay.mp4"),
            "stills_dir": str(stills_dir),
        },
        "parameters": {
            "grid_step_px": int(args.grid_step_px),
            "max_points": int(args.max_points),
            "torchhub_repo": str(args.torchhub_repo),
            "torchhub_model": str(args.torchhub_model),
            "torchhub_source": str(args.torchhub_source),
            "backward_tracking": bool(args.backward_tracking),
        },
    }
    (args.output_dir / "qc_cotracker_object_tracks_v5.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
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
