#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AnchorPoint:
    track_id: str
    anchor_frame_idx: int
    anchor_index: int
    object_xyz: np.ndarray
    evidence: str
    confidence: float


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_intrinsics(dataset: Path, manifest: dict) -> tuple[float, float, float, float]:
    qc = dataset / "qc_bundlesdf_dataset_v3.json"
    if qc.exists():
        values = load_json(qc).get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return tuple(float(v) for v in values)
    values = manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return tuple(float(v) for v in values)
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3):
        raise RuntimeError(f"{dataset / 'cam_K.txt'} must be a 3x3 matrix")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def prompt_height(prompt_width: int, source_size: tuple[int, int]) -> int:
    source_w, source_h = source_size
    return int(round(prompt_width * source_h / source_w))


def load_depth(dataset: Path, index: int) -> np.ndarray:
    path = dataset / "depth" / f"{index:06d}.png"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"failed to read depth {path}")
    return depth.astype(np.float64) / 1000.0


def sample_depth(depth_m: np.ndarray, xy: np.ndarray, radius: int) -> float | None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    x0 = max(0, x - radius)
    x1 = min(depth_m.shape[1], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(depth_m.shape[0], y + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0.05)]
    if values.size == 0:
        return None
    return float(np.median(values))


def unproject(source_xy: np.ndarray, z: float, intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.asarray([(source_xy[0] - cx) * z / fx, (source_xy[1] - cy) * z / fy, z], dtype=np.float64)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homog = np.r_[point.astype(np.float64), 1.0]
    return (transform @ homog)[:3]


def project(cam_xyz: np.ndarray, intrinsics: tuple[float, float, float, float]) -> np.ndarray | None:
    if not np.isfinite(cam_xyz).all() or cam_xyz[2] <= 0.0:
        return None
    fx, fy, cx, cy = intrinsics
    return np.asarray([fx * cam_xyz[0] / cam_xyz[2] + cx, fy * cam_xyz[1] / cam_xyz[2] + cy], dtype=np.float64)


def load_pose(bundlesdf_output: Path, index: int) -> np.ndarray:
    path = bundlesdf_output / "ob_in_cam" / f"{index:06d}.txt"
    if not path.exists():
        raise RuntimeError(f"missing BundleSDF pose {path}")
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise RuntimeError(f"BundleSDF pose must be finite 4x4: {path}")
    return pose


def point_files(point_root: Path, include_track_ids: set[str]) -> list[tuple[Path, dict]]:
    files = sorted(point_root.glob("*/object_point_prompts_vlm.json"))
    if not files:
        raise RuntimeError(f"no object_point_prompts_vlm.json files under {point_root}")
    selected = []
    for path in files:
        payload = load_json(path)
        track_id = str(payload.get("track_id", ""))
        if track_id in include_track_ids:
            selected.append((path, payload))
    missing = sorted(include_track_ids.difference({str(payload.get("track_id", "")) for _, payload in selected}))
    if missing:
        raise RuntimeError(f"missing included VLM point tracks: {missing}")
    return selected


def build_anchor_points(
    args: argparse.Namespace,
    entries_by_frame: dict[int, dict],
    source_size: tuple[int, int],
    intrinsics: tuple[float, float, float, float],
    point_payloads: list[tuple[Path, dict]],
) -> list[AnchorPoint]:
    anchors: list[AnchorPoint] = []
    for _, payload in point_payloads:
        track_id = str(payload["track_id"])
        p_width = int(payload["prompt_image_width"])
        p_height = prompt_height(p_width, source_size)
        scale = np.asarray([source_size[0] / p_width, source_size[1] / p_height], dtype=np.float64)
        rows = payload.get("point_prompts")
        if not isinstance(rows, list):
            raise RuntimeError(f"{track_id} point prompt payload lacks point_prompts")
        for row in rows:
            frame_idx = int(row["frame_idx"])
            if frame_idx not in entries_by_frame or not row.get("target_visible"):
                continue
            if args.anchor_frames and frame_idx not in set(args.anchor_frames):
                continue
            positives = row.get("positive_points") or []
            if not positives:
                continue
            index = int(entries_by_frame[frame_idx]["index"])
            depth = load_depth(args.dataset, index)
            ob_in_cam = load_pose(args.bundlesdf_output, index)
            cam_in_ob = np.linalg.inv(ob_in_cam)
            for point_i, point in enumerate(positives):
                prompt_xy = np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
                source_xy = prompt_xy * scale
                z = sample_depth(depth, source_xy, int(args.depth_sample_radius_px))
                if z is None:
                    continue
                cam_xyz = unproject(source_xy, z, intrinsics)
                object_xyz = transform_point(cam_in_ob, cam_xyz)
                anchors.append(
                    AnchorPoint(
                        track_id=track_id,
                        anchor_frame_idx=frame_idx,
                        anchor_index=point_i,
                        object_xyz=object_xyz,
                        evidence=str(point.get("evidence", "")),
                        confidence=float(row.get("confidence", 0.5)),
                    )
                )
    if len(anchors) < int(args.min_anchor_points):
        raise RuntimeError(f"only {len(anchors)} anchor points survived depth lifting")
    return anchors


def reproject_anchors(
    anchors: list[AnchorPoint],
    ob_in_cam: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    source_size: tuple[int, int],
    prompt_size: tuple[int, int],
    depth_m: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    source_to_prompt = np.asarray([prompt_size[0] / source_size[0], prompt_size[1] / source_size[1]], dtype=np.float64)
    rows = []
    occupied: set[tuple[int, int]] = set()
    for anchor in anchors:
        cam_xyz = transform_point(ob_in_cam, anchor.object_xyz)
        source_xy = project(cam_xyz, intrinsics)
        if source_xy is None:
            continue
        if source_xy[0] < 0 or source_xy[0] >= source_size[0] or source_xy[1] < 0 or source_xy[1] >= source_size[1]:
            continue
        observed_depth = sample_depth(depth_m, source_xy, int(args.reprojection_depth_radius_px))
        if observed_depth is None:
            continue
        depth_error = abs(float(observed_depth) - float(cam_xyz[2]))
        if depth_error > float(args.reprojection_depth_tolerance_m):
            continue
        prompt_xy = source_xy * source_to_prompt
        cell = (int(round(prompt_xy[0] / args.dedup_grid_px)), int(round(prompt_xy[1] / args.dedup_grid_px)))
        if cell in occupied:
            continue
        occupied.add(cell)
        rows.append(
            {
                "x": float(np.clip(prompt_xy[0], 0.0, prompt_size[0] - 1.0)),
                "y": float(np.clip(prompt_xy[1], 0.0, prompt_size[1] - 1.0)),
                "evidence": (
                    f"{anchor.track_id} point {anchor.anchor_index} lifted from frame "
                    f"{anchor.anchor_frame_idx}, depth residual {depth_error:.3f}m: {anchor.evidence}"
                ),
            }
        )
    return rows


def bbox_from_points(points: list[dict], prompt_size: tuple[int, int], margin_px: float) -> list[float]:
    xy = np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float64)
    lo = xy.min(axis=0) - float(margin_px)
    hi = xy.max(axis=0) + float(margin_px)
    return [
        float(np.clip(lo[0], 0.0, prompt_size[0] - 1.0)),
        float(np.clip(lo[1], 0.0, prompt_size[1] - 1.0)),
        float(np.clip(hi[0], 0.0, prompt_size[0] - 1.0)),
        float(np.clip(hi[1], 0.0, prompt_size[1] - 1.0)),
    ]


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest or (args.dataset / "manifest.json"))
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest must contain a nonempty frames list")
    entries_by_frame = {int(entry["frame_idx"]): entry for entry in entries}
    selected_entries = [
        entry
        for entry in entries
        if int(entry["frame_idx"]) >= int(args.frame_start) and int(entry["frame_idx"]) <= int(args.frame_end)
    ]
    if not selected_entries:
        raise RuntimeError("no manifest frames selected")
    first_rgb = cv2.imread(str(args.dataset / "rgb" / f"{int(selected_entries[0]['index']):06d}.png"), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError("failed to read first RGB frame")
    source_size = (int(first_rgb.shape[1]), int(first_rgb.shape[0]))
    prompt_size = (int(args.prompt_image_width), prompt_height(int(args.prompt_image_width), source_size))
    intrinsics = load_intrinsics(args.dataset, manifest)
    include_track_ids = set(args.include_track_ids)
    payloads = point_files(args.point_root, include_track_ids)
    anchors = build_anchor_points(args, entries_by_frame, source_size, intrinsics, payloads)

    point_rows = []
    visible_count = 0
    for entry in selected_entries:
        frame_idx = int(entry["frame_idx"])
        index = int(entry["index"])
        ob_in_cam = load_pose(args.bundlesdf_output, index)
        depth = load_depth(args.dataset, index)
        positives = reproject_anchors(anchors, ob_in_cam, intrinsics, source_size, prompt_size, depth, args)
        target_visible = len(positives) >= int(args.min_positive_points)
        bbox = bbox_from_points(positives, prompt_size, float(args.bbox_margin_px)) if target_visible else []
        if target_visible:
            visible_count += 1
        point_rows.append(
            {
                "frame_idx": frame_idx,
                "target_visible": bool(target_visible),
                "positive_points": positives if target_visible else [],
                "negative_points": [],
                "bbox_xyxy": bbox,
                "visual_evidence": (
                    "VLM-labeled surface points were lifted through metric depth and BundleSDF object pose, "
                    "then reprojected into this frame."
                ),
                "confidence": float(np.median([anchor.confidence for anchor in anchors])) if target_visible else 0.0,
            }
        )

    output = {
        "status": "ok",
        "backend": "BundleSDF-pose propagation of VLM surface point prompts",
        "track_id": args.track_id,
        "description": args.description,
        "source_point_root": str(args.point_root),
        "included_track_ids": sorted(include_track_ids),
        "dataset": str(args.dataset),
        "bundlesdf_output": str(args.bundlesdf_output),
        "prompt_image_width": int(prompt_size[0]),
        "prompt_image_height": int(prompt_size[1]),
        "frames_prompted": int(len(point_rows)),
        "visible_frames": int(visible_count),
        "anchor_points": [
            {
                "track_id": anchor.track_id,
                "anchor_frame_idx": int(anchor.anchor_frame_idx),
                "anchor_index": int(anchor.anchor_index),
                "object_xyz": anchor.object_xyz.astype(float).tolist(),
                "evidence": anchor.evidence,
                "confidence": float(anchor.confidence),
            }
            for anchor in anchors
        ],
        "point_prompts": point_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "object_point_prompts_vlm.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in {"point_prompts", "anchor_points"}}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bundlesdf-output", type=Path, required=True)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-track-ids", nargs="+", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frames", type=int, nargs="*")
    parser.add_argument("--prompt-image-width", type=int, default=960)
    parser.add_argument("--depth-sample-radius-px", type=int, default=2)
    parser.add_argument("--reprojection-depth-radius-px", type=int, default=2)
    parser.add_argument("--reprojection-depth-tolerance-m", type=float, default=0.035)
    parser.add_argument("--bbox-margin-px", type=float, default=90.0)
    parser.add_argument("--dedup-grid-px", type=float, default=6.0)
    parser.add_argument("--min-positive-points", type=int, default=4)
    parser.add_argument("--min-anchor-points", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
