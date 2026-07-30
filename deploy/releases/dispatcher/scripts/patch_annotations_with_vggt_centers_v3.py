#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_vggt_centers(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    required = {"frame_idx", "camera_centers_aligned"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    centers = blob["camera_centers_aligned"].astype(np.float64)
    if centers.shape != (len(frame_idx), 3):
        raise RuntimeError(f"camera_centers_aligned shape {centers.shape} does not match frame_idx")
    return {int(idx): centers[i] for i, idx in enumerate(frame_idx)}


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations must contain a nonempty frames list")
    centers = load_vggt_centers(args.vggt_archive)
    changed = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        if idx < int(args.frame_start) or idx > int(args.frame_end):
            continue
        if idx not in centers:
            raise RuntimeError(f"VGGT archive has no aligned center for frame {idx}")
        camera = frame.get("camera")
        if not isinstance(camera, dict) or "T_world_camera_metric" not in camera:
            raise RuntimeError(f"frame {idx} missing camera.T_world_camera_metric")
        T = np.asarray(camera["T_world_camera_metric"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"frame {idx} camera transform must be finite 4x4")
        old = T[:3, 3].copy()
        T[:3, 3] = centers[idx]
        camera["T_world_camera_metric"] = T.astype(float).tolist()
        camera["translation_source_before_vggt_patch"] = old.astype(float).tolist()
        camera["translation_source_after_vggt_patch"] = centers[idx].astype(float).tolist()
        changed.append(
            {
                "frame_idx": idx,
                "old_center_world_m": old.astype(float).tolist(),
                "new_center_world_m": centers[idx].astype(float).tolist(),
                "translation_delta_m": float(np.linalg.norm(centers[idx] - old)),
            }
        )
    if not changed:
        raise RuntimeError("no frames were patched")
    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "patch_annotations_with_vggt_centers_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "changed_frames": int(len(changed)),
        "translation_delta_median_m": float(np.median([row["translation_delta_m"] for row in changed])),
        "translation_delta_p95_m": float(np.percentile([row["translation_delta_m"] for row in changed], 95.0)),
        "rows": changed,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
