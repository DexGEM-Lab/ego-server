#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_frames(available: list[int], max_frames: int) -> list[int]:
    if len(available) <= max_frames:
        return available
    idx = np.linspace(0, len(available) - 1, max_frames, dtype=int)
    return [available[int(i)] for i in idx]


def run(args: argparse.Namespace) -> dict[str, Any]:
    ann = load_json(args.visible_geometry_annotations)
    depth_blob = np.load(args.depth_npz)
    frame_idx = depth_blob["frame_idx"].astype(int)
    depth = depth_blob["depth"].astype(np.float32)
    intr = depth_blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    frame_to_i = {int(f): int(i) for i, f in enumerate(frame_idx)}
    rows = []
    candidates = []
    for frame in ann.get("frames", []):
        if not isinstance(frame, dict):
            continue
        idx = int(frame.get("frame_idx", -1))
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if not isinstance(obj, dict):
                continue
            if obj.get("status") == "visible_metric_surface_measurement" and obj.get("mask_path") and idx in frame_to_i:
                candidates.append((idx, frame, obj))
    if not candidates:
        raise ContractError("no_visible_metric_surface_candidates")
    selected_ids = set(select_frames([idx for idx, _frame, _obj in candidates], int(args.max_frames)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb"
    mask_dir = args.output_dir / "mask"
    depth_dir = args.output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    K_written = False
    intrinsics_rows = []
    for idx, frame, obj in candidates:
        if idx not in selected_ids:
            continue
        depth_i = frame_to_i[idx]
        raw_rgb = cv2.imread(str(frame.get("raw_frame_path")), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if raw_rgb is None or mask is None:
            continue
        depth_m = depth[depth_i]
        if mask.shape != depth_m.shape:
            mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
        if raw_rgb.shape[:2] != depth_m.shape:
            raw_rgb = cv2.resize(raw_rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        rgb_path = rgb_dir / f"{idx:06d}.jpg"
        mask_path = mask_dir / f"{idx:06d}.png"
        depth_path = depth_dir / f"{idx:06d}.png"
        cv2.imwrite(str(rgb_path), raw_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        cv2.imwrite(str(mask_path), np.where(mask > 0, 255, 0).astype(np.uint8))
        cv2.imwrite(str(depth_path), depth_mm)
        fx, fy, cx, cy = intr[depth_i].tolist()
        if not K_written:
            K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            np.savetxt(args.output_dir / "cam_K.txt", K)
            K_written = True
        intrinsics_rows.append({"frame_idx": int(idx), "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)})
        rows.append(
            {
                "index": int(idx),
                "frame_idx": int(idx),
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "depth": str(depth_path),
                "intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
                "source_visible_geometry_status": obj.get("status"),
            }
        )
    if not rows:
        raise ContractError("no_heightfield_rows_written")
    manifest = {
        "schema": "v21_heightfield_dataset_manifest.v0",
        "status": "ok",
        "method": "export_v21_heightfield_dataset",
        "visible_geometry_annotations": str(args.visible_geometry_annotations),
        "depth_npz": str(args.depth_npz),
        "dataset_dir": str(args.output_dir),
        "frames": rows,
        "intrinsics_rows": intrinsics_rows,
        "claim_scope": "Dataset adapter for observed heightfield mesh reconstruction from accepted masks and selected metric depth. It is not itself a mesh or pose result.",
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V21 accepted mask+metric depth rows to heightfield reconstruction dataset format.")
    parser.add_argument("--visible-geometry-annotations", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
