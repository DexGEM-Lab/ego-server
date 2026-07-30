#!/usr/bin/env python3
"""Warp a HOT3D-Clips fisheye RGB V19 input into a pinhole V19 input.

This is a camera adapter only. It consumes HOT3D camera calibration for the
selected stream, uses the same fisheye->pinhole warp as the HOT3D toolkit
(`convert_to_pinhole_camera` + `warp_image`), and writes a V19-style input video,
raw-frame manifest, and calibration contract. It must not consume HOT3D hand,
object, MANO, contact, or visibility annotations as V19 predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ovr624_distort(p: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    if coeffs.shape != (12,):
        raise RuntimeError(f"FISHEYE624 requires 12 distortion coefficients, got {coeffs.shape}")
    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = coeffs.tolist()
    r2 = np.sum(p * p, axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    uv = p * radial
    x = uv[..., 0].copy()
    y = uv[..., 1].copy()
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2_flat = x2 + y2
    x += 2.0 * p2 * xy + p1 * (r2_flat + 2.0 * x2)
    y += 2.0 * p1 * xy + p2 * (r2_flat + 2.0 * y2)
    r4_flat = r2_flat * r2_flat
    x += s1 * r2_flat + s2 * r4_flat
    y += s3 * r2_flat + s4 * r4_flat
    return np.stack((x, y), axis=-1)


def fisheye624_pinhole_maps(
    width: int,
    height: int,
    f: float,
    cx: float,
    cy: float,
    coeffs: np.ndarray,
    focal_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dst_fx = float(f) * float(focal_scale)
    dst_fy = float(f) * float(focal_scale)
    dst_cx = float(cx)
    dst_cy = float(cy)
    px, py = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (px - dst_cx) / dst_fx
    y = (py - dst_cy) / dst_fy
    norm = np.sqrt(x * x + y * y + 1.0)
    vx = x / norm
    vy = y / norm
    vz = 1.0 / norm
    r = np.sqrt(vx * vx + vy * vy)
    scale = np.arctan2(r, vz) / np.maximum(r, np.finfo(np.float64).tiny)
    projected = np.stack((vx * scale, vy * scale), axis=-1)
    distorted = ovr624_distort(projected, coeffs)
    map_x = (distorted[..., 0] * float(f) + float(cx)).astype(np.float32)
    map_y = (distorted[..., 1] * float(f) + float(cy)).astype(np.float32)
    valid = (map_x >= -0.5) & (map_x <= width - 0.5) & (map_y >= -0.5) & (map_y <= height - 0.5)
    meta = {
        "dst_fx": dst_fx,
        "dst_fy": dst_fy,
        "dst_cx": dst_cx,
        "dst_cy": dst_cy,
        "source_valid_fraction": float(np.mean(valid)),
        "map_x_min_max": [float(np.nanmin(map_x)), float(np.nanmax(map_x))],
        "map_y_min_max": [float(np.nanmin(map_y)), float(np.nanmax(map_y))],
    }
    return map_x, map_y, meta


def camera_signature(cam: dict[str, Any], stream_id: str, focal_scale: float) -> tuple[Any, ...]:
    calib = cam["calibration"]
    params = tuple(float(x) for x in calib["projection_params"])
    return (
        stream_id,
        calib.get("projection_model_type"),
        int(calib["image_width"]),
        int(calib["image_height"]),
        params,
        float(focal_scale),
    )


def parse_fisheye624(cam: dict[str, Any]) -> tuple[int, int, float, float, float, np.ndarray]:
    calib = cam["calibration"]
    if calib.get("projection_model_type") != "CameraModelType.FISHEYE624":
        raise RuntimeError(f"expected CameraModelType.FISHEYE624, got {calib.get('projection_model_type')}")
    params = [float(x) for x in calib["projection_params"]]
    if len(params) != 15:
        raise RuntimeError(f"expected 15 FISHEYE624 projection params, got {len(params)}")
    f, cx, cy = params[:3]
    coeffs = np.asarray(params[3:], dtype=np.float64)
    return int(calib["image_width"]), int(calib["image_height"]), f, cx, cy, coeffs


def make_review(originals: list[np.ndarray], undistorted: list[np.ndarray], labels: list[str], output: Path) -> None:
    tiles: list[np.ndarray] = []
    for orig, und, label in zip(originals, undistorted, labels):
        h = min(orig.shape[0], und.shape[0])
        if orig.shape[0] != h:
            orig = cv2.resize(orig, (int(orig.shape[1] * h / orig.shape[0]), h), interpolation=cv2.INTER_AREA)
        if und.shape[0] != h:
            und = cv2.resize(und, (int(und.shape[1] * h / und.shape[0]), h), interpolation=cv2.INTER_AREA)
        tile = np.hstack([orig, und])
        banner = np.zeros((44, tile.shape[1], 3), dtype=np.uint8)
        banner[:] = (8, 8, 8)
        cv2.putText(banner, f"{label}: original fisheye | pinhole warp", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([banner, tile]))
    width = min(1600, max(t.shape[1] for t in tiles))
    norm_tiles = []
    for t in tiles:
        if t.shape[1] != width:
            scale = width / t.shape[1]
            t = cv2.resize(t, (width, int(round(t.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        norm_tiles.append(t)
    sheet = np.vstack(norm_tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write review sheet {output}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_root / "input" / "raw_frame_manifest" / "manifest.json"
    gt_path = args.input_root / "evaluation" / "hot3d_gt" / "hot3d_clip_gt_sidecar.json"
    manifest = load_json(manifest_path)
    gt = load_json(gt_path)
    frames = manifest.get("frames")
    gt_frames = gt.get("frames")
    if not isinstance(frames, list) or not isinstance(gt_frames, list) or len(frames) != len(gt_frames):
        raise RuntimeError("manifest and HOT3D sidecar must contain same-length frame lists")
    gt_by_idx = {int(row["frame_idx"]): row for row in gt_frames if isinstance(row, dict)}

    rgb_dir = args.output_root / "input" / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.output_root / "state" / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    video_path = input_dir / f"{manifest.get('clip_id', args.input_root.name)}_{args.stream_id.replace('-', '_')}_pinhole.mp4"
    writer: cv2.VideoWriter | None = None
    out_rows: list[dict[str, Any]] = []
    map_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    review_orig: list[np.ndarray] = []
    review_und: list[np.ndarray] = []
    review_labels: list[str] = []
    calibration_records: list[dict[str, Any]] = []

    try:
        for frame in frames:
            idx = int(frame["frame_idx"])
            gt_row = gt_by_idx[idx]
            cam = gt_row.get("json", {}).get("cameras.json", {}).get(args.stream_id)
            if not isinstance(cam, dict):
                raise RuntimeError(f"frame {idx} missing cameras.json stream {args.stream_id}")
            sig = camera_signature(cam, args.stream_id, args.focal_scale)
            if sig not in map_cache:
                width, height, f, cx, cy, coeffs = parse_fisheye624(cam)
                map_cache[sig] = fisheye624_pinhole_maps(width, height, f, cx, cy, coeffs, args.focal_scale)
                calibration_records.append(
                    {
                        "frame_idx_first_seen": idx,
                        "stream_id": args.stream_id,
                        "source_projection_model_type": "CameraModelType.FISHEYE624",
                        "source_projection_params": list(sig[4]),
                        "width": width,
                        "height": height,
                        "pinhole_fx_fy_cx_cy": [
                            map_cache[sig][2]["dst_fx"],
                            map_cache[sig][2]["dst_fy"],
                            map_cache[sig][2]["dst_cx"],
                            map_cache[sig][2]["dst_cy"],
                        ],
                        "map_meta": map_cache[sig][2],
                    }
                )
            map_x, map_y, map_meta = map_cache[sig]
            image = cv2.imread(str(frame["raw_frame_path"]))
            if image is None:
                raise FileNotFoundError(frame["raw_frame_path"])
            undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            if writer is None:
                h, w = undistorted.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(manifest.get("fps", args.fps)), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer {video_path}")
            writer.write(undistorted)
            out_path = rgb_dir / f"{idx:06d}.jpg"
            if not cv2.imwrite(str(out_path), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f"failed to write {out_path}")
            out_row = dict(frame)
            out_row.update(
                {
                    "rgb": str(out_path.relative_to(args.output_root)),
                    "raw_frame_path": str(out_path),
                    "source_width": int(undistorted.shape[1]),
                    "source_height": int(undistorted.shape[0]),
                    "manifest_width": int(undistorted.shape[1]),
                    "manifest_height": int(undistorted.shape[0]),
                    "hot3d_camera_adapter": "fisheye624_to_pinhole_v1",
                    "hot3d_source_raw_frame_path": frame["raw_frame_path"],
                    "hot3d_pinhole_fx_fy_cx_cy": [map_meta["dst_fx"], map_meta["dst_fy"], map_meta["dst_cx"], map_meta["dst_cy"]],
                }
            )
            out_rows.append(out_row)
            if idx in set(args.review_frames):
                review_orig.append(image)
                review_und.append(undistorted)
                review_labels.append(f"frame {idx:06d}")
    finally:
        if writer is not None:
            writer.release()

    if not out_rows:
        raise RuntimeError("no frames adapted")
    first_k = out_rows[0]["hot3d_pinhole_fx_fy_cx_cy"]
    out_manifest = dict(manifest)
    out_manifest.update(
        {
            "schema": "v19_hot3d_pinhole_raw_frame_manifest_v1",
            "source_manifest": str(manifest_path),
            "source_input_root": str(args.input_root),
            "video": str(video_path),
            "frame_count": len(out_rows),
            "width": int(out_rows[0]["manifest_width"]),
            "height": int(out_rows[0]["manifest_height"]),
            "hot3d_camera_adapter": "fisheye624_to_pinhole_v1",
            "stream_id": args.stream_id,
            "causal_boundary": "Consumes HOT3D camera calibration only; hand/object/MANO/contact annotations remain evaluation-only.",
            "frames": out_rows,
        }
    )
    write_json(args.output_root / "input" / "raw_frame_manifest" / "manifest.json", out_manifest)
    calib = {
        "schema": "v19_hot3d_pinhole_camera_calibration_contract_v1",
        "method": "build_v19_hot3d_pinhole_adapter",
        "claim_scope": "sensor camera adaptation only; not a hand/object/contact prediction",
        "source": "HOT3D cameras.json calibration, FISHEYE624 -> PinholePlane using HOT3D toolkit equations",
        "input_root": str(args.input_root),
        "stream_id": args.stream_id,
        "focal_scale": float(args.focal_scale),
        "image_width": int(out_rows[0]["manifest_width"]),
        "image_height": int(out_rows[0]["manifest_height"]),
        "K": [[float(first_k[0]), 0.0, float(first_k[2])], [0.0, float(first_k[1]), float(first_k[3])], [0.0, 0.0, 1.0]],
        "intrinsics_fx_fy_cx_cy": [float(x) for x in first_k],
        "calibration_records": calibration_records,
    }
    write_json(calib_dir / "v19_hot3d_pinhole_camera_calibration_contract.json", calib)
    np.savez_compressed(
        calib_dir / "v19_hot3d_pinhole_camera_intrinsics.npz",
        intrinsics_fx_fy_cx_cy=np.asarray([out_rows[i]["hot3d_pinhole_fx_fy_cx_cy"] for i in range(len(out_rows))], dtype=np.float32),
    )
    review_path = eval_dir / "hot3d_pinhole_adapter_review.jpg"
    if review_orig:
        make_review(review_orig, review_und, review_labels, review_path)
    report = {
        "status": "ok",
        "method": "build_v19_hot3d_pinhole_adapter",
        "claim_scope": calib["claim_scope"],
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "output_video": str(video_path),
        "raw_frame_manifest": str(args.output_root / "input" / "raw_frame_manifest" / "manifest.json"),
        "calibration_contract": str(calib_dir / "v19_hot3d_pinhole_camera_calibration_contract.json"),
        "review": str(review_path) if review_orig else None,
        "frame_count": len(out_rows),
        "unique_camera_calibrations": len(map_cache),
        "pinhole_fx_fy_cx_cy_first": [float(x) for x in first_k],
        "map_meta_first": calibration_records[0]["map_meta"] if calibration_records else None,
    }
    write_json(eval_dir / "hot3d_pinhole_adapter_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--review-frames", type=int, nargs="*", default=[0, 50, 100, 149])
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
