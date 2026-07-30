#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def read_manifest(path: Path, frame_start: int, frame_end: int) -> list[dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    rows = [row for row in frames if int(frame_start) <= int(row["frame_idx"]) <= int(frame_end)]
    rows.sort(key=lambda row: int(row["frame_idx"]))
    if not rows:
        raise RuntimeError(f"no frames in requested range {frame_start}:{frame_end}")
    return rows


def localize_path(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src, dst in ((local_root, remote_root), (remote_root, local_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def source_intrinsics_from_vggt(
    intrinsic_vggt: np.ndarray,
    source_width: int,
    source_height: int,
    target_size: int,
) -> list[float]:
    if source_width >= source_height:
        new_width = int(target_size)
        new_height = round(source_height * (new_width / source_width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(source_width * (new_height / source_height) / 14) * 14
    if new_width <= 0 or new_height <= 0:
        raise RuntimeError("invalid VGGT preprocessing dimensions")
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    sx = new_width / float(source_width)
    sy = new_height / float(source_height)
    return [
        float(intrinsic_vggt[0, 0] / sx),
        float(intrinsic_vggt[1, 1] / sy),
        float((intrinsic_vggt[0, 2] - pad_left) / sx),
        float((intrinsic_vggt[1, 2] - pad_top) / sy),
    ]


def vggt_depth_rows(
    archive: Path | None,
    frame_indices: list[int],
    source_width: int,
    source_height: int,
    target_size: int,
) -> dict[int, dict]:
    if archive is None:
        return {}
    blob = np.load(archive)
    required = {"frame_idx", "vertex_offsets", "object_points_vggt", "extrinsic", "intrinsic", "sim3_scale"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{archive} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    points = blob["object_points_vggt"].astype(np.float64)
    extrinsic = blob["extrinsic"].astype(np.float64)
    intrinsic = blob["intrinsic"].astype(np.float64)
    sim3_scale = float(blob["sim3_scale"][0])
    out: dict[int, dict] = {}
    for frame_idx in frame_indices:
        hits = np.where(frames == int(frame_idx))[0]
        if len(hits) != 1:
            raise RuntimeError(f"VGGT archive has {len(hits)} rows for frame {frame_idx}")
        i = int(hits[0])
        pts = points[int(offsets[i]) : int(offsets[i + 1])]
        pts_cam = pts @ extrinsic[i, :3, :3].T + extrinsic[i, :3, 3][None, :]
        pts_cam = pts_cam[np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 0)]
        if len(pts_cam) == 0:
            raise RuntimeError(f"VGGT frame {frame_idx} has no positive camera points")
        K4 = source_intrinsics_from_vggt(intrinsic[i], source_width, source_height, target_size)
        out[int(frame_idx)] = {
            "vggt_unscaled_depth_median_m": float(np.median(pts_cam[:, 2])),
            "vggt_sim3_scaled_depth_median_m": float(np.median(pts_cam[:, 2]) * sim3_scale),
            "vggt_source_intrinsics_fx_fy_cx_cy": K4,
            "sim3_scale": float(sim3_scale),
        }
    return out


def to_numpy_depth(depth: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(depth, torch.Tensor):
        arr = depth.detach().float().cpu().numpy()
    else:
        arr = np.asarray(depth, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Depth Pro returned invalid depth shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("Depth Pro depth contains non-finite values")
    return arr.astype(np.float32)


def to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().reshape(-1)[0])
    arr = np.asarray(value)
    if arr.size != 1:
        raise RuntimeError(f"expected scalar, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth
    return cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def colorize_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise RuntimeError("cannot colorize empty depth")
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-3
    scaled = np.clip((depth - lo) / (hi - lo), 0, 1)
    color = cv2.applyColorMap((255 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[mask <= 0] = (0.35 * color[mask <= 0]).astype(np.uint8)
    return color


def render_review(rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, row: dict) -> np.ndarray:
    image = rgb.copy()
    depth_color = colorize_depth(depth, mask)
    overlay = image.copy()
    overlay[mask > 0] = depth_color[mask > 0]
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)
    x0, y0, x1, y1 = mask_bbox(mask)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
    text = (
        f"frame {row['frame_idx']}  focal {row['depthpro_focal_px']:.0f}px  "
        f"mask depth {row['depthpro_mask_depth_median_m']:.3f}m  "
        f"width {row['width_from_depthpro_focal_m']:.3f}m"
    )
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.depthpro_repo is not None:
        sys.path.insert(0, str(args.depthpro_repo / "src"))
    import depth_pro  # noqa: PLC0415

    rows_in = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    frame_indices = [int(row["frame_idx"]) for row in rows_in]
    source_width = int(args.source_width)
    source_height = int(args.source_height)
    vggt = vggt_depth_rows(args.vggt_archive, frame_indices, source_width, source_height, int(args.target_size))

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("Depth Pro metric-source diagnostic requires CUDA unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    depth_stack = []
    rows = []

    for entry in rows_in:
        frame_idx = int(entry["frame_idx"])
        rgb_path = localize_path(str(entry["rgb"]), args.remote_root, args.local_root)
        mask_path = localize_path(str(entry["mask"]), args.remote_root, args.local_root)
        da_depth_path = localize_path(str(entry["depth"]), args.remote_root, args.local_root)
        image_pil, _, exif_focal = depth_pro.load_rgb(rgb_path)
        image_tensor = transform(image_pil).to(device)
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=exif_focal)
        depth = resize_depth(to_numpy_depth(prediction["depth"]), (source_height, source_width))
        focal_px = to_float(prediction["focallength_px"])
        if focal_px <= 0 or not np.isfinite(focal_px):
            raise RuntimeError(f"invalid Depth Pro focal for frame {frame_idx}: {focal_px}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        da_depth = cv2.imread(str(da_depth_path), cv2.IMREAD_UNCHANGED)
        if mask is None or rgb is None or da_depth is None:
            raise RuntimeError(f"failed to read RGB/mask/depth for frame {frame_idx}")
        if mask.shape != depth.shape:
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}: {mask.shape} vs {depth.shape}")
        object_pixels = (mask > 0) & np.isfinite(depth) & (depth > 0)
        if int(object_pixels.sum()) < int(args.min_mask_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few valid Depth Pro mask pixels")
        x0, y0, x1, y1 = mask_bbox(mask)
        mask_width = int(x1 - x0)
        mask_height = int(y1 - y0)
        depth_values = depth[object_pixels].astype(np.float64)
        da_values = da_depth.astype(np.float64)[mask > 0] / 1000.0
        da_values = da_values[np.isfinite(da_values) & (da_values > 0)]
        if da_values.size == 0:
            raise RuntimeError(f"frame {frame_idx} has no valid Depth Anything pixels")
        depth_median = float(np.median(depth_values))
        vrow = vggt.get(frame_idx, {})
        vggt_depth = vrow.get("vggt_unscaled_depth_median_m")
        row = {
            "frame_idx": frame_idx,
            "rgb": str(rgb_path),
            "mask": str(mask_path),
            "depthpro_focal_px": focal_px,
            "depthpro_mask_depth_median_m": depth_median,
            "depthpro_mask_depth_p05_m": float(np.percentile(depth_values, 5.0)),
            "depthpro_mask_depth_p95_m": float(np.percentile(depth_values, 95.0)),
            "depth_anything_mask_depth_median_m": float(np.median(da_values)),
            "mask_width_px": mask_width,
            "mask_height_px": mask_height,
            "width_from_depthpro_focal_m": float(mask_width * depth_median / focal_px),
            "height_from_depthpro_focal_m": float(mask_height * depth_median / focal_px),
            "vggt": vrow,
        }
        if vggt_depth is not None:
            vfx, vfy = vrow["vggt_source_intrinsics_fx_fy_cx_cy"][:2]
            row["depthpro_minus_vggt_unscaled_depth_m"] = float(depth_median - float(vggt_depth))
            row["width_from_vggt_depth_vggt_focal_m"] = float(mask_width * float(vggt_depth) / float(vfx))
            row["height_from_vggt_depth_vggt_focal_m"] = float(mask_height * float(vggt_depth) / float(vfy))
        rows.append(row)
        depth_stack.append(depth.astype(np.float16))
        review = render_review(rgb, mask, depth, row)
        cv2.imwrite(str(still_dir / f"frame_{frame_idx:06d}.png"), review)

    depth_archive = args.output_dir / "depthpro_metric_depth_v3.npz"
    np.savez_compressed(
        depth_archive,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        depth=np.stack(depth_stack, axis=0),
        source_size=np.asarray([source_width, source_height], dtype=np.int32),
        focal_px=np.asarray([row["depthpro_focal_px"] for row in rows], dtype=np.float32),
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "run_depthpro_metric_source_v3",
        "claim_tested": "Depth Pro provides an independent RGB-only metric depth and focal source for object scale ownership",
        "manifest": str(args.manifest),
        "vggt_archive": str(args.vggt_archive) if args.vggt_archive else None,
        "frames": int(len(rows)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "depth_archive": str(depth_archive),
        "stills_dir": str(still_dir),
        "depthpro_focal_px": summarize([row["depthpro_focal_px"] for row in rows]),
        "depthpro_mask_depth_median_m": summarize([row["depthpro_mask_depth_median_m"] for row in rows]),
        "depth_anything_mask_depth_median_m": summarize([row["depth_anything_mask_depth_median_m"] for row in rows]),
        "width_from_depthpro_focal_m": summarize([row["width_from_depthpro_focal_m"] for row in rows]),
        "height_from_depthpro_focal_m": summarize([row["height_from_depthpro_focal_m"] for row in rows]),
        "depthpro_minus_vggt_unscaled_depth_m": summarize(
            [row["depthpro_minus_vggt_unscaled_depth_m"] for row in rows if "depthpro_minus_vggt_unscaled_depth_m" in row]
        ),
        "rows": rows,
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_depthpro_metric_source_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--vggt-archive", type=Path)
    parser.add_argument("--depthpro-repo", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-mask-pixels", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
