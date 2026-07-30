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


def to_numpy(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.squeeze(arr)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"UniDepth {name} contains non-finite values")
    return arr.astype(np.float32)


def infer_unidepth(model: object, image: Image.Image, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    image_t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    with torch.no_grad():
        prediction = model.infer(image_t)
    if not isinstance(prediction, dict):
        raise RuntimeError(f"UniDepth infer returned {type(prediction)}")
    depth_key = "depth" if "depth" in prediction else "depths" if "depths" in prediction else None
    if depth_key is None:
        raise RuntimeError(f"UniDepth prediction lacks depth key: {sorted(prediction)}")
    depth = to_numpy(prediction[depth_key], "depth")
    if depth.ndim != 2:
        raise RuntimeError(f"UniDepth depth has invalid shape {depth.shape}")
    intrinsics = None
    for key in ("intrinsics", "K", "camera"):
        if key in prediction:
            candidate = to_numpy(prediction[key], key)
            if candidate.shape == (3, 3):
                intrinsics = candidate.astype(np.float64)
                break
            if candidate.size == 9:
                intrinsics = candidate.reshape(3, 3).astype(np.float64)
                break
    return depth, intrinsics


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32)
    return cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def colorize_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("cannot colorize empty depth")
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-3
    scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
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
        f"frame {row['frame_idx']}  focal {row['unidepth_focal_px']:.0f}px  "
        f"mask depth {row['unidepth_mask_depth_median_m']:.3f}m  "
        f"width {row['width_from_unidepth_focal_m']:.3f}m"
    )
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def load_model(model_id: str, device: torch.device) -> object:
    from unidepth.models import UniDepthV2  # noqa: PLC0415

    if hasattr(UniDepthV2, "from_pretrained"):
        model = UniDepthV2.from_pretrained(model_id)
    else:
        raise RuntimeError("UniDepthV2.from_pretrained is missing in this checkout")
    return model.to(device).eval()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.unidepth_repo is not None:
        sys.path.insert(0, str(args.unidepth_repo))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("UniDepth metric-source diagnostic requires CUDA unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = load_model(args.model_id, device)
    rows_in = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    frame_indices = [int(row["frame_idx"]) for row in rows_in]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    depth_png_dir = args.output_dir / "depth"
    depth_png_dir.mkdir(exist_ok=True)
    depth_stack = []
    focal_px = []
    intrinsics_stack = []
    rows = []
    for entry in rows_in:
        frame_idx = int(entry["frame_idx"])
        rgb_path = localize_path(str(entry["rgb"]), args.remote_root, args.local_root)
        mask_path = localize_path(str(entry["mask"]), args.remote_root, args.local_root)
        manifest_depth_path = None
        if "depth" in entry and entry["depth"]:
            manifest_depth_path = localize_path(str(entry["depth"]), args.remote_root, args.local_root)
        image = Image.open(rgb_path).convert("RGB")
        depth_raw, intrinsics = infer_unidepth(model, image, device)
        depth = resize_depth(depth_raw, (int(args.source_height), int(args.source_width)))
        if intrinsics is None:
            raise RuntimeError(f"UniDepth returned no intrinsics for frame {frame_idx}")
        fx = float(intrinsics[0, 0]) * (int(args.source_width) / float(depth_raw.shape[1]))
        fy = float(intrinsics[1, 1]) * (int(args.source_height) / float(depth_raw.shape[0]))
        cx = float(intrinsics[0, 2]) * (int(args.source_width) / float(depth_raw.shape[1]))
        cy = float(intrinsics[1, 2]) * (int(args.source_height) / float(depth_raw.shape[0]))
        focal = float(np.sqrt(max(1e-9, fx * fy)))
        if not np.isfinite(focal) or focal <= 0.0:
            raise RuntimeError(f"invalid UniDepth focal for frame {frame_idx}: {focal}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        manifest_depth = cv2.imread(str(manifest_depth_path), cv2.IMREAD_UNCHANGED) if manifest_depth_path is not None else None
        if mask is None or rgb is None:
            raise RuntimeError(f"failed to read RGB or mask for frame {frame_idx}")
        if mask.shape != depth.shape:
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}: {mask.shape} vs {depth.shape}")
        object_pixels = (mask > 0) & np.isfinite(depth) & (depth > 0.0)
        if int(object_pixels.sum()) < int(args.min_mask_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few valid UniDepth mask pixels")
        x0, y0, x1, y1 = mask_bbox(mask)
        mask_width = int(x1 - x0)
        mask_height = int(y1 - y0)
        depth_values = depth[object_pixels].astype(np.float64)
        depth_median = float(np.median(depth_values))
        row = {
            "frame_idx": frame_idx,
            "rgb": str(rgb_path),
            "mask": str(mask_path),
            "unidepth_focal_px": focal,
            "unidepth_fx_px": fx,
            "unidepth_fy_px": fy,
            "unidepth_cx_px": cx,
            "unidepth_cy_px": cy,
            "unidepth_mask_depth_median_m": depth_median,
            "unidepth_mask_depth_p05_m": float(np.percentile(depth_values, 5.0)),
            "unidepth_mask_depth_p95_m": float(np.percentile(depth_values, 95.0)),
            "mask_width_px": mask_width,
            "mask_height_px": mask_height,
            "width_from_unidepth_focal_m": float(mask_width * depth_median / focal),
            "height_from_unidepth_focal_m": float(mask_height * depth_median / focal),
        }
        if manifest_depth is not None:
            manifest_values = manifest_depth.astype(np.float64)[mask > 0] / 1000.0
            manifest_values = manifest_values[np.isfinite(manifest_values) & (manifest_values > 0.0)]
            if manifest_values.size == 0:
                raise RuntimeError(f"frame {frame_idx} has no valid manifest-depth pixels")
            row["manifest_mask_depth_median_m"] = float(np.median(manifest_values))
        rows.append(row)
        focal_px.append(focal)
        intrinsics_stack.append([fx, fy, cx, cy])
        depth_stack.append(depth.astype(np.float16))
        depth_png_path = depth_png_dir / f"{int(entry.get('index', len(rows) - 1)):06d}.png"
        depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if not cv2.imwrite(str(depth_png_path), depth_mm):
            raise RuntimeError(f"failed to write {depth_png_path}")
        row["depth_png"] = str(depth_png_path)
        cv2.imwrite(str(still_dir / f"frame_{frame_idx:06d}.png"), render_review(rgb, mask, depth, row))

    depth_archive = args.output_dir / "unidepth_metric_depth_v3.npz"
    np.savez_compressed(
        depth_archive,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        depth=np.stack(depth_stack, axis=0),
        source_size=np.asarray([int(args.source_width), int(args.source_height)], dtype=np.int32),
        focal_px=np.asarray(focal_px, dtype=np.float32),
        intrinsics_fx_fy_cx_cy=np.asarray(intrinsics_stack, dtype=np.float32),
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "run_unidepth_metric_source_v3",
        "claim_tested": "UniDepth provides an independent RGB-only metric depth and focal source for object scale ownership",
        "model_id": str(args.model_id),
        "manifest": str(args.manifest),
        "frames": int(len(rows)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "depth_archive": str(depth_archive),
        "stills_dir": str(still_dir),
        "unidepth_focal_px": summarize([row["unidepth_focal_px"] for row in rows]),
        "unidepth_mask_depth_median_m": summarize([row["unidepth_mask_depth_median_m"] for row in rows]),
        "manifest_mask_depth_median_m": summarize(
            [row["manifest_mask_depth_median_m"] for row in rows if "manifest_mask_depth_median_m" in row]
        ),
        "width_from_unidepth_focal_m": summarize([row["width_from_unidepth_focal_m"] for row in rows]),
        "height_from_unidepth_focal_m": summarize([row["height_from_unidepth_focal_m"] for row in rows]),
        "rows": rows,
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_unidepth_metric_source_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--unidepth-repo", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--min-mask-pixels", type=int, default=1000)
    parser.add_argument("--model-id", default="lpiccinelli/unidepth-v2-vitl14")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
