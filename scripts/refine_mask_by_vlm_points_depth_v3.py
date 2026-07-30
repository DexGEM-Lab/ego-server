#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fuse_v1_full_fidelity import load_json


def load_depth(path: Path) -> dict:
    blob = np.load(path)
    return {
        "frame_to_i": {int(frame_idx): i for i, frame_idx in enumerate(blob["frame_idx"].astype(int))},
        "depth": blob["depth"].astype(np.float32),
        "source_size": tuple(int(v) for v in blob["source_size"].tolist()),
    }


def prompt_by_frame(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"point prompt file has no point_prompts list: {path}")
    return {int(row["frame_idx"]): row for row in rows}, payload


def source_to_mask_xy(points: list[dict], prompt_size: tuple[int, int], mask_shape: tuple[int, int]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    scale = np.asarray([mask_shape[1] / prompt_size[0], mask_shape[0] / prompt_size[1]], dtype=np.float32)
    return xy * scale[None, :]


def sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    samples = []
    for dy in (-1, 0, 1):
        yy = np.clip(y + dy, 0, depth.shape[0] - 1)
        for dx in (-1, 0, 1):
            xx = np.clip(x + dx, 0, depth.shape[1] - 1)
            samples.append(depth[yy, xx])
    return np.median(np.stack(samples, axis=1), axis=1)


def component_records(mask: np.ndarray, depth: np.ndarray, positive_xy: np.ndarray, depth_center: float, depth_tol: float) -> list[dict]:
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    records = []
    px = np.clip(np.rint(positive_xy[:, 0]).astype(int), 0, mask.shape[1] - 1) if len(positive_xy) else np.asarray([], dtype=int)
    py = np.clip(np.rint(positive_xy[:, 1]).astype(int), 0, mask.shape[0] - 1) if len(positive_xy) else np.asarray([], dtype=int)
    for label in range(1, count):
        comp = labels == label
        comp_depth = depth[comp]
        median_depth = float(np.median(comp_depth)) if comp_depth.size else float("nan")
        positive_hits = int(np.sum(labels[py, px] == label)) if len(positive_xy) else 0
        depth_ok = abs(median_depth - depth_center) <= depth_tol
        records.append(
            {
                "label": int(label),
                "area_px": int(stats[label, cv2.CC_STAT_AREA]),
                "center_xy": [float(centers[label, 0]), float(centers[label, 1])],
                "median_depth_m": median_depth,
                "positive_hits": positive_hits,
                "keep": bool(positive_hits > 0 or depth_ok),
            }
        )
    return records


def draw_review(base_image: np.ndarray, original: np.ndarray, refined: np.ndarray, prompt: dict, output: Path) -> None:
    tile = base_image.copy()
    overlay = tile.copy()
    overlay[original] = (0.35 * overlay[original] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
    overlay[refined] = (0.25 * overlay[refined] + 0.75 * np.asarray([0, 220, 255])).astype(np.uint8)
    tile = overlay
    for point in prompt.get("positive_points", []):
        cv2.circle(tile, (int(round(float(point["x"]))), int(round(float(point["y"])))), 8, (0, 255, 0), -1)
    for point in prompt.get("negative_points", []):
        cv2.circle(tile, (int(round(float(point["x"]))), int(round(float(point["y"])))), 8, (0, 0, 255), -1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), tile):
        raise RuntimeError(f"failed to write review image {output}")


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = annotations["frames"]
    by_frame = {int(frame["frame_idx"]): frame for frame in frames}
    prompts, prompt_payload = prompt_by_frame(args.point_prompts)
    depth = load_depth(args.metric_depth_npz)
    prompt_size = (int(prompt_payload["prompt_image_width"]), int(round(int(prompt_payload["prompt_image_width"]) * 9 / 16)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    kept = 0
    for frame_idx, prompt in sorted(prompts.items()):
        if frame_idx not in by_frame:
            continue
        if frame_idx not in depth["frame_to_i"]:
            reports.append({"frame_idx": frame_idx, "reason": "no_metric_depth"})
            continue
        frame = by_frame[frame_idx]
        obj = frame.get("object", {})
        if obj.get("status") != "measured_vlm_points_sam" or not obj.get("mask_path"):
            reports.append({"frame_idx": frame_idx, "reason": "no_measured_point_mask"})
            continue
        raw = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"failed to read mask {obj['mask_path']}")
        mask = raw > 0
        dep = depth["depth"][depth["frame_to_i"][frame_idx]]
        if dep.shape != mask.shape:
            dep = cv2.resize(dep, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        pos_xy = source_to_mask_xy(prompt.get("positive_points", []), prompt_size, mask.shape)
        pos_depth = sample_depth(dep, pos_xy)
        finite = pos_depth[np.isfinite(pos_depth)]
        if finite.size == 0:
            reports.append({"frame_idx": frame_idx, "reason": "no_positive_depth"})
            continue
        center = float(np.median(finite))
        records = component_records(mask, dep, pos_xy, center, float(args.depth_tolerance_m))
        count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep_labels = {int(record["label"]) for record in records if record["keep"]}
        refined = np.isin(labels, list(keep_labels)) if keep_labels else np.zeros_like(mask)
        if int(refined.sum()) < int(args.min_area_px):
            reports.append(
                {
                    "frame_idx": frame_idx,
                    "reason": "refined_area_underconstrained",
                    "original_area_px": int(mask.sum()),
                    "refined_area_px": int(refined.sum()),
                    "positive_depth_median_m": center,
                    "components": records,
                }
            )
            continue
        out_mask = args.output_dir / "object_masks" / f"{frame_idx:06d}.png"
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_mask), refined.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write mask {out_mask}")
        original_area = int(mask.sum())
        refined_area = int(refined.sum())
        obj.update(
            {
                "status": "measured_vlm_points_sam_depth_refined",
                "mask_path": str(out_mask),
                "area_px": refined_area,
                "bbox_xyxy": [
                    float(np.where(refined)[1].min()),
                    float(np.where(refined)[0].min()),
                    float(np.where(refined)[1].max()),
                    float(np.where(refined)[0].max()),
                ],
                "depth_refinement": {
                    "positive_depth_median_m": center,
                    "depth_tolerance_m": float(args.depth_tolerance_m),
                    "original_area_px": original_area,
                    "refined_area_px": refined_area,
                    "components": records,
                },
            }
        )
        kept += 1
        reports.append(
            {
                "frame_idx": frame_idx,
                "reason": "ok",
                "original_area_px": original_area,
                "refined_area_px": refined_area,
                "positive_depth_median_m": center,
                "components": records,
            }
        )
        review_src = cv2.imread(str(args.review_stills_dir / f"{frame_idx:06d}.jpg"), cv2.IMREAD_COLOR) if args.review_stills_dir else None
        if review_src is not None:
            draw_review(review_src, mask, refined, prompt, args.output_dir / "review_stills" / f"{frame_idx:06d}.jpg")
    annotations_path = args.output_dir / "annotations_depth_refined.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "backend": "metric-depth compatible refinement of VLM point SAM masks",
        "annotations": str(annotations_path),
        "point_prompts": str(args.point_prompts),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frames_seen": len(prompts),
        "frames_kept": kept,
        "reports": reports,
    }
    (args.output_dir / "qc_depth_refined_masks.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "reports"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.18)
    parser.add_argument("--min-area-px", type=int, default=200)
    parser.add_argument("--review-stills-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
