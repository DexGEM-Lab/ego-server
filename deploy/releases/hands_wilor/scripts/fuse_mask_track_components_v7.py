#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def frame_rows(path: Path) -> list[dict]:
    rows = load_json(path).get("frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return rows


def manifest_by_frame(path: Path) -> dict[int, dict]:
    return {int(row["frame_idx"]): row for row in frame_rows(path)}


def prompt_by_source_frame(path: Path) -> tuple[dict[int, dict], dict]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain a nonempty point_prompts list")
    out = {}
    for row in rows:
        source_idx = int(row.get("source_frame_idx", row["frame_idx"]))
        out[source_idx] = row
    return out, payload


def read_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    if shape_hw is not None and tuple(mask.shape[:2]) != tuple(shape_hw):
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def mask_from_track(track: dict, frame_idx: int, shape_hw: tuple[int, int]) -> np.ndarray:
    row = track.get(str(frame_idx)) or track.get(frame_idx)
    if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
        return np.zeros(shape_hw, dtype=bool)
    return read_mask(Path(str(row["mask_path"])), shape_hw)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0


def point_xy(prompt: dict, key: str, mask_shape: tuple[int, int], prompt_payload: dict) -> np.ndarray:
    rows = prompt.get(key, [])
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    prompt_w = int(prompt_payload.get("prompt_image_width", mask_shape[1]))
    prompt_h = int(round(prompt_w * mask_shape[0] / mask_shape[1]))
    scale = np.asarray([mask_shape[1] / float(prompt_w), mask_shape[0] / float(prompt_h)], dtype=np.float32)
    xy = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)
    return xy * scale[None, :]


def count_hits(mask: np.ndarray, xy: np.ndarray) -> int:
    if len(xy) == 0:
        return 0
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return int(np.count_nonzero(mask[y, x]))


def component_rows(
    repair: np.ndarray,
    parent: np.ndarray,
    prompt: dict | None,
    prompt_payload: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    count, labels, stats, centers = cv2.connectedComponentsWithStats(repair.astype(np.uint8), 8)
    parent_distance = cv2.distanceTransform((~parent).astype(np.uint8), cv2.DIST_L2, 3)
    positive_xy = point_xy(prompt, "positive_points", repair.shape, prompt_payload) if prompt is not None else np.zeros((0, 2), dtype=np.float32)
    negative_xy = point_xy(prompt, "negative_points", repair.shape, prompt_payload) if prompt is not None else np.zeros((0, 2), dtype=np.float32)
    selected = np.zeros(repair.shape, dtype=bool)
    rows = []
    for label in range(1, count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        pos_hits = count_hits(comp, positive_xy)
        neg_hits = count_hits(comp, negative_xy)
        min_parent_distance_px = float(np.min(parent_distance[comp])) if np.any(parent) else float("inf")
        prompt_supported = bool(prompt is not None and pos_hits >= int(args.min_prompt_positive_hits) and neg_hits <= int(args.max_negative_hits))
        parent_supported = bool(min_parent_distance_px <= float(args.max_parent_distance_px))
        area_supported = bool(area >= int(args.min_component_area_px))
        keep = bool(area_supported and (prompt_supported or (prompt is None and parent_supported)))
        if keep:
            selected |= comp
        rows.append(
            {
                "component_label": int(label),
                "area_px": area,
                "center_xy": [float(centers[label, 0]), float(centers[label, 1])],
                "bbox_xywh": [int(v) for v in stats[label].tolist()],
                "positive_hits": int(pos_hits),
                "positive_points": int(len(positive_xy)),
                "negative_hits": int(neg_hits),
                "negative_points": int(len(negative_xy)),
                "min_parent_distance_px": min_parent_distance_px,
                "area_supported": area_supported,
                "prompt_supported": prompt_supported,
                "parent_supported": parent_supported,
                "selected": keep,
            }
        )
    return selected, rows


def render_review(rgb_path: Path, parent: np.ndarray, repair: np.ndarray, exclusion: np.ndarray, fused: np.ndarray, output: Path) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read RGB frame: {rgb_path}")
    if tuple(image.shape[:2]) != tuple(fused.shape):
        image = cv2.resize(image, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = image.astype(np.float32)
    colors = [
        (parent, np.asarray([0, 0, 255], dtype=np.float32)),
        (repair, np.asarray([255, 255, 0], dtype=np.float32)),
        (exclusion, np.asarray([0, 255, 0], dtype=np.float32)),
        (fused, np.asarray([0, 180, 255], dtype=np.float32)),
    ]
    for mask, color in colors:
        overlay[mask] = overlay[mask] * 0.58 + color * 0.42
    cv2.putText(
        overlay,
        "parent red, added cyan, exclusion green, fused orange",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.clip(overlay, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"failed to write review: {output}")


def run(args: argparse.Namespace) -> dict:
    parent_manifest = manifest_by_frame(args.parent_manifest)
    repair_track = load_json(args.repair_track)
    exclusion_track = load_json(args.exclusion_track) if args.exclusion_track is not None else {}
    prompts, prompt_payload = prompt_by_source_frame(args.point_prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    review_dir = args.output_dir / "review"
    depth_dir = args.output_dir / "depth"
    rgb_dir = args.output_dir / "rgb"
    for directory in (mask_dir, review_dir, depth_dir, rgb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    out_rows = []
    qc_rows = []
    for out_i, frame_idx in enumerate(range(int(args.frame_start), int(args.frame_end) + 1)):
        parent_row = parent_manifest.get(frame_idx)
        if parent_row is None:
            raise RuntimeError(f"parent manifest lacks frame {frame_idx}")
        parent_mask_path = Path(str(parent_row.get("mask") or parent_row.get("source_mask")))
        parent = read_mask(parent_mask_path)
        repair_raw = mask_from_track(repair_track, frame_idx, parent.shape)
        exclusion = dilate(mask_from_track(exclusion_track, frame_idx, parent.shape), int(args.exclusion_dilate_px))
        repair_candidate = repair_raw & ~exclusion & ~parent
        selected_repair, components = component_rows(repair_candidate, parent, prompts.get(frame_idx), prompt_payload, args)
        selected_repair = morph_close(selected_repair, int(args.close_px)) & ~exclusion
        fused = parent | selected_repair
        if int(np.count_nonzero(fused)) < int(args.min_fused_area_px):
            raise RuntimeError(f"frame {frame_idx} fused mask has too few pixels")
        mask_path = mask_dir / f"{out_i:06d}.png"
        if not cv2.imwrite(str(mask_path), fused.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        rgb_src = Path(str(parent_row["rgb"]))
        depth_src = Path(str(parent_row["depth"]))
        rgb_dst = rgb_dir / f"{out_i:06d}{rgb_src.suffix or '.png'}"
        depth_dst = depth_dir / f"{out_i:06d}{depth_src.suffix or '.png'}"
        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(depth_src, depth_dst)
        render_review(rgb_src, parent, selected_repair, exclusion, fused, review_dir / f"{frame_idx:06d}.jpg")
        out_row = dict(parent_row)
        out_row.update(
            {
                "index": int(out_i),
                "frame_idx": int(frame_idx),
                "rgb": str(rgb_dst),
                "depth": str(depth_dst),
                "mask": str(mask_path),
                "source_parent_mask": str(parent_mask_path),
                "source_repair_track": str(args.repair_track),
                "source_exclusion_track": str(args.exclusion_track) if args.exclusion_track is not None else None,
                "track_id": str(prompt_payload.get("track_id", "fused_mask_track")),
                "label": str(prompt_payload.get("description", "fused object mask")),
            }
        )
        out_rows.append(out_row)
        qc_rows.append(
            {
                "frame_idx": int(frame_idx),
                "parent_pixels": int(np.count_nonzero(parent)),
                "repair_raw_pixels": int(np.count_nonzero(repair_raw)),
                "exclusion_pixels": int(np.count_nonzero(exclusion)),
                "selected_repair_pixels": int(np.count_nonzero(selected_repair)),
                "fused_pixels": int(np.count_nonzero(fused)),
                "components": components,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": out_rows}, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "fuse_mask_track_components_v7",
        "parent_manifest": str(args.parent_manifest),
        "repair_track": str(args.repair_track),
        "exclusion_track": str(args.exclusion_track) if args.exclusion_track is not None else None,
        "point_prompts": str(args.point_prompts),
        "manifest": str(manifest_path),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(out_rows)),
        "selection_contract": {
            "min_component_area_px": int(args.min_component_area_px),
            "min_prompt_positive_hits": int(args.min_prompt_positive_hits),
            "max_negative_hits": int(args.max_negative_hits),
            "max_parent_distance_px": float(args.max_parent_distance_px),
            "exclusion_dilate_px": int(args.exclusion_dilate_px),
        },
        "rows": qc_rows,
    }
    (args.output_dir / "qc_fuse_mask_track_components_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "method", "manifest", "frames")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--repair-track", type=Path, required=True)
    parser.add_argument("--exclusion-track", type=Path)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-component-area-px", type=int, default=120)
    parser.add_argument("--min-prompt-positive-hits", type=int, default=1)
    parser.add_argument("--max-negative-hits", type=int, default=0)
    parser.add_argument("--max-parent-distance-px", type=float, default=18.0)
    parser.add_argument("--exclusion-dilate-px", type=int, default=4)
    parser.add_argument("--close-px", type=int, default=1)
    parser.add_argument("--min-fused-area-px", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
