#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def mask_bool(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8) > 0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_pad_bbox(box: tuple[int, int, int, int], width: int, height: int, pad_frac: float = 0.08) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    side = int(round(max(bw, bh) * (1.0 + 2.0 * pad_frac)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx1 = sx0 + side
    sy1 = sy0 + side
    if sx0 < 0:
        sx1 -= sx0
        sx0 = 0
    if sy0 < 0:
        sy1 -= sy0
        sy0 = 0
    if sx1 > width:
        sx0 -= sx1 - width
        sx1 = width
    if sy1 > height:
        sy0 -= sy1 - height
        sy1 = height
    sx0 = max(0, sx0)
    sy0 = max(0, sy0)
    sx1 = min(width, sx1)
    sy1 = min(height, sy1)
    return sx0, sy0, sx1, sy1


def make_rgba_crop(raw_path: Path, mask_path: Path, out_path: Path) -> dict[str, Any]:
    raw = Image.open(raw_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L").resize(raw.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    box = bbox_from_mask(mask)
    if box is None:
        raise RuntimeError(f"empty mask: {mask_path}")
    crop_box = square_pad_bbox(box, raw.width, raw.height)
    raw_arr = np.asarray(raw, dtype=np.uint8).copy()
    raw_arr[~mask] = 0
    rgba = Image.fromarray(np.dstack([raw_arr, mask.astype(np.uint8) * 255]), mode="RGBA")
    crop = rgba.crop(crop_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return {
        "raw_image": str(raw_path),
        "mask": str(mask_path),
        "crop_rgba": str(out_path),
        "source_size": [raw.width, raw.height],
        "mask_bbox_xyxy": list(box),
        "crop_bbox_xyxy": list(crop_box),
        "mask_area_px": int(mask.sum()),
        "crop_size": list(crop.size),
    }


def find_depth_fused_object(report_path: Path, object_id: str) -> dict[str, Any]:
    report = load_json(report_path)
    for row in report.get("object_rows", []):
        if isinstance(row, dict) and row.get("object_id") == object_id:
            return row
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-fused-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_compact_rigid_completion"))
    parser.add_argument("--selected-frame-idx", type=int, default=None, help="documented override when the top support mask is visually/geometrically invalid")
    parser.add_argument("--selection-note", default=None)
    args = parser.parse_args()

    ann = load_json(args.annotations)
    object_safe = safe_id(args.object_id.replace("object:", "object_"))
    out_dir = args.output_root / args.case / object_safe / "evidence_bundle"
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for frame in ann.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx"))
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if not isinstance(obj, dict) or obj.get("object_id") != args.object_id:
                continue
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            raw_text = str(frame.get("raw_frame_path") or "")
            mask_text = str(obj.get("mask_path") or "")
            if not raw_text or not mask_text:
                continue
            raw_path = Path(raw_text)
            mask_path = Path(mask_text)
            if not raw_path.is_file() or not mask_path.is_file():
                continue
            try:
                mask = mask_bool(mask_path)
                mask_area = int(mask.sum())
            except Exception:
                mask_area = 0
            visible_vertices = int(geom.get("vertex_count") or len(geom.get("world_vertices_sample_m") or []))
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "raw_frame_path": str(raw_path),
                    "mask_path": str(mask_path),
                    "visible_depth_vertex_count": visible_vertices,
                    "mask_area_px": mask_area,
                    "camera": frame.get("camera"),
                    "visible_geometry_candidate": geom,
                    "object_bbox_xyxy": obj.get("bbox_xyxy"),
                }
            )

    if not candidates:
        raise RuntimeError(f"no evidence candidates for {args.case} {args.object_id}")
    candidates.sort(key=lambda r: (-int(r["visible_depth_vertex_count"]), -int(r["mask_area_px"]), int(r["frame_idx"])))
    if args.selected_frame_idx is not None:
        matches = [c for c in candidates if int(c["frame_idx"]) == int(args.selected_frame_idx)]
        if not matches:
            raise RuntimeError(f"selected override frame {args.selected_frame_idx} is not a candidate for {args.object_id}")
        selected = dict(matches[0])
        selection_rule = "documented_selected_frame_override_due_invalid_top_mask_or_conditioning_evidence"
    else:
        selected = dict(candidates[0])
        selection_rule = "max_visible_depth_vertex_count_then_mask_area_then_earliest_frame"
    selected_crop = crop_dir / f"frame_{int(selected['frame_idx']):06d}_{object_safe}_rgba.png"
    selected["trellis_conditioning_crop"] = make_rgba_crop(Path(selected["raw_frame_path"]), Path(selected["mask_path"]), selected_crop)

    depth_fused = find_depth_fused_object(args.depth_fused_report, args.object_id)
    partial_mesh_paths = {}
    if isinstance(depth_fused.get("mesh_reconstruction"), dict):
        for key in ["fused_point_cloud_path", "poisson_mesh_path", "convex_hull_mesh_path"]:
            value = depth_fused["mesh_reconstruction"].get(key)
            if value:
                partial_mesh_paths[key] = value
    report = {
        "method": "build_v18_compact_rigid_evidence_bundle",
        "status": "ok",
        "case": args.case,
        "object_id": args.object_id,
        "output_dir": str(out_dir),
        "selection_rule": selection_rule,
        "selection_note": args.selection_note,
        "selected_frame_idx": int(selected["frame_idx"]),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidate_summary": {
            "visible_depth_vertex_count_max": int(candidates[0]["visible_depth_vertex_count"]),
            "visible_depth_vertex_count_median": float(np.median([c["visible_depth_vertex_count"] for c in candidates])),
            "mask_area_px_max": int(max(c["mask_area_px"] for c in candidates)),
        },
        "depth_fused_object_row": depth_fused,
        "partial_metric_geometry_paths": partial_mesh_paths,
        "all_candidate_frames": [
            {k: c[k] for k in ["frame_idx", "visible_depth_vertex_count", "mask_area_px", "raw_frame_path", "mask_path"]}
            for c in candidates
        ],
    }
    (out_dir / "evidence_bundle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
