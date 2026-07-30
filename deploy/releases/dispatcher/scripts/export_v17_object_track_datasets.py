#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np


STATUS = "v17_object_track_datasets_for_material_correspondence"
CLAIM = (
    "This artifact exports per-object RGB/mask/depth datasets for material-correspondence tracking. "
    "It does not run tracking and does not solve object pose."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def raw_frame_map(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = require_dict(load_json(path), f"{path}")
    frames = require_list(payload.get("frames"), f"{path}.frames")
    out: dict[int, dict[str, Any]] = {}
    for i, raw in enumerate(frames):
        row = require_dict(raw, f"{path}.frames[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"{path}.frames[{i}].frame_idx")
        rgb = Path(require_str(row.get("rgb"), f"{path}.frames[{i}].rgb"))
        if not rgb.exists():
            raise RuntimeError(f"raw RGB frame is missing: {rgb}")
        out[frame_idx] = row
    return out, payload


def load_depth(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=False)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing depth keys: {missing}")
    frame_idx = blob["frame_idx"].astype(np.int64)
    depth = blob["depth"].astype(np.float32)
    intrinsics = blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if depth.ndim != 3 or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(f"{path} has invalid depth/intrinsics shapes")
    return {
        "frame_idx": frame_idx,
        "depth": depth,
        "intrinsics": intrinsics,
        "frame_to_i": {int(idx): int(i) for i, idx in enumerate(frame_idx.tolist())},
    }


def write_depth_mm(path: Path, depth_m: np.ndarray) -> None:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float64) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"failed to write depth image: {path}")


def resize_mask_to_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape_hw:
        return mask
    return cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def resize_rgb_to_shape(rgb: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == shape_hw:
        return rgb
    return cv2.resize(rgb, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)


def object_frames(timeline: dict[str, Any], object_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw_frame in enumerate(require_list(timeline.get("frames"), "timeline.frames")):
        frame = require_dict(raw_frame, f"timeline.frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"timeline.frames[{i}].frame_idx")
        for j, raw_obj in enumerate(require_list(frame.get("objects"), f"timeline.frames[{i}].objects")):
            obj = require_dict(raw_obj, f"timeline.frames[{i}].objects[{j}]")
            if obj.get("object_id") == object_id:
                row = dict(obj)
                row["frame_idx"] = frame_idx
                rows.append(row)
    return rows


def export_object_dataset(
    case: str,
    object_id: str,
    object_rows: list[dict[str, Any]],
    raw_frames: dict[int, dict[str, Any]],
    depth: dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    track_id = object_id.removeprefix("object:")
    dataset_dir = output_root / case / track_id
    rgb_dir = dataset_dir / "rgb"
    mask_dir = dataset_dir / "masks"
    depth_dir = dataset_dir / "depth"
    for directory in (rgb_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for obj in object_rows:
        frame_idx = require_int(obj.get("frame_idx"), f"{case} {object_id} frame_idx")
        if obj.get("visible") is not True:
            rejected.append({"frame_idx": frame_idx, "reason": "object_not_visible", "annotation_ready": False})
            continue
        if frame_idx not in depth["frame_to_i"]:
            rejected.append({"frame_idx": frame_idx, "reason": "metric_depth_missing_for_frame", "annotation_ready": False})
            continue
        raw_row = raw_frames.get(frame_idx)
        if raw_row is None:
            rejected.append({"frame_idx": frame_idx, "reason": "raw_frame_missing", "annotation_ready": False})
            continue
        mask_path = Path(require_str(obj.get("mask_path"), f"{case} {object_id} frame {frame_idx} mask_path"))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            rejected.append({"frame_idx": frame_idx, "reason": f"mask_read_failed:{mask_path}", "annotation_ready": False})
            continue
        depth_i = int(depth["frame_to_i"][frame_idx])
        depth_m = depth["depth"][depth_i]
        shape_hw = (int(depth_m.shape[0]), int(depth_m.shape[1]))
        mask = resize_mask_to_shape(mask, shape_hw)
        mask_pixels = int(np.count_nonzero(mask > 0))
        if mask_pixels < int(args.min_mask_pixels):
            rejected.append({"frame_idx": frame_idx, "reason": "too_few_mask_pixels", "mask_pixels": mask_pixels, "annotation_ready": False})
            continue
        src_rgb = Path(require_str(raw_row.get("rgb"), f"{case} raw frame {frame_idx}.rgb"))
        rgb = cv2.imread(str(src_rgb), cv2.IMREAD_COLOR)
        if rgb is None:
            rejected.append({"frame_idx": frame_idx, "reason": f"rgb_read_failed:{src_rgb}", "annotation_ready": False})
            continue
        rgb = resize_rgb_to_shape(rgb, shape_hw)
        out_i = len(rows)
        rgb_path = rgb_dir / f"{out_i:06d}.jpg"
        out_mask_path = mask_dir / f"{out_i:06d}.png"
        depth_path = depth_dir / f"{out_i:06d}.png"
        if not cv2.imwrite(str(rgb_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"failed to write RGB frame: {rgb_path}")
        if not cv2.imwrite(str(out_mask_path), (mask > 0).astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write mask frame: {out_mask_path}")
        write_depth_mm(depth_path, depth_m)
        fx, fy, cx, cy = [float(v) for v in depth["intrinsics"][depth_i].tolist()]
        rows.append(
            {
                "index": out_i,
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "mask": str(out_mask_path),
                "depth": str(depth_path),
                "source_rgb": str(src_rgb),
                "source_mask": str(mask_path),
                "mask_pixels": mask_pixels,
                "intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )
    if len(rows) < int(args.min_exported_frames):
        status = "rejected_insufficient_exported_frames"
    else:
        status = STATUS
    if rows:
        intr = np.median(np.asarray([row["intrinsics_fx_fy_cx_cy"] for row in rows], dtype=np.float64), axis=0)
        cam_k = np.asarray([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
        np.savetxt(dataset_dir / "cam_K.txt", cam_k, fmt="%.10f")
    manifest = {
        "method": "export_v17_object_track_datasets",
        "status": status,
        "claim": CLAIM,
        "case": case,
        "object_id": object_id,
        "track_id": track_id,
        "dataset_dir": str(dataset_dir),
        "frame_count": len(rows),
        "rejected_frame_count": len(rejected),
        "first_frame": rows[0]["frame_idx"] if rows else None,
        "last_frame": rows[-1]["frame_idx"] if rows else None,
        "frames": rows,
        "rejected_rows": rejected,
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(dataset_dir / "manifest.json", manifest)
    return manifest


def build_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    measure_manifest_path = args.measurement_store_root / case / "v17_measurement_manifest.json"
    measure_manifest = require_dict(load_json(measure_manifest_path), f"{case} measurement manifest")
    v16_manifest = require_dict(load_json(Path(require_str(measure_manifest.get("manifest"), f"{case}.manifest"))), f"{case} v16 manifest")
    raw_frame_manifest = Path(require_str(measure_manifest.get("v16_root"), f"{case}.v16_root")) / "raw_frame_manifest" / "manifest.json"
    raw_frames, raw_payload = raw_frame_map(raw_frame_manifest)
    timeline_path = args.multi_object_timeline_root / case / "v17_multi_object_timeline.json"
    timeline = require_dict(load_json(timeline_path), f"{case} multi-object timeline")
    depth = load_depth(args.depth_root / case / "unidepth_metric" / "unidepth_metric_depth_v3.npz")
    object_ids = [
        require_str(row.get("object_id"), f"{case}.objects[{i}].object_id")
        for i, row in enumerate(require_list(timeline.get("objects"), f"{case}.objects"))
    ]
    manifests: list[dict[str, Any]] = []
    for object_id in object_ids:
        rows = object_frames(timeline, object_id)
        manifests.append(export_object_dataset(case, object_id, rows, raw_frames, depth, args.output_root, args))
    output = {
        "method": "export_v17_object_track_datasets",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_measurement_manifest": str(measure_manifest_path),
        "source_v16_manifest": str(measure_manifest.get("manifest")),
        "source_raw_frame_manifest": str(raw_frame_manifest),
        "source_multi_object_timeline": str(timeline_path),
        "clip": v16_manifest.get("clip"),
        "raw_video": v16_manifest.get("raw_video"),
        "raw_frame_manifest": {"status": raw_payload.get("status"), "frames": len(raw_frames)},
        "depth_frame_count": int(len(depth["frame_idx"])),
        "object_count": len(manifests),
        "exported_object_count": int(sum(1 for row in manifests if row["frame_count"] >= int(args.min_exported_frames))),
        "total_exported_frames": int(sum(int(row["frame_count"]) for row in manifests)),
        "total_rejected_frames": int(sum(int(row["rejected_frame_count"]) for row in manifests)),
        "objects": [
            {
                "object_id": row["object_id"],
                "manifest": str(Path(row["dataset_dir"]) / "manifest.json"),
                "frame_count": row["frame_count"],
                "rejected_frame_count": row["rejected_frame_count"],
                "status": row["status"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
            for row in manifests
        ],
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / case / "v17_object_track_dataset_summary.json", output)
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [build_case(args, case) for case in args.case]
    payload = {
        "method": "export_v17_object_track_datasets",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "cases": [
            {
                "case": case["case"],
                "summary": str(args.output_root / case["case"] / "v17_object_track_dataset_summary.json"),
                "object_count": case["object_count"],
                "exported_object_count": case["exported_object_count"],
                "total_exported_frames": case["total_exported_frames"],
                "total_rejected_frames": case["total_rejected_frames"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
            for case in cases
        ],
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_object_track_dataset_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--multi-object-timeline-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    parser.add_argument(
        "--depth-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument("--case", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-mask-pixels", type=int, default=200)
    parser.add_argument("--min-exported-frames", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
