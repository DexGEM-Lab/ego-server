#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_vggt_wilor_object_annotations_v3 import load_mask_track, load_vggt, video_info
from run_v1_wilor_colmap import caption_for_frame, load_actions


def object_row(frame_idx: int, mask_track: dict[int, dict], source_size: tuple[int, int], label: str, track_id: str) -> dict:
    row = mask_track.get(frame_idx)
    if row is None or not row.get("visible") or not row.get("mask_path"):
        return {"label": label, "track_id": track_id, "status": "unobserved_no_verified_mask"}
    mask_path = Path(str(row["mask_path"]))
    if not mask_path.exists():
        raise RuntimeError(f"missing object mask for frame {frame_idx}: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object mask {mask_path}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError(f"empty object mask {mask_path}")
    return {
        "label": label,
        "track_id": track_id,
        "status": "measured_vlm_sam2_mask_unidepth_observed_surface",
        "mask_path": str(mask_path),
        "mask_image_size": [int(mask.shape[1]), int(mask.shape[0])],
        "source_image_size": [int(source_size[0]), int(source_size[1])],
        "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        "center_xy": [float(np.mean(xs)), float(np.mean(ys))],
        "area_px": int(len(xs)),
        "mesh_status": "mask_depth_evidence_waiting_for_complete_mesh_alignment",
        "sam2_reason": row.get("reason"),
    }


def run(args: argparse.Namespace) -> dict:
    info = video_info(args.video)
    vggt = load_vggt(args.vggt_archive)
    actions = load_actions(args.actions_json) if args.actions_json is not None else []
    mask_track = load_mask_track(args.mask_track)
    frame_indices = [int(idx) for idx in vggt["frame_idx"].tolist()]
    if args.frame_start is not None:
        frame_indices = [idx for idx in frame_indices if idx >= int(args.frame_start)]
    if args.frame_end is not None:
        frame_indices = [idx for idx in frame_indices if idx <= int(args.frame_end)]
    if not frame_indices:
        raise RuntimeError("no VGGT frames remain after frame filtering")
    frames = []
    for idx in frame_indices:
        transform = np.asarray(vggt["transforms"][idx], dtype=np.float64)
        intrinsics = np.asarray(vggt["intrinsics"][idx], dtype=np.float64)
        caption = caption_for_frame(actions, idx) if actions else ""
        if not caption:
            caption = args.default_caption
        frames.append(
            {
                "frame_idx": int(idx),
                "time_s": float(idx / info["fps"]),
                "caption": caption,
                "camera": {
                    "T_world_camera_metric": transform.astype(float).tolist(),
                    "position_world_m": transform[:3, 3].astype(float).tolist(),
                    "vggt_source_intrinsics_fx_fy_cx_cy": intrinsics.astype(float).tolist(),
                    "pose_source_status": "v3_vggt_native_metric_depth_scaled_local_world",
                    "vggt_to_meters": float(vggt["vggt_to_meters"]),
                    "anchor_frame": int(vggt["anchor_frame"]),
                },
                "hands": [],
                "object": object_row(idx, mask_track, (int(info["width"]), int(info["height"])), args.object_label, args.track_id),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "annotations_v3_vggt_object_skeleton.json"
    out_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    measured = [frame for frame in frames if str(frame["object"].get("status")) == "measured_vlm_sam2_mask_unidepth_observed_surface"]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "build_vggt_object_skeleton_generic_v3",
        "video": str(args.video),
        "vggt_archive": str(args.vggt_archive),
        "mask_track": str(args.mask_track),
        "annotations": str(out_path),
        "frames": int(len(frames)),
        "measured_object_frames": int(len(measured)),
        "measured_object_frame_indices": [int(frame["frame_idx"]) for frame in measured],
        "object_label": str(args.object_label),
        "track_id": str(args.track_id),
        "vggt_to_meters": float(vggt["vggt_to_meters"]),
        "anchor_frame": int(vggt["anchor_frame"]),
    }
    (args.output_dir / "qc_vggt_object_skeleton_generic_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-label", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--actions-json", type=Path)
    parser.add_argument("--default-caption", required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
