#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_vggt_wilor_object_annotations_v3 import load_mask_track, load_vggt, object_row, video_info
from run_v1_wilor_colmap import caption_for_frame, load_actions


def run(args: argparse.Namespace) -> dict:
    info = video_info(args.video)
    vggt = load_vggt(args.vggt_archive)
    actions = load_actions(args.actions_json)
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
        T = np.asarray(vggt["transforms"][idx], dtype=np.float64)
        intr = np.asarray(vggt["intrinsics"][idx], dtype=np.float64)
        caption = caption_for_frame(actions, idx)
        if not caption:
            caption = args.default_caption
        frames.append(
            {
                "frame_idx": int(idx),
                "time_s": float(idx / info["fps"]),
                "caption": caption,
                "camera": {
                    "T_world_camera_metric": T.astype(float).tolist(),
                    "position_world_m": T[:3, 3].astype(float).tolist(),
                    "vggt_source_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
                    "pose_source_status": "v3_vggt_native_metric_depth_scaled_local_world",
                    "vggt_to_meters": float(vggt["vggt_to_meters"]),
                    "anchor_frame": int(vggt["anchor_frame"]),
                },
                "hands": [],
                "object": object_row(idx, mask_track, (int(info["width"]), int(info["height"]))),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations_v3_vggt_object_skeleton.json"
    annotations_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    measured_object = [
        frame for frame in frames if frame["object"].get("status") == "measured_vlm_sam2_mask_unidepth_observed_surface"
    ]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "build_vggt_object_annotation_skeleton_v3",
        "video": str(args.video),
        "vggt_archive": str(args.vggt_archive),
        "mask_track": str(args.mask_track),
        "annotations": str(annotations_path),
        "frames": int(len(frames)),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "camera": {
            "scale_status": "VGGT native camera scaled by verified object-mask metric-depth/VGGT depth ratio",
            "vggt_to_meters": float(vggt["vggt_to_meters"]),
            "anchor_frame": int(vggt["anchor_frame"]),
        },
        "hands": {
            "status": "empty_skeleton_waiting_for_real_mano_stream",
            "rows": 0,
        },
        "object": {
            "measured_mask_frames": int(len(measured_object)),
            "measured_mask_frame_indices": [int(frame["frame_idx"]) for frame in measured_object],
            "status": "sparse_observed_surface_only_until_dense_masks_and_complete_mesh_are_available",
        },
    }
    (args.output_dir / "qc_vggt_object_annotation_skeleton_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--actions-json", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--default-caption", default="Peeling wild rice stem by hand")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
