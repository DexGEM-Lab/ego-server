#!/usr/bin/env python3
"""V21 → V18 pipeline layout bridge.

Creates the V16/V17/V18 pipeline directory structure that the V18 evidence
builders expect, populated from V21 measurements. This bridges V21's
measurement format into the V18 pipeline's consumption layout.

Creates:
  v18_pipeline_layout/<case>/
    annotations_v16_full.json (minimal V16 annotation from V21)
    v16_full_pipeline_manifest.json
    v18_annotation_state.json
    annotations_v18_full.json (copy of V21's V18-compatible annotation)
    renders/
      overlay_mano_object.mp4 (from V21 overlay render)
      reconstruction_3d_world.mp4 (placeholder or V21 world)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np


def load_json(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None


def run(args):
    run_root = Path(args.run_root).resolve()
    case = run_root.name
    obj = args.object_id
    
    layout_dir = run_root / "v18_pipeline_layout" / case
    layout_dir.mkdir(parents=True, exist_ok=True)
    
    # Load V21 annotation
    ann_path = run_root / "state/annotations_v18_full_mano.json"
    if not ann_path.exists():
        ann_path = run_root / "state/annotations_v18_compatible.json"
    ann = load_json(ann_path)
    
    # 1. annotations_v18_full.json
    v18_ann_path = layout_dir / "annotations_v18_full.json"
    v18_ann_path.write_text(json.dumps(ann, indent=2))
    
    # 2. v16_full_pipeline_manifest.json
    manifest = load_json(run_root / "input/raw_frame_manifest/manifest.json")
    v16_manifest = {
        "case_id": case,
        "frame_count": len(manifest["frames"]),
        "fps": manifest.get("fps", 25.0),
        "width": manifest["frames"][0]["source_width"],
        "height": manifest["frames"][0]["source_height"],
        "video_path": str(run_root / "input/clips"),
        "annotation_file": str(v18_ann_path),
        "method": "v21_bridge_to_v18_layout",
    }
    (layout_dir / "v16_full_pipeline_manifest.json").write_text(json.dumps(v16_manifest, indent=2))
    
    # 3. annotations_v16_full.json (V16-compatible minimal annotation)
    v16_ann = {
        "case_id": case,
        "frame_count": len(manifest["frames"]),
        "fps": manifest.get("fps", 25.0),
        "width": manifest["frames"][0]["source_width"],
        "height": manifest["frames"][0]["source_height"],
        "frames": [],
    }
    for frame in ann["frames"]:
        fidx = frame["frame_idx"]
        v16_hands = []
        for h in frame.get("hands", []):
            mc = h.get("mano_candidate", h.get("metric_mano_state", {}))
            v16_hands.append({
                "hand_side": h.get("hand_side", h.get("side", "right")),
                "bbox_xyxy": h.get("bbox_xyxy", [0,0,0,0]),
                "backend": "WiLoR_v21",
                "joints3d_camera": mc.get("joints3d_camera", mc.get("joints3d_camera_metric", [])),
                "cam_t": mc.get("cam_t", mc.get("cam_t_metric", [0,0,1.5])),
                "source_intrinsics": mc.get("source_intrinsics", mc.get("intrinsics_manifest", [])),
                "detector_score": mc.get("detector_score", 0.5),
            })
        v16_objects = []
        for o in frame.get("objects", []):
            v16_objects.append({
                "object_id": o.get("object_id"),
                "track_id": o.get("track_id"),
                "label": o.get("label", o.get("track_id", "")),
                "bbox_xyxy": o.get("bbox_xyxy", [0,0,0,0]),
                "mask_path": o.get("mask_path", ""),
                "visible": o.get("visible", True),
            })
        v16_ann["frames"].append({
            "frame_idx": fidx,
            "hands": v16_hands,
            "objects": v16_objects,
            "camera": frame.get("camera", {}),
        })
    (layout_dir / "annotations_v16_full.json").write_text(json.dumps(v16_ann, indent=2))
    
    # 4. v18_annotation_state.json (V18 state consumed by occlusion scripts)
    v18_state = {
        "case_id": case,
        "frame_count": len(manifest["frames"]),
        "fps": manifest.get("fps", 25.0),
        "frames": [],
    }
    for frame in ann["frames"]:
        fidx = frame["frame_idx"]
        state_hands = []
        for h in frame.get("hands", []):
            state_hands.append({
                "hand_side": h.get("hand_side", h.get("side", "right")),
                "bbox_xyxy": h.get("bbox_xyxy", [0,0,0,0]),
                "visibility_state": h.get("visibility_state", "visible"),
                "mano_candidate": h.get("mano_candidate", {}),
                "metric_mano_state": h.get("metric_mano_state", {}),
            })
        state_objects = []
        for o in frame.get("objects", []):
            state_objects.append({
                "object_id": o.get("object_id"),
                "track_id": o.get("track_id"),
                "label": o.get("label", ""),
                "bbox_xyxy": o.get("bbox_xyxy", [0,0,0,0]),
                "mask_path": o.get("mask_path", ""),
                "visible_geometry_candidate": o.get("visible_geometry_candidate", {}),
                "reconstructed_geometry_pose": o.get("reconstructed_geometry_pose", {}),
            })
        v18_state["frames"].append({
            "frame_idx": fidx,
            "hands": state_hands,
            "objects": state_objects,
            "camera": frame.get("camera", {}),
        })
    (layout_dir / "v18_annotation_state.json").write_text(json.dumps(v18_state, indent=2))
    
    # 5. Copy overlay render as V16 overlay
    renders_dir = layout_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    overlay_src = run_root / "renders/v21_overlay.mp4"
    if overlay_src.exists():
        shutil.copy2(overlay_src, renders_dir / "overlay_mano_object.mp4")
    
    print(json.dumps({"status": "ok", "layout_dir": str(layout_dir), "case": case}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    run(args)
