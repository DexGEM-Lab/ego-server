#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    review = load_json(args.segmentation_review)
    raw_manifest = load_json(Path(str(input_manifest["raw_frame_manifest"])))
    accepted = [row for row in review.get("tracks", []) if isinstance(row, dict) and str(row.get("decision", "")).startswith("accept")]
    if not accepted:
        raise ContractError("no_accepted_segmentation_tracks")
    accepted_by_track = {str(row["track_id"]): row for row in accepted}
    track_payloads = {track_id: load_json(Path(str(row["track_path"]))) for track_id, row in accepted_by_track.items()}
    frames_out = []
    for frame in raw_manifest.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        objects = []
        for track_id, track in track_payloads.items():
            row = track.get(str(frame_idx))
            if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
                continue
            review_row = accepted_by_track[track_id]
            objects.append(
                {
                    "object_id": review_row.get("target_object_id") or track_id,
                    "track_id": track_id,
                    "visible": True,
                    "mask_path": row["mask_path"],
                    "bbox_xyxy": row.get("bbox_xyxy"),
                    "center_xy": row.get("center_xy"),
                    "area_px": row.get("area_px"),
                    "segmentation_state": "accepted_sam2_proper_owlv2_bbox_prompt_visible_mask_evidence",
                    "geometry_state": "not_reconstructed_yet",
                    "pose_state": "not_solved_yet",
                    "uncertainty": {
                        "mask_contamination_review": review_row.get("decision"),
                        "flags": review_row.get("flags", []),
                        "depth_camera_selected": False,
                        "mesh_pose_solved": False,
                    },
                }
            )
        frames_out.append({"frame_idx": frame_idx, "time_s": frame.get("time_s"), "rgb": frame.get("rgb"), "objects": objects, "hands": []})
    annotations = {
        "schema": "v21_renderable_annotations.v0",
        "mode": "v21_infer",
        "case_id": input_manifest.get("case_id"),
        "input_manifest": str(args.input_manifest),
        "source_video": input_manifest.get("primary_video"),
        "timeline": {
            "frame_count": int(len(frames_out)),
            "fps": float(input_manifest["primary_video_metadata"]["fps"]),
            "resolution": [int(input_manifest["primary_video_metadata"]["width"]), int(input_manifest["primary_video_metadata"]["height"])],
        },
        "accepted_segmentation_tracks": list(accepted_by_track.values()),
        "frames": frames_out,
        "claim_scope": "Renderable V21 annotation state from accepted SAM2 masks propagated from approved OWLv2 bbox prompts. It does not yet contain metric MANO, object mesh pose, contact, occlusion ownership, or nonpenetration claims.",
    }
    state = load_json(args.state_in) if args.state_in and args.state_in.exists() else {"schema": "v21_physical_state.v0"}
    state.update(
        {
            "schema": "v21_physical_state.v0",
            "status": "segmentation_measured_depth_hand_geometry_pending",
            "case_id": input_manifest.get("case_id"),
            "run_root": input_manifest.get("run_root"),
            "segmentation": {
                "state": "sam2_proper_owlv2_bbox_prompt_segmentation_accepted_for_visible_mask_evidence",
                "review_report": str(args.segmentation_review),
                "accepted_track_count": int(len(accepted)),
                "accepted_tracks": list(accepted_by_track),
            },
            "camera_depth": state.get("camera_depth", {"state": "unmeasured", "required_for_metric_claims": True}),
            "hands": state.get("hands", {"state": "unmeasured", "metric_mano_required_for_contact": True}),
            "objects": {
                "state": "visible_masks_measured_geometry_pose_pending",
                "accepted_visible_mask_tracks": list(accepted_by_track),
                "mesh_pose_required_for_object_pose_claims": True,
            },
            "render_inputs": {"annotations": str(args.output_annotations)},
            "renderer_boundary": "V21 render currently consumes accepted SAM2 proper segmentation masks only; final physical render must be extended with selected depth/camera, MANO, and object mesh pose states.",
        }
    )
    write_json(args.output_annotations, annotations)
    write_json(args.output_state, state)
    summary = {
        "schema": "v21_segmentation_state_assembly_summary.v0",
        "status": "ok",
        "method": "assemble_v21_segmentation_state",
        "annotations": str(args.output_annotations),
        "state": str(args.output_state),
        "frames": int(len(frames_out)),
        "accepted_track_count": int(len(accepted)),
        "claim_scope": annotations["claim_scope"],
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble renderable V21 state from accepted SAM2 segmentation evidence.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-review", type=Path, required=True)
    parser.add_argument("--state-in", type=Path)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
