#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    state = load_json(args.v21_state)
    ann = load_json(args.v21_annotations)
    depth_selection = load_json(args.depth_selection_report)
    primary = depth_selection.get("selected_primary_depth_camera")
    if not isinstance(primary, dict) or not primary.get("depth_archive"):
        raise ContractError("depth_selection_has_no_primary_depth_archive")
    accepted = ann.get("accepted_segmentation_tracks")
    if not isinstance(accepted, list) or not accepted:
        raise ContractError("annotations_have_no_accepted_segmentation_tracks")
    outputs = []
    for track in accepted:
        if not isinstance(track, dict):
            continue
        track_id = str(track["track_id"])
        track_path = Path(str(track["track_path"]))
        object_id = str(track.get("target_object_id") or track_id)
        output_dir = args.output_root / track_id
        cmd_args = [
            "--case",
            str(ann.get("case_id") or state.get("case_id") or "v21_case"),
            "--track-id",
            track_id,
            "--object-id",
            object_id,
            "--raw-frame-manifest",
            str(args.source_frame_manifest),
            "--sam2-track-json",
            str(track_path),
            "--depth-npz",
            str(primary["depth_archive"]),
            "--output-dir",
            str(output_dir),
            "--object-plan",
            str(args.object_plan),
            "--allow-camera-frame-world",
            "--pixel-stride",
            str(args.pixel_stride),
            "--max-points",
            str(args.max_points),
            "--min-valid-points",
            str(args.min_valid_points),
            "--min-depth-m",
            str(args.min_depth_m),
            "--max-depth-m",
            str(args.max_depth_m),
        ]
        command = [sys.executable, "scripts/build_v19_visible_geometry_from_sam2_depth.py", *cmd_args]
        subprocess.run(command, check=True)
        report = load_json(output_dir / "v19_visible_geometry_adapter_report.json")
        outputs.append(
            {
                "track_id": track_id,
                "object_id": object_id,
                "visible_geometry_report": str(output_dir / "v19_visible_geometry_adapter_report.json"),
                "annotations": report.get("outputs", {}).get("annotations"),
                "visible_metric_frame_count": int(report.get("visible_metric_frame_count", 0)),
                "output_frame_count": int(report.get("output_frame_count", 0)),
                "claim_scope": report.get("claim_scope"),
            }
        )
    if not outputs:
        raise ContractError("no_visible_surface_outputs")
    summary = {
        "schema": "v21_visible_surface_from_depth_summary.v0",
        "status": "ok",
        "method": "run_v21_visible_surface_from_depth",
        "case_id": ann.get("case_id") or state.get("case_id"),
        "v21_state": str(args.v21_state),
        "v21_annotations": str(args.v21_annotations),
        "depth_selection_report": str(args.depth_selection_report),
        "primary_depth_candidate_id": primary.get("candidate_id"),
        "primary_depth_archive": primary.get("depth_archive"),
        "outputs": outputs,
        "claim_scope": "V21 visible-surface geometry from accepted masks and selected metric depth. This is surfel/centroid measurement for object geometry/pose stages, not completed object mesh or rigid pose.",
    }
    write_json(args.output_summary, summary)
    state["objects"] = state.get("objects", {}) if isinstance(state.get("objects"), dict) else {}
    state["objects"].update(
        {
            "state": "visible_metric_surfaces_measured_mesh_pose_pending",
            "visible_surface_summary": str(args.output_summary),
            "mesh_pose_required_for_object_pose_claims": True,
        }
    )
    write_json(args.v21_state, state)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V21 visible object surface measurements from accepted masks and selected metric depth.")
    parser.add_argument("--v21-state", type=Path, required=True)
    parser.add_argument("--v21-annotations", type=Path, required=True)
    parser.add_argument("--depth-selection-report", type=Path, required=True)
    parser.add_argument("--source-frame-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=1600)
    parser.add_argument("--min-valid-points", type=int, default=40)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
