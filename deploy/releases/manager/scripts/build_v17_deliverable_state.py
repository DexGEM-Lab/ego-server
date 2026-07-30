#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    write_json,
)


STATUS = "v17_deliverable_state_manifest_qc"
CLAIM = (
    "This artifact assembles the V17 render-state manifest from the interior-owned baked hand "
    "annotations and the multi-object world mesh archive. It verifies frame-count agreement with the "
    "raw video contract and emits one manifest per case for the duration-honest render path. It is a "
    "render-input assembly layer: visual quality, object completeness, contact closure, and all V17 "
    "readiness flags are decided elsewhere and remain false here."
)

OBJECT_LIMIT_FIELDS = {
    "multi_object_timeline_ready": False,
    "object_schema_status": "multi_object_partial_reconstruction_stream",
    "missing_multi_object_roster_required": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "object_geometry_status": "partial_full_interval_reconstruction_subset",
}


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def case_manifest(case: str, args: argparse.Namespace) -> dict[str, Any]:
    v16_manifest_path = existing_path(
        args.v16_root / case / "v16_full_pipeline_manifest.json",
        f"{case} v16 manifest",
    )
    v16 = require_dict(load_json(v16_manifest_path), f"{case} v16 manifest")
    raw_video = require_dict(v16.get("raw_video"), f"{case} raw_video")
    raw_frame_count = require_int(raw_video.get("frame_count"), f"{case} raw frame count")
    annotations_path = existing_path(
        args.hand_annotations_root / case / "annotations_v17_interior_owned_hands.json",
        f"{case} baked hand annotations",
    )
    annotations = require_dict(load_json(annotations_path), f"{case} baked annotations")
    frames = require_list(annotations.get("frames"), f"{case} annotation frames")
    if len(frames) != raw_frame_count:
        raise RuntimeError(
            f"{case} baked annotations have {len(frames)} frames but raw video has {raw_frame_count}"
        )
    baking = require_dict(annotations.get("v17_hand_state_baking"), f"{case} hand baking stamp")
    archive_report_path = existing_path(
        args.mesh_archive_root / case / "v17_multi_object_world_mesh_archive_report.json",
        f"{case} multi-object mesh archive report",
    )
    archive_report = require_dict(load_json(archive_report_path), f"{case} mesh archive report")
    archive_path_raw = archive_report.get("archive_path")
    if archive_path_raw is None:
        raise RuntimeError(
            f"{case} multi-object mesh archive has no frames; refusing to assemble a render state "
            "without any reconstructed object stream"
        )
    archive_path = existing_path(Path(require_str(archive_path_raw, "archive path")), f"{case} mesh archive")
    coverage = archive_report.get("archive_coverage_fraction_of_annotation")
    manifest = {
        "case": case,
        "status": STATUS,
        "claim": CLAIM,
        "artifact_status": "partial",
        "artifact_kind": "v17_deliverable_render_state",
        "delivery_role": "render_input_pending_quality_review",
        "method": "build_v17_deliverable_state",
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FIELDS,
        "object_pose_semantics": (
            "Object stream contains only evaluator-accepted full-interval BundleSDF reconstructions; "
            "objects without accepted reconstructions are absent from the mesh archive."
        ),
        "object_geometry_semantics": (
            "Per-frame world meshes come from canonical reconstructed geometry posed by "
            "T_world_camera @ ob_in_cam; deformation states are not represented."
        ),
        "v16_manifest": str(v16_manifest_path),
        "raw_frame_count": raw_frame_count,
        "annotations": str(annotations_path),
        "object_mesh_archive": str(archive_path),
        "object_mesh_archive_report": str(archive_report_path),
        "object_archive_frame_count": require_int(
            archive_report.get("archive_frame_count"),
            "archive frame count",
        ),
        "object_archive_coverage_fraction": coverage,
        "archived_object_count": require_int(
            archive_report.get("archived_object_count"),
            "archived object count",
        ),
        "baked_hand_rows": require_int(baking.get("baked_hand_rows"), "baked hand rows"),
        "kept_prior_hand_rows": require_int(baking.get("kept_prior_hand_rows"), "kept prior hand rows"),
        "solver_status": "v17_interior_owned_hand_graph_plus_full_interval_reconstruction_subset",
        **FALSE_READY,
    }
    out_path = args.output_root / case / "v17_deliverable_state_manifest.json"
    write_json(out_path, manifest)
    return {**manifest, "manifest_path": str(out_path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_manifest(case, args) for case in args.cases]
    summary = {
        "method": "build_v17_deliverable_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "manifest_path": require_str(case.get("manifest_path"), "manifest path"),
                "raw_frame_count": case["raw_frame_count"],
                "object_archive_frame_count": case["object_archive_frame_count"],
                "archived_object_count": case["archived_object_count"],
                "baked_hand_rows": case["baked_hand_rows"],
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_deliverable_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v16-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"),
    )
    parser.add_argument(
        "--hand-annotations-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_interior_owned_hand_annotations"),
    )
    parser.add_argument(
        "--mesh-archive-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_world_mesh_archive"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_deliverable_state"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
