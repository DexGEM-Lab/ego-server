#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


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


def video_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {path}")
    meta = {
        "path": str(path),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    meta["duration_s"] = float(meta["frame_count"] / meta["fps"]) if meta["fps"] > 0 else None
    return meta


def same_timeline(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        int(a.get("frame_count", -1)) == int(b.get("frame_count", -2))
        and abs(float(a.get("fps", -1.0)) - float(b.get("fps", -2.0))) < 1e-3
        and int(a.get("width", -1)) == int(b.get("width", -2))
        and int(a.get("height", -1)) == int(b.get("height", -2))
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.input_manifest)
    primary_video = Path(str(manifest["primary_video"]))
    primary_meta = video_meta(primary_video)
    candidates: list[dict[str, Any]] = []
    candidates.append(
        {
            "candidate_id": "monocular_depthpro_unidepth_baseline",
            "kind": "monocular_rgb_metric_depth_baseline",
            "source_video": str(primary_video),
            "required": True,
            "selection_rule": "must_run_before_assisted_depth_selection",
            "status": "registered_not_run",
        }
    )
    stereo_right = manifest.get("stereo_right_video")
    if stereo_right:
        right_path = Path(str(stereo_right))
        right_meta = video_meta(right_path)
        synchronized = same_timeline(primary_meta, right_meta)
        candidates.append(
            {
                "candidate_id": "stereo_disparity_candidate",
                "kind": "stereo_or_side_by_side_depth_candidate",
                "left_video": str(primary_video),
                "right_video": str(right_path),
                "right_video_metadata": right_meta,
                "synchronized_shape_fps_count": bool(synchronized),
                "calibration_status": "unknown_unless_sidecar_manifest_supplies_K_baseline_rectification",
                "metric_depth_available": False,
                "default_backend": "opencv_sgbm_smoke_then_raft_stereo_or_igev_when_calibration_available",
                "status": "registered_not_run",
                "claim_scope": "Without K/baseline/rectification this is relative inverse-depth evidence only; it cannot support metric geometry/contact claims.",
            }
        )
    multiview = manifest.get("multiview_sources")
    if isinstance(multiview, list) and multiview:
        synchronized_count = 0
        for row in multiview:
            meta = row.get("metadata") if isinstance(row, dict) else None
            if isinstance(meta, dict) and same_timeline(primary_meta, meta):
                synchronized_count += 1
        candidates.append(
            {
                "candidate_id": "multiview_camera_depth_candidate",
                "kind": "multiview_camera_depth_candidate",
                "source_count": int(len(multiview)),
                "synchronized_with_primary_count": int(synchronized_count),
                "default_backend": "vggt_existing_runner_then_mast3r_external_candidate",
                "calibration_status": "not_found_in_input_manifest",
                "metric_depth_available": False,
                "status": "registered_not_run",
                "claim_scope": "Uncalibrated multiview can provide relative camera/depth constraints; metric use requires scale/intrinsics validation against monocular/native evidence.",
            }
        )
    report = {
        "schema": "v21_depth_modality_report.v0",
        "status": "ok",
        "method": "build_v21_depth_modality_report",
        "case_id": manifest.get("case_id"),
        "run_root": manifest.get("run_root"),
        "input_manifest": str(args.input_manifest),
        "primary_video": str(primary_video),
        "primary_video_metadata": primary_meta,
        "detected_modalities": {
            "monocular_rgb": True,
            "stereo_pair": bool(stereo_right),
            "multiview": bool(isinstance(multiview, list) and multiview),
            "native_metric_depth": False,
            "rgbd": False,
        },
        "candidates": candidates,
        "monocular_baseline_policy": "Depth Pro and UniDepth must run first or be explicitly failed before native/stereo/multiview candidates can be selected as primary.",
        "assisted_not_worse_policy": "Native/RGB-D/stereo/multiview candidates that are worse than monocular baseline require calibration/registration/focal/scale diagnosis and tuning before downweighting or rejection.",
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify V21 depth/camera modalities and register candidate backends.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
