#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, load_json, load_npz_depth, read_depth_image_m, write_json


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ContractError(f"could_not_open_video: {path}")
    meta = {
        "path": str(path),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if meta["width"] <= 0 or meta["height"] <= 0 or meta["frame_count"] <= 0:
        raise ContractError(f"invalid_video_metadata: {meta}")
    return meta


def npz_candidate(candidate_id: str, path: Path, method_family: str, source_modality: str, role: str, weight: float) -> dict[str, Any]:
    data = load_npz_depth(path)
    depth = data["depth"]
    valid = np.isfinite(depth) & (depth > 0)
    return {
        "candidate_id": candidate_id,
        "method_family": method_family,
        "source_modality": source_modality,
        "depth_npz": str(path),
        "frame_indices": data["frame_idx"].astype(int).tolist(),
        "frame_count": int(depth.shape[0]),
        "depth_shape_hw": [int(depth.shape[1]), int(depth.shape[2])],
        "valid_fraction": float(valid.mean()),
        "intrinsics_available": data["intrinsics_fx_fy_cx_cy"] is not None,
        "source_size_available": data["source_size"] is not None,
        "candidate_role_hint": role,
        "prior_weight": float(weight),
        "accepted_depth": False,
        "evaluation_reference_allowed_in_prediction": False,
    }


def image_sequence_candidate(candidate_id: str, manifest: Path, method_family: str, source_modality: str, role: str, weight: float, depth_scale_to_m: float | None, semantics: str | None) -> dict[str, Any]:
    payload = load_json(manifest)
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"depth_image_manifest_requires_frames: {manifest}")
    depth_paths: list[str] = []
    frame_indices: list[int] = []
    valid_fractions: list[float] = []
    shapes = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("depth_path") or row.get("depth") or row.get("path")
        if raw is None:
            raise ContractError(f"depth_manifest_row_missing_depth_path: {manifest}")
        path = Path(raw)
        if not path.is_absolute():
            path = manifest.parent / path
        depth = read_depth_image_m(path, semantics, depth_scale_to_m)
        depth_paths.append(str(path))
        frame_indices.append(int(row.get("frame_idx", row.get("index", len(frame_indices)))))
        valid_fractions.append(float((np.isfinite(depth) & (depth > 0)).mean()))
        shapes.add(tuple(depth.shape))
    if len(shapes) != 1:
        raise ContractError(f"depth_image_manifest_shape_mismatch: {manifest}")
    shape = next(iter(shapes))
    return {
        "candidate_id": candidate_id,
        "method_family": method_family,
        "source_modality": source_modality,
        "depth_paths": depth_paths,
        "frame_indices": frame_indices,
        "frame_count": len(depth_paths),
        "depth_shape_hw": [int(shape[0]), int(shape[1])],
        "valid_fraction": float(np.mean(valid_fractions)),
        "depth_scale_to_m": depth_scale_to_m,
        "depth_semantics": semantics,
        "candidate_role_hint": role,
        "prior_weight": float(weight),
        "accepted_depth": False,
        "evaluation_reference_allowed_in_prediction": False,
    }


def parse_candidate(raw: str) -> dict[str, str]:
    parts = raw.split("|")
    if len(parts) < 4:
        raise ContractError("--candidate format: id|kind|path|method_family[|source_modality|role|weight|scale|semantics]")
    keys = ["candidate_id", "kind", "path", "method_family", "source_modality", "role", "weight", "scale", "semantics"]
    return {key: parts[i] for i, key in enumerate(keys) if i < len(parts)}


def detect_modality(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    source_notes = []
    input_video_meta = None
    if args.input_video is not None:
        input_video_meta = video_metadata(args.input_video)
        source_notes.append(f"input_video={args.input_video}")
    source_modalities = {str(row.get("source_modality")) for row in candidates}
    report = {
        "schema": "v20_depth_modality_report.v0",
        "input_video": input_video_meta,
        "has_native_depth": "native_depth" in source_modalities,
        "has_rgbd_stream": "rgbd" in source_modalities,
        "has_stereo_pair": bool(args.stereo_right_video) or "stereo" in source_modalities,
        "has_calibration": bool(args.calibration) or any(row.get("intrinsics_available") for row in candidates),
        "has_camera_trajectory": bool(args.camera_npz),
        "rgb_only": not candidates,
        "registered_candidate_count": len(candidates),
        "source_notes": source_notes,
    }
    if args.stereo_right_video is not None:
        report["stereo_right_video"] = video_metadata(args.stereo_right_video)
    if args.calibration is not None:
        report["calibration"] = str(args.calibration)
    if args.camera_npz is not None:
        report["camera_npz"] = str(args.camera_npz)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw in args.candidate or []:
        spec = parse_candidate(raw)
        kind = spec["kind"].strip().lower()
        candidate_id = spec["candidate_id"]
        if candidate_id.lower().startswith("gt") or "ground_truth" in candidate_id.lower() or "oracle" in candidate_id.lower():
            raise ContractError(f"reference_or_oracle_depth_candidate_forbidden: {candidate_id}")
        path = Path(spec["path"])
        source_modality = spec.get("source_modality", "monocular_or_external")
        role = spec.get("role", "retained_uncertain")
        weight = float(spec.get("weight", 1.0))
        scale = float(spec["scale"]) if spec.get("scale") not in (None, "", "none") else None
        semantics = spec.get("semantics")
        if kind == "npz":
            candidates.append(npz_candidate(candidate_id, path, spec["method_family"], source_modality, role, weight))
        elif kind in {"images", "image_manifest"}:
            candidates.append(image_sequence_candidate(candidate_id, path, spec["method_family"], source_modality, role, weight, scale, semantics))
        else:
            raise ContractError(f"unsupported_depth_candidate_kind: {kind}")
    if not candidates and args.require_candidate:
        raise ContractError("v20_depth_candidate_registry_failed: no_real_depth_candidates_provided")
    modality = detect_modality(args, candidates)
    registry = {
        "schema": "v20_depth_candidate_registry.v0",
        "claim_scope": "Registry contains prediction-side depth observations only; selection is separate and residual-driven.",
        "evaluation_reference_policy": "Eval-ref-derived depth is forbidden in prediction candidates; eval refs may be read only by evaluation.",
        "candidates": candidates,
        "candidate_count": len(candidates),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "depth_modality_report.json", modality)
    write_json(args.output_dir / "depth_candidate_registry.json", registry)
    return {"modality": modality, "registry": registry}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V20 prediction-side depth modality report and candidate registry.")
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--stereo-right-video", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--camera-npz", type=Path, default=None)
    parser.add_argument("--candidate", action="append", help="id|kind|path|method_family[|source_modality|role|weight|scale|semantics]")
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(result["registry"]["candidate_count"])
