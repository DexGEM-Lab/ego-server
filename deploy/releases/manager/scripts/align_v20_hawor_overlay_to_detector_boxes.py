#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, numeric_summary, write_json


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_side_tracks(raw: list[str]) -> dict[str, list[float]]:
    mapping: dict[str, list[float]] = {}
    for item in raw:
        if ":" not in item:
            raise ContractError(f"side_track_mapping_requires_side_colon_track_key: {item}")
        side, keys = item.split(":", 1)
        side = side.strip().lower()
        if side not in {"left", "right"}:
            raise ContractError(f"side_track_mapping_side_must_be_left_or_right: {item}")
        wanted = []
        for key in keys.split(","):
            key = key.strip()
            if key:
                wanted.append(float(key))
        if not wanted:
            raise ContractError(f"side_track_mapping_missing_track_key: {item}")
        mapping.setdefault(side, []).extend(wanted)
    return mapping


def detection_score(row: dict[str, Any], box: np.ndarray) -> float:
    if box.size >= 5 and np.isfinite(box[4]):
        return float(box[4])
    return 0.0


def track_rows_by_frame(model_tracks_path: Path, side_tracks: dict[str, list[float]]) -> dict[str, dict[int, dict[str, Any]]]:
    tracks = np.load(model_tracks_path, allow_pickle=True).item()
    if not isinstance(tracks, dict):
        raise ContractError(f"hawor_model_tracks_not_dict: {model_tracks_path}")
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for side, wanted_keys in side_tracks.items():
        selected_keys = []
        for wanted_key in wanted_keys:
            selected_key = None
            for key in tracks.keys():
                if abs(float(key) - wanted_key) < 1.0e-4:
                    selected_key = key
                    break
            if selected_key is None:
                raise ContractError(f"hawor_track_key_missing_for_side: {side}:{wanted_key}")
            selected_keys.append(selected_key)
        frame_map: dict[int, dict[str, Any]] = {}
        for selected_key in selected_keys:
            rows = tracks[selected_key]
            for row in rows:
                if not isinstance(row, dict) or not row.get("det", False):
                    continue
                box = np.asarray(row.get("det_box"), dtype=float).reshape(-1)
                if box.size < 4 or not np.isfinite(box[:4]).all():
                    continue
                frame_idx = int(row.get("frame", len(frame_map)))
                score = detection_score(row, box)
                current = frame_map.get(frame_idx)
                if current is not None and score < float(current.get("score", 0.0)):
                    continue
                frame_map[frame_idx] = {
                    "box_xyxy_source": box[:4].astype(float).tolist(),
                    "score": score,
                    "source_track_key": float(selected_key),
                }
        out[side] = frame_map
    return out


def hand_side(hand: dict[str, Any]) -> str:
    return str(hand.get("hand_side") or hand.get("side") or "unknown").lower()


def valid_uv(raw: Any) -> np.ndarray | None:
    arr = np.asarray(raw if raw is not None else [], dtype=float)
    if arr.shape != (21, 2):
        return None
    valid = np.isfinite(arr).all(axis=1)
    if np.count_nonzero(valid) < 5:
        return None
    return arr


def bbox_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid = np.isfinite(uv).all(axis=1)
    if np.count_nonzero(valid) < 5:
        return None
    pts = uv[valid]
    lo = np.percentile(pts, 5.0, axis=0)
    hi = np.percentile(pts, 95.0, axis=0)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any((hi - lo) < 1.0):
        return None
    return lo, hi


def frame_lookup_from_raw_manifest(annotations: dict[str, Any], annotations_path: Path) -> dict[int, dict[str, Any]]:
    raw_path = annotations.get("raw_frame_manifest") if isinstance(annotations, dict) else None
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.is_absolute() and not path.exists():
        candidate = annotations_path.parent / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        return {}
    payload = load_json(path)
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if isinstance(row, dict):
            frame_idx = int(row.get("frame_idx", row.get("index", len(out))))
            out[frame_idx] = row
    return out


def scale_box_to_manifest(box: list[float], frame: dict[str, Any], raw_frame: dict[str, Any] | None = None) -> np.ndarray:
    raw_frame = raw_frame or {}
    source_w = float(raw_frame.get("source_width") or frame.get("source_width") or frame.get("source_image_size", [0, 0])[0] or frame.get("camera", {}).get("source_width") or 0)
    source_h = float(raw_frame.get("source_height") or frame.get("source_height") or frame.get("source_image_size", [0, 0])[1] or frame.get("camera", {}).get("source_height") or 0)
    manifest_w = float(raw_frame.get("manifest_width") or frame.get("manifest_width") or frame.get("source_size", [0, 0])[0] or 0)
    manifest_h = float(raw_frame.get("manifest_height") or frame.get("manifest_height") or frame.get("source_size", [0, 0])[1] or 0)
    if source_w <= 0 or source_h <= 0:
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        intr = camera.get("intrinsics_fx_fy_cx_cy")
        if isinstance(intr, list) and len(intr) == 4:
            manifest_w = manifest_w or float(intr[2]) * 2.0
            manifest_h = manifest_h or float(intr[3]) * 2.0
        source_w = manifest_w
        source_h = manifest_h
    if manifest_w <= 0 or manifest_h <= 0:
        manifest_w = source_w
        manifest_h = source_h
    sx = manifest_w / source_w
    sy = manifest_h / source_h
    arr = np.asarray(box, dtype=float).copy()
    arr[[0, 2]] *= sx
    arr[[1, 3]] *= sy
    return arr


def align_uv_to_box(uv: np.ndarray, target_box: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]] | tuple[None, dict[str, Any]]:
    src_bbox = bbox_from_uv(uv)
    if src_bbox is None:
        return None, {"status": "skipped_invalid_source_uv"}
    src_lo, src_hi = src_bbox
    src_center = 0.5 * (src_lo + src_hi)
    src_size = np.maximum(src_hi - src_lo, 1.0)
    target_lo = target_box[:2]
    target_hi = target_box[2:4]
    target_center = 0.5 * (target_lo + target_hi)
    target_size = np.maximum(target_hi - target_lo, 1.0) * float(args.target_box_fraction)
    scale = float(np.median(target_size / src_size))
    scale = float(np.clip(scale, float(args.min_scale), float(args.max_scale)))
    aligned = (uv - src_center.reshape(1, 2)) * scale + target_center.reshape(1, 2)
    valid = np.isfinite(uv).all(axis=1)
    center_before = float(np.linalg.norm(src_center - target_center))
    aligned_bbox = bbox_from_uv(aligned)
    center_after = None
    if aligned_bbox is not None:
        aligned_center = 0.5 * (aligned_bbox[0] + aligned_bbox[1])
        center_after = float(np.linalg.norm(aligned_center - target_center))
    return aligned, {
        "status": "aligned_to_detector_box_for_overlay_only",
        "scale": scale,
        "source_center_error_px_before": center_before,
        "source_center_error_px_after": center_after,
        "target_box_xyxy": target_box.astype(float).tolist(),
        "valid_joint_count": int(np.count_nonzero(valid)),
    }


def align(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    ensure_no_gt_in_prediction(annotations, "hawor_overlay_alignment_input")
    frames = annotations.get("frames") if isinstance(annotations, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ContractError("annotations_have_no_frames")
    side_tracks = parse_side_tracks(args.side_track)
    tracks = track_rows_by_frame(args.model_tracks, side_tracks)
    raw_frames = frame_lookup_from_raw_manifest(annotations, args.annotations)
    aligned_rows = []
    skipped_rows = []
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
        for hand in hands:
            side = hand_side(hand)
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            if side not in tracks or frame_idx not in tracks[side]:
                skipped_rows.append({"frame_idx": frame_idx, "hand_side": side, "reason": "missing_detector_box"})
                continue
            uv = valid_uv(metric.get("joints_2d_px"))
            if uv is None:
                skipped_rows.append({"frame_idx": frame_idx, "hand_side": side, "reason": "invalid_projected_mano_uv"})
                continue
            target_box = scale_box_to_manifest(tracks[side][frame_idx]["box_xyxy_source"], frame, raw_frames.get(frame_idx))
            aligned, info = align_uv_to_box(uv, target_box, args)
            if aligned is None:
                skipped_rows.append({"frame_idx": frame_idx, "hand_side": side, **info})
                continue
            metric["joints_2d_px_detector_aligned"] = aligned.astype(float).tolist()
            metric["overlay_alignment"] = {
                **info,
                "source": "hawor_detector_box_plus_mano_projected_skeleton_similarity_alignment",
                "claim_scope": "2D overlay alignment only; it does not repair metric MANO camera-space state and must not support contact or nonpenetration claims.",
            }
            metric["support_state"] = "prediction_side_hawor_mano_metric_uncertain_detector_aligned_overlay_only"
            hand["metric_mano_state"] = metric
            hand["hand_geometry_source"] = "prediction_side_hawor_mano_with_detector_aligned_overlay_only"
            aligned_rows.append({"frame_idx": frame_idx, "hand_side": side, **info})
    annotations["hand_overlay_alignment"] = {
        "method": "align_v20_hawor_overlay_to_detector_boxes",
        "claim_scope": "HaWoR MANO overlay is similarity-aligned to HaWoR detector boxes for visualization only because camera-space projection was misaligned in resized VTLA frames.",
        "aligned_rows": len(aligned_rows),
        "skipped_rows": len(skipped_rows),
    }
    report = {
        "schema": "v20_hawor_detector_overlay_alignment_report.v0",
        "status": "ok",
        "method": "align_v20_hawor_overlay_to_detector_boxes",
        "annotation_ready": True,
        "output_annotations": str(args.output_annotations),
        "aligned_rows": len(aligned_rows),
        "skipped_rows": len(skipped_rows),
        "center_error_before_px": numeric_summary([row.get("source_center_error_px_before") for row in aligned_rows]),
        "center_error_after_px": numeric_summary([row.get("source_center_error_px_after") for row in aligned_rows]),
        "scale": numeric_summary([row.get("scale") for row in aligned_rows]),
        "skipped_preview": skipped_rows[:50],
        "claim_scope": "2D detector-box overlay alignment improves visible hand annotation placement only; metric MANO/contact/nonpenetration remain uncertain.",
    }
    ensure_no_gt_in_prediction(annotations, "hawor_overlay_alignment_output")
    ensure_no_gt_in_prediction(report, "hawor_overlay_alignment_report")
    write_json(args.output_annotations, annotations)
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align HaWoR MANO overlay joints to HaWoR detector boxes for visualization only.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model-tracks", type=Path, required=True)
    parser.add_argument("--side-track", action="append", required=True, help="Mapping like left:1.0 and right:2.0")
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--target-box-fraction", type=float, default=0.82)
    parser.add_argument("--min-scale", type=float, default=0.15)
    parser.add_argument("--max-scale", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    align(parse_args())
