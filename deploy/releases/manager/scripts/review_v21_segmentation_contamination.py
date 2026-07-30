#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def resolve_current_output_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    candidates = [path]
    text = str(raw)
    if text.startswith("outputs/"):
        candidates.append(Path("output") / Path(text).relative_to("outputs"))
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        candidates.append(Path("output") / Path(text[len(historical_prefix) :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def numeric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def mask_info(mask_path: Path) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ContractError(f"could_not_read_mask: {mask_path}")
    valid = mask > 0
    area = int(np.count_nonzero(valid))
    if area == 0:
        return {"area_px": 0, "bbox_xyxy": None, "center_xy": None}
    ys, xs = np.nonzero(valid)
    return {
        "area_px": area,
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        "center_xy": [float(xs.mean()), float(ys.mean())],
    }


def render_sheet(primary_video: Path, track: dict[str, Any], frames: list[int], output: Path, render_width: int) -> None:
    cap = cv2.VideoCapture(str(primary_video))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {primary_video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_height = int(round(render_width * src_h / src_w))
    panels = []
    try:
        for frame_idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            row = track.get(str(frame_idx), {})
            mask_path = row.get("mask_path") if isinstance(row, dict) else None
            resolved_mask_path = resolve_current_output_path(mask_path)
            if resolved_mask_path:
                mask = cv2.imread(str(resolved_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (render_width, render_height), interpolation=cv2.INTER_NEAREST) > 0
                    tint = np.zeros_like(frame)
                    tint[:, :, 1] = 255
                    frame[mask] = cv2.addWeighted(frame, 0.55, tint, 0.45, 0.0)[mask]
            cv2.putText(frame, f"frame {frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(frame)
    finally:
        cap.release()
    if not panels:
        raise ContractError("no_panels_for_review_sheet")
    rows = []
    for i in range(0, len(panels), 3):
        chunk = panels[i : i + 3]
        while len(chunk) < 3:
            chunk.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise ContractError(f"could_not_write_sheet: {output}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    sam2_summary = load_json(args.sam2_summary)
    primary_video = resolve_current_output_path(input_manifest["primary_video"])
    if primary_video is None:
        raise ContractError("input_manifest_missing_primary_video")
    tracks_out: list[dict[str, Any]] = []
    for track_row in sam2_summary.get("tracks", []):
        if not isinstance(track_row, dict):
            continue
        track_id = str(track_row["track_id"])
        track_path = resolve_current_output_path(track_row["sam2_track"])
        if track_path is None:
            raise ContractError(f"missing_sam2_track_path_for_track: {track_id}")
        track = load_json(track_path)
        visible_frames = []
        areas = []
        centers = []
        bboxes = []
        missing_masks = 0
        for key, value in sorted(track.items(), key=lambda item: int(item[0])):
            frame_idx = int(key)
            if not isinstance(value, dict) or not value.get("visible"):
                continue
            mask_path = value.get("mask_path")
            if not mask_path:
                missing_masks += 1
                continue
            resolved_mask_path = resolve_current_output_path(mask_path)
            if resolved_mask_path is None:
                missing_masks += 1
                continue
            info = mask_info(resolved_mask_path)
            if info["area_px"] <= 0:
                continue
            visible_frames.append(frame_idx)
            areas.append(float(info["area_px"]))
            centers.append(info["center_xy"])
            bboxes.append(info["bbox_xyxy"])
        center_steps = []
        for a, b in zip(centers, centers[1:]):
            center_steps.append(float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
        area_jumps = []
        for a, b in zip(areas, areas[1:]):
            area_jumps.append(float(abs(b - a) / max(1.0, a)))
        review_frames = []
        if visible_frames:
            review_frames = sorted(set([visible_frames[0], visible_frames[len(visible_frames) // 2], visible_frames[-1]]))
            if len(visible_frames) >= 6:
                for frac in [0.2, 0.4, 0.6, 0.8]:
                    review_frames.append(visible_frames[int(frac * (len(visible_frames) - 1))])
                review_frames = sorted(set(review_frames))
        sheet_path = args.output_dir / track_id / "segmentation_contamination_review.jpg"
        if review_frames:
            render_sheet(primary_video, track, review_frames, sheet_path, int(args.render_width))
        visible_fraction = len(visible_frames) / max(1, int(sam2_summary.get("frame_count", len(track))))
        decision = "accept_for_v21_visible_mask_evidence"
        flags = []
        if visible_fraction < float(args.min_visible_fraction):
            decision = "reject_or_reprompt_low_visible_fraction"
            flags.append("low_visible_fraction")
        if areas and max(areas) > float(args.max_area_fraction) * int(input_manifest["primary_video_metadata"]["width"]) * int(input_manifest["primary_video_metadata"]["height"]):
            decision = "reject_or_reprompt_area_too_large"
            flags.append("area_too_large_possible_background_contamination")
        if area_jumps and np.percentile(area_jumps, 95) > float(args.max_area_jump_p95):
            flags.append("large_area_jumps_review_temporal_drift")
        if center_steps and np.percentile(center_steps, 95) > float(args.max_center_step_p95_px):
            flags.append("large_center_jumps_review_temporal_drift")
        tracks_out.append(
            {
                "track_id": track_id,
                "target_object_id": track_row.get("target_object_id"),
                "track_path": str(track_path),
                "review_sheet": str(sheet_path) if review_frames else None,
                "visible_frames": int(len(visible_frames)),
                "visible_fraction": float(visible_fraction),
                "missing_masks": int(missing_masks),
                "area_px": numeric_summary(areas),
                "center_step_px": numeric_summary(center_steps),
                "relative_area_jump": numeric_summary(area_jumps),
                "decision": decision,
                "flags": flags,
                "claim_scope": "Programmatic contamination review plus sheet for human inspection. Acceptance means mask evidence may feed V21 visible geometry, not that object pose/mesh is solved.",
            }
        )
    accepted = [row for row in tracks_out if str(row["decision"]).startswith("accept")]
    report = {
        "schema": "v21_segmentation_contamination_review.v0",
        "status": "ok" if accepted else "no_accepted_tracks",
        "method": "review_v21_segmentation_contamination",
        "case_id": input_manifest.get("case_id"),
        "input_manifest": str(args.input_manifest),
        "sam2_summary": str(args.sam2_summary),
        "tracks": tracks_out,
        "accepted_track_count": int(len(accepted)),
        "claim_scope": "V21 segmentation acceptance gate. Accepted tracks are visible mask evidence only and still require depth/camera validation before geometry/pose use.",
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review V21 SAM2 masks for obvious contamination/drift before geometry use.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--sam2-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--min-visible-fraction", type=float, default=0.20)
    parser.add_argument("--max-area-fraction", type=float, default=0.35)
    parser.add_argument("--max-area-jump-p95", type=float, default=2.5)
    parser.add_argument("--max-center-step-p95-px", type=float, default=220.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
