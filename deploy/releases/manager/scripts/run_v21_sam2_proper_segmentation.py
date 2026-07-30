#!/usr/bin/env python3
"""V21 SAM2.1 video segmentation from approved OWLv2 bbox prompts.

Runs SAM2.1 video predictor on source-resolution frames for pixel-accurate
object segmentation. Bbox prompts must come from the V21 approved OWLv2 bbox
prompt artifact. Point prompts and SAM2 RGB baseline fallback are not accepted.

Output:
  measurements/object_tracks/sam2_proper/<track_id>/sam2_masks/XXXXXX.png
  measurements/object_tracks/sam2_proper/<track_id>/sam2_track.json
  measurements/object_tracks/sam2_proper/<track_id>/segmentation_report.json
  measurements/object_tracks/sam2_proper_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from tqdm import tqdm


class ContractError(RuntimeError):
    pass


def validate_compute_target(raw: str | None, allow_local_heavy: bool) -> str:
    target = " ".join(str(raw or "").split())
    if not target:
        raise ContractError("missing_compute_target: pass --compute-target or set V21_COMPUTE_TARGET before running SAM2 inference")
    lowered = target.lower()
    if any(token in lowered for token in ["local", "workstation", "laptop"]) and not allow_local_heavy:
        raise ContractError("local_heavy_inference_not_authorized: pass --allow-local-heavy only with explicit user approval")
    return target


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def sanitize_box(box: Any, width: int | None = None, height: int | None = None) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    vals = [float(v) for v in box]
    if not all(np.isfinite(vals)):
        return None
    x1, y1, x2, y2 = vals
    if width is not None:
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    if height is not None:
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_boxes_from_approved_prompts(path: Path, object_id: str) -> tuple[dict[int, list[list[float]]], str, str, list[dict[str, Any]]]:
    payload = load_json(path)
    method = str(payload.get("method", "approved_bbox_prompts"))
    if "groundingdino" in method.lower() or "grounding_dino" in method.lower():
        raise ContractError(f"groundingdino_bbox_prompts_disabled_for_v21: {path}")
    prompts: dict[int, list[list[float]]] = {}
    prompt_rows: list[dict[str, Any]] = []
    rows = payload.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"approved_bbox_prompt_file_has_no_prompts: {path}")
    for row in rows:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        row_object_id = str(row.get("track_id") or row.get("object_id") or row.get("target_object_id") or "")
        if row_object_id and object_id not in {row_object_id, row_object_id.replace("object:", "")}:
            continue
        box = sanitize_box(row.get("bbox_xyxy"))
        if box is None:
            continue
        frame_idx = int(row["frame_idx"])
        prompts.setdefault(frame_idx, []).append(box)
        prompt_rows.append(row)
    if not prompts:
        raise ContractError(f"approved_bbox_prompt_file_has_no_usable_boxes_for_object: object={object_id} path={path}")
    return prompts, method, str(path), prompt_rows


def run_sam2_video(frames_dir: Path, output_dir: Path, box_prompts: dict[int, list[list[float]]], checkpoint_path: str, model_cfg: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run SAM2.1 video predictor with box prompts and write V21 track rows."""
    import torch  # type: ignore[import-not-found]
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=checkpoint_path,
        device="cuda",
        vos_optimized=False,
    )

    frame_paths = sorted(glob.glob(str(frames_dir / "*.jpg"))) or sorted(glob.glob(str(frames_dir / "*.png")))
    if not frame_paths:
        raise ContractError(f"sam2_input_frames_missing: {frames_dir}")
    print(f"  SAM2: {len(frame_paths)} frames", flush=True)

    inference_state = predictor.init_state(video_path=str(frames_dir), async_loading_frames=False)

    for frame_idx, boxes in sorted(box_prompts.items()):
        for box in boxes:
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=int(frame_idx),
                box=np.asarray(box, dtype=np.float32),
                obj_id=0,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    track: dict[str, Any] = {}
    mask_frames: list[dict[str, Any]] = []
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        masks = (mask_logits > 0).cpu().numpy()
        for i, _obj_id in enumerate(obj_ids):
            mask = masks[i, 0].astype(bool)
            mask_path = output_dir / f"{int(frame_idx):06d}.png"
            cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
            area = int(mask.sum())
            if area > 0:
                ys, xs = np.where(mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                center = [float(xs.mean()), float(ys.mean())]
                visible = True
            else:
                bbox = None
                center = None
                visible = False
            row = {
                "frame_idx": int(frame_idx),
                "visible": bool(visible),
                "mask_path": str(mask_path),
                "bbox_xyxy": bbox,
                "center_xy": center,
                "area_px": area,
                "segmentation_source": "sam2_proper_owlv2_bbox_prompt",
            }
            track[str(int(frame_idx))] = row
            mask_frames.append({"frame_idx": int(frame_idx), "mask_pixels": area, "mask_bbox": bbox, "mask_path": str(mask_path)})

    predictor.clear_all_prompts(inference_state)
    del predictor
    torch.cuda.empty_cache()
    return track, mask_frames


def validate_masks_against_depth(mask_dir: Path, depth_npz_path: Path, sample_frames: int = 10) -> dict[str, Any]:
    """Check if mask boundaries align with depth discontinuities."""
    if not depth_npz_path.exists():
        return {"status": "missing_depth_npz", "path": str(depth_npz_path)}
    depth_data = np.load(depth_npz_path)
    depth = depth_data["depth"]
    mask_files = sorted(glob.glob(str(mask_dir / "*.png")))
    if not mask_files:
        return {"status": "no_masks"}
    sample_indices = np.linspace(0, len(mask_files) - 1, min(sample_frames, len(mask_files)), dtype=int)
    edge_alignments = []
    for si in sample_indices:
        mf = mask_files[si]
        fidx = int(Path(mf).stem)
        if fidx >= depth.shape[0]:
            continue
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE) > 127
        if mask.sum() < 10:
            continue
        d_frame = depth[fidx].astype(float)
        if d_frame.shape != mask.shape:
            d_frame = cv2.resize(d_frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) - cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        boundary_depths = d_frame[boundary > 0]
        valid = boundary_depths[boundary_depths > 0.1]
        if len(valid) > 5:
            edge_alignments.append({"frame_idx": fidx, "boundary_depth_std": float(np.std(valid)), "boundary_depth_range": float(valid.max() - valid.min())})
    return {
        "status": "ok",
        "sample_count": len(edge_alignments),
        "median_boundary_depth_std": float(np.median([e["boundary_depth_std"] for e in edge_alignments])) if edge_alignments else None,
        "samples": edge_alignments[:5],
    }


def resolve_rgb_path(run_root: Path, repo_root: Path, frame_row: dict[str, Any]) -> Path | None:
    fidx = int(frame_row["frame_idx"])
    candidates = [run_root / f"input/source_frame_manifest/rgb/{fidx:06d}.jpg"]
    if frame_row.get("rgb"):
        raw = Path(str(frame_row["rgb"]))
        candidates.extend([raw if raw.is_absolute() else repo_root / raw, run_root / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_overlay(run_root: Path, repo_root: Path, manifest: dict[str, Any], track: dict[str, Any], output_path: Path, fps: float) -> dict[str, Any]:
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    mask_frames_drawn = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for frame_row in manifest.get("frames", []):
            if not isinstance(frame_row, dict) or frame_row.get("frame_idx") is None:
                continue
            frame_idx = int(frame_row["frame_idx"])
            rgb_path = resolve_rgb_path(run_root, repo_root, frame_row)
            if rgb_path is None:
                continue
            image = cv2.imread(str(rgb_path))
            if image is None:
                continue
            track_row = track.get(str(frame_idx))
            if isinstance(track_row, dict) and track_row.get("mask_path"):
                mask = cv2.imread(str(track_row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    valid = mask > 127
                    if valid.any():
                        tint = np.zeros_like(image)
                        tint[:, :, 1] = 255
                        image[valid] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[valid]
                        contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
                        mask_frames_drawn += 1
            cv2.putText(image, f"frame {frame_idx}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, f"frame {frame_idx}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(image.shape[1]), int(image.shape[0])))
                if not writer.isOpened():
                    raise ContractError(f"could_not_open_overlay_writer: {output_path}")
            writer.write(image)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    if frames_written == 0:
        raise ContractError("sam2_proper_overlay_no_frames_written")
    return {"overlay": str(output_path), "frames_written": int(frames_written), "mask_frames_drawn": int(mask_frames_drawn)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    repo_root = Path.cwd()
    obj = args.object_id
    manifest = load_json(run_root / "input/raw_frame_manifest/manifest.json")
    compute_target = validate_compute_target(args.compute_target, bool(args.allow_local_heavy))
    total_frames = len(manifest["frames"])

    sam2_input_dir = run_root / "logs/sam2_input_frames"
    sam2_input_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {total_frames} source-resolution frames for SAM2...", flush=True)
    for fm in tqdm(manifest["frames"], desc="Preparing frames"):
        fidx = int(fm["frame_idx"])
        src_path = run_root / f"input/source_frame_manifest/rgb/{fidx:06d}.jpg"
        if not src_path.exists() and fm.get("rgb"):
            raw = Path(str(fm["rgb"]))
            src_path = raw if raw.is_absolute() else repo_root / raw
        if src_path.exists():
            img = cv2.imread(str(src_path))
            if img is not None:
                cv2.imwrite(str(sam2_input_dir / f"{fidx:06d}.jpg"), img)

    approved_prompt_path = Path(args.approved_bbox_prompts)
    box_prompts, bbox_method, bbox_source, prompt_rows = load_boxes_from_approved_prompts(approved_prompt_path, obj)
    print(f"Box prompts on {len(box_prompts)} seed frames from {bbox_method}: {list(sorted(box_prompts.keys()))}", flush=True)

    track_root = run_root / f"measurements/object_tracks/sam2_proper/{obj}"
    mask_output_dir = track_root / "sam2_masks"
    started = time.time()
    track, mask_frames = run_sam2_video(sam2_input_dir, mask_output_dir, box_prompts, args.sam2_checkpoint, args.sam2_model_cfg)
    elapsed = time.time() - started
    print(f"SAM2 done in {elapsed:.1f}s: {len(mask_frames)} masks", flush=True)

    track_path = track_root / "sam2_track.json"
    write_json(track_path, track)

    depth_path = Path(args.depth_npz) if args.depth_npz else run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"
    validation = validate_masks_against_depth(mask_output_dir, depth_path)
    fps = float(manifest.get("fps", 25.0) or 25.0)
    overlay_qc = render_overlay(run_root, repo_root, manifest, track, track_root / "sam2_proper_overlay.mp4", fps)

    pixels = [m["mask_pixels"] for m in mask_frames]
    visible_frames = int(sum(1 for row in track.values() if isinstance(row, dict) and row.get("visible")))
    report = {
        "schema": "v21_sam2_proper_segmentation.v1",
        "status": "ok",
        "method": "sam2.1_video_predictor_with_approved_owlv2_bbox_prompts",
        "groundingdino_used": False,
        "compute_target": compute_target,
        "object_id": obj,
        "track_id": obj,
        "bbox_prompt_source": bbox_source,
        "bbox_prompt_method": bbox_method,
        "bbox_prompt_seed_frames": list(sorted(box_prompts.keys())),
        "approved_bbox_prompt_count": int(sum(len(v) for v in box_prompts.values())),
        "approved_bbox_prompts": prompt_rows,
        "elapsed_s": elapsed,
        "frame_count": len(mask_frames),
        "visible_frames": visible_frames,
        "mask_stats": {
            "median_pixels": float(np.median(pixels)) if pixels else 0,
            "min_pixels": min(pixels) if pixels else 0,
            "max_pixels": max(pixels) if pixels else 0,
        },
        "depth_validation": validation,
        "outputs": {
            "sam2_track": str(track_path),
            "sam2_mask_dir": str(mask_output_dir),
            "overlay": overlay_qc["overlay"],
        },
        "overlay_qc": overlay_qc,
        "claim_scope": "SAM2 mask evidence from approved OWLv2 bbox prompts. This is segmentation evidence only; it requires contamination review before geometry use.",
    }
    report_path = track_root / "segmentation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)

    qc_path = track_root / "qc_sam2_proper.json"
    write_json(
        qc_path,
        {
            "schema": "v21_sam2_proper_qc.v0",
            "status": "ok",
            "track_id": obj,
            "frames": int(len(mask_frames)),
            "visible_frames": visible_frames,
            "mask_dir": str(mask_output_dir),
            "sam2_track": str(track_path),
            "overlay": overlay_qc["overlay"],
            "depth_validation": validation,
        },
    )

    summary_path = Path(args.output_summary) if args.output_summary else run_root / "measurements/object_tracks/sam2_proper_summary.json"
    summary = {
        "schema": "v21_sam2_proper_summary.v0",
        "status": "ok",
        "method": "run_v21_sam2_proper_segmentation",
        "case_id": manifest.get("case_id"),
        "compute_target": compute_target,
        "run_root": str(run_root),
        "raw_frame_manifest": str(run_root / "input/raw_frame_manifest/manifest.json"),
        "frame_count": int(len(mask_frames)),
        "tracks": [
            {
                "track_id": obj,
                "target_object_id": obj,
                "approved_bbox_prompts": str(approved_prompt_path),
                "sam2_output_dir": str(track_root),
                "sam2_mask_dir": str(mask_output_dir),
                "sam2_track": str(track_path),
                "sam2_qc": str(qc_path),
                "segmentation_report": str(report_path),
                "overlay": overlay_qc["overlay"],
                "visible_frames": visible_frames,
                "frame_count": int(len(mask_frames)),
                "prompt_frames": list(sorted(box_prompts.keys())),
            }
        ],
        "claim_scope": "Active V21 SAM2 mask track from approved OWLv2 bbox prompts only.",
    }
    write_json(summary_path, summary)
    report["summary"] = str(summary_path)
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--approved-bbox-prompts", required=True, help="Approved V21 OWLv2 bbox prompt JSON from approve_v21_owlv2_bbox_prompts.py.")
    ap.add_argument("--output-summary", help="Path for measurements/object_tracks/sam2_proper_summary.json.")
    ap.add_argument("--sam2-checkpoint", default="/mnt/user-home/zjh/ego-pipeline/ego-object/work/models/sam2/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--depth-npz")
    ap.add_argument("--compute-target", default=os.environ.get("V21_COMPUTE_TARGET"), help="Required explicit compute target label, e.g. A800/server job id.")
    ap.add_argument("--allow-local-heavy", action="store_true", help="Only use with explicit user approval for local heavy inference.")
    args = ap.parse_args()
    run(args)
