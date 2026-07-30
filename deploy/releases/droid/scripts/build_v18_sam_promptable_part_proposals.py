#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_sam_promptable_part_proposal_probe"
CLAIM = (
    "This artifact probes whether local promptable SAM assets can produce generic within-object segmentation proposals "
    "for part/relative-motion objects. The proposals are not referring/open-vocabulary part tracks and are not accepted "
    "as V18 part-mask evidence, geometry, pose, or contact readiness."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def text_font(size: int) -> Any:
    for raw in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        path = Path(raw)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int]) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def import_sam_model(checkpoint: Path, model_type: str, device: str):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-not-found]

    if model_type not in sam_model_registry:
        raise RuntimeError(f"SAM model_type {model_type} is not in sam_model_registry")
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def object_ids_requiring_parts(blocker_report: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for raw in require_list(blocker_report.get("object_rows"), "blocker object_rows"):
        row = require_dict(raw, "blocker row")
        out.add(require_str(row.get("object_id"), "blocker object_id"))
    return out


def annotation_object_rows(annotation: dict[str, Any], object_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        raw_frame_path = require_str(frame.get("raw_frame_path"), "raw_frame_path")
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            obj = require_dict(raw_obj, "object")
            object_id = str(obj.get("object_id"))
            if object_id not in object_ids:
                continue
            if obj.get("visibility_state") != "visible" or obj.get("renderable_mask") is not True:
                continue
            mask_path = obj.get("mask_path")
            if not isinstance(mask_path, str) or not Path(mask_path).exists():
                continue
            row = {
                "frame_idx": frame_idx,
                "raw_frame_path": raw_frame_path,
                "object_id": object_id,
                "track_id": obj.get("track_id"),
                "name": obj.get("name"),
                "bbox_xyxy": obj.get("bbox_xyxy"),
                "geometry_scope": obj.get("geometry_scope"),
                "mask_path": mask_path,
            }
            rows[object_id].append(row)
    return rows


def quantile_select(rows: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    if len(rows) <= max_count:
        return rows
    # Prefer rows that already have depth-backed visible surface, then spread selected frames over time.
    surface_rows = [row for row in rows if row.get("geometry_scope") == "visible_surface_depth_backed"]
    source = surface_rows if len(surface_rows) >= max_count else rows
    source = sorted(source, key=lambda row: require_int(row.get("frame_idx"), "frame_idx"))
    if max_count == 1:
        return [source[len(source) // 2]]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for i in range(max_count):
        pos = round(i * (len(source) - 1) / float(max_count - 1))
        idx = int(pos)
        while idx in used and idx + 1 < len(source):
            idx += 1
        used.add(idx)
        selected.append(source[idx])
    return selected


def load_mask(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def prompt_points_from_mask(mask: np.ndarray, max_points: int, min_distance_px: float) -> list[tuple[int, int]]:
    if mask.sum() == 0:
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[tuple[int, int]] = []
    suppressed = np.zeros(mask.shape, dtype=bool)
    for _ in range(max_points):
        score = np.where(mask & ~suppressed, dist, -1.0)
        flat = int(np.argmax(score))
        value = float(score.flat[flat])
        if value <= 0.0:
            break
        y, x = np.unravel_index(flat, mask.shape)
        points.append((int(x), int(y)))
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        suppressed |= (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= min_distance_px**2
    return points


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (mask.astype(np.uint8) * 255)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write mask {path}")


def proposal_state(containment: float, whole_fraction: float, min_containment: float, min_fraction: float, max_fraction: float) -> str:
    if containment < min_containment:
        return "rejected_not_contained_in_whole_object_mask"
    if whole_fraction < min_fraction:
        return "rejected_too_small_for_part_proposal"
    if whole_fraction > max_fraction:
        return "rejected_near_whole_object_mask_not_part"
    return "promptable_sam_proposal_not_referring_part_track"


def render_sheet(case: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    font = text_font(16)
    small = text_font(12)
    for row in rows[:24]:
        raw_path = Path(require_str(row.get("raw_frame_path"), "raw_frame_path"))
        image = Image.open(raw_path).convert("RGB")
        image.thumbnail((360, 220))
        canvas = Image.new("RGB", (380, 280), (20, 20, 24))
        canvas.paste(image, (10, 10))
        draw = ImageDraw.Draw(canvas)
        draw_label(draw, (10, 236), f"f{row.get('frame_idx')} {row.get('name')}", small, (255, 255, 255))
        draw_label(draw, (10, 256), f"saved proposal masks={row.get('saved_promptable_proposal_mask_count')}", small, (180, 220, 255))
        thumbs.append(canvas)
    if not thumbs:
        thumbs.append(Image.new("RGB", (380, 280), (20, 20, 24)))
    cols = 2
    rows_n = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 380, rows_n * 280 + 44), (12, 12, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{case} SAM promptable part proposals (proposal-only, not accepted tracks)", font=font, fill=(255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 380
        y = 44 + (idx // cols) * 280
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def process_frame(predictor: Any, row: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    raw_path = Path(require_str(row.get("raw_frame_path"), "raw_frame_path"))
    image = np.asarray(Image.open(raw_path).convert("RGB"))
    mask = load_mask(Path(require_str(row.get("mask_path"), "mask_path")), image.shape[:2])
    object_area = int(mask.sum())
    points = prompt_points_from_mask(mask, int(args.points_per_frame), float(args.prompt_min_distance_px))
    predictor.set_image(image)
    proposal_rows: list[dict[str, Any]] = []
    raw_candidate_count = 0
    saved_count = 0
    for point_index, (x, y) in enumerate(points):
        masks, scores, _logits = predictor.predict(
            point_coords=np.asarray([[x, y]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
        for mask_index, candidate_mask in enumerate(masks):
            raw_candidate_count += 1
            proposal = np.asarray(candidate_mask, dtype=bool)
            area = int(proposal.sum())
            if area == 0 or object_area == 0:
                containment = 0.0
                whole_fraction = 0.0
            else:
                overlap = int(np.logical_and(proposal, mask).sum())
                containment = overlap / float(area)
                whole_fraction = overlap / float(object_area)
            state = proposal_state(
                containment,
                whole_fraction,
                float(args.min_containment),
                float(args.min_whole_fraction),
                float(args.max_whole_fraction),
            )
            proposal_path = None
            if state == "promptable_sam_proposal_not_referring_part_track" and saved_count < int(args.max_saved_proposals_per_frame):
                saved_count += 1
                proposal_path = output_dir / require_str(row.get("object_id"), "object_id").replace(":", "_") / f"{require_int(row.get('frame_idx'), 'frame_idx'):06d}_proposal_{saved_count:02d}.png"
                save_mask(proposal_path, proposal)
            proposal_rows.append(
                {
                    "point_index": point_index,
                    "point_xy": [x, y],
                    "sam_mask_index": mask_index,
                    "sam_score": float(scores[mask_index]) if len(scores) > mask_index else None,
                    "proposal_area_px": area,
                    "whole_object_area_px": object_area,
                    "containment_in_whole_object_mask": containment,
                    "whole_object_overlap_fraction": whole_fraction,
                    "bbox_xyxy": bbox_from_mask(proposal),
                    "proposal_state": state,
                    "proposal_mask_path": str(proposal_path) if proposal_path is not None else None,
                    "accepted_as_part_track": False,
                }
            )
    counts = Counter(str(item["proposal_state"]) for item in proposal_rows)
    return {
        **row,
        "prompt_points": [[x, y] for x, y in points],
        "prompt_point_count": len(points),
        "raw_sam_mask_candidate_count": raw_candidate_count,
        "saved_promptable_proposal_mask_count": saved_count,
        "proposal_state_counts": dict(sorted(counts.items())),
        "proposal_rows": proposal_rows,
        "accepted_part_track_count": 0,
        "semantic_part_label_ready": False,
        "mask_evidence_created": False,
    }


def case_report(case: str, predictor: Any, args: argparse.Namespace) -> dict[str, Any]:
    blocker_path = args.part_object_blockers_root / case / "v18_part_object_blocker_manifest_report.json"
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    blocker_report = require_dict(load_json(blocker_path), f"{case} blocker report")
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    required_ids = object_ids_requiring_parts(blocker_report)
    object_rows = annotation_object_rows(annotation, required_ids)
    frame_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    output_dir = args.output_root / case / "proposal_masks"
    for object_id in sorted(required_ids):
        available_rows = sorted(object_rows.get(object_id, []), key=lambda row: require_int(row.get("frame_idx"), "frame_idx"))
        selected = quantile_select(available_rows, int(args.max_frames_per_object))
        processed: list[dict[str, Any]] = []
        for row in selected:
            processed_row = process_frame(predictor, row, args, output_dir)
            processed.append(processed_row)
            frame_records.append(processed_row)
        object_state = "promptable_sam_proposals_available_not_referring_part_tracks" if sum(int(row["saved_promptable_proposal_mask_count"]) for row in processed) > 0 else "no_promptable_sam_proposals_selected"
        object_records.append(
            {
                "object_id": object_id,
                "available_whole_object_mask_frame_count": len(available_rows),
                "selected_frame_count": len(selected),
                "saved_promptable_proposal_mask_count": sum(int(row["saved_promptable_proposal_mask_count"]) for row in processed),
                "raw_sam_mask_candidate_count": sum(int(row["raw_sam_mask_candidate_count"]) for row in processed),
                "object_probe_state": object_state,
                "accepted_part_track_count": 0,
                "mask_evidence_created": False,
            }
        )
    sheet_path = args.output_root / case / "v18_sam_promptable_part_proposals_sheet.jpg"
    render_sheet(case, frame_records, sheet_path)
    proposal_counts = Counter()
    for row in frame_records:
        proposal_counts.update(require_dict(row.get("proposal_state_counts"), "proposal counts"))
    report = {
        "method": "build_v18_sam_promptable_part_proposals",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"part_object_blockers": str(blocker_path), "annotation_state": str(annotation_path)},
        "sam_backend": {"package": "segment_anything", "model_type": args.sam_model_type, "checkpoint": str(args.sam_checkpoint), "device": args.device},
        "object_count": len(object_records),
        "selected_frame_count": len(frame_records),
        "prompt_point_count": sum(int(row["prompt_point_count"]) for row in frame_records),
        "raw_sam_mask_candidate_count": sum(int(row["raw_sam_mask_candidate_count"]) for row in frame_records),
        "saved_promptable_proposal_mask_count": sum(int(row["saved_promptable_proposal_mask_count"]) for row in frame_records),
        "proposal_state_counts": dict(sorted(proposal_counts.items())),
        "object_records": object_records,
        "frame_records": frame_records,
        "proposal_sheet": str(sheet_path),
        "accepted_part_track_count": 0,
        "semantic_part_label_ready_count": 0,
        "mask_evidence_created_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_sam_promptable_part_proposals_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM promptable proposal probe requested CUDA but torch.cuda.is_available() is false")
    if not args.sam_checkpoint.exists():
        raise RuntimeError(f"SAM checkpoint missing: {args.sam_checkpoint}")
    predictor = import_sam_model(args.sam_checkpoint, args.sam_model_type, args.device)
    reports = [case_report(case, predictor, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    proposal_counts: Counter[str] = Counter()
    for report in reports:
        proposal_counts.update(require_dict(report.get("proposal_state_counts"), "proposal state counts"))
    summary = {
        "method": "build_v18_sam_promptable_part_proposals",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count": sum(require_int(report.get("object_count"), "object count") for report in reports),
        "selected_frame_count": sum(require_int(report.get("selected_frame_count"), "selected frame count") for report in reports),
        "prompt_point_count": sum(require_int(report.get("prompt_point_count"), "prompt point count") for report in reports),
        "raw_sam_mask_candidate_count": sum(require_int(report.get("raw_sam_mask_candidate_count"), "raw candidate count") for report in reports),
        "saved_promptable_proposal_mask_count": sum(require_int(report.get("saved_promptable_proposal_mask_count"), "saved proposal mask count") for report in reports),
        "proposal_state_counts": dict(sorted(proposal_counts.items())),
        "accepted_part_track_count": 0,
        "semantic_part_label_ready_count": 0,
        "mask_evidence_created_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_sam_promptable_part_proposals_report.json"),
                "selected_frame_count": report["selected_frame_count"],
                "saved_promptable_proposal_mask_count": report["saved_promptable_proposal_mask_count"],
                "accepted_part_track_count": 0,
                "mask_evidence_created_count": 0,
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_sam_promptable_part_proposals_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals"))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("/home/yiwen/ego_annotation/checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames-per-object", type=int, default=3)
    parser.add_argument("--points-per-frame", type=int, default=5)
    parser.add_argument("--prompt-min-distance-px", type=float, default=48.0)
    parser.add_argument("--min-containment", type=float, default=0.75)
    parser.add_argument("--min-whole-fraction", type=float, default=0.02)
    parser.add_argument("--max-whole-fraction", type=float, default=0.92)
    parser.add_argument("--max-saved-proposals-per-frame", type=int, default=8)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
