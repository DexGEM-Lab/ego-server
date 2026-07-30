#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Probe full-frame open-vocabulary boxes for late trash lid/top/rim geometry.

This is a discriminator for the trash MANO blocker after the old object-mask-gated
SAM2 lid track was shown to recover only partial side/rim/background masks. It
uses schema-derived phrases and OWLv2 on full frames, not the stale object mask
crop, to ask whether model-produced boxes can localize the actual manipulated
lid/top/rim in late frames.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DEFAULT_SCHEMA = Path("/data2/ego_annotation_outputs/v18_physical_state_schema/trash_1050/v18_physical_state_schema_report.json")
DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "trash_1050/annotations_v18_full.json"
)
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_trash_late_lid_open_vocab_box_probe_v1")
DEFAULT_MODEL = Path("/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d")
TARGET_FRAMES = [901, 956, 958, 1003]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def frame_by_idx(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for frame in as_list(annotations.get("frames")):
        if isinstance(frame, dict) and frame.get("frame_idx") is not None:
            out[int(frame["frame_idx"])] = frame
    return out


def schema_phrases(schema: dict[str, Any], object_id: str) -> tuple[list[str], dict[str, Any]]:
    row = None
    for candidate in as_list(schema.get("object_rows")):
        if isinstance(candidate, dict) and candidate.get("object_id") == object_id:
            row = candidate
            break
    if row is None:
        raise RuntimeError(f"object_id not found in schema: {object_id}")
    # Phrases are derived from the VLM/schema language for this object plus generic part nouns
    # already present in that language. They are not category-specific control branches.
    text_sources = [str(row.get("name") or ""), str(row.get("physical_notes") or "")]
    text_sources.extend(str(x) for x in as_list(row.get("primary_articulation_evidence_terms")))
    text_sources.extend(str(x) for x in as_list(row.get("part_or_relative_motion_evidence_terms")))
    text = " ".join(text_sources).lower().replace("/", " ")
    candidates = []
    if "trash" in text and "lid" in text:
        candidates.extend(["trash can lid", "lid of a trash can"])
    if "pink" in text and "lid" in text:
        candidates.extend(["pink lid", "pink trash can lid"])
    if "rim" in text:
        candidates.extend(["trash can rim", "lid rim", "pink rim"])
    if "top" in text or "lid" in text:
        candidates.extend(["lid top", "top of a trash can", "round lid"])
    # Keep an explicit base part prompt because it was the prior accepted source.
    candidates.append("lid")
    deduped: list[str] = []
    for phrase in candidates:
        phrase = " ".join(phrase.split())
        if phrase and phrase not in deduped:
            deduped.append(phrase)
    return [f"a photo of a {phrase}" for phrase in deduped], {"schema_row": row, "source_text": text_sources, "phrases": deduped}


def run_frame(processor: Any, model: Any, image: Image.Image, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=float(threshold),
        target_sizes=[(image.height, image.width)],
        text_labels=[prompts],
    )[0]
    rows = []
    for box_tensor, score_tensor, text_label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = [float(v) for v in box_tensor.detach().cpu().tolist()]
        x1 = max(0.0, min(float(image.width), x1))
        x2 = max(0.0, min(float(image.width), x2))
        y1 = max(0.0, min(float(image.height), y1))
        y2 = max(0.0, min(float(image.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        rows.append(
            {
                "text_label": str(text_label),
                "owlv2_score": float(score_tensor.detach().cpu().item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "box_area_fraction": float(area / max(1.0, image.width * image.height)),
                "center_xy": [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
            }
        )
    rows.sort(key=lambda r: float(r["owlv2_score"]), reverse=True)
    return rows


def draw_sheet(frames: list[int], detections_by_frame: dict[int, list[dict[str, Any]]], annotations_by_frame: dict[int, dict[str, Any]], output_path: Path, max_boxes: int) -> None:
    colors = [(0, 255, 255), (255, 0, 255), (255, 230, 0), (0, 255, 80), (255, 120, 0), (120, 180, 255)]
    tiles = []
    font = ImageFont.load_default()
    for frame_idx in frames:
        frame = annotations_by_frame[frame_idx]
        image = Image.open(str(frame["raw_frame_path"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        rows = detections_by_frame.get(frame_idx, [])[:max_boxes]
        for i, row in enumerate(rows):
            color = colors[i % len(colors)]
            box = [int(round(float(v))) for v in row["bbox_xyxy"]]
            draw.rectangle(box, outline=color, width=3)
            label = f"{i+1}:{row['text_label'].replace('a photo of a ', '')} {row['owlv2_score']:.2f}"
            draw.rectangle((box[0], max(0, box[1] - 13), min(image.width, box[0] + 6 * len(label)), box[1]), fill=(0, 0, 0))
            draw.text((box[0] + 2, max(0, box[1] - 12)), label, fill=color, font=font)
        title = f"frame {frame_idx}: full-frame OWLv2 schema-derived lid/rim/top boxes"
        draw.rectangle((4, 4, min(image.width - 4, 4 + 6 * len(title)), 18), fill=(0, 0, 0))
        draw.text((6, 6), title, fill=(255, 255, 255), font=font)
        tiles.append(image)
    if not tiles:
        return
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    cols = min(2, len(tiles))
    rows_n = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * w, rows_n * h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--owlv2-model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--object-id", default="object:pink_lid_trash_can_second")
    ap.add_argument("--frames", type=int, nargs="+", default=TARGET_FRAMES)
    ap.add_argument("--threshold", type=float, default=0.025)
    ap.add_argument("--max-boxes-per-frame", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    schema = load_json(args.schema)
    annotations = load_json(args.annotations)
    frames = frame_by_idx(annotations)
    prompts, prompt_source = schema_phrases(schema, args.object_id)
    processor = Owlv2Processor.from_pretrained(str(args.owlv2_model), local_files_only=True)
    model: Any = Owlv2ForObjectDetection.from_pretrained(str(args.owlv2_model), local_files_only=True).to(args.device)
    model.eval()
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in args.frames:
        if frame_idx not in frames:
            raise RuntimeError(f"frame missing from annotations: {frame_idx}")
        image = Image.open(str(frames[frame_idx]["raw_frame_path"])).convert("RGB")
        detections_by_frame[int(frame_idx)] = run_frame(processor, model, image, prompts, float(args.threshold))
    sheet_path = args.output_root / "trash_1050" / "late_lid_open_vocab_box_probe_sheet.jpg"
    draw_sheet(list(map(int, args.frames)), detections_by_frame, frames, sheet_path, int(args.max_boxes_per_frame))
    report = {
        "method": "probe_v18_trash_late_lid_open_vocab_boxes",
        "status": "ok",
        "claim_scope": "Full-frame OWLv2 box probe only. It proposes model boxes for late lid/top/rim segmentation; it does not create masks, geometry, MANO correction, or object pose.",
        "inputs": {"schema": str(args.schema), "annotations": str(args.annotations), "owlv2_model": str(args.owlv2_model), "object_id": str(args.object_id)},
        "parameters": {"frames": list(map(int, args.frames)), "threshold": float(args.threshold), "max_boxes_per_frame": int(args.max_boxes_per_frame)},
        "prompt_source": prompt_source,
        "text_prompts": prompts,
        "detection_counts_by_frame": {str(k): len(v) for k, v in detections_by_frame.items()},
        "detections_by_frame": {str(k): v for k, v in detections_by_frame.items()},
        "review_sheet": str(sheet_path),
    }
    report_path = args.output_root / "trash_1050" / "v18_trash_late_lid_open_vocab_box_probe_report.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "review_sheet": str(sheet_path), "detection_counts_by_frame": report["detection_counts_by_frame"]}, indent=2))


if __name__ == "__main__":
    main()
