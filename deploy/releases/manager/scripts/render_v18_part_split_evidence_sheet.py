#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


STATUS = "v18_part_split_evidence_sheet"
CLAIM = (
    "This sheet visualizes accepted part-mask overlap evidence for V18 part-split candidates. It is a QC artifact, "
    "not part geometry or part pose."
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


def finite_float(value: Any, label: str) -> float:
    if value is None:
        raise RuntimeError(f"{label} missing")
    return float(value)


def normalize_path(path: str) -> Path:
    candidates = [
        Path(path),
        Path(path.replace("/mnt/user-home/yiwen/ego_annotation_remote/data", "/data2/ego_annotation_outputs")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def text_font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def overlay_mask(base: Image.Image, mask_path: Path, color: tuple[int, int, int], alpha_value: int) -> Image.Image:
    if not mask_path.exists():
        return base
    mask = Image.open(mask_path).convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    alpha = mask.point([alpha_value if p > 0 else 0 for p in range(256)])
    layer = Image.new("RGB", base.size, color)
    return Image.composite(layer, base, alpha)


def raw_frame_index(annotation: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        out[require_int(frame.get("frame_idx"), "frame_idx")] = str(frame.get("raw_frame_path"))
    return out


def accepted_samples(report: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw_obj in require_list(report.get("object_rows"), "object rows"):
        obj = require_dict(raw_obj, "object row")
        object_id = str(obj.get("object_id"))
        for raw_eval in require_list(obj.get("candidate_part_track_evaluations", []), "track evaluations"):
            ev = require_dict(raw_eval, "track evaluation")
            if ev.get("accepted_as_part_evidence") is not True:
                continue
            samples = [require_dict(raw_sample, "sample") for raw_sample in require_list(ev.get("samples"), "samples")]
            if not samples:
                continue
            sample = max(samples, key=lambda item: float(item.get("part_containment_in_object", 0.0)))
            out.append(
                {
                    "object_id": object_id,
                    "track_label": ev.get("track_label"),
                    "track_path": ev.get("track_path"),
                    "frame_idx": sample.get("frame_idx"),
                    "part_mask_path": sample.get("part_mask_path"),
                    "object_mask_path": sample.get("object_mask_path"),
                    "part_containment_in_object": sample.get("part_containment_in_object"),
                    "object_coverage_by_part": sample.get("object_coverage_by_part"),
                    "iou": sample.get("iou"),
                }
            )
    return out[:max_samples]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text(xy, text, font=font, fill=(255, 255, 255))


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.part_split_root / case / "v18_part_split_evidence_report.json"
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    report = require_dict(load_json(report_path), f"{case} report")
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    frames = raw_frame_index(annotation)
    samples = accepted_samples(report, args.max_tiles)
    output_dir = args.part_split_root / case
    output_path = output_dir / "v18_part_split_evidence_sheet.jpg"
    if not samples:
        sheet = Image.new("RGB", (args.tile_width, args.tile_height), (20, 20, 24))
        draw = ImageDraw.Draw(sheet)
        draw_label(draw, (16, 16), f"{case}: no accepted part-mask overlap evidence", text_font(22))
        sheet.save(output_path, quality=90)
        tile_count = 0
    else:
        font = text_font(16)
        tiles: list[Image.Image] = []
        for sample in samples:
            frame_idx = require_int(sample.get("frame_idx"), "sample frame_idx")
            raw_path = normalize_path(frames[frame_idx])
            image = Image.open(raw_path).convert("RGB")
            image = overlay_mask(image, normalize_path(str(sample.get("object_mask_path"))), (70, 180, 255), 80)
            image = overlay_mask(image, normalize_path(str(sample.get("part_mask_path"))), (255, 80, 80), 130)
            image.thumbnail((args.tile_width, args.tile_height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (args.tile_width, args.tile_height), (0, 0, 0))
            tile.paste(image, ((args.tile_width - image.width) // 2, (args.tile_height - image.height) // 2))
            draw = ImageDraw.Draw(tile)
            containment = finite_float(sample.get("part_containment_in_object"), "part_containment_in_object")
            coverage = finite_float(sample.get("object_coverage_by_part"), "object_coverage_by_part")
            label = f"f{frame_idx} {sample.get('track_label')} | contain={containment:.2f} objcov={coverage:.2f}"
            draw_label(draw, (8, 8), label[:110], font)
            draw_label(draw, (8, args.tile_height - 28), "blue=whole object, red=accepted part mask", font)
            tiles.append(tile)
        cols = min(args.columns, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * args.tile_width, rows * args.tile_height), (0, 0, 0))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * args.tile_width, (i // cols) * args.tile_height))
        sheet.save(output_path, quality=90)
        tile_count = len(tiles)
    qc = {
        "method": "render_v18_part_split_evidence_sheet",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"part_split_evidence_report": str(report_path), "annotation_state": str(annotation_path)},
        "output_sheet": str(output_path),
        "accepted_sample_tile_count": tile_count,
        "part_geometry_extraction_ready": False,
        "part_pose_ready": False,
    }
    write_json(output_dir / "v18_part_split_evidence_sheet_qc.json", qc)
    return qc


def render(args: argparse.Namespace) -> dict[str, Any]:
    qcs = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_part_split_evidence_sheet",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(qcs),
        "accepted_sample_tile_count": sum(require_int(qc.get("accepted_sample_tile_count"), "tile count") for qc in qcs),
        "outputs": qcs,
        "part_geometry_extraction_ready": False,
        "part_pose_ready": False,
    }
    write_json(args.part_split_root / "v18_part_split_evidence_sheet_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-split-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--max-tiles", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2))


if __name__ == "__main__":
    main()
