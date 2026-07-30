#!/usr/bin/env python3
"""Render V18 occlusion-owner acceptance audit.

This artifact tests whether any occlusion-owner row is promotable under a
conservative intersection of temporal graph selection, foreground depth support,
mesh support, and source acceptance gates. It does not relax strict ownership or
claim solved occlusion ownership.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bb = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=bg)
    draw.text((x, y), text, fill=fill, font=fnt)


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[int, int, int, int]:
    return int(round(box[0] * sx)), int(round(box[1] * sy)), int(round(box[2] * sx)), int(round(box[3] * sy))


def scale_point(point: tuple[float, float], sx: float, sy: float) -> tuple[int, int]:
    return int(round(point[0] * sx)), int(round(point[1] * sy))


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}", "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path),
    ], check=True)


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def hand_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(hand.get("hand_side") or hand.get("side")): hand for hand in frame.get("hands", []) if isinstance(hand, dict)}


def object_by_id(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id")): obj for obj in frame.get("objects", []) if isinstance(obj, dict)}


def classify_row(row: dict[str, Any]) -> tuple[str, bool]:
    gate = row.get("acceptance_gate", {}) if isinstance(row.get("acceptance_gate"), dict) else {}
    selected = bool(row.get("selected_by_occlusion_graph"))
    exact_depth = bool(gate.get("exact_foreground_depth_support"))
    contradiction = int(gate.get("same_frame_foreground_contradiction_count") or 0) > 0 or "foreground_depth_contradicts_candidate" in (row.get("acceptance_blockers") or [])
    mesh_support = finite_float(gate.get("mesh_temporal_support"), 0.0)
    mesh_ok = mesh_support >= finite_float(gate.get("mesh_support_threshold"), 0.5)
    margin = gate.get("temporal_graph_margin")
    margin_ok = isinstance(margin, (int, float)) and math.isfinite(float(margin)) and float(margin) >= finite_float(gate.get("temporal_graph_margin_threshold"), 0.25)
    source_depth_ok = bool(gate.get("source_depth_order_resolved"))
    source_owner_ok = bool(gate.get("source_occluder_owner_accepted"))
    strict_promotable = selected and exact_depth and not contradiction and mesh_ok and margin_ok and source_depth_ok and source_owner_ok
    if strict_promotable:
        return "strict_promotable_owner", True
    if contradiction:
        return "foreground_depth_contradicts_candidate", False
    if exact_depth and mesh_ok and not selected:
        return "direct_depth_mesh_support_not_temporal_selected", False
    if selected and mesh_ok and margin_ok and not exact_depth:
        return "temporal_selected_mesh_margin_supported_depth_missing", False
    if selected and mesh_ok and not margin_ok:
        return "temporal_selected_mesh_supported_margin_low", False
    if selected and not mesh_ok:
        return "temporal_selected_mesh_support_low_or_missing", False
    if exact_depth and not mesh_ok:
        return "direct_depth_support_mesh_low_or_missing", False
    return "not_selected_no_direct_depth_support", False


def audit_rows(graph_report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in graph_report.get("rows", []) if isinstance(graph_report.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        gate = row.get("acceptance_gate", {}) if isinstance(row.get("acceptance_gate"), dict) else {}
        depth_evidence = row.get("depth_pair_evidence", {}) if isinstance(row.get("depth_pair_evidence"), dict) else {}
        category, promotable = classify_row(row)
        out.append({
            "frame_idx": int(row.get("frame_idx", -1)),
            "hand_side": str(row.get("hand_side")),
            "object_id": str(row.get("object_id")),
            "object_name": row.get("object_name"),
            "category": category,
            "strict_promotable_owner": promotable,
            "accepted_occlusion_owner": bool(row.get("accepted_occlusion_owner")),
            "selected_by_temporal_graph": bool(row.get("selected_by_occlusion_graph")),
            "exact_foreground_depth_support": bool(gate.get("exact_foreground_depth_support")),
            "same_frame_foreground_contradiction_count": int(gate.get("same_frame_foreground_contradiction_count") or 0),
            "mesh_temporal_support": gate.get("mesh_temporal_support"),
            "mesh_support_threshold": gate.get("mesh_support_threshold"),
            "temporal_graph_margin": gate.get("temporal_graph_margin"),
            "temporal_graph_margin_threshold": gate.get("temporal_graph_margin_threshold"),
            "source_depth_order_resolved": bool(gate.get("source_depth_order_resolved")),
            "source_occluder_owner_accepted": bool(gate.get("source_occluder_owner_accepted")),
            "depth_pair_evidence_state": row.get("depth_pair_evidence_state"),
            "hand_box_coverage_by_object_box": row.get("hand_box_coverage_by_object_box"),
            "bbox_iou": row.get("bbox_iou"),
            "object_depth_median_m": depth_evidence.get("object_depth_median_m"),
            "hand_metric_depth_state": depth_evidence.get("hand_metric_depth_state"),
            "acceptance_blockers": row.get("acceptance_blockers") if isinstance(row.get("acceptance_blockers"), list) else [],
            "evidence_scope": "acceptance_audit_only_not_owner_assignment_or_pose_fill",
        })
    return out


def color_for_category(category: str) -> tuple[int, int, int]:
    if category == "strict_promotable_owner":
        return (80, 255, 120)
    if category == "direct_depth_mesh_support_not_temporal_selected":
        return (60, 220, 255)
    if category == "temporal_selected_mesh_margin_supported_depth_missing":
        return (255, 120, 220)
    if category == "temporal_selected_mesh_supported_margin_low":
        return (255, 190, 70)
    if category == "temporal_selected_mesh_support_low_or_missing":
        return (180, 120, 255)
    if category == "foreground_depth_contradicts_candidate":
        return (255, 70, 70)
    return (170, 170, 170)


def row_priority(category: str) -> int:
    order = {
        "strict_promotable_owner": 0,
        "direct_depth_mesh_support_not_temporal_selected": 1,
        "temporal_selected_mesh_margin_supported_depth_missing": 2,
        "foreground_depth_contradicts_candidate": 3,
        "temporal_selected_mesh_supported_margin_low": 4,
        "temporal_selected_mesh_support_low_or_missing": 5,
    }
    return order.get(category, 9)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    raw_video = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    source_w = finite_float(raw_video.get("width"), 1920.0)
    source_h = finite_float(raw_video.get("height"), 1080.0)
    fps = finite_float(ann.get("fps"), 30.0)
    graph_path = args.owner_graph_root / case / "v18_occlusion_owner_graph_report.json"
    graph_report = load_json(graph_path)
    rows = audit_rows(graph_report)
    frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame_rows[int(row["frame_idx"])].append(row)
    case_dir = args.output_root / case / "occlusion_owner_acceptance_audit"
    frame_dir = case_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    counts: Counter[str] = Counter()
    category_counts = Counter(str(row["category"]) for row in rows)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw_path).convert("RGB") if raw_path.exists() else Image.new("RGB", (int(source_w), int(source_h)), (12, 12, 12))
        sx = image.size[0] / source_w if source_w > 0 else 1.0
        sy = image.size[1] / source_h if source_h > 0 else 1.0
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.size[0], 58), fill=(0, 0, 0))
        draw.text((12, 8), f"V18 occlusion owner acceptance audit {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.text((12, 34), "green=strict promotable; cyan=direct depth+mesh not selected; magenta=graph selected+mesh but depth missing; red=depth contradiction", fill=(230, 230, 230), font=small)
        hands = hand_by_side(frame)
        objects = object_by_id(frame)
        visible_rows = sorted(frame_rows.get(frame_idx, []), key=lambda r: row_priority(str(r.get("category"))))[: args.max_rows_per_frame]
        if not visible_rows:
            counts["frames_without_candidate_rows"] += 1
        y = 64
        for row in visible_rows:
            category = str(row.get("category"))
            color = color_for_category(category)
            side = str(row.get("hand_side"))
            oid = str(row.get("object_id"))
            hand = hands.get(side, {})
            obj = objects.get(oid, {})
            hbox = bbox_tuple(hand.get("bbox_xyxy"))
            obox = bbox_tuple(obj.get("bbox_xyxy"))
            if hbox is not None:
                draw.rectangle(scale_box(hbox, sx, sy), outline=color, width=4)
                hp = scale_point(center(hbox), sx, sy)
            else:
                hp = (80, y + 30)
            if obox is not None:
                draw.rectangle(scale_box(obox, sx, sy), outline=(80, 210, 255), width=3)
                op = scale_point(center(obox), sx, sy)
            else:
                op = (min(image.size[0] - 40, hp[0] + 180), hp[1])
            draw.line((hp[0], hp[1], op[0], op[1]), fill=color, width=4)
            draw.ellipse((hp[0] - 5, hp[1] - 5, hp[0] + 5, hp[1] + 5), fill=color)
            draw.ellipse((op[0] - 5, op[1] - 5, op[0] + 5, op[1] + 5), fill=(80, 210, 255))
            label = (
                f"{side}->{str(row.get('object_name') or oid).replace('object:', '')} {category} "
                f"mesh={finite_float(row.get('mesh_temporal_support'), 0.0):.2f} "
                f"margin={row.get('temporal_graph_margin')} blockers={','.join(str(b) for b in (row.get('acceptance_blockers') or [])[:2])}"
            )
            draw_label(draw, (12, y), label[:155], small, color, (0, 0, 0))
            y += 20
            counts[f"draw::{category}"] += 1
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_occlusion_owner_acceptance_audit.mp4"
    encode_video(frame_dir, video_path, fps)
    report = {
        "method": "render_v18_occlusion_owner_acceptance_audit",
        "case": case,
        "claim_scope": "full_timeline_audit_of_occlusion_owner_acceptance_intersection_not_owner_assignment_not_pose_fill",
        "source_owner_graph_report": str(graph_path),
        "frame_count": len(frames),
        "fps": fps,
        "candidate_rows": len(rows),
        "strict_promotable_owner_rows": sum(1 for row in rows if row.get("strict_promotable_owner") is True),
        "accepted_occlusion_owner_rows_in_source_graph": sum(1 for row in rows if row.get("accepted_occlusion_owner") is True),
        "category_counts": dict(sorted(category_counts.items())),
        "acceptance_blocker_counts": graph_report.get("acceptance_blocker_counts"),
        "rows": rows,
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_occlusion_owner_acceptance_audit_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_occlusion_owner_acceptance_audit",
        "status": "occlusion_owner_acceptance_audit_not_owner_assignment",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_occlusion_owner_acceptance_audit_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--max-rows-per-frame", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
