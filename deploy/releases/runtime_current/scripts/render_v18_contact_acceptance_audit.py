#!/usr/bin/env python3
"""Render V18 contact acceptance audit.

This tests the intersection of contact graph acceptance and local
nonpenetration evidence. It does not accept contact when the mesh is open or the
signed-distance evidence is local/incomplete.
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


def nonpenetration_index(path: Path, key_name: str) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    report = load_json(path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")), str(row.get("object_id")))] = row
    return out


def contact_assignments(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        if not isinstance(hand_graph, dict):
            continue
        for row in hand_graph.get("assignments", []):
            if not isinstance(row, dict):
                continue
            oid = row.get("chosen_owner_object_id")
            if isinstance(oid, str) and oid.startswith("object:"):
                rows.append(row)
    return rows


def classify_contact(row: dict[str, Any], signed: dict[str, Any] | None, triangle: dict[str, Any] | None) -> tuple[str, bool]:
    accepted = bool(row.get("accepted_contact_owner"))
    signed_pen = bool(signed and signed.get("local_penetration_detected"))
    tri_pen = bool(triangle and triangle.get("local_triangle_penetration_detected"))
    triangle_available = triangle is not None
    signed_available = signed is not None
    watertight = bool(triangle and triangle.get("mesh_watertight_by_edges"))
    signed_complete = bool(signed and signed.get("signed_nonpenetration_complete"))
    triangle_complete = bool(triangle and triangle.get("triangle_nonpenetration_complete"))
    strict = accepted and signed_available and triangle_available and not signed_pen and not tri_pen and watertight and signed_complete and triangle_complete
    if strict:
        return "strict_promotable_contact", True
    if not accepted:
        return "graph_selected_not_contact_accepted", False
    if signed_pen or tri_pen:
        return "source_graph_candidate_local_penetration_veto", False
    if not signed_available or not triangle_available:
        return "source_graph_candidate_missing_local_nonpenetration_evidence", False
    if not watertight or not signed_complete or not triangle_complete:
        return "source_graph_candidate_local_no_penetration_open_mesh_not_strict", False
    return "source_graph_candidate_unclassified_not_strict", False


def audit_rows(contact_report: dict[str, Any], signed_rows: dict[tuple[int, str, str], dict[str, Any]], triangle_rows: dict[tuple[int, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in contact_assignments(contact_report):
        frame_idx = int(row.get("frame_idx", -1))
        side = str(row.get("hand_side"))
        oid = str(row.get("chosen_owner_object_id"))
        key = (frame_idx, side, oid)
        signed = signed_rows.get(key)
        triangle = triangle_rows.get(key)
        category, strict = classify_contact(row, signed, triangle)
        source_claim = str(row.get("contact_owner_claim") or "")
        if source_claim.startswith("accepted_contact_owner"):
            source_claim = source_claim.replace("accepted_contact_owner", "source_graph_contact_candidate", 1)
        rows.append({
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": oid,
            "category": category,
            "strict_promotable_contact": strict,
            "selected_by_contact_graph": True,
            "source_graph_contact_candidate_before_physical_veto": bool(row.get("accepted_contact_owner")),
            "contact_owner_claim": source_claim,
            "contact_owner_claim_context": "source_contact_graph_candidate_before_local_nonpenetration_and_completeness_veto_not_final_contact_acceptance",
            "unary_energy_margin": row.get("unary_energy_margin"),
            "min_hand_surface_to_object_mesh_m": row.get("min_hand_surface_to_object_mesh_m"),
            "signed_available": signed is not None,
            "signed_complete": bool(signed and signed.get("signed_nonpenetration_complete")),
            "signed_local_penetration_detected": bool(signed and signed.get("local_penetration_detected")),
            "signed_min_local_distance_m": signed.get("min_local_signed_distance_m") if signed else None,
            "signed_semantics": signed.get("local_signed_distance_semantics") if signed else None,
            "triangle_available": triangle is not None,
            "triangle_complete": bool(triangle and triangle.get("triangle_nonpenetration_complete")),
            "triangle_mesh_watertight_by_edges": bool(triangle and triangle.get("mesh_watertight_by_edges")),
            "triangle_local_penetration_detected": bool(triangle and triangle.get("local_triangle_penetration_detected")),
            "triangle_min_local_distance_m": triangle.get("min_local_triangle_signed_distance_m") if triangle else None,
            "triangle_boundary_edge_count": triangle.get("boundary_edge_count") if triangle else None,
            "triangle_semantics": triangle.get("local_triangle_signed_distance_semantics") if triangle else None,
            "state_role": "contact_acceptance_audit_not_contact_assignment_not_complete_nonpenetration",
        })
    return rows


def color_for_category(category: str) -> tuple[int, int, int]:
    if category == "strict_promotable_contact":
        return (80, 255, 120)
    if category == "source_graph_candidate_local_no_penetration_open_mesh_not_strict":
        return (80, 210, 255)
    if category == "source_graph_candidate_local_penetration_veto":
        return (255, 70, 70)
    if category == "graph_selected_not_contact_accepted":
        return (255, 190, 70)
    if category == "source_graph_candidate_missing_local_nonpenetration_evidence":
        return (190, 120, 255)
    return (180, 180, 180)


def row_priority(category: str) -> int:
    order = {
        "strict_promotable_contact": 0,
        "source_graph_candidate_local_no_penetration_open_mesh_not_strict": 1,
        "source_graph_candidate_local_penetration_veto": 2,
        "graph_selected_not_contact_accepted": 3,
        "source_graph_candidate_missing_local_nonpenetration_evidence": 4,
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
    contact_path = args.contact_graph_root / case / "v18_contact_ownership_graph_report.json"
    signed_path = args.signed_nonpenetration_root / case / "v18_signed_nonpenetration_evidence_report.json"
    triangle_path = args.triangle_nonpenetration_root / case / "v18_triangle_nonpenetration_evidence_report.json"
    contact_report = load_json(contact_path)
    signed_rows = nonpenetration_index(signed_path, "signed")
    triangle_rows = nonpenetration_index(triangle_path, "triangle")
    rows = audit_rows(contact_report, signed_rows, triangle_rows)
    frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame_rows[int(row["frame_idx"])].append(row)
    case_dir = args.output_root / case / "contact_acceptance_audit"
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
        draw.text((12, 8), f"V18 contact acceptance audit {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.text((12, 34), "green=strict; cyan=local no-penetration but open/incomplete mesh; red=local penetration veto; yellow=graph selected not accepted", fill=(230, 230, 230), font=small)
        hands = hand_by_side(frame)
        objects = object_by_id(frame)
        visible_rows = sorted(frame_rows.get(frame_idx, []), key=lambda r: row_priority(str(r.get("category"))))[: args.max_rows_per_frame]
        if not visible_rows:
            counts["frames_without_selected_contact_rows"] += 1
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
                f"{side}->{oid.replace('object:', '')} {category} "
                f"signed_pen={row.get('signed_local_penetration_detected')} tri_pen={row.get('triangle_local_penetration_detected')} "
                f"watertight={row.get('triangle_mesh_watertight_by_edges')}"
            )
            draw_label(draw, (12, y), label[:155], small, color, (0, 0, 0))
            y += 20
            counts[f"draw::{category}"] += 1
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_contact_acceptance_audit.mp4"
    encode_video(frame_dir, video_path, fps)
    report = {
        "method": "render_v18_contact_acceptance_audit",
        "case": case,
        "claim_scope": "full_timeline_audit_of_contact_graph_acceptance_and_local_nonpenetration_not_contact_assignment_not_complete_sdf",
        "source_contact_graph_report": str(contact_path),
        "source_signed_nonpenetration_report": str(signed_path),
        "source_triangle_nonpenetration_report": str(triangle_path),
        "frame_count": len(frames),
        "fps": fps,
        "selected_contact_rows": len(rows),
        "strict_promotable_contact_rows": sum(1 for row in rows if row.get("strict_promotable_contact") is True),
        "source_graph_contact_candidate_rows_before_physical_veto": sum(1 for row in rows if row.get("source_graph_contact_candidate_before_physical_veto") is True),
        "category_counts": dict(sorted(category_counts.items())),
        "rows": rows,
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_contact_acceptance_audit_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_contact_acceptance_audit",
        "status": "contact_acceptance_audit_not_contact_assignment_not_complete_sdf",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_contact_acceptance_audit_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--contact-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--signed-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--triangle-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--max-rows-per-frame", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
