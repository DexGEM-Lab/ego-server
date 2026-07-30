#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def project_camera(points_cam: np.ndarray, intr: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intr]
    z = np.maximum(points_cam[:, 2], 1e-9)
    return np.stack([fx * points_cam[:, 0] / z + cx, fy * points_cam[:, 1] / z + cy], axis=1)


def find_frame(frames: list[dict[str, Any]], frame_idx: int) -> dict[str, Any] | None:
    for frame in frames:
        if int(frame.get("frame_idx")) == frame_idx:
            return frame
    return None


def draw_panel(frame: dict[str, Any], hand_side: str, constraint_row: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    raw = Image.open(frame["raw_frame_path"]).convert("RGB")
    raw = raw.resize(size, Image.Resampling.BILINEAR)
    sx = size[0] / raw.width if raw.width else 1.0
    sy = size[1] / raw.height if raw.height else 1.0
    # raw has already been resized; compute scale from original stored separately
    orig = Image.open(frame["raw_frame_path"]).convert("RGB")
    sx = size[0] / orig.width
    sy = size[1] / orig.height
    panel = orig.resize(size, Image.Resampling.BILINEAR).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if obj.get("object_id") == constraint_row.get("object_id") and obj.get("mask_path"):
            try:
                mask = Image.open(obj["mask_path"]).convert("L").resize(size, Image.Resampling.NEAREST)
                arr = np.asarray(mask) > 0
                ys, xs = np.where(arr)
                if xs.size:
                    # sparse mask points are enough for review; draw translucent fill and bbox
                    rgba = Image.new("RGBA", size, (255, 140, 0, 70))
                    overlay.alpha_composite(Image.composite(rgba, Image.new("RGBA", size, (0, 0, 0, 0)), mask))
                    od.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 180, 0, 255), width=2)
            except Exception:
                pass
    hand = None
    for h in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if h.get("hand_side") == hand_side:
            hand = h
            break
    if hand:
        intr = hand.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
        verts = np.asarray(hand.get("vertices_camera_sample_m") or [], dtype=float)
        joints = np.asarray(hand.get("joints_current_v18_camera_m") or [], dtype=float)
        if intr and verts.ndim == 2 and verts.shape[1] == 3 and len(verts):
            pts = project_camera(verts, intr)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            for x, y in pts:
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    od.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(0, 255, 255, 190))
        if intr and joints.ndim == 2 and joints.shape[1] == 3 and len(joints):
            pts = project_camera(joints, intr)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            for x, y in pts:
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    od.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 255, 0, 230), outline=(0, 0, 0, 255))
    panel = Image.alpha_composite(panel, overlay).convert("RGB")
    d = ImageDraw.Draw(panel)
    text = (
        f"f{constraint_row['frame_idx']} {hand_side} | {constraint_row['candidate_application_state']}\n"
        f"AABB verts {constraint_row['aabb_candidate_vertex_count']} | nearest min {constraint_row['nearest_surface_unsigned_m']['min']:.4f} m\n"
        f"H': unchanged; uncertainty/no signed correction"
    )
    d.rectangle([0, 0, size[0], 54], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(255, 255, 255))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--constraint-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=216)
    args = parser.parse_args()
    annotations = load_json(args.annotations)
    constraints = load_json(args.constraint_report)
    rows = [r for r in constraints.get("constraint_rows", []) if r.get("candidate_application_state") != "no_penetration_no_coordinate_change_needed"]
    if not rows:
        rows = constraints.get("constraint_rows", [])[:12]
    panels = []
    for row in rows[:24]:
        frame = find_frame(annotations.get("frames", []), int(row["frame_idx"]))
        if frame is None:
            continue
        panels.append(draw_panel(frame, row["hand_side"], row, (args.panel_width, args.panel_height)))
    if not panels:
        raise RuntimeError("no panels rendered")
    cols = min(4, len(panels))
    rows_n = int(np.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * args.panel_width, rows_n * args.panel_height), (20, 20, 20))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * args.panel_width, (i // cols) * args.panel_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    summary = {
        "method": "render_v18_mano_object_constraint_review",
        "status": "ok",
        "output": str(args.output),
        "panel_count": len(panels),
        "rendered_rows": [{"frame_idx": int(r["frame_idx"]), "hand_side": r["hand_side"], "state": r["candidate_application_state"]} for r in rows[:len(panels)]],
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
