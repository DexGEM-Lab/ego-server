#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
SIDES = ("left", "right")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    if points_camera.ndim != 2 or points_camera.shape[1] != 3 or intrinsics.shape != (4,):
        return None
    if np.any(points_camera[:, 2] <= 1e-6):
        return None
    fx, fy, cx, cy = intrinsics.astype(float)
    out = np.empty((len(points_camera), 2), dtype=np.float64)
    out[:, 0] = fx * points_camera[:, 0] / points_camera[:, 2] + cx
    out[:, 1] = fy * points_camera[:, 1] / points_camera[:, 2] + cy
    return out if np.isfinite(out).all() else None


def current_hands_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        side = str(hand.get("hand_side") or hand.get("side") or "").lower()
        if side in SIDES:
            out[side] = hand
    return out


def reference_projection(hand: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else None
    if mano is None:
        return None, None
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics")
    if not (isinstance(joints, list) and len(joints) == 21 and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return None, str(mano.get("source")) if mano.get("source") is not None else None
    points = np.asarray(joints, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)[None, :]
    return project(points, np.asarray(intr, dtype=np.float64)), str(mano.get("source")) if mano.get("source") is not None else None


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bb = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=bg)
    draw.text((x, y), text, font=fnt, fill=fill)


def scale_points_to_image(points: np.ndarray, image_size: tuple[int, int], source_size: tuple[float, float] = (1920.0, 1080.0)) -> np.ndarray:
    sx = image_size[0] / source_size[0]
    sy = image_size[1] / source_size[1]
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def draw_skeleton(draw: ImageDraw.ImageDraw, pts: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    if pts.shape != (21, 2):
        return
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAND_EDGES:
        draw.line((ipts[a][0], ipts[a][1], ipts[b][0], ipts[b][1]), fill=color, width=width)
    for x, y in ipts:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, color: tuple[int, int, int], image_size: tuple[int, int]) -> None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return
    vals = [finite_float(v) for v in bbox]
    if not all(math.isfinite(v) for v in vals):
        return
    sx = image_size[0] / 1920.0
    sy = image_size[1] / 1080.0
    x0, y0, x1, y1 = vals
    left, right = sorted((x0 * sx, x1 * sx))
    top, bottom = sorted((y0 * sy, y1 * sy))
    if right > left and bottom > top:
        draw.rectangle((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))), outline=color, width=3)


def collect_rows(ann: dict[str, Any], z: np.lib.npyio.NpzFile, img_focal: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intr = np.asarray([img_focal, img_focal, 960.0, 540.0], dtype=np.float64)
    by_frame = {int(frame.get("frame_idx", -1)): frame for frame in ann.get("frames", []) if isinstance(frame, dict)}
    for row_idx, (frame_idx, side_raw, joints_camera) in enumerate(zip(np.asarray(z["frame_idx"], dtype=int), np.asarray(z["side"]), np.asarray(z["joints_hawor_camera_m"], dtype=np.float64))):
        side = "left" if int(side_raw) == 0 else "right"
        frame = by_frame.get(int(frame_idx))
        if frame is None:
            continue
        hawor_proj = project(joints_camera, intr)
        hands = current_hands_by_side(frame)
        ref, ref_source = reference_projection(hands.get(side, {})) if side in hands else (None, None)
        if hawor_proj is None or ref is None:
            continue
        residual = np.linalg.norm(hawor_proj - ref, axis=1)
        rows.append({
            "row_idx": int(row_idx),
            "frame_idx": int(frame_idx),
            "side": side,
            "residual_median_px": float(np.median(residual)),
            "residual_p95_px": float(np.percentile(residual, 95.0)),
            "reference_source_backend": ref_source,
        })
    return rows


def render_tile(ann: dict[str, Any], z: np.lib.npyio.NpzFile, row: dict[str, Any], tile_w: int, tile_h: int, title: str) -> Image.Image:
    frame = ann["frames"][int(row["frame_idx"])]
    raw = Path(str(frame.get("raw_frame_path")))
    image = Image.open(raw).convert("RGB") if raw.exists() else Image.new("RGB", (1920, 1080), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    side = str(row["side"])
    row_idx = int(row["row_idx"])
    img_focal = 2304.0
    hawor_proj = project(np.asarray(z["joints_hawor_camera_m"], dtype=np.float64)[row_idx], np.asarray([img_focal, img_focal, 960.0, 540.0], dtype=np.float64))
    hands = current_hands_by_side(frame)
    ref, ref_source = reference_projection(hands.get(side, {})) if side in hands else (None, None)
    if side in hands:
        draw_bbox(draw, hands[side].get("bbox_xyxy"), (255, 190, 40), image.size)
    if ref is not None:
        draw_skeleton(draw, scale_points_to_image(ref, image.size), (255, 170, 30), 3)
    if hawor_proj is not None:
        draw_skeleton(draw, scale_points_to_image(hawor_proj, image.size), (60, 230, 255), 4)
    color = (255, 80, 80) if float(row["residual_median_px"]) > 200.0 else (80, 255, 120)
    draw.rectangle((0, 0, image.size[0], 84), fill=(0, 0, 0))
    draw.text((12, 10), f"{title}: frame {row['frame_idx']} {side} median {row['residual_median_px']:.1f}px p95 {row['residual_p95_px']:.1f}px", fill=color, font=font(30))
    draw.text((12, 48), f"cyan=HaWoR bridge projection; orange=current visible hand candidate projection ({ref_source}); orange box=current hand bbox", fill=(230, 230, 230), font=font(20))
    return image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)


def run(args: argparse.Namespace) -> dict[str, Any]:
    ann = load_json(args.source_root / args.case / "annotations_v18_full.json")
    bridge_report_path = args.output_root / "hawor_bridge_state" / args.case / "v18_hawor_bridge_state_report.json"
    bridge_report = load_json(bridge_report_path)
    npz_path = Path(str(bridge_report.get("bridge_candidate_npz")))
    z = np.load(npz_path)
    rows = collect_rows(ann, z, 2304.0)
    rows_sorted = sorted(rows, key=lambda r: r["residual_median_px"])
    low = rows_sorted[: args.count]
    high = rows_sorted[-args.count :][::-1]
    selected = [("low", row) for row in low] + [("high", row) for row in high]
    tile_w, tile_h = 640, 360
    sheet = Image.new("RGB", (tile_w * args.count, tile_h * 2), (15, 15, 15))
    for i, (kind, row) in enumerate(selected):
        x = (i % args.count) * tile_w
        y = (0 if kind == "low" else 1) * tile_h
        sheet.paste(render_tile(ann, z, row, tile_w, tile_h, kind), (x, y))
    out_dir = args.output_root / "hawor_bridge_state" / args.case
    out_path = out_dir / "v18_hawor_bridge_residual_review_sheet.jpg"
    sheet.save(out_path, quality=92)
    report = {
        "method": "render_v18_hawor_bridge_review",
        "case": args.case,
        "claim_scope": "visual_QC_of_HaWoR_bridge_residuals_not_foundation_acceptance",
        "bridge_report": str(bridge_report_path),
        "bridge_npz": str(npz_path),
        "review_sheet": str(out_path),
        "reference_rows": len(rows),
        "selected_low_residual_rows": low,
        "selected_high_residual_rows": high,
    }
    write_json(out_dir / "v18_hawor_bridge_review_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--case", default="trash_1050")
    parser.add_argument("--count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
