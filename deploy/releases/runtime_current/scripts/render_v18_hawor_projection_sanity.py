#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
SIDE_COLOR = {"left": (30, 230, 80), "right": (255, 150, 20)}


def font(size: int) -> ImageFont.ImageFont:
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def project(points: np.ndarray, intr: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intr.astype(float)
    out = np.empty((len(points), 2), dtype=np.float64)
    z = points[:, 2]
    out[:, 0] = fx * points[:, 0] / z + cx
    out[:, 1] = fy * points[:, 1] / z + cy
    return out


def inside_fraction(points2d: np.ndarray, width: int, height: int) -> float:
    if points2d.size == 0:
        return 0.0
    ok = (points2d[:, 0] >= 0) & (points2d[:, 0] < width) & (points2d[:, 1] >= 0) & (points2d[:, 1] < height)
    return float(np.mean(ok))


def read_video_frame(video: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"cannot read frame {frame_idx} from {video}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    bbox = draw.textbbox(xy, text, font=fnt)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text(xy, text, font=fnt, fill=fill)


def render_frame(base: Image.Image, rows: list[dict[str, Any]], intr: np.ndarray, track_records: list[dict[str, Any]] | None = None) -> tuple[Image.Image, list[dict[str, Any]]]:
    img = base.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    fnt = font(18)
    small = font(14)
    width, height = img.size
    reports: list[dict[str, Any]] = []
    if track_records:
        for tr in track_records:
            box = tr.get("box_xyxyscore")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            score = float(box[4]) if len(box) >= 5 else float("nan")
            handedness = tr.get("det_handedness")
            if isinstance(handedness, list) and handedness:
                hand_txt = f"h={handedness[0]:.0f}"
            else:
                hand_txt = "h=?"
            draw.rectangle((x1, y1, x2, y2), outline=(80, 180, 255), width=3)
            draw_label(draw, (int(max(0, min(width - 260, x1))), int(max(36, min(height - 28, y1)))), f"track {tr.get('track_id')} {hand_txt} s={score:.2f}", small, (80, 180, 255))
    for row in rows:
        side = str(row["side"])
        color = SIDE_COLOR.get(side, (255, 255, 255))
        joints = np.asarray(row["joints_camera"], dtype=np.float64)
        verts = np.asarray(row["vertices_camera"], dtype=np.float64)
        if joints.shape != (21, 3) or verts.ndim != 2 or verts.shape[1] != 3:
            reports.append({"side": side, "status": "bad_shape"})
            continue
        if np.any(joints[:, 2] <= 1e-6) or np.any(verts[:, 2] <= 1e-6):
            reports.append({"side": side, "status": "nonpositive_depth", "joint_z_min": float(joints[:, 2].min()), "vertex_z_min": float(verts[:, 2].min())})
            continue
        pts = project(joints, intr)
        stride = max(1, len(verts) // 180)
        vpts = project(verts[::stride], intr)
        for x, y in vpts:
            if -20 <= x < width + 20 and -20 <= y < height + 20:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        for a, b in HAND_EDGES:
            ax, ay = pts[a]
            bx, by = pts[b]
            draw.line((ax, ay, bx, by), fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        wrist = tuple(int(round(v)) for v in pts[0])
        draw_label(draw, (max(0, min(width - 200, wrist[0] + 8)), max(30, min(height - 30, wrist[1] + 8))), f"HaWoR {side}", small, color)
        reports.append({
            "side": side,
            "status": "projected",
            "joint_inside_image_fraction": inside_fraction(pts, width, height),
            "vertex_sample_inside_image_fraction": inside_fraction(vpts, width, height),
            "median_joint_depth_m": float(np.median(joints[:, 2])),
            "median_vertex_depth_m": float(np.median(verts[:, 2])),
            "joint_bbox_xyxy": [float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))],
        })
    draw.rectangle((0, 0, width, 34), fill=(0, 0, 0))
    return img, reports


def make_contact_sheet(images: list[Image.Image], labels: list[str], out: Path) -> None:
    if not images:
        return
    thumb_w = 640
    thumb_h = int(round(images[0].height * thumb_w / images[0].width))
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    fnt = font(18)
    for i, img in enumerate(images):
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + 28)
        sheet.paste(thumb, (x, y + 28))
        draw.text((x + 8, y + 4), labels[i], font=fnt, fill=(255, 255, 255))
    sheet.save(out, quality=92)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.bridge_npz)
    frame_idx = np.asarray(z["frame_idx"], dtype=np.int32)
    side_code = np.asarray(z["side"], dtype=np.int32)
    vertices = np.asarray(z["vertices_hawor_camera_m"], dtype=np.float64)
    joints = np.asarray(z["joints_hawor_camera_m"], dtype=np.float64)
    frame_to_rows: dict[int, list[dict[str, Any]]] = {}
    for i, f in enumerate(frame_idx):
        side = "left" if int(side_code[i]) == 0 else "right" if int(side_code[i]) == 1 else str(side_code[i])
        frame_to_rows.setdefault(int(f), []).append({"side": side, "vertices_camera": vertices[i], "joints_camera": joints[i]})
    frame_to_tracks: dict[int, list[dict[str, Any]]] = {}
    if args.hawor_tracks_npy is not None:
        tracks_obj = np.load(args.hawor_tracks_npy, allow_pickle=True)
        tracks = tracks_obj.item() if getattr(tracks_obj, "shape", None) == () else tracks_obj
        if not isinstance(tracks, dict):
            raise RuntimeError(f"expected dict-like HaWoR tracks object in {args.hawor_tracks_npy}")
        for track_id, records in tracks.items():
            for rec in records:
                f = int(rec.get("frame", -1))
                if f < 0:
                    continue
                box = rec.get("det_box")
                handedness = rec.get("det_handedness")
                frame_to_tracks.setdefault(f, []).append({
                    "track_id": str(track_id),
                    "det": bool(rec.get("det")),
                    "box_xyxyscore": np.asarray(box).reshape(-1).astype(float).tolist() if box is not None else None,
                    "det_handedness": np.asarray(handedness).reshape(-1).astype(float).tolist() if handedness is not None else None,
                })
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    focal = float(np.asarray(z.get("img_focal", np.asarray([2304.0]))).reshape(-1)[0]) if "img_focal" in z.files else float(args.focal)
    intr = np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float64)
    rendered: list[Image.Image] = []
    labels: list[str] = []
    frame_reports: list[dict[str, Any]] = []
    for f in args.frames:
        if f < 0 or f >= total:
            frame_reports.append({"frame_idx": int(f), "status": "outside_video_range"})
            continue
        base = read_video_frame(args.video, int(f))
        img, rows = render_frame(base, frame_to_rows.get(int(f), []), intr, frame_to_tracks.get(int(f), []))
        out_path = args.output_dir / f"{int(f):06d}_hawor_projection.jpg"
        img.save(out_path, quality=92)
        rendered.append(img)
        labels.append(f"frame {int(f)}")
        frame_reports.append({"frame_idx": int(f), "output": str(out_path), "hand_reports": rows, "track_records": frame_to_tracks.get(int(f), [])})
    sheet_path = args.output_dir / "hawor_projection_contact_sheet.jpg"
    make_contact_sheet(rendered, labels, sheet_path)
    report = {
        "method": "render_v18_hawor_projection_sanity",
        "case": args.case,
        "video": str(args.video),
        "bridge_npz": str(args.bridge_npz),
        "hawor_tracks_npy": str(args.hawor_tracks_npy) if args.hawor_tracks_npy is not None else None,
        "output_dir": str(args.output_dir),
        "contact_sheet": str(sheet_path),
        "video_frame_count": total,
        "video_size": [width, height],
        "intrinsics_used": [float(v) for v in intr.tolist()],
        "frames": frame_reports,
        "claim_scope": "visual_projection_sanity_only_not_foundation_acceptance",
    }
    (args.output_dir / "hawor_projection_sanity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bridge-npz", type=Path, required=True)
    parser.add_argument("--hawor-tracks-npy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--focal", type=float, default=2304.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
