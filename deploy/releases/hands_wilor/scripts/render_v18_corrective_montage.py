#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
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


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", size, (35, 20, 20))
        d = ImageDraw.Draw(img)
        d.text((10, 38), f"missing: {path}", fill=(255, 120, 120), font=font(12))
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (0, 0, 0))
    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, size[0], 24), fill=(0, 0, 0))
    d.text((8, 5), label, fill=(255, 255, 255), font=font(12))
    return canvas


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


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    fps = float(ann.get("fps") or 30.0)
    case_dir = args.output_root / case / "corrective_montage"
    frame_dir = case_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    panel_size = (480, 270)
    labels_paths = [
        ("previous V18 overlay", args.previous_root / case / "overlay_frames"),
        ("graph-driven hand/object state", args.output_root / case / "corrective_overlay_frames"),
        ("temporal 2D MANO smoothing", args.output_root / case / "temporal_hand_pose_smoothing" / "frames"),
        ("HaWoR prior / provisioning failure", args.output_root / case / "hawor_ghost_attempt" / "frames"),
        ("tentative occlusion owner", args.output_root / case / "occlusion_owner_best_effort" / "frames"),
        ("occlusion acceptance audit", args.output_root / case / "occlusion_owner_acceptance_audit" / "frames"),
        ("contact acceptance audit", args.output_root / case / "contact_acceptance_audit" / "frames"),
        ("contact + local nonpenetration", args.output_root / case / "contact_nonpenetration_state" / "frames"),
        ("generic rigid fused-canonical SE3", args.output_root / case / "rigid_se3_attempt" / "world_frames"),
        ("frame-local visible surface", args.output_root / case / "visible_surface_state" / "world_frames"),
        ("geometry coverage audit", args.output_root / case / "geometry_coverage_audit" / "frames"),
        ("rigid SE3 residual check", args.output_root / case / "rigid_se3_residual_check" / "world_frames"),
        ("nonpenetration translation candidates", args.output_root / case / "nonpenetration_repair_proposal" / "diagnostic_xz_frames"),
    ]
    rows = int((len(labels_paths) + 1) // 2)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        idx = int(frame.get("frame_idx", 0))
        name = f"{idx:06d}.jpg"
        sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * rows + 34), (8, 8, 10))
        d = ImageDraw.Draw(sheet)
        d.rectangle((0, 0, sheet.size[0], 34), fill=(0, 0, 0))
        d.text((10, 8), f"V18 corrective montage {case} frame {idx+1}/{len(frames)} — changed mechanisms, not full closure", fill=(255, 255, 255), font=font(16))
        for i, (label, directory) in enumerate(labels_paths):
            img = panel(directory / name, label, panel_size)
            x = (i % 2) * panel_size[0]
            y = 34 + (i // 2) * panel_size[1]
            sheet.paste(img, (x, y))
        sheet.save(frame_dir / name, quality=88)
    video_path = case_dir / "v18_corrective_montage.mp4"
    encode_video(frame_dir, video_path, fps)
    report = {
        "method": "render_v18_corrective_montage",
        "case": case,
        "claim_scope": "full_timeline_visual_index_of_changed_corrective_mechanisms_not_full_v18_closure",
        "frame_count": len(frames),
        "fps": fps,
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "panels": [label for label, _directory in labels_paths],
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_corrective_montage_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_corrective_montage",
        "status": "corrective_montage_not_full_v18_closure",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_corrective_montage_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--previous-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
