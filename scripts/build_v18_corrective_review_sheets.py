#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def load_panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", size, (35, 20, 20))
        d = ImageDraw.Draw(img)
        d.text((12, 12), f"missing: {path}", fill=(255, 120, 120), font=font(16))
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (0, 0, 0))
    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, size[0], 28), fill=(0, 0, 0))
    d.text((8, 6), label, fill=(255, 255, 255), font=font(15))
    return canvas


def text_panel(label: str, lines: list[str], size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (12, 14, 20))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, size[0], 30), fill=(0, 0, 0))
    d.text((8, 7), label, fill=(255, 255, 255), font=font(15))
    y = 48
    for line in lines:
        d.text((14, y), line, fill=(230, 230, 230), font=font(18))
        y += 30
    return canvas


def make_sheet(case: str, frame_idx: int, output_root: Path, corrective_root: Path, previous_root: Path) -> Path:
    frame = f"{frame_idx:06d}.jpg"
    panels: list[Image.Image] = [
        load_panel(previous_root / case / "overlay_frames" / frame, "previous V18 overlay (V16-base)", (640, 280)),
        load_panel(corrective_root / case / "corrective_overlay_frames" / frame, "corrective graph-driven overlay", (640, 280)),
        load_panel(corrective_root / case / "hawor_ghost_attempt" / "frames" / frame, "HaWoR ghost / execution failure", (640, 280)),
        load_panel(corrective_root / case / "occlusion_owner_best_effort" / "frames" / frame, "tentative occlusion-owner best effort", (640, 280)),
        load_panel(corrective_root / case / "rigid_se3_attempt" / "world_frames" / frame, "generic rigid SE3 fused-canonical attempt", (640, 280)),
        load_panel(corrective_root / case / "visible_surface_state" / "world_frames" / frame, "frame-local visible surface state", (640, 280)),
        text_panel("scope / interpretation", [
            "Graph render drives hand boxes + shifted MANO skeletons.",
            "Owner panel shows tentative graph-selected owners with blockers.",
            "Rigid SE3 tests fused canonical geometry under generic prior.",
            "Visible-surface panel shows frame-local RGBD geometry evidence.",
            "HaWoR panel shows prior where available or provisioning failure.",
            "None claim full V18 closure or solved occlusion/contact.",
        ], (640, 280)),
        text_panel("machine-readable state", [
            "annotations_v18_corrective_state.json contains these deltas:",
            "graph_shifted_mano, rigid_stable_pose, visible_surface,",
            "hawor_prior/provisioning_failure, tentative_occlusion_owner.",
            "Source full annotation remains referenced, not duplicated here.",
        ], (640, 280)),
    ]
    panel_size = (640, 280)
    sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * 4 + 42), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    d.rectangle((0, 0, sheet.size[0], 42), fill=(0, 0, 0))
    d.text((12, 10), f"V18 corrective review — {case} frame {frame_idx}", fill=(255, 255, 255), font=font(20))
    for i, panel in enumerate(panels):
        x = (i % 2) * panel_size[0]
        y = 42 + (i // 2) * panel_size[1]
        sheet.paste(panel, (x, y))
    out = output_root / f"{case}_{frame_idx:06d}_corrective_review.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return out


def parse_frame_spec(specs: Iterable[str]) -> list[tuple[str, int]]:
    out = []
    for spec in specs:
        case, frame_s = spec.split(":", 1)
        out.append((case, int(frame_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--previous-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/review_sheets"))
    parser.add_argument("--frames", nargs="+", default=["trash_1050:53", "trash_1050:840", "trash_1050:850", "trash_1050:856", "trash_1050:872", "task5_tomato_960:780"])
    args = parser.parse_args()
    for case, frame_idx in parse_frame_spec(args.frames):
        print(make_sheet(case, frame_idx, args.output_root, args.corrective_root, args.previous_root))


if __name__ == "__main__":
    main()
