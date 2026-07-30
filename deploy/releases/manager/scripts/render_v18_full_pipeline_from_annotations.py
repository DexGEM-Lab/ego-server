#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from argparse import Namespace
from pathlib import Path

from run_v18_full_pipeline import render_overlay, render_world, write_json, ffprobe_frame_count


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def compose_side_by_side(overlay: Path, world: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(overlay), '-i', str(world),
        '-filter_complex', '[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[left];[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[right];[left][right]hstack=inputs=2[v]',
        '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(output)
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--v16-root', type=Path, default=Path('/data2/ego_annotation_outputs/v16_full_pipeline'))
    args = ap.parse_args()
    ann = load_json(args.annotations)
    case_dir = args.output_root / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / 'annotations_v18_full.json', ann)
    render_args = Namespace(output_root=args.output_root, v16_root=args.v16_root)
    overlay_qc = render_overlay(args.case, ann, render_args)
    world_qc = render_world(args.case, ann, render_args)
    overlay = Path(overlay_qc['output_video'])
    world = Path(world_qc['output_video'])
    side = case_dir / 'v18_side_by_side.mp4'
    compose_side_by_side(overlay, world, side)
    frame_count = len(ann.get('frames', []))
    summary = {
        'method': 'render_v18_full_pipeline_from_annotations',
        'status': 'ok',
        'case': args.case,
        'annotations': str(args.annotations),
        'output_annotations': str(case_dir / 'annotations_v18_full.json'),
        'overlay': overlay_qc,
        'world': world_qc,
        'side_by_side_video': str(side),
        'expected_frame_count': frame_count,
        'overlay_frame_count': overlay_qc.get('frame_count'),
        'world_frame_count': world_qc.get('frame_count'),
        'side_by_side_frame_count': ffprobe_frame_count(side),
        'frame_count_match': overlay_qc.get('frame_count') == frame_count and world_qc.get('frame_count') == frame_count and ffprobe_frame_count(side) == frame_count,
        'claim_scope': 'Renders consume an already-built V18 annotation JSON; this does not rebuild unrelated modules.',
    }
    write_json(case_dir / 'render_from_annotations_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
