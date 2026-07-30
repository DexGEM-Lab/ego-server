#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    load_json,
    require_dict,
    require_int,
    require_str,
    write_json,
)


STATUS = "v17_deliverable_render_qc"
CLAIM = (
    "This artifact renders the V17 deliverable state (interior-owned baked hands plus the "
    "multi-object reconstructed object stream) into full-duration overlay, 3D world, and "
    "side-by-side videos, then verifies frame-count and duration agreement with the raw video. "
    "It is a duration-honest render pass: visual quality, object completeness, and contact closure "
    "are reviewed separately, the QC banner states the open limits, and all V17 readiness flags "
    "remain false."
)

BANNER_LINES = (
    "V17 PARTIAL: interior-owned hands + reconstructed-object subset",
    "object_geometry_complete=false | deliverable_ready=false | quality review pending",
)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    try:
        info = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()
    if info["fps"] <= 0 or info["frame_count"] <= 0:
        raise RuntimeError(f"invalid video metadata for {path}")
    return info


def draw_banner(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(1.0, w / 1700.0))
    line_h = int(round(28 * scale))
    banner_h = int(round(16 * scale + line_h * len(BANNER_LINES)))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0.0, out)
    y = int(round(22 * scale))
    for line in BANNER_LINES:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return out


def write_banner_video(src: Path, dst: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open render {src}")
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    frames_written = 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open writer {tmp}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(draw_banner(frame))
                frames_written += 1
        finally:
            writer.release()
    finally:
        cap.release()
    tmp.replace(dst)
    return {"path": str(dst), "frames_written": frames_written}


def render_case(manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    state = require_dict(load_json(manifest_path), f"{manifest_path} deliverable state")
    case = require_str(state.get("case"), "case")
    v16 = require_dict(load_json(Path(require_str(state.get("v16_manifest"), "v16 manifest path"))), "v16 manifest")
    clip = existing_path(Path(require_str(v16.get("clip"), "clip path")), f"{case} raw clip")
    raw = video_info(clip)
    raw_frame_count = require_int(state.get("raw_frame_count"), "raw_frame_count")
    if raw["frame_count"] != raw_frame_count:
        raise RuntimeError(f"{case} raw clip has {raw['frame_count']} frames, manifest says {raw_frame_count}")
    case_dir = args.output_root / case
    render_dir = case_dir / "render_tmp"
    final_dir = case_dir / "renders"
    final_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        "scripts/fuse_v1_full_fidelity.py",
        "--clip",
        str(clip),
        "--output-dir",
        str(render_dir),
        "--render-only-annotations",
        require_str(state.get("annotations"), "annotations path"),
        "--object-mesh-npz",
        require_str(state.get("object_mesh_archive"), "object mesh archive"),
        "--render-width",
        str(int(args.render_width)),
    ]
    completed = subprocess.run(cmd, cwd=str(args.repo_root), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case} render failed rc={completed.returncode}\nstdout tail: {completed.stdout[-1500:]}\n"
            f"stderr tail: {completed.stderr[-1500:]}"
        )
    names = {
        "overlay": ("overlay_mano_object.mp4", "v17_overlay_mano_object.mp4"),
        "world": ("reconstruction_3d_world.mp4", "v17_reconstruction_3d_world.mp4"),
        "side_by_side": ("side_by_side.mp4", "v17_side_by_side.mp4"),
    }
    render_qc: dict[str, Any] = {}
    for key, (src_name, dst_name) in names.items():
        src = existing_path(render_dir / src_name, f"{case} render output {src_name}")
        banner = write_banner_video(src, final_dir / dst_name)
        out_info = video_info(Path(banner["path"]))
        render_qc[key] = {
            **banner,
            **out_info,
            "frame_count_match": bool(out_info["frame_count"] == raw_frame_count),
            "duration_s": out_info["frame_count"] / out_info["fps"],
            "raw_duration_s": raw["frame_count"] / raw["fps"],
        }
    duration_pass = all(row["frame_count_match"] for row in render_qc.values())
    report = {
        "method": "render_v17_deliverable_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "deliverable_state_manifest": str(manifest_path),
        "raw_frame_count": raw_frame_count,
        "render_width": int(args.render_width),
        "banner_lines": list(BANNER_LINES),
        "renders": render_qc,
        "duration_render_qc_pass": duration_pass,
        "visual_quality_qc_pass": False,
        "object_archive_coverage_fraction": state.get("object_archive_coverage_fraction"),
        "archived_object_count": state.get("archived_object_count"),
        **FALSE_READY,
    }
    if not duration_pass:
        raise RuntimeError(f"{case} render duration QC failed: {json.dumps(render_qc)[:600]}")
    write_json(case_dir / "v17_deliverable_render_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    reports = [
        render_case(
            existing_path(
                args.deliverable_state_root / case / "v17_deliverable_state_manifest.json",
                f"{case} deliverable state manifest",
            ),
            args,
        )
        for case in args.cases
    ]
    summary = {
        "method": "render_v17_deliverable_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "duration_render_qc_pass": bool(all(r.get("duration_render_qc_pass") is True for r in reports)),
        "visual_quality_qc_pass": False,
        "cases": [
            {
                "case": require_str(r.get("case"), "case"),
                "raw_frame_count": r["raw_frame_count"],
                "duration_render_qc_pass": r["duration_render_qc_pass"],
                "renders": {k: v["path"] for k, v in require_dict(r.get("renders"), "renders").items()},
                **FALSE_READY,
            }
            for r in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_deliverable_render_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument(
        "--deliverable-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_deliverable_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_deliverable_renders"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
