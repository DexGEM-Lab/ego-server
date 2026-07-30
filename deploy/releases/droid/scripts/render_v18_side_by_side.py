#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_full_duration_status_side_by_side"
CLAIM = (
    "This render is a full-duration V18 side-by-side status video: left is raw-frame 2D status overlay, "
    "right is abstract world/status visualization. It is not a complete object-pose deliverable."
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


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def ffprobe_frame_count(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def compose_video(overlay_path: Path, world_path: Path, output_path: Path, width_each: int, height: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[left];"
        f"[1:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(overlay_path),
        "-i",
        str(world_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.render_root / case
    overlay_qc_path = case_dir / "v18_status_overlay_qc.json"
    world_qc_path = case_dir / "v18_world_status_qc.json"
    overlay_qc = require_dict(load_json(overlay_qc_path), f"{case} overlay qc")
    world_qc = require_dict(load_json(world_qc_path), f"{case} world qc")
    overlay_path = Path(str(overlay_qc.get("output_video")))
    world_path = Path(str(world_qc.get("output_video")))
    if not overlay_path.exists():
        raise RuntimeError(f"{case}: missing overlay video {overlay_path}")
    if not world_path.exists():
        raise RuntimeError(f"{case}: missing world/status video {world_path}")
    expected_count = require_int(overlay_qc.get("state_frame_count"), "overlay state_frame_count")
    world_count = require_int(world_qc.get("state_frame_count"), "world state_frame_count")
    if world_count != expected_count:
        raise RuntimeError(f"{case}: overlay/world expected frame counts differ: {expected_count} vs {world_count}")
    overlay_probe = ffprobe_frame_count(overlay_path)
    world_probe = ffprobe_frame_count(world_path)
    if overlay_probe != expected_count or world_probe != expected_count:
        raise RuntimeError(f"{case}: input frame-count QC failed: overlay={overlay_probe}, world={world_probe}, expected={expected_count}")
    output_path = case_dir / "v18_status_side_by_side.mp4"
    start = time.perf_counter()
    compose_video(overlay_path, world_path, output_path, args.width_each, args.height)
    elapsed = time.perf_counter() - start
    output_probe = ffprobe_frame_count(output_path)
    qc = {
        "method": "render_v18_side_by_side",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "status_overlay_qc": str(overlay_qc_path),
            "world_status_qc": str(world_qc_path),
            "status_overlay_video": str(overlay_path),
            "world_status_video": str(world_path),
        },
        "output_video": str(output_path),
        "expected_frame_count": expected_count,
        "overlay_video_frame_count": overlay_probe,
        "world_video_frame_count": world_probe,
        "side_by_side_frame_count": output_probe,
        "frame_count_match": output_probe == expected_count,
        "elapsed_s": elapsed,
        "canvas_size": [args.width_each * 2, args.height],
        "render_contract": {
            "full_duration": True,
            "same_frame_count_as_raw": output_probe == expected_count,
            "left_panel": "raw-frame status overlay",
            "right_panel": "abstract world/status visualization",
            "pose_filled_through_occlusion": False,
            "contact_factor_ready_rows": 0,
            "not_complete_object_pose_deliverable": True,
        },
        **FALSE_READY,
    }
    write_json(case_dir / "v18_status_side_by_side_qc.json", qc)
    return qc


def render(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    qcs = [render_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "render_v18_side_by_side",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(qcs),
        "elapsed_s": elapsed,
        "all_frame_counts_match": all(bool(qc.get("frame_count_match")) for qc in qcs),
        "outputs": [
            {
                "case": qc["case"],
                "output_video": qc["output_video"],
                "qc_path": str(args.render_root / str(qc["case"]) / "v18_status_side_by_side_qc.json"),
                "expected_frame_count": qc["expected_frame_count"],
                "side_by_side_frame_count": qc["side_by_side_frame_count"],
                "frame_count_match": qc["frame_count_match"],
                **FALSE_READY,
            }
            for qc in qcs
        ],
        **FALSE_READY,
    }
    write_json(args.render_root / "v18_status_side_by_side_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_renders"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--width-each", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2))


if __name__ == "__main__":
    main()
