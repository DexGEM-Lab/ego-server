#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANO_MODEL = Path("/data/dex_home/yiwen/hand_trajectory_loader/assets/mano/models/MANO_RIGHT.pkl")


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_path(raw: object, key: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{key} must be a non-empty path string")
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"{key} does not exist: {path}")
    return path


def shell_token(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_command(argv: list[str], dry_run: bool) -> None:
    print(" ".join(shell_token(arg) for arg in argv))
    if dry_run:
        return
    subprocess.run(argv, check=True)


def video_shape(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def check_video(
    path: Path,
    expected_frames: int,
    expected_fps: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict:
    shape = video_shape(path)
    if shape["frames"] != int(expected_frames):
        raise RuntimeError(f"{path}: frame count {shape['frames']} != {expected_frames}")
    if abs(float(shape["fps"]) - float(expected_fps)) > 0.05:
        raise RuntimeError(f"{path}: fps {shape['fps']} != {expected_fps}")
    if expected_width is not None and shape["width"] != int(expected_width):
        raise RuntimeError(f"{path}: width {shape['width']} != {expected_width}")
    if expected_height is not None and shape["height"] != int(expected_height):
        raise RuntimeError(f"{path}: height {shape['height']} != {expected_height}")
    return shape


def accepted_report(path: Path, label: str) -> dict:
    report = load_json(path)
    if report.get("status") != "accepted" or not bool(report.get("annotation_ready", False)):
        raise RuntimeError(f"{label} report is not accepted and annotation-ready: {path}")
    return report


def require_full_fidelity_replay(report: dict, path: Path) -> None:
    replay_controls = report.get("replay_controls", {})
    if not bool(replay_controls.get("full_fidelity_zbuffer", False)):
        raise RuntimeError(f"deliverable rendering requires full-fidelity replay, got diagnostic replay controls: {path}")


def run(args: argparse.Namespace) -> dict:
    replay = accepted_report(args.replay_report, "replay")
    require_full_fidelity_replay(replay, args.replay_report)
    physics = accepted_report(args.physics_report, "physics")
    mesh_archive = require_path(replay.get("aligned_mesh_archive"), "replay.aligned_mesh_archive")
    contact_report = require_path(physics.get("contact_report"), "physics.contact_report")
    if int(replay["frame_start"]) != int(physics["frame_start"]) or int(replay["frame_end"]) != int(physics["frame_end"]):
        raise RuntimeError("replay and physics reports cover different frame windows")
    if not args.manifest.exists():
        raise RuntimeError(f"manifest does not exist: {args.manifest}")
    if not args.annotations.exists():
        raise RuntimeError(f"annotations do not exist: {args.annotations}")
    if not args.mano_model.exists():
        raise RuntimeError(f"MANO model does not exist: {args.mano_model}")

    frame_start = int(replay["frame_start"])
    frame_end = int(replay["frame_end"])
    expected_frames = frame_end - frame_start + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlay"
    world_dir = args.output_dir / "world"
    overlay_cmd = [
        sys.executable,
        str(args.scripts_dir / "render_mesh_surface_contact_review_v3.py"),
        "--manifest",
        str(args.manifest),
        "--annotations",
        str(args.annotations),
        "--contact-report",
        str(contact_report),
        "--object-mesh-npz",
        str(mesh_archive),
        "--output-dir",
        str(overlay_dir),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--output-fps",
        str(args.output_fps),
        "--render-width",
        str(args.overlay_width),
        "--max-mesh-edges",
        str(args.max_overlay_mesh_edges),
    ]
    world_cmd = [
        sys.executable,
        str(args.scripts_dir / "render_world_reconstruction_v3.py"),
        "--manifest",
        str(args.manifest),
        "--annotations",
        str(args.annotations),
        "--object-mesh-npz",
        str(mesh_archive),
    ]
    if args.append_report is not None:
        if not args.append_report.exists():
            raise RuntimeError(f"append report does not exist: {args.append_report}")
        world_cmd.extend(["--append-report", str(args.append_report)])
    if args.dynamics_report is not None:
        if not args.dynamics_report.exists():
            raise RuntimeError(f"dynamics report does not exist: {args.dynamics_report}")
        world_cmd.extend(["--dynamics-report", str(args.dynamics_report)])
    world_cmd.extend([
        "--contact-report",
        str(contact_report),
        "--mano-model",
        str(args.mano_model),
        "--output-dir",
        str(world_dir),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--output-fps",
        str(args.output_fps),
        "--output-width",
        str(args.side_by_side_width),
        "--panel-width",
        str(args.world_panel_width),
        "--panel-height",
        str(args.world_panel_height),
        "--caption-height",
        str(args.caption_height),
        "--max-overlay-mesh-edges",
        str(args.max_overlay_mesh_edges),
        "--caption-prefix",
        args.caption_prefix,
        "--include-camera-in-focus",
    ])
    run_command(overlay_cmd, bool(args.dry_run))
    run_command(world_cmd, bool(args.dry_run))

    overlay_video = overlay_dir / "mesh_surface_contact_review.mp4"
    side_by_side_video = world_dir / "world_reconstruction_side_by_side.mp4"
    world_video = world_dir / "world_reconstruction_3d.mp4"
    structural_qc = None
    if not args.dry_run:
        structural_qc = {
            "expected_frames": expected_frames,
            "expected_fps": float(args.output_fps),
            "overlay": check_video(overlay_video, expected_frames, float(args.output_fps), int(args.overlay_width), None),
            "side_by_side": check_video(
                side_by_side_video,
                expected_frames,
                float(args.output_fps),
                int(args.side_by_side_width),
                int(args.world_panel_height + args.caption_height),
            ),
            "world_3d": check_video(
                world_video,
                expected_frames,
                float(args.output_fps),
                int(args.world_panel_width),
                int(args.world_panel_height),
            ),
        }

    report = {
        "status": "dry_run" if args.dry_run else "ok",
        "method": "render_mesh_candidate_deliverables",
        "claim_tested": "an accepted object mesh candidate can be rendered as MANO/object overlay, standalone world 3D animation, and side-by-side presentation with semantic captions",
        "replay_report": str(args.replay_report),
        "physics_report": str(args.physics_report),
        "mesh_archive": str(mesh_archive),
        "contact_report": str(contact_report),
        "append_report": str(args.append_report) if args.append_report is not None else None,
        "dynamics_report": str(args.dynamics_report) if args.dynamics_report is not None else None,
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "mano_model": str(args.mano_model),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "videos": {
            "overlay": str(overlay_video),
            "side_by_side": str(side_by_side_video),
            "world_3d": str(world_video),
        },
        "renderer_reports": {
            "overlay": str(overlay_dir / "review_manifest.json"),
            "world": str(world_dir / "render_manifest.json"),
        },
        "structural_qc": structural_qc,
    }
    report_path = args.output_dir / "v7_candidate_deliverables_manifest.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--physics-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--append-report", type=Path)
    parser.add_argument("--dynamics-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--mano-model", type=Path, default=DEFAULT_MANO_MODEL)
    parser.add_argument("--output-fps", type=float, default=6.0)
    parser.add_argument("--overlay-width", type=int, default=1280)
    parser.add_argument("--side-by-side-width", type=int, default=1920)
    parser.add_argument("--world-panel-width", type=int, default=960)
    parser.add_argument("--world-panel-height", type=int, default=720)
    parser.add_argument("--caption-height", type=int, default=58)
    parser.add_argument("--max-overlay-mesh-edges", type=int, default=260)
    parser.add_argument("--caption-prefix", default="V7 mesh-backed reconstruction")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
