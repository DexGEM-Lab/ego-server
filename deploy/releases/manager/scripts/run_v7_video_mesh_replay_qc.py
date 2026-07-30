#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def metric(report: dict, key: str, stat: str) -> float:
    raw = report.get(key)
    if not isinstance(raw, dict) or stat not in raw:
        raise RuntimeError(f"report lacks metric {key}.{stat}")
    return float(raw[stat])


def run_command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def mesh_archive_frames(path: Path, frame_start: int, frame_end: int) -> list[int]:
    with np.load(path, allow_pickle=False) as archive:
        if "frame_idx" not in archive.files:
            raise RuntimeError(f"mesh archive lacks frame_idx: {path}")
        frames = [int(frame) for frame in archive["frame_idx"].tolist() if int(frame_start) <= int(frame) <= int(frame_end)]
    if not frames:
        raise RuntimeError(f"mesh archive has no frames in range {frame_start}-{frame_end}: {path}")
    return frames


def selected_frames(args: argparse.Namespace, mesh_archive: Path) -> list[int] | None:
    if not args.archive_frames_only:
        return None
    return mesh_archive_frames(mesh_archive, int(args.frame_start), int(args.frame_end))


def run_zbuffer(args: argparse.Namespace, mesh_archive: Path, output_dir: Path, frames: list[int] | None) -> dict:
    argv = [
        sys.executable,
        str(args.scripts_dir / "render_mesh_zbuffer_qc_v3.py"),
        "--mesh-archive",
        str(mesh_archive),
        "--manifest",
        str(args.manifest),
        "--annotations",
        str(args.annotations),
        "--metric-depth-npz",
        str(args.metric_depth_npz),
        "--intrinsics-source",
        args.intrinsics_source,
        "--frame-start",
        str(args.frame_start),
        "--frame-end",
        str(args.frame_end),
        "--max-faces",
        str(args.max_faces),
        "--zbuffer-surface-mode",
        args.zbuffer_surface_mode,
        "--vertex-splat-radius-px",
        str(args.vertex_splat_radius_px),
        "--output-dir",
        str(output_dir),
    ]
    if frames is not None:
        argv.append("--frames")
        argv.extend(str(frame) for frame in frames)
    run_command(argv)
    return load_json(output_dir / "qc_mesh_zbuffer_projection_v3.json")


def same_path(actual: object, expected: Path, key: str) -> None:
    if not isinstance(actual, str) or not actual:
        raise RuntimeError(f"z-buffer report lacks {key}")
    actual_path = Path(actual)
    if not actual_path.exists():
        raise RuntimeError(f"z-buffer report {key} does not exist: {actual_path}")
    if actual_path.resolve() != expected.resolve():
        raise RuntimeError(f"z-buffer report {key} mismatch: {actual_path} != {expected}")


def manifest_frame_indices(path: Path, frame_start: int, frame_end: int, requested_frames: list[int] | None = None) -> list[int]:
    payload = load_json(path)
    manifest_rows = payload.get("frames")
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    selected = []
    requested = None if requested_frames is None else {int(frame) for frame in requested_frames}
    for frame in manifest_rows:
        idx = int(frame["frame_idx"])
        if requested is not None and idx not in requested:
            continue
        if int(frame_start) <= idx <= int(frame_end):
            selected.append(idx)
    if not selected:
        raise RuntimeError(f"{path} has no frames in range {frame_start}-{frame_end}")
    return selected


def validate_zbuffer_report(args: argparse.Namespace, path: Path, mesh_archive: Path, frames: list[int] | None) -> tuple[dict, Path, Path]:
    if not path.exists():
        raise RuntimeError(f"z-buffer report does not exist: {path}")
    report = load_json(path)
    if report.get("status") != "ok":
        raise RuntimeError(f"z-buffer report is not ok: {path}")
    if report.get("method") != "mesh_zbuffer_projection_qc_v3":
        raise RuntimeError(f"z-buffer report has wrong method: {path}")
    same_path(report.get("mesh_archive"), mesh_archive, "mesh_archive")
    same_path(report.get("manifest"), args.manifest, "manifest")
    same_path(report.get("annotations"), args.annotations, "annotations")
    same_path(report.get("metric_depth_npz"), args.metric_depth_npz, "metric_depth_npz")
    if str(report.get("intrinsics_source")) != str(args.intrinsics_source):
        raise RuntimeError("z-buffer report intrinsics_source mismatch")
    if int(report.get("vertex_splat_radius_px", -1)) != int(args.vertex_splat_radius_px):
        raise RuntimeError("z-buffer report vertex_splat_radius_px mismatch")
    if str(report.get("zbuffer_surface_mode", "triangles-plus-vertices")) != str(args.zbuffer_surface_mode):
        raise RuntimeError("z-buffer report zbuffer_surface_mode mismatch")
    if "full_fidelity_zbuffer" not in report or "max_faces" not in report:
        raise RuntimeError("z-buffer report lacks full-fidelity render contract")
    expected_full_fidelity = bool(int(args.max_faces) == 0)
    if bool(report["full_fidelity_zbuffer"]) != expected_full_fidelity:
        raise RuntimeError("z-buffer report full_fidelity_zbuffer mismatch")
    expected_max_faces = None if int(args.max_faces) == 0 else int(args.max_faces)
    actual_max_faces = report.get("max_faces")
    normalized_actual_max_faces = None if actual_max_faces is None else int(actual_max_faces)
    if normalized_actual_max_faces != expected_max_faces:
        raise RuntimeError("z-buffer report max_faces mismatch")
    expected_frames = manifest_frame_indices(args.manifest, int(args.frame_start), int(args.frame_end), frames)
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("z-buffer report lacks rows")
    actual_frames = [int(row["frame_idx"]) for row in rows]
    if actual_frames != expected_frames:
        raise RuntimeError(f"z-buffer report frame rows mismatch: {actual_frames} != {expected_frames}")
    if int(report.get("frames", -1)) != len(expected_frames):
        raise RuntimeError("z-buffer report frame count mismatch")
    video_raw = report.get("video")
    if not isinstance(video_raw, str) or not video_raw:
        raise RuntimeError("z-buffer report lacks video")
    video = Path(video_raw)
    if not video.exists():
        raise RuntimeError(f"z-buffer video does not exist: {video}")
    return report, path, video


def write_report(path: Path, report: dict) -> dict:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run(args: argparse.Namespace) -> dict:
    if not args.video_mesh_archive.exists():
        raise RuntimeError(f"video mesh archive does not exist: {args.video_mesh_archive}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zbuffer_dir = args.output_dir / "video_mesh_zbuffer_qc"
    frames = selected_frames(args, args.video_mesh_archive)
    if args.video_mesh_zbuffer_report is None:
        zbuffer = run_zbuffer(args, args.video_mesh_archive, zbuffer_dir, frames)
        zbuffer_report_path = zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"
        zbuffer_video_path = zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"
        zbuffer_cache = {"used": False}
    else:
        zbuffer, zbuffer_report_path, zbuffer_video_path = validate_zbuffer_report(
            args,
            args.video_mesh_zbuffer_report,
            args.video_mesh_archive,
            frames,
        )
        zbuffer_cache = {
            "used": True,
            "report": str(zbuffer_report_path),
            "video": str(zbuffer_video_path),
        }

    iou_median = metric(zbuffer, "silhouette_mask_iou", "median")
    visible_inside_median = metric(zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    zbuffer_abs_p95_median = metric(zbuffer, "zbuffer_depth_abs_p95_m", "median")
    pass_rows = {
        "silhouette_iou_median": bool(iou_median >= float(args.min_iou_median)),
        "visible_inside_median": bool(visible_inside_median >= float(args.min_visible_inside_median)),
        "zbuffer_abs_p95_median": bool(zbuffer_abs_p95_median <= float(args.max_zbuffer_abs_p95_median_m)),
    }
    accepted = all(pass_rows.values())
    replay_controls = {
        "samples": None,
        "max_faces": int(args.max_faces),
        "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
        "zbuffer_surface_mode": str(args.zbuffer_surface_mode),
        "full_fidelity_zbuffer": bool(args.max_faces == 0),
    }
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_video_mesh_replay_qc",
        "claim_tested": (
            "a video-derived object mesh archive can enter V7 delivery only when it "
            "directly replays against model-produced object masks, metric depth, "
            "camera poses, temporal factors, and physics checks"
        ),
        "video_mesh_archive": str(args.video_mesh_archive),
        "aligned_mesh_archive": str(args.video_mesh_archive),
        "mesh_prior": None,
        "observed_mesh_archive": str(args.video_mesh_archive),
        "zbuffer_report": str(zbuffer_report_path),
        "zbuffer_video": str(zbuffer_video_path),
        "video_mesh_zbuffer_cache": zbuffer_cache,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "evaluated_frames": frames if frames is not None else None,
        "replay_controls": replay_controls,
        "metrics": {
            "silhouette_iou_median": iou_median,
            "visible_inside_median": visible_inside_median,
            "zbuffer_abs_p95_median_m": zbuffer_abs_p95_median,
        },
        "thresholds": {
            "min_iou_median": float(args.min_iou_median),
            "min_visible_inside_median": float(args.min_visible_inside_median),
            "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
        },
        "pass": pass_rows,
        "delivery_pass_keys": [
            "silhouette_iou_median",
            "visible_inside_median",
            "zbuffer_abs_p95_median",
        ],
        "next_required_if_accepted": [
            "topology-independent track-surface QC",
            "mesh-surface contact recomputation",
            "selected-contact SDF when contact evidence is present",
            "full-hand SDF",
            "stakeholder render inspection",
        ],
    }
    return write_report(args.output_dir / "qc_v7_video_mesh_replay.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-mesh-archive", type=Path, required=True)
    parser.add_argument("--video-mesh-zbuffer-report", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="annotation-vggt")
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--zbuffer-surface-mode", choices=("triangles", "triangles-plus-vertices"), default="triangles-plus-vertices")
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--archive-frames-only", action="store_true")
    parser.add_argument("--min-iou-median", type=float, default=0.900)
    parser.add_argument("--min-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--max-zbuffer-abs-p95-median-m", type=float, default=0.010)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
