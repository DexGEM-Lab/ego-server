#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata for {path}")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def check_video(path: Path, raw: VideoInfo) -> dict[str, Any]:
    info = video_info(path)
    return {
        "path": str(path),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "frame_count": info.frame_count,
        "raw_frame_count": raw.frame_count,
        "frame_count_match": info.frame_count == raw.frame_count,
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=str(cwd), check=True)


QC_BANNER_LINES = (
    "V17 QC ONLY: duration check, not visual-quality pass",
    "annotation_ready=false | v3_solver_complete=false | single-object stream",
)

OBJECT_LIMIT_KEYS = (
    "multi_object_timeline_ready",
    "object_schema_status",
    "missing_multi_object_roster_required",
    "object_geometry_complete",
    "object_pose_requirement_met",
    "object_geometry_status",
)


def require_limit_field(source: dict[str, Any], key: str, origin: Path) -> Any:
    if key not in source:
        raise RuntimeError(f"{origin} is missing required V17 limitation field {key}")
    return source[key]


def limitation_fields(state: dict[str, Any], solver_report: dict[str, Any], origin: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in OBJECT_LIMIT_KEYS:
        if key in solver_report:
            out[key] = solver_report[key]
        else:
            out[key] = require_limit_field(state, key, origin)
    return out


def draw_qc_banner(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.55, min(1.0, w / 1600.0))
    line_h = int(round(30 * scale))
    banner_h = int(round(18 * scale + line_h * len(QC_BANNER_LINES)))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, out, 0.32, 0.0, out)
    y = int(round(24 * scale))
    for line in QC_BANNER_LINES:
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return out


def write_qc_video(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {src}")
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"invalid video metadata for {src}")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer {tmp}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(draw_qc_banner(frame))
        finally:
            writer.release()
    finally:
        cap.release()
    tmp.replace(dst)


def inspection_frames(state: dict[str, Any], raw: VideoInfo) -> list[int]:
    frames = {0, raw.frame_count // 4, raw.frame_count // 2, (3 * raw.frame_count) // 4, raw.frame_count - 1}
    solver_report = state.get("solver_report")
    if isinstance(solver_report, str):
        report = load_json(Path(solver_report))
        rows = (report.get("contact_after") or {}).get("rows") if isinstance(report, dict) else None
        if isinstance(rows, list):
            contact_frames = sorted(
                {
                    int(row["frame_idx"])
                    for row in rows
                    if isinstance(row, dict) and isinstance(row.get("frame_idx"), int)
                }
            )
            if len(contact_frames) <= 24:
                frames.update(contact_frames)
            elif contact_frames:
                pick = np.linspace(0, len(contact_frames) - 1, 24).round().astype(int)
                frames.update(contact_frames[int(i)] for i in pick)
    return sorted(frame for frame in frames if 0 <= frame < raw.frame_count)


def sheet_filename(method_name: str) -> str:
    if "contact_mode_factor" in method_name:
        return "v17_contact_mode_factor_side_by_side_sheet.jpg"
    if "graph" in method_name:
        return "v17_graph_anchor_side_by_side_sheet.jpg"
    return "v17_anchor_side_by_side_sheet.jpg"


def visual_inspection_sheet(video: Path, output: Path, frames: list[int]) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {video}")
    thumbs: list[np.ndarray] = []
    try:
        for frame_idx in frames:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
                raise RuntimeError(f"failed to seek {video} to frame {frame_idx}")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read {video} frame {frame_idx}")
            thumb_w = 960
            thumb_h = int(round(thumb_w * frame.shape[0] / frame.shape[1]))
            thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            thumb = draw_qc_banner(thumb)
            frame_label_y = thumb.shape[0] - 18
            cv2.putText(
                thumb,
                f"frame {frame_idx}",
                (18, frame_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                thumb,
                f"frame {frame_idx}",
                (18, frame_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
    finally:
        cap.release()
    if not thumbs:
        raise RuntimeError("no inspection frames selected")
    cols = 1 if len(thumbs) <= 3 else 2
    rows = int(math.ceil(len(thumbs) / cols))
    h = max(thumb.shape[0] for thumb in thumbs)
    w = max(thumb.shape[1] for thumb in thumbs)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h : r * h + thumb.shape[0], c * w : c * w + thumb.shape[1]] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write {output}")
    return {"path": str(output), "sampled_frames": [int(frame) for frame in frames]}


def valid_video(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        video_info(path)
    except RuntimeError:
        path.unlink()
        return False
    return True


def existing_render_source(render_dir: Path, final_dir: Path, src_name: str, dst_name: str) -> Path | None:
    tmp_source = render_dir / src_name
    if valid_video(tmp_source):
        return tmp_source
    legacy_names = {
        "qc_overlay_mano_object_multi.mp4": "overlay_mano_object_multi.mp4",
        "qc_world_reconstruction_3d_v17.mp4": "world_reconstruction_3d_v17.mp4",
        "qc_side_by_side_v17.mp4": "side_by_side_v17.mp4",
    }
    legacy_name = legacy_names.get(dst_name)
    if legacy_name is not None:
        legacy = final_dir / legacy_name
        if valid_video(legacy):
            return legacy
    dst = final_dir / dst_name
    return dst if valid_video(dst) else None


def remove_non_qc_visual_artifacts(case_dir: Path, render_dir: Path, final_dir: Path, current_sheet: Path) -> list[str]:
    stale_paths = [
        final_dir / "overlay_mano_object_multi.mp4",
        final_dir / "world_reconstruction_3d_v17.mp4",
        final_dir / "side_by_side_v17.mp4",
        render_dir / "overlay_mano_object.mp4",
        render_dir / "reconstruction_3d_world.mp4",
        render_dir / "side_by_side.mp4",
        case_dir / "v17_graph_anchor_side_by_side_sheet.jpg",
        case_dir / "v17_anchor_side_by_side_sheet.jpg",
    ]
    removed: list[str] = []
    for path in stale_paths:
        if path == current_sheet:
            continue
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def render_case(args: argparse.Namespace, case_manifest: Path, output_root: Path) -> dict[str, Any]:
    state = load_json(case_manifest)
    v16 = load_json(Path(state["v16_manifest"]))
    clip = Path(v16["clip"])
    raw = video_info(clip)
    if int(state["raw_frame_count"]) != raw.frame_count:
        raise RuntimeError(f"{case_manifest} raw_frame_count does not match clip metadata")
    case_dir = output_root / str(state["case"])
    render_dir = case_dir / "render_tmp"
    final_dir = case_dir / "renders"
    final_dir.mkdir(parents=True, exist_ok=True)
    if not bool(args.reuse_existing_final_renders):
        render_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                str(args.python),
                "scripts/fuse_v1_full_fidelity.py",
                "--clip",
                str(clip),
                "--output-dir",
                str(render_dir),
                "--render-only-annotations",
                str(state["annotations"]),
                "--object-mesh-npz",
                str(state["object_mesh_archive"]),
                "--render-width",
                str(args.render_width),
            ],
            args.repo_root,
        )
    names = {
        "overlay": ("overlay_mano_object.mp4", "qc_overlay_mano_object_multi.mp4"),
        "world": ("reconstruction_3d_world.mp4", "qc_world_reconstruction_3d_v17.mp4"),
        "side_by_side": ("side_by_side.mp4", "qc_side_by_side_v17.mp4"),
    }
    render_qc: dict[str, Any] = {}
    for key, (src_name, dst_name) in names.items():
        src = (
            existing_render_source(render_dir, final_dir, src_name, dst_name)
            if bool(args.reuse_existing_final_renders)
            else render_dir / src_name
        )
        if src is None:
            raise RuntimeError(f"no existing render available for {final_dir / dst_name}")
        dst = final_dir / dst_name
        if not src.exists():
            raise RuntimeError(f"renderer did not produce {src}")
        write_qc_video(src, dst)
        render_qc[key] = check_video(dst, raw)
    duration_render_qc_pass = all(row["frame_count_match"] for row in render_qc.values())
    solver_report_path = Path(state["solver_report"]) if isinstance(state.get("solver_report"), str) else None
    solver_report = load_json(solver_report_path) if solver_report_path is not None and solver_report_path.exists() else {}
    limits = limitation_fields(state, solver_report, case_manifest)
    sheet = visual_inspection_sheet(
        Path(render_qc["side_by_side"]["path"]),
        case_dir / sheet_filename(str(args.method_name)),
        inspection_frames(state, raw),
    )
    removed_non_qc_visual_artifacts = remove_non_qc_visual_artifacts(
        case_dir,
        render_dir,
        final_dir,
        Path(sheet["path"]),
    )
    report = {
        "case": state["case"],
        "status": "duration_render_qc_pass" if duration_render_qc_pass else "duration_render_qc_failed",
        "artifact_status": "partial",
        "artifact_kind": "duration_qc_render",
        "delivery_role": "qc_only_not_v17_closure",
        "render_qc_scope": "duration_only_not_visual_quality",
        "visible_qc_banner": True,
        "qc_banner_text": list(QC_BANNER_LINES),
        "non_qc_visual_artifacts_removed": removed_non_qc_visual_artifacts,
        "method": args.method_name,
        "clip": str(clip),
        "annotations": state["annotations"],
        "object_mesh_archive": state["object_mesh_archive"],
        "raw_video": raw.__dict__,
        "render_qc": render_qc,
        "frame_count_match": duration_render_qc_pass,
        "solver_status": state.get("solver_status"),
        "solver_artifact_status": state.get("artifact_status"),
        "solver_artifact_kind": state.get("artifact_kind"),
        "solver_delivery_role": state.get("delivery_role"),
        "solver_completeness": state.get("solver_completeness"),
        "solver_report": state.get("solver_report"),
        "v3_solver_complete": bool(state.get("v3_solver_complete")) if "v3_solver_complete" in state else None,
        "duration_render_qc_pass": bool(duration_render_qc_pass),
        "visual_quality_qc_pass": False,
        "stage9_visual_deliverable_ready": False,
        "annotation_ready": bool(solver_report.get("annotation_ready")),
        "deliverable_ready": bool(solver_report.get("deliverable_ready")),
        "accuracy_target_met": bool(solver_report.get("accuracy_target_met")),
        **limits,
        "visual_inspection_sheet": sheet,
    }
    write_json(case_dir / "v17_render_manifest.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.repo_root = Path(args.repo_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [render_case(args, manifest, args.output_root) for manifest in args.case_manifests]
    duration_render_qc_pass = all(bool(row["duration_render_qc_pass"]) for row in reports)
    summary_limits = {
        "multi_object_timeline_ready": bool(all(row["multi_object_timeline_ready"] for row in reports)),
        "object_schema_status": "single_manipulated_object_qc",
        "missing_multi_object_roster_required": bool(any(row["missing_multi_object_roster_required"] for row in reports)),
        "object_geometry_complete": bool(all(row["object_geometry_complete"] for row in reports)),
        "object_pose_requirement_met": bool(all(row["object_pose_requirement_met"] for row in reports)),
        "object_geometry_status": "partial_visible_surface_or_local_patch_qc",
    }
    summary = {
        "status": "duration_render_qc_pass" if duration_render_qc_pass else "duration_render_qc_failed",
        "artifact_status": "partial",
        "artifact_kind": "duration_qc_render_collection",
        "delivery_role": "qc_only_not_v17_closure",
        "render_qc_scope": "duration_only_not_visual_quality",
        "visible_qc_banner": True,
        "qc_banner_text": list(QC_BANNER_LINES),
        "duration_render_qc_status": "pass" if duration_render_qc_pass else "fail",
        "visual_quality_qc_pass": False,
        "stage9_visual_deliverable_ready": False,
        "annotation_ready": bool(all(row["annotation_ready"] for row in reports)),
        "deliverable_ready": bool(all(row["deliverable_ready"] for row in reports)),
        "accuracy_target_met": bool(all(row["accuracy_target_met"] for row in reports)),
        "v3_solver_complete": bool(all(row.get("v3_solver_complete") for row in reports)),
        **summary_limits,
        "method": args.method_name,
        "cases": reports,
    }
    write_json(args.output_root / "v17_render_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--method-name", default="render_v17_full_state")
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_full_state"))
    parser.add_argument("--reuse-existing-final-renders", action="store_true")
    parser.add_argument(
        "--case-manifests",
        type=Path,
        nargs="+",
        default=[
            Path("/data2/ego_annotation_outputs/v17_full_state/trash_1050/v17_full_state_manifest.json"),
            Path("/data2/ego_annotation_outputs/v17_full_state/task5_tomato_960/v17_full_state_manifest.json"),
        ],
    )
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
