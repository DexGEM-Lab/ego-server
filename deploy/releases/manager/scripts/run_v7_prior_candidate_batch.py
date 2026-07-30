#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
REQUIRED_TARGET_KEYS = (
    "observed_mesh_archive",
    "manifest",
    "annotations",
    "metric_depth_npz",
    "frame_start",
    "frame_end",
    "intrinsics_source",
    "physics_intrinsics_source",
    "baseline_zbuffer_json",
)


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


def require_float(raw: object, key: str) -> float:
    if isinstance(raw, bool):
        raise RuntimeError(f"{key} must be a number")
    value = float(raw)
    if not value > 0.0:
        raise RuntimeError(f"{key} must be positive")
    return value


def require_int(raw: object, key: str) -> int:
    if isinstance(raw, bool):
        raise RuntimeError(f"{key} must be an integer")
    value = int(raw)
    if value <= 0:
        raise RuntimeError(f"{key} must be positive")
    return value


def parse_candidate(raw: str) -> tuple[str, str, Path, str]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--candidate must have format target_id|candidate_name|mesh_path|note")
    target_id, name, mesh, note = [part.strip() for part in parts]
    if not target_id or not name or not mesh:
        raise RuntimeError("--candidate target_id, candidate_name, and mesh_path must be non-empty")
    return target_id, name, Path(mesh), note


def read_candidate_file(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"candidate file does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        try:
            parse_candidate(stripped)
        except RuntimeError as exc:
            raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
        rows.append(stripped)
    return rows


def read_observed_cache_file(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise RuntimeError(f"observed-cache file does not exist: {path}")
    caches = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) not in {2, 3}:
            raise RuntimeError(f"{path}:{line_number}: row must have target_id|report_path or target_id|report_path|video_path")
        target_id, report_path = parts[:2]
        if not target_id or not report_path:
            raise RuntimeError(f"{path}:{line_number}: target_id and report_path must be non-empty")
        if target_id in caches:
            raise RuntimeError(f"{path}:{line_number}: duplicate observed-cache target_id: {target_id}")
        report = require_path(report_path, f"{path}:{line_number}.report")
        video = parts[2] if len(parts) == 3 else ""
        caches[target_id] = {
            "target_id": target_id,
            "report": str(report),
            "video": video,
            "status": "cached",
        }
    return caches


def candidate_rows(args: argparse.Namespace) -> list[str]:
    rows = []
    for path in args.candidate_file:
        rows.extend(read_candidate_file(path))
    rows.extend(args.candidate)
    if not rows:
        raise RuntimeError("provide at least one --candidate or --candidate-file row")
    return rows


def shell_token(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_command(argv: list[str], dry_run: bool) -> None:
    print(" ".join(shell_token(arg) for arg in argv))
    if dry_run:
        return
    subprocess.run(argv, check=True)


def run_observed_target_cache(args: argparse.Namespace, target_id: str, target: dict) -> dict:
    cache_dir = args.output_root / "_observed_target_zbuffer_cache" / target_id
    report_path = cache_dir / "qc_mesh_zbuffer_projection_v3.json"
    video_path = cache_dir / "mesh_zbuffer_projection_qc.mp4"
    argv = [
        sys.executable,
        str(args.scripts_dir / "render_mesh_zbuffer_qc_v3.py"),
        "--mesh-archive",
        str(target["observed_mesh_archive"]),
        "--manifest",
        str(target["manifest"]),
        "--annotations",
        str(target["annotations"]),
        "--metric-depth-npz",
        str(target["metric_depth_npz"]),
        "--intrinsics-source",
        str(target["intrinsics_source"]),
        "--frame-start",
        str(target["frame_start"]),
        "--frame-end",
        str(target["frame_end"]),
        "--max-faces",
        str(args.max_faces),
        "--vertex-splat-radius-px",
        str(args.vertex_splat_radius_px),
        "--output-dir",
        str(cache_dir),
    ]
    run_command(argv, bool(args.dry_run))
    return {
        "target_id": target_id,
        "report": str(report_path),
        "video": str(video_path),
        "status": "dry_run" if args.dry_run else "ok",
    }


def validate_target(target_id: str, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise RuntimeError(f"target {target_id} must be a JSON object")
    missing = [key for key in REQUIRED_TARGET_KEYS if key not in raw]
    if missing:
        raise RuntimeError(f"target {target_id} lacks keys: {', '.join(missing)}")
    target = dict(raw)
    for key in ("observed_mesh_archive", "manifest", "annotations", "metric_depth_npz", "baseline_zbuffer_json"):
        target[key] = require_path(target[key], f"{target_id}.{key}")
    target["frame_start"] = int(target["frame_start"])
    target["frame_end"] = int(target["frame_end"])
    if target["frame_end"] < target["frame_start"]:
        raise RuntimeError(f"target {target_id} has inverted frame range")
    if target["intrinsics_source"] not in {"manifest", "annotation-vggt"}:
        raise RuntimeError(f"target {target_id} has invalid intrinsics_source: {target['intrinsics_source']}")
    if target["physics_intrinsics_source"] not in {"annotation-vggt", "hand", "cli"}:
        raise RuntimeError(f"target {target_id} has invalid physics_intrinsics_source: {target['physics_intrinsics_source']}")
    if "track_qc" in target:
        target["track_qc"] = validate_track_qc(target_id, target["track_qc"])
    return target


def validate_track_qc(target_id: str, raw: object) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError(f"target {target_id}.track_qc must be a JSON object or null")
    required = (
        "pair_factors_json",
        "frame_start",
        "frame_end",
        "min_tracks",
        "min_edges",
        "max_pair_factor_residual_m",
        "max_track_surface_distance_m",
        "max_pair_residual_p95_m",
        "max_correction_displacement_p95_m",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise RuntimeError(f"target {target_id}.track_qc lacks keys: {', '.join(missing)}")
    track_qc = dict(raw)
    track_qc["pair_factors_json"] = require_path(track_qc["pair_factors_json"], f"{target_id}.track_qc.pair_factors_json")
    track_qc["frame_start"] = int(track_qc["frame_start"])
    track_qc["frame_end"] = int(track_qc["frame_end"])
    if track_qc["frame_end"] < track_qc["frame_start"]:
        raise RuntimeError(f"target {target_id}.track_qc has inverted frame range")
    track_qc["min_tracks"] = require_int(track_qc["min_tracks"], f"{target_id}.track_qc.min_tracks")
    track_qc["min_edges"] = require_int(track_qc["min_edges"], f"{target_id}.track_qc.min_edges")
    track_qc["max_pair_factor_residual_m"] = require_float(
        track_qc["max_pair_factor_residual_m"],
        f"{target_id}.track_qc.max_pair_factor_residual_m",
    )
    track_qc["max_track_surface_distance_m"] = require_float(
        track_qc["max_track_surface_distance_m"],
        f"{target_id}.track_qc.max_track_surface_distance_m",
    )
    track_qc["max_pair_residual_p95_m"] = require_float(
        track_qc["max_pair_residual_p95_m"],
        f"{target_id}.track_qc.max_pair_residual_p95_m",
    )
    track_qc["max_correction_displacement_p95_m"] = require_float(
        track_qc["max_correction_displacement_p95_m"],
        f"{target_id}.track_qc.max_correction_displacement_p95_m",
    )
    return track_qc


def candidate_output_dir(root: Path, target_id: str, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    if not safe:
        raise RuntimeError("candidate name produced an empty output directory")
    return root / target_id / safe


def run_replay(
    args: argparse.Namespace,
    target_id: str,
    name: str,
    mesh: Path,
    note: str,
    target: dict,
    observed_cache: dict | None,
) -> dict:
    if not mesh.exists():
        raise RuntimeError(f"candidate mesh does not exist: {mesh}")
    out_dir = candidate_output_dir(args.output_root, target_id, name)
    if args.candidate_kind == "generated_prior":
        argv = [
            sys.executable,
            str(args.scripts_dir / "run_v7_generated_prior_replay_qc.py"),
            "--mesh-prior",
            str(mesh),
            "--observed-mesh-archive",
            str(target["observed_mesh_archive"]),
            "--manifest",
            str(target["manifest"]),
            "--annotations",
            str(target["annotations"]),
            "--metric-depth-npz",
            str(target["metric_depth_npz"]),
            "--frame-start",
            str(target["frame_start"]),
            "--frame-end",
            str(target["frame_end"]),
            "--intrinsics-source",
            str(target["intrinsics_source"]),
            "--output-dir",
            str(out_dir),
            "--samples",
            str(args.samples),
        ]
        report_path = out_dir / "qc_v7_generated_prior_replay.json"
    elif args.candidate_kind == "video_mesh":
        argv = [
            sys.executable,
            str(args.scripts_dir / "run_v7_video_mesh_replay_qc.py"),
            "--video-mesh-archive",
            str(mesh),
            "--manifest",
            str(target["manifest"]),
            "--annotations",
            str(target["annotations"]),
            "--metric-depth-npz",
            str(target["metric_depth_npz"]),
            "--frame-start",
            str(target["frame_start"]),
            "--frame-end",
            str(target["frame_end"]),
            "--intrinsics-source",
            str(target["intrinsics_source"]),
            "--output-dir",
            str(out_dir),
        ]
        report_path = out_dir / "qc_v7_video_mesh_replay.json"
    else:
        raise RuntimeError(f"unsupported candidate_kind: {args.candidate_kind}")
    baseline_zbuffer_json = target["baseline_zbuffer_json"]
    if observed_cache is not None:
        if args.candidate_kind == "generated_prior":
            argv.extend(["--observed-target-zbuffer-report", str(observed_cache["report"])])
        else:
            argv.extend(["--video-mesh-zbuffer-report", str(observed_cache["report"])])
        baseline_zbuffer_json = Path(str(observed_cache["report"]))
    if args.max_faces:
        argv.extend(["--max-faces", str(args.max_faces)])
    if args.vertex_splat_radius_px:
        argv.extend(["--vertex-splat-radius-px", str(args.vertex_splat_radius_px)])
    run_command(argv, bool(args.dry_run))
    result = {
        "target_id": target_id,
        "candidate_name": name,
        "note": note,
        "candidate_kind": args.candidate_kind,
        "mesh_prior": str(mesh) if args.candidate_kind == "generated_prior" else None,
        "video_mesh_archive": str(mesh) if args.candidate_kind == "video_mesh" else None,
        "output_dir": str(out_dir),
        "report": str(report_path),
        "baseline_zbuffer_json": str(baseline_zbuffer_json),
        "replay_controls": {
            "samples": int(args.samples),
            "max_faces": int(args.max_faces),
            "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
            "full_fidelity_zbuffer": bool(args.max_faces == 0),
        },
    }
    if not args.dry_run:
        report = load_json(report_path)
        result["status"] = report.get("status")
        result["annotation_ready"] = bool(report.get("annotation_ready", False))
        result["metrics"] = report.get("metrics")
        result["pass"] = report.get("pass")
    return result


def summary_value(report: dict, section: str, key: str) -> float:
    payload = report.get(section)
    if not isinstance(payload, dict) or key not in payload:
        raise RuntimeError(f"track QC report lacks {section}.{key}")
    return float(payload[key])


def run_track_qc(args: argparse.Namespace, replay_result: dict, target: dict) -> dict | None:
    track_qc = target.get("track_qc")
    if track_qc is None:
        return None
    track_dir = Path(replay_result["output_dir"]) / "track_qc"
    track_report = track_dir / "qc_v7_candidate_track_surface.json"
    replay_status = replay_result.get("status")
    if args.dry_run and replay_status is None:
        return {
            "status": "dry_run_replay_not_executed",
            "reason": "dry-run mode prints the replay command without creating the replay report required by track QC",
            "output_dir": str(track_dir),
            "report": str(track_report),
        }
    if replay_status != "accepted" or not bool(replay_result.get("annotation_ready", False)):
        return {
            "status": "skipped_replay_not_accepted",
            "reason": f"track QC requires accepted replay, got {replay_status}",
            "output_dir": str(track_dir),
            "report": str(track_report),
        }
    replay_controls = replay_result.get("replay_controls", {})
    if not bool(replay_controls.get("full_fidelity_zbuffer", False)):
        return {
            "status": "skipped_diagnostic_replay",
            "reason": "track QC requires full-fidelity z-buffer replay; bounded-face replay is diagnostic only",
            "output_dir": str(track_dir),
            "report": str(track_report),
        }
    replay_report = load_json(Path(replay_result["report"]))
    aligned_mesh_archive = replay_report.get("aligned_mesh_archive")
    if not isinstance(aligned_mesh_archive, str) or not aligned_mesh_archive:
        raise RuntimeError(f"accepted replay report lacks aligned_mesh_archive: {replay_result['report']}")
    argv = [
        sys.executable,
        str(args.scripts_dir / "check_v7_candidate_track_surface_qc.py"),
        "--candidate-mesh-archive",
        aligned_mesh_archive,
        "--pair-factors-json",
        str(track_qc["pair_factors_json"]),
        "--frame-start",
        str(track_qc["frame_start"]),
        "--frame-end",
        str(track_qc["frame_end"]),
        "--max-track-surface-distance-m",
        str(track_qc["max_track_surface_distance_m"]),
        "--max-pair-factor-residual-m",
        str(track_qc["max_pair_factor_residual_m"]),
        "--min-edges",
        str(track_qc["min_edges"]),
        "--min-tracks",
        str(track_qc["min_tracks"]),
        "--max-pair-residual-p95-m",
        str(track_qc["max_pair_residual_p95_m"]),
        "--max-correction-displacement-p95-m",
        str(track_qc["max_correction_displacement_p95_m"]),
        "--output-dir",
        str(track_dir),
    ]
    run_command(argv, bool(args.dry_run))
    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(track_dir),
            "report": str(track_report),
        }
    report = load_json(track_report)
    accepted_edge_count = int(report.get("accepted_edge_count", 0))
    track_count = int(report.get("track_count", 0))
    pair_p95 = summary_value(report, "pair_residual_m", "p95")
    correction_p95 = summary_value(report, "correction_displacement_m", "p95")
    checks = report.get("pass")
    if not isinstance(checks, dict):
        raise RuntimeError(f"track QC report lacks pass checks: {track_report}")
    accepted = report.get("status") == "accepted" and bool(report.get("annotation_ready", False))
    return {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "output_dir": str(track_dir),
        "report": str(track_report),
        "metrics": {
            "track_count": track_count,
            "accepted_edge_count": accepted_edge_count,
            "pair_residual_p95_m": pair_p95,
            "correction_displacement_p95_m": correction_p95,
        },
        "thresholds": {
            "min_tracks": int(track_qc["min_tracks"]),
            "min_edges": int(track_qc["min_edges"]),
            "max_pair_factor_residual_m": float(track_qc["max_pair_factor_residual_m"]),
            "max_track_surface_distance_m": float(track_qc["max_track_surface_distance_m"]),
            "max_pair_residual_p95_m": float(track_qc["max_pair_residual_p95_m"]),
            "max_correction_displacement_p95_m": float(track_qc["max_correction_displacement_p95_m"]),
        },
        "pass": checks,
    }


def run_physics(args: argparse.Namespace, replay_result: dict, target: dict, track_qc: dict | None) -> dict | None:
    if not args.run_physics:
        return None
    physics_dir = Path(replay_result["output_dir"]) / "physics_qc"
    physics_report = physics_dir / "qc_v7_candidate_physics.json"
    replay_status = replay_result.get("status")
    if args.dry_run and replay_status is None:
        return {
            "status": "dry_run_replay_not_executed",
            "reason": "dry-run mode prints the replay command without creating the replay report required by physics QC",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    if replay_status != "accepted":
        return {
            "status": "skipped_replay_not_accepted",
            "reason": f"physics QC requires accepted replay, got {replay_status}",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    if target.get("track_qc") is not None and (
        track_qc is None or track_qc.get("status") != "accepted" or not bool(track_qc.get("annotation_ready", False))
    ):
        track_status = None if track_qc is None else track_qc.get("status")
        return {
            "status": "skipped_track_not_accepted",
            "reason": f"physics QC requires accepted track QC for this target, got {track_status}",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    replay_controls = replay_result.get("replay_controls", {})
    if not bool(replay_controls.get("full_fidelity_zbuffer", False)):
        return {
            "status": "skipped_diagnostic_replay",
            "reason": "physics QC requires full-fidelity z-buffer replay; bounded-face replay is diagnostic only",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    argv = [
        sys.executable,
        str(args.scripts_dir / "run_v7_candidate_physics_qc.py"),
        "--replay-report",
        str(replay_result["report"]),
        "--annotations",
        str(target["annotations"]),
        "--manifest",
        str(target["manifest"]),
        "--metric-depth-npz",
        str(target["metric_depth_npz"]),
        "--intrinsics-source",
        str(target["physics_intrinsics_source"]),
        "--output-dir",
        str(physics_dir),
        "--output-json",
        str(physics_report),
        "--selected-contact-sdf-pitch-m",
        str(args.selected_contact_sdf_pitch_m),
        "--full-window-sdf-pitch-m",
        str(args.full_window_sdf_pitch_m),
        "--max-selected-contact-abs-sdf-p95-m",
        str(args.max_selected_contact_abs_sdf_p95_m),
        "--min-selected-contact-near-surface-fraction",
        str(args.min_selected_contact_near_surface_fraction),
        "--max-selected-contact-penetration-fraction",
        str(args.max_selected_contact_penetration_fraction),
        "--max-full-hand-penetration-fraction",
        str(args.max_full_hand_penetration_fraction),
    ]
    try:
        run_command(argv, bool(args.dry_run))
    except subprocess.CalledProcessError as exc:
        if not physics_report.exists():
            raise
        physics = load_json(physics_report)
        return {
            "status": physics.get("status", "error"),
            "annotation_ready": bool(physics.get("annotation_ready", False)),
            "output_dir": str(physics_dir),
            "report": str(physics_report),
            "metrics": physics.get("metrics"),
            "pass": physics.get("pass"),
            "error": str(exc),
        }
    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(physics_dir),
            "report": str(physics_report),
        }
    physics = load_json(physics_report)
    return {
        "status": physics.get("status"),
        "annotation_ready": bool(physics.get("annotation_ready", False)),
        "output_dir": str(physics_dir),
        "report": str(physics_report),
        "metrics": physics.get("metrics"),
        "pass": physics.get("pass"),
    }


def run_deliverables(args: argparse.Namespace, replay_result: dict, physics: dict | None, target: dict) -> dict | None:
    if not args.render_deliverables:
        return None
    render_dir = Path(replay_result["output_dir"]) / "deliverables"
    render_report = render_dir / "v7_candidate_deliverables_manifest.json"
    if physics is None:
        return {
            "status": "skipped_physics_not_requested",
            "reason": "delivery rendering requires --run-physics and accepted physics QC",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    if physics.get("status") != "accepted" or not bool(physics.get("annotation_ready", False)):
        return {
            "status": "skipped_physics_not_accepted",
            "reason": f"delivery rendering requires accepted physics QC, got {physics.get('status')}",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    argv = [
        sys.executable,
        str(args.scripts_dir / "render_v7_candidate_deliverables.py"),
        "--replay-report",
        str(replay_result["report"]),
        "--physics-report",
        str(physics["report"]),
        "--manifest",
        str(target["manifest"]),
        "--annotations",
        str(load_json(Path(physics["report"])).get("annotations") or target["annotations"]),
        "--output-dir",
        str(render_dir),
        "--output-fps",
        str(args.render_fps),
        "--caption-prefix",
        "V7 mesh-backed reconstruction",
    ]
    run_command(argv, bool(args.dry_run))
    if args.dry_run:
        return {
            "status": "dry_run",
            "output_dir": str(render_dir),
            "report": str(render_report),
        }
    rendered = load_json(render_report)
    return {
        "status": rendered.get("status"),
        "output_dir": str(render_dir),
        "report": str(render_report),
        "videos": rendered.get("videos"),
        "structural_qc": rendered.get("structural_qc"),
    }


def fmt_metric(value: object, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value) * float(scale)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_batch_summary(args: argparse.Namespace, report: dict) -> None:
    if args.dry_run:
        return
    rows = []
    delivered = 0
    physics_rejected = 0
    for result in report["candidates"]:
        track = result.get("track_qc") or {}
        physics = result.get("physics_qc") or {}
        deliverables = result.get("deliverables") or {}
        full_delivery = (
            result.get("status") == "accepted"
            and track.get("status") in {None, "accepted"}
            and physics.get("status") == "accepted"
            and deliverables.get("status") == "ok"
        )
        if full_delivery:
            delivered += 1
        if physics.get("status") == "rejected":
            physics_rejected += 1
        metrics = result.get("metrics") or {}
        track_metrics = track.get("metrics") or {}
        physics_metrics = physics.get("metrics") or {}
        rows.append(
            [
                result["target_id"],
                result["candidate_name"],
                result.get("candidate_kind", ""),
                "yes" if full_delivery else "no",
                str(result.get("status", "")),
                str(track.get("status", "")),
                str(physics.get("status", "")),
                str(deliverables.get("status", "")),
                fmt_metric(metrics.get("silhouette_iou_median"), digits=3),
                fmt_metric(metrics.get("zbuffer_abs_p95_median_m"), scale=1000.0, digits=2),
                fmt_metric(track_metrics.get("pair_residual_p95_m"), scale=1000.0, digits=2),
                fmt_metric(physics_metrics.get("reliable_temporal_contact_rows"), digits=0),
                fmt_metric(physics_metrics.get("selected_contact_abs_sdf_p95_m"), scale=1000.0, digits=2),
                fmt_metric(physics_metrics.get("full_window_hand_penetration_fraction"), scale=100.0, digits=2),
            ]
        )
    lines = [
        "# V7 Candidate Batch Summary",
        "",
        f"Batch root: `{args.output_root}`",
        "",
        f"Full-delivery rows: {delivered}/{len(report['candidates'])}",
        f"Physics-rejected rows: {physics_rejected}/{len(report['candidates'])}",
        "",
        "| Target | Candidate | Source | Full delivery | Replay | Track | Physics | Deliverables | Replay IoU | Replay depth p95 mm | Track p95 mm | Contact rows | Contact SDF p95 mm | Full-hand penetration % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    summary_md = args.output_root / "qc_v7_prior_candidate_batch_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matrix(args: argparse.Namespace, results: list[dict]) -> None:
    if args.dry_run:
        return
    entries = []
    for result in results:
        entries.extend(
            [
                "--entry",
                "|".join(
                    [
                        f"{result['target_id']}_{result['candidate_name']}",
                        result["baseline_zbuffer_json"],
                        result["report"],
                        result["note"],
                    ]
                ),
            ]
        )
    if not entries:
        return
    argv = [
        sys.executable,
        str(args.scripts_dir / "summarize_v7_prior_replay_matrix.py"),
        *entries,
        "--output-json",
        str(args.output_root / "qc_v7_prior_candidate_batch_matrix.json"),
        "--output-md",
        str(args.output_root / "qc_v7_prior_candidate_batch_matrix.md"),
    ]
    run_command(argv, False)


def write_visual_qc(args: argparse.Namespace) -> None:
    if args.dry_run or args.skip_visual_qc:
        return
    argv = [
        sys.executable,
        str(args.scripts_dir / "render_v7_prior_batch_visual_qc.py"),
        "--batch-json",
        str(args.output_root / "qc_v7_prior_candidate_batch.json"),
        "--output-png",
        str(args.output_root / "qc_v7_prior_candidate_batch_visual_sheet.png"),
        "--output-json",
        str(args.output_root / "qc_v7_prior_candidate_batch_visual_sheet.json"),
        "--tile-width",
        str(args.visual_qc_tile_width),
        "--tile-height",
        str(args.visual_qc_tile_height),
        "--label-height",
        str(args.visual_qc_label_height),
        "--label-chars",
        str(args.visual_qc_label_chars),
        "--label-scale",
        str(args.visual_qc_label_scale),
    ]
    run_command(argv, False)


def run(args: argparse.Namespace) -> dict:
    targets_payload = load_json(args.targets_json)
    targets = {target_id: validate_target(target_id, raw) for target_id, raw in targets_payload.items()}
    raw_candidates = candidate_rows(args)
    output_dirs = {}
    for raw in raw_candidates:
        target_id, name, mesh, _note = parse_candidate(raw)
        if target_id not in targets:
            raise RuntimeError(f"candidate {name} references unknown target_id: {target_id}")
        out_dir = candidate_output_dir(args.output_root, target_id, name)
        previous = output_dirs.get(out_dir)
        if previous is not None:
            raise RuntimeError(
                "multiple V7 candidates resolve to the same output directory: "
                f"{out_dir} for {previous} and {target_id}|{name}|{mesh}"
            )
        output_dirs[out_dir] = f"{target_id}|{name}|{mesh}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    target_ids = sorted({parse_candidate(raw)[0] for raw in raw_candidates})
    observed_caches = {}
    for path in args.observed_cache_file:
        file_caches = read_observed_cache_file(path)
        for target_id, cache in file_caches.items():
            if target_id in observed_caches:
                raise RuntimeError(f"duplicate observed-cache target_id across files: {target_id}")
            observed_caches[target_id] = cache
    if not args.dry_run:
        missing_cache_targets = [target_id for target_id in target_ids if target_id not in observed_caches]
        for target_id in missing_cache_targets:
            observed_caches[target_id] = run_observed_target_cache(args, target_id, targets[target_id])
    results = []
    for raw in raw_candidates:
        target_id, name, mesh, note = parse_candidate(raw)
        target = targets[target_id]
        result = run_replay(args, target_id, name, mesh, note, target, observed_caches.get(target_id))
        track = run_track_qc(args, result, target)
        if track is not None:
            result["track_qc"] = track
        physics = run_physics(args, result, target, track)
        if physics is not None:
            result["physics_qc"] = physics
        deliverables = run_deliverables(args, result, physics, target)
        if deliverables is not None:
            result["deliverables"] = deliverables
        results.append(result)
    report = {
        "status": "dry_run" if args.dry_run else "ok",
        "method": "run_v7_prior_candidate_batch",
        "targets_json": str(args.targets_json),
        "candidate_files": [str(path) for path in args.candidate_file],
        "candidate_kind": args.candidate_kind,
        "output_root": str(args.output_root),
        "replay_controls": {
            "samples": int(args.samples),
            "max_faces": int(args.max_faces),
            "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
            "full_fidelity_zbuffer": bool(args.max_faces == 0),
        },
        "track_qc_enabled": any(target.get("track_qc") is not None for target in targets.values()),
        "physics_enabled": bool(args.run_physics),
        "deliverable_rendering_enabled": bool(args.render_deliverables),
        "observed_target_zbuffer_cache": observed_caches,
        "candidates": results,
    }
    report_path = args.output_root / "qc_v7_prior_candidate_batch.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_batch_summary(args, report)
    write_matrix(args, results)
    write_visual_qc(args)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-json", type=Path, default=REPO_DIR / "configs" / "v7_prior_replay_targets.json")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-file", type=Path, action="append", default=[])
    parser.add_argument("--candidate-kind", choices=("generated_prior", "video_mesh"), default="generated_prior")
    parser.add_argument("--observed-cache-file", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--run-physics", action="store_true")
    parser.add_argument("--render-deliverables", action="store_true")
    parser.add_argument("--render-fps", type=float, default=6.0)
    parser.add_argument("--skip-visual-qc", action="store_true")
    parser.add_argument("--visual-qc-tile-width", type=int, default=640)
    parser.add_argument("--visual-qc-tile-height", type=int, default=360)
    parser.add_argument("--visual-qc-label-height", type=int, default=145)
    parser.add_argument("--visual-qc-label-chars", type=int, default=92)
    parser.add_argument("--visual-qc-label-scale", type=float, default=0.60)
    parser.add_argument("--sdf-pitch-m", type=float, default=None)
    parser.add_argument("--selected-contact-sdf-pitch-m", type=float, default=0.001)
    parser.add_argument("--full-window-sdf-pitch-m", type=float, default=0.003)
    parser.add_argument("--max-selected-contact-abs-sdf-p95-m", type=float, default=0.006)
    parser.add_argument("--min-selected-contact-near-surface-fraction", type=float, default=0.75)
    parser.add_argument("--max-selected-contact-penetration-fraction", type=float, default=0.10)
    parser.add_argument("--max-full-hand-penetration-fraction", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sdf_pitch_m is not None:
        args.selected_contact_sdf_pitch_m = float(args.sdf_pitch_m)
        args.full_window_sdf_pitch_m = float(args.sdf_pitch_m)
    return args


if __name__ == "__main__":
    run(parse_args())
