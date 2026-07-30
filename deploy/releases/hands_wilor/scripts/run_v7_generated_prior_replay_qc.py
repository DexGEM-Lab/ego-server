#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def run_zbuffer(args: argparse.Namespace, mesh_archive: Path, output_dir: Path) -> dict:
    run_command(
        [
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
            "--vertex-splat-radius-px",
            str(args.vertex_splat_radius_px),
            "--output-dir",
            str(output_dir),
        ]
    )
    return load_json(output_dir / "qc_mesh_zbuffer_projection_v3.json")


def same_path(actual: object, expected: Path, key: str) -> None:
    if not isinstance(actual, str) or not actual:
        raise RuntimeError(f"observed target cache lacks {key}")
    actual_path = Path(actual)
    if not actual_path.exists():
        raise RuntimeError(f"observed target cache {key} does not exist: {actual_path}")
    if actual_path.resolve() != expected.resolve():
        raise RuntimeError(f"observed target cache {key} mismatch: {actual_path} != {expected}")


def manifest_frame_indices(path: Path, frame_start: int, frame_end: int) -> list[int]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    out = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        if int(frame_start) <= idx <= int(frame_end):
            out.append(idx)
    if not out:
        raise RuntimeError(f"{path} has no frames in range {frame_start}-{frame_end}")
    return out


def validate_observed_zbuffer_cache(args: argparse.Namespace, path: Path) -> tuple[dict, Path, Path]:
    if not path.exists():
        raise RuntimeError(f"observed target z-buffer report does not exist: {path}")
    report = load_json(path)
    if report.get("status") != "ok":
        raise RuntimeError(f"observed target z-buffer report is not ok: {path}")
    if report.get("method") != "mesh_zbuffer_projection_qc_v3":
        raise RuntimeError(f"observed target z-buffer report has wrong method: {path}")
    same_path(report.get("mesh_archive"), args.observed_mesh_archive, "mesh_archive")
    same_path(report.get("manifest"), args.manifest, "manifest")
    same_path(report.get("annotations"), args.annotations, "annotations")
    same_path(report.get("metric_depth_npz"), args.metric_depth_npz, "metric_depth_npz")
    if str(report.get("intrinsics_source")) != str(args.intrinsics_source):
        raise RuntimeError("observed target z-buffer report intrinsics_source mismatch")
    if int(report.get("vertex_splat_radius_px", -1)) != int(args.vertex_splat_radius_px):
        raise RuntimeError("observed target z-buffer report vertex_splat_radius_px mismatch")
    if "full_fidelity_zbuffer" not in report or "max_faces" not in report:
        raise RuntimeError("observed target z-buffer report lacks full-fidelity render contract")
    expected_full_fidelity = bool(int(args.max_faces) == 0)
    if bool(report["full_fidelity_zbuffer"]) != expected_full_fidelity:
        raise RuntimeError("observed target z-buffer report full_fidelity_zbuffer mismatch")
    expected_max_faces = None if int(args.max_faces) == 0 else int(args.max_faces)
    actual_max_faces = report.get("max_faces")
    if actual_max_faces is None:
        normalized_actual_max_faces = None
    else:
        normalized_actual_max_faces = int(actual_max_faces)
    if normalized_actual_max_faces != expected_max_faces:
        raise RuntimeError("observed target z-buffer report max_faces mismatch")
    expected_frames = manifest_frame_indices(args.manifest, int(args.frame_start), int(args.frame_end))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("observed target z-buffer report lacks rows")
    actual_frames = [int(row["frame_idx"]) for row in rows]
    if actual_frames != expected_frames:
        raise RuntimeError(f"observed target z-buffer report frame rows mismatch: {actual_frames} != {expected_frames}")
    if int(report.get("frames", -1)) != len(expected_frames):
        raise RuntimeError("observed target z-buffer report frame count mismatch")
    video_raw = report.get("video")
    if not isinstance(video_raw, str) or not video_raw:
        raise RuntimeError("observed target z-buffer report lacks video")
    video = Path(video_raw)
    if not video.exists():
        raise RuntimeError(f"observed target z-buffer video does not exist: {video}")
    return report, path, video


def write_report(path: Path, report: dict) -> dict:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_archive = args.output_dir / "aligned_prior_meshes_world.npz"
    align_report = args.output_dir / "qc_aligned_mesh_prior_v7.json"
    observed_zbuffer_dir = args.output_dir / "observed_target_zbuffer_qc"
    zbuffer_dir = args.output_dir / "zbuffer_qc"
    if args.observed_target_zbuffer_report is None:
        observed_zbuffer = run_zbuffer(args, args.observed_mesh_archive, observed_zbuffer_dir)
        observed_zbuffer_report_path = observed_zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"
        observed_zbuffer_video_path = observed_zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"
        observed_target_cache = {"used": False}
    else:
        observed_zbuffer, observed_zbuffer_report_path, observed_zbuffer_video_path = validate_observed_zbuffer_cache(
            args,
            args.observed_target_zbuffer_report,
        )
        observed_target_cache = {
            "used": True,
            "report": str(observed_zbuffer_report_path),
            "video": str(observed_zbuffer_video_path),
        }
    observed_iou = metric(observed_zbuffer, "silhouette_mask_iou", "median")
    observed_visible_inside = metric(observed_zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    observed_zbuffer_abs_p95 = metric(observed_zbuffer, "zbuffer_depth_abs_p95_m", "median")
    observed_target_pass = {
        "observed_silhouette_iou_median": bool(observed_iou >= float(args.min_observed_iou_median)),
        "observed_visible_inside_median": bool(observed_visible_inside >= float(args.min_observed_visible_inside_median)),
        "observed_zbuffer_abs_p95_median": bool(observed_zbuffer_abs_p95 <= float(args.max_observed_zbuffer_abs_p95_median_m)),
    }
    replay_controls = {
        "samples": int(args.samples),
        "max_faces": int(args.max_faces),
        "vertex_splat_radius_px": int(args.vertex_splat_radius_px),
        "full_fidelity_zbuffer": bool(args.max_faces == 0),
    }
    mesh_source = args.mesh_prior if args.mesh_prior is not None else args.prealigned_mesh_archive
    if mesh_source is None:
        raise RuntimeError("either --mesh-prior or --prealigned-mesh-archive must be supplied")
    if (args.prealigned_mesh_archive is None) != (args.prealigned_report is None):
        raise RuntimeError("--prealigned-mesh-archive and --prealigned-report must be supplied together")
    if args.mesh_prior is not None and args.prealigned_mesh_archive is not None:
        raise RuntimeError("--mesh-prior and --prealigned-mesh-archive are mutually exclusive")

    if not all(observed_target_pass.values()):
        delivery_pass_keys = [
            "visible_surface_coverage_p95",
            "silhouette_iou_median",
            "visible_inside_median",
            "zbuffer_abs_p95_median",
        ]
        return write_report(
            args.output_dir / "qc_v7_generated_prior_replay.json",
            {
                "status": "invalid_observed_target",
                "annotation_ready": False,
                "method": "run_v7_generated_prior_replay_qc",
                "claim_tested": "a generated object mesh prior can be accepted as object geometry only after its observed target replay is internally consistent and the prior replay passes",
                "mesh_prior": str(mesh_source),
                "observed_mesh_archive": str(args.observed_mesh_archive),
                "observed_target_zbuffer_report": str(observed_zbuffer_report_path),
                "observed_target_zbuffer_video": str(observed_zbuffer_video_path),
                "observed_target_zbuffer_cache": observed_target_cache,
                "frame_start": int(args.frame_start),
                "frame_end": int(args.frame_end),
                "replay_controls": replay_controls,
                "metrics": {
                    "observed_target_silhouette_iou_median": observed_iou,
                    "observed_target_visible_inside_median": observed_visible_inside,
                    "observed_target_zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
                    "alignment_bidirectional_p95_m": None,
                    "visible_surface_coverage_p95_m": None,
                    "hidden_surface_conflict_p95_m": None,
                    "silhouette_iou_median": None,
                    "visible_inside_median": None,
                    "zbuffer_abs_p95_median_m": None,
                },
                "observed_target_metrics": {
                    "silhouette_iou_median": observed_iou,
                    "visible_inside_median": observed_visible_inside,
                    "zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
                },
                "thresholds": {
                    "min_observed_iou_median": float(args.min_observed_iou_median),
                    "min_observed_visible_inside_median": float(args.min_observed_visible_inside_median),
                    "max_observed_zbuffer_abs_p95_median_m": float(args.max_observed_zbuffer_abs_p95_median_m),
                    "max_alignment_p95_m": float(args.max_alignment_p95_m),
                    "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
                    "min_iou_median": float(args.min_iou_median),
                    "min_visible_inside_median": float(args.min_visible_inside_median),
                    "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
                },
                "pass": {
                    "strict_full_surface_alignment_p95": None,
                    "visible_surface_coverage_p95": None,
                    "silhouette_iou_median": None,
                    "visible_inside_median": None,
                    "zbuffer_abs_p95_median": None,
                },
                "reason": "observed target mesh does not replay against the supplied mask/depth/camera contract, so prior acceptance would be causally uninterpretable",
                "observed_target_pass": observed_target_pass,
                "invalid_observed_target_keys": [key for key, value in observed_target_pass.items() if not value],
                "delivery_pass_keys": delivery_pass_keys,
                "not_evaluated_delivery_keys": delivery_pass_keys,
                "rejection_stage": "observed_target_replay",
            },
        )
    if args.prealigned_mesh_archive is None:
        run_command(
            [
                sys.executable,
                str(args.scripts_dir / "archive_aligned_mesh_prior_v7.py"),
                "--mesh-prior",
                str(args.mesh_prior),
                "--observed-mesh-archive",
                str(args.observed_mesh_archive),
                "--frame-start",
                str(args.frame_start),
                "--frame-end",
                str(args.frame_end),
                "--output-mesh-archive",
                str(aligned_archive),
                "--output-json",
                str(align_report),
                "--samples",
                str(args.samples),
                "--max-bidirectional-p95-m",
                str(args.max_alignment_p95_m),
                "--max-visible-surface-p95-m",
                str(args.max_visible_surface_p95_m),
            ]
        )
    else:
        if not args.prealigned_mesh_archive.exists():
            raise RuntimeError(f"prealigned mesh archive does not exist: {args.prealigned_mesh_archive}")
        if not args.prealigned_report.exists():
            raise RuntimeError(f"prealigned report does not exist: {args.prealigned_report}")
        aligned_archive = args.prealigned_mesh_archive
        align_report = args.prealigned_report
    align = load_json(align_report)
    alignment_p95 = metric(align, "alignment_bidirectional_p95_m", "p95")
    visible_surface_p95 = metric(align, "visible_surface_coverage_p95_m", "p95")
    hidden_surface_p95 = metric(align, "hidden_surface_conflict_p95_m", "p95")
    pass_rows = {
        "strict_full_surface_alignment_p95": bool(alignment_p95 <= float(args.max_alignment_p95_m)),
        "visible_surface_coverage_p95": bool(visible_surface_p95 <= float(args.max_visible_surface_p95_m)),
    }
    delivery_pass_keys = [
        "visible_surface_coverage_p95",
        "silhouette_iou_median",
        "visible_inside_median",
        "zbuffer_abs_p95_median",
    ]
    not_evaluated_delivery_keys = []
    if not pass_rows["visible_surface_coverage_p95"]:
        not_evaluated_delivery_keys = [
            "silhouette_iou_median",
            "visible_inside_median",
            "zbuffer_abs_p95_median",
        ]
        pass_rows.update({key: None for key in not_evaluated_delivery_keys})
        report = {
            "status": "rejected",
            "annotation_ready": False,
            "method": "run_v7_generated_prior_replay_qc",
            "claim_tested": "a generated object mesh prior can be accepted as object geometry only after its visible surface covers measured geometry and image-depth replay passes",
            "mesh_prior": str(mesh_source),
            "observed_mesh_archive": str(args.observed_mesh_archive),
            "observed_target_zbuffer_report": str(observed_zbuffer_report_path),
            "observed_target_zbuffer_video": str(observed_zbuffer_video_path),
            "observed_target_zbuffer_cache": observed_target_cache,
            "aligned_mesh_archive": str(aligned_archive),
            "alignment_report": str(align_report),
            "zbuffer_report": None,
            "zbuffer_video": None,
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "replay_controls": replay_controls,
            "metrics": {
                "observed_target_visible_inside_median": observed_visible_inside,
                "observed_target_silhouette_iou_median": observed_iou,
                "observed_target_zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
                "alignment_bidirectional_p95_m": alignment_p95,
                "visible_surface_coverage_p95_m": visible_surface_p95,
                "hidden_surface_conflict_p95_m": hidden_surface_p95,
                "silhouette_iou_median": None,
                "visible_inside_median": None,
                "zbuffer_abs_p95_median_m": None,
            },
            "thresholds": {
                "min_observed_visible_inside_median": float(args.min_observed_visible_inside_median),
                "min_observed_iou_median": float(args.min_observed_iou_median),
                "max_observed_zbuffer_abs_p95_median_m": float(args.max_observed_zbuffer_abs_p95_median_m),
                "max_alignment_p95_m": float(args.max_alignment_p95_m),
                "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
                "min_iou_median": float(args.min_iou_median),
                "min_visible_inside_median": float(args.min_visible_inside_median),
                "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
            },
            "pass": pass_rows,
            "observed_target_pass": observed_target_pass,
            "strict_full_surface_alignment_is_diagnostic": True,
            "bounded_zbuffer_is_diagnostic": bool(args.max_faces != 0),
            "delivery_pass_keys": delivery_pass_keys,
            "not_evaluated_delivery_keys": not_evaluated_delivery_keys,
            "rejection_stage": "visible_surface_alignment",
            "reason": (
                "visible-surface coverage failed before image-depth replay; "
                "z-buffer rendering cannot make the generated mesh satisfy the visible-surface delivery factor"
            ),
            "next_required_if_accepted": [
                "mesh-surface contact recomputation",
                "selected-contact SDF",
                "full-hand SDF",
                "stakeholder render inspection",
            ],
        }
        return write_report(args.output_dir / "qc_v7_generated_prior_replay.json", report)
    zbuffer = run_zbuffer(args, aligned_archive, zbuffer_dir)
    iou_median = metric(zbuffer, "silhouette_mask_iou", "median")
    visible_inside_median = metric(zbuffer, "visible_silhouette_inside_mask_fraction", "median")
    zbuffer_abs_p95_median = metric(zbuffer, "zbuffer_depth_abs_p95_m", "median")
    pass_rows.update(
        {
            "silhouette_iou_median": bool(iou_median >= float(args.min_iou_median)),
            "visible_inside_median": bool(visible_inside_median >= float(args.min_visible_inside_median)),
            "zbuffer_abs_p95_median": bool(zbuffer_abs_p95_median <= float(args.max_zbuffer_abs_p95_median_m)),
        }
    )
    accepted = all(pass_rows[key] for key in delivery_pass_keys)
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_generated_prior_replay_qc",
        "claim_tested": "a generated object mesh prior can be accepted as object geometry only after its visible surface covers measured geometry and image-depth replay passes",
        "mesh_prior": str(mesh_source),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "observed_target_zbuffer_report": str(observed_zbuffer_report_path),
        "observed_target_zbuffer_video": str(observed_zbuffer_video_path),
        "observed_target_zbuffer_cache": observed_target_cache,
        "aligned_mesh_archive": str(aligned_archive),
        "alignment_report": str(align_report),
        "zbuffer_report": str(zbuffer_dir / "qc_mesh_zbuffer_projection_v3.json"),
        "zbuffer_video": str(zbuffer_dir / "mesh_zbuffer_projection_qc.mp4"),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "replay_controls": replay_controls,
        "metrics": {
            "observed_target_visible_inside_median": observed_visible_inside,
            "observed_target_silhouette_iou_median": observed_iou,
            "observed_target_zbuffer_abs_p95_median_m": observed_zbuffer_abs_p95,
            "alignment_bidirectional_p95_m": alignment_p95,
            "visible_surface_coverage_p95_m": visible_surface_p95,
            "hidden_surface_conflict_p95_m": hidden_surface_p95,
            "silhouette_iou_median": iou_median,
            "visible_inside_median": visible_inside_median,
            "zbuffer_abs_p95_median_m": zbuffer_abs_p95_median,
        },
        "thresholds": {
            "min_observed_visible_inside_median": float(args.min_observed_visible_inside_median),
            "min_observed_iou_median": float(args.min_observed_iou_median),
            "max_observed_zbuffer_abs_p95_median_m": float(args.max_observed_zbuffer_abs_p95_median_m),
            "max_alignment_p95_m": float(args.max_alignment_p95_m),
            "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
            "min_iou_median": float(args.min_iou_median),
            "min_visible_inside_median": float(args.min_visible_inside_median),
            "max_zbuffer_abs_p95_median_m": float(args.max_zbuffer_abs_p95_median_m),
        },
        "pass": pass_rows,
        "observed_target_pass": observed_target_pass,
        "strict_full_surface_alignment_is_diagnostic": True,
        "bounded_zbuffer_is_diagnostic": bool(args.max_faces != 0),
        "delivery_pass_keys": delivery_pass_keys,
        "not_evaluated_delivery_keys": not_evaluated_delivery_keys,
        "next_required_if_accepted": [
            "mesh-surface contact recomputation",
            "selected-contact SDF",
            "full-hand SDF",
            "stakeholder render inspection",
        ],
    }
    return write_report(args.output_dir / "qc_v7_generated_prior_replay.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path)
    parser.add_argument("--prealigned-mesh-archive", type=Path)
    parser.add_argument("--prealigned-report", type=Path)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--observed-target-zbuffer-report", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="annotation-vggt")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--vertex-splat-radius-px", type=int, default=0)
    parser.add_argument("--min-observed-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--min-observed-iou-median", type=float, default=0.900)
    parser.add_argument("--max-observed-zbuffer-abs-p95-median-m", type=float, default=0.010)
    parser.add_argument("--max-alignment-p95-m", type=float, default=0.010)
    parser.add_argument("--max-visible-surface-p95-m", type=float, default=0.010)
    parser.add_argument("--min-iou-median", type=float, default=0.900)
    parser.add_argument("--min-visible-inside-median", type=float, default=0.900)
    parser.add_argument("--max-zbuffer-abs-p95-median-m", type=float, default=0.010)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
