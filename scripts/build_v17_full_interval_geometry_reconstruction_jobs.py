#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from build_v17_geometry_reconstruction_jobs import (
    FALSE_READY,
    existing_path,
    finite_float,
    intrinsics_summary,
    load_json,
    rectify_frame,
    require_dict,
    require_int,
    require_list,
    require_str,
    stable_seed,
    summarize,
    target_intrinsics,
    write_cam_k,
    write_json,
)


STATUS = "v17_full_interval_geometry_reconstruction_jobs_qc"
CLAIM = (
    "This artifact prepares full-active-interval RGBD reconstruction jobs for hidden-topology object "
    "solvers, one per exported object-track dataset, using the same constant-intrinsics rectification "
    "contract as the seed-window jobs. It tests whether full-interval unknown-object reconstruction is "
    "feasible per object; it is a solver input layer, not object geometry reconstruction, and it does "
    "not change V17 readiness."
)


def case_objects(summary: dict[str, Any], case: str) -> list[dict[str, Any]]:
    rows = []
    for i, raw in enumerate(require_list(summary.get("objects"), f"{case} object dataset rows")):
        row = require_dict(raw, f"{case} object dataset rows[{i}]")
        frame_count = require_int(row.get("frame_count"), f"{case} object row {i} frame_count")
        if frame_count <= 0:
            continue
        rows.append(row)
    return rows


def contiguous_segments(frames: list[dict[str, Any]], max_gap: int) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_idx: int | None = None
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        if prev_idx is not None and frame_idx - prev_idx > max_gap:
            segments.append(current)
            current = []
        current.append(frame)
        prev_idx = frame_idx
    if current:
        segments.append(current)
    return segments


def build_job(
    *,
    case: str,
    object_row: dict[str, Any],
    segment_frames: list[dict[str, Any]],
    object_manifest: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_id = require_str(object_row.get("object_id"), "object_id")
    track_id = object_id.split(":", 1)[1] if ":" in object_id else object_id
    frames = segment_frames
    if len(frames) < int(args.min_job_frames):
        raise RuntimeError(
            f"{case} {track_id} has {len(frames)} exported frames, below minimum {args.min_job_frames}"
        )
    if int(args.max_job_frames) > 0 and len(frames) > int(args.max_job_frames):
        pick = np.linspace(0, len(frames) - 1, int(args.max_job_frames)).round().astype(np.int64)
        frames = [frames[int(i)] for i in pick]
        frame_subsampled = True
    else:
        frame_subsampled = False
    first_frame = require_int(frames[0].get("frame_idx"), "first frame_idx")
    last_frame = require_int(frames[-1].get("frame_idx"), "last frame_idx")
    target_k = target_intrinsics(frames)
    job_id = f"{case}_{track_id}_full_{first_frame:06d}_{last_frame:06d}"
    job_dir = output_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    rgb_dir = job_dir / "rgb"
    mask_dir = job_dir / "masks"
    depth_dir = job_dir / "depth"
    rectified_rows: list[dict[str, Any]] = []
    dropped_frames: list[dict[str, Any]] = []
    out_i = 0
    for frame in frames:
        row = rectify_frame(
            frame,
            output_index=out_i,
            target_k=target_k,
            output_rgb=rgb_dir / f"{out_i:06d}.png",
            output_mask=mask_dir / f"{out_i:06d}.png",
            output_depth=depth_dir / f"{out_i:06d}.png",
            max_eval_points=int(args.max_eval_points),
            seed=stable_seed(case, job_id, frame["frame_idx"]),
            raster_scale=int(args.raster_scale),
        )
        frame_residual_p95 = finite_float(
            row["rectification_nearest_3d_residual_m"].get("p95"),
            "frame rectification p95",
        )
        frame_inside = finite_float(row.get("projected_inside_fraction"), "frame projected fraction")
        if (
            frame_residual_p95 <= float(args.max_rectification_residual_p95_m)
            and frame_inside >= float(args.min_projected_inside_fraction)
        ):
            rectified_rows.append(row)
            out_i += 1
        else:
            for path in [
                rgb_dir / f"{out_i:06d}.png",
                mask_dir / f"{out_i:06d}.png",
                depth_dir / f"{out_i:06d}.png",
            ]:
                path.unlink(missing_ok=True)
            dropped_frames.append(
                {
                    "frame_idx": require_int(frame.get("frame_idx"), "frame_idx"),
                    "rectification_nearest_3d_residual_p95_m": frame_residual_p95,
                    "projected_inside_fraction": frame_inside,
                    "reason": "frame_fails_rectification_contract",
                }
            )
    if len(rectified_rows) < int(args.min_job_frames):
        raise RuntimeError(
            f"{case} {track_id} keeps {len(rectified_rows)} rectification-valid frames "
            f"(dropped {len(dropped_frames)}), below minimum {args.min_job_frames}"
        )
    scaled_k = target_k.astype(np.float64).copy()
    scaled_k *= float(args.raster_scale)
    write_cam_k(job_dir / "cam_K.txt", scaled_k)
    residual_p95 = [
        finite_float(row["rectification_nearest_3d_residual_m"].get("p95"), "rectification p95")
        for row in rectified_rows
    ]
    inside_fraction = [
        finite_float(row.get("projected_inside_fraction"), "projected_inside_fraction")
        for row in rectified_rows
    ]
    ray_preserving = bool(
        residual_p95
        and max(residual_p95) <= float(args.max_rectification_residual_p95_m)
        and min(inside_fraction) >= float(args.min_projected_inside_fraction)
    )
    kept_fraction = float(len(rectified_rows) / (len(rectified_rows) + len(dropped_frames)))
    if kept_fraction < float(args.min_kept_frame_fraction):
        ray_preserving = False
    job_status = "ready_for_unknown_object_rgbd_solver" if ray_preserving else "rejected_rectification_residual"
    manifest = {
        "method": "build_v17_full_interval_geometry_reconstruction_jobs",
        "status": job_status,
        "claim": CLAIM,
        "case": case,
        "job_id": job_id,
        "object_id": object_id,
        "track_id": track_id,
        "window_id": f"full_{first_frame:06d}_{last_frame:06d}",
        "solver_backend_contract": "BundleSDF-compatible RGBD folder with constant cam_K.txt; no solver output included",
        "dataset_dir": str(job_dir),
        "cam_k": str(job_dir / "cam_K.txt"),
        "source_object_track_manifest": require_str(
            object_manifest.get("dataset_dir"),
            "object manifest dataset_dir",
        )
        + "/manifest.json",
        "frame_count": len(rectified_rows),
        "dropped_frame_count": len(dropped_frames),
        "kept_frame_fraction": kept_fraction,
        "dropped_frames": dropped_frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frame_subsampled": frame_subsampled,
        "source_exported_frame_count": require_int(object_manifest.get("frame_count"), "object manifest frame_count"),
        "source_rejected_frame_count": require_int(
            object_manifest.get("rejected_frame_count"),
            "object manifest rejected_frame_count",
        ),
        "source_intrinsics": intrinsics_summary(frames),
        "rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in scaled_k.tolist()],
        "base_rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in target_k.tolist()],
        "raster_scale": int(args.raster_scale),
        "frames": rectified_rows,
        "rectification_nearest_3d_residual_p95_m": summarize(residual_p95),
        "projected_inside_fraction": summarize(inside_fraction),
        "readiness_checks": {
            "constant_intrinsics_written": True,
            "source_rays_preserved_by_rectified_depth": ray_preserving,
            "hidden_topology_solver_has_run": False,
            "hidden_topology_reconstructed": False,
            "mesh_projection_qc_passed": False,
        },
        "solver_job_ready": ray_preserving,
        "hidden_topology_reconstructed": False,
        **FALSE_READY,
    }
    write_json(job_dir / "manifest.json", {"frames": rectified_rows})
    write_json(job_dir / "v17_geometry_reconstruction_job.json", manifest)
    return manifest


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.object_track_dataset_root / case / "v17_object_track_dataset_summary.json",
        f"{case} object-track dataset summary",
    )
    summary = require_dict(load_json(summary_path), f"{case} object-track dataset summary")
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for object_row in case_objects(summary, case):
        object_id_filter = require_str(object_row.get("object_id"), "object_id")
        track_filter = object_id_filter.split(":", 1)[1] if ":" in object_id_filter else object_id_filter
        if args.only_tracks and track_filter not in set(args.only_tracks):
            continue
        manifest_path = existing_path(
            Path(require_str(object_row.get("manifest"), "object row manifest path")),
            f"{case} object manifest",
        )
        object_manifest = require_dict(load_json(manifest_path), f"{case} object manifest")
        if bool(args.reports_only):
            object_id = require_str(object_row.get("object_id"), "object_id")
            track_id = object_id.split(":", 1)[1] if ":" in object_id else object_id
            job_paths = sorted((args.output_root / case).glob(f"{case}_{track_id}_full_*/v17_geometry_reconstruction_job.json"))
            if not job_paths:
                skipped.append(
                    {
                        "object_id": object_id,
                        "reason": "no existing full-interval job manifest under reports-only mode",
                        **FALSE_READY,
                    }
                )
                continue
            for job_path in job_paths:
                jobs.append(require_dict(load_json(job_path), f"{case} {track_id} existing job manifest"))
            continue
        all_frames = [
            require_dict(raw, "object-track frame")
            for raw in require_list(object_manifest.get("frames"), "object-track frames")
        ]
        object_id = require_str(object_row.get("object_id"), "object_id")
        track_id = object_id.split(":", 1)[1] if ":" in object_id else object_id
        for stale in sorted((args.output_root / case).glob(f"{case}_{track_id}_full_*")):
            if stale.is_dir():
                shutil.rmtree(stale)
        for segment in contiguous_segments(all_frames, int(args.max_segment_gap_frames)):
            try:
                jobs.append(
                    build_job(
                        case=case,
                        object_row=object_row,
                        segment_frames=segment,
                        object_manifest=object_manifest,
                        output_dir=args.output_root / case,
                        args=args,
                    )
                )
            except RuntimeError as exc:
                skipped.append(
                    {
                        "object_id": require_str(object_row.get("object_id"), "object_id"),
                        "reason": str(exc),
                        **FALSE_READY,
                    }
                )
    ready_jobs = [job for job in jobs if job.get("solver_job_ready") is True]
    report = {
        "method": "build_v17_full_interval_geometry_reconstruction_jobs",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_object_track_dataset_summary": str(summary_path),
        "job_count": len(jobs),
        "solver_job_ready_count": len(ready_jobs),
        "skipped_job_count": len(skipped),
        "jobs": [
            {
                "job_id": require_str(job.get("job_id"), "job_id"),
                "object_id": require_str(job.get("object_id"), "object_id"),
                "track_id": require_str(job.get("track_id"), "track_id"),
                "window_id": require_str(job.get("window_id"), "window_id"),
                "job_path": str(
                    Path(require_str(job.get("dataset_dir"), "dataset_dir"))
                    / "v17_geometry_reconstruction_job.json"
                ),
                "dataset_dir": require_str(job.get("dataset_dir"), "dataset_dir"),
                "frame_count": require_int(job.get("frame_count"), "frame_count"),
                "first_frame": require_int(job.get("first_frame"), "first_frame"),
                "last_frame": require_int(job.get("last_frame"), "last_frame"),
                "frame_subsampled": bool(job.get("frame_subsampled") is True),
                "solver_job_ready": bool(job.get("solver_job_ready") is True),
                "rectification_nearest_3d_residual_p95_m": require_dict(
                    job.get("rectification_nearest_3d_residual_p95_m"),
                    "job rectification residual",
                ),
                "projected_inside_fraction": require_dict(
                    job.get("projected_inside_fraction"),
                    "job projected fraction",
                ),
                **FALSE_READY,
            }
            for job in jobs
        ],
        "skipped_jobs": skipped,
        "hidden_topology_reconstructed_job_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_full_interval_geometry_reconstruction_jobs_report.json", report)
    # evaluator-compatible alias so build_v17_geometry_reconstruction_results can consume this root directly
    write_json(args.output_root / case / "v17_geometry_reconstruction_jobs_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v17_full_interval_geometry_reconstruction_jobs",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "job_count": sum(require_int(case.get("job_count"), "case job_count") for case in cases),
        "solver_job_ready_count": sum(
            require_int(case.get("solver_job_ready_count"), "case ready count") for case in cases
        ),
        "skipped_job_count": sum(require_int(case.get("skipped_job_count"), "case skipped count") for case in cases),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "job_count": require_int(case.get("job_count"), "case job_count"),
                "solver_job_ready_count": require_int(case.get("solver_job_ready_count"), "case ready count"),
                "jobs": [
                    {
                        "job_id": job["job_id"],
                        "frame_count": job["frame_count"],
                        "solver_job_ready": job["solver_job_ready"],
                    }
                    for job in require_list(case.get("jobs"), "case jobs")
                ],
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_full_interval_geometry_reconstruction_jobs_summary.json", summary)
    # evaluator-compatible alias so build_v17_geometry_reconstruction_results can consume this root directly
    write_json(args.output_root / "v17_geometry_reconstruction_jobs_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-track-dataset-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_interval_geometry_reconstruction_jobs"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument(
        "--only-tracks",
        nargs="*",
        default=None,
        help="restrict (re-)export to these track ids; other objects keep existing job dirs untouched",
    )
    parser.add_argument("--min-job-frames", type=int, default=20)
    parser.add_argument(
        "--max-segment-gap-frames",
        type=int,
        default=30,
        help="split an object's exported frames into separate jobs at gaps larger than this",
    )
    parser.add_argument(
        "--max-job-frames",
        type=int,
        default=0,
        help="0 keeps all exported frames; positive values subsample evenly for runtime control",
    )
    parser.add_argument("--max-eval-points", type=int, default=5000)
    parser.add_argument("--max-rectification-residual-p95-m", type=float, default=0.003)
    parser.add_argument("--min-projected-inside-fraction", type=float, default=0.995)
    parser.add_argument(
        "--min-kept-frame-fraction",
        type=float,
        default=0.75,
        help="job is not solver-ready if more than this fraction of exported frames fail rectification",
    )
    parser.add_argument("--raster-scale", type=int, default=2)
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="rebuild case reports and summary from existing per-job manifests without re-rectifying frames",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
