#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quote(parts: list[str]) -> str:
    return " ".join("'" + part.replace("'", "'\\''") + "'" for part in parts)


def py(parts: list[str]) -> str:
    return "PYTHONPATH=scripts " + quote(["uv", "run", "python", *parts])


def commands(job: dict, args: argparse.Namespace) -> dict:
    adapt = py(
        [
            "scripts/adapt_sam2_track_to_annotations_v3.py",
            "--base-annotations",
            args.base_annotations,
            "--sam2-track",
            str(Path(job["sam2_output_dir"]) / "sam2_track.json"),
            "--point-prompts",
            job["point_prompts"],
            "--output-json",
            job["adapted_annotations"],
            "--frame-start",
            str(job["frame_start"]),
            "--frame-end",
            str(job["frame_end"]),
            "--remote-output-root",
            str(args.remote_output_root),
            "--local-output-root",
            str(args.local_output_root),
        ]
    )
    mesh = py(
        [
            "scripts/reconstruct_object_mesh_v2.py",
            "--annotations",
            job["adapted_annotations"],
            "--droid-npz",
            args.droid_npz,
            "--droid-reconstruction",
            args.droid_reconstruction,
            "--metric-depth-npz",
            args.metric_depth_npz,
            "--output-dir",
            job["mesh_dir"],
            "--droid-to-meters",
            str(args.droid_to_meters),
            "--depth-source",
            "metric_depth",
            "--mask-stride",
            str(args.mask_stride),
            "--review-stride",
            str(args.review_stride),
        ]
    )
    reliability = py(
        [
            "scripts/diagnose_hand_contact_reliability_v3.py",
            "--annotations",
            job["adapted_annotations"],
            "--metric-depth-npz",
            args.metric_depth_npz,
            "--object-mesh-npz",
            str(Path(job["mesh_dir"]) / "dynamic_object_meshes.npz"),
            "--output-json",
            job["contact_reliability_json"],
            "--frame-start",
            str(job["frame_start"]),
            "--frame-end",
            str(job["frame_end"]),
        ]
    )
    return {"adapt_annotations": adapt, "mesh_metric_depth": mesh, "contact_reliability": reliability}


def run(args: argparse.Namespace) -> dict:
    sam2_manifest = load_json(args.sam2_manifest)
    jobs = sam2_manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"SAM2 manifest has no jobs: {args.sam2_manifest}")
    post_jobs = []
    for job in jobs:
        post_job = dict(job)
        post_job["postprocess_commands"] = commands(post_job, args)
        post_jobs.append(post_job)
    out = {
        "status": "ok",
        "sam2_manifest": str(args.sam2_manifest),
        "base_annotations": str(args.base_annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "droid_npz": str(args.droid_npz),
        "droid_reconstruction": str(args.droid_reconstruction),
        "droid_to_meters": float(args.droid_to_meters),
        "jobs": post_jobs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "jobs": len(post_jobs)}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-manifest", type=Path, required=True)
    parser.add_argument("--base-annotations", required=True)
    parser.add_argument("--metric-depth-npz", required=True)
    parser.add_argument("--droid-npz", required=True)
    parser.add_argument("--droid-reconstruction", required=True)
    parser.add_argument("--droid-to-meters", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mask-stride", type=int, default=8)
    parser.add_argument("--review-stride", type=int, default=30)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
