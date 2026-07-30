#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def visible_prompt_frames(payload: dict) -> list[int]:
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError("point prompt payload lacks point_prompts")
    frames = [
        int(row["frame_idx"])
        for row in rows
        if row.get("target_visible") and row.get("positive_points")
    ]
    if not frames:
        raise RuntimeError(f"no visible prompt frames for {payload.get('track_id')}")
    return sorted(set(frames))


def command(job: dict, repo_dir: Path, script_path: str) -> str:
    args = [
        "bash",
        script_path,
        str(repo_dir),
        job["remote_clip"],
        job["remote_point_prompts"],
        job["remote_sam2_output_dir"],
        job["remote_checkpoint"],
        str(job["frame_start"]),
        str(job["frame_end"]),
    ]
    return " ".join("'" + part.replace("'", "'\\''") + "'" for part in args)


def remote_path(path: Path, local_root: Path, remote_root: Path) -> str:
    rel = path.resolve().relative_to(local_root.resolve())
    return str(remote_root / rel)


def run(args: argparse.Namespace) -> dict:
    point_files = sorted(args.point_root.glob("*/object_point_prompts_vlm.json"))
    if not point_files:
        raise RuntimeError(f"no point prompt files under {args.point_root}")
    jobs = []
    for point_file in point_files:
        payload = load_json(point_file)
        frames = visible_prompt_frames(payload)
        track_id = str(payload["track_id"])
        surface_dir = args.output_root / track_id
        local_sam2_dir = surface_dir / "sam2"
        job = {
            "track_id": track_id,
            "description": str(payload["description"]),
            "clip": str(args.clip),
            "remote_clip": remote_path(args.clip, args.local_data_root, args.remote_data_root),
            "point_prompts": str(point_file),
            "remote_point_prompts": remote_path(point_file, args.local_data_root, args.remote_data_root),
            "prompt_frames": frames,
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "checkpoint": str(args.checkpoint),
            "remote_checkpoint": remote_path(args.checkpoint, args.local_data_root, args.remote_data_root),
            "sam2_output_dir": str(local_sam2_dir),
            "remote_sam2_output_dir": remote_path(local_sam2_dir, args.local_data_root, args.remote_data_root),
            "adapted_annotations": str(surface_dir / "annotations_sam2_vlm_points.json"),
            "metric_depth_dir": str(surface_dir / "metric_depth"),
            "mesh_dir": str(surface_dir / "mesh_metric_depth"),
            "contact_reliability_json": str(surface_dir / "contact_reliability.json"),
        }
        job["remote_command"] = command(job, args.repo_dir, args.remote_script)
        jobs.append(job)
    manifest = {
        "status": "ok",
        "clip": str(args.clip),
        "base_annotations": str(args.base_annotations),
        "point_root": str(args.point_root),
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "local_data_root": str(args.local_data_root),
        "remote_data_root": str(args.remote_data_root),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "jobs": jobs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "jobs": len(jobs), "track_ids": [j["track_id"] for j in jobs]}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-data-root", type=Path, default=Path("/data2"))
    parser.add_argument("--remote-data-root", type=Path, default=Path("/data2"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/repo"))
    parser.add_argument("--remote-script", default="scripts/remote_run_sam2_surface_track.sh")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
