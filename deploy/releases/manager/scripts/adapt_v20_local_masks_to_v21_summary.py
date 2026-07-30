#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    report = load_json(args.local_mask_report)
    tracks = []
    for row in report.get("tracks", []):
        if not isinstance(row, dict):
            continue
        track_id = str(row["track_id"])
        track_path = Path(str(row["output_track"]))
        tracks.append(
            {
                "track_id": track_id,
                "target_object_id": row.get("object_id"),
                "prompt_path": None,
                "sam2_output_dir": str(track_path.parent.parent),
                "sam2_track": str(track_path),
                "sam2_qc": str(args.local_mask_report),
                "overlay": None,
                "visible_frames": int(row.get("visible_frames", 0)),
                "frame_count": int(row.get("frames", 0)),
                "prompt_frames": [int(anchor["frame_idx"]) for anchor in row.get("anchors", []) if isinstance(anchor, dict) and "frame_idx" in anchor],
                "candidate_kind": "prompt_conditioned_local_grabcut_rgb_only",
            }
        )
    if not tracks:
        raise ContractError("local_mask_report_has_no_tracks")
    summary = {
        "schema": "v21_sam2_rgb_baseline_summary.v0",
        "status": "ok",
        "method": "adapt_v20_local_masks_to_v21_summary",
        "case_id": input_manifest.get("case_id"),
        "compute_target": "local_cpu_allowed",
        "input_manifest": str(args.input_manifest),
        "raw_frame_manifest": str(args.source_frame_manifest),
        "primary_video": str(input_manifest["primary_video"]),
        "frame_count": int(input_manifest["raw_frame_manifest_summary"]["frame_count"]),
        "tracks": tracks,
        "claim_scope": "V21 RGB-only local prompt-conditioned segmentation candidate. It is a fallback candidate when SAM2 video propagation disconnects; masks still require V21 contamination review.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt prompt-conditioned local mask report into V21 segmentation-candidate summary schema.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-frame-manifest", type=Path, required=True)
    parser.add_argument("--local-mask-report", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
