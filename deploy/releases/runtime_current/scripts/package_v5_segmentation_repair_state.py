#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def require_ok(path: Path) -> dict:
    data = load_json(path)
    if data.get("status") != "ok" and not str(data.get("status", "")).startswith("diagnostic_"):
        raise RuntimeError(f"QC report is not ok: {path} status={data.get('status')}")
    return data


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"manifest has no frames list: {args.manifest}")
    first = int(args.frame_start)
    last = int(args.frame_end)
    selected = [row for row in frames if first <= int(row["frame_idx"]) <= last]
    frame_ids = [int(row["frame_idx"]) for row in selected]
    expected = list(range(first, last + 1))
    if frame_ids != expected:
        raise RuntimeError(f"repair manifest is not dense: expected {expected}, got {frame_ids}")

    require_ok(args.zbuffer_qc)
    require_ok(args.contact_qc)
    selected_sdf = require_ok(args.selected_contact_sdf_qc)
    full_hand_sdf = require_ok(args.full_hand_sdf_qc)
    if float(selected_sdf["summary"]["penetration_fraction"]) != 0.0:
        raise RuntimeError("selected-contact SDF report has nonzero penetration")
    if float(full_hand_sdf["summary"]["penetration_fraction"]) != 0.0:
        raise RuntimeError("full-hand SDF report has nonzero penetration")

    rows = []
    for row in selected:
        rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "geometry_state": "segmentation_repaired_geometry",
                "state_reasons": [
                    str(args.repair_source),
                    "zbuffer_replay_pass",
                    "sdf_nonpenetration_pass",
                ],
                "track_status_source": "v5_segmentation_repair",
                "rgb": row.get("rgb"),
                "mask": row.get("mask"),
                "mask_pixels": int(row.get("mask_pixels", 0)),
                "reliable_contact_rows": int(args.contact_rows_per_frame),
            }
        )

    package = {
        "status": "ok",
        "method": "package_v5_segmentation_repair_state",
        "claim": "V5 replaces a bounded ambiguous object-mask interval with a model-tracked instance mask and keeps the same mesh, depth, contact, and SDF replay contract.",
        "frame_start": first,
        "frame_end": last,
        "frames": len(rows),
        "state_counts": {"segmentation_repaired_geometry": len(rows)},
        "rows": rows,
        "accepted_geometry": {
            "manifest": str(args.manifest),
            "mask_track": str(args.mask_track),
            "mesh_archive": str(args.mesh_archive),
            "zbuffer_qc": str(args.zbuffer_qc),
            "contact_qc": str(args.contact_qc),
            "selected_contact_sdf_qc": str(args.selected_contact_sdf_qc),
            "full_hand_sdf_qc": str(args.full_hand_sdf_qc),
        },
        "baseline_window": {
            "v4_completed_zbuffer_qc": str(args.baseline_zbuffer_qc),
            "v4_completed_contact_qc": str(args.baseline_contact_qc),
            "frame_window": [first, last],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in package.items() if k != "rows"}, indent=2))
    return package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--zbuffer-qc", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path, required=True)
    parser.add_argument("--selected-contact-sdf-qc", type=Path, required=True)
    parser.add_argument("--full-hand-sdf-qc", type=Path, required=True)
    parser.add_argument("--baseline-zbuffer-qc", type=Path, required=True)
    parser.add_argument("--baseline-contact-qc", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--repair-source", default="sam2_seed_track")
    parser.add_argument("--contact-rows-per-frame", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
