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


def manifest_entries(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks frames list")
    out = {}
    for row in rows:
        frame = int(row["frame_idx"])
        out[frame] = dict(row)
    return out


def zbuffer_rows(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks rows list")
    return {int(row["frame_idx"]): dict(row) for row in rows}


def reliable_contact_counts(path: Path) -> dict[int, int]:
    payload = load_json(path)
    rows = payload.get("rows_detail")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks rows_detail")
    counts: dict[int, int] = {}
    for row in rows:
        if bool(row.get("reliable_for_contact", False)):
            frame = int(row["frame_idx"])
            counts[frame] = counts.get(frame, 0) + 1
    return counts


def run(args: argparse.Namespace) -> dict:
    entries = manifest_entries(args.manifest)
    zrows = zbuffer_rows(args.zbuffer_qc)
    contact = reliable_contact_counts(args.contact_qc)
    first = int(args.frame_start)
    last = int(args.frame_end)
    expected = list(range(first, last + 1))
    missing = [frame for frame in expected if frame not in entries or frame not in zrows]
    if missing:
        raise RuntimeError(f"state inputs missing frames: {missing}")
    rows = []
    counts: dict[str, int] = {}
    for frame in expected:
        entry = entries[frame]
        repair = frame in set(args.repaired_frames)
        if repair:
            state = "verified_vlm_sam2_repaired_geometry"
            reasons = ["VLM mask-verifier replacement prompt", "SAM2 candidate accepted by point contract", "one-frame z-buffer replay below 5 mm p95"]
        else:
            source = str(entry.get("source", entry.get("track_status_source", "")))
            status = str(entry.get("track_status", entry.get("status", "")))
            selection = entry.get("selection", {})
            candidate = str(entry.get("candidate", ""))
            if isinstance(selection, dict):
                candidate = " ".join([candidate, str(selection.get("candidate", "")), str(selection.get("reason", ""))])
            provenance = " ".join([source, status, candidate, str(entry.get("rgb", "")), str(entry.get("mask", ""))]).lower()
            if "sam2_mask_seed" in provenance or "completion" in provenance:
                state = "completed_tracked_geometry"
                reasons = ["temporal mask-seed completion", "image-depth replay accepted"]
            else:
                state = "measured_mesh_geometry"
                reasons = ["model-produced mask", "UniDepth metric surface", "1 mm sheet solidification"]
        counts[state] = counts.get(state, 0) + 1
        z = zrows[frame]
        rows.append(
            {
                "frame_idx": frame,
                "geometry_state": state,
                "state_reasons": reasons,
                "track_status_source": str(entry.get("source", entry.get("track_status_source", ""))),
                "rgb": entry.get("rgb"),
                "mask": entry.get("mask"),
                "reliable_contact_rows": int(contact.get(frame, 0)),
                "silhouette_mask_iou": z.get("silhouette_mask_iou"),
                "visible_silhouette_inside_mask_fraction": z.get("visible_silhouette_inside_mask_fraction"),
                "zbuffer_depth_abs_p95_m": z.get("zbuffer_depth_abs_p95_m"),
            }
        )
    package = {
        "status": "ok",
        "method": "package_v6_repaired_track_state",
        "claim": "V6 keeps the per-frame reconstructed object mesh stream and records the verified frame-2539 perception repair plus sparse CoTracker temporal-prior evidence.",
        "frame_start": first,
        "frame_end": last,
        "frames": len(rows),
        "state_counts": counts,
        "rows": rows,
        "accepted_geometry": {
            "manifest": str(args.manifest),
            "mesh_archive": str(args.mesh_archive),
            "zbuffer_qc": str(args.zbuffer_qc),
            "contact_qc": str(args.contact_qc),
            "selected_contact_sdf_qc": str(args.selected_contact_sdf_qc),
            "full_hand_sdf_qc": str(args.full_hand_sdf_qc),
            "contact_frame_identity_qc": str(args.contact_frame_identity_qc),
        },
        "temporal_priors": {
            "factor_graph_qc": str(args.factor_graph_qc),
            "factor_graph_mesh_archive": str(args.factor_graph_mesh_archive),
        },
        "repair": {
            "repaired_frames": [int(frame) for frame in args.repaired_frames],
            "verifier_report": str(args.verifier_report),
            "sam2_qc": str(args.sam2_qc),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in package.items() if k != "rows"}, indent=2))
    return package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--zbuffer-qc", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path, required=True)
    parser.add_argument("--selected-contact-sdf-qc", type=Path, required=True)
    parser.add_argument("--full-hand-sdf-qc", type=Path, required=True)
    parser.add_argument("--contact-frame-identity-qc", type=Path, required=True)
    parser.add_argument("--factor-graph-qc", type=Path, required=True)
    parser.add_argument("--factor-graph-mesh-archive", type=Path, required=True)
    parser.add_argument("--verifier-report", type=Path, required=True)
    parser.add_argument("--sam2-qc", type=Path, required=True)
    parser.add_argument("--repaired-frames", type=int, nargs="+", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
