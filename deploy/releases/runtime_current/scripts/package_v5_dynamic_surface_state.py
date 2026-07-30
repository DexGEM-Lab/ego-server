#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def frames_by_key(path: Path, key: str) -> dict[int, dict]:
    rows = load_json(path).get(key)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} must contain nonempty {key}")
    return {int(row["frame_idx"]): row for row in rows}


def frame_entries(path: Path) -> list[dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    return [dict(entry) for entry in frames]


def contact_counts(path: Path) -> dict[int, int]:
    payload = load_json(path)
    rows = payload.get("rows_detail")
    if rows is None:
        rows = payload.get("geometry_backed_rows_preview")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain rows_detail or geometry_backed_rows_preview")
    counts: dict[int, int] = {}
    for row in rows:
        if bool(row.get("reliable_for_contact", False)):
            frame_idx = int(row["frame_idx"])
            counts[frame_idx] = counts.get(frame_idx, 0) + 1
    return counts


def state_for_frame(entry: dict, observability: dict | None, contact_count: int) -> tuple[str, list[str]]:
    source = str(entry.get("track_status_source", "unlabeled"))
    if source != "measured":
        return "completed_geometry", [source]
    if observability is None:
        return "unclassified_measured_geometry", ["missing_observability_row"]
    if bool(observability.get("map_observable", False)):
        return "map_observable_measured_geometry", []
    reasons = [str(value) for value in observability.get("reject_reasons", [])]
    if contact_count > 0:
        return "ambiguous_contact_geometry", reasons
    return "ambiguous_measured_geometry", reasons


def run(args: argparse.Namespace) -> dict:
    entries = frame_entries(args.manifest)
    observability_rows = frames_by_key(args.observability_qc, "frame_rows")
    contact = contact_counts(args.contact_qc)
    first = int(args.frame_start)
    last = int(args.frame_end)
    selected = [entry for entry in entries if first <= int(entry["frame_idx"]) <= last]
    expected = list(range(first, last + 1))
    actual = [int(entry["frame_idx"]) for entry in selected]
    if actual != expected:
        raise RuntimeError(f"manifest sequence is not dense: expected {expected}, got {actual}")

    rows = []
    state_counts: dict[str, int] = {}
    for entry in selected:
        frame_idx = int(entry["frame_idx"])
        obs = observability_rows.get(frame_idx)
        state, reasons = state_for_frame(entry, obs, int(contact.get(frame_idx, 0)))
        state_counts[state] = state_counts.get(state, 0) + 1
        row = {
            "frame_idx": frame_idx,
            "geometry_state": state,
            "state_reasons": reasons,
            "track_status_source": str(entry.get("track_status_source", "unlabeled")),
            "rgb": entry.get("rgb"),
            "mask": entry.get("mask"),
            "reliable_contact_rows": int(contact.get(frame_idx, 0)),
        }
        if obs is not None:
            for key in (
                "map_observable",
                "stable_neighbor_count",
                "pca_extent_ratio_to_median",
                "pca_extent_max_abs_log_to_median",
                "silhouette_mask_iou",
                "visible_silhouette_inside_mask_fraction",
                "zbuffer_depth_abs_p95_m",
            ):
                row[key] = obs.get(key)
        rows.append(row)

    package = {
        "status": "ok",
        "method": "package_v5_dynamic_surface_state",
        "claim": "V5 keeps the completed V4 per-frame mesh archive as accepted measured/completed geometry and attaches map-observability state for downstream dynamic fitting.",
        "frame_start": first,
        "frame_end": last,
        "frames": len(rows),
        "state_counts": state_counts,
        "rows": rows,
        "accepted_geometry": {
            "manifest": str(args.manifest),
            "mesh_archive": str(args.mesh_archive),
            "zbuffer_qc": str(args.zbuffer_qc),
            "contact_qc": str(args.contact_qc),
            "selected_contact_sdf_qc": str(args.selected_contact_sdf_qc),
            "full_hand_sdf_qc": str(args.full_hand_sdf_qc),
        },
        "observability": {
            "qc": str(args.observability_qc),
            "candidate_windows": load_json(args.observability_qc).get("candidate_windows", []),
        },
        "rigid_map_falsifications": {
            "full_window": str(args.full_window_rigid_map_qc),
            "observable_window": str(args.observable_window_rigid_map_qc),
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
    parser.add_argument("--observability-qc", type=Path, required=True)
    parser.add_argument("--full-window-rigid-map-qc", type=Path, required=True)
    parser.add_argument("--observable-window-rigid-map-qc", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
