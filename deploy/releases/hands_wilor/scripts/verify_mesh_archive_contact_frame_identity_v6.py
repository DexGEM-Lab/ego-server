#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def mesh_archive_hashes(path: Path) -> dict[int, str]:
    if not path.exists():
        raise RuntimeError(f"mesh archive does not exist: {path}")
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"]
    faces = blob["faces"]
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError(f"{path} offsets do not match frame count")
    hashes: dict[int, str] = {}
    for i, frame in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(vertices[v0:v1]).view(np.uint8))
        h.update(np.ascontiguousarray(faces[f0:f1]).view(np.uint8))
        hashes[int(frame)] = h.hexdigest()
    return hashes


def selected_contact_frames(contact_report: Path) -> list[int]:
    payload = load_json(contact_report)
    rows = payload.get("rows_detail")
    if not isinstance(rows, list):
        raise RuntimeError(f"{contact_report} lacks rows_detail")
    frames = sorted({int(row["frame_idx"]) for row in rows if bool(row.get("reliable_for_contact", False))})
    if not frames:
        raise RuntimeError(f"{contact_report} contains no reliable_for_contact rows")
    return frames


def run(args: argparse.Namespace) -> dict:
    old_hashes = mesh_archive_hashes(args.old_mesh_archive)
    new_hashes = mesh_archive_hashes(args.new_mesh_archive)
    contact_frames = selected_contact_frames(args.contact_report)
    missing = [frame for frame in contact_frames if frame not in old_hashes or frame not in new_hashes]
    changed = [
        frame
        for frame in contact_frames
        if frame in old_hashes and frame in new_hashes and old_hashes[frame] != new_hashes[frame]
    ]
    replaced_frames = sorted(set(new_hashes).symmetric_difference(set(old_hashes)) | {f for f in old_hashes if f in new_hashes and old_hashes[f] != new_hashes[f]})
    report = {
        "status": "ok" if not missing and not changed else "failed",
        "old_mesh_archive": str(args.old_mesh_archive),
        "new_mesh_archive": str(args.new_mesh_archive),
        "contact_report": str(args.contact_report),
        "reliable_contact_frames": contact_frames,
        "changed_reliable_contact_frames": changed,
        "missing_reliable_contact_frames": missing,
        "changed_or_added_archive_frames": replaced_frames,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing or changed:
        raise RuntimeError(f"contact-frame mesh identity failed: {args.output_json}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-mesh-archive", type=Path, required=True)
    parser.add_argument("--new-mesh-archive", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
