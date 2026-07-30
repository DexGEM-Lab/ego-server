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


def reliable_contact_frames(path: Path) -> list[int]:
    payload = load_json(path)
    rows = payload.get("rows_detail")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks rows_detail")
    frames = sorted({int(row["frame_idx"]) for row in rows if bool(row.get("reliable_for_contact", False))})
    if not frames:
        raise RuntimeError(f"{path} contains no reliable_for_contact rows")
    return frames


def run(args: argparse.Namespace) -> dict:
    source = load_json(args.source_sdf_report)
    proof = load_json(args.identity_report)
    contact_frames = reliable_contact_frames(args.contact_report)
    source_frames = [int(frame) for frame in source.get("frames", [])]
    changed_contact = [int(frame) for frame in proof.get("changed_reliable_contact_frames", [])]
    missing_contact = [int(frame) for frame in proof.get("missing_reliable_contact_frames", [])]
    if changed_contact or missing_contact:
        raise RuntimeError(f"identity report does not prove unchanged contact frames: {args.identity_report}")
    if sorted(source_frames) != sorted(contact_frames):
        raise RuntimeError(
            "source SDF frames do not match reliable contact frames: "
            f"source={source_frames} contact={contact_frames}"
        )
    report = dict(source)
    report["annotations"] = str(args.annotations)
    report["mesh_archive"] = str(args.new_mesh_archive)
    report["contact_report"] = str(args.contact_report)
    report["source_sdf_report"] = str(args.source_sdf_report)
    report["identity_report"] = str(args.identity_report)
    report["composition_proof"] = {
        "basis": (
            "The new mesh archive differs from the source archive only outside frames selected by "
            "reliable_for_contact rows. SDF samples in this report are defined only on those selected "
            "contact frames, so their values are unchanged from the source SDF report."
        ),
        "reliable_contact_frames": contact_frames,
        "changed_or_added_archive_frames": [int(frame) for frame in proof.get("changed_or_added_archive_frames", [])],
        "changed_reliable_contact_frames": changed_contact,
        "missing_reliable_contact_frames": missing_contact,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "method": report.get("method"),
                "frames": report.get("frames"),
                "summary": report.get("summary"),
                "composition_proof": report["composition_proof"],
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sdf-report", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--new-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
