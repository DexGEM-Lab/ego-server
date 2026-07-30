#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def frame_index(frames: list[dict[str, Any]], context: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"{context} frame row {row_i} is not a JSON object")
        idx = frame.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{context} frame row {row_i} has invalid frame_idx {idx!r}")
        out[idx] = frame
    return out


def anchor_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("anchors")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} has no anchors list")
    out: dict[int, dict[str, Any]] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} anchor row {row_i} is not a JSON object")
        idx = row.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} anchor row {row_i} has invalid frame_idx {idx!r}")
        out[idx] = row
    return out


def repair_frames(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} has no frames list")
    return frame_index(rows, str(path))


def contact_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain a JSON list")
    out: dict[int, list[dict[str, Any]]] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} contact row {row_i} is not a JSON object")
        idx = row.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} contact row {row_i} has invalid frame_idx {idx!r}")
        out.setdefault(idx, []).append(row)
    return out


def finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def valid_hand(hand: dict[str, Any], max_residual_px: float) -> bool:
    if hand.get("measurement_available") is False:
        return False
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return False
    value = finite_float(residual.get("median"))
    if value is None:
        return False
    return value <= float(max_residual_px)


def annotate_hand_state(hand: dict[str, Any], state: str, source: str) -> dict[str, Any]:
    out = copy.deepcopy(hand)
    out["v17_state"] = state
    out["v17_state_source"] = source
    return out


def solve(args: argparse.Namespace) -> dict[str, Any]:
    base_payload = load_json(args.base_annotations)
    base_frames_raw = base_payload.get("frames")
    if not isinstance(base_frames_raw, list):
        raise RuntimeError(f"{args.base_annotations} has no frames list")
    base_frames = frame_index(base_frames_raw, str(args.base_annotations))
    repairs = repair_frames(args.hand_repair_annotations)
    anchors = anchor_rows(args.anchor_qc)
    contacts = contact_rows(args.hand_repair_contact_measurements)

    output_frames: list[dict[str, Any]] = []
    anchor_summary: list[dict[str, Any]] = []
    repaired_frame_count = 0
    unresolved_frame_count = 0
    for idx in sorted(base_frames):
        frame = copy.deepcopy(base_frames[idx])
        anchor = anchors.get(idx)
        repair = repairs.get(idx)
        frame_contacts = contacts.get(idx, [])
        if repair is not None:
            solved_hands = [
                annotate_hand_state(hand, "measured_repaired", "selected_v17_hamer_anchor_repair")
                for hand in repair.get("hands", [])
                if isinstance(hand, dict) and valid_hand(hand, float(args.max_hand_residual_px))
            ]
            if not solved_hands:
                raise RuntimeError(f"repair frame {idx} produced no valid hand states")
            frame["hands"] = solved_hands
            repaired_frame_count += 1
        else:
            frame["hands"] = [
                annotate_hand_state(hand, "measured_accepted", "v16_measurement_retained")
                for hand in frame.get("hands", [])
                if isinstance(hand, dict) and valid_hand(hand, float(args.max_hand_residual_px))
            ]
        frame["v17_contact_measurements"] = frame_contacts
        if anchor is not None:
            failures = anchor.get("failures")
            if not isinstance(failures, list):
                raise RuntimeError(f"anchor {idx} has invalid failures list")
            status = "accepted" if not failures else "rejected_unresolved"
            if status == "rejected_unresolved":
                unresolved_frame_count += 1
            frame["v17_anchor_status"] = {
                "status": status,
                "failures": failures,
                "expected_visible_hands": anchor.get("expected_visible_hands"),
                "expected_contact": anchor.get("expected_contact"),
            }
            anchor_summary.append({"frame_idx": idx, "status": status, "failures": failures})
        output_frames.append(frame)

    output_payload = copy.deepcopy(base_payload)
    output_payload["method"] = "assemble_v17_full_timeline_state"
    output_payload["base_annotations"] = str(args.base_annotations)
    output_payload["hand_repair_annotations"] = str(args.hand_repair_annotations)
    output_payload["anchor_qc"] = str(args.anchor_qc)
    output_payload["solver_contract"] = {
        "state_scope": "full_timeline_state_export_with_anchor_graph_hand_repairs",
        "retained_object_state": "v16_object_mesh_stream",
        "known_missing_v17_requirements": [
            "continuous MANO pose optimization",
            "multi-object geometry solve",
            "contact latent-state optimization",
            "tomato persistent canonical-shape solve",
        ],
    }
    output_payload["frames"] = output_frames
    write_json(args.output_annotations, output_payload)

    manifest = {
        "status": "ok",
        "method": "assemble_v17_full_timeline_state",
        "output_annotations": str(args.output_annotations),
        "frame_count": len(output_frames),
        "repaired_frame_count": repaired_frame_count,
        "unresolved_anchor_frame_count": unresolved_frame_count,
        "anchor_summary": anchor_summary,
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--hand-repair-annotations", type=Path, required=True)
    parser.add_argument("--hand-repair-contact-measurements", type=Path, required=True)
    parser.add_argument("--anchor-qc", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-hand-residual-px", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
