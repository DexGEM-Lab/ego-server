#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS = "multi_object_mask_timeline_not_pose_or_geometry"
CLAIM = (
    "This artifact materializes the V17 object roster as a full-frame multi-object mask-evidence timeline. "
    "It is not object-pose annotation because object mesh geometry and pose variables remain absent."
)


@dataclass(frozen=True)
class TrackBundle:
    object_id: str
    track_id: str
    track_path: Path
    track_payload: dict[str, Any]
    active_intervals: tuple[tuple[int, int], ...]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a JSON boolean")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def finite_number_list(value: Any, label: str, length: int) -> list[float]:
    row = require_list(value, label)
    if len(row) != length:
        raise RuntimeError(f"{label} must have length {length}")
    return [finite_number(item, f"{label}[{i}]") for i, item in enumerate(row)]


def object_track_id(object_id: str) -> str:
    prefix = "object:"
    if not object_id.startswith(prefix):
        raise RuntimeError(f"object_id must start with {prefix}: {object_id}")
    return object_id[len(prefix) :]


def local_mask_path(track_path: Path, source_mask_path: str | None, frame_idx: int) -> str | None:
    if source_mask_path is None:
        return None
    source = Path(source_mask_path)
    filename = source.name or f"{frame_idx:06d}.png"
    local = track_path.parent / "sam2_masks" / filename
    if not local.exists():
        raise RuntimeError(f"local SAM2 mask path is missing for frame {frame_idx}: {local}")
    return str(local)


def visible_track_row(track_path: Path, track_payload: dict[str, Any], frame_idx: int) -> dict[str, Any] | None:
    raw = require_dict(track_payload.get(str(frame_idx)), f"{track_path} frame {frame_idx}")
    visible = require_bool(raw.get("visible"), f"{track_path} frame {frame_idx}.visible")
    if not visible:
        return None
    bbox = finite_number_list(raw.get("bbox_xyxy"), f"{track_path} frame {frame_idx}.bbox_xyxy", 4)
    center = finite_number_list(raw.get("center_xy"), f"{track_path} frame {frame_idx}.center_xy", 2)
    area = finite_number(raw.get("area_px"), f"{track_path} frame {frame_idx}.area_px")
    mask_source = require_str(raw.get("mask_path"), f"{track_path} frame {frame_idx}.mask_path")
    mask_local = local_mask_path(track_path, mask_source, frame_idx)
    return {
        "visible": True,
        "bbox_xyxy": bbox,
        "center_xy": center,
        "area_px": area,
        "source_mask_path": mask_source,
        "mask_path": mask_local,
        "mask_path_status": "local_path_verified",
    }


def frame_in_intervals(frame_idx: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= frame_idx <= end for start, end in intervals)


def parse_intervals(raw: Any, label: str) -> tuple[tuple[int, int], ...]:
    rows = require_list(raw, label)
    intervals: list[tuple[int, int]] = []
    for i, item in enumerate(rows):
        pair = require_list(item, f"{label}[{i}]")
        if len(pair) != 2:
            raise RuntimeError(f"{label}[{i}] must have two endpoints")
        start = require_int(pair[0], f"{label}[{i}][0]")
        end = require_int(pair[1], f"{label}[{i}][1]")
        if end < start:
            raise RuntimeError(f"{label}[{i}] has end before start")
        intervals.append((start, end))
    return tuple(intervals)


def active_roster_rows(roster: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(roster):
        row = require_dict(item, f"object roster row {i}")
        if row.get("source") != "vlm_object_plan":
            continue
        count = row.get("active_frame_count")
        if count is None or require_int(count, f"object roster row {i}.active_frame_count") <= 0:
            continue
        rows.append(row)
    return rows


def sam2_source_roots(manifest: dict[str, Any]) -> list[Path]:
    sources = require_list(manifest.get("sam2_multiobject_sources"), "sam2_multiobject_sources")
    roots: list[Path] = []
    for i, source in enumerate(sources):
        row = require_dict(source, f"sam2_multiobject_sources[{i}]")
        if row.get("status") != "ok":
            continue
        path = Path(require_str(row.get("path"), f"sam2_multiobject_sources[{i}].path"))
        if not path.exists():
            raise RuntimeError(f"SAM2 root does not exist: {path}")
        roots.append(path)
    if not roots:
        raise RuntimeError("no ok SAM2 multi-object roots in manifest")
    return roots


def load_track_bundle(row: dict[str, Any], sam2_roots: list[Path]) -> TrackBundle:
    object_id = require_str(row.get("object_id"), "object_id")
    track_id = object_track_id(object_id)
    candidates = [root / track_id / "sam2" / "sam2_track.json" for root in sam2_roots]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise RuntimeError(f"expected exactly one SAM2 track for {object_id}, found {existing}")
    track_path = existing[0]
    qc_path = track_path.parents[2] / "qc_sam2_multiobject_points.json"
    qc = require_dict(load_json(qc_path), f"{qc_path}")
    active_by_track = require_dict(qc.get("active_intervals_by_track"), f"{qc_path}.active_intervals_by_track")
    intervals = parse_intervals(active_by_track.get(track_id), f"{qc_path}.active_intervals_by_track.{track_id}")
    track_payload = require_dict(load_json(track_path), f"{track_path}")
    return TrackBundle(
        object_id=object_id,
        track_id=track_id,
        track_path=track_path,
        track_payload=track_payload,
        active_intervals=intervals,
    )


def object_state(
    row: dict[str, Any],
    bundle: TrackBundle,
    frame_idx: int,
) -> dict[str, Any] | None:
    if not frame_in_intervals(frame_idx, bundle.active_intervals):
        return None
    visible = visible_track_row(bundle.track_path, bundle.track_payload, frame_idx)
    state: dict[str, Any] = {
        "object_id": bundle.object_id,
        "track_id": bundle.track_id,
        "name": row.get("name"),
        "frame_idx": frame_idx,
        "active": True,
        "active_intervals": [list(pair) for pair in bundle.active_intervals],
        "role_status": row.get("role_status"),
        "physical_notes": row.get("physical_notes"),
        "sam2_track": str(bundle.track_path),
        "mask_evidence_status": "visible_mask" if visible is not None else "active_interval_no_visible_mask",
        "geometry_state": "mask_only_no_mesh",
        "pose_state": "no_object_pose_variable",
        "mesh_state": "missing_object_mesh",
        "multi_object_timeline_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
    }
    if visible is not None:
        state.update(visible)
    else:
        state.update(
            {
                "visible": False,
                "bbox_xyxy": None,
                "center_xy": None,
                "area_px": None,
                "mask_path": None,
                "mask_path_status": "no_visible_mask_for_active_frame",
            }
        )
    return state


def case_timeline(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = require_dict(load_json(manifest_path), f"{manifest_path}")
    case = require_str(manifest.get("case"), f"{manifest_path}.case")
    v16_manifest_path = Path(require_str(manifest.get("manifest"), f"{manifest_path}.manifest"))
    v16_manifest = require_dict(load_json(v16_manifest_path), f"{v16_manifest_path}")
    raw_video = require_dict(v16_manifest.get("raw_video"), f"{v16_manifest_path}.raw_video")
    frame_count = require_int(raw_video.get("frame_count"), f"{case}.raw_video.frame_count")
    finite_number(raw_video.get("fps"), f"{case}.raw_video.fps")

    roster_path = Path(require_str(manifest.get("object_roster"), f"{manifest_path}.object_roster"))
    roster = require_list(load_json(roster_path), f"{roster_path}")
    rows = active_roster_rows(roster)
    sam2_roots = sam2_source_roots(manifest)
    bundles = [load_track_bundle(row, sam2_roots) for row in rows]
    row_by_object_id = {require_str(row.get("object_id"), "object_id"): row for row in rows}

    frames: list[dict[str, Any]] = []
    visible_counts = {bundle.object_id: 0 for bundle in bundles}
    active_counts = {bundle.object_id: 0 for bundle in bundles}
    missing_visible_counts = {bundle.object_id: 0 for bundle in bundles}
    for frame_idx in range(frame_count):
        objects: list[dict[str, Any]] = []
        for bundle in bundles:
            state = object_state(row_by_object_id[bundle.object_id], bundle, frame_idx)
            if state is None:
                continue
            objects.append(state)
            active_counts[bundle.object_id] += 1
            if state["visible"]:
                visible_counts[bundle.object_id] += 1
            else:
                missing_visible_counts[bundle.object_id] += 1
        frames.append({"frame_idx": frame_idx, "objects": objects})

    object_rows = []
    for bundle in bundles:
        row = row_by_object_id[bundle.object_id]
        object_rows.append(
            {
                "object_id": bundle.object_id,
                "track_id": bundle.track_id,
                "name": row.get("name"),
                "source": row.get("source"),
                "active_intervals": [list(pair) for pair in bundle.active_intervals],
                "active_frame_count": active_counts[bundle.object_id],
                "visible_mask_frame_count": visible_counts[bundle.object_id],
                "active_without_visible_mask_frame_count": missing_visible_counts[bundle.object_id],
                "sam2_track": str(bundle.track_path),
                "geometry_state": "mask_only_no_mesh",
                "pose_state": "no_object_pose_variable",
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )

    payload = {
        "method": "build_v17_multi_object_timeline",
        "case": case,
        "status": STATUS,
        "claim": CLAIM,
        "frame_count": frame_count,
        "raw_video": raw_video,
        "source_manifest": str(manifest_path),
        "object_roster": str(roster_path),
        "sam2_roots": [str(path) for path in sam2_roots],
        "object_count": len(object_rows),
        "object_frame_rows": sum(active_counts.values()),
        "visible_mask_frame_rows": sum(visible_counts.values()),
        "active_without_visible_mask_frame_rows": sum(missing_visible_counts.values()),
        "objects": object_rows,
        "frames": frames,
        "multi_object_timeline_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    write_json(output_root / case / "v17_multi_object_timeline.json", payload)
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.measurement_store_root / "v17_measurement_store_summary.json"
    summary = require_dict(load_json(summary_path), f"{summary_path}")
    cases = require_list(summary.get("cases"), f"{summary_path}.cases")
    case_outputs: list[dict[str, Any]] = []
    for i, raw_case in enumerate(cases):
        row = require_dict(raw_case, f"{summary_path}.cases[{i}]")
        case = require_str(row.get("case"), f"{summary_path}.cases[{i}].case")
        manifest_path = args.measurement_store_root / case / "v17_measurement_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"measurement manifest is missing for {case}: {manifest_path}")
        case_outputs.append(case_timeline(manifest_path, args.output_root))

    payload = {
        "method": "build_v17_multi_object_timeline",
        "status": STATUS,
        "claim": CLAIM,
        "measurement_store_summary": str(summary_path),
        "case_count": len(case_outputs),
        "cases": [
            {
                "case": case["case"],
                "timeline_path": str(args.output_root / case["case"] / "v17_multi_object_timeline.json"),
                "frame_count": case["frame_count"],
                "object_count": case["object_count"],
                "object_frame_rows": case["object_frame_rows"],
                "visible_mask_frame_rows": case["visible_mask_frame_rows"],
                "active_without_visible_mask_frame_rows": case["active_without_visible_mask_frame_rows"],
                "multi_object_timeline_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "annotation_ready": False,
            }
            for case in case_outputs
        ],
        "multi_object_timeline_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_multi_object_timeline_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
