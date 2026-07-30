#!/usr/bin/env python3
"""Build the V19 render-consumed rigid object state.

This adapter moves the rigid-object render boundary out of private measurement
reports.  It reads the V19 annotation backbone, completed mesh contract,
corrected rigid pose trajectory, and hand/object uncertainty sidecars, then
writes one explicit state JSON consumed by ``render_v19_rigid_state_artifact.py``.

The state may still preserve provenance paths back to measurements, but the
renderer consumes this state document as its authority for which geometry and
pose trajectory are visible in the final artifact.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ACCEPTED_RIGID_POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
    "completed_temporal_rigid_pose_uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--object-label", default=None)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, default=None)
    parser.add_argument("--constraint-report", type=Path, required=True)
    parser.add_argument("--temporal-mano-state", type=Path, default=None)
    parser.add_argument("--hidden-volume-validation", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite absolute path prefixes while building state, useful when testing remote-run artifacts locally.",
    )
    parser.add_argument(
        "--allow-missing-poses",
        action="store_true",
        help="Write explicit missing-pose uncertainty instead of failing when the rigid trajectory is not full-timeline.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise RuntimeError(f"invalid --path-rewrite {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        if not old:
            raise RuntimeError(f"invalid --path-rewrite {item!r}; OLD is empty")
        rewrites.append((old.rstrip("/"), new.rstrip("/")))
    return rewrites


def rewrite_path(path: Path | str | None, rewrites: list[tuple[str, str]]) -> Path | None:
    if path is None:
        return None
    text = str(path)
    for old, new in rewrites:
        if text == old or text.startswith(old + "/"):
            text = new + text[len(old) :]
            break
    return Path(text)


def completion_report_completed_mesh(path: Path, rewrites: list[tuple[str, str]]) -> Path:
    data = load_json(path)
    outputs = data.get("outputs") if isinstance(data, dict) else None
    if not isinstance(outputs, dict):
        raise RuntimeError(f"completion report {path} has no outputs object")
    value = outputs.get("completed_mesh_labeled") or outputs.get("completed_mesh")
    if not value:
        raise RuntimeError(f"completion report {path} has no completed mesh output")
    out = rewrite_path(str(value), rewrites)
    assert out is not None
    return out


def same_existing_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


def validate_completed_mesh(completed_mesh: Path, expected_mesh: Path) -> None:
    if completed_mesh.name == "trellis_mesh.ply" or any(part.startswith("trellis_") for part in completed_mesh.parts):
        raise RuntimeError(
            "render-state mesh contract broken: completed mesh must be the P13 completed-canonical mesh, "
            f"not raw TRELLIS output ({completed_mesh})"
        )
    if not same_existing_file(completed_mesh, expected_mesh) and completed_mesh.resolve(strict=False) != expected_mesh.resolve(strict=False):
        raise RuntimeError(
            "render-state mesh contract broken: pose rows are in the P13 completed-canonical frame, "
            f"but completed_mesh={completed_mesh} differs from completion outputs.completed_mesh_labeled={expected_mesh}"
        )
    if not completed_mesh.exists() or completed_mesh.stat().st_size <= 0:
        raise RuntimeError(f"completed mesh is missing or empty: {completed_mesh}")


def accepted_pose_rows(pose_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in pose_data.get("pose_rows", []) if isinstance(pose_data.get("pose_rows"), list) else []:
        if not isinstance(raw, dict):
            continue
        if raw.get("status") not in ACCEPTED_RIGID_POSE_STATUSES:
            continue
        frame_idx = int(raw.get("frame_idx"))
        rot = np.asarray(raw.get("rotation_world_from_completed_canonical_matrix"), dtype=np.float64)
        trans = np.asarray(raw.get("translation_world_m"), dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,) or not np.isfinite(rot).all() or not np.isfinite(trans).all():
            raise RuntimeError(f"invalid rigid pose row for frame {frame_idx}")
        rows.append(dict(raw))
    rows.sort(key=lambda row: int(row["frame_idx"]))
    return rows


def frame_ids_from_annotations(annotations: dict[str, Any]) -> list[int]:
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations contain no frames")
    out: list[int] = []
    for pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"annotation frame at position {pos} is not an object")
        out.append(int(frame.get("frame_idx", pos)))
    return out


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rewrites = parse_rewrites(list(args.path_rewrite or []))
    annotations_path = rewrite_path(args.annotations, rewrites)
    pose_report_path = rewrite_path(args.pose_report, rewrites)
    completion_report_path = rewrite_path(args.completion_report, rewrites)
    constraint_report_path = rewrite_path(args.constraint_report, rewrites)
    temporal_mano_path = rewrite_path(args.temporal_mano_state, rewrites)
    hidden_validation_path = rewrite_path(args.hidden_volume_validation, rewrites)
    if annotations_path is None or pose_report_path is None or completion_report_path is None or constraint_report_path is None:
        raise RuntimeError("required input path unexpectedly resolved to None")

    annotations = load_json(annotations_path)
    pose_data = load_json(pose_report_path)
    completion_data = load_json(completion_report_path)
    constraint_data = load_json(constraint_report_path)
    if not isinstance(annotations, dict) or not isinstance(pose_data, dict) or not isinstance(completion_data, dict) or not isinstance(constraint_data, dict):
        raise RuntimeError("annotations, pose report, completion report, and constraint report must be JSON objects")

    expected_mesh = completion_report_completed_mesh(completion_report_path, rewrites)
    completed_mesh = rewrite_path(args.completed_mesh, rewrites) if args.completed_mesh is not None else expected_mesh
    if completed_mesh is None:
        raise RuntimeError("completed mesh could not be resolved")
    validate_completed_mesh(completed_mesh, expected_mesh)

    frame_ids = frame_ids_from_annotations(annotations)
    pose_rows = accepted_pose_rows(pose_data)
    pose_frame_ids = {int(row["frame_idx"]) for row in pose_rows}
    missing_pose_frames = sorted(set(frame_ids).difference(pose_frame_ids))
    if missing_pose_frames and not args.allow_missing_poses:
        preview = missing_pose_frames[:20]
        raise RuntimeError(
            f"rigid render state is not full-timeline: {len(missing_pose_frames)} annotation frames lack accepted rigid poses; "
            f"first missing frames={preview}. P15 must complete uncertain poses before P19."
        )

    constraint_rows = constraint_data.get("constraint_rows") if isinstance(constraint_data.get("constraint_rows"), list) else []
    temporal_payload = load_optional_json(temporal_mano_path)
    hidden_payload = load_optional_json(hidden_validation_path)

    output = rewrite_path(args.output, rewrites)
    if output is None:
        raise RuntimeError("output path resolved to None")
    object_label = args.object_label or args.object_id
    state = {
        "status": "ok",
        "method": "build_v19_rigid_render_state",
        "claim_scope": "explicit render-consumed state for a rigid object body, pose trajectory, MANO/contact uncertainty, and projection contract",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "object_label": str(object_label),
        "created_unix_s": time.time(),
        "inputs": {
            "annotations": str(annotations_path),
            "pose_report": str(pose_report_path),
            "completion_report": str(completion_report_path),
            "completed_mesh": str(completed_mesh),
            "constraint_report": str(constraint_report_path),
            "temporal_mano_state": str(temporal_mano_path) if temporal_mano_path is not None else None,
            "hidden_volume_validation": str(hidden_validation_path) if hidden_validation_path is not None else None,
        },
        "annotation_backbone": {
            "path": str(annotations_path),
            "frame_count": len(frame_ids),
            "frame_ids": frame_ids,
            "raw_video": annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else None,
        },
        "object_geometry": {
            "state": "completed_canonical_rigid_mesh",
            "completed_mesh_path": str(completed_mesh),
            "completion_report_path": str(completion_report_path),
            "completion_outputs": completion_data.get("outputs") if isinstance(completion_data.get("outputs"), dict) else {},
            "mesh_frame": "completed_canonical",
            "raw_trellis_mesh_is_renderable": False,
        },
        "object_pose_trajectory": {
            "state": "full_timeline_rigid_pose_trajectory" if not missing_pose_frames else "rigid_pose_trajectory_with_explicit_missing_frames",
            "pose_report_path": str(pose_report_path),
            "accepted_statuses": sorted(ACCEPTED_RIGID_POSE_STATUSES),
            "pose_rows": pose_rows,
            "frame_count_with_pose": len(pose_frame_ids),
            "missing_pose_frames": missing_pose_frames,
            "full_timeline_rigid_pose_completion": pose_data.get("full_timeline_rigid_pose_completion"),
            "correction_summary": pose_data.get("correction_summary"),
        },
        "mano_constraint_state": {
            "state": "constraint_rows_embedded_for_render_labels",
            "constraint_report_path": str(constraint_report_path),
            "constraint_rows": constraint_rows,
        },
        "temporal_mano_state": {
            "state": "embedded" if temporal_payload is not None else "not_supplied",
            "path": str(temporal_mano_path) if temporal_mano_path is not None else None,
            "payload": temporal_payload,
        },
        "hidden_volume_validation": {
            "state": "embedded" if hidden_payload is not None else "not_supplied",
            "path": str(hidden_validation_path) if hidden_validation_path is not None else None,
            "payload": hidden_payload,
        },
        "projection_contract": {
            "intrinsics_coordinate_rule": "Use camera/hand fx_fy_cx_cy in the annotation frame's source coordinate system; scale fx,cx by rendered_width/source_width and fy,cy by rendered_height/source_height before projection.",
            "source_size_fields": ["frame.source_width/source_height", "raw_video.width/height"],
            "render_size_source": "actual decoded raw_frame_path image size",
            "forbidden_projection_rule": "Do not project a 1408x1408/source-size K directly onto a 960x960 rendered frame.",
        },
        "outputs": {"render_state_json": str(output)},
        "evidence": {
            "total_frames": len(frame_ids),
            "pose_rows": len(pose_rows),
            "constraint_rows": len(constraint_rows),
            "temporal_mano_rows": len(temporal_payload.get("per_frame_states", [])) if isinstance(temporal_payload, dict) and isinstance(temporal_payload.get("per_frame_states"), list) else 0,
            "missing_pose_frame_count": len(missing_pose_frames),
        },
        "total_elapsed_s": time.time() - started,
    }
    write_json(output, state)
    return state


def main() -> None:
    state = build(parse_args())
    print(json.dumps({
        "status": state["status"],
        "method": state["method"],
        "output": state["outputs"]["render_state_json"],
        "evidence": state["evidence"],
    }, indent=2))


if __name__ == "__main__":
    main()
