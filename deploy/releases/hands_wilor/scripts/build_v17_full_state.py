#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_v16_full_pipeline import load_mesh_archive, save_mesh_archive

OBJECT_LIMIT_FLAGS: dict[str, Any] = {
    "multi_object_timeline_ready": False,
    "object_schema_status": "single_manipulated_object_qc",
    "missing_multi_object_roster_required": True,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "object_geometry_status": "partial_visible_surface_or_local_patch_qc",
}

OBJECT_GEOMETRY_SEMANTICS = (
    "Current object geometry can be a visible surface, local contact patch, or legacy single-object mesh stream; "
    "complete manipulated-object mesh reconstruction remains open."
)
OBJECT_POSE_SEMANTICS = (
    "Legacy object center, extent, and local-surface fields are QC evidence fields; complete manipulated-object "
    "geometry and pose estimation remain open."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def object_limit_payload() -> dict[str, Any]:
    return {
        **OBJECT_LIMIT_FLAGS,
        "semantics": "The frame keeps the legacy singular object stream; simultaneous object states remain unimplemented.",
        "geometry_semantics": OBJECT_GEOMETRY_SEMANTICS,
        "pose_semantics": OBJECT_POSE_SEMANTICS,
    }


def apply_object_limit_payload(obj: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    out.update(object_limit_payload())
    return out


def root_limit_payload() -> dict[str, Any]:
    return {
        "artifact_status": "partial",
        "artifact_kind": "full_timeline_evidence_qc_annotation",
        "delivery_role": "qc_only_not_v17_closure",
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "object_pose_semantics": OBJECT_POSE_SEMANTICS,
        "object_geometry_semantics": OBJECT_GEOMETRY_SEMANTICS,
    }


def write_mesh_archive_metadata(path: Path, metadata: dict[str, Any]) -> str:
    with np.load(path) as blob:
        arrays = {name: blob[name] for name in blob.files}
    metadata_path = path.with_name(f"{path.name}.metadata.json")
    payload = {
        **metadata,
        "metadata_path": str(metadata_path),
        "npz_metadata_key": "v17_archive_metadata_json",
    }
    arrays["v17_archive_metadata_json"] = np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    write_json(metadata_path, payload)
    return str(metadata_path)


def array3(value: object) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def frames_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise RuntimeError(f"{path} must contain a frames list")
    return payload, payload["frames"]


def frames_by_index(path: Path) -> dict[int, dict[str, Any]]:
    _payload, frames = frames_payload(path)
    out: dict[int, dict[str, Any]] = {}
    for row_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"{path} frame row {row_i} is not a JSON object")
        idx = frame.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} frame row {row_i} has invalid frame_idx {idx!r}")
        out[idx] = frame
    return out


def graph_selected_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    states = payload.get("states") if isinstance(payload, dict) else None
    if not isinstance(states, list):
        raise RuntimeError(f"{path} must contain graph states")
    out: dict[int, dict[str, Any]] = {}
    for row_i, state in enumerate(states):
        if not isinstance(state, dict):
            raise RuntimeError(f"{path} state row {row_i} is not a JSON object")
        idx = state.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} state row {row_i} has invalid frame_idx {idx!r}")
        selected_id = state.get("selected_measurement_id")
        selected_row = None
        for candidate in state.get("candidates") or []:
            row = candidate.get("row") if isinstance(candidate, dict) else None
            if isinstance(row, dict) and row.get("measurement_id") == selected_id:
                selected_row = row
                break
        out[idx] = {**state, "selected_row": selected_row}
    return out


def load_local_patch_states(paths: tuple[Path, ...]) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    states: dict[int, dict[str, Any]] = {}
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"{path} must contain a JSON list")
        for row_i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"{path} row {row_i} is not a JSON object")
            if row.get("annotation_ready") is not True:
                continue
            idx = row.get("frame_idx")
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise RuntimeError(f"{path} row {row_i} has invalid frame_idx {idx!r}")
            mesh_archive = row.get("mesh_archive")
            if not isinstance(mesh_archive, str) or not mesh_archive:
                raise RuntimeError(f"{path} row {row_i} missing mesh_archive")
            archive_meshes = load_mesh_archive(Path(mesh_archive))
            if idx not in archive_meshes:
                raise RuntimeError(f"{mesh_archive} has no local patch mesh for frame {idx}")
            states[idx] = row
            meshes[idx] = archive_meshes[idx]
    return states, meshes


def load_persistent_surfaces(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    state = load_json(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    npz_path = state.get("canonical_mesh_npz")
    if not isinstance(npz_path, str) or not npz_path:
        raise RuntimeError(f"{path} missing canonical_mesh_npz")
    data = np.load(npz_path)
    required = {"vertices", "faces", "frame_idx", "frame_vertex_start", "frame_vertex_end"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(f"{npz_path} missing keys: {missing}")
    frame_rows = state.get("frame_rows")
    if not isinstance(frame_rows, list):
        raise RuntimeError(f"{path} missing frame_rows")
    vertices = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int32)
    frame_idx = data["frame_idx"].astype(int)
    starts = data["frame_vertex_start"].astype(np.int64)
    ends = data["frame_vertex_end"].astype(np.int64)
    centers = data["object_center_world_m"].astype(np.float64) if "object_center_world_m" in data.files else None
    if centers is not None and centers.shape != (len(frame_idx), 3):
        raise RuntimeError(f"{npz_path} has invalid object_center_world_m shape {centers.shape}")
    if len(frame_rows) != len(frame_idx):
        raise RuntimeError(f"{path} frame_rows and mesh frame_idx lengths differ")
    rows_by_frame: dict[int, dict[str, Any]] = {}
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    face_offset = 0
    for i, raw in enumerate(frame_rows):
        if not isinstance(raw, dict):
            raise RuntimeError(f"{path} frame row {i} is not a JSON object")
        idx = int(frame_idx[i])
        row_face_count = int(raw.get("surface_faces") or 0)
        if row_face_count <= 0:
            raise RuntimeError(f"{path} frame {idx} has invalid surface_faces")
        v0, v1 = int(starts[i]), int(ends[i])
        frame_faces = faces[face_offset : face_offset + row_face_count] - v0
        frame_vertices = vertices[v0:v1]
        center = centers[i] if centers is not None else array3(raw.get("object_center_world_m"))
        if center is None:
            raise RuntimeError(f"{path} frame {idx} has no object_center_world_m for world mesh placement")
        frame_vertices = frame_vertices + center[None, :]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"{path} frame {idx} has empty mesh")
        if int(frame_faces.min()) < 0 or int(frame_faces.max()) >= len(frame_vertices):
            raise RuntimeError(f"{path} frame {idx} has invalid face indices")
        row = dict(raw)
        row["measurement_type"] = "object_persistent_visible_surface_state"
        row["source_file"] = str(path)
        row["canonical_mesh_npz"] = npz_path
        rows_by_frame[idx] = row
        meshes[idx] = (frame_vertices, frame_faces.astype(np.int32))
        face_offset += row_face_count
    if face_offset != len(faces):
        raise RuntimeError(f"{path} consumed {face_offset} faces but archive has {len(faces)}")
    return rows_by_frame, meshes


def replace_anchor_hands(frames: list[dict[str, Any]], repair_by_frame: dict[int, dict[str, Any]]) -> int:
    replaced = 0
    for frame in frames:
        idx = int(frame["frame_idx"])
        repair = repair_by_frame.get(idx)
        if repair is None:
            continue
        hands = repair.get("hands")
        if not isinstance(hands, list):
            raise RuntimeError(f"repair frame {idx} has no hands list")
        frame["hands"] = hands
        frame["v17_hand_state_source"] = {
            "status": "replaced_with_v17_anchor_repair",
            "source_frame": idx,
            "hand_count": len(hands),
        }
        replaced += 1
    return replaced


def mark_graph_contacts(frames: list[dict[str, Any]], graph_rows: dict[int, dict[str, Any]]) -> int:
    marked = 0
    for frame in frames:
        idx = int(frame["frame_idx"])
        row = graph_rows.get(idx)
        if row is None:
            continue
        selected = row.get("selected_row")
        frame["v17_contact_state"] = {
            "status": row.get("status"),
            "expected_contact": row.get("expected_contact"),
            "selected_measurement_id": row.get("selected_measurement_id"),
            "selected_contact_state_measurement": row.get("selected_contact_state_measurement"),
            "selected_requires_temporal_validation": row.get("selected_requires_temporal_validation"),
            "selected_source": selected.get("source_contact_measurements") if isinstance(selected, dict) else None,
            "local_patch_state_id": selected.get("local_patch_state_id") if isinstance(selected, dict) else None,
        }
        marked += 1
    return marked


def patch_object_states(
    frames: list[dict[str, Any]],
    persistent_rows: dict[int, dict[str, Any]],
    local_patch_rows: dict[int, dict[str, Any]],
) -> tuple[int, int]:
    persistent_count = 0
    patch_count = 0
    for frame in frames:
        idx = int(frame["frame_idx"])
        obj = dict(frame.get("object") or {})
        obj = apply_object_limit_payload(obj)
        persistent = persistent_rows.get(idx)
        local_patch = local_patch_rows.get(idx)
        if persistent is not None:
            obj["v17_shape_state"] = {
                "status": "persistent_visible_surface",
                "object_id": persistent.get("object_id"),
                "object_center_world_m": persistent.get("object_center_world_m"),
                "surface_vertices": persistent.get("surface_vertices"),
                "surface_faces": persistent.get("surface_faces"),
                "source_mask_path": persistent.get("source_mask_path"),
                "canonical_mesh_npz": persistent.get("canonical_mesh_npz"),
            }
            obj["mesh_state"] = "v17_persistent_visible_surface"
            persistent_count += 1
        if local_patch is not None:
            obj["v17_local_contact_patch_state"] = {
                "status": local_patch.get("status"),
                "entity_id": local_patch.get("entity_id"),
                "mesh_vertices": local_patch.get("mesh_vertices"),
                "mesh_faces": local_patch.get("mesh_faces"),
                "patch_mask_path": local_patch.get("patch_mask_path"),
                "source_object_mask_path": local_patch.get("source_object_mask_path"),
                "contact_state_measurement": local_patch.get("contact_state_measurement"),
                "hand_object_mesh_distance_m": local_patch.get("hand_object_mesh_distance_m"),
            }
            obj["mesh_state"] = "v17_local_deformable_contact_patch"
            patch_count += 1
        frame["object"] = obj
    return persistent_count, patch_count


def annotate_v17_captions(frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        caption = str(frame.get("caption") or "").strip()
        labels: list[str] = ["V17"]
        contact = frame.get("v17_contact_state")
        if isinstance(contact, dict):
            status = contact.get("status")
            if status == "accepted_contact":
                labels.append("contact")
            elif status == "accepted_no_contact":
                labels.append("no-contact")
        obj = frame.get("object")
        if isinstance(obj, dict):
            if obj.get("v17_local_contact_patch_state") is not None:
                labels.append("local object patch")
            if obj.get("v17_shape_state") is not None:
                labels.append("persistent object mesh")
        if not caption:
            if "contact" in labels:
                caption = "Accepted hand-object contact state from V17 graph evidence."
            elif "no-contact" in labels:
                caption = "Accepted no-contact state from V17 graph evidence."
            elif "persistent object mesh" in labels:
                caption = "Persistent object surface state carried through the full timeline."
            else:
                caption = "Full-video V17 evidence/QC state; annotation closure remains open."
        frame["caption"] = f"{'; '.join(labels)}: {caption}"
        frame["objects_status"] = {
            "status": "single_manipulated_object_qc",
            **object_limit_payload(),
        }


def merged_mesh_archive(
    base_archive: Path,
    output_archive: Path,
    frame_count: int,
    persistent_meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    local_patch_meshes: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    meshes = load_mesh_archive(base_archive)
    for idx, mesh in persistent_meshes.items():
        meshes[idx] = mesh
    for idx, mesh in local_patch_meshes.items():
        meshes[idx] = mesh
    frames = sorted(meshes)
    save_mesh_archive(output_archive, frames, [meshes[idx][0] for idx in frames], [meshes[idx][1] for idx in frames])
    metadata = {
        "output_archive": str(output_archive),
        "artifact_status": "partial",
        "artifact_kind": "full_timeline_evidence_qc_mesh_archive",
        "delivery_role": "qc_only_not_v17_closure",
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "frame_count": int(frame_count),
        "mesh_frames": len(frames),
        "missing_mesh_frame_count": int(frame_count - len(frames)),
        "first_frame": int(frames[0]) if frames else None,
        "last_frame": int(frames[-1]) if frames else None,
        "mesh_semantics": OBJECT_GEOMETRY_SEMANTICS,
        "persistent_replaced_frames": len(persistent_meshes),
        "local_patch_replaced_frames": len(local_patch_meshes),
    }
    metadata["metadata_path"] = write_mesh_archive_metadata(output_archive, metadata)
    return metadata


def case_spec(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "trash_1050":
        return {
            "v16_manifest": args.trash_v16_manifest,
            "hand_repair_annotations": args.trash_hand_repair_annotations,
            "contact_graph": args.trash_contact_graph,
            "local_patch_states": args.trash_local_patch_states,
            "persistent_shape": None,
        }
    if name == "task5_tomato_960":
        return {
            "v16_manifest": args.tomato_v16_manifest,
            "hand_repair_annotations": None,
            "contact_graph": args.tomato_contact_graph,
            "local_patch_states": (),
            "persistent_shape": args.tomato_persistent_shape,
        }
    raise RuntimeError(f"unknown case {name}")


def build_case(name: str, spec: dict[str, Any], output_root: Path) -> dict[str, Any]:
    manifest = load_json(spec["v16_manifest"])
    annotations_path = Path(manifest["annotations"])
    base_archive = Path(manifest["object_mesh_archive"])
    payload, frames = frames_payload(annotations_path)
    if int(manifest["raw_frame_count"]) != len(frames):
        raise RuntimeError(f"{name} annotation count does not match manifest raw frame count")

    repair_count = 0
    if spec["hand_repair_annotations"] is not None:
        repair_count = replace_anchor_hands(frames, frames_by_index(spec["hand_repair_annotations"]))

    graph_count = 0
    if spec["contact_graph"] is not None:
        graph_count = mark_graph_contacts(frames, graph_selected_rows(spec["contact_graph"]))

    local_rows: dict[int, dict[str, Any]] = {}
    local_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if spec["local_patch_states"]:
        local_rows, local_meshes = load_local_patch_states(tuple(spec["local_patch_states"]))

    persistent_rows: dict[int, dict[str, Any]] = {}
    persistent_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if spec["persistent_shape"] is not None:
        persistent_rows, persistent_meshes = load_persistent_surfaces(spec["persistent_shape"])

    persistent_count, patch_count = patch_object_states(frames, persistent_rows, local_rows)
    annotate_v17_captions(frames)
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    annotations_out = case_dir / "annotations_v17_full.json"
    payload["frames"] = frames
    payload.update(root_limit_payload())
    payload["v17_state_note"] = {
        "status": "evidence_layer_qc_state",
        "artifact_kind": "full_timeline_evidence_qc_state",
        "delivery_role": "qc_only_not_v17_closure",
        "claim": "full-timeline evidence/QC state assembled from accepted V17 measurements; annotation closure and the integrated nonlinear solver remain open",
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "object_pose_semantics": OBJECT_POSE_SEMANTICS,
        "object_geometry_semantics": OBJECT_GEOMETRY_SEMANTICS,
    }
    write_json(annotations_out, payload)

    mesh_archive = case_dir / "object_meshes_v17_full.npz"
    mesh_report = merged_mesh_archive(base_archive, mesh_archive, int(manifest["raw_frame_count"]), persistent_meshes, local_meshes)
    report = {
        "case": name,
        "status": "evidence_qc_state_built",
        "artifact_status": "partial",
        "artifact_kind": "full_timeline_evidence_qc_state",
        "delivery_role": "qc_only_not_v17_closure",
        "method": "build_v17_full_state",
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "object_pose_semantics": OBJECT_POSE_SEMANTICS,
        "object_geometry_semantics": OBJECT_GEOMETRY_SEMANTICS,
        "v16_manifest": str(spec["v16_manifest"]),
        "raw_frame_count": int(manifest["raw_frame_count"]),
        "annotations": str(annotations_out),
        "object_mesh_archive": str(mesh_archive),
        "hand_repair_frames": repair_count,
        "contact_graph_frames": graph_count,
        "persistent_object_frames": persistent_count,
        "local_contact_patch_frames": patch_count,
        "mesh_report": mesh_report,
        "solver_status": "integrated_full_timeline_factor_graph_open",
    }
    write_json(case_dir / "v17_full_state_manifest.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [build_case(name, case_spec(name, args), args.output_root) for name in args.cases]
    built = all(row["status"] == "evidence_qc_state_built" for row in reports)
    summary = {
        "status": "evidence_qc_state_built_collection" if built else "failed",
        "artifact_status": "partial",
        "artifact_kind": "full_timeline_evidence_qc_state_collection",
        "delivery_role": "qc_only_not_v17_closure",
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        "multi_object_timeline_ready": False,
        "object_schema_status": "single_manipulated_object_qc",
        "missing_multi_object_roster_required": True,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "object_geometry_status": "partial_visible_surface_or_local_patch_qc",
        "method": "build_v17_full_state",
        "cases": reports,
    }
    write_json(args.output_root / "v17_full_state_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_full_state"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument(
        "--trash-v16-manifest",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v16_full_pipeline/trash_1050/v16_full_pipeline_manifest.json"),
    )
    parser.add_argument(
        "--trash-hand-repair-annotations",
        type=Path,
        default=Path(
            "/data2/ego_annotation_outputs/v17_hand_evidence/trash_1050/"
            "anchor_graph_repair_v1/annotations_v17_anchor_graph_repair.json"
        ),
    )
    parser.add_argument(
        "--trash-contact-graph",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_measurements/trash_1050/anchor_contact_state_graph_v4.json"),
    )
    parser.add_argument(
        "--trash-local-patch-states",
        type=Path,
        nargs="*",
        default=[
            Path(
                "/data2/ego_annotation_outputs/v17_object_plan/trash_1050/"
                "local_contact_patch_black_bag_182_graph_hand_v1/local_contact_patch_states.json"
            ),
            Path(
                "/data2/ego_annotation_outputs/v17_object_plan/trash_1050/"
                "local_contact_patch_white_bag_856_graph_hand_v1/local_contact_patch_states.json"
            ),
        ],
    )
    parser.add_argument(
        "--tomato-v16-manifest",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v16_full_pipeline/task5_tomato_960/v16_full_pipeline_manifest.json"),
    )
    parser.add_argument(
        "--tomato-persistent-shape",
        type=Path,
        default=Path(
            "/data2/ego_annotation_outputs/v17_object_plan/task5_tomato_960/"
            "persistent_object_shape_obj_tomato_v1/persistent_object_shape_state.json"
        ),
    )
    parser.add_argument(
        "--tomato-contact-graph",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_measurements/task5_tomato_960/anchor_contact_state_graph_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
