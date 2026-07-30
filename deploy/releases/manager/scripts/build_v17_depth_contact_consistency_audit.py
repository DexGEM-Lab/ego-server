#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingImports]

from build_v17_geometry_reconstruction_results import load_obj_mesh, transform_points


STATUS = "v17_depth_contact_consistency_audit_qc"
CLAIM = (
    "This artifact checks whether accepted short-segment reconstruction meshes, UniDepth visible surfaces, "
    "legacy contact-mode gaps, and graph MANO hand geometry live in one metric depth/contact state. It is an "
    "audit of depth ownership, not a repair and not a solver."
)

FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


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


def optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return require_str(value, label)


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a JSON boolean")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def shared_depth_state_ready(row: dict[str, Any]) -> bool:
    checks = require_dict(row.get("same_depth_state_checks"), "same depth checks")
    return bool(
        checks.get("mesh_matches_visible_unidepth") is True
        and checks.get("legacy_object_depth_matches_visible_unidepth") is True
        and checks.get("all_hand_depths_match_visible_unidepth") is True
    )


def shared_depth_contact_ready(row: dict[str, Any]) -> bool:
    return bool(
        shared_depth_state_ready(row)
        and require_int(
            row.get("reconstructed_mesh_contact_candidate_rows"),
            "reconstructed mesh contact candidate rows",
        )
        > 0
    )


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    if not vals:
        return {"count": 0}

    def pct(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

    return {
        "count": len(vals),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 32) - 1)


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    valid = np.isfinite(area) & (area > 0.0)
    face_ids = np.flatnonzero(valid)
    if len(face_ids) == 0:
        raise RuntimeError("mesh has no finite-area faces")
    probs = area[valid] / float(np.sum(area[valid]))
    rng = np.random.default_rng(seed)
    replace = int(count) > len(face_ids)
    chosen = rng.choice(face_ids, size=int(count), replace=replace, p=probs)
    tri = vertices[faces[chosen]]
    r1 = rng.random(len(chosen))
    r2 = rng.random(len(chosen))
    s1 = np.sqrt(r1)
    return (1.0 - s1)[:, None] * tri[:, 0] + (s1 * (1.0 - r2))[:, None] * tri[:, 1] + (s1 * r2)[:, None] * tri[:, 2]


def read_depth_m(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"failed to read depth image: {path}")
    if depth.ndim != 2:
        raise RuntimeError(f"depth image must be single-channel: {path}")
    return depth.astype(np.float64) / 1000.0


def frame_index(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {require_int(frame.get("frame_idx"), "frame_idx"): frame for frame in frames}


def object_track_frames(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        require_int(row.get("frame_idx"), "track frame_idx"): row
        for row in require_list(manifest.get("frames"), "object-track frames")
    }


def contact_mode_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "contact-mode rows")):
        row = require_dict(raw, f"contact-mode rows[{i}]")
        out[(require_int(row.get("frame_idx"), "contact-mode frame_idx"), require_str(row.get("side"), "contact-mode side"))] = row
    return out


def multi_object_contact_index(report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "multi-object contact rows")):
        row = require_dict(raw, f"multi-object contact rows[{i}]")
        out[
            (
                require_int(row.get("frame_idx"), "multi-object contact frame_idx"),
                require_str(row.get("object_id"), "multi-object contact object_id"),
                require_str(row.get("hand_side"), "multi-object contact hand_side"),
            )
        ] = row
    return out


def object_owner_state(frame: dict[str, Any], object_id: str, track_id: str) -> dict[str, Any]:
    legacy = require_dict(frame.get("object"), "legacy object")
    legacy_object_id = optional_str(legacy.get("object_id"), "legacy object_id")
    legacy_track_id = optional_str(legacy.get("track_id"), "legacy track_id")
    legacy_label = optional_str(legacy.get("label"), "legacy label")
    objects = []
    reconstructed_present = False
    reconstructed_visible = False
    for i, raw in enumerate(require_list(frame.get("objects"), "multi-object frame objects")):
        row = require_dict(raw, f"multi-object frame objects[{i}]")
        row_object_id = require_str(row.get("object_id"), "multi-object frame object_id")
        row_track_id = require_str(row.get("track_id"), "multi-object frame track_id")
        active = require_bool(row.get("active"), "multi-object frame active")
        visible = require_bool(row.get("visible"), "multi-object frame visible")
        objects.append(
            {
                "object_id": row_object_id,
                "track_id": row_track_id,
                "name": optional_str(row.get("name"), "multi-object frame name"),
                "active": active,
                "visible": visible,
                "mask_evidence_status": optional_str(
                    row.get("mask_evidence_status"),
                    "multi-object mask_evidence_status",
                ),
            }
        )
        if row_object_id == object_id:
            reconstructed_present = bool(reconstructed_present or active)
            reconstructed_visible = bool(reconstructed_visible or (active and visible))
    legacy_matches = bool(legacy_object_id == object_id or legacy_track_id == track_id)
    return {
        "reconstructed_object_id": object_id,
        "reconstructed_track_id": track_id,
        "legacy_single_object": {
            "object_id": legacy_object_id,
            "track_id": legacy_track_id,
            "label": legacy_label,
        },
        "legacy_single_object_matches_reconstructed_object": legacy_matches,
        "multi_object_frame_objects": objects,
        "reconstructed_object_present_in_multi_object_timeline": reconstructed_present,
        "reconstructed_object_visible_in_multi_object_timeline": reconstructed_visible,
    }


def hand_depth_rows(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hand in require_list(frame.get("hands"), "frame hands"):
        hand_row = require_dict(hand, "hand")
        side = require_str(hand_row.get("side"), "hand side")
        source_vertices = np.asarray(hand_row.get("vertices_source_camera_m", []), dtype=np.float64)
        world_vertices = np.asarray(hand_row.get("vertices_world_m", []), dtype=np.float64)
        if source_vertices.ndim != 2 or source_vertices.shape[1] != 3 or len(source_vertices) == 0:
            continue
        if world_vertices.ndim != 2 or world_vertices.shape[1] != 3 or len(world_vertices) == 0:
            continue
        rows.append(
            {
                "side": side,
                "source_depth_m": summarize([float(v) for v in source_vertices[:, 2].tolist()]),
                "world_vertices": world_vertices,
            }
        )
    return rows


def accepted_reconstruction_jobs(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        require_dict(row, "geometry reconstruction result job")
        for row in require_list(report.get("jobs"), "geometry reconstruction result jobs")
        if require_dict(row, "geometry reconstruction result job").get("accepted_reconstruction_result") is True
    ]


def frame_row(
    *,
    case: str,
    job: dict[str, Any],
    projection_row: dict[str, Any],
    annotation_frame: dict[str, Any],
    object_track_frame: dict[str, Any],
    contact_rows: dict[tuple[int, str], dict[str, Any]],
    multi_object_contact_rows: dict[tuple[int, str, str], dict[str, Any]],
    mesh_samples: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(projection_row.get("frame_idx"), "projection frame_idx")
    dataset_index = require_int(projection_row.get("dataset_index"), "projection dataset_index")
    pose_path = Path(require_str(job.get("bundlesdf_output_dir"), "bundlesdf_output_dir")) / "ob_in_cam" / f"{dataset_index:06d}.txt"
    object_in_cam = np.loadtxt(pose_path).astype(np.float64)
    if object_in_cam.shape != (4, 4) or not np.isfinite(object_in_cam).all():
        raise RuntimeError(f"invalid BundleSDF pose: {pose_path}")
    camera = require_dict(annotation_frame.get("camera"), "annotation camera")
    T_world_camera = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
        raise RuntimeError(f"invalid T_world_camera_metric for frame {frame_idx}")
    mesh_camera = transform_points(mesh_samples, object_in_cam)
    mesh_world = transform_points(mesh_camera, T_world_camera)
    mesh_tree = cKDTree(mesh_world)
    object_depth = read_depth_m(Path(require_str(object_track_frame.get("depth"), "object-track depth")))
    mask = cv2.imread(require_str(object_track_frame.get("mask"), "object-track mask"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object-track mask: {object_track_frame.get('mask')}")
    valid = (mask > 0) & np.isfinite(object_depth) & (object_depth > 0.05)
    if not np.any(valid):
        raise RuntimeError(f"object-track frame {frame_idx} has no valid mask depth")
    visible_depth_m = object_depth[valid]
    object_center_depth = optional_float(
        require_dict(annotation_frame.get("object"), "legacy object").get("depth_m"),
        "legacy object depth_m",
    )
    legacy_mesh_depth = optional_float(
        require_dict(require_dict(annotation_frame.get("object"), "legacy object").get("mesh_qc"), "legacy mesh_qc").get("depth_median_m"),
        "legacy mesh_qc depth_median_m",
    )
    hand_rows = []
    reconstructed_contact_candidates = 0
    multi_object_contact_candidates = 0
    near_reconstructed_mesh_rows = 0
    legacy_contact_ready_rows = 0
    for hand in hand_depth_rows(annotation_frame):
        side = require_str(hand.get("side"), "hand side")
        world_vertices = np.asarray(hand["world_vertices"], dtype=np.float64)
        hand_to_mesh = mesh_tree.query(world_vertices, k=1, workers=-1)[0]
        hand_tree = cKDTree(world_vertices)
        mesh_to_hand = hand_tree.query(mesh_world, k=1, workers=-1)[0]
        min_distance = float(min(float(np.min(hand_to_mesh)), float(np.min(mesh_to_hand))))
        contact_row = contact_rows.get((frame_idx, side), {})
        contact_gap = optional_float(contact_row.get("gap_p05_m"), "contact-mode gap_p05_m")
        legacy_contact_ready = bool(contact_row.get("contact_factor_ready") is True)
        if legacy_contact_ready:
            legacy_contact_ready_rows += 1
        multi_contact_row = multi_object_contact_rows.get(
            (frame_idx, require_str(job.get("object_id"), "job object_id"), side),
            {},
        )
        multi_object_contact_ready = bool(multi_contact_row.get("contact_factor_ready") is True)
        if multi_object_contact_ready:
            multi_object_contact_candidates += 1
        near_reconstructed = bool(min_distance <= float(args.near_reconstructed_mesh_m))
        reconstructed_contact = bool(
            near_reconstructed
            and legacy_contact_ready
            and contact_gap is not None
            and contact_gap <= float(args.near_legacy_contact_m)
        )
        if near_reconstructed:
            near_reconstructed_mesh_rows += 1
        if reconstructed_contact:
            reconstructed_contact_candidates += 1
        hand_rows.append(
            {
                "side": side,
                "source_depth_m": hand["source_depth_m"],
                "reconstructed_mesh_distance_m": {
                    "min_symmetric": min_distance,
                    "hand_to_mesh": summarize([float(v) for v in hand_to_mesh.tolist()]),
                    "mesh_to_hand": summarize([float(v) for v in mesh_to_hand.tolist()]),
                },
                "legacy_contact_mode": {
                    "mode": contact_row.get("mode"),
                    "contact_factor_ready": legacy_contact_ready,
                    "gap_p05_m": contact_gap,
                    "gap_min_m": optional_float(contact_row.get("gap_min_m"), "contact-mode gap_min_m"),
                    "mask_distance_median_px": optional_float(
                        contact_row.get("mask_distance_median_px"),
                        "contact-mode mask_distance_median_px",
                    ),
                },
                "multi_object_contact_evidence": {
                    "contact_mode_state": multi_contact_row.get("contact_mode_state"),
                    "geometry_source": multi_contact_row.get("geometry_source"),
                    "contact_factor_ready": multi_object_contact_ready,
                    "visible_surface_distance_candidate": bool(
                        multi_contact_row.get("visible_surface_distance_candidate") is True
                    ),
                    "contact_distance_candidate": bool(multi_contact_row.get("contact_distance_candidate") is True),
                    "min_symmetric_distance_m": optional_float(
                        multi_contact_row.get("min_symmetric_distance_m"),
                        "multi-object min_symmetric_distance_m",
                    ),
                },
                "near_reconstructed_mesh": near_reconstructed,
                "reconstructed_mesh_contact_candidate": reconstructed_contact,
                **FALSE_READY,
            }
        )
    hand_medians = [
        finite_float(row["source_depth_m"].get("median"), "hand depth median")
        for row in hand_rows
        if row["source_depth_m"].get("count", 0) > 0
    ]
    observed_depth_median = float(np.median(visible_depth_m))
    mesh_depth_median = float(np.median(mesh_camera[:, 2]))
    front_surface_depth_abs_p95 = finite_float(
        projection_row.get("front_surface_depth_abs_p95_m"),
        "front surface depth abs p95",
    )
    depth_delta_values = hand_medians + ([object_center_depth] if object_center_depth is not None else [])
    depth_deltas = [abs(float(value) - observed_depth_median) for value in depth_delta_values]
    row = {
        "case": case,
        "job_id": require_str(job.get("job_id"), "job_id"),
        "object_id": require_str(job.get("object_id"), "object_id"),
        "track_id": require_str(job.get("track_id"), "track_id"),
        "frame_idx": frame_idx,
        "dataset_index": dataset_index,
        "pose_path": str(pose_path),
        "reconstructed_mesh_camera_depth_m": summarize([float(v) for v in mesh_camera[:, 2].tolist()]),
        "visible_object_unidepth_m": summarize([float(v) for v in visible_depth_m.tolist()]),
        "reconstructed_mesh_front_surface_depth_abs_m": {
            "median": finite_float(
                projection_row.get("front_surface_depth_abs_median_m"),
                "front surface depth abs median",
            ),
            "p95": front_surface_depth_abs_p95,
            "sample_count": require_int(
                projection_row.get("front_surface_depth_sample_count"),
                "front surface depth sample count",
            ),
        },
        "legacy_object_center_depth_m": object_center_depth,
        "legacy_object_mesh_depth_median_m": legacy_mesh_depth,
        "hand_rows": hand_rows,
        "hand_row_count": len(hand_rows),
        "near_reconstructed_mesh_hand_rows": near_reconstructed_mesh_rows,
        "reconstructed_mesh_contact_candidate_rows": reconstructed_contact_candidates,
        "legacy_contact_ready_hand_rows": legacy_contact_ready_rows,
        "multi_object_reconstructed_object_contact_candidate_rows": multi_object_contact_candidates,
        "object_owner_state": object_owner_state(
            annotation_frame,
            require_str(job.get("object_id"), "job object_id"),
            require_str(job.get("track_id"), "job track_id"),
        ),
        "same_depth_state_checks": {
            "mesh_matches_visible_unidepth": bool(
                front_surface_depth_abs_p95 <= float(args.max_same_state_depth_delta_m)
            ),
            "legacy_object_depth_matches_visible_unidepth": bool(
                object_center_depth is not None
                and abs(object_center_depth - observed_depth_median) <= float(args.max_same_state_depth_delta_m)
            ),
            "all_hand_depths_match_visible_unidepth": bool(
                hand_medians
                and max(abs(value - observed_depth_median) for value in hand_medians)
                <= float(args.max_same_state_depth_delta_m)
            ),
        },
        "max_depth_delta_to_visible_unidepth_m": max(depth_deltas) if depth_deltas else None,
        **FALSE_READY,
    }
    row["shared_depth_state_ready"] = shared_depth_state_ready(row)
    row["shared_depth_contact_state_ready"] = shared_depth_contact_ready(row)
    return row


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    reconstruction_path = existing_path(
        args.geometry_reconstruction_results_root / case / "v17_geometry_reconstruction_results_report.json",
        f"{case} geometry reconstruction results report",
    )
    annotation_path = existing_path(
        args.graph_root / case / "annotations_v17_full_timeline_graph.json",
        f"{case} graph annotations",
    )
    contact_path = existing_path(
        args.contact_mode_graph_root / case / "v17_contact_mode_graph_report.json",
        f"{case} contact-mode graph report",
    )
    multi_object_contact_path = existing_path(
        args.multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
        f"{case} multi-object contact evidence report",
    )
    reconstruction = require_dict(load_json(reconstruction_path), f"{case} reconstruction results")
    annotation = require_dict(load_json(annotation_path), f"{case} graph annotations")
    contact = require_dict(load_json(contact_path), f"{case} contact-mode report")
    multi_object_contact = require_dict(
        load_json(multi_object_contact_path),
        f"{case} multi-object contact evidence",
    )
    annotation_by_frame = frame_index(
        [require_dict(row, f"{case} annotation frames") for row in require_list(annotation.get("frames"), "annotation frames")]
    )
    contact_rows = contact_mode_index(contact)
    multi_object_contact_rows = multi_object_contact_index(multi_object_contact)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for job in accepted_reconstruction_jobs(reconstruction):
        object_manifest_path = existing_path(
            args.object_track_dataset_root / case / require_str(job.get("track_id"), "job track_id") / "manifest.json",
            f"{case} {job.get('track_id')} object-track manifest",
        )
        object_manifest = require_dict(load_json(object_manifest_path), "object-track manifest")
        object_track_by_frame = object_track_frames(object_manifest)
        vertices, faces = load_obj_mesh(Path(require_str(job.get("mesh_path"), "job mesh_path")))
        mesh_samples = sample_mesh_surface(
            vertices,
            faces,
            int(args.mesh_surface_samples),
            stable_seed(case, job.get("job_id"), "mesh_surface"),
        )
        job_rows = []
        for projection in require_list(
            require_dict(job.get("projection_qc"), "job projection_qc").get("rows"),
            "projection rows",
        ):
            projection_row = require_dict(projection, "projection row")
            frame_idx = require_int(projection_row.get("frame_idx"), "projection frame_idx")
            if frame_idx not in annotation_by_frame:
                raise RuntimeError(f"{case} frame {frame_idx} missing graph annotation")
            if frame_idx not in object_track_by_frame:
                raise RuntimeError(f"{case} frame {frame_idx} missing object-track manifest row")
            row = frame_row(
                case=case,
                job=job,
                projection_row=projection_row,
                annotation_frame=annotation_by_frame[frame_idx],
                object_track_frame=object_track_by_frame[frame_idx],
                contact_rows=contact_rows,
                multi_object_contact_rows=multi_object_contact_rows,
                mesh_samples=mesh_samples,
                args=args,
            )
            rows.append(row)
            job_rows.append(row)
        jobs.append(
            {
                "job_id": require_str(job.get("job_id"), "job_id"),
                "object_id": require_str(job.get("object_id"), "object_id"),
                "track_id": require_str(job.get("track_id"), "track_id"),
                "first_frame": require_int(job.get("first_frame"), "first_frame"),
                "last_frame": require_int(job.get("last_frame"), "last_frame"),
                "frame_count": require_int(job.get("frame_count"), "frame_count"),
                "accepted_reconstruction_result": True,
                "evaluated_frame_count": len(job_rows),
                "near_reconstructed_mesh_hand_rows": sum(
                    require_int(row.get("near_reconstructed_mesh_hand_rows"), "near reconstructed rows")
                    for row in job_rows
                ),
                "reconstructed_mesh_contact_candidate_rows": sum(
                    require_int(row.get("reconstructed_mesh_contact_candidate_rows"), "contact candidate rows")
                    for row in job_rows
                ),
                "legacy_contact_ready_hand_rows": sum(
                    require_int(row.get("legacy_contact_ready_hand_rows"), "legacy contact ready hand rows")
                    for row in job_rows
                ),
                "multi_object_reconstructed_object_contact_candidate_rows": sum(
                    require_int(
                        row.get("multi_object_reconstructed_object_contact_candidate_rows"),
                        "multi-object contact candidate rows",
                    )
                    for row in job_rows
                ),
                "legacy_owner_mismatch_frame_count": sum(
                    1
                    for row in job_rows
                    if require_dict(row.get("object_owner_state"), "object owner state").get(
                        "legacy_single_object_matches_reconstructed_object"
                    )
                    is False
                ),
                "reconstructed_mesh_to_hand_min_m": summarize(
                    [
                        finite_float(hand["reconstructed_mesh_distance_m"]["min_symmetric"], "min reconstructed distance")
                        for row in job_rows
                        for hand in require_list(row.get("hand_rows"), "hand rows")
                    ]
                ),
                "visible_unidepth_m": summarize(
                    [
                        finite_float(row["visible_object_unidepth_m"].get("median"), "visible depth median")
                        for row in job_rows
                    ]
                ),
                "reconstructed_mesh_camera_depth_m": summarize(
                    [
                        finite_float(row["reconstructed_mesh_camera_depth_m"].get("median"), "mesh depth median")
                        for row in job_rows
                    ]
                ),
                "reconstructed_mesh_front_surface_depth_abs_p95_m": summarize(
                    [
                        finite_float(
                            require_dict(
                                row.get("reconstructed_mesh_front_surface_depth_abs_m"),
                                "front surface depth",
                            ).get("p95"),
                            "front surface depth p95",
                        )
                        for row in job_rows
                    ]
                ),
                "legacy_object_center_depth_m": summarize(
                    [
                        finite_float(row.get("legacy_object_center_depth_m"), "legacy object center depth")
                        for row in job_rows
                        if row.get("legacy_object_center_depth_m") is not None
                    ]
                ),
                "hand_source_depth_m": summarize(
                    [
                        finite_float(hand["source_depth_m"].get("median"), "hand source depth median")
                        for row in job_rows
                        for hand in require_list(row.get("hand_rows"), "hand rows")
                        if hand["source_depth_m"].get("count", 0) > 0
                    ]
                ),
                "max_depth_delta_to_visible_unidepth_m": summarize(
                    [
                        finite_float(row.get("max_depth_delta_to_visible_unidepth_m"), "max depth delta")
                        for row in job_rows
                        if row.get("max_depth_delta_to_visible_unidepth_m") is not None
                    ]
                ),
                "shared_depth_state_ready_frame_count": sum(1 for row in job_rows if shared_depth_state_ready(row)),
                "depth_owner_incompatibility_count": sum(
                    1
                    for row in job_rows
                    if any(
                        value is False
                        for value in require_dict(row.get("same_depth_state_checks"), "same depth checks").values()
                    )
                ),
                "shared_depth_contact_state_ready": any(shared_depth_contact_ready(row) for row in job_rows),
                **FALSE_READY,
            }
        )
    report = {
        "method": "build_v17_depth_contact_consistency_audit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_geometry_reconstruction_results_report": str(reconstruction_path),
        "source_graph_annotations": str(annotation_path),
        "source_contact_mode_graph_report": str(contact_path),
        "source_multi_object_contact_evidence_report": str(multi_object_contact_path),
        "accepted_reconstruction_job_count": len(jobs),
        "evaluated_frame_count": len(rows),
        "evaluated_hand_rows": sum(len(require_list(row.get("hand_rows"), "hand rows")) for row in rows),
        "near_reconstructed_mesh_hand_rows": sum(
            require_int(row.get("near_reconstructed_mesh_hand_rows"), "near reconstructed mesh rows")
            for row in rows
        ),
        "reconstructed_mesh_contact_candidate_rows": sum(
            require_int(row.get("reconstructed_mesh_contact_candidate_rows"), "mesh contact candidate rows")
            for row in rows
        ),
        "legacy_contact_ready_hand_rows": sum(
            require_int(row.get("legacy_contact_ready_hand_rows"), "legacy contact ready hand rows")
            for row in rows
        ),
        "multi_object_reconstructed_object_contact_candidate_rows": sum(
            require_int(
                row.get("multi_object_reconstructed_object_contact_candidate_rows"),
                "multi-object reconstructed object contact candidate rows",
            )
            for row in rows
        ),
        "legacy_owner_mismatch_frame_count": sum(
            1
            for row in rows
            if require_dict(row.get("object_owner_state"), "object owner state").get(
                "legacy_single_object_matches_reconstructed_object"
            )
            is False
        ),
        "shared_depth_state_ready_frame_count": sum(1 for row in rows if shared_depth_state_ready(row)),
        "depth_owner_incompatibility_count": sum(
            1
            for row in rows
            if any(
                value is False
                for value in require_dict(row.get("same_depth_state_checks"), "same depth checks").values()
            )
        ),
        "reconstructed_mesh_to_hand_min_m": summarize(
            [
                finite_float(hand["reconstructed_mesh_distance_m"]["min_symmetric"], "min reconstructed distance")
                for row in rows
                for hand in require_list(row.get("hand_rows"), "hand rows")
            ]
        ),
        "visible_unidepth_m": summarize(
            [finite_float(row["visible_object_unidepth_m"].get("median"), "visible depth median") for row in rows]
        ),
        "reconstructed_mesh_camera_depth_m": summarize(
            [
                finite_float(row["reconstructed_mesh_camera_depth_m"].get("median"), "mesh depth median")
                for row in rows
            ]
        ),
        "reconstructed_mesh_front_surface_depth_abs_p95_m": summarize(
            [
                finite_float(
                    require_dict(row.get("reconstructed_mesh_front_surface_depth_abs_m"), "front surface depth").get(
                        "p95"
                    ),
                    "front surface depth p95",
                )
                for row in rows
            ]
        ),
        "legacy_object_center_depth_m": summarize(
            [
                finite_float(row.get("legacy_object_center_depth_m"), "legacy object center depth")
                for row in rows
                if row.get("legacy_object_center_depth_m") is not None
            ]
        ),
        "hand_source_depth_m": summarize(
            [
                finite_float(hand["source_depth_m"].get("median"), "hand source depth median")
                for row in rows
                for hand in require_list(row.get("hand_rows"), "hand rows")
                if hand["source_depth_m"].get("count", 0) > 0
            ]
        ),
        "max_depth_delta_to_visible_unidepth_m": summarize(
            [
                finite_float(row.get("max_depth_delta_to_visible_unidepth_m"), "max depth delta")
                for row in rows
                if row.get("max_depth_delta_to_visible_unidepth_m") is not None
            ]
        ),
        "jobs": jobs,
        "rows": rows,
        "parameters": {
            "mesh_surface_samples": int(args.mesh_surface_samples),
            "near_reconstructed_mesh_m": float(args.near_reconstructed_mesh_m),
            "near_legacy_contact_m": float(args.near_legacy_contact_m),
            "max_same_state_depth_delta_m": float(args.max_same_state_depth_delta_m),
        },
        "shared_depth_contact_state_ready": any(shared_depth_contact_ready(row) for row in rows),
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_depth_contact_consistency_audit_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.geometry_reconstruction_results_root / "v17_geometry_reconstruction_results_summary.json",
        "geometry reconstruction results summary",
    )
    summary = require_dict(load_json(summary_path), "geometry reconstruction results summary")
    reports = [
        build_case(
            require_str(require_dict(row, f"summary cases[{i}]").get("case"), f"summary cases[{i}].case"),
            args,
        )
        for i, row in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_depth_contact_consistency_audit",
        "status": STATUS,
        "claim": CLAIM,
        "source_geometry_reconstruction_results_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_depth_contact_consistency_audit_report.json"
                ),
                "accepted_reconstruction_job_count": require_int(
                    report.get("accepted_reconstruction_job_count"),
                    "accepted reconstruction job count",
                ),
                "evaluated_frame_count": require_int(report.get("evaluated_frame_count"), "evaluated frame count"),
                "evaluated_hand_rows": require_int(report.get("evaluated_hand_rows"), "evaluated hand rows"),
                "near_reconstructed_mesh_hand_rows": require_int(
                    report.get("near_reconstructed_mesh_hand_rows"),
                    "near reconstructed mesh hand rows",
                ),
                "reconstructed_mesh_contact_candidate_rows": require_int(
                    report.get("reconstructed_mesh_contact_candidate_rows"),
                    "reconstructed mesh contact candidate rows",
                ),
                "legacy_contact_ready_hand_rows": require_int(
                    report.get("legacy_contact_ready_hand_rows"),
                    "legacy contact ready hand rows",
                ),
                "multi_object_reconstructed_object_contact_candidate_rows": require_int(
                    report.get("multi_object_reconstructed_object_contact_candidate_rows"),
                    "multi-object reconstructed object contact candidate rows",
                ),
                "legacy_owner_mismatch_frame_count": require_int(
                    report.get("legacy_owner_mismatch_frame_count"),
                    "legacy owner mismatch frame count",
                ),
                "shared_depth_state_ready_frame_count": require_int(
                    report.get("shared_depth_state_ready_frame_count"),
                    "shared depth ready frame count",
                ),
                "depth_owner_incompatibility_count": require_int(
                    report.get("depth_owner_incompatibility_count"),
                    "depth owner incompatibility count",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "accepted_reconstruction_job_count": sum(
            require_int(report.get("accepted_reconstruction_job_count"), "accepted reconstruction job count")
            for report in reports
        ),
        "evaluated_frame_count": sum(
            require_int(report.get("evaluated_frame_count"), "evaluated frame count") for report in reports
        ),
        "evaluated_hand_rows": sum(
            require_int(report.get("evaluated_hand_rows"), "evaluated hand rows") for report in reports
        ),
        "near_reconstructed_mesh_hand_rows": sum(
            require_int(report.get("near_reconstructed_mesh_hand_rows"), "near reconstructed mesh hand rows")
            for report in reports
        ),
        "reconstructed_mesh_contact_candidate_rows": sum(
            require_int(report.get("reconstructed_mesh_contact_candidate_rows"), "mesh contact candidates")
            for report in reports
        ),
        "legacy_contact_ready_hand_rows": sum(
            require_int(report.get("legacy_contact_ready_hand_rows"), "legacy contact ready hand rows")
            for report in reports
        ),
        "multi_object_reconstructed_object_contact_candidate_rows": sum(
            require_int(
                report.get("multi_object_reconstructed_object_contact_candidate_rows"),
                "multi-object reconstructed object contact candidate rows",
            )
            for report in reports
        ),
        "legacy_owner_mismatch_frame_count": sum(
            require_int(report.get("legacy_owner_mismatch_frame_count"), "legacy owner mismatch frame count")
            for report in reports
        ),
        "shared_depth_state_ready_frame_count": sum(
            require_int(report.get("shared_depth_state_ready_frame_count"), "shared depth ready frames")
            for report in reports
        ),
        "depth_owner_incompatibility_count": sum(
            require_int(report.get("depth_owner_incompatibility_count"), "depth incompatibility count")
            for report in reports
        ),
        "shared_depth_contact_state_ready": any(
            bool(report.get("shared_depth_contact_state_ready") is True) for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_depth_contact_consistency_audit_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-reconstruction-results-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results"),
    )
    parser.add_argument(
        "--object-track-dataset-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--contact-mode-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_graph"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit"),
    )
    parser.add_argument("--mesh-surface-samples", type=int, default=20000)
    parser.add_argument("--near-reconstructed-mesh-m", type=float, default=0.02)
    parser.add_argument("--near-legacy-contact-m", type=float, default=0.02)
    parser.add_argument("--max-same-state-depth-delta-m", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
