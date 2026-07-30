#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from diagnose_mesh_surface_contact_v3 import camera_points, load_mesh_archive


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def source_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def hand_vertices_camera(hand: dict, T_world_camera: np.ndarray) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        arr = np.asarray(hand.get(key, []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    for key in ("vertices_world_m", "vertices_world_m_sample"):
        arr = np.asarray(hand.get(key, []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return camera_points(arr, T_world_camera)
    raise RuntimeError("hand has no usable MANO vertices")


def selected_vertex_ids(row: dict) -> np.ndarray:
    source = str(row.get("selected_patch_source"))
    if source == "anatomical_patch":
        key = "anatomical_patch_vertex_ids"
    elif source == "best_patch":
        key = "best_patch_vertex_ids"
    else:
        raise RuntimeError(f"unsupported selected patch source {source!r}")
    ids = np.asarray(row.get(key, []), dtype=int)
    if ids.ndim != 1 or len(ids) == 0:
        raise RuntimeError(f"row {row.get('frame_idx')} has no selected patch vertex ids")
    return ids


def frame_map(annotations: dict) -> dict[int, dict]:
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def contact_rows(report: dict) -> list[dict]:
    rows = [row for row in report.get("rows_detail", []) if bool(row.get("reliable_for_contact", False))]
    return sorted(rows, key=lambda row: (str(row.get("track_id")), str(row.get("selected_patch_source")), str(row.get("selected_patch_region")), int(row["frame_idx"]), int(row["hand_idx"])))


def summarize(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def build_rows(args: argparse.Namespace) -> list[dict]:
    annotations = frame_map(load_json(args.annotations))
    meshes = load_mesh_archive(args.object_mesh_npz)
    report = load_json(args.contact_report)
    rows = []
    for contact in contact_rows(report):
        frame_idx = int(contact["frame_idx"])
        frame = annotations[frame_idx]
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        hand = frame["hands"][int(contact["hand_idx"])]
        hand_vertices = hand_vertices_camera(hand, T_world_camera)
        patch_ids = selected_vertex_ids(contact)
        patch_camera = hand_vertices[patch_ids]
        object_world, _ = meshes[frame_idx]
        object_camera = camera_points(object_world, T_world_camera)
        nearest_distance, nearest_ids = cKDTree(object_camera).query(patch_camera, k=1)
        object_patch_camera = object_camera[nearest_ids]
        hand_center_camera = np.median(patch_camera, axis=0)
        object_center_camera = np.median(object_patch_camera, axis=0)
        hand_center_world = source_to_world(hand_center_camera[None, :], T_world_camera)[0]
        object_center_world = source_to_world(object_center_camera[None, :], T_world_camera)[0]
        rows.append(
            {
                "frame_idx": frame_idx,
                "hand_idx": int(contact["hand_idx"]),
                "side": str(contact.get("side")),
                "track_id": contact.get("track_id"),
                "selected_patch_source": contact.get("selected_patch_source"),
                "selected_patch_region": contact.get("selected_patch_region"),
                "selected_patch_vertices": int(len(patch_ids)),
                "nearest_distance_median_m": float(np.median(nearest_distance)),
                "nearest_distance_p95_m": float(np.percentile(nearest_distance, 95.0)),
                "hand_patch_center_world_m": hand_center_world.astype(float).tolist(),
                "object_patch_center_world_m": object_center_world.astype(float).tolist(),
                "patch_center_gap_world_m": float(np.linalg.norm(hand_center_world - object_center_world)),
                "hand_patch_center_camera_m": hand_center_camera.astype(float).tolist(),
                "object_patch_center_camera_m": object_center_camera.astype(float).tolist(),
                "patch_center_gap_camera_m": float(np.linalg.norm(hand_center_camera - object_center_camera)),
                "contact_depth_bias_m": contact.get("mano_minus_metric_depth_median_m"),
                "contact_reprojection_px": contact.get("median_joint_reprojection_px"),
            }
        )
    if not rows:
        raise RuntimeError("no contact kinematic rows")
    return rows


def pair_rows(rows: list[dict], fps: float, max_gap_frames: int) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("track_id")),
            str(row.get("selected_patch_source")),
            str(row.get("selected_patch_region")),
        )
        grouped.setdefault(key, []).append(row)
    pairs = []
    for key, group in grouped.items():
        group.sort(key=lambda row: int(row["frame_idx"]))
        for prev, cur in zip(group[:-1], group[1:]):
            gap_frames = int(cur["frame_idx"]) - int(prev["frame_idx"])
            if gap_frames <= 0 or gap_frames > int(max_gap_frames):
                continue
            dt = gap_frames / float(fps)
            hp = np.asarray(prev["hand_patch_center_world_m"], dtype=float)
            hc = np.asarray(cur["hand_patch_center_world_m"], dtype=float)
            op = np.asarray(prev["object_patch_center_world_m"], dtype=float)
            oc = np.asarray(cur["object_patch_center_world_m"], dtype=float)
            hand_velocity = (hc - hp) / dt
            object_velocity = (oc - op) / dt
            relative_velocity = hand_velocity - object_velocity
            pairs.append(
                {
                    "from_frame": int(prev["frame_idx"]),
                    "to_frame": int(cur["frame_idx"]),
                    "track_id": key[0],
                    "selected_patch_source": key[1],
                    "selected_patch_region": key[2],
                    "gap_frames": gap_frames,
                    "hand_patch_speed_world_m_s": float(np.linalg.norm(hand_velocity)),
                    "object_patch_speed_world_m_s": float(np.linalg.norm(object_velocity)),
                    "relative_patch_speed_world_m_s": float(np.linalg.norm(relative_velocity)),
                    "hand_patch_step_world_m": float(np.linalg.norm(hc - hp)),
                    "object_patch_step_world_m": float(np.linalg.norm(oc - op)),
                    "relative_patch_step_world_m": float(np.linalg.norm((hc - hp) - (oc - op))),
                    "gap_change_m": float(cur["patch_center_gap_world_m"] - prev["patch_center_gap_world_m"]),
                }
            )
    return pairs


def run(args: argparse.Namespace) -> dict:
    rows = build_rows(args)
    pairs = pair_rows(rows, float(args.fps), int(args.max_gap_frames))
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_contact_kinematics_v3",
        "annotations": str(args.annotations),
        "contact_report": str(args.contact_report),
        "object_mesh_npz": str(args.object_mesh_npz),
        "rows": rows,
        "pairs": pairs,
        "summary_rows": {
            "patch_center_gap_world_m": summarize([row["patch_center_gap_world_m"] for row in rows]),
            "nearest_distance_p95_m": summarize([row["nearest_distance_p95_m"] for row in rows]),
        },
        "summary_pairs": {
            "relative_patch_speed_world_m_s": summarize([row["relative_patch_speed_world_m_s"] for row in pairs]),
            "relative_patch_step_world_m": summarize([row["relative_patch_step_world_m"] for row in pairs]),
            "gap_change_m": summarize([abs(row["gap_change_m"]) for row in pairs]),
        },
        "thresholds": {
            "fps": float(args.fps),
            "max_gap_frames": int(args.max_gap_frames),
        },
        "interpretation": (
            "This diagnostic checks whether accepted contact patches move together in the stored world frame. "
            "Large relative patch speed means the visual contact row is a local geometric proximity observation, "
            "not yet a physically regularized hand-object trajectory."
        ),
    }
    save_json(args.output_json, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "pairs", "skipped"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-gap-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
