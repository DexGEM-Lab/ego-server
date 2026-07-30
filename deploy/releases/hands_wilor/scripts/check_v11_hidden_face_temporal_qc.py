#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_factor_graph_v6 import report_pair_rows
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def rows_by_frame(report: dict) -> dict[int, dict]:
    rows = report.get("rows")
    if rows is None:
        rows = report.get("output_frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("append report lacks nonempty rows")
    out: dict[int, dict] = {}
    for row in rows:
        frame = int(row["frame_idx"])
        if frame in out:
            raise RuntimeError(f"duplicate append row for frame {frame}")
        out[frame] = row
    return out


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("cannot sample empty mesh")
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not bool(valid.any()):
        raise RuntimeError("mesh has no positive-area faces")
    tri = tri[valid]
    areas = areas[valid]
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(tri), size=int(count), replace=True, p=areas / areas.sum())
    chosen = tri[ids]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])


def connected_components(face_count: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(int(face_count), dtype=np.int64)
    rank = np.zeros(int(face_count), dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in np.asarray(edges, dtype=np.int64):
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(int(face_count))], dtype=np.int64)
    unique = {root: i for i, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([unique[int(root)] for root in roots], dtype=np.int64)


def face_adjacency_labels(faces: np.ndarray) -> np.ndarray:
    edge_owner: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        verts = [int(face[0]), int(face[1]), int(face[2])]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            key = (a, b) if a < b else (b, a)
            prev = edge_owner.get(key)
            if prev is None:
                edge_owner[key] = int(face_id)
            else:
                edges.append((prev, int(face_id)))
    if not edges:
        return np.arange(len(faces), dtype=np.int64)
    return connected_components(len(faces), np.asarray(edges, dtype=np.int64))


def hidden_submesh(meshes: dict[int, tuple[np.ndarray, np.ndarray]], append_rows: dict[int, dict], frame: int) -> tuple[np.ndarray, np.ndarray]:
    if frame not in meshes:
        raise RuntimeError(f"mesh archive lacks frame {frame}")
    if frame not in append_rows:
        raise RuntimeError(f"append report lacks frame {frame}")
    vertices, faces = meshes[frame]
    row = append_rows[frame]
    observed_vertices = int(row["observed_vertices"])
    observed_faces = int(row["observed_faces"])
    archive_vertices = int(row["archive_vertices"])
    archive_faces = int(row["archive_faces"])
    if archive_vertices != len(vertices) or archive_faces != len(faces):
        raise RuntimeError(f"append report/archive count mismatch for frame {frame}")
    hidden_vertices = np.asarray(vertices[observed_vertices:], dtype=np.float64)
    hidden_faces = np.asarray(faces[observed_faces:], dtype=np.int64) - observed_vertices
    if hidden_faces.size and (hidden_faces.min() < 0 or hidden_faces.max() >= len(hidden_vertices)):
        raise RuntimeError(f"hidden face index out of range for frame {frame}")
    return hidden_vertices, hidden_faces.astype(np.int32)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(np.sum(area[np.isfinite(area)]))


def component_rows(faces: np.ndarray) -> list[dict[str, Any]]:
    if len(faces) == 0:
        return []
    labels = face_adjacency_labels(faces)
    rows = []
    for label in sorted(set(labels.tolist())):
        face_count = int(np.count_nonzero(labels == int(label)))
        rows.append({"component": int(label), "faces": face_count})
    rows.sort(key=lambda row: int(row["faces"]), reverse=True)
    return rows


def finite_summary(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return summarize(arr)


def pair_transforms(path: Path | None) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]]:
    if path is None:
        return {}
    report = load_json(path)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]] = {}
    for row in report_pair_rows(report, path):
        source = int(row["source_frame"])
        target = int(row["target_frame"])
        if not row.get("rigid_factor_ready"):
            continue
        rot = np.asarray(row["rotation"], dtype=np.float64)
        trans = np.asarray(row["translation_m"], dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,):
            raise RuntimeError(f"invalid pair transform for {source}->{target}")
        out[(source, target)] = (rot, trans, row)
    return out


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rot + trans[None, :]


def symmetric_surface_distance(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        raise RuntimeError("cannot compare empty hidden point sets")
    target_tree = cKDTree(target)
    source_tree = cKDTree(source)
    source_to_target, _ = target_tree.query(source, k=1)
    target_to_source, _ = source_tree.query(target, k=1)
    return source_to_target.astype(np.float64), target_to_source.astype(np.float64)


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    append_report = load_json(args.append_report)
    append_rows = rows_by_frame(append_report)
    frames = [frame for frame in sorted(meshes) if int(args.frame_start) <= frame <= int(args.frame_end)]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames selected")

    frame_reports = []
    hidden_samples: dict[int, np.ndarray] = {}
    for frame in frames:
        vertices, faces = hidden_submesh(meshes, append_rows, frame)
        components = component_rows(faces)
        largest_faces = int(components[0]["faces"]) if components else 0
        face_count = int(len(faces))
        row: dict[str, Any] = {
            "frame_idx": int(frame),
            "hidden_vertices": int(len(vertices)),
            "hidden_faces": face_count,
            "hidden_surface_area_m2": surface_area(vertices, faces),
            "component_count": int(len(components)),
            "largest_component_faces": largest_faces,
            "largest_component_face_fraction": None if face_count == 0 else float(largest_faces / face_count),
            "components": components[: int(args.max_reported_components)],
        }
        if face_count > 0:
            sample_count = min(int(args.sample_points), max(int(args.min_sample_points), face_count))
            hidden_samples[frame] = sample_surface(vertices, faces, sample_count, seed=int(args.seed) + frame)
            row["sample_points"] = int(sample_count)
        else:
            row["sample_points"] = 0
        frame_reports.append(row)

    transforms = pair_transforms(args.pair_factors_json)
    pair_reports = []
    evaluated_distances = []
    face_count_log_steps = []
    hidden_pairs = 0
    evaluated_pairs = 0
    for source, target in zip(frames[:-1], frames[1:]):
        source_faces = int(next(row["hidden_faces"] for row in frame_reports if int(row["frame_idx"]) == source))
        target_faces = int(next(row["hidden_faces"] for row in frame_reports if int(row["frame_idx"]) == target))
        if source_faces == 0 or target_faces == 0:
            pair_reports.append(
                {
                    "source_frame": int(source),
                    "target_frame": int(target),
                    "status": "skipped_empty_hidden_surface",
                    "source_hidden_faces": source_faces,
                    "target_hidden_faces": target_faces,
                }
            )
            continue
        hidden_pairs += 1
        if (source, target) not in transforms:
            pair_reports.append(
                {
                    "source_frame": int(source),
                    "target_frame": int(target),
                    "status": "missing_observed_motion_factor",
                    "source_hidden_faces": source_faces,
                    "target_hidden_faces": target_faces,
                }
            )
            continue
        rot, trans, transform_row = transforms[(source, target)]
        transformed_source = transform_points(hidden_samples[source], rot, trans)
        target_points = hidden_samples[target]
        source_to_target, target_to_source = symmetric_surface_distance(transformed_source, target_points)
        combined = np.concatenate([source_to_target, target_to_source])
        face_log_step = float(abs(np.log(max(source_faces, 1) / max(target_faces, 1))))
        evaluated_pairs += 1
        evaluated_distances.extend(combined.tolist())
        face_count_log_steps.append(face_log_step)
        pair_reports.append(
            {
                "source_frame": int(source),
                "target_frame": int(target),
                "status": "evaluated",
                "source_hidden_faces": source_faces,
                "target_hidden_faces": target_faces,
                "observed_motion_track_count": int(transform_row.get("track_count", 0)),
                "observed_motion_inlier_residual_m": transform_row.get("inlier_residual_m", {}),
                "source_to_target_m": finite_summary(source_to_target),
                "target_to_source_m": finite_summary(target_to_source),
                "symmetric_distance_m": finite_summary(combined),
                "hidden_face_count_abs_log_step": face_log_step,
            }
        )

    hidden_frame_count = int(sum(int(row["hidden_faces"]) > 0 for row in frame_reports))
    hidden_pair_coverage = None if hidden_pairs == 0 else float(evaluated_pairs / hidden_pairs)
    distance_summary = finite_summary(np.asarray(evaluated_distances, dtype=np.float64))
    face_step_summary = finite_summary(np.asarray(face_count_log_steps, dtype=np.float64))
    checks = {
        "min_hidden_frames": hidden_frame_count >= int(args.min_hidden_frames),
        "min_hidden_pair_coverage": False
        if hidden_pair_coverage is None
        else hidden_pair_coverage >= float(args.min_hidden_pair_coverage),
        "max_symmetric_distance_p95": False
        if distance_summary["p95"] is None
        else float(distance_summary["p95"]) <= float(args.max_symmetric_distance_p95_m),
        "max_hidden_face_count_abs_log_step": False
        if face_step_summary["p95"] is None
        else float(face_step_summary["p95"]) <= float(args.max_hidden_face_count_abs_log_step),
    }
    accepted = bool(all(checks.values()))
    if hidden_frame_count < int(args.min_hidden_frames):
        status = "no_hidden_geometry"
    else:
        status = "accepted" if accepted else "rejected"
    report = {
        "status": status,
        "annotation_ready": bool(accepted),
        "method": "check_v11_hidden_face_temporal_qc",
        "claim_tested": (
            "appended hidden faces should remain temporally stable under model-produced observed-surface motion factors; "
            "visible replay alone cannot validate unseen geometry"
        ),
        "mesh_archive": str(args.mesh_archive),
        "append_report": str(args.append_report),
        "pair_factors_json": None if args.pair_factors_json is None else str(args.pair_factors_json),
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "frame_count": int(len(frames)),
        "hidden_frame_count": hidden_frame_count,
        "hidden_pair_count": int(hidden_pairs),
        "evaluated_hidden_pair_count": int(evaluated_pairs),
        "hidden_pair_coverage": hidden_pair_coverage,
        "symmetric_hidden_surface_distance_m": distance_summary,
        "hidden_face_count_abs_log_step": face_step_summary,
        "checks": checks,
        "thresholds": {
            "min_hidden_frames": int(args.min_hidden_frames),
            "min_hidden_pair_coverage": float(args.min_hidden_pair_coverage),
            "max_symmetric_distance_p95_m": float(args.max_symmetric_distance_p95_m),
            "max_hidden_face_count_abs_log_step": float(args.max_hidden_face_count_abs_log_step),
        },
        "parameters": {
            "sample_points": int(args.sample_points),
            "min_sample_points": int(args.min_sample_points),
            "seed": int(args.seed),
        },
        "frames": frame_reports,
        "pairs": pair_reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "qc_v11_hidden_face_temporal.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frames", "pairs"}}, indent=2))
    if args.fail_on_rejected and status != "accepted":
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"hidden-face temporal QC {status}; failed checks: {failed}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--append-report", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--min-hidden-frames", type=int, default=2)
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--min-sample-points", type=int, default=1000)
    parser.add_argument("--max-reported-components", type=int, default=8)
    parser.add_argument("--min-hidden-pair-coverage", type=float, default=1.0)
    parser.add_argument("--max-symmetric-distance-p95-m", type=float, default=0.02)
    parser.add_argument("--max-hidden-face-count-abs-log-step", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--fail-on-rejected", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
