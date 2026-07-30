#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float32)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError(f"{path} has invalid offset lengths")
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx.tolist()):
        meshes[int(idx)] = (
            vertices[int(vertex_offsets[i]) : int(vertex_offsets[i + 1])],
            faces[int(face_offsets[i]) : int(face_offsets[i + 1])],
        )
    return meshes


def save_mesh_archive(path: Path, meshes: dict[int, tuple[np.ndarray, np.ndarray]], frame_order: list[int]) -> None:
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame_idx in frame_order:
        if frame_idx not in meshes:
            raise RuntimeError(f"missing mesh frame {frame_idx}")
        vertices, faces = meshes[frame_idx]
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError(f"empty mesh frame {frame_idx}")
        vertices_all.append(np.asarray(vertices, dtype=np.float32))
        faces_all.append(np.asarray(faces, dtype=np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_order, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def parse_completion(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise RuntimeError("--completion must be source=manifest=mesh_archive")
    source, manifest, mesh_archive = parts
    if not source:
        raise RuntimeError("completion source label is empty")
    return source, Path(manifest), Path(mesh_archive)


def frame_entries(manifest: dict) -> dict[int, dict]:
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest must contain nonempty frames list")
    return {int(entry["frame_idx"]): dict(entry) for entry in entries}


def run(args: argparse.Namespace) -> dict:
    base_manifest = load_json(args.base_manifest)
    base_entries = frame_entries(base_manifest)
    base_meshes = load_mesh_archive(args.base_mesh_archive)

    entries = {frame: dict(entry) for frame, entry in base_entries.items()}
    meshes = dict(base_meshes)
    provenance: dict[int, dict] = {
        int(frame): {"source": "measured", "manifest": str(args.base_manifest), "mesh_archive": str(args.base_mesh_archive)}
        for frame in base_entries
    }
    replacements = []

    for spec in args.completion:
        source, manifest_path, mesh_path = parse_completion(spec)
        completion_entries = frame_entries(load_json(manifest_path))
        completion_meshes = load_mesh_archive(mesh_path)
        for frame_idx, entry in completion_entries.items():
            if frame_idx not in completion_meshes:
                raise RuntimeError(f"completion mesh archive {mesh_path} lacks frame {frame_idx}")
            entries[int(frame_idx)] = entry
            meshes[int(frame_idx)] = completion_meshes[int(frame_idx)]
            provenance[int(frame_idx)] = {
                "source": source,
                "manifest": str(manifest_path),
                "mesh_archive": str(mesh_path),
            }
            replacements.append({"frame_idx": int(frame_idx), "source": source})

    frame_order = [frame for frame in sorted(entries) if int(args.frame_start) <= frame <= int(args.frame_end)]
    expected = list(range(int(args.frame_start), int(args.frame_end) + 1))
    if frame_order != expected:
        missing = sorted(set(expected).difference(frame_order))
        extra = sorted(set(frame_order).difference(expected))
        raise RuntimeError(f"assembled frame sequence is not dense; missing={missing[:20]} extra={extra[:20]}")
    for frame_idx in frame_order:
        if frame_idx not in meshes:
            raise RuntimeError(f"assembled mesh archive lacks frame {frame_idx}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame_idx in enumerate(frame_order):
        entry = dict(entries[frame_idx])
        entry["index"] = index
        entry["track_status_source"] = provenance[frame_idx]["source"]
        rows.append(entry)

    assembled_manifest = dict(base_manifest)
    assembled_manifest["backend"] = "assemble_completed_track_v4"
    assembled_manifest["dataset_dir"] = str(args.output_dir)
    assembled_manifest["frames"] = rows
    assembled_manifest["assembly"] = {
        "base_manifest": str(args.base_manifest),
        "base_mesh_archive": str(args.base_mesh_archive),
        "replacements": replacements,
        "frame_provenance": {str(frame): provenance[frame] for frame in frame_order},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(assembled_manifest, indent=2), encoding="utf-8")

    mesh_path = args.output_dir / "solidified_sheet_object_meshes_world.npz"
    save_mesh_archive(mesh_path, meshes, frame_order)
    report = {
        "status": "ok",
        "method": "assemble_completed_track_v4",
        "manifest": str(manifest_path),
        "mesh_archive": str(mesh_path),
        "frames": int(len(frame_order)),
        "first_frame": int(frame_order[0]),
        "last_frame": int(frame_order[-1]),
        "replacements": replacements,
        "frame_provenance": {str(frame): provenance[frame] for frame in frame_order},
    }
    (args.output_dir / "qc_assemble_completed_track_v4.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-mesh-archive", type=Path, required=True)
    parser.add_argument("--completion", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
