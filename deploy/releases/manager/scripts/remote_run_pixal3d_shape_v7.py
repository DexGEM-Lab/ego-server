#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


def parse_case(raw: str) -> tuple[str, Path, int]:
    parts = raw.split("|")
    if len(parts) != 3:
        raise RuntimeError("--case must have format name|image_path|seed")
    name, image, seed = [part.strip() for part in parts]
    if not name:
        raise RuntimeError("case name is empty")
    return name, Path(image), int(seed)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces)
        ]
        if not meshes:
            raise RuntimeError(f"GLB contains no triangle geometry: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported mesh object type from {path}: {type(loaded).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"mesh has no vertices/faces: {path}")
    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("mesh vertices are invalid")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("mesh faces are invalid")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if float(extent.max()) <= 0.0:
        raise RuntimeError("mesh extent is degenerate")
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent_model_units": [float(x) for x in extent],
        "center_model_units": [float(x) for x in vertices.mean(axis=0)],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def run_case(args: argparse.Namespace, name: str, image_path: Path, seed: int) -> dict:
    if not image_path.exists():
        raise RuntimeError(f"{name}: image does not exist: {image_path}")
    input_image = Image.open(image_path).convert("RGBA")
    alpha_pixels = int(np.count_nonzero(np.asarray(input_image)[..., 3] > 0))
    if alpha_pixels < int(args.min_alpha_pixels):
        raise RuntimeError(f"{name}: RGBA alpha mask has too few object pixels")
    case_dir = args.output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input_rgba.png"
    input_image.save(input_path)
    glb_path = case_dir / "pixal3d_mesh.glb"
    cmd = [
        str(args.python),
        str(args.repo / "inference.py"),
        "--image",
        str(input_path),
        "--output",
        str(glb_path),
        "--seed",
        str(seed),
        "--model_path",
        args.model,
        "--low_vram",
        "--resolution",
        str(args.resolution),
        "--fov",
        str(args.fov),
    ]
    env = os.environ.copy()
    env["ATTN_BACKEND"] = args.attn_backend
    env["SPARSE_ATTN_BACKEND"] = args.sparse_attn_backend
    subprocess.run(cmd, check=True, cwd=str(args.repo), env=env)
    if not glb_path.exists():
        raise RuntimeError(f"{name}: Pixal3D did not write expected GLB: {glb_path}")
    mesh = load_mesh(glb_path)
    ply_path = case_dir / "pixal3d_mesh.ply"
    mesh.export(ply_path)
    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "alpha_pixels": alpha_pixels,
        "seed": int(seed),
        "model": args.model,
        "glb": str(glb_path),
        "mesh": str(ply_path),
        "mesh_stats": mesh_stats(mesh),
    }
    (case_dir / "qc_pixal3d_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Pixal3D, but torch reports no CUDA device")
    case_reports = [run_case(args, *parse_case(raw)) for raw in args.case]
    report = {
        "status": "ok",
        "method": "remote_run_pixal3d_shape_v7",
        "repo": str(args.repo),
        "model": args.model,
        "resolution": int(args.resolution),
        "low_vram": True,
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
    }
    (args.output_dir / "qc_pixal3d_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", default="TencentARC/Pixal3D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=0.8575560450553894)
    parser.add_argument("--attn-backend", default="flash_attn_3")
    parser.add_argument("--sparse-attn-backend", default="flash_attn_3")
    parser.add_argument("--min-alpha-pixels", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
