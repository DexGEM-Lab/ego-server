#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


def parse_case(raw: str) -> tuple[str, Path]:
    parts = raw.split("|")
    if len(parts) != 2:
        raise RuntimeError("--case must have format name|image_path")
    name, image = [part.strip() for part in parts]
    if not name:
        raise RuntimeError("case name is empty")
    return name, Path(image)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0]
        if not meshes:
            raise RuntimeError(f"GLB contains no mesh geometry: {path}")
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
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent_model_units": [float(x) for x in extent],
        "center_model_units": [float(x) for x in vertices.mean(axis=0)],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def load_spar3d(repo: Path):
    sys.path.insert(0, str(repo))
    from spar3d.system import SPAR3D
    from spar3d.utils import foreground_crop

    return SPAR3D, foreground_crop


def run_case(args: argparse.Namespace, model, foreground_crop, name: str, image_path: Path) -> dict:
    if not image_path.exists():
        raise RuntimeError(f"{name}: image does not exist: {image_path}")
    input_image = Image.open(image_path).convert("RGBA")
    alpha_pixels = int(np.count_nonzero(np.asarray(input_image)[..., 3] > 0))
    if alpha_pixels < int(args.min_alpha_pixels):
        raise RuntimeError(f"{name}: RGBA alpha mask has too few object pixels")
    case_dir = args.output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    processed_image = foreground_crop(input_image, args.foreground_ratio)
    run_dir = case_dir / "0"
    run_dir.mkdir(parents=True, exist_ok=True)
    processed_image.save(run_dir / "input.png")
    device_type = "cuda" if args.device.startswith("cuda") else args.device
    with torch.no_grad():
        with (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        ):
            mesh, glob_dict = model.run_image(
                processed_image,
                bake_resolution=args.texture_resolution,
                remesh=args.remesh_option,
                vertex_count=-1,
                return_points=True,
            )
    mesh_path = case_dir / "0" / "mesh.glb"
    points_path = case_dir / "0" / "points.ply"
    mesh.export(mesh_path, include_normals=True)
    point_clouds = glob_dict.get("point_clouds", [])
    if point_clouds:
        point_clouds[0].export(points_path)
    points_path = case_dir / "0" / "points.ply"
    if not mesh_path.exists():
        raise RuntimeError(f"{name}: SPAR3D did not write expected mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    ply_path = case_dir / "spar3d_mesh.ply"
    glb_path = case_dir / "spar3d_mesh.glb"
    mesh.export(ply_path)
    mesh.export(glb_path)
    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "alpha_pixels": alpha_pixels,
        "glb": str(glb_path),
        "mesh": str(ply_path),
        "points": str(points_path) if points_path.exists() else None,
        "mesh_stats": mesh_stats(mesh),
    }
    (case_dir / "qc_spar3d_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    SPAR3D, foreground_crop = load_spar3d(args.repo)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SPAR3D, but torch reports no CUDA device")
    model = SPAR3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=args.low_vram_mode,
    )
    model.to(args.device)
    model.eval()
    case_reports = [run_case(args, model, foreground_crop, *parse_case(raw)) for raw in args.case]
    report = {
        "status": "ok",
        "method": "remote_run_spar3d_shape_v7",
        "repo": str(args.repo),
        "pretrained_model": args.pretrained_model,
        "cases": case_reports,
    }
    (args.output_dir / "qc_spar3d_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="stabilityai/stable-point-aware-3d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--foreground-ratio", type=float, default=1.05)
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--remesh-option", choices=("none", "triangle", "quad"), default="none")
    parser.add_argument("--low-vram-mode", action="store_true")
    parser.add_argument("--min-alpha-pixels", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
