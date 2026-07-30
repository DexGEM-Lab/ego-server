#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces)]
        if not meshes:
            raise RuntimeError(f"mesh scene contains no triangle geometry: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported mesh type from {path}: {type(loaded).__name__}")
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


def load_pipeline(repo: Path):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh

    return Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh


def run_case(args: argparse.Namespace, pipeline, name: str, image_path: Path, seed: int) -> dict:
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
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(int(seed))
    outputs = pipeline(
        image=str(input_path),
        generator=generator,
        num_inference_steps=int(args.steps),
        octree_resolution=int(args.octree_resolution),
        num_chunks=int(args.num_chunks),
        output_type="mesh",
    )[0]
    mesh = args.export_to_trimesh([outputs])[0]
    if mesh is None:
        raise RuntimeError(f"{name}: Hunyuan3D 2.1 surface extraction returned no mesh")
    raw_glb = case_dir / "hunyuan21_raw.glb"
    mesh.export(raw_glb)
    mesh = load_mesh(raw_glb)
    ply_path = case_dir / "hunyuan21_mesh.ply"
    glb_path = case_dir / "hunyuan21_mesh.glb"
    mesh.export(ply_path)
    mesh.export(glb_path)
    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "alpha_pixels": alpha_pixels,
        "seed": int(seed),
        "mesh": str(ply_path),
        "glb": str(glb_path),
        "mesh_stats": mesh_stats(mesh),
    }
    (case_dir / "qc_hunyuan21_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Hunyuan3D 2.1, but torch reports no CUDA device")
    Pipeline, export_to_trimesh = load_pipeline(args.repo)
    args.export_to_trimesh = export_to_trimesh
    pipeline = Pipeline.from_pretrained(
        args.model,
        subfolder=args.subfolder,
        variant=args.variant,
        device=args.device,
    )
    pipeline.to(args.device)
    case_reports = [run_case(args, pipeline, *parse_case(raw)) for raw in args.case]
    report = {
        "status": "ok",
        "method": "remote_run_hunyuan21_shape_v7",
        "repo": str(args.repo),
        "model": args.model,
        "subfolder": args.subfolder,
        "variant": args.variant,
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
    }
    (args.output_dir / "qc_hunyuan21_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-1")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=200000)
    parser.add_argument("--min-alpha-pixels", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
