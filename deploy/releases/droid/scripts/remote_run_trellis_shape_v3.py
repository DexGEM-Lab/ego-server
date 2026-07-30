#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import types
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import trimesh


def mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices_raw = mesh.vertices
    faces_raw = mesh.faces
    if isinstance(vertices_raw, torch.Tensor):
        vertices = vertices_raw.detach().cpu().numpy()
    else:
        vertices = np.asarray(vertices_raw)
    if isinstance(faces_raw, torch.Tensor):
        faces = faces_raw.detach().cpu().numpy()
    else:
        faces = np.asarray(faces_raw)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("TRELLIS mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("TRELLIS mesh has no valid triangular faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("TRELLIS mesh vertices contain non-finite values")
    return vertices.astype(float), faces.astype(np.int64)


def load_image_pipeline(repo: Path):
    trellis_root = repo / "trellis"
    pipelines_root = trellis_root / "pipelines"
    image_pipeline_path = pipelines_root / "trellis_image_to_3d.py"
    if not image_pipeline_path.exists():
        raise RuntimeError(f"TRELLIS image pipeline missing: {image_pipeline_path}")

    trellis_pkg = types.ModuleType("trellis")
    trellis_pkg.__path__ = [str(trellis_root)]
    trellis_pkg.__package__ = "trellis"
    sys.modules["trellis"] = trellis_pkg

    pipelines_pkg = types.ModuleType("trellis.pipelines")
    pipelines_pkg.__path__ = [str(pipelines_root)]
    pipelines_pkg.__package__ = "trellis.pipelines"
    sys.modules["trellis.pipelines"] = pipelines_pkg

    spec = importlib.util.spec_from_file_location("trellis.pipelines.trellis_image_to_3d", image_pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TRELLIS image pipeline spec: {image_pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TrellisImageTo3DPipeline


def run(args: argparse.Namespace) -> dict:
    os.environ["ATTN_BACKEND"] = args.attn_backend
    os.environ["SPCONV_ALGO"] = args.spconv_algo
    sys.path.insert(0, str(args.repo))

    # Load the image pipeline without executing TRELLIS package initializers.
    # Those initializers import the text pipeline and Open3D, unused here.
    TrellisImageTo3DPipeline = load_image_pipeline(args.repo)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGBA")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": args.sparse_cfg},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
    )
    meshes = outputs.get("mesh")
    if not meshes:
        raise RuntimeError("TRELLIS returned no mesh output")
    mesh = meshes[0]
    vertices, faces = mesh_arrays(mesh)

    mesh_path = args.output_dir / args.mesh_name
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(mesh_path))
    gaussian_path = None
    gaussians = outputs.get("gaussian")
    if gaussians:
        gaussian_path = args.output_dir / "trellis_gaussian.ply"
        gaussians[0].save_ply(str(gaussian_path))

    glb_path = None
    if args.export_glb:
        if not gaussians:
            raise RuntimeError("--export-glb requires a gaussian output")
        from trellis.utils import postprocessing_utils

        glb_path = args.output_dir / "trellis_mesh.glb"
        glb = postprocessing_utils.to_glb(gaussians[0], mesh, simplify=args.glb_simplify, texture_size=args.texture_size)
        glb.export(str(glb_path))

    report = {
        "status": "ok",
        "method": "remote_run_trellis_shape_v3",
        "repo": str(args.repo),
        "model": args.model,
        "image": str(args.image),
        "seed": int(args.seed),
        "sparse_steps": int(args.sparse_steps),
        "sparse_cfg": float(args.sparse_cfg),
        "slat_steps": int(args.slat_steps),
        "slat_cfg": float(args.slat_cfg),
        "mesh": str(mesh_path),
        "gaussian": str(gaussian_path) if gaussian_path else None,
        "glb": str(glb_path) if glb_path else None,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent_model_units": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        "center_model_units": vertices.mean(axis=0).astype(float).tolist(),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
    }
    (args.output_dir / "qc_trellis_shape_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--mesh-name", default="trellis_mesh.ply")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--sparse-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument("--attn-backend", choices=("xformers", "flash-attn"), default="xformers")
    parser.add_argument("--spconv-algo", choices=("native", "auto"), default="native")
    parser.add_argument("--export-glb", action="store_true")
    parser.add_argument("--glb-simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
