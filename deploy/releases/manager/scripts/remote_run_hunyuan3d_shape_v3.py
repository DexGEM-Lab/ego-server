#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA")
    return image


def mesh_stats(mesh) -> dict:
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("generated mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("generated mesh has no valid faces")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if not np.isfinite(extent).all() or float(extent.max()) <= 0.0:
        raise RuntimeError("generated mesh extent is invalid")
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent_model_units": [float(x) for x in extent],
        "center_model_units": [float(x) for x in vertices.mean(axis=0)],
    }


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.repo))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.manual_seed(args.seed)
    if args.mode == "single":
        if len(args.image) != 1:
            raise RuntimeError("single mode requires exactly one --image")
        image = load_image(args.image[0])
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model,
            subfolder=args.subfolder,
            variant=args.variant,
        )
        if args.flashvdm:
            pipeline.enable_flashvdm()
        mesh = pipeline(
            image=image,
            num_inference_steps=args.steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            generator=generator,
            output_type="trimesh",
        )[0]
    else:
        if len(args.image) < 2:
            raise RuntimeError("multiview mode requires at least two --image values")
        views = ["front", "left", "back", "right"]
        images = {view: load_image(path) for view, path in zip(views, args.image)}
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model,
            subfolder=args.subfolder,
            variant=args.variant,
        )
        if args.flashvdm:
            pipeline.enable_flashvdm()
        mesh = pipeline(
            image=images,
            num_inference_steps=args.steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            generator=generator,
            output_type="trimesh",
        )[0]

    mesh_path = args.output_dir / args.mesh_name
    mesh.export(mesh_path)
    report = {
        "status": "ok",
        "method": "remote_run_hunyuan3d_shape_v3",
        "repo": str(args.repo),
        "mode": args.mode,
        "model": args.model,
        "subfolder": args.subfolder,
        "variant": args.variant,
        "images": [str(path) for path in args.image],
        "steps": int(args.steps),
        "octree_resolution": int(args.octree_resolution),
        "num_chunks": int(args.num_chunks),
        "seed": int(args.seed),
        "mesh": str(mesh_path),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
        "mesh_stats": mesh_stats(mesh),
    }
    (args.output_dir / "qc_hunyuan3d_shape_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "multiview"), required=True)
    parser.add_argument("--model", default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv-turbo")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--octree-resolution", type=int, default=380)
    parser.add_argument("--num-chunks", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mesh-name", default="mesh.glb")
    parser.add_argument("--flashvdm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
