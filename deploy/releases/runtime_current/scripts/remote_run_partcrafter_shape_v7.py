#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


def parse_case(raw: str) -> tuple[str, Path, int, int]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--case must have format name|image_path|seed|num_parts")
    name, image, seed, num_parts = [part.strip() for part in parts]
    if not name:
        raise RuntimeError("case name is empty")
    parsed_parts = int(num_parts)
    if parsed_parts < 1:
        raise RuntimeError(f"{name}: num_parts must be positive")
    return name, Path(image), int(seed), parsed_parts


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("mesh vertices are invalid")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("mesh faces are invalid")
    if not np.isfinite(vertices).all():
        raise RuntimeError("mesh vertices contain non-finite values")
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


def validate_rgba(path: Path, min_alpha_pixels: int) -> int:
    if not path.exists():
        raise RuntimeError(f"image does not exist: {path}")
    rgba = Image.open(path).convert("RGBA")
    alpha_pixels = int(np.count_nonzero(np.asarray(rgba)[..., 3] > 0))
    if alpha_pixels < int(min_alpha_pixels):
        raise RuntimeError(f"RGBA alpha mask has too few object pixels: {alpha_pixels}")
    return alpha_pixels


def composite_parts(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not parts:
        raise RuntimeError("PartCrafter returned no mesh parts")
    loaded_parts = []
    for idx, mesh in enumerate(parts):
        if mesh is None:
            raise RuntimeError(f"PartCrafter returned None for part {idx}")
        stats = mesh_stats(mesh)
        if stats["vertices"] < 4 or stats["faces"] < 4:
            raise RuntimeError(f"PartCrafter returned degenerate mesh for part {idx}: {stats}")
        loaded_parts.append(mesh)
    merged = trimesh.util.concatenate(loaded_parts)
    mesh_stats(merged)
    return merged


def run_case(args: argparse.Namespace, pipe, name: str, image_path: Path, seed: int, num_parts: int) -> dict:
    if num_parts > int(args.max_parts):
        raise RuntimeError(f"{name}: num_parts {num_parts} exceeds max_parts {args.max_parts}")
    alpha_pixels = validate_rgba(image_path, int(args.min_alpha_pixels))
    case_dir = args.output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    from accelerate.utils import set_seed
    from src.utils.image_utils import prepare_image

    set_seed(seed)
    image = prepare_image(
        str(image_path),
        bg_color=np.array([1.0, 1.0, 1.0]),
        rmbg_net=None,
        device=args.device,
    )
    start = time.time()
    output = pipe(
        image=[image] * num_parts,
        attention_kwargs={"num_parts": int(num_parts)},
        num_tokens=int(args.num_tokens),
        generator=torch.Generator(device=pipe.device).manual_seed(int(seed)),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        max_num_expanded_coords=int(args.max_num_expanded_coords),
        use_flash_decoder=bool(args.use_flash_decoder),
    )
    elapsed = time.time() - start
    parts = list(output.meshes)
    if len(parts) != num_parts:
        raise RuntimeError(f"{name}: expected {num_parts} parts, got {len(parts)}")
    merged = composite_parts(parts)

    part_reports = []
    for idx, mesh in enumerate(parts):
        glb_path = case_dir / f"part_{idx:02d}.glb"
        mesh.export(glb_path)
        part_reports.append({"index": int(idx), "glb": str(glb_path), "mesh_stats": mesh_stats(mesh)})

    glb_path = case_dir / "partcrafter_mesh.glb"
    ply_path = case_dir / "partcrafter_mesh.ply"
    merged.export(glb_path)
    merged.export(ply_path)
    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "alpha_pixels": int(alpha_pixels),
        "seed": int(seed),
        "num_parts": int(num_parts),
        "elapsed_s": float(elapsed),
        "glb": str(glb_path),
        "mesh": str(ply_path),
        "parts": part_reports,
        "mesh_stats": mesh_stats(merged),
    }
    (case_dir / "qc_partcrafter_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PartCrafter, but torch reports no CUDA device")
    sys.path.insert(0, str(args.repo))

    from huggingface_hub import snapshot_download
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.repo / "pretrained_weights" / "PartCrafter"
    snapshot_download(repo_id=args.model, local_dir=weights_dir, max_workers=1)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipe = PartCrafterPipeline.from_pretrained(str(weights_dir)).to(args.device, dtype)

    case_reports = [run_case(args, pipe, *parse_case(raw)) for raw in args.case]
    report = {
        "status": "ok",
        "method": "remote_run_partcrafter_shape_v7",
        "repo": str(args.repo),
        "model": args.model,
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device": args.device,
        },
    }
    (args.output_dir / "qc_partcrafter_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="wgsxm/PartCrafter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--max-num-expanded-coords", type=int, default=100000000)
    parser.add_argument("--use-flash-decoder", action="store_true")
    parser.add_argument("--max-parts", type=int, default=16)
    parser.add_argument("--min-alpha-pixels", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
