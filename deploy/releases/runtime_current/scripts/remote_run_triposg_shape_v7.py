#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    name, image, seed = parts
    if not name.strip():
        raise RuntimeError("case name is empty")
    return name.strip(), Path(image), int(seed)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0]
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


def load_repo_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "scripts"))

    from huggingface_hub import snapshot_download
    from briarmbg import BriaRMBG
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    inference_module = load_repo_module(args.repo / "scripts" / "inference_triposg.py", "triposg_inference_script")
    run_triposg = inference_module.run_triposg

    args.output_dir.mkdir(parents=True, exist_ok=True)
    triposg_weights_dir = args.repo / "pretrained_weights" / "TripoSG"
    rmbg_weights_dir = args.repo / "pretrained_weights" / "RMBG-1.4"
    snapshot_download(repo_id=args.triposg_model, local_dir=triposg_weights_dir)
    snapshot_download(repo_id=args.rmbg_model, local_dir=rmbg_weights_dir)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rmbg_net = BriaRMBG.from_pretrained(str(rmbg_weights_dir)).to(device)
    rmbg_net.eval()
    pipe = TripoSGPipeline.from_pretrained(str(triposg_weights_dir)).to(device, dtype)

    case_reports = []
    for name, image_path, seed in [parse_case(raw) for raw in args.case]:
        case_dir = args.output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        input_image = Image.open(image_path).convert("RGBA")
        if int(np.count_nonzero(np.asarray(input_image)[..., 3] > 0)) < int(args.min_alpha_pixels):
            raise RuntimeError(f"{name}: RGBA alpha mask has too few object pixels")
        mesh = run_triposg(
            pipe=pipe,
            image_input=str(image_path),
            rmbg_net=rmbg_net,
            seed=int(seed),
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            faces=int(args.faces),
        )
        glb_path = case_dir / "triposg_mesh.glb"
        ply_path = case_dir / "triposg_mesh.ply"
        mesh.export(glb_path)
        mesh.export(ply_path)
        report = {
            "status": "ok",
            "name": name,
            "image": str(image_path),
            "seed": int(seed),
            "glb": str(glb_path),
            "mesh": str(ply_path),
            "mesh_stats": mesh_stats(load_mesh(glb_path)),
        }
        (case_dir / "qc_triposg_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        case_reports.append(report)

    report = {
        "status": "ok",
        "method": "remote_run_triposg_shape_v7",
        "repo": str(args.repo),
        "triposg_model": args.triposg_model,
        "rmbg_model": args.rmbg_model,
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device": str(device),
        },
    }
    (args.output_dir / "qc_triposg_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--triposg-model", default="VAST-AI/TripoSG")
    parser.add_argument("--rmbg-model", default="briaai/RMBG-1.4")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--min-alpha-pixels", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
