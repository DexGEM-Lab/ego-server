#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.asarray(mask, dtype=np.uint8) > 0


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def export_raw_mesh(mesh_obj, path: Path) -> dict:
    vertices = as_numpy(mesh_obj.vertices).astype(np.float64)
    faces = as_numpy(mesh_obj.faces).astype(np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("SAM3D mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("SAM3D mesh has no valid triangular faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("SAM3D mesh vertices contain non-finite values")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    attrs = getattr(mesh_obj, "vertex_attrs", None)
    if attrs is not None:
        attrs_np = as_numpy(attrs)
        if attrs_np.ndim == 2 and attrs_np.shape[0] == vertices.shape[0] and attrs_np.shape[1] >= 3:
            colors = np.clip(attrs_np[:, :3], 0.0, 1.0)
            mesh.visual.vertex_colors = np.concatenate(
                [(colors * 255.0).astype(np.uint8), np.full((len(colors), 1), 255, dtype=np.uint8)],
                axis=1,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent_model_units": [float(x) for x in extent],
        "center_model_units": [float(x) for x in vertices.mean(axis=0)],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def tensor_list(value) -> list[float] | None:
    if value is None:
        return None
    arr = as_numpy(value).astype(float).reshape(-1)
    return [float(x) for x in arr]


def parse_case(raw: str) -> tuple[str, Path, Path, int]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--case must have format name|image_path|mask_path|seed")
    name, image, mask, seed = parts
    if not name.strip():
        raise RuntimeError("case name is empty")
    return name.strip(), Path(image), Path(mask), int(seed)


def run_one_case(inference, output_root: Path, name: str, image_path: Path, mask_path: Path, seed: int, args: argparse.Namespace) -> dict:
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    mask = load_mask(mask_path)
    if image.shape[:2] != mask.shape:
        raise RuntimeError(f"{name}: image shape {image.shape[:2]} differs from mask shape {mask.shape}")
    if int(mask.sum()) < int(args.min_mask_pixels):
        raise RuntimeError(f"{name}: mask has too few foreground pixels: {int(mask.sum())}")

    output = inference(image, mask, seed=int(seed))
    mesh_list = output.get("mesh")
    if not mesh_list:
        raise RuntimeError(f"{name}: SAM3D returned no mesh output")
    mesh_path = case_dir / args.mesh_name
    mesh_stats = export_raw_mesh(mesh_list[0], mesh_path)

    glb_path = case_dir / args.glb_name
    glb = output.get("glb")
    if glb is None:
        raise RuntimeError(f"{name}: SAM3D returned no GLB output")
    glb.export(str(glb_path))

    gs_path = None
    if output.get("gs") is not None:
        gs_path = case_dir / args.gaussian_name
        output["gs"].save_ply(str(gs_path))

    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "mask": str(mask_path),
        "seed": int(seed),
        "mask_pixels": int(mask.sum()),
        "image_shape_h_w": [int(image.shape[0]), int(image.shape[1])],
        "mesh": str(mesh_path),
        "glb": str(glb_path),
        "gaussian": str(gs_path) if gs_path else None,
        "mesh_stats": mesh_stats,
        "pose": {
            "rotation": tensor_list(output.get("rotation")),
            "translation": tensor_list(output.get("translation")),
            "scale": tensor_list(output.get("scale")),
        },
    }
    (case_dir / "qc_sam3d_objects_mesh_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "notebook"))

    from inference import Inference

    cases = [parse_case(raw) for raw in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference = Inference(str(args.config), compile=bool(args.compile))
    case_reports = [run_one_case(inference, args.output_dir, name, image, mask, seed, args) for name, image, mask, seed in cases]

    report = {
        "status": "ok",
        "method": "remote_run_sam3d_objects_mesh_v7",
        "repo": str(args.repo),
        "config": str(args.config),
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    (args.output_dir / "qc_sam3d_objects_mesh_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-name", default="sam3d_mesh.ply")
    parser.add_argument("--glb-name", default="sam3d_mesh.glb")
    parser.add_argument("--gaussian-name", default="sam3d_gaussian.ply")
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
