#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


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


def mesh_stats(path: Path) -> dict:
    mesh = load_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if float(extent.max()) <= 0.0:
        raise RuntimeError(f"degenerate mesh extent: {path}")
    return {
        "mesh": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_model_units": [float(x) for x in extent],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def load_sequence_report(path: Path, sequence_dir_override: Path | None) -> dict:
    report = load_json(path)
    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) < 6:
        raise RuntimeError(f"{path} does not describe a Mesh4D-ready six-frame sequence")
    sequence_dir = sequence_dir_override if sequence_dir_override is not None else Path(str(report.get("sequence_dir", "")))
    if not sequence_dir.exists():
        raise RuntimeError(f"Mesh4D sequence path does not exist: {sequence_dir}")
    rgba_paths = [sequence_dir / f"{int(row['sequence_index'])}.png" for row in frames]
    missing = [str(p) for p in rgba_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"missing RGBA sequence files: {missing[:3]}")
    report["sequence_dir"] = str(sequence_dir)
    report["dataset_root"] = str(sequence_dir.parent.parent)
    for row in frames:
        row["mesh4d_rgba"] = str(sequence_dir / f"{int(row['sequence_index'])}.png")
    return report


def copy_dataset(sequence_report: dict, work_data_root: Path) -> Path:
    group = str(sequence_report["group_name"])
    sequence = str(sequence_report["sequence_name"])
    src = Path(str(sequence_report["sequence_dir"]))
    dst = work_data_root / group / sequence
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return work_data_root


def make_runtime_config(repo: Path, dataset_root: Path, output_dir: Path, config_path: Path) -> Path:
    source = repo / "hy3dshape" / "configs" / "infer.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["dataset"]["params"]["dataset_path"] = str(dataset_root)
    config["dataset"]["params"]["cfg"] = str(repo / "configs" / "OBJVERSE" / "train" / "infer.yaml")
    config["dataset"]["params"]["log_dir"] = str(output_dir / "infer_log_dir")
    config["dataset"]["params"]["num_workers"] = 0
    config["dataset"]["params"]["val_num_workers"] = 0
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def first_rgba_frame(sequence_report: dict) -> Path:
    frames = sequence_report["frames"]
    frames = sorted(frames, key=lambda row: int(row["sequence_index"]))
    return Path(str(frames[0]["mesh4d_rgba"]))


def run(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Mesh4D, but torch reports no CUDA device")
    sequence_report = load_sequence_report(args.sequence_json, args.sequence_dir)
    repo = args.repo.resolve()
    hy3dshape_root = repo / "hy3dshape"
    if not hy3dshape_root.exists():
        raise RuntimeError(f"Mesh4D hy3dshape directory does not exist: {hy3dshape_root}")
    os.chdir(hy3dshape_root)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(hy3dshape_root))
    sys.path.insert(0, str(repo / "hy3dpaint"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
    from hy3dshape.utils.misc import instantiate_from_config, instantiate_non_trainable_model
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from hy3dshape.pipelines_video_newvae_all_nonalign_infer import Hunyuan3DDiTFlowMatchingPipeline as Mesh4DPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = copy_dataset(sequence_report, args.output_dir / "DATA")
    config_path = make_runtime_config(repo, dataset_root, args.output_dir, args.output_dir / "mesh4d_infer_runtime.yaml")

    generator = torch.Generator(device=args.device).manual_seed(int(args.seed))
    image_path = first_rgba_frame(sequence_report)
    image = Image.open(image_path).convert("RGBA")
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.shape_model,
        subfolder=args.shape_subfolder,
        variant=args.shape_variant,
        device=args.device,
    )
    mesh_output = shape_pipe(
        image=image,
        generator=generator,
        num_inference_steps=int(args.shape_steps),
        octree_resolution=int(args.shape_octree_resolution),
        num_chunks=int(args.shape_num_chunks),
        output_type="mesh",
    )[0]
    gen_mesh = export_to_trimesh([mesh_output])[0]
    if gen_mesh is None:
        raise RuntimeError("Mesh4D initial shape generation returned no mesh")
    initial_mesh = args.output_dir / "initial_shape_mesh.ply"
    gen_mesh.export(initial_mesh)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    denoiser = instantiate_from_config(config["model"]["params"]["denoiser_cfg"])
    ckpt = torch.load(args.denoiser_ckpt, map_location="cpu")
    state = ckpt["state_dict"]
    state = {k.replace("model.", ""): v for k, v in state.items()}
    denoiser.load_state_dict(state, strict=False)
    denoiser = denoiser.to(args.device).half()

    vae = instantiate_non_trainable_model(config["model"]["params"]["first_stage_config"])
    conditioner = instantiate_from_config(config["model"]["params"]["cond_stage_config"])
    static_conditioner = instantiate_non_trainable_model(config["model"]["params"]["cond_stage_config_2"])
    image_processor = instantiate_from_config(config["model"]["params"]["image_processor_cfg"])
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    pipeline_cfg = config["model"]["params"]["pipeline_cfg"]
    if pipeline_cfg["target"] != "hy3dshape.pipelines_video_newvae_all_nonalign_infer.Hunyuan3DDiTFlowMatchingPipeline":
        raise RuntimeError(f"unexpected Mesh4D pipeline target: {pipeline_cfg['target']}")
    pipeline_params = dict(pipeline_cfg["params"])
    latent_shape = tuple(int(value) for value in pipeline_params["latent_shape"])
    if len(latent_shape) != 3:
        raise RuntimeError(f"expected Mesh4D latent_shape to have 3 dimensions, got {latent_shape}")
    pipeline = Mesh4DPipeline(
        vae=vae,
        model=denoiser,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
        cond_stage_model_2=static_conditioner,
        z_scale_factor=config["model"]["params"]["z_scale_factor"],
        **pipeline_params,
    )
    pipeline.device = torch.device(args.device)
    pipeline.dtype = torch.float16
    conditioner.disable_drop = True

    data_module = instantiate_from_config(config["dataset"])
    loader = data_module.val_dataloader()
    batches = list(loader)
    if len(batches) != 1:
        raise RuntimeError(f"expected exactly one Mesh4D batch, got {len(batches)}")
    batch = batches[0]
    batch = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
    with torch.amp.autocast(device_type="cuda"):
        metrics = pipeline(
            batch=batch,
            output_path=str(args.output_dir),
            generator=generator,
            gen_mesh=load_mesh(initial_mesh),
            gen_mesh_list=None,
            not_simplify=True,
            combine_glb=False,
            num_inference_steps=int(args.mesh4d_steps),
            guidance_scale=float(args.guidance_scale),
        )

    model_name = str(batch["model_name"][0])
    mesh_dir = args.output_dir / model_name / "gen"
    mesh_paths = sorted(mesh_dir.glob("gen_*.obj"))
    if len(mesh_paths) != 6:
        raise RuntimeError(f"expected 6 Mesh4D gen_*.obj outputs under {mesh_dir}, got {len(mesh_paths)}")
    frames = sorted(sequence_report["frames"], key=lambda row: int(row["sequence_index"]))
    frame_reports = []
    for idx, path in enumerate(mesh_paths):
        row = frames[idx]
        frame_reports.append(
            {
                "sequence_index": int(row["sequence_index"]),
                "frame_idx": int(row["frame_idx"]),
                "source_index": int(row["source_index"]),
                **mesh_stats(path),
            }
        )
    report = {
        "status": "ok",
        "method": "remote_run_mesh4d_sequence_v7",
        "repo": str(repo),
        "sequence_json": str(args.sequence_json),
        "dataset_root": str(dataset_root),
        "runtime_config": str(config_path),
        "initial_mesh": str(initial_mesh),
        "model_name": model_name,
        "mesh_dir": str(mesh_dir),
        "pipeline_params": json_ready(pipeline_params),
        "frames": frame_reports,
        "metrics": json_ready(metrics),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
    }
    (args.output_dir / "qc_mesh4d_sequence_v7.json").write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(report), indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sequence-json", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--denoiser-ckpt", type=Path, required=True)
    parser.add_argument("--shape-model", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--shape-subfolder", default="hunyuan3d-dit-v2-1")
    parser.add_argument("--shape-variant", default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape-steps", type=int, default=30)
    parser.add_argument("--shape-octree-resolution", type=int, default=256)
    parser.add_argument("--shape-num-chunks", type=int, default=200000)
    parser.add_argument("--mesh4d-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
