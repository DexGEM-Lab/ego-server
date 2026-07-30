#!/usr/bin/env python3
"""V21 UniDepth V2 depth runner.

Runs UniDepth V2 on all frames to produce a second metric depth
source for comparison with DepthPro.

Output:
  measurements/depth_candidates/unidepth_v2/unidepth_v2_depth.npz
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

UNIDEPTH_REPO = Path(os.environ.get("V22_UNIDEPTH_REPO", "/home/zjh/ego-annation-checkpoints/unidepth_repo"))
UNIDEPTH_MODEL = Path(os.environ.get("V22_UNIDEPTH_MODEL", "/home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected"))


def resolve_manifest_path(run_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    candidates = [path]
    text = str(raw)
    if text.startswith("outputs/"):
        candidates.append(Path("output") / Path(text).relative_to("outputs"))
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        candidates.append(Path("output") / Path(text[len(historical_prefix) :]))
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, run_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def run(args):
    run_root = Path(args.run_root).resolve()

    if not UNIDEPTH_REPO.exists():
        raise FileNotFoundError(f"unidepth_repo_not_found: {UNIDEPTH_REPO}")
    if not UNIDEPTH_MODEL.exists():
        raise FileNotFoundError(f"unidepth_model_not_found: {UNIDEPTH_MODEL}")

    # Load UniDepth from the declared server-side source/checkpoint paths.
    inference_stubs = Path(__file__).resolve().parents[1] / "third_party_inference_stubs"
    sys.path.insert(0, str(inference_stubs))
    sys.path.insert(1, str(UNIDEPTH_REPO))
    config_path = Path(os.environ.get("V22_UNIDEPTH_CONFIG", str(UNIDEPTH_MODEL / "config.json")))
    if not config_path.exists():
        config_path = UNIDEPTH_REPO / "configs" / "config_v2_vits14.json"
    config = json.loads(config_path.read_text())
    from unidepth.models import UniDepthV2
    model = UniDepthV2(config)
    from safetensors.torch import load_file
    state = load_file(str(UNIDEPTH_MODEL / "model.safetensors"))
    model.load_state_dict(state, strict=False)
    model = model.to("cuda").eval()
    print("UniDepth V2 loaded!", flush=True)

    # Load manifest
    manifest = json.loads((run_root / "input/raw_frame_manifest/manifest.json").read_text())

    depths = []
    fidxs = []
    intrinsics_list = []

    for fm in tqdm(manifest["frames"], desc="UniDepth V2"):
        fidx = fm["frame_idx"]
        rgb_path = resolve_manifest_path(run_root, fm["rgb"])
        image = Image.open(rgb_path).convert("RGB")

        rgb = np.array(image, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda")

        with torch.no_grad():
            predictions = model.infer(rgb_tensor)

        depth = predictions["depth"][0, 0].cpu().numpy().astype(np.float32)
        depths.append(depth.astype(np.float16))
        fidxs.append(fidx)

        # UniDepth also predicts intrinsics
        if "intrinsics" in predictions:
            intr = predictions["intrinsics"][0].cpu().numpy()
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            intrinsics_list.append([fx, fy, cx, cy])

    depths = np.stack(depths)
    output = {
        "depth": depths,
        "frame_idx": np.array(fidxs, dtype=np.int32),
    }
    if intrinsics_list:
        output["intrinsics_fx_fy_cx_cy"] = np.array(intrinsics_list, dtype=np.float32)

    out_dir = run_root / "measurements/depth_candidates/unidepth_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_dir / "unidepth_v2_depth.npz"), **output)
    report = {
        "schema": "v21_unidepth_v2_candidate.v0",
        "status": "ok",
        "method": "run_v21_unidepth",
        "unidepth_repo": str(UNIDEPTH_REPO),
        "unidepth_model": str(UNIDEPTH_MODEL),
        "unidepth_config": str(config_path),
        "depth_archive": str(out_dir / "unidepth_v2_depth.npz"),
        "frame_count": int(len(depths)),
        "first_frame": int(fidxs[0]) if fidxs else None,
        "last_frame": int(fidxs[-1]) if fidxs else None,
        "claim_scope": "UniDepth v2 monocular metric depth and intrinsics candidate. This is a depth/camera observation, not object pose or contact evidence.",
    }
    (out_dir / "qc_unidepth_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Comparison with DepthPro
    dp_path = run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"
    if dp_path.exists():
        dp = np.load(str(dp_path))
        dp_depth = dp["depth"]

        residuals = []
        for i in range(0, len(fidxs), max(1, len(fidxs)//10)):
            dp_frame = dp_depth[i]
            ud_frame = depths[i]
            if dp_frame.shape != ud_frame.shape:
                from PIL import Image as PILImage
                ud_frame_resized = np.array(PILImage.fromarray(ud_frame).resize((dp_frame.shape[1], dp_frame.shape[0])))
            else:
                ud_frame_resized = ud_frame
            valid = (dp_frame > 0.1) & (ud_frame_resized > 0.1)
            if valid.sum() > 100:
                residual = np.abs(dp_frame[valid] - ud_frame_resized[valid])
                residuals.append(float(np.median(residual)))

        comparison_report = {
            "method": "depthpro_vs_unidepth_v2",
            "median_residual_m": float(np.median(residuals)) if residuals else None,
            "sample_count": len(residuals),
        }
        (out_dir / "depthpro_vs_unidepth_report.json").write_text(json.dumps(comparison_report, indent=2))
        print(f"DepthPro vs UniDepth: median residual {comparison_report['median_residual_m']:.3f}m" if residuals else "No comparison", flush=True)

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    run(args)
