#!/usr/bin/env python3
"""Run UniDepth as a resident multi-item batch worker.

This is the first real model-stage resident worker target for the batch API
architecture. It loads UniDepth once, consumes one or more frame batches from
multiple items, writes per-item depth archives, and records ownership fields.
Use this on the remote A800 environment, not on the local workstation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_UNIDEPTH_REPO = Path(os.environ.get("V22_UNIDEPTH_REPO", "/home/zjh/ego-annation-checkpoints/unidepth_repo"))
DEFAULT_UNIDEPTH_MODEL = Path(os.environ.get("V22_UNIDEPTH_MODEL", "/home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected"))
DEFAULT_STAGE_ID = "unidepth_v2_depth_resident"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def resolve_path(run_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, run_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def resize_hw(width: int, source_width: int, source_height: int) -> tuple[int, int]:
    height = int(round(float(width) * float(source_height) / float(source_width)))
    return int(width), max(2, height + height % 2)


def load_model(repo: Path, model_dir: Path, config_path: Path | None, device: str):
    import torch

    if not repo.exists():
        raise FileNotFoundError(f"unidepth_repo_not_found: {repo}")
    if not model_dir.exists():
        raise FileNotFoundError(f"unidepth_model_not_found: {model_dir}")
    inference_stubs = Path(__file__).resolve().parents[1] / "third_party_inference_stubs"
    sys.path.insert(0, str(inference_stubs))
    sys.path.insert(1, str(repo))
    resolved_config = config_path or Path(os.environ.get("V22_UNIDEPTH_CONFIG", str(model_dir / "config.json")))
    if not resolved_config.exists():
        resolved_config = repo / "configs" / "config_v2_vits14.json"
    config = json.loads(resolved_config.read_text(encoding="utf-8"))
    from unidepth.models import UniDepthV2
    from safetensors.torch import load_file

    model = UniDepthV2(config)
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, resolved_config, torch


def iter_rows(request: dict[str, Any]) -> list[dict[str, Any]]:
    job_id = str(request.get("job_id") or "resident_unidepth_job")
    stage_id = str(request.get("stage_id") or DEFAULT_STAGE_ID)
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(request.get("items") or []):
        if not isinstance(item, dict):
            raise RuntimeError(f"items[{item_index}] must be an object")
        item_id = str(item.get("item_id") or f"item_{item_index:06d}")
        raw_run_root = item.get("run_root")
        if not raw_run_root:
            raise RuntimeError(f"items[{item_index}].run_root is required")
        run_root = Path(str(raw_run_root)).expanduser().resolve()
        manifest_path = Path(str(item.get("raw_frame_manifest") or run_root / "input" / "raw_frame_manifest" / "manifest.json"))
        manifest_path = resolve_path(run_root, manifest_path)
        manifest = load_json(manifest_path)
        frames = manifest.get("frames") if isinstance(manifest.get("frames"), list) else []
        max_frames = item.get("max_frames", request.get("max_frames_per_item"))
        if max_frames is not None:
            frames = frames[: int(max_frames)]
        output_dir = Path(str(item.get("output_dir") or run_root / "measurements" / "depth_candidates" / "unidepth_v2_resident"))
        for local_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            frame_idx = int(frame.get("frame_idx", frame.get("index", local_index)))
            rgb_raw = frame.get("rgb") or frame.get("raw_frame_path")
            if not rgb_raw:
                raise RuntimeError(f"{item_id} frame {local_index} lacks rgb path")
            batch_index = len(rows) // max(1, int(request.get("batch_size", 4)))
            batch_id = str(request.get("batch_id_prefix") or f"{job_id}_{stage_id}_batch_{batch_index:05d}")
            rows.append(
                {
                    "row_id": f"{batch_id}_{item_id}_{frame_idx:06d}",
                    "job_id": job_id,
                    "item_id": item_id,
                    "batch_id": batch_id,
                    "stage_id": stage_id,
                    "agent_id": str(item.get("agent_id") or request.get("agent_id") or "resident_unidepth_agent"),
                    "attempt_id": str(item.get("attempt_id") or request.get("attempt_id") or "attempt_0001"),
                    "run_root": str(run_root),
                    "raw_frame_manifest": str(manifest_path),
                    "frame_idx": frame_idx,
                    "time_s": frame.get("time_s"),
                    "rgb_path": str(resolve_path(run_root, rgb_raw)),
                    "output_dir": str(output_dir),
                    "source_width": frame.get("source_width") or frame.get("width") or manifest.get("render_width"),
                    "source_height": frame.get("source_height") or frame.get("height") or manifest.get("render_height"),
                }
            )
    if not rows:
        raise RuntimeError("request produced no frame rows")
    return rows


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def load_image_tensor(row: dict[str, Any], resize_width: int | None, torch: Any):
    import numpy as np
    from PIL import Image

    image = Image.open(row["rgb_path"]).convert("RGB")
    original_size = image.size
    if resize_width is not None:
        width, height = resize_hw(int(resize_width), image.width, image.height)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return tensor, {"original_size": list(original_size), "inference_size": [image.width, image.height]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    request = load_json(args.request)
    rows = iter_rows(request)
    batch_size = int(request.get("batch_size") or args.batch_size)
    if batch_size <= 0:
        raise RuntimeError("batch_size must be positive")
    device = str(request.get("device") or args.device)
    repo = Path(str(request.get("unidepth_repo") or args.unidepth_repo)).expanduser()
    model_dir = Path(str(request.get("unidepth_model") or args.unidepth_model)).expanduser()
    config_path = Path(str(request["unidepth_config"])).expanduser() if request.get("unidepth_config") else None
    output_root = Path(str(request.get("output_root") or args.output_root or Path(rows[0]["run_root"]) / "resident_unidepth_batch"))
    output_root.mkdir(parents=True, exist_ok=True)
    worker_id = str(request.get("worker_id") or args.worker_id)
    stage_id = str(request.get("stage_id") or DEFAULT_STAGE_ID)
    resize_width = request.get("resize_width", args.resize_width)
    resize_width_i = int(resize_width) if resize_width is not None and int(resize_width) > 0 else None

    started = time.time()
    load_started = utc_now()
    model, resolved_config, torch = load_model(repo, model_dir, config_path, device)
    load_finished = utc_now()
    model_identity = {
        "model_name": "UniDepthV2",
        "model_version": "unidepth_v2",
        "unidepth_repo": str(repo),
        "unidepth_model": str(model_dir),
        "unidepth_config": str(resolved_config),
        "stage_id": stage_id,
        "worker_id": worker_id,
        "device": device,
    }

    per_item: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"depth": [], "frame_idx": [], "intrinsics": [], "rows": []})
    batch_reports: list[dict[str, Any]] = []
    for batch_no, batch_rows in enumerate(chunks(rows, batch_size)):
        tensors = []
        image_meta = []
        for row in batch_rows:
            tensor, meta = load_image_tensor(row, resize_width_i, torch)
            tensors.append(tensor)
            image_meta.append(meta)
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise RuntimeError(f"mixed tensor shapes in batch {batch_no}: {sorted(shapes)}; set resize_width")
        rgb_tensor = torch.stack(tensors, dim=0).to(device)
        with torch.inference_mode():
            predictions = model.infer(rgb_tensor)
        depth = predictions["depth"][:, 0].detach().cpu().numpy()
        intrinsics = predictions.get("intrinsics")
        intr_np = intrinsics.detach().cpu().numpy() if intrinsics is not None else None
        batch_id = str(batch_rows[0]["batch_id"])
        batch_report = {
            "batch_id": batch_id,
            "batch_no": batch_no,
            "batch_size": len(batch_rows),
            "rows": [],
        }
        for pos, row in enumerate(batch_rows):
            item_bucket = per_item[str(row["item_id"])]
            item_bucket["depth"].append(depth[pos].astype("float16"))
            item_bucket["frame_idx"].append(int(row["frame_idx"]))
            if intr_np is not None:
                intr = intr_np[pos]
                item_bucket["intrinsics"].append([float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])])
            output_dir = Path(str(row["output_dir"]))
            row_report = {
                "row_id": row["row_id"],
                "job_id": row["job_id"],
                "item_id": row["item_id"],
                "batch_id": batch_id,
                "stage_id": stage_id,
                "agent_id": row["agent_id"],
                "worker_id": worker_id,
                "attempt_id": row["attempt_id"],
                "frame_idx": row["frame_idx"],
                "rgb_path": row["rgb_path"],
                "output_dir": str(output_dir),
                "image_meta": image_meta[pos],
                "status": "ok",
            }
            item_bucket["rows"].append(row_report)
            batch_report["rows"].append(row_report)
        batch_reports.append(batch_report)

    item_reports = []
    import numpy as np

    for item_id, bucket in sorted(per_item.items()):
        output_dir = Path(str(bucket["rows"][0]["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "depth": np.stack(bucket["depth"]).astype("float16"),
            "frame_idx": np.asarray(bucket["frame_idx"], dtype=np.int32),
        }
        if bucket["intrinsics"]:
            payload["intrinsics_fx_fy_cx_cy"] = np.asarray(bucket["intrinsics"], dtype=np.float32)
        npz_path = output_dir / "unidepth_v2_depth_resident.npz"
        np.savez_compressed(str(npz_path), **payload)
        qc = {
            "schema": "v22_resident_unidepth_item.v0",
            "status": "ok",
            "job_id": rows[0]["job_id"],
            "item_id": item_id,
            "stage_id": stage_id,
            "worker_id": worker_id,
            "model_identity": model_identity,
            "depth_archive": str(npz_path),
            "frame_count": len(bucket["frame_idx"]),
            "rows": bucket["rows"],
            "claim_scope": "Resident UniDepth depth/intrinsics candidate; not object pose, hand state, or contact evidence.",
        }
        qc_path = output_dir / "qc_unidepth_v2_resident.json"
        write_json(qc_path, qc)
        item_reports.append({"item_id": item_id, "status": "ok", "depth_archive": str(npz_path), "qc": str(qc_path), "frame_count": len(bucket["frame_idx"])})

    if torch.cuda.is_available() and device.startswith("cuda"):
        gpu_residency = {
            "device": device,
            "memory_allocated_mb": float(torch.cuda.memory_allocated() / (1024 * 1024)),
            "memory_reserved_mb": float(torch.cuda.memory_reserved() / (1024 * 1024)),
        }
    else:
        gpu_residency = {"device": device, "memory_allocated_mb": None, "memory_reserved_mb": None}
    report = {
        "schema": "v22_resident_unidepth_batch_worker.v0",
        "status": "ok",
        "job_id": rows[0]["job_id"],
        "stage_id": stage_id,
        "worker_id": worker_id,
        "model_identity": model_identity,
        "model_load_count": 1,
        "batch_inference_count": len(batch_reports),
        "batch_sizes": [row["batch_size"] for row in batch_reports],
        "rows_inferred": len(rows),
        "load_started_utc": load_started,
        "load_finished_utc": load_finished,
        "elapsed_s": float(time.time() - started),
        "gpu_residency": gpu_residency,
        "resize_width": resize_width_i,
        "items": item_reports,
        "batches": batch_reports,
        "claim_scope": "One resident UniDepth model instance consumed multiple true frame batches while preserving item boundaries.",
    }
    report_path = output_root / "resident_unidepth_worker_report.json"
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path), "model_load_count": 1, "batch_inference_count": len(batch_reports), "rows_inferred": len(rows)}, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--worker-id", default="unidepth_resident_worker_000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional inference resize width for mixed-shape smoke batches.")
    parser.add_argument("--unidepth-repo", type=Path, default=DEFAULT_UNIDEPTH_REPO)
    parser.add_argument("--unidepth-model", type=Path, default=DEFAULT_UNIDEPTH_MODEL)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
