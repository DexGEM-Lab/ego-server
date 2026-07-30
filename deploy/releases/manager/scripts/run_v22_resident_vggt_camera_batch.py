#!/usr/bin/env python3
"""Run a resident VGGT/Omega-style D4 camera geometry batch stage.

This worker establishes the DROID replacement contract for camera trajectory:
multiple item windows are stacked as one tensor with shape [B, S, 3, H, W], one
already-loaded backend consumes that tensor, and outputs are split back into
per-item DROID-compatible camera trajectory artifacts.

The default `contract` backend is dependency-light and deterministic. It proves
batch shape, item isolation, and downstream artifact compatibility without local
heavy inference. Real `vggt`/`vggt_omega` backends are hooks for the A800 runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v22_model_request_helpers import write_vggt_camera_request

DEFAULT_STAGE_ID = "vggt_omega_camera_geometry_resident"
DEFAULT_WORKER_ID = "vggt_omega_camera_resident_worker_000"


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


def resolve_path(run_root: Path, raw: Any) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [run_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_checkpoint_path(raw: Any, *, request_path: Path | None = None) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    if request_path is not None:
        candidates.append(request_path.parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def request_optional_bool(request: dict[str, Any], key: str) -> bool | None:
    if key not in request:
        return None
    value = request[key]
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"{key} must be a JSON boolean, got {type(value).__name__}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_model_version(*, checkpoint: Path | None, model_id: str, model_file: str, allow_remote_model_download: bool) -> str:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.exists():
            return f"checkpoint={resolved};status=missing"
        stat = resolved.stat()
        return f"checkpoint={resolved};size_bytes={stat.st_size};sha256={file_sha256(resolved)}"
    if allow_remote_model_download:
        return f"remote_model={model_id}/resolve/main/{model_file}"
    return "checkpoint=missing"


def instantiate_vggt_model(model_cls: Any, target_size: int) -> Any:
    try:
        signature = inspect.signature(model_cls)
    except (TypeError, ValueError):
        return model_cls()
    for parameter_name in ("img_size", "image_size", "target_size"):
        if parameter_name in signature.parameters:
            return model_cls(**{parameter_name: int(target_size)})
    return model_cls()


def resolve_calibration_contract(run_root: Path, raw: Any | None) -> Path | None:
    if raw is not None:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = run_root / path
        return path.resolve()
    candidate = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    return candidate.resolve() if candidate.exists() else None


def matrix_to_quat_xyzw(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise RuntimeError(f"rotation must be finite 3x3, got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise RuntimeError("invalid quaternion from rotation matrix")
    quat /= norm
    return quat.astype(float).tolist()


def video_meta_from_manifest(manifest: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    video = manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
    first = frames[0] if frames else {}
    width = video.get("width") or first.get("source_width") or first.get("manifest_width") or first.get("width")
    height = video.get("height") or first.get("source_height") or first.get("manifest_height") or first.get("height")
    fps = manifest.get("fps") or video.get("fps") or 30.0
    frame_count = manifest.get("frame_count") or video.get("frame_count") or len(frames)
    return {"width": int(width), "height": int(height), "fps": float(fps), "frame_count": int(frame_count)}


def pick_frame_rows(
    frames: list[dict[str, Any]],
    *,
    frame_indices: list[int] | None,
    sequence_length: int,
    start_position: int,
    stride: int,
) -> list[dict[str, Any]]:
    if frame_indices is not None:
        by_idx = {int(row.get("frame_idx", pos)): row for pos, row in enumerate(frames)}
        missing = [idx for idx in frame_indices if idx not in by_idx]
        if missing:
            raise RuntimeError(f"requested frame indices absent from manifest: {missing}")
        picked = [by_idx[idx] for idx in frame_indices]
    else:
        if start_position < 0 or stride <= 0:
            raise RuntimeError("start_position must be non-negative and stride must be positive")
        picked = frames[start_position : start_position + sequence_length * stride : stride]
    if len(picked) != sequence_length:
        raise RuntimeError(f"sequence row has {len(picked)} frames, expected {sequence_length}")
    return picked


@dataclass(frozen=True)
class SequenceRow:
    row_id: str
    job_id: str
    item_id: str
    batch_id: str
    stage_id: str
    agent_id: str
    attempt_id: str
    run_root: Path
    raw_frame_manifest: Path
    output_dir: Path
    calibration_contract: Path | None
    frames: list[dict[str, Any]]
    video_meta: dict[str, Any]
    full_source_timeline: bool
    model_request_path: Path | None


def iter_sequence_rows(request: dict[str, Any]) -> list[SequenceRow]:
    job_id = str(request.get("job_id") or "vggt_camera_batch_job")
    stage_id = str(request.get("stage_id") or DEFAULT_STAGE_ID)
    sequence_length = int(request.get("sequence_length") or 0)
    if sequence_length <= 0:
        raise RuntimeError("request.sequence_length must be positive; mixed implicit lengths are not supported")
    stride = int(request.get("frame_stride") or 1)
    rows: list[SequenceRow] = []
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise RuntimeError("request.items must be a non-empty list")
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"items[{item_index}] must be an object")
        item_id = str(item.get("item_id") or f"item_{item_index:06d}")
        if not item.get("run_root"):
            raise RuntimeError(f"items[{item_index}].run_root is required")
        run_root = Path(str(item.get("run_root"))).expanduser().resolve()
        manifest_path = Path(str(item.get("raw_frame_manifest") or run_root / "input" / "raw_frame_manifest" / "manifest.json"))
        manifest_path = resolve_path(run_root, manifest_path)
        manifest = load_json(manifest_path)
        frames = manifest.get("frames") if isinstance(manifest.get("frames"), list) else []
        if not frames:
            raise RuntimeError(f"raw frame manifest has no frames: {manifest_path}")
        explicit_indices = item.get("frame_indices")
        frame_indices = [int(idx) for idx in explicit_indices] if isinstance(explicit_indices, list) else None
        picked = pick_frame_rows(
            frames,
            frame_indices=frame_indices,
            sequence_length=int(item.get("sequence_length") or sequence_length),
            start_position=int(item.get("start_position") or 0),
            stride=int(item.get("frame_stride") or stride),
        )
        if len(picked) != sequence_length:
            raise RuntimeError("mixed sequence lengths in one tensor batch are rejected; bucket by sequence_length")
        meta = video_meta_from_manifest(manifest, frames)
        frame_idx = [int(row.get("frame_idx", pos)) for pos, row in enumerate(picked)]
        expected_all = [int(row.get("frame_idx", pos)) for pos, row in enumerate(frames)]
        full_source = len(frame_idx) == len(expected_all) and frame_idx == expected_all
        batch_index = len(rows) // max(1, int(request.get("batch_size", len(items))))
        batch_id = str(item.get("batch_id") or f"{job_id}_{stage_id}_batch_{batch_index:05d}")
        output_dir = Path(str(item.get("output_dir") or run_root / "measurements" / "camera_trajectory" / "vggt_omega_full_frame")).expanduser()
        if not output_dir.is_absolute():
            output_dir = (run_root / output_dir).resolve()
        calibration_contract = resolve_calibration_contract(
            run_root,
            item.get("calibration_contract", request.get("calibration_contract")),
        )
        request_name = item.get("compat_request_name", request.get("compat_request_name", "droid"))
        model_request_path = run_root / "requests" / f"{request_name}.json" if request_name else None
        rows.append(
            SequenceRow(
                row_id=f"{batch_id}_{item_id}_seq0",
                job_id=job_id,
                item_id=item_id,
                batch_id=batch_id,
                stage_id=stage_id,
                agent_id=str(item.get("agent_id") or request.get("agent_id") or "vggt_camera_batch_agent"),
                attempt_id=str(item.get("attempt_id") or request.get("attempt_id") or "attempt_0001"),
                run_root=run_root,
                raw_frame_manifest=manifest_path,
                output_dir=output_dir,
                calibration_contract=calibration_contract,
                frames=picked,
                video_meta=meta,
                full_source_timeline=full_source,
                model_request_path=model_request_path,
            )
        )
    return rows


def chunks(rows: list[SequenceRow], size: int) -> list[list[SequenceRow]]:
    if size <= 0:
        raise RuntimeError("batch_size must be positive")
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def padded_resize_size(width: int, height: int, target_size: int, patch_multiple: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0 or target_size <= 0 or patch_multiple <= 0:
        raise RuntimeError("invalid image dimensions for VGGT preprocessing")
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / patch_multiple) * patch_multiple
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / patch_multiple) * patch_multiple
    new_width = max(patch_multiple, min(target_size, new_width))
    new_height = max(patch_multiple, min(target_size, new_height))
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    return int(new_width), int(new_height), int(pad_left), int(pad_top)


def source_intrinsics_from_padded(
    intrinsic: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_size: int,
    patch_multiple: int,
) -> list[float]:
    new_width, new_height, pad_left, pad_top = padded_resize_size(source_width, source_height, target_size, patch_multiple)
    sx = new_width / float(source_width)
    sy = new_height / float(source_height)
    return [
        float(intrinsic[0, 0] / sx),
        float(intrinsic[1, 1] / sy),
        float((intrinsic[0, 2] - pad_left) / sx),
        float((intrinsic[1, 2] - pad_top) / sy),
    ]


def load_image_tensor(path: Path, target_size: int, patch_multiple: int, torch: Any | None) -> tuple[Any, dict[str, Any]]:
    image = Image.open(path).convert("RGB")
    source_width, source_height = image.size
    new_width, new_height, pad_left, pad_top = padded_resize_size(source_width, source_height, target_size, patch_multiple)
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(image, (pad_left, pad_top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1)) if torch is None else torch.from_numpy(arr).permute(2, 0, 1)
    return tensor, {
        "source_size": [source_width, source_height],
        "inference_size": [target_size, target_size],
        "resized_size": [new_width, new_height],
        "pad_left_top": [pad_left, pad_top],
        "patch_multiple": patch_multiple,
    }


def frame_rgb_path(row: dict[str, Any], run_root: Path) -> Path:
    raw = row.get("rgb") or row.get("raw_frame_path") or row.get("image_path")
    if not raw:
        raise RuntimeError(f"frame {row.get('frame_idx')} lacks an RGB path")
    path = resolve_path(run_root, raw)
    if not path.exists():
        raise FileNotFoundError(f"RGB frame path does not exist: {path}")
    return path


def load_sequence_tensor(row: SequenceRow, target_size: int, patch_multiple: int, torch: Any | None) -> tuple[Any, list[dict[str, Any]]]:
    tensors = []
    metas = []
    for frame in row.frames:
        tensor, meta = load_image_tensor(frame_rgb_path(frame, row.run_root), target_size, patch_multiple, torch)
        tensors.append(tensor)
        metas.append(meta)
    sequence = np.stack(tensors, axis=0) if torch is None else torch.stack(tensors, dim=0)
    return sequence, metas


class ContractCameraBackend:
    model_name = "vggt_camera_contract_backend"
    model_version = "v0"

    def __init__(self, step_m: float = 0.01) -> None:
        self.step_m = float(step_m)

    def infer(self, batch_tensor: Any, rows: list[SequenceRow], torch: Any | None) -> dict[str, np.ndarray]:
        shape = tuple(int(x) for x in batch_tensor.shape)
        if len(shape) != 5:
            raise RuntimeError(f"contract backend expected [B,S,3,H,W], got {shape}")
        batch, seq, _channels, height, width = shape
        extrinsic = np.zeros((batch, seq, 3, 4), dtype=np.float32)
        intrinsic = np.zeros((batch, seq, 3, 3), dtype=np.float32)
        depth = np.ones((batch, seq, height, width), dtype=np.float16)
        depth_conf = np.ones((batch, seq, height, width), dtype=np.float16)
        for bi, row in enumerate(rows):
            digest = hashlib.sha256(row.item_id.encode("utf-8")).hexdigest()
            item_offset_m = (int(digest[:8], 16) % 1000) / 100000.0
            for si in range(seq):
                center = np.asarray([self.step_m * si, item_offset_m, 0.0], dtype=np.float32)
                extrinsic[bi, si, :3, :3] = np.eye(3, dtype=np.float32)
                extrinsic[bi, si, :3, 3] = -center
                intrinsic[bi, si] = np.asarray(
                    [[0.8 * width, 0.0, 0.5 * width], [0.0, 0.8 * height, 0.5 * height], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                depth[bi, si] = np.float16(1.0 + 0.001 * si)
        return {"extrinsic": extrinsic, "intrinsic": intrinsic, "depth": depth, "depth_conf": depth_conf}


class VggtCameraBackend:
    def __init__(
        self,
        *,
        variant: str,
        repo_root: Path,
        checkpoint: Path | None,
        model_id: str,
        model_file: str,
        device: str,
        allow_remote_model_download: bool = False,
        target_size: int = 518,
    ) -> None:
        self.variant = variant
        self.model_name = variant
        self.repo_root = repo_root
        self.checkpoint = checkpoint.expanduser().resolve() if checkpoint is not None else None
        self.model_id = model_id
        self.model_file = model_file
        self.device = device
        self.allow_remote_model_download = bool(allow_remote_model_download)
        self.target_size = int(target_size)
        self.model_version = checkpoint_model_version(
            checkpoint=self.checkpoint,
            model_id=self.model_id,
            model_file=self.model_file,
            allow_remote_model_download=self.allow_remote_model_download,
        )
        self._loaded = False

    def _load(self, torch: Any) -> None:
        if self._loaded:
            return
        if self.variant == "vggt":
            vggt_root = self.repo_root / "third_party" / "vggt"
            if vggt_root.exists():
                sys.path.insert(0, str(vggt_root))
            try:
                from vggt.models.vggt import VGGT
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            except ModuleNotFoundError as exc:
                raise RuntimeError(f"VGGT import failed; provide third_party/vggt or install vggt in the selected runtime: {exc}") from exc

            if self.checkpoint is None and not self.allow_remote_model_download:
                raise RuntimeError("vggt backend requires --checkpoint unless --allow-remote-model-download is set explicitly")
            self.model = instantiate_vggt_model(VGGT, self.target_size)
            if self.checkpoint is not None:
                state = torch.load(str(self.checkpoint), map_location="cpu")
            else:
                checkpoint_url = f"https://huggingface.co/{self.model_id}/resolve/main/{self.model_file}"
                state = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
            self.model.load_state_dict(state)
            self.pose_converter = pose_encoding_to_extri_intri
        elif self.variant == "vggt_omega":
            omega_root = self.repo_root / "third_party" / "vggt-omega"
            if omega_root.exists():
                sys.path.insert(0, str(omega_root))
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.pose_enc import encoding_to_camera

            if self.checkpoint is None:
                raise RuntimeError("vggt_omega backend requires --checkpoint")
            self.model = VGGTOmega()
            self.model.load_state_dict(torch.load(str(self.checkpoint), map_location="cpu"))
            self.pose_converter = encoding_to_camera
        else:
            raise RuntimeError(f"unsupported backend variant: {self.variant}")
        self.model = self.model.to(self.device).eval()
        self._loaded = True

    def infer(self, batch_tensor: Any, rows: list[SequenceRow], torch: Any | None) -> dict[str, np.ndarray]:
        if torch is None:
            raise RuntimeError("real VGGT/Omega backends require torch; use backend=contract for dependency-light local tests")
        self._load(torch)
        images = batch_tensor.to(self.device, non_blocking=True)
        cuda_device = str(self.device).startswith("cuda")
        amp_dtype = torch.bfloat16 if cuda_device and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if cuda_device else "cpu", dtype=amp_dtype, enabled=cuda_device):
                predictions = self.model(images)
        extrinsic, intrinsic = self.pose_converter(predictions["pose_enc"], images.shape[-2:])
        depth = predictions.get("depth")
        depth_conf = predictions.get("depth_conf")
        if depth is None:
            depth_np = np.empty((len(rows), images.shape[1], 0, 0), dtype=np.float16)
        else:
            if depth.ndim == 5 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth_np = depth.detach().float().cpu().numpy().astype(np.float16)
        if depth_conf is None:
            depth_conf_np = np.empty((len(rows), images.shape[1], 0, 0), dtype=np.float16)
        else:
            depth_conf_np = depth_conf.detach().float().cpu().numpy().astype(np.float16)
        return {
            "extrinsic": extrinsic.detach().float().cpu().numpy().astype(np.float32),
            "intrinsic": intrinsic.detach().float().cpu().numpy().astype(np.float32),
            "depth": depth_np,
            "depth_conf": depth_conf_np,
        }


def backend_from_request(args: argparse.Namespace, request: dict[str, Any]) -> Any:
    backend = str(request.get("backend") or args.backend)
    if backend == "contract":
        return ContractCameraBackend(step_m=float(request.get("contract_step_m", args.contract_step_m)))
    if backend in {"vggt", "vggt_omega"}:
        checkpoint = request.get("checkpoint") or args.checkpoint
        checkpoint_path = resolve_checkpoint_path(checkpoint, request_path=args.request) if checkpoint else None
        request_download_permission = request_optional_bool(request, "allow_remote_model_download")
        allow_remote_model_download = bool(args.allow_remote_model_download) or bool(request_download_permission)
        return VggtCameraBackend(
            variant=backend,
            repo_root=Path(str(request.get("repo_root") or args.repo_root)).resolve(),
            checkpoint=checkpoint_path,
            model_id=str(request.get("model_id") or args.model_id),
            model_file=str(request.get("model_file") or args.model_file),
            device=str(request.get("device") or args.device),
            allow_remote_model_download=allow_remote_model_download,
            target_size=int(request.get("target_size") or args.target_size),
        )
    raise RuntimeError(f"unknown backend: {backend}")


def camera_centers_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    centers = []
    for row in extrinsic:
        rotation = row[:3, :3]
        translation = row[:3, 3]
        centers.append(-rotation.T @ translation)
    return np.asarray(centers, dtype=np.float64)


def make_local_world_poses(extrinsic: np.ndarray, scale: float, anchor_i: int = 0) -> np.ndarray:
    centers = camera_centers_from_extrinsic(extrinsic)
    anchor_rotation = extrinsic[anchor_i, :3, :3]
    anchor_center = centers[anchor_i]
    transforms = []
    for i in range(len(extrinsic)):
        rotation_world_camera = anchor_rotation @ extrinsic[i, :3, :3].T
        translation_world_camera = float(scale) * (anchor_rotation @ (centers[i] - anchor_center))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_world_camera
        transform[:3, 3] = translation_world_camera
        transforms.append(transform)
    return np.stack(transforms, axis=0)


def camera_backend_label(backend_name: str) -> str:
    if backend_name == "vggt":
        return "vggt_resident_batch"
    if backend_name == "vggt_omega":
        return "vggt_omega_resident_batch"
    return backend_name


def write_item_camera_outputs(
    *,
    row: SequenceRow,
    prediction: dict[str, np.ndarray],
    image_meta: list[dict[str, Any]],
    backend_name: str,
    backend_version: str,
    worker_id: str,
    batch_id: str,
    batch_tensor_shape: list[int],
    target_size: int,
    patch_multiple: int,
    translation_scale: float,
    scale_status: str,
) -> dict[str, Any]:
    row.output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = np.asarray([int(frame.get("frame_idx", i)) for i, frame in enumerate(row.frames)], dtype=np.int32)
    extrinsic = np.asarray(prediction["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(prediction["intrinsic"], dtype=np.float32)
    depth = np.asarray(prediction.get("depth", np.empty((len(frame_idx), 0, 0))), dtype=np.float16)
    depth_conf = np.asarray(prediction.get("depth_conf", np.empty((len(frame_idx), 0, 0))), dtype=np.float16)
    transforms = make_local_world_poses(extrinsic.astype(np.float64), float(translation_scale), 0).astype(np.float32)
    source_intrinsics = []
    for i, meta in enumerate(image_meta):
        width, height = meta["source_size"]
        source_intrinsics.append(
            source_intrinsics_from_padded(
                intrinsic[i],
                source_width=int(width),
                source_height=int(height),
                target_size=int(target_size),
                patch_multiple=int(patch_multiple),
            )
        )
    source_intrinsics_np = np.asarray(source_intrinsics, dtype=np.float32)
    translations = transforms[:, :3, 3]
    step = np.linalg.norm(np.diff(translations, axis=0), axis=1) if len(translations) > 1 else np.zeros(0, dtype=np.float32)
    pose_rows = []
    for i, idx in enumerate(frame_idx.tolist()):
        quat = matrix_to_quat_xyzw(transforms[i, :3, :3])
        pose = transforms[i, :3, 3].astype(float).tolist() + quat
        pose_rows.append(
            {
                "frame_idx": int(idx),
                "pose_world_camera_xyzw": pose,
                "T_world_camera": transforms[i].astype(float).tolist(),
                "source_intrinsics_fx_fy_cx_cy": source_intrinsics_np[i].astype(float).tolist(),
                "pose_source": backend_name,
            }
        )
    npz_path = row.output_dir / "droid_dense_trajectory.npz"
    np.savez_compressed(
        npz_path,
        frame_idx=frame_idx,
        pose_world_camera_xyzw=np.asarray([entry["pose_world_camera_xyzw"] for entry in pose_rows], dtype=np.float32),
        T_world_camera=transforms.astype(np.float32),
        intrinsics_source=np.median(source_intrinsics_np, axis=0).astype(np.float32),
        intrinsics_source_per_frame=source_intrinsics_np.astype(np.float32),
        fps=np.asarray([float(row.video_meta["fps"])], dtype=np.float32),
        backend=np.asarray([backend_name]),
    )
    dense_json_path = row.output_dir / "droid_dense_trajectory.json"
    write_json(dense_json_path, {"frames": pose_rows})
    geometry_npz = row.output_dir / "vggt_camera_geometry.npz"
    payload = {
        "frame_idx": frame_idx,
        "extrinsic_camera_from_world": extrinsic.astype(np.float32),
        "intrinsic_padded": intrinsic.astype(np.float32),
        "T_world_camera": transforms.astype(np.float32),
        "source_intrinsics_fx_fy_cx_cy": source_intrinsics_np.astype(np.float32),
        "translation_scale": np.asarray([float(translation_scale)], dtype=np.float32),
    }
    if depth.size:
        payload["depth"] = depth
    if depth_conf.size:
        payload["depth_conf"] = depth_conf
    np.savez_compressed(geometry_npz, **payload)
    qc = {
        "status": "ok",
        "schema": "v22_vggt_camera_trajectory_qc.v0",
        "method": "v22_resident_vggt_camera_batch",
        "backend": backend_name,
        "backend_version": backend_version,
        "worker_id": worker_id,
        "job_id": row.job_id,
        "item_id": row.item_id,
        "batch_id": batch_id,
        "stage_id": row.stage_id,
        "processed_frames": int(len(frame_idx)),
        "dense_trajectory_frames": int(len(frame_idx)),
        "expected_raw_manifest_frames": int(row.video_meta["frame_count"]),
        "full_source_timeline": bool(row.full_source_timeline),
        "pose_convention": "T_world_camera, pose vector [tx, ty, tz, qx, qy, qz, qw]",
        "calibration_source": "VGGT/Omega predicted intrinsics mapped from padded inference image to source pixels",
        "trajectory_path_length": float(step.sum()),
        "median_step": float(np.median(step)) if step.size else 0.0,
        "p95_step": float(np.percentile(step, 95)) if step.size else 0.0,
        "scale_status": scale_status,
        "batch_tensor_shape": batch_tensor_shape,
        "outputs": {
            "dense_npz": str(npz_path),
            "dense_json": str(dense_json_path),
            "keyframe_reconstruction": None,
            "keyframes": None,
            "vggt_camera_geometry": str(geometry_npz),
        },
        "claim_scope": "D4 camera trajectory candidate from a resident VGGT/Omega-style tensor-batch worker; video-derived uncertain gauge, not fixed-gauge metric accuracy.",
    }
    qc_path = row.output_dir / "droid_qc.json"
    write_json(qc_path, qc)
    stage = {
        "schema": "v22_camera_trajectory_stage.v0",
        "status": "ok",
        "run_root": str(row.run_root),
        "clip": None,
        "calibration_contract": str(row.calibration_contract) if row.calibration_contract is not None else None,
        "camera_backend": camera_backend_label(backend_name),
        "replacement_for": "D4_droid_head_camera_trajectory",
        "backend": backend_name,
        "backend_version": backend_version,
        "worker_id": worker_id,
        "job_id": row.job_id,
        "item_id": row.item_id,
        "batch_id": batch_id,
        "stage_id": row.stage_id,
        "outputs": {
            "output_dir": str(row.output_dir),
            "dense_npz": str(npz_path),
            "dense_json": str(dense_json_path),
            "qc_json": str(qc_path),
            "vggt_camera_geometry": str(geometry_npz),
        },
        "model_request": str(row.model_request_path) if row.model_request_path is not None else None,
        "claim_scope": qc["claim_scope"],
        "gauge_declaration": {
            "trajectory_frame": "VGGT/Omega local anchor world gauge",
            "scale_status": scale_status,
            "metric_anchor_needed": "device VIO/SLAM/IMU, fiducial/mocap, known-size scene/object measurement, or benchmark GT",
        },
        "batch_tensor_shape": batch_tensor_shape,
        "source_frame_indices": frame_idx.astype(int).tolist(),
    }
    stage_path = row.output_dir / "v22_camera_trajectory_stage.json"
    write_json(stage_path, stage)
    if row.model_request_path is not None:
        write_vggt_camera_request(
            row.run_root,
            output_dir=row.output_dir,
            request_path=row.model_request_path,
            calibration_contract=row.calibration_contract,
            backend=backend_name,
            stage="D4_camera_trajectory",
            parameters={
                "input_contract": {"required_fields": ["input_video", "camera", "output_dir"]},
                "batch_contract": {"tensor_shape": "[B,S,3,H,W]", "sequence_length": int(len(frame_idx)), "item_isolation": True},
                "compatibility": {"writes_droid_compatible_camera_artifacts": True},
            },
        )
    return {
        "item_id": row.item_id,
        "status": "ok",
        "output_dir": str(row.output_dir),
        "dense_npz": str(npz_path),
        "dense_json": str(dense_json_path),
        "qc_json": str(qc_path),
        "stage_json": str(stage_path),
        "model_request": str(row.model_request_path) if row.model_request_path is not None else None,
        "frame_count": int(len(frame_idx)),
        "source_frame_indices": frame_idx.astype(int).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    request = load_json(args.request)
    backend_name = str(request.get("backend") or args.backend)
    torch = None
    if backend_name != "contract":
        import torch as torch_module

        torch = torch_module
    rows = iter_sequence_rows(request)
    batch_size = int(request.get("batch_size") or args.batch_size)
    target_size = int(request.get("target_size") or args.target_size)
    patch_multiple = int(request.get("patch_multiple") or args.patch_multiple)
    output_root = Path(str(request.get("output_root") or args.output_root or Path(rows[0].run_root) / "resident_vggt_camera_batch")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worker_id = str(request.get("worker_id") or args.worker_id)
    translation_scale = float(request.get("translation_scale", args.translation_scale))
    scale_status = str(request.get("scale_status") or args.scale_status)
    backend = backend_from_request(args, request)
    started = time.time()
    load_started = utc_now()
    model_identity = {
        "model_name": getattr(backend, "model_name", str(request.get("backend") or args.backend)),
        "model_version": getattr(backend, "model_version", "unknown"),
        "stage_id": str(request.get("stage_id") or DEFAULT_STAGE_ID),
        "worker_id": worker_id,
        "backend": str(request.get("backend") or args.backend),
        "device": str(request.get("device") or args.device),
    }
    load_finished = utc_now()
    item_reports: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    for batch_no, batch_rows in enumerate(chunks(rows, batch_size)):
        sequences = []
        sequence_metas = []
        for row in batch_rows:
            sequence, metas = load_sequence_tensor(row, target_size, patch_multiple, torch)
            sequences.append(sequence)
            sequence_metas.append(metas)
        shapes = {tuple(int(x) for x in tensor.shape) for tensor in sequences}
        if len(shapes) != 1:
            raise RuntimeError(f"mixed sequence tensor shapes in batch {batch_no}: {sorted(shapes)}")
        batch_tensor = np.stack(sequences, axis=0) if torch is None else torch.stack(sequences, dim=0)
        batch_tensor_shape = [int(x) for x in batch_tensor.shape]
        prediction = backend.infer(batch_tensor, batch_rows, torch)
        if tuple(prediction["extrinsic"].shape[:3]) != (len(batch_rows), batch_tensor_shape[1], 3):
            raise RuntimeError(f"backend returned invalid extrinsic shape {prediction['extrinsic'].shape}")
        batch_id = batch_rows[0].batch_id
        batch_item_reports = []
        for item_pos, row in enumerate(batch_rows):
            item_prediction = {
                "extrinsic": prediction["extrinsic"][item_pos],
                "intrinsic": prediction["intrinsic"][item_pos],
                "depth": prediction.get("depth", np.empty((len(row.frames), 0, 0)))[item_pos] if "depth" in prediction else np.empty((len(row.frames), 0, 0)),
                "depth_conf": prediction.get("depth_conf", np.empty((len(row.frames), 0, 0)))[item_pos] if "depth_conf" in prediction else np.empty((len(row.frames), 0, 0)),
            }
            item_report = write_item_camera_outputs(
                row=row,
                prediction=item_prediction,
                image_meta=sequence_metas[item_pos],
                backend_name=model_identity["model_name"],
                backend_version=model_identity["model_version"],
                worker_id=worker_id,
                batch_id=batch_id,
                batch_tensor_shape=batch_tensor_shape,
                target_size=target_size,
                patch_multiple=patch_multiple,
                translation_scale=translation_scale,
                scale_status=scale_status,
            )
            item_reports.append(item_report)
            batch_item_reports.append(item_report)
        batch_reports.append(
            {
                "batch_id": batch_id,
                "batch_no": batch_no,
                "batch_size": len(batch_rows),
                "batch_tensor_shape": batch_tensor_shape,
                "items": batch_item_reports,
                "status": "ok",
            }
        )
    if torch is not None and torch.cuda.is_available() and str(request.get("device") or args.device).startswith("cuda"):
        gpu_residency = {
            "device": str(request.get("device") or args.device),
            "memory_allocated_mb": float(torch.cuda.memory_allocated() / (1024 * 1024)),
            "memory_reserved_mb": float(torch.cuda.memory_reserved() / (1024 * 1024)),
        }
    else:
        gpu_residency = {"device": str(request.get("device") or args.device), "memory_allocated_mb": None, "memory_reserved_mb": None}
    report = {
        "schema": "v22_resident_vggt_camera_batch_worker.v0",
        "status": "ok",
        "job_id": rows[0].job_id,
        "stage_id": rows[0].stage_id,
        "worker_id": worker_id,
        "model_identity": model_identity,
        "model_load_count": 1,
        "batch_inference_count": len(batch_reports),
        "batch_sizes": [row["batch_size"] for row in batch_reports],
        "batch_tensor_shapes": [row["batch_tensor_shape"] for row in batch_reports],
        "rows_inferred": len(rows),
        "items": item_reports,
        "batches": batch_reports,
        "target_size": target_size,
        "patch_multiple": patch_multiple,
        "sequence_length": int(request.get("sequence_length")),
        "load_started_utc": load_started,
        "load_finished_utc": load_finished,
        "elapsed_s": float(time.time() - started),
        "gpu_residency": gpu_residency,
        "claim_scope": "One resident VGGT/Omega-style camera worker consumed true [B,S,3,H,W] sequence batches while preserving item boundaries and writing DROID-compatible D4 artifacts.",
    }
    report_path = output_root / "resident_vggt_camera_worker_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "model_load_count": 1,
                "batch_inference_count": len(batch_reports),
                "batch_tensor_shapes": report["batch_tensor_shapes"],
                "rows_inferred": len(rows),
            },
            indent=2,
        )
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--worker-id", default=DEFAULT_WORKER_ID)
    parser.add_argument("--backend", choices=("contract", "vggt", "vggt_omega"), default="contract")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--patch-multiple", type=int, default=14)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--scale-status", default="video_derived_uncertain_without_external_metric_anchor")
    parser.add_argument("--contract-step-m", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-remote-model-download", action="store_true")
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--model-file", default="model.pt")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
