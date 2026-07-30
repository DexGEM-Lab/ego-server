#!/usr/bin/env python3
"""Materialize Feishu Ray model responses into existing annotation-stage artifacts."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.call_feishu_ray_service import ServiceCallerError, call_service_arrays, retryable_response_info

DEFAULT_PROFILE = REPO_ROOT / "configs" / "feishu_ray_services.json"
DEFAULT_WILOR_ROOT = Path(os.environ.get("V22_WILOR_DIR", "/home/zjh/ego-annation-checkpoints/wilor_model"))
SERVICE_IMAGE_WIDTH = 960
SERVICE_IMAGE_HEIGHT = 540
DROID_TARGET_AREA = 384 * 512
DROID_MODEL_REVISION = "droid-v1"
DROID_PINNED_RELEASE = "9c989fadce9f02ff02dbc9ff8e8075e86d0a0513"
DROID_SESSION_OPTIONS = {
    # Use upstream motion/keyframe selection. Retaining every frame as a
    # keyframe makes DROID's factor graph scale with the full video and can
    # exhaust an 80GB replica during finalize; dense_source carries skipped
    # frames separately.
    "filter_thresh": 8.0,
    "warmup": 8,
    "keyframe_thresh": 4.0,
    "frontend_thresh": 16.0,
    "frontend_window": 25,
    "frontend_radius": 2,
    "frontend_nms": 1,
    "backend_thresh": 22.0,
    "backend_radius": 2,
    "backend_nms": 3,
    "upsample": True,
    "beta": 0.3,
    "stereo": False,
}
DROID_COMPENSATIONS = {
    "pinned_service_release": DROID_PINNED_RELEASE,
    "rgb_channel_compensation": (
        "submit channel-symmetric grayscale RGB so the deployed extra R/B reversal is invariant"
    ),
    "dynamic_pixel_compensation": "zero every model-grid pixel where the submitted dynamic-ignore mask is positive",
    "mask_polarity_compensation": "submit float32 positive=ignore mask without inversion",
    "motion_filter_mode": "use deployed upstream motion filtering; keyframe_added is an uncertain per-frame measurement",
    "dense_timeline": "deployed dense_source records every admitted source frame for trajectory filling",
    "buffer_right_sizing": (
        "allocate only max(frame_count + 1, warmup + 1) slots instead of the historical 1024-slot default"
    ),
}
OWNERSHIP_KEYS = (
    "request_id",
    "job_id",
    "item_id",
    "stage_id",
    "source_id",
    "schema_version",
    "source_timestamp_s",
    "submitted_at",
)

ServiceCall = Callable[..., dict[str, Any]]


class FeishuRayAdapterError(RuntimeError):
    """A service response cannot truthfully drive an annotation artifact."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        response_received: bool | None = None,
        response_status: int | None = None,
        response_headers: Mapping[str, str] | None = None,
        raw_response_bytes: bytes | None = None,
        retryable: bool = False,
        retry_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.response_received = response_received
        self.response_status = response_status
        self.response_headers = dict(response_headers) if response_headers is not None else None
        self.raw_response_bytes = bytes(raw_response_bytes) if raw_response_bytes is not None else None
        self.retryable = bool(retryable)
        self.retry_attempts = int(retry_attempts)


def adapter_int(value: Any, *, code: str, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeishuRayAdapterError(code, message)
    return value


def adapter_float(value: Any, *, code: str, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeishuRayAdapterError(code, message)
    try:
        return float(value)
    except OverflowError as exc:
        raise FeishuRayAdapterError(code, message) from exc


def adapter_string(value: Any, *, code: str, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeishuRayAdapterError(code, message)
    return value


def exception_evidence(error: Exception) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "code": error.code if isinstance(error, FeishuRayAdapterError) else "unexpected_session_failure",
        "message": str(error),
    }


def add_exception_note(error: BaseException, note: str) -> None:
    """Attach diagnostic context without ever replacing the primary exception."""
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            try:
                add_note(str(note))
                return
            except BaseException:
                pass
        notes = getattr(error, "__notes__", None)
        if not isinstance(notes, list):
            notes = []
            setattr(error, "__notes__", notes)
        notes.append(str(note))
    except BaseException:
        return


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeishuRayAdapterError("invalid_json_object", f"expected JSON object: {path}")
    return payload


def load_profile(path: Path) -> dict[str, Any]:
    payload = load_json_object(path)
    if payload.get("schema") != "ego.annotation.feishu_ray_services_profile.v1":
        raise FeishuRayAdapterError("invalid_service_profile", f"unsupported profile schema: {payload.get('schema')}")
    services = payload.get("services")
    if not isinstance(services, dict):
        raise FeishuRayAdapterError("invalid_service_profile", f"profile lacks services: {path}")
    return payload


def profile_base_url(profile: dict[str, Any], service: str, override: str | None = None) -> str:
    if override:
        return override.rstrip("/")
    services = profile.get("services")
    row = services.get(service) if isinstance(services, dict) else None
    if not isinstance(row, dict) or not isinstance(row.get("base_url"), str):
        raise FeishuRayAdapterError("service_endpoint_missing", f"profile lacks base URL for {service}")
    return str(row["base_url"]).rstrip("/")


def make_ownership(
    *,
    job_id: str,
    item_id: str,
    stage_id: str,
    source_id: str,
    source_timestamp_s: float | None,
) -> dict[str, Any]:
    return {
        "request_id": f"{job_id}-{stage_id}-{uuid4().hex[:16]}",
        "job_id": job_id,
        "item_id": item_id,
        "stage_id": stage_id,
        "source_id": source_id,
        "schema_version": "ego.model-service.v1",
        "source_timestamp_s": source_timestamp_s,
        "submitted_at": utc_now(),
    }


def ownership_matches(expected: Mapping[str, Any], actual: Any) -> bool:
    if not isinstance(actual, Mapping):
        return False
    for key in OWNERSHIP_KEYS:
        lhs = expected.get(key)
        rhs = actual.get(key)
        if key == "source_timestamp_s" and lhs is not None and rhs is not None:
            if (
                isinstance(lhs, bool)
                or isinstance(rhs, bool)
                or not isinstance(lhs, (int, float))
                or not isinstance(rhs, (int, float))
            ):
                return False
            try:
                timestamp_matches = math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1.0e-9)
            except OverflowError:
                return False
            if not timestamp_matches:
                return False
        elif lhs != rhs:
            return False
    return True


def require_success(
    report: dict[str, Any],
    *,
    expected_ownership: Mapping[str, Any],
    route: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    raw_status = report.get("http_status")
    status = adapter_int(
        0 if raw_status is None else raw_status,
        code="feishu_ray_response_envelope_invalid",
        message=f"{route}: response HTTP status is invalid",
    )
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise FeishuRayAdapterError("feishu_ray_response_metadata_missing", f"{route}: response metadata is missing")
    error = metadata.get("error")
    if report.get("status") != "ok" or status < 200 or status >= 300:
        retryable, _, _ = retryable_response_info(report)
        raise FeishuRayAdapterError(
            "feishu_ray_http_error",
            f"{route}: HTTP {status}; error={error!r}",
            retryable=retryable,
        )
    if isinstance(error, dict):
        code = str(error.get("code") or "service_error")
        retryable, _, _ = retryable_response_info(report)
        raise FeishuRayAdapterError(
            f"feishu_ray_{code}",
            f"{route}: {error.get('message') or error}",
            retryable=retryable,
        )
    if error is not None:
        raise FeishuRayAdapterError(
            "feishu_ray_response_envelope_invalid",
            f"{route}: invalid error field {error!r}",
        )
    if not ownership_matches(expected_ownership, metadata.get("ownership")):
        raise FeishuRayAdapterError(
            "feishu_ray_ownership_mismatch",
            f"{route}: response ownership does not match request ownership",
        )
    result = metadata.get("result")
    if result is None:
        result = metadata.get("camera_state")
    if not isinstance(result, dict):
        raise FeishuRayAdapterError("feishu_ray_result_missing", f"{route}: response result is missing")
    if result.get("ownership") is not None and not ownership_matches(expected_ownership, result.get("ownership")):
        raise FeishuRayAdapterError(
            "feishu_ray_ownership_mismatch",
            f"{route}: nested result ownership does not match request ownership",
        )
    arrays: dict[str, dict[str, Any]] = {}
    array_rows = report.get("arrays")
    if not isinstance(array_rows, list):
        raise FeishuRayAdapterError("feishu_ray_invalid_array_part", f"{route}: response arrays must be a list")
    for row in array_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise FeishuRayAdapterError("feishu_ray_invalid_array_part", f"{route}: malformed array part")
        name = str(row["name"])
        if name in arrays:
            raise FeishuRayAdapterError("feishu_ray_duplicate_array", f"{route}: duplicate array {name}")
        arrays[name] = row
    return metadata, result, arrays


def decode_array(
    arrays: Mapping[str, dict[str, Any]],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtypes: tuple[str, ...] = ("float32",),
) -> np.ndarray:
    row = arrays.get(name)
    if row is None:
        raise FeishuRayAdapterError("feishu_ray_array_missing", f"response lacks array {name}")
    dtype = str(row.get("dtype"))
    if dtype not in dtypes:
        raise FeishuRayAdapterError(
            "feishu_ray_array_dtype_mismatch",
            f"array {name} dtype {dtype}, expected one of {dtypes}",
        )
    try:
        observed_shape = tuple(int(dim) for dim in row.get("shape", ()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeishuRayAdapterError("feishu_ray_array_shape_invalid", f"array {name} has an invalid shape") from exc
    if shape is not None and observed_shape != shape:
        raise FeishuRayAdapterError(
            "feishu_ray_array_shape_mismatch",
            f"array {name} shape {observed_shape}, expected {shape}",
        )
    data = row.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise FeishuRayAdapterError("feishu_ray_array_bytes_missing", f"array {name} has no decoded bytes")
    try:
        array = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(observed_shape)
    except (TypeError, ValueError) as exc:
        raise FeishuRayAdapterError("feishu_ray_array_decode_failed", f"array {name}: {exc}") from exc
    return np.array(array, copy=True)


def require_finite(array: np.ndarray, name: str) -> None:
    value = np.asanyarray(array)
    if value.ndim >= 3:
        for index in range(value.shape[0]):
            if not np.isfinite(value[index]).all():
                raise FeishuRayAdapterError("feishu_ray_nonfinite_array", f"array {name}[{index}] contains NaN or infinity")
    elif not np.isfinite(value).all():
        raise FeishuRayAdapterError("feishu_ray_nonfinite_array", f"array {name} contains NaN or infinity")


def service_spatial(source_width: int, source_height: int) -> dict[str, Any]:
    if source_width <= 0 or source_height <= 0:
        raise FeishuRayAdapterError(
            "invalid_source_image_size",
            f"source image size must be positive, got {source_width}x{source_height}",
        )
    scale_x = SERVICE_IMAGE_WIDTH / float(source_width)
    scale_y = SERVICE_IMAGE_HEIGHT / float(source_height)
    inverse_x = source_width / float(SERVICE_IMAGE_WIDTH)
    inverse_y = source_height / float(SERVICE_IMAGE_HEIGHT)
    return {
        "source_size": {"width": int(source_width), "height": int(source_height)},
        "model_size": {"width": SERVICE_IMAGE_WIDTH, "height": SERVICE_IMAGE_HEIGHT},
        "color_space": "RGB",
        "pixel_transform": {
            "source_to_model": [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            "model_to_source": [[inverse_x, 0.0, 0.0], [0.0, inverse_y, 0.0], [0.0, 0.0, 1.0]],
            "resize_mode": "anisotropic_fixed_service_grid",
            "crop_xywh": None,
            "pad_ltrb": None,
        },
        "K_px": None,
    }


def resize_rgb_to_service(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise FeishuRayAdapterError(
            "invalid_source_rgb",
            f"source RGB must be uint8[H,W,3], got dtype={source.dtype} shape={source.shape}",
        )
    image = Image.fromarray(source, mode="RGB")
    resized = image.resize((SERVICE_IMAGE_WIDTH, SERVICE_IMAGE_HEIGHT), resample=Image.Resampling.BILINEAR)
    model_rgb = np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    if model_rgb.shape != (SERVICE_IMAGE_HEIGHT, SERVICE_IMAGE_WIDTH, 3):
        raise AssertionError(f"service RGB resize returned {model_rgb.shape}")
    return model_rgb


def resample_model_scalar_to_source(array: np.ndarray, *, source_width: int, source_height: int) -> np.ndarray:
    from PIL import Image

    model_array = np.asarray(array, dtype=np.float32)
    if model_array.shape != (SERVICE_IMAGE_HEIGHT, SERVICE_IMAGE_WIDTH):
        raise FeishuRayAdapterError(
            "feishu_ray_array_shape_mismatch",
            f"model scalar grid shape {model_array.shape}, expected {(SERVICE_IMAGE_HEIGHT, SERVICE_IMAGE_WIDTH)}",
        )
    image = Image.fromarray(model_array, mode="F")
    resized = image.resize((int(source_width), int(source_height)), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.float32))


def lift_model_intrinsics(K_model: np.ndarray, spatial: Mapping[str, Any]) -> np.ndarray:
    transform = np.asarray(spatial["pixel_transform"]["model_to_source"], dtype=np.float64)
    K_source = transform @ np.asarray(K_model, dtype=np.float64)
    return K_source.astype(np.asarray(K_model).dtype, copy=False)


def lift_model_boxes(boxes: np.ndarray, spatial: Mapping[str, Any]) -> np.ndarray:
    source_boxes = np.asarray(boxes, dtype=np.float64).copy()
    transform = np.asarray(spatial["pixel_transform"]["model_to_source"], dtype=np.float64)
    if source_boxes.ndim != 2 or source_boxes.shape[1:] != (4,):
        raise FeishuRayAdapterError("feishu_ray_invalid_hand_boxes", f"invalid box shape {source_boxes.shape}")
    source_boxes[:, [0, 2]] *= transform[0, 0]
    source_boxes[:, [1, 3]] *= transform[1, 1]
    return source_boxes.astype(np.float32)


def lift_model_masks(
    masks: np.ndarray,
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    from PIL import Image

    model_masks = np.asarray(masks, dtype=np.uint8)
    if model_masks.ndim != 3 or model_masks.shape[1:] != (SERVICE_IMAGE_HEIGHT, SERVICE_IMAGE_WIDTH):
        raise FeishuRayAdapterError(
            "feishu_ray_array_shape_mismatch",
            f"model mask grid shape {model_masks.shape}, expected [N,{SERVICE_IMAGE_HEIGHT},{SERVICE_IMAGE_WIDTH}]",
        )
    source_masks = np.empty((model_masks.shape[0], int(source_height), int(source_width)), dtype=np.uint8)
    for index, mask in enumerate(model_masks):
        resized = Image.fromarray(mask, mode="L").resize(
            (int(source_width), int(source_height)),
            resample=Image.Resampling.NEAREST,
        )
        source_masks[index] = np.asarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(source_masks)


def resolve_manifest_rgb(run_root: Path, repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    text = str(raw)
    candidates = [path]
    if text.startswith("outputs/"):
        candidates.append(Path("output") / Path(text).relative_to("outputs"))
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        candidates.append(Path("output") / Path(text[len(historical_prefix) :]))
    if not path.is_absolute():
        candidates.extend([repo_root / path, run_root / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"raw manifest RGB file not found: {raw}")


def call_typed(
    caller: ServiceCall,
    *,
    base_url: str,
    route: str,
    metadata: dict[str, Any],
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]],
    timeout_s: float,
    retry_events: list[dict[str, Any]] | None = None,
    retry_max_wait_s: float = 0.0,
    retry_initial_delay_s: float = 1.0,
    allow_retryable: bool = True,
) -> dict[str, Any]:
    """Synchronously call one route, waiting only on explicit retryable admission errors.

    A response that reached the caller is never retried merely because it is an
    error. Retry requires the service's explicit retryable marker (or a standard
    admission HTTP status). DROID finalize disables this path because any
    received finalize response retires the session and is terminal.
    """
    if retry_max_wait_s < 0.0 or retry_initial_delay_s < 0.0:
        raise FeishuRayAdapterError("invalid_retry_policy", "retry delays and maximum wait must be non-negative")
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            report = caller(
                base_url=base_url,
                route=route,
                metadata=metadata,
                arrays=arrays,
                timeout_s=timeout_s,
            )
        except ServiceCallerError as exc:
            if exc.response_received is not None and type(exc.response_received) is not bool:
                raise FeishuRayAdapterError(
                    "feishu_ray_response_receipt_invalid",
                    f"{route}: caller returned invalid response_received provenance {exc.response_received!r}",
                    response_received=None,
                    response_status=exc.response_status,
                    response_headers=exc.response_headers,
                    raw_response_bytes=exc.raw_response_bytes,
                ) from exc
            raise FeishuRayAdapterError(
                exc.code,
                f"{route}: {exc}",
                response_received=exc.response_received,
                response_status=exc.response_status,
                response_headers=exc.response_headers,
                raw_response_bytes=exc.raw_response_bytes,
            ) from exc
        retryable, retry_after_s, error_code = retryable_response_info(report)
        if not allow_retryable or not retryable:
            if retry_events and attempt:
                retry_events[-1]["terminal_attempt"] = attempt
            return report
        elapsed_s = time.monotonic() - started
        if retry_max_wait_s > 0.0 and elapsed_s >= retry_max_wait_s:
            event = {
                "event": "retryable_response_exhausted",
                "route": route,
                "attempt": attempt + 1,
                "error_code": error_code,
                "elapsed_s": float(elapsed_s),
                "max_wait_s": float(retry_max_wait_s),
                "ownership": dict(metadata.get("ownership") or {}),
            }
            if retry_events is not None:
                retry_events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            return report
        fallback_delay = min(30.0, float(retry_initial_delay_s) * (2.0 ** min(attempt, 8)))
        delay_s = retry_after_s if retry_after_s is not None else fallback_delay
        if retry_max_wait_s > 0.0:
            delay_s = min(delay_s, max(0.0, retry_max_wait_s - elapsed_s))
        event = {
            "event": "retryable_response_waiting",
            "route": route,
            "attempt": attempt + 1,
            "error_code": error_code,
            "delay_s": float(delay_s),
            "elapsed_s": float(elapsed_s),
            "ownership": dict(metadata.get("ownership") or {}),
        }
        if retry_events is not None:
            retry_events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if delay_s > 0.0:
            time.sleep(delay_s)
        attempt += 1


def decode_unidepth_response(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    _, result, arrays = require_success(report, expected_ownership=ownership, route="/unidepth.infer")
    depth = decode_array(arrays, "depth_m", shape=(height, width), dtypes=("float32",))
    confidence = decode_array(arrays, "confidence", shape=(height, width), dtypes=("float32",))
    K_px = decode_array(arrays, "K_px", shape=(3, 3), dtypes=("float32", "float64"))
    require_finite(depth, "depth_m")
    require_finite(confidence, "confidence")
    require_finite(K_px, "K_px")
    if np.any(depth <= 0.0):
        raise FeishuRayAdapterError("feishu_ray_invalid_depth", "UniDepth depth_m must be strictly positive")
    if float(np.max(depth)) > float(np.finfo(np.float16).max):
        raise FeishuRayAdapterError("feishu_ray_depth_float16_overflow", "UniDepth depth exceeds legacy float16 archive range")
    if K_px[0, 0] <= 0.0 or K_px[1, 1] <= 0.0 or not np.allclose(K_px[2], [0.0, 0.0, 1.0], atol=1.0e-5):
        raise FeishuRayAdapterError("feishu_ray_invalid_intrinsics", f"invalid UniDepth K_px: {K_px.tolist()}")
    spatial = result.get("spatial")
    if isinstance(spatial, dict):
        model_size = spatial.get("model_size")
        expected_size = {"width": width, "height": height}
        if isinstance(model_size, dict) and model_size != expected_size:
            raise FeishuRayAdapterError(
                "feishu_ray_spatial_mismatch",
                f"UniDepth result model_size {model_size} != {expected_size}",
            )
    return depth, confidence, K_px, result


def write_unidepth_artifact(
    *,
    output_dir: Path,
    frame_idx: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    frame_idx = np.asarray(frame_idx, dtype=np.int32)
    depth = np.asanyarray(depth)
    confidence = np.asanyarray(confidence)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if depth.ndim != 3 or confidence.shape != depth.shape or depth.shape[0] == 0:
        raise FeishuRayAdapterError("unidepth_artifact_shape_mismatch", "depth/confidence must be non-empty matching [N,H,W] arrays")
    if frame_idx.shape != (depth.shape[0],) or intrinsics.shape != (depth.shape[0], 4):
        raise FeishuRayAdapterError("unidepth_artifact_timeline_mismatch", "frame_idx/intrinsics do not match depth timeline")
    require_finite(depth, "depth")
    require_finite(confidence, "confidence")
    require_finite(intrinsics, "intrinsics_fx_fy_cx_cy")
    for index in range(depth.shape[0]):
        if np.any(depth[index] <= 0.0):
            raise FeishuRayAdapterError("unidepth_artifact_invalid_values", f"depth[{index}] must be positive")
        if float(np.max(depth[index])) > float(np.finfo(np.float16).max):
            raise FeishuRayAdapterError("feishu_ray_depth_float16_overflow", f"depth[{index}] exceeds float16 range")
    if np.any(intrinsics[:, :2] <= 0.0):
        raise FeishuRayAdapterError("unidepth_artifact_invalid_values", "focal values must be positive")
    confidence_min = math.inf
    confidence_max = -math.inf
    confidence_frame_medians: list[float] = []
    for index in range(confidence.shape[0]):
        frame_confidence = np.asarray(confidence[index], dtype=np.float64)
        confidence_min = min(confidence_min, float(np.min(frame_confidence)))
        confidence_max = max(confidence_max, float(np.max(frame_confidence)))
        confidence_frame_medians.append(float(np.median(frame_confidence)))
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "unidepth_v2_depth.npz"
    staging = output_dir / f".{archive.name}.staging-{uuid4().hex}.npz"
    try:
        np.savez_compressed(
            staging,
            depth=depth.astype(np.float16, copy=False),
            confidence=confidence.astype(np.float32, copy=False),
            frame_idx=frame_idx,
            intrinsics_fx_fy_cx_cy=intrinsics,
            source_size=np.asarray([depth.shape[2], depth.shape[1]], dtype=np.int32),
        )
        os.replace(staging, archive)
    except Exception:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        raise
    report = {
        "schema": "v21_unidepth_v2_candidate.v0",
        "status": "ok",
        "method": "feishu_ray_unidepth_adapter",
        "model_revision": provenance.get("model_revision"),
        "service_profile": provenance.get("service_profile"),
        "service_base_url": provenance.get("service_base_url"),
        "depth_archive": str(archive),
        "depth_archive_sha256": sha256_file(archive),
        "frame_count": int(depth.shape[0]),
        "first_frame": int(frame_idx[0]) if len(frame_idx) else None,
        "last_frame": int(frame_idx[-1]) if len(frame_idx) else None,
        "confidence": {
            "min": confidence_min,
            "median_of_frame_medians": float(np.median(np.asarray(confidence_frame_medians, dtype=np.float64))),
            "max": confidence_max,
            "archive_key": "confidence",
            "semantics": (
                "raw finite UniDepth v2 exp(logconfidence) score trained against depth-error magnitude; "
                "unbounded, lower generally indicates better confidence/error evidence, and not a calibrated probability"
            ),
            "normalization": "none; source-grid resampling only, with finite service values preserved as float32",
            "downstream_use": "relative score/quantile reasoning only unless separately calibrated",
        },
        "trace_batch_ids": provenance.get("trace_batch_ids", []),
        "service_frames": provenance.get("service_frames", []),
        "claim_scope": "UniDepth metric depth/intrinsics candidate with service confidence; not object pose or contact evidence.",
    }
    write_json(output_dir / "qc_unidepth_v2.json", report)
    return report


def validated_manifest_frames(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise FeishuRayAdapterError("raw_frame_manifest_empty", "raw frame manifest has no frames")
    previous_timestamp = -math.inf
    fps = float(manifest.get("fps") or 30.0)
    if not math.isfinite(fps) or fps <= 0.0:
        raise FeishuRayAdapterError("raw_frame_manifest_invalid", f"invalid manifest fps {fps}")
    validated: list[dict[str, Any]] = []
    for position, row in enumerate(frames):
        if not isinstance(row, dict) or row.get("frame_idx") is None or row.get("rgb") is None:
            raise FeishuRayAdapterError("raw_frame_manifest_invalid", f"invalid raw frame row: {row!r}")
        frame_idx = int(row["frame_idx"])
        if frame_idx != position:
            raise FeishuRayAdapterError(
                "raw_frame_manifest_order_invalid",
                f"frame row {position} has frame_idx={frame_idx}; expected a contiguous zero-based timeline",
            )
        timestamp = float(row.get("time_s", frame_idx / fps))
        if not math.isfinite(timestamp) or (position > 0 and timestamp <= previous_timestamp):
            raise FeishuRayAdapterError(
                "raw_frame_manifest_order_invalid",
                f"frame {frame_idx} has non-finite or decreasing timestamp {timestamp}",
            )
        previous_timestamp = timestamp
        validated.append(row)
    return validated


def run_unidepth(args: argparse.Namespace, *, caller: ServiceCall = call_service_arrays) -> dict[str, Any]:
    from PIL import Image

    started = time.time()
    run_root = args.run_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    manifest = load_json_object(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    frames = validated_manifest_frames(manifest)
    profile = load_profile(args.profile.expanduser().resolve())
    base_url = profile_base_url(profile, "unidepth", args.base_url)
    job_id = str(args.job_id or manifest.get("case_id") or run_root.name)
    intrinsics: list[list[float]] = []
    frame_indices: list[int] = []
    batch_ids: list[str] = []
    retry_events: list[dict[str, Any]] = []
    service_frames: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".feishu_unidepth_", dir=run_root) as scratch_raw:
        scratch = Path(scratch_raw)
        depth_store: np.memmap | None = None
        confidence_store: np.memmap | None = None
        expected_hw: tuple[int, int] | None = None
        for position, row in enumerate(frames):
            frame_idx = int(row["frame_idx"])
            timestamp = float(row.get("time_s", frame_idx / float(manifest.get("fps") or 30.0)))
            rgb_path = resolve_manifest_rgb(run_root, repo_root, str(row["rgb"]))
            with Image.open(rgb_path) as image:
                rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
            source_height, source_width = rgb.shape[:2]
            if expected_hw is None:
                expected_hw = (source_height, source_width)
                depth_store = np.memmap(
                    scratch / "depth.float16.mmap",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(frames), source_height, source_width),
                )
                confidence_store = np.memmap(
                    scratch / "confidence.float32.mmap",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(frames), source_height, source_width),
                )
            elif expected_hw != (source_height, source_width):
                raise FeishuRayAdapterError(
                    "unidepth_frame_shape_changed",
                    f"frame {frame_idx} shape {(source_height, source_width)} != {expected_hw}",
                )
            spatial = service_spatial(source_width, source_height)
            model_rgb = resize_rgb_to_service(rgb)
            ownership = make_ownership(
                job_id=job_id,
                item_id=f"frame-{frame_idx:06d}",
                stage_id="unidepth.infer",
                source_id=f"frame-{frame_idx:06d}",
                source_timestamp_s=timestamp,
            )
            metadata = {
                "ownership": ownership,
                "spatial": spatial,
                "model_revision": "unidepth-v2-vitl14-corrected",
                "options": {},
            }
            response = call_typed(
                caller,
                base_url=base_url,
                route="/unidepth.infer",
                metadata=metadata,
                arrays={"rgb": (model_rgb.tobytes(), tuple(model_rgb.shape), "uint8")},
                timeout_s=float(args.timeout_s),
                retry_events=retry_events,
                retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)),
                retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)),
            )
            model_depth, model_confidence, K_model, result = decode_unidepth_response(
                response,
                ownership=ownership,
                height=SERVICE_IMAGE_HEIGHT,
                width=SERVICE_IMAGE_WIDTH,
            )
            depth = resample_model_scalar_to_source(
                model_depth,
                source_width=source_width,
                source_height=source_height,
            )
            confidence = resample_model_scalar_to_source(
                model_confidence,
                source_width=source_width,
                source_height=source_height,
            )
            K_source = lift_model_intrinsics(K_model, spatial)
            if depth_store is None or confidence_store is None:
                raise AssertionError("UniDepth stores were not initialized")
            depth_store[position] = depth.astype(np.float16)
            confidence_store[position] = np.asarray(confidence, dtype=np.float32)
            intrinsics.append(
                [
                    float(K_source[0, 0]),
                    float(K_source[1, 1]),
                    float(K_source[0, 2]),
                    float(K_source[1, 2]),
                ]
            )
            frame_indices.append(frame_idx)
            trace = result.get("trace")
            if isinstance(trace, dict) and trace.get("batch_id"):
                batch_ids.append(str(trace["batch_id"]))
            service_frames.append(
                {
                    "frame_idx": frame_idx,
                    "ownership": ownership,
                    "model_revision": result.get("model_revision"),
                    "trace": trace if isinstance(trace, dict) else None,
                    "spatial": spatial,
                    "K_model_px": K_model.astype(float).tolist(),
                    "K_source_px": K_source.astype(float).tolist(),
                    "depth_response_space": "model_grid",
                    "depth_artifact_space": "source_grid",
                    "confidence_semantics": (
                        "raw finite UniDepth v2 exp(logconfidence) score trained against depth-error magnitude; "
                        "unbounded, lower generally indicates better confidence/error evidence, not probability"
                    ),
                    "confidence_transform": "model-grid values resampled spatially only; no value normalization or calibration",
                }
            )
        if depth_store is None or confidence_store is None:
            raise AssertionError("UniDepth stores are empty")
        depth_store.flush()
        confidence_store.flush()
        report = write_unidepth_artifact(
            output_dir=run_root / "measurements" / "depth_candidates" / "unidepth_v2",
            frame_idx=np.asarray(frame_indices, dtype=np.int32),
            depth=depth_store,
            confidence=confidence_store,
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            provenance={
                "model_revision": "unidepth-v2-vitl14-corrected",
                "service_profile": profile.get("profile"),
                "service_base_url": base_url,
                "trace_batch_ids": batch_ids,
                "retry_events": retry_events,
                "service_frames": service_frames,
            },
        )
        del depth_store, confidence_store
    report["elapsed_s"] = float(time.time() - started)
    write_json(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "qc_unidepth_v2.json", report)
    return report


def decode_hands_response(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    height: int,
    width: int,
) -> dict[str, Any]:
    _, result, arrays = require_success(report, expected_ownership=ownership, route="/hands.detect")
    detection = result.get("detection")
    if not isinstance(detection, dict):
        raise FeishuRayAdapterError("feishu_ray_detection_missing", "hands.detect result lacks detection")
    n_hands = int(detection.get("n_hands", -1))
    if n_hands < 0:
        raise FeishuRayAdapterError("feishu_ray_detection_count_invalid", f"invalid n_hands={n_hands}")
    boxes = decode_array(arrays, "boxes", shape=(n_hands, 4), dtypes=("float32",))
    scores = decode_array(arrays, "scores", shape=(n_hands,), dtypes=("float32",))
    sides = decode_array(arrays, "sides", shape=(n_hands,), dtypes=("uint8",))
    masks = decode_array(arrays, "masks", shape=(n_hands, height, width), dtypes=("uint8",))
    visibility = decode_array(arrays, "visibility", shape=(n_hands,), dtypes=("float32",))
    uncertainty = decode_array(arrays, "uncertainty", shape=(n_hands,), dtypes=("float32",))
    for name, array in (("boxes", boxes), ("scores", scores), ("visibility", visibility), ("uncertainty", uncertainty)):
        require_finite(array, name)
    if n_hands:
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise FeishuRayAdapterError("feishu_ray_invalid_hand_boxes", "hands.detect returned a non-positive box")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise FeishuRayAdapterError("feishu_ray_invalid_hand_scores", "hands.detect scores must be in [0,1]")
        if np.any((sides != 0) & (sides != 1)):
            raise FeishuRayAdapterError("feishu_ray_invalid_hand_side", "hands.detect sides must be 0 or 1")
        if np.any((masks != 0) & (masks != 1)):
            raise FeishuRayAdapterError("feishu_ray_invalid_hand_masks", "hands.detect masks must be binary")
        if np.any(visibility < 0.0) or np.any(visibility > 1.0) or np.any(uncertainty < 0.0):
            raise FeishuRayAdapterError("feishu_ray_invalid_hand_uncertainty", "invalid visibility/uncertainty")
    return {
        "n_hands": n_hands,
        "boxes": boxes,
        "scores": scores,
        "sides": sides,
        "masks": masks,
        "visibility": visibility,
        "uncertainty": uncertainty,
        "result": result,
    }


def project_full_image(points: np.ndarray, cam_t: np.ndarray, focal: float, img_size: np.ndarray) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(focal)
    K[1, 1] = float(focal)
    K[0, 2] = float(img_size[0]) / 2.0
    K[1, 2] = float(img_size[1]) / 2.0
    translated = np.asarray(points, dtype=np.float32) + np.asarray(cam_t, dtype=np.float32)[None]
    z = translated[:, 2:3]
    if np.any(z <= 1.0e-6) or not np.isfinite(translated).all():
        raise FeishuRayAdapterError("feishu_ray_invalid_wilor_camera_geometry", "WiLoR joints project behind the camera")
    normalized = translated / z
    return (K @ normalized.T).T[:, :2]


def require_rotation_matrices(array: np.ndarray, name: str) -> None:
    rotations = np.asarray(array, dtype=np.float64).reshape(-1, 3, 3)
    gram = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
    determinants = np.linalg.det(rotations)
    if not np.allclose(gram, np.eye(3, dtype=np.float64)[None], atol=1.0e-3, rtol=1.0e-3) or not np.allclose(
        determinants,
        1.0,
        atol=1.0e-3,
        rtol=1.0e-3,
    ):
        raise FeishuRayAdapterError("feishu_ray_invalid_wilor_rotation", f"{name} contains a non-rotation matrix")


def build_wilor_candidate(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    detector_score: float,
    bbox_xyxy: np.ndarray,
    detector_visibility: float,
    detector_uncertainty: float,
    expected_side: int,
    img_size: np.ndarray,
) -> dict[str, Any]:
    _, result, arrays = require_success(report, expected_ownership=ownership, route="/wilor.reconstruct")
    mano = result.get("mano")
    if not isinstance(mano, dict):
        raise FeishuRayAdapterError("feishu_ray_mano_missing", "wilor.reconstruct result lacks MANO output")
    handedness = int(result.get("handedness", -1))
    if handedness != int(expected_side) or handedness not in {0, 1}:
        raise FeishuRayAdapterError(
            "feishu_ray_handedness_mismatch",
            f"WiLoR handedness {handedness} != detector side {expected_side}",
        )
    global_orient = decode_array(arrays, "global_orient", shape=(1, 3, 3), dtypes=("float32",))
    hand_pose = decode_array(arrays, "hand_pose", shape=(15, 3, 3), dtypes=("float32",))
    betas = decode_array(arrays, "betas", shape=(10,), dtypes=("float32",))
    vertices = decode_array(arrays, "vertices", shape=(778, 3), dtypes=("float32",))
    joints = decode_array(arrays, "joints", shape=(21, 3), dtypes=("float32",))
    cam_t = decode_array(arrays, "cam_t_full", shape=(3,), dtypes=("float32",))
    pred_cam = decode_array(arrays, "pred_cam", shape=(3,), dtypes=("float32",))
    confidence = decode_array(arrays, "confidence", shape=(1,), dtypes=("float32",))
    uncertainty = decode_array(arrays, "uncertainty", shape=(1,), dtypes=("float32",))
    keypoints_row = arrays.get("keypoints_2d")
    if keypoints_row is None:
        raise FeishuRayAdapterError("feishu_ray_array_missing", "response lacks keypoints_2d")
    keypoints_shape = tuple(int(dim) for dim in keypoints_row.get("shape", ()))
    if keypoints_shape != (778, 2):
        raise FeishuRayAdapterError("feishu_ray_array_shape_mismatch", f"keypoints_2d shape {keypoints_shape}, expected (778, 2)")
    projected_surface = decode_array(arrays, "keypoints_2d", shape=keypoints_shape, dtypes=("float32",))
    for name, array in (
        ("global_orient", global_orient),
        ("hand_pose", hand_pose),
        ("betas", betas),
        ("vertices", vertices),
        ("joints", joints),
        ("cam_t_full", cam_t),
        ("pred_cam", pred_cam),
        ("confidence", confidence),
        ("uncertainty", uncertainty),
        ("keypoints_2d", projected_surface),
    ):
        require_finite(array, name)
    require_rotation_matrices(global_orient, "global_orient")
    require_rotation_matrices(hand_pose, "hand_pose")
    focal = float(mano.get("focal_length", 0.0))
    if int(mano.get("n_vertices", -1)) != 778:
        raise FeishuRayAdapterError("feishu_ray_invalid_wilor_geometry", f"WiLoR MANO vertex count is {mano.get('n_vertices')}")
    if (
        not math.isfinite(focal)
        or focal <= 0.0
        or float(confidence[0]) < 0.0
        or float(confidence[0]) > 1.0
        or float(uncertainty[0]) < 0.0
    ):
        raise FeishuRayAdapterError("feishu_ray_invalid_wilor_uncertainty", "invalid WiLoR focal/confidence/uncertainty")
    if np.any((vertices + cam_t[None])[:, 2] <= 1.0e-6):
        raise FeishuRayAdapterError("feishu_ray_invalid_wilor_camera_geometry", "WiLoR mesh projects behind the camera")
    source_img_size = np.asarray(img_size, dtype=np.float32)
    joints2d = project_full_image(joints, cam_t, focal, source_img_size)
    expected_surface = project_full_image(vertices, cam_t, focal, source_img_size)
    if not np.allclose(projected_surface, expected_surface, atol=1.0e-3, rtol=1.0e-4):
        max_error = float(np.max(np.abs(projected_surface - expected_surface)))
        raise FeishuRayAdapterError(
            "feishu_ray_wilor_projection_mismatch",
            f"WiLoR source-pixel surface projection differs from returned geometry by {max_error:.6g}px",
        )
    side = "right" if handedness == 1 else "left"
    return {
        "backend": "WiLoR/Feishu-Ray",
        "side": side,
        "coordinate_space": "source_pixels",
        "source_image_size": {"width": int(source_img_size[0]), "height": int(source_img_size[1])},
        "detector_score": float(detector_score),
        "bbox_xyxy": np.asarray(bbox_xyxy, dtype=np.float32).astype(float).tolist(),
        "cam_t": cam_t.astype(float).tolist(),
        "pred_cam": pred_cam.astype(float).tolist(),
        "focal_length": focal,
        "joints3d_camera": joints.astype(float).tolist(),
        "joints2d": joints2d.astype(float).tolist(),
        "joints2d_coordinate_space": "source_pixels",
        "projected_surface_2d": projected_surface.astype(float).tolist(),
        "projected_surface_2d_coordinate_space": "source_pixels",
        "mano_params": {
            "representation": "rotation_matrices_pose2rot_false",
            "global_orient": global_orient.astype(float).tolist(),
            "hand_pose": hand_pose.astype(float).tolist(),
            "betas": betas.astype(float).tolist(),
        },
        # Existing fusion treats these as root-relative MANO geometry and adds cam_t.
        "vertices_camera": vertices.astype(float).tolist(),
        "vertices_camera_sample": vertices[::10].astype(float).tolist(),
        "filter_status": "measured_raw",
        "detector_visibility": float(detector_visibility),
        "detector_uncertainty": float(detector_uncertainty),
        "service_confidence": float(confidence[0]),
        "service_uncertainty": float(uncertainty[0]),
        "projected_surface_2d_shape": list(projected_surface.shape),
        "service_provenance": {
            "ownership": dict(ownership),
            "model_revision": result.get("model_revision"),
            "trace": result.get("trace"),
            "reconstruction_metadata_semantics": "box_center_box_size_img_size_and_2d_projections_are_source_pixels",
            "geometry_handedness_semantics": (
                "service_reflected_canonical_right_vertices_and_joints_along_x_before_camera_lift"
                if handedness == 0
                else "service_native_right_hand_geometry"
            ),
        },
    }


def patch_wilor_preprocessor_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def resolve_wilor_config(wilor_root: Path, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if os.environ.get("V22_WILOR_CONFIG"):
        candidates.append(Path(os.environ["V22_WILOR_CONFIG"]).expanduser())
    if os.environ.get("V22_WILOR_CHECKPOINT_ROOT"):
        candidates.append(Path(os.environ["V22_WILOR_CHECKPOINT_ROOT"]).expanduser() / "model_config.yaml")
    candidates.extend(
        [
            wilor_root / "pretrained_models" / "model_config.yaml",
            wilor_root / "model_config.yaml",
            wilor_root.parent / "wilor" / "model_config.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"WiLoR preprocessing config not found; checked {[str(path) for path in candidates]}")


def load_wilor_preprocessor(wilor_root: Path, config_path: Path):
    patch_wilor_preprocessor_imports()
    root = wilor_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WiLoR preprocessing repository missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wilor.configs import get_config
    from wilor.datasets.vitdet_dataset import ViTDetDataset

    previous = Path.cwd()
    os.chdir(root)
    try:
        config = get_config(str(config_path), update_cachedir=True)
        config.defrost()
        if "BBOX_SHAPE" not in config.MODEL:
            config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()
    finally:
        os.chdir(previous)
    return config, ViTDetDataset


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def depth_archive_for_diagnosis(run_root: Path) -> str | None:
    report_path = run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json"
    if report_path.is_file():
        report = load_json_object(report_path)
        value = report.get("primary_depth_archive")
        if isinstance(value, str) and Path(value).is_file():
            return value
    for path in (
        run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz",
        run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz",
    ):
        if path.is_file():
            return str(path)
    return None


def detector_visibility_state(visibility: float) -> str:
    return "visible" if float(visibility) >= 0.5 else "partially_visible"


def build_detector_side_states(
    observations: list[dict[str, Any]],
    *,
    ownership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for side_index, side_name in ((0, "left"), (1, "right")):
        matches = [row for row in observations if int(row["side_index"]) == side_index]
        if matches:
            primary = max(matches, key=lambda row: float(row["score"]))
            state = str(primary["visibility_state"])
            states.append(
                {
                    "side": side_name,
                    "visibility_state": state,
                    "occlusion_state": (
                        "observed_visible_surface_no_occluder_inferred"
                        if state == "visible"
                        else "partial_detector_visibility_occluder_unresolved"
                    ),
                    "occluder_owner": None,
                    "occluder_owner_status": "not_inferred_from_detector_only",
                    "detector_observation_indices": [int(row["detection_index"]) for row in matches],
                    "primary_detection_index": int(primary["detection_index"]),
                    "detector_visibility": float(primary["visibility"]),
                    "detector_uncertainty": float(primary["uncertainty"]),
                    "ownership": dict(ownership),
                }
            )
        else:
            states.append(
                {
                    "side": side_name,
                    "visibility_state": "unresolved",
                    "occlusion_state": "missing_detector_evidence_occluded_vs_out_of_frame_unresolved",
                    "occluder_owner": None,
                    "occluder_owner_status": "unresolved_no_depth_order",
                    "detector_observation_indices": [],
                    "primary_detection_index": None,
                    "detector_visibility": None,
                    "detector_uncertainty": None,
                    "ownership": dict(ownership),
                }
            )
    return states


def write_hands_detector_artifact(
    *,
    output_dir: Path,
    detector_frames: list[dict[str, Any]],
    mask_records: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    service_profile: str | None,
    service_base_url: str,
) -> tuple[Path, Path]:
    packed_width = (int(source_width) * int(source_height) + 7) // 8
    if mask_records:
        packed_masks = np.stack([np.asarray(row["packed_mask"], dtype=np.uint8) for row in mask_records])
    else:
        packed_masks = np.empty((0, packed_width), dtype=np.uint8)
    if packed_masks.shape != (len(mask_records), packed_width):
        raise FeishuRayAdapterError(
            "hands_detector_mask_archive_shape_mismatch",
            f"packed masks have shape {packed_masks.shape}, expected {(len(mask_records), packed_width)}",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "hands_source_masks.npz"
    np.savez_compressed(
        mask_path,
        masks_packbits=packed_masks,
        frame_idx=np.asarray([row["frame_idx"] for row in mask_records], dtype=np.int32),
        detection_idx=np.asarray([row["detection_index"] for row in mask_records], dtype=np.int16),
        side=np.asarray([row["side_index"] for row in mask_records], dtype=np.uint8),
        score=np.asarray([row["score"] for row in mask_records], dtype=np.float32),
        visibility=np.asarray([row["visibility"] for row in mask_records], dtype=np.float32),
        uncertainty=np.asarray([row["uncertainty"] for row in mask_records], dtype=np.float32),
        bbox_xyxy_source=np.asarray([row["bbox_xyxy_source"] for row in mask_records], dtype=np.float32).reshape(-1, 4),
        source_size=np.asarray([source_width, source_height], dtype=np.int32),
        model_size=np.asarray([SERVICE_IMAGE_WIDTH, SERVICE_IMAGE_HEIGHT], dtype=np.int32),
        mask_bit_count=np.asarray(source_width * source_height, dtype=np.int64),
        packbits_bitorder=np.asarray("little"),
    )
    artifact_path = output_dir / "hands_detector_timeline.json"
    artifact = {
        "schema": "ego.annotation.hands_detector_timeline.v1",
        "backend": "hands.detect/Feishu-Ray",
        "service_profile": service_profile,
        "service_base_url": service_base_url,
        "frame_count": len(detector_frames),
        "coordinate_semantics": {
            "request_rgb": "uint8_RGB_model_grid",
            "response_boxes": "model_pixels_lifted_to_source_pixels",
            "response_masks": "model_grid_nearest_neighbor_lifted_to_source_grid",
            "artifact_boxes": "source_pixels_xyxy",
            "artifact_masks": "source_grid_binary_packbits_row_major",
            "model_size": {"width": SERVICE_IMAGE_WIDTH, "height": SERVICE_IMAGE_HEIGHT},
            "source_size": {"width": int(source_width), "height": int(source_height)},
        },
        "visibility_state_semantics": {
            "visible": "detector observation with visibility >= 0.5",
            "partially_visible": "detector observation with visibility < 0.5",
            "occluded": "reserved for later temporal/depth-order evidence; not inferred by this detector adapter",
            "out_of_frame": "reserved for later timeline evidence; not inferred from detector absence",
            "unresolved": "no same-side detector observation; occluded versus out-of-frame is unresolved",
        },
        "mask_archive": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "array_key": "masks_packbits",
            "shape": [int(value) for value in packed_masks.shape],
            "dtype": "uint8",
            "encoding": "numpy.packbits_axis_flat_bitorder_little",
            "decoded_mask_shape": [int(source_height), int(source_width)],
            "coordinate_space": "source_grid",
        },
        "frames": detector_frames,
        "claim_scope": "Full-timeline source-grid detector evidence; no occluded hand pose, occluder ownership, contact, or MANO acceptance.",
    }
    write_json(artifact_path, artifact)
    return artifact_path, mask_path


def run_wilor(args: argparse.Namespace, *, caller: ServiceCall = call_service_arrays) -> dict[str, Any]:
    import cv2

    from scripts.run_v21_wilor_hand_candidates import diagnose_hand_candidates

    started = time.time()
    run_root = args.run_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    manifest = load_json_object(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    frames = validated_manifest_frames(manifest)
    profile = load_profile(args.profile.expanduser().resolve())
    hands_base_url = profile_base_url(profile, "hands_wilor", args.base_url)
    wilor_base_url = profile_base_url(profile, "wilor", args.base_url)
    job_id = str(args.job_id or manifest.get("case_id") or run_root.name)
    wilor_root = args.wilor_root.expanduser().resolve()
    wilor_config = resolve_wilor_config(wilor_root, args.wilor_config)
    config, dataset_class = load_wilor_preprocessor(wilor_root, wilor_config)
    raw_frames: list[dict[str, Any]] = []
    detector_frames: list[dict[str, Any]] = []
    mask_records: list[dict[str, Any]] = []
    detected_frames = 0
    detected_hands = 0
    service_calls = {"hands.detect": 0, "wilor.reconstruct": 0}
    retry_events: list[dict[str, Any]] = []
    expected_hw: tuple[int, int] | None = None
    for frame_row in frames:
        frame_idx = int(frame_row["frame_idx"])
        timestamp = float(frame_row.get("time_s", frame_idx / float(manifest.get("fps") or 30.0)))
        rgb_path = resolve_manifest_rgb(run_root, repo_root, str(frame_row["rgb"]))
        frame_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FeishuRayAdapterError("raw_frame_decode_failed", f"could not decode frame {rgb_path}")
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        source_height, source_width = rgb.shape[:2]
        if expected_hw is None:
            expected_hw = (source_height, source_width)
        elif expected_hw != (source_height, source_width):
            raise FeishuRayAdapterError(
                "wilor_frame_shape_changed",
                f"frame {frame_idx} shape {(source_height, source_width)} != {expected_hw}",
            )
        spatial = service_spatial(source_width, source_height)
        model_rgb = resize_rgb_to_service(rgb)
        detect_ownership = make_ownership(
            job_id=job_id,
            item_id=f"frame-{frame_idx:06d}",
            stage_id="hands.detect",
            source_id=f"frame-{frame_idx:06d}",
            source_timestamp_s=timestamp,
        )
        detect_metadata = {
            "ownership": detect_ownership,
            "spatial": spatial,
            "model_revision": "hands-yolo-sam2.1-hiera-l",
            "options": {},
        }
        detect_report = call_typed(
            caller,
            base_url=hands_base_url,
            route="/hands.detect",
            metadata=detect_metadata,
            arrays={"rgb": (model_rgb.tobytes(), tuple(model_rgb.shape), "uint8")},
            timeout_s=float(args.timeout_s),
            retry_events=retry_events,
            retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)),
            retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)),
        )
        service_calls["hands.detect"] += 1
        detection = decode_hands_response(
            detect_report,
            ownership=detect_ownership,
            height=SERVICE_IMAGE_HEIGHT,
            width=SERVICE_IMAGE_WIDTH,
        )
        model_boxes = np.asarray(detection["boxes"], dtype=np.float32)
        source_boxes = lift_model_boxes(model_boxes, spatial)
        source_masks = lift_model_masks(
            detection["masks"],
            source_width=source_width,
            source_height=source_height,
        )
        detector_observations: list[dict[str, Any]] = []
        for det_index in range(int(detection["n_hands"])):
            mask_index = len(mask_records)
            visibility = float(detection["visibility"][det_index])
            uncertainty = float(detection["uncertainty"][det_index])
            score = float(detection["scores"][det_index])
            side_index = int(detection["sides"][det_index])
            packed_mask = np.packbits(source_masks[det_index].reshape(-1), bitorder="little")
            mask_record = {
                "packed_mask": packed_mask,
                "frame_idx": frame_idx,
                "detection_index": det_index,
                "side_index": side_index,
                "score": score,
                "visibility": visibility,
                "uncertainty": uncertainty,
                "bbox_xyxy_source": source_boxes[det_index].astype(float).tolist(),
            }
            mask_records.append(mask_record)
            detector_observations.append(
                {
                    "detection_index": det_index,
                    "side": "right" if side_index == 1 else "left",
                    "side_index": side_index,
                    "score": score,
                    "visibility": visibility,
                    "visibility_state": detector_visibility_state(visibility),
                    "uncertainty": uncertainty,
                    "bbox_xyxy_model": model_boxes[det_index].astype(float).tolist(),
                    "bbox_xyxy_source": source_boxes[det_index].astype(float).tolist(),
                    "bbox_coordinate_space": "source_pixels",
                    "mask_archive_index": mask_index,
                    "mask_coordinate_space": "source_grid",
                    "mask_source_pixel_count": int(np.count_nonzero(source_masks[det_index])),
                    "ownership": dict(detect_ownership),
                }
            )
        detection_result = detection["result"]
        detector_frames.append(
            {
                "frame_idx": frame_idx,
                "time_s": timestamp,
                "ownership": detect_ownership,
                "model_revision": detection_result.get("model_revision"),
                "trace": detection_result.get("trace"),
                "spatial": spatial,
                "observations": detector_observations,
                "side_states": build_detector_side_states(detector_observations, ownership=detect_ownership),
            }
        )
    if expected_hw is None:
        raise AssertionError("hands detector source timeline is empty")
    detector_output_dir = run_root / "measurements" / "hand_detections" / "feishu_ray_hands"
    detector_path, detector_masks_path = write_hands_detector_artifact(
        output_dir=detector_output_dir,
        detector_frames=detector_frames,
        mask_records=mask_records,
        source_width=expected_hw[1],
        source_height=expected_hw[0],
        service_profile=profile.get("profile"),
        service_base_url=hands_base_url,
    )

    persisted_detector = load_json_object(detector_path)
    persisted_frames = persisted_detector.get("frames")
    if not isinstance(persisted_frames, list) or len(persisted_frames) != len(frames):
        raise FeishuRayAdapterError(
            "hands_detector_timeline_mismatch",
            f"persisted detector timeline has {len(persisted_frames) if isinstance(persisted_frames, list) else None} frames, expected {len(frames)}",
        )
    for position, frame_row in enumerate(frames):
        frame_idx = int(frame_row["frame_idx"])
        timestamp = float(frame_row.get("time_s", frame_idx / float(manifest.get("fps") or 30.0)))
        detector_frame = persisted_frames[position]
        if not isinstance(detector_frame, dict) or int(detector_frame.get("frame_idx", -1)) != frame_idx:
            raise FeishuRayAdapterError(
                "hands_detector_timeline_mismatch",
                f"persisted detector row {position} does not match source frame {frame_idx}",
            )
        observations = detector_frame.get("observations")
        if not isinstance(observations, list):
            raise FeishuRayAdapterError("hands_detector_timeline_mismatch", f"frame {frame_idx} observations are invalid")
        spatial = detector_frame.get("spatial")
        detect_ownership = detector_frame.get("ownership")
        if not isinstance(spatial, dict) or not isinstance(detect_ownership, dict):
            raise FeishuRayAdapterError("hands_detector_timeline_mismatch", f"frame {frame_idx} provenance is invalid")
        source_boxes = np.asarray([row["bbox_xyxy_source"] for row in observations], dtype=np.float32).reshape(-1, 4)
        detector_sides = np.asarray([row["side_index"] for row in observations], dtype=np.float32)
        rgb_path = resolve_manifest_rgb(run_root, repo_root, str(frame_row["rgb"]))
        frame_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FeishuRayAdapterError("raw_frame_decode_failed", f"could not decode frame {rgb_path}")
        source_height, source_width = frame_bgr.shape[:2]
        if expected_hw != (source_height, source_width):
            raise FeishuRayAdapterError(
                "wilor_frame_shape_changed",
                f"frame {frame_idx} shape {(source_height, source_width)} != {expected_hw}",
            )
        hands: list[dict[str, Any]] = []
        if observations:
            dataset = dataset_class(
                config,
                frame_bgr,
                source_boxes,
                detector_sides,
                rescale_factor=float(args.rescale_factor),
                fp16=False,
            )
            if len(dataset) != len(observations):
                raise FeishuRayAdapterError(
                    "wilor_crop_count_mismatch",
                    f"preprocessor produced {len(dataset)} crops for {len(observations)} detections",
                )
            for det_index in range(len(dataset)):
                observation = observations[det_index]
                if not isinstance(observation, dict) or int(observation.get("detection_index", -1)) != det_index:
                    raise FeishuRayAdapterError(
                        "hands_detector_timeline_mismatch",
                        f"frame {frame_idx} detection row {det_index} is invalid",
                    )
                crop_row = dataset[det_index]
                crop = np.ascontiguousarray(as_numpy(crop_row["img"]), dtype=np.float32)
                if crop.shape != (3, 256, 256) or not np.isfinite(crop).all():
                    raise FeishuRayAdapterError("wilor_crop_invalid", f"crop {det_index} has shape {crop.shape}")
                box_center = as_numpy(crop_row["box_center"]).astype(np.float32).reshape(2)
                box_size = float(as_numpy(crop_row["box_size"]).reshape(-1)[0])
                img_size = as_numpy(crop_row["img_size"]).astype(np.float32).reshape(2)
                handedness = int(round(float(as_numpy(crop_row["right"]).reshape(-1)[0])))
                expected_handedness = int(observation["side_index"])
                if (
                    not np.isfinite(box_center).all()
                    or not math.isfinite(box_size)
                    or box_size <= 0.0
                    or not np.isfinite(img_size).all()
                    or not np.allclose(img_size, [source_width, source_height], atol=1.0e-4)
                    or handedness != expected_handedness
                ):
                    raise FeishuRayAdapterError(
                        "wilor_crop_metadata_invalid",
                        f"crop {det_index} has center={box_center.tolist()} size={box_size} image={img_size.tolist()} side={handedness}/{expected_handedness}",
                    )
                reconstruct_ownership = make_ownership(
                    job_id=job_id,
                    item_id=f"frame-{frame_idx:06d}-hand-{det_index}",
                    stage_id="wilor.reconstruct",
                    source_id=f"frame-{frame_idx:06d}-crop-{det_index}",
                    source_timestamp_s=timestamp,
                )
                reconstruct_metadata = {
                    "ownership": reconstruct_ownership,
                    "handedness": handedness,
                    "box_center": box_center.astype(float).tolist(),
                    "box_size": box_size,
                    "img_size": img_size.astype(float).tolist(),
                    "source_K_px": None,
                    "model_revision": "wilor-final-v1",
                    "options": {},
                }
                reconstruct_report = call_typed(
                    caller,
                    base_url=wilor_base_url,
                    route="/wilor.reconstruct",
                    metadata=reconstruct_metadata,
                    arrays={"crop": (crop.tobytes(), tuple(crop.shape), "float32")},
                    timeout_s=float(args.timeout_s),
                    retry_events=retry_events,
                    retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)),
                    retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)),
                )
                service_calls["wilor.reconstruct"] += 1
                candidate = build_wilor_candidate(
                    reconstruct_report,
                    ownership=reconstruct_ownership,
                    detector_score=float(observation["score"]),
                    bbox_xyxy=source_boxes[det_index],
                    detector_visibility=float(observation["visibility"]),
                    detector_uncertainty=float(observation["uncertainty"]),
                    expected_side=handedness,
                    img_size=img_size,
                )
                candidate["crop_metadata"] = {
                    "box_center": box_center.astype(float).tolist(),
                    "box_size": box_size,
                    "img_size": img_size.astype(float).tolist(),
                    "coordinate_space": "source_pixels",
                    "source_to_model": spatial["pixel_transform"]["source_to_model"],
                    "model_to_source": spatial["pixel_transform"]["model_to_source"],
                    "detector_timeline": str(detector_path),
                    "detector_observation_index": det_index,
                }
                hands.append(candidate)
        if hands:
            detected_frames += 1
            detected_hands += len(hands)
        raw_frames.append(
            {
                "frame_idx": frame_idx,
                "time_s": timestamp,
                "raw_hands": hands,
                "hands_detection": {
                    "ownership": detect_ownership,
                    "model_revision": detector_frame.get("model_revision"),
                    "trace": detector_frame.get("trace"),
                    "n_hands": len(observations),
                    "spatial": spatial,
                    "boxes_coordinate_space": "source_pixels",
                    "masks_coordinate_space": "source_grid",
                    "detector_timeline": str(detector_path),
                },
            }
        )

    output_dir = run_root / "measurements" / "hand_candidates" / "wilor_v21"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "wilor_raw_hands.json"
    raw_payload = {
        "schema": "v21_wilor_hand_candidates.v0",
        "backend": "WiLoR/Feishu-Ray",
        "wilor_root": str(wilor_root),
        "wilor_config": str(wilor_config),
        "service_profile": profile.get("profile"),
        "service_base_url": wilor_base_url,
        "frame_count": len(raw_frames),
        "hands_detector_timeline": str(detector_path),
        "hands_detector_masks": str(detector_masks_path),
        "coordinate_space": "source_pixels",
        "frames": raw_frames,
        "retry_events": retry_events,
    }
    write_json(raw_path, raw_payload)
    diagnosis = diagnose_hand_candidates(raw_frames, depth_archive_for_diagnosis(run_root), manifest)
    diagnosis["backend"] = "WiLoR/Feishu-Ray"
    diagnosis["service_calls"] = service_calls
    diagnosis["candidate_state"] = "remote_model_raw_evidence_not_accepted_mano_state"
    write_json(output_dir / "wilor_v21_diagnosis.json", diagnosis)
    qc = {
        "status": "ok",
        "method": "feishu_ray_wilor_adapter",
        "run_root": str(run_root),
        "processed_frames": len(raw_frames),
        "frames_with_hands": detected_frames,
        "detection_rate": detected_frames / max(1, len(raw_frames)),
        "detected_hands": detected_hands,
        "mean_hands_per_frame": detected_hands / max(1, len(raw_frames)),
        "elapsed_s": float(time.time() - started),
        "compute_target": args.compute_target,
        "service_profile": profile.get("profile"),
        "service_base_url": wilor_base_url,
        "service_calls": service_calls,
        "retry_events": retry_events,
        "hands_detector_path": str(detector_path),
        "hands_detector_sha256": sha256_file(detector_path),
        "hands_detector_masks_path": str(detector_masks_path),
        "hands_detector_masks_sha256": sha256_file(detector_masks_path),
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "diagnosis_path": str(output_dir / "wilor_v21_diagnosis.json"),
    }
    write_json(output_dir / "wilor_qc.json", qc)
    return qc


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def droid_spatial(source_width: int, source_height: int) -> dict[str, Any]:
    if source_width <= 0 or source_height <= 0:
        raise FeishuRayAdapterError(
            "invalid_source_image_size",
            f"source image size must be positive, got {source_width}x{source_height}",
        )
    target_scale = math.sqrt(DROID_TARGET_AREA / float(source_width * source_height))
    provisional_width = int(source_width * target_scale)
    provisional_height = int(source_height * target_scale)
    model_width = provisional_width - provisional_width % 8
    model_height = provisional_height - provisional_height % 8
    if model_width <= 0 or model_height <= 0:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_grid_invalid",
            f"DROID target-area rule produced {model_width}x{model_height} from {source_width}x{source_height}",
        )
    scale_x = provisional_width / float(source_width)
    scale_y = provisional_height / float(source_height)
    inverse_x = source_width / float(provisional_width)
    inverse_y = source_height / float(provisional_height)
    return {
        "target_area": DROID_TARGET_AREA,
        "source_size": {"width": int(source_width), "height": int(source_height)},
        "provisional_target_size": {"width": provisional_width, "height": provisional_height},
        "model_size": {"width": model_width, "height": model_height},
        "color_space": "RGB",
        "pixel_transform": {
            "source_to_model": [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            "model_to_source": [[inverse_x, 0.0, 0.0], [0.0, inverse_y, 0.0], [0.0, 0.0, 1.0]],
            "resize_mode": "target_area_384x512_resize_then_bottom_right_crop_hw_to_multiple_of_8",
            "crop_xywh": [0.0, 0.0, float(model_width), float(model_height)],
            "pad_ltrb": None,
        },
        "dimension_rule": (
            "provisional=int(source*sqrt((384*512)/(source_h*source_w))); "
            "resize to provisional, then crop bottom/right remainder so model dimensions are multiples of 8"
        ),
    }


def resize_rgb_to_droid_model(rgb: np.ndarray, spatial: Mapping[str, Any]) -> np.ndarray:
    import cv2

    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise FeishuRayAdapterError(
            "invalid_source_rgb",
            f"source RGB must be uint8[H,W,3], got dtype={source.dtype} shape={source.shape}",
        )
    model_size = spatial.get("model_size")
    if not isinstance(model_size, Mapping):
        raise FeishuRayAdapterError("feishu_ray_droid_model_grid_invalid", "DROID spatial metadata lacks model_size")
    model_width = int(model_size["width"])
    model_height = int(model_size["height"])
    provisional_size = spatial.get("provisional_target_size")
    if not isinstance(provisional_size, Mapping):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_grid_invalid",
            "DROID spatial metadata lacks provisional_target_size",
        )
    provisional_width = int(provisional_size["width"])
    provisional_height = int(provisional_size["height"])
    resized = cv2.resize(source, (provisional_width, provisional_height), interpolation=cv2.INTER_AREA)
    model_rgb = np.ascontiguousarray(resized[:model_height, :model_width], dtype=np.uint8)
    if model_rgb.shape != (model_height, model_width, 3):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_grid_invalid",
            f"DROID RGB resize returned {model_rgb.shape}, expected {(model_height, model_width, 3)}",
        )
    return model_rgb


def grayscale_and_zero_droid_rgb(model_rgb: np.ndarray, dynamic_ignore_mask: np.ndarray) -> np.ndarray:
    import cv2

    rgb = np.asarray(model_rgb)
    mask = np.asarray(dynamic_ignore_mask, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise FeishuRayAdapterError("feishu_ray_droid_rgb_invalid", f"invalid model RGB dtype/shape {rgb.dtype}/{rgb.shape}")
    if mask.shape != rgb.shape[:2] or not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_values_invalid",
            f"model-grid dynamic-ignore mask is invalid: shape={mask.shape} rgb={rgb.shape}",
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    submitted_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    submitted_rgb[mask > 0.0] = 0
    submitted_rgb = np.ascontiguousarray(submitted_rgb, dtype=np.uint8)
    if not (
        np.array_equal(submitted_rgb[:, :, 0], submitted_rgb[:, :, 1])
        and np.array_equal(submitted_rgb[:, :, 1], submitted_rgb[:, :, 2])
    ):
        raise AssertionError("DROID grayscale compatibility transform is not channel-symmetric")
    return submitted_rgb


def source_and_model_intrinsics(
    calibration_path: Path,
    *,
    source_width: int,
    source_height: int,
    spatial: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    calibration = load_json_object(calibration_path)
    values = calibration.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_calibration_invalid",
            f"calibration lacks intrinsics_fx_fy_cx_cy: {calibration_path}",
        )
    try:
        fx, fy, cx, cy = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_calibration_invalid",
            f"calibration intrinsics are not numeric: {values!r}",
        ) from exc
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0.0 or fy <= 0.0:
        raise FeishuRayAdapterError("feishu_ray_droid_calibration_invalid", f"invalid source intrinsics {values!r}")
    declared_source = calibration.get("source_size")
    if isinstance(declared_source, Mapping):
        declared_width = int(declared_source.get("width") or 0)
        declared_height = int(declared_source.get("height") or 0)
        if (declared_width, declared_height) != (source_width, source_height):
            raise FeishuRayAdapterError(
                "feishu_ray_droid_calibration_grid_mismatch",
                f"calibration source size {declared_width}x{declared_height} != frame grid {source_width}x{source_height}",
            )
    K_source = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    source_to_model = np.asarray(spatial["pixel_transform"]["source_to_model"], dtype=np.float64)
    K_model = source_to_model @ K_source
    if (
        not np.isfinite(K_model).all()
        or K_model[0, 0] <= 0.0
        or K_model[1, 1] <= 0.0
        or not np.allclose(K_model[2], [0.0, 0.0, 1.0], atol=1.0e-9)
    ):
        raise FeishuRayAdapterError("feishu_ray_droid_calibration_invalid", "model-grid intrinsics are invalid")
    provenance = {
        "calibration_contract": str(calibration_path),
        "calibration_contract_sha256": sha256_file(calibration_path),
        "calibration_source": calibration.get("intrinsics_source") or calibration.get("method"),
        "K_source_px": K_source.astype(float).tolist(),
        "K_model_px": K_model.astype(float).tolist(),
        "source_to_model": source_to_model.astype(float).tolist(),
        "model_to_source": spatial["pixel_transform"]["model_to_source"],
        "pixel_transform": dict(spatial["pixel_transform"]),
        "target_area": int(spatial["target_area"]),
        "provisional_target_size": dict(spatial["provisional_target_size"]),
        "dimension_rule": spatial["dimension_rule"],
        "source_size": {"width": source_width, "height": source_height},
        "model_size": dict(spatial["model_size"]),
        "intrinsics_request_semantics": "camera.intrinsics is model-grid [fx,fy,cx,cy]; camera.K_px retains source-grid K",
    }
    return K_source, K_model, provenance


def resolve_local_artifact(path_value: Any, *, owner_path: Path, run_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise FeishuRayAdapterError("hands_detector_mask_artifact_missing", f"invalid artifact path in {owner_path}")
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [owner_path.parent / path, run_root / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FeishuRayAdapterError("hands_detector_mask_artifact_missing", f"artifact is missing: {path_value}")


def prepare_droid_dynamic_ignore_masks(
    *,
    run_root: Path,
    frames: list[dict[str, Any]],
    fps: float,
    source_width: int,
    source_height: int,
    spatial: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cv2

    detector_path = run_root / "measurements" / "hand_detections" / "feishu_ray_hands" / "hands_detector_timeline.json"
    detector = load_json_object(detector_path)
    if detector.get("schema") != "ego.annotation.hands_detector_timeline.v1":
        raise FeishuRayAdapterError(
            "hands_detector_timeline_mismatch",
            f"unsupported detector timeline schema {detector.get('schema')!r}",
        )
    detector_frames = detector.get("frames")
    if not isinstance(detector_frames, list) or len(detector_frames) != len(frames):
        raise FeishuRayAdapterError(
            "hands_detector_timeline_mismatch",
            f"detector timeline has {len(detector_frames) if isinstance(detector_frames, list) else None} rows, expected {len(frames)}",
        )
    mask_contract = detector.get("mask_archive")
    if not isinstance(mask_contract, Mapping):
        raise FeishuRayAdapterError("hands_detector_mask_artifact_missing", "detector timeline lacks mask_archive")
    mask_path = resolve_local_artifact(mask_contract.get("path"), owner_path=detector_path, run_root=run_root)
    expected_archive_hash = mask_contract.get("sha256")
    archive_hash = sha256_file(mask_path)
    if not isinstance(expected_archive_hash, str) or archive_hash != expected_archive_hash:
        raise FeishuRayAdapterError(
            "hands_detector_mask_hash_mismatch",
            f"detector mask archive hash mismatch: expected={expected_archive_hash} actual={archive_hash}",
        )
    try:
        with np.load(mask_path, allow_pickle=False) as archive:
            required = {
                "masks_packbits",
                "frame_idx",
                "detection_idx",
                "side",
                "source_size",
                "mask_bit_count",
                "packbits_bitorder",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise FeishuRayAdapterError(
                    "hands_detector_mask_archive_shape_mismatch",
                    f"detector mask archive is missing keys {missing}",
                )
            packed_masks = np.asarray(archive["masks_packbits"], dtype=np.uint8)
            mask_frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
            mask_detection_idx = np.asarray(archive["detection_idx"], dtype=np.int64)
            mask_side = np.asarray(archive["side"], dtype=np.int64)
            archive_source_size = np.asarray(archive["source_size"], dtype=np.int64).reshape(-1)
            mask_bit_count = int(np.asarray(archive["mask_bit_count"]).reshape(-1)[0])
            bitorder = str(np.asarray(archive["packbits_bitorder"]).reshape(-1)[0])
    except FeishuRayAdapterError:
        raise
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise FeishuRayAdapterError(
            "hands_detector_mask_archive_shape_mismatch",
            f"could not decode detector mask archive: {exc}",
        ) from exc
    packed_width = (source_width * source_height + 7) // 8
    record_count = packed_masks.shape[0] if packed_masks.ndim == 2 else -1
    if (
        packed_masks.shape != (record_count, packed_width)
        or mask_frame_idx.shape != (record_count,)
        or mask_detection_idx.shape != (record_count,)
        or mask_side.shape != (record_count,)
        or archive_source_size.tolist() != [source_width, source_height]
        or mask_bit_count != source_width * source_height
        or bitorder != "little"
    ):
        raise FeishuRayAdapterError(
            "hands_detector_mask_archive_shape_mismatch",
            f"detector mask archive does not match source grid {source_width}x{source_height}",
        )
    model_width = int(spatial["model_size"]["width"])
    model_height = int(spatial["model_size"]["height"])
    input_dir = run_root / "measurements" / "camera_trajectory" / "droid_service_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    submitted_path = input_dir / "droid_submitted_dynamic_ignore_masks.npy"
    submitted_store = np.lib.format.open_memmap(
        submitted_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frames), model_height, model_width),
    )
    seen_archive_indices: set[int] = set()
    frame_provenance: list[dict[str, Any]] = []
    for position, (source_row, detector_row) in enumerate(zip(frames, detector_frames)):
        frame_idx = int(source_row["frame_idx"])
        timestamp = float(source_row.get("time_s", frame_idx / fps))
        if not isinstance(detector_row, Mapping) or int(detector_row.get("frame_idx", -1)) != frame_idx:
            raise FeishuRayAdapterError(
                "hands_detector_timeline_mismatch",
                f"detector row {position} does not match source frame {frame_idx}",
            )
        try:
            detector_timestamp = float(detector_row.get("time_s"))
        except (TypeError, ValueError) as exc:
            raise FeishuRayAdapterError(
                "hands_detector_timeline_mismatch",
                f"detector frame {frame_idx} has invalid timestamp",
            ) from exc
        if not math.isclose(detector_timestamp, timestamp, rel_tol=0.0, abs_tol=1.0e-7):
            raise FeishuRayAdapterError(
                "hands_detector_timeline_mismatch",
                f"detector timestamp {detector_timestamp} != source timestamp {timestamp} at frame {frame_idx}",
            )
        observations = detector_row.get("observations")
        if not isinstance(observations, list):
            raise FeishuRayAdapterError(
                "hands_detector_timeline_mismatch",
                f"detector frame {frame_idx} observations are invalid",
            )
        source_union = np.zeros((source_height, source_width), dtype=np.uint8)
        archive_indices: list[int] = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise FeishuRayAdapterError("hands_detector_timeline_mismatch", f"invalid observation at frame {frame_idx}")
            archive_index = int(observation.get("mask_archive_index", -1))
            detection_index = int(observation.get("detection_index", -1))
            side_index = int(observation.get("side_index", -1))
            if archive_index < 0 or archive_index >= record_count or archive_index in seen_archive_indices:
                raise FeishuRayAdapterError(
                    "hands_detector_timeline_mismatch",
                    f"invalid or duplicate mask archive index {archive_index} at frame {frame_idx}",
                )
            if (
                int(mask_frame_idx[archive_index]) != frame_idx
                or int(mask_detection_idx[archive_index]) != detection_index
                or int(mask_side[archive_index]) != side_index
            ):
                raise FeishuRayAdapterError(
                    "hands_detector_timeline_mismatch",
                    f"mask archive row {archive_index} ownership does not match frame {frame_idx} detection {detection_index}",
                )
            decoded = np.unpackbits(
                packed_masks[archive_index],
                bitorder="little",
                count=source_width * source_height,
            ).reshape(source_height, source_width)
            expected_count = observation.get("mask_source_pixel_count")
            if expected_count is not None and int(expected_count) != int(np.count_nonzero(decoded)):
                raise FeishuRayAdapterError(
                    "hands_detector_timeline_mismatch",
                    f"mask pixel count mismatch for archive row {archive_index}",
                )
            source_union |= decoded.astype(np.uint8, copy=False)
            seen_archive_indices.add(archive_index)
            archive_indices.append(archive_index)
        provisional_width = int(spatial["provisional_target_size"]["width"])
        provisional_height = int(spatial["provisional_target_size"]["height"])
        provisional_mask = cv2.resize(
            source_union.astype(np.float32),
            (provisional_width, provisional_height),
            interpolation=cv2.INTER_AREA,
        )
        submitted_mask = np.ascontiguousarray(
            np.clip(provisional_mask[:model_height, :model_width], 0.0, 1.0),
            dtype=np.float32,
        )
        submitted_store[position] = submitted_mask
        frame_provenance.append(
            {
                "frame_idx": frame_idx,
                "source_timestamp_s": timestamp,
                "detector_mask_archive_indices": archive_indices,
                "source_union_sha256": sha256_array(source_union),
                "source_union_dynamic_pixels": int(np.count_nonzero(source_union)),
                "source_union_dynamic_fraction": float(np.mean(source_union)),
                "submitted_model_mask_sha256": sha256_array(submitted_mask),
                "submitted_model_mask_positive_pixels": int(np.count_nonzero(submitted_mask > 0.0)),
                "submitted_model_mask_ignore_pixels": int(np.count_nonzero(submitted_mask > 0.0)),
                "submitted_model_mask_mean": float(np.mean(submitted_mask)),
            }
        )
    if seen_archive_indices != set(range(record_count)):
        missing_indices = sorted(set(range(record_count)).difference(seen_archive_indices))
        raise FeishuRayAdapterError(
            "hands_detector_timeline_mismatch",
            f"detector timeline did not reference mask archive rows {missing_indices[:20]}",
        )
    submitted_store.flush()
    del submitted_store
    submitted_hash = sha256_file(submitted_path)
    provenance_path = input_dir / "droid_submitted_dynamic_ignore_masks.json"
    provenance = {
        "schema": "ego.annotation.feishu_ray_droid_inputs.v1",
        "status": "prepared",
        "source_detector_timeline": {
            "path": str(detector_path),
            "sha256": sha256_file(detector_path),
            "schema": detector.get("schema"),
        },
        "source_detector_mask_archive": {
            "path": str(mask_path),
            "sha256": archive_hash,
            "encoding": "numpy.packbits_axis_flat_bitorder_little",
            "source_value_semantics": "1=dynamic_ignore,0=static_keep",
        },
        "union_policy": "bitwise OR of every persisted D6a detection mask owned by the source frame",
        "source_size": {"width": source_width, "height": source_height},
        "model_size": {"width": model_width, "height": model_height},
        "provisional_target_size": dict(spatial["provisional_target_size"]),
        "pixel_transform": dict(spatial["pixel_transform"]),
        "mask_resize": "OpenCV INTER_AREA to target-area provisional grid, then exact bottom/right crop to DROID model grid",
        "submitted_timeline": {
            "path": str(submitted_path),
            "sha256": submitted_hash,
            "shape": [len(frames), model_height, model_width],
            "dtype": "float32",
            "value_semantics": "positive=ignore,0=retain",
            "array_is_exact_submitted_tensor_timeline": True,
        },
        "dynamic_pixel_zeroing_rule": "submitted_mask > 0.0",
        "frames": frame_provenance,
    }
    write_json(provenance_path, provenance)
    dynamic_mask = {
        "status": "applied",
        "path": str(submitted_path),
        "sha256": submitted_hash,
        "shape": [len(frames), model_height, model_width],
        "dtype": "float32",
        "source": "persisted_d6a_hands_detector_union",
        "source_value_semantics": "1=dynamic_ignore,0=static_keep",
        "submitted_value_semantics": "positive=ignore,0=retain",
        "service_consumption_semantics": "positive=ignore,0=retain",
        "source_to_submitted_conversion": "area_resample_positive_preserved",
        "source_detector_timeline": str(detector_path),
        "source_detector_timeline_sha256": sha256_file(detector_path),
        "source_detector_mask_archive": str(mask_path),
        "source_detector_mask_archive_sha256": archive_hash,
        "submitted_timeline_provenance": str(provenance_path),
        "submitted_timeline_provenance_sha256": sha256_file(provenance_path),
        "dynamic_pixel_zeroing_rule": "submitted_mask > 0.0",
    }
    return dynamic_mask, provenance


def require_droid_json_envelope(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    route: str,
) -> dict[str, Any]:
    raw_status = report.get("http_status")
    status = adapter_int(
        0 if raw_status is None else raw_status,
        code="feishu_ray_response_envelope_invalid",
        message=f"{route}: response HTTP status is invalid",
    )
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise FeishuRayAdapterError("feishu_ray_response_metadata_missing", f"{route}: JSON response metadata is missing")
    if report.get("status") != "ok" or status < 200 or status >= 300:
        raise FeishuRayAdapterError("feishu_ray_http_error", f"{route}: HTTP {status}; error={metadata.get('error')!r}")
    error = metadata.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "service_error")
        raise FeishuRayAdapterError(f"feishu_ray_{code}", f"{route}: {error.get('message') or error}")
    if error is not None:
        raise FeishuRayAdapterError("feishu_ray_response_envelope_invalid", f"{route}: invalid error field {error!r}")
    if not ownership_matches(ownership, metadata.get("ownership")):
        raise FeishuRayAdapterError("feishu_ray_ownership_mismatch", f"{route}: response ownership does not match request")
    arrays = report.get("arrays")
    if not isinstance(arrays, list) or arrays:
        raise FeishuRayAdapterError("feishu_ray_response_envelope_invalid", f"{route}: expected a JSON-only response")
    return metadata


def validate_droid_create_response(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
) -> str:
    metadata = require_droid_json_envelope(
        report,
        ownership=ownership,
        route="/droid.create_session",
    )
    session_id = metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise FeishuRayAdapterError("feishu_ray_droid_session_missing", "droid.create_session returned no session_id")
    if metadata.get("status") is not None or metadata.get("camera_state") is not None:
        raise FeishuRayAdapterError("feishu_ray_response_envelope_invalid", "droid.create_session returned a non-create envelope")
    return session_id


def validate_droid_push_response(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    session_id: str,
    frame_id: str,
    source_timestamp_s: float,
    expected_keyframe_count: int,
) -> dict[str, Any]:
    metadata = require_droid_json_envelope(report, ownership=ownership, route="/droid.push_frame")
    status = metadata.get("status")
    if not isinstance(status, Mapping):
        raise FeishuRayAdapterError("feishu_ray_droid_push_status_missing", "droid.push_frame returned no status")
    if not ownership_matches(ownership, status.get("ownership")):
        raise FeishuRayAdapterError("feishu_ray_ownership_mismatch", "droid.push_frame nested status ownership mismatch")
    response_session_id = adapter_string(
        status.get("session_id"),
        code="feishu_ray_droid_push_identity_mismatch",
        message="droid.push_frame returned a non-string session_id",
    )
    response_frame_id = adapter_string(
        status.get("frame_id"),
        code="feishu_ray_droid_push_identity_mismatch",
        message="droid.push_frame returned a non-string frame_id",
    )
    if response_session_id != session_id or response_frame_id != frame_id:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_push_identity_mismatch",
            f"droid.push_frame returned session/frame {response_session_id}/{response_frame_id}",
        )
    response_timestamp = adapter_float(
        status.get("source_timestamp_s"),
        code="feishu_ray_droid_push_identity_mismatch",
        message="push status timestamp is invalid",
    )
    if not math.isclose(response_timestamp, source_timestamp_s, rel_tol=0.0, abs_tol=1.0e-9):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_push_identity_mismatch",
            f"push status timestamp {response_timestamp} != {source_timestamp_s}",
        )
    validity = status.get("validity")
    if not isinstance(validity, Mapping):
        raise FeishuRayAdapterError("feishu_ray_droid_push_status_missing", "push status lacks validity")
    validity_frame_id = adapter_string(
        validity.get("frame_id"),
        code="feishu_ray_droid_push_identity_mismatch",
        message="push validity returned a non-string frame_id",
    )
    validity_timestamp = adapter_float(
        validity.get("source_timestamp_s"),
        code="feishu_ray_droid_push_identity_mismatch",
        message="push validity timestamp is invalid",
    )
    if (
        validity_frame_id != frame_id
        or not math.isclose(validity_timestamp, source_timestamp_s, rel_tol=0.0, abs_tol=1.0e-9)
        or validity.get("admitted") is not True
        or not isinstance(validity.get("keyframe_added"), bool)
        or (
            validity.get("keyframe_added") is True
            and validity.get("skip_reason") is not None
        )
        or (
            validity.get("keyframe_added") is False
            and not isinstance(validity.get("skip_reason"), str)
        )
    ):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_push_validity_invalid",
            f"frame {frame_id} returned invalid admitted/keyframe validity: {dict(validity)}",
        )
    keyframe_count = adapter_int(
        status.get("keyframe_count", -1),
        code="feishu_ray_droid_all_keyframe_compensation_failed",
        message=f"frame {frame_id} returned an invalid keyframe_count",
    )
    if keyframe_count < expected_keyframe_count:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_keyframe_count_regressed",
            f"frame {frame_id} returned keyframe_count={keyframe_count}, below previous {expected_keyframe_count}",
        )
    return dict(status)


def validate_droid_finalize_compatibility(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    session_id: str,
    expected_timeline: list[dict[str, Any]],
    expected_model_K: np.ndarray,
    expected_model_size: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_droid_finalize(report, ownership=ownership, expected_timeline=expected_timeline)
    state = validated["state"]
    response_session_id = adapter_string(
        state.get("session_id"),
        code="feishu_ray_droid_session_mismatch",
        message="droid.finalize returned a non-string session_id",
    )
    if response_session_id != session_id:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_session_mismatch",
            f"droid.finalize session {response_session_id!r} != {session_id!r}",
        )
    if state.get("model_revision") != DROID_MODEL_REVISION:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_revision_mismatch",
            f"droid.finalize model revision {state.get('model_revision')!r} != {DROID_MODEL_REVISION!r}",
        )
    keyframes = validated["keyframe_mapping"]
    expected_by_source = {row["source_frame_id"]: row for row in expected_timeline}
    for index, mapping in enumerate(keyframes):
        keyframe_index = adapter_int(
            mapping.get("keyframe_index", -1),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"finalize keyframe index is invalid at row {index}",
        )
        mapping_timestamp = adapter_float(
            mapping.get("source_timestamp_s"),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"finalize keyframe timestamp is invalid at row {index}",
        )
        mapping_source_id = adapter_string(
            mapping.get("source_frame_id"),
            code="feishu_ray_droid_keyframe_mapping_mismatch",
            message=f"finalize keyframe source_frame_id is invalid at row {index}",
        )
        expected = expected_by_source.get(mapping_source_id)
        if expected is None:
            raise FeishuRayAdapterError(
                "feishu_ray_droid_keyframe_mapping_mismatch",
                f"finalize keyframe source {mapping_source_id!r} is outside the submitted timeline",
            )
        expected_timestamp = adapter_float(
            expected.get("source_timestamp_s"),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected keyframe timestamp is invalid for {mapping_source_id}",
        )
        if keyframe_index != index or not math.isclose(
            mapping_timestamp,
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise FeishuRayAdapterError(
                "feishu_ray_droid_keyframe_mapping_mismatch",
                f"finalize keyframe mapping diverged at row {index}",
            )
    try:
        expected_K = np.asarray(expected_model_K, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_grid_invalid",
            "expected DROID model-grid intrinsics are invalid",
        ) from exc
    intrinsics = np.asarray(validated["intrinsics_px"], dtype=np.float64)
    if expected_K.shape != (3, 3) or not np.allclose(
        intrinsics,
        expected_K[None],
        atol=1.0e-5,
        rtol=1.0e-6,
    ):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_intrinsics_mismatch",
            "finalize intrinsics do not equal the model-grid intrinsics submitted at create_session",
        )
    if not isinstance(expected_model_size, Mapping):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_grid_invalid",
            "expected DROID model-grid size is invalid",
        )
    expected_disparity_shape = (
        adapter_int(
            expected_model_size.get("height"),
            code="feishu_ray_droid_model_grid_invalid",
            message="expected DROID model-grid height is invalid",
        )
        // 8,
        adapter_int(
            expected_model_size.get("width"),
            code="feishu_ray_droid_model_grid_invalid",
            message="expected DROID model-grid width is invalid",
        )
        // 8,
    )
    if tuple(validated["disparities"].shape[1:]) != expected_disparity_shape:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_disparity_shape",
            f"finalize disparity grid {validated['disparities'].shape[1:]} != expected {expected_disparity_shape}",
        )
    return validated


def preserve_failed_droid_finalize(
    report: Mapping[str, Any],
    *,
    run_root: Path,
    ownership: Mapping[str, Any],
    session_id: str | None,
    error: Exception,
    service_provenance: Mapping[str, Any],
    route: str = "/droid.finalize",
    failure_family: str = "feishu_ray_droid_finalize",
    failure_status: str = "failed_finalize_validation",
    manifest_name: str = "failed_finalize.json",
    extra_fields: Mapping[str, Any] | None = None,
) -> Path:
    request_id = str(ownership.get("request_id") or uuid4().hex)
    failure_dir = run_root / "failures" / failure_family / request_id
    failure_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = failure_dir / "response_metadata.json"
    try:
        metadata_text = json.dumps(report.get("metadata"), indent=2, ensure_ascii=False, allow_nan=True)
    except (TypeError, ValueError) as exc:
        metadata_text = json.dumps(
            {"serialization_error": str(exc), "repr": repr(report.get("metadata"))},
            indent=2,
            ensure_ascii=False,
        )
    metadata_path.write_text(metadata_text, encoding="utf-8")
    array_artifacts: list[dict[str, Any]] = []
    array_rows = report.get("arrays")
    if not isinstance(array_rows, list):
        array_rows = []
    for index, row in enumerate(array_rows):
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or f"array_{index}")
        safe_name = "".join(character if character.isalnum() or character in "_.-" else "_" for character in name)
        raw_shape = row.get("shape", ())
        shape_error: str | None = None
        try:
            shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            shape = ()
            shape_error = f"invalid declared shape: {exc}"
        dtype = str(row.get("dtype"))
        data = row.get("data")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            continue
        try:
            if shape_error is not None:
                raise ValueError(shape_error)
            array = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raw_path = failure_dir / f"{index:02d}_{safe_name}.bin"
            raw_path.write_bytes(bytes(data))
            array_artifacts.append(
                {
                    "name": name,
                    "path": str(raw_path),
                    "shape": list(shape),
                    "declared_shape_repr": repr(raw_shape),
                    "dtype": dtype,
                    "sha256": sha256_file(raw_path),
                    "typed_decode_error": str(exc),
                }
            )
            continue
        array_path = failure_dir / f"{index:02d}_{safe_name}.npy"
        np.save(array_path, array, allow_pickle=False)
        array_artifacts.append(
            {
                "name": name,
                "path": str(array_path),
                "shape": list(shape),
                "dtype": dtype,
                "sha256": sha256_file(array_path),
                "finite": bool(np.isfinite(array).all()) if array.dtype.kind in {"b", "i", "u", "f", "c"} else None,
            }
        )
    failure = {
        "schema": "ego.annotation.feishu_ray_failed_service_evidence.v1",
        "status": failure_status,
        "route": route,
        "validation_error": exception_evidence(error),
        "ownership": dict(ownership),
        "session_id": session_id,
        "http_status": report.get("http_status"),
        "content_type": report.get("content_type"),
        "response_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "typed_arrays": array_artifacts,
        "service_provenance": dict(service_provenance),
        "successful_d4_artifacts_published": False,
    }
    if extra_fields is not None:
        failure.update(dict(extra_fields))
    write_json(failure_dir / manifest_name, failure)
    return failure_dir


def preserve_failed_droid_finalize_call(
    *,
    run_root: Path,
    ownership: Mapping[str, Any],
    session_id: str | None,
    error: Exception,
    response_received: bool | None,
    response_status: int | None,
    response_headers: Mapping[str, str] | None,
    raw_response_bytes: bytes | None,
    service_provenance: Mapping[str, Any] | None,
    route: str = "/droid.finalize",
    failure_family: str = "feishu_ray_droid_finalize",
    failure_status: str = "failed_finalize_call",
    manifest_name: str = "failed_finalize.json",
    extra_fields: Mapping[str, Any] | None = None,
) -> Path:
    request_id = str(ownership.get("request_id") or uuid4().hex)
    failure_dir = run_root / "failures" / failure_family / request_id
    failure_dir.mkdir(parents=True, exist_ok=False)
    headers_artifact: dict[str, Any] | None = None
    if response_headers is not None:
        headers_path = failure_dir / "response_headers.json"
        write_json(headers_path, dict(response_headers))
        headers_artifact = {
            "path": str(headers_path),
            "sha256": sha256_file(headers_path),
            "size_bytes": headers_path.stat().st_size,
            "header_count": len(response_headers),
        }
    raw_response_artifact: dict[str, Any] | None = None
    if raw_response_bytes is not None:
        raw_response_path = failure_dir / "raw_response.bin"
        raw_response_path.write_bytes(bytes(raw_response_bytes))
        raw_response_artifact = {
            "path": str(raw_response_path),
            "sha256": sha256_file(raw_response_path),
            "size_bytes": raw_response_path.stat().st_size,
        }
    unavailable_response_evidence = ["decoded_response_metadata", "typed_response_arrays"]
    if response_status is None:
        unavailable_response_evidence.append("http_status")
    if response_headers is None:
        unavailable_response_evidence.append("response_headers")
    if raw_response_bytes is None:
        unavailable_response_evidence.append("raw_response_bytes")
    failure = {
        "schema": "ego.annotation.feishu_ray_failed_service_evidence.v1",
        "status": failure_status,
        "route": route,
        "validation_error": exception_evidence(error),
        "ownership": dict(ownership),
        "session_id": session_id,
        "response_received": response_received if type(response_received) is bool else None,
        "invalid_response_received_repr": (
            None if response_received is None or type(response_received) is bool else repr(response_received)
        ),
        "http_status": response_status,
        "response_headers": headers_artifact,
        "raw_response_bytes": raw_response_artifact,
        "response_metadata": None,
        "typed_arrays": [],
        "unavailable_response_evidence": unavailable_response_evidence,
        "caller_evidence_limit": (
            "The caller returned no decoded report. Available transport response evidence is preserved; "
            "unavailable response evidence is listed explicitly."
        ),
        "service_provenance": dict(service_provenance or {}),
        "successful_d4_artifacts_published": False,
    }
    if extra_fields is not None:
        failure.update(dict(extra_fields))
    write_json(failure_dir / manifest_name, failure)
    return failure_dir


def droid_create_failure_fields(
    *,
    report: Any,
    trusted_session_id: str | None,
    response_received: bool | None,
    cleanup_attempt_required: bool,
) -> dict[str, Any]:
    metadata = report.get("metadata") if isinstance(report, Mapping) else None
    observed_session_id = metadata.get("session_id") if isinstance(metadata, Mapping) else None
    has_trusted_handle = isinstance(trusted_session_id, str) and bool(trusted_session_id)
    return {
        "response_received": response_received if type(response_received) is bool else None,
        "observed_session_id_type": type(observed_session_id).__name__,
        "observed_session_id_repr": repr(observed_session_id),
        "trusted_session_id": trusted_session_id if has_trusted_handle else None,
        "session_handle_status": (
            "trusted_exact_string_matching_success_json_envelope"
            if has_trusted_handle
            else "unresolved_no_trusted_session_handle"
        ),
        "retirement_status": (
            "cleanup_required_for_trusted_session_handle"
            if has_trusted_handle and cleanup_attempt_required
            else "unresolved_no_trusted_session_handle"
        ),
        "cleanup_attempt_required": bool(has_trusted_handle and cleanup_attempt_required),
        "cleanup_attempted": False,
        "cleanup_lifecycle": None,
        "cleanup_not_attempted_reason": (
            None
            if has_trusted_handle and cleanup_attempt_required
            else (
                "No exact non-empty string session handle was established from a matching "
                "successful JSON-only create envelope."
            )
        ),
        "service_contract_limitation": (
            "The service exposes no abort route; without a trusted session handle, possible "
            "server-side retirement cannot be confirmed."
        ),
    }


def update_failed_droid_create_cleanup(
    failure_dir: Path,
    *,
    cleanup_path: Path | None,
) -> None:
    manifest_path = failure_dir / "failed_create.json"
    record = load_json_object(manifest_path)
    record["cleanup_attempted"] = True
    record["cleanup_not_attempted_reason"] = None
    if cleanup_path is None or not cleanup_path.is_file():
        record["retirement_status"] = "unresolved_cleanup_record_unavailable"
        record["cleanup_lifecycle"] = None
    else:
        cleanup = load_json_object(cleanup_path)
        record["retirement_status"] = cleanup.get("retirement_status", "unresolved")
        record["cleanup_lifecycle"] = {
            "path": str(cleanup_path),
            "sha256": sha256_file(cleanup_path),
        }
    write_json(manifest_path, record)


def validate_droid_finalize_service_result(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    session_id: str,
) -> Mapping[str, Any]:
    raw_status = report.get("http_status")
    status = adapter_int(
        0 if raw_status is None else raw_status,
        code="feishu_ray_response_envelope_invalid",
        message="/droid.finalize: response HTTP status is invalid",
    )
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise FeishuRayAdapterError(
            "feishu_ray_response_metadata_missing",
            "/droid.finalize: cleanup response metadata is missing",
        )
    error = metadata.get("error")
    if report.get("status") != "ok" or status < 200 or status >= 300:
        raise FeishuRayAdapterError(
            "feishu_ray_http_error",
            f"/droid.finalize: cleanup HTTP {status}; error={error!r}",
        )
    if isinstance(error, dict):
        code = str(error.get("code") or "service_error")
        raise FeishuRayAdapterError(f"feishu_ray_{code}", f"/droid.finalize: {error.get('message') or error}")
    if error is not None:
        raise FeishuRayAdapterError(
            "feishu_ray_response_envelope_invalid",
            f"/droid.finalize: invalid cleanup error field {error!r}",
        )
    if not ownership_matches(ownership, metadata.get("ownership")):
        raise FeishuRayAdapterError(
            "feishu_ray_ownership_mismatch",
            "/droid.finalize: cleanup response ownership does not match request",
        )
    result = metadata.get("camera_state")
    if result is None:
        result = metadata.get("result")
    if not isinstance(result, Mapping):
        raise FeishuRayAdapterError(
            "feishu_ray_result_missing",
            "/droid.finalize: cleanup response lacks a successful service result",
        )
    nested_ownership = result.get("ownership")
    if not ownership_matches(ownership, nested_ownership):
        raise FeishuRayAdapterError(
            "feishu_ray_ownership_mismatch",
            "/droid.finalize: cleanup nested result ownership is missing or does not match request",
        )
    cleanup_session_id = adapter_string(
        result.get("session_id"),
        code="feishu_ray_droid_session_mismatch",
        message="/droid.finalize: cleanup returned a non-string session_id",
    )
    if cleanup_session_id != session_id:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_session_mismatch",
            f"/droid.finalize: cleanup session {cleanup_session_id!r} != {session_id!r}",
        )
    if result.get("model_revision") != DROID_MODEL_REVISION:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_model_revision_mismatch",
            f"/droid.finalize: cleanup model revision {result.get('model_revision')!r} != {DROID_MODEL_REVISION!r}",
        )
    return result


def attempt_droid_session_cleanup(
    *,
    caller: ServiceCall,
    run_root: Path,
    base_url: str,
    timeout_s: float,
    job_id: str,
    session_id: str,
    trigger_route: str,
    trigger_error: Exception,
) -> Path:
    cleanup_ownership = make_ownership(
        job_id=job_id,
        item_id=f"{job_id}-cleanup",
        stage_id="droid.finalize",
        source_id=job_id,
        source_timestamp_s=None,
    )
    cleanup_error: dict[str, str] | None = None
    response_summary: dict[str, Any] | None = None
    retirement_confirmed = False
    try:
        cleanup_report = call_typed(
            caller,
            base_url=base_url,
            route="/droid.finalize",
            metadata={
                "ownership": cleanup_ownership,
                "session_id": session_id,
                "model_revision": DROID_MODEL_REVISION,
            },
            arrays={},
            timeout_s=timeout_s,
            allow_retryable=False,
        )
        cleanup_result = validate_droid_finalize_service_result(
            cleanup_report,
            ownership=cleanup_ownership,
            session_id=session_id,
        )
        retirement_confirmed = True
        response_summary = {
            "http_status": cleanup_report.get("http_status"),
            "content_type": cleanup_report.get("content_type"),
            "successful_service_result": True,
            "outer_ownership_matched": True,
            "nested_ownership_matched": True,
            "session_id": cleanup_result.get("session_id"),
            "model_revision": cleanup_result.get("model_revision"),
            "geometry_validated_or_published": False,
        }
    except Exception as exc:
        cleanup_error = {
            "code": exc.code if isinstance(exc, FeishuRayAdapterError) else "unexpected_cleanup_failure",
            "message": str(exc),
        }
    record = {
        "schema": "ego.annotation.feishu_ray_droid_cleanup_lifecycle.v1",
        "session_id": session_id,
        "attempted": True,
        "retirement_status": "confirmed_by_successful_finalize_result" if retirement_confirmed else "unresolved",
        "successful_service_result_confirmed_retirement": retirement_confirmed,
        "trigger": {
            "route": trigger_route,
            "error": exception_evidence(trigger_error),
        },
        "cleanup": {
            "route": "/droid.finalize",
            "ownership": cleanup_ownership,
            "response": response_summary,
            "error": cleanup_error,
        },
        "successful_d4_artifacts_published": False,
        "service_contract_limitation": (
            "The service exposes no abort route; if this cleanup finalize fails, the server session may remain resident."
        ),
    }
    cleanup_dir = (
        run_root
        / "failures"
        / "feishu_ray_droid_session_cleanup"
        / str(cleanup_ownership["request_id"])
    )
    cleanup_path = cleanup_dir / "cleanup_lifecycle.json"
    write_json(cleanup_path, record)
    return cleanup_path


def cleanup_droid_session_without_masking(
    *,
    original_error: Exception,
    caller: ServiceCall,
    run_root: Path,
    base_url: str,
    timeout_s: float,
    job_id: str,
    session_id: str,
    trigger_route: str,
) -> Path | None:
    try:
        cleanup_path = attempt_droid_session_cleanup(
            caller=caller,
            run_root=run_root,
            base_url=base_url,
            timeout_s=timeout_s,
            job_id=job_id,
            session_id=session_id,
            trigger_route=trigger_route,
            trigger_error=original_error,
        )
    except Exception as cleanup_record_error:
        add_exception_note(
            original_error,
            f"DROID cleanup lifecycle record could not be persisted: {cleanup_record_error}",
        )
        return None
    add_exception_note(original_error, f"DROID cleanup lifecycle: {cleanup_path}")
    return cleanup_path


def resolve_droid_clip(run_root: Path, manifest: Mapping[str, Any]) -> Path:
    candidates: list[Any] = []
    input_manifest_path = run_root / "input" / "input_manifest.json"
    if input_manifest_path.is_file():
        input_manifest = load_json_object(input_manifest_path)
        candidates.extend([input_manifest.get("primary_video"), input_manifest.get("clip_video")])
    candidates.append(manifest.get("clip"))
    frames = manifest.get("frames")
    if isinstance(frames, list) and frames and isinstance(frames[0], Mapping):
        candidates.append(frames[0].get("source_video"))
    for raw in candidates:
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw).expanduser()
        for candidate in ([path] if path.is_absolute() else [run_root / path, REPO_ROOT / path]):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError("prepared DROID clip is missing from input/raw manifests")


def run_droid(args: argparse.Namespace, *, caller: ServiceCall = call_service_arrays) -> dict[str, Any]:
    import fcntl

    lock_path = Path(
        os.environ.get(
            "EGO_DROID_MAINTENANCE_LOCK",
            "/home/zjh/ego-service-redeploy/droid_maintenance.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        return _run_droid_unlocked(args, caller=caller)
    finally:
        os.close(lock_fd)


def _run_droid_unlocked(args: argparse.Namespace, *, caller: ServiceCall = call_service_arrays) -> dict[str, Any]:
    from PIL import Image

    started = time.time()
    run_root = args.run_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    manifest = load_json_object(manifest_path)
    frames = validated_manifest_frames(manifest)
    fps = float(manifest.get("fps") or (manifest.get("video") or {}).get("fps") or 30.0)
    profile = load_profile(args.profile.expanduser().resolve())
    base_url = profile_base_url(profile, "droid", args.base_url)
    services = profile.get("services")
    profile_row = services.get("droid") if isinstance(services, Mapping) else None
    expected_routes = ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
    if (
        not isinstance(profile_row, Mapping)
        or profile_row.get("routes") != expected_routes
        or profile_row.get("model_revision") != DROID_MODEL_REVISION
        or profile_row.get("pinned_service_release") != DROID_PINNED_RELEASE
        or not isinstance(profile_row.get("canonical_semantics"), Mapping)
        or not isinstance(profile_row.get("deployed_compatibility_semantics"), Mapping)
    ):
        raise FeishuRayAdapterError(
            "invalid_service_profile",
            "DROID profile must pin the deployed DROID release, model droid-v1, canonical routes, and explicit compatibility semantics",
        )
    job_id = str(args.job_id or manifest.get("case_id") or run_root.name)
    output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    if output_dir.exists():
        raise FeishuRayAdapterError(
            "feishu_ray_droid_output_not_fresh",
            f"refusing to overwrite an existing D4 output directory: {output_dir}",
        )
    first_rgb_path = resolve_manifest_rgb(run_root, repo_root, str(frames[0]["rgb"]))
    with Image.open(first_rgb_path) as image:
        first_rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    source_height, source_width = first_rgb.shape[:2]
    spatial = droid_spatial(source_width, source_height)
    calibration_path = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    K_source, K_model, camera_provenance = source_and_model_intrinsics(
        calibration_path,
        source_width=source_width,
        source_height=source_height,
        spatial=spatial,
    )
    dynamic_mask, mask_provenance = prepare_droid_dynamic_ignore_masks(
        run_root=run_root,
        frames=frames,
        fps=fps,
        source_width=source_width,
        source_height=source_height,
        spatial=spatial,
    )
    submitted_masks = np.load(Path(dynamic_mask["path"]), mmap_mode="r", allow_pickle=False)
    expected_mask_shape = (
        len(frames),
        int(spatial["model_size"]["height"]),
        int(spatial["model_size"]["width"]),
    )
    if not isinstance(submitted_masks, np.ndarray) or submitted_masks.shape != expected_mask_shape or submitted_masks.dtype != np.float32:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_timeline_mismatch",
            f"persisted submitted mask timeline has dtype/shape {getattr(submitted_masks, 'dtype', None)}/{getattr(submitted_masks, 'shape', None)}",
        )
    expected_timeline = [
        {
            "frame_idx": int(row["frame_idx"]),
            "source_frame_id": f"frame-{int(row['frame_idx'])}",
            "source_timestamp_s": float(row.get("time_s", int(row["frame_idx"]) / fps)),
        }
        for row in frames
    ]
    session_options = dict(DROID_SESSION_OPTIONS)
    session_options["buffer"] = max(len(frames) + 1, int(session_options["warmup"]) + 1)
    create_ownership = make_ownership(
        job_id=job_id,
        item_id=job_id,
        stage_id="droid.create_session",
        source_id=job_id,
        source_timestamp_s=None,
    )
    create_metadata = {
        "ownership": create_ownership,
        "camera": {
            "intrinsics": [
                float(K_model[0, 0]),
                float(K_model[1, 1]),
                float(K_model[0, 2]),
                float(K_model[1, 2]),
            ],
            "source_size": {"width": source_width, "height": source_height},
            "pixel_transform": spatial["pixel_transform"],
            "K_px": K_source.astype(float).tolist(),
        },
        "image_shape": {
            "height": int(spatial["model_size"]["height"]),
            "width": int(spatial["model_size"]["width"]),
        },
        "model_revision": DROID_MODEL_REVISION,
        "options": session_options,
    }
    lifecycle: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []
    last_keyframe_count = 0
    create_service_provenance = {
        "service_profile": profile.get("profile"),
        "service_base_url": base_url,
        "profile_droid": dict(profile_row) if isinstance(profile_row, Mapping) else None,
        "pinned_service_release": DROID_PINNED_RELEASE,
        "model_revision": DROID_MODEL_REVISION,
        "compatibility_compensations": dict(DROID_COMPENSATIONS),
        "session_options": session_options,
        "create_ownership": create_ownership,
        "create_request_metadata": create_metadata,
        "retry_events": retry_events,
        "camera": camera_provenance,
        "dynamic_mask": dynamic_mask,
    }
    create_report: Any = None
    create_call_returned = False
    trusted_session_id: str | None = None
    try:
        create_report = call_typed(
            caller,
            base_url=base_url,
            route="/droid.create_session",
            metadata=create_metadata,
            arrays={},
            timeout_s=float(args.timeout_s),
            retry_events=retry_events,
            retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)),
            retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)),
        )
        create_call_returned = True
        if not isinstance(create_report, Mapping):
            raise FeishuRayAdapterError(
                "feishu_ray_response_envelope_invalid",
                "/droid.create_session: caller returned a non-mapping response report",
                response_received=True,
            )
        create_envelope = require_droid_json_envelope(
            create_report,
            ownership=create_ownership,
            route="/droid.create_session",
        )
        candidate_session_id = create_envelope.get("session_id")
        if isinstance(candidate_session_id, str) and candidate_session_id:
            trusted_session_id = candidate_session_id
        session_id = validate_droid_create_response(create_report, ownership=create_ownership)
    except Exception as exc:
        if create_call_returned and isinstance(exc, FeishuRayAdapterError) and exc.response_received is None:
            exc.response_received = True
            if isinstance(create_report, Mapping) and type(create_report.get("http_status")) is int:
                exc.response_status = int(create_report["http_status"])
        response_received = True if create_call_returned else getattr(exc, "response_received", None)
        failure_fields = droid_create_failure_fields(
            report=create_report,
            trusted_session_id=trusted_session_id,
            response_received=response_received,
            cleanup_attempt_required=trusted_session_id is not None,
        )
        failure_dir: Path | None = None
        try:
            if isinstance(create_report, Mapping):
                failure_dir = preserve_failed_droid_finalize(
                    create_report,
                    run_root=run_root,
                    ownership=create_ownership,
                    session_id=trusted_session_id,
                    error=exc,
                    service_provenance=create_service_provenance,
                    route="/droid.create_session",
                    failure_family="feishu_ray_droid_create",
                    failure_status="failed_create_validation",
                    manifest_name="failed_create.json",
                    extra_fields={
                        **failure_fields,
                        "unavailable_response_evidence": ["response_headers", "raw_response_bytes"],
                    },
                )
            else:
                failure_dir = preserve_failed_droid_finalize_call(
                    run_root=run_root,
                    ownership=create_ownership,
                    session_id=None,
                    error=exc,
                    response_received=response_received,
                    response_status=getattr(exc, "response_status", None),
                    response_headers=getattr(exc, "response_headers", None),
                    raw_response_bytes=getattr(exc, "raw_response_bytes", None),
                    service_provenance=create_service_provenance,
                    route="/droid.create_session",
                    failure_family="feishu_ray_droid_create",
                    failure_status="failed_create_call",
                    manifest_name="failed_create.json",
                    extra_fields={
                        **failure_fields,
                        "returned_report": (
                            {"python_type": type(create_report).__name__, "repr": repr(create_report)}
                            if create_call_returned
                            else None
                        ),
                    },
                )
        except Exception as evidence_error:
            add_exception_note(exc, f"DROID create-session evidence could not be persisted: {evidence_error}")
        else:
            add_exception_note(exc, f"DROID failed create-session evidence: {failure_dir}")
        if trusted_session_id is not None:
            cleanup_path = cleanup_droid_session_without_masking(
                original_error=exc,
                caller=caller,
                run_root=run_root,
                base_url=base_url,
                timeout_s=float(args.timeout_s),
                job_id=job_id,
                session_id=trusted_session_id,
                trigger_route="/droid.create_session",
            )
            if failure_dir is not None:
                try:
                    update_failed_droid_create_cleanup(failure_dir, cleanup_path=cleanup_path)
                except Exception as update_error:
                    add_exception_note(
                        exc,
                        f"DROID create-session evidence cleanup link could not be updated: {update_error}",
                    )
        else:
            add_exception_note(
                exc,
                "DROID cleanup finalize not attempted because create_session established no trusted exact-string session handle.",
            )
        raise
    lifecycle.append({"route": "/droid.create_session", "ownership": create_ownership, "session_id": session_id})
    service_frames: list[dict[str, Any]] = []
    session_trigger_route = "/droid.push_frame"
    normal_finalize_call_started = False
    finalize_ownership: dict[str, Any] = {}
    service_provenance: dict[str, Any] | None = None
    preparation_wall_s = 0.0
    push_wall_s = 0.0

    def prepare_frame(position: int) -> dict[str, Any]:
        frame_row = frames[position]
        expected = expected_timeline[position]
        frame_idx = int(frame_row["frame_idx"])
        preparation_started = time.perf_counter()
        rgb_path = resolve_manifest_rgb(run_root, repo_root, str(frame_row["rgb"]))
        with Image.open(rgb_path) as image:
            source_rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if source_rgb.shape[:2] != (source_height, source_width):
            raise FeishuRayAdapterError(
                "droid_frame_shape_changed",
                f"frame {frame_idx} shape {source_rgb.shape[:2]} != {(source_height, source_width)}",
            )
        model_rgb = resize_rgb_to_droid_model(source_rgb, spatial)
        submitted_mask = np.ascontiguousarray(submitted_masks[position], dtype=np.float32)
        expected_mask_hash = mask_provenance["frames"][position]["submitted_model_mask_sha256"]
        actual_mask_hash = sha256_array(submitted_mask)
        if actual_mask_hash != expected_mask_hash:
            raise FeishuRayAdapterError(
                "feishu_ray_droid_mask_hash_mismatch",
                f"submitted mask frame {frame_idx} hash changed after persistence",
            )
        submitted_rgb = grayscale_and_zero_droid_rgb(model_rgb, submitted_mask)
        return {
            "frame_idx": frame_idx,
            "timestamp": float(expected["source_timestamp_s"]),
            "frame_id": expected["source_frame_id"],
            "rgb_path": rgb_path,
            "source_rgb_sha256": sha256_file(rgb_path),
            "submitted_rgb": submitted_rgb,
            "submitted_rgb_sha256": sha256_array(submitted_rgb),
            "submitted_mask": submitted_mask,
            "submitted_mask_sha256": actual_mask_hash,
            "preparation_elapsed_s": time.perf_counter() - preparation_started,
        }

    prefetch_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="droid-frame-prefetch",
    )
    try:
        prepared_future = prefetch_executor.submit(prepare_frame, 0)
        for position, expected in enumerate(expected_timeline):
            prepared = prepared_future.result()
            if position + 1 < len(expected_timeline):
                # Only preparation is overlapped. Stateful DROID pushes remain
                # strictly ordered and single-flight within this session.
                prepared_future = prefetch_executor.submit(prepare_frame, position + 1)
            preparation_wall_s += float(prepared["preparation_elapsed_s"])
            frame_idx = int(prepared["frame_idx"])
            timestamp = float(prepared["timestamp"])
            frame_id = str(prepared["frame_id"])
            rgb_path = Path(prepared["rgb_path"])
            submitted_rgb = prepared["submitted_rgb"]
            submitted_mask = prepared["submitted_mask"]
            actual_mask_hash = str(prepared["submitted_mask_sha256"])
            push_ownership = make_ownership(
                job_id=job_id,
                item_id=f"frame-{frame_idx:06d}",
                stage_id="droid.push_frame",
                source_id=frame_id,
                source_timestamp_s=timestamp,
            )
            push_metadata = {
                "ownership": push_ownership,
                "session_id": session_id,
                "frame_id": frame_id,
                "source_timestamp_s": timestamp,
                "model_revision": DROID_MODEL_REVISION,
            }
            push_started = time.perf_counter()
            push_report = call_typed(
                caller,
                base_url=base_url,
                route="/droid.push_frame",
                metadata=push_metadata,
                arrays={
                    "rgb": (submitted_rgb.tobytes(), tuple(submitted_rgb.shape), "uint8"),
                    "static_confidence_mask": (
                        submitted_mask.tobytes(),
                        tuple(submitted_mask.shape),
                        "float32",
                    ),
                },
                timeout_s=float(args.timeout_s),
                retry_events=retry_events,
                retry_max_wait_s=float(getattr(args, "retry_max_wait_s", 0.0)),
                retry_initial_delay_s=float(getattr(args, "retry_initial_delay_s", 1.0)),
            )
            push_wall_s += time.perf_counter() - push_started
            push_status = validate_droid_push_response(
                push_report,
                ownership=push_ownership,
                session_id=session_id,
                frame_id=frame_id,
                source_timestamp_s=timestamp,
                expected_keyframe_count=max(1, last_keyframe_count),
            )
            last_keyframe_count = int(push_status["keyframe_count"])
            trace = push_status.get("trace")
            service_frames.append(
                {
                    "frame_idx": frame_idx,
                    "source_frame_id": frame_id,
                    "source_timestamp_s": timestamp,
                    "ownership": push_ownership,
                    "source_rgb_path": str(rgb_path),
                    "source_rgb_sha256": str(prepared["source_rgb_sha256"]),
                    "submitted_rgb_sha256": str(prepared["submitted_rgb_sha256"]),
                    "submitted_mask_sha256": actual_mask_hash,
                    "submitted_mask_positive_is_ignore": True,
                    "submitted_rgb_channel_symmetric": True,
                    "submitted_dynamic_pixels_zeroed_where": "submitted_mask > 0.0",
                    "keyframe_count": int(push_status["keyframe_count"]),
                    "trace": dict(trace) if isinstance(trace, Mapping) else None,
                }
            )
            lifecycle.append(
                {
                    "route": "/droid.push_frame",
                    "ownership": push_ownership,
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "source_timestamp_s": timestamp,
                    "keyframe_added": bool(push_status["validity"]["keyframe_added"]),
                    "keyframe_count": int(push_status["keyframe_count"]),
                }
            )
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
        prefetch_executor = None
        session_trigger_route = "/droid.finalize"
        finalize_ownership = make_ownership(
            job_id=job_id,
            item_id=job_id,
            stage_id="droid.finalize",
            source_id=job_id,
            source_timestamp_s=None,
        )
        service_provenance = {
            "service_profile": profile.get("profile"),
            "service_base_url": base_url,
            "profile_droid": dict(profile_row) if isinstance(profile_row, Mapping) else None,
            "pinned_service_release": DROID_PINNED_RELEASE,
            "model_revision": DROID_MODEL_REVISION,
            "compatibility_compensations": dict(DROID_COMPENSATIONS),
            "session_options": session_options,
            "create_ownership": create_ownership,
            "finalize_ownership": finalize_ownership,
            "session_id": session_id,
            "lifecycle": lifecycle,
            "service_frames": service_frames,
            "retry_events": retry_events,
            "caller_timing": {
                "frame_preparation_wall_s": preparation_wall_s,
                "push_call_wall_s": push_wall_s,
                "prefetch_workers": 1,
                "ordered_single_flight_push": True,
            },
            "camera": camera_provenance,
            "dynamic_mask": dynamic_mask,
        }
        normal_finalize_call_started = True
        finalize_report = call_typed(
            caller,
            base_url=base_url,
            route="/droid.finalize",
            metadata={
                "ownership": finalize_ownership,
                "session_id": session_id,
                "model_revision": DROID_MODEL_REVISION,
            },
            arrays={},
            timeout_s=float(args.timeout_s),
            retry_events=retry_events,
            allow_retryable=False,
        )
    except Exception as exc:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True, cancel_futures=True)
        if normal_finalize_call_started:
            response_received = getattr(exc, "response_received", None)
            try:
                evidence_dir = preserve_failed_droid_finalize_call(
                    run_root=run_root,
                    ownership=finalize_ownership,
                    session_id=session_id,
                    error=exc,
                    response_received=response_received,
                    response_status=getattr(exc, "response_status", None),
                    response_headers=getattr(exc, "response_headers", None),
                    raw_response_bytes=getattr(exc, "raw_response_bytes", None),
                    service_provenance=service_provenance,
                )
            except Exception as evidence_error:
                add_exception_note(exc, f"DROID finalize call evidence could not be persisted: {evidence_error}")
            else:
                add_exception_note(exc, f"DROID failed finalize call evidence: {evidence_dir}")
            if response_received is False:
                cleanup_droid_session_without_masking(
                    original_error=exc,
                    caller=caller,
                    run_root=run_root,
                    base_url=base_url,
                    timeout_s=float(args.timeout_s),
                    job_id=job_id,
                    session_id=session_id,
                    trigger_route=session_trigger_route,
                )
            else:
                add_exception_note(
                    exc,
                    f"DROID cleanup finalize not retried because normal finalize response_received is {response_received!r}.",
                )
        else:
            cleanup_droid_session_without_masking(
                original_error=exc,
                caller=caller,
                run_root=run_root,
                base_url=base_url,
                timeout_s=float(args.timeout_s),
                job_id=job_id,
                session_id=session_id,
                trigger_route=session_trigger_route,
            )
        raise
    lifecycle.append({"route": "/droid.finalize", "ownership": finalize_ownership, "session_id": session_id})
    if service_provenance is None:
        raise AssertionError("DROID service provenance was not initialized")
    try:
        if not isinstance(finalize_report, Mapping):
            raise FeishuRayAdapterError(
                "feishu_ray_response_envelope_invalid",
                "/droid.finalize: caller returned a non-mapping response report",
            )
        validate_droid_finalize_compatibility(
            finalize_report,
            ownership=finalize_ownership,
            session_id=session_id,
            expected_timeline=expected_timeline,
            expected_model_K=K_model,
            expected_model_size=spatial["model_size"],
        )
        clip = resolve_droid_clip(run_root, manifest)
        materialized = materialize_droid_finalize(
            finalize_report,
            ownership=finalize_ownership,
            expected_timeline=expected_timeline,
            output_dir=output_dir,
            fps=fps,
            clip=clip,
            clip_sha256=sha256_file(clip),
            dynamic_mask=dynamic_mask,
            camera_provenance=camera_provenance,
            service_provenance=service_provenance,
        )
    except Exception as exc:
        try:
            if isinstance(finalize_report, Mapping):
                evidence_dir = preserve_failed_droid_finalize(
                    finalize_report,
                    run_root=run_root,
                    ownership=finalize_ownership,
                    session_id=session_id,
                    error=exc,
                    service_provenance=service_provenance,
                )
            else:
                evidence_dir = preserve_failed_droid_finalize_call(
                    run_root=run_root,
                    ownership=finalize_ownership,
                    session_id=session_id,
                    error=exc,
                    response_received=True,
                    response_status=None,
                    response_headers=None,
                    raw_response_bytes=None,
                    service_provenance=service_provenance,
                )
        except Exception as evidence_error:
            add_exception_note(exc, f"DROID finalize evidence could not be persisted: {evidence_error}")
        else:
            add_exception_note(exc, f"DROID failed finalize evidence: {evidence_dir}")
        raise
    del submitted_masks
    return {
        "status": "ok",
        "method": "feishu_ray_droid_adapter",
        "run_root": str(run_root),
        "processed_frames": len(frames),
        "session_id": session_id,
        "service_profile": profile.get("profile"),
        "service_base_url": base_url,
        "pinned_service_release": DROID_PINNED_RELEASE,
        "compatibility_compensations": dict(DROID_COMPENSATIONS),
        "dynamic_mask": dynamic_mask,
        "shared_geometry_manifest": materialized["manifest_path"],
        "shared_geometry_manifest_sha256": materialized["manifest_sha256"],
        "elapsed_s": float(time.time() - started),
        "caller_timing": {
            "frame_preparation_wall_s": preparation_wall_s,
            "push_call_wall_s": push_wall_s,
            "prefetch_workers": 1,
            "ordered_single_flight_push": True,
        },
        "retry_events": retry_events,
    }


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise FeishuRayAdapterError("feishu_ray_droid_invalid_rotation", "DROID returned a zero quaternion")
    quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def validate_droid_finalize(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    expected_timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(expected_timeline, list) or not expected_timeline:
        raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", "expected source timeline is empty")
    for index, row in enumerate(expected_timeline):
        if not isinstance(row, dict) or row.get("source_timestamp_s") is None:
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"expected timeline row {index} is invalid")
        adapter_string(
            row.get("source_frame_id"),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected timeline source_frame_id {index} is invalid",
        )
        timestamp = adapter_float(
            row["source_timestamp_s"],
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected timeline timestamp {index} is invalid",
        )
        if not math.isfinite(timestamp):
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"expected timeline timestamp {index} is non-finite")
    _, state, arrays = require_success(report, expected_ownership=ownership, route="/droid.finalize")
    if not ownership_matches(ownership, state.get("ownership")):
        raise FeishuRayAdapterError(
            "feishu_ray_ownership_mismatch",
            "/droid.finalize: nested result ownership is required and must match request ownership",
        )
    adapter_string(
        state.get("session_id"),
        code="feishu_ray_droid_session_mismatch",
        message="droid.finalize returned a non-string session_id",
    )
    uncertainty = state.get("uncertainty")
    if not isinstance(uncertainty, dict):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_nonfinite_geometry",
            "DROID finalize result lacks uncertainty metadata",
        )
    finite_pose_ratio = adapter_float(
        uncertainty.get("finite_pose_ratio", 0.0),
        code="feishu_ray_droid_finalize_metadata_invalid",
        message="DROID finite_pose_ratio is not numeric",
    )
    if finite_pose_ratio != 1.0:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_nonfinite_geometry",
            f"DROID finite_pose_ratio is {finite_pose_ratio}",
        )
    T_world_camera = decode_array(arrays, "T_world_camera", dtypes=("float32", "float64"))
    T_camera_world = decode_array(arrays, "T_camera_world", dtypes=("float32", "float64"))
    if T_world_camera.ndim != 3 or T_world_camera.shape[1:] != (4, 4) or T_camera_world.shape != T_world_camera.shape:
        raise FeishuRayAdapterError("feishu_ray_droid_pose_shape", "DROID pose arrays must be matching [N,4,4]")
    for name, array in (("T_world_camera", T_world_camera), ("T_camera_world", T_camera_world)):
        if not np.isfinite(array).all():
            raise FeishuRayAdapterError("feishu_ray_droid_nonfinite_geometry", f"DROID {name} contains NaN or infinity")
        if not np.allclose(array[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
            raise FeishuRayAdapterError("feishu_ray_droid_invalid_se3", f"DROID {name} has invalid homogeneous rows")
    identity = np.matmul(T_world_camera, T_camera_world)
    if not np.allclose(identity, np.eye(4, dtype=np.float64)[None], atol=1.0e-5, rtol=1.0e-5):
        raise FeishuRayAdapterError("feishu_ray_droid_pose_inverse_mismatch", "T_world_camera and T_camera_world disagree")
    for name, array in (("T_world_camera", T_world_camera), ("T_camera_world", T_camera_world)):
        rotations = array[:, :3, :3]
        gram = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
        determinants = np.linalg.det(rotations)
        if not np.allclose(gram, np.eye(3, dtype=np.float64)[None], atol=1.0e-4, rtol=1.0e-4) or not np.allclose(
            determinants,
            1.0,
            atol=1.0e-4,
            rtol=1.0e-4,
        ):
            raise FeishuRayAdapterError("feishu_ray_droid_invalid_se3", f"DROID {name} rotation blocks are not proper rotations")
    dense_mapping = state.get("dense_mapping")
    if not isinstance(dense_mapping, list) or len(dense_mapping) != len(expected_timeline) or len(dense_mapping) != T_world_camera.shape[0]:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_incomplete_timeline",
            f"dense mapping/pose count {len(dense_mapping) if isinstance(dense_mapping, list) else None}/{T_world_camera.shape[0]} != expected {len(expected_timeline)}",
        )
    for index, (mapping, expected) in enumerate(zip(dense_mapping, expected_timeline)):
        expected_frame_idx = adapter_int(
            expected.get("frame_idx", index),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected frame index is invalid at row {index}",
        )
        if expected_frame_idx != index:
            raise FeishuRayAdapterError(
                "feishu_ray_droid_incomplete_timeline",
                f"expected source timeline is not contiguous from zero at row {index}",
            )
        if not isinstance(mapping, dict):
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"dense mapping row {index} is invalid")
        dense_index = adapter_int(
            mapping.get("dense_index", -1),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"dense mapping index is invalid at row {index}",
        )
        if dense_index != index:
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"dense index mismatch at {index}")
        mapping_source_id = adapter_string(
            mapping.get("source_frame_id"),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"dense mapping source_frame_id is invalid at row {index}",
        )
        if mapping_source_id != expected["source_frame_id"]:
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"source frame mismatch at {index}")
        mapping_timestamp = adapter_float(
            mapping.get("source_timestamp_s"),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"dense mapping timestamp is invalid at row {index}",
        )
        expected_timestamp = adapter_float(
            expected.get("source_timestamp_s"),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected dense timestamp is invalid at row {index}",
        )
        if not math.isclose(
            mapping_timestamp,
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise FeishuRayAdapterError("feishu_ray_droid_incomplete_timeline", f"source timestamp mismatch at {index}")
    keyframe_mapping = state.get("keyframe_mapping")
    if not isinstance(keyframe_mapping, list) or not keyframe_mapping:
        raise FeishuRayAdapterError("feishu_ray_droid_keyframes_missing", "DROID keyframe mapping is empty")
    expected_by_source = {row["source_frame_id"]: row for row in expected_timeline}
    keyframe_sources: set[str] = set()
    for index, mapping in enumerate(keyframe_mapping):
        if not isinstance(mapping, dict):
            raise FeishuRayAdapterError("feishu_ray_droid_keyframe_mapping_mismatch", f"invalid keyframe mapping row {index}")
        keyframe_index = adapter_int(
            mapping.get("keyframe_index", -1),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"keyframe mapping index is invalid at row {index}",
        )
        if keyframe_index != index:
            raise FeishuRayAdapterError("feishu_ray_droid_keyframe_mapping_mismatch", f"invalid keyframe mapping row {index}")
        source_id = adapter_string(
            mapping.get("source_frame_id"),
            code="feishu_ray_droid_keyframe_mapping_mismatch",
            message=f"keyframe source_frame_id is invalid at row {index}",
        )
        expected = expected_by_source.get(source_id)
        if expected is None or source_id in keyframe_sources:
            raise FeishuRayAdapterError("feishu_ray_droid_keyframe_mapping_mismatch", f"unknown or duplicate keyframe source {source_id}")
        keyframe_sources.add(source_id)
        mapping_timestamp = adapter_float(
            mapping.get("source_timestamp_s"),
            code="feishu_ray_droid_finalize_metadata_invalid",
            message=f"keyframe mapping timestamp is invalid at row {index}",
        )
        expected_timestamp = adapter_float(
            expected.get("source_timestamp_s"),
            code="feishu_ray_droid_incomplete_timeline",
            message=f"expected keyframe timestamp is invalid at row {index}",
        )
        if not math.isclose(
            mapping_timestamp,
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise FeishuRayAdapterError("feishu_ray_droid_keyframe_mapping_mismatch", f"keyframe timestamp mismatch at {index}")
    disparities = decode_array(arrays, "disparities", dtypes=("float32",))
    intrinsics = decode_array(arrays, "intrinsics_px", dtypes=("float32", "float64"))
    if disparities.ndim != 3 or disparities.shape[0] != len(keyframe_mapping):
        raise FeishuRayAdapterError("feishu_ray_droid_disparity_shape", "DROID disparities do not match keyframes")
    if intrinsics.shape != (len(keyframe_mapping), 3, 3):
        raise FeishuRayAdapterError("feishu_ray_droid_intrinsics_shape", "DROID intrinsics do not match keyframes")
    if not np.isfinite(disparities).all() or np.any(disparities <= 0.0):
        raise FeishuRayAdapterError("feishu_ray_droid_nonfinite_geometry", "DROID disparities must be finite and positive")
    if (
        not np.isfinite(intrinsics).all()
        or np.any(intrinsics[:, 0, 0] <= 0.0)
        or np.any(intrinsics[:, 1, 1] <= 0.0)
        or not np.allclose(intrinsics[:, 2], [0.0, 0.0, 1.0], atol=1.0e-6)
    ):
        raise FeishuRayAdapterError("feishu_ray_droid_invalid_intrinsics", "DROID intrinsics must be finite pinhole matrices with positive focal values")
    return {
        "state": state,
        "T_world_camera": T_world_camera.astype(np.float64),
        "T_camera_world": T_camera_world.astype(np.float64),
        "disparities": disparities.astype(np.float32),
        "intrinsics_px": intrinsics.astype(np.float64),
        "dense_mapping": dense_mapping,
        "keyframe_mapping": keyframe_mapping,
    }


def validate_dynamic_mask_contract(dynamic_mask: dict[str, Any], *, expected_frames: int) -> dict[str, Any]:
    status = str(dynamic_mask.get("status") or "")
    if status == "not_provided":
        return dict(dynamic_mask)
    if status not in {"applied", "applied_from_hawor_preparation"}:
        raise FeishuRayAdapterError("feishu_ray_droid_mask_status_invalid", f"unknown dynamic mask status {status!r}")
    source_semantics = dynamic_mask.get("source_value_semantics")
    submitted_semantics = dynamic_mask.get("submitted_value_semantics")
    service_semantics = dynamic_mask.get("service_consumption_semantics")
    conversion = dynamic_mask.get("source_to_submitted_conversion")
    if source_semantics != "1=dynamic_ignore,0=static_keep":
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_semantics_unclear",
            "dynamic mask must declare source_value_semantics=1=dynamic_ignore,0=static_keep",
        )
    allowed_conversions = {
        "1=ignore,0=retain": {"identity", "area_resample_positive_preserved"},
        "positive=ignore,0=retain": {"area_resample_positive_preserved"},
        "1=retain,0=ignore": {"one_minus_source"},
    }
    if submitted_semantics not in allowed_conversions or service_semantics != submitted_semantics:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_semantics_unclear",
            "submitted and service mask value semantics must agree explicitly",
        )
    if conversion not in allowed_conversions[submitted_semantics]:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_semantics_unclear",
            f"mask conversion {conversion!r} is inconsistent with submitted semantics {submitted_semantics!r}",
        )
    raw_path = dynamic_mask.get("path")
    expected_hash = dynamic_mask.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_hash, str) or not expected_hash:
        raise FeishuRayAdapterError("feishu_ray_droid_mask_artifact_missing", "applied dynamic mask requires path and sha256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FeishuRayAdapterError("feishu_ray_droid_mask_artifact_missing", f"dynamic mask file is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_hash_mismatch",
            f"dynamic mask hash mismatch: expected={expected_hash} actual={actual_hash}",
        )
    try:
        masks = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FeishuRayAdapterError("feishu_ray_droid_mask_decode_failed", f"could not load dynamic mask: {exc}") from exc
    if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] != int(expected_frames):
        shape = tuple(masks.shape) if isinstance(masks, np.ndarray) else None
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_timeline_mismatch",
            f"dynamic mask shape {shape} does not cover {expected_frames} frames",
        )
    if masks.dtype.kind not in {"b", "u", "i", "f"} or not np.isfinite(masks).all():
        raise FeishuRayAdapterError("feishu_ray_droid_mask_values_invalid", "dynamic mask must be finite numeric data")
    minimum = float(masks.min()) if masks.size else 0.0
    maximum = float(masks.max()) if masks.size else 0.0
    if minimum < 0.0 or maximum > 1.0:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_mask_values_invalid",
            f"dynamic mask values must be in [0,1], got [{minimum},{maximum}]",
        )
    return {
        **dynamic_mask,
        "path": str(path),
        "shape": [int(value) for value in masks.shape],
        "dtype": str(masks.dtype),
    }


def build_droid_finalize_staging(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    expected_timeline: list[dict[str, Any]],
    staging_dir: Path,
    published_output_dir: Path,
    fps: float,
    clip: Path,
    clip_sha256: str | None,
    dynamic_mask: dict[str, Any],
    camera_provenance: Mapping[str, Any] | None = None,
    service_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_droid_finalize(report, ownership=ownership, expected_timeline=expected_timeline)
    dynamic_mask = validate_dynamic_mask_contract(dynamic_mask, expected_frames=len(expected_timeline))
    camera_provenance = dict(camera_provenance or {})
    service_provenance = dict(service_provenance or {})
    T_world_camera = validated["T_world_camera"]
    frame_idx = np.asarray([int(row.get("frame_idx", index)) for index, row in enumerate(expected_timeline)], dtype=np.int32)
    poses = np.zeros((len(frame_idx), 7), dtype=np.float32)
    poses[:, :3] = T_world_camera[:, :3, 3].astype(np.float32)
    for index, matrix in enumerate(T_world_camera):
        poses[index, 3:] = matrix_to_quaternion_xyzw(matrix[:3, :3]).astype(np.float32)
    keyframes = validated["keyframe_mapping"]
    keyframe_frame_idx: list[int] = []
    timeline_by_id = {row["source_frame_id"]: int(row.get("frame_idx", index)) for index, row in enumerate(expected_timeline)}
    for row in keyframes:
        source_id = row["source_frame_id"]
        if source_id not in timeline_by_id:
            raise FeishuRayAdapterError("feishu_ray_droid_keyframe_mapping_mismatch", f"unknown keyframe source {source_id}")
        keyframe_frame_idx.append(timeline_by_id[source_id])
    staging_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = staging_dir / "droid_dense_trajectory.npz"
    published_trajectory_path = published_output_dir / trajectory_path.name
    intrinsics = validated["intrinsics_px"]
    source_K_raw = camera_provenance.get("K_source_px")
    source_K = np.asarray(source_K_raw, dtype=np.float64) if source_K_raw is not None else None
    if source_K is not None:
        if (
            source_K.shape != (3, 3)
            or not np.isfinite(source_K).all()
            or source_K[0, 0] <= 0.0
            or source_K[1, 1] <= 0.0
            or not np.allclose(source_K[2], [0.0, 0.0, 1.0], atol=1.0e-6)
        ):
            raise FeishuRayAdapterError("feishu_ray_droid_invalid_intrinsics", "source camera provenance contains invalid K_source_px")
        intrinsics_4 = np.asarray(
            [source_K[0, 0], source_K[1, 1], source_K[0, 2], source_K[1, 2]],
            dtype=np.float32,
        )
    else:
        intrinsics_4 = np.asarray(
            [
                np.median(intrinsics[:, 0, 0]),
                np.median(intrinsics[:, 1, 1]),
                np.median(intrinsics[:, 0, 2]),
                np.median(intrinsics[:, 1, 2]),
            ],
            dtype=np.float32,
        )
    np.savez_compressed(
        trajectory_path,
        frame_idx=frame_idx,
        pose_world_camera_xyzw=poses,
        T_world_camera=T_world_camera.astype(np.float32),
        intrinsics_source=intrinsics_4,
        fps=np.asarray([float(fps)], dtype=np.float32),
    )
    trajectory_json = staging_dir / "droid_dense_trajectory.json"
    published_trajectory_json = published_output_dir / trajectory_json.name
    write_json(
        trajectory_json,
        {
            "frames": [
                {
                    "frame_idx": int(frame_idx[index]),
                    "pose_world_camera_xyzw": poses[index].astype(float).tolist(),
                    "T_world_camera": T_world_camera[index].astype(float).tolist(),
                }
                for index in range(len(frame_idx))
            ]
        },
    )
    reconstruction_path = staging_dir / "droid_keyframe_reconstruction.npz"
    published_reconstruction_path = published_output_dir / reconstruction_path.name
    np.savez_compressed(
        reconstruction_path,
        tstamps=np.asarray(keyframe_frame_idx, dtype=np.int32),
        disps=validated["disparities"],
        disps_low=validated["disparities"],
        intrinsics=np.stack(
            [intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]],
            axis=1,
        ).astype(np.float32),
        depth_level=np.asarray("network_stride_8"),
    )
    keyframes_path = staging_dir / "droid_keyframes.json"
    published_keyframes_path = published_output_dir / keyframes_path.name
    write_json(
        keyframes_path,
        {
            "keyframes": [
                {
                    "keyframe_index": int(row.get("keyframe_index", index)),
                    "source_frame_idx": int(keyframe_frame_idx[index]),
                    "source_frame_id": row["source_frame_id"],
                    "source_timestamp_s": adapter_float(
                        row.get("source_timestamp_s"),
                        code="feishu_ray_droid_finalize_metadata_invalid",
                        message=f"keyframe source timestamp is invalid at row {index}",
                    ),
                }
                for index, row in enumerate(keyframes)
            ]
        },
    )
    qc_path = staging_dir / "droid_qc.json"
    published_qc_path = published_output_dir / qc_path.name
    qc = {
        "status": "ok",
        "backend": "feishu_ray_droid",
        "processed_frames": len(frame_idx),
        "trajectory_path": str(published_trajectory_path),
        "keyframe_count": len(keyframes),
        "scale_status": validated["state"].get("uncertainty", {}).get("scale_status"),
        "uncertainty": validated["state"].get("uncertainty"),
        "dynamic_mask": dynamic_mask,
        "camera_provenance": camera_provenance,
        "service_provenance": service_provenance,
        "claim_scope": "Mask-aware DROID camera geometry in an arbitrary monocular gauge; no metric scale claim.",
    }
    write_json(qc_path, qc)
    stage_path = staging_dir / "v22_camera_trajectory_stage.json"
    run_root = published_output_dir.parents[2]
    stage = {
        "schema": "v22_camera_trajectory_stage.v0",
        "status": "ok",
        "run_root": str(run_root),
        "clip": str(clip),
        "calibration_contract": camera_provenance.get("calibration_contract"),
        "camera_backend": "droid",
        "execution_backend": "feishu_ray",
        "replacement_for": "D4_droid_head_camera_trajectory",
        "dynamic_mask": dynamic_mask,
        "outputs": {
            "output_dir": str(published_output_dir),
            "dense_npz": str(published_trajectory_path),
            "dense_json": str(published_trajectory_json),
            "qc_json": str(published_qc_path),
            "shared_geometry_manifest": str(published_output_dir / "droid_shared_geometry.json"),
        },
        "claim_scope": "D4 Feishu Ray DROID camera trajectory in an arbitrary monocular gauge; metric scale is estimated downstream from depth/disparity evidence.",
        "gauge_declaration": {
            "trajectory_frame": "DROID arbitrary world gauge",
            "scale_status": "video_derived_uncertain_without_external_metric_anchor",
            "metric_anchor_needed": "device VIO/SLAM/IMU, fiducial/mocap, known-size geometry, or fixed-gauge benchmark GT",
        },
        "service_provenance": service_provenance,
    }
    write_json(stage_path, stage)
    artifacts = {
        "dense_trajectory": {
            "path": str(published_trajectory_path),
            "sha256": sha256_file(trajectory_path),
            "frame_idx_key": "frame_idx",
            "trajectory_key": "pose_world_camera_xyzw",
            "matrix_key": "T_world_camera",
            "fps_key": "fps",
            "trajectory_for_hawor": "pose_world_camera_xyzw",
        },
        "dense_trajectory_json": {
            "path": str(published_trajectory_json),
            "sha256": sha256_file(trajectory_json),
        },
        "keyframe_reconstruction": {
            "path": str(published_reconstruction_path),
            "sha256": sha256_file(reconstruction_path),
            "timestamp_key": "tstamps",
            "disparity_key": "disps",
            "low_resolution_disparity_key": "disps_low",
            "intrinsics_key": "intrinsics",
            "depth_level_key": "depth_level",
        },
        "keyframes": {"path": str(published_keyframes_path), "sha256": sha256_file(keyframes_path)},
        "droid_qc": {"path": str(published_qc_path), "sha256": sha256_file(qc_path)},
    }
    if str(dynamic_mask.get("status", "")).startswith("applied") and dynamic_mask.get("path"):
        artifacts["dynamic_mask"] = dict(dynamic_mask)
    manifest_path = staging_dir / "droid_shared_geometry.json"
    published_manifest_path = published_output_dir / manifest_path.name
    manifest = {
        "schema": "v22_shared_droid_geometry.v1",
        "status": "ok",
        "backend": "droid",
        "execution_backend": "feishu_ray",
        "clip": str(clip),
        "clip_sha256": clip_sha256,
        "processed_frames": len(frame_idx),
        "full_source_timeline": True,
        "droid_invocation": {
            "class": "feishu_ray.droid_session",
            "instance_count": 1,
            "track_call_count": len(frame_idx),
            "terminate_call_count": 1,
            "session_id": validated["state"].get("session_id"),
        },
        "pose_contract": {
            "raw_terminate_vector": "[tx, ty, tz, qx, qy, qz, qw] world-from-camera",
            "hawor_consumption_key": "dense_trajectory.pose_world_camera_xyzw",
            "d4_matrix_key": "dense_trajectory.T_world_camera",
            "conversion_rule": "Both values are materialized from the service T_world_camera after inverse-consistency validation.",
        },
        "scale_contract": {
            "droid_scale": "arbitrary_video_gauge",
            "metric_scale": "computed downstream from DROID disparity and metric depth; not part of DROID service output",
        },
        "dynamic_mask": dynamic_mask,
        "camera_provenance": camera_provenance,
        "service_provenance": service_provenance,
        "artifacts": artifacts,
        "consumers": ["D4_camera_trajectory", "HaWoR_SLAM_adapter", "D7_hybrid_fusion_via_HaWoR_world_npz"],
    }
    write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(published_manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_droid_staged_publication(
    *,
    staging_dir: Path,
    published_output_dir: Path,
    materialized: Mapping[str, Any],
) -> None:
    expected_files = {
        "droid_dense_trajectory.npz",
        "droid_dense_trajectory.json",
        "droid_keyframe_reconstruction.npz",
        "droid_keyframes.json",
        "droid_qc.json",
        "droid_shared_geometry.json",
        "v22_camera_trajectory_stage.json",
    }
    observed_files = {path.name for path in staging_dir.iterdir() if path.is_file()}
    if observed_files != expected_files or any(path.is_dir() for path in staging_dir.iterdir()):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_publication_invalid",
            f"staged D4 files {sorted(observed_files)} != {sorted(expected_files)}",
        )
    try:
        with np.load(staging_dir / "droid_dense_trajectory.npz", allow_pickle=False) as trajectory:
            if not {"frame_idx", "pose_world_camera_xyzw", "T_world_camera", "intrinsics_source", "fps"}.issubset(
                trajectory.files
            ):
                raise ValueError("dense trajectory archive lacks canonical arrays")
        with np.load(staging_dir / "droid_keyframe_reconstruction.npz", allow_pickle=False) as reconstruction:
            if not {"tstamps", "disps", "disps_low", "intrinsics", "depth_level"}.issubset(
                reconstruction.files
            ):
                raise ValueError("keyframe reconstruction archive lacks canonical arrays")
        for filename in (
            "droid_dense_trajectory.json",
            "droid_keyframes.json",
            "droid_qc.json",
        ):
            load_json_object(staging_dir / filename)
        stage = load_json_object(staging_dir / "v22_camera_trajectory_stage.json")
        if (
            stage.get("status") != "ok"
            or stage.get("camera_backend") != "droid"
            or stage.get("execution_backend") != "feishu_ray"
            or (stage.get("outputs") or {}).get("dense_json") != str(published_output_dir / "droid_dense_trajectory.json")
        ):
            raise ValueError("camera trajectory stage contract is invalid")
        staged_manifest = load_json_object(staging_dir / "droid_shared_geometry.json")
    except (OSError, ValueError) as exc:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_publication_invalid",
            f"staged D4 artifact validation failed: {exc}",
        ) from exc
    expected_manifest = {
        key: value
        for key, value in materialized.items()
        if key not in {"manifest_path", "manifest_sha256"}
    }
    if staged_manifest != expected_manifest:
        raise FeishuRayAdapterError(
            "feishu_ray_droid_publication_invalid",
            "staged D4 manifest does not match the validated in-memory manifest",
        )
    artifacts = staged_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_publication_invalid",
            "staged D4 manifest lacks artifact evidence",
        )
    canonical_artifacts = {
        "dense_trajectory": "droid_dense_trajectory.npz",
        "dense_trajectory_json": "droid_dense_trajectory.json",
        "keyframe_reconstruction": "droid_keyframe_reconstruction.npz",
        "keyframes": "droid_keyframes.json",
        "droid_qc": "droid_qc.json",
    }
    for artifact_name, filename in canonical_artifacts.items():
        row = artifacts.get(artifact_name)
        staged_path = staging_dir / filename
        expected_published_path = published_output_dir / filename
        if (
            not isinstance(row, Mapping)
            or row.get("path") != str(expected_published_path)
            or row.get("sha256") != sha256_file(staged_path)
        ):
            raise FeishuRayAdapterError(
                "feishu_ray_droid_publication_invalid",
                f"staged D4 artifact evidence is invalid for {artifact_name}",
            )
    if materialized.get("manifest_path") != str(published_output_dir / "droid_shared_geometry.json") or materialized.get(
        "manifest_sha256"
    ) != sha256_file(staging_dir / "droid_shared_geometry.json"):
        raise FeishuRayAdapterError(
            "feishu_ray_droid_publication_invalid",
            "staged D4 manifest path or hash is invalid",
        )


def materialize_droid_finalize(
    report: dict[str, Any],
    *,
    ownership: Mapping[str, Any],
    expected_timeline: list[dict[str, Any]],
    output_dir: Path,
    fps: float,
    clip: Path,
    clip_sha256: str | None,
    dynamic_mask: dict[str, Any],
    camera_provenance: Mapping[str, Any] | None = None,
    service_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FeishuRayAdapterError(
            "feishu_ray_droid_output_not_fresh",
            f"refusing to overwrite an existing D4 output directory: {output_dir}",
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{uuid4().hex}")
    try:
        materialized = build_droid_finalize_staging(
            report,
            ownership=ownership,
            expected_timeline=expected_timeline,
            staging_dir=staging_dir,
            published_output_dir=output_dir,
            fps=fps,
            clip=clip,
            clip_sha256=clip_sha256,
            dynamic_mask=dynamic_mask,
            camera_provenance=camera_provenance,
            service_provenance=service_provenance,
        )
        validate_droid_staged_publication(
            staging_dir=staging_dir,
            published_output_dir=output_dir,
            materialized=materialized,
        )
        os.replace(staging_dir, output_dir)
        return materialized
    except Exception as exc:
        for partial_path in (staging_dir, output_dir):
            if not partial_path.exists():
                continue
            try:
                if partial_path.is_dir():
                    shutil.rmtree(partial_path)
                else:
                    partial_path.unlink()
            except OSError as cleanup_error:
                add_exception_note(exc, f"D4 publication cleanup failed for {partial_path}: {cleanup_error}")
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("unidepth", "wilor", "droid"):
        stage = subparsers.add_parser(name)
        stage.add_argument("--run-root", type=Path, required=True)
        stage.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        stage.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
        stage.add_argument("--base-url", default=None)
        stage.add_argument("--job-id", default=None)
        stage.add_argument("--timeout-s", type=float, default=120.0)
        stage.add_argument("--retry-max-wait-s", type=float, default=0.0, help="Maximum cumulative wait for explicit retryable service responses; 0 means wait indefinitely")
        stage.add_argument("--retry-initial-delay-s", type=float, default=1.0)
    wilor = subparsers.choices["wilor"]
    wilor.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    wilor.add_argument("--wilor-config", type=Path, default=None)
    wilor.add_argument("--rescale-factor", type=float, default=2.0)
    wilor.add_argument("--compute-target", default="A800 Feishu Ray GPU1; local CPU preprocessing only")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        runners = {"unidepth": run_unidepth, "wilor": run_wilor, "droid": run_droid}
        report = runners[args.stage](args)
    except (FeishuRayAdapterError, FileNotFoundError) as exc:
        code = exc.code if isinstance(exc, FeishuRayAdapterError) else "file_not_found"
        print(json.dumps({"status": "failed", "code": code, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
