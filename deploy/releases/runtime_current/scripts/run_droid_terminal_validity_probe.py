#!/usr/bin/env python3
"""Replay the failed d1h4 D1/0.5 terminal stream through a direct DROID adapter.

This is intentionally not a Ray Serve deployment or endpoint probe. It creates a
fresh resident ``DroidAdapter`` and replays payloads 0000--0011 one at a time.
The curve and serving-default option profiles use the same payload bytes,
corrected model-grid camera, serial admission, and deterministic seed. The probe
observes frontend/pre-backend, backend(7), backend(12), filler, conversion, and
``CameraState`` finite boundaries without changing adapter logic.

The two normal runs for one profile must have identical boundary fingerprints;
otherwise the emitted verdict is ``nondeterministic``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ego_annotation.serving.contracts import (
    CameraState,
    ContractValidationError,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidFinalizeRequest,
    DroidFrameRequest,
    DroidImageShape,
    DroidSessionOptions,
    ImageSize,
    Ownership,
    PixelTransform,
    TensorPayload,
)
from ego_annotation.serving.droid import DroidAdapter, build_droid_model_config


D1H4_ROOT = Path(
    "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/"
    "runs/d1h4_20260721T183900Z/droid"
)
DEFAULT_MANIFEST = D1H4_ROOT / "payload_manifest.json"
DEFAULT_OUTPUT = Path(
    "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/"
    "benchmarks/droid_terminal_validity_probe/verdict.json"
)
# The V19 DROID calibration used by the V22 trajectory run. The payload corpus
# resizes its 1920×1080 source frames to the 568×320 DROID model grid.
_SOURCE_FOCAL_1920 = 1681.2077272759332
_SOURCE_SIZE = ImageSize(width=1920, height=1080)
_MODEL_SIZE = ImageSize(width=568, height=320)
_MODEL_GRID_INTRINSICS = (
    _SOURCE_FOCAL_1920 * _MODEL_SIZE.width / _SOURCE_SIZE.width,
    _SOURCE_FOCAL_1920 * _MODEL_SIZE.height / _SOURCE_SIZE.height,
    _MODEL_SIZE.width / 2.0,
    _MODEL_SIZE.height / 2.0,
)

OPTION_PROFILES: dict[str, DroidSessionOptions] = {
    "curve": DroidSessionOptions(
        buffer=128, filter_thresh=1.0, keyframe_thresh=2.0, warmup=2,
    ),
    "serving-defaults": DroidSessionOptions(
        buffer=1024, filter_thresh=2.4, keyframe_thresh=4.0, warmup=8,
        frontend_thresh=16.0, backend_thresh=22.0, upsample=True, beta=0.3,
    ),
}


@dataclass(frozen=True)
class StreamPayload:
    payload_id: str
    source_frame_index: int
    timestamp_s: float
    rgb_path: Path
    mask_path: Path
    rgb_sha256: str
    mask_sha256: str

    @property
    def frame_id(self) -> str:
        return f"0-{self.payload_id.removeprefix('payload-')}-{self.source_frame_index}"


@dataclass(frozen=True)
class FirstBadRow:
    row_index: int
    payload_id: str | None
    source_frame_index: int | None
    source_timestamp_s: float | None


@dataclass(frozen=True)
class FiniteCheck:
    stage: str
    total_rows: int
    finite_rows: int
    first_bad: FirstBadRow | None

    @property
    def all_finite(self) -> bool:
        return self.finite_rows == self.total_rows

    def to_wire(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "finite_rows": self.finite_rows,
            "first_bad": asdict(self.first_bad) if self.first_bad else None,
        }


@dataclass
class ProbeRecorder:
    """Keeps the first observation at every terminal boundary, never overwriting it."""

    checks: dict[str, FiniteCheck] = field(default_factory=dict)

    def check(self, stage: str, values: Any, mappings: Sequence[StreamPayload]) -> FiniteCheck:
        if stage in self.checks:
            return self.checks[stage]
        array = _to_numpy(values)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)
        rows = int(array.shape[0])
        flattened = array.reshape(rows, -1)
        row_finite = [bool(np.isfinite(flattened[index]).all()) for index in range(rows)]
        bad_index = next((index for index in range(rows) if not row_finite[index]), None)
        first_bad = None
        if bad_index is not None:
            payload = mappings[bad_index] if bad_index < len(mappings) else None
            first_bad = FirstBadRow(
                row_index=bad_index,
                payload_id=payload.payload_id if payload else None,
                source_frame_index=payload.source_frame_index if payload else None,
                source_timestamp_s=payload.timestamp_s if payload else None,
            )
        result = FiniteCheck(stage, rows, sum(row_finite), first_bad)
        self.checks[stage] = result
        return result

    def first_nonfinite(self) -> tuple[str, FiniteCheck] | None:
        # backend(7) and backend(12) are two observations of the one
        # ``backend_poses`` boundary; the earlier call wins when both are bad.
        for stage, boundary in (
            ("frontend_pre_backend_poses", "frontend_pre_backend"),
            ("frontend_pre_backend_disparities", "frontend_pre_backend"),
            ("backend_7", "backend_poses"),
            ("backend_12", "backend_poses"),
            ("filler_trajectory", "filler_trajectory"),
            ("pose_conversion", "pose_conversion"),
            ("camera_state_validation", "camera_state_validation"),
        ):
            check = self.checks.get(stage)
            if check is not None and not check.all_finite:
                return boundary, check
        return None


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy, torch, or lietorch-like tensor data without copying semantics."""
    current = value
    if hasattr(current, "data") and not isinstance(current, np.ndarray):
        current = current.data
    for name in ("detach", "cpu"):
        method = getattr(current, name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current)


def _payload_from_mapping(mapping: Mapping[str, Any]) -> StreamPayload:
    required = ("payload_id", "source_frame_index", "timestamp_s", "rgb_path", "mask_path", "rgb_sha256", "mask_sha256")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"payload manifest entry lacks {missing}")
    return StreamPayload(
        payload_id=str(mapping["payload_id"]),
        source_frame_index=int(mapping["source_frame_index"]),
        timestamp_s=float(mapping["timestamp_s"]),
        rgb_path=Path(str(mapping["rgb_path"])),
        mask_path=Path(str(mapping["mask_path"])),
        rgb_sha256=str(mapping["rgb_sha256"]),
        mask_sha256=str(mapping["mask_sha256"]),
    )


def load_exact_d1_stream(manifest_path: Path) -> tuple[StreamPayload, ...]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("payloads") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("d1h4 payload manifest lacks a payloads list")
    selected = tuple(_payload_from_mapping(item) for item in entries[:12] if isinstance(item, Mapping))
    expected_frames = tuple(range(60, 94, 3))
    if len(selected) != 12 or tuple(p.source_frame_index for p in selected) != expected_frames:
        raise ValueError("probe requires exact d1h4 payloads 0000--0011 / source frames 60--93")
    if tuple(round(p.timestamp_s, 1) for p in selected) != tuple(round(2.0 + 0.1 * i, 1) for i in range(12)):
        raise ValueError("probe requires exact d1h4 timestamps 2.0--3.1 seconds")
    for payload in selected:
        for path, digest in ((payload.rgb_path, payload.rgb_sha256), (payload.mask_path, payload.mask_sha256)):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"payload hash mismatch for {path}: {actual} != {digest}")
    return selected


def exact_camera() -> DroidCamera:
    """Return the V22-validated DROID model-grid camera, never the old 408.96 K."""
    return DroidCamera(
        intrinsics=_MODEL_GRID_INTRINSICS,
        source_size=_MODEL_SIZE,
        pixel_transform=PixelTransform.identity(),
    )


def exact_options(profile: str) -> DroidSessionOptions:
    try:
        return OPTION_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown DROID option profile {profile!r}; expected {sorted(OPTION_PROFILES)}") from error


def journal_capacity(options: DroidSessionOptions) -> int:
    """Keep the bounded idempotency journal valid for the profile's declared buffer."""
    return options.buffer + 1


def _array_stats(values: Any) -> dict[str, Any]:
    """Finite/numeric summary without presenting a tensor dump as an explanation."""
    array = _to_numpy(values).astype(np.float64, copy=False)
    finite = np.isfinite(array)
    count = int(array.size)
    finite_values = array[finite]
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "value_count": count,
        "finite_count": int(finite.sum()),
        "finite_ratio": float(finite.mean()) if count else 0.0,
    }
    if finite_values.size:
        summary.update({
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "median": float(np.median(finite_values)),
        })
    return summary


def model_grid_calibration_comparison(camera: DroidCamera, image_shape: DroidImageShape) -> dict[str, Any]:
    """Attest the actual V19→V22 DROID resize contract used by this replay."""
    expected_model_px = np.asarray(_MODEL_GRID_INTRINSICS, dtype=np.float64)
    passed_model_px = np.asarray(camera.intrinsics, dtype=np.float64)
    if image_shape.shape_hwc[:2] != (_MODEL_SIZE.height, _MODEL_SIZE.width):
        raise ValueError(f"probe requires model grid 320x568, got {image_shape.height}x{image_shape.width}")
    ratio = passed_model_px / expected_model_px
    return {
        "source": "v19_droid_calibration_contract_used_by_v22_trajectory",
        "source_focal_px": _SOURCE_FOCAL_1920,
        "source_size_px": {"width": _SOURCE_SIZE.width, "height": _SOURCE_SIZE.height},
        "model_size_px": {"width": image_shape.width, "height": image_shape.height},
        "resize_scale_source_to_model": {"x": image_shape.width / _SOURCE_SIZE.width, "y": image_shape.height / _SOURCE_SIZE.height},
        "expected_model_intrinsics_px": {"fx": float(expected_model_px[0]), "fy": float(expected_model_px[1]), "cx": float(expected_model_px[2]), "cy": float(expected_model_px[3]), "units": "model_image_pixels"},
        "passed_session_intrinsics_px": {"fx": float(passed_model_px[0]), "fy": float(passed_model_px[1]), "cx": float(passed_model_px[2]), "cy": float(passed_model_px[3]), "units": "model_image_pixels"},
        "passed_over_expected_ratio": {"fx": float(ratio[0]), "fy": float(ratio[1]), "cx": float(ratio[2]), "cy": float(ratio[3])},
        "classification": "model_pixel_calibration_agreement" if np.allclose(ratio, 1.0) else "model_pixel_calibration_mismatch",
    }


def frontend_state_stats(state: Any, calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the finite frontend gauge immediately before the terminal backend begins."""
    n_key = int(state.video.counter.value)
    poses = _to_numpy(state.video.poses[:n_key])
    disps = _to_numpy(state.video.disps[:n_key])
    internal_intrinsics = _to_numpy(state.video.intrinsics[:n_key])
    translations = poses[:, :3] if poses.ndim == 2 else poses
    quaternions = poses[:, 3:7] if poses.ndim == 2 and poses.shape[1] >= 7 else np.empty((0,))
    translation_norm = np.linalg.norm(translations, axis=1) if translations.ndim == 2 else np.empty((0,))
    quaternion_norm = np.linalg.norm(quaternions, axis=1) if quaternions.ndim == 2 else np.empty((0,))
    finite_pose = bool(np.isfinite(poses).all())
    finite_disps = bool(np.isfinite(disps).all())
    positive_disparity_ratio = float((disps > 0).mean()) if disps.size else 0.0
    quaternion_unit = bool(quaternion_norm.size and np.isfinite(quaternion_norm).all() and np.max(np.abs(quaternion_norm - 1.0)) < 0.05)
    nontrivial_motion = bool(translation_norm.size and np.isfinite(translation_norm).all() and float(translation_norm.max()) > 1e-6)
    if finite_pose and finite_disps and positive_disparity_ratio == 1.0 and quaternion_unit and nontrivial_motion:
        gauge_assessment = "numerically_plausible_monocular_gauge; metric scale remains unobservable"
    else:
        gauge_assessment = "frontend gauge is not numerically plausible before terminal BA"
    return {
        "keyframe_count": n_key,
        "session_camera_intrinsics_px": calibration["passed_session_intrinsics_px"],
        "internal_video_intrinsics_grid_px": _array_stats(internal_intrinsics),
        "options": state.options.to_wire(),
        "poses": _array_stats(poses),
        "translation": {**_array_stats(translations), "norm": _array_stats(translation_norm)},
        "quaternion": {**_array_stats(quaternions), "norm": _array_stats(quaternion_norm), "approximately_unit": quaternion_unit},
        "disparities": {**_array_stats(disps), "positive_ratio": positive_disparity_ratio},
        "gauge_assessment": gauge_assessment,
    }


def first_nonfinite_ba_iteration(iterations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for observation in iterations:
        pose = observation.get("poses")
        if isinstance(pose, Mapping) and pose.get("finite_count") != pose.get("value_count"):
            return dict(observation)
    return None


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _source_payloads_for_state(state: Any, stream: Sequence[StreamPayload], *, dense: bool) -> tuple[StreamPayload, ...]:
    source = state.dense_source if dense else state.keyframe_source
    by_timestamp = {item.timestamp_s: item for item in stream}
    mapped: list[StreamPayload] = []
    for _frame_id, timestamp in source:
        item = by_timestamp.get(float(timestamp))
        if item is not None:
            mapped.append(item)
    return tuple(mapped)


def _payload_array(payload: TensorPayload) -> np.ndarray:
    return np.frombuffer(bytes(payload.data), dtype=np.dtype(payload.dtype)).reshape(payload.shape)


def _replace_callable(target: Any, attribute: str, replacement: Any) -> Callable[[], None]:
    previous = getattr(target, attribute)
    setattr(target, attribute, replacement)

    def restore() -> None:
        setattr(target, attribute, previous)

    return restore


def _instrument_terminal_boundaries(
    adapter: DroidAdapter,
    state: Any,
    stream: Sequence[StreamPayload],
    recorder: ProbeRecorder,
    ba_observations: list[dict[str, Any]],
) -> Callable[[], None]:
    """Observe adapter call boundaries by replacement, leaving adapter source untouched.

    The recovered backend calls ``video.ba`` exactly once per ``update_lowmem``
    iteration.  Wrapping that instance callable therefore reports the first
    post-BA pose state without copying or editing DROID's iteration loop.
    """
    del adapter  # The fresh state owns the wrapped callables; adapter source is untouched.
    restores: list[Callable[[], None]] = []
    original_backend = state.backend
    original_ba = state.video.ba
    active_backend_steps: int | None = None
    active_iteration = 0

    def checked_ba(*args: Any, **kwargs: Any) -> Any:
        nonlocal active_iteration
        result = original_ba(*args, **kwargs)
        if active_backend_steps is not None:
            active_iteration += 1
            n_key = int(state.video.counter.value)
            stage = f"backend_{active_backend_steps}_iteration_{active_iteration:02d}"
            recorder.check(stage, state.video.poses[:n_key], _source_payloads_for_state(state, stream, dense=False))
            ba_observations.append({
                "backend_call_steps": active_backend_steps,
                "iteration": active_iteration,
                "poses": _array_stats(state.video.poses[:n_key]),
            })
        return result

    restores.append(_replace_callable(state.video, "ba", checked_ba))

    def checked_backend(iterations: int, *args: Any, **kwargs: Any) -> Any:
        nonlocal active_backend_steps, active_iteration
        active_backend_steps, active_iteration = iterations, 0
        try:
            result = original_backend(iterations, *args, **kwargs)
        finally:
            active_backend_steps = None
        if iterations in (7, 12):
            n_key = int(state.video.counter.value)
            recorder.check(
                f"backend_{iterations}",
                state.video.poses[:n_key],
                _source_payloads_for_state(state, stream, dense=False),
            )
        return result

    restores.append(_replace_callable(state, "backend", checked_backend))
    original_filler = state.filler

    def checked_filler(*args: Any, **kwargs: Any) -> Any:
        trajectory = original_filler(*args, **kwargs)
        recorder.check("filler_trajectory", trajectory.data, _source_payloads_for_state(state, stream, dense=True))
        return trajectory

    restores.append(_replace_callable(state, "filler", checked_filler))

    # The helper and CameraState are globals resolved at runtime in droid.py.
    import ego_annotation.serving.droid as droid_module

    original_conversion = droid_module.camera_from_world_xyzw_to_world_camera_matrix
    conversion_calls = 0

    def checked_conversion(poses: Any) -> Any:
        nonlocal conversion_calls
        converted = original_conversion(poses)
        conversion_calls += 1
        if conversion_calls == 1:  # _finalize_session_locked converts dense before keyframes.
            recorder.check("pose_conversion", converted, _source_payloads_for_state(state, stream, dense=True))
        return converted

    restores.append(_replace_callable(droid_module, "camera_from_world_xyzw_to_world_camera_matrix", checked_conversion))
    original_camera_state = droid_module.CameraState

    def checked_camera_state(*args: Any, **kwargs: Any) -> CameraState:
        payload = kwargs.get("T_world_camera")
        mappings = _source_payloads_for_state(state, stream, dense=True)
        try:
            value = original_camera_state(*args, **kwargs)
        except ContractValidationError:
            if payload is not None:
                recorder.check("camera_state_validation", _payload_array(payload), mappings)
            raise
        if payload is not None:
            recorder.check("camera_state_validation", _payload_array(payload), mappings)
        return value

    restores.append(_replace_callable(droid_module, "CameraState", checked_camera_state))

    def restore_all() -> None:
        for restore in reversed(restores):
            restore()

    return restore_all


def _boundary_fingerprint(checks: Mapping[str, FiniteCheck]) -> dict[str, Any]:
    return {
        stage: check.to_wire()
        for stage, check in sorted(checks.items())
        if stage in {
            "frontend_pre_backend_poses", "frontend_pre_backend_disparities",
            "backend_7", "backend_12", "filler_trajectory", "pose_conversion", "camera_state_validation",
        }
    }


def _verdict(
    recorder: ProbeRecorder,
    *,
    pre_backend: Mapping[str, Any] | None = None,
    ba_iterations: Sequence[Mapping[str, Any]] = (),
    exception: Exception | None = None,
) -> dict[str, Any]:
    first = recorder.first_nonfinite()
    if first is None:
        boundary = "none/finite-success"
        first_bad = None
    else:
        boundary, check = first
        first_bad = asdict(check.first_bad) if check.first_bad else None
    return {
        "boundary": boundary,
        "first_bad": first_bad,
        "finite_counts": _boundary_fingerprint(recorder.checks),
        "pre_backend": dict(pre_backend) if pre_backend is not None else None,
        "ba_iterations": [dict(row) for row in ba_iterations],
        "first_nonfinite_ba_iteration": first_nonfinite_ba_iteration(ba_iterations),
        "finalize_exception": repr(exception) if exception else None,
    }


def run_once(
    *,
    stream: Sequence[StreamPayload],
    weights: str,
    source_release: str,
    source_digest: str,
    source_amendment: str,
    device: str,
    seed: int,
    options_profile: str,
    adapter_factory: Callable[..., DroidAdapter] = DroidAdapter,
) -> dict[str, Any]:
    """Drive a fresh adapter through the full direct-adapter lifecycle exactly once."""
    seed_everything(seed)
    options = exact_options(options_profile)
    config = build_droid_model_config(
        weights=weights,
        model_revision="droid-v1",
        device=device,
        assigned_gpu=7,
        replica_id="droid-terminal-validity-probe-gpu7",
        droid_source_release_path=source_release,
        droid_source_digest=source_digest,
        droid_source_amendment_id=source_amendment,
        max_sessions=1,
        max_queued_frames_per_session=1,
        max_fnet_batch_size=1,
        max_result_journal_entries_per_session=journal_capacity(options),
    )
    adapter = adapter_factory(config)
    create = adapter.create_session(DroidCreateSessionRequest(
        ownership=Ownership("terminal-probe-create", "droid-terminal-probe", "d1h4-0.5", "droid.create_session", "d1h4"),
        camera=exact_camera(), image_shape=DroidImageShape(height=320, width=568),
        options=options, model_revision="droid-v1",
    ))
    if create.session_id is None:
        raise RuntimeError(f"probe create failed: {create.error}")
    session_id = create.session_id
    for index, payload in enumerate(stream):
        request = DroidFrameRequest(
            ownership=Ownership(
                request_id=f"terminal-probe-push-{index:04d}", job_id="droid-terminal-probe",
                item_id=payload.payload_id, stage_id="droid.push_frame",
                source_id=f"d1h4:{payload.source_frame_index}", source_timestamp_s=payload.timestamp_s,
            ),
            session_id=session_id, frame_id=payload.frame_id, source_timestamp_s=payload.timestamp_s,
            rgb=TensorPayload(payload.rgb_path.read_bytes(), (320, 568, 3), "uint8"),
            static_confidence_mask=TensorPayload(payload.mask_path.read_bytes(), (320, 568), "float32"),
            model_revision="droid-v1",
        )
        prepared = adapter.admit_frame(request)
        if not hasattr(prepared, "request"):
            raise RuntimeError(f"probe admission returned terminal response at {payload.payload_id}: {prepared}")
        response = asyncio.run(adapter.push_frame_batch([prepared]))[0]  # type: ignore[arg-type]
        if response.error is not None:
            raise RuntimeError(f"probe push failed at {payload.payload_id}: {response.error}")
    state = adapter._sessions[session_id]  # direct-adapter probe intentionally owns this fresh session.
    calibration = model_grid_calibration_comparison(state.camera, state.image_shape)
    pre_backend = frontend_state_stats(state, calibration)
    recorder = ProbeRecorder()
    keyframe_payloads = _source_payloads_for_state(state, stream, dense=False)
    n_key = int(state.video.counter.value)
    recorder.check("frontend_pre_backend_poses", state.video.poses[:n_key], keyframe_payloads)
    recorder.check("frontend_pre_backend_disparities", state.video.disps[:n_key], keyframe_payloads)
    ba_iterations: list[dict[str, Any]] = []
    restore = _instrument_terminal_boundaries(adapter, state, stream, recorder, ba_iterations)
    exception: Exception | None = None
    try:
        response = asyncio.run(adapter.finalize(DroidFinalizeRequest(
            ownership=Ownership("terminal-probe-finalize", "droid-terminal-probe", "d1h4-0.5", "droid.finalize", "d1h4"),
            session_id=session_id, model_revision="droid-v1",
        )))
        if response.camera_state is None and response.error is not None:
            exception = RuntimeError(response.error.message)
    except Exception as exc:  # CameraState can reject by design; the recorder identifies its boundary.
        exception = exc
    finally:
        restore()
    return _verdict(
        recorder,
        pre_backend={"calibration_comparison": calibration, "frontend_state": pre_backend},
        ba_iterations=ba_iterations,
        exception=exception,
    )


def combine_reruns(primary: Mapping[str, Any], rerun: Mapping[str, Any]) -> dict[str, Any]:
    """Classify only stable boundary evidence as a causal verdict."""
    # Determinism for this experiment is a stable validity boundary on identical
    # bytes/K/seed. Detailed floating-point statistics may vary slightly across
    # CUDA kernels without changing any finite/non-finite conclusion.
    stable = {
        "boundary": primary.get("boundary"),
        "first_bad": primary.get("first_bad"),
        "finite_counts": primary.get("finite_counts"),
    } == {
        "boundary": rerun.get("boundary"),
        "first_bad": rerun.get("first_bad"),
        "finite_counts": rerun.get("finite_counts"),
    }
    verdict = dict(primary)
    verdict["determinism_rerun"] = dict(rerun)
    verdict["deterministic"] = stable
    if not stable:
        verdict["boundary"] = "nondeterministic"
    return verdict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--droid-source-release", required=True)
    parser.add_argument("--droid-source-digest", required=True)
    parser.add_argument("--droid-source-amendment", default="recovered-hawor-droid-core-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--options-profile", choices=sorted(OPTION_PROFILES), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stream = load_exact_d1_stream(args.manifest)
    primary = run_once(
        stream=stream, weights=args.weights, source_release=args.droid_source_release,
        source_digest=args.droid_source_digest, source_amendment=args.droid_source_amendment,
        device=args.device, seed=args.seed, options_profile=args.options_profile,
    )
    rerun = run_once(
        stream=stream, weights=args.weights, source_release=args.droid_source_release,
        source_digest=args.droid_source_digest, source_amendment=args.droid_source_amendment,
        device=args.device, seed=args.seed, options_profile=args.options_profile,
    )
    verdict = combine_reruns(primary, rerun)
    verdict.update({
        "schema": "ego.droid-terminal-validity-probe.v1",
        "stream": [{"payload_id": item.payload_id, "source_frame_index": item.source_frame_index,
                    "source_timestamp_s": item.timestamp_s} for item in stream],
        "seed": args.seed,
        "device": args.device,
        "options_profile": args.options_profile,
        "options": exact_options(args.options_profile).to_wire(),
        "camera": model_grid_calibration_comparison(exact_camera(), DroidImageShape(height=320, width=568)),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["boundary"] != "nondeterministic" else 2


if __name__ == "__main__":
    raise SystemExit(main())
