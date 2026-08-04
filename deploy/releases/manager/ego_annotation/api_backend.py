"""Real multipart HTTP backend for the frozen 3572551 service routes.

The live path uses only route-specific metadata/part adapters from
``live_wire.py``. The internal generic envelope codec is intentionally not
referenced here. HTTP 200 is not success until the route result/error and exact
ownership have been decoded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urljoin, urlsplit

import numpy as np

from ego_annotation.api_routes import RouteSpec, route_for
from ego_annotation.live_wire import LiveRouteResponse, LiveWireError, decode_live_response, encode_live_request
from ego_annotation.multipart import DecodedPart, decode_raw_multipart
from ego_annotation.scripted.contracts import AlgorithmRequest, AlgorithmResult, ClientRequestTiming, NativeBatchTrace
from ego_annotation.typed_contracts import (
    DroidCapabilities,
    DroidCreateInput,
    DroidCreateOutput,
    DroidFinalizeInput,
    DroidFinalizeOutput,
    DroidPushInput,
    DroidPushOutput,
    HandDetections,
    HandsOutput,
    HaworTrackOutput,
    InfillerOutput,
    ManoBatch,
    CosmosOutput,
    STAGE_INPUT_TYPES,
    TypedContractError,
    TypedTensor,
    UniDepthOutput,
    WiLoROutput,
)


class ApiBackendError(RuntimeError):
    """Base class for explicit remote backend failures."""


class ApiTransportError(ApiBackendError):
    """HTTP connection or non-success response failure."""


class ApiProtocolError(ApiBackendError):
    """Malformed or identity-changing route response."""


class CapabilityMismatchError(ApiProtocolError):
    """Frozen DROID contract cannot prove required RGB-D consumption."""


class CosmosDisabledError(ApiBackendError):
    """Cosmos is explicitly unavailable and cannot silently fabricate semantics."""


class RemoteDroidFailure(ApiBackendError):
    """A classified DROID failure preserved for a later recovery policy."""


def _wilor_full_camera_source_projection(
    points: np.ndarray,
    cam_t_full: np.ndarray,
    focal_length_px: float,
    source_size_px: tuple[int, int],
) -> np.ndarray:
    """Project root-relative MANO geometry with WiLoR's returned full camera."""

    geometry = np.asarray(points, dtype=np.float64)
    camera = np.asarray(cam_t_full, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] != 3 or camera.shape != (3,):
        raise ApiProtocolError("WiLoR full-camera projection received invalid geometry or translation shape")
    width, height = (int(source_size_px[0]), int(source_size_px[1]))
    focal = float(focal_length_px)
    if width <= 0 or height <= 0 or not np.isfinite(focal) or focal <= 0.0:
        raise ApiProtocolError("WiLoR full-camera projection requires positive source geometry and focal length")
    if not np.isfinite(geometry).all() or not np.isfinite(camera).all():
        raise ApiProtocolError("WiLoR full-camera projection requires finite geometry and cam_t_full")
    translated = geometry + camera[None, :]
    if np.any(translated[:, 2] <= 1e-8):
        raise ApiProtocolError("WiLoR full-camera projection produced non-positive depth")
    principal = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    projected = focal * translated[:, :2] / translated[:, 2:3] + principal[None, :]
    return np.ascontiguousarray(projected.astype(np.float32))


@dataclass(frozen=True)
class DroidRecoveryPolicy:
    """Reserved remote recovery branches; no guessed service parameters."""

    oom_filter_thresh: float | None = None
    exclusive_fixed_release: str | None = None
    keyframe_retry_filter_thresh: float | None = None
    max_keyframe_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_keyframe_retries != 1:
            raise ValueError("DROID keyframe recovery is exactly one bounded retry")
        if self.oom_filter_thresh is not None and self.oom_filter_thresh <= 0:
            raise ValueError("OOM filter_thresh must be positive")
        if self.keyframe_retry_filter_thresh is not None and self.keyframe_retry_filter_thresh <= 0:
            raise ValueError("keyframe retry filter_thresh must be positive")

    def on_oom(self) -> dict[str, object]:
        return {"classification": "remote_droid_oom", "filter_thresh": self.oom_filter_thresh, "exclusive_fixed_release": self.exclusive_fixed_release, "automatic_retry": False}

    def on_finalize_keyframes(self, keyframe_count: int, retries_used: int = 0) -> dict[str, object]:
        if keyframe_count > 1:
            return {"classification": "accepted", "automatic_retry": False}
        if retries_used == 0 and self.keyframe_retry_filter_thresh is not None:
            return {"classification": "remote_droid_insufficient_keyframes", "action": "retry_lower_filter_thresh", "filter_thresh": self.keyframe_retry_filter_thresh, "automatic_retry": False, "max_retries": 1}
        return {"classification": "remote_droid_insufficient_keyframes", "action": "insufficient_trajectory", "preserve_sole_measured_pose": True, "skip_pairwise_ba_and_filler": True, "automatic_retry": False}


A800_SERVICE_PORTS: Mapping[str, int] = {
    "unidepth.infer": 28000,
    "hands.detect": 28001,
    "wilor.reconstruct": 28004,
    "droid.create_session": 28002,
    "droid.push_frame": 28002,
    "droid.finalize": 28002,
    "hawor.infer_tracks": 28003,
    "hawor_infiller.fill": 28003,
    "cosmos3.reason": 28006,
}


@dataclass(frozen=True)
class ApiBackendConfig:
    """Backend-private HTTP configuration, restricted to A800 localhost."""

    base_url: str
    timeout_s: float = 86400.0
    user_agent: str = "ego-annotation-api-backend/1"
    recovery: DroidRecoveryPolicy = field(default_factory=DroidRecoveryPolicy)
    require_a800_localhost: bool = True
    cosmos_enabled: bool = False
    service_origins: Mapping[str, str] = field(default_factory=dict)
    stage_capture_root: str | None = None
    stage_capture_limit: int = 1
    stage_capture_limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ApiBackend base_url must be an absolute http(s) origin")
        if self.require_a800_localhost and parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("ApiBackend requests must originate from A800 localhost")
        if parsed.path not in {"", "/"}:
            raise ValueError("ApiBackend base_url must be an origin without a private route path")
        if self.base_url.endswith("/"):
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if self.timeout_s <= 0:
            raise ValueError("ApiBackend timeout_s must be positive")
        if isinstance(self.stage_capture_limit, bool) or not isinstance(self.stage_capture_limit, int) or self.stage_capture_limit <= 0:
            raise ValueError("stage_capture_limit must be a positive integer")
        for stage_id, limit in self.stage_capture_limits.items():
            if stage_id not in {"hawor.infer_tracks", "hawor_infiller.fill", "cosmos3.reason", "droid.finalize"} or isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("stage_capture_limits must name explicitly capturable stages with positive integer limits")
        for stage_id, origin in self.service_origins.items():
            parsed_origin = urlsplit(origin)
            if stage_id not in A800_SERVICE_PORTS or parsed_origin.hostname not in {"127.0.0.1", "localhost"} or parsed_origin.path not in {"", "/"}:
                raise ValueError("service origins must use known A800 localhost origins")

    def origin_for(self, stage_id: str) -> str:
        if stage_id in self.service_origins:
            return self.service_origins[stage_id].rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.port is not None and parsed.port != A800_SERVICE_PORTS[stage_id]:
            return self.base_url
        return f"{parsed.scheme}://{parsed.hostname}:{A800_SERVICE_PORTS[stage_id]}"

    @classmethod
    def for_stage(cls, stage_id: str, *, host: str = "127.0.0.1", **kwargs: object) -> "ApiBackendConfig":
        if stage_id not in A800_SERVICE_PORTS:
            raise ValueError(f"no A800 service port for {stage_id!r}")
        return cls(base_url=f"http://{host}:{A800_SERVICE_PORTS[stage_id]}", **kwargs)


class HttpOpener(Protocol):
    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        ...


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _strict_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiProtocolError(f"Cosmos {field_name} must be non-empty text")
    return value


def _strict_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ApiProtocolError(f"Cosmos {field_name} must be a non-negative integer")
    return value


def _strict_nonnegative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ApiProtocolError(f"Cosmos {field_name} must be a finite non-negative number")
    return float(value)


def _strict_mapping(value: object, field_name: str) -> dict[str, object]:
    if type(value) is not dict or not value or any(not isinstance(key, str) or not key for key in value):
        raise ApiProtocolError(f"Cosmos {field_name} must be a non-empty object with text keys")
    return dict(value)


class ApiBackend:
    """Execute one typed request against its fixed frozen service route."""

    def __init__(self, config: ApiBackendConfig, *, opener: HttpOpener | None = None) -> None:
        self.config = config
        self._opener = opener or _default_opener
        self._sessions: dict[str, dict[str, Any]] = {}
        self._capture_lock = threading.Lock()
        self._capture_counts: dict[str, int] = {}
        self._service_batch_trace_lock = threading.Lock()
        self._service_batch_traces: list[dict[str, object]] = []

    @property
    def service_batch_traces(self) -> tuple[dict[str, object], ...]:
        """Thread-safe snapshots of actual service traces from decoded responses."""
        with self._service_batch_trace_lock:
            return tuple({**record, "trace": dict(record["trace"])} for record in self._service_batch_traces)

    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        result, _timing = self.execute_timed(request)
        return result

    def execute_timed(self, request: AlgorithmRequest[Any]) -> tuple[AlgorithmResult[Any], ClientRequestTiming]:
        """Execute without shared timing state; return one immutable timing record.

        ``transport_wait_s`` is deliberately measured around the complete opener
        call, including upload, manager queueing, server compute, and response
        bytes. It is never presented as server-only compute.
        """
        total_started = time.monotonic()
        prepare_started = total_started
        route = route_for(request.algorithm_id)
        if request.algorithm_id == "cosmos3.reason" and not self.config.cosmos_enabled:
            raise CosmosDisabledError("semantic lane is explicitly disabled; no Cosmos fallback or fabricated rows")
        self._validate_request(request, route)
        body, content_type, metadata, part_names = encode_live_request(request)
        prepare_s = time.monotonic() - prepare_started
        transport_started = time.monotonic()
        response_body, response_type = self._post(route, body, content_type, request=request)
        transport_s = time.monotonic() - transport_started
        decode_started = time.monotonic()
        try:
            decoded = decode_live_response(request.algorithm_id, getattr(request.input, "ownership"), response_body, response_type)
        except LiveWireError as exc:
            raise ApiProtocolError(str(exc)) from exc
        self._record_service_batch_trace(request, decoded)
        output = self._route_output(request, decoded)
        self._validate_droid_capability(request, output)
        self._capture_stage_exchange(request, body, content_type, metadata, part_names, response_body, response_type, decoded)
        self._record_droid_lifecycle(request, output)
        result = AlgorithmResult.from_request(request, output=output, native_batch_trace=NativeBatchTrace.from_work(request.work))
        decode_s = time.monotonic() - decode_started
        timing = ClientRequestTiming(prepare_s, transport_s, decode_s, time.monotonic() - total_started)
        return result, timing

    def _record_service_batch_trace(self, request: AlgorithmRequest[Any], decoded: LiveRouteResponse) -> None:
        """Keep only service-returned complete batch traces; never infer one locally."""
        candidates = (
            ("route_metadata.trace", decoded.route_metadata.get("trace")),
            ("result.trace", decoded.result.get("trace") if isinstance(decoded.result, Mapping) else None),
        )
        required = {"batch_id", "request_count", "effective_work_units", "replica_id"}
        for location, candidate in candidates:
            if not isinstance(candidate, Mapping) or not required.issubset(candidate):
                continue
            # Stateless services expose ``forward_count``. DROID's truthful
            # callback trace names the fused model count
            # ``fnet_forward_count`` and separately reports session-local work;
            # retain that original schema rather than fabricating a generic
            # count or dropping the callback entirely.
            if "forward_count" not in candidate and "fnet_forward_count" not in candidate:
                continue
            trace = dict(candidate)
            if not isinstance(trace["batch_id"], str) or not trace["batch_id"]:
                continue
            with self._service_batch_trace_lock:
                self._service_batch_traces.append(
                    {
                        "stage_id": request.algorithm_id,
                        "case_id": request.case_id,
                        "item_id": request.item_id,
                        "source_id": request.source_id,
                        "trace_location": location,
                        "trace": trace,
                    }
                )
            return

    def _capture_stage_exchange(
        self,
        request: AlgorithmRequest[Any],
        request_body: bytes,
        request_content_type: str,
        request_metadata: Mapping[str, object],
        request_parts: tuple[str, ...],
        response_body: bytes,
        response_content_type: str,
        decoded: LiveRouteResponse,
    ) -> None:
        default_allowed = {"hawor.infer_tracks", "hawor_infiller.fill", "cosmos3.reason"}
        explicit_droid_finalize = (
            request.algorithm_id == "droid.finalize"
            and request.algorithm_id in self.config.stage_capture_limits
        )
        if self.config.stage_capture_root is None or (request.algorithm_id not in default_allowed and not explicit_droid_finalize):
            return
        stage_name = request.algorithm_id.replace(".", "_")
        with self._capture_lock:
            limit = self.config.stage_capture_limits.get(request.algorithm_id, self.config.stage_capture_limit)
            if self._capture_counts.get(request.algorithm_id, 0) >= limit:
                return
            # Reserve the stage budget before any filesystem work so concurrent
            # HaWoR calls cannot all capture themselves as the representative.
            self._capture_counts[request.algorithm_id] = self._capture_counts.get(request.algorithm_id, 0) + 1
            try:
                root = Path(self.config.stage_capture_root)
                root.mkdir(parents=True, exist_ok=True)
                request_hash = hashlib.sha256(request_body).hexdigest()
                response_hash = hashlib.sha256(response_body).hexdigest()
                address = root / stage_name / request_hash / response_hash
                captured_request = decode_raw_multipart(request_body, request_content_type)
                if tuple(captured_request.parts) != request_parts:
                    raise ApiProtocolError("captured request multipart part order mismatch")
                manifest = {
                    "schema": "ego.annotation.stage_exchange.v1",
                    "algorithm_id": request.algorithm_id,
                    "model_revision": request.model_revision,
                    "ownership": {
                        "case_id": request.case_id,
                        "item_id": request.item_id,
                        "source_id": request.source_id,
                        "scope": getattr(request.input, "ownership").scope,
                    },
                    "source_timeline": request.timeline.to_mapping(),
                    "request": {
                        "content_type": request_content_type,
                        "size_bytes": len(request_body),
                        "sha256": request_hash,
                        "parts": {name: dict(part.descriptor) for name, part in captured_request.parts.items()},
                        "metadata": dict(request_metadata),
                    },
                    "response": {
                        "content_type": response_content_type,
                        "size_bytes": len(response_body),
                        "sha256": response_hash,
                        "metadata": dict(decoded.route_metadata),
                        "parts": {name: dict(part.descriptor) for name, part in decoded.parts.items()},
                    },
                }
                if address.exists():
                    if (address / "request.multipart").read_bytes() != request_body or (address / "response.multipart").read_bytes() != response_body:
                        raise ApiProtocolError("stage capture hash collision has non-identical bytes")
                else:
                    parent = address.parent
                    parent.mkdir(parents=True, exist_ok=True)
                    temporary = Path(tempfile.mkdtemp(prefix=f".{response_hash}.", dir=parent))
                    try:
                        self._write_capture_file(temporary / "request.multipart", request_body)
                        self._write_capture_file(temporary / "response.multipart", response_body)
                        self._write_capture_file(temporary / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8"))
                        directory_fd = os.open(temporary, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                        os.replace(temporary, address)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                self._write_capture_index(root)
            except Exception:
                self._capture_counts[request.algorithm_id] -= 1
                raise

    @staticmethod
    def _write_capture_file(path: Path, data: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_capture_index(self, root: Path) -> None:
        entries: list[dict[str, object]] = []
        for manifest_path in sorted(root.glob("*/*/*/manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries.append({
                "algorithm_id": payload["algorithm_id"],
                "request_sha256": payload["request"]["sha256"],
                "response_sha256": payload["response"]["sha256"],
                "manifest": str(manifest_path.relative_to(root)),
            })
        data = json.dumps({"schema": "ego.annotation.stage_capture_index.v1", "entries": entries}, indent=2, ensure_ascii=True).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(prefix=".fixture_index.", dir=root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, root / "fixture_index.json")
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _validate_request(self, request: AlgorithmRequest[Any], route: RouteSpec) -> None:
        expected = STAGE_INPUT_TYPES.get(request.algorithm_id)
        if expected is None or not isinstance(request.input, expected):
            raise ApiProtocolError(f"{request.algorithm_id} requires its registered typed input")
        if request.work.native_batch_cap != route.native_batch_cap or request.work.outer_item_batch_size != 1:
            raise ApiProtocolError("request native batch cap/outer item cardinality disagrees with fixed route")
        ownership = getattr(request.input, "ownership", None)
        if ownership is None or ownership.case_id != request.case_id or ownership.item_id != request.item_id or ownership.source_id != request.source_id:
            raise ApiProtocolError("typed input ownership does not match AlgorithmRequest identity")
        if self._strict_droid(request):
            try:
                DroidCapabilities.frozen_3572551().require_rgbd()
            except TypedContractError as exc:
                raise CapabilityMismatchError("remote_droid_capability_mismatch: " + str(exc)) from exc
        if isinstance(request.input, DroidPushInput):
            session = self._sessions.get(request.input.session_id)
            if session is None:
                raise ApiProtocolError("DROID push arrived before create_session")
            if request.input.frame_index <= session["last_frame_index"]:
                raise ApiProtocolError(f"DROID push frame order violation: frame {request.input.frame_index} is not after {session['last_frame_index']}")
        if isinstance(request.input, DroidFinalizeInput):
            session = self._sessions.get(request.input.session_id)
            if session is None or session["pushed_frames"] <= 0:
                raise ApiProtocolError("DROID finalize requires a created session with pushed frames")

    @staticmethod
    def _strict_droid(request: AlgorithmRequest[Any]) -> bool:
        return isinstance(request.input, (DroidCreateInput, DroidPushInput, DroidFinalizeInput)) and bool(getattr(request.input, "require_rgbd_capability", True))

    def _post(self, route: RouteSpec, body: bytes, content_type: str, *, request: AlgorithmRequest[Any]) -> tuple[bytes, str]:
        url = urljoin(self.config.origin_for(route.stage_id) + "/", route.path.lstrip("/"))
        headers = {
            "Content-Type": content_type,
            "Accept": "multipart/form-data, application/json",
            "User-Agent": self.config.user_agent,
            "X-Ego-Video-Job-Id": request.case_id,
            "X-Ego-Video-Item-Id": request.item_id,
            "X-Ego-Stage-Id": request.algorithm_id,
        }
        http_request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        # Capacity retries live in the local admission queue. This call waits
        # for one queued service answer and does not add another retry layer.
        backoff_s: tuple[int, ...] = ()
        for retry_index in range(len(backoff_s) + 1):
            try:
                response = self._opener(http_request, self.config.timeout_s)
                status = int(getattr(response, "status", response.getcode() if hasattr(response, "getcode") else 0))
                response_type = str(response.headers.get("Content-Type", ""))
                response_body = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                retry_complete = str(exc.headers.get("X-Ego-Admission-Retry-Complete", "")).strip() == "1"
                if exc.code == 429 and not retry_complete and retry_index < len(backoff_s):
                    time.sleep(backoff_s[retry_index])
                    continue
                # DROID finalize model failures carry the numeric failure family
                # after the ownership envelope. Preserve enough response text for
                # the client to discriminate a nonfinite trajectory and perform
                # its one bounded fresh-session replay.
                detail_limit = 4000 if route.stage_id == "droid.finalize" else 500
                raise ApiTransportError(f"{route.stage_id} HTTP {exc.code}: {detail[:detail_limit]}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ApiTransportError(f"{route.stage_id} HTTP transport failed: {exc}") from exc
            if status < 200 or status >= 300:
                raise ApiTransportError(f"{route.stage_id} HTTP status {status}")
            return response_body, response_type
        raise AssertionError("HTTP retry loop exhausted without a response or transport failure")

    def _route_output(self, request: AlgorithmRequest[Any], decoded: LiveRouteResponse) -> object:
        result = decoded.result or {}
        if request.algorithm_id == "unidepth.infer":
            value = request.input
            depth = self._route_tensor(decoded, "depth_m", units="metres", frame="depth_grid", order="yx", tag="unidepth_metric_depth_v1", add_batch=True)
            K = self._route_tensor(decoded, "K_px", units="pixels", frame="depth_grid", order="yx", tag="unidepth_full_K_v1", add_batch=True)
            confidence = self._route_tensor(decoded, "confidence", units="probability", frame="depth_grid", order="yx", tag="unidepth_confidence_v1", add_batch=True)
            return UniDepthOutput(value.ownership, depth, K, confidence, value.frame_indices, value.timestamps_s, value.spatial, str(result.get("model_revision", request.model_revision)))
        if request.algorithm_id == "hands.detect":
            value = request.input
            # ``hands-yolo-v2`` is deliberately locator-only: do not require or
            # synthesize segmentation masks before the unchanged WiLoR/HaWoR paths.
            detections = HandDetections(
                self._route_tensor(decoded, "boxes", units="pixels", frame="source", order="tkf", tag="hand_boxes_v1", add_batch=True),
                self._route_tensor(decoded, "scores", units="probability", frame="source", order="tk", tag="hand_scores_v1", add_batch=True),
                self._route_tensor(decoded, "sides", units="class_id", frame="source", order="tk", tag="hand_sides_v1", add_batch=True),
                self._route_tensor(decoded, "visibility", units="fraction", frame="source", order="tk", tag="hand_visibility_v1", add_batch=True),
                self._route_tensor(decoded, "uncertainty", units="score", frame="source", order="tk", tag="hand_uncertainty_v1", add_batch=True),
            )
            return HandsOutput(value.ownership, detections, value.frame_indices, value.timestamps_s, value.spatial, str(result.get("model_revision", request.model_revision)))
        if request.algorithm_id == "wilor.reconstruct":
            value = request.input
            def batched(name: str, units: str, order: str, tag: str, *, already: bool = False) -> TypedTensor:
                return self._route_tensor(decoded, name, units=units, frame="source_camera", order=order, tag=tag, add_batch=not already)
            vertices = batched("vertices", "metres", "bvx", "mano_vertices_root_relative_v1")
            joints = batched("joints", "metres", "bjx", "mano_joints_root_relative_v1")
            cam_t_full = batched("cam_t_full", "virtual_camera", "bx", "wilor_full_camera_translation_v1")
            pred_cam = batched("pred_cam", "weak_perspective", "bx", "wilor_crop_weak_camera_v1")
            returned_surface = batched("keypoints_2d", "pixels", "bvu", "wilor_returned_surface_keypoints_source_px_v1")
            mano_metadata = result.get("mano")
            if not isinstance(mano_metadata, Mapping):
                raise ApiProtocolError("WiLoR response lacks MANO full-camera metadata")
            try:
                focal_length_px = float(mano_metadata["focal_length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiProtocolError("WiLoR response lacks a numeric MANO focal_length") from exc
            if not np.isfinite(focal_length_px) or focal_length_px <= 0.0:
                raise ApiProtocolError("WiLoR MANO focal_length must be finite and positive")
            if vertices.shape[0] != len(value.crop_transforms) or joints.shape[0] != len(value.crop_transforms) or cam_t_full.shape != (len(value.crop_transforms), 3):
                raise ApiProtocolError("WiLoR full-camera geometry batch disagrees with submitted crops")
            if returned_surface.shape != vertices.shape[:-1] + (2,):
                raise ApiProtocolError("WiLoR returned keypoints_2d must match the 778-vertex MANO surface axis")
            projected_vertices: list[np.ndarray] = []
            projected_joints: list[np.ndarray] = []
            maximum_keypoint_residual_px = 0.0
            for slot, transform in enumerate(value.crop_transforms):
                source_size = (transform.source_to_crop.source_width_px, transform.source_to_crop.source_height_px)
                projected_surface = _wilor_full_camera_source_projection(vertices.array[slot], cam_t_full.array[slot], focal_length_px, source_size)
                returned_surface_slot = np.asarray(returned_surface.array[slot], dtype=np.float32)
                if not np.isfinite(returned_surface_slot).all():
                    raise ApiProtocolError("WiLoR returned keypoints_2d must be finite source pixels")
                residual = float(np.max(np.abs(projected_surface - returned_surface_slot))) if returned_surface_slot.size else 0.0
                maximum_keypoint_residual_px = max(maximum_keypoint_residual_px, residual)
                if not np.allclose(projected_surface, returned_surface_slot, atol=1.0e-3, rtol=1.0e-4):
                    raise ApiProtocolError(f"WiLoR returned keypoints_2d disagree with full-camera surface projection by {residual:.6g}px")
                projected_vertices.append(returned_surface_slot)
                projected_joints.append(_wilor_full_camera_source_projection(joints.array[slot], cam_t_full.array[slot], focal_length_px, source_size))
            projection_provenance = {
                "projection": "wilor_returned_full_camera_focal_cam_t_source_image_center",
                "focal_length_px": focal_length_px,
                "cam_t_full_digest": cam_t_full.canonical_tensor_digest,
                "returned_surface_keypoints_validated": True,
                "maximum_surface_keypoint_residual_px": maximum_keypoint_residual_px,
                "metric_world_lift": "unchanged_unidepth_wrist_ray_lift",
            }
            mano = ManoBatch(
                batched("global_orient", "rotation", "bxy", "mano_global_orient_v1", already=True),
                batched("hand_pose", "rotation", "bjxy", "mano_hand_pose_v1"),
                batched("betas", "shape", "bd", "mano_betas_v1"),
                vertices,
                joints,
                cam_t_full,
                pred_cam,
                TypedTensor(np.stack(projected_vertices), "pixels", "source_pixels", "bvu", "wilor_returned_surface_keypoints_source_px_v1", projection_provenance),
                TypedTensor(np.stack(projected_joints), "pixels", "source_pixels", "bju", "wilor_full_camera_projected_joints_source_px_v1", projection_provenance),
                batched("confidence", "probability", "b", "mano_confidence_v1", already=True),
                batched("uncertainty", "score", "b", "mano_uncertainty_v1", already=True),
            )
            handedness = tuple(transform.side for transform in value.crop_transforms)
            return WiLoROutput(value.ownership, handedness, mano, str(result.get("model_revision", request.model_revision)))
        if request.algorithm_id == "hawor.infer_tracks":
            value = request.input
            observed = self._route_tensor(decoded, "observed", units="boolean", frame="source", order="bt", tag="hawor_observed_v1")
            states = result.get("occlusion_state", ["unresolved"] * int(np.prod(observed.shape)))
            encoded = np.asarray([{"visible": 0, "partially_visible": 1, "occluded": 2, "out_of_frame": 3, "unresolved": 4}.get(str(item), 4) for item in states], dtype=np.uint8).reshape(observed.shape)
            occlusion = TypedTensor(encoded, units="state_id", coordinate_frame="source", tensor_index_order="bt", semantic_tag="hawor_occlusion_state_v1", provenance={"route": request.algorithm_id})
            return HaworTrackOutput(
                value.ownership,
                self._route_tensor(decoded, "root_orient", units="rotation", frame="camera", order="btxy", tag="hawor_root_orient_v1"),
                self._route_tensor(decoded, "hand_pose", units="rotation", frame="camera", order="btjxy", tag="hawor_hand_pose_v1"),
                self._route_tensor(decoded, "trans", units="metres", frame="camera", order="btx", tag="hawor_translation_v1"),
                self._route_tensor(decoded, "betas", units="shape", frame="mano", order="btd", tag="hawor_betas_v1"),
                self._route_tensor(decoded, "vertices", units="metres", frame="camera", order="btvx", tag="hawor_vertices_v1"),
                self._route_tensor(decoded, "joints", units="metres", frame="camera", order="btjx", tag="hawor_joints_v1"),
                observed, occlusion,
                self._route_tensor(decoded, "uncertainty", units="metres", frame="camera", order="bt", tag="hawor_uncertainty_v1"),
                str(result.get("model_revision", request.model_revision)),
            )
        if request.algorithm_id == "hawor_infiller.fill":
            value = request.input
            root = self._route_array(decoded, "root_orient")
            hand = self._route_array(decoded, "hand_pose")
            trans = self._route_array(decoded, "trans")
            betas = self._route_array(decoded, "betas")
            if root.shape[-2:] != (3, 3) or hand.shape[-2:] != (3, 3):
                raise ApiProtocolError("infiller rotation arrays must be matrices for 6D conversion")
            root6 = root[..., :2].reshape(*root.shape[:-2], 6)
            hand6 = hand[..., :2].reshape(*hand.shape[:-3], 15 * 6)
            state = np.concatenate((trans, betas, root6, hand6), axis=-1).astype(np.float32, copy=False)
            state_tensor = TypedTensor(state, units="coupled_mano_state", coordinate_frame="camera", tensor_index_order="htd", semantic_tag="infiller_two_hand_109d_v1", provenance={"route": request.algorithm_id})
            return InfillerOutput(value.ownership, state_tensor, self._route_tensor(decoded, "observed", units="boolean", frame="source", order="ht", tag="infiller_observed_v1"), self._route_tensor(decoded, "inferred", units="boolean", frame="source", order="ht", tag="infiller_inferred_v1"), self._route_tensor(decoded, "uncertainty", units="metres", frame="camera", order="ht", tag="infiller_uncertainty_v1"), str(result.get("model_revision", request.model_revision)))
        if request.algorithm_id == "cosmos3.reason":
            value = request.input
            required = {
                "ownership", "text", "finish_reason", "stop_reason", "prompt_tokens", "completion_tokens",
                "total_tokens", "timings", "trace", "media_provenance", "model_revision",
            }
            if type(result) is not dict or set(result) != required:
                raise ApiProtocolError("Cosmos response result fields do not match the frozen schema")
            if type(result["ownership"]) is not dict or result["ownership"] != decoded.ownership:
                raise ApiProtocolError("Cosmos nested result ownership mismatch")
            text = _strict_text(result["text"], "text")
            finish_reason = _strict_text(result["finish_reason"], "finish_reason")
            stop_reason = result["stop_reason"]
            if stop_reason is not None and not isinstance(stop_reason, str):
                raise ApiProtocolError("Cosmos stop_reason must be text or null")
            prompt_tokens = _strict_nonnegative_int(result["prompt_tokens"], "prompt_tokens")
            completion_tokens = _strict_nonnegative_int(result["completion_tokens"], "completion_tokens")
            total_tokens = _strict_nonnegative_int(result["total_tokens"], "total_tokens")
            timings = _strict_mapping(result["timings"], "timings")
            trace = _strict_mapping(result["trace"], "trace")
            raw_provenance = result["media_provenance"]
            if type(raw_provenance) is not list or len(raw_provenance) != len(value.media):
                raise ApiProtocolError("Cosmos media_provenance cardinality mismatch")
            provenance: list[dict[str, object]] = []
            for index, (raw, media, source_index) in enumerate(zip(raw_provenance, value.media, value.source_frame_indices)):
                item = _strict_mapping(raw, f"media_provenance[{index}]")
                if set(item) != {"kind", "media_type", "source_index", "bytes"}:
                    raise ApiProtocolError("Cosmos media_provenance field set mismatch")
                if (
                    item["kind"] != "image"
                    or item["media_type"] != media.media_type
                    or type(item["source_index"]) is not int
                    or item["source_index"] != source_index
                    or type(item["bytes"]) is not int
                    or item["bytes"] != len(media.data)
                ):
                    raise ApiProtocolError("Cosmos media_provenance order/source/type/size mismatch")
                provenance.append(item)
            try:
                return CosmosOutput(
                    value.ownership,
                    text,
                    finish_reason,
                    stop_reason,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    {str(key): _strict_nonnegative_number(raw, f"timings.{key}") for key, raw in timings.items()},
                    trace,
                    tuple(provenance),
                    _strict_text(result["model_revision"], "model_revision"),
                )
            except TypedContractError as exc:
                raise ApiProtocolError(str(exc)) from exc
        if request.algorithm_id == "droid.create_session":
            session_id = result.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ApiProtocolError("DROID create response has no session_id")
            return DroidCreateOutput(request.input.ownership, session_id, DroidCapabilities.frozen_3572551())
        if request.algorithm_id == "droid.push_frame":
            status = result.get("status")
            if not isinstance(status, Mapping):
                raise ApiProtocolError("DROID push response has no route-specific status")
            return DroidPushOutput(request.input.ownership, request.input.session_id, request.input.frame_index, bool(status.get("validity", {}).get("admitted", False)), int(status.get("keyframe_count", 0)), DroidCapabilities.frozen_3572551())  # type: ignore[arg-type]
        if request.algorithm_id == "droid.finalize":
            return self._decode_droid_finalize(request, decoded)
        raise ApiProtocolError(f"{request.algorithm_id} route result adapter is not configured")

    @staticmethod
    def _route_array(decoded: LiveRouteResponse, name: str) -> np.ndarray:
        part = decoded.parts.get(name)
        if not isinstance(part, DecodedPart):
            raise ApiProtocolError(f"route response missing binary part {name}")
        descriptor = part.descriptor
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise ApiProtocolError(f"route response part {name} lacks shape/dtype")
        try:
            return np.frombuffer(part.data, dtype=np.dtype(dtype)).reshape(tuple(int(dim) for dim in shape)).copy()
        except (TypeError, ValueError) as exc:
            raise ApiProtocolError(f"route response part {name} bytes do not match shape/dtype") from exc

    @classmethod
    def _route_tensor(cls, decoded: LiveRouteResponse, name: str, *, units: str, frame: str, order: str, tag: str, add_batch: bool = False) -> TypedTensor:
        array = cls._route_array(decoded, name)
        if add_batch:
            array = array[None]
        return TypedTensor(array, units=units, coordinate_frame=frame, tensor_index_order=order, semantic_tag=tag, provenance={"route": decoded.route_metadata.get("model_revision", "frozen"), "field": name})

    def _validate_droid_capability(self, request: AlgorithmRequest[Any], output: object) -> None:
        if not isinstance(request.input, (DroidCreateInput, DroidPushInput, DroidFinalizeInput)):
            return
        capabilities = getattr(output, "capabilities", None)
        if not isinstance(capabilities, DroidCapabilities):
            raise CapabilityMismatchError("remote_droid_capability_mismatch: no contract-derived DROID profile")
        if self._strict_droid(request):
            try:
                capabilities.require_rgbd()
            except TypedContractError as exc:
                raise CapabilityMismatchError("remote_droid_capability_mismatch: " + str(exc)) from exc
        else:
            if not getattr(request.input, "allow_monocular_droid_smoke", False):
                raise CapabilityMismatchError("DROID monocular smoke requires explicit allow_monocular_droid_smoke=True")
            if capabilities.native_sensor_depth_consumed:
                raise CapabilityMismatchError("diagnostic monocular mode cannot claim native sensor depth")

    def _record_droid_lifecycle(self, request: AlgorithmRequest[Any], output: object) -> None:
        if isinstance(request.input, DroidCreateInput):
            self._sessions[output.session_id] = {"last_frame_index": -1, "pushed_frames": 0, "K_droid_input": request.input.K_droid_input}  # type: ignore[attr-defined]
        elif isinstance(request.input, DroidPushInput):
            session = self._sessions[request.input.session_id]
            session["last_frame_index"] = int(request.input.frame_index)
            session["pushed_frames"] += 1
        elif isinstance(request.input, DroidFinalizeInput):
            self._sessions.pop(request.input.session_id, None)

    def _decode_droid_finalize(self, request: AlgorithmRequest[Any], decoded: LiveRouteResponse) -> DroidFinalizeOutput:
        result = decoded.result or {}
        required = ("T_world_camera", "T_camera_world", "disparities")
        if any(name not in decoded.parts for name in required):
            raise ApiProtocolError("DROID finalize camera_state is missing named binary pose/disparity parts")
        response_mapping = result.get("keyframe_mapping") if isinstance(result, Mapping) else None
        dense_mapping = result.get("dense_mapping") if isinstance(result, Mapping) else None
        def tensor(name: str, *, units: str, frame: str, order: str, tag: str) -> TypedTensor:
            part = decoded.parts.get(name)
            if not isinstance(part, DecodedPart):
                raise ApiProtocolError(f"DROID finalize missing binary part {name}")
            descriptor = part.descriptor
            shape = descriptor.get("shape")
            dtype = descriptor.get("dtype")
            if not isinstance(shape, list) or not isinstance(dtype, str):
                raise ApiProtocolError(f"DROID finalize part {name} lacks shape/dtype")
            try:
                import numpy as np
                array = np.frombuffer(part.data, dtype=np.dtype(dtype)).reshape(tuple(int(dim) for dim in shape)).copy()
            except (TypeError, ValueError) as exc:
                raise ApiProtocolError(f"DROID finalize part {name} has invalid tensor bytes") from exc
            provenance = {"route": "droid.finalize", "field": name}
            if response_mapping is not None:
                provenance["keyframe_source_indices"] = response_mapping
            if dense_mapping is not None:
                provenance["dense_source_indices"] = dense_mapping
            return TypedTensor(array, units=units, coordinate_frame=frame, tensor_index_order=order, semantic_tag=tag, provenance=provenance)
        session = self._sessions.get(request.input.session_id, {})
        K = np.asarray(session.get("K_droid_input", (1.0, 1.0, 0.0, 0.0)), dtype=np.float32)
        intrinsics = TypedTensor(K, units="pixels", coordinate_frame="droid_input", tensor_index_order="four", semantic_tag="droid_full_K_v1", provenance={"route": "droid.finalize"})
        caps = DroidCapabilities.frozen_3572551()
        diagnostic = bool(getattr(request.input, "allow_monocular_droid_smoke", False))
        keyframes = int(result.get("keyframe_mapping", result.get("keyframe_count", 0)) if isinstance(result.get("keyframe_mapping", result.get("keyframe_count", 0)), int) else len(result.get("keyframe_mapping", ())))
        scale_mode = "up_to_scale_monocular" if diagnostic else "metric_rgbd_unidepth"
        scale_provenance = {
            "scale_source": "diagnostic_monocular_gauge" if diagnostic else "native_sensor_depth_plus_unidepth",
            "convention": "T_world_camera maps camera homogeneous points into world; inverse is T_camera_world",
            "route": "droid.finalize",
        }
        return DroidFinalizeOutput(request.input.ownership, request.input.session_id, tensor("T_world_camera", units="metres", frame="world_from_camera", order="tyx", tag="droid_T_world_camera_v1"), tensor("T_camera_world", units="metres", frame="camera_from_world", order="tyx", tag="droid_T_camera_world_v1"), intrinsics, tensor("disparities", units="inverse_metres", frame="droid_model", order="tyx", tag="droid_sensor_disparity_v1"), keyframes, scale_mode, caps, acceptance=not diagnostic, diagnostic_only=diagnostic, scale_provenance=scale_provenance)

    @staticmethod
    def classify_droid_failure(message: str, *, keyframe_count: int | None = None) -> str | None:
        lowered = message.lower()
        if "out of memory" in lowered or re.search(r"\boom\b", lowered):
            return "remote_droid_oom"
        if keyframe_count is not None and keyframe_count <= 1:
            return "remote_droid_insufficient_keyframes"
        if "keyframe" in lowered and ("insufficient" in lowered or "one or fewer" in lowered):
            return "remote_droid_insufficient_keyframes"
        return None


__all__ = ["A800_SERVICE_PORTS", "ApiBackend", "ApiBackendConfig", "ApiBackendError", "ApiProtocolError", "ApiTransportError", "CapabilityMismatchError", "CosmosDisabledError", "DroidRecoveryPolicy", "RemoteDroidFailure"]
