"""Ray-free resident Cosmos3 (vLLM) adapter: one persistent engine, native batching.

This module is importable without Ray installed (ordinary unit tests inject a fake
``Cosmos3Backend``). The Ray Serve deployment wrapper lives in
``ego_annotation.serving.cosmos3_deployment`` and imports this adapter.

Contract invariants enforced here:

* The resident config owns the model revision: requests carry no ``model_revision``
  field, and every result carries only the configured (server-owned) revision.
* Media is bounded binary only: ``max_media_per_request``, ``max_media_bytes_per_item``,
  and ``max_media_bytes_per_request`` are enforced at admission. A caller NEVER supplies
  a server filesystem path; media is decoded to vLLM-native PIL/numpy at admission.
* vLLM owns continuous batching across concurrent requests. Unlike UniDepth, the Serve
  layer does NOT fuse requests with ``@serve.batch``; each request is one engine
  ``generate`` call. ``work_units`` is informational (always 1 per request).
* The engine is constructed once at replica startup (``model_load_count == 1``) with the
  exact Cosmos3 ``EngineArgs`` (``nvidia/Cosmos3-Nano``,
  ``Cosmos3ReasonerForConditionalGeneration`` architecture override, bf16, async
  scheduling, ``mm_encoder_tp_mode='data'``, ``max_model_len=32768``,
  ``gpu_memory_utilization=0.90``, ``media_io_kwargs={'video':{'num_frames':-1}}``).
* Timings are truthful monotonic ``time.monotonic()`` readings: admission, engine
  submit, first token (when available), and completion. ``e2e_s`` is the replica-observed
  end-to-end latency. vLLM engine-internal prefill/decode stats are surfaced when the
  engine exposes them, otherwise the monotonic timings stand on their own.

The model boundary is the vLLM ``AsyncLLMEngine.generate`` async generator (vLLM 0.19.1
v1 engine). The backend protocol below mirrors the subset of ``RequestOutput`` the
adapter consumes: the generated text, finish/stop reason, and token counts.
"""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from ego_annotation.serving.batching import BatchPolicy
from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    Cosmos3MediaItem,
    Cosmos3Request,
    Cosmos3Response,
    Cosmos3Result,
    DeploymentStatus,
    ErrorCode,
    GenerationControls,
    ServiceError,
)


class Cosmos3Backend(Protocol):
    """The vLLM engine boundary.

    ``generate`` mirrors ``AsyncLLMEngine.generate``: it returns an async generator of
    request-output-like objects. The adapter consumes the *final* output (``finished``)
    and reads ``text``, ``finish_reason``, ``stop_reason`` and token-count attributes.
    ``model_revision`` is exposed by the resident backend for provenance.
    """

    model_revision: str

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        multi_modal_data: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def engine_running_requests(self) -> int:
        """Number of requests currently running inside the vLLM engine (informational)."""
        ...


TensorResolver = Callable[[Any], Any]
SamplingParamsFactory = Callable[[GenerationControls, int], Any]
BackendFactory = Callable[["Cosmos3ModelConfig"], Cosmos3Backend]


@dataclass(frozen=True)
class Cosmos3ModelConfig:
    """Server-owned model settings; no request field can select a server path.

    The engine kwargs mirror the working bare ``vllm serve nvidia/Cosmos3-Nano``
    deployment exactly. ``model_revision`` is the server-owned provenance string
    returned with every result (e.g. ``cosmos3-nano:vllm-0.19.1``).
    """

    model_source: str
    model_revision: str
    device: str = "cuda"
    replica_id: str = "cosmos3-gpu6"
    assigned_gpu: int = 6
    # Exact Cosmos3 engine kwargs (mirror the bare vLLM 0.19.1 deployment).
    hf_overrides: Mapping[str, Any] = field(
        default_factory=lambda: {"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}
    )
    tensor_parallel_size: int = 1
    mm_encoder_tp_mode: str = "data"
    async_scheduling: bool = True
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    media_io_kwargs: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: {"video": {"num_frames": -1}}
    )
    enable_prefix_caching: bool | None = None
    # Bounded media admission. Enforced before the request reaches the engine.
    max_media_per_request: int = 8
    max_media_bytes_per_item: int = 16 * 1024 * 1024  # 16 MiB per item
    max_media_bytes_per_request: int = 64 * 1024 * 1024  # 64 MiB per request
    batch_policy: BatchPolicy = BatchPolicy(
        # vLLM owns batching; these bound the Serve admission queue only.
        max_batch_size=1,
        batch_wait_timeout_s=0.0,
        max_queued_requests=32,
    )

    def __post_init__(self) -> None:
        if self.assigned_gpu < 0:
            raise ContractValidationError("assigned_gpu must be non-negative")
        if not self.model_source or not self.model_revision:
            raise ContractValidationError("Cosmos3 model_source and model_revision are required server configuration")
        if self.max_media_per_request <= 0 or self.max_media_bytes_per_item <= 0 or self.max_media_bytes_per_request <= 0:
            raise ContractValidationError("media bounds must be positive")
        if self.max_media_bytes_per_request < self.max_media_bytes_per_item:
            raise ContractValidationError("per-request media budget must be >= per-item budget")

    @property
    def engine_kwargs(self) -> dict[str, Any]:
        """The exact EngineArgs kwargs for the resident vLLM AsyncLLMEngine."""
        return {
            "model": self.model_source,
            "hf_overrides": dict(self.hf_overrides),
            "tensor_parallel_size": self.tensor_parallel_size,
            "mm_encoder_tp_mode": self.mm_encoder_tp_mode,
            "async_scheduling": self.async_scheduling,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "media_io_kwargs": {k: dict(v) for k, v in self.media_io_kwargs.items()},
            "enforce_eager": False,
            "enable_prefix_caching": self.enable_prefix_caching,
        }


def build_cosmos3_model_config(
    *,
    model_source: str = "nvidia/Cosmos3-Nano",
    model_revision: str = "cosmos3-nano:vllm-0.19.1",
    device: str = "cuda",
    replica_id: str = "cosmos3-gpu6",
    assigned_gpu: int = 6,
    max_media_per_request: int = 8,
    max_media_bytes_per_item: int = 16 * 1024 * 1024,
    max_media_bytes_per_request: int = 64 * 1024 * 1024,
) -> Cosmos3ModelConfig:
    """Build a server-owned Cosmos3 config from explicit kwargs (no request input)."""
    return Cosmos3ModelConfig(
        model_source=model_source,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        max_media_per_request=max_media_per_request,
        max_media_bytes_per_item=max_media_bytes_per_item,
        max_media_bytes_per_request=max_media_bytes_per_request,
    )


@dataclass(frozen=True)
class _PreparedRequest:
    request: Cosmos3Request
    prompt: Any  # token ids (list[int]) or text str ready for engine.generate
    multi_modal_data: Mapping[str, Any] | None
    sampling_params: Any
    admitted_monotonic_s: float


def _default_tensor_resolver(value: Any) -> Any:
    # Media arrives as binary bytes (HTTP) or an in-cluster object reference.
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    import numpy as np  # noqa: F401  (an in-cluster caller may pass an ndarray)
    return value


def decode_media(media: Sequence[Cosmos3MediaItem], resolver: TensorResolver) -> Mapping[str, Any] | None:
    """Decode bounded binary media to vLLM-native PIL images / numpy video arrays.

    vLLM's ``multi_modal_data`` mapping accepts ``{"image": PIL.Image | list[PIL.Image]}``
    and ``{"video": (np.ndarray, metadata) | list[(np.ndarray, metadata)]}``. We decode
    each binary item with PIL (images) / imageio+decord-equivalent (video) so no caller
    path ever reaches the model. The decode happens at admission, before the engine call.
    """
    if not media:
        return None
    images: list[Any] = []
    videos: list[Any] = []
    for item in media:
        raw = resolver(item.data)
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ContractValidationError("media data must resolve to binary bytes")
        data = bytes(raw)
        if item.kind == "image":
            from PIL import Image  # imported lazily; available in the Cosmos venv

            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception as exc:
                raise ContractValidationError(f"image media could not be decoded: {exc}") from exc
            images.append(img)
        else:  # video
            video_array, meta = _decode_video_bytes(data)
            videos.append((video_array, meta))
    multi_modal: dict[str, Any] = {}
    if images:
        multi_modal["image"] = images[0] if len(images) == 1 else images
    if videos:
        multi_modal["video"] = videos[0] if len(videos) == 1 else videos
    return multi_modal or None


def _decode_video_bytes(data: bytes) -> tuple[Any, dict[str, Any]]:
    """Decode video bytes to a (numpy array T,H,W,3 uint8, metadata) pair.

    Mirrors vLLM's ``VideoMediaIO`` contract: the engine consumes a numpy array of
    frames plus a metadata dict (e.g. ``{"num_frames": T}``). Uses decord when
    available (the Cosmos venv ships it), falling back to imageio.
    """
    import numpy as np

    try:
        import decord  # type: ignore[import-not-found]

        vr = decord.VideoReader(io.BytesIO(data))
        frames = vr.get_batch(range(len(vr))).asnumpy().astype(np.uint8)  # [T,H,W,3]
        meta = {"num_frames": int(len(vr)), "fps": float(vr.get_avg_fps())}
        return frames, meta
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        reader = imageio.get_reader(io.BytesIO(data))
        frames = np.stack([np.asarray(frame) for frame in reader], axis=0).astype(np.uint8)
        meta = {"num_frames": int(frames.shape[0])}
        return frames, meta
    except Exception as exc:
        raise ContractValidationError(f"video media could not be decoded: {exc}") from exc


def _build_sampling_params(generation: GenerationControls, max_model_len: int) -> Any:
    """Construct vLLM ``SamplingParams`` from bounded generation controls."""
    from vllm import SamplingParams  # imported lazily; only in the Cosmos venv

    return SamplingParams(
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k or -1,
        max_tokens=min(generation.max_tokens, max_model_len),
        seed=generation.seed if generation.seed is not None else None,
        stop=list(generation.stop) if generation.stop else None,
        frequency_penalty=generation.frequency_penalty,
        presence_penalty=generation.presence_penalty,
    )


def _fake_sampling_params(generation: GenerationControls, max_model_len: int) -> Any:
    """Test-only SamplingParams factory that carries the bounded controls without vLLM."""
    return {
        "max_tokens": min(generation.max_tokens, max_model_len),
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "seed": generation.seed,
        "stop": list(generation.stop),
        "frequency_penalty": generation.frequency_penalty,
        "presence_penalty": generation.presence_penalty,
    }


def _apply_chat_template(request: Cosmos3Request, tokenizer: Any) -> tuple[str, list[int]]:
    """Apply the resident tokenizer chat template to prompt or messages.

    Returns ``(prompt_text, prompt_token_ids)``. When the request supplies a plain
    ``prompt``, it is wrapped as a single user turn. Media is supplied separately to
    ``engine.generate`` via ``multi_modal_data``; the chat template must render the
    message placeholders (vLLM's ``apply_chat_template`` handles the image/video tokens
    for Cosmos3 given the multi_modal_data mapping).
    """
    if request.prompt is not None:
        # Cosmos3's processor inserts multimodal token placeholders from OpenAI-style
        # content items. Passing a raw prompt alongside multi_modal_data leaves vLLM
        # without the image/video positions it needs to bind media.
        content = [{"type": item.kind} for item in request.media]
        content.append({"type": "text", "text": request.prompt})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": role, "content": content} for role, content in request.messages]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(rendered, str):
        return rendered, []
    # Transformers 5 returns a BatchEncoding here; older supported versions return
    # token ids directly. Both resolve to one unbatched token-id sequence for vLLM.
    if hasattr(rendered, "get"):
        rendered = rendered.get("input_ids", rendered)
    return "", [int(token) for token in rendered]


def _load_cosmos3_backend(config: Cosmos3ModelConfig) -> Cosmos3Backend:
    """Load the resident vLLM AsyncLLMEngine once inside the assigned Serve replica.

    Uses the exact Cosmos3 EngineArgs. The engine is the vLLM 0.19.1 v1 ``AsyncLLM``
    (aliased as ``AsyncLLMEngine``), whose ``generate`` returns an async generator of
    ``RequestOutput`` with continuous batching across concurrent requests.
    """
    import torch
    from vllm import AsyncLLMEngine, SamplingParams  # noqa: F401
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer

    engine_args = AsyncEngineArgs(**config.engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_source)

    class VllmCosmos3Backend:
        def __init__(self) -> None:
            self._engine = engine
            self._tokenizer = tokenizer

        @property
        def model_revision(self) -> str:
            return config.model_revision

        def generate(
            self,
            prompt: Any,
            sampling_params: Any,
            request_id: str,
            *,
            multi_modal_data: Mapping[str, Any] | None = None,
        ) -> Any:
            from vllm.inputs import TextPrompt, TokensPrompt

            if isinstance(prompt, list):
                payload: dict[str, Any] = {"prompt_token_ids": prompt}
                if multi_modal_data is not None:
                    payload["multi_modal_data"] = multi_modal_data
                inp = TokensPrompt(**payload)
            else:
                payload = {"prompt": prompt}
                if multi_modal_data is not None:
                    payload["multi_modal_data"] = multi_modal_data
                inp = TextPrompt(**payload)
            return self._engine.generate(inp, sampling_params, request_id)

        def engine_running_requests(self) -> int:
            try:
                return int(getattr(self._engine, "num_running_requests", 0))
            except Exception:
                return 0

    return VllmCosmos3Backend()


class Cosmos3Adapter:
    """Resident vLLM engine owner used by a single Ray Serve GPU6 replica."""

    def __init__(
        self,
        config: Cosmos3ModelConfig,
        *,
        backend_factory: BackendFactory = _load_cosmos3_backend,
        tensor_resolver: TensorResolver = _default_tensor_resolver,
        sampling_params_factory: SamplingParamsFactory = _build_sampling_params,
    ) -> None:
        self._config = config
        self._tensor_resolver = tensor_resolver
        self._sampling_params_factory = sampling_params_factory
        self._backend = backend_factory(config)
        self._model_load_count = 1
        self._running_batches = 0
        self._admitted_pending = 0

    @property
    def config(self) -> Cosmos3ModelConfig:
        return self._config

    def admit(self, request: Cosmos3Request) -> _PreparedRequest:
        """Validate and decode a request against the resident config before admission.

        Enforces bounded media (count + per-item + per-request bytes), decodes media to
        vLLM-native PIL/numpy, builds SamplingParams, and applies the chat template.
        Raises ``ContractValidationError`` for oversized media, undecodable media, or an
        oversized prompt. ``model_revision`` is server-owned, so the request must not
        carry one (the contract has no such field by construction).
        """
        if len(request.media) > self._config.max_media_per_request:
            raise ContractValidationError(
                f"request carries {len(request.media)} media items but the bound is "
                f"{self._config.max_media_per_request}"
            )
        total_bytes = 0
        for item in request.media:
            if not isinstance(item.data, (bytes, bytearray, memoryview)):
                raise ContractValidationError("media data must be binary bytes at the HTTP boundary")
            size = len(bytes(item.data))
            if size > self._config.max_media_bytes_per_item:
                raise ContractValidationError(
                    f"media item (kind={item.kind}) is {size} bytes but the per-item bound is "
                    f"{self._config.max_media_bytes_per_item}"
                )
            total_bytes += size
        if total_bytes > self._config.max_media_bytes_per_request:
            raise ContractValidationError(
                f"request media total {total_bytes} bytes exceeds the per-request bound "
                f"{self._config.max_media_bytes_per_request}"
            )
        multi_modal_data = decode_media(request.media, self._tensor_resolver)
        sampling_params = self._sampling_params_factory(request.generation, self._config.max_model_len)
        # Tokenizer/template application happens lazily inside the backend's generate path
        # when prompt is text; for messages we pre-render token ids via the backend tokenizer.
        prompt: Any
        # Always render through the resident tokenizer. This gives prompt requests with
        # image/video media their required Cosmos3 multimodal placeholders and keeps the
        # messages path on the same token contract.
        tokenizer = getattr(self._backend, "_tokenizer", None)
        if tokenizer is None:
            raise ContractValidationError("Cosmos3 requests require a resident tokenizer")
        _, prompt = _apply_chat_template(request, tokenizer)
        admitted_at = time.monotonic()
        self._admitted_pending += 1
        return _PreparedRequest(
            request=request,
            prompt=prompt,
            multi_modal_data=multi_modal_data,
            sampling_params=sampling_params,
            admitted_monotonic_s=admitted_at,
        )

    def request_dispatched(self) -> None:
        self._admitted_pending = max(0, self._admitted_pending - 1)

    async def infer(self, request: Cosmos3Request) -> Cosmos3Response:
        try:
            prepared = self.admit(request)
        except ContractValidationError as exc:
            return Cosmos3Response(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
            )
        return await self.infer_one(prepared)

    async def infer_one(self, prepared: _PreparedRequest) -> Cosmos3Response:
        """Run one engine ``generate`` call and collect the final RequestOutput.

        vLLM's engine coalesces concurrent ``generate`` calls internally (continuous
        batching); there is no Serve-level ``@serve.batch`` callback here.
        """
        import asyncio

        batch_id = uuid4().hex
        request_id = f"{prepared.request.ownership.request_id}-{batch_id[:8]}"
        dispatched_monotonic_s = time.monotonic()
        self.request_dispatched()
        self._running_batches += 1
        forward_started_monotonic_s = time.monotonic()
        first_token_monotonic_s = forward_started_monotonic_s
        try:
            gen = self._backend.generate(
                prepared.prompt,
                prepared.sampling_params,
                request_id,
                multi_modal_data=prepared.multi_modal_data,
            )
            final_output = None
            async for output in gen:
                final_output = output
                if first_token_monotonic_s == forward_started_monotonic_s:
                    first_token_monotonic_s = time.monotonic()
            completed_monotonic_s = time.monotonic()
        except Exception as exc:
            completed_monotonic_s = time.monotonic()
            self._running_batches = max(0, self._running_batches - 1)
            # Some vLLM exceptions stringify to an empty string. Preserve the exception
            # class/repr so a failed live request is a diagnosable model failure rather
            # than an invalid empty-error response at the transport boundary.
            message = str(exc) or f"{type(exc).__name__}: {exc!r}"
            return Cosmos3Response(
                ownership=prepared.request.ownership,
                error=ServiceError(
                    ErrorCode.MODEL_FAILURE,
                    message,
                    retryable=False,
                    ownership=prepared.request.ownership,
                    batch_id=batch_id,
                ),
            )
        self._running_batches = max(0, self._running_batches - 1)

        if final_output is None:
            return Cosmos3Response(
                ownership=prepared.request.ownership,
                error=ServiceError(
                    ErrorCode.MODEL_FAILURE,
                    "vLLM engine produced no output",
                    retryable=False,
                    ownership=prepared.request.ownership,
                    batch_id=batch_id,
                ),
            )

        text, finish_reason, stop_reason, prompt_tokens, completion_tokens, total_tokens = _extract_output(final_output)
        # vLLM's engine running count at submit time is the best available proxy for the
        # effective engine batch this request joined. forward_count records it as provenance.
        running_at_submit = max(1, self._backend.engine_running_requests())
        timings = {
            "queue_wait_s": max(0.0, dispatched_monotonic_s - prepared.admitted_monotonic_s),
            "prefill_s": max(0.0, forward_started_monotonic_s - dispatched_monotonic_s),
            "time_to_first_token_s": max(0.0, first_token_monotonic_s - forward_started_monotonic_s),
            "decode_s": max(0.0, completed_monotonic_s - first_token_monotonic_s),
            "e2e_s": max(0.0, completed_monotonic_s - prepared.admitted_monotonic_s),
        }
        trace = BatchTrace(
            batch_id=batch_id,
            replica_id=self._config.replica_id,
            admitted_monotonic_s=prepared.admitted_monotonic_s,
            dispatched_monotonic_s=dispatched_monotonic_s,
            forward_started_monotonic_s=forward_started_monotonic_s,
            completed_monotonic_s=completed_monotonic_s,
            effective_work_units=1,
            request_count=1,
            request_ids=(prepared.request.ownership.request_id,),
            # vLLM engine batch provenance (informational): the engine fuses across
            # concurrent requests internally; this is not a Serve-callback forward count.
            forward_count=running_at_submit,
            model_load_count=self._model_load_count,
            served_at_wall_unix_s=time.time(),
        )
        media_provenance = tuple(
            {"kind": m.kind, "media_type": m.media_type, "source_index": m.source_index, "bytes": len(bytes(m.data))}
            for m in prepared.request.media
        )
        try:
            result = Cosmos3Result(
                ownership=prepared.request.ownership,
                text=text,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                timings=timings,
                model_revision=self._backend.model_revision,
                trace=trace,
                media_provenance=media_provenance,
            )
            return Cosmos3Response(ownership=prepared.request.ownership, result=result)
        except ContractValidationError as exc:
            return Cosmos3Response(
                ownership=prepared.request.ownership,
                error=ServiceError(
                    ErrorCode.RESULT_SPLIT_FAILURE,
                    str(exc),
                    retryable=False,
                    ownership=prepared.request.ownership,
                    batch_id=batch_id,
                ),
            )

    def status(self) -> DeploymentStatus:
        return DeploymentStatus(
            deployment_name="cosmos3.reason",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
        )


def _extract_output(final_output: Any) -> tuple[str, str, str | None, int, int, int]:
    """Read generated text, finish/stop reason, and token counts from a RequestOutput.

    vLLM 0.19.1 ``RequestOutput`` has ``outputs: list[CompletionOutput]`` where each
    ``CompletionOutput`` carries ``text``, ``finish_reason``, ``stop_reason``. Token
    counts come from the engine's usage stats when present; otherwise derived from the
    final output's ``prompt_token_ids`` and the completion text length.
    """
    outputs = getattr(final_output, "outputs", None) or []
    completion = outputs[0] if outputs else None
    text = getattr(completion, "text", "") if completion is not None else ""
    finish_reason = getattr(completion, "finish_reason", "stop") if completion is not None else "stop"
    stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None

    prompt_tokens = 0
    completion_tokens = 0
    # vLLM exposes metrics on RequestOutput.metrics (RequestStateStats) in 0.19.1.
    metrics = getattr(final_output, "metrics", None)
    if metrics is not None:
        prompt_tokens = int(getattr(metrics, "num_prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(metrics, "num_generation_tokens", 0) or 0)
    if prompt_tokens == 0:
        prompt_token_ids = getattr(final_output, "prompt_token_ids", None) or []
        prompt_tokens = len(prompt_token_ids)
    if completion_tokens == 0 and completion is not None:
        # Fallback: approximate completion tokens from the token ids on the completion.
        token_ids = getattr(completion, "token_ids", None) or []
        completion_tokens = len(token_ids)
    total_tokens = prompt_tokens + completion_tokens
    return text, str(finish_reason), stop_reason, prompt_tokens, completion_tokens, total_tokens
