"""Ray Serve deployment-only module for the resident Cosmos3 (vLLM) model API.

This module imports Ray Serve at the top level and is therefore a **deployment-only
import path**: ordinary adapter/contract unit tests never import it. ``serve run`` and
``serve deploy`` resolve ``ego_annotation.serving.cosmos3_deployment:app`` against the
GPU6 cluster's dedicated Python 3.13 interpreter (Torch 2.10.0+cu128, vLLM 0.19.1,
Ray 2.55.1).

Design invariants:

* One Ray cluster per physical GPU group (see ``lifecycle.py``). The GPU6 cluster starts
  with ``CUDA_VISIBLE_DEVICES=6`` and advertises one native Ray GPU (``--num-gpus=1``),
  so this deployment requests ``num_gpus=1`` and Ray owns the physical GPU6.
* The resident vLLM ``AsyncLLMEngine`` is constructed once at replica startup with the
  exact Cosmos3 ``EngineArgs`` and stays resident; ``model_load_count`` does not increase
  with request count.
* vLLM owns continuous batching across concurrent requests. Unlike the UniDepth
  deployment, this module does NOT use ``@serve.batch``: each request is one engine
  ``generate`` call and the vLLM engine coalesces concurrent requests internally.
  ``max_ongoing_requests``/``max_queued_requests`` bound the Serve admission queue.
* Requests arrive as multipart binary HTTP carrying bounded image/video media parts; the
  deployment parses the multipart body, reconstructs the contract request, admits it
  (media bounds + decode + revision check), and forwards it to the engine. The response
  is scalar (text/token/timing/provenance) and is returned as a multipart metadata part.
* No caller-controlled server path is ever accepted: media travels as inline binary.
"""
from __future__ import annotations

import os
from typing import Any

from ray import serve
from starlette.responses import JSONResponse, Response

from ego_annotation.serving.batch_snapshot import snapshot_collection
from ego_annotation.serving.contracts import (
    ContractValidationError,
    Cosmos3MediaItem,
    Cosmos3Request,
    Cosmos3Response,
    ErrorCode,
    GenerationControls,
    Ownership,
    ServiceError,
)
from ego_annotation.serving.cosmos3 import Cosmos3Adapter, build_cosmos3_model_config
from ego_annotation.serving.transport import (
    multipart_asgi_response,
    parse_cosmos3_request,
)


def _config_from_env() -> Any:
    return build_cosmos3_model_config(
        model_source=os.environ.get("EGO_COSMOS3_MODEL", "nvidia/Cosmos3-Nano"),
        model_revision=os.environ.get("EGO_COSMOS3_REVISION", "cosmos3-nano:vllm-0.19.1"),
        replica_id=os.environ.get("EGO_COSMOS3_REPLICA_ID", "cosmos3-gpu6"),
        assigned_gpu=int(os.environ.get("EGO_COSMOS3_GPU", "6")),
        max_media_per_request=int(os.environ.get("EGO_COSMOS3_MAX_MEDIA", "8")),
        max_media_bytes_per_item=int(os.environ.get("EGO_COSMOS3_MAX_MEDIA_BYTES_ITEM", str(16 * 1024 * 1024))),
        max_media_bytes_per_request=int(os.environ.get("EGO_COSMOS3_MAX_MEDIA_BYTES_REQ", str(64 * 1024 * 1024))),
    )


@serve.deployment(
    name="cosmos3.reason",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/home/ylang/.cache/huggingface",
                "EGO_COSMOS3_MODEL": "/home/ylang/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
            }
        },
    },
    # vLLM owns the running batch; these bound the Serve admission queue only.
    max_ongoing_requests=16,
    max_queued_requests=32,
)
class Cosmos3Deployment:
    def __init__(self) -> None:
        # The resident vLLM AsyncLLMEngine loads once here and stays resident.
        self.adapter = Cosmos3Adapter(_config_from_env())

    async def reason(self, request: Cosmos3Request) -> Cosmos3Response:
        # vLLM's engine owns batching; no @serve.batch callback here.
        return await self.adapter.infer(request)

    async def __call__(self, request: Any) -> Response:
        path = request.url.path if hasattr(request, "url") else ""
        if getattr(request, "method", "").upper() == "GET" and path.rstrip("/").endswith("/-/batch-snapshot"):
            return JSONResponse(snapshot_collection(self.adapter.batch_snapshot()), headers={"Cache-Control": "no-store"})
        body = await request.body()
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        try:
            metadata, media_items = parse_cosmos3_request(body, content_type)
            ownership = Ownership.from_mapping(metadata["ownership"])
        except (ValueError, KeyError, ContractValidationError) as exc:
            return _error_response(ownership=None, exc=exc)
        try:
            media = tuple(
                Cosmos3MediaItem(
                    kind=item["kind"],
                    data=item["data"],
                    media_type=item["media_type"],
                    source_index=item["source_index"],
                )
                for item in media_items
            )
            parsed = Cosmos3Request(
                ownership=ownership,
                prompt=metadata.get("prompt"),
                messages=tuple(
                    (str(m.get("role")), m.get("content"))
                    for m in metadata.get("messages", [])
                    if isinstance(m, dict)
                ),
                media=media,
                generation=GenerationControls.from_mapping(metadata.get("generation")),
            )
        except ContractValidationError as exc:
            return _error_response(ownership=ownership, exc=exc)
        response = await self.reason(parsed)
        return _response_to_wire(response)


def _error_response(ownership: Any, exc: Exception) -> Response:
    code = ErrorCode.VALIDATION if isinstance(exc, (ContractValidationError, ValueError, KeyError)) else ErrorCode.TRANSPORT
    err = (
        ServiceError(code, str(exc), retryable=False, ownership=ownership)
        if ownership is not None
        else ServiceError(code, str(exc), retryable=False)
    )
    return multipart_asgi_response(
        {"ownership": ownership.to_wire() if ownership is not None else None, "result": None, "error": err.to_wire()},
        {},
    )


def _response_to_wire(response: Cosmos3Response) -> Response:
    return multipart_asgi_response(response.to_wire(), {})


# A bound Ray Serve Application. ``serve run ego_annotation.serving.cosmos3_deployment:app``
# deploys this against the GPU6 cluster.
app: Any = Cosmos3Deployment.bind()
