"""Typed model-serving foundation with no import-time Ray dependency.

Ordinary adapter/contract/transport unit tests import from this package without Ray
installed. The Ray Serve deployment wrapper lives in ``ego_annotation.serving.deployment``
and is a deployment-only import path.

The caller-facing boundary is the typed multi-cluster gateway
(``gateway.ModelServiceGateway``) over the stable public router (``router``).
"""
from ego_annotation.serving.batching import (
    BatchPolicy,
    assert_one_forward,
    canonical_batch_size_fn,
    total_work,
)
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
    HandDetection,
    HandSide,
    HandsDetectRequest,
    HandsDetectResponse,
    HandsDetectResult,
    ImageSize,
    ManoOutput,
    Ownership,
    PixelTransform,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
    UniDepthResult,
    WiLoRReconstructRequest,
    WiLoRReconstructResponse,
    WiLoRReconstructResult,
    reject_filesystem_fields,
    utc_now,
    # Stateful DROID session API.
    CameraState,
    DenseSourceMapping,
    DROID_RGB_DTYPE,
    DroidBatchTrace,
    DroidPhaseTiming,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    DroidImageShape,
    DroidSessionOptions,
    DroidUncertainty,
    FrameValidity,
    KeyframeSourceMapping,
    StepStatus,
)

__all__ = [
    "BatchPolicy", "BatchTrace", "ContractValidationError", "Cosmos3MediaItem", "Cosmos3Request",
    "Cosmos3Response", "Cosmos3Result", "DeploymentStatus", "ErrorCode", "GenerationControls", "HandDetection",
    "HandSide", "HandsDetectRequest", "HandsDetectResponse", "HandsDetectResult", "ImageSize", "ManoOutput",
    "Ownership", "PixelTransform", "ServiceError", "SpatialMetadata", "TensorPayload", "UniDepthRequest",
    "UniDepthResponse", "UniDepthResult", "WiLoRReconstructRequest", "WiLoRReconstructResponse",
    "WiLoRReconstructResult", "assert_one_forward", "canonical_batch_size_fn",
    "reject_filesystem_fields", "total_work", "utc_now",
    # DROID session API.
    "CameraState", "DenseSourceMapping", "DROID_RGB_DTYPE", "DroidBatchTrace", "DroidPhaseTiming", "DroidCamera",
    "DroidCreateSessionRequest", "DroidCreateSessionResponse", "DroidFinalizeRequest",
    "DroidFinalizeResponse", "DroidFrameRequest", "DroidFrameResponse", "DroidImageShape",
    "DroidSessionOptions", "DroidUncertainty", "FrameValidity", "KeyframeSourceMapping", "StepStatus",
]
