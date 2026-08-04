"""One service-side pending queue budget expressed in image equivalents."""
from __future__ import annotations

IMAGE_QUEUE_BUDGET = 8000
REQUEST_IMAGE_EQUIVALENTS = {
    "unidepth.infer": 1,
    "hands.detect": 1,
    "wilor.reconstruct": 1,
    "hawor.infer_tracks": 16,
    "hawor_infiller.fill": 120,
    "cosmos3.reason": 8,
    "droid.infer": 256,
}


def queued_request_capacity(stage_id: str) -> int:
    """Return the smallest request queue that can hold 8000 images/frames."""
    images_per_request = REQUEST_IMAGE_EQUIVALENTS[stage_id]
    return (IMAGE_QUEUE_BUDGET + images_per_request - 1) // images_per_request
