from ego_annotation.serving.queue_budget import (
    IMAGE_QUEUE_BUDGET,
    REQUEST_IMAGE_EQUIVALENTS,
    queued_request_capacity,
)


def test_all_service_queues_hold_at_least_8000_images() -> None:
    assert IMAGE_QUEUE_BUDGET == 8000
    expected = {
        "unidepth.infer": 8000,
        "hands.detect": 8000,
        "wilor.reconstruct": 8000,
        "hawor.infer_tracks": 500,
        "hawor_infiller.fill": 67,
        "cosmos3.reason": 1000,
        "droid.infer": 32,
    }
    assert {stage: queued_request_capacity(stage) for stage in REQUEST_IMAGE_EQUIVALENTS} == expected
    for stage, images_per_request in REQUEST_IMAGE_EQUIVALENTS.items():
        queued = queued_request_capacity(stage)
        assert queued * images_per_request >= IMAGE_QUEUE_BUDGET
        assert (queued - 1) * images_per_request < IMAGE_QUEUE_BUDGET
