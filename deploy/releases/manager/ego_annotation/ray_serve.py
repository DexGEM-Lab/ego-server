"""Ray Serve deployment wrapper for annotation jobs."""
from __future__ import annotations

from typing import Any

from ego_annotation.jobs import AnnotationJobRunner
from ego_annotation.models import AnnotationJobRequest


def build_deployment() -> Any:
    try:
        from ray import serve
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in deployment env
        raise RuntimeError(
            "Ray Serve is required for resident GPU actor deployment. Install the service extra on the private GPU fleet."
        ) from exc

    @serve.deployment(name="annotation-job-runner")
    class AnnotationJobDeployment:
        def __init__(self) -> None:
            self.runner = AnnotationJobRunner()

        async def __call__(self, request: Any) -> dict[str, Any]:
            payload = await request.json()
            job = AnnotationJobRequest.from_mapping(payload)
            return self.runner.run(job).to_dict()

    return AnnotationJobDeployment
