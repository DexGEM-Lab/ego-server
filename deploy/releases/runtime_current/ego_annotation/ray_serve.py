"""Ray Serve deployment for the pre-existing annotation job API only.

The resident, model-native UniDepth serving stack has moved to
``ego_annotation.serving.deployment`` (a deployment-only import path that imports Ray
Serve at module top level). This module retains the annotation-job deployment so the
existing annotation API path is unchanged.

Importing this module is GPU/Ray-free. Ray decorators and the annotation runner are
constructed only when ``build_deployment`` is called.
"""
from __future__ import annotations

from typing import Any

from ego_annotation.jobs import AnnotationJobRunner
from ego_annotation.models import AnnotationJobRequest


def build_deployment() -> Any:
    """Compatibility deployment for the pre-existing annotation job API only."""
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
