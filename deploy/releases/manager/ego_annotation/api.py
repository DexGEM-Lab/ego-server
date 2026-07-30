"""FastAPI ingress for annotation jobs."""
from __future__ import annotations

from typing import Any

from ego_annotation.jobs import AnnotationJobRunner
from ego_annotation.models import AnnotationJobRequest
from ego_annotation.schema import PUBLIC_ANNOTATION_ENDPOINT


def create_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in deployment env
        raise RuntimeError(
            "FastAPI is required for POST /v1/annotation-jobs. Install the service extra before starting the API."
        ) from exc

    app = FastAPI(title="Ego Annotation API", version="1.0.0-alpha")
    runner = AnnotationJobRunner()

    @app.post(PUBLIC_ANNOTATION_ENDPOINT)
    def create_annotation_job(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = AnnotationJobRequest.from_mapping(payload)
            return runner.run(request).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
