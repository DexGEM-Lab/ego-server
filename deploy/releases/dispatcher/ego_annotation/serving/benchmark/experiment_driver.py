"""Detached Ray 2.55 Serve driver for allowlisted experimental applications.

This replaces the invalid/blocking ``serve run --port`` invocation.  The driver
attests the release before importing any application module, connects to the exact
GCS address, starts Serve detached on the exact HTTP port, and returns after a
non-blocking named deployment is submitted.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.release import verify_release


EXPERIMENT_APPLICATIONS = {
    "unidepth": "ego_annotation.serving.deployment:app",
    "droid": "ego_annotation.serving.droid_deployment:app",
    # GPU3 has two independent applications on the same isolated Serve proxy.
    "hawor": "ego_annotation.serving.hawor_deployment:app",
    # GPU1-equivalent experiment: one deployment exposes Hands/SAM2 and WiLoR.
    "hands": "ego_annotation.serving.hands_deployment:hands_app",
}


def _load_application(release_root: Path, import_path: str):
    verified = verify_release(release_root)
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("import path must be module:attribute")
    # Remove cwd/PYTHONPATH shadows and import only from the attested root.
    sys.path[:] = [str(release_root)] + [p for p in sys.path if Path(p or os.curdir).resolve() != release_root]
    module = importlib.import_module(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        module_file.relative_to(verified.module_root)
    except ValueError as exc:
        raise RuntimeError(f"application module imported outside verified release root: {module_file}") from exc
    return getattr(module, attribute)


def run_driver(
    *, release_root: str | Path, gcs_address: str, http_port: int, app_choice: str,
    app_name: str, route_prefix: str,
) -> None:
    if app_choice not in EXPERIMENT_APPLICATIONS:
        raise ValueError(f"unsupported experimental application {app_choice!r}")
    import_path = EXPERIMENT_APPLICATIONS[app_choice]
    root = Path(release_root).resolve(strict=True)
    verified = verify_release(root)
    os.chdir(root)
    # Ray is intentionally imported only after release verification.
    import ray  # type: ignore
    from ray import serve  # type: ignore

    ray.init(address=gcs_address, namespace=f"ego-experiment-{app_name}", ignore_reinit_error=True)
    serve.start(
        detached=True,
        http_options={"host": "0.0.0.0", "port": int(http_port)},
    )
    application = _load_application(root, import_path)
    if app_choice == "hawor":
        deployment = importlib.import_module("ego_annotation.serving.hawor_deployment")
        deployment_file = Path(getattr(deployment, "__file__", "")).resolve()
        try:
            deployment_file.relative_to(verified.module_root)
        except ValueError as exc:
            raise RuntimeError(f"HaWoR deployment imported outside verified release root: {deployment_file}") from exc
        serve.run(application, blocking=False, name=f"{app_name}-tracks", route_prefix="/hawor.infer_tracks")
        serve.run(deployment.infiller_app, blocking=False, name=f"{app_name}-infiller", route_prefix="/hawor_infiller.fill")
        return
    serve.run(application, blocking=False, name=app_name, route_prefix=route_prefix)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one detached experimental Ray Serve application")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--gcs-address", required=True)
    parser.add_argument("--http-port", required=True, type=int)
    parser.add_argument("--app-choice", required=True, choices=tuple(EXPERIMENT_APPLICATIONS))
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--route-prefix", default="/")
    args = parser.parse_args(argv)
    run_driver(
        release_root=args.release_root, gcs_address=args.gcs_address, http_port=args.http_port,
        app_choice=args.app_choice, app_name=args.app_name, route_prefix=args.route_prefix,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - Ray-only process
    raise SystemExit(main())
