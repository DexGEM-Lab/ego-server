"""Deploy one non-Cosmos Ego model-service group into an existing Ray head.

The driver uses Ray Serve's detached control plane and then exits; model replicas and
proxy remain owned by the scoped Ray cluster. GPU3 is the only group with two
independent applications and therefore must deploy both explicitly.
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence


def _validate_target(gpu_id: int, *, address: str, dashboard_address: str, port: int, combined: bool) -> None:
    """Reject an accidental driver invocation against another Ray head."""
    from ego_annotation.serving.lifecycle import COMMITTED_GPU_GROUPS

    group = next((candidate for candidate in COMMITTED_GPU_GROUPS if candidate.gpu_id == gpu_id), None)
    if group is None or (address, dashboard_address, port) != (
        group.lifecycle.gcs_address, group.lifecycle.dashboard_address, group.lifecycle.ports.serve_http_port,
    ):
        raise ValueError(f"GPU{gpu_id} endpoint tuple does not match committed lifecycle ownership")
    if combined and gpu_id != 1:
        raise ValueError("--combined is valid only for the GPU1 Hands rollback application")


def deploy_group(gpu_id: int, *, address: str, dashboard_address: str, port: int, combined: bool = False) -> None:
    _validate_target(gpu_id, address=address, dashboard_address=dashboard_address, port=port, combined=combined)
    import ray
    from ray import serve

    ray.init(address=address, ignore_reinit_error=True)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": port})

    if gpu_id == 0:
        from ego_annotation.serving.deployment import app

        serve.run(app, name="ego-unidepth", route_prefix="/")
    elif gpu_id == 1:
        if combined:
            from ego_annotation.serving.hands_deployment import hands_app

            serve.run(hands_app, name="ego-hands-wilor", route_prefix="/")
        else:
            from ego_annotation.serving.hands_deployment import hands_only_app

            serve.run(hands_only_app, name="ego-hands", route_prefix="/")
    elif gpu_id == 4:
        from ego_annotation.serving.hands_deployment import wilor_only_app

        serve.run(wilor_only_app, name="ego-wilor", route_prefix="/")
    elif gpu_id == 2:
        from ego_annotation.serving.droid_deployment import app

        serve.run(app, name="ego-droid-service", route_prefix="/")
    elif gpu_id == 3:
        from ego_annotation.serving.hawor_deployment import app, infiller_app

        serve.run(app, name="hawor-infer-tracks", route_prefix="/hawor.infer_tracks")
        serve.run(infiller_app, name="hawor-infiller-fill", route_prefix="/hawor_infiller.fill")
    else:
        raise ValueError(f"serve_group_driver supports GPU0/1/2/3/4, got GPU{gpu_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy one Ego Serve group into its existing Ray head")
    parser.add_argument("--gpu-id", type=int, required=True, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--address", required=True, help="Exact Ray GCS address for this group")
    parser.add_argument("--dashboard-address", required=True, help="Exact dashboard address, validated against lifecycle ownership")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--combined", action="store_true", help="GPU1 rollback: deploy combined Hands+WiLoR application")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deploy_group(
        args.gpu_id, address=args.address, dashboard_address=args.dashboard_address,
        port=args.port, combined=args.combined,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
