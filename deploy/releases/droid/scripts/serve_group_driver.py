"""Deploy one non-Cosmos Ego model-service group into an existing Ray head.

The driver uses Ray Serve's detached control plane and then exits; model replicas and
proxy remain owned by the scoped Ray cluster. GPU3 is the only group with two
independent applications and therefore must deploy both explicitly.
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence


def deploy_group(gpu_id: int, *, address: str, port: int) -> None:
    import ray
    from ray import serve

    ray.init(address=address, ignore_reinit_error=True)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": port})

    if gpu_id == 0:
        from ego_annotation.serving.deployment import app

        serve.run(app, name="ego-unidepth", route_prefix="/")
    elif gpu_id == 1:
        from ego_annotation.serving.hands_deployment import hands_app

        serve.run(hands_app, name="ego-hands-wilor", route_prefix="/")
    elif gpu_id == 2:
        from ego_annotation.serving.droid_deployment import app

        serve.run(app, name="ego-droid-service", route_prefix="/")
    elif gpu_id == 3:
        from ego_annotation.serving.hawor_deployment import app, infiller_app

        serve.run(app, name="hawor-infer-tracks", route_prefix="/hawor.infer_tracks")
        serve.run(infiller_app, name="hawor-infiller-fill", route_prefix="/hawor_infiller.fill")
    else:
        raise ValueError(f"serve_group_driver supports GPU0/1/2/3, got GPU{gpu_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy one Ego Serve group into its existing Ray head")
    parser.add_argument("--gpu-id", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--address", required=True)
    parser.add_argument("--port", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deploy_group(args.gpu_id, address=args.address, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
