#!/usr/bin/env python3
"""Run one V22 pipeline under the API-owned remote admission boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The current API-Ify client exposes DROID as the three lifecycle routes
# droid.create_session/push_frame/finalize.  The admission proxy only exposes
# the legacy atomic /droid.infer endpoint, so leave DROID on its native A800
# localhost origin instead of injecting the stale alias and failing client-side
# stage validation.  Stateless stages continue to use the admission proxy.
API_IFY_STAGE_IDS = (
    "unidepth.infer",
    "hands.detect",
    "wilor.reconstruct",
    "hawor.infer_tracks",
    "hawor_infiller.fill",
    "cosmos3.reason",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation_admission_proxy import running_proxy
from scripts.package_v22_annotation_result import PackageError, create_result_package


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    separator = next((index for index, value in enumerate(sys.argv[1:]) if value == "--"), None)
    if separator is None:
        raise SystemExit("run_v22_api_job_with_admission requires -- followed by the pipeline command")
    separator += 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--algorithm-inflight-multiplier", type=int, required=True)
    parser.add_argument("--upstream-endpoints-json", default="{}")
    parser.add_argument("--api-ify", action="store_true", help="Route the frozen API-Ify runner through the shared proxy")
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=0)
    parser.add_argument("--package-root", type=Path, default=None, help="Package the completed single-item run under this directory")
    parser.add_argument("--package-name", default=None, help="Optional downloadable package basename")
    args = parser.parse_args(sys.argv[1:separator])
    command = sys.argv[separator + 1 :]
    if not command:
        raise SystemExit("missing pipeline command after --")
    return args, command


def _run_root_from_command(command: list[str]) -> Path | None:
    try:
        index = command.index("--run-root")
        return Path(command[index + 1]).expanduser()
    except (ValueError, IndexError):
        return None


def main() -> int:
    args, command = parse_args()
    if args.algorithm_inflight_multiplier <= 0:
        raise SystemExit("--algorithm-inflight-multiplier must be positive")
    try:
        upstream_overrides = json.loads(args.upstream_endpoints_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --upstream-endpoints-json: {exc}") from exc
    if not isinstance(upstream_overrides, dict):
        raise SystemExit("--upstream-endpoints-json must be a JSON object")
    args.lock_root.mkdir(parents=True, exist_ok=True)
    # The wrapper owns no fixed per-algorithm reservation. Its local proxy
    # groups fully buffered stateless requests by native batch cap and retries
    # only rejected 429 items; DROID occupies one bounded single-push slot per
    # request, and Cosmos continues to use vLLM-owned scheduling.
    with running_proxy(
        host=args.proxy_host,
        port=args.proxy_port,
        profile=args.profile,
        multiplier=args.algorithm_inflight_multiplier,
        events_path=args.events_path,
        lock_root=args.lock_root,
        upstream_overrides={str(key): str(value) for key, value in upstream_overrides.items()},
    ) as proxy_url:
        if args.api_ify:
            command = [
                *command,
                "--service-origins-json",
                json.dumps({stage_id: proxy_url for stage_id in API_IFY_STAGE_IDS}, sort_keys=True),
            ]
        else:
            command = [
                *command,
                "--feishu-unidepth-base-url",
                proxy_url,
                "--feishu-hands-wilor-base-url",
                proxy_url,
                "--feishu-droid-base-url",
                proxy_url,
                "--feishu-hawor-base-url",
                proxy_url,
            ]
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(args.repo_root.resolve()), child_env.get("PYTHONPATH", "")) if value
        )
        # The client reads this after completion to retain the proxy scheduler's
        # real release sequence. It is an observer path, not a service input.
        child_env["EGO_ANNOTATION_ADMISSION_EVENTS_PATH"] = str(args.events_path.resolve())
        proc = subprocess.run(command, cwd=str(args.repo_root), env=child_env, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)
    if args.package_root is not None:
        run_root = _run_root_from_command(command)
        if run_root is None:
            print(json.dumps({"status": "error", "error": "--package-root requires child --run-root"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            package = create_result_package(run_root, args.package_root, package_name=args.package_name)
        except (PackageError, OSError, ValueError) as exc:
            print(json.dumps({"status": "error", "error": f"result packaging failed: {exc}"}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({
            "status": "ok",
            "run_root": str(run_root.resolve()),
            "package_path": package.get("package_path"),
            "report_path": package.get("final_report_path"),
            "video_path": package.get("final_video_path"),
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
