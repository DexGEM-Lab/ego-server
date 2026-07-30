"""Build or verify an immutable recovered DROID source release without model loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.droid_source import (
    EXPECTED_CORE_GROUP_DIGEST,
    RECOVERED_AMENDMENT_ID,
    build_droid_source_release,
    verify_droid_source_release,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--candidate-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--origin-evidence-json", type=Path, required=True)
    build.add_argument("--amendment-id", default=RECOVERED_AMENDMENT_ID)
    build.add_argument("--expected-core-group-digest", default=EXPECTED_CORE_GROUP_DIGEST)
    verify = sub.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--expected-digest")
    verify.add_argument("--amendment-id", default=RECOVERED_AMENDMENT_ID)
    verify.add_argument("--expected-core-group-digest", default=EXPECTED_CORE_GROUP_DIGEST)
    args = parser.parse_args(argv)
    if args.command == "build":
        origin = json.loads(args.origin_evidence_json.read_text(encoding="utf-8"))
        release = build_droid_source_release(
            args.candidate_root, args.output_root, origin_evidence=origin,
            amendment_id=args.amendment_id, expected_core_group_digest=args.expected_core_group_digest,
        )
    else:
        release = verify_droid_source_release(
            args.release, expected_digest=args.expected_digest, expected_amendment_id=args.amendment_id,
            expected_core_group_digest=args.expected_core_group_digest,
        )
    print(json.dumps({
        "schema": "ego.droid-source-release-result.v1", "path": str(release.path),
        "droid_slam_root": str(release.source_root), "source_digest": release.source_digest,
        "amendment_id": release.amendment_id, "core_group_digest": release.core_group_digest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
