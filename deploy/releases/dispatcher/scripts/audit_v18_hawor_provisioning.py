#!/usr/bin/env python3
"""Audit local HaWoR provisioning for V18.

This does not install HaWoR or substitute another hand model. It records whether
required HaWoR repo/weights and MANO assets exist in configured and plausible
local paths.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "bytes": st.st_size if path.is_file() else None,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(st.st_mtime)),
    }


def search_exact(roots: list[Path], name: str, max_hits: int, max_seconds: float) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    start = time.perf_counter()
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob(name):
                s = str(path)
                if "/frames/" in s or "/world_frames/" in s or "/overlay_frames/" in s or "/corrective_montage/frames/" in s:
                    continue
                hits.append(file_info(path))
                if len(hits) >= max_hits or (time.perf_counter() - start) > max_seconds:
                    return hits
        except (PermissionError, OSError):
            continue
        if (time.perf_counter() - start) > max_seconds:
            return hits
    return hits


def search_dirs(roots: list[Path], dirname: str, max_hits: int, max_seconds: float) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    start = time.perf_counter()
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob(dirname):
                if path.is_dir():
                    hits.append(file_info(path))
                    if len(hits) >= max_hits or (time.perf_counter() - start) > max_seconds:
                        return hits
        except (PermissionError, OSError):
            continue
        if (time.perf_counter() - start) > max_seconds:
            return hits
    return hits


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    roots = [p for p in args.search_roots if p.exists()]
    hawor_repo = args.hawor_root
    required = {
        "configured_hawor_repo": file_info(hawor_repo),
        "configured_hawor_git": file_info(hawor_repo / ".git"),
        "configured_hawor_checkpoint": file_info(hawor_repo / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"),
        "configured_infiller_weight": file_info(hawor_repo / "weights" / "hawor" / "checkpoints" / "infiller.pt"),
        "configured_model_config": file_info(hawor_repo / "weights" / "hawor" / "model_config.yaml"),
        "configured_mano_right": file_info(args.mano_right),
        "configured_mano_left": file_info(args.mano_left),
    }
    searches = {
        "MANO_LEFT.pkl": search_exact(roots, "MANO_LEFT.pkl", args.max_hits, args.max_search_seconds),
        "MANO_RIGHT.pkl": search_exact(roots, "MANO_RIGHT.pkl", args.max_hits, args.max_search_seconds),
        "hawor.ckpt": search_exact(roots, "hawor.ckpt", args.max_hits, args.max_search_seconds),
        "infiller.pt": search_exact(roots, "infiller.pt", args.max_hits, args.max_search_seconds),
        "HaWoR_dirs": search_dirs(roots, "HaWoR", args.max_hits, args.max_search_seconds),
    }
    missing_required = [name for name, info in required.items() if not info.get("exists")]
    status = "blocked_missing_required_hawor_assets" if missing_required else "provisioning_files_present_not_execution_validated"
    report = {
        "method": "audit_v18_hawor_provisioning",
        "status": status,
        "claim_scope": "local_filesystem_provisioning_audit_only_no_hawor_execution_no_model_substitution",
        "configured_paths": {
            "hawor_root": str(args.hawor_root),
            "mano_right": str(args.mano_right),
            "mano_left": str(args.mano_left),
            "search_roots": [str(p) for p in roots],
        },
        "required": required,
        "missing_required": missing_required,
        "search_hits": searches,
        "interpretation": "Task5 HaWoR cannot be run locally unless HaWoR repo/weights and MANO_LEFT.pkl are provisioned; WiLoR MANO_RIGHT alone is insufficient." if missing_required else "Required files are present; execution still requires environment/GPU validation.",
        "elapsed_s": time.perf_counter() - start,
    }
    out = args.output_root / "hawor_provisioning_audit" / "v18_hawor_provisioning_audit_report.json"
    write_json(out, report)
    md = args.output_root / "hawor_provisioning_audit" / "V18_HAWOR_PROVISIONING_AUDIT.md"
    lines = [
        "# V18 HaWoR provisioning audit",
        "",
        "This is local provisioning evidence only. It does not run HaWoR and does not substitute another hand model.",
        "",
        f"Status: `{status}`",
        f"Missing required: `{missing_required}`",
        "",
        "## Required paths",
    ]
    for name, info in required.items():
        lines.append(f"- {name}: `{info['path']}` exists=`{info['exists']}`")
    lines += ["", "## Search hits"]
    for name, hits in searches.items():
        lines.append(f"- {name}: {len(hits)} hit(s)")
        for hit in hits[: args.max_hits]:
            lines.append(f"  - `{hit['path']}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--hawor-root", type=Path, default=Path(os.environ.get("EGO_HAWOR_REPO", "/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/third_party/HaWoR")))
    parser.add_argument("--mano-right", type=Path, default=Path(os.environ.get("EGO_MANO_RIGHT", "/home/yiwen/ego_annotation/third_party/WiLoR/mano_data/MANO_RIGHT.pkl")))
    parser.add_argument("--mano-left", type=Path, default=Path(os.environ.get("EGO_MANO_LEFT", "/home/yiwen/ego_annotation/third_party/WiLoR/mano_data/MANO_LEFT.pkl")))
    parser.add_argument("--search-roots", nargs="+", type=Path, default=[Path("/home/yiwen/ego_annotation"), Path("/home/yiwen"), Path("/mnt/user-home/yiwen"), Path("/data2")])
    parser.add_argument("--max-hits", type=int, default=20)
    parser.add_argument("--max-search-seconds", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
