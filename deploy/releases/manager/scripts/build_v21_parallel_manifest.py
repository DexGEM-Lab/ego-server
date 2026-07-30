#!/usr/bin/env python3
"""Build a V21 parallel batch manifest from a dataset root or file list."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
EGOSCALE_30H_CANDIDATES = (
    "egoscale 30h",
    "egoscale_30h",
    "egoscale-30h",
    "egoscale30h",
    "EgoScale 30h",
    "EgoScale_30h",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_default_egoscale_root(base: Path) -> Path:
    for name in EGOSCALE_30H_CANDIDATES:
        candidate = base / name
        if candidate.exists():
            return candidate
    normalized_targets = {re.sub(r"[^a-z0-9]+", "", name.lower()) for name in EGOSCALE_30H_CANDIDATES}
    if base.exists():
        for child in sorted(base.iterdir()):
            if child.is_dir() and re.sub(r"[^a-z0-9]+", "", child.name.lower()) in normalized_targets:
                return child
    return base / EGOSCALE_30H_CANDIDATES[0]


def safe_id(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("._-")
    return out or "entry"


def unique_case_id(raw: str, used: set[str]) -> str:
    base = safe_id(raw)
    if base not in used:
        used.add(base)
        return base
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}_{digest}"
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{base}_{digest}_{suffix}"
    used.add(candidate)
    return candidate


def read_manifest_input(path: Path) -> list[Path]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            blob = load_json(path)
            rows = blob.get("entries", blob) if isinstance(blob, dict) else blob
        if not isinstance(rows, list):
            raise RuntimeError(f"manifest input must contain a list or entries list: {path}")
        out: list[Path] = []
        for row in rows:
            if isinstance(row, str):
                out.append(Path(row).expanduser())
            elif isinstance(row, dict):
                raw = row.get("input_video") or row.get("video") or row.get("path")
                if raw:
                    out.append(Path(str(raw)).expanduser())
        return out
    return [Path(line.strip()).expanduser() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def discover_videos(data_root: Path, recursive: bool) -> list[Path]:
    if data_root.is_file():
        if data_root.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"data root is a file but not a supported video: {data_root}")
        return [data_root]
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    iterator = data_root.rglob("*") if recursive else data_root.glob("*")
    videos = [p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda p: str(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None, help="Dataset root. If omitted, resolve EgoScale 30h under --data-base.")
    p.add_argument("--data-base", type=Path, default=Path("~/data"), help="Base directory used to resolve the EgoScale 30h default.")
    p.add_argument("--manifest-input", type=Path, default=None, help="Optional text/json/jsonl file with explicit input videos.")
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--parallelism", type=int, default=64)
    p.add_argument("--case-prefix", default="egoscale30h")
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-entries", type=int, default=0)
    p.add_argument("--run-root-template", default="{batch_root}/entries/{case_id}")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.expanduser()
    output = args.output.expanduser() if args.output else batch_root / "batch_manifest.json"
    data_root = args.data_root.expanduser() if args.data_root else resolve_default_egoscale_root(args.data_base.expanduser())

    videos = read_manifest_input(args.manifest_input.expanduser()) if args.manifest_input else discover_videos(data_root, args.recursive)
    if args.max_entries and args.max_entries > 0:
        videos = videos[: args.max_entries]
    if not videos:
        raise RuntimeError(f"no videos found for V21 parallel manifest under {data_root}")

    used_case_ids: set[str] = set()
    entries: list[dict[str, Any]] = []
    for idx, video in enumerate(videos):
        try:
            rel = video.resolve().relative_to(data_root.resolve())
            raw_case = str(rel.with_suffix(""))
        except Exception:
            raw_case = video.stem
        case_id = unique_case_id(f"{args.case_prefix}_{raw_case}", used_case_ids)
        run_root = args.run_root_template.format(batch_root=str(batch_root), case_id=case_id, index=idx)
        entries.append(
            {
                "data_entry_id": f"entry_{idx:06d}",
                "case_id": case_id,
                "input_video": str(video),
                "run_root": run_root,
                "status": "queued",
                "priority": 0,
            }
        )

    payload = {
        "schema": "v21_parallel_batch_manifest.v1",
        "created_at": utc_now(),
        "data_root": str(data_root),
        "batch_root": str(batch_root),
        "parallelism": int(args.parallelism),
        "dataset": {
            "name": "egoscale_30h",
            "location_hint": "remote zjh@115.190.235.210:~/data under EgoScale 30h dataset",
        },
        "runner_contract": {
            "prompt_template": ".pi/prompts/v21_parallel_runner.md",
            "gpu_wrapper": "scripts/v21_gpu_wrapper.py",
            "pipeline": "pipeline.md",
        },
        "entries": entries,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"status": "ok", "manifest": str(output), "entries": len(entries), "parallelism": int(args.parallelism)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
