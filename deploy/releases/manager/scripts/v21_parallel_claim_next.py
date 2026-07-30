#!/usr/bin/env python3
"""Atomically claim and update V21 parallel batch entries."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_event(manifest: Path, event: dict[str, Any]) -> None:
    log_dir = manifest.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runner_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


@contextmanager
def locked_manifest(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        manifest = load_json(path)
        yield manifest
        write_json_atomic(path, manifest)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def find_entry(entries: list[dict[str, Any]], case_id: str | None, data_entry_id: str | None) -> dict[str, Any]:
    for entry in entries:
        if case_id and entry.get("case_id") == case_id:
            return entry
        if data_entry_id and entry.get("data_entry_id") == data_entry_id:
            return entry
    raise RuntimeError(f"entry not found: case_id={case_id!r} data_entry_id={data_entry_id!r}")


def summarize(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "queued"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runner-id", required=True)
    p.add_argument("--claim", action="store_true")
    p.add_argument("--complete", action="store_true")
    p.add_argument("--fail", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--case-id", default=None)
    p.add_argument("--data-entry-id", default=None)
    p.add_argument("--reason", default="")
    p.add_argument("--artifact", action="append", default=[])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    actions = [args.claim, args.complete, args.fail, args.status]
    if sum(bool(x) for x in actions) != 1:
        raise SystemExit("choose exactly one of --claim, --complete, --fail, or --status")
    manifest_path = args.manifest.expanduser()

    with locked_manifest(manifest_path) as manifest:
        entries_raw = manifest.get("entries")
        if not isinstance(entries_raw, list):
            raise RuntimeError(f"manifest has no entries list: {manifest_path}")
        entries: list[dict[str, Any]] = entries_raw
        now = utc_now()
        event: dict[str, Any]
        result: dict[str, Any]

        if args.status:
            result = {"status": "ok", "counts": summarize(entries), "entries": len(entries)}
            print(json.dumps(result, sort_keys=True))
            return 0

        if args.claim:
            candidates = [entry for entry in entries if str(entry.get("status", "queued")) == "queued"]
            candidates.sort(key=lambda e: (int(e.get("priority", 0)), str(e.get("data_entry_id", ""))))
            if not candidates:
                result = {"status": "empty", "counts": summarize(entries)}
                event = {"event": "claim_empty", "runner_id": args.runner_id, "at": now, "counts": result["counts"]}
                append_event(manifest_path, event)
                print(json.dumps(result, sort_keys=True))
                return 0
            entry = candidates[0]
            entry["status"] = "running"
            entry["runner_id"] = args.runner_id
            entry["claimed_at"] = now
            entry["claim_pid"] = os.getpid()
            event = {"event": "claimed", "runner_id": args.runner_id, "at": now, "data_entry_id": entry.get("data_entry_id"), "case_id": entry.get("case_id")}
            append_event(manifest_path, event)
            result = {"status": "claimed", "entry": entry}
            print(json.dumps(result, sort_keys=True))
            return 0

        entry = find_entry(entries, args.case_id, args.data_entry_id)
        if args.complete:
            entry["status"] = "completed"
            entry["completed_at"] = now
            entry["completed_by"] = args.runner_id
            if args.artifact:
                entry["artifacts"] = args.artifact
            event = {"event": "completed", "runner_id": args.runner_id, "at": now, "data_entry_id": entry.get("data_entry_id"), "case_id": entry.get("case_id"), "artifacts": args.artifact}
            append_event(manifest_path, event)
            print(json.dumps({"status": "completed", "entry": entry}, sort_keys=True))
            return 0

        if args.fail:
            entry["status"] = "failed"
            entry["failed_at"] = now
            entry["failed_by"] = args.runner_id
            entry["failure_reason"] = args.reason or "unspecified_failure"
            event = {"event": "failed", "runner_id": args.runner_id, "at": now, "data_entry_id": entry.get("data_entry_id"), "case_id": entry.get("case_id"), "reason": entry["failure_reason"]}
            append_event(manifest_path, event)
            print(json.dumps({"status": "failed", "entry": entry}, sort_keys=True))
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
