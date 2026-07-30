#!/usr/bin/env python3
"""Write a copy of a frame manifest containing only selected frame indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated frame_idx values.")
    args = parser.parse_args()
    payload: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    wanted = {int(v) for v in args.frames.split(",") if v.strip()}
    frames = [row for row in payload.get("frames", []) if int(row.get("frame_idx", row.get("index", -1))) in wanted]
    if len(frames) != len(wanted):
        found = {int(row.get("frame_idx", row.get("index", -1))) for row in frames}
        missing = sorted(wanted - found)
        raise RuntimeError(f"selected frames missing from manifest: {missing}")
    payload["frames"] = frames
    payload["filtered_from"] = str(args.input)
    payload["filter_frame_idx"] = sorted(wanted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "frame_count": len(frames)}, indent=2))


if __name__ == "__main__":
    main()
