#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args: argparse.Namespace) -> dict:
    rows = json.loads(args.input_track.read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise RuntimeError(f"mask track must be a frame-keyed object: {args.input_track}")
    out = {}
    for frame, row in rows.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid row for frame {frame}")
        item = dict(row)
        raw = item.get("mask_path")
        if raw:
            name = Path(str(raw)).name
            local = args.local_mask_dir / name
            if not local.exists():
                raise RuntimeError(f"localized mask path missing for frame {frame}: {local}")
            item["mask_path"] = str(local)
            item["remote_mask_path"] = str(raw)
        out[str(int(frame))] = item
    args.output_track.parent.mkdir(parents=True, exist_ok=True)
    args.output_track.write_text(json.dumps(out, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "localize_mask_track_paths_v5",
        "input_track": str(args.input_track),
        "output_track": str(args.output_track),
        "local_mask_dir": str(args.local_mask_dir),
        "frames": int(len(out)),
        "visible_frames": int(sum(1 for row in out.values() if row.get("visible"))),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-track", type=Path, required=True)
    parser.add_argument("--local-mask-dir", type=Path, required=True)
    parser.add_argument("--output-track", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
