#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_mapping(path: Path) -> dict[int, int]:
    payload = load_json(path)
    raw = payload.get("local_frame_mapping")
    if not isinstance(raw, dict):
        raise RuntimeError(f"{path} has no local_frame_mapping object")
    mapping = {int(k): int(v) for k, v in raw.items()}
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("source frame mapping is not one-to-one")
    return mapping


def run(args: argparse.Namespace) -> dict:
    track = load_json(args.sam2_track_json)
    mapping = parse_mapping(args.point_prompts)
    mask_dir = args.output_dir / "sam2_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    remapped: dict[str, dict] = {}
    rows = []
    for local_frame, source_frame in sorted(mapping.items()):
        key = str(local_frame)
        if key not in track:
            raise RuntimeError(f"SAM2 track missing local frame {local_frame}")
        row = dict(track[key])
        if row.get("visible"):
            src_mask = Path(str(row["mask_path"]))
            if not src_mask.exists() and args.local_mask_dir is not None:
                src_mask = args.local_mask_dir / f"{local_frame:06d}.png"
            if not src_mask.exists():
                raise FileNotFoundError(src_mask)
            dst_mask = mask_dir / f"{source_frame:06d}.png"
            shutil.copy2(src_mask, dst_mask)
            row["mask_path"] = str(dst_mask)
        row["local_frame_idx"] = int(local_frame)
        row["source_frame_idx"] = int(source_frame)
        remapped[str(source_frame)] = row
        rows.append(row)
    output = {
        "status": "ok",
        "method": "remap_sam2_track_to_source_frames_v7",
        "source_track_json": str(args.sam2_track_json),
        "point_prompts": str(args.point_prompts),
        "frame_start": int(min(mapping.values())),
        "frame_end": int(max(mapping.values())),
        "frames": int(len(mapping)),
        "visible_frames": int(sum(1 for row in rows if bool(row.get("visible")))),
        "track": remapped,
    }
    save_json(args.output_json, output)
    if args.output_track_json is not None:
        save_json(args.output_track_json, remapped)
    print(json.dumps({k: output[k] for k in ("status", "frames", "visible_frames", "frame_start", "frame_end")}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-track-json", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-track-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-mask-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
