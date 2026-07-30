#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fuse_v1_full_fidelity import read_video_frame
from run_v1_wilor_colmap import open_video


COLORS = [
    (40, 220, 255),
    (70, 170, 255),
    (80, 255, 120),
    (255, 190, 70),
    (255, 90, 210),
    (180, 120, 255),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def localize(path: str, remote_root: Path, local_root: Path) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    text = str(raw)
    prefix = str(remote_root)
    if text.startswith(prefix):
        candidate = local_root / text[len(prefix) :].lstrip("/")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_tracks(root: Path, remote_root: Path, local_root: Path) -> list[tuple[str, dict]]:
    tracks = []
    for track_dir in sorted(root.iterdir()):
        track_json = track_dir / "sam2" / "sam2_track.json"
        if track_json.exists():
            tracks.append((track_dir.name, load_json(track_json)))
    if not tracks:
        raise RuntimeError(f"no sam2_track.json files found under {root}")
    return tracks


def mask_entry(track: dict, frame_idx: int) -> dict | None:
    entry = track.get(str(frame_idx))
    if not isinstance(entry, dict) or not entry.get("visible"):
        return None
    path = entry.get("mask_path")
    if not path:
        return None
    return entry


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    sel = mask > 0
    out = image.copy()
    out[sel] = (out[sel].astype(np.float32) * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def put_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def panel_for_track(
    raw: np.ndarray,
    track_id: str,
    track: dict,
    frame_idx: int,
    color: tuple[int, int, int],
    render_size: tuple[int, int],
    remote_root: Path,
    local_root: Path,
) -> np.ndarray:
    panel = cv2.resize(raw, render_size, interpolation=cv2.INTER_AREA)
    entry = mask_entry(track, frame_idx)
    if entry is None:
        put_label(panel, f"{frame_idx} {track_id}: no accepted mask")
        return panel
    mask_path = localize(str(entry["mask_path"]), remote_root, local_root)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    panel = overlay_mask(panel, mask, color, 0.50)
    area = int((mask > 0).sum())
    put_label(panel, f"{frame_idx} {track_id}: area={area}")
    return panel


def run(args: argparse.Namespace) -> dict:
    tracks = load_tracks(args.sam2_root, args.remote_root, args.local_root)
    frame_set = set()
    for _, track in tracks:
        for key, entry in track.items():
            if isinstance(entry, dict) and entry.get("visible"):
                frame_set.add(int(key))
    if args.frames:
        frames = [int(v) for v in args.frames]
    else:
        frames = sorted(frame_set)
    if not frames:
        raise RuntimeError("no visible mask frames found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(args.clip)
    render_h = int(round(args.render_width * info.height / info.width))
    render_size = (int(args.render_width), render_h)
    written = []
    try:
        for frame_idx in frames:
            raw = read_video_frame(cap, frame_idx)
            raw_panel = cv2.resize(raw, render_size, interpolation=cv2.INTER_AREA)
            put_label(raw_panel, f"{frame_idx} raw")
            panels = [raw_panel]
            for i, (track_id, track) in enumerate(tracks):
                panels.append(
                    panel_for_track(
                        raw,
                        track_id,
                        track,
                        frame_idx,
                        COLORS[i % len(COLORS)],
                        render_size,
                        args.remote_root,
                        args.local_root,
                    )
                )
            rows = []
            for start in range(0, len(panels), args.columns):
                row = panels[start : start + args.columns]
                while len(row) < args.columns:
                    row.append(np.zeros_like(raw_panel))
                rows.append(np.hstack(row))
            grid = np.vstack(rows)
            out = args.output_dir / f"sam2_surface_mask_grid_{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write {out}")
            written.append(str(out))
    finally:
        cap.release()
    result = {"status": "ok", "frames": frames, "tracks": [t[0] for t in tracks], "written": written}
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--remote-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
