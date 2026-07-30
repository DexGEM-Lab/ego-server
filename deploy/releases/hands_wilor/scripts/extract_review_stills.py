from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def frame_at(path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} from {path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="still")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in args.frames:
        frame = frame_at(args.video, frame_idx)
        out = args.output_dir / f"{args.prefix}_{frame_idx:04d}.jpg"
        if not cv2.imwrite(str(out), frame):
            raise RuntimeError(f"failed to write {out}")
        print(out)


if __name__ == "__main__":
    main()
