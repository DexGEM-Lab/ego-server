#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from fuse_v1_full_fidelity import read_video_frame
from run_v1_wilor_colmap import open_video


def draw_prompt(image, row: dict, track_id: str) -> None:
    for point in row.get("positive_points", []):
        xy = (int(round(float(point["x"]))), int(round(float(point["y"]))))
        cv2.circle(image, xy, 8, (0, 255, 0), -1)
        cv2.circle(image, xy, 10, (0, 80, 0), 2)
    for point in row.get("negative_points", []):
        xy = (int(round(float(point["x"]))), int(round(float(point["y"]))))
        cv2.circle(image, xy, 8, (0, 0, 255), 2)
    bbox = row.get("bbox_xyxy") or []
    if len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    label = f"{int(row['frame_idx'])} {track_id} visible={bool(row['target_visible'])} conf={float(row['confidence']):.2f}"
    cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict:
    payload = json.loads(args.point_prompts.read_text(encoding="utf-8"))
    track_id = str(payload["track_id"])
    rows = payload.get("point_prompts")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"point prompt file has no rows: {args.point_prompts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    written = []
    try:
        for row in rows:
            frame_idx = int(row["frame_idx"])
            image = read_video_frame(cap, frame_idx)
            image = cv2.resize(image, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
            draw_prompt(image, row, track_id)
            out = args.output_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(out), image):
                raise RuntimeError(f"failed to write {out}")
            written.append(str(out))
    finally:
        cap.release()
    result = {"status": "ok", "track_id": track_id, "frames": len(written), "output_dir": str(args.output_dir)}
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
