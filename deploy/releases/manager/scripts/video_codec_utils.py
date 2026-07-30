#!/usr/bin/env python3
"""Video writing helpers for browser/VSCode-compatible MP4 outputs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2


class H264VideoWriter:
    """OpenCV-style writer that finalizes MP4 files as H.264/AVC.

    OpenCV's portable MP4 writer commonly emits MPEG-4 Part 2 (mp4v), which is
    a valid MP4 but often fails in VSCode Remote/Chromium preview. This class
    writes an intermediate OpenCV file, then transcodes the final path to H.264
    (avc1), yuv420p, with faststart metadata.
    """

    def __init__(self, output_path: str | Path, fps: float, frame_size: tuple[int, int], *, crf: int = 20) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.crf = int(crf)
        self._tmp_path = self.output_path.with_name(f".{self.output_path.stem}.opencv_mp4v_tmp_{os.getpid()}.mp4")
        self._writer = cv2.VideoWriter(
            str(self._tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(frame_size[0]), int(frame_size[1])),
        )
        self._released = False

    def isOpened(self) -> bool:  # noqa: N802 - match cv2.VideoWriter API
        return bool(self._writer.isOpened())

    def write(self, frame) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._writer.release()
        if not self._tmp_path.exists() or self._tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"intermediate_video_missing_or_empty: {self._tmp_path}")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self._tmp_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        finally:
            self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "H264VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def video_codec_summary(path: str | Path) -> dict[str, str]:
    """Return ffprobe codec metadata for verification."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string,pix_fmt,width,height,duration,nb_frames",
        "-of",
        "default=nw=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    result: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result
