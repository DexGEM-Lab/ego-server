#!/usr/bin/env python3
"""Upload helpers for the annotation API service."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import AsyncIterable, Iterable


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class UploadError(RuntimeError):
    pass


def clean_upload_filename(raw: str | None, *, default: str) -> str:
    name = Path(raw or default).name
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._-")
    if not cleaned:
        cleaned = default
    suffix = Path(cleaned).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        cleaned = f"{Path(cleaned).stem or 'video'}.mp4"
    return cleaned[:128]


def upload_destination(output_root: Path, job_id: str, filename: str) -> Path:
    upload_dir = output_root.expanduser().resolve() / "_uploads" / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / clean_upload_filename(filename, default=f"{job_id}.mp4")


def write_upload_chunks(chunks: Iterable[bytes], destination: Path) -> dict[str, int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as handle:
        for chunk in chunks:
            if not chunk:
                continue
            handle.write(chunk)
            size += len(chunk)
    if size <= 0:
        raise UploadError("empty_upload")
    return {"path": str(destination), "size_bytes": size}


async def write_upload_stream(chunks: AsyncIterable[bytes], destination: Path) -> dict[str, int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as handle:
        async for chunk in chunks:
            if not chunk:
                continue
            handle.write(chunk)
            size += len(chunk)
    if size <= 0:
        raise UploadError("empty_upload")
    return {"path": str(destination), "size_bytes": size}
