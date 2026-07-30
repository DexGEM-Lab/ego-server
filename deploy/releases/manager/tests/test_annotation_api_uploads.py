from __future__ import annotations

import asyncio
from pathlib import Path

from scripts.annotation_api_uploads import UploadError, clean_upload_filename, upload_destination, write_upload_chunks, write_upload_stream


def test_clean_upload_filename_keeps_video_suffix() -> None:
    assert clean_upload_filename("../clip.mov", default="job.mp4") == "clip.mov"
    assert clean_upload_filename("bad name?.txt", default="job.mp4") == "bad_name_.mp4"
    assert clean_upload_filename("", default="job.mp4") == "job.mp4"


def test_upload_destination_is_scoped_by_job(tmp_path: Path) -> None:
    path = upload_destination(tmp_path, "job_001", "../clip.mp4")
    assert path == (tmp_path / "_uploads" / "job_001" / "clip.mp4").resolve()
    info = write_upload_chunks([b"abc", b"", b"def"], path)
    assert info["size_bytes"] == 6
    assert path.read_bytes() == b"abcdef"


def test_empty_upload_rejected(tmp_path: Path) -> None:
    path = upload_destination(tmp_path, "job_002", "clip.mp4")
    try:
        write_upload_chunks([b""], path)
    except UploadError as exc:
        assert str(exc) == "empty_upload"
    else:
        raise AssertionError("empty upload accepted")


def test_async_upload_stream_writes_incrementally(tmp_path: Path) -> None:
    async def chunks():
        yield b"ab"
        yield b""
        yield b"cd"

    path = upload_destination(tmp_path, "job_003", "clip.mp4")
    info = asyncio.run(write_upload_stream(chunks(), path))
    assert info["size_bytes"] == 4
    assert path.read_bytes() == b"abcd"
