from __future__ import annotations

import json
from pathlib import Path

from scripts.run_v22_resident_unidepth_batch import iter_rows, resize_hw


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_item(tmp_path: Path, item_id: str, frame_count: int) -> Path:
    run_root = tmp_path / item_id
    frames = []
    for idx in range(frame_count):
        rgb = run_root / "input" / "raw_frame_manifest" / "rgb" / f"{idx:06d}.jpg"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"fake-jpeg")
        frames.append(
            {
                "frame_idx": idx,
                "time_s": idx / 30.0,
                "rgb": str(rgb),
                "source_width": 640,
                "source_height": 480,
            }
        )
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {"schema": "v22_raw_frame_manifest.v0", "frame_count": frame_count, "frames": frames},
    )
    return run_root


def test_iter_rows_preserves_item_and_batch_boundaries(tmp_path: Path) -> None:
    run_a = make_item(tmp_path, "item_a", 3)
    run_b = make_item(tmp_path, "item_b", 2)
    request = {
        "job_id": "job_u",
        "stage_id": "unidepth_v2_depth_resident",
        "batch_size": 2,
        "agent_id": "agent_7",
        "attempt_id": "attempt_0003",
        "items": [
            {"item_id": "item_a", "run_root": str(run_a), "max_frames": 3},
            {"item_id": "item_b", "run_root": str(run_b), "max_frames": 2},
        ],
    }
    rows = iter_rows(request)
    assert len(rows) == 5
    assert [row["item_id"] for row in rows] == ["item_a", "item_a", "item_a", "item_b", "item_b"]
    assert [row["batch_id"] for row in rows] == [
        "job_u_unidepth_v2_depth_resident_batch_00000",
        "job_u_unidepth_v2_depth_resident_batch_00000",
        "job_u_unidepth_v2_depth_resident_batch_00001",
        "job_u_unidepth_v2_depth_resident_batch_00001",
        "job_u_unidepth_v2_depth_resident_batch_00002",
    ]
    assert {row["job_id"] for row in rows} == {"job_u"}
    assert {row["stage_id"] for row in rows} == {"unidepth_v2_depth_resident"}
    assert {row["agent_id"] for row in rows} == {"agent_7"}
    assert {row["attempt_id"] for row in rows} == {"attempt_0003"}
    assert all(Path(row["rgb_path"]).exists() for row in rows)


def test_resize_hw_keeps_even_height() -> None:
    assert resize_hw(321, 640, 480) == (321, 242)
