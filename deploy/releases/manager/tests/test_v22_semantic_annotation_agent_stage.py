from __future__ import annotations

from scripts.run_v22_semantic_annotation_agent_stage import analysis_grid, points_to_rows, validate_points


def test_semantic_agent_grid_is_two_fps_native_frame_indices() -> None:
    grid = analysis_grid(frame_count=120, fps=30.0, analysis_fps=2.0)
    assert len(grid) == 8
    assert grid[0] == {"analysis_index": 0, "frame_idx": 0, "time_sec": 0.0}
    assert grid[1] == {"analysis_index": 1, "frame_idx": 15, "time_sec": 0.5}
    assert grid[-1] == {"analysis_index": 7, "frame_idx": 105, "time_sec": 3.5}


def test_semantic_points_normalize_to_caption_rows() -> None:
    grid = analysis_grid(frame_count=60, fps=30.0, analysis_fps=2.0)
    raw = {
        "points": [
            {
                "analysis_index": point["analysis_index"],
                "frame_idx": point["frame_idx"],
                "time_sec": point["time_sec"],
                "left_hand": {"in_frame": "yes", "contact": "yes", "object": "cloth", "rigidity": "flexible", "assembly": "no", "contact_location": "edge"},
                "right_hand": {"in_frame": "yes", "contact": "no", "object": "none", "rigidity": "unknown", "assembly": "unknown", "contact_location": "none"},
                "understanding": "left hand holds cloth while right hand is visible",
            }
            for point in grid
        ]
    }
    points = validate_points(raw, grid)
    rows = points_to_rows(points, frame_count=60, fps=30.0)
    assert len(rows) == len(grid)
    assert rows[0]["start_frame"] == 0
    assert rows[0]["end_frame"] == 15
    assert rows[-1]["start_frame"] == 45
    assert rows[-1]["end_frame"] == 60
    assert rows[0]["caption"] == "left hand holds cloth while right hand is visible"
    assert rows[0]["per_hand"]["left"]["action_state"] == "contacting_or_operating_object"
    assert rows[0]["per_hand"]["left"]["object_rigidity"] == "flexible"
