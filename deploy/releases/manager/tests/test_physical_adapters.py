from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ego_annotation.full_video_timeline import InMemoryFrameSource
from ego_annotation.physical_adapters import HAND_EDGES, PhysicalAdapterError, PhysicalArtifactAdapter, _camera_centric_world_view, _build_world_view, _build_world_views, _draw_camera, _draw_projected_hand, _draw_projected_hand_2d, _project_world, _world_canvas, _world_to_camera_display, project_points, transform_points


def test_transform_points_applies_world_from_camera_pose() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (2.0, -1.0, 0.5)

    result = transform_points(points, pose)

    np.testing.assert_allclose(result, [[3.0, 1.0, 3.5], [2.0, -1.0, 1.5]])


def test_project_points_keeps_depth_validity() -> None:
    points = np.asarray([[1.0, 2.0, 2.0], [1.0, 1.0, -1.0]], dtype=np.float32)
    k = np.asarray([[100.0, 0.0, 10.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    result = project_points(points, k)

    np.testing.assert_allclose(result[0], [60.0, 120.0, 1.0])
    assert result[1, 2] == 0.0


def test_transform_rejects_bad_pose() -> None:
    with pytest.raises(PhysicalAdapterError):
        transform_points(np.zeros((2, 3), dtype=np.float32), np.eye(3, dtype=np.float32))


def test_world_canvas_skips_unobserved_nan_points() -> None:
    points = [
        np.asarray([[np.nan, np.nan, np.nan], [0.1, 0.2, 0.3]], dtype=np.float32),
        np.full((2, 3), np.nan, dtype=np.float32),
    ]
    camera_centers = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.2]], dtype=np.float32)

    canvas = _world_canvas(points, camera_centers, (64, 64))

    assert canvas.shape == (64, 64, 3)
    assert canvas.dtype == np.uint8


def test_projected_hand_draws_mano_bones_and_vertex_outline() -> None:
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    joints = np.column_stack((np.linspace(-0.08, 0.08, 21), np.linspace(-0.05, 0.05, 21), np.ones(21))).astype(np.float32)
    vertices = np.column_stack((
        np.cos(np.linspace(0, 2 * np.pi, 778)) * 0.12,
        np.sin(np.linspace(0, 2 * np.pi, 778)) * 0.08,
        np.ones(778),
    )).astype(np.float32)
    k = np.asarray([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    counts = _draw_projected_hand(canvas, vertices, joints, k, (76, 220, 92))

    assert counts["skeleton_edges"] == len(HAND_EDGES)
    assert counts["mesh_outlines"] == 1
    assert counts["surface_vertices"] > 100
    assert np.count_nonzero(canvas) > 500


def test_direct_crop_projection_draws_full_sized_hand_away_from_image_center_bug() -> None:
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    joints = np.column_stack((np.linspace(190.0, 290.0, 21), np.linspace(70.0, 180.0, 21))).astype(np.float32)
    vertices = np.column_stack((
        240.0 + np.cos(np.linspace(0, 2 * np.pi, 778)) * 55.0,
        125.0 + np.sin(np.linspace(0, 2 * np.pi, 778)) * 70.0,
    )).astype(np.float32)

    counts = _draw_projected_hand_2d(canvas, vertices, joints, (42, 205, 255))

    assert counts["skeleton_edges"] == len(HAND_EDGES)
    assert counts["mesh_outlines"] == 1
    ys, xs = np.nonzero(np.any(canvas != 0, axis=2))
    assert xs.min() > 150
    assert xs.max() - xs.min() > 90
    assert ys.max() - ys.min() > 100


def test_combined_render_is_the_only_video_and_contains_full_timeline(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    frame_count = 3
    source = InMemoryFrameSource(
        [np.full((120, 160, 3), 40 + index * 20, dtype=np.uint8) for index in range(frame_count)],
        fps=5.0,
        source_id="combined-render-fixture",
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 0.2, frame_count)
    valid = np.ones(frame_count, dtype=bool)
    joints = np.zeros((2, frame_count, 21, 3), dtype=np.float32)
    vertices = np.zeros((2, frame_count, 778, 3), dtype=np.float32)
    for side in range(2):
        joints[side, :, :, 0] = np.linspace(-0.05, 0.05, 21) + side * 0.08
        joints[side, :, :, 1] = np.linspace(-0.04, 0.04, 21)
        joints[side, :, :, 2] = 1.0
        vertices[side, :, :, 0] = np.cos(np.linspace(0.0, 2.0 * np.pi, 778)) * 0.07 + side * 0.08
        vertices[side, :, :, 1] = np.sin(np.linspace(0.0, 2.0 * np.pi, 778)) * 0.05
        vertices[side, :, :, 2] = 1.0
    tensor = lambda array, provenance=None: SimpleNamespace(array=array, provenance=provenance or {})
    coverage = SimpleNamespace(
        source_frame_count=frame_count,
        submitted_count=frame_count,
        unannotated_range=None,
        pose_valid=tuple(True for _ in range(frame_count)),
        pose_sampled=tuple(True for _ in range(frame_count)),
        to_wire=lambda: {"status": "complete", "reason": None},
    )
    state = SimpleNamespace(
        frame_count=frame_count,
        source_timeline=source.timeline,
        canonical_K=SimpleNamespace(k_canonical=np.asarray([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float32)),
        droid_records=SimpleNamespace(final=SimpleNamespace(output=SimpleNamespace(T_world_camera=tensor(poses, {"droid_pose_valid": valid}))), coverage=coverage),
        timeline_inference=SimpleNamespace(
            valid=tensor(np.ones((2, frame_count), dtype=bool)),
            vertices_camera_m=tensor(vertices),
            joints_camera_m=tensor(joints),
            uncertainty_m=tensor(np.full((2, frame_count), 0.01, dtype=np.float32)),
        ),
        semantic_rows=({"start_frame": 0, "end_frame": frame_count, "caption": "Cosmos says the hands manipulate the visible object.", "claim_scope": "semantic_only_not_physical_evidence"},),
        acceptance=SimpleNamespace(accepted=True, diagnostic_only=False, reasons=()),
        hawor_geometry_diagnostics={"status": "ok"},
    )

    result = PhysicalArtifactAdapter(render_size=(160, 120)).render(state, source, tmp_path)

    assert result.combined_video.endswith("renders/v22_combined.mp4")
    assert sorted(path.name for path in (tmp_path / "renders").glob("*.mp4")) == ["v22_combined.mp4"]
    capture = cv2.VideoCapture(result.combined_video)
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frame_count
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
    ok, rendered = capture.read()
    capture.release()
    assert ok and rendered is not None
    assert np.count_nonzero(rendered[:, :160]) > 1000
    assert np.count_nonzero(rendered[:, 160:]) > 1000
    report = json.loads((tmp_path / "renders" / "physical_adapter_report.json").read_text(encoding="utf-8"))
    assert report["output_video"] == result.combined_video
    assert report["pane_layout"] == {"left": "source_overlay_mano_and_cosmos_caption", "right": "metric_world_camera_and_mano"}
    assert report["semantic_render"]["coverage"]["fraction"] == 1.0
    with np.load(result.state_npz, allow_pickle=False) as physical:
        assert physical["frame_idx"].shape == (frame_count,)
        assert physical["left_vertices_world_m"].shape == (frame_count, 778, 3)
        assert np.count_nonzero(physical["left_valid"]) == frame_count


def test_world_canvas_is_perspective_3d_with_metric_grid_and_trajectory() -> None:
    camera = np.asarray([[0.0, 0.2, 0.0], [0.2, 0.25, 0.4], [0.5, 0.3, 0.8]], dtype=np.float32)
    joints = np.column_stack((np.linspace(0.1, 0.3, 21), np.linspace(0.0, 0.2, 21), np.linspace(0.5, 0.7, 21))).astype(np.float32)
    world_history = [joints[None, ...], np.full((1, 21, 3), np.nan, dtype=np.float32)]
    view = _build_world_view(camera, world_history)

    canvas = _world_canvas([joints, np.empty((0, 3), dtype=np.float32)], camera, (640, 360), view=view, frame_index=2)

    assert canvas.shape == (360, 640, 3)
    assert view.span >= 0.5
    assert not np.allclose(view.eye, view.target)
    assert np.count_nonzero(canvas != 28) > 1000


def test_local_world_view_rejects_remote_outlier_from_frame_fit() -> None:
    camera = np.column_stack((np.linspace(0.0, 1.0, 100), np.zeros(100), np.zeros(100)))
    camera[50] = (1000.0, 1000.0, 1000.0)
    hands = np.full((100, 21, 3), np.nan, dtype=np.float32)
    hands[:, :, 0] = camera[:, None, 0] + np.linspace(-0.1, 0.1, 21)
    hands[:, :, 1] = np.linspace(-0.15, 0.15, 21)
    hands[:, :, 2] = 1.0

    global_view = _build_world_view(camera, [hands], (1600, 900))
    local_view = _build_world_view(camera, [hands], (1600, 900), frame_index=0, window_frames=10)

    assert local_view.span < global_view.span / 10.0
    pixels, valid = _project_world(np.concatenate(([camera[0]], hands[0]), axis=0), local_view, (1600, 900))
    assert valid.all()
    assert np.all((pixels >= 0) & (pixels < (1600, 900)))


def test_local_world_views_include_current_hand_outside_previous_window() -> None:
    camera = np.zeros((100, 3), dtype=np.float64)
    hands = np.full((100, 21, 3), np.nan, dtype=np.float64)
    hands[90, :, 0] = 4.0 + np.linspace(-0.1, 0.1, 21)
    hands[90, :, 1] = np.linspace(-0.15, 0.15, 21)
    hands[90, :, 2] = 1.0

    view = _build_world_views(camera, [hands], (1600, 900), window_frames=10)[90]
    points, valid = _project_world(np.concatenate(([camera[90]], hands[90]), axis=0), view, (1600, 900))

    assert valid.all()
    assert np.all((points[:, 0] >= 0) & (points[:, 0] < 1600))
    assert np.all((points[:, 1] >= 0) & (points[:, 1] < 900))
    assert np.max(np.ptp(points[1:], axis=0)) >= 50.0


def test_no_valid_hand_falls_back_to_finite_camera_and_camera_geometry_scale() -> None:
    camera = np.column_stack((np.linspace(0.0, 2.0, 12), np.zeros(12), np.linspace(0.0, 1.0, 12)))
    hands = np.full((12, 21, 3), np.nan, dtype=np.float64)
    view = _build_world_views(camera, [hands], (1600, 900), window_frames=6)[5]

    pixels, valid = _project_world(camera[5:6], view, (1600, 900))
    assert valid.all()
    assert view.camera_scale > 0.0
    assert view.camera_scale >= 0.10 * view.span

    canvas = np.full((900, 1600, 3), 28, dtype=np.uint8)
    _draw_camera(canvas, camera[5], None, view)
    assert np.count_nonzero(np.any(canvas != 28, axis=2)) > 20


def test_camera_centric_transform_keeps_current_camera_at_origin() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (3.0, -2.0, 4.0)
    pose[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    origin = _world_to_camera_display(pose[:3, 3][None], pose)
    point = _world_to_camera_display(np.asarray([[5.0, 0.0, 6.0]]), pose)

    np.testing.assert_allclose(origin, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(point, [[2.0, -2.0, 2.0]])


def test_camera_centric_display_has_fixed_view_and_stable_current_hand() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = (0.0, 2.0, 5.0)
    local_hand = np.column_stack((np.linspace(-0.10, 0.10, 21), np.linspace(-0.15, 0.15, 21), np.full(21, 0.8)))
    hands = np.stack([local_hand + poses[index, :3, 3] for index in range(3)])
    right = hands + np.asarray([0.18, 0.0, 0.0])

    view, local_hands, reference, display_rotation = _camera_centric_world_view(poses, [hands, right], (1600, 900))
    first_pixels, first_valid = _project_world(local_hands[0][0], view, (1600, 900))
    last_pixels, last_valid = _project_world(local_hands[0][-1], view, (1600, 900))

    assert reference == 1
    np.testing.assert_allclose(display_rotation @ display_rotation.T, np.eye(3), atol=1e-8)
    assert first_valid.all() and last_valid.all()
    np.testing.assert_allclose(local_hands[0][0], local_hands[0][-1], atol=1e-12)
    np.testing.assert_allclose(first_pixels, last_pixels, atol=1e-10)
    central_pixels = []
    for hand in local_hands[:2]:
        pixels, valid = _project_world(hand[reference], view, (1600, 900))
        assert valid.all()
        central_pixels.append(pixels)
    camera_pixels, camera_valid = _project_world(np.zeros((1, 3)), view, (1600, 900))
    bilateral_centroid = np.mean(np.concatenate(central_pixels, axis=0), axis=0)
    assert camera_valid.all()
    assert camera_pixels[0, 0] < bilateral_centroid[0]
    assert min(np.max(np.ptp(pixels, axis=0)) for pixels in central_pixels) >= 50.0


def test_camera_centric_historical_path_moves_while_current_camera_stays_fixed() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = (0.0, 1.0, 2.0)
    centers = poses[:, :3, 3]

    at_one = _world_to_camera_display(centers[:2], poses[1])
    at_two = _world_to_camera_display(centers, poses[2])

    np.testing.assert_allclose(at_one[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(at_two[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(at_one[0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(at_two[0], [-2.0, 0.0, 0.0])


def test_world_view_autofit_keeps_camera_and_hands_inside_perspective_pane() -> None:
    camera = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.2, 6.0], [7.0, -0.1, 11.0]], dtype=np.float32)
    left = np.asarray([[[8.5 + x, 1.0 + y, 12.0 + z] for x, y, z in zip(np.linspace(-0.2, 0.2, 21), np.linspace(-0.3, 0.3, 21), np.linspace(-0.1, 0.1, 21))]], dtype=np.float32)
    right = np.asarray([[[-1.5 + x, 0.5 + y, -0.8 + z] for x, y, z in zip(np.linspace(-0.2, 0.2, 21), np.linspace(-0.3, 0.3, 21), np.linspace(-0.1, 0.1, 21))]], dtype=np.float32)
    size = (640, 360)

    view = _build_world_view(camera, [left, right], size)
    extent = np.concatenate((camera, left.reshape(-1, 3), right.reshape(-1, 3)), axis=0)
    pixels, valid = _project_world(extent, view, size)

    assert valid.all()
    assert np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < size[0]))
    assert np.all((pixels[:, 1] >= 0) & (pixels[:, 1] < size[1]))
