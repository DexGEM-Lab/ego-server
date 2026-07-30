from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from scripts import run_v22_api_egoscale30h_batch as batch


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        prepare_python="prepare-python",
        pipeline_python="pipeline-python",
        repo_root=tmp_path / "repo",
        render_width=None,
        service_timeout_s=321.0,
        feishu_service_profile=tmp_path / "feishu.json",
        feishu_unidepth_base_url=None,
        feishu_hands_wilor_base_url=None,
        feishu_droid_base_url=None,
        feishu_hawor_base_url=None,
        hawor_root=tmp_path / "hawor",
        rerun_completed=False,
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        pipeline_concurrency=2,
        submission_mode="bounded",
        total_request_limit=128,
        algorithm_inflight_multiplier=2,
        admission_proxy_host="127.0.0.1",
        admission_proxy_port=0,
        rapid_active_limit=2,
        tmux_session="test_batch",
        single_item_index=None,
        skip_item_index=[],
        retry_max_wait_s=0.0,
        retry_initial_delay_s=0.0,
    )


def test_rapid_tmux_admission_creates_one_complete_process_per_video(monkeypatch, tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.submission_mode = "rapid_tmux"
    args.output_root.mkdir(parents=True)
    videos = []
    for index in range(2):
        path = args.dataset_root / f"video_{index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video-{index}".encode())
        videos.append(path.resolve())
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        if command[:2] == ["tmux", "new-window"]:
            index = int(command[-1].split("--single-item-index", 1)[1].split()[0])
            launch_token = command[-1].split("--launch-token", 1)[1].split()[0]
            result_path = args.output_root / "items" / f"item_{index:06d}" / "item_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({"status": "completed", "item_index": index, "launch_token": launch_token}), encoding="utf-8")
        return type("Completed", (), {"returncode": 1 if command[:2] == ["tmux", "has-session"] else 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    result = batch.rapidly_admit_tmux_requests(args, videos, started=1.0)

    assert result == 0
    new_windows = [command for command in calls if command[:2] == ["tmux", "new-window"]]
    assert len(new_windows) == 2
    assert all("flock -x 9" in command[-1] for command in new_windows)
    admission_rows = [json.loads(line) for line in (args.output_root / "dataset_admission.jsonl").read_text().splitlines()]
    assert [row["status"] for row in admission_rows] == ["admitted", "admitted"]
    assert [row["complete_pipeline_slot"] for row in admission_rows] == [0, 1]
    assert {row["complete_pipeline_active_limit"] for row in admission_rows} == {2}
    summary = json.loads((args.output_root / "dataset_batch_summary.json").read_text())
    assert summary["submission_mode"] == "rapid_tmux"
    assert summary["admitted_count"] == 2
    assert summary["complete_pipeline_active_limit"] == 2


def test_resume_admission_skips_explicitly_active_item(monkeypatch, tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.submission_mode = "rapid_tmux"
    args.skip_item_index = [1]
    args.output_root.mkdir(parents=True)
    videos = []
    for index in range(3):
        path = args.dataset_root / f"video_{index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video-{index}".encode())
        videos.append(path.resolve())
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        if command[:2] == ["tmux", "new-window"]:
            index = int(command[-1].split("--single-item-index", 1)[1].split()[0])
            launch_token = command[-1].split("--launch-token", 1)[1].split()[0]
            result_path = args.output_root / "items" / f"item_{index:06d}" / "item_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({"status": "completed", "item_index": index, "launch_token": launch_token}), encoding="utf-8")
        return type("Completed", (), {"returncode": 1 if command[:2] == ["tmux", "has-session"] else 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    result = batch.rapidly_admit_tmux_requests(args, videos, started=1.0)

    assert result == 0
    new_windows = [command for command in calls if command[:2] == ["tmux", "new-window"]]
    assert len(new_windows) == 2
    assert all("item_000001" not in command for command in new_windows)
    admission_rows = [json.loads(line) for line in (args.output_root / "dataset_admission.jsonl").read_text().splitlines()]
    assert [row["status"] for row in admission_rows] == ["admitted", "skipped_active", "admitted"]
    summary = json.loads((args.output_root / "dataset_batch_summary.json").read_text())
    assert summary["video_count"] == 3
    assert summary["admitted_count"] == 2
    assert summary["skipped_active_count"] == 1


def test_rapid_admission_rolls_after_terminal_child_result(monkeypatch, tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.submission_mode = "rapid_tmux"
    args.total_request_limit = 1
    args.output_root.mkdir(parents=True)
    videos = []
    for index in range(3):
        path = args.dataset_root / f"video_{index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video-{index}".encode())
        videos.append(path.resolve())
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        if command[:2] == ["tmux", "new-window"]:
            index = int(command[-1].split("--single-item-index", 1)[1].split()[0])
            launch_token = command[-1].split("--launch-token", 1)[1].split()[0]
            result_path = args.output_root / "items" / f"item_{index:06d}" / "item_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({"status": "completed", "item_index": index, "launch_token": launch_token}), encoding="utf-8")
        return type("Completed", (), {"returncode": 1 if command[:2] == ["tmux", "has-session"] else 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    assert batch.rapidly_admit_tmux_requests(args, videos, started=1.0) == 0
    new_windows = [command for command in calls if command[:2] == ["tmux", "new-window"]]
    indices = [int(command[-1].split("--single-item-index", 1)[1].split()[0]) for command in new_windows]
    assert indices == [0, 1, 2]
    events = [json.loads(line) for line in (args.output_root / "dataset_request_events.jsonl").read_text().splitlines()]
    assert sum(row["status"] == "request_released" for row in events) == 3


def test_pipeline_command_is_complete_non_cosmos_feishu_request(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    command = batch.pipeline_command(args, tmp_path / "video.mp4", tmp_path / "run", "case-1")

    assert command[:2] == ["pipeline-python", "scripts/run_v22_minimal_annotation_pipeline.py"]
    assert command[command.index("--model-execution") + 1] == "feishu_ray"
    assert command[command.index("--camera-backend") + 1] == "droid"
    assert "--run-camera-trajectory" in command
    assert "--run-hawor-metric-hands" in command
    assert "--run-hybrid-hands" in command
    assert "--run-gt-free-drift-self-calibration" in command
    assert "--run-self-consistency-qc" in command
    assert "--run-evaluator" in command
    assert "--write-product-bundle" in command
    assert "--skip-cosmos" in command
    assert "--run-captioning" not in command
    assert not any("1810" in value for value in command)
    assert not any("vggt" in value.lower() for value in command)


def test_child_request_command_forwards_outer_admission_parameters(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    command = batch.child_request_command(args, 4, tmp_path / "video.mp4")
    assert command[command.index("--total-request-limit") + 1] == "128"
    assert command[command.index("--algorithm-inflight-multiplier") + 1] == "2"


def test_algorithm_admission_limits_are_retired_for_batch_scheduler() -> None:
    assert batch.algorithm_admission_limits(2) == {}


def test_batch_admission_proxy_uses_real_profile_wilor_service() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = make_args(repo_root / ".test-tmp")
    args.feishu_service_profile = repo_root / "configs" / "feishu_ray_services.json"

    upstreams = batch.resolve_service_upstreams(args)

    assert batch.ROUTE_TO_SERVICE["/hands.detect"] == "hands_wilor"
    assert batch.ROUTE_TO_SERVICE["/wilor.reconstruct"] == "wilor"
    assert upstreams["hands_wilor"] == "http://127.0.0.1:28001"
    assert upstreams["wilor"] == "http://127.0.0.1:28004"


def test_algorithm_admission_proxy_forwards_route_and_body(tmp_path: Path) -> None:
    received: list[tuple[str, bytes]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body))
            self.send_response(201)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy = batch.AlgorithmAdmissionProxy(
        ("127.0.0.1", 0),
        upstreams={"unidepth": f"http://{host}:{port}"},
        limits=batch.algorithm_admission_limits(2),
        events_path=tmp_path / "proxy_events.jsonl",
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        proxy_host, proxy_port = proxy.server_address[:2]
        connection = http.client.HTTPConnection(proxy_host, proxy_port, timeout=5.0)
        connection.request("POST", "/unidepth.infer", body=b"payload", headers={"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        assert response.status == 201
        assert response.read() == b"ok"
        connection.close()
        assert received == [("/unidepth.infer", b"payload")]
        events = [json.loads(line) for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()]
        event = next(row for row in events if row["event"] == "algorithm_request_forwarded")
        assert event["limit_name"] == "unidepth.infer"
        assert event["configured_limit"] is None
        assert event["batch_cap"] == 8
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()


def test_algorithm_admission_proxy_does_not_block_at_retired_route_limit(tmp_path: Path) -> None:
    active = 0
    entered_count = 0
    max_active = 0
    active_lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    class BlockingHandler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            nonlocal active, entered_count, max_active
            self.rfile.read(int(self.headers["Content-Length"]))
            with active_lock:
                active += 1
                entered_count += 1
                max_active = max(max_active, active)
                (first_entered if entered_count == 1 else second_entered).set()
            release.wait(timeout=5.0)
            with active_lock:
                active -= 1
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), BlockingHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy = batch.AlgorithmAdmissionProxy(
        ("127.0.0.1", 0),
        upstreams={"unidepth": f"http://{host}:{port}"},
        limits={"unidepth.infer": 1},
        events_path=tmp_path / "proxy_events.jsonl",
        batch_caps={"/unidepth.infer": 2},
        batch_waits={"/unidepth.infer": 0.2},
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    responses: list[int] = []

    def call_proxy() -> None:
        proxy_host, proxy_port = proxy.server_address[:2]
        connection = http.client.HTTPConnection(proxy_host, proxy_port, timeout=5.0)
        connection.request("POST", "/unidepth.infer", body=b"x")
        response = connection.getresponse()
        responses.append(response.status)
        response.read()
        connection.close()

    first = threading.Thread(target=call_proxy)
    second = threading.Thread(target=call_proxy)
    try:
        first.start()
        second.start()
        assert first_entered.wait(timeout=2.0)
        assert second_entered.wait(timeout=2.0)
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert responses == [200, 200]
        assert max_active == 2
    finally:
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()


def test_request_release_requires_matching_launch_token(tmp_path: Path) -> None:
    result = tmp_path / "item_result.json"
    result.write_text(json.dumps({"status": "completed", "launch_token": "old"}), encoding="utf-8")
    assert batch._fresh_item_result(result, "new") is None
    assert batch._fresh_item_result(result, "old") == {"status": "completed", "launch_token": "old"}


def test_dead_tmux_child_releases_request_as_failed_launcher(monkeypatch, tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    active = {
        7: {
            "case_id": "case-7",
            "video": "/input.mp4",
            "launch_token": "token-7",
            "tmux_session": "batch",
            "tmux_window": "item-7",
        }
    }
    monkeypatch.setattr(batch, "tmux_window_exists", lambda *_: False)
    assert batch.wait_for_one_request(active, output_root=tmp_path, events_path=events) == 7
    result = json.loads((tmp_path / "items" / "item_000007" / "item_result.json").read_text())
    assert result["status"] == "failed_launcher"
    assert result["launch_token"] == "token-7"
    assert active == {}


def test_complete_request_runs_prepare_then_one_whole_pipeline(monkeypatch, tmp_path: Path) -> None:
    args = make_args(tmp_path)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path, log_path: Path) -> dict[str, object]:
        calls.append(command)
        return {"status": "ok", "returncode": 0, "log": str(log_path)}

    monkeypatch.setattr(batch, "run_command", fake_run)
    monkeypatch.setattr(batch, "completed_attempt", lambda *args, **kwargs: True)
    result = batch.run_complete_request(args, video=video, item_index=7, output_root=tmp_path / "output")

    assert result["status"] == "completed"
    assert len(calls) == 2
    assert calls[0][1] == "scripts/prepare_v22_single_video_run.py"
    assert calls[1][1] == "scripts/run_v22_minimal_annotation_pipeline.py"
    assert result["run_root"].endswith("items/item_000007/attempt_0001")
    assert result["video_identity"]["sha256"] == batch.sha256_file(video)


def write_valid_delivery(run_root: Path, *, case_id: str = "case", frame_count: int = 3) -> None:
    required = batch.required_delivery_paths(run_root)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix not in {".json", ".npz"}:
            path.write_bytes(b"video")
    product = run_root / "product_bundle" / "case" / "manifest.json"
    product.parent.mkdir(parents=True, exist_ok=True)
    errors = product.parent / "events" / "errors.ndjson"
    errors.parent.mkdir(parents=True)
    errors.write_text("", encoding="utf-8")
    semantic_video = run_root / "renders" / "v22_semantic_subtitle.mp4"
    semantic_video.parent.mkdir(parents=True, exist_ok=True)
    semantic_video.write_bytes(b"semantic-video")
    semantic_review = run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"
    semantic_review.parent.mkdir(parents=True, exist_ok=True)
    semantic_review.write_text('{"semantic_rows":[{"start_frame":0,"end_frame":3}]}', encoding="utf-8")
    product.write_text(json.dumps({
        "schema": "ego.annotation.output",
        "status": "completed",
        "tables": {
            "frames": {"rows": frame_count},
            "head_camera": {"rows": frame_count},
            "hand_states": {"rows": frame_count * 2},
            "semantic_clips": {"rows": 1},
        },
        "events": {"errors": {"rows": 0, "ndjson": str(errors)}},
        "errors_count": 0,
    }), encoding="utf-8")
    required[0].write_text(
        json.dumps(
            {
                "status": "ok",
                "steps": [{"status": "skipped_prepared_input"}, {"status": "ok"}],
                "ffprobe_overlay": {"ffprobe": {"streams": [{"nb_read_frames": str(frame_count)}]}},
                "product_manifest_path": str(product),
                "enabled_stages": {"captioning": True},
                "renders": {"semantic_subtitle": str(semantic_video)},
                "semantic_review": str(semantic_review),
            }
        ),
        encoding="utf-8",
    )
    required[1].write_text(
        json.dumps(
            {
                "schema": "v22_input_manifest.v0",
                "case_id": case_id,
                "source_fingerprint": {"path": "/input/video.mp4", "sha256": "input-hash"},
            }
        ),
        encoding="utf-8",
    )
    required[2].write_text(json.dumps({"frame_count": frame_count, "frames": [{}] * frame_count}), encoding="utf-8")
    required[3].write_text(json.dumps({"status": "ok", "processed_frames": frame_count}), encoding="utf-8")
    np.savez_compressed(
        required[4],
        frame_idx=np.arange(frame_count, dtype=np.int32),
        T_world_camera=np.tile(np.eye(4, dtype=np.float32), (frame_count, 1, 1)),
        pose_world_camera_xyzw=np.zeros((frame_count, 7), dtype=np.float32),
        droid_pose_valid=np.ones(frame_count, dtype=np.uint8),
    )
    np.savez_compressed(
        required[5],
        tstamps=np.asarray([0], dtype=np.int32),
        disps=np.ones((1, 2, 2), dtype=np.float32),
        intrinsics=np.ones((1, 4), dtype=np.float32),
    )
    required[7].write_text(json.dumps({"status": "ok", "frames": frame_count}), encoding="utf-8")
    required[10].write_text(json.dumps({"video_frame_count": frame_count}), encoding="utf-8")
    vertices = np.zeros((frame_count, 778, 3), dtype=np.float32)
    vertices[0, 0, 0] = 1.0
    joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
    joints[0, 0, 0] = 1.0
    trans = np.zeros((frame_count, 3), dtype=np.float32)
    valid = np.zeros(frame_count, dtype=np.uint8)
    valid[0] = 1
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    np.savez_compressed(
        required[6],
        frame_idx=np.arange(frame_count, dtype=np.int32),
        R_c2w=np.tile(np.eye(3, dtype=np.float32), (frame_count, 1, 1)),
        t_c2w=np.zeros((frame_count, 3), dtype=np.float32),
        left_vertices_world_m=vertices,
        left_joints_world_m=joints,
        left_trans_world_m=trans,
        left_valid=valid,
        left_faces=faces,
        right_vertices_world_m=vertices,
        right_joints_world_m=joints,
        right_trans_world_m=trans,
        right_valid=valid,
        right_faces=faces,
    )


def write_cosmos_enabled_product(run_root: Path, *, frame_count: int = 3) -> tuple[Path, Path]:
    pipeline_path = run_root / "annotation_pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline.update(
        {
            "execution_topology": "complete_feishu_ray_with_cosmos_semantics",
            "enabled_stages": {"captioning": True},
            "renders": {**pipeline.get("renders", {}), "semantic_subtitle": str(run_root / "renders" / "v22_semantic_subtitle.mp4")},
            "semantic_review": str(run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"),
        }
    )
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    product_path = Path(pipeline["product_manifest_path"])
    errors_path = product_path.parent / "events" / "errors.ndjson"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_video = run_root / "renders" / "v22_semantic_subtitle.mp4"
    semantic_video.parent.mkdir(parents=True, exist_ok=True)
    semantic_video.write_bytes(b"semantic-video")
    semantic_review = run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"
    semantic_review.parent.mkdir(parents=True, exist_ok=True)
    semantic_review.write_text('{"semantic_rows":[{"start_frame":0,"end_frame":3}]}', encoding="utf-8")
    errors = [{"code": "offline_evaluator_gt_unavailable", "severity": "degraded"}]
    errors_path.write_text("".join(json.dumps(row) + "\n" for row in errors), encoding="utf-8")
    product_path.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.output",
                "status": "completed_with_errors",
                "tables": {
                    "frames": {"rows": frame_count},
                    "head_camera": {"rows": frame_count},
                    "hand_states": {"rows": frame_count * 2},
                    "semantic_clips": {"rows": 1},
                },
                "events": {"errors": {"rows": len(errors), "ndjson": str(errors_path)}},
                "errors_count": len(errors),
            }
        ),
        encoding="utf-8",
    )
    return product_path, errors_path


def test_completed_attempt_validates_real_arrays_and_accepts_skip_prepare(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root)
    monkeypatch.setattr(batch, "ffprobe_frame_count", lambda _path: 3)

    assert batch.completed_attempt(run_root, expected_case_id="case")
    (run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz").write_bytes(b"truncated")
    assert not batch.completed_attempt(run_root, expected_case_id="case")


def test_hawor_validator_accepts_nan_camera_tail_and_rejects_endpoint_fill(tmp_path: Path) -> None:
    frame_count = 1025
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root, frame_count=frame_count)
    droid_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    T = np.tile(np.eye(4, dtype=np.float32)[None], (frame_count, 1, 1))
    pose = np.zeros((frame_count, 7), dtype=np.float32)
    valid = np.zeros(frame_count, dtype=np.uint8)
    valid[:1024] = 1
    T[1024:] = np.nan
    pose[1024:] = np.nan
    np.savez_compressed(droid_path, frame_idx=np.arange(frame_count, dtype=np.int32), T_world_camera=T, pose_world_camera_xyzw=pose, droid_pose_valid=valid)
    qc_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "qc_hawor_world_hands.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["status"] = "completed_with_partial_camera_coverage"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    hawor_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz"
    with np.load(hawor_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["R_c2w"][1024:] = np.nan
    payload["t_c2w"][1024:] = np.nan
    payload["camera_valid"] = valid
    for side in ("left", "right"):
        payload[f"{side}_vertices_world_m"][1024:] = np.nan
        payload[f"{side}_joints_world_m"][1024:] = np.nan
        payload[f"{side}_trans_world_m"][1024:] = np.nan
        payload[f"{side}_valid"][1024:] = 0
    np.savez_compressed(hawor_path, **payload)

    assert batch.validate_droid_artifacts(run_root, frame_count)
    assert batch.validate_hawor_artifacts(run_root, frame_count, None)

    qc["status"] = "ok"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    assert not batch.validate_hawor_artifacts(run_root, frame_count, None)
    qc["status"] = "completed_with_partial_camera_coverage"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    payload["R_c2w"][1024:] = np.eye(3, dtype=np.float32)
    np.savez_compressed(hawor_path, **payload)
    assert not batch.validate_hawor_artifacts(run_root, frame_count, None)


def test_droid_validator_rejects_maskless_filled_tail(tmp_path: Path) -> None:
    frame_count = 1025
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root, frame_count=frame_count)
    droid_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    T = np.tile(np.eye(4, dtype=np.float32)[None], (frame_count, 1, 1))
    pose = np.zeros((frame_count, 7), dtype=np.float32)
    # A plausible but forbidden endpoint-filled 1025th pose cannot become full coverage.
    T[1024] = T[1023]
    pose[1024] = pose[1023]
    np.savez_compressed(droid_path, frame_idx=np.arange(frame_count, dtype=np.int32), T_world_camera=T, pose_world_camera_xyzw=pose)

    assert batch.droid_camera_validity(run_root, frame_count) is None
    assert not batch.validate_droid_artifacts(run_root, frame_count)


def test_droid_validator_rejects_incorrect_coverage_masks(tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root, frame_count=1025)
    droid_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    T = np.tile(np.eye(4, dtype=np.float32)[None], (1025, 1, 1))
    pose = np.zeros((1025, 7), dtype=np.float32)
    np.savez_compressed(droid_path, frame_idx=np.arange(1025, dtype=np.int32), T_world_camera=T, pose_world_camera_xyzw=pose, droid_pose_valid=np.ones(1025, dtype=np.uint8))

    assert batch.droid_camera_validity(run_root, 1025) is None
    assert not batch.validate_droid_artifacts(run_root, 1025)


def test_droid_validator_requires_all_true_mask_at_or_below_capacity(tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root, frame_count=360)
    droid_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    T = np.tile(np.eye(4, dtype=np.float32)[None], (360, 1, 1))
    pose = np.zeros((360, 7), dtype=np.float32)
    mask = np.ones(360, dtype=np.uint8)
    mask[-1] = 0
    np.savez_compressed(droid_path, frame_idx=np.arange(360, dtype=np.int32), T_world_camera=T, pose_world_camera_xyzw=pose, droid_pose_valid=mask)

    assert batch.droid_camera_validity(run_root, 360) is None
    assert not batch.validate_droid_artifacts(run_root, 360)


def test_completed_attempt_rejects_missing_cosmos_artifact(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root)
    monkeypatch.setattr(batch, "ffprobe_frame_count", lambda _path: 3)
    (run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json").unlink()
    assert not batch.completed_attempt(run_root, expected_case_id="case")


def test_completed_attempt_requires_enabled_cosmos_semantic_artifacts(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root)
    product_path, errors_path = write_cosmos_enabled_product(run_root)
    monkeypatch.setattr(batch, "ffprobe_frame_count", lambda _path: 3)

    assert batch.completed_attempt(run_root, expected_case_id="case")

    product = json.loads(product_path.read_text(encoding="utf-8"))
    product["errors_count"] = 2
    product["events"]["errors"]["rows"] = 2
    product_path.write_text(json.dumps(product), encoding="utf-8")
    errors_path.write_text(
        json.dumps({"code": "offline_evaluator_gt_unavailable", "severity": "degraded"})
        + "\n"
        + json.dumps({"code": "model_failure", "severity": "error"})
        + "\n",
        encoding="utf-8",
    )
    assert not batch.completed_attempt(run_root, expected_case_id="case")

    product_path, _ = write_cosmos_enabled_product(run_root)
    pipeline_path = run_root / "annotation_pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["enabled_stages"]["captioning"] = False
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    assert not batch.completed_attempt(run_root, expected_case_id="case")


def test_completed_attempt_binds_video_identity(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "attempt_0001"
    write_valid_delivery(run_root, case_id="case-a")
    monkeypatch.setattr(batch, "ffprobe_frame_count", lambda _path: 3)

    assert batch.completed_attempt(run_root, expected_case_id="case-a")
    assert not batch.completed_attempt(run_root, expected_case_id="case-b")


def test_reserve_attempt_root_is_atomic_and_next_number_is_observable(tmp_path: Path) -> None:
    item_root = tmp_path / "item_000001"
    first = batch.reserve_attempt_root(item_root)
    second = batch.reserve_attempt_root(item_root)

    assert first.name == "attempt_0001"
    assert second.name == "attempt_0002"
    assert batch.next_attempt_root(item_root) == item_root / "attempt_0003"


def test_zero_argument_traversal_uses_fixed_manager_contract(monkeypatch) -> None:
    monkeypatch.delenv(batch.DATASET_ROOT_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["run_v22_api_egoscale30h_batch.py"])
    args = batch.parse_args()
    assert args.dataset_root == batch.FIXED_DATASET_ROOT
    assert args.output_root is None
    assert args.submission_mode == "api_http"
    assert args.api_base_url == batch.FIXED_API_BASE_URL
    assert args.api_client_concurrency == batch.DEFAULT_API_CLIENT_CONCURRENCY == 0
    assert args.stability_video_limit == 0
    assert args.total_request_limit == 128
    assert args.algorithm_inflight_multiplier == 2


def test_zero_argument_traversal_allows_inventory_root_override(monkeypatch, tmp_path: Path) -> None:
    inventory_root = tmp_path / "egoscale_demo_30h"
    monkeypatch.setenv(batch.DATASET_ROOT_ENV, str(inventory_root))
    monkeypatch.setattr(sys, "argv", ["run_v22_api_egoscale30h_batch.py"])

    assert batch.parse_args().dataset_root == inventory_root


@pytest.mark.parametrize(
    "flag",
    ["--dataset-root", "--output-root", "--api-base-url", "--total-request-limit", "--submission-mode"],
)
def test_production_traversal_rejects_business_parameters(monkeypatch, flag: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_v22_api_egoscale30h_batch.py", flag, "override"])
    with pytest.raises(SystemExit) as exc_info:
        batch.parse_args()
    assert exc_info.value.code == 2


def test_api_http_product_acceptance_requires_truthful_cosmos_summary() -> None:
    complete = {
        "cosmos": {
            "status": "enabled",
            "request_count": 3,
            "semantic_row_count": 2,
            "review_json": "/remote/review.json",
            "captioned_combined_video": "/remote/v22_combined.mp4",
        }
    }
    assert batch.cosmos_api_summary_is_complete(complete)
    assert batch.cosmos_api_summary_is_complete({"cosmos": {**complete["cosmos"], "status": "completed_with_anomalies", "anomaly_count": 2}})
    assert not batch.cosmos_api_summary_is_complete({"cosmos": {**complete["cosmos"], "semantic_row_count": 0}})
    assert not batch.cosmos_api_summary_is_complete({"cosmos": {**complete["cosmos"], "request_count": True}})
    assert not batch.cosmos_api_summary_is_complete({"cosmos": {**complete["cosmos"], "status": "disabled"}})
    diagnostic_complete = {
        "status": "diagnostic_or_uncertain",
        "acceptance": {"accepted": False, "diagnostic_only": True, "scale_mode": "up_to_scale_monocular"},
        **complete,
    }
    assert batch.cosmos_api_summary_is_complete(diagnostic_complete)


def test_api_http_mode_submits_each_video_to_single_item_manager(tmp_path: Path) -> None:
    received: list[tuple[str, bytes]] = []

    class ApiHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body))
            payload = json.dumps({
                "status": "ok",
                "job_id": "accepted",
                "summary": {
                    "cosmos": {
                        "status": "enabled",
                        "request_count": 2,
                        "semantic_row_count": 1,
                        "review_json": "/remote/review.json",
                        "captioned_combined_video": "/remote/v22_combined.mp4",
                    }
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = argparse.Namespace(
            output_root=tmp_path / "api_run",
            api_base_url=f"http://127.0.0.1:{server.server_address[1]}",
            api_model_backend="api_ify",
            api_diagnostic_monocular=True,
            api_client_concurrency=0,
            api_request_timeout_s=30.0,
            api_job_prefix="fresh_run",
            total_request_limit=128,
            algorithm_inflight_multiplier=2,
        )
        args.output_root.mkdir()
        videos = []
        for index in range(3):
            video = tmp_path / f"video {index}.mp4"
            video.write_bytes(f"video-{index}".encode())
            videos.append(video)

        assert batch.run_api_http_requests(args, videos, started=time.time()) == 0
        assert len(received) == 3
        assert all(path == "/v1/annotation-jobs" for path, _body in received)
        assert all(b"name=\"file\"" in body for _path, body in received)
        assert all(b"name=\"request\"" not in body for _path, body in received)
        summary = json.loads((args.output_root / "dataset_batch_summary.json").read_text())
        assert summary["status"] == "completed"
        assert summary["terminal_count"] == 3
        assert summary["submitted_count"] == 3
        assert summary["stability_control"]["mode"] == "full_dataset_producer"
        assert summary["stability_control"]["producer_only"] is True
        assert summary["stability_control"]["stability_reached"] is False
        events = [json.loads(line) for line in (args.output_root / "dataset_request_events.jsonl").read_text().splitlines()]
        terminals = [row for row in events if row["event"] == "terminal"]
        assert len(terminals) == 3
        assert all(row["measurement_phase"] == "producer" for row in terminals)
        assert summary["admission_owner"] == "single_item_api_manager"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_service_lane_summary_includes_cosmos_request_trace() -> None:
    traces = []
    for stage in ("unidepth.infer", "hands.detect", "wilor.reconstruct", "droid.create_session", "droid.push_frame", "droid.finalize", "hawor.infer_tracks", "hawor_infiller.fill", "cosmos3.reason"):
        traces.append({"stage_id": stage, "request_count": 1, "started_monotonic_s": 1.0, "completed_monotonic_s": 2.0})
    lanes = batch.aggregate_video_service_lane_traces({"performance": {"request_traces": traces}})
    assert lanes["cosmos3"]["request_count"] == 1
    assert lanes["cosmos3"]["stage_ids"] == ["cosmos3.reason"]
