#!/usr/bin/env python3
"""Traverse EgoScale30h and run one complete annotation request per video.

The zero-argument production command uses the default A800 EgoScale30h mirror
(or ``EGO_API_IFY_DATASET_ROOT``), automatically creates a fresh timestamped
output root, and uploads every video to the fixed single-item manager at
``127.0.0.1:8092``. The manager owns all pipeline and admission configuration.
Historical direct modes remain internal compatibility paths.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego_annotation.fps_config import DEFAULT_FPS_CONDITION, get_fps_condition
from scripts.annotation_admission_proxy import AdmissionServer, ROUTE_TO_LIMIT_NAME, ROUTE_TO_SERVICE
from scripts.summarize_fps_production_condition import stability_windows


STATE_LOCK = threading.Lock()
FIXED_DATASET_ROOT = Path("/home/zjh/data/egoscale_demo_30h")
FIXED_API_BASE_URL = os.environ.get("EGO_API_IFY_API_BASE_URL", "http://127.0.0.1:8092")
FIXED_OUTPUT_PARENT = Path("/home/zjh/data")
FIXED_API_JOB_ROOT = Path(os.environ.get("EGO_API_IFY_API_JOB_ROOT", "/home/zjh/data/v22_api_release_da17415/jobs"))
FIXED_MANAGER_TOTAL_REQUEST_LIMIT = int(os.environ.get("EGO_API_IFY_TOTAL_REQUEST_LIMIT", "128"))
FIXED_MANAGER_ALGORITHM_MULTIPLIER = int(os.environ.get("EGO_API_IFY_ALGORITHM_INFLIGHT_MULTIPLIER", "2"))
FIXED_API_REQUEST_TIMEOUT_S = 28800.0
DATASET_ROOT_ENV = "EGO_API_IFY_DATASET_ROOT"
STABILITY_VIDEO_LIMIT_ENV = "EGO_API_IFY_STABILITY_VIDEO_LIMIT"
STABILITY_WARMUP_COUNT_ENV = "EGO_API_IFY_STABILITY_WARMUP_COUNT"
STABILITY_WINDOW_SIZE_ENV = "EGO_API_IFY_STABILITY_WINDOW_SIZE"
STABILITY_TOLERANCE_ENV = "EGO_API_IFY_STABILITY_TOLERANCE"
API_CLIENT_CONCURRENCY_ENV = "EGO_API_IFY_API_CLIENT_CONCURRENCY"
OUTPUT_ROOT_ENV = "EGO_API_IFY_OUTPUT_ROOT"
DEFAULT_STABILITY_WARMUP_COUNT = 4
DEFAULT_STABILITY_WINDOW_SIZE = 4
DEFAULT_STABILITY_TOLERANCE = 0.10
DEFAULT_API_CLIENT_CONCURRENCY = 0
VIDEO_STREAM_SERVICE_STAGES: dict[str, tuple[str, ...]] = {
    "unidepth": ("unidepth.infer",),
    "hands.detect": ("hands.detect",),
    "wilor": ("wilor.reconstruct",),
    "droid": ("droid.create_session", "droid.push_frame", "droid.finalize"),
    "hawor.track": ("hawor.infer_tracks",),
    "hawor.infiller": ("hawor_infiller.fill",),
    "cosmos3": ("cosmos3.reason",),
}

# The compatibility batch admission proxy shares its route/service/limit
# ownership with the API-Ify admission proxy.  Keeping one imported mapping
# prevents an independently launched full-dataset manager from drifting away
# from the profile's WiLoR service during a port split.


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with STATE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            finally:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def algorithm_admission_limits(multiplier: int) -> dict[str, int]:
    """Compatibility summary for the retired fixed B-layer limit policy."""
    if multiplier <= 0:
        raise ValueError("--algorithm-inflight-multiplier must be positive")
    return {}


def resolve_service_upstreams(args: argparse.Namespace) -> dict[str, str]:
    profile = load_json(Path(args.feishu_service_profile))
    services = profile.get("services") if isinstance(profile, dict) else None
    if not isinstance(services, dict):
        raise ValueError(f"invalid Feishu service profile: {args.feishu_service_profile}")
    overrides = {
        "unidepth": args.feishu_unidepth_base_url,
        "hands_wilor": args.feishu_hands_wilor_base_url,
        "droid": args.feishu_droid_base_url,
        "hawor": args.feishu_hawor_base_url,
    }
    result: dict[str, str] = {}
    for service in set(ROUTE_TO_SERVICE.values()):
        row = services.get(service)
        value = overrides.get(service) or (row.get("base_url") if isinstance(row, dict) else None)
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            if service == "cosmos3":
                continue
            raise ValueError(f"missing upstream base URL for {service}")
        result[service] = value.rstrip("/")
    return result


class AlgorithmAdmissionProxy(AdmissionServer):
    """Compatibility name for the shared client batch scheduler."""

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstreams: dict[str, str],
        limits: dict[str, int] | None = None,
        events_path: Path,
        batch_caps: dict[str, int] | None = None,
        batch_waits: dict[str, float] | None = None,
    ):
        super().__init__(
            address,
            upstreams=upstreams,
            limits=limits,
            events_path=events_path,
            batch_caps=batch_caps,
            batch_waits=batch_waits,
        )


@contextlib.contextmanager
def run_algorithm_admission_proxy(args: argparse.Namespace) -> Iterator[str]:
    upstreams = resolve_service_upstreams(args)
    server = AlgorithmAdmissionProxy(
        (str(args.admission_proxy_host), int(args.admission_proxy_port)),
        upstreams=upstreams,
        events_path=args.output_root / "algorithm_admission_events.jsonl",
    )
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, name="algorithm_admission_proxy", daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextlib.contextmanager
def acquire_algorithm_slots(
    output_root: Path,
    *,
    item_index: int,
    multiplier: int,
) -> Iterator[dict[str, Any]]:
    """Compatibility context with no fixed per-algorithm reservation.

    The route-level client scheduler groups fully buffered requests and treats
    429 as the service capacity signal.  DROID ordering remains protocol-owned
    by each child and by the shared proxy's local lifecycle marker.
    """
    if multiplier <= 0:
        raise ValueError("--algorithm-inflight-multiplier must be positive")
    yield {
        "limits": {},
        "locks": {},
        "route_level_proxy": True,
        "admission_policy": "client_batch_scheduler",
        "item_index": int(item_index),
        "output_root": str(output_root),
    }


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished = time.time()
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "started_at_unix": started,
        "finished_at_unix": finished,
        "elapsed_s": float(finished - started),
        "log": str(log_path),
        "command": command,
    }


def discover_videos(dataset_root: Path, limit: int | None) -> list[Path]:
    videos = sorted(path.resolve() for path in dataset_root.rglob("*.mp4") if path.is_file())
    if limit is not None:
        videos = videos[: int(limit)]
    if not videos:
        raise FileNotFoundError(f"no mp4 videos under {dataset_root}")
    return videos


def required_delivery_paths(run_root: Path) -> tuple[Path, ...]:
    return (
        run_root / "annotation_pipeline_manifest.json",
        run_root / "input" / "input_manifest.json",
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json",
        run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz",
        run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_keyframe_reconstruction.npz",
        run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz",
        run_root / "measurements" / "hand_candidates" / "hawor_world" / "qc_hawor_world_hands.json",
        run_root / "renders" / "v22_overlay.mp4",
        run_root / "renders" / "v22_world_head_hand_3d.mp4",
        run_root / "renders" / "v22_world_head_hand_3d_report.json",
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_identity(video: Path) -> dict[str, Any]:
    stat = video.stat()
    return {"path": str(video.resolve()), "size_bytes": int(stat.st_size), "sha256": sha256_file(video)}


def manifest_overlay_frame_count(manifest: dict[str, Any]) -> int | None:
    probe = manifest.get("ffprobe_overlay")
    ffprobe = probe.get("ffprobe") if isinstance(probe, dict) else None
    streams = ffprobe.get("streams") if isinstance(ffprobe, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return None
    raw = streams[0].get("nb_read_frames")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def load_jsonl_objects(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None
                rows.append(value)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return rows


def cosmos_enabled_product_is_complete(
    product_path: Path,
    product: dict[str, Any],
    pipeline_manifest: dict[str, Any],
    expected_frames: int,
) -> bool:
    """Require real semantic rows and full-duration Cosmos artifacts."""
    if product.get("schema") != "ego.annotation.output":
        return False
    enabled = pipeline_manifest.get("enabled_stages")
    if not isinstance(enabled, dict) or enabled.get("captioning") is not True:
        return False
    tables = product.get("tables")
    if not isinstance(tables, dict):
        return False
    for name, expected in (("frames", expected_frames), ("head_camera", expected_frames), ("hand_states", expected_frames * 2)):
        row = tables.get(name)
        if not isinstance(row, dict) or row.get("rows") != expected:
            return False
    semantic = tables.get("semantic_clips")
    if not isinstance(semantic, dict) or not isinstance(semantic.get("rows"), int) or semantic["rows"] <= 0:
        return False
    renders = pipeline_manifest.get("renders")
    if not isinstance(renders, dict):
        return False
    for value in (renders.get("semantic_subtitle"), pipeline_manifest.get("semantic_review")):
        if not isinstance(value, str) or not Path(value).is_file():
            return False
    events = product.get("events")
    errors_meta = events.get("errors") if isinstance(events, dict) else None
    if not isinstance(errors_meta, dict) or not isinstance(errors_meta.get("ndjson"), str):
        return False
    errors_path = Path(errors_meta["ndjson"])
    if not errors_path.is_absolute():
        errors_path = product_path.parent / errors_path
    errors = load_jsonl_objects(errors_path)
    return errors is not None and len(errors) == errors_meta.get("rows") and not any(row.get("severity") == "error" for row in errors)


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def droid_camera_validity(run_root: Path, expected_frames: int) -> np.ndarray | None:
    """Require the frozen DROID coverage policy; a finite maskless tail is invalid."""
    import numpy as np

    trajectory_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    try:
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            key = next((name for name in ("droid_pose_valid", "camera_valid") if name in trajectory.files), None)
            if key is None:
                return None
            valid = np.asarray(trajectory[key], dtype=bool).reshape(-1)
    except (OSError, ValueError, KeyError):
        return None
    if valid.shape != (expected_frames,):
        return None
    submitted = expected_frames if expected_frames <= 1024 else 1024
    expected = np.zeros(expected_frames, dtype=bool)
    expected[:submitted] = True
    return valid if np.array_equal(valid, expected) else None


def validate_droid_artifacts(run_root: Path, expected_frames: int) -> bool:
    import numpy as np

    droid_manifest = load_json(run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json")
    if droid_manifest is None or droid_manifest.get("status") != "ok":
        return False
    if int(droid_manifest.get("processed_frames", -1)) != expected_frames:
        return False
    trajectory_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.npz"
    reconstruction_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_keyframe_reconstruction.npz"
    try:
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            if not {"frame_idx", "T_world_camera", "pose_world_camera_xyzw"}.issubset(trajectory.files):
                return False
            frame_idx = np.asarray(trajectory["frame_idx"])
            poses = np.asarray(trajectory["T_world_camera"])
            pose_vectors = np.asarray(trajectory["pose_world_camera_xyzw"])
            if frame_idx.shape != (expected_frames,) or not np.array_equal(frame_idx, np.arange(expected_frames, dtype=frame_idx.dtype)):
                return False
            if poses.shape != (expected_frames, 4, 4) or pose_vectors.shape != (expected_frames, 7):
                return False
            valid = droid_camera_validity(run_root, expected_frames)
            if valid is None:
                return False
            if not np.isfinite(poses[valid]).all() or not np.isfinite(pose_vectors[valid]).all():
                return False
            if np.isfinite(poses[~valid]).any() or np.isfinite(pose_vectors[~valid]).any():
                return False
        with np.load(reconstruction_path, allow_pickle=False) as reconstruction:
            if not {"tstamps", "disps", "intrinsics"}.issubset(reconstruction.files):
                return False
            disps = np.asarray(reconstruction["disps"])
            if disps.ndim != 3 or disps.shape[0] <= 0 or not np.isfinite(disps).all() or np.any(disps <= 0.0):
                return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def validate_hawor_artifacts(run_root: Path, expected_frames: int, expected_video_sha256: str | None) -> bool:
    import numpy as np

    archive_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz"
    qc_path = run_root / "measurements" / "hand_candidates" / "hawor_world" / "qc_hawor_world_hands.json"
    qc = load_json(qc_path)
    if qc is None:
        return False
    if int(qc.get("frame_count", qc.get("frames", -1))) != expected_frames:
        return False
    declared_archive = qc.get("output_npz")
    if declared_archive and Path(str(declared_archive)).resolve() != archive_path.resolve():
        return False
    required = {
        "frame_idx",
        "R_c2w",
        "t_c2w",
        "left_vertices_world_m",
        "left_joints_world_m",
        "left_trans_world_m",
        "left_valid",
        "left_faces",
        "right_vertices_world_m",
        "right_joints_world_m",
        "right_trans_world_m",
        "right_valid",
        "right_faces",
    }
    try:
        camera_valid = droid_camera_validity(run_root, expected_frames)
        if camera_valid is None:
            return False
        expected_qc_status = "ok" if bool(np.all(camera_valid)) else "completed_with_partial_camera_coverage"
        if qc.get("status") != expected_qc_status:
            return False
        with np.load(archive_path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                return False
            frame_idx = np.asarray(archive["frame_idx"])
            if frame_idx.shape != (expected_frames,) or not np.array_equal(frame_idx, np.arange(expected_frames, dtype=frame_idx.dtype)):
                return False
            R_c2w = np.asarray(archive["R_c2w"])
            t_c2w = np.asarray(archive["t_c2w"])
            if R_c2w.shape != (expected_frames, 3, 3) or t_c2w.shape != (expected_frames, 3):
                return False
            if not np.isfinite(R_c2w[camera_valid]).all() or not np.isfinite(t_c2w[camera_valid]).all():
                return False
            if np.isfinite(R_c2w[~camera_valid]).any() or np.isfinite(t_c2w[~camera_valid]).any():
                return False
            if (~camera_valid).any() and "camera_valid" not in archive.files:
                return False
            if "camera_valid" in archive.files and not np.array_equal(np.asarray(archive["camera_valid"]).reshape(-1).astype(bool), camera_valid):
                return False
            for side in ("left", "right"):
                vertices = np.asarray(archive[f"{side}_vertices_world_m"])
                joints = np.asarray(archive[f"{side}_joints_world_m"])
                trans = np.asarray(archive[f"{side}_trans_world_m"])
                valid = np.asarray(archive[f"{side}_valid"]).reshape(-1)
                faces = np.asarray(archive[f"{side}_faces"])
                if vertices.shape != (expected_frames, 778, 3) or joints.shape != (expected_frames, 21, 3) or trans.shape != (expected_frames, 3) or valid.shape != (expected_frames,):
                    return False
                if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0 or np.any(faces < 0) or np.any(faces >= 778):
                    return False
                selected = valid.astype(bool)
                if np.any(selected & ~camera_valid):
                    return False
                if not np.isnan(vertices[~camera_valid]).all() or not np.isnan(joints[~camera_valid]).all() or not np.isnan(trans[~camera_valid]).all():
                    return False
                if selected.any():
                    if not np.isfinite(vertices[selected]).all() or not np.isfinite(joints[selected]).all() or not np.isfinite(trans[selected]).all():
                        return False
                    if not np.any(np.abs(vertices[selected]) > 1.0e-7):
                        return False
            if expected_video_sha256 is not None and "video_sha256" in archive.files:
                declared_hash = str(np.asarray(archive["video_sha256"]).reshape(-1)[0])
                if declared_hash and declared_hash != expected_video_sha256:
                    return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def completed_attempt(
    run_root: Path,
    *,
    expected_case_id: str | None = None,
    expected_video: dict[str, Any] | None = None,
) -> bool:
    paths = required_delivery_paths(run_root)
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        return False
    manifest = load_json(paths[0])
    input_manifest = load_json(paths[1])
    raw_manifest = load_json(paths[2])
    world_report = load_json(paths[10])
    if any(value is None for value in (manifest, input_manifest, raw_manifest, world_report)):
        return False
    assert manifest is not None and input_manifest is not None and raw_manifest is not None and world_report is not None
    if expected_case_id is not None and input_manifest.get("case_id") != expected_case_id:
        return False
    source_fingerprint = input_manifest.get("source_fingerprint")
    if expected_video is not None:
        if not isinstance(source_fingerprint, dict):
            return False
        if source_fingerprint.get("path") != expected_video.get("path") or source_fingerprint.get("sha256") != expected_video.get("sha256"):
            return False
    if manifest.get("status") != "ok":
        return False
    steps = manifest.get("steps")
    allowed_step_statuses = {"ok", "skipped_prepared_input"}
    if not isinstance(steps, list) or not steps or any(
        not isinstance(row, dict) or row.get("status") not in allowed_step_statuses for row in steps
    ):
        return False
    expected_frames = raw_manifest.get("frame_count")
    if expected_frames is None and isinstance(raw_manifest.get("frames"), list):
        expected_frames = len(raw_manifest["frames"])
    try:
        expected_frames = int(expected_frames)
        world_frames = int(world_report.get("video_frame_count"))
    except (TypeError, ValueError):
        return False
    if expected_frames <= 0 or world_frames != expected_frames:
        return False
    if manifest_overlay_frame_count(manifest) != expected_frames:
        return False
    if ffprobe_frame_count(paths[8]) != expected_frames or ffprobe_frame_count(paths[9]) != expected_frames:
        return False
    if not validate_droid_artifacts(run_root, expected_frames):
        return False
    expected_hash = expected_video.get("sha256") if expected_video is not None else None
    if not validate_hawor_artifacts(run_root, expected_frames, expected_hash):
        return False
    product_path = manifest.get("product_manifest_path")
    if not isinstance(product_path, str) or not product_path:
        return False
    product_manifest_path = Path(product_path)
    product = load_json(product_manifest_path)
    if product is None:
        return False
    allowed_product_statuses = {"ok", "completed", "completed_with_degraded_outputs", "completed_with_errors"}
    return product.get("status") in allowed_product_statuses and cosmos_enabled_product_is_complete(
        product_manifest_path,
        product,
        manifest,
        expected_frames,
    )


def find_completed_attempt(
    item_root: Path,
    *,
    expected_case_id: str | None = None,
    expected_video: dict[str, Any] | None = None,
) -> Path | None:
    for path in sorted(item_root.glob("attempt_*"), reverse=True):
        if path.is_dir() and completed_attempt(path, expected_case_id=expected_case_id, expected_video=expected_video):
            return path
    return None


def next_attempt_root(item_root: Path) -> Path:
    attempt_numbers: list[int] = []
    for path in item_root.glob("attempt_*"):
        try:
            attempt_numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return item_root / f"attempt_{max(attempt_numbers, default=0) + 1:04d}"


def reserve_attempt_root(item_root: Path) -> Path:
    item_root.mkdir(parents=True, exist_ok=True)
    for attempt_number in range(1, 1_000_000):
        candidate = item_root / f"attempt_{attempt_number:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not reserve a fresh attempt under {item_root}")


def prepare_command(args: argparse.Namespace, video: Path, run_root: Path, case_id: str) -> list[str]:
    command = [
        args.prepare_python,
        "scripts/prepare_v22_single_video_run.py",
        "--case-id",
        case_id,
        "--input-video",
        str(video),
        "--run-root",
        str(run_root),
    ]
    if args.render_width is not None:
        command.extend(["--render-width", str(args.render_width)])
    return command


def pipeline_command(args: argparse.Namespace, video: Path, run_root: Path, case_id: str) -> list[str]:
    command = [
        args.pipeline_python,
        "scripts/run_v22_minimal_annotation_pipeline.py",
        "--case-id",
        case_id,
        "--input-video",
        str(video),
        "--run-root",
        str(run_root),
        "--skip-prepare",
        "--model-execution",
        "feishu_ray",
        "--camera-backend",
        "droid",
        "--run-camera-trajectory",
        "--run-hawor-metric-hands",
        "--run-hybrid-hands",
        "--run-gt-free-drift-self-calibration",
        "--run-self-consistency-qc",
        "--run-evaluator",
        "--write-product-bundle",
        "--skip-cosmos",
        "--service-timeout-s",
        str(args.service_timeout_s),
        "--retry-max-wait-s",
        str(float(getattr(args, "retry_max_wait_s", 0.0))),
        "--retry-initial-delay-s",
        str(float(getattr(args, "retry_initial_delay_s", 1.0))),
        "--feishu-service-profile",
        str(args.feishu_service_profile),
        "--hawor-root",
        str(args.hawor_root),
    ]
    endpoint_overrides = (
        ("--feishu-unidepth-base-url", args.feishu_unidepth_base_url),
        ("--feishu-hands-wilor-base-url", args.feishu_hands_wilor_base_url),
        ("--feishu-droid-base-url", args.feishu_droid_base_url),
        ("--feishu-hawor-base-url", args.feishu_hawor_base_url),
    )
    for flag, value in endpoint_overrides:
        if value:
            command.extend([flag, str(value)])
    if args.render_width is not None:
        command.extend(["--render-width", str(args.render_width)])
    return command


def child_request_command(args: argparse.Namespace, item_index: int, video: Path, *, launch_token: str | None = None) -> list[str]:
    """Build one independent complete-pipeline process command for tmux admission."""
    command = [
        str(Path(args.pipeline_python).expanduser()),
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--repo-root",
        str(args.repo_root),
        "--pipeline-python",
        str(args.pipeline_python),
        "--prepare-python",
        str(args.prepare_python),
        "--service-timeout-s",
        str(float(args.service_timeout_s)),
        "--total-request-limit",
        str(int(args.total_request_limit)),
        "--algorithm-inflight-multiplier",
        str(int(args.algorithm_inflight_multiplier)),
        "--retry-max-wait-s",
        str(float(getattr(args, "retry_max_wait_s", 0.0))),
        "--retry-initial-delay-s",
        str(float(getattr(args, "retry_initial_delay_s", 1.0))),
        "--single-item-index",
        str(int(item_index)),
        "--single-video",
        str(video),
    ]
    if launch_token is not None:
        command.extend(["--launch-token", str(launch_token)])
    if args.render_width is not None:
        command.extend(["--render-width", str(args.render_width)])
    if args.rerun_completed:
        command.append("--rerun-completed")
    command.extend(["--feishu-service-profile", str(args.feishu_service_profile), "--hawor-root", str(args.hawor_root)])
    for flag, value in (
        ("--feishu-unidepth-base-url", args.feishu_unidepth_base_url),
        ("--feishu-hands-wilor-base-url", args.feishu_hands_wilor_base_url),
        ("--feishu-droid-base-url", args.feishu_droid_base_url),
        ("--feishu-hawor-base-url", args.feishu_hawor_base_url),
    ):
        if value:
            command.extend([flag, str(value)])
    return command


def write_item_result(output_root: Path, row: dict[str, Any]) -> Path:
    item_root = output_root / "items" / f"item_{int(row['item_index']):06d}"
    result_path = item_root / "item_result.json"
    write_json_atomic(result_path, row)
    return result_path


def run_managed_complete_request(
    args: argparse.Namespace,
    *,
    video: Path,
    item_index: int,
    output_root: Path,
) -> dict[str, Any]:
    with acquire_algorithm_slots(
        output_root,
        item_index=item_index,
        multiplier=int(getattr(args, "algorithm_inflight_multiplier", 2)),
    ) as reservation:
        row = run_complete_request(args, video=video, item_index=item_index, output_root=output_root)
        row["algorithm_admission"] = reservation
        return row


def run_single_item_child(args: argparse.Namespace, videos: list[Path]) -> int:
    index = int(args.single_item_index)
    video = Path(args.single_video).expanduser().resolve() if args.single_video else None
    if video is None:
        if index < 0 or index >= len(videos):
            raise IndexError(f"--single-item-index {index} is outside discovered video count {len(videos)}")
        video = videos[index]
    if not video.is_file():
        raise FileNotFoundError(f"single child video is missing: {video}")
    started = time.time()
    try:
        row = run_managed_complete_request(args, video=video, item_index=index, output_root=args.output_root)
    except Exception as exc:
        row = {
            "status": "failed_launcher",
            "item_index": index,
            "video": str(video),
            "error": repr(exc),
            "finished_at": utc_now(),
        }
    row["launch_token"] = str(getattr(args, "launch_token", ""))
    row["child_pid"] = os.getpid()
    row["child_elapsed_s"] = float(time.time() - started)
    row["item_result_path"] = str(write_item_result(args.output_root, row))
    append_jsonl(args.output_root / "dataset_request_events.jsonl", row)
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    return 1 if str(row.get("status", "")).startswith("failed") else 0


def ensure_tmux_session(session_name: str) -> None:
    probe = subprocess.run(["tmux", "has-session", "-t", session_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if probe.returncode == 0:
        return
    created = subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "controller", "bash"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if created.returncode != 0:
        raise RuntimeError(f"could not create tmux session {session_name}: {created.stderr.strip()}")


def _fresh_item_result(path: Path, launch_token: str) -> dict[str, Any] | None:
    result = load_json(path)
    if result is None or str(result.get("launch_token") or "") != str(launch_token):
        return None
    return result


def tmux_window_exists(session_name: str, window_name: str) -> bool:
    probe = subprocess.run(
        ["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return probe.returncode == 0 and window_name in set(probe.stdout.splitlines())


def wait_for_one_request(
    active: dict[int, dict[str, Any]],
    *,
    output_root: Path,
    events_path: Path,
) -> int:
    """Block until one admitted child publishes its terminal result."""
    while True:
        for index, state in list(active.items()):
            result_path = output_root / "items" / f"item_{index:06d}" / "item_result.json"
            result = _fresh_item_result(result_path, str(state["launch_token"]))
            if result is None and not tmux_window_exists(str(state["tmux_session"]), str(state["tmux_window"])):
                result = {
                    "status": "failed_launcher",
                    "item_index": index,
                    "case_id": state["case_id"],
                    "video": state["video"],
                    "launch_token": state["launch_token"],
                    "error": "tmux child exited without publishing item_result.json",
                    "finished_at": utc_now(),
                }
                write_json_atomic(result_path, result)
                append_jsonl(events_path, result)
            if result is None:
                continue
            release = {
                "event": "request_released",
                "status": "request_released",
                "item_index": index,
                "case_id": state["case_id"],
                "request_result": str(result_path),
                "released_at": utc_now(),
                "result_status": result.get("status"),
            }
            append_jsonl(events_path, release)
            del active[index]
            return index
        time.sleep(0.25)


def rapidly_admit_tmux_requests(args: argparse.Namespace, videos: list[Path], *, started: float) -> dict[str, Any]:
    """Run a rolling top-level request window over complete video annotations.

    At most ``total_request_limit`` children are admitted at once.  The next
    video is not submitted until one current child writes ``item_result.json``.
    The existing complete-pipeline flock remains an independent compute cap;
    algorithm reservations are acquired inside each child before the internal
    pipeline starts.
    """
    ensure_tmux_session(str(args.tmux_session))
    events_path = args.output_root / "dataset_request_events.jsonl"
    admission_path = args.output_root / "dataset_admission.jsonl"
    slot_dir = args.output_root / ".rapid_complete_pipeline_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    active_limit = int(args.rapid_active_limit)
    total_request_limit = int(args.total_request_limit)
    if total_request_limit <= 0:
        raise ValueError("--total-request-limit must be positive")
    admissions: list[dict[str, Any]] = []
    active: dict[int, dict[str, Any]] = {}
    skipped_item_indices = set(getattr(args, "skip_item_index", []) or [])

    for index, video in enumerate(videos):
        while len(active) >= total_request_limit:
            wait_for_one_request(active, output_root=args.output_root, events_path=events_path)
        case_id = f"egoscale30h_{index:06d}_{video.stem}"
        if index in skipped_item_indices:
            row = {"event": "skipped_active", "status": "skipped_active", "item_index": index, "case_id": case_id, "video": str(video), "admitted_at": utc_now()}
            admissions.append(row)
            append_jsonl(admission_path, row)
            append_jsonl(events_path, row)
            continue
        item_root = args.output_root / "items" / f"item_{index:06d}"
        if any(item_root.glob("attempt_*")):
            identity = video_identity(video)
            existing = find_completed_attempt(item_root, expected_case_id=case_id, expected_video=identity)
        else:
            stat = video.stat()
            identity = {"path": str(video.resolve()), "size_bytes": int(stat.st_size), "sha256_deferred_to_child": True}
            existing = None
        if existing is not None and not args.rerun_completed:
            row = {"event": "skipped_completed", "status": "skipped_completed", "item_index": index, "case_id": case_id, "video": str(video), "run_root": str(existing), "admitted_at": utc_now()}
            admissions.append(row)
            append_jsonl(admission_path, row)
            append_jsonl(events_path, row)
            continue
        item_root.mkdir(parents=True, exist_ok=True)
        log_path = item_root / "logs" / "tmux_admission.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        launch_token = uuid.uuid4().hex
        command = child_request_command(args, index, video, launch_token=launch_token)
        slot_index = index % active_limit
        slot_lock = slot_dir / f"slot_{slot_index:03d}.lock"
        shell_command = f"exec 9> {shlex.quote(str(slot_lock))}; flock -x 9; exec {shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
        window_name = f"item_{index:06d}_{launch_token[:8]}"[:48]
        launched = subprocess.run(["tmux", "new-window", "-d", "-t", str(args.tmux_session), "-n", window_name, "bash", "-lc", shell_command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if launched.returncode != 0:
            row = {"event": "admission_failed", "status": "failed_admission", "item_index": index, "case_id": case_id, "video": str(video), "error": launched.stderr.strip(), "command": command, "admitted_at": utc_now()}
            admissions.append(row)
            append_jsonl(admission_path, row)
            append_jsonl(events_path, row)
            continue
        row = {"event": "admitted", "status": "admitted", "item_index": index, "case_id": case_id, "video": str(video), "video_identity": identity, "tmux_session": str(args.tmux_session), "tmux_window": window_name, "complete_pipeline_slot": slot_index, "complete_pipeline_active_limit": active_limit, "slot_lock": str(slot_lock), "log": str(log_path), "command": command, "admitted_at": utc_now(), "total_request_limit": total_request_limit}
        admissions.append(row)
        active[index] = {"case_id": case_id, "video": str(video), "launch_token": launch_token, "tmux_session": str(args.tmux_session), "tmux_window": window_name}
        append_jsonl(admission_path, row)

    while active:
        wait_for_one_request(active, output_root=args.output_root, events_path=events_path)
    summary = summarize(args, videos, admissions, started=started, include_items=True)
    summary.update({"status": "admitted", "submission_mode": "rapid_tmux", "tmux_session": str(args.tmux_session), "admitted_count": sum(1 for row in admissions if row.get("status") == "admitted"), "skipped_completed_count": sum(1 for row in admissions if row.get("status") == "skipped_completed"), "skipped_active_count": sum(1 for row in admissions if row.get("status") == "skipped_active"), "complete_pipeline_active_limit": active_limit, "total_request_limit": total_request_limit, "algorithm_inflight_multiplier": int(args.algorithm_inflight_multiplier), "algorithm_admission_limits": algorithm_admission_limits(int(args.algorithm_inflight_multiplier)), "slot_directory": str(slot_dir), "algorithm_admission_directory": str(args.output_root / ".algorithm_admission"), "admission_path": str(admission_path), "completion_state": "children_write_item_result_and_dataset_request_events"})
    write_json_atomic(args.output_root / "dataset_batch_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if not any(str(row.get("status", "")).startswith("failed") for row in admissions) else 1


def run_complete_request(
    args: argparse.Namespace,
    *,
    video: Path,
    item_index: int,
    output_root: Path,
) -> dict[str, Any]:
    case_id = f"egoscale30h_{item_index:06d}_{video.stem}"
    item_root = output_root / "items" / f"item_{item_index:06d}"
    identity = video_identity(video)
    existing = find_completed_attempt(item_root, expected_case_id=case_id, expected_video=identity)
    if existing is not None and not args.rerun_completed:
        return {
            "status": "skipped_completed",
            "case_id": case_id,
            "item_index": item_index,
            "video": str(video),
            "video_identity": identity,
            "run_root": str(existing),
            "finished_at": utc_now(),
        }

    run_root = reserve_attempt_root(item_root)
    request_started = time.time()
    prepare = run_command(
        prepare_command(args, video, run_root, case_id),
        cwd=args.repo_root,
        log_path=run_root / "logs" / "launcher_prepare.log",
    )
    if prepare["status"] != "ok":
        return {
            "status": "failed_prepare",
            "case_id": case_id,
            "item_index": item_index,
            "video": str(video),
            "video_identity": identity,
            "run_root": str(run_root),
            "prepare": prepare,
            "elapsed_s": float(time.time() - request_started),
            "finished_at": utc_now(),
        }

    pipeline = run_command(
        pipeline_command(args, video, run_root, case_id),
        cwd=args.repo_root,
        log_path=run_root / "logs" / "launcher_pipeline.log",
    )
    physically_complete = pipeline["status"] == "ok" and completed_attempt(
        run_root,
        expected_case_id=case_id,
        expected_video=identity,
    )
    return {
        "status": "completed" if physically_complete else "failed_pipeline",
        "case_id": case_id,
        "item_index": item_index,
        "video": str(video),
        "video_identity": identity,
        "run_root": str(run_root),
        "prepare": prepare,
        "pipeline": pipeline,
        "physical_delivery_complete": physically_complete,
        "elapsed_s": float(time.time() - request_started),
        "finished_at": utc_now(),
    }


def summarize(
    args: argparse.Namespace,
    videos: list[Path],
    results: list[dict[str, Any]],
    *,
    started: float,
    include_items: bool = False,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    terminal_count = sum(counts.values())
    failures = sum(count for status, count in counts.items() if status.startswith("failed"))
    summary = {
        "schema": "v22_feishu_ray_egoscale30h_batch.v1",
        "status": "running" if terminal_count < len(videos) else "completed_with_errors" if failures else "completed",
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "model_execution": "feishu_ray",
        "cosmos_enabled": False,
        "scheduling_unit": "one_complete_per_video_annotation_pipeline_request",
        "service_call_semantics": "synchronous_call_wait_for_response; endpoint_queueing_owned_by_ray_serve",
        "pipeline_concurrency": int(args.pipeline_concurrency),
        "total_request_limit": int(args.total_request_limit),
        "algorithm_inflight_multiplier": int(args.algorithm_inflight_multiplier),
        "algorithm_admission_limits": algorithm_admission_limits(int(args.algorithm_inflight_multiplier)),
        "algorithm_admission_mode": "manager_route_proxy_plus_droid_session_lock",
        "algorithm_admission_events": str(args.output_root / "algorithm_admission_events.jsonl"),
        "video_count": len(videos),
        "terminal_count": terminal_count,
        "status_counts": counts,
        "started_at_unix": started,
        "updated_at": utc_now(),
        "elapsed_s": float(time.time() - started),
        "item_results_jsonl": str(args.output_root / "dataset_request_events.jsonl"),
    }
    if include_items:
        summary["items"] = sorted(results, key=lambda row: int(row.get("item_index", -1)))
    return summary


def api_job_id(args: argparse.Namespace, *, item_index: int, video: Path) -> str:
    prefix = args.api_job_prefix or args.output_root.name
    raw = f"{prefix}_{item_index:06d}_{video.stem}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return (cleaned[:96] or f"egoscale30h_{item_index:06d}")


def api_annotation_url(args: argparse.Namespace) -> str:
    return f"{str(args.api_base_url).rstrip('/')}/v1/annotation-jobs"


def cosmos_api_summary_is_complete(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    cosmos = summary.get("cosmos")
    if not isinstance(cosmos, dict):
        return False
    return (
        cosmos.get("status") in {"enabled", "completed_with_anomalies"}
        and isinstance(cosmos.get("request_count"), int)
        and not isinstance(cosmos["request_count"], bool)
        and cosmos["request_count"] > 0
        and isinstance(cosmos.get("semantic_row_count"), int)
        and not isinstance(cosmos["semantic_row_count"], bool)
        and cosmos["semantic_row_count"] > 0
        and all(isinstance(cosmos.get(key), str) and bool(cosmos[key]) for key in ("review_json", "captioned_combined_video"))
    )


def summarize_api_http(
    args: argparse.Namespace,
    videos: list[Path],
    results: list[dict[str, Any]],
    *,
    started: float,
    include_items: bool = False,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    terminal_count = len(results)
    failed = sum(count for status, count in counts.items() if status.startswith("failed"))
    stability_control = dict(getattr(args, "stability_control", {}))
    submitted_count = int(stability_control.get("submitted_count", len(videos)))
    submission_closed = bool(stability_control.get("submission_closed", False))
    if not submission_closed or terminal_count < submitted_count:
        status = "running"
    elif failed:
        status = "completed_with_failures"
    else:
        status = "completed"
    fps_condition = getattr(args, "fps_condition", get_fps_condition(DEFAULT_FPS_CONDITION))
    summary: dict[str, Any] = {
        "status": status,
        "submission_mode": "api_http",
        "api_base_url": str(args.api_base_url),
        "api_route": "/v1/annotation-jobs",
        "model_backend": str(args.api_model_backend),
        "diagnostic_monocular": bool(args.api_diagnostic_monocular),
        "cosmos_enabled": True,
        "fps_sampling": {"condition": fps_condition.name, "unidepth_fps": fps_condition.unidepth_fps, "droid_submission": "source_indices_[0,min(N,1024)); droid_fps config is not applied", "source": "ego_annotation.fps_config built-in manager condition"},
        "item_batch_size": 1,
        "outer_http_client_concurrency": int(args.api_client_concurrency),
        "stability_window": {
            "video_limit": int(getattr(args, "stability_video_limit", 0)),
            "warmup_video_count": int(getattr(args, "stability_warmup_count", DEFAULT_STABILITY_WARMUP_COUNT)),
            "window_size": int(getattr(args, "stability_window_size", DEFAULT_STABILITY_WINDOW_SIZE)),
            "tolerance": float(getattr(args, "stability_tolerance", DEFAULT_STABILITY_TOLERANCE)),
            "limit_semantics": "0 means traverse the dataset until stable; positive value caps rolling submissions",
        },
        "stability_control": stability_control,
        "manager_total_request_limit": int(args.total_request_limit),
        "manager_algorithm_inflight_multiplier": int(args.algorithm_inflight_multiplier),
        "admission_owner": "single_item_api_manager",
        "eligible_video_count": len(videos),
        "submitted_count": submitted_count,
        "terminal_count": terminal_count,
        "status_counts": counts,
        "started_at_unix": started,
        "updated_at": utc_now(),
        "elapsed_s": float(time.time() - started),
        "item_results_jsonl": str(args.output_root / "dataset_request_events.jsonl"),
    }
    if include_items:
        summary["items"] = sorted(results, key=lambda row: int(row.get("item_index", -1)))
    return summary


def aggregate_video_service_lane_traces(summary: Any) -> dict[str, dict[str, Any]]:
    """Collapse one complete-video response's stage traces into service lanes."""
    if not isinstance(summary, dict):
        return {}
    performance = summary.get("performance")
    traces = performance.get("request_traces") if isinstance(performance, dict) else None
    if not isinstance(traces, list):
        return {}
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        if isinstance(trace, dict) and isinstance(trace.get("stage_id"), str):
            by_stage.setdefault(str(trace["stage_id"]), []).append(trace)
    lanes: dict[str, dict[str, Any]] = {}
    for service, stage_ids in VIDEO_STREAM_SERVICE_STAGES.items():
        selected = [trace for stage_id in stage_ids for trace in by_stage.get(stage_id, ())]
        present = {str(trace.get("stage_id")) for trace in selected}
        missing = [stage_id for stage_id in stage_ids if stage_id not in present]
        starts = [float(trace["started_monotonic_s"]) for trace in selected if isinstance(trace.get("started_monotonic_s"), (int, float))]
        finishes = [float(trace["completed_monotonic_s"]) for trace in selected if isinstance(trace.get("completed_monotonic_s"), (int, float))]
        if missing or not starts or not finishes:
            continue
        started = min(starts)
        completed = max(finishes)
        lanes[service] = {
            "service": service,
            "stage_ids": list(stage_ids),
            "request_count": sum(int(trace.get("request_count") or 0) for trace in selected),
            "work_units": sum(int(trace.get("request_count") or 0) * int((trace.get("native_work_shape") or [1])[0] if isinstance(trace.get("native_work_shape"), list) and trace.get("native_work_shape") else 1) for trace in selected),
            "started_monotonic_s": started,
            "completed_monotonic_s": completed,
            "total_wall_s": max(0.0, completed - started),
        }
    return lanes


def build_service_performance_metrics(results: list[dict[str, Any]], *, output_root: Path, manager_limit: int, algorithm_multiplier: int, outer_limit: int) -> dict[str, Any]:
    """Write observer-only service metrics; unavailable server/GPU fields stay explicit."""
    capacities = {"unidepth": 32, "hands.detect": 32, "wilor": 64, "hawor.track": 16, "hawor.infiller": 8, "cosmos3": 32, "droid": 8}
    services: dict[str, dict[str, Any]] = {}
    for service, stage_ids in VIDEO_STREAM_SERVICE_STAGES.items():
        lanes = [row.get("service_lane_traces", {}).get(service) for row in results if isinstance(row.get("service_lane_traces"), dict)]
        lanes = [lane for lane in lanes if isinstance(lane, dict)]
        intervals = [(float(lane["started_monotonic_s"]), float(lane["completed_monotonic_s"])) for lane in lanes if isinstance(lane.get("started_monotonic_s"), (int, float)) and isinstance(lane.get("completed_monotonic_s"), (int, float)) and float(lane["completed_monotonic_s"]) >= float(lane["started_monotonic_s"])]
        points = sorted((value, delta) for start, end in intervals for value, delta in ((start, 1), (end, -1)))
        active = 0
        observed_max = 0
        for _value, delta in points:
            active += delta
            observed_max = max(observed_max, active)
        request_count = sum(int(lane.get("request_count") or 0) for lane in lanes)
        wall_s = sum(max(0.0, float(lane.get("total_wall_s") if lane.get("total_wall_s") is not None else float(lane["completed_monotonic_s"]) - float(lane["started_monotonic_s"]))) for lane in lanes)
        work_units = sum(int(lane.get("work_units") or lane.get("request_count") or 0) for lane in lanes)
        services[service] = {
            "service": service, "stage_ids": list(stage_ids), "request_count": request_count,
            "native_work_units": work_units, "active_interval_sum_s": wall_s,
            "observed_req_per_s": (request_count / wall_s) if wall_s else None,
            "effective_img_per_s": (work_units / wall_s) if wall_s else None,
            "configured_capacity": capacities.get(service), "observed_max_in_flight": observed_max,
            "capacity_comparison": {"configured": capacities.get(service), "observed_max": observed_max, "within_configured": observed_max <= capacities.get(service, observed_max)},
            "queue_depth": None, "pending_requests": None, "running_batches": None,
            "queue_metrics_status": "unavailable_without_manager_or_service_telemetry",
            "gpu": {"gpu_id": None, "utilization_pct": None, "memory_used_bytes": None, "power_watts": None, "status": "unavailable_without_service_observer"},
        }
    metrics = {
        "schema": "ego.annotation.service_performance_observer.v1",
        "observer_scope": "client-visible request traces and archived admission events; no service-internal compute claim",
        "design_limits": {"outer_concurrency": outer_limit, "manager_total_requests": manager_limit, "algorithm_multiplier": algorithm_multiplier, "unidepth": 32, "hands": 32, "wilor": 64, "hawor": 16, "infiller": 8, "cosmos": 32, "droid_absolute": 8},
        "services": services,
        "admission_telemetry": {"path": str(output_root / "algorithm_admission_events.jsonl"), "status": "archived_separately"},
        "nvidia_smi": {"status": "not_collected_by_client_only_task", "gpu_id": None, "utilization_pct": None, "memory_used_bytes": None, "power_watts": None},
    }
    write_json_atomic(output_root / "service_performance_metrics.json", metrics)
    return metrics


def archive_condition_algorithm_events(output_root: Path, results: list[dict[str, Any]], *, source: Path) -> dict[str, Any]:
    video_job_ids = {str(row.get("job_id") or row.get("request_token")) for row in results if row.get("job_id") or row.get("request_token")}
    destination = output_root / "algorithm_admission_events.jsonl"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    matched = 0
    malformed = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as target:
        if source.is_file():
            with source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    if isinstance(row, dict) and str(row.get("video_job_id") or "") in video_job_ids:
                        target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        matched += 1
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, destination)
    return {"source": str(source), "destination": str(destination), "video_job_count": len(video_job_ids), "matched_event_count": matched, "malformed_source_lines": malformed, "status": "ok" if source.is_file() else "missing_source"}


async def run_api_http_requests_async(args: argparse.Namespace, videos: list[Path], *, started: float) -> int:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("--submission-mode api_http requires aiohttp") from exc

    events_path = args.output_root / "dataset_request_events.jsonl"
    summary_path = args.output_root / "dataset_batch_summary.json"
    results: list[dict[str, Any]] = []
    configured_client_limit = int(args.api_client_concurrency)
    client_limit = configured_client_limit if configured_client_limit > 0 else len(videos)
    producer_only = int(getattr(args, "stability_video_limit", 0)) == 0
    args.stability_control = {
        "mode": "full_dataset_producer" if producer_only else "rolling_fixed_inflight",
        "producer_only": producer_only,
        "target_inflight": client_limit,
        "max_submission_count": len(videos),
        "submitted_count": 0,
        "measurement_terminal_count": 0,
        "drain_terminal_count": 0,
        "stability_reached": False,
        "stable_at_completion_count": None,
        "stable_window": None,
        "service_stability": {},
        "required_services": list(VIDEO_STREAM_SERVICE_STAGES),
        "submission_closed": False,
    }
    write_json_atomic(summary_path, summarize_api_http(args, videos, results, started=started))
    request_url = api_annotation_url(args)
    client_slots = asyncio.Semaphore(min(client_limit, len(videos))) if client_limit > 0 else None
    timeout = aiohttp.ClientTimeout(total=float(args.api_request_timeout_s))
    connector = aiohttp.TCPConnector(limit=client_limit if client_limit > 0 else 0, force_close=False)

    @contextlib.asynccontextmanager
    async def acquire_outer_client_slot() -> Iterator[None]:
        if client_slots is None:
            yield
            return
        async with client_slots:
            yield

    async def submit_one(session: Any, item_index: int, video: Path) -> dict[str, Any]:
        job_id = api_job_id(args, item_index=item_index, video=video)
        fps_condition = getattr(args, "fps_condition", get_fps_condition(DEFAULT_FPS_CONDITION))
        queued_at_unix = time.time()
        queued_at = utc_now()
        append_jsonl(
            events_path,
            {
                "event": "queued",
                "status": "queued",
                "item_index": item_index,
                "request_token": job_id,
                "video": str(video),
                "queued_at": queued_at,
                "queued_at_unix": queued_at_unix,
            },
        )
        queued_started = time.monotonic()
        async with acquire_outer_client_slot():
            request_started = time.monotonic()
            request_started_at_unix = time.time()
            append_jsonl(
                events_path,
                {
                    "event": "request_started",
                    "status": "request_started",
                    "item_index": item_index,
                    "request_token": job_id,
                    "video": str(video),
                    "queued_at": queued_at,
                    "request_started_at": utc_now(),
                    "request_started_at_unix": request_started_at_unix,
                    "client_queue_wait_s": float(request_started - queued_started),
                    "request_url": request_url,
                },
            )
            submit_started = request_started
            upload_prepare_started = time.monotonic()
            upload_prepare_s = 0.0
            manager_http_wait_s = 0.0
            submitter_response_decode_s = 0.0
            try:
                with video.open("rb") as handle:
                    form = aiohttp.FormData()
                    form.add_field("file", handle, filename=video.name, content_type="video/mp4")
                    upload_prepare_s = time.monotonic() - upload_prepare_started
                    manager_http_started = time.monotonic()
                    async with session.post(request_url, data=form, headers={"Accept": "application/json", "X-Ego-Api-Ify-Fps-Condition": fps_condition.name}) as response:
                        body_bytes = await response.read()
                        manager_http_wait_s = time.monotonic() - manager_http_started
                        decode_started = time.monotonic()
                        body = body_bytes.decode("utf-8", errors="replace")
                        try:
                            payload = json.loads(body)
                        except json.JSONDecodeError:
                            payload = None
                        submitter_response_decode_s = time.monotonic() - decode_started
                        http_status = int(response.status)
                submit_total_wall_s = time.monotonic() - submit_started
                if 200 <= http_status < 300 and isinstance(payload, dict):
                    response_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
                    cosmos_complete = cosmos_api_summary_is_complete(response_summary)
                    row = {
                        "event": "terminal",
                        "status": "completed" if cosmos_complete else "failed_product",
                        "item_index": item_index,
                        "request_token": job_id,
                        "job_id": payload.get("job_id"),
                        "video": str(video),
                        "http_status": http_status,
                        "response_status": payload.get("status"),
                        "download_url": payload.get("download_url"),
                        "package_path": payload.get("package_path"),
                        "remote_run_root": payload.get("remote_run_root"),
                        "api_run_root": payload.get("run_root"),
                        "summary_accepted": response_summary.get("accepted"),
                        "cosmos": response_summary.get("cosmos"),
                        "cosmos_product_complete": cosmos_complete,
                        "service_lane_traces": aggregate_video_service_lane_traces(response_summary),
                        "elapsed_s": float(time.monotonic() - request_started),
                        "upload_prepare_s": float(upload_prepare_s),
                        "manager_http_wait_s": float(manager_http_wait_s),
                        "submitter_response_decode_s": float(submitter_response_decode_s),
                        "total_submit_wall_s": float(submit_total_wall_s),
                        "finished_at": utc_now(),
                        "finished_at_unix": time.time(),
                    }
                else:
                    row = {
                        "event": "terminal",
                        "status": "failed_http",
                        "item_index": item_index,
                        "request_token": job_id,
                        "job_id": payload.get("job_id") if isinstance(payload, dict) else None,
                        "video": str(video),
                        "http_status": http_status,
                        "response": payload if isinstance(payload, dict) else body[-4000:],
                        "elapsed_s": float(time.monotonic() - request_started),
                        "upload_prepare_s": float(upload_prepare_s),
                        "manager_http_wait_s": float(manager_http_wait_s),
                        "submitter_response_decode_s": float(submitter_response_decode_s),
                        "total_submit_wall_s": float(time.monotonic() - submit_started),
                        "finished_at": utc_now(),
                        "finished_at_unix": time.time(),
                    }
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                row = {
                    "event": "terminal",
                    "status": "failed_client",
                    "item_index": item_index,
                    "request_token": job_id,
                    "job_id": None,
                    "video": str(video),
                    "error": repr(exc),
                    "elapsed_s": float(time.monotonic() - request_started),
                    "upload_prepare_s": float(upload_prepare_s),
                    "manager_http_wait_s": float(manager_http_wait_s),
                    "submitter_response_decode_s": float(submitter_response_decode_s),
                    "total_submit_wall_s": float(time.monotonic() - submit_started),
                    "finished_at": utc_now(),
                    "finished_at_unix": time.time(),
                }
            row["request_id"] = job_id
            row["timing_schema"] = "submitter_timing.v1"
            return row

    next_index = 0
    in_flight: set[asyncio.Task[Any]] = set()
    measurement_completion_times: list[float] = []
    service_completion_times: dict[str, list[float]] = {service: [] for service in VIDEO_STREAM_SERVICE_STAGES}
    latest_stability_evaluation: dict[str, Any] | None = None

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        def submit_next() -> None:
            nonlocal next_index
            task = asyncio.create_task(submit_one(session, next_index, videos[next_index]))
            in_flight.add(task)
            next_index += 1
            args.stability_control["submitted_count"] = next_index

        while next_index < len(videos) and len(in_flight) < client_limit:
            submit_next()
        if next_index >= len(videos):
            args.stability_control["submission_closed"] = True
        write_json_atomic(summary_path, summarize_api_http(args, videos, results, started=started))

        while in_flight:
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            in_flight.difference_update(done)
            completed_rows = sorted((task.result() for task in done), key=lambda row: float(row.get("finished_at_unix") or 0.0))
            for row in completed_rows:
                if producer_only:
                    row["measurement_phase"] = "producer"
                    args.stability_control["measurement_terminal_count"] += 1
                elif args.stability_control["stability_reached"]:
                    row["measurement_phase"] = "drain_after_stability"
                    args.stability_control["drain_terminal_count"] += 1
                else:
                    row["measurement_phase"] = "measurement"
                    args.stability_control["measurement_terminal_count"] += 1
                    if row.get("status") == "completed" and row.get("finished_at_unix") is not None:
                        measurement_completion_times.append(float(row["finished_at_unix"]))
                        lane_traces = row.get("service_lane_traces") if isinstance(row.get("service_lane_traces"), dict) else {}
                        for service in VIDEO_STREAM_SERVICE_STAGES:
                            lane = lane_traces.get(service) if isinstance(lane_traces.get(service), dict) else None
                            completed_lane = lane.get("completed_monotonic_s") if lane is not None else None
                            if isinstance(completed_lane, (int, float)):
                                service_completion_times[service].append(float(completed_lane))
                        stability_options = {
                            "warmup_count": int(getattr(args, "stability_warmup_count", DEFAULT_STABILITY_WARMUP_COUNT)),
                            "window_size": int(getattr(args, "stability_window_size", DEFAULT_STABILITY_WINDOW_SIZE)),
                            "tolerance": float(getattr(args, "stability_tolerance", DEFAULT_STABILITY_TOLERANCE)),
                        }
                        latest_stability_evaluation = stability_windows(measurement_completion_times, **stability_options)
                        service_stability = {
                            service: stability_windows(times, **stability_options)
                            for service, times in service_completion_times.items()
                        }
                        args.stability_control["latest_stability_evaluation"] = latest_stability_evaluation
                        args.stability_control["service_stability"] = service_stability
                        args.stability_control["missing_service_lane_markers"] = {
                            service: len(measurement_completion_times) - len(times)
                            for service, times in service_completion_times.items()
                        }
                        all_services_stable = all(
                            service_stability[service]["stable"]
                            and len(service_completion_times[service]) == len(measurement_completion_times)
                            for service in VIDEO_STREAM_SERVICE_STAGES
                        )
                        if latest_stability_evaluation["stable"] and all_services_stable:
                            args.stability_control["stability_reached"] = True
                            args.stability_control["stable_at_completion_count"] = len(measurement_completion_times)
                            args.stability_control["stable_window"] = latest_stability_evaluation
                            args.stability_control["submission_closed"] = True
                append_jsonl(events_path, row)
                results.append(row)
                write_json_atomic(summary_path, summarize_api_http(args, videos, results, started=started))

            while (
                (producer_only or not args.stability_control["stability_reached"])
                and next_index < len(videos)
                and len(in_flight) < client_limit
            ):
                submit_next()
            if next_index >= len(videos):
                args.stability_control["submission_closed"] = True
            write_json_atomic(summary_path, summarize_api_http(args, videos, results, started=started))

    args.stability_control["submission_closed"] = True
    if latest_stability_evaluation is not None and args.stability_control["stable_window"] is None:
        args.stability_control["latest_stability_evaluation"] = latest_stability_evaluation
    archive = archive_condition_algorithm_events(args.output_root, results, source=FIXED_API_JOB_ROOT / "_algorithm_admission_events.jsonl")
    service_metrics = build_service_performance_metrics(
        results, output_root=args.output_root, manager_limit=int(args.total_request_limit),
        algorithm_multiplier=int(args.algorithm_inflight_multiplier), outer_limit=int(args.api_client_concurrency),
    )
    summary = summarize_api_http(args, videos, results, started=started, include_items=True)
    summary["algorithm_event_archive"] = archive
    summary["service_performance_metrics"] = service_metrics
    summary["service_performance_metrics_path"] = str(args.output_root / "service_performance_metrics.json")
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    terminal_failures = [
        row
        for row in results
        if row.get("measurement_phase") in {"measurement", "producer"} and str(row.get("status", "")).startswith("failed")
    ]
    return 1 if terminal_failures else 0


def run_api_http_requests(args: argparse.Namespace, videos: list[Path], *, started: float) -> int:
    return asyncio.run(run_api_http_requests_async(args, videos, started=started))


def _environment_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _environment_float(name: str, default: float, *, minimum_exclusive: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value <= minimum_exclusive:
        raise ValueError(f"{name} must be > {minimum_exclusive}")
    return value


def parse_args() -> argparse.Namespace:
    # The production traversal is intentionally a closed, zero-argument entry.
    # Legacy launcher helpers remain in this module for compatibility, but no
    # command-line switch can select or reconfigure them.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    fps_condition = get_fps_condition(os.environ.get("EGO_API_IFY_FPS_CONDITION", DEFAULT_FPS_CONDITION))
    stability_video_limit = _environment_int(STABILITY_VIDEO_LIMIT_ENV, 0, minimum=0)
    stability_warmup_count = _environment_int(STABILITY_WARMUP_COUNT_ENV, DEFAULT_STABILITY_WARMUP_COUNT, minimum=0)
    stability_window_size = _environment_int(STABILITY_WINDOW_SIZE_ENV, DEFAULT_STABILITY_WINDOW_SIZE, minimum=2)
    stability_tolerance = _environment_float(STABILITY_TOLERANCE_ENV, DEFAULT_STABILITY_TOLERANCE, minimum_exclusive=0.0)
    api_client_concurrency = _environment_int(API_CLIENT_CONCURRENCY_ENV, DEFAULT_API_CLIENT_CONCURRENCY, minimum=0)
    configured_dataset_root = os.environ.get(DATASET_ROOT_ENV, "").strip()
    configured_output_root = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    return argparse.Namespace(
        dataset_root=Path(configured_dataset_root) if configured_dataset_root else FIXED_DATASET_ROOT,
        output_root=Path(configured_output_root) if configured_output_root else None,
        repo_root=repo_root,
        pipeline_python="/home/zjh/miniconda3/envs/ego_foundation/bin/python",
        prepare_python="/home/zjh/miniconda3/envs/hamer/bin/python",
        pipeline_concurrency=2,
        submission_mode="api_http",
        api_base_url=FIXED_API_BASE_URL,
        api_client_concurrency=api_client_concurrency,
        api_model_backend="api_ify",
        api_diagnostic_monocular=True,
        api_request_timeout_s=FIXED_API_REQUEST_TIMEOUT_S,
        api_job_prefix=None,
        fps_condition=fps_condition,
        stability_video_limit=stability_video_limit,
        stability_warmup_count=stability_warmup_count,
        stability_window_size=stability_window_size,
        stability_tolerance=stability_tolerance,
        rapid_active_limit=16,
        tmux_session="ego_annotation_batch",
        single_item_index=None,
        single_video=None,
        launch_token="",
        skip_item_index=[],
        max_items=None,
        total_request_limit=FIXED_MANAGER_TOTAL_REQUEST_LIMIT,
        algorithm_inflight_multiplier=FIXED_MANAGER_ALGORITHM_MULTIPLIER,
        admission_proxy_host="127.0.0.1",
        admission_proxy_port=0,
        service_timeout_s=86400.0,
        retry_max_wait_s=0.0,
        retry_initial_delay_s=1.0,
        render_width=None,
        rerun_completed=False,
        feishu_service_profile=repo_root / "configs" / "feishu_ray_services.json",
        feishu_unidepth_base_url=None,
        feishu_hands_wilor_base_url=None,
        feishu_droid_base_url=None,
        feishu_hawor_base_url=None,
        hawor_root=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel/third_party/algorithms/hawor"),
    )


def run_bounded_requests(args: argparse.Namespace, videos: list[Path], *, started: float) -> int:
    results: list[dict[str, Any]] = []
    events_path = args.output_root / "dataset_request_events.jsonl"
    summary_path = args.output_root / "dataset_batch_summary.json"
    write_json_atomic(summary_path, summarize(args, videos, results, started=started))
    worker_count = min(int(args.pipeline_concurrency), int(args.total_request_limit), len(videos))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="complete_annotation") as executor:
        futures = {
            executor.submit(run_managed_complete_request, args, video=video, item_index=index, output_root=args.output_root): (index, video)
            for index, video in enumerate(videos)
        }
        for future in as_completed(futures):
            index, video = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"status": "failed_launcher", "item_index": index, "video": str(video), "error": repr(exc), "finished_at": utc_now()}
            results.append(row)
            append_jsonl(events_path, row)
            write_json_atomic(summary_path, summarize(args, videos, results, started=started))
    summary = summarize(args, videos, results, started=started, include_items=True)
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if any(str(row.get("status", "")).startswith("failed") for row in results) else 0


def main() -> int:
    args = parse_args()
    if args.pipeline_concurrency <= 0:
        raise ValueError("--pipeline-concurrency must be positive")
    if args.total_request_limit <= 0:
        raise ValueError("--total-request-limit must be positive")
    if args.algorithm_inflight_multiplier <= 0:
        raise ValueError("--algorithm-inflight-multiplier must be positive")
    if args.api_client_concurrency < 0:
        raise ValueError("--api-client-concurrency must be zero (unbounded) or positive")
    if args.api_request_timeout_s <= 0:
        raise ValueError("--api-request-timeout-s must be positive")
    if args.admission_proxy_port < 0 or args.admission_proxy_port > 65535:
        raise ValueError("--admission-proxy-port must be between 0 and 65535")
    if args.rapid_active_limit <= 0:
        raise ValueError("--rapid-active-limit must be positive")
    parsed_api_url = urlsplit(str(args.api_base_url))
    if parsed_api_url.scheme not in {"http", "https"} or not parsed_api_url.netloc or parsed_api_url.path not in {"", "/"}:
        raise ValueError("--api-base-url must be an HTTP(S) origin without a path")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.output_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_root = FIXED_OUTPUT_PARENT / f"v22_api_full_dataset_{stamp}"
    args.output_root = args.output_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.feishu_service_profile = args.feishu_service_profile.expanduser().resolve()
    args.hawor_root = args.hawor_root.expanduser().resolve()
    if args.submission_mode != "api_http" and not args.feishu_service_profile.is_file():
        raise FileNotFoundError(f"Feishu service profile is missing: {args.feishu_service_profile}")
    if args.submission_mode == "api_http" and args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"api_http requires a fresh empty --output-root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.single_video is not None:
        videos = [args.single_video.expanduser().resolve()]
    else:
        videos = discover_videos(args.dataset_root, args.max_items)
        if args.stability_video_limit > 0:
            videos = videos[: args.stability_video_limit]
    args.skip_item_index = sorted(set(args.skip_item_index))
    invalid_skip_indices = [index for index in args.skip_item_index if index < 0 or index >= len(videos)]
    if invalid_skip_indices:
        raise ValueError(f"--skip-item-index values are outside discovered video count {len(videos)}: {invalid_skip_indices}")
    if args.skip_item_index and args.submission_mode != "rapid_tmux":
        raise ValueError("--skip-item-index is supported only with --submission-mode rapid_tmux")
    if args.single_item_index is not None and args.submission_mode == "api_http":
        raise ValueError("internal --single-item-index is not part of the api_http path")
    started = time.time()
    if args.submission_mode == "api_http":
        return run_api_http_requests(args, videos, started=started)
    if args.single_item_index is not None:
        return run_single_item_child(args, videos)
    with run_algorithm_admission_proxy(args) as proxy_base_url:
        args.feishu_unidepth_base_url = proxy_base_url
        args.feishu_hands_wilor_base_url = proxy_base_url
        args.feishu_droid_base_url = proxy_base_url
        args.feishu_hawor_base_url = proxy_base_url
        if args.submission_mode == "rapid_tmux":
            return rapidly_admit_tmux_requests(args, videos, started=started)
        return run_bounded_requests(args, videos, started=started)


if __name__ == "__main__":
    raise SystemExit(main())
