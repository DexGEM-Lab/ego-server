#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_path(raw: object, key: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{key} must be a non-empty path string")
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"{key} does not exist: {path}")
    return path


def read_video_frame(path: Path, frame_index: int | None) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"invalid video shape for {path}: {width}x{height}, frames={frames}")
        index = frames // 2 if frame_index is None else int(frame_index)
        index = max(0, min(index, frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read frame {index} from {path}")
        return frame, {"frames": frames, "fps": fps, "width": width, "height": height, "frame_index": index}
    finally:
        cap.release()


def resized(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def put_text_block(image: np.ndarray, lines: list[str], x: int, y: int, scale: float, color: tuple[int, int, int]) -> int:
    line_height = max(14, int(round(25 * scale)))
    yy = int(y)
    for line in lines:
        cv2.putText(image, line, (int(x), yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, line, (int(x), yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        yy += line_height
    return yy


def short_name(name: str, max_chars: int) -> str:
    if len(name) <= max_chars:
        return name
    keep = max(8, (max_chars - 3) // 2)
    return f"{name[:keep]}...{name[-keep:]}"


def format_optional_metric(value: object, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def make_text_tile(width: int, height: int, lines: list[str]) -> np.ndarray:
    tile = np.full((int(height), int(width), 3), 232, dtype=np.uint8)
    put_text_block(tile, lines, 18, 34, 0.58, (35, 35, 35))
    return tile


def candidate_label(candidate: dict, replay: dict, max_chars: int) -> list[str]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else replay.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"candidate lacks metrics: {candidate.get('candidate_name')}")
    failed = []
    not_evaluated = []
    pass_rows = candidate.get("pass") if isinstance(candidate.get("pass"), dict) else replay.get("pass")
    if isinstance(pass_rows, dict):
        failed = [key for key, value in pass_rows.items() if value is False]
        not_evaluated = [key for key, value in pass_rows.items() if value is None]
    name = short_name(str(candidate.get("candidate_name", "candidate")), max_chars)
    kind = str(candidate.get("candidate_kind") or "generated_prior")
    stage = replay.get("rejection_stage") or "replay"
    first = f"{candidate.get('target_id')} | {kind} | {candidate.get('status')} | {stage} | {name}"
    observed_failure = replay.get("invalid_observed_target_keys") or []
    visible_p95 = metrics.get("visible_surface_coverage_p95_m")
    numbers = (
        f"visible_p95={format_optional_metric(visible_p95, 3)}m "
        f"IoU={format_optional_metric(metrics.get('silhouette_iou_median'), 3)} "
        f"depth_p95={format_optional_metric(metrics.get('zbuffer_abs_p95_median_m'), 3)}m"
    )
    fail_text = "observed failed: " + ", ".join(observed_failure[:2]) if observed_failure else "failed: " + ", ".join(failed[:4]) if failed else "failed: none"
    if not_evaluated:
        fail_text = (fail_text + "; not evaluated: " + ", ".join(not_evaluated[:3]))[: max_chars * 2]
    lines = []
    for text in (first, numbers, fail_text):
        lines.extend(textwrap.wrap(text, width=max_chars) or [""])
    return lines[:4]


def render_row(candidate: dict, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    replay = load_json(require_path(candidate.get("report"), "candidate.report"))
    kind = str(candidate.get("candidate_kind") or "generated_prior")
    if kind == "video_mesh":
        observed_video_key = "zbuffer_video"
        candidate_header = "video mesh replay"
    else:
        observed_video_key = "observed_target_zbuffer_video"
        candidate_header = "generated prior replay"
    observed_path = require_path(replay.get(observed_video_key), f"replay.{observed_video_key}")
    observed, observed_shape = read_video_frame(observed_path, args.frame_index)
    prior_path = replay.get("zbuffer_video")
    if isinstance(prior_path, str) and prior_path:
        prior, prior_shape = read_video_frame(require_path(prior_path, "replay.zbuffer_video"), args.frame_index)
        prior_video = prior_path
    else:
        stage = str(replay.get("rejection_stage") or "rejected before z-buffer")
        lines = [
            "prior z-buffer not rendered",
            stage,
            "observed target failed" if stage == "observed_target_replay" else "visible-surface coverage already failed",
        ]
        prior = make_text_tile(args.tile_width, args.tile_height, lines)
        prior_shape = {
            "frames": 0,
            "fps": 0.0,
            "width": int(args.tile_width),
            "height": int(args.tile_height),
            "frame_index": None,
        }
        prior_video = None
    observed = resized(observed, args.tile_width, args.tile_height)
    prior = resized(prior, args.tile_width, args.tile_height)
    row_h = int(args.tile_height + args.label_height)
    row_w = int(args.tile_width * 2)
    row = np.full((row_h, row_w, 3), 245, dtype=np.uint8)
    row[: args.tile_height, : args.tile_width] = observed
    row[: args.tile_height, args.tile_width :] = prior
    observed_header = "video mesh replay" if kind == "video_mesh" else "observed target replay"
    cv2.putText(row, observed_header, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(row, observed_header, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        row,
        candidate_header,
        (args.tile_width + 14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        row,
        "generated prior replay",
        (args.tile_width + 14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.line(row, (0, args.tile_height), (row_w, args.tile_height), (210, 210, 210), 1)
    put_text_block(
        row,
        candidate_label(candidate, replay, args.label_chars),
        14,
        args.tile_height + 24,
        args.label_scale,
        (35, 35, 35),
    )
    return row, {
        "target_id": candidate.get("target_id"),
        "candidate_name": candidate.get("candidate_name"),
        "status": candidate.get("status"),
        "annotation_ready": bool(candidate.get("annotation_ready", False)),
        "observed_video": str(observed_path),
        "prior_video": prior_video,
        "observed_frame": observed_shape,
        "prior_frame": prior_shape,
        "metrics": candidate.get("metrics"),
        "pass": candidate.get("pass"),
    }


def run(args: argparse.Namespace) -> dict:
    batch = load_json(args.batch_json)
    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"batch JSON has no candidates: {args.batch_json}")
    rows = []
    rendered = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("candidate entry must be a JSON object")
        row, row_report = render_row(candidate, args)
        rows.append(row)
        rendered.append(row_report)
    sheet = np.vstack(rows)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_png), sheet):
        raise RuntimeError(f"failed to write {args.output_png}")
    report = {
        "status": "ok",
        "method": "render_v7_prior_batch_visual_qc",
        "claim_tested": "candidate replay rejections are visually inspectable beside the measured observed-target replay",
        "batch_json": str(args.batch_json),
        "output_png": str(args.output_png),
        "candidates": rendered,
        "sheet_shape": {"height": int(sheet.shape[0]), "width": int(sheet.shape[1])},
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=270)
    parser.add_argument("--label-height", type=int, default=135)
    parser.add_argument("--label-chars", type=int, default=84)
    parser.add_argument("--label-scale", type=float, default=0.56)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
