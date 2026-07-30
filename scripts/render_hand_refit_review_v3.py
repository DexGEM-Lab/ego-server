#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_hand_streams_scale055_v3 import load_frame_window
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask


HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video}")
    return frame


def draw_mask(frame: np.ndarray, ann: dict, args: argparse.Namespace) -> None:
    obj = ann.get("object", {})
    if not obj.get("mask_path"):
        return
    mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
    mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    color = np.zeros_like(frame)
    color[:, :] = (80, 80, 230)
    frame[:] = np.where(mask[..., None], (0.58 * frame + 0.42 * color).astype(np.uint8), frame)


def draw_hand(frame: np.ndarray, hand: dict) -> None:
    measured = bool(hand.get("measurement_available", False))
    side = str(hand.get("side", "unknown"))
    color = (70, 220, 80) if side == "right" else (235, 160, 60)
    if not measured:
        color = (125, 125, 125)
    raw = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if raw.shape == (21, 2):
        for p in raw:
            cv2.circle(frame, tuple(np.rint(p).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.rint(p).astype(int)), 5, (35, 35, 35), 1, cv2.LINE_AA)
    if joints.shape == (21, 3) and intr.shape == (4,):
        proj = project_points(joints, intr)
        for a, b in HAND_EDGES:
            pa = tuple(np.rint(proj[a]).astype(int))
            pb = tuple(np.rint(proj[b]).astype(int))
            cv2.line(frame, pa, pb, color, 2 if measured else 1, cv2.LINE_AA)
        for p in proj:
            cv2.circle(frame, tuple(np.rint(p).astype(int)), 3, color, -1, cv2.LINE_AA)
    text = f"{side} score={float(hand.get('detector_score', 0.0)):.2f}"
    refit = hand.get("v3_target_similarity_refit")
    if isinstance(refit, dict):
        text += f" r={float(refit.get('median_reprojection_after_px', 0.0)):.1f}px gap={float(refit.get('mano_minus_unidepth_after_m', 0.0))*1000:.0f}mm"
    anchor = raw[0] if raw.shape == (21, 2) else np.asarray([24.0, 40.0])
    cv2.putText(frame, text, tuple(np.rint(anchor + np.asarray([8.0, -16.0])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict:
    frames = load_frame_window(args.annotations, args.frame_start, args.frame_end)
    reliable_frames: set[int] = set()
    if args.contact_qc is not None:
        qc = load_json(args.contact_qc)
        reliable_frames = {
            int(row["frame_idx"])
            for row in qc["streams"][args.stream_name]["rows_preview"]
            if bool(row.get("reliable_for_contact", False))
        }
    review_frames = sorted(set(args.extra_frame or []) | reliable_frames)
    if not review_frames:
        review_frames = list(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for frame_idx in review_frames:
        ann = frames.get(frame_idx)
        if ann is None:
            continue
        image = read_frame(args.video, frame_idx)
        draw_mask(image, ann, args)
        for hand in ann.get("hands", []):
            draw_hand(image, hand)
        out = args.output_dir / f"frame_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(out), image):
            raise RuntimeError(f"failed to write {out}")
        written.append(str(out))
    report = {
        "status": "ok",
        "annotations": str(args.annotations),
        "contact_qc": str(args.contact_qc) if args.contact_qc is not None else None,
        "stream_name": args.stream_name,
        "review_frames": review_frames,
        "written": written,
        "interpretation": "White dots are measured 2D keypoints when present; colored skeleton is the MANO projection; red overlay is the object mask.",
    }
    report_path = args.output_dir / "review_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path)
    parser.add_argument("--stream-name", default="hand_stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--extra-frame", action="append", type=int)
    parser.add_argument("--remote-output-root", default="/mnt/user-home/yiwen/ego_annotation_remote/outputs")
    parser.add_argument("--local-output-root", default="/data2/ego_annotation_outputs")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
