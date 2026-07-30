#!/usr/bin/env python3
"""Publish a V19 render branch as clear full-duration user-facing videos.

The script does not change physical state.  It consumes existing state-driven
render videos, adds a stable explanatory legend/metric banner, writes a
publication report, optional review stills, and can atomically update canonical
``v19_overlay.mp4``, ``v19_world.mp4``, and ``v19_side_by_side.mp4`` paths,
using symlinks when the filesystem preserves them and real copies otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_NAMES = {
    "overlay": "v19_overlay.mp4",
    "world": "v19_world.mp4",
    "side_by_side": "v19_side_by_side.mp4",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metric_value(summary: dict[str, Any], key: str, stat: str = "median") -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get(stat)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def side_summary(interval_state: dict[str, Any], side: str) -> dict[str, Any]:
    for row in interval_state.get("intervals", []) if isinstance(interval_state.get("intervals"), list) else []:
        if isinstance(row, dict) and str(row.get("interval_id", "")).startswith(f"{side}_"):
            return row
    return {}


def contact_semantics(summary: dict[str, Any]) -> str:
    """Human-facing contact wording from the metric source-gap model.

    This is not an acceptance gate and not a calibrated probability.  It exposes
    the physical meaning already represented by
    build_v19_source_gap_contact_likelihood_state.py: source/object surfaces are
    compared against the combined metric uncertainty
    sqrt(hand_sigma^2 + object_sigma^2 + depth_order_sigma^2).  The wording
    must never claim contact ownership or signed nonpenetration.
    """
    raw_counts = summary.get("contact_likelihood_state_counts")
    counts = raw_counts if isinstance(raw_counts, dict) else {}
    total = sum(int(v) for v in counts.values() if isinstance(v, (int, float)))
    compatible = int(counts.get("near_contact_compatible_by_source_gap_only") or 0)
    within2 = int(counts.get("near_contact_uncertain_within_2sigma") or 0)
    unlikely = int(counts.get("contact_unlikely_source_gap_exceeds_3sigma") or 0)
    z = metric_value(summary, "source_gap_z_median")
    compat_score = metric_value(summary, "contact_compatibility_score_median")
    if total > 0 and compatible >= 0.5 * total and z is not None and z <= 1.0:
        return "near-contact compatible; ownership/NP unresolved"
    if total > 0 and (compatible + within2) >= 0.5 * total and z is not None and z <= 2.0:
        return "near-contact uncertain; ownership/NP unresolved"
    if total > 0 and unlikely >= 0.5 * total and z is not None and z >= 3.0:
        return "contact unlikely by source gap"
    if total == 0 and compat_score is not None and compat_score >= 0.5:
        return "near-contact compatible; ownership/NP unresolved"
    return "contact uncertain"


def summarize_interval(interval_state: Path | None) -> dict[str, Any]:
    if interval_state is None:
        return {"summary_text": "interval metrics unavailable", "sides": {}}
    payload = load_json(interval_state)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if isinstance(payload.get("per_frame_states"), list) and summary:
        normal = metric_value(summary, "contact_normal_abs_after_median")
        tangent = metric_value(summary, "contact_tangent_after_median")
        distance = metric_value(summary, "contact_distance_after_median")
        if distance is None:
            distance = metric_value(summary, "contact_after_median")
        source_gap = metric_value(summary, "source_hand_to_object_surface_distance_median")
        if source_gap is None:
            source_gap = metric_value(summary, "contact_distance_before_median")
        if source_gap is None:
            source_gap = metric_value(summary, "contact_before_median")
        shift = metric_value(summary, "metric_joint_shift_px")
        if shift is None:
            shift = metric_value(summary, "visible_joint_shift_px_median")
        rows = summary.get("rows_out") or summary.get("rows") or summary.get("optimized_rows") or summary.get("row_count")
        state_kind = str(payload.get("method") or "")
        contact_prob = metric_value(summary, "contact_compatibility_score_median")
        if contact_prob is None:
            contact_prob = metric_value(summary, "contact_likelihood_median")
        source_gap_z = metric_value(summary, "source_gap_z_median")
        parts = []
        split_metric_surface = summary.get("split_state_policy") == "metric_mano_preserved_contact_surface_posterior"
        if rows is not None:
            parts.append(f"rows {rows}")
        if state_kind.startswith("v19_direct_object_surface_contact_posterior_state") or state_kind.startswith("v19_visible_object_surface_contact_posterior_state"):
            if source_gap is not None:
                parts.append(f"source gap {source_gap * 1000.0:.1f}mm")
            elif distance is not None:
                parts.append(f"source gap {distance * 1000.0:.1f}mm")
            if normal is not None:
                parts.append(f"normal {normal * 1000.0:.1f}mm")
            if contact_prob is not None:
                parts.append(f"contact compat~{contact_prob:.3f}")
            if source_gap_z is not None:
                parts.append(f"gap z {source_gap_z:.1f}")
        else:
            if split_metric_surface:
                if source_gap is not None:
                    parts.append(f"source gap {source_gap * 1000.0:.1f}mm")
                if normal is not None:
                    parts.append(f"posterior normal {normal * 1000.0:.1f}mm")
                if tangent is not None:
                    parts.append(f"tangent {tangent * 1000.0:.1f}mm")
                parts.append("metric MANO preserved")
            else:
                if normal is not None:
                    parts.append(f"normal {normal * 1000.0:.1f}mm")
                if tangent is not None:
                    parts.append(f"tangent {tangent * 1000.0:.1f}mm")
                elif distance is not None:
                    parts.append(f"surface {distance * 1000.0:.1f}mm")
        if shift is not None:
            parts.append(f"candidate shift {shift:.1f}px" if split_metric_surface else f"joint shift {shift:.1f}px")
        if "metric_mano_preserved" in json.dumps(payload.get("per_frame_states", [])[:1]) and "metric MANO preserved" not in parts:
            parts.append("metric MANO preserved")
        contact_text = contact_semantics(summary)
        parts.append(contact_text)
        return {
            "summary_text": " | ".join(parts),
            "interval_state": str(interval_state),
            "state_kind": payload.get("method"),
            "per_frame_rows": len(payload.get("per_frame_states", [])),
            "contact_semantics": contact_text,
            "contact_likelihood_state_counts": summary.get("contact_likelihood_state_counts"),
        }
    sides: dict[str, Any] = {}
    tokens: list[str] = []
    for side in ("left", "right"):
        row = side_summary(payload, side)
        gap = metric_value(row, "contact_patch_final_abs_normal_gap_m")
        shift = metric_value(row, "visible_joint_shift_max_px")
        trans = metric_value(row, "translation_delta_norm_m")
        closed = row.get("active_set_closed")
        sides[side] = {
            "contact_patch_final_abs_normal_gap_m_median": gap,
            "visible_joint_shift_max_px_median": shift,
            "translation_delta_norm_m_median": trans,
            "active_set_closed": closed,
        }
        gap_txt = "gap=?" if gap is None else f"gap {gap * 1000.0:.1f}mm"
        shift_txt = "shift=?" if shift is None else f"shift {shift:.1f}px"
        closed_txt = f"closed {closed}" if closed is not None else "closed ?"
        tokens.append(f"{side[0].upper()}: {gap_txt}, {shift_txt}, {closed_txt}")
    return {"summary_text": " | ".join(tokens), "sides": sides, "interval_state": str(interval_state)}


def put_text_fit(image: np.ndarray, text: str, org: tuple[int, int], max_width: int, scale: float, color: tuple[int, int, int], thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    use_scale = scale
    while use_scale > 0.35:
        (tw, _), _ = cv2.getTextSize(text, font, use_scale, thickness)
        if tw <= max_width:
            break
        use_scale *= 0.92
    cv2.putText(image, text, org, font, use_scale, color, thickness, cv2.LINE_AA)


def banner(width: int, title: str, subtitle: str, metrics: str, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:] = (12, 12, 12)
    cv2.rectangle(out, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    put_text_fit(out, title, (14, 27), width - 28, 0.68, (255, 255, 255), 2)
    put_text_fit(out, subtitle, (14, 51), width - 28, 0.48, (215, 230, 255), 1)
    put_text_fit(out, metrics, (14, height - 13), width - 28, 0.45, (160, 240, 255), 1)
    return out


def annotate_frame(frame: np.ndarray, title: str, subtitle: str, metrics: str, banner_h: int) -> np.ndarray:
    top = banner(frame.shape[1], title, subtitle, metrics, banner_h)
    return np.vstack([top, frame])


def write_video_with_banner(
    src: Path,
    dst: Path,
    title: str,
    subtitle: str,
    metrics: str,
    still_frames: set[int],
    still_dir: Path,
    banner_h: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source video {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + banner_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open writer {dst}")
    written = 0
    stills: list[str] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = annotate_frame(frame, title, subtitle, metrics, banner_h)
            writer.write(annotated)
            if written in still_frames:
                still_dir.mkdir(parents=True, exist_ok=True)
                out = still_dir / f"{dst.stem}_frame_{written:06d}.jpg"
                if not cv2.imwrite(str(out), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {out}")
                stills.append(str(out))
            written += 1
    finally:
        writer.release()
        cap.release()
    return {
        "source": str(src),
        "output": str(dst),
        "source_frames_reported": frame_count,
        "frames_written": int(written),
        "fps": fps,
        "source_size": [width, height],
        "output_size": [width, height + banner_h],
        "stills": stills,
    }


def atomic_symlink(target: Path, link: Path) -> str:
    """Publish a canonical render path atomically.

    Runtime delivery filesystems used for A800/truenas runs have shown degraded
    POSIX symlink behavior: a Python-visible symlink can later appear to normal
    file consumers as a zero-byte regular file.  Canonical render paths are the
    user-facing artifact, so publish them as real copies instead of symlinks.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError(f"canonical render source {target} is missing or empty")
    tmp = link.with_name(f".{link.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    shutil.copy2(target, tmp)
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"canonical copy for {link} is empty after copying {target}")
    os.replace(tmp, link)
    if link.is_symlink() or not link.exists() or link.stat().st_size <= 0:
        raise RuntimeError(f"canonical render {link} is not a non-empty regular file after publication")
    return "copy"


def publish(args: argparse.Namespace) -> dict[str, Any]:
    metrics = summarize_interval(args.interval_state)
    subtitle = args.subtitle or "green=rigid object | orange halo=current/source MANO | cyan/yellow=optimized interval hypothesis | UNCERTAIN=not accepted contact closure"
    title = args.title
    out_dir = args.output_dir
    still_dir = out_dir / "review_stills"
    stills = {int(x) for x in args.still_frames}
    videos = {
        "overlay": args.overlay,
        "world": args.world,
        "side_by_side": args.side_by_side,
    }
    outputs: dict[str, Any] = {}
    for kind, src in videos.items():
        dst = out_dir / VIDEO_NAMES[kind]
        outputs[kind] = write_video_with_banner(
            src=src,
            dst=dst,
            title=title if kind != "world" else f"{title} - world view",
            subtitle=subtitle,
            metrics=metrics["summary_text"],
            still_frames=stills,
            still_dir=still_dir,
            banner_h=int(args.banner_height),
        )
    canonical_updates: dict[str, Any] = {}
    if args.canonical_dir is not None:
        for kind, name in VIDEO_NAMES.items():
            target = (out_dir / name).resolve()
            link = args.canonical_dir / name
            mode = "not_replaced"
            if args.replace_canonical:
                mode = atomic_symlink(target, link)
            canonical_updates[str(link)] = {"target": str(target), "mode": mode}
    report = {
        "status": "ok",
        "method": "publish_v19_render_artifact",
        "claim_scope": "presentation/publishing of existing state-driven render; physical marks come from source render videos",
        "title": title,
        "subtitle": subtitle,
        "metrics": metrics,
        "outputs": outputs,
        "canonical_updates": canonical_updates,
        "replace_canonical": bool(args.replace_canonical),
    }
    write_json(out_dir / "v19_published_render_report.json", report)
    print(json.dumps(report, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--side-by-side", type=Path, required=True)
    parser.add_argument("--interval-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--replace-canonical", action="store_true")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle")
    parser.add_argument("--banner-height", type=int, default=78)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[690, 700, 720, 725])
    return parser.parse_args()


def main() -> None:
    publish(parse_args())


if __name__ == "__main__":
    main()
