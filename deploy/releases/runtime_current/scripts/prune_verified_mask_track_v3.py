#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_stats(mask_path: Path) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {mask_path}")
    binary = mask > 0
    area = int(binary.sum())
    if area == 0:
        raise RuntimeError(f"empty mask {mask_path}")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([], dtype=np.int32)
    order = np.argsort(component_areas)[::-1]
    main_area = int(component_areas[order[0]]) if order.size else 0
    second_area = int(component_areas[order[1]]) if order.size > 1 else 0
    ys, xs = np.where(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bbox_area = int((x1 - x0) * (y1 - y0))
    return {
        "area_px": area,
        "component_count": int(count - 1),
        "main_component_area_px": main_area,
        "second_component_area_px": second_area,
        "second_over_main": float(second_area / max(1, main_area)),
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "bbox_fill": float(area / max(1, bbox_area)),
    }


def verdict_by_frame(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    payload = load_json(path)
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list):
        raise RuntimeError(f"verdict file has no verdicts list: {path}")
    return {int(row["frame_idx"]): row for row in verdicts}


def keep_frame(entry: dict, stats: dict, verdict: dict | None, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    if not entry.get("visible"):
        reasons.append("track_marked_not_visible")
    if verdict is not None:
        if not bool(verdict.get("target_visible")):
            reasons.append("vlm_target_not_visible")
        if not bool(verdict.get("mask_correct_for_track")):
            reasons.append("vlm_rejected_mask")
        if float(verdict.get("confidence", 0.0)) < float(args.min_verifier_confidence):
            reasons.append("low_verifier_confidence")
    if int(stats["component_count"]) > int(args.max_components):
        reasons.append("too_many_components")
    if float(stats["second_over_main"]) > float(args.max_second_over_main):
        reasons.append("large_secondary_component")
    if float(stats["bbox_fill"]) < float(args.min_bbox_fill):
        reasons.append("low_bbox_fill")
    return len(reasons) == 0, reasons


def run(args: argparse.Namespace) -> dict:
    track = load_json(args.mask_track)
    verdicts = verdict_by_frame(args.verdicts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "selected_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    pruned = {}
    rows = []
    kept = 0
    for key, entry in sorted(track.items(), key=lambda item: int(item[0])):
        frame_idx = int(key)
        row = {"frame_idx": frame_idx}
        if not entry.get("visible"):
            pruned[key] = {"visible": False, "prune_reasons": ["track_marked_not_visible"], "source_entry": entry}
            row.update({"kept": False, "prune_reasons": ["track_marked_not_visible"]})
            rows.append(row)
            continue
        source_mask = Path(str(entry["mask_path"]))
        stats = mask_stats(source_mask)
        verdict = verdicts.get(frame_idx)
        keep, reasons = keep_frame(entry, stats, verdict, args)
        row.update({"kept": keep, "prune_reasons": reasons, "stats": stats, "verdict": verdict})
        if keep:
            target_mask = mask_dir / f"{frame_idx:06d}.png"
            shutil.copy2(source_mask, target_mask)
            out_entry = dict(entry)
            out_entry["mask_path"] = str(target_mask)
            out_entry["mask_quality"] = stats
            if verdict is not None:
                out_entry["vlm_verdict"] = verdict
            pruned[key] = out_entry
            kept += 1
        else:
            pruned[key] = {
                "visible": False,
                "prune_reasons": reasons,
                "source_entry": entry,
                "mask_quality": stats,
                "vlm_verdict": verdict,
            }
        rows.append(row)
    track_path = args.output_dir / "sam2_vlm_selected_track_pruned.json"
    track_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "method": "prune_verified_mask_track_v3",
        "source_mask_track": str(args.mask_track),
        "verdicts": str(args.verdicts) if args.verdicts else None,
        "output_track": str(track_path),
        "frames": int(len(track)),
        "kept_frames": int(kept),
        "rejected_frames": int(len(track) - kept),
        "thresholds": {
            "max_components": int(args.max_components),
            "max_second_over_main": float(args.max_second_over_main),
            "min_bbox_fill": float(args.min_bbox_fill),
            "min_verifier_confidence": float(args.min_verifier_confidence),
        },
        "rows": rows,
    }
    (args.output_dir / "qc_pruned_mask_track_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "rows"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--max-second-over-main", type=float, default=0.04)
    parser.add_argument("--min-bbox-fill", type=float, default=0.28)
    parser.add_argument("--min-verifier-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
