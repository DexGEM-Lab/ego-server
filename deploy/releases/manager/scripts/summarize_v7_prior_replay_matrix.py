#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def median(report: dict, key: str) -> float | None:
    value = report.get(key)
    if not isinstance(value, dict) or "median" not in value:
        return None
    return float(value["median"])


def parse_entry(raw: str) -> tuple[str, Path, Path, str]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--entry must have format name|baseline_zbuffer_json|prior_replay_json|note")
    name, baseline, prior, note = parts
    if not name.strip():
        raise RuntimeError("entry name is empty")
    return name.strip(), Path(baseline), Path(prior), note.strip()


def format_optional_float(value: object, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def failed_or_not_evaluated(pass_rows: dict, delivery_keys: list[str]) -> str:
    failed = [key for key in delivery_keys if pass_rows.get(key) is False]
    not_evaluated = [key for key in delivery_keys if pass_rows.get(key) is None]
    labels = [*failed, *[f"{key}: not evaluated" for key in not_evaluated]]
    return ", ".join(labels)


def summarize_entry(name: str, baseline_path: Path, prior_path: Path, note: str) -> dict:
    baseline = load_json(baseline_path)
    prior = load_json(prior_path)
    metrics = prior.get("metrics")
    pass_rows = prior.get("pass")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"prior report lacks metrics object: {prior_path}")
    if not isinstance(pass_rows, dict):
        raise RuntimeError(f"prior report lacks pass object: {prior_path}")
    return {
        "name": name,
        "note": note,
        "baseline": {
            "path": str(baseline_path),
            "frames": int(baseline.get("frames", 0)),
            "silhouette_iou_median": median(baseline, "silhouette_mask_iou"),
            "visible_inside_median": median(baseline, "visible_silhouette_inside_mask_fraction"),
            "zbuffer_abs_p95_median_m": median(baseline, "zbuffer_depth_abs_p95_m"),
            "video": baseline.get("video"),
        },
        "prior": {
            "path": str(prior_path),
            "status": prior.get("status"),
            "annotation_ready": bool(prior.get("annotation_ready", False)),
            "source_kind": "video_mesh" if prior.get("method") == "run_v7_video_mesh_replay_qc" else "generated_prior",
            "mesh_prior": prior.get("mesh_prior"),
            "video_mesh_archive": prior.get("video_mesh_archive"),
            "frame_start": prior.get("frame_start"),
            "frame_end": prior.get("frame_end"),
            "metrics": metrics,
            "thresholds": prior.get("thresholds"),
            "pass": pass_rows,
            "invalid_observed_target_keys": prior.get("invalid_observed_target_keys") or [],
            "delivery_pass_keys": prior.get("delivery_pass_keys"),
            "zbuffer_video": prior.get("zbuffer_video"),
        },
    }


def write_markdown(path: Path, report: dict) -> None:
    rows = [
        "| Sample | Source | Baseline IoU | Baseline depth p95 m | Candidate status | Observed-control failure | Visible p95 m | Candidate IoU | Candidate depth p95 m | Failed or unevaluated delivery keys | Note |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in report["entries"]:
        baseline = entry["baseline"]
        prior = entry["prior"]
        metrics = prior["metrics"]
        pass_rows = prior["pass"]
        delivery_keys = prior.get("delivery_pass_keys") or []
        observed_failure = ", ".join(prior.get("invalid_observed_target_keys") or [])
        rows.append(
            "| {name} | {source} | {biou:.3f} | {bdepth:.4f} | {status} | {observed_failure} | {visible} | {piou} | {pdepth} | {failed} | {note} |".format(
                name=entry["name"],
                source=prior.get("source_kind") or "unknown",
                biou=float(baseline["silhouette_iou_median"]),
                bdepth=float(baseline["zbuffer_abs_p95_median_m"]),
                status=prior["status"],
                observed_failure=observed_failure or "none",
                visible=format_optional_float(metrics.get("visible_surface_coverage_p95_m"), 4),
                piou=format_optional_float(metrics.get("silhouette_iou_median"), 3),
                pdepth=format_optional_float(metrics.get("zbuffer_abs_p95_median_m"), 4),
                failed=failed_or_not_evaluated(pass_rows, delivery_keys),
                note=entry["note"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    entries = [summarize_entry(*parse_entry(raw)) for raw in args.entry]
    if not entries:
        raise RuntimeError("at least one --entry is required")
    report = {
        "status": "ok",
        "method": "summarize_v7_prior_replay_matrix",
        "claim_tested": "object mesh candidates must pass their source-appropriate replay contract across representative manipulation clips",
        "entries": entries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md is not None:
        write_markdown(args.output_md, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
