#!/usr/bin/env python3
"""Write structural QC records for existing V21 atomic algorithm overlays.

This script does not perform visual acceptance. It records file existence,
video metadata, deprecated status, and explicit tuning debt only when an audit
row already marks tuning as required by large-deviation evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ffprobe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
    except Exception as exc:
        return {"exists": True, "ffprobe_error": str(exc)}
    if proc.returncode != 0:
        return {"exists": True, "ffprobe_error": proc.stderr.strip()}
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    return {
        "exists": True,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "nb_frames": int(stream.get("nb_frames") or 0) if str(stream.get("nb_frames") or "").isdigit() else None,
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration": float(stream.get("duration")) if stream.get("duration") not in (None, "N/A") else None,
    }


def write_tuning_stub(row: dict[str, Any]) -> str | None:
    if row.get("deprecated") or row.get("optional") or not row.get("tuning_required"):
        return None
    tuning_dir = Path(str(row["tuning_dir"]))
    attempt = tuning_dir / "attempt_000.json"
    if attempt.exists():
        return str(attempt)
    payload = {
        "schema": "v21_atomic_tuning_record.v0",
        "status": "blocked_large_deviation_tuning_not_recorded",
        "algorithm_id": row["algorithm_id"],
        "family": row["family"],
        "case": row["case"],
        "data_path": row.get("data_path"),
        "overlay_path": row.get("overlay_path"),
        "physical_variable_blocked": row["family"],
        "observation": "The audit row marks this atom as requiring tuning, but no sample-bound parameter-changing attempt was recorded.",
        "interpretation": "This is a tuning debt record triggered by explicit large-deviation evidence, not by the mere existence of an output.",
        "required_next_intervention": "Run a causally targeted internal parameter/model/prompt/calibration change for this same atomic algorithm, then replace this stub with measured before/after residuals and visual review.",
        "invalid_data": bool(row.get("invalid_data")),
        "invalid_data_method": row.get("invalid_data_method"),
        "claim_scope": "Explicitly records missing tuning evidence. It does not mark the algorithm as tuned, run, or accepted.",
    }
    write_json(attempt, payload)
    return str(attempt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit = load_json(args.audit)
    written_qc: list[str] = []
    written_tuning: list[str] = []
    for row in audit.get("rows", []):
        if not isinstance(row, dict):
            continue
        overlay_candidates = [row.get("overlay_path"), row.get("native_overlay_path")]
        overlay_path = None
        for raw in overlay_candidates:
            if not raw:
                continue
            candidate = Path(str(raw))
            if candidate.exists():
                overlay_path = candidate
                break
        if overlay_path is not None and overlay_path.exists() and not row.get("deprecated"):
            qc_path = Path(str(row.get("qc_path") or (overlay_path.parent / "qc.json")))
            if not qc_path.exists():
                qc = {
                    "schema": "v21_atomic_overlay_qc.v0",
                    "status": "structural_only_not_visual_acceptance",
                    "case": row.get("case"),
                    "algorithm_id": row.get("algorithm_id"),
                    "family": row.get("family"),
                    "runner_agent": row.get("runner_agent", audit.get("runner_agent", "runner_agent")),
                    "source": row.get("source"),
                    "data_path": row.get("data_path"),
                    "overlay_path": str(overlay_path),
                    "video_metadata": ffprobe(overlay_path),
                    "visual_quality_reviewed": False,
                    "accepted_for_downstream_physical_claim": False,
                    "claim_scope": "Structural overlay QC only. A human/Pi visual review must inspect target correctness, alignment, and physical semantics before acceptance.",
                }
                write_json(qc_path, qc)
                written_qc.append(str(qc_path))
        tuning = write_tuning_stub(row)
        if tuning:
            written_tuning.append(tuning)
    summary = {
        "schema": "v21_atomic_overlay_qc_materialization.v0",
        "status": "ok",
        "audit_input": str(args.audit),
        "runner_agent": audit.get("runner_agent", "runner_agent"),
        "qc_written_count": len(written_qc),
        "tuning_stub_written_count": len(written_tuning),
        "qc_written": written_qc,
        "tuning_stubs_written": written_tuning,
        "claim_scope": "Materializes structural QC and explicit large-deviation tuning debt records. It does not run models, generate missing overlays, or accept visual quality.",
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("outputs/v21_per_algorithm_results/atomic_algorithm_overlay_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v21_per_algorithm_results/atomic_overlay_qc_materialization.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
