#!/usr/bin/env python3
"""V21 OWLv2 keyframe bbox proposals.

This is the default V21 open-vocabulary bbox detector. GroundingDINO is
intentionally not used by this runner. The script requires either a local
OWLv2 model path or an explicit --allow-download opt-in.

Output:
  measurements/object_candidates/owlv2_bbox_proposals.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


class ContractError(RuntimeError):
    pass


COMMON_OWLV2_MODEL_PATHS = [
    Path(os.environ.get("V21_OWLV2_MODEL", "")) if os.environ.get("V21_OWLV2_MODEL") else None,
    Path(os.environ.get("OWLV2_MODEL", "")) if os.environ.get("OWLV2_MODEL") else None,
    Path("/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    Path("/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble"),
    Path("/mnt/user-home/zjh/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble"),
    Path("/home/zjh/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble"),
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_compute_target(raw: str | None, allow_local_heavy: bool) -> str:
    target = " ".join(str(raw or "").split())
    if not target:
        raise ContractError("missing_compute_target: pass --compute-target or set V21_COMPUTE_TARGET before running OWLv2 inference")
    lowered = target.lower()
    if any(token in lowered for token in ["local", "workstation", "laptop"]) and not allow_local_heavy:
        raise ContractError("local_heavy_inference_not_authorized: pass --allow-local-heavy only with explicit user approval")
    return target


def resolve_model_path(raw: str | None, allow_download: bool) -> str:
    if raw:
        path = Path(raw).expanduser()
        try:
            if path.exists():
                return str(path)
        except PermissionError:
            if not allow_download:
                raise
        if allow_download:
            # Treat non-existing explicit values such as google/owlv2-base-patch16-ensemble
            # as HuggingFace model IDs instead of continuing into inaccessible local caches.
            return str(raw)
        raise ContractError(f"owlv2_model_path_missing: {path}")
    for candidate in COMMON_OWLV2_MODEL_PATHS:
        if candidate is None or not str(candidate):
            continue
        try:
            if candidate.exists():
                return str(candidate)
        except PermissionError:
            continue
    if allow_download:
        return "google/owlv2-base-patch16-ensemble"
    searched = [str(p) for p in COMMON_OWLV2_MODEL_PATHS if p is not None and str(p)]
    raise ContractError(
        "missing_owlv2_model_path: provide --owlv2-model or V21_OWLV2_MODEL, "
        "or explicitly pass --allow-download on an authorized compute target. searched=" + json.dumps(searched)
    )


def object_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
    rows = root.get("objects") if isinstance(root, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def prompt_terms_from_plan(plan_path: Path | None) -> list[str]:
    if plan_path is None or not plan_path.exists():
        return []
    plan = load_json(plan_path)
    terms: list[str] = []
    for row in object_rows(plan):
        for key in ["description", "object_id", "track_id", "target_object_id"]:
            value = row.get(key)
            if value:
                terms.append(str(value).replace("object:", "").replace("_", " "))
        for key in ["open_vocab_prompts", "prompts", "text_prompts"]:
            values = row.get(key)
            if isinstance(values, list):
                terms.extend(str(v) for v in values if v)
    deduped: list[str] = []
    for term in terms:
        term = " ".join(term.lower().split())
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def normalize_prompts(raw_prompts: list[str], plan_path: Path | None) -> list[str]:
    terms = []
    terms.extend(raw_prompts)
    terms.extend(prompt_terms_from_plan(plan_path))
    deduped: list[str] = []
    for term in terms:
        term = " ".join(str(term).strip().split())
        if not term:
            continue
        if not term.lower().startswith("a photo of"):
            term = f"a photo of a {term}"
        if term not in deduped:
            deduped.append(term)
    if not deduped:
        raise ContractError("no_owlv2_text_prompts: pass --text-prompt or provide an object plan with descriptions/prompts")
    return deduped


def frames_from_keyframe_report(path: Path) -> list[int]:
    report = load_json(path)
    selected = report.get("selected_keyframes")
    if not isinstance(selected, list) or not selected:
        raise ContractError(f"keyframe_selection_report_has_no_selected_keyframes: {path}")
    frames: list[int] = []
    for row in selected:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            frames.append(int(row["frame_idx"]))
        elif isinstance(row, int):
            frames.append(int(row))
        else:
            raise ContractError(f"invalid_keyframe_selection_row: {row}")
    deduped: list[int] = []
    for frame_idx in frames:
        if frame_idx not in deduped:
            deduped.append(frame_idx)
    return deduped


def keyframe_indices(frames: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    if args.keyframe_selection_report:
        wanted = set(frames_from_keyframe_report(Path(args.keyframe_selection_report)))
        selected = [i for i, row in enumerate(frames) if int(row.get("frame_idx", i)) in wanted]
        missing = sorted(wanted - {int(frames[i].get("frame_idx", i)) for i in selected})
        if missing:
            raise ContractError(f"keyframe_selection_frames_missing_from_manifest: {missing}")
        return selected
    if args.frames:
        wanted = {int(v) for v in args.frames.split(",") if str(v).strip()}
        return [i for i, row in enumerate(frames) if int(row.get("frame_idx", i)) in wanted or i in wanted]
    if args.frame_stride is not None and args.frame_stride > 0:
        return list(range(0, len(frames), int(args.frame_stride)))
    count = max(1, min(int(args.num_keyframes), len(frames)))
    if count >= len(frames):
        return list(range(len(frames)))
    step = max(1, len(frames) // count)
    return list(range(0, len(frames), step))[:count]


def resolve_frame_path(run_root: Path, row: dict[str, Any]) -> Path:
    frame_idx = int(row.get("frame_idx", row.get("index", 0)))
    candidates = [
        run_root / f"input/source_frame_manifest/rgb/{frame_idx:06d}.jpg",
        run_root / f"input/raw_frame_manifest/rgb/{frame_idx:06d}.jpg",
    ]
    raw_rgb = row.get("rgb") or row.get("raw_frame_path")
    if raw_rgb:
        raw_path = Path(str(raw_rgb))
        candidates.append(raw_path if raw_path.is_absolute() else Path.cwd() / raw_path)
        candidates.append(run_root / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ContractError(f"frame_image_missing: frame={frame_idx} candidates={[str(c) for c in candidates]}")


def detect_frame(processor: Any, model: Any, image: Image.Image, prompts: list[str], threshold: float, torch_module: Any) -> list[dict[str, Any]]:
    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(model.device)
    with torch_module.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=float(threshold),
        target_sizes=[(image.height, image.width)],
        text_labels=[prompts],
    )[0]
    rows: list[dict[str, Any]] = []
    for box_tensor, score_tensor, text_label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = [float(v) for v in box_tensor.detach().cpu().tolist()]
        x1 = max(0.0, min(float(image.width), x1))
        x2 = max(0.0, min(float(image.width), x2))
        y1 = max(0.0, min(float(image.height), y1))
        y2 = max(0.0, min(float(image.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        rows.append(
            {
                "label": str(text_label),
                "text_label": str(text_label),
                "score": float(score_tensor.detach().cpu().item()),
                "owlv2_score": float(score_tensor.detach().cpu().item()),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "box_area_fraction": float(area / max(1.0, image.width * image.height)),
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    manifest_path = Path(args.raw_frame_manifest) if args.raw_frame_manifest else run_root / "input/raw_frame_manifest/manifest.json"
    manifest = load_json(manifest_path)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"raw_manifest_has_no_frames: {manifest_path}")
    plan_path = Path(args.object_plan) if args.object_plan else None
    prompts = normalize_prompts(args.text_prompt or [], plan_path)
    compute_target = validate_compute_target(args.compute_target, bool(args.allow_local_heavy))
    model_id = resolve_model_path(str(args.owlv2_model) if args.owlv2_model else None, bool(args.allow_download))

    import torch  # type: ignore[import-not-found]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ContractError("cuda_requested_but_unavailable")

    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore[import-not-found]

    local_files_only = not bool(args.allow_download)
    processor = Owlv2Processor.from_pretrained(model_id, local_files_only=local_files_only)
    model: Any = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=local_files_only).to(args.device)
    model.eval()

    selected = keyframe_indices(frames, args)
    out_frames: list[dict[str, Any]] = []
    for ordinal in selected:
        row = frames[ordinal]
        frame_idx = int(row.get("frame_idx", ordinal))
        image_path = resolve_frame_path(run_root, row)
        image = Image.open(str(image_path)).convert("RGB")
        detections = detect_frame(processor, model, image, prompts, float(args.threshold), torch)
        if args.max_boxes_per_frame > 0:
            detections = detections[: int(args.max_boxes_per_frame)]
        out_frames.append(
            {
                "frame_idx": frame_idx,
                "frame_ordinal": int(ordinal),
                "image_path": str(image_path),
                "image_width": int(image.width),
                "image_height": int(image.height),
                "detections": detections,
            }
        )

    report = {
        "schema": "v21_owlv2_bbox_proposals.v0",
        "status": "ok",
        "method": "owlv2_segmentation_stable_keyframe_bbox_proposals" if args.keyframe_selection_report else "owlv2_keyframe_bbox_proposals",
        "disabled_replacement_for": "groundingdino_default_bbox_path",
        "claim_scope": "Open-vocabulary bbox proposals for V21 target discovery and SAM2 prompts. Boxes are image evidence only; they are not masks, object geometry, pose, or contact evidence.",
        "run_root": str(run_root),
        "raw_frame_manifest": str(manifest_path),
        "object_plan": str(plan_path) if plan_path else None,
        "keyframe_selection_report": str(args.keyframe_selection_report) if args.keyframe_selection_report else None,
        "owlv2_model": model_id,
        "allow_download": bool(args.allow_download),
        "device": str(args.device),
        "compute_target": compute_target,
        "threshold": float(args.threshold),
        "text_prompts": prompts,
        "keyframe_count": len(out_frames),
        "frames": out_frames,
        "total_detections": int(sum(len(row["detections"]) for row in out_frames)),
    }
    output_path = Path(args.output) if args.output else run_root / "measurements/object_candidates/owlv2_bbox_proposals.json"
    write_json(output_path, report)
    print(json.dumps({"status": "ok", "output": str(output_path), "total_detections": report["total_detections"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--raw-frame-manifest")
    parser.add_argument("--object-plan")
    parser.add_argument("--output")
    parser.add_argument("--owlv2-model")
    parser.add_argument("--allow-download", action="store_true", help="Allow HuggingFace download. Use only on an authorized compute target.")
    parser.add_argument("--compute-target", default=os.environ.get("V21_COMPUTE_TARGET"), help="Required explicit compute target label, e.g. A800/server job id.")
    parser.add_argument("--allow-local-heavy", action="store_true", help="Only use with explicit user approval for local heavy inference.")
    parser.add_argument("--text-prompt", action="append", default=[], help="Open-vocabulary target phrase. May be passed multiple times.")
    parser.add_argument("--num-keyframes", type=int, default=25)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--frames", help="Comma-separated frame_idx or ordinal list.")
    parser.add_argument("--keyframe-selection-report", type=Path, help="Segmentation-stable keyframe report with selected_keyframes[].frame_idx.")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-boxes-per-frame", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
