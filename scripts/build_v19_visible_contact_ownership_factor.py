#!/usr/bin/env python3
"""Build V19 visible ownership and latent contact_patch factors from image-space MANO/object evidence.

The factor is deliberately object/category agnostic.  It does not infer contact
from labels.  It uses the current MANO projection as hand image support, the
model-produced visible object mask as object support, and emits:

* visible_ownership rows whose non_object_owned mask marks target-object pixels
  plausibly owned by the visible hand surface.  The interval solver uses this to
  quarantine hard object-surface constraints behind the hand.
* contact_patch rows where projected hand support overlaps or lies close to the
  visible object mask.  These rows are latent/uncertain contact hypotheses, not
  accepted persistent contact anchors.

This is a measurement bridge for the V19 workbench: it creates the contact source
that the older V18 contact_patch builder only promoted from existing annotations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

CONTACT_STATES = {
    "likely_contact",
    "possible_contact",
    "no_contact",
    "unresolved",
}

OCCLUSION_RELATIONS = {
    "hand_in_front_of_object",
    "object_in_front_of_hand",
    "object_partially_occluded_by_hand",
    "no_visible_occlusion",
    "unresolved",
}

DEPTH_RELIABILITY_STATES = {
    "hand_depth_unreliable",
    "object_depth_reliable",
    "mixed_or_unresolved",
    "not_evaluated",
}

NO_QUARANTINE_OCCLUSIONS = {"object_in_front_of_hand", "no_visible_occlusion"}
HAND_FRONT_OCCLUSIONS = {"hand_in_front_of_object", "object_partially_occluded_by_hand"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--target-entity-id", required=True)
    p.add_argument("--frame-span", type=int, nargs=2, required=True, metavar=("START", "END"))
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--hand-radius-px", type=int, default=10)
    p.add_argument("--hand-line-radius-px", type=int, default=6)
    p.add_argument("--ownership-dilation-px", type=int, default=8)
    p.add_argument("--contact-image-band-px", type=int, default=18)
    p.add_argument("--min-contact-image-px", type=int, default=12)
    p.add_argument("--contact-weight", type=float, default=2.5e4)
    p.add_argument("--raw-proposal-weight-factor", type=float, default=0.65)
    p.add_argument("--contact-patch-band-m", type=float, default=0.18)
    p.add_argument("--contact-patch-target-margin-m", type=float, default=0.004)
    p.add_argument("--contact-support-uncertainty-m", type=float, default=0.060)
    p.add_argument("--agent-interaction-judgment", type=Path, default=None, help="Optional Pi-agent-authored interval contact/occlusion judgment JSON. When supplied, it calibrates contact priors, support uncertainty, and ownership quarantine policy instead of relying on projection adjacency alone.")
    p.add_argument("--agent-judgment-can-create-contact", action=argparse.BooleanOptionalAction, default=True, help="Allow a likely/possible agent contact judgment to emit a contact_patch prior even when the image-proximity pixel threshold is not met. The row still uses projected MANO/object geometry for targets; this only creates the prior/switch.")
    p.add_argument("--max-vertices", type=int, default=160)
    p.add_argument("--review-frames", type=int, nargs="*", default=[691, 700, 720, 725])
    return p.parse_args()


def target_object(frame: dict[str, Any], target_entity_id: str) -> dict[str, Any] | None:
    target_id = str(target_entity_id)
    bare = target_id.split(":", 1)[-1]
    for obj in as_list(frame.get("objects")):
        if not isinstance(obj, dict):
            continue
        ids = {str(obj.get("object_id")), str(obj.get("track_id")), f"object:{obj.get('track_id')}", f"object:{obj.get('object_id')}"}
        if target_id in ids or bare in ids:
            return obj
    return None


def target_matches(raw: Any, target_entity_id: str) -> bool:
    if raw in (None, ""):
        return True
    target = str(target_entity_id)
    bare = target.split(":", 1)[-1]
    val = str(raw)
    val_bare = val.split(":", 1)[-1]
    return val in {target, bare, f"object:{bare}"} or val_bare == bare


def finite_float(value: Any, *, field: str, path: Path) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"agent interaction judgment field {field!r} is not numeric in {path}: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"agent interaction judgment field {field!r} is not finite in {path}: {value!r}")
    return out


def load_agent_interaction_judgments(path: Path | None, *, case: str, target_entity_id: str, frame_span: tuple[int, int], sides: set[str]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {"enabled": False, "path": None, "segments": 0, "frame_side_count": 0}
    if not path.exists():
        raise FileNotFoundError(f"missing agent interaction judgment JSON: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"agent interaction judgment must be a JSON object: {path}")
    if payload.get("status") not in (None, "ok"):
        raise ValueError(f"agent interaction judgment status is not ok in {path}: {payload.get('status')!r}")
    if payload.get("case") not in (None, case):
        raise ValueError(f"agent interaction judgment case {payload.get('case')!r} does not match {case!r} in {path}")
    rows = payload.get("interaction_judgments", payload.get("judgments", payload.get("segments")))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"agent interaction judgment has no non-empty interaction_judgments/judgments/segments list: {path}")
    start_default, end_default = frame_span
    index: dict[tuple[int, str], dict[str, Any]] = {}
    kept_segments = 0
    state_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f"agent interaction judgment segment is not an object in {path}: {raw!r}")
        if not target_matches(raw.get("target_entity_id", raw.get("object_id", raw.get("track_id"))), target_entity_id):
            continue
        hand_side = str(raw.get("hand_side", raw.get("side", "")))
        if hand_side not in sides:
            continue
        s = int(raw.get("frame_start", raw.get("start_frame", start_default)))
        e = int(raw.get("frame_end", raw.get("end_frame", end_default)))
        if e < s:
            raise ValueError(f"agent interaction judgment has inverted interval {s}>{e} in {path}: {raw}")
        s = max(s, start_default)
        e = min(e, end_default)
        if e < s:
            continue
        contact_state = str(raw.get("contact_state", "")).strip()
        if contact_state not in CONTACT_STATES:
            raise ValueError(f"agent interaction judgment contact_state must be one of {sorted(CONTACT_STATES)} in {path}: {raw}")
        occlusion_relation = str(raw.get("occlusion_relation", raw.get("visibility_relation", ""))).strip()
        if occlusion_relation not in OCCLUSION_RELATIONS:
            raise ValueError(f"agent interaction judgment occlusion_relation must be one of {sorted(OCCLUSION_RELATIONS)} in {path}: {raw}")
        depth_reliability = str(raw.get("depth_reliability", "mixed_or_unresolved")).strip()
        if depth_reliability not in DEPTH_RELIABILITY_STATES:
            raise ValueError(f"agent interaction judgment depth_reliability must be one of {sorted(DEPTH_RELIABILITY_STATES)} in {path}: {raw}")
        prior = finite_float(raw.get("contact_prior_probability"), field="contact_prior_probability", path=path)
        if not (0.0 <= prior <= 1.0):
            raise ValueError(f"agent interaction judgment contact_prior_probability must be in [0,1] in {path}: {raw}")
        support_unc = finite_float(raw.get("contact_support_uncertainty_m"), field="contact_support_uncertainty_m", path=path)
        if support_unc < 0.0:
            raise ValueError(f"agent interaction judgment contact_support_uncertainty_m must be nonnegative in {path}: {raw}")
        weight_mult = finite_float(raw.get("contact_weight_multiplier", 1.0), field="contact_weight_multiplier", path=path)
        if weight_mult < 0.0:
            raise ValueError(f"agent interaction judgment contact_weight_multiplier must be nonnegative in {path}: {raw}")
        normalized = {
            **raw,
            "judgment_id": str(raw.get("judgment_id", f"{hand_side}_{s}_{e}")),
            "frame_start": int(s),
            "frame_end": int(e),
            "hand_side": hand_side,
            "target_entity_id": str(target_entity_id),
            "contact_state": contact_state,
            "occlusion_relation": occlusion_relation,
            "depth_reliability": depth_reliability,
            "contact_prior_probability": float(prior),
            "contact_support_uncertainty_m": float(support_unc),
            "contact_weight_multiplier": float(weight_mult),
            "source_path": str(path),
        }
        kept_segments += 1
        state_counts[contact_state] += 1
        occlusion_counts[occlusion_relation] += 1
        for frame_idx in range(s, e + 1):
            key = (frame_idx, hand_side)
            if key in index:
                raise ValueError(f"overlapping agent interaction judgments for frame/side {key} in {path}: {index[key]} and {raw}")
            index[key] = normalized
    if kept_segments == 0:
        raise ValueError(f"agent interaction judgment contained no segments for case={case} target={target_entity_id} sides={sorted(sides)} span={frame_span}: {path}")
    return index, {
        "enabled": True,
        "path": str(path),
        "segments": int(kept_segments),
        "frame_side_count": int(len(index)),
        "contact_state_counts": dict(state_counts),
        "occlusion_relation_counts": dict(occlusion_counts),
    }


def ownership_quarantine_mode(judgment: dict[str, Any] | None) -> str:
    if not isinstance(judgment, dict):
        return "measurement_default"
    explicit = judgment.get("ownership_quarantine") or judgment.get("object_surface_policy")
    if isinstance(explicit, str):
        norm = explicit.strip()
        if norm in {"hand_projected", "measurement_default", "none", "unresolved"}:
            return norm
        if norm in {"quarantine_projected_hand", "quarantine_under_hand"}:
            return "hand_projected"
        if norm in {"do_not_quarantine", "object_visible"}:
            return "none"
    occlusion = str(judgment.get("occlusion_relation") or "")
    if occlusion in NO_QUARANTINE_OCCLUSIONS:
        return "none"
    if occlusion in HAND_FRONT_OCCLUSIONS:
        return "hand_projected"
    return "unresolved"


def contact_row_state_from_judgment(judgment: dict[str, Any] | None, *, image_candidate: bool, allow_agent_create: bool) -> tuple[bool, str]:
    if not isinstance(judgment, dict):
        return bool(image_candidate), "active_contact_patch" if image_candidate else "missing_contact_patch_row"
    state = str(judgment.get("contact_state") or "unresolved")
    if state == "no_contact":
        return True, "inactive_no_contact_by_agent_judgment"
    if state in {"likely_contact", "possible_contact"}:
        if image_candidate or allow_agent_create:
            return True, "active_contact_patch"
        return True, "inactive_agent_contact_without_image_support"
    if image_candidate:
        return True, "active_contact_patch"
    return True, "inactive_unresolved_contact_by_agent_judgment"


def combine_contact_prior(image_prior: float, judgment: dict[str, Any] | None) -> float:
    image_prior = float(np.clip(image_prior, 0.0, 1.0))
    if not isinstance(judgment, dict):
        return image_prior
    agent_prior = float(np.clip(float(judgment["contact_prior_probability"]), 0.0, 1.0))
    state = str(judgment.get("contact_state") or "unresolved")
    if state == "likely_contact":
        return max(image_prior, agent_prior)
    if state == "possible_contact":
        return max(min(image_prior, 0.65), agent_prior)
    if state == "no_contact":
        return min(image_prior, agent_prior)
    return agent_prior


def agent_judgment_provenance(judgment: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(judgment, dict):
        return {"agent_interaction_judgment": None}
    keys = [
        "source_path",
        "judgment_id",
        "frame_start",
        "frame_end",
        "contact_state",
        "occlusion_relation",
        "depth_reliability",
        "contact_prior_probability",
        "contact_support_uncertainty_m",
        "evidence",
        "uncertainty",
    ]
    return {"agent_interaction_judgment": {k: judgment.get(k) for k in keys if k in judgment}}


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def project_camera(points_camera: np.ndarray, intr: list[float], width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_camera, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(intr) != 4:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0,), dtype=bool)
    fx, fy, cx, cy = [float(x) for x in intr]
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1.0e-5)
    u_float = fx * pts[:, 0] / np.maximum(z, 1.0e-6) + cx
    v_float = fy * pts[:, 1] / np.maximum(z, 1.0e-6) + cy
    # V19 review/mask frames may be a scaled version of source intrinsics.
    scale_x = width / max(1.0, 2.0 * cx)
    scale_y = height / max(1.0, 2.0 * cy)
    u = np.rint(u_float * scale_x).astype(int)
    v = np.rint(v_float * scale_y).astype(int)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def draw_hand_mask(hand: dict[str, Any], width: int, height: int, *, point_radius: int, line_radius: int) -> tuple[np.ndarray, dict[str, Any]]:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or metric.get("v19_camera_intrinsics_fx_fy_cx_cy")
    joints = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
    verts = np.asarray(metric.get("vertices_camera_sample_m") or [], dtype=float)
    mask = np.zeros((height, width), dtype=np.uint8)
    joint_u = joint_v = np.zeros((0,), dtype=int)
    joint_valid = np.zeros((0,), dtype=bool)
    if joints.shape == (21, 3) and isinstance(intr, list) and len(intr) == 4:
        joint_u, joint_v, joint_valid = project_camera(joints, intr, width, height)
        for a, b in HAND_EDGES:
            if a < len(joint_valid) and b < len(joint_valid) and joint_valid[a] and joint_valid[b]:
                cv2.line(mask, (int(joint_u[a]), int(joint_v[a])), (int(joint_u[b]), int(joint_v[b])), 255, max(1, int(line_radius)))
        for u, v, ok in zip(joint_u, joint_v, joint_valid):
            if ok:
                cv2.circle(mask, (int(u), int(v)), max(1, int(point_radius)), 255, -1)
    vert_valid_count = 0
    if verts.ndim == 2 and verts.shape[1] == 3 and isinstance(intr, list) and len(intr) == 4:
        vu, vv, vvld = project_camera(verts, intr, width, height)
        vert_valid_count = int(np.count_nonzero(vvld))
        for u, v, ok in zip(vu, vv, vvld):
            if ok:
                cv2.circle(mask, (int(u), int(v)), max(1, int(point_radius // 2)), 255, -1)
    return mask > 0, {
        "joint_valid_count": int(np.count_nonzero(joint_valid)),
        "vertex_valid_count": int(vert_valid_count),
        "hand_support_px": int(np.count_nonzero(mask)),
    }


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def proximity_stats(hand_mask: np.ndarray, object_mask: np.ndarray, band_px: int) -> dict[str, Any]:
    hand_px = int(np.count_nonzero(hand_mask))
    object_px = int(np.count_nonzero(object_mask))
    overlap_px = int(np.count_nonzero(hand_mask & object_mask))
    if hand_px == 0 or object_px == 0:
        return {"hand_px": hand_px, "object_px": object_px, "overlap_px": overlap_px, "near_px": 0, "min_distance_px": None, "median_near_distance_px": None}
    obj_u8 = object_mask.astype(np.uint8)
    dist = cv2.distanceTransform((1 - obj_u8).astype(np.uint8), cv2.DIST_L2, 3)
    dvals = dist[hand_mask]
    finite = dvals[np.isfinite(dvals)]
    near = finite <= float(band_px)
    return {
        "hand_px": hand_px,
        "object_px": object_px,
        "overlap_px": overlap_px,
        "near_px": int(np.count_nonzero(near)),
        "min_distance_px": float(np.min(finite)) if finite.size else None,
        "median_near_distance_px": float(np.median(finite[near])) if np.any(near) else None,
    }


def save_bool_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def make_review(raw_path: Path, object_mask: np.ndarray, hand_mask: np.ndarray, non_object_mask: np.ndarray, out: Path, title: str) -> None:
    img = cv2.imread(str(raw_path))
    if img is None:
        img = np.zeros((object_mask.shape[0], object_mask.shape[1], 3), dtype=np.uint8)
    if img.shape[:2] != object_mask.shape:
        img = cv2.resize(img, (object_mask.shape[1], object_mask.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = img.copy()
    overlay[object_mask] = (0.65 * overlay[object_mask] + 0.35 * np.array([40, 255, 80])).astype(np.uint8)
    overlay[hand_mask] = (0.65 * overlay[hand_mask] + 0.35 * np.array([255, 255, 0])).astype(np.uint8)
    overlay[non_object_mask] = (0.35 * overlay[non_object_mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(overlay, title[:110], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])


def build(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.annotations)
    frames = as_list(payload.get("frames")) if isinstance(payload, dict) else []
    start, end = [int(x) for x in args.frame_span]
    out_case = args.output_root / args.case
    mask_dir = out_case / "ownership_masks"
    review_dir = out_case / "review_frames"
    ownership_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    side_set = set(str(s) for s in args.sides)
    judgment_index, judgment_summary = load_agent_interaction_judgments(
        args.agent_interaction_judgment,
        case=str(args.case),
        target_entity_id=str(args.target_entity_id),
        frame_span=(start, end),
        sides=side_set,
    )
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < start or frame_idx > end:
            continue
        obj = target_object(frame, str(args.target_entity_id))
        if not isinstance(obj, dict) or not bool(obj.get("visible", True)):
            diagnostics.append({"frame_idx": frame_idx, "state": "missing_visible_target_object"})
            continue
        if obj.get("rigid_pose_observation_eligible") is False:
            diagnostics.append({
                "frame_idx": frame_idx,
                "state": "skipped_object_mask_ineligible_for_contact_ownership",
                "reason": obj.get("rigid_pose_observation_reason"),
                "object_area_px": obj.get("area_px"),
            })
            continue
        mask_path_raw = obj.get("mask_path")
        if not isinstance(mask_path_raw, str) or not Path(mask_path_raw).exists():
            diagnostics.append({"frame_idx": frame_idx, "state": "missing_object_mask", "mask_path": mask_path_raw})
            continue
        object_mask = load_mask(Path(mask_path_raw))
        height, width = object_mask.shape
        object_contact_band = dilate(object_mask, int(args.contact_image_band_px))
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side") or "")
            if side not in side_set:
                continue
            hand_mask, hand_diag = draw_hand_mask(hand, width, height, point_radius=int(args.hand_radius_px), line_radius=int(args.hand_line_radius_px))
            judgment = judgment_index.get((frame_idx, side))
            quarantine_mode = ownership_quarantine_mode(judgment)
            hand_for_ownership = dilate(hand_mask, int(args.ownership_dilation_px))
            if quarantine_mode == "none":
                non_object_owned = np.zeros_like(object_mask, dtype=bool)
            else:
                non_object_owned = object_mask & hand_for_ownership
            visible_object_owned = object_mask & ~non_object_owned
            constraint_eligible = visible_object_owned.copy()
            prox = proximity_stats(hand_mask, object_mask, int(args.contact_image_band_px))
            contact_image_px = int(prox.get("overlap_px") or 0) + int(prox.get("near_px") or 0)
            image_candidate = contact_image_px >= int(args.min_contact_image_px)
            emit_contact_row, contact_state = contact_row_state_from_judgment(
                judgment,
                image_candidate=bool(image_candidate),
                allow_agent_create=bool(args.agent_judgment_can_create_contact),
            )
            side_mask_dir = mask_dir / side
            non_object_path = side_mask_dir / f"{frame_idx:06d}_non_object_owned.png"
            visible_object_path = side_mask_dir / f"{frame_idx:06d}_visible_object_owned.png"
            constraint_path = side_mask_dir / f"{frame_idx:06d}_constraint_eligible_entity.png"
            hand_support_path = side_mask_dir / f"{frame_idx:06d}_projected_mano_hand_support.png"
            save_bool_mask(non_object_path, non_object_owned)
            save_bool_mask(visible_object_path, visible_object_owned)
            save_bool_mask(constraint_path, constraint_eligible)
            save_bool_mask(hand_support_path, hand_mask)
            counts = {
                "non_object_owned_px": int(np.count_nonzero(non_object_owned)),
                "visible_object_owned_px": int(np.count_nonzero(visible_object_owned)),
                "constraint_eligible_entity_px": int(np.count_nonzero(constraint_eligible)),
                **hand_diag,
                **prox,
                "contact_image_px": int(contact_image_px),
            }
            judgment_fields = agent_judgment_provenance(judgment)
            ownership_rows.append({
                "factor_family": "visible_ownership",
                "target_entity_id": str(args.target_entity_id),
                "frame_idx": int(frame_idx),
                "hand_side": side,
                "variable_affected": "O_t_surface_visibility",
                "observation_type": "projected_mano_visible_hand_ownership_mask",
                "residual_or_quarantine_rule": "quarantine object hard-surface constraints whose projected support overlaps the projected MANO hand support; this marks hand-owned/uncertain first-surface pixels, not accepted contact",
                "provenance": {
                    "annotations": str(args.annotations),
                    "object_mask_path": str(mask_path_raw),
                    "hand_support_mask_path": str(hand_support_path),
                    "reason": "current MANO projection supplies hand-owned visible support where monocular depth at hand pixels is unreliable; Pi-agent interaction judgment configures whether this support quarantines object-surface constraints",
                    **judgment_fields,
                },
                "rendered_uncertainty_channel": "visible object pixels under projected hand support are rendered/treated as ownership-uncertain rather than hard object surface when the agent interaction judgment says the hand is in front or unresolved",
                "state": "active_visible_ownership",
                "agent_ownership_quarantine_mode": str(quarantine_mode),
                "agent_contact_state": judgment.get("contact_state") if isinstance(judgment, dict) else None,
                "agent_occlusion_relation": judgment.get("occlusion_relation") if isinstance(judgment, dict) else None,
                "agent_depth_reliability": judgment.get("depth_reliability") if isinstance(judgment, dict) else None,
                "non_object_owned_mask_path": str(non_object_path),
                "visible_object_owned_mask_path": str(visible_object_path),
                "constraint_eligible_entity_mask_path": str(constraint_path),
                "adjusted_entity_mask_path": str(constraint_path),
                "counts": counts,
            })
            min_dist = prox.get("min_distance_px")
            overlap_fraction = float(counts["overlap_px"]) / max(1.0, float(counts["object_px"]))
            near_fraction = float(counts["near_px"]) / max(1.0, float(counts["hand_px"]))
            if min_dist is None:
                proximity_fraction = 0.0
            else:
                proximity_fraction = max(0.0, min(1.0, 1.0 - float(min_dist) / max(1.0, float(args.contact_image_band_px))))
            image_contact_prior = max(0.05, min(0.65, 0.10 + 0.35 * proximity_fraction + 0.20 * min(1.0, 4.0 * overlap_fraction) + 0.10 * near_fraction))
            contact_prior = combine_contact_prior(float(image_contact_prior), judgment)
            support_uncertainty_m = float(args.contact_support_uncertainty_m)
            weight_multiplier = 1.0
            if isinstance(judgment, dict):
                support_uncertainty_m = float(judgment["contact_support_uncertainty_m"])
                weight_multiplier = float(judgment.get("contact_weight_multiplier", 1.0))
            active_weight = float(args.contact_weight) * max(float(args.raw_proposal_weight_factor), contact_prior) * float(weight_multiplier)
            if contact_state != "active_contact_patch":
                active_weight = 0.0
            if emit_contact_row:
                blockers = ["no_persistent_object_frame_contact_anchor", "raw_depth_at_hand_pixels_quarantined"]
                if isinstance(judgment, dict):
                    blockers.append("agent_interval_judgment_not_a_persistent_object_frame_anchor")
                    if judgment.get("contact_state") == "no_contact":
                        blockers.append("agent_judged_no_contact")
                    if contact_state == "inactive_agent_contact_without_image_support":
                        blockers.append("agent_contact_judgment_without_projected_image_support")
                contact_rows.append({
                    "factor_family": "contact_patch",
                    "target_entity_id": str(args.target_entity_id),
                    "frame_idx": int(frame_idx),
                    "hand_side": side,
                    "variable_affected": "H_t",
                    "observation_type": "agent_judged_interaction_plus_projected_mano_object_image_adjacency",
                    "residual_or_quarantine_rule": "Pi-agent interval contact/occlusion judgment sets the latent contact prior and support uncertainty; projected MANO/object geometry selects candidate local contact vertices when active; no persistent object-frame contact anchor is claimed",
                    "provenance": {
                        "annotations": str(args.annotations),
                        "object_mask_path": str(mask_path_raw),
                        "hand_support_mask_path": str(hand_support_path),
                        "non_object_owned_mask_path": str(non_object_path),
                        "reason": "Pi-agent visual interaction judgment supplies the contact/occlusion switch; projected MANO/object image adjacency supplies local geometric support and remains uncertain under raw hand-depth ambiguity",
                        **judgment_fields,
                    },
                    "rendered_uncertainty_channel": "agent-judged contact/occlusion prior consumed as a soft graph switch; final render must still show posterior/uncertainty rather than treating the judgment as metric truth",
                    "state": str(contact_state),
                    "weight": float(active_weight),
                    "contact_patch_base_weight": float(args.contact_weight),
                    "contact_patch_band_m": float(args.contact_patch_band_m),
                    "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
                    "contact_patch_support_uncertainty_m": float(support_uncertainty_m),
                    "object_support_uncertainty_m": float(support_uncertainty_m),
                    "contact_patch_deadband_m": float(args.contact_patch_target_margin_m + support_uncertainty_m),
                    "max_vertices": int(args.max_vertices),
                    "contact_state_prior_probability": float(contact_prior),
                    "contact_patch_prior_probability": float(contact_prior),
                    "image_contact_prior_probability": float(image_contact_prior),
                    "agent_contact_state": judgment.get("contact_state") if isinstance(judgment, dict) else None,
                    "agent_occlusion_relation": judgment.get("occlusion_relation") if isinstance(judgment, dict) else None,
                    "agent_depth_reliability": judgment.get("depth_reliability") if isinstance(judgment, dict) else None,
                    "agent_contact_weight_multiplier": float(weight_multiplier),
                    "contact_anchor_state": "agent_interval_prior_no_stable_pose_anchor" if isinstance(judgment, dict) else "image_adjacency_only_no_stable_pose_anchor",
                    "contact_anchor_residual_allowed": False,
                    "contact_anchor_blockers": blockers,
                    "contact_pose_anchor_key": None,
                    "image_contact_counts": counts,
                    "local_patch_sample_count": int(contact_image_px),
                    "local_patch_temporal_sample_count": 0,
                    "local_patch_support_state": "agent_judgment_with_projected_mano_object_image_adjacency" if isinstance(judgment, dict) else "projected_mano_object_image_adjacency",
                    "local_patch_support_consumed": bool(contact_state == "active_contact_patch"),
                    "local_patch_support_uncertainty_m": float(support_uncertainty_m),
                    "global_object_support_uncertainty_m": float(support_uncertainty_m),
                })
            diagnostics.append({
                "frame_idx": frame_idx,
                "hand_side": side,
                "image_candidate_contact": bool(image_candidate),
                "emitted_contact_row": bool(emit_contact_row),
                "contact_row_state": str(contact_state),
                "agent_judgment_id": judgment.get("judgment_id") if isinstance(judgment, dict) else None,
                "agent_ownership_quarantine_mode": str(quarantine_mode),
                "contact_prior_probability": float(contact_prior),
                "image_contact_prior_probability": float(image_contact_prior),
                "counts": counts,
            })
            if frame_idx in set(int(x) for x in args.review_frames):
                raw_path = Path(str(frame.get("raw_frame_path", "")))
                make_review(raw_path, object_mask, hand_mask, non_object_owned, review_dir / f"{frame_idx:06d}_{side}_contact_ownership.jpg", f"{frame_idx} {side} green=obj yellow=MANO red=hand-owned object")
    duplicate_keys = [(r["frame_idx"], r["hand_side"]) for r in contact_rows]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError("duplicate contact rows emitted")
    payload = {
        "method": "v19_visible_contact_ownership_factor_from_projected_mano_object_mask_and_agent_interaction_judgment",
        "case": str(args.case),
        "target_entity_id": str(args.target_entity_id),
        "claim_scope": "Pi-agent interval interaction judgments supply explicit contact/occlusion priors; projected MANO/object image adjacency supplies local support and ownership masks. The report does not assert persistent contact anchors, exact hand depth, or object pose correction.",
        "inputs": {"annotations": str(args.annotations), "agent_interaction_judgment": None if args.agent_interaction_judgment is None else str(args.agent_interaction_judgment)},
        "parameters": {
            "frame_span": [start, end],
            "sides": list(args.sides),
            "hand_radius_px": int(args.hand_radius_px),
            "hand_line_radius_px": int(args.hand_line_radius_px),
            "ownership_dilation_px": int(args.ownership_dilation_px),
            "contact_image_band_px": int(args.contact_image_band_px),
            "min_contact_image_px": int(args.min_contact_image_px),
            "contact_weight": float(args.contact_weight),
            "contact_patch_band_m": float(args.contact_patch_band_m),
            "contact_patch_target_margin_m": float(args.contact_patch_target_margin_m),
            "contact_support_uncertainty_m": float(args.contact_support_uncertainty_m),
            "agent_judgment_can_create_contact": bool(args.agent_judgment_can_create_contact),
        },
        "agent_interaction_judgment_summary": judgment_summary,
        "summary": {
            "ownership_row_count": len(ownership_rows),
            "contact_patch_row_count": len(contact_rows),
            "active_contact_patch_row_count": int(sum(str(r.get("state")) == "active_contact_patch" for r in contact_rows)),
            "inactive_contact_patch_row_count": int(sum(str(r.get("state")) != "active_contact_patch" for r in contact_rows)),
            "contact_patch_state_counts": dict(Counter(str(r.get("state")) for r in contact_rows)),
            "agent_ownership_quarantine_mode_counts": dict(Counter(str(r.get("agent_ownership_quarantine_mode")) for r in ownership_rows)),
            "agent_contact_state_counts": dict(Counter(str(r.get("agent_contact_state")) for r in contact_rows if r.get("agent_contact_state") is not None)),
            "agent_occlusion_relation_counts": dict(Counter(str(r.get("agent_occlusion_relation")) for r in ownership_rows if r.get("agent_occlusion_relation") is not None)),
            "non_object_owned_px_total": int(sum(((r.get("counts") or {}).get("non_object_owned_px") or 0) for r in ownership_rows)),
            "contact_image_px_total": int(sum(((r.get("image_contact_counts") or {}).get("contact_image_px") or 0) for r in contact_rows)),
            "frames_with_active_contact_patch": sorted(set(int(r["frame_idx"]) for r in contact_rows if str(r.get("state")) == "active_contact_patch")),
            "frames_with_any_contact_patch_row": sorted(set(int(r["frame_idx"]) for r in contact_rows)),
        },
        "ownership_rows": ownership_rows,
        "factor_rows": contact_rows,
        "diagnostics_preview": diagnostics[:160],
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args)
    out = args.output_root / args.case / "v19_visible_contact_ownership_factor_report.json"
    write_json(out, payload)
    print(json.dumps({"status": "ok", "report": str(out), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
