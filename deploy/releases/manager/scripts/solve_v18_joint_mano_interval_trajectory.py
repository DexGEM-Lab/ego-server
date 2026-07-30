#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Solve a continuous MANO interval trajectory with object/visual constraints.

This is the workbench solver path, not a diagnostic classifier.  It jointly
optimizes, over a contiguous interval, current-space MANO root translation,
root/wrist orientation delta, and finger articulation delta.  The zero state is
side-specific exact HaWoR MANO replay mapped to the current V18 bridge surface.

The physical objective is:
  - stay close to the current visible/depth hand observation,
  - stay temporally smooth in root motion, wrist orientation, and articulation,
  - move penetrating MANO surface out of trusted observed object surface, and
  - ignore hidden/free-space-conflicted completion volume as a force.

If this solver fails to make a coherent rendered hand trajectory, that is a
failure of a named scientific assumption, not an acceptance/count result.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_mano_object_constraint_state import frame_intrinsics, project  # noqa: E402
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import (  # noqa: E402
    VERTEX_OBSERVED_SUPPORTED,
    classify_object_vertices_against_depth,
    face_provenance,
)
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    HAND_EDGES,
    bridge_vertices_and_joints,
    load_source_arrays,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    rotvec_to_matrix,
    similarity_from_to,
    source_npz_for_hand,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    inverse_object,
    load_json,
    load_mesh,
    numeric_summary,
    object_vec_to_world,
    pose_map,
    write_json,
)

SANITIZED_ANNOTATION_ROOT = Path("/data2/ego_annotation_outputs/v18_full_pipeline_sanitized_base_for_hprime")
REJECTED_ANNOTATION_PATH_MARKERS = (
    "v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard",
    "verified_hprime_final",
    "hprime_final",
)
DEFAULT_ANNOTATIONS = SANITIZED_ANNOTATION_ROOT / "task5_tomato_960/annotations_v18_full.json"
DEFAULT_POSE_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json"
)
DEFAULT_MESH = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply"
)
DEFAULT_DEPTH = Path(
    "/data2/ego_annotation_outputs/v18_unidepth_extension/complete_depth_root/task5_tomato_960/"
    "unidepth_metric/unidepth_metric_depth_v3.npz"
)
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_v1")
DEFAULT_LEFT_MANO = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl")
DEFAULT_WILOR_ROOT = Path("third_party/WiLoR")


@dataclass(frozen=True)
class FrameHandRow:
    frame_idx: int
    side: str
    frame: dict[str, Any]
    current_vertices_world: np.ndarray
    current_joints_world: np.ndarray
    raw_vertices_world: np.ndarray
    raw_joints_world: np.ndarray
    root_orient_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    betas: np.ndarray
    trans_world_m: np.ndarray
    similarity_scale: float
    similarity_rotation_raw_to_current: np.ndarray
    similarity_translation_raw_to_current: np.ndarray
    source_hawor_npz: Path
    source_frame_index: int
    constraint_indices: np.ndarray
    constraint_normals_world: np.ndarray
    constraint_depths_m: np.ndarray
    observed_initial_measure: dict[str, Any]
    observed_constraint_count: int
    object_rotation_world_from_object: np.ndarray
    object_translation_world_m: np.ndarray
    face_strict_observed_raw: np.ndarray
    face_strict_observed: np.ndarray
    hand_owned_quarantined_face_count: int
    surface_eligibility_npz_path: str | None
    surface_eligibility_mode: str | None
    observed_surface_support_uncertainty_m: float
    surface_eligible_face_count: int
    surface_input_face_count: int
    surface_applied_face_delta: int
    visible_ownership_non_object_mask_path: str | None
    visible_ownership_object_owned_mask_path: str | None
    visible_ownership_constraint_eligible_mask_path: str | None
    visible_ownership_non_object_owned_px: int
    visible_ownership_object_owned_px: int
    visible_ownership_constraint_eligible_px: int
    visible_ownership_quarantined_face_count: int
    visible_object_mask_path: str | None
    visible_object_mask_face_count_raw: int
    visible_object_mask_face_count: int
    visible_surface_track_factor_state: str | None
    visible_surface_track_mask_path: str | None
    visible_surface_track_npz_path: str | None
    visible_surface_track_valid_depth_pixels: int
    visible_surface_track_quarantined_face_count: int
    visible_surface_depth_order_vertex_indices: np.ndarray
    visible_surface_depth_order_depth_m: np.ndarray
    visible_surface_depth_order_initial_delta_m: np.ndarray
    visible_surface_depth_order_initial_measure: dict[str, Any]
    hand_observation_visibility_factor_state: str | None
    hand_observation_visibility_candidate_px: int
    hand_observation_visibility_weight_multiplier: float
    contact_patch_factor_state: str | None
    contact_patch_vertex_indices: np.ndarray
    contact_patch_target_world_m: np.ndarray
    contact_patch_normal_world: np.ndarray
    contact_patch_initial_distance_m: np.ndarray
    contact_patch_weight: float
    contact_patch_prior_probability: float
    contact_patch_band_m: float
    contact_patch_target_margin_m: float
    contact_patch_support_uncertainty_m: float
    contact_patch_support_uncertainty_source: str | None
    local_patch_support_state: str | None
    local_patch_support_consumed: bool
    local_patch_support_uncertainty_m: float | None
    global_object_support_uncertainty_m: float | None
    local_patch_sample_count: int
    local_patch_temporal_sample_count: int
    contact_anchor_state: str | None
    contact_anchor_residual_allowed: bool
    contact_anchor_blockers: list[str]
    contact_pose_anchor_key: str | None
    joint_visibility_weights: np.ndarray
    joint_depth_residual_m: np.ndarray
    hand_ray_shift_prior_world_m: np.ndarray
    hand_ray_shift_prior_source_m: float | None
    hand_ray_shift_prior_weight: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="task5_tomato_960")
    p.add_argument("--object-id", default="object:obj_tomato")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    p.add_argument("--completed-mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument(
        "--completion-report",
        type=Path,
        default=None,
        help="Optional P13 compact-rigid completion report. When supplied, --completed-mesh must equal outputs.completed_mesh_labeled.",
    )
    p.add_argument("--depth-npz", type=Path, action="append", default=None, help="Depth NPZ path(s). Defaults to the task5 complete-depth source only when omitted; explicit paths replace that default for other cases.")
    p.add_argument("--hand-depth-repair-graph", type=Path, default=None, help="Optional prior source with per-frame hand_ray_shift_m camera-ray observations from the V17 hand-depth repair graph.")
    p.add_argument("--use-hand-ray-shift-prior", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--initialize-hand-ray-shift", action=argparse.BooleanOptionalAction, default=False, help="Initialize MANO translation deltas from the hand-ray depth repair observation for a discriminating repair test.")
    p.add_argument("--hand-ray-shift-prior-weight", type=float, default=2.5e3)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    p.add_argument("--wilor-mano-right", type=Path, default=None)
    p.add_argument("--wilor-mano-left", type=Path, default=DEFAULT_LEFT_MANO)
    p.add_argument("--hawor-left-shapedirs-x-fix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--start-frame", type=int, default=453)
    p.add_argument("--end-frame", type=int, default=508)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--support-margin-m", type=float, default=0.015)
    p.add_argument("--free-space-margin-m", type=float, default=0.025)
    p.add_argument("--penetration-epsilon-m", type=float, default=1.0e-5)
    p.add_argument("--max-constraints-per-frame", type=int, default=96)
    p.add_argument("--max-optimizer-iterations", type=int, default=120)
    p.add_argument("--visible-shift-limit-px", type=float, default=12.0)
    p.add_argument("--depth-shift-limit-m", type=float, default=0.035)
    p.add_argument("--max-translation-m", type=float, default=0.045)
    p.add_argument("--max-root-delta-rad", type=float, default=0.30)
    p.add_argument("--max-pose-delta-rad", type=float, default=0.45)
    p.add_argument("--translation-prior-weight", type=float, default=2.0e3)
    p.add_argument("--root-prior-weight", type=float, default=1.5e2)
    p.add_argument("--pose-prior-weight", type=float, default=7.5e1)
    p.add_argument("--smooth-weight", type=float, default=5.0e3)
    p.add_argument("--accel-weight", type=float, default=1.0e4)
    p.add_argument("--observed-penetration-weight", type=float, default=3.0e5)
    p.add_argument("--zero-surface-mode", choices=("bridge_delta", "similarity_mapped_raw"), default="bridge_delta", help="bridge_delta uses the current V18 bridge as the zero surface and maps MANO deltas onto it; similarity_mapped_raw optimizes the similarity-transformed MANO surface directly.")
    p.add_argument("--dense-observed-surface-barrier", action=argparse.BooleanOptionalAction, default=True, help="Apply a tangent-plane nonpenetration barrier to every MANO vertex whose nearest object face is observed-supported, not only the current active penetrating subset.")
    p.add_argument("--dense-observed-penetration-weight", type=float, default=3.0e5)
    p.add_argument("--optimize-object-translation", action=argparse.BooleanOptionalAction, default=False, help="Jointly solve a small per-frame object translation delta so hand/object residual can expose tomato pose-depth alignment error instead of forcing all correction into MANO.")
    p.add_argument("--hand-owned-object-depth-quarantine", action=argparse.BooleanOptionalAction, default=False, help="Do not treat object faces as trusted observed-surface constraints when their projected depth is plausibly owned by the visible/current hand surface.")
    p.add_argument("--hand-owned-quarantine-radius-px", type=float, default=3.0)
    p.add_argument("--hand-owned-quarantine-depth-margin-m", type=float, default=0.005, help="Required camera-depth foreground separation for hand-owned object-depth quarantine.")
    p.add_argument("--hand-owned-quarantine-hand-depth-support-m", type=float, default=0.030)
    p.add_argument("--max-object-translation-m", type=float, default=0.015, help="Object translation uncertainty bound; default equals the observed-depth support margin scale.")
    p.add_argument("--object-translation-prior-weight", type=float, default=4.0e3)
    p.add_argument("--object-smooth-weight", type=float, default=8.0e3)
    p.add_argument("--visibility-weighted-hand-observation", action=argparse.BooleanOptionalAction, default=False, help="Use metric depth support to reduce HaWoR joint/articulation anchoring for occluded or depth-inconsistent fingers while preserving visible joints.")
    p.add_argument("--visible-joint-depth-margin-m", type=float, default=0.030)
    p.add_argument("--occluded-joint-observation-weight", type=float, default=0.12)
    p.add_argument("--front-inconsistent-joint-observation-weight", type=float, default=0.35)
    p.add_argument("--invalid-joint-observation-weight", type=float, default=0.35)
    p.add_argument("--visible-hinge-weight", type=float, default=8.0e2)
    p.add_argument("--depth-hinge-weight", type=float, default=2.0e4)
    p.add_argument("--bound-hinge-weight", type=float, default=3.0e3)
    p.add_argument("--sample-vertex-count-for-render", type=int, default=160)
    p.add_argument("--active-set-iterations", type=int, default=6, help="Maximum active-set passes. Each pass optimizes, remeasures full observed-surface penetration, and expands constraints. A closed pass adds zero constraints.")
    p.add_argument("--visible-ownership-factor-report", type=Path, default=None, help="Optional reusable visible ownership factor report. Its non_object_owned masks quarantine hard object constraints; its visible_object_owned masks replace visible-object masks for depth-order/gating when present.")
    p.add_argument("--surface-eligibility-factor-report", type=Path, default=None, help="Optional reusable surface eligibility factor report. Its eligible_hard_observed face masks define which object faces may exert hard MANO constraints.")
    p.add_argument("--surface-eligibility-mode", choices=("replace", "intersect"), default="intersect", help="How to apply surface eligibility to the current trusted face set when a factor report is supplied. Default intersects to preserve existing ownership/visibility quarantines unless replacement is explicitly justified.")
    p.add_argument("--observed-surface-support-uncertainty-m", type=float, default=0.0, help="Default support uncertainty slack for observed object surface nonpenetration. Row-level surface_support_uncertainty_m / observed_surface_support_uncertainty_m from surface_eligibility factors overrides this value.")
    p.add_argument("--visible-surface-track-factor-report", type=Path, default=None, help="Optional reusable visible-surface track factor report. Active rows provide model-mask/metric-depth first-surface observations for MANO depth-order and hidden-volume quarantine.")
    p.add_argument("--factor-report", type=Path, action="append", default=None, help="Generic reusable factor report(s). Rows are dispatched by factor_family, enabling ownership, surface_eligibility, visible_surface_track, hand_observation_visibility, hand_depth_shift_prior, and contact_patch factors through one interface.")
    p.add_argument("--contact-patch-weight", type=float, default=0.0, help="Default weight for active contact_patch factor rows. A row-level weight overrides this value. The residual keeps selected MANO vertices near an observed object surface patch; existing nonpenetration prevents crossing.")
    p.add_argument("--contact-patch-band-m", type=float, default=0.020, help="Current-state max distance for selecting MANO vertices that define the local contact patch.")
    p.add_argument("--contact-patch-target-margin-m", type=float, default=0.0025, help="Allowed hand-to-patch distance before the two-sided contact residual is active.")
    p.add_argument("--contact-patch-support-uncertainty-m", type=float, default=0.0, help="Independent object/patch support uncertainty added to the contact_patch deadband. Row-level object_support_uncertainty_m overrides this value. Use this to represent latent/sliding contact support rather than hard current-surface anchoring.")
    p.add_argument("--contact-patch-residual-mode", choices=("deadband_tube", "support_scaled_attraction"), default="deadband_tube", help="deadband_tube preserves the original bounded-contact residual: no force inside target margin plus support uncertainty. support_scaled_attraction treats active contact as a soft normal-manifold likelihood whose precision is reduced by object/patch support uncertainty, so it can test whether contact evidence can move H_t without claiming millimetre object support.")
    p.add_argument("--max-contact-patch-vertices", type=int, default=96, help="Maximum MANO vertices selected for each contact_patch factor row.")
    p.add_argument("--optimize-contact-state", action=argparse.BooleanOptionalAction, default=False, help="Optimize a per-row latent contact probability C_t for contact_patch rows. The contact residual is weighted by posterior C_t, while observation priors and temporal continuity keep plausible contacts active.")
    p.add_argument("--contact-state-prior-residual-scale-m", type=float, default=0.010, help="Physical residual scale used to convert contact row weights into contact-state prior strength. Deviating C_t from its observation prior by 1 costs the same as this many metres of contact residual.")
    p.add_argument("--contact-state-temporal-strength", type=float, default=1.0, help="Multiplier on same-side adjacent-frame C_t smoothness, using the same physical weight scale as the contact-state prior.")
    p.add_argument("--contact-state-geometry-likelihood", action=argparse.BooleanOptionalAction, default=False, help="Add a metric contact-compatibility observation on C_t from the selected current MANO-to-patch distance relative to the row's target margin plus object/patch support uncertainty. This makes C_t an evidence-updated switch instead of only a prior/temporal scalar.")
    p.add_argument("--require-contact-patch-pose-anchor", action=argparse.BooleanOptionalAction, default=False, help="Fail loudly if an active contact_patch row is consumed without explicit stable contact-anchor support. Use only to test a persistent object-frame A_t/pose-anchor mechanism; local visible-surface contact rows must not be silently upgraded to point anchors.")
    p.add_argument("--visible-ownership-face-overlap-dilation-px", type=int, default=2, help="Pixel dilation for deciding whether any projected face support sample overlaps non-object-owned ownership pixels.")
    p.add_argument("--visible-object-mask-report", type=Path, default=None, help="Legacy visible entity mask report. Prefer --factor-report with factor_family=visible_surface_track; this path remains only for reproducing earlier mask/depth ablations.")
    p.add_argument("--visible-object-mask-gate", action=argparse.BooleanOptionalAction, default=False, help="Legacy gate: trust observed object mesh faces only when their projected center lies inside the model-produced visible entity mask.")
    p.add_argument("--visible-mask-quarantine-signed-mesh", action=argparse.BooleanOptionalAction, default=False, help="On frames with a visible entity mask or active visible-surface factor, do not use compact-mesh signed nonpenetration as a trusted force; rely on visible first-surface constraints instead.")
    p.add_argument("--visible-object-mask-dilation-px", type=int, default=2)
    p.add_argument("--visible-surface-depth-order-term", dest="visible_surface_depth_order_term", action=argparse.BooleanOptionalAction, default=False, help="Penalize MANO vertices that project inside a visible first-surface mask but remain in front of the observed metric depth beyond a margin. Active visible_surface_track factors enable this residual automatically.")
    p.add_argument("--visible-surface-depth-order-margin-m", dest="visible_surface_depth_order_margin_m", type=float, default=0.010)
    p.add_argument("--visible-surface-depth-order-weight", dest="visible_surface_depth_order_weight", type=float, default=2.0e4)
    p.add_argument("--max-visible-surface-depth-vertices", dest="max_visible_surface_depth_vertices", type=int, default=160)
    p.add_argument("--gate-translation-with-visible-surface-support", action=argparse.BooleanOptionalAction, default=False, help="Output gate: when selected visible-surface depth-order support is at or below the threshold, preserve the source HaWoR wrist/root translation while keeping optimized wrist-relative MANO articulation. This prevents ungrounded contact/temporal terms from moving the global hand state.")
    p.add_argument("--freeze-translation-without-visible-surface-support", action=argparse.BooleanOptionalAction, default=False, help="In-solver gate: rows whose selected visible-surface depth-order support count is at or below --translation-gate-min-visible-surface-depth-vertices cannot use global MANO translation during optimization. This tests whether unsupported latent translation contaminates wrist-relative articulation before the output gate projects the wrist/root back to HaWoR.")
    p.add_argument("--translation-gate-min-visible-surface-depth-vertices", type=int, default=0, help="Rows with selected visible-surface depth-order vertex count <= this value are translation-gated when --gate-translation-with-visible-surface-support is enabled, and translation-frozen during optimization when --freeze-translation-without-visible-surface-support is enabled.")
    p.add_argument("--visible-lid-depth-order-term", dest="visible_surface_depth_order_term", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--visible-lid-depth-order-margin-m", dest="visible_surface_depth_order_margin_m", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--visible-lid-depth-order-weight", dest="visible_surface_depth_order_weight", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--max-visible-lid-depth-vertices", dest="max_visible_surface_depth_vertices", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    return p.parse_args()


def completion_report_completed_mesh(path: Path) -> Path:
    data = load_json(path)
    outputs = data.get("outputs") if isinstance(data, dict) else None
    if not isinstance(outputs, dict):
        raise RuntimeError(f"completion report {path} has no outputs object")
    value = outputs.get("completed_mesh_labeled") or outputs.get("completed_mesh")
    if not value:
        raise RuntimeError(f"completion report {path} has no completed mesh output")
    return Path(str(value))


def same_mesh_path(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


def validate_completed_mesh_contract(completed_mesh: Path, completion_report: Path | None) -> Path | None:
    if completed_mesh.name == "trellis_mesh.ply" or any(part.startswith("trellis_") for part in completed_mesh.parts):
        raise RuntimeError(
            "completed mesh frame mismatch: this solver consumes a P13 completed-canonical mesh, "
            f"not raw TRELLIS model output ({completed_mesh})"
        )
    if completion_report is None:
        return None
    expected = completion_report_completed_mesh(completion_report)
    if not same_mesh_path(completed_mesh, expected) and completed_mesh.resolve(strict=False) != expected.resolve(strict=False):
        raise RuntimeError(
            "completed mesh frame mismatch: pose rows and object-contact constraints are in the P13 completed-canonical frame, "
            f"but --completed-mesh={completed_mesh} differs from {completion_report} outputs.completed_mesh_labeled={expected}"
        )
    if not completed_mesh.exists() or completed_mesh.stat().st_size <= 0:
        raise RuntimeError(f"completed mesh {completed_mesh} is missing or empty")
    return expected


def project_world(points_world: np.ndarray, frame: dict[str, Any], side: str) -> np.ndarray | None:
    intr = frame_intrinsics(frame, side)
    if intr is None:
        return None
    r_c2w, t_c2w = frame_camera_pose(frame)
    return project(points_world, r_c2w, t_c2w, intr)


def world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    r_c2w, t_c2w = frame_camera_pose(frame)
    return (points_world - t_c2w[None, :]) @ r_c2w


def load_visible_object_mask_paths(report_path: Path | None) -> dict[int, Path]:
    if report_path is None:
        return {}
    if not report_path.exists():
        raise FileNotFoundError(f"missing visible object mask report: {report_path}")
    payload = load_json(report_path)
    rows = []
    if isinstance(payload, dict):
        rows.extend(as_list(payload.get("saved_mask_rows_after_start")))
        if not rows:
            rows.extend(as_list(payload.get("target_mask_rows")))
        if not rows:
            rows.extend(as_list(payload.get("surface_rows")))
    out: dict[int, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        mask_path = row.get("saved_mask_path") or row.get("mask_path")
        if not mask_path:
            continue
        path = Path(str(mask_path))
        if path.exists():
            out[int(row["frame_idx"])] = path
    return out


def load_binary_mask(mask_path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    cached = cache.get(mask_path)
    if cached is not None:
        return cached
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    cache[mask_path] = mask
    return mask


FACTOR_REQUIRED_FIELDS = (
    "factor_family",
    "target_entity_id",
    "frame_idx",
    "hand_side",
    "variable_affected",
    "observation_type",
    "residual_or_quarantine_rule",
    "provenance",
    "rendered_uncertainty_channel",
)


def validate_factor_row_contract(row: dict[str, Any], *, expected_family: str, target_entity_id: str, report_path: Path) -> tuple[int, str]:
    missing = [field for field in FACTOR_REQUIRED_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"{expected_family} factor row lacks required fields {missing} in {report_path}: {row}")
    family = str(row.get("factor_family") or "")
    if family != expected_family:
        raise ValueError(f"factor row family {family!r} does not match expected {expected_family!r} in {report_path}: {row}")
    if str(row.get("target_entity_id")) != str(target_entity_id):
        raise ValueError(f"{expected_family} factor target {row.get('target_entity_id')} does not match solver target {target_entity_id} in {report_path}")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(f"{expected_family} factor row has empty/non-dict provenance in {report_path}: {row}")
    return int(row["frame_idx"]), str(row["hand_side"])


def load_factor_rows(report_path: Path | None, row_key: str, *, expected_family: str, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if report_path is None:
        return {}
    if not report_path.exists():
        raise FileNotFoundError(f"missing {expected_family} factor report: {report_path}")
    payload = load_json(report_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{expected_family} factor report is not a JSON object: {report_path}")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in as_list(payload.get(row_key)):
        if not isinstance(row, dict):
            continue
        key = validate_factor_row_contract(row, expected_family=expected_family, target_entity_id=target_entity_id, report_path=report_path)
        if key in out:
            raise ValueError(f"duplicate {expected_family} factor row for {key} while reading {report_path}")
        out[key] = row
    return out


def load_visible_ownership_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    return load_factor_rows(report_path, "ownership_rows", expected_family="visible_ownership", target_entity_id=target_entity_id)


def load_surface_eligibility_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    return load_factor_rows(report_path, "factor_rows", expected_family="surface_eligibility", target_entity_id=target_entity_id)


def load_visible_surface_track_rows(report_path: Path | None, *, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    return load_factor_rows(report_path, "factor_rows", expected_family="visible_surface_track", target_entity_id=target_entity_id)


def load_generic_factor_reports(report_paths: list[Path] | None, *, target_entity_id: str) -> dict[str, dict[tuple[int, str], dict[str, Any]]]:
    out: dict[str, dict[tuple[int, str], dict[str, Any]]] = {
        "visible_ownership": {},
        "surface_eligibility": {},
        "visible_surface_track": {},
        "hand_observation_visibility": {},
        "hand_depth_shift_prior": {},
        "contact_patch": {},
    }
    for report_path in list(report_paths or []):
        if not report_path.exists():
            raise FileNotFoundError(f"missing generic factor report: {report_path}")
        payload = load_json(report_path)
        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            rows.extend([r for r in as_list(payload.get("ownership_rows")) if isinstance(r, dict)])
            rows.extend([r for r in as_list(payload.get("factor_rows")) if isinstance(r, dict)])
        for row in rows:
            family = str(row.get("factor_family") or "")
            if family not in out:
                continue
            key = validate_factor_row_contract(row, expected_family=family, target_entity_id=target_entity_id, report_path=report_path)
            if key in out[family]:
                raise ValueError(f"duplicate generic {family} factor row for {key} while reading {report_path}")
            out[family][key] = row
    return out


def merge_factor_row_maps(family: str, specific: dict[tuple[int, str], dict[str, Any]], generic: dict[tuple[int, str], dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    merged = dict(generic)
    overlap = sorted(set(merged) & set(specific))
    if overlap:
        raise ValueError(f"factor family {family} supplied by both generic and family-specific reports for keys: {overlap[:5]}")
    merged.update(specific)
    return merged


def visible_surface_track_mask_for_row(row: dict[str, Any] | None, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not isinstance(row, dict):
        return None, {"state": "missing_visible_surface_track_row"}
    state = str(row.get("surface_state") or row.get("state") or "unknown")
    raw = row.get("surface_mask_path")
    if state != "active_visible_surface":
        return None, {
            "state": state,
            "surface_mask_path": raw if isinstance(raw, str) else None,
            "visible_surface_npz_path": row.get("visible_surface_npz_path") if isinstance(row.get("visible_surface_npz_path"), str) else None,
            "valid_depth_pixels": int(row.get("valid_depth_pixels", 0) or 0),
            "quarantine_hidden_volume": bool(row.get("quarantine_hidden_volume", False)),
        }
    if not isinstance(raw, str) or not Path(raw).exists():
        raise FileNotFoundError(f"active visible-surface factor row has no readable mask path: {raw}")
    mask = load_binary_mask(Path(raw), cache)
    return mask, {
        "state": state,
        "surface_mask_path": raw,
        "visible_surface_npz_path": row.get("visible_surface_npz_path") if isinstance(row.get("visible_surface_npz_path"), str) else None,
        "valid_depth_pixels": int(row.get("valid_depth_pixels", 0) or 0),
        "quarantine_hidden_volume": bool(row.get("quarantine_hidden_volume", True)),
    }


def surface_eligibility_mask_for_row(row: dict[str, Any] | None, expected_count: int, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not isinstance(row, dict):
        return None, {"state": "missing_surface_eligibility_row"}
    raw = row.get("face_state_npz_path")
    if not isinstance(raw, str) or not Path(raw).exists():
        return None, {"state": "missing_surface_eligibility_npz", "face_state_npz_path": raw if isinstance(raw, str) else None}
    path = Path(raw)
    if path not in cache:
        with np.load(path) as data:
            if "eligible_hard_observed" not in data:
                raise KeyError(f"surface eligibility NPZ lacks eligible_hard_observed: {path}")
            cache[path] = np.asarray(data["eligible_hard_observed"], dtype=bool)
    mask = cache[path]
    if mask.shape != (int(expected_count),):
        return None, {"state": "surface_eligibility_shape_mismatch", "face_state_npz_path": str(path), "mask_count": int(mask.size), "expected_count": int(expected_count)}
    try:
        support_uncertainty_m = float(row.get("observed_surface_support_uncertainty_m", row.get("surface_support_uncertainty_m", row.get("object_support_uncertainty_m", 0.0))) or 0.0)
    except Exception:
        support_uncertainty_m = 0.0
    return mask.copy(), {"state": "ok", "face_state_npz_path": str(path), "eligible_hard_observed_count": int(np.count_nonzero(mask)), "observed_surface_support_uncertainty_m": max(0.0, support_uncertainty_m)}


def visible_ownership_masks_for_row(row: dict[str, Any] | None, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if not isinstance(row, dict):
        return None, None, {"state": "missing_visible_ownership_row"}
    non_object_raw = row.get("non_object_owned_mask_path")
    constraint_raw = row.get("constraint_eligible_entity_mask_path") or row.get("adjusted_entity_mask_path") or row.get("visible_object_owned_mask_path")
    visible_object_raw = row.get("visible_object_owned_mask_path")
    if not isinstance(non_object_raw, str) or not Path(non_object_raw).exists():
        raise FileNotFoundError(f"visible ownership row has no readable non_object_owned_mask_path: {non_object_raw}")
    if not isinstance(constraint_raw, str) or not Path(constraint_raw).exists():
        raise FileNotFoundError(f"visible ownership row has no readable constraint_eligible_entity/adjusted_entity mask path: {constraint_raw}")
    non_object_mask = load_binary_mask(Path(non_object_raw), cache)
    constraint_mask = load_binary_mask(Path(constraint_raw), cache)
    raw_counts = row.get("counts")
    counts = raw_counts if isinstance(raw_counts, dict) else {}
    return non_object_mask, constraint_mask, {
        "state": "ok",
        "non_object_owned_mask_path": non_object_raw if isinstance(non_object_raw, str) else None,
        "constraint_eligible_entity_mask_path": constraint_raw if isinstance(constraint_raw, str) else None,
        "visible_object_owned_mask_path": visible_object_raw if isinstance(visible_object_raw, str) else None,
        "non_object_owned_px": int(counts.get("non_object_owned_px", int(non_object_mask.sum()) if non_object_mask is not None else 0)),
        "visible_object_owned_px": int(counts.get("visible_object_owned_px", 0)),
        "constraint_eligible_entity_px": int(counts.get("constraint_eligible_entity_px", int(constraint_mask.sum()) if constraint_mask is not None else 0)),
    }


def hand_depth_shift_prior_for_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"state": "missing_hand_depth_shift_prior_row", "camera_z_shift_m": 0.0, "weight": None}
    state = str(row.get("state") or "active_hand_depth_shift_prior")
    try:
        shift = float(row.get("camera_z_shift_m", 0.0) or 0.0)
    except Exception:
        shift = 0.0
    raw_weight = row.get("weight")
    weight: float | None
    if raw_weight in (None, ""):
        weight = None
    else:
        try:
            weight = float(raw_weight)
        except Exception as exc:
            raise ValueError(f"hand_depth_shift_prior row has invalid weight {raw_weight!r}: {row}") from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"hand_depth_shift_prior row has invalid nonnegative finite weight {raw_weight!r}: {row}")
    return {"state": state, "camera_z_shift_m": shift, "weight": weight}


def hand_observation_visibility_for_row(row: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"state": "missing_hand_observation_visibility_row", "candidate_px": 0, "weight_multiplier": 1.0}
    state = str(row.get("state") or row.get("visibility_state") or "active_hand_observation_visibility")
    candidate_px = int(row.get("candidate_px") or row.get("candidate_non_object_owned_px") or 0)
    raw_mult = row.get("joint_observation_weight_multiplier", row.get("weight_multiplier", args.occluded_joint_observation_weight))
    try:
        multiplier = float(raw_mult)
    except Exception:
        multiplier = float(args.occluded_joint_observation_weight)
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    return {"state": state, "candidate_px": candidate_px, "weight_multiplier": multiplier}


def contact_patch_for_row(row: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "state": "missing_contact_patch_row",
            "weight": 0.0,
            "band_m": float(args.contact_patch_band_m),
            "target_margin_m": float(args.contact_patch_target_margin_m),
            "support_uncertainty_m": float(args.contact_patch_support_uncertainty_m),
            "support_uncertainty_source": "default_arg_missing_contact_patch_row",
            "local_patch_support_state": None,
            "local_patch_support_consumed": False,
            "local_patch_support_uncertainty_m": None,
            "global_object_support_uncertainty_m": None,
            "local_patch_sample_count": 0,
            "local_patch_temporal_sample_count": 0,
            "max_vertices": int(args.max_contact_patch_vertices),
            "prior_probability": 0.0,
            "contact_anchor_state": None,
            "contact_anchor_residual_allowed": False,
            "contact_anchor_blockers": [],
            "contact_pose_anchor_key": None,
        }
    state = str(row.get("state") or "active_contact_patch")
    try:
        raw_weight = float(row.get("weight", args.contact_patch_weight) or 0.0)
    except Exception:
        raw_weight = float(args.contact_patch_weight)
    try:
        base_weight = float(row.get("contact_patch_base_weight", row.get("base_weight", raw_weight)) or raw_weight)
    except Exception:
        base_weight = raw_weight
    weight = base_weight if bool(args.optimize_contact_state) else raw_weight
    try:
        band_m = float(row.get("contact_patch_band_m", row.get("band_m", args.contact_patch_band_m)) or args.contact_patch_band_m)
    except Exception:
        band_m = float(args.contact_patch_band_m)
    try:
        target_margin_m = float(row.get("contact_patch_target_margin_m", row.get("target_margin_m", args.contact_patch_target_margin_m)) or args.contact_patch_target_margin_m)
    except Exception:
        target_margin_m = float(args.contact_patch_target_margin_m)
    try:
        max_vertices = int(row.get("max_vertices", args.max_contact_patch_vertices) or args.max_contact_patch_vertices)
    except Exception:
        max_vertices = int(args.max_contact_patch_vertices)
    try:
        support_uncertainty_m = float(row.get("object_support_uncertainty_m", row.get("contact_patch_support_uncertainty_m", row.get("support_uncertainty_m", args.contact_patch_support_uncertainty_m))) or args.contact_patch_support_uncertainty_m)
    except Exception:
        support_uncertainty_m = float(args.contact_patch_support_uncertainty_m)
    raw_prior = row.get("contact_state_prior_probability", row.get("contact_patch_prior_probability", row.get("latent_contact_confidence", None)))
    try:
        prior_probability = float(raw_prior) if raw_prior is not None else (1.0 if max(0.0, weight) > 0.0 and state == "active_contact_patch" else 0.0)
    except Exception:
        prior_probability = 1.0 if max(0.0, weight) > 0.0 and state == "active_contact_patch" else 0.0
    def optional_float(raw: Any) -> float | None:
        try:
            val = float(raw)
        except Exception:
            return None
        return val if np.isfinite(val) else None

    blockers = row.get("contact_anchor_blockers")
    local_unc = optional_float(row.get("local_patch_support_uncertainty_m"))
    global_unc = optional_float(row.get("global_object_support_uncertainty_m"))
    try:
        local_count = int(row.get("local_patch_sample_count", 0) or 0)
    except Exception:
        local_count = 0
    try:
        temporal_count = int(row.get("local_patch_temporal_sample_count", 0) or 0)
    except Exception:
        temporal_count = 0
    return {
        "state": state,
        "weight": max(0.0, weight),
        "prior_probability": float(np.clip(prior_probability, 0.0, 1.0)),
        "band_m": max(0.0, band_m),
        "target_margin_m": max(0.0, target_margin_m),
        "support_uncertainty_m": max(0.0, support_uncertainty_m),
        "support_uncertainty_source": row.get("contact_patch_support_uncertainty_source") if isinstance(row.get("contact_patch_support_uncertainty_source"), str) else None,
        "local_patch_support_state": row.get("local_patch_support_state") if isinstance(row.get("local_patch_support_state"), str) else None,
        "local_patch_support_consumed": bool(row.get("local_patch_support_consumed")),
        "local_patch_support_uncertainty_m": local_unc,
        "global_object_support_uncertainty_m": global_unc,
        "local_patch_sample_count": max(0, local_count),
        "local_patch_temporal_sample_count": max(0, temporal_count),
        "max_vertices": max(0, max_vertices),
        "contact_anchor_state": row.get("contact_anchor_state"),
        "contact_anchor_residual_allowed": bool(row.get("contact_anchor_residual_allowed")),
        "contact_anchor_blockers": blockers if isinstance(blockers, list) else [],
        "contact_pose_anchor_key": row.get("contact_pose_anchor_key") if isinstance(row.get("contact_pose_anchor_key"), str) else None,
    }


def visible_ownership_quarantine_faces(
    *,
    frame: dict[str, Any],
    side: str,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    object_pose: tuple[np.ndarray, np.ndarray],
    face_strict_observed: np.ndarray,
    non_object_owned_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    if non_object_owned_mask is None or not np.any(strict):
        return strict, 0
    r_obj, t_obj = object_pose
    tri = object_vertices[object_faces]
    # A face-center-only mask test misses thin hand/object boundary ownership
    # evidence whenever the triangle covers non-object-owned pixels but its
    # center projects outside that small region.  Use a small fixed support set
    # that is still category-agnostic: vertices, edge midpoints, and center.
    samples_obj = np.stack(
        [
            tri[:, 0],
            tri[:, 1],
            tri[:, 2],
            0.5 * (tri[:, 0] + tri[:, 1]),
            0.5 * (tri[:, 1] + tri[:, 2]),
            0.5 * (tri[:, 2] + tri[:, 0]),
            tri.mean(axis=1),
        ],
        axis=1,
    )
    samples_world = samples_obj.reshape(-1, 3) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv = project_world(samples_world, frame, side)
    inside_flat = mask_membership(non_object_owned_mask, uv, int(args.visible_ownership_face_overlap_dilation_px))
    if inside_flat.shape[0] != samples_world.shape[0]:
        return strict, 0
    inside = inside_flat.reshape(len(object_faces), -1).any(axis=1)
    q = strict & inside
    strict[q] = False
    return strict, int(np.count_nonzero(q))


def mask_membership(mask: np.ndarray, uv: np.ndarray | None, dilation_px: int = 0) -> np.ndarray:
    if uv is None:
        return np.zeros((0,), dtype=bool)
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return np.zeros((len(uv),), dtype=bool)
    height, width = mask.shape
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside = np.zeros((len(uv),), dtype=bool)
    if not np.any(valid):
        return inside
    radius = max(0, int(dilation_px))
    x0 = np.clip(u[valid] - radius, 0, width - 1)
    x1 = np.clip(u[valid] + radius, 0, width - 1)
    y0 = np.clip(v[valid] - radius, 0, height - 1)
    y1 = np.clip(v[valid] + radius, 0, height - 1)
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    area = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
    inside[np.where(valid)[0]] = area > 0
    return inside


def visible_object_mask_face_gate(
    *,
    frame: dict[str, Any],
    side: str,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    object_pose: tuple[np.ndarray, np.ndarray],
    face_strict_observed: np.ndarray,
    mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int]:
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    raw_count = int(np.count_nonzero(strict))
    if mask is None or raw_count == 0:
        return strict, raw_count, raw_count
    if bool(args.visible_mask_quarantine_signed_mesh):
        strict[:] = False
        return strict, raw_count, 0
    if not bool(args.visible_object_mask_gate):
        return strict, raw_count, raw_count
    r_obj, t_obj = object_pose
    face_centers_world = object_vertices[object_faces].mean(axis=1) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv = project_world(face_centers_world, frame, side)
    inside = mask_membership(mask, uv, int(args.visible_object_mask_dilation_px))
    if inside.shape[0] == strict.shape[0]:
        strict &= inside
    return strict, raw_count, int(np.count_nonzero(strict))


def visible_surface_depth_order_constraints(
    *,
    frame: dict[str, Any],
    side: str,
    vertices_world: np.ndarray,
    mask: np.ndarray | None,
    depth_row: dict[str, Any] | None,
    args: argparse.Namespace,
    enabled: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if enabled is None:
        enabled = bool(args.visible_surface_depth_order_term)
    if mask is None or depth_row is None or not bool(enabled):
        empty = np.zeros((0,), dtype=float)
        return np.zeros((0,), dtype=np.int64), empty, empty, {
            "finite_inside_count": 0,
            "hand_behind_observed_surface_count": 0,
            "hand_in_front_of_observed_surface_count": 0,
            "hand_near_observed_surface_depth_count": 0,
            "depth_delta_hand_minus_surface_m": numeric_summary(empty),
        }
    uv = project_world(vertices_world, frame, side)
    if uv is None:
        empty = np.zeros((0,), dtype=float)
        return np.zeros((0,), dtype=np.int64), empty, empty, {
            "finite_inside_count": 0,
            "hand_behind_observed_surface_count": 0,
            "hand_in_front_of_observed_surface_count": 0,
            "hand_near_observed_surface_depth_count": 0,
            "depth_delta_hand_minus_surface_m": numeric_summary(empty),
        }
    depth = np.asarray(depth_row.get("depth"), dtype=np.float32)
    if depth.ndim != 2:
        empty = np.zeros((0,), dtype=float)
        return np.zeros((0,), dtype=np.int64), empty, empty, {
            "finite_inside_count": 0,
            "hand_behind_observed_surface_count": 0,
            "hand_in_front_of_observed_surface_count": 0,
            "hand_near_observed_surface_depth_count": 0,
            "depth_delta_hand_minus_surface_m": numeric_summary(empty),
        }
    height, width = depth.shape
    inside = mask_membership(mask, uv, int(args.visible_object_mask_dilation_px))
    cam = world_to_camera(vertices_world, frame)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = inside & (cam[:, 2] > 1.0e-5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if np.any(valid):
        z_surface_all = depth[v[valid], u[valid]].astype(float)
        valid_ids = np.where(valid)[0]
        finite = np.isfinite(z_surface_all) & (z_surface_all > 1.0e-5)
        valid_ids = valid_ids[finite]
        z_surface_all = z_surface_all[finite]
    else:
        valid_ids = np.zeros((0,), dtype=np.int64)
        z_surface_all = np.zeros((0,), dtype=float)
    if valid_ids.size == 0:
        empty = np.zeros((0,), dtype=float)
        return np.zeros((0,), dtype=np.int64), empty, empty, {
            "finite_inside_count": 0,
            "hand_behind_observed_surface_count": 0,
            "hand_in_front_of_observed_surface_count": 0,
            "hand_near_observed_surface_depth_count": 0,
            "depth_delta_hand_minus_surface_m": numeric_summary(empty),
        }
    delta = cam[valid_ids, 2].astype(float) - z_surface_all
    margin = float(args.visible_surface_depth_order_margin_m)
    measure = {
        "finite_inside_count": int(valid_ids.size),
        "hand_behind_observed_surface_count": int(np.count_nonzero(delta > margin)),
        "hand_in_front_of_observed_surface_count": int(np.count_nonzero(delta < -margin)),
        "hand_near_observed_surface_depth_count": int(np.count_nonzero(np.abs(delta) <= margin)),
        "depth_delta_hand_minus_surface_m": numeric_summary(delta),
    }
    violation = np.maximum(0.0, (z_surface_all - margin) - cam[valid_ids, 2].astype(float))
    order = np.argsort(violation)[::-1]
    cap = max(0, int(args.max_visible_surface_depth_vertices))
    if cap and len(order) > cap:
        order = order[:cap]
    selected_ids = valid_ids[order].astype(np.int64)
    selected_surface_depth = z_surface_all[order].astype(float)
    selected_delta = delta[order].astype(float)
    return selected_ids, selected_surface_depth, selected_delta, measure


def sample_ids(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def joint_visibility_from_metric_depth(
    row_frame: dict[str, Any],
    side: str,
    joints_world: np.ndarray,
    depth_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-openpose-joint observation weights from first-surface depth.

    A joint close to the observed depth surface keeps full HaWoR anchoring.  A
    joint behind the observed first surface is treated as occluded by foreground
    geometry; a joint far in front of the observed depth is depth-inconsistent.
    Both cases reduce the zero-state observation force without removing temporal
    coherence or object nonpenetration.
    """
    weights = np.ones((21,), dtype=float)
    residual = np.full((21,), np.nan, dtype=float)
    if depth_row is None:
        weights[:] = float(args.invalid_joint_observation_weight)
        return weights, residual
    uv = project_world(joints_world, row_frame, side)
    if uv is None:
        weights[:] = float(args.invalid_joint_observation_weight)
        return weights, residual
    depth = np.asarray(depth_row.get("depth"), dtype=np.float32)
    if depth.ndim != 2:
        weights[:] = float(args.invalid_joint_observation_weight)
        return weights, residual
    height, width = depth.shape
    cam = world_to_camera(joints_world, row_frame)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (cam[:, 2] > 1.0e-5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_depth = np.zeros((21,), dtype=bool)
    valid_depth[valid] = np.isfinite(depth[v[valid], u[valid]]) & (depth[v[valid], u[valid]] > 1.0e-5)
    valid = valid & valid_depth
    weights[~valid] = float(args.invalid_joint_observation_weight)
    if np.any(valid):
        residual[valid] = cam[valid, 2] - depth[v[valid], u[valid]].astype(float)
        margin = float(args.visible_joint_depth_margin_m)
        behind = valid & (residual > margin)
        in_front = valid & (residual < -margin)
        weights[behind] = float(args.occluded_joint_observation_weight)
        weights[in_front] = float(args.front_inconsistent_joint_observation_weight)
    return np.clip(weights, 0.0, 1.0), residual


def hand_owned_object_depth_quarantine(
    *,
    frame: dict[str, Any],
    side: str,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    object_pose: tuple[np.ndarray, np.ndarray],
    hand_vertices_world: np.ndarray,
    face_strict_observed: np.ndarray,
    depth_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    """Quarantine object faces whose first-surface depth is plausibly hand-owned.

    Object-depth classification alone assumes a projected first surface belongs
    to the object.  During a grasp, a depth-supported hand surface can occupy the
    same pixels and be in front of, or at the same depth as, the object face. In
    that case the face is an occlusion/ownership uncertainty and should not push
    MANO as a trusted observed-object nonpenetration constraint.
    """
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    if not bool(args.hand_owned_object_depth_quarantine) or depth_row is None or not np.any(strict):
        return strict, 0
    depth = np.asarray(depth_row.get("depth"), dtype=np.float32)
    if depth.ndim != 2:
        return strict, 0
    height, width = depth.shape
    r_obj, t_obj = object_pose
    face_centers_world = object_vertices[object_faces].mean(axis=1) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv_face = project_world(face_centers_world, frame, side)
    uv_hand = project_world(hand_vertices_world, frame, side)
    if uv_face is None or uv_hand is None:
        return strict, 0
    cam_face = world_to_camera(face_centers_world, frame)
    cam_hand = world_to_camera(hand_vertices_world, frame)
    uh = np.rint(uv_hand[:, 0]).astype(int)
    vh = np.rint(uv_hand[:, 1]).astype(int)
    hand_valid = (cam_hand[:, 2] > 1.0e-5) & (uh >= 0) & (uh < width) & (vh >= 0) & (vh < height)
    hand_ids = np.where(hand_valid)[0]
    if hand_ids.size == 0:
        return strict, 0
    uf = np.rint(uv_face[:, 0]).astype(int)
    vf = np.rint(uv_face[:, 1]).astype(int)
    face_valid = strict & (cam_face[:, 2] > 1.0e-5) & (uf >= 0) & (uf < width) & (vf >= 0) & (vf < height)
    if np.any(face_valid):
        face_depth = depth[vf[face_valid], uf[face_valid]].astype(float)
        tmp = np.zeros((len(face_valid),), dtype=bool)
        tmp[face_valid] = np.isfinite(face_depth) & (face_depth > 1.0e-5)
        face_valid = face_valid & tmp
    face_ids = np.where(face_valid)[0]
    if face_ids.size == 0:
        return strict, 0
    face_uv = uv_face[face_ids]
    face_z = cam_face[face_ids, 2]
    face_observed_depth = depth[vf[face_ids], uf[face_ids]].astype(float)
    q = np.zeros((face_ids.size,), dtype=bool)
    radius2 = float(args.hand_owned_quarantine_radius_px) ** 2
    z_margin = float(args.hand_owned_quarantine_depth_margin_m)
    support = float(args.hand_owned_quarantine_hand_depth_support_m)
    for hid in hand_ids.astype(int):
        d2 = np.sum((face_uv - uv_hand[hid]) ** 2, axis=1)
        hand_matches_face_depth = np.abs(float(cam_hand[hid, 2]) - face_observed_depth) <= support
        q |= (d2 <= radius2) & hand_matches_face_depth & (cam_hand[hid, 2] <= face_z - z_margin)
    quarantined_ids = face_ids[q]
    strict[quarantined_ids] = False
    return strict, int(quarantined_ids.size)


def load_models(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    patch_legacy_mano_loader()
    mano_cls = load_wilor_mano_class(args.wilor_root)
    right_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    models: dict[str, Any] = {}
    if "right" in args.sides:
        if not right_path.exists():
            raise FileNotFoundError(f"missing right MANO model: {right_path}")
        models["right"] = mano_cls(model_path=str(right_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    if "left" in args.sides:
        left_path = args.wilor_mano_left
        if left_path is None or not left_path.exists():
            raise FileNotFoundError(f"missing left MANO model: {left_path}")
        left_model = mano_cls(model_path=str(left_path), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        if bool(args.hawor_left_shapedirs_x_fix):
            with torch.no_grad():
                left_model.shapedirs[:, 0, :] *= -1
        models["left"] = left_model
    for m in models.values():
        m.eval()
    return models


OPENPOSE_FINGER_GROUPS = [
    np.arange(1, 5, dtype=np.int64),
    np.arange(5, 9, dtype=np.int64),
    np.arange(9, 13, dtype=np.int64),
    np.arange(13, 17, dtype=np.int64),
    np.arange(17, 21, dtype=np.int64),
]


def infer_pose_joint_finger_groups(model: Any, base_root_mat: torch.Tensor, base_pose_mat: torch.Tensor, betas: torch.Tensor, trans: torch.Tensor) -> np.ndarray:
    """Map each MANO internal pose joint to the output finger group it moves most.

    The WiLoR wrapper returns OpenPose-ordered 21 hand joints, while MANO's 15
    hand_pose rotations use MANO's internal kinematic order.  A small local
    perturbation gives a model-specific mapping without relying on undocumented
    joint-name assumptions.
    """
    device = base_pose_mat.device
    out_groups: list[int] = []
    with torch.no_grad():
        base = model(global_orient=base_root_mat[:1], hand_pose=base_pose_mat[:1], betas=betas[:1], transl=trans[:1], return_verts=True, pose2rot=False)
        base_j = base.joints[0, :21]
        for pose_i in range(15):
            delta = torch.zeros((1, 15, 3), dtype=torch.float32, device=device)
            delta[0, pose_i, 0] = 0.08
            posed = rotvec_to_matrix(delta) @ base_pose_mat[:1]
            hyp = model(global_orient=base_root_mat[:1], hand_pose=posed, betas=betas[:1], transl=trans[:1], return_verts=True, pose2rot=False)
            disp = torch.linalg.norm(hyp.joints[0, :21] - base_j, dim=1).detach().cpu().numpy().astype(float)
            scores = [float(np.mean(disp[g])) for g in OPENPOSE_FINGER_GROUPS]
            out_groups.append(int(np.argmax(scores)))
    return np.asarray(out_groups, dtype=np.int64)


def observed_constraints_for_hand(
    *,
    vertices_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    scene: Any,
    face_strict_observed: np.ndarray,
    frame_idx: int,
    max_constraints: int,
    eps: float,
    support_uncertainty_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    r_obj, t_obj = pose
    vertices_object = inverse_object(vertices_world, r_obj, t_obj)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    hard_eps = float(eps) + max(0.0, float(support_uncertainty_m))
    penetrating = np.where(signed > hard_eps)[0]
    if penetrating.size == 0:
        measure = {
            "frame_idx": frame_idx,
            "penetrating_vertex_count": 0,
            "observed_supported_penetrating_vertex_count": 0,
            "observed_supported_penetration_m": numeric_summary(np.asarray([], dtype=float)),
            "observed_surface_support_uncertainty_m": max(0.0, float(support_uncertainty_m)),
        }
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), measure
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = face_strict_observed[primitive_ids[valid]]
    obs_idx = penetrating[observed]
    obs_depth = signed[obs_idx]
    if len(obs_idx) > int(max_constraints):
        order = np.argsort(obs_depth)[::-1][: int(max_constraints)]
        obs_idx = obs_idx[order]
        obs_depth = obs_depth[order]
    normals_world = np.zeros((0, 3), dtype=float)
    depths = np.zeros((0,), dtype=float)
    if len(obs_idx):
        closest_obs = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[obs_idx], dtype=np.float32)))["points"].numpy().astype(float)
        disp = closest_obs - vertices_object[obs_idx]
        norms = np.linalg.norm(disp, axis=1)
        good = norms > 1.0e-12
        normals_world = object_vec_to_world(disp[good] / norms[good, None], r_obj)
        depths = obs_depth[good]
        obs_idx = obs_idx[good]
    measure = {
        "frame_idx": frame_idx,
        "penetrating_vertex_count": int(penetrating.size),
        "observed_supported_penetrating_vertex_count": int(len(obs_idx)),
        "observed_supported_penetration_m": numeric_summary(depths),
        "observed_surface_support_uncertainty_m": max(0.0, float(support_uncertainty_m)),
    }
    return obs_idx.astype(np.int64), normals_world.astype(float), depths.astype(float), measure


def load_hand_ray_shift_priors(path: Path | None) -> dict[tuple[int, str], float]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[tuple[int, str], float] = {}
    for row in payload.get("rows", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and isinstance(row.get("hand_ray_shift_m"), (int, float)):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = float(row["hand_ray_shift_m"])
    return out


def build_rows(args: argparse.Namespace, side: str) -> tuple[list[FrameHandRow], dict[str, Any], Any]:
    annotations = load_json(args.annotations)
    frames = [f for f in as_list(annotations.get("frames")) if isinstance(f, dict)]
    frames_by_idx = {int(f["frame_idx"]): f for f in frames}
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    validate_completed_mesh_contract(args.completed_mesh, args.completion_report)
    mesh = load_mesh(args.completed_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_obj = vertices_object[faces]
    face_normals_object = np.cross(tri_obj[:, 1] - tri_obj[:, 0], tri_obj[:, 2] - tri_obj[:, 0])
    face_normal_norms = np.linalg.norm(face_normals_object, axis=1)
    valid_face_normals = face_normal_norms > 1.0e-12
    face_normals_object[valid_face_normals] /= face_normal_norms[valid_face_normals, None]
    face_normals_object[~valid_face_normals] = 0.0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_object.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    depth_paths = list(args.depth_npz or [DEFAULT_DEPTH])
    depth_rows = load_depth_sources(depth_paths)
    visible_mask_paths = load_visible_object_mask_paths(args.visible_object_mask_report)
    generic_factor_rows = load_generic_factor_reports(args.factor_report, target_entity_id=str(args.object_id))
    visible_ownership_rows = merge_factor_row_maps("visible_ownership", load_visible_ownership_rows(args.visible_ownership_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows["visible_ownership"])
    surface_eligibility_rows = merge_factor_row_maps("surface_eligibility", load_surface_eligibility_rows(args.surface_eligibility_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows["surface_eligibility"])
    visible_surface_track_rows = merge_factor_row_maps("visible_surface_track", load_visible_surface_track_rows(args.visible_surface_track_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows["visible_surface_track"])
    hand_observation_visibility_rows = generic_factor_rows["hand_observation_visibility"]
    hand_depth_shift_prior_rows = generic_factor_rows["hand_depth_shift_prior"]
    contact_patch_rows = generic_factor_rows["contact_patch"]
    visible_mask_cache: dict[Path, np.ndarray] = {}
    surface_eligibility_cache: dict[Path, np.ndarray] = {}
    hand_ray_shift_priors = load_hand_ray_shift_priors(args.hand_depth_repair_graph)
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    rows: list[FrameHandRow] = []
    object_depth_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame_idx in range(int(args.start_frame), int(args.end_frame) + 1):
        frame = frames_by_idx.get(frame_idx)
        pose = poses.get(frame_idx)
        if frame is None or pose is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_frame_or_object_pose"})
            continue
        vertex_classes, obj_summary = classify_object_vertices_against_depth(
            frame=frame,
            vertices_object=vertices_object,
            pose=pose,
            depth_row=depth_rows.get(frame_idx),
            support_margin_m=float(args.support_margin_m),
            free_space_margin_m=float(args.free_space_margin_m),
        )
        object_depth_summaries.append(obj_summary)
        prov = face_provenance(vertex_classes, faces)
        strict_raw = np.asarray(prov["observed_supported_strict"], dtype=bool)
        hand = None
        for h in as_list(frame.get("hands")):
            if isinstance(h, dict) and str(h.get("hand_side")) == side:
                hand = h
                break
        if hand is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_hand"})
            continue
        arrays = bridge_vertices_and_joints(hand, bridge_cache)
        source_info = source_npz_for_hand(hand)
        if arrays is None or source_info is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_bridge_or_source"})
            continue
        current_vertices, current_joints = arrays
        strict, hand_owned_quarantined = hand_owned_object_depth_quarantine(
            frame=frame,
            side=side,
            object_vertices=vertices_object,
            object_faces=faces,
            object_pose=pose,
            hand_vertices_world=current_vertices,
            face_strict_observed=strict_raw,
            depth_row=depth_rows.get(frame_idx),
            args=args,
        )
        ownership_row = visible_ownership_rows.get((frame_idx, side))
        ownership_non_object_mask, ownership_constraint_eligible_mask, ownership_diag = visible_ownership_masks_for_row(ownership_row, visible_mask_cache)
        strict, visible_ownership_quarantined = visible_ownership_quarantine_faces(
            frame=frame,
            side=side,
            object_vertices=vertices_object,
            object_faces=faces,
            object_pose=pose,
            face_strict_observed=strict,
            non_object_owned_mask=ownership_non_object_mask,
            args=args,
        )
        visible_mask_path = visible_mask_paths.get(frame_idx)
        visible_mask = None if visible_mask_path is None else load_binary_mask(visible_mask_path, visible_mask_cache)
        visible_surface_row = visible_surface_track_rows.get((frame_idx, side))
        if (args.visible_surface_track_factor_report is not None or generic_factor_rows["visible_surface_track"]) and visible_surface_row is None:
            raise ValueError(f"visible-surface track factor missing row for frame={frame_idx} side={side}")
        visible_surface_mask, visible_surface_diag = visible_surface_track_mask_for_row(visible_surface_row, visible_mask_cache)
        visible_surface_active = visible_surface_diag.get("state") == "active_visible_surface" and visible_surface_mask is not None
        if visible_surface_active:
            visible_mask = visible_surface_mask
            visible_mask_path = Path(str(visible_surface_diag.get("surface_mask_path")))
        if ownership_constraint_eligible_mask is not None:
            visible_mask = ownership_constraint_eligible_mask if visible_mask is None else (visible_mask & ownership_constraint_eligible_mask)
        strict, visible_mask_face_count_raw, visible_mask_face_count = visible_object_mask_face_gate(
            frame=frame,
            side=side,
            object_vertices=vertices_object,
            object_faces=faces,
            object_pose=pose,
            face_strict_observed=strict,
            mask=visible_mask,
            args=args,
        )
        visible_surface_track_quarantined_face_count = 0
        if visible_surface_active and bool(visible_surface_diag.get("quarantine_hidden_volume", True)):
            visible_surface_track_quarantined_face_count = int(np.count_nonzero(strict))
            strict[:] = False
            visible_mask_face_count = 0
        surface_row = surface_eligibility_rows.get((frame_idx, side))
        surface_mask, surface_diag = surface_eligibility_mask_for_row(surface_row, len(faces), surface_eligibility_cache)
        surface_input_face_count = int(np.count_nonzero(strict))
        if args.surface_eligibility_factor_report is not None and surface_mask is None:
            raise ValueError(f"surface eligibility factor could not supply frame={frame_idx} side={side}: {surface_diag}")
        if surface_mask is not None:
            if str(args.surface_eligibility_mode) == "replace":
                strict = surface_mask.astype(bool)
            else:
                strict = strict & surface_mask.astype(bool)
        surface_eligible_face_count = int(np.count_nonzero(surface_mask)) if surface_mask is not None else 0
        surface_applied_face_delta = int(np.count_nonzero(strict)) - surface_input_face_count
        observed_surface_support_uncertainty_m = float(surface_diag.get("observed_surface_support_uncertainty_m", args.observed_surface_support_uncertainty_m) or 0.0)
        source_path, source_frame = source_info
        source = load_source_arrays(source_cache, source_path)
        required = [
            f"{side}_vertices_world_m",
            f"{side}_joints_world_m",
            f"{side}_root_orient_axis_angle",
            f"{side}_hand_pose_axis_angle",
            f"{side}_betas",
            f"{side}_trans_world_m",
        ]
        missing = [key for key in required if key not in source]
        if missing:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "source_missing_arrays", "missing": missing})
            continue
        raw_vertices = np.asarray(source[f"{side}_vertices_world_m"][source_frame], dtype=float)
        raw_joints = np.asarray(source[f"{side}_joints_world_m"][source_frame], dtype=float)
        scale, rot, sim_trans, _err = similarity_from_to(raw_vertices, current_vertices)
        cidx, normals, depths, measure = observed_constraints_for_hand(
            vertices_world=current_vertices,
            pose=pose,
            scene=scene,
            face_strict_observed=strict,
            frame_idx=frame_idx,
            max_constraints=int(args.max_constraints_per_frame),
            eps=float(args.penetration_epsilon_m),
            support_uncertainty_m=observed_surface_support_uncertainty_m,
        )
        if bool(args.visibility_weighted_hand_observation):
            joint_visibility_weights, joint_depth_residual = joint_visibility_from_metric_depth(frame, side, current_joints, depth_rows.get(frame_idx), args)
        else:
            joint_visibility_weights = np.ones((21,), dtype=float)
            joint_depth_residual = np.full((21,), np.nan, dtype=float)
        hand_visibility_diag = hand_observation_visibility_for_row(hand_observation_visibility_rows.get((frame_idx, side)), args)
        if hand_visibility_diag.get("state") == "active_hand_observation_visibility":
            joint_visibility_weights = np.minimum(joint_visibility_weights, float(hand_visibility_diag.get("weight_multiplier", 1.0)))
        surface_depth_idx, surface_depth_m, surface_depth_delta, surface_depth_measure = visible_surface_depth_order_constraints(
            frame=frame,
            side=side,
            vertices_world=current_vertices,
            mask=visible_mask,
            depth_row=depth_rows.get(frame_idx),
            args=args,
            enabled=bool(args.visible_surface_depth_order_term) or bool(visible_surface_active),
        )
        r_obj, t_obj = pose
        ray_shift = hand_ray_shift_priors.get((frame_idx, side))
        hand_depth_shift_diag = hand_depth_shift_prior_for_row(hand_depth_shift_prior_rows.get((frame_idx, side)))
        contact_patch_diag = contact_patch_for_row(contact_patch_rows.get((frame_idx, side)), args)
        contact_patch_idx = np.zeros((0,), dtype=np.int64)
        contact_patch_targets = np.zeros((0, 3), dtype=float)
        contact_patch_normals = np.zeros((0, 3), dtype=float)
        contact_patch_distances = np.zeros((0,), dtype=float)
        if contact_patch_diag.get("state") == "active_contact_patch" and float(contact_patch_diag.get("weight", 0.0)) > 0.0:
            if bool(args.require_contact_patch_pose_anchor) and not bool(contact_patch_diag.get("contact_anchor_residual_allowed")):
                raise ValueError(
                    "active contact_patch row lacks stable contact-anchor support; refusing to upgrade local visible-surface contact to persistent A_t pose anchor "
                    f"for frame={frame_idx} side={side} state={contact_patch_diag.get('contact_anchor_state')} blockers={contact_patch_diag.get('contact_anchor_blockers')}"
                )
            contact_patch_idx, contact_patch_targets, contact_patch_normals, contact_patch_distances = contact_patch_targets_from_vertices(
                current_vertices,
                scene,
                np.asarray(r_obj, dtype=float),
                np.asarray(t_obj, dtype=float),
                strict.astype(bool),
                face_normals_object.astype(float),
                max_vertices=int(contact_patch_diag.get("max_vertices", args.max_contact_patch_vertices)),
                band_m=float(contact_patch_diag.get("band_m", args.contact_patch_band_m)),
            )
        r_c2w_frame, _t_c2w_frame = frame_camera_pose(frame)
        # V17 hand_ray_shift_m is a camera-ray depth repair observation.  The
        # direction that reduced current observed-surface residual in the
        # workbench probe is the negative camera-z shift.  Generic hand-depth
        # shift factors instead specify camera_z_shift_m directly: positive
        # moves the hand away from the camera, behind the visible first surface.
        ray_prior_world = np.zeros(3, dtype=float)
        ray_prior_source: float | None = None
        ray_prior_weight = float(args.hand_ray_shift_prior_weight)
        if bool(args.use_hand_ray_shift_prior) and ray_shift is not None:
            ray_prior_world = -float(ray_shift) * np.asarray(r_c2w_frame[:, 2], dtype=float)
            ray_prior_source = float(ray_shift)
        if hand_depth_shift_diag.get("state") == "active_hand_depth_shift_prior":
            camera_z_shift = float(hand_depth_shift_diag.get("camera_z_shift_m", 0.0))
            ray_prior_world = camera_z_shift * np.asarray(r_c2w_frame[:, 2], dtype=float)
            ray_prior_source = camera_z_shift
            if hand_depth_shift_diag.get("weight") is not None:
                ray_prior_weight = float(hand_depth_shift_diag["weight"])
        rows.append(
            FrameHandRow(
                frame_idx=frame_idx,
                side=side,
                frame=frame,
                current_vertices_world=current_vertices,
                current_joints_world=current_joints,
                raw_vertices_world=raw_vertices,
                raw_joints_world=raw_joints,
                root_orient_axis_angle=np.asarray(source[f"{side}_root_orient_axis_angle"][source_frame], dtype=float),
                hand_pose_axis_angle=np.asarray(source[f"{side}_hand_pose_axis_angle"][source_frame], dtype=float),
                betas=np.asarray(source[f"{side}_betas"][source_frame], dtype=float),
                trans_world_m=np.asarray(source[f"{side}_trans_world_m"][source_frame], dtype=float),
                similarity_scale=float(scale),
                similarity_rotation_raw_to_current=rot.astype(float),
                similarity_translation_raw_to_current=np.asarray(sim_trans, dtype=float),
                source_hawor_npz=source_path,
                source_frame_index=int(source_frame),
                constraint_indices=cidx,
                constraint_normals_world=normals,
                constraint_depths_m=depths,
                observed_initial_measure=measure,
                observed_constraint_count=int(len(depths)),
                object_rotation_world_from_object=np.asarray(r_obj, dtype=float),
                object_translation_world_m=np.asarray(t_obj, dtype=float),
                face_strict_observed_raw=strict_raw.astype(bool),
                face_strict_observed=strict.astype(bool),
                hand_owned_quarantined_face_count=int(hand_owned_quarantined),
                surface_eligibility_npz_path=surface_diag.get("face_state_npz_path"),
                surface_eligibility_mode=(str(args.surface_eligibility_mode) if surface_mask is not None else None),
                observed_surface_support_uncertainty_m=float(observed_surface_support_uncertainty_m),
                surface_eligible_face_count=int(surface_eligible_face_count),
                surface_input_face_count=int(surface_input_face_count),
                surface_applied_face_delta=int(surface_applied_face_delta),
                visible_ownership_non_object_mask_path=ownership_diag.get("non_object_owned_mask_path"),
                visible_ownership_object_owned_mask_path=ownership_diag.get("visible_object_owned_mask_path"),
                visible_ownership_constraint_eligible_mask_path=ownership_diag.get("constraint_eligible_entity_mask_path"),
                visible_ownership_non_object_owned_px=int(ownership_diag.get("non_object_owned_px", 0)),
                visible_ownership_object_owned_px=int(ownership_diag.get("visible_object_owned_px", 0)),
                visible_ownership_constraint_eligible_px=int(ownership_diag.get("constraint_eligible_entity_px", 0)),
                visible_ownership_quarantined_face_count=int(visible_ownership_quarantined),
                visible_object_mask_path=None if visible_mask_path is None else str(visible_mask_path),
                visible_object_mask_face_count_raw=int(visible_mask_face_count_raw),
                visible_object_mask_face_count=int(visible_mask_face_count),
                visible_surface_track_factor_state=visible_surface_diag.get("state") if (args.visible_surface_track_factor_report is not None or generic_factor_rows["visible_surface_track"]) else None,
                visible_surface_track_mask_path=visible_surface_diag.get("surface_mask_path"),
                visible_surface_track_npz_path=visible_surface_diag.get("visible_surface_npz_path"),
                visible_surface_track_valid_depth_pixels=int(visible_surface_diag.get("valid_depth_pixels", 0) or 0),
                visible_surface_track_quarantined_face_count=int(visible_surface_track_quarantined_face_count),
                visible_surface_depth_order_vertex_indices=surface_depth_idx.astype(np.int64),
                visible_surface_depth_order_depth_m=surface_depth_m.astype(float),
                visible_surface_depth_order_initial_delta_m=surface_depth_delta.astype(float),
                visible_surface_depth_order_initial_measure=surface_depth_measure,
                hand_observation_visibility_factor_state=hand_visibility_diag.get("state") if hand_visibility_diag.get("state") != "missing_hand_observation_visibility_row" else None,
                hand_observation_visibility_candidate_px=int(hand_visibility_diag.get("candidate_px", 0)),
                hand_observation_visibility_weight_multiplier=float(hand_visibility_diag.get("weight_multiplier", 1.0)),
                contact_patch_factor_state=contact_patch_diag.get("state") if contact_patch_diag.get("state") != "missing_contact_patch_row" else None,
                contact_patch_vertex_indices=contact_patch_idx.astype(np.int64),
                contact_patch_target_world_m=contact_patch_targets.astype(float),
                contact_patch_normal_world=contact_patch_normals.astype(float),
                contact_patch_initial_distance_m=contact_patch_distances.astype(float),
                contact_patch_weight=float(contact_patch_diag.get("weight", 0.0)),
                contact_patch_prior_probability=float(contact_patch_diag.get("prior_probability", 0.0)),
                contact_patch_band_m=float(contact_patch_diag.get("band_m", args.contact_patch_band_m)),
                contact_patch_target_margin_m=float(contact_patch_diag.get("target_margin_m", args.contact_patch_target_margin_m)),
                contact_patch_support_uncertainty_m=float(contact_patch_diag.get("support_uncertainty_m", args.contact_patch_support_uncertainty_m)),
                contact_patch_support_uncertainty_source=contact_patch_diag.get("support_uncertainty_source"),
                local_patch_support_state=contact_patch_diag.get("local_patch_support_state"),
                local_patch_support_consumed=bool(contact_patch_diag.get("local_patch_support_consumed")),
                local_patch_support_uncertainty_m=contact_patch_diag.get("local_patch_support_uncertainty_m"),
                global_object_support_uncertainty_m=contact_patch_diag.get("global_object_support_uncertainty_m"),
                local_patch_sample_count=int(contact_patch_diag.get("local_patch_sample_count", 0)),
                local_patch_temporal_sample_count=int(contact_patch_diag.get("local_patch_temporal_sample_count", 0)),
                contact_anchor_state=contact_patch_diag.get("contact_anchor_state"),
                contact_anchor_residual_allowed=bool(contact_patch_diag.get("contact_anchor_residual_allowed")),
                contact_anchor_blockers=list(contact_patch_diag.get("contact_anchor_blockers", [])),
                contact_pose_anchor_key=contact_patch_diag.get("contact_pose_anchor_key"),
                joint_visibility_weights=joint_visibility_weights.astype(float),
                joint_depth_residual_m=joint_depth_residual.astype(float),
                hand_ray_shift_prior_world_m=ray_prior_world.astype(float),
                hand_ray_shift_prior_source_m=ray_prior_source,
                hand_ray_shift_prior_weight=float(ray_prior_weight),
            )
        )
    meta = {
        "side": side,
        "requested_start_frame": int(args.start_frame),
        "requested_end_frame": int(args.end_frame),
        "row_count": int(len(rows)),
        "skipped": skipped,
        "object_depth_summaries": object_depth_summaries[:5],
    }
    return rows, meta, scene


def active_constraints_from_vertices(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, max_constraints: int, eps: float, reference_vertices_world: np.ndarray | None = None, object_translation_delta_world: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    hard_eps = float(eps) + max(0.0, float(row.observed_surface_support_uncertainty_m))
    penetrating = np.where(signed > hard_eps)[0]
    if penetrating.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    idx = penetrating[observed]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    closest_obj = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[idx], dtype=np.float32)))["points"].numpy().astype(float)
    disp = closest_obj - vertices_object[idx]
    norms = np.linalg.norm(disp, axis=1)
    good = norms > 1.0e-12
    idx = idx[good]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    normals_world = object_vec_to_world(disp[good] / norms[good, None], row.object_rotation_world_from_object)
    depths = signed[idx]
    reference = row.current_vertices_world if reference_vertices_world is None else reference_vertices_world
    reference_to_query = vertices_world[idx] - reference[idx] - obj_delta[None, :]
    required = np.sum(normals_world * reference_to_query, axis=1) + depths
    order = np.argsort(required)[::-1]
    if len(order) > int(max_constraints):
        order = order[: int(max_constraints)]
    return idx[order].astype(np.int64), normals_world[order].astype(float), required[order].astype(float)


def contact_patch_targets_from_vertices(
    vertices_world: np.ndarray,
    scene: Any,
    object_rotation_world_from_object: np.ndarray,
    object_translation_world_m: np.ndarray,
    face_strict_observed: np.ndarray,
    face_normals_object: np.ndarray,
    *,
    max_vertices: int,
    band_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(max_vertices) <= 0 or float(band_m) <= 0.0 or not np.any(face_strict_observed):
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    vertices_object = inverse_object(vertices_world, object_rotation_world_from_object, object_translation_world_m)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = np.asarray(face_strict_observed, dtype=bool)[primitive_ids[valid]]
    closest_obj = closest["points"].numpy().astype(float)
    closest_world = closest_obj @ np.asarray(object_rotation_world_from_object, dtype=float).T + np.asarray(object_translation_world_m, dtype=float)[None, :]
    distances = np.linalg.norm(np.asarray(vertices_world, dtype=float) - closest_world, axis=1)
    candidate = observed & np.isfinite(distances) & (distances <= float(band_m))
    idx = np.where(candidate)[0].astype(np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    order = np.argsort(distances[idx])[: int(max_vertices)]
    idx = idx[order]
    primitive = primitive_ids[idx]
    normals_obj = np.asarray(face_normals_object, dtype=float)[primitive]
    normal_norms = np.linalg.norm(normals_obj, axis=1)
    good = normal_norms > 1.0e-9
    idx = idx[good]
    primitive = primitive[good]
    normals_obj = normals_obj[good]
    normal_norms = normal_norms[good]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    normals_obj = normals_obj / normal_norms[:, None]
    normals = object_vec_to_world(normals_obj, object_rotation_world_from_object)
    return idx.astype(np.int64), closest_world[idx].astype(float), normals.astype(float), distances[idx].astype(float)


def dense_observed_surface_constraints_from_vertices(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, reference_vertices_world: np.ndarray | None = None, object_translation_delta_world: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearized observed-surface barrier for every MANO vertex near a trusted face.

    The active set only constrains vertices already known to penetrate.  This
    dense barrier adds tangent-plane inequalities for all vertices whose nearest
    object face is depth-observed.  A vertex currently outside the object gets a
    negative required displacement, so it contributes zero loss unless the
    optimizer moves it inward across that observed surface.  A vertex currently
    inside gets a positive required displacement along the outward normal.
    """
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    idx = np.where(observed)[0].astype(np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    closest_obj = closest["points"].numpy().astype(float)[idx]
    disp = closest_obj - vertices_object[idx]
    norms = np.linalg.norm(disp, axis=1)
    good = norms > 1.0e-12
    idx = idx[good]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    signed_idx = signed[idx]
    normal_obj = disp[good] / norms[good, None]
    outside = signed_idx < 0.0
    normal_obj[outside] *= -1.0
    normals_world = object_vec_to_world(normal_obj, row.object_rotation_world_from_object)
    reference = row.current_vertices_world if reference_vertices_world is None else reference_vertices_world
    reference_to_query = vertices_world[idx] - reference[idx] - obj_delta[None, :]
    required = np.sum(normals_world * reference_to_query, axis=1) + signed_idx
    return idx.astype(np.int64), normals_world.astype(float), required.astype(float)


def merge_constraints(base_idx: np.ndarray, base_normals: np.ndarray, base_depths: np.ndarray, new_idx: np.ndarray, new_normals: np.ndarray, new_depths: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best: dict[int, tuple[np.ndarray, float]] = {}
    for idx, normal, depth in zip(base_idx.astype(int), base_normals, base_depths.astype(float)):
        best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    for idx, normal, depth in zip(new_idx.astype(int), new_normals, new_depths.astype(float)):
        old = best.get(int(idx))
        if old is None or float(depth) > old[1]:
            best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    if not best:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[: int(cap)]
    idx = np.asarray([k for k, _ in items], dtype=np.int64)
    normals = np.stack([v[0] for _, v in items]).astype(float)
    depths = np.asarray([v[1] for _, v in items], dtype=float)
    return idx, normals, depths


def full_observed_surface_measure(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, eps: float, object_translation_delta_world: np.ndarray | None = None, face_strict_observed: np.ndarray | None = None) -> dict[str, Any]:
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    face_mask = row.face_strict_observed if face_strict_observed is None else np.asarray(face_strict_observed, dtype=bool)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    if penetrating.size == 0:
        return {
            "penetrating_vertex_count": 0,
            "observed_supported_penetrating_vertex_count": 0,
            "observed_supported_penetration_m": numeric_summary(np.asarray([], dtype=float)),
        }
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_mask))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = face_mask[primitive_ids[valid]]
    observed_depths = signed[penetrating][observed]
    return {
        "penetrating_vertex_count": int(penetrating.size),
        "observed_supported_penetrating_vertex_count": int(np.count_nonzero(observed)),
        "observed_supported_penetration_m": numeric_summary(observed_depths),
    }


def contact_patch_anchor_coherence(rows: list[FrameHandRow]) -> dict[str, Any]:
    centroids_object: list[np.ndarray] = []
    row_spreads: list[float] = []
    support_uncertainties: list[float] = []
    anchor_states: Counter[str] = Counter()
    allowed_count = 0
    active_count = 0
    for row in rows:
        if row.contact_anchor_state:
            anchor_states[str(row.contact_anchor_state)] += 1
        if row.contact_anchor_residual_allowed:
            allowed_count += 1
        if row.contact_patch_factor_state != "active_contact_patch" or len(row.contact_patch_target_world_m) == 0:
            continue
        active_count += 1
        targets_object = inverse_object(
            row.contact_patch_target_world_m,
            row.object_rotation_world_from_object,
            row.object_translation_world_m,
        )
        centroid = np.median(targets_object, axis=0)
        centroids_object.append(centroid.astype(float))
        spread = np.linalg.norm(targets_object - centroid[None, :], axis=1)
        row_spreads.extend(spread.astype(float).tolist())
        support_uncertainties.append(float(row.contact_patch_support_uncertainty_m))
    if centroids_object:
        centroid_arr = np.asarray(centroids_object, dtype=float)
        median_centroid = np.median(centroid_arr, axis=0)
        centroid_dispersion = np.linalg.norm(centroid_arr - median_centroid[None, :], axis=1)
    else:
        median_centroid = np.zeros((3,), dtype=float)
        centroid_dispersion = np.asarray([], dtype=float)
    centroid_summary = numeric_summary(np.asarray(centroid_dispersion, dtype=float))
    support_summary = numeric_summary(np.asarray(support_uncertainties, dtype=float))
    dispersion_p95 = centroid_summary.get("p95") if isinstance(centroid_summary, dict) else None
    support_p95 = support_summary.get("p95") if isinstance(support_summary, dict) else None
    return {
        "active_contact_patch_rows_with_targets": int(active_count),
        "contact_anchor_state_counts": dict(sorted(anchor_states.items())),
        "contact_anchor_residual_allowed_count": int(allowed_count),
        "object_frame_contact_centroid_median": median_centroid.astype(float).tolist(),
        "object_frame_centroid_dispersion_m": centroid_summary,
        "row_patch_spread_m": numeric_summary(np.asarray(row_spreads, dtype=float)),
        "support_uncertainty_m": support_summary,
        "centroid_dispersion_p95_exceeds_support_p95": bool(
            dispersion_p95 is not None and support_p95 is not None and float(dispersion_p95) > float(support_p95)
        ),
        "claim_scope": "Diagnostic support for a persistent object-frame A_t contact anchor. If stable anchor rows are absent or dispersion exceeds support, local contact must remain a sliding/bounded visible-surface hypothesis rather than a point-anchor force on H_t.",
    }


def optimize_rows(rows: list[FrameHandRow], model: Any, args: argparse.Namespace, device: torch.device, scene: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {"status": "no_rows"}, []
    b = len(rows)
    root = torch.tensor(np.stack([r.root_orient_axis_angle for r in rows]).reshape(b, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r.hand_pose_axis_angle for r in rows]).reshape(b, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r.betas for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r.trans_world_m for r in rows]), dtype=torch.float32, device=device)
    base_root_mat = rotvec_to_matrix(root)
    base_pose_mat = rotvec_to_matrix(pose)
    with torch.no_grad():
        base_out = model(global_orient=base_root_mat, hand_pose=base_pose_mat, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_base_vertices = base_out.vertices.detach().cpu().numpy().astype(float)
        raw_base_joints = base_out.joints.detach().cpu().numpy().astype(float)
    replay_vertex_err = [np.linalg.norm(raw_base_vertices[i] - rows[i].raw_vertices_world, axis=1) for i in range(b)]
    replay_joint_err = [np.linalg.norm(raw_base_joints[i] - rows[i].raw_joints_world, axis=1) for i in range(b)]
    replay_ok = max(float(np.median(e)) for e in replay_vertex_err) <= 1.0e-5 and max(float(np.median(e)) for e in replay_joint_err) <= 1.0e-5

    root_delta = torch.zeros((b, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    pose_delta = torch.zeros((b, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    hand_ray_shift_prior_t = torch.tensor(np.stack([r.hand_ray_shift_prior_world_m for r in rows]), dtype=torch.float32, device=device)
    hand_ray_shift_prior_weight_t = torch.tensor(np.asarray([float(r.hand_ray_shift_prior_weight) for r in rows], dtype=float), dtype=torch.float32, device=device)
    trans_init = hand_ray_shift_prior_t.detach().clone() if bool(args.initialize_hand_ray_shift) else torch.zeros((b, 3), dtype=torch.float32, device=device)
    trans_delta = trans_init.clone().detach().requires_grad_(True)
    translation_support_count_np = np.asarray([len(r.visible_surface_depth_order_vertex_indices) for r in rows], dtype=int)
    if bool(args.freeze_translation_without_visible_surface_support):
        translation_allowed_np = translation_support_count_np > int(args.translation_gate_min_visible_surface_depth_vertices)
    else:
        translation_allowed_np = np.ones((b,), dtype=bool)
    translation_allowed_t = torch.tensor(translation_allowed_np.astype(np.float32), dtype=torch.float32, device=device).reshape(b, 1)
    translation_allowed_bool_t = torch.tensor(translation_allowed_np, dtype=torch.bool, device=device)
    object_trans_delta = torch.zeros((b, 3), dtype=torch.float32, device=device, requires_grad=bool(args.optimize_object_translation))
    contact_prior_np = np.asarray([float(np.clip(r.contact_patch_prior_probability, 0.0, 1.0)) for r in rows], dtype=float)
    contact_geometry_target_np = []
    for r in rows:
        if r.contact_patch_factor_state == "active_contact_patch" and len(r.contact_patch_initial_distance_m):
            deadband = max(float(r.contact_patch_target_margin_m + r.contact_patch_support_uncertainty_m), 1.0e-6)
            distance = float(np.median(np.asarray(r.contact_patch_initial_distance_m, dtype=float)))
            target = float(np.exp(-0.5 * (distance / deadband) ** 2))
        else:
            target = float(np.clip(r.contact_patch_prior_probability, 0.0, 1.0))
        contact_geometry_target_np.append(float(np.clip(target, 1.0e-4, 1.0 - 1.0e-4)))
    contact_geometry_target_np = np.asarray(contact_geometry_target_np, dtype=float)
    contact_prior_t = torch.tensor(contact_prior_np, dtype=torch.float32, device=device)
    contact_geometry_target_t = torch.tensor(contact_geometry_target_np, dtype=torch.float32, device=device)
    contact_logit_init = torch.logit(torch.tensor(np.clip(contact_prior_np, 1.0e-4, 1.0 - 1.0e-4), dtype=torch.float32, device=device))
    contact_logit = contact_logit_init.clone().detach().requires_grad_(bool(args.optimize_contact_state))
    hand_ray_shift_prior_active = (torch.linalg.norm(hand_ray_shift_prior_t, dim=1) > 1.0e-9) & (hand_ray_shift_prior_weight_t > 0.0)
    optim_params = [root_delta, pose_delta, trans_delta]
    if bool(args.optimize_object_translation):
        optim_params.append(object_trans_delta)
    if bool(args.optimize_contact_state):
        optim_params.append(contact_logit)
    optimizer = torch.optim.LBFGS(optim_params, lr=0.35, max_iter=int(args.max_optimizer_iterations), line_search_fn="strong_wolfe")

    current_vertices_t = [torch.tensor(r.current_vertices_world, dtype=torch.float32, device=device) for r in rows]
    current_joints_t = [torch.tensor(r.current_joints_world, dtype=torch.float32, device=device) for r in rows]
    raw_base_vertices_t = torch.tensor(raw_base_vertices, dtype=torch.float32, device=device)
    raw_base_joints_t = torch.tensor(raw_base_joints, dtype=torch.float32, device=device)
    sim_scale_np = np.asarray([r.similarity_scale for r in rows], dtype=float).reshape(b, 1, 1)
    sim_rot_np = np.stack([r.similarity_rotation_raw_to_current for r in rows]).astype(float)
    sim_trans_np = np.stack([r.similarity_translation_raw_to_current for r in rows]).astype(float)
    sim_scale_t = torch.tensor(sim_scale_np, dtype=torch.float32, device=device)
    sim_rot_t = torch.tensor(sim_rot_np, dtype=torch.float32, device=device)
    sim_trans_t = torch.tensor(sim_trans_np, dtype=torch.float32, device=device).reshape(b, 1, 3)
    zero_surface_mode = str(args.zero_surface_mode)
    if zero_surface_mode == "similarity_mapped_raw":
        reference_vertices_np = sim_scale_np * np.matmul(raw_base_vertices, np.transpose(sim_rot_np, (0, 2, 1))) + sim_trans_np[:, None, :]
        reference_joints_np = sim_scale_np * np.matmul(raw_base_joints, np.transpose(sim_rot_np, (0, 2, 1))) + sim_trans_np[:, None, :]
    else:
        reference_vertices_np = np.stack([r.current_vertices_world for r in rows]).astype(float)
        reference_joints_np = np.stack([r.current_joints_world for r in rows]).astype(float)
    reference_vertices_t = [torch.tensor(reference_vertices_np[i], dtype=torch.float32, device=device) for i in range(b)]
    active_constraint_indices: list[np.ndarray] = []
    active_constraint_normals: list[np.ndarray] = []
    active_constraint_depths: list[np.ndarray] = []
    dense_constraint_indices: list[np.ndarray] = []
    dense_constraint_normals: list[np.ndarray] = []
    dense_constraint_depths: list[np.ndarray] = []
    reference_observed_measures: list[dict[str, Any]] = []
    reference_raw_observed_measures: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        c_idx, c_normals, c_depths = active_constraints_from_vertices(
            reference_vertices_np[i], r, scene, int(args.max_constraints_per_frame), float(args.penetration_epsilon_m), reference_vertices_np[i]
        )
        active_constraint_indices.append(c_idx)
        active_constraint_normals.append(c_normals)
        active_constraint_depths.append(c_depths)
        if bool(args.dense_observed_surface_barrier):
            d_idx, d_normals, d_depths = dense_observed_surface_constraints_from_vertices(reference_vertices_np[i], r, scene, reference_vertices_np[i])
        else:
            d_idx = np.zeros((0,), dtype=np.int64)
            d_normals = np.zeros((0, 3), dtype=float)
            d_depths = np.zeros((0,), dtype=float)
        dense_constraint_indices.append(d_idx)
        dense_constraint_normals.append(d_normals)
        dense_constraint_depths.append(d_depths)
        reference_observed_measures.append(full_observed_surface_measure(reference_vertices_np[i], r, scene, float(args.penetration_epsilon_m)))
        reference_raw_observed_measures.append(full_observed_surface_measure(reference_vertices_np[i], r, scene, float(args.penetration_epsilon_m), face_strict_observed=r.face_strict_observed_raw))
    pose_joint_finger_groups = infer_pose_joint_finger_groups(model, base_root_mat, base_pose_mat, betas, trans)
    joint_visibility_weights_np = np.stack([r.joint_visibility_weights for r in rows]).astype(float)
    pose_visibility_weights_np = np.ones((b, 15), dtype=float)
    for pose_i, group_i in enumerate(pose_joint_finger_groups.astype(int)):
        pose_visibility_weights_np[:, pose_i] = np.mean(joint_visibility_weights_np[:, OPENPOSE_FINGER_GROUPS[group_i]], axis=1)
    hand_observation_weight_multiplier_np = np.asarray([float(r.hand_observation_visibility_weight_multiplier) if r.hand_observation_visibility_factor_state == "active_hand_observation_visibility" else 1.0 for r in rows], dtype=float)
    joint_visibility_weights_t = torch.tensor(joint_visibility_weights_np, dtype=torch.float32, device=device)
    pose_visibility_weights_t = torch.tensor(pose_visibility_weights_np, dtype=torch.float32, device=device)
    hand_observation_weight_multiplier_t = torch.tensor(hand_observation_weight_multiplier_np, dtype=torch.float32, device=device)
    intr_t: list[torch.Tensor | None] = []
    base_uv: list[torch.Tensor | None] = []
    r_c2w_t: list[torch.Tensor] = []
    t_c2w_t: list[torch.Tensor] = []
    base_depth: list[torch.Tensor] = []
    visible_surface_depth_order_indices_t: list[torch.Tensor] = []
    visible_surface_depth_order_depth_t: list[torch.Tensor] = []
    visible_surface_depth_order_initial_counts: list[int] = []
    contact_patch_indices_t: list[torch.Tensor] = []
    contact_patch_targets_t: list[torch.Tensor] = []
    contact_patch_normals_t: list[torch.Tensor] = []
    contact_patch_weights_t: list[torch.Tensor] = []
    contact_patch_margins_t: list[torch.Tensor] = []
    contact_patch_support_uncertainty_t: list[torch.Tensor] = []
    for row in rows:
        intr = frame_intrinsics(row.frame, row.side)
        intr_t.append(None if intr is None else torch.tensor(intr, dtype=torch.float32, device=device))
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        base_uv.append(None if uv0 is None else torch.tensor(uv0, dtype=torch.float32, device=device))
        r_c2w, t_c2w = frame_camera_pose(row.frame)
        r_c2w_t.append(torch.tensor(r_c2w, dtype=torch.float32, device=device))
        t_c2w_t.append(torch.tensor(t_c2w, dtype=torch.float32, device=device))
        base_depth.append(torch.tensor(world_to_camera(row.current_joints_world, row.frame)[:, 2], dtype=torch.float32, device=device))
        visible_surface_depth_order_indices_t.append(torch.tensor(row.visible_surface_depth_order_vertex_indices, dtype=torch.long, device=device))
        visible_surface_depth_order_depth_t.append(torch.tensor(row.visible_surface_depth_order_depth_m, dtype=torch.float32, device=device))
        visible_surface_depth_order_initial_counts.append(int((row.visible_surface_depth_order_initial_measure or {}).get("finite_inside_count", 0)))
        contact_patch_indices_t.append(torch.tensor(row.contact_patch_vertex_indices, dtype=torch.long, device=device))
        contact_patch_targets_t.append(torch.tensor(row.contact_patch_target_world_m, dtype=torch.float32, device=device))
        contact_patch_normals_t.append(torch.tensor(row.contact_patch_normal_world, dtype=torch.float32, device=device))
        contact_patch_weights_t.append(torch.tensor(float(row.contact_patch_weight), dtype=torch.float32, device=device))
        contact_patch_margins_t.append(torch.tensor(float(row.contact_patch_target_margin_m), dtype=torch.float32, device=device))
        contact_patch_support_uncertainty_t.append(torch.tensor(float(row.contact_patch_support_uncertainty_m), dtype=torch.float32, device=device))
    contact_active_np = np.asarray([
        bool(r.contact_patch_factor_state == "active_contact_patch" and len(r.contact_patch_vertex_indices) and float(r.contact_patch_weight) > 0.0)
        for r in rows
    ], dtype=bool)
    contact_active_t = torch.tensor(contact_active_np, dtype=torch.bool, device=device)
    contact_weight_np = np.asarray([float(r.contact_patch_weight) for r in rows], dtype=float)
    contact_prior_strength_np = contact_weight_np * float(args.contact_state_prior_residual_scale_m) ** 2
    contact_prior_strength_t = torch.tensor(contact_prior_strength_np, dtype=torch.float32, device=device)
    contact_temporal_pairs: list[tuple[int, int]] = []
    by_side: dict[str, list[tuple[int, int]]] = {}
    for i, r in enumerate(rows):
        by_side.setdefault(str(r.side), []).append((int(r.frame_idx), i))
    for items in by_side.values():
        items = sorted(items)
        for (f0, i0), (f1, i1) in zip(items[:-1], items[1:]):
            if f1 == f0 + 1 and contact_active_np[i0] and contact_active_np[i1]:
                contact_temporal_pairs.append((i0, i1))
    contact_temporal_pairs_t = torch.tensor(contact_temporal_pairs, dtype=torch.long, device=device) if contact_temporal_pairs else torch.zeros((0, 2), dtype=torch.long, device=device)

    def effective_trans_delta() -> torch.Tensor:
        return trans_delta * translation_allowed_t

    def hypothesis() -> tuple[torch.Tensor, torch.Tensor]:
        new_root = rotvec_to_matrix(root_delta) @ base_root_mat
        new_pose = rotvec_to_matrix(pose_delta) @ base_pose_mat
        out = model(global_orient=new_root, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        eff_trans_delta = effective_trans_delta()
        if zero_surface_mode == "similarity_mapped_raw":
            mapped_vertices = sim_scale_t * torch.matmul(out.vertices, sim_rot_t.transpose(1, 2)) + sim_trans_t
            mapped_joints = sim_scale_t * torch.matmul(out.joints, sim_rot_t.transpose(1, 2)) + sim_trans_t
            verts = mapped_vertices + eff_trans_delta[:, None, :]
            joints = mapped_joints + eff_trans_delta[:, None, :]
        else:
            raw_delta_vertices = out.vertices - raw_base_vertices_t
            raw_delta_joints = out.joints - raw_base_joints_t
            mapped_vertices = sim_scale_t * torch.matmul(raw_delta_vertices, sim_rot_t.transpose(1, 2))
            mapped_joints = sim_scale_t * torch.matmul(raw_delta_joints, sim_rot_t.transpose(1, 2))
            verts = torch.stack(current_vertices_t, dim=0) + mapped_vertices + eff_trans_delta[:, None, :]
            joints = torch.stack(current_joints_t, dim=0) + mapped_joints + eff_trans_delta[:, None, :]
        return verts, joints

    def project_torch(points_world: torch.Tensor, i: int) -> torch.Tensor | None:
        intr = intr_t[i]
        if intr is None:
            return None
        cam = torch.matmul(points_world - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
        z = cam[:, 2].clamp_min(1.0e-5)
        fx, fy, cx, cy = intr
        return torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], dim=-1)

    def temporal_terms(x: torch.Tensor, weight: float) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        vel = x[1:] - x[:-1]
        loss = float(weight) * torch.mean(vel * vel)
        if x.shape[0] > 2:
            acc = x[2:] - 2.0 * x[1:-1] + x[:-2]
            loss = loss + float(args.accel_weight) * torch.mean(acc * acc)
        return loss

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        hyp_vertices, hyp_joints = hypothesis()
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        contact_prob = torch.sigmoid(contact_logit) if bool(args.optimize_contact_state) else torch.ones((b,), dtype=torch.float32, device=device)
        if bool(args.optimize_contact_state) and torch.any(contact_active_t):
            contact_delta = contact_prob - contact_prior_t
            active_count = torch.clamp(torch.sum(contact_active_t.to(torch.float32)), min=1.0)
            loss = loss + torch.sum(contact_prior_strength_t[contact_active_t] * contact_delta[contact_active_t] * contact_delta[contact_active_t]) / active_count
            if bool(args.contact_state_geometry_likelihood):
                contact_geometry_delta = contact_prob - contact_geometry_target_t
                loss = loss + torch.sum(contact_prior_strength_t[contact_active_t] * contact_geometry_delta[contact_active_t] * contact_geometry_delta[contact_active_t]) / active_count
            if contact_temporal_pairs_t.numel() > 0:
                i0 = contact_temporal_pairs_t[:, 0]
                i1 = contact_temporal_pairs_t[:, 1]
                pair_strength = 0.5 * (contact_prior_strength_t[i0] + contact_prior_strength_t[i1]) * float(args.contact_state_temporal_strength)
                pair_delta = contact_prob[i1] - contact_prob[i0]
                loss = loss + torch.mean(pair_strength * pair_delta * pair_delta)
        obs_mult = hand_observation_weight_multiplier_t
        eff_trans_delta = effective_trans_delta()
        trans_prior_num = torch.sum(obs_mult[:, None] * eff_trans_delta * eff_trans_delta)
        trans_prior_den = torch.clamp(torch.sum(obs_mult) * 3.0, min=1.0)
        loss = loss + float(args.translation_prior_weight) * trans_prior_num / trans_prior_den
        root_prior_num = torch.sum(obs_mult[:, None, None] * root_delta * root_delta)
        root_prior_den = torch.clamp(torch.sum(obs_mult) * 9.0, min=1.0)
        loss = loss + float(args.root_prior_weight) * root_prior_num / root_prior_den
        pose_prior_num = torch.sum(pose_visibility_weights_t[:, :, None] * pose_delta * pose_delta)
        pose_prior_den = torch.clamp(torch.sum(pose_visibility_weights_t) * 3.0, min=1.0)
        loss = loss + float(args.pose_prior_weight) * pose_prior_num / pose_prior_den
        loss = loss + temporal_terms(eff_trans_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(root_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(pose_delta, float(args.smooth_weight))
        if bool(args.optimize_object_translation):
            loss = loss + float(args.object_translation_prior_weight) * torch.mean(object_trans_delta * object_trans_delta)
            loss = loss + temporal_terms(object_trans_delta, float(args.object_smooth_weight))
        hand_ray_shift_prior_active_supported = hand_ray_shift_prior_active & translation_allowed_bool_t
        if torch.any(hand_ray_shift_prior_active_supported):
            diff = eff_trans_delta[hand_ray_shift_prior_active_supported] - hand_ray_shift_prior_t[hand_ray_shift_prior_active_supported]
            weights = hand_ray_shift_prior_weight_t[hand_ray_shift_prior_active_supported].reshape(-1, 1)
            denom = torch.clamp(torch.tensor(float(diff.numel()), dtype=torch.float32, device=device), min=1.0)
            loss = loss + torch.sum(weights * diff * diff) / denom
        trans_norm = torch.linalg.norm(eff_trans_delta, dim=1)
        object_trans_norm = torch.linalg.norm(object_trans_delta, dim=1)
        root_norm = torch.linalg.norm(root_delta.reshape(b, 3), dim=1)
        pose_norm = torch.linalg.norm(pose_delta, dim=2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(trans_norm - float(args.max_translation_m)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(object_trans_norm - float(args.max_object_translation_m)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(root_norm - float(args.max_root_delta_rad)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(pose_norm - float(args.max_pose_delta_rad)) ** 2)
        for i, row in enumerate(rows):
            if len(active_constraint_indices[i]):
                ids = torch.tensor(active_constraint_indices[i], dtype=torch.long, device=device)
                normals = torch.tensor(active_constraint_normals[i], dtype=torch.float32, device=device)
                depths = torch.tensor(active_constraint_depths[i], dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - reference_vertices_t[i][ids] - object_trans_delta[i].reshape(1, 3)
                residual = torch.relu(depths - float(rows[i].observed_surface_support_uncertainty_m) - torch.sum(normals * moved, dim=1))
                loss = loss + float(args.observed_penetration_weight) * torch.mean(residual * residual)
            if len(dense_constraint_indices[i]):
                ids = torch.tensor(dense_constraint_indices[i], dtype=torch.long, device=device)
                normals = torch.tensor(dense_constraint_normals[i], dtype=torch.float32, device=device)
                depths = torch.tensor(dense_constraint_depths[i], dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - reference_vertices_t[i][ids] - object_trans_delta[i].reshape(1, 3)
                residual = torch.relu(depths - float(rows[i].observed_surface_support_uncertainty_m) - torch.sum(normals * moved, dim=1))
                active_count = torch.clamp(torch.sum((residual > 0.0).to(torch.float32)), min=1.0)
                loss = loss + float(args.dense_observed_penetration_weight) * torch.sum(residual * residual) / active_count
            uv = project_torch(hyp_joints[i], i)
            base_uv_i = base_uv[i]
            obs_w = joint_visibility_weights_t[i]
            obs_den = torch.clamp(torch.sum(obs_w), min=1.0)
            if uv is not None and base_uv_i is not None:
                shift = torch.linalg.norm(uv - base_uv_i, dim=1)
                visible_hinge = torch.relu(shift - float(args.visible_shift_limit_px)) ** 2
                loss = loss + float(args.visible_hinge_weight) * torch.sum(obs_w * visible_hinge) / obs_den
            cam = torch.matmul(hyp_joints[i] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
            depth_shift = torch.abs(cam[:, 2] - base_depth[i])
            depth_hinge = torch.relu(depth_shift - float(args.depth_shift_limit_m)) ** 2
            loss = loss + float(args.depth_hinge_weight) * torch.sum(obs_w * depth_hinge) / obs_den
            visible_depth_order_enabled = bool(args.visible_surface_depth_order_term) or rows[i].visible_surface_track_factor_state == "active_visible_surface"
            if visible_depth_order_enabled and len(visible_surface_depth_order_indices_t[i]):
                ids = visible_surface_depth_order_indices_t[i]
                surface_depth = visible_surface_depth_order_depth_t[i]
                cam_v = torch.matmul(hyp_vertices[i, ids] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
                residual = torch.relu((surface_depth - float(args.visible_surface_depth_order_margin_m)) - cam_v[:, 2])
                active_count = torch.clamp(torch.sum((residual > 0.0).to(torch.float32)), min=1.0)
                loss = loss + float(args.visible_surface_depth_order_weight) * torch.sum(residual * residual) / active_count
            if rows[i].contact_patch_factor_state == "active_contact_patch" and len(contact_patch_indices_t[i]) and float(rows[i].contact_patch_weight) > 0.0:
                ids = contact_patch_indices_t[i]
                targets = contact_patch_targets_t[i] + object_trans_delta[i].reshape(1, 3)
                normals = contact_patch_normals_t[i]
                normal_gap = torch.sum((hyp_vertices[i, ids] - targets) * normals, dim=1)
                deadband = contact_patch_margins_t[i] + contact_patch_support_uncertainty_t[i]
                if str(args.contact_patch_residual_mode) == "support_scaled_attraction":
                    precision_scale = (contact_patch_margins_t[i].clamp_min(1.0e-6) / deadband.clamp_min(1.0e-6)) ** 2
                    residual = normal_gap
                    loss = loss + contact_patch_weights_t[i] * contact_prob[i] * precision_scale * torch.mean(residual * residual)
                else:
                    residual = torch.relu(torch.abs(normal_gap) - deadband)
                    loss = loss + contact_patch_weights_t[i] * contact_prob[i] * torch.mean(residual * residual)
        loss.backward()
        return loss

    active_set_added_counts: list[int] = []
    active_set_closed = False
    active_set_pass_count = 0
    if replay_ok:
        for active_iter in range(max(1, int(args.active_set_iterations))):
            active_set_pass_count = active_iter + 1
            if active_iter > 0:
                optimizer = torch.optim.LBFGS(optim_params, lr=0.25, max_iter=int(args.max_optimizer_iterations), line_search_fn="strong_wolfe")
            optimizer.step(closure)
            with torch.no_grad():
                hyp_vertices_t, _hyp_joints_t = hypothesis()
                hyp_vertices_np = hyp_vertices_t.detach().cpu().numpy().astype(float)
                object_trans_np_active = object_trans_delta.detach().cpu().numpy().astype(float)
            added_total = 0
            for i, row in enumerate(rows):
                new_idx, new_normals, new_depths = active_constraints_from_vertices(
                    hyp_vertices_np[i],
                    row,
                    scene,
                    int(args.max_constraints_per_frame),
                    float(args.penetration_epsilon_m),
                    reference_vertices_np[i],
                    object_trans_np_active[i],
                )
                before = len(active_constraint_indices[i])
                merged = merge_constraints(
                    active_constraint_indices[i],
                    active_constraint_normals[i],
                    active_constraint_depths[i],
                    new_idx,
                    new_normals,
                    new_depths,
                    max(int(args.max_constraints_per_frame), before),
                )
                active_constraint_indices[i], active_constraint_normals[i], active_constraint_depths[i] = merged
                added_total += max(0, len(active_constraint_indices[i]) - before)
                if bool(args.dense_observed_surface_barrier):
                    dense_constraint_indices[i], dense_constraint_normals[i], dense_constraint_depths[i] = dense_observed_surface_constraints_from_vertices(
                        hyp_vertices_np[i], row, scene, reference_vertices_np[i], object_trans_np_active[i]
                    )
            active_set_added_counts.append(int(added_total))
            if added_total == 0:
                active_set_closed = True
                break
    with torch.no_grad():
        hyp_vertices_t, hyp_joints_t = hypothesis()
        hyp_vertices = hyp_vertices_t.detach().cpu().numpy().astype(float)
        hyp_joints = hyp_joints_t.detach().cpu().numpy().astype(float)
        trans_np = effective_trans_delta().detach().cpu().numpy().astype(float)
        latent_trans_np = trans_delta.detach().cpu().numpy().astype(float)
        object_trans_np = object_trans_delta.detach().cpu().numpy().astype(float)
        contact_posterior_np = (torch.sigmoid(contact_logit) if bool(args.optimize_contact_state) else contact_prior_t).detach().cpu().numpy().astype(float)
        root_np = root_delta.detach().cpu().numpy().reshape(b, 3).astype(float)
        pose_np = pose_delta.detach().cpu().numpy().astype(float)

    render_ids = sample_ids(778, int(args.sample_vertex_count_for_render))
    states: list[dict[str, Any]] = []
    initial_obs_max: list[float] = []
    final_linear_residual_max: list[float] = []
    final_full_observed_max: list[float] = []
    final_raw_observed_max: list[float] = []
    visible_max: list[float] = []
    depth_max: list[float] = []
    trans_max: list[float] = []
    object_trans_max: list[float] = []
    root_max: list[float] = []
    pose_max: list[float] = []
    visible_surface_depth_order_selected_count: list[float] = []
    visible_surface_depth_order_initial_in_front_count: list[float] = []
    visible_surface_depth_order_final_in_front_count: list[float] = []
    visible_surface_depth_order_final_delta_min: list[float] = []
    contact_patch_final_abs_normal_gap: list[float] = []
    output_translation_gate_applied_count = 0
    output_translation_gate_shift_norm: list[float] = []
    output_translation_gate_support_count: list[float] = []
    corrected_frames = 0
    for i, row in enumerate(rows):
        if len(active_constraint_indices[i]):
            moved = hyp_vertices[i, active_constraint_indices[i]] - reference_vertices_np[i, active_constraint_indices[i]] - object_trans_np[i][None, :]
            residual = np.maximum(0.0, active_constraint_depths[i] - np.sum(active_constraint_normals[i] * moved, axis=1))
        else:
            residual = np.zeros((0,), dtype=float)
        init_measure = reference_observed_measures[i]
        init_raw_measure = reference_raw_observed_measures[i]
        init_max = float((init_measure.get("observed_supported_penetration_m") or {}).get("max") or 0.0)
        final_max = float(np.max(residual)) if residual.size else 0.0
        full_post = full_observed_surface_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m), object_trans_np[i])
        full_raw_post = full_observed_surface_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m), object_trans_np[i], face_strict_observed=row.face_strict_observed_raw)
        full_post_max = float((full_post.get("observed_supported_penetration_m") or {}).get("max") or 0.0)
        full_raw_post_max = float((full_raw_post.get("observed_supported_penetration_m") or {}).get("max") or 0.0)
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        uv1 = project_world(hyp_joints[i], row.frame, row.side)
        if uv0 is not None and uv1 is not None:
            shift = np.linalg.norm(uv1 - uv0, axis=1)
            shift_max = float(np.max(shift))
            shift_med = float(np.median(shift))
        else:
            shift = np.zeros((0,), dtype=float)
            shift_max = float("nan")
            shift_med = float("nan")
        cam0 = world_to_camera(row.current_joints_world, row.frame)
        cam1 = world_to_camera(hyp_joints[i], row.frame)
        dshift = np.abs(cam1[:, 2] - cam0[:, 2])
        surface_ids = row.visible_surface_depth_order_vertex_indices.astype(int)
        if surface_ids.size:
            cam_v_final = world_to_camera(hyp_vertices[i, surface_ids], row.frame)[:, 2]
            surface_final_delta = cam_v_final.astype(float) - row.visible_surface_depth_order_depth_m.astype(float)
            surface_initial_delta = row.visible_surface_depth_order_initial_delta_m.astype(float)
            surface_initial_in_front = int(np.count_nonzero(surface_initial_delta < -float(args.visible_surface_depth_order_margin_m)))
            surface_final_in_front = int(np.count_nonzero(surface_final_delta < -float(args.visible_surface_depth_order_margin_m)))
            surface_final_summary = numeric_summary(surface_final_delta)
            visible_surface_depth_order_selected_count.append(float(surface_ids.size))
            visible_surface_depth_order_initial_in_front_count.append(float(surface_initial_in_front))
            visible_surface_depth_order_final_in_front_count.append(float(surface_final_in_front))
            visible_surface_depth_order_final_delta_min.append(float(np.min(surface_final_delta)))
            additional_camera_z_to_clear = float(max(0.0, -float(args.visible_surface_depth_order_margin_m) - float(np.min(surface_final_delta))))
        else:
            surface_initial_delta = np.zeros((0,), dtype=float)
            surface_final_delta = np.zeros((0,), dtype=float)
            surface_initial_in_front = 0
            surface_final_in_front = 0
            surface_final_summary = numeric_summary(np.zeros((0,), dtype=float))
            additional_camera_z_to_clear = 0.0
        r_c2w_frame, _t_c2w_frame = frame_camera_pose(row.frame)
        camera_z_axis_world = np.asarray(r_c2w_frame[:, 2], dtype=float)
        optimized_camera_z_shift_m = float(np.dot(trans_np[i], camera_z_axis_world))
        lateral_translation = trans_np[i] - optimized_camera_z_shift_m * camera_z_axis_world
        lateral_norm = float(np.linalg.norm(lateral_translation))
        max_translation = max(0.0, float(args.max_translation_m))
        max_camera_z_inside_translation_bound = math.sqrt(max(0.0, max_translation * max_translation - lateral_norm * lateral_norm))
        translation_bound_remaining_camera_z_m = float(max_camera_z_inside_translation_bound - optimized_camera_z_shift_m)
        if len(row.contact_patch_vertex_indices):
            cp_targets = row.contact_patch_target_world_m + object_trans_np[i][None, :]
            cp_gap = np.sum((hyp_vertices[i, row.contact_patch_vertex_indices.astype(int)] - cp_targets) * row.contact_patch_normal_world, axis=1)
            cp_gap_summary = numeric_summary(cp_gap)
            contact_patch_final_abs_normal_gap.extend(np.abs(cp_gap).astype(float).tolist())
        else:
            cp_gap_summary = numeric_summary(np.zeros((0,), dtype=float))
        tnorm = float(np.linalg.norm(trans_np[i]))
        otnorm = float(np.linalg.norm(object_trans_np[i]))
        rnorm = float(np.linalg.norm(root_np[i]))
        pnorm = float(np.max(np.linalg.norm(pose_np[i], axis=1)))
        changed = tnorm > 1.0e-4 or rnorm > 1.0e-4 or pnorm > 1.0e-4
        if changed:
            corrected_frames += 1
        initial_obs_max.append(init_max)
        final_linear_residual_max.append(final_max)
        final_full_observed_max.append(full_post_max)
        final_raw_observed_max.append(full_raw_post_max)
        if np.isfinite(shift_max):
            visible_max.append(shift_max)
        depth_max.append(float(np.max(dshift)))
        trans_max.append(tnorm)
        object_trans_max.append(otnorm)
        root_max.append(rnorm)
        pose_max.append(pnorm)
        state_joints_world = hyp_joints[i].astype(float).copy()
        state_vertices_world = hyp_vertices[i].astype(float).copy()
        state_translation_world = trans_np[i].astype(float).copy()
        optimizer_translation_support_gate = {
            "enabled": bool(args.freeze_translation_without_visible_surface_support),
            "frozen": bool(args.freeze_translation_without_visible_surface_support) and int(surface_ids.size) <= int(args.translation_gate_min_visible_surface_depth_vertices),
            "min_visible_surface_depth_vertices": int(args.translation_gate_min_visible_surface_depth_vertices),
            "selected_visible_surface_depth_vertices": int(surface_ids.size),
            "policy": "global MANO translation optimized only for rows with visible-surface support above threshold" if bool(args.freeze_translation_without_visible_surface_support) else "global MANO translation optimized normally",
        }
        output_translation_gate = {
            "enabled": bool(args.gate_translation_with_visible_surface_support),
            "applied": False,
            "reason": "disabled" if not bool(args.gate_translation_with_visible_surface_support) else "support_count_above_threshold",
            "min_visible_surface_depth_vertices": int(args.translation_gate_min_visible_surface_depth_vertices),
            "selected_visible_surface_depth_vertices": int(surface_ids.size),
            "articulation_policy": "raw optimizer output",
        }
        if bool(args.gate_translation_with_visible_surface_support) and int(surface_ids.size) <= int(args.translation_gate_min_visible_surface_depth_vertices):
            gate_shift = row.current_joints_world[0].astype(float) - state_joints_world[0].astype(float)
            state_joints_world = state_joints_world + gate_shift[None, :]
            state_vertices_world = state_vertices_world + gate_shift[None, :]
            state_translation_world = state_translation_world + gate_shift
            output_translation_gate_applied_count += 1
            output_translation_gate_shift_norm.append(float(np.linalg.norm(gate_shift)))
            output_translation_gate_support_count.append(float(surface_ids.size))
            output_translation_gate = {
                "enabled": True,
                "applied": True,
                "reason": "visible_surface_support_at_or_below_threshold",
                "min_visible_surface_depth_vertices": int(args.translation_gate_min_visible_surface_depth_vertices),
                "selected_visible_surface_depth_vertices": int(surface_ids.size),
                "baseline_wrist_world_m": row.current_joints_world[0].astype(float).tolist(),
                "raw_optimizer_wrist_world_m": hyp_joints[i, 0].astype(float).tolist(),
                "applied_world_shift_m": gate_shift.astype(float).tolist(),
                "applied_world_shift_norm_m": float(np.linalg.norm(gate_shift)),
                "articulation_policy": "preserve optimized wrist-relative MANO articulation; preserve source HaWoR wrist/root translation",
            }
        state_camera_z_shift_m = float(np.dot(state_translation_world, camera_z_axis_world))
        state_lateral_translation = state_translation_world - state_camera_z_shift_m * camera_z_axis_world
        state_lateral_norm = float(np.linalg.norm(state_lateral_translation))
        state_max_camera_z_inside_translation_bound = math.sqrt(max(0.0, max_translation * max_translation - state_lateral_norm * state_lateral_norm))
        state_translation_bound_remaining_camera_z_m = float(state_max_camera_z_inside_translation_bound - state_camera_z_shift_m)
        states.append(
            {
                "frame_idx": int(row.frame_idx),
                "hand_side": row.side,
                "temporal_mano_state": "joint_continuous_mano_trajectory_correction",
                "source_hawor_npz": str(row.source_hawor_npz),
                "source_frame_index": int(row.source_frame_index),
                "optimized_translation_world_m": state_translation_world.astype(float).tolist(),
                "raw_optimizer_translation_world_m": trans_np[i].astype(float).tolist(),
                "latent_optimizer_translation_world_m": latent_trans_np[i].astype(float).tolist(),
                "optimizer_translation_support_gate": optimizer_translation_support_gate,
                "output_translation_gate": output_translation_gate,
                "hand_ray_shift_prior_translation_world_m": row.hand_ray_shift_prior_world_m.astype(float).tolist(),
                "hand_ray_shift_prior_source_m": row.hand_ray_shift_prior_source_m,
                "hand_ray_shift_prior_weight": float(row.hand_ray_shift_prior_weight),
                "optimized_object_translation_world_m": object_trans_np[i].astype(float).tolist(),
                "joint_visibility_weights": row.joint_visibility_weights.astype(float).tolist(),
                "joint_depth_residual_m": [None if not np.isfinite(x) else float(x) for x in row.joint_depth_residual_m],
                "pose_visibility_weights": pose_visibility_weights_np[i].astype(float).tolist(),
                "optimized_root_delta_axis_angle_rad": root_np[i].astype(float).tolist(),
                "optimized_hand_pose_delta_axis_angle_rad": pose_np[i].reshape(-1).astype(float).tolist(),
                "optimized_joints_world_m": state_joints_world.astype(float).tolist(),
                "optimized_vertices_world_sample_m": state_vertices_world[render_ids].astype(float).tolist(),
                "optimized_vertices_sample_ids": render_ids.astype(int).tolist(),
                "initial_observed_surface_penetration_m": init_measure.get("observed_supported_penetration_m"),
                "initial_raw_observed_surface_penetration_m": init_raw_measure.get("observed_supported_penetration_m"),
                "current_bridge_observed_surface_penetration_m": row.observed_initial_measure.get("observed_supported_penetration_m"),
                "hand_owned_quarantined_face_count": int(row.hand_owned_quarantined_face_count),
                "surface_eligibility_npz_path": row.surface_eligibility_npz_path,
                "surface_eligibility_mode": row.surface_eligibility_mode,
                "observed_surface_support_uncertainty_m": float(row.observed_surface_support_uncertainty_m),
                "surface_eligible_face_count": int(row.surface_eligible_face_count),
                "surface_input_face_count": int(row.surface_input_face_count),
                "surface_applied_face_delta": int(row.surface_applied_face_delta),
                "visible_ownership_non_object_mask_path": row.visible_ownership_non_object_mask_path,
                "visible_ownership_object_owned_mask_path": row.visible_ownership_object_owned_mask_path,
                "visible_ownership_constraint_eligible_mask_path": row.visible_ownership_constraint_eligible_mask_path,
                "visible_ownership_non_object_owned_px": int(row.visible_ownership_non_object_owned_px),
                "visible_ownership_object_owned_px": int(row.visible_ownership_object_owned_px),
                "visible_ownership_constraint_eligible_px": int(row.visible_ownership_constraint_eligible_px),
                "visible_ownership_quarantined_face_count": int(row.visible_ownership_quarantined_face_count),
                "visible_object_mask_path": row.visible_object_mask_path,
                "visible_object_mask_face_count_raw": int(row.visible_object_mask_face_count_raw),
                "visible_object_mask_face_count": int(row.visible_object_mask_face_count),
                "visible_surface_track_factor_state": row.visible_surface_track_factor_state,
                "visible_surface_track_mask_path": row.visible_surface_track_mask_path,
                "visible_surface_track_npz_path": row.visible_surface_track_npz_path,
                "visible_surface_track_valid_depth_pixels": int(row.visible_surface_track_valid_depth_pixels),
                "visible_surface_track_quarantined_face_count": int(row.visible_surface_track_quarantined_face_count),
                "visible_surface_depth_order_initial": row.visible_surface_depth_order_initial_measure,
                "visible_surface_depth_order_selected_vertex_count": int(surface_ids.size),
                "visible_surface_depth_order_selected_vertex_ids": surface_ids.astype(int).tolist(),
                "visible_surface_depth_order_selected_surface_depth_m": row.visible_surface_depth_order_depth_m.astype(float).tolist(),
                "visible_surface_depth_order_selected_initial_delta_hand_minus_surface_m": surface_initial_delta.astype(float).tolist(),
                "visible_surface_depth_order_selected_final_in_front_count": int(surface_final_in_front),
                "visible_surface_depth_order_selected_initial_in_front_count": int(surface_initial_in_front),
                "visible_surface_depth_order_selected_final_delta_hand_minus_surface_m": surface_final_summary,
                "visible_surface_depth_order_selected_final_delta_values_m": surface_final_delta.astype(float).tolist(),
                "visible_surface_depth_order_additional_camera_z_to_clear_selected_m": float(additional_camera_z_to_clear),
                "optimized_translation_camera_z_m": float(state_camera_z_shift_m),
                "optimized_translation_lateral_norm_m": float(state_lateral_norm),
                "translation_bound_remaining_camera_z_m": float(state_translation_bound_remaining_camera_z_m),
                "raw_optimizer_translation_camera_z_m": float(optimized_camera_z_shift_m),
                "raw_optimizer_translation_lateral_norm_m": float(lateral_norm),
                "raw_optimizer_translation_bound_remaining_camera_z_m": float(translation_bound_remaining_camera_z_m),
                "hand_observation_visibility_factor_state": row.hand_observation_visibility_factor_state,
                "hand_observation_visibility_candidate_px": int(row.hand_observation_visibility_candidate_px),
                "hand_observation_visibility_weight_multiplier": float(row.hand_observation_visibility_weight_multiplier),
                "contact_patch_factor_state": row.contact_patch_factor_state,
                "contact_patch_vertex_count": int(len(row.contact_patch_vertex_indices)),
                "contact_patch_vertex_ids": row.contact_patch_vertex_indices.astype(int).tolist(),
                "contact_patch_target_world_m": row.contact_patch_target_world_m.astype(float).tolist(),
                "contact_patch_normal_world": row.contact_patch_normal_world.astype(float).tolist(),
                "contact_patch_initial_distance_values_m": row.contact_patch_initial_distance_m.astype(float).tolist(),
                "contact_patch_initial_distance_m": numeric_summary(row.contact_patch_initial_distance_m),
                "contact_patch_weight": float(row.contact_patch_weight),
                "contact_patch_prior_probability": float(row.contact_patch_prior_probability),
                "contact_patch_geometry_target_probability": float(contact_geometry_target_np[i]),
                "contact_patch_posterior_probability": float(contact_posterior_np[i]),
                "contact_patch_state_optimized": bool(args.optimize_contact_state),
                "contact_patch_band_m": float(row.contact_patch_band_m),
                "contact_patch_target_margin_m": float(row.contact_patch_target_margin_m),
                "contact_patch_support_uncertainty_m": float(row.contact_patch_support_uncertainty_m),
                "contact_patch_support_uncertainty_source": row.contact_patch_support_uncertainty_source,
                "local_patch_support_state": row.local_patch_support_state,
                "local_patch_support_consumed": bool(row.local_patch_support_consumed),
                "local_patch_support_uncertainty_m": row.local_patch_support_uncertainty_m,
                "global_object_support_uncertainty_m": row.global_object_support_uncertainty_m,
                "local_patch_sample_count": int(row.local_patch_sample_count),
                "local_patch_temporal_sample_count": int(row.local_patch_temporal_sample_count),
                "contact_patch_deadband_m": float(row.contact_patch_target_margin_m + row.contact_patch_support_uncertainty_m),
                "contact_patch_residual_mode": str(args.contact_patch_residual_mode),
                "contact_anchor_state": row.contact_anchor_state,
                "contact_anchor_residual_allowed": bool(row.contact_anchor_residual_allowed),
                "contact_anchor_blockers": list(row.contact_anchor_blockers),
                "contact_pose_anchor_key": row.contact_pose_anchor_key,
                "contact_patch_final_normal_gap_m": cp_gap_summary,
                "final_active_constraint_residual_after_solver_m": numeric_summary(residual),
                "full_observed_surface_penetration_after_solver_m": full_post.get("observed_supported_penetration_m"),
                "full_raw_observed_surface_penetration_after_solver_m": full_raw_post.get("observed_supported_penetration_m"),
                "full_observed_supported_penetrating_vertex_count_after_solver": int(full_post.get("observed_supported_penetrating_vertex_count", 0)),
                "visible_joint_shift_px": {"count": int(len(shift)), "median": shift_med, "max": shift_max},
                "joint_camera_depth_shift_m": {"count": int(len(dshift)), "median": float(np.median(dshift)), "max": float(np.max(dshift))},
                "delta_norms": {"translation_m": tnorm, "object_translation_m": otnorm, "root_rad": rnorm, "max_pose_joint_rad": pnorm},
            }
        )
    interval = {
        "hand_side": rows[0].side,
        "start_frame": int(rows[0].frame_idx),
        "end_frame": int(rows[-1].frame_idx),
        "frame_count": int(len(rows)),
        "solver": "joint_root_translation_root_orientation_and_articulation",
        "zero_surface_mode": zero_surface_mode,
        "optimizer_ran": bool(replay_ok),
        "replay_ok": bool(replay_ok),
        "raw_replay_vertex_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in replay_vertex_err], dtype=float)),
        "raw_replay_joint_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in replay_joint_err], dtype=float)),
        "corrected_frame_count": int(corrected_frames),
        "initial_observed_surface_penetration_max_m": numeric_summary(np.asarray(initial_obs_max, dtype=float)),
        "final_active_constraint_residual_after_solver_max_m": numeric_summary(np.asarray(final_linear_residual_max, dtype=float)),
        "full_observed_surface_penetration_after_solver_max_m": numeric_summary(np.asarray(final_full_observed_max, dtype=float)),
        "full_raw_observed_surface_penetration_after_solver_max_m": numeric_summary(np.asarray(final_raw_observed_max, dtype=float)),
        "hand_owned_quarantined_face_count": numeric_summary(np.asarray([r.hand_owned_quarantined_face_count for r in rows], dtype=float)),
        "visible_object_mask_face_count_raw": numeric_summary(np.asarray([r.visible_object_mask_face_count_raw for r in rows], dtype=float)),
        "visible_object_mask_face_count": numeric_summary(np.asarray([r.visible_object_mask_face_count for r in rows], dtype=float)),
        "visible_surface_track_active_row_count": int(sum(r.visible_surface_track_factor_state == "active_visible_surface" for r in rows)),
        "visible_surface_track_quarantined_face_count": numeric_summary(np.asarray([r.visible_surface_track_quarantined_face_count for r in rows], dtype=float)),
        "visible_surface_depth_order_selected_vertex_count": numeric_summary(np.asarray(visible_surface_depth_order_selected_count, dtype=float)),
        "translation_optimization_support_gate_enabled": bool(args.freeze_translation_without_visible_surface_support),
        "translation_optimization_support_gate_min_visible_surface_depth_vertices": int(args.translation_gate_min_visible_surface_depth_vertices),
        "translation_optimization_support_gate_frozen_count": int(np.count_nonzero(~translation_allowed_np)) if bool(args.freeze_translation_without_visible_surface_support) else 0,
        "visible_surface_depth_order_selected_initial_in_front_count": numeric_summary(np.asarray(visible_surface_depth_order_initial_in_front_count, dtype=float)),
        "visible_surface_depth_order_selected_final_in_front_count": numeric_summary(np.asarray(visible_surface_depth_order_final_in_front_count, dtype=float)),
        "visible_surface_depth_order_selected_final_delta_min_m": numeric_summary(np.asarray(visible_surface_depth_order_final_delta_min, dtype=float)),
        "visible_joint_shift_max_px": numeric_summary(np.asarray(visible_max, dtype=float)),
        "joint_camera_depth_shift_max_m": numeric_summary(np.asarray(depth_max, dtype=float)),
        "translation_delta_norm_m": numeric_summary(np.asarray(trans_max, dtype=float)),
        "output_translation_gate_enabled": bool(args.gate_translation_with_visible_surface_support),
        "output_translation_gate_min_visible_surface_depth_vertices": int(args.translation_gate_min_visible_surface_depth_vertices),
        "output_translation_gate_applied_count": int(output_translation_gate_applied_count),
        "output_translation_gate_shift_norm_m": numeric_summary(np.asarray(output_translation_gate_shift_norm, dtype=float)),
        "output_translation_gate_selected_support_count": numeric_summary(np.asarray(output_translation_gate_support_count, dtype=float)),
        "object_translation_delta_norm_m": numeric_summary(np.asarray(object_trans_max, dtype=float)),
        "root_delta_norm_rad": numeric_summary(np.asarray(root_max, dtype=float)),
        "pose_delta_max_joint_norm_rad": numeric_summary(np.asarray(pose_max, dtype=float)),
        "active_set_added_constraint_counts": active_set_added_counts,
        "active_set_pass_count": int(active_set_pass_count),
        "active_set_closed": bool(active_set_closed),
        "active_constraint_count_final": numeric_summary(np.asarray([len(x) for x in active_constraint_indices], dtype=float)),
        "visibility_weighted_hand_observation_enabled": bool(args.visibility_weighted_hand_observation),
        "hand_observation_visibility_factor_active_row_count": int(sum(r.hand_observation_visibility_factor_state == "active_hand_observation_visibility" for r in rows)),
        "hand_observation_visibility_candidate_px": numeric_summary(np.asarray([r.hand_observation_visibility_candidate_px for r in rows], dtype=float)),
        "contact_patch_factor_active_row_count": int(sum(r.contact_patch_factor_state == "active_contact_patch" for r in rows)),
        "contact_patch_state_optimized": bool(args.optimize_contact_state),
        "contact_patch_residual_mode": str(args.contact_patch_residual_mode),
        "contact_patch_prior_probability": numeric_summary(np.asarray([r.contact_patch_prior_probability for r in rows if r.contact_patch_factor_state == "active_contact_patch"], dtype=float)),
        "contact_patch_geometry_target_probability": numeric_summary(contact_geometry_target_np[[i for i, r in enumerate(rows) if r.contact_patch_factor_state == "active_contact_patch"]].astype(float) if any(r.contact_patch_factor_state == "active_contact_patch" for r in rows) else np.asarray([], dtype=float)),
        "contact_patch_posterior_probability": numeric_summary(contact_posterior_np[[i for i, r in enumerate(rows) if r.contact_patch_factor_state == "active_contact_patch"]].astype(float) if any(r.contact_patch_factor_state == "active_contact_patch" for r in rows) else np.asarray([], dtype=float)),
        "contact_patch_posterior_minus_prior": numeric_summary(np.asarray([float(contact_posterior_np[i]) - float(r.contact_patch_prior_probability) for i, r in enumerate(rows) if r.contact_patch_factor_state == "active_contact_patch"], dtype=float)),
        "contact_patch_temporal_pair_count": int(contact_temporal_pairs_t.shape[0]),
        "contact_patch_vertex_count": numeric_summary(np.asarray([len(r.contact_patch_vertex_indices) for r in rows], dtype=float)),
        "contact_patch_initial_distance_m": numeric_summary(np.concatenate([r.contact_patch_initial_distance_m for r in rows if len(r.contact_patch_initial_distance_m)]).astype(float) if any(len(r.contact_patch_initial_distance_m) for r in rows) else np.asarray([], dtype=float)),
        "contact_patch_final_abs_normal_gap_m": numeric_summary(np.asarray(contact_patch_final_abs_normal_gap, dtype=float)),
        "contact_patch_support_uncertainty_m": numeric_summary(np.asarray([r.contact_patch_support_uncertainty_m for r in rows if r.contact_patch_factor_state == "active_contact_patch"], dtype=float)),
        "contact_patch_support_uncertainty_source_counts": dict(Counter(str(r.contact_patch_support_uncertainty_source) for r in rows if r.contact_patch_factor_state == "active_contact_patch")),
        "local_patch_support_state_counts": dict(Counter(str(r.local_patch_support_state) for r in rows if r.contact_patch_factor_state == "active_contact_patch")),
        "local_patch_support_consumed_count": int(sum(bool(r.local_patch_support_consumed) for r in rows if r.contact_patch_factor_state == "active_contact_patch")),
        "local_patch_support_uncertainty_m": numeric_summary(np.asarray([r.local_patch_support_uncertainty_m for r in rows if r.contact_patch_factor_state == "active_contact_patch" and r.local_patch_support_uncertainty_m is not None], dtype=float)),
        "global_object_support_uncertainty_m": numeric_summary(np.asarray([r.global_object_support_uncertainty_m for r in rows if r.contact_patch_factor_state == "active_contact_patch" and r.global_object_support_uncertainty_m is not None], dtype=float)),
        "local_patch_sample_count": numeric_summary(np.asarray([r.local_patch_sample_count for r in rows if r.contact_patch_factor_state == "active_contact_patch"], dtype=float)),
        "local_patch_temporal_sample_count": numeric_summary(np.asarray([r.local_patch_temporal_sample_count for r in rows if r.contact_patch_factor_state == "active_contact_patch"], dtype=float)),
        "contact_patch_anchor_coherence": contact_patch_anchor_coherence(rows),
        "hand_observation_weight_multiplier": numeric_summary(hand_observation_weight_multiplier_np),
        "joint_visibility_weight": numeric_summary(joint_visibility_weights_np.reshape(-1)),
        "pose_visibility_weight": numeric_summary(pose_visibility_weights_np.reshape(-1)),
        "pose_joint_finger_groups": pose_joint_finger_groups.astype(int).tolist(),
        "hand_ray_shift_prior_enabled": bool(args.use_hand_ray_shift_prior),
        "hand_ray_shift_prior_count": int(sum(np.linalg.norm(r.hand_ray_shift_prior_world_m) > 1.0e-9 for r in rows)),
        "object_translation_optimized": bool(args.optimize_object_translation),
        "hand_owned_object_depth_quarantine_enabled": bool(args.hand_owned_object_depth_quarantine),
        "surface_eligibility_factor_enabled": args.surface_eligibility_factor_report is not None or any(r.surface_eligibility_mode is not None for r in rows),
        "surface_eligibility_mode": str(args.surface_eligibility_mode),
        "observed_surface_support_uncertainty_m": numeric_summary(np.asarray([r.observed_surface_support_uncertainty_m for r in rows], dtype=float)),
        "visible_ownership_factor_enabled": args.visible_ownership_factor_report is not None or any(r.visible_ownership_non_object_mask_path is not None or r.visible_ownership_object_owned_mask_path is not None or r.visible_ownership_constraint_eligible_mask_path is not None for r in rows),
        "visible_surface_track_factor_enabled": args.visible_surface_track_factor_report is not None or any(r.visible_surface_track_factor_state is not None for r in rows),
        "visible_object_mask_gate_enabled": bool(args.visible_object_mask_gate),
        "visible_mask_quarantine_signed_mesh_enabled": bool(args.visible_mask_quarantine_signed_mesh),
        "visible_surface_depth_order_term_enabled": bool(args.visible_surface_depth_order_term),
        "visible_object_mask_report": None if args.visible_object_mask_report is None else str(args.visible_object_mask_report),
        "dense_observed_surface_barrier_enabled": bool(args.dense_observed_surface_barrier),
        "dense_observed_constraint_count_final": numeric_summary(np.asarray([len(x) for x in dense_constraint_indices], dtype=float)),
    }
    return interval, states


def reject_rejected_annotation_path(path: Path) -> None:
    raw = str(path)
    hits = [marker for marker in REJECTED_ANNOTATION_PATH_MARKERS if marker in raw]
    if hits:
        raise ValueError(
            "rejected H-prime/final-v7 annotation source supplied to interval MANO solver: "
            f"{path}. Use sanitized non-H-prime annotations under {SANITIZED_ANNOTATION_ROOT}."
        )


def main() -> None:
    args = parse_args()
    reject_rejected_annotation_path(args.annotations)
    visible_surface_factor_supplied = args.visible_surface_track_factor_report is not None or bool(args.factor_report)
    if (bool(args.visible_object_mask_gate) or bool(args.visible_surface_depth_order_term)) and args.visible_object_mask_report is None and not visible_surface_factor_supplied:
        raise ValueError("visible object mask terms require --visible-object-mask-report or a visible_surface_track factor report")
    device = torch.device(args.device)
    models = load_models(args, device)
    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    build_meta: dict[str, Any] = {}
    for side in args.sides:
        rows, meta, scene = build_rows(args, side)
        build_meta[side] = meta
        if not rows:
            intervals.append({"hand_side": side, "start_frame": int(args.start_frame), "end_frame": int(args.end_frame), "frame_count": 0, "solver": "joint_root_translation_root_orientation_and_articulation", "state": "no_rows"})
            continue
        interval, states = optimize_rows(rows, models[side], args, device, scene)
        interval["interval_id"] = f"{side}_{rows[0].frame_idx:04d}_{rows[-1].frame_idx:04d}_joint_mano"
        for st in states:
            st["interval_id"] = interval["interval_id"]
        intervals.append(interval)
        per_frame_states.extend(states)
    report = {
        "method": "solve_v18_joint_mano_interval_trajectory",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": "Continuous interval MANO trajectory correction candidate: root translation, root orientation, and finger articulation optimized jointly against visible/depth compatibility and trusted observed object surface.",
        "inputs": {"annotations": str(args.annotations), "pose_report": str(args.pose_report), "completed_mesh": str(args.completed_mesh), "completion_report": None if args.completion_report is None else str(args.completion_report), "completion_report_completed_mesh_labeled": None if args.completion_report is None else str(completion_report_completed_mesh(args.completion_report)), "depth_npz": [str(p) for p in list(args.depth_npz or [DEFAULT_DEPTH])], "visible_object_mask_report": None if args.visible_object_mask_report is None else str(args.visible_object_mask_report), "visible_ownership_factor_report": None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report), "surface_eligibility_factor_report": None if args.surface_eligibility_factor_report is None else str(args.surface_eligibility_factor_report), "visible_surface_track_factor_report": None if args.visible_surface_track_factor_report is None else str(args.visible_surface_track_factor_report), "factor_report": None if args.factor_report is None else [str(p) for p in args.factor_report]},
        "parameters": {k: ([str(x) for x in v] if k == "factor_report" and v is not None else (str(v) if isinstance(v, Path) else v)) for k, v in vars(args).items() if k not in {"depth_npz"}},
        "build_meta": build_meta,
        "summary": {"interval_count": int(len(intervals)), "per_frame_state_count": int(len(per_frame_states)), "frame_span": [int(args.start_frame), int(args.end_frame)], "sides": list(args.sides)},
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "scientific_test": "If this sequence still looks incoherent after rendering, the remaining failure is not that the optimizer was missing root/articulation coupling; it is a conflict among hand observation, observed tomato geometry, camera/depth alignment, and interval observability.",
    }
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v18_joint_mano_interval_trajectory_state.json"
    write_json(out, report)
    print(json.dumps({"output": str(out), "summary": report["summary"], "intervals": intervals}, indent=2)[:6000])


if __name__ == "__main__":
    main()
