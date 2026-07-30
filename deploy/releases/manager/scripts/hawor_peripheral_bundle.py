"""Reserved HaWoR-equivalent peripheral orchestration contract.

The current API path owns model calls and remains the default.  This module
makes the legacy peripheral state transition explicit so a future switch can
replace one bundle between D5b and D7 instead of changing each service adapter.
It is intentionally planning-only until the reserved profile is activated.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

HaworPeripheralProfile = Literal["api_adapter", "legacy_equivalent_reserved"]
DEFAULT_PROFILE: HaworPeripheralProfile = "api_adapter"
BUNDLE_SCHEMA = "v22_hawor_peripheral_bundle.v1"

LEGACY_STAGES = (
    "track_identity_and_bbox_interpolation",
    "native_16_frame_crop_chunks",
    "camera_to_world_wrist_aware_lift",
    "filling_preprocess_lerp_slerp_canonicalization",
    "missing_interval_common_anchor_selection",
    "infiller_temporal_windows",
    "ordered_pred_valid_state_update",
    "mano_replay_and_world_materialization",
)


@dataclass(frozen=True)
class HaworPeripheralBundle:
    profile: HaworPeripheralProfile
    default_active: bool
    intervention: str
    stage_order: tuple[str, ...]
    consumes: tuple[str, ...]
    publishes: tuple[str, ...]
    switch_point: str
    current_path_unchanged: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage_order"] = list(self.stage_order)
        payload["consumes"] = list(self.consumes)
        payload["publishes"] = list(self.publishes)
        return payload


def resolve_bundle(profile: str | None) -> HaworPeripheralBundle:
    selected = profile or DEFAULT_PROFILE
    if selected not in ("api_adapter", "legacy_equivalent_reserved"):
        raise ValueError(f"unknown HaWoR peripheral profile: {selected}")
    if selected == "api_adapter":
        return HaworPeripheralBundle(
            profile="api_adapter",
            default_active=True,
            intervention="none; existing D5b API adapter and D7 fusion remain authoritative",
            stage_order=(),
            consumes=("D6 detector timeline", "D4 shared camera geometry", "D3 metric depth", "D5b API HaWoR output"),
            publishes=("existing hawor_world_hands.npz", "existing D7 hybrid fusion inputs"),
            switch_point="D5b_hawor_adapter -> D7_hybrid_hand_fusion",
            current_path_unchanged=True,
        )
    return HaworPeripheralBundle(
        profile="legacy_equivalent_reserved",
        default_active=False,
        intervention="reserved; no runtime activation in this release",
        stage_order=LEGACY_STAGES,
        consumes=("source RGB timeline", "D6 detector/track evidence", "D4 camera trajectory", "D3 calibration/depth", "HaWoR MANO assets"),
        publishes=("full-timeline track/crop provenance", "legacy-equivalent HaWoR state", "MANO mesh/joints in world frame", "D7-compatible hand archive"),
        switch_point="D5b_hawor_adapter -> D7_hybrid_hand_fusion",
        current_path_unchanged=True,
    )


def write_bundle_plan(run_root: Path, profile: str | None) -> Path:
    bundle = resolve_bundle(profile)
    path = run_root / "state" / "hawor_peripheral_bundle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "status": "reserved" if bundle.profile == "legacy_equivalent_reserved" else "active_current_path",
                "bundle": bundle.as_dict(),
                "activation_rule": "A future release must implement and validate every stage_order item before changing profile semantics; this file alone never changes execution.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
