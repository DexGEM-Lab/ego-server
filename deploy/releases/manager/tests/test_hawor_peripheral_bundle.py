from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hawor_peripheral_bundle import LEGACY_STAGES, resolve_bundle, write_bundle_plan


def test_default_profile_is_non_intervening(tmp_path: Path) -> None:
    bundle = resolve_bundle(None)
    assert bundle.profile == "api_adapter"
    assert bundle.current_path_unchanged is True
    assert bundle.default_active is True
    assert bundle.stage_order == ()
    path = write_bundle_plan(tmp_path, None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "active_current_path"
    assert payload["bundle"]["switch_point"] == "D5b_hawor_adapter -> D7_hybrid_hand_fusion"


def test_reserved_profile_contains_full_peripheral_order_without_activation(tmp_path: Path) -> None:
    bundle = resolve_bundle("legacy_equivalent_reserved")
    assert bundle.default_active is False
    assert bundle.current_path_unchanged is True
    assert bundle.stage_order == LEGACY_STAGES
    path = write_bundle_plan(tmp_path, "legacy_equivalent_reserved")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "reserved"
    assert payload["activation_rule"].startswith("A future release must implement")


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown HaWoR peripheral profile"):
        resolve_bundle("unexpected")
