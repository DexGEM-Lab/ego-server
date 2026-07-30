from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from ego_annotation.serving.contracts import (
    DroidCamera,
    DroidCreateSessionRequest,
    DroidFinalizeRequest,
    DroidImageShape,
    DroidSessionOptions,
    ImageSize,
    Ownership,
    PixelTransform,
)
from ego_annotation.serving.droid import DroidAdapter, build_droid_model_config


def _owner(request_id: str, stage: str) -> Ownership:
    return Ownership(request_id, "job", "item", stage, "source")


def _create(request_id: str = "create") -> DroidCreateSessionRequest:
    return DroidCreateSessionRequest(
        ownership=_owner(request_id, "droid.create_session"),
        camera=DroidCamera(
            intrinsics=(60.0, 60.0, 8.0, 8.0),
            source_size=ImageSize(width=16, height=16),
            pixel_transform=PixelTransform.identity(),
        ),
        image_shape=DroidImageShape(16, 16),
        model_revision="droid-test",
        options=DroidSessionOptions(buffer=8),
    )


def _adapter() -> DroidAdapter:
    config = build_droid_model_config(
        weights="server-owned.pth",
        model_revision="droid-test",
        device="cpu",
        max_sessions=2,
        max_buffer_slots=8,
        max_result_journal_entries_per_session=16,
    )

    def session_factory(*_args: object) -> dict[str, object]:
        return {
            "video": SimpleNamespace(counter=SimpleNamespace(value=0)),
            "filter": object(),
            "frontend": object(),
            "backend": MagicMock(),
            "filler": object(),
        }

    return DroidAdapter(
        config,
        backend_factory=lambda _config: object(),
        session_factory=session_factory,
        continuation_fn=lambda *_args: (0, False),
    )


def test_idle_reaper_frees_orphan_resources_and_returns_capacity() -> None:
    adapter = _adapter()
    created = adapter.create_session(_create())
    assert created.session_id is not None
    session_id = created.session_id
    state = adapter._sessions[session_id]

    # The deployment records the timestamp only after it has a successful response
    # to return. Before the TTL nothing is reclaimed; just beyond it, every
    # DepthVideo/backend/filler reference is released through `_mark_terminal`.
    assert adapter.response_sent(session_id, now_monotonic=100.0)
    assert adapter.reap_idle_sessions(now_monotonic=160.0) == 0
    assert adapter.reap_idle_sessions(now_monotonic=160.001) == 1
    assert state.lifecycle.value == "quarantined"
    assert state.video is state.motion_state is state.frontend is state.backend is state.filler is None
    assert adapter.status().active_sessions == 0
    assert session_id in adapter._terminal_lru


def test_idle_reaper_never_takes_an_inflight_session() -> None:
    adapter = _adapter()
    created = adapter.create_session(_create())
    assert created.session_id is not None
    state = adapter._sessions[created.session_id]
    assert adapter.response_sent(created.session_id, now_monotonic=100.0)
    state.in_flight = True
    assert adapter.reap_idle_sessions(now_monotonic=1000.0) == 0
    assert state.lifecycle.value == "open"
    assert state.video is not None


def test_fewer_than_two_keyframes_remains_terminal_unresolved() -> None:
    adapter = _adapter()
    created = adapter.create_session(_create())
    assert created.session_id is not None
    state = adapter._sessions[created.session_id]
    state.video.counter.value = 1
    backend = state.backend

    response = asyncio.run(adapter.finalize(DroidFinalizeRequest(
        ownership=_owner("finalize", "droid.finalize"),
        session_id=created.session_id,
        model_revision="droid-test",
    )))

    assert response.error is not None and response.error.code.value == "unresolved"
    backend.assert_not_called()
    assert state.lifecycle.value == "unresolved"
    assert state.video is None


def test_droid_wait_and_reaper_defaults_are_reported() -> None:
    config = _adapter().config
    assert config.fnet_batch_wait_timeout_s == 0.5
    assert config.idle_session_ttl_s == 60.0
    assert config.idle_reaper_interval_s == 10.0
    assert config.runtime_config_wire()["fnet_batch_wait_timeout_ms"] == 500.0
