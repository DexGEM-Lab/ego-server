from __future__ import annotations

import threading
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from ego_annotation.cosmos_semantics import CosmosSemanticError, parse_boundary_output, parse_boundary_output_with_anomalies, parse_coarse_output, parse_coarse_output_with_anomalies, refinement_indices, run_cosmos_semantics, sample_source_indices
from ego_annotation.full_video_timeline import FullVideoDriverConfig, FullVideoTimelineDriver, InMemoryFrameSource
from ego_annotation.scripted.contracts import AlgorithmResult
from ego_annotation.typed_contracts import CosmosOutput


def tagged_rows(state: str = "idle", count: int = 8) -> str:
    contact = "yes" if state == "holding" else "no"
    obj = "cup" if state == "holding" else "none"
    location = "rim" if state == "holding" else "none"
    understanding = "holding cup" if state == "holding" else "hands away from cup"
    return "\n".join(
        f"I={slot}|LI=yes|LC={contact}|LO={obj}|LR=rigid|LA=no|LL={location}|RI=yes|RC=no|RO=none|RR=rigid|RA=no|RL=none|U={understanding}"
        for slot in range(count)
    )


def response_output(request, text: str, *, finish: str = "stop") -> CosmosOutput:
    trace = {"batch_id": "batch", "replica_id": "gpu6", "admitted_monotonic_s": 1.0, "dispatched_monotonic_s": 1.1, "forward_started_monotonic_s": 1.2, "completed_monotonic_s": 1.3, "effective_work_units": 1, "request_count": 1, "forward_count": 1, "model_load_count": 1}
    timings = {"queue_wait_s": 0.1, "prefill_s": 0.1, "time_to_first_token_s": 0.1, "decode_s": 0.1, "e2e_s": 0.4}
    provenance = tuple({"kind": "image", "media_type": media.media_type, "source_index": index, "bytes": len(media.data)} for media, index in zip(request.input.media, request.input.source_frame_indices))
    return CosmosOutput(request.input.ownership, text, finish, None, 10, 5, 15, timings, trace, provenance, "cosmos3-test")


class TaggedClient:
    def __init__(self, first: str | None = None, repair: str | None = None) -> None:
        self.requests = []
        self.first, self.repair = first, repair

    def execute(self, request):
        self.requests.append(request)
        scope = request.input.ownership.scope
        if scope.endswith("coarse:0000") and self.first is not None:
            text = self.first
        elif scope.endswith("coarse:0000:repair:1") and self.repair is not None:
            text = self.repair
        elif ":coarse:" in scope:
            text = tagged_rows(
                "holding" if request.input.source_frame_indices[0] >= 360 else "idle",
                len(request.input.source_frame_indices),
            )
        else:
            text = f"S={len(request.input.source_frame_indices) - 1}|C=0.90|E=cup contact begins"
        return AlgorithmResult.from_request(request, output=response_output(request, text))


def source_24s() -> InMemoryFrameSource:
    return InMemoryFrameSource([np.zeros((2, 3, 3), dtype=np.uint8)] * 720, fps=30.0, source_id="video")


def source_seconds(seconds: int) -> InMemoryFrameSource:
    return InMemoryFrameSource([np.zeros((2, 3, 3), dtype=np.uint8)] * (seconds * 30), fps=30.0, source_id="video")


def source_12s() -> InMemoryFrameSource:
    return source_seconds(12)


def test_keyed_coarse_parser_accepts_exact_lines_and_text_fence() -> None:
    indices = tuple(range(0, 240, 30))
    rows = parse_coarse_output(tagged_rows(), indices, 30.0, image_offset=0)
    fenced = parse_coarse_output(f"```text\n{tagged_rows()}\n```", indices, 30.0, image_offset=0)
    assert [row["frame_idx"] for row in rows] == list(indices)
    assert rows == fenced
    assert rows[0]["left_hand"]["contact"] == "no"


@pytest.mark.parametrize(
    "broken",
    [
        lambda text: text.replace("LC=no", "LC=contact", 1),
        lambda text: text.replace("LO=none", "LO=object", 1),
        lambda text: text.replace("I=0|LI=yes", "LI=yes|I=0", 1),
        lambda text: text.replace("I=0", "I=1", 1),
        lambda text: text.replace("LL=none", "LL=bad|extra", 1),
        lambda text: "\n" + text,
    ],
)
def test_keyed_coarse_rejects_placeholder_order_identity_delimiter_and_extra_lines(broken) -> None:
    with pytest.raises(CosmosSemanticError):
        parse_coarse_output(broken(tagged_rows()), tuple(range(0, 240, 30)), 30.0, image_offset=0)


def test_unsupported_keyed_enums_normalize_unknown_with_exact_provenance_and_no_repair() -> None:
    raw = tagged_rows().replace("LA=no", "LA=black", 1).replace("RA=no", "RA=metallic", 1)
    parsed = parse_coarse_output_with_anomalies(raw, tuple(range(0, 240, 30)), 30.0, image_offset=0)

    assert parsed.rows[0]["left_hand"]["assembly"] == "unknown"
    assert parsed.rows[0]["right_hand"]["assembly"] == "unknown"
    assert parsed.anomalies == (
        {"kind": "unsupported_enum", "raw_field": "LA", "typed_path": "left_hand.assembly", "raw_value": "black", "normalized_value": "unknown", "request_local_slot": 0, "source_frame_idx": 0},
        {"kind": "unsupported_enum", "raw_field": "RA", "typed_path": "right_hand.assembly", "raw_value": "metallic", "normalized_value": "unknown", "request_local_slot": 0, "source_frame_idx": 0},
    )
    result = run_cosmos_semantics(TaggedClient(first=raw), source_12s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    assert result.repair_request_count == 0
    assert result.anomalies[0]["request_scope"] == "cosmos3.reason:coarse:0000"
    assert result.rows[0]["semantic_anomalies"][1]["raw_value"] == "metallic"
    from concurrent.futures import Future
    future = Future()
    future.set_result((result, SimpleNamespace()))
    status, _, _, review, _ = FullVideoTimelineDriver._finish_semantic(future)
    assert status == "completed_with_anomalies"
    assert review["anomaly_count"] == 2
    assert review["anomaly_ledger"][0]["raw_value"] == "black"
    assert review["coarse_rows"][0]["left_hand_assembly"] == "unknown"
    assert review["segments"][0]["left_hand"]["contact"] == "no"
    assert review["semantic_rows"][0]["caption"].startswith("Left hand")
    assert "boundary_refinements" in review


def test_keyed_boundary_maps_local_slot_and_rejects_nonexact_grammar() -> None:
    parsed = parse_boundary_output("S=2|C=0.90|E=cup contact begins", (0, 10, 20))
    assert parsed["change_frame_idx"] == 20
    assert parsed["confidence"] == 0.9
    for bad in ("S=2|C=high|E=x", "S=3|C=0.90|E=x", "C=0.90|S=2|E=x", "S=2|C=0.90|E=bad|extra", "```json\nS=2|C=0.90|E=x\n```"):
        with pytest.raises(CosmosSemanticError):
            parse_boundary_output(bad, (0, 10, 20))


def test_refinement_grammar_exposes_explicit_empty_enum_anomaly_list() -> None:
    parsed = parse_boundary_output_with_anomalies("S=2|C=0.90|E=black assembly label is not a typed field", (0, 10, 20))
    assert parsed.record["change_frame_idx"] == 20
    assert parsed.anomalies == ()

    result = run_cosmos_semantics(TaggedClient(), source_24s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    assert result.refinements[0]["semantic_anomalies"] == []
    assert not [item for item in result.anomalies if item.get("phase") == "boundary"]


@pytest.mark.parametrize(
    ("seconds", "expected_groups"),
    [
        (3, [(0, 30, 60)]),
        (5, [(0, 30, 60, 90, 120)]),
        (8, [tuple(range(0, 240, 30))]),
        (12, [tuple(range(0, 240, 30)), tuple(range(120, 360, 30))]),
    ],
)
def test_coarse_short_video_and_tail_groups_use_only_real_distinct_frames(seconds: int, expected_groups: list[tuple[int, ...]]) -> None:
    client = TaggedClient()
    result = run_cosmos_semantics(client, source_seconds(seconds), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)

    coarse_requests = [request for request in client.requests if ":coarse:" in request.input.ownership.scope]
    assert result.coarse_request_count == len(expected_groups)
    assert [request.input.source_frame_indices for request in coarse_requests] == expected_groups
    assert all(len(request.input.source_frame_indices) <= 8 for request in coarse_requests)
    assert all(len(set(request.input.source_frame_indices)) == len(request.input.source_frame_indices) for request in coarse_requests)
    assert f"Return exactly {len(expected_groups[0])} tagged" in coarse_requests[0].input.prompt


def test_variable_length_coarse_parser_rejects_extra_rows_and_wrong_local_identity() -> None:
    indices = (0, 30, 60)
    with pytest.raises(CosmosSemanticError, match="exactly 3"):
        parse_coarse_output(tagged_rows(count=4), indices, 30.0, image_offset=0)
    with pytest.raises(CosmosSemanticError, match="identity"):
        parse_coarse_output(tagged_rows(count=3).replace("I=2", "I=3"), indices, 30.0, image_offset=0)


def test_1fps_coarse_and_changed_interval_refinement_are_preserved() -> None:
    client = TaggedClient()
    result = run_cosmos_semantics(client, source_24s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    assert sample_source_indices(720, 30.0) == tuple(range(0, 720, 30))
    assert result.coarse_request_count == 3
    assert result.refinement_request_count == 1
    assert [request.input.source_frame_indices for request in client.requests[:3]] == [tuple(range(0, 240, 30)), tuple(range(240, 480, 30)), tuple(range(480, 720, 30))]
    assert client.requests[3].input.source_frame_indices == refinement_indices(450, 480)
    assert result.rows[-1]["end_frame"] == 720
    assert result.rows[0]["caption"] == "Left hand visible_no_contact none; right hand visible_no_contact none."
    assert result.coarse_rows[0]["left_hand_in_frame"] == "yes"
    assert "left_hand" not in result.coarse_rows[0]
    assert result.segments[0]["source"] == "cosmos_gallery_until_boundary"
    boundary = result.refinements[0]
    assert boundary["previous_label"]["left_hand"]["contact"] == "no"
    assert boundary["next_label"]["left_hand"]["contact"] == "yes"
    assert boundary["confidence"] == "high"
    assert boundary["confidence_score"] == 0.9


def test_single_keyed_repair_changes_prompt_scope_not_media_and_records_trace() -> None:
    client = TaggedClient(first=tagged_rows().replace("LC=no", "LC=contact", 1), repair=tagged_rows())
    result = run_cosmos_semantics(client, source_12s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    first, repair = client.requests[:2]
    assert result.repair_request_count == 1
    assert repair.input.ownership.scope == "cosmos3.reason:coarse:0000:repair:1"
    assert [asset.data for asset in first.input.media] == [asset.data for asset in repair.input.media]
    assert repair.input.prompt != first.input.prompt
    assert "LC echoes literal field placeholder" in str(result.attempts[0]["validation_error"])
    assert result.attempts[1]["repair_count"] == 1


def test_single_keyed_repair_fails_closed_after_one_attempt() -> None:
    client = TaggedClient(first=tagged_rows().replace("LC=no", "LC=contact", 1), repair=tagged_rows().replace("LC=no", "LC=contact", 1))
    with pytest.raises(CosmosSemanticError, match="repair failed after one correction") as caught:
        run_cosmos_semantics(client, source_12s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    assert len(caught.value.attempts) == 2
    assert [request.input.ownership.scope for request in client.requests] == ["cosmos3.reason:coarse:0000", "cosmos3.reason:coarse:0000:repair:1"]


def test_nonstop_generation_never_enters_repair() -> None:
    class Truncated(TaggedClient):
        def execute(self, request):
            self.requests.append(request)
            return AlgorithmResult.from_request(request, output=response_output(request, tagged_rows(), finish="length"))
    client = Truncated()
    with pytest.raises(CosmosSemanticError, match="truncated|incomplete"):
        run_cosmos_semantics(client, source_12s(), case_id="case", item_id="item", revision="cosmos3-test", gallery_width=3)
    assert len(client.requests) == 1


def test_driver_starts_cosmos_independently_and_joins_before_return() -> None:
    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = FullVideoDriverConfig(cosmos_enabled=True, require_rgbd_capability=False, allow_monocular_droid_smoke=True)
    started, physical, release = threading.Event(), threading.Event(), threading.Event()
    sentinel = object()
    def fake_cosmos(self, source, case_id, item_id):
        started.set(); assert release.wait(timeout=2); return SimpleNamespace(), SimpleNamespace()
    def fake_physical(self, source, *, case_id, item_id, semantic_future, module_timings_s):
        assert started.wait(timeout=2); physical.set(); release.set(); semantic_future.result(timeout=2); return sentinel
    driver._run_cosmos = MethodType(fake_cosmos, driver)
    driver._run_physical = MethodType(fake_physical, driver)
    assert driver.run(SimpleNamespace(), case_id="case") is sentinel
    assert physical.is_set()
