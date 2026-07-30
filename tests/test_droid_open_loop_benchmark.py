"""Pure accounting tests for the external live-DROID offered-load driver."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ego_annotation.serving.contracts import (
    DroidBatchTrace,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidFinalizeRequest,
    DroidFrameRequest,
    DroidImageShape,
    ErrorCode,
    FrameValidity,
    Ownership,
    ServerIdentity,
    StepStatus,
    TensorPayload,
)
from ego_annotation.serving.transport import parse_multipart_request_fields


_MODULE_PATH = Path(__file__).parents[1] / "benchmarks/ray_serve/benchmark_droid_open_loop.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_droid_open_loop", _MODULE_PATH)
assert _SPEC and _SPEC.loader
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def runtime_identity(replica_id: str, gpu_id: int | None = None) -> ServerIdentity:
    if gpu_id is None:
        gpu_id = 7 if replica_id.endswith("a") else 4
    return ServerIdentity(
        experiment_id="exp", replica_id=replica_id, assigned_gpu=gpu_id, worker_pid=123,
        gcs_address=f"127.0.0.1:{34000 + gpu_id}", http_port=36000 + gpu_id,
        temp_dir=f"/tmp/ego-droid-scaling/exp/gpu{gpu_id}", model_revision="droid-v1",
        checkpoint_digest="checkpoint", schema_version="ego.model-service.v1",
        release_sha="release", release_digest="release", cuda_uuid=f"GPU-{gpu_id}",
        module_root="/releases/release", dependency_digest="source-digest", dependency_root="/sources/source-digest/droid_slam", source_amendment_id="recovered-hawor-droid-core-v1",
    )


def make_endpoint(base_url: str, replica_id: str, gpu_id: int | None = None):
    return bench.ReplicaEndpoint(base_url, replica_id, "droid-v1", runtime_identity(replica_id, gpu_id))


def record(operation: str, outcome: str, latency: float, trace: dict[str, object] | None = None):
    return bench.RequestRecord(
        operation=operation,
        level="sessions-2_waves-per-s-1",
        request_id=f"{operation}-{outcome}-{latency}",
        session_id="session" if operation != "create_session" else None,
        scheduled_s=10.0 if operation == "push_frame" else None,
        sent_s=10.0,
        completed_s=10.0 + latency,
        latency_s=latency,
        http_status=200 if outcome == "completed" else 429,
        outcome=outcome,
        error_code=None if outcome == "completed" else "backpressure",
        error_message=None,
        trace=trace,
    )


def test_summary_keeps_offered_rejected_and_fused_counts_separate() -> None:
    trace = {
        "batch_id": "fused-batch",
        "effective_work_units": 2,
        "fnet_forward_count": 1,
        "session_local_forward_count": 3,
        "model_load_count": 1,
    }
    rows = bench.summarize([
        record("push_frame", "completed", 0.2, trace),
        record("push_frame", "completed", 0.4, trace),
        record("push_frame", "rejected", 0.01),
    ])
    row = rows[0]
    assert row["offered_count"] == 3
    assert row["admitted_count"] == 2
    assert row["completed_count"] == 2
    assert row["rejected_count"] == 1
    assert row["fused_batch_size_distribution"] == {2: 1}
    assert row["fused_forward_count"] == 1
    assert row["session_local_forward_count"] == 6
    assert row["model_load_counts"] == [1]
    assert row["failure_modes"] == {"backpressure": 1}


def test_summary_keeps_server_allocator_high_water_per_level() -> None:
    first = record("push_frame", "completed", 0.1)
    second = record("push_frame", "completed", 0.2)
    first.diagnostics = {"allocator_memory": {"allocated_bytes": 10, "reserved_bytes": 20, "max_allocated_bytes": 30, "max_reserved_bytes": 40}}
    second.diagnostics = {"allocator_memory": {"allocated_bytes": 11, "reserved_bytes": 19, "max_allocated_bytes": 31, "max_reserved_bytes": 39}}
    summary = bench.summarize([first, second])[0]
    assert summary["allocator_memory"] == {
        "allocated_bytes": 11, "reserved_bytes": 20, "max_allocated_bytes": 31, "max_reserved_bytes": 40,
    }


def test_summary_filters_null_diagnostics_and_retains_terminal_finite_pose_ratios() -> None:
    push_with_null_allocator = record("push_frame", "completed", 0.1)
    push_with_null_allocator.diagnostics = {"allocator_memory": None}
    terminal_a = record("finalize", "completed", 0.2)
    terminal_b = record("finalize", "completed", 0.3)
    terminal_a.finite_pose_ratio = 1.0
    terminal_b.finite_pose_ratio = 0.875

    rows = {row["operation"]: row for row in bench.summarize([push_with_null_allocator, terminal_a, terminal_b])}

    assert rows["push_frame"]["allocator_memory"] == {
        "allocated_bytes": None, "reserved_bytes": None, "max_allocated_bytes": None, "max_reserved_bytes": None,
    }
    assert rows["finalize"]["finite_pose_ratio"] == {
        "count": 2, "min": 0.875, "max": 1.0, "values": [1.0, 0.875],
    }


def test_multipart_push_body_carries_binary_mask_metadata() -> None:
    body, content_type = bench.multipart_body(
        {"session_id": "s"},
        {"static_confidence_mask": (b"mask", (2, 2), "float32")},
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="static_confidence_mask"; shape="2,2"; dtype="float32"' in body
    assert b"mask" in body


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        content: bytes | None = None,
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content if content is not None else json.dumps(payload).encode()
        self.headers = {"content-type": content_type}
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeClient:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.urls: list[str] = []

    async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> FakeResponse:
        self.urls.append(url)
        metadata, arrays = parse_multipart_request_fields(content, headers["Content-Type"])
        result = self.handler(url, metadata, arrays)
        if isinstance(result, Exception):
            raise result
        if result._payload is not None and "server_identity" not in result._payload:
            base = url.rsplit("/", 1)[0]
            replica_id = "replica-a" if "replica-a" in base else "replica-b"
            result._payload["server_identity"] = runtime_identity(replica_id).to_wire()
            result.content = json.dumps(result._payload).encode()
            result.text = result.content.decode()
        return result


def typed_error(
    metadata: dict[str, Any], code: ErrorCode, message: str, *, retryable: bool = False, terminal: bool = False,
) -> FakeResponse:
    return FakeResponse(payload={
        "ownership": metadata["ownership"],
        "error": {
            "code": code.value,
            "message": message,
            "retryable": retryable,
            "ownership": metadata["ownership"],
            "batch_id": None,
        },
        "terminal": terminal,
    })


def make_create(request_id: str) -> DroidCreateSessionRequest:
    return DroidCreateSessionRequest(
        ownership=Ownership(request_id, "job", "item", "droid.create_session", "source"),
        camera=DroidCamera.from_mapping(bench.camera_contract()),
        image_shape=DroidImageShape(320, 568),
        model_revision="droid-v1",
    )


def make_push(session_id: str, request_id: str) -> DroidFrameRequest:
    rgb = bytes(320 * 568 * 3)
    return DroidFrameRequest(
        ownership=Ownership(request_id, "job", "item", "droid.push_frame", "source", source_timestamp_s=1.0),
        session_id=session_id,
        frame_id=f"frame-{request_id}",
        source_timestamp_s=1.0,
        rgb=TensorPayload(rgb, (320, 568, 3), "uint8"),
        model_revision="droid-v1",
    )


def make_finalize(session_id: str, request_id: str) -> DroidFinalizeRequest:
    return DroidFinalizeRequest(
        ownership=Ownership(request_id, "job", "item", "droid.finalize", "source"),
        session_id=session_id,
        model_revision="droid-v1",
    )


def test_sticky_two_replica_router_never_moves_session_and_retires_terminal_mapping() -> None:
    create_counts = {"http://replica-a:36007": 0, "http://replica-b:36004": 0}

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        base = url.rsplit("/", 1)[0]
        if url.endswith("create_session"):
            create_counts[base] += 1
            return FakeResponse(payload={
                "ownership": metadata["ownership"],
                "session_id": f"{"a" if "replica-a" in base else "b"}-session-{create_counts[base]}",
                "error": None,
            })
        if url.endswith("push_frame"):
            return typed_error(metadata, ErrorCode.BACKPRESSURE, "busy", retryable=True)
        return typed_error(metadata, ErrorCode.UNRESOLVED, "one keyframe", terminal=True)

    async def scenario() -> None:
        client = FakeClient(handler)
        endpoints = (
            make_endpoint("http://replica-a:36007", "replica-a"),
            make_endpoint("http://replica-b:36004", "replica-b"),
        )
        router = bench.StickyDroidRouter(endpoints)
        first = await router.create_session(client, make_create("create-a"))
        second = await router.create_session(client, make_create("create-b"))
        assert first.response.session_id == "a-session-1"
        assert second.response.session_id == "b-session-1"

        await router.push_frame(client, make_push("a-session-1", "push-a"))
        await router.push_frame(client, make_push("b-session-1", "push-b"))
        assert client.urls[-2].startswith("http://replica-a:36007/")
        assert client.urls[-1].startswith("http://replica-b:36004/")

        terminal = await router.finalize(client, make_finalize("a-session-1", "finalize-a"))
        assert terminal.response.error.code is ErrorCode.UNRESOLVED
        assert client.urls[-1].startswith("http://replica-a:36007/")
        # Terminal sessions leave active affinity but retain a bounded retry route
        # so the server-side terminal journal remains reachable.
        assert await router.endpoint_for_session("a-session-1") == endpoints[0]
        assert await router.active_session_count() == 1
        assert await router.terminal_route_count() == 1
        retry = await router.finalize(client, make_finalize("a-session-1", "finalize-a"))
        assert retry.response.error.code is ErrorCode.UNRESOLVED
        assert client.urls[-1] == "http://replica-a:36007/droid.finalize"
        assert await router.endpoint_for_session("b-session-1") == endpoints[1]

    asyncio.run(scenario())


def test_endpoint_failure_is_explicit_and_does_not_migrate_sticky_session() -> None:
    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse | Exception:
        if url.endswith("create_session"):
            return FakeResponse(payload={
                "ownership": metadata["ownership"], "session_id": "sticky", "error": None,
            })
        if url.startswith("http://replica-a:36007/"):
            return ConnectionError("replica-a unavailable")
        raise AssertionError("router migrated mutable session to replica-b")

    async def scenario() -> None:
        client = FakeClient(handler)
        endpoints = (
            make_endpoint("http://replica-a:36007", "replica-a"),
            make_endpoint("http://replica-b:36004", "replica-b"),
        )
        router = bench.StickyDroidRouter(endpoints)
        created = await router.create_session(client, make_create("create"))
        assert created.response.session_id == "sticky"
        first = await router.push_frame(client, make_push("sticky", "push"))
        second = await router.push_frame(client, make_push("sticky", "push"))
        assert first.parse_error.startswith("transport:")
        assert second.parse_error.startswith("transport:")
        assert client.urls[-2:] == [
            "http://replica-a:36007/droid.push_frame", "http://replica-a:36007/droid.push_frame",
        ]
        assert await router.endpoint_for_session("sticky") == endpoints[0]

    asyncio.run(scenario())


def test_http_200_wrong_replica_trace_is_semantically_rejected() -> None:
    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            return FakeResponse(payload={
                "ownership": metadata["ownership"], "session_id": "sticky", "error": None,
            })
        owner = Ownership.from_mapping(metadata["ownership"])
        trace = DroidBatchTrace(
            batch_id="batch", replica_id="replica-b",
            admitted_monotonic_s=1.0, dispatched_monotonic_s=1.0,
            fnet_forward_started_monotonic_s=1.0, fnet_completed_monotonic_s=2.0,
            completed_monotonic_s=2.0, fnet_forward_count=1,
            session_local_forward_count=1, request_count=1,
            effective_work_units=1, model_load_count=1, session_ids=("sticky",),
        )
        status = StepStatus(
            ownership=owner, session_id="sticky", frame_id=metadata["frame_id"],
            source_timestamp_s=float(metadata["source_timestamp_s"]),
            validity=FrameValidity(metadata["frame_id"], float(metadata["source_timestamp_s"]), True, True),
            keyframe_count=1, trace=trace,
        )
        return FakeResponse(payload={
            "ownership": metadata["ownership"], "status": status.to_wire(), "error": None,
        })

    async def scenario() -> None:
        endpoint = make_endpoint("http://replica-a:36007", "replica-a")
        router = bench.StickyDroidRouter((endpoint,))
        client = FakeClient(handler)
        await router.create_session(client, make_create("create"))
        call = await router.push_frame(client, make_push("sticky", "push"))
        assert call.response is None
        assert "trace replica disagrees" in call.parse_error
        assert await router.endpoint_for_session("sticky") == endpoint

    asyncio.run(scenario())


def test_http_200_malformed_finalize_is_failed_and_mapping_is_retained() -> None:
    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            return FakeResponse(payload={
                "ownership": metadata["ownership"], "session_id": "sticky", "error": None,
            })
        return FakeResponse(
            status_code=200,
            payload=None,
            content=b"this is not multipart",
            content_type="multipart/form-data; boundary=missing",
        )

    async def scenario() -> None:
        endpoint = make_endpoint("http://replica-a:36007", "replica-a")
        router = bench.StickyDroidRouter((endpoint,))
        client = FakeClient(handler)
        await router.create_session(client, make_create("create"))
        call = await router.finalize(client, make_finalize("sticky", "finalize"))
        record = bench.record_from_call(
            call, level="malformed", request_id="finalize", session_id="sticky", scheduled_s=None,
        )
        assert call.http_status == 200
        assert call.response is None
        assert record.outcome == "failed"
        assert record.error_code == "semantic_parse_failure"
        assert record.semantic_valid is False
        assert await router.endpoint_for_session("sticky") == endpoint

    asyncio.run(scenario())


def test_replica_endpoints_load_plan_identities_and_reject_shared_gpu(tmp_path) -> None:
    identities = [runtime_identity("replica-a", 7), runtime_identity("replica-b", 4)]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "corpus_digest": "corpus", "measurement_interval_id": "interval",
        "replicas": [{"expected_server_identity": identity.to_wire()} for identity in identities],
    }))
    args = type("Args", (), {
        "endpoint": "http://replica-a:36007,http://replica-b:36004",
        "replica_ids": None,
        "runtime_identities": plan_path,
        "model_revision": "droid-v1",
        "corpus_digest": "corpus", "measurement_interval_id": "interval",
    })()
    endpoints = bench.replica_endpoints(args)
    assert [endpoint.runtime_identity.assigned_gpu for endpoint in endpoints] == [7, 4]
    assert [endpoint.expected_replica_id for endpoint in endpoints] == ["replica-a", "replica-b"]

    shared = [runtime_identity("replica-a", 7), runtime_identity("replica-b", 7)]
    plan_path.write_text(json.dumps({
        "corpus_digest": "corpus", "measurement_interval_id": "interval",
        "replicas": [{"expected_server_identity": identity.to_wire()} for identity in shared],
    }))
    args.endpoint = "http://replica-a:36007,http://replica-b:36007"
    try:
        bench.replica_endpoints(args)
    except ValueError as exc:
        assert "disjoint physical GPUs" in str(exc)
    else:
        raise AssertionError("shared physical GPU identity was accepted")


def test_replica_endpoints_reject_missing_dependency_source_identity(tmp_path) -> None:
    identity = ServerIdentity(**{**runtime_identity("replica-a").__dict__, "dependency_digest": None})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
        {"expected_server_identity": identity.to_wire()},
    ]}))
    args = SimpleNamespace(
        endpoint="http://replica-a:36007", replica_ids=None, runtime_identities=plan_path,
        model_revision="droid-v1", corpus_digest="corpus", measurement_interval_id="interval",
    )
    with __import__("pytest").raises(ValueError, match="dependency source identity"):
        bench.replica_endpoints(args)


def test_wrong_worker_runtime_identity_is_rejected_before_sticky_result() -> None:
    wrong = ServerIdentity(**{**runtime_identity("replica-a").__dict__, "checkpoint_digest": "wrong"})

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            return FakeResponse(payload={
                "ownership": metadata["ownership"], "session_id": "sticky", "error": None,
                "server_identity": wrong.to_wire(),
            })
        raise AssertionError("create should fail identity validation")

    async def scenario() -> None:
        router = bench.StickyDroidRouter((make_endpoint("http://replica-a:36007", "replica-a"),))
        call = await router.create_session(FakeClient(handler), make_create("create"))
        assert call.response is None
        assert "runtime identity mismatch" in call.parse_error

    asyncio.run(scenario())


def test_backpressure_typed_identity_is_semantic_evidence_not_parse_failure() -> None:
    endpoint = make_endpoint("http://replica-a:36007", "replica-a")

    def handler(_url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        response = typed_error(metadata, ErrorCode.BACKPRESSURE, "bounded admission", retryable=True)
        response.status_code = 429
        response._payload["server_identity"] = endpoint.runtime_identity.to_wire()
        response.content = json.dumps(response._payload).encode()
        return response

    async def scenario() -> None:
        call = await bench._post_typed(FakeClient(handler), endpoint, "push_frame", make_push("sticky", "push"))
        item = bench.record_from_call(call, level="D2", request_id="push", session_id="sticky", scheduled_s=1.0)
        assert call.http_status == 429
        assert call.parse_error is None
        assert call.response.error.code is ErrorCode.BACKPRESSURE
        assert item.outcome == "rejected" and item.semantic_valid is True
        assert item.replica_id == "replica-a"

    asyncio.run(scenario())


def test_nonterminal_finalize_error_retries_on_same_replica_until_server_terminal_evidence() -> None:
    endpoint = make_endpoint("http://replica-a:36007", "replica-a")
    attempts = 0

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        nonlocal attempts
        if url.endswith("create_session"):
            return FakeResponse(payload={"ownership": metadata["ownership"], "session_id": "sticky", "error": None})
        attempts += 1
        return typed_error(
            metadata, ErrorCode.UNRESOLVED, "actor state still open" if attempts == 1 else "retired",
            terminal=attempts == 2,
        )

    async def scenario() -> None:
        client = FakeClient(handler)
        router = bench.StickyDroidRouter((endpoint,))
        await router.create_session(client, make_create("create"))
        first = await router.finalize(client, make_finalize("sticky", "f1"))
        assert first.response.terminal is False
        assert await router.endpoint_for_session("sticky") == endpoint
        second = await router.finalize(client, make_finalize("sticky", "f2"))
        assert second.response.terminal is True
        assert await router.endpoint_for_session("sticky") == endpoint
        assert await router.active_session_count() == 0
        assert await router.terminal_route_count() == 1
        assert client.urls[-2:] == [
            "http://replica-a:36007/droid.finalize", "http://replica-a:36007/droid.finalize",
        ]

    asyncio.run(scenario())


def test_replica_endpoints_reject_cross_replica_checkpoint_source_and_schema(tmp_path) -> None:
    identities = [runtime_identity("replica-a", 7), runtime_identity("replica-b", 4)]
    plan_path = tmp_path / "plan.json"
    args = type("Args", (), {
        "endpoint": "http://replica-a:36007,http://replica-b:36004",
        "replica_ids": None, "runtime_identities": plan_path, "model_revision": "droid-v1",
        "corpus_digest": "corpus", "measurement_interval_id": "interval",
    })()
    for field, value in (
        ("checkpoint_digest", "other"), ("release_sha", "other-source"), ("schema_version", "other-schema"),
        ("dependency_digest", "other-source-bundle"),
    ):
        bad = ServerIdentity(**{**identities[1].__dict__, field: value})
        plan_path.write_text(json.dumps({"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
            {"expected_server_identity": identities[0].to_wire()},
            {"expected_server_identity": bad.to_wire()},
        ]}))
        try:
            bench.replica_endpoints(args)
        except ValueError as exc:
            assert field.split("_")[0] in str(exc)
        else:
            raise AssertionError(f"cross-replica {field} mismatch was accepted")


def test_replica_endpoints_reject_unequal_dependency_roots(tmp_path) -> None:
    identities = [runtime_identity("replica-a", 7), runtime_identity("replica-b", 4)]
    mismatched = ServerIdentity(**{
        **identities[1].__dict__, "dependency_root": "/sources/other-source/droid_slam",
    })
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
        {"expected_server_identity": identities[0].to_wire()},
        {"expected_server_identity": mismatched.to_wire()},
    ]}))
    args = SimpleNamespace(
        endpoint="http://replica-a:36007,http://replica-b:36004", replica_ids=None,
        runtime_identities=plan_path, model_revision="droid-v1", corpus_digest="corpus",
        measurement_interval_id="interval",
    )
    with __import__("pytest").raises(ValueError, match="dependency_root"):
        bench.replica_endpoints(args)


def test_benchmark_rejects_corpus_or_measurement_interval_that_differs_from_plan(tmp_path) -> None:
    identity = runtime_identity("replica-a", 7)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "corpus_digest": "corpus-a", "measurement_interval_id": "interval-a",
        "replicas": [{"expected_server_identity": identity.to_wire()}],
    }))
    args = type("Args", (), {
        "endpoint": "http://replica-a:36007", "replica_ids": None,
        "runtime_identities": plan_path, "model_revision": "droid-v1",
        "corpus_digest": "corpus-b", "measurement_interval_id": "interval-a",
    })()
    try:
        bench.replica_endpoints(args)
    except ValueError as exc:
        assert "corpus_digest" in str(exc)
    else:
        raise AssertionError("benchmark accepted a corpus different from the launch plan")
    args.corpus_digest = "corpus-a"
    args.measurement_interval_id = "interval-b"
    try:
        bench.replica_endpoints(args)
    except ValueError as exc:
        assert "measurement_interval_id" in str(exc)
    else:
        raise AssertionError("benchmark accepted a measurement interval different from the launch plan")


def test_push_offer_rate_uses_actual_submission_window_not_drain() -> None:
    records = [
        bench.RequestRecord("push_frame", "D2", "a", "s-a", 9.0, 10.0, 100.0, 90.0, 200, "completed", None, None, None, replica_id="replica-a", semantic_valid=True),
        bench.RequestRecord("push_frame", "D2", "b", "s-b", 10.0, 11.0, 101.0, 90.0, 200, "completed", None, None, None, replica_id="replica-b", semantic_valid=True),
    ]
    row = bench.summarize(records)[0]
    assert row["run_start_s"] == 9.0
    assert row["first_submit_s"] == 10.0 and row["final_submit_s"] == 11.0
    assert row["actual_offer_window_s"] == 1.0
    assert row["drain_end_s"] == 101.0 and row["drain_duration_s"] == 90.0
    assert row["offered_rate_per_s"] == 2.0
    assert row["per_replica_actual_offer"]["replica-a"]["submitted_count"] == 1


def test_finalizations_are_gathered_in_parallel_without_losing_sticky_affinity(monkeypatch) -> None:
    endpoints = (make_endpoint("http://replica-a:36007", "replica-a"), make_endpoint("http://replica-b:36004", "replica-b"))
    final_started = asyncio.Event()
    active_finalizes = 0
    max_active_finalizes = 0

    class ParallelClient(FakeClient):
        async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> FakeResponse:
            nonlocal active_finalizes, max_active_finalizes
            self.urls.append(url)
            metadata, arrays = parse_multipart_request_fields(content, headers["Content-Type"])
            result = self.handler(url, metadata, arrays)
            if url.endswith("finalize"):
                active_finalizes += 1
                max_active_finalizes = max(max_active_finalizes, active_finalizes)
                if active_finalizes == 2:
                    final_started.set()
                await final_started.wait()
                active_finalizes -= 1
            if result._payload is not None and "server_identity" not in result._payload:
                replica_id = "replica-a" if "replica-a" in url else "replica-b"
                result._payload["server_identity"] = runtime_identity(replica_id).to_wire()
                result.content = json.dumps(result._payload).encode()
            return result

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            assert metadata["options"]["buffer"] == 128
            return FakeResponse(payload={"ownership": metadata["ownership"], "session_id": f"s-{'a' if 'replica-a' in url else 'b'}", "error": None})
        if url.endswith("push_frame"):
            return typed_error(metadata, ErrorCode.BACKPRESSURE, "busy", retryable=True)
        return typed_error(metadata, ErrorCode.UNRESOLVED, "retired", terminal=True)

    monkeypatch.setattr(bench, "read_payload", lambda _payload: (b"rgb", b"mask"))
    payload = bench.CachedPayload("p", 0, 0.0, "rgb", "mask", "unused", "unused")
    args = type("Args", (), {"model_revision": "droid-v1", "start_delay_s": 0.0, "waves": 1, "session_buffer": 128})()
    async def scenario() -> None:
        records = await bench.benchmark_level(args, ParallelClient(handler), bench.StickyDroidRouter(endpoints), [payload], 0, 2, 1.0)
        assert len([record for record in records if record.operation == "finalize"]) == 2

    asyncio.run(scenario())
    assert max_active_finalizes == 2


def test_one_session_level_is_labeled_one_replica_control_with_explicit_one_zero_assignment() -> None:
    endpoints = (make_endpoint("http://replica-a:36007", "replica-a"), make_endpoint("http://replica-b:36004", "replica-b"))

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            owner = "a" if "replica-a" in url else "b"
            return FakeResponse(payload={"ownership": metadata["ownership"], "session_id": f"only-{owner}", "error": None})
        if url.endswith("push_frame"):
            return typed_error(metadata, ErrorCode.BACKPRESSURE, "bounded", retryable=True)
        return typed_error(metadata, ErrorCode.UNRESOLVED, "terminal", terminal=True)

    monkeypatch_payload = bench.read_payload
    bench.read_payload = lambda _payload: (b"rgb", b"mask")
    try:
        args = type("Args", (), {"model_revision": "droid-v1", "start_delay_s": 0.0, "waves": 1})()
        payload = bench.CachedPayload("p", 0, 0.0, "rgb", "mask", "unused", "unused")

        async def scenario() -> list[bench.RequestRecord]:
            router = bench.StickyDroidRouter(endpoints)
            first = await bench.benchmark_level(args, FakeClient(handler), router, [payload], 0, 1, 1.0)
            second = await bench.benchmark_level(args, FakeClient(handler), router, [payload], 1, 1, 2.0)
            return first + second

        records = asyncio.run(scenario())
    finally:
        bench.read_payload = monkeypatch_payload
    assert {record.treatment_role for record in records} == {"one_replica_control"}
    assignments = {tuple(sorted((record.planned_session_assignment or {}).items())) for record in records}
    assert assignments == {
        (("replica-a", 1), ("replica-b", 0)),
        (("replica-a", 0), ("replica-b", 1)),
    }
    pushes = [row for row in bench.summarize(records) if row["operation"] == "push_frame"]
    assert {push["treatment_role"] for push in pushes} == {"one_replica_control"}
    assert {tuple(sorted(push["planned_session_assignment"].items())) for push in pushes} == assignments
    assert any(push["per_replica_actual_offer"]["replica-a"]["submitted_count"] == 1 for push in pushes)
    assert any(push["per_replica_actual_offer"]["replica-b"]["submitted_count"] == 1 for push in pushes)


def test_level_rejects_missing_expected_replica_before_offer() -> None:
    endpoints = (make_endpoint("http://replica-a:36007", "replica-a"), make_endpoint("http://replica-b:36004", "replica-b"))

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if "replica-b" in url:
            return typed_error(metadata, ErrorCode.BACKPRESSURE, "full", retryable=True)
        return FakeResponse(payload={"ownership": metadata["ownership"], "session_id": "only-a", "error": None})

    args = type("Args", (), {"model_revision": "droid-v1", "start_delay_s": 0.0, "waves": 1})()
    async def scenario() -> None:
        with __import__("pytest").raises(RuntimeError, match="expected replica completeness"):
            await bench.benchmark_level(args, FakeClient(handler), bench.StickyDroidRouter(endpoints), [], 0, 2, 1.0)

    asyncio.run(scenario())


def test_terminal_route_tombstones_are_bounded_and_retries_stay_on_original_endpoint() -> None:
    endpoints = (make_endpoint("http://replica-a:36007", "replica-a"), make_endpoint("http://replica-b:36004", "replica-b"))

    def handler(url: str, metadata: dict[str, Any], _arrays: Any) -> FakeResponse:
        if url.endswith("create_session"):
            return FakeResponse(payload={"ownership": metadata["ownership"], "session_id": metadata["ownership"]["request_id"], "error": None})
        return typed_error(metadata, ErrorCode.UNRESOLVED, "terminal", terminal=True)

    async def scenario() -> None:
        client = FakeClient(handler)
        router = bench.StickyDroidRouter(endpoints, max_terminal_routes=1)
        await router.create_session(client, make_create("s-a"))
        await router.finalize(client, make_finalize("s-a", "f-a"))
        retry = await router.finalize(client, make_finalize("s-a", "f-a"))
        assert retry.response.terminal is True
        assert client.urls[-1] == "http://replica-a:36007/droid.finalize"
        await router.create_session(client, make_create("s-b"))
        await router.finalize(client, make_finalize("s-b", "f-b"))
        assert await router.terminal_route_count() == 1
        assert await router.endpoint_for_session("s-a") is None
        assert await router.endpoint_for_session("s-b") == endpoints[1]

    asyncio.run(scenario())


def test_replica_endpoint_config_requires_contract_and_rejects_duplicate_id_url_or_cuda_identity(tmp_path) -> None:
    identities = [runtime_identity("replica-a", 7), runtime_identity("replica-b", 4)]
    plan_path = tmp_path / "plan.json"
    args = type("Args", (), {
        "endpoint": "http://replica-a:36007,http://replica-b:36004", "replica_ids": None,
        "runtime_identities": plan_path, "model_revision": "droid-v1",
        "corpus_digest": "corpus", "measurement_interval_id": "interval",
    })()
    for mutation, expected in (
        ({"corpus_digest": ""}, "corpus_digest"),
        ({"measurement_interval_id": ""}, "measurement_interval_id"),
    ):
        plan = {"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
            {"expected_server_identity": identity.to_wire()} for identity in identities
        ]}
        plan.update(mutation)
        plan_path.write_text(json.dumps(plan))
        with __import__("pytest").raises(ValueError, match=expected):
            bench.replica_endpoints(args)
    plan_path.write_text(json.dumps({"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
        {"expected_server_identity": identity.to_wire()} for identity in identities
    ]}))
    args.corpus_digest = ""
    with __import__("pytest").raises(ValueError, match="corpus_digest"):
        bench.replica_endpoints(args)
    args.corpus_digest = "corpus"
    args.measurement_interval_id = ""
    with __import__("pytest").raises(ValueError, match="measurement_interval_id"):
        bench.replica_endpoints(args)
    args.measurement_interval_id = "interval"
    args.replica_ids = "same,same"
    with __import__("pytest").raises(ValueError, match="replica ids"):
        bench.replica_endpoints(args)
    args.replica_ids = None
    args.endpoint = "http://replica-a:36007,http://replica-a:36007"
    with __import__("pytest").raises(ValueError, match="endpoints"):
        bench.replica_endpoints(args)
    args.endpoint = "http://replica-a:36007,http://replica-b:36004"
    duplicate_cuda = ServerIdentity(**{**identities[1].__dict__, "cuda_uuid": "GPU-7"})
    plan_path.write_text(json.dumps({"corpus_digest": "corpus", "measurement_interval_id": "interval", "replicas": [
        {"expected_server_identity": identities[0].to_wire()}, {"expected_server_identity": duplicate_cuda.to_wire()},
    ]}))
    with __import__("pytest").raises(ValueError, match="CUDA identities"):
        bench.replica_endpoints(args)


def test_equal_replica_offer_acceptance_requires_exact_work_and_only_tolerates_timestamp_jitter() -> None:
    def offered(replica: str, request_id: str, sent: float) -> bench.RequestRecord:
        return bench.RequestRecord("push_frame", "D2", request_id, request_id, 0.0, sent, sent + 1.0, 1.0, 200,
                                   "completed", None, None, None, replica_id=replica, semantic_valid=True)
    accepted = [offered("replica-a", "a0", 0.00), offered("replica-a", "a1", 1.00),
                offered("replica-b", "b0", 0.02), offered("replica-b", "b1", 1.02)]
    bench.validate_equal_replica_offers(accepted, expected_replica_ids=("replica-a", "replica-b"))
    with __import__("pytest").raises(RuntimeError, match="unequal"):
        bench.validate_equal_replica_offers(accepted[:-1], expected_replica_ids=("replica-a", "replica-b"))
    staggered = [offered("replica-a", "a0", 0.00), offered("replica-b", "b0", 0.10)]
    with __import__("pytest").raises(RuntimeError, match="common start"):
        bench.validate_equal_replica_offers(staggered, expected_replica_ids=("replica-a", "replica-b"))


def test_payload_corpus_digest_excludes_cache_paths_and_is_stable_for_equal_content() -> None:
    base = bench.CachedPayload("payload-0000", 3, 0.1, "rgb", "mask", "/run-a/rgb", "/run-a/mask")
    relocated = bench.CachedPayload("payload-0000", 3, 0.1, "rgb", "mask", "/another-root/rgb", "/another-root/mask")
    assert bench._payload_corpus_digest([base]) == bench._payload_corpus_digest([relocated])


def test_run_aligned_nvml_labels_d1_d2_d4_and_rejects_uuid_mismatch(tmp_path) -> None:
    from ego_annotation.serving.benchmark.measurement import NvmlSampler, validate_gpu_samples

    times = iter([0.0, 0.1, 0.2, 0.5, 0.9, 1.0, 1.1])
    sampler = NvmlSampler(
        gpu_ids=(7,), gpu_uuids={7: "GPU-7"}, experiment_id="exp", release_digest="release",
        sample_fn=lambda _gpu: {"gpu_uuid": "GPU-7", "utilization_gpu_pct": 50, "memory_used_bytes": 123},
        clock=lambda: next(times), interval_s=1000,
    )
    sampler.start()
    for level in ("D1", "D2", "D4"):
        sampler.set_level(level)
        sampler.sample(level=level)
    sampler.stop()
    path = sampler.write(tmp_path / "gpu_samples.json")
    evidence = validate_gpu_samples(
        path, gpu_ids=(7,), gpu_uuids={7: "GPU-7"}, experiment_id="exp", release_digest="release",
        run_start_s=0.1, run_end_s=0.9, min_samples_per_gpu=2,
    )
    assert {sample["level"] for sample in evidence["samples"]} >= {"D1", "D2", "D4"}
    payload = json.loads(path.read_text())
    payload["samples"][1]["gpu_uuid"] = "GPU-wrong"
    path.write_text(json.dumps(payload))
    try:
        validate_gpu_samples(
            path, gpu_ids=(7,), gpu_uuids={7: "GPU-7"}, experiment_id="exp", release_digest="release",
            run_start_s=0.1, run_end_s=0.9, min_samples_per_gpu=2,
        )
    except ValueError as exc:
        assert "UUID mismatch" in str(exc)
    else:
        raise AssertionError("wrong NVML GPU UUID was accepted")
