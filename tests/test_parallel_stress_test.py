from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import httpx

from scripts.parallel_stress_test import PreparedOperation, post_raw, run_once


def run(coro):
    return asyncio.run(coro)


def test_run_once_starts_mock_service_operations_concurrently_and_collects_results() -> None:
    async def scenario() -> None:
        started: set[str] = set()
        release = asyncio.Event()

        def operation(name: str) -> PreparedOperation:
            async def invoke(_client: Any, _run_id: str) -> dict[str, Any]:
                started.add(name)
                await release.wait()
                return {"service": name, "endpoint": f"http://fake/{name}", "success": True, "latency_ms": 1.0, "http_status": 200, "result_hash": name, "batch_wait_ms": 2.0}

            return PreparedOperation(name, f"http://fake/{name}", invoke, {"fixture": name})

        operations = {name: operation(name) for name in ("unidepth", "hands", "cosmos")}
        task = asyncio.create_task(run_once(operations, object(), "parallel-r1"))
        for _ in range(20):
            if started == set(operations):
                break
            await asyncio.sleep(0)
        assert started == set(operations), "all calls must be runnable before any fake endpoint completes"
        release.set()
        wall_ms, rows = await task
        assert wall_ms >= 0
        assert [row["service"] for row in rows] == list(operations)
        assert all(row["success"] for row in rows)
        assert [row["gpu_mapping"] for row in rows] == [0, 1, 6]
        assert all(row["payload"]["fixture"] == row["service"] for row in rows)

    run(scenario())


def test_run_once_turns_one_mock_endpoint_failure_into_a_service_record() -> None:
    async def broken(_client: Any, _run_id: str) -> dict[str, Any]:
        raise RuntimeError("fake endpoint unavailable")

    operation = PreparedOperation("droid", "http://fake/droid", broken, {"manifest": "fake"})
    _wall_ms, rows = run(run_once({"droid": operation}, object(), "parallel-r1"))
    assert len(rows) == 1
    assert rows[0]["success"] is False
    assert rows[0]["typed_error"]["code"] == "client_preparation"
    assert "fake endpoint unavailable" in rows[0]["typed_error"]["message"]
    assert rows[0]["gpu_mapping"] == 2


def test_post_raw_collects_mock_http_response_hash_and_batch_wait() -> None:
    class FakeClient:
        async def post(self, endpoint: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
            assert endpoint == "http://fake/hands"
            assert content == b"real-request"
            assert headers["Content-Type"] == "application/json"
            return httpx.Response(
                200,
                json={"result": {"trace": {"admitted_monotonic_s": 10.0, "dispatched_monotonic_s": 10.025}}},
                headers={"content-type": "application/json"},
            )

    row = run(post_raw(FakeClient(), service="hands", endpoint="http://fake/hands", body=b"real-request", content_type="application/json"))
    assert row["success"] is True
    assert row["result_hash"] == hashlib.sha256(b'{"result":{"trace":{"admitted_monotonic_s":10.0,"dispatched_monotonic_s":10.025}}}').hexdigest()
    assert abs(row["batch_wait_ms"] - 25.0) < 1e-9
    assert row["response"]["result"]["trace"]["dispatched_monotonic_s"] == 10.025
