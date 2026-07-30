from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx

from scripts import serve_v22_annotation_api as api


def test_http_manager_queues_second_job_at_total_capacity(monkeypatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        api.configure_admission_limits(total_request_limit=1, algorithm_inflight_multiplier=2)
        monkeypatch.setattr(api, "DEFAULT_OUTPUT_ROOT", tmp_path)
        entered_count = 0
        entered_lock = threading.Lock()
        first_entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def fake_run_annotation_job(req: api.AnnotationJobRequest) -> api.AnnotationJobResponse:
            nonlocal entered_count
            with entered_lock:
                entered_count += 1
                if entered_count == 1:
                    loop.call_soon_threadsafe(first_entered.set)
            release.wait(timeout=5.0)
            return api.AnnotationJobResponse(
                job_id=req.job_id or "manager-test",
                status="ok",
                run_root=str(tmp_path / "run"),
                manifest_path=str(tmp_path / "run" / "annotation_pipeline_manifest.json"),
                download_url="/v1/downloads/manager-test.zip",
                elapsed_s=0.0,
                summary={"accepted": False, "diagnostic_only": True},
            )

        monkeypatch.setattr(api, "run_annotation_job", fake_run_annotation_job)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/annotation-jobs",
                    files={"file": ("one.mp4", b"one", "video/mp4")},
                )
            )
            await asyncio.wait_for(first_entered.wait(), timeout=2.0)
            second = asyncio.create_task(
                client.post(
                    "/v1/annotation-jobs",
                    files={"file": ("two.mp4", b"two", "video/mp4")},
                )
            )
            try:
                await asyncio.wait_for(asyncio.shield(second), timeout=0.05)
            except asyncio.TimeoutError:
                pass
            else:
                raise AssertionError("second HTTP manager request entered before the first permit was released")
            assert not second.done()
            with entered_lock:
                assert entered_count == 1
            release.set()
            first_response = await first
            second_response = await second
            assert first_response.status_code == 200
            assert second_response.status_code == 200
            with entered_lock:
                assert entered_count == 2

    try:
        asyncio.run(scenario())
    finally:
        api.configure_admission_limits(total_request_limit=128, algorithm_inflight_multiplier=2)


def test_http_manager_caps_1701_requests_at_128_active(monkeypatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        api.configure_admission_limits(total_request_limit=128, algorithm_inflight_multiplier=2)
        monkeypatch.setattr(api, "DEFAULT_OUTPUT_ROOT", tmp_path)
        active = 0
        max_active = 0
        entered = 0
        active_lock = asyncio.Lock()
        capacity_reached = asyncio.Event()
        release = asyncio.Event()

        async def fake_run_in_threadpool(_func: object, req: api.AnnotationJobRequest) -> api.AnnotationJobResponse:
            nonlocal active, max_active, entered
            async with active_lock:
                active += 1
                entered += 1
                max_active = max(max_active, active)
                if active == 128:
                    capacity_reached.set()
            await release.wait()
            async with active_lock:
                active -= 1
            return api.AnnotationJobResponse(
                job_id=req.job_id or "manager-stress",
                status="ok",
                run_root=str(tmp_path / "run"),
                manifest_path=str(tmp_path / "run" / "annotation_pipeline_manifest.json"),
                download_url="/v1/downloads/manager-stress.zip",
                elapsed_s=0.0,
                summary={"accepted": False, "diagnostic_only": True},
            )

        monkeypatch.setattr(api, "run_in_threadpool", fake_run_in_threadpool)
        transport = httpx.ASGITransport(app=api.app)
        request_count = 1701
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/annotation-jobs",
                        files={"file": (f"item-{index}.mp4", b"item", "video/mp4")},
                    )
                )
                for index in range(request_count)
            ]
            await asyncio.wait_for(capacity_reached.wait(), timeout=10.0)
            assert max_active == 128
            assert sum(not task.done() for task in tasks) >= request_count - 128
            release.set()
            responses = await asyncio.gather(*tasks)
            assert len(responses) == request_count
            assert all(response.status_code == 200 for response in responses)
            assert entered == request_count
            assert max_active == 128

    try:
        asyncio.run(scenario())
    finally:
        api.configure_admission_limits(total_request_limit=128, algorithm_inflight_multiplier=2)
