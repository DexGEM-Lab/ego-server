"""Fixed-wave HTTP process bursts for proving Serve physical batch formation.

A coroutine ``gather`` does not make request arrivals simultaneous when every task
constructs a multi-megabyte multipart body on its shared event loop.  This module
builds every body before a wave has a release clock, then assigns one independent
OS process to each already-connected request.  The children wait on a common
monotonic target and only write their prebuilt body after that target.
"""
from __future__ import annotations

import argparse
import http.client
import json
import multiprocessing as mp
import queue
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from ego_annotation.serving.benchmark.generator import _record_from_response
from ego_annotation.serving.benchmark.manifest import PayloadItem, PayloadManifest
from ego_annotation.serving.benchmark.metrics import ItemRecord
from ego_annotation.serving.contracts import ErrorCode, ServiceError
from ego_annotation.serving.binary_envelope import CONTENT_TYPE as BINARY_ENVELOPE_CONTENT_TYPE, content_type_is_binary_envelope
from ego_annotation.serving.gateway import (
    EnvelopeHttpBody,
    GatewayResponse,
    _build_generic_envelope,
    _build_generic_multipart,
    _gateway_response_from_multipart,
    _parse_generic_envelope,
)
from ego_annotation.serving.transport import parse_multipart_response


class IncompleteWaveEvidenceError(RuntimeError):
    """The parent could not prove that every process completed one wave."""


@dataclass(frozen=True)
class PrebuiltWaveRequest:
    item: PayloadItem
    body: bytes | tuple[memoryview, ...]
    content_type: str
    wire_format: str
    build_started_s: float
    build_completed_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item.item_id,
            "request_id": self.item.ownership.request_id,
            "payload_bytes": _body_length(self.body),
            "wire_format": self.wire_format,
            "build_started_s": self.build_started_s,
            "build_completed_s": self.build_completed_s,
        }


@dataclass(frozen=True)
class ChildSubmission:
    child_id: str
    ready_s: float | None
    submit_s: float | None
    response_s: float | None
    status_code: int | None
    response_headers: Mapping[str, str]
    response_body: bytes
    error: str | None = None


@dataclass(frozen=True)
class WaveRequestEvidence:
    wave_index: int
    item_id: str
    request_id: str
    build_started_s: float
    build_completed_s: float
    child_ready_s: float
    release_target_s: float
    submit_s: float
    response_s: float
    process_exit_code: int
    http_status: int
    batch_id: str
    batch_size: int
    allocator_memory: Mapping[str, int | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_index": self.wave_index,
            "item_id": self.item_id,
            "request_id": self.request_id,
            "build_started_s": self.build_started_s,
            "build_completed_s": self.build_completed_s,
            "child_ready_s": self.child_ready_s,
            "release_target_s": self.release_target_s,
            "submit_s": self.submit_s,
            "response_s": self.response_s,
            "process_exit_code": self.process_exit_code,
            "http_status": self.http_status,
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "allocator_memory": dict(self.allocator_memory),
        }


@dataclass(frozen=True)
class WaveEvidence:
    wave_index: int
    requested_wave_size: int
    release_target_s: float
    request_evidence: tuple[WaveRequestEvidence, ...]
    physical_batches: tuple[dict[str, object], ...]
    submit_spread_s: float
    synchronization_window_s: float
    submissions_within_synchronization_window: bool
    dominant_observed_batch_size: int | None
    requested_wave_matches_dominant_batch_size: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_index": self.wave_index,
            "requested_wave_size": self.requested_wave_size,
            "release_target_s": self.release_target_s,
            "request_evidence": [record.to_dict() for record in self.request_evidence],
            "physical_batches": list(self.physical_batches),
            "submit_spread_s": self.submit_spread_s,
            "synchronization_window_s": self.synchronization_window_s,
            "submissions_within_synchronization_window": self.submissions_within_synchronization_window,
            "dominant_observed_batch_size": self.dominant_observed_batch_size,
            "requested_wave_matches_dominant_batch_size": self.requested_wave_matches_dominant_batch_size,
        }


@dataclass(frozen=True)
class FixedWaveRun:
    records: tuple[ItemRecord, ...]
    waves: tuple[WaveEvidence, ...]
    wire_format: str = "multipart"

    def to_dict(self) -> dict[str, object]:
        return {
            "wire_format": self.wire_format,
            "records": [record.to_dict() for record in self.records],
            "waves": [wave.to_dict() for wave in self.waves],
        }


def _multipart_for_item(item: PayloadItem) -> tuple[bytes, str]:
    """Build the exact gateway wire body without an event-loop-side send path."""
    request = item.to_gateway_request()
    metadata: dict[str, Any] = {
        "ownership": request.ownership.to_wire(),
        "api_name": request.api_name.value,
        "metadata": {key: value for key, value in request.metadata.items() if key != "work_units"},
        "work_units": request.work_units,
    }
    if request.spatial is not None:
        metadata["spatial"] = request.spatial.to_wire()
    if request.model_revision is not None:
        metadata["model_revision"] = request.model_revision
    parts = [(part.name, bytes(part.data), part.shape, part.dtype) for part in request.parts]
    return _build_generic_multipart(metadata, parts)


def _envelope_for_item(item: PayloadItem) -> tuple[tuple[memoryview, ...], str]:
    """Build the same request as the gateway, preserving header/tensor vectors."""
    request = item.to_gateway_request()
    metadata: dict[str, Any] = {
        "ownership": request.ownership.to_wire(),
        "api_name": request.api_name.value,
        "metadata": {key: value for key, value in request.metadata.items() if key != "work_units"},
        "work_units": request.work_units,
    }
    if request.spatial is not None:
        metadata["spatial"] = request.spatial.to_wire()
    if request.model_revision is not None:
        metadata["model_revision"] = request.model_revision
    parts = [(part.name, bytes(part.data), part.shape, part.dtype) for part in request.parts]
    envelope: EnvelopeHttpBody = _build_generic_envelope(metadata, parts)
    return envelope.iovecs, BINARY_ENVELOPE_CONTENT_TYPE


def _body_length(body: bytes | tuple[memoryview, ...]) -> int:
    return len(body) if isinstance(body, bytes) else sum(vector.nbytes for vector in body)


def prebuild_wave_bodies(
    items: Sequence[PayloadItem], *, wire_format: str = "multipart", clock: Callable[[], float] = time.monotonic,
    build_body: Callable[[PayloadItem], tuple[bytes | tuple[memoryview, ...], str]] | None = None,
) -> tuple[PrebuiltWaveRequest, ...]:
    """Complete all CPU multipart construction before a process wave can release."""
    if wire_format not in {"multipart", "envelope"}:
        raise ValueError("wire_format must be multipart or envelope")
    selected_builder = build_body or (_multipart_for_item if wire_format == "multipart" else _envelope_for_item)
    prepared: list[PrebuiltWaveRequest] = []
    for item in items:
        started = clock()
        body, content_type = selected_builder(item)
        completed = clock()
        prepared.append(PrebuiltWaveRequest(item, body, content_type, wire_format, started, completed))
    return tuple(prepared)


def _connection_for_endpoint(endpoint: str, timeout_s: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"endpoint must be an absolute HTTP(S) URL, got {endpoint!r}")
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return connection_type(parsed.hostname, parsed.port, timeout=timeout_s), path


def _wave_child(
    child_id: str,
    endpoint: str,
    body: bytes | tuple[memoryview, ...],
    content_type: str,
    timeout_s: float,
    release_event: Any,
    release_target: Any,
    ready_pipe: Any,
    result_queue: Any,
) -> None:
    """Connect before readiness; after release this process only sends prebuilt bytes."""
    connection: http.client.HTTPConnection | None = None
    try:
        connection, path = _connection_for_endpoint(endpoint, timeout_s)
        connection.connect()
        ready_s = time.monotonic()
        ready_pipe.send(("ready", ready_s, None))
        release_event.wait()
        target_s = float(release_target.value)
        # A clock target, rather than an event alone, prevents parent wake-up order
        # from becoming the order in which request bodies are written.
        while time.monotonic() < target_s:
            pass
        submit_s = time.monotonic()
        if isinstance(body, bytes):
            connection.request("POST", path, body=body, headers={"Content-Type": content_type})
        else:
            # http.client has no writev API, but separate ``send`` calls preserve
            # the prebuilt vectors and avoid assembling multi-megabyte tensors.
            connection.putrequest("POST", path)
            connection.putheader("Content-Type", content_type)
            connection.putheader("Content-Length", str(_body_length(body)))
            connection.endheaders()
            for vector in body:
                connection.send(vector)
        response = connection.getresponse()
        response_body = response.read()
        response_s = time.monotonic()
        result_queue.put(ChildSubmission(
            child_id=child_id, ready_s=ready_s, submit_s=submit_s, response_s=response_s,
            status_code=int(response.status), response_headers=dict(response.getheaders()),
            response_body=response_body,
        ))
    except BaseException as exc:
        try:
            ready_pipe.send(("error", None, f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
        result_queue.put(ChildSubmission(
            child_id=child_id, ready_s=None, submit_s=None, response_s=None, status_code=None,
            response_headers={}, response_body=b"", error=f"{type(exc).__name__}: {exc}",
        ))
    finally:
        if connection is not None:
            connection.close()
        ready_pipe.close()


def validate_complete_wave_evidence(
    expected_child_ids: Sequence[str], submissions: Mapping[str, ChildSubmission], exit_codes: Mapping[str, int | None],
) -> None:
    """Reject partial, failed, or clockless child evidence instead of reporting a wave."""
    expected = set(expected_child_ids)
    actual = set(submissions)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise IncompleteWaveEvidenceError(f"incomplete child evidence: missing={missing}, extra={extra}")
    failed = [child_id for child_id in expected_child_ids if submissions[child_id].error]
    if failed:
        raise IncompleteWaveEvidenceError(f"child transport failures: {failed}")
    unfinished = [child_id for child_id in expected_child_ids if exit_codes.get(child_id) != 0]
    if unfinished:
        raise IncompleteWaveEvidenceError(f"children did not exit cleanly: {unfinished}")
    clockless = [
        child_id for child_id in expected_child_ids
        if submissions[child_id].ready_s is None or submissions[child_id].submit_s is None or submissions[child_id].response_s is None
    ]
    if clockless:
        raise IncompleteWaveEvidenceError(f"children missing required clocks: {clockless}")


def _submission_content_type(submission: ChildSubmission) -> str | None:
    return next(
        (value for name, value in submission.response_headers.items() if name.lower() == "content-type"),
        None,
    )


def _response_evidence(submission: ChildSubmission, content_type: str | None) -> str:
    body_prefix = submission.response_body[:512].decode("utf-8", errors="replace")
    return (
        f"http_status={submission.status_code} content_type={content_type!r} "
        f"body_length={len(submission.response_body)} body_prefix={body_prefix!r}"
    )


def _parsed_error_evidence(error: ServiceError) -> str:
    return f"parsed_error_code={error.code.value!r} parsed_error_message={error.message!r}"


def _response_from_submission(item: PayloadItem, submission: ChildSubmission, wire_format: str = "multipart") -> GatewayResponse:
    assert submission.status_code is not None and submission.submit_s is not None and submission.response_s is not None
    transport_ms = (submission.response_s - submission.submit_s) * 1000.0
    content_type = _submission_content_type(submission)
    evidence = _response_evidence(submission, content_type)
    if wire_format not in {"multipart", "envelope"}:
        raise ValueError("wire_format must be multipart or envelope")
    def parse_response() -> tuple[dict[str, Any], Mapping[str, tuple[Any, tuple[int, ...], str]]]:
        if wire_format == "envelope":
            if not content_type_is_binary_envelope(content_type):
                raise ValueError(f"expected binary envelope response, got {content_type!r}")
            return _parse_generic_envelope(submission.response_body)
        return parse_multipart_response(submission.response_body, content_type or "")
    if submission.status_code >= 400:
        parsed_error_evidence = ""
        try:
            meta, _ = parse_response()
            if meta.get("error"):
                parsed_error_evidence = f" {_parsed_error_evidence(ServiceError.from_wire(meta['error']))}"
        except Exception as exc:
            label = "multipart_parse_exception" if wire_format == "multipart" else "envelope_parse_exception"
            parsed_error_evidence = f" {label}={type(exc).__name__}: {exc}"
        return GatewayResponse(
            ownership=item.ownership,
            error=ServiceError(
                ErrorCode.TRANSPORT,
                f"HTTP response rejected: {evidence}{parsed_error_evidence}",
                retryable=False,
                ownership=item.ownership,
            ),
            attempts=1,
            last_status_code=submission.status_code,
            transport_ms=transport_ms,
        )
    try:
        meta, arrays = parse_response()
    except Exception as exc:
        return GatewayResponse(
            ownership=item.ownership,
            error=ServiceError(
                ErrorCode.TRANSPORT,
                f"invalid multipart response: {evidence} parse_exception={type(exc).__name__}: {exc}",
                retryable=False,
                ownership=item.ownership,
            ),
            attempts=1,
            last_status_code=submission.status_code,
            transport_ms=transport_ms,
        )
    if meta.get("error"):
        error = ServiceError.from_wire(meta["error"])
        return GatewayResponse(
            ownership=item.ownership,
            error=ServiceError(
                error.code,
                f"{_parsed_error_evidence(error)} {evidence}",
                retryable=error.retryable,
                ownership=error.ownership,
                batch_id=error.batch_id,
            ),
            attempts=1,
            last_status_code=submission.status_code,
            transport_ms=transport_ms,
        )
    return _gateway_response_from_multipart(meta, arrays, item.ownership, 1, submission.status_code, transport_ms)


def _run_one_fixed_wave(
    endpoint: str, prepared: Sequence[PrebuiltWaveRequest], *, wave_index: int, release_lead_s: float,
    synchronization_window_s: float, timeout_s: float, clock: Callable[[], float], process_context: Any,
) -> tuple[list[ItemRecord], WaveEvidence]:
    if release_lead_s <= 0:
        raise ValueError("release_lead_s must be positive")
    release_event = process_context.Event()
    release_target = process_context.Value("d", 0.0)
    result_queue = process_context.Queue()
    processes: dict[str, Any] = {}
    readiness: dict[str, Any] = {}
    expected_ids: list[str] = []
    for index, request in enumerate(prepared):
        child_id = f"wave-{wave_index}-request-{index}"
        parent_pipe, child_pipe = process_context.Pipe(duplex=False)
        process = process_context.Process(
            target=_wave_child,
            args=(child_id, endpoint, request.body, request.content_type, timeout_s, release_event, release_target, child_pipe, result_queue),
            name=f"ego-service-burst-{child_id}",
        )
        process.start()
        child_pipe.close()
        processes[child_id] = process
        readiness[child_id] = parent_pipe
        expected_ids.append(child_id)

    try:
        readiness_deadline = clock() + timeout_s
        for child_id in expected_ids:
            remaining = readiness_deadline - clock()
            if remaining <= 0 or not readiness[child_id].poll(remaining):
                raise IncompleteWaveEvidenceError(f"child did not establish its connection before release: {child_id}")
            kind, _, error = readiness[child_id].recv()
            if kind != "ready":
                raise IncompleteWaveEvidenceError(f"child failed before release: {child_id}: {error}")

        # The release clock is deliberately created *after* all multipart bodies
        # and all independent connections exist.
        target_s = clock() + release_lead_s
        release_target.value = target_s
        release_event.set()

        submissions: dict[str, ChildSubmission] = {}
        response_deadline = target_s + timeout_s
        while len(submissions) < len(expected_ids):
            remaining = response_deadline - clock()
            if remaining <= 0:
                break
            try:
                submission = result_queue.get(timeout=remaining)
            except queue.Empty:
                break
            submissions[submission.child_id] = submission

        for process in processes.values():
            process.join(timeout=max(0.0, response_deadline - clock()))
        exit_codes = {child_id: process.exitcode for child_id, process in processes.items()}
        validate_complete_wave_evidence(expected_ids, submissions, exit_codes)

        records: list[ItemRecord] = []
        request_evidence: list[WaveRequestEvidence] = []
        for child_id, request in zip(expected_ids, prepared):
            submission = submissions[child_id]
            response = _response_from_submission(request.item, submission, request.wire_format)
            assert submission.submit_s is not None and submission.response_s is not None and submission.ready_s is not None
            record = _record_from_response(request.item, response, target_s, submission.submit_s, submission.response_s)
            if record.outcome != "completed" or record.batch_id is None or record.batch_size is None:
                raise IncompleteWaveEvidenceError(
                    f"server did not supply complete batch evidence for {child_id}: outcome={record.outcome}, batch_id={record.batch_id}, batch_size={record.batch_size}"
                )
            records.append(record)
            request_evidence.append(WaveRequestEvidence(
                wave_index=wave_index, item_id=request.item.item_id, request_id=request.item.ownership.request_id,
                build_started_s=request.build_started_s, build_completed_s=request.build_completed_s,
                child_ready_s=submission.ready_s, release_target_s=target_s, submit_s=submission.submit_s,
                response_s=submission.response_s, process_exit_code=exit_codes[child_id] or 0,
                http_status=submission.status_code or 0, batch_id=record.batch_id, batch_size=record.batch_size,
                allocator_memory={
                    "allocated_bytes": record.allocator_allocated_bytes,
                    "reserved_bytes": record.allocator_reserved_bytes,
                    "max_allocated_bytes": record.allocator_max_allocated_bytes,
                    "max_reserved_bytes": record.allocator_max_reserved_bytes,
                },
            ))

        submit_spread_s = max(record.submit_s for record in request_evidence) - min(record.submit_s for record in request_evidence)
        if submit_spread_s > synchronization_window_s:
            raise IncompleteWaveEvidenceError(
                f"process submissions exceeded the {synchronization_window_s * 1000.0:.1f} ms wave window: {submit_spread_s * 1000.0:.3f} ms"
            )

        physical: dict[str, dict[str, Any]] = {}
        for record in records:
            assert record.batch_id is not None and record.batch_size is not None
            existing = physical.setdefault(record.batch_id, {"batch_id": record.batch_id, "batch_size": record.batch_size, "member_item_ids": []})
            if existing["batch_size"] != record.batch_size:
                raise IncompleteWaveEvidenceError(f"server batch {record.batch_id!r} has inconsistent batch sizes")
            existing["member_item_ids"].append(record.item_id)
        physical_batches = tuple(physical[batch_id] for batch_id in sorted(physical))
        size_counts = Counter(int(batch["batch_size"]) for batch in physical_batches)
        dominant = max(size_counts, key=lambda size: (size_counts[size], size)) if size_counts else None
        return records, WaveEvidence(
            wave_index=wave_index, requested_wave_size=len(prepared), release_target_s=target_s,
            request_evidence=tuple(request_evidence), physical_batches=physical_batches,
            submit_spread_s=submit_spread_s, synchronization_window_s=synchronization_window_s,
            submissions_within_synchronization_window=True, dominant_observed_batch_size=dominant,
            requested_wave_matches_dominant_batch_size=dominant == len(prepared),
        )
    finally:
        for ready in readiness.values():
            ready.close()
        for process in processes.values():
            if process.is_alive():
                process.terminate()
            process.join()


def run_fixed_process_service_waves(
    endpoint: str, manifest: PayloadManifest, *, wave_size: int, wave_count: int,
    release_lead_s: float = 0.05, synchronization_window_s: float = 0.020, timeout_s: float = 90.0,
    clock: Callable[[], float] = time.monotonic,
    process_context: Any | None = None,
    wire_format: str = "multipart",
) -> FixedWaveRun:
    """Run B=8/B=16 waves whose actual server batch traces remain the evidence."""
    if wave_size not in {8, 16} or wave_count < 1 or len(manifest.items) < wave_size * wave_count:
        raise ValueError("waves require B=8/16 and distinct corpus items for every offered request")
    if timeout_s <= 0 or synchronization_window_s <= 0:
        raise ValueError("timeout_s and synchronization_window_s must be positive")
    if wire_format not in {"multipart", "envelope"}:
        raise ValueError("wire_format must be multipart or envelope")
    context = process_context or mp.get_context("spawn")
    records: list[ItemRecord] = []
    waves: list[WaveEvidence] = []
    for wave_index in range(wave_count):
        items = manifest.items[wave_index * wave_size:(wave_index + 1) * wave_size]
        prepared = prebuild_wave_bodies(items, wire_format=wire_format, clock=clock)
        wave_records, wave = _run_one_fixed_wave(
            endpoint, prepared, wave_index=wave_index, release_lead_s=release_lead_s,
            synchronization_window_s=synchronization_window_s, timeout_s=timeout_s, clock=clock, process_context=context,
        )
        records.extend(wave_records)
        waves.append(wave)
    return FixedWaveRun(tuple(records), tuple(waves), wire_format=wire_format)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a fixed service burst with an explicit, recorded wire format."""
    parser = argparse.ArgumentParser(description="Run fixed-wave UniDepth HTTP bursts")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--payload-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wave-size", required=True, type=int, choices=(8, 16))
    parser.add_argument("--wave-count", required=True, type=int)
    parser.add_argument("--wire-format", required=True, choices=("multipart", "envelope"))
    args = parser.parse_args(argv)
    try:
        from ego_annotation.serving.benchmark.manifest import load_payload_manifest

        manifest = load_payload_manifest(args.payload_manifest, expected_api=None)
        run = run_fixed_process_service_waves(
            args.endpoint, manifest, wave_size=args.wave_size, wave_count=args.wave_count,
            wire_format=args.wire_format,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, IncompleteWaveEvidenceError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse.error exits


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
