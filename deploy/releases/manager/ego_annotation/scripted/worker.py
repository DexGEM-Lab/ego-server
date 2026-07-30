"""Persistent local worker IPC for script-owned algorithm stages."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from ego_annotation.scripted.contracts import (
    AlgorithmRequest,
    AlgorithmResult,
    ContractError,
    FrameTimelineMetadata,
    NativeBatchTrace,
    NativeWorkDescription,
    StageMetadata,
)


class WorkerProtocolError(RuntimeError):
    """Raised when a worker violates the framed request/result contract."""


TOutput = TypeVar("TOutput")


def _timeline_from_mapping(value: Mapping[str, object]) -> FrameTimelineMetadata:
    return FrameTimelineMetadata(
        source_id=str(value["source_id"]),
        frame_indices=tuple(int(item) for item in value["frame_indices"]),  # type: ignore[index]
        timestamps_s=tuple(float(item) for item in value["timestamps_s"]),  # type: ignore[index]
        source_sha256=str(value["source_sha256"]) if value.get("source_sha256") is not None else None,
        width_px=int(value["width_px"]) if value.get("width_px") is not None else None,
        height_px=int(value["height_px"]) if value.get("height_px") is not None else None,
        fps=float(value["fps"]) if value.get("fps") is not None else None,
        timeline_mode=str(value.get("timeline_mode", "dense")),
    )


def _work_from_mapping(value: Mapping[str, object]) -> NativeWorkDescription:
    return NativeWorkDescription(
        work_unit_type=str(value["work_unit_type"]),
        compatibility_key=str(value["compatibility_key"]),
        native_batch_axis=int(value["native_batch_axis"]) if value.get("native_batch_axis") is not None else None,
        native_batch_size=int(value["native_batch_size"]),
        native_batch_cap=int(value["native_batch_cap"]),
        native_shape=tuple(int(item) for item in value["native_shape"]),  # type: ignore[index]
        chunk_length=int(value["chunk_length"]) if value.get("chunk_length") is not None else None,
        temporal_window=int(value["temporal_window"]) if value.get("temporal_window") is not None else None,
        outer_item_batch_size=int(value.get("outer_item_batch_size", 1)),
    )


def _request_from_mapping(value: Mapping[str, object]) -> AlgorithmRequest[object]:
    stage_value = value["stage"]
    timeline_value = value["timeline"]
    work_value = value["work"]
    if not isinstance(stage_value, Mapping) or not isinstance(timeline_value, Mapping) or not isinstance(work_value, Mapping):
        raise WorkerProtocolError("request metadata must be mappings")
    return AlgorithmRequest(
        algorithm_id=str(value["algorithm_id"]),
        model_revision=str(value["model_revision"]),
        case_id=str(value["case_id"]),
        item_id=str(value["item_id"]),
        source_id=str(value["source_id"]),
        timeline=_timeline_from_mapping(timeline_value),
        stage=StageMetadata(
            stage_id=str(stage_value["stage_id"]),
            owner=str(stage_value["owner"]),
            ownership_scope=str(stage_value["ownership_scope"]),
            model_revision=str(stage_value["model_revision"]),
        ),
        work=_work_from_mapping(work_value),
        input=value.get("input"),
        options=dict(value.get("options") or {}),  # type: ignore[arg-type]
    )


class PersistentScriptBackend(Generic[TOutput]):
    """Execute one typed request per line through one resident script process."""

    def __init__(self, command: Sequence[str]) -> None:
        if not command or any(not isinstance(token, str) or not token for token in command):
            raise ValueError("worker command must be a non-empty string sequence")
        self.command = tuple(command)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise WorkerProtocolError("worker pipes were not created")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._lock = threading.Lock()
        self._closed = False

    def execute(self, request: AlgorithmRequest[object]) -> AlgorithmResult[TOutput]:
        if self._closed:
            raise WorkerProtocolError("worker backend is closed")
        request_mapping = request.to_mapping()
        request_id = uuid.uuid4().hex
        frame = {"protocol": "ego.scripted.worker.v1", "request_id": request_id, "kind": "execute", "request": request_mapping}
        encoded = json.dumps(frame, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._stdin.write(encoded + "\n")
            self._stdin.flush()
            line = self._stdout.readline()
        if not line:
            code = self._process.poll()
            raise WorkerProtocolError(f"worker ended before reply (returncode={code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError("worker emitted invalid JSONL") from exc
        if not isinstance(response, Mapping) or response.get("request_id") != request_id:
            raise WorkerProtocolError("worker response request identity mismatch")
        if response.get("status") != "ok":
            raise WorkerProtocolError(str(response.get("error") or "worker rejected request"))
        result_mapping = response.get("result")
        if not isinstance(result_mapping, Mapping):
            raise WorkerProtocolError("worker result is not a mapping")
        return self._result_from_mapping(request, result_mapping)

    @staticmethod
    def _result_from_mapping(request: AlgorithmRequest[object], value: Mapping[str, object]) -> AlgorithmResult[object]:
        timeline_value = value.get("timeline")
        trace_value = value.get("native_batch_trace")
        if not isinstance(timeline_value, Mapping) or not isinstance(trace_value, Mapping):
            raise WorkerProtocolError("worker result is missing timeline or native batch trace")
        identity_fields = ("algorithm_id", "model_revision", "case_id", "item_id", "source_id")
        for field_name in identity_fields:
            if value.get(field_name) != getattr(request, field_name):
                raise WorkerProtocolError(f"worker changed request identity field {field_name}")
        trace = NativeBatchTrace(
            work_unit_type=str(trace_value["work_unit_type"]),
            compatibility_key=str(trace_value["compatibility_key"]),
            native_shape=tuple(int(item) for item in trace_value["native_shape"]),  # type: ignore[index]
            native_batch_axis=int(trace_value["native_batch_axis"]) if trace_value.get("native_batch_axis") is not None else None,
            native_batch_size=int(trace_value["native_batch_size"]),
            native_batch_cap=int(trace_value["native_batch_cap"]),
            execution_units=int(trace_value["execution_units"]),
        )
        result = AlgorithmResult.from_request(
            request,
            output=value.get("output"),
            uncertainty=dict(value.get("uncertainty") or {}),  # type: ignore[arg-type]
            visibility=dict(value.get("visibility") or {}),  # type: ignore[arg-type]
            native_batch_trace=trace,
            provenance=tuple(dict(item) for item in (value.get("provenance") or ())),  # type: ignore[arg-type]
        )
        if result.timeline.to_mapping() != _timeline_from_mapping(timeline_value).to_mapping():
            raise WorkerProtocolError("worker changed the source timeline")
        if result.algorithm_id != request.algorithm_id or result.item_id != request.item_id:
            raise WorkerProtocolError("worker changed request identity")
        if result.native_batch_trace != NativeBatchTrace.from_work(request.work):
            raise WorkerProtocolError("worker changed native batch semantics")
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.close()
        finally:
            self._process.terminate()
            self._process.wait()
            self._stdout.close()

    def __enter__(self) -> "PersistentScriptBackend[TOutput]":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _echo_result(request: AlgorithmRequest[object]) -> dict[str, object]:
    result = AlgorithmResult.from_request(
        request,
        output={"echo": request.input},
        provenance=({"worker": "echo", "execution_units": 1},),
    )
    return result.to_mapping()


def run_worker(*, echo: bool) -> int:
    if not echo:
        raise SystemExit("a concrete script handler is required")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
            if not isinstance(frame, Mapping) or frame.get("kind") != "execute":
                raise WorkerProtocolError("unsupported worker frame")
            request_value = frame.get("request")
            if not isinstance(request_value, Mapping):
                raise WorkerProtocolError("worker request is not a mapping")
            request = _request_from_mapping(request_value)
            response: dict[str, object] = {
                "request_id": frame.get("request_id"),
                "status": "ok",
                "result": _echo_result(request),
            }
        except (ContractError, KeyError, TypeError, ValueError, WorkerProtocolError) as exc:
            response = {"request_id": frame.get("request_id") if isinstance(locals().get("frame"), Mapping) else None, "status": "error", "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo", action="store_true")
    args = parser.parse_args(argv)
    return run_worker(echo=args.echo)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PersistentScriptBackend", "WorkerProtocolError", "main"]
