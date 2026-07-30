#!/usr/bin/env python3
"""CPU-only transport-cost benchmark for one preserved UniDepth corpus tensor.

This is a measurement/prototype tool.  It imports no Ray client, sends no HTTP
request, and never initializes CUDA.  The envelope path is an experimental
in-memory vectored representation; it is not a production wire-format change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping

# Permit direct ``python scripts/benchmark_unidepth_host_path.py`` execution from
# a source checkout without installing the package.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from ego_annotation.serving.benchmark.manifest import load_payload_manifest
from ego_annotation.serving.binary_envelope import build_binary_envelope, parse_binary_envelope
from ego_annotation.serving.router import ModelApiName
from ego_annotation.serving.transport import (
    _assemble_multipart,
    build_multipart_request,
    build_multipart_response,
    parse_multipart_request,
    parse_multipart_response,
)


DEFAULT_CORPUS = "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/benchmarks/unidepth_v22_multivideo_2000_20260720T202304Z/unidepth.infer.json"
WARMUP_ITERATIONS = 20


def _summary(samples_ns: list[int]) -> dict[str, float]:
    ordered = sorted(samples_ns)
    count = len(ordered)
    if not count:
        raise ValueError("benchmark has no samples")

    def percentile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, count - 1)
        return (ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)) / 1_000_000.0

    return {
        "mean_ms": statistics.fmean(samples_ns) / 1_000_000.0,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
    }


def _measure(operation: Callable[[], object], iterations: int) -> dict[str, float]:
    for _ in range(WARMUP_ITERATIONS):
        operation()
    samples: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = operation()
        elapsed = time.perf_counter_ns() - started
        if result is None:
            raise RuntimeError("benchmark operation unexpectedly returned None")
        samples.append(elapsed)
    return _summary(samples)


def _metadata(item_id: str) -> dict[str, object]:
    return {
        "schema_version": "ego.model-service.v1",
        "ownership": {
            "request_id": "host-path-benchmark-request",
            "job_id": "host-path-benchmark",
            "item_id": item_id,
            "stage_id": "unidepth.infer",
            "source_id": "preserved-corpus",
        },
        "model_revision": "unidepth-v2-vitl14-corrected",
    }


def _response_metadata(item_id: str) -> dict[str, object]:
    return {
        "ownership": {"request_id": "host-path-benchmark-request", "item_id": item_id},
        "result": {"model_revision": "unidepth-v2-vitl14-corrected"},
    }


def _envelope_parts(metadata: Mapping[str, object], arrays: Mapping[str, tuple[object, tuple[int, ...], str]]) -> dict[str, tuple[object, tuple[int, ...], str]]:
    result: dict[str, tuple[object, tuple[int, ...], str]] = {
        "metadata": (json.dumps(metadata, separators=(",", ":")).encode("utf-8"), (), "application/json"),
    }
    result.update(arrays)
    return result


def _load_fixture(corpus_path: Path) -> tuple[dict[str, object], bytes, tuple[int, int, int], dict[str, tuple[object, tuple[int, ...], str]]]:
    manifest = load_payload_manifest(corpus_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=1)
    item = manifest.items[0]
    rgb_part = item.parts[0]
    rgb = rgb_part.data
    if rgb_part.shape != (540, 960, 3) or rgb_part.dtype != "uint8" or len(rgb) != 1_555_200:
        raise ValueError("preserved corpus fixture must be a 540x960x3 uint8 tensor (1,555,200 bytes)")

    # Values derive from the preserved RGB tensor outside the timed region.  The
    # transport cost depends on shape/dtype/byte count, while this preserves real
    # corpus-backed memory rather than introducing a synthetic random payload.
    real_rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(rgb_part.shape)
    depth = np.ascontiguousarray(real_rgb[..., 0], dtype=np.float32) / np.float32(255.0) + np.float32(0.1)
    confidence = np.ascontiguousarray(real_rgb[..., 1], dtype=np.float32) / np.float32(255.0)
    intrinsics = np.array(((480.0, 0.0, 479.5), (0.0, 480.0, 269.5), (0.0, 0.0, 1.0)), dtype=np.float32)
    arrays: dict[str, tuple[object, tuple[int, ...], str]] = {
        "depth_m": (depth, tuple(depth.shape), "float32"),
        "K_px": (intrinsics, tuple(intrinsics.shape), "float32"),
        "confidence": (confidence, tuple(confidence.shape), "float32"),
    }
    return _metadata(item.item_id), rgb, rgb_part.shape, arrays


def _table_row(name: str, current: dict[str, float], prototype: dict[str, float]) -> dict[str, object]:
    saved = current["mean_ms"] - prototype["mean_ms"]
    return {
        "component": name,
        "current_stack_ms": current,
        "prototype_ms": prototype,
        "projected_saving_ms_per_request": saved,
        "projected_saving_ms_per_s_at_15_img_s": saved * 15.0,
    }


def run_benchmark(corpus_path: Path, iterations: int) -> dict[str, Any]:
    request_metadata, rgb, rgb_shape, response_arrays = _load_fixture(corpus_path)
    response_metadata = _response_metadata(str(request_metadata["ownership"]["item_id"]))
    rgb_array = np.frombuffer(rgb, dtype=np.uint8).reshape(rgb_shape)
    request_arrays = {"rgb": (rgb_array, rgb_shape, "uint8")}

    response_bytes = {
        name: (np.ascontiguousarray(value).tobytes(), shape, dtype)
        for name, (value, shape, dtype) in response_arrays.items()
    }
    request_body, request_content_type = build_multipart_request(request_metadata, rgb=rgb, rgb_shape=rgb_shape, rgb_dtype="uint8")
    response_body, response_content_type = build_multipart_response(response_metadata, response_bytes)
    request_envelope = build_binary_envelope(_envelope_parts(request_metadata, request_arrays))
    response_envelope = build_binary_envelope(_envelope_parts(response_metadata, response_arrays))

    response_parts = _envelope_parts(response_metadata, response_bytes)
    response_multipart_specs = [("metadata", response_parts["metadata"][0], {"content_type": "application/json"})]
    response_multipart_specs.extend(
        (name, data, {"shape": shape, "dtype": dtype}) for name, (data, shape, dtype) in response_bytes.items()
    )

    two_mib = np.arange((2 * 1024 * 1024) // 4, dtype=np.float32).reshape(512, 1024)
    noncontiguous_two_mib = two_mib[:, ::-1]

    current_request_build = _measure(
        lambda: build_multipart_request(request_metadata, rgb=rgb, rgb_shape=rgb_shape, rgb_dtype="uint8"), iterations
    )
    prototype_request_build = _measure(
        lambda: build_binary_envelope(_envelope_parts(request_metadata, request_arrays)), iterations
    )
    current_request_parse = _measure(lambda: parse_multipart_request(request_body, request_content_type), iterations)
    prototype_request_parse = _measure(
        lambda: (lambda env: (json.loads(bytes(env.parts[0].data)), env.parts[1].data, env.parts[1].shape, env.parts[1].dtype))(
            parse_binary_envelope(request_envelope.header, tuple(part.data for part in request_envelope.parts))
        ),
        iterations,
    )
    current_response_build = _measure(lambda: build_multipart_response(response_metadata, response_bytes), iterations)
    prototype_response_build = _measure(
        lambda: build_binary_envelope(_envelope_parts(response_metadata, response_arrays)), iterations
    )
    current_response_parse = _measure(lambda: parse_multipart_response(response_body, response_content_type), iterations)
    prototype_response_parse = _measure(
        lambda: (lambda env: (json.loads(bytes(env.parts[0].data)), {part.name: (part.data, part.shape, part.dtype) for part in env.parts[1:]}))(
            parse_binary_envelope(response_envelope.header, tuple(part.data for part in response_envelope.parts))
        ),
        iterations,
    )
    tensor_copy = _measure(lambda: np.ascontiguousarray(noncontiguous_two_mib).tobytes(), iterations)
    tensor_view = _measure(lambda: memoryview(two_mib).cast("B"), iterations)
    multipart_concat = _measure(lambda: _assemble_multipart("hostpath", response_multipart_specs), iterations)
    envelope_header = _measure(lambda: build_binary_envelope(response_parts), iterations)
    fp16_convert = _measure(lambda: (response_arrays["depth_m"][0].astype(np.float16), response_arrays["confidence"][0].astype(np.float16)), iterations)

    rows = [
        _table_row("request codec: build 1.5 MB RGB body", current_request_build, prototype_request_build),
        _table_row("request codec: parse 1.5 MB RGB body", current_request_parse, prototype_request_parse),
        _table_row("response codec: assemble 4.2 MB float32 body", current_response_build, prototype_response_build),
        _table_row("response codec: parse 4.2 MB float32 body", current_response_parse, prototype_response_parse),
        _table_row("2 MiB tensor materialization: ascontiguousarray().tobytes()", tensor_copy, tensor_view),
        _table_row("response framing: full multipart bytearray concatenation", multipart_concat, envelope_header),
    ]
    depth_confidence_f32_bytes = sum(response_arrays[name][0].nbytes for name in ("depth_m", "confidence"))
    depth_confidence_f16_bytes = depth_confidence_f32_bytes // 2
    return {
        "schema": "ego.unidepth-host-path-benchmark.v1",
        "mode": "cpu-only/no-ray/no-http/no-gpu",
        "iterations": iterations,
        "warmup_iterations": WARMUP_ITERATIONS,
        "corpus": {"manifest": str(corpus_path), "item_id": request_metadata["ownership"]["item_id"], "rgb_bytes": len(rgb)},
        "wire_sizes": {
            "request_multipart_bytes": len(request_body),
            "request_tensor_bytes": len(rgb),
            "response_multipart_bytes": len(response_body),
            "response_tensor_bytes": sum(len(data) for data, _, _ in response_bytes.values()),
            "response_float32_depth_confidence_bytes": depth_confidence_f32_bytes,
            "response_float16_depth_confidence_bytes": depth_confidence_f16_bytes,
            "depth_confidence_payload_delta_bytes": depth_confidence_f32_bytes - depth_confidence_f16_bytes,
            "depth_confidence_payload_delta_percent": 50.0,
        },
        "components": rows,
        "fp16_depth_confidence_conversion_ms": fp16_convert,
        "notes": [
            "Prototype timing is build/parse of header plus vectored memoryviews; it does not include a socket syscall.",
            "Rows overlap by design: codec assembly includes full framing/copy cost, while the framing row isolates that copy lever.",
            "Depth/confidence values are float32 projections of the real preserved RGB tensor generated outside timed loops; no model inference runs.",
        ],
    }


def _format_table(result: Mapping[str, Any]) -> str:
    lines = [
        "component | current mean/p50/p95 ms | prototype mean/p50/p95 ms | saving ms/request | saving ms/s @15 img/s",
        "--- | --- | --- | --- | ---",
    ]
    for row in result["components"]:
        current = row["current_stack_ms"]
        prototype = row["prototype_ms"]
        lines.append(
            f"{row['component']} | {current['mean_ms']:.3f}/{current['p50_ms']:.3f}/{current['p95_ms']:.3f} | "
            f"{prototype['mean_ms']:.3f}/{prototype['p50_ms']:.3f}/{prototype['p95_ms']:.3f} | "
            f"{row['projected_saving_ms_per_request']:.3f} | {row['projected_saving_ms_per_s_at_15_img_s']:.3f}"
        )
    fp16 = result["fp16_depth_confidence_conversion_ms"]
    wire = result["wire_sizes"]
    lines.append(
        f"fp16 depth+confidence conversion | {fp16['mean_ms']:.3f}/{fp16['p50_ms']:.3f}/{fp16['p95_ms']:.3f} | n/a | "
        f"payload delta {wire['depth_confidence_payload_delta_bytes']} bytes ({wire['depth_confidence_payload_delta_percent']:.0f}%) | n/a"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS), help="preserved ego.benchmark-payload-source.v1 UniDepth descriptor")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True, help="result JSON path")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    result = run_benchmark(args.corpus, args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_format_table(result))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
