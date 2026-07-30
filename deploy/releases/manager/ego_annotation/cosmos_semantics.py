"""Source-backed, bounded Cosmos semantic reasoning for one immutable timeline.

The semantic lane is deliberately independent from physical state.  It samples
source frames, asks Cosmos for a strict tagged text grammar, and produces only full-timeline
caption rows with explicit semantic provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ego_annotation.api_routes import route_for
from ego_annotation.scripted.contracts import AlgorithmRequest, AlgorithmResult, NativeWorkDescription, StageMetadata
from ego_annotation.typed_contracts import BinaryAsset, CosmosGeneration, CosmosInput, CosmosOutput, Ownership


class CosmosSemanticError(RuntimeError):
    """A source, request, service-result, or coverage contract failed."""

    def __init__(self, message: str, *, attempts: Sequence[Mapping[str, object]] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(dict(attempt) for attempt in attempts)


HAND_KEYS = ("left_hand", "right_hand")
HAND_FIELDS = ("in_frame", "contact", "object", "rigidity", "assembly", "contact_location")
ENUMS = {
    "in_frame": {"yes", "no", "unknown"},
    "contact": {"yes", "no", "unknown"},
    "rigidity": {"rigid", "flexible", "mixed", "unknown"},
    "assembly": {"yes", "no", "unknown"},
}

COARSE_REQUEST_MEDIA_COUNT = 8

COARSE_PROMPT = """Return exactly __MEDIA_COUNT__ tagged plain-text records and nothing else. Do not use JSON, quotes, brackets, Markdown, prose before/after records, or blank lines. Return I=0 through I=__LAST_SLOT__ exactly once, in order, one line each.
Each line has exactly these keys once and in exactly this order: I,LI,LC,LO,LR,LA,LL,RI,RC,RO,RR,RA,RL,U. Use a pipe between key=value pairs.
Concrete valid example: I=0|LI=no|LC=no|LO=footwear|LR=flexible|LA=unknown|LL=floor|RI=no|RC=no|RO=unknown|RR=unknown|RA=unknown|RL=drawer|U=footwear visible near a drawer
LI, LC, LA, RI, RC, RA must each be exactly yes, no, or unknown. LR and RR must each be exactly rigid, flexible, mixed, or unknown. LO, LL, RO, RL, U must be nonempty bounded text. Do not emit literal placeholder words contact, object, rigidity, assembly, in_frame, or contact_location as values. The characters |, ~, and newline are forbidden inside values. Do not use disassembled.
IMAGE_GRID: __IMAGE_GRID__"""

BOUNDARY_PROMPT = """Return exactly one tagged plain-text boundary record and nothing else. Do not use JSON, quotes, brackets, Markdown, prose before/after the record, or blank lines.
Exact grammar: S=local_slot|C=confidence|E=short evidence
Concrete valid example: S=7|C=0.90|E=hand first holds the notebook
S must be one supplied request-local slot. C must be a finite decimal from 0 through 1. E must be nonempty bounded evidence text and cannot contain |, ~, or newlines.
PREVIOUS_STATE: __PREVIOUS_STATE__
NEXT_STATE: __NEXT_STATE__
REFINEMENT_GRID: __IMAGE_GRID__"""

COARSE_REPAIR_PROMPT = """CORRECTION REQUEST. The prior response failed this local validation: {validation_error}
Return exactly __MEDIA_COUNT__ plain-text records, I=0 through I=__LAST_SLOT__ in order, one line each: no JSON, brackets, quotes, Markdown, prose, blank lines, missing, extra, or reordered keys.
Each line must exactly use I|LI|LC|LO|LR|LA|LL|RI|RC|RO|RR|RA|RL|U in that order. Concrete valid line: I=0|LI=no|LC=no|LO=footwear|LR=flexible|LA=unknown|LL=floor|RI=no|RC=no|RO=unknown|RR=unknown|RA=unknown|RL=drawer|U=footwear visible near a drawer
LI/LC/LA/RI/RC/RA are yes|no|unknown; LR/RR are rigid|flexible|mixed|unknown; LO/LL/RO/RL/U are bounded nonempty delimiter-free strings. Never emit placeholders such as contact, object, rigidity, assembly, in_frame, contact_location, or disassembled.
IMAGE_GRID: __IMAGE_GRID__"""

BOUNDARY_REPAIR_PROMPT = """CORRECTION REQUEST. The prior response failed this local validation: {validation_error}
Return exactly one plain-text record, no JSON, Markdown, prose, or blank lines: S=local_slot|C=confidence|E=short evidence. Concrete valid line: S=7|C=0.90|E=hand first holds the notebook. S is one supplied local slot 0 through {last_slot}; C is finite decimal [0,1]; E is bounded nonempty delimiter-free text.
REFINEMENT_GRID: __IMAGE_GRID__"""


class TimelineLike(Protocol):
    source_id: str
    source_sha256: str
    frame_count: int
    fps: float
    width_px: int
    height_px: int

    def metadata(self, indices: Sequence[int] | None = None): ...


class FrameSourceLike(Protocol):
    timeline: TimelineLike

    def read_rgb(self, frame_index: int) -> np.ndarray: ...


class StageClientLike(Protocol):
    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]: ...


@dataclass(frozen=True)
class CosmosCoarseParse:
    """A structurally valid coarse response plus nonfatal enum anomalies."""

    rows: tuple[dict[str, object], ...]
    anomalies: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class CosmosBoundaryParse:
    """A boundary record with an explicit no-enum anomaly contract.

    The boundary grammar has only slot/confidence/evidence fields.  It carries no
    typed hand enum and therefore cannot normalize one silently.
    """

    record: Mapping[str, object]
    anomalies: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.anomalies:
            raise CosmosSemanticError("boundary grammar cannot carry typed enum anomalies")


@dataclass(frozen=True)
class CosmosSemanticResult:
    rows: tuple[Mapping[str, object], ...]
    request_count: int
    coarse_request_count: int
    refinement_request_count: int
    repair_request_count: int
    sampled_source_indices: tuple[int, ...]
    outputs: tuple[CosmosOutput, ...]
    refinements: tuple[Mapping[str, object], ...]
    attempts: tuple[Mapping[str, object], ...]
    anomalies: tuple[Mapping[str, object], ...]
    coarse_rows: tuple[Mapping[str, object], ...] = ()
    segments: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        validate_semantic_coverage(self.rows, self.rows[-1]["end_frame"] if self.rows else 0)
        if self.request_count != self.coarse_request_count + self.refinement_request_count + self.repair_request_count:
            raise CosmosSemanticError("Cosmos request accounting mismatch")
        if len(self.outputs) != self.request_count or len(self.attempts) != self.request_count:
            raise CosmosSemanticError("Cosmos response trace accounting mismatch")


def sample_source_indices(frame_count: int, fps: float) -> tuple[int, ...]:
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise CosmosSemanticError("source timeline must be positive")
    sample_count = max(1, int(math.ceil(frame_count / fps - 1e-9)))
    indices: list[int] = []
    for second in range(sample_count):
        index = min(frame_count - 1, int(math.floor(second * fps + 0.5)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return tuple(indices)


def group_bounded(indices: Sequence[int], size: int = 8) -> tuple[tuple[int, ...], ...]:
    if size <= 0 or size > 8:
        raise CosmosSemanticError("Cosmos media groups must contain 1..8 images")
    return tuple(tuple(int(value) for value in indices[start:start + size]) for start in range(0, len(indices), size))


def refinement_indices(start_frame: int, end_frame: int, *, limit: int = 8) -> tuple[int, ...]:
    if start_frame < 0 or end_frame <= start_frame or limit < 2 or limit > 8:
        raise CosmosSemanticError("transition refinement bounds are invalid")
    count = min(limit, end_frame - start_frame + 1)
    values = np.linspace(start_frame, end_frame, num=count)
    indices = tuple(dict.fromkeys(int(round(value)) for value in values))
    if indices[0] != start_frame or indices[-1] != end_frame:
        raise CosmosSemanticError("transition refinement must preserve interval endpoints")
    return indices


def _jpeg(frame_rgb: np.ndarray, width: int) -> bytes:
    import cv2

    frame = np.asarray(frame_rgb)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise CosmosSemanticError("Cosmos source frame must be uint8 RGB")
    if width <= 0:
        raise CosmosSemanticError("Cosmos gallery width must be positive")
    if frame.shape[1] > width:
        height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok or encoded.size == 0:
        raise CosmosSemanticError("JPEG encoding failed")
    return bytes(encoded)


def _grid(indices: Sequence[int], fps: float, *, image_offset: int = 0) -> str:
    return "; ".join(f"{image_offset + slot}@{index}@{index / fps:.6f}" for slot, index in enumerate(indices))


def _coarse_request_grid(media_count: int) -> str:
    if not 1 <= media_count <= COARSE_REQUEST_MEDIA_COUNT:
        raise CosmosSemanticError("coarse Cosmos request must contain 1..8 images")
    return "; ".join(f"{slot}@{slot}@{float(slot):.6f}" for slot in range(media_count))


def _coarse_prompt(template: str, media_count: int) -> str:
    if not 1 <= media_count <= COARSE_REQUEST_MEDIA_COUNT:
        raise CosmosSemanticError("coarse Cosmos request must contain 1..8 images")
    return (template
            .replace("__MEDIA_COUNT__", str(media_count))
            .replace("__LAST_SLOT__", str(media_count - 1))
            .replace("__IMAGE_GRID__", _coarse_request_grid(media_count)))


def _complete_coarse_group(sampled: Sequence[int], group_index: int, group: Sequence[int]) -> tuple[tuple[int, ...], int, int]:
    """Return real tail frames plus the available distinct preceding context.

    Cosmos accepts one through eight images. Short videos therefore use their
    complete sampled timeline directly; an undersized tail uses as much real
    preceding context as exists rather than inventing duplicate padding.
    """
    real = tuple(group)
    group_start = group_index * COARSE_REQUEST_MEDIA_COUNT
    if not 1 <= len(real) <= COARSE_REQUEST_MEDIA_COUNT or tuple(sampled[group_start:group_start + len(real)]) != real:
        raise CosmosSemanticError("coarse Cosmos group does not match the sampled timeline")
    prefix_count = COARSE_REQUEST_MEDIA_COUNT - len(real)
    prior = tuple(sampled[:group_start])
    context = prior[-min(prefix_count, len(prior)):] if prefix_count else ()
    request_indices = context + real
    request_offset, retain_from = group_start - len(context), len(context)
    if (not 1 <= len(request_indices) <= COARSE_REQUEST_MEDIA_COUNT
            or any(type(index) is not int or index < 0 for index in request_indices)
            or tuple(sorted(set(request_indices))) != request_indices):
        raise CosmosSemanticError("coarse Cosmos request must contain distinct increasing source frames")
    return request_indices, request_offset, retain_from


def _request(
    source: FrameSourceLike,
    indices: tuple[int, ...],
    *,
    prompt: str,
    case_id: str,
    item_id: str,
    revision: str,
    scope: str,
    gallery_width: int,
    max_tokens: int,
) -> AlgorithmRequest[CosmosInput]:
    timeline = source.timeline
    assets = tuple(
        BinaryAsset(
            _jpeg(source.read_rgb(index), gallery_width),
            "image/jpeg",
            f"{timeline.source_id}:frame:{index:06d}",
            (index,),
        )
        for index in indices
    )
    ownership = Ownership(case_id, item_id, timeline.source_id, route_for("cosmos3.reason").owner, f"cosmos3.reason:{scope}")
    value = CosmosInput(ownership, prompt, (), CosmosGeneration(max_tokens), assets, indices)
    envelope_indices = tuple(range(indices[0], indices[-1] + 1))
    route = route_for("cosmos3.reason")
    return AlgorithmRequest(
        algorithm_id="cosmos3.reason",
        model_revision=revision,
        case_id=case_id,
        item_id=item_id,
        source_id=timeline.source_id,
        timeline=timeline.metadata(envelope_indices),
        stage=StageMetadata("cosmos3.reason", route.owner, ownership.scope, revision),
        work=NativeWorkDescription("cosmos3.reason", f"cosmos3.reason:{revision}:{scope}", None, 1, route.native_batch_cap, (1,), outer_item_batch_size=1),
        input=value,
    )


def _require_complete_output(output: CosmosOutput) -> None:
    if output.finish_reason != "stop":
        raise CosmosSemanticError(f"Cosmos generation was truncated or incomplete: finish_reason={output.finish_reason!r}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


_WHOLE_JSON_FENCE = re.compile(
    r"\A\s*```(?:json|JSON)\r?\n(?P<body>.*?)(?:\r?\n)```\s*\Z",
    re.DOTALL,
)


def _strict_json(text: str) -> object:
    if not isinstance(text, str):
        raise CosmosSemanticError("Cosmos response must be JSON text")
    candidate = text
    try:
        return json.loads(
            candidate,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as raw_exc:
        fence = _WHOLE_JSON_FENCE.fullmatch(text)
        if fence is None:
            raise CosmosSemanticError(f"Cosmos returned malformed, duplicate-key, or non-finite JSON: {raw_exc}") from raw_exc
        candidate = fence.group("body")
    try:
        return json.loads(
            candidate,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CosmosSemanticError(f"Cosmos returned malformed, duplicate-key, or non-finite JSON: {exc}") from exc


def _strict_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CosmosSemanticError(f"{context} must be a finite numeric primitive")
    return float(value)


def _strict_hand(value: object, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(HAND_FIELDS):
        raise CosmosSemanticError(f"{context} must contain the exact hand-state fields")
    result: dict[str, str] = {}
    for field in HAND_FIELDS:
        raw = value[field]
        if not isinstance(raw, str) or not raw.strip():
            raise CosmosSemanticError(f"{context}.{field} must be non-empty text")
        text = raw.strip()
        if field in ENUMS and text not in ENUMS[field]:
            raise CosmosSemanticError(f"{context}.{field} has unsupported label {text!r}")
        if field in {"object", "contact_location"} and len(text) > 120:
            raise CosmosSemanticError(f"{context}.{field} is unbounded")
        result[field] = text
    return result


_WHOLE_TEXT_FENCE = re.compile(r"\A```text\r?\n(?P<body>.*?)(?:\r?\n)```\Z", re.DOTALL)
_KEYED_FIELDS = ("I", "LI", "LC", "LO", "LR", "LA", "LL", "RI", "RC", "RO", "RR", "RA", "RL", "U")
_LITERAL_PLACEHOLDERS = {"contact", "object", "rigidity", "assembly", "in_frame", "contact_location", "slot", "understanding"}
_BOUNDARY_KEYED = re.compile(r"\AS=(?P<slot>[0-9]+)\|C=(?P<confidence>(?:0(?:\.\d+)?|1(?:\.0+)?))\|E=(?P<evidence>[^|~\r\n]+)\Z")


def _tagged_text(text: str, context: str) -> str:
    if not isinstance(text, str):
        raise CosmosSemanticError(f"Cosmos {context} response must be text")
    match = _WHOLE_TEXT_FENCE.fullmatch(text)
    candidate = match.group("body") if match else text
    if candidate.startswith(("\n", "\r")) or candidate.endswith(("\n", "\r")):
        raise CosmosSemanticError(f"Cosmos {context} response has blank or extra lines")
    return candidate


def _tagged_value(value: str, context: str, *, max_length: int, allow_placeholder: bool = False) -> str:
    if not value.strip() or len(value.strip()) > max_length or any(char in value for char in "|~\r\n"):
        raise CosmosSemanticError(f"{context} must be bounded nonempty delimiter-free text")
    result = value.strip()
    if not allow_placeholder and result.lower() in _LITERAL_PLACEHOLDERS:
        raise CosmosSemanticError(f"{context} echoes literal field placeholder {result!r}")
    return result


def parse_coarse_output_with_anomalies(text: str, indices: Sequence[int], fps: float, *, image_offset: int) -> CosmosCoarseParse:
    """Parse the exact keyed grammar while preserving unsupported enum evidence.

    Grammar, cardinality, order, identity, delimiters, and text bounds are
    transport/structure contracts and remain fail-closed.  An unsupported value
    in a typed enum is instead a model-semantic anomaly: its typed value becomes
    ``unknown`` and the exact wire value is retained for review.
    """
    source_indices = tuple(indices)
    if not 1 <= len(source_indices) <= COARSE_REQUEST_MEDIA_COUNT:
        raise CosmosSemanticError("coarse Cosmos rows must bind 1..8 real source frames")
    if any(type(index) is not int or index < 0 for index in source_indices) or tuple(sorted(set(source_indices))) != source_indices:
        raise CosmosSemanticError("coarse Cosmos rows must bind strictly increasing source frames")
    lines = _tagged_text(text, "coarse").splitlines()
    if len(lines) != len(source_indices) or any(not line for line in lines):
        raise CosmosSemanticError(f"Cosmos coarse output must contain exactly {len(source_indices)} nonempty keyed lines")
    rows: list[dict[str, object]] = []
    anomalies: list[Mapping[str, object]] = []
    enum_fields = {
        "LI": ("left_hand", "in_frame", {"yes", "no", "unknown"}),
        "LC": ("left_hand", "contact", {"yes", "no", "unknown"}),
        "LA": ("left_hand", "assembly", {"yes", "no", "unknown"}),
        "RI": ("right_hand", "in_frame", {"yes", "no", "unknown"}),
        "RC": ("right_hand", "contact", {"yes", "no", "unknown"}),
        "RA": ("right_hand", "assembly", {"yes", "no", "unknown"}),
        "LR": ("left_hand", "rigidity", {"rigid", "flexible", "mixed", "unknown"}),
        "RR": ("right_hand", "rigidity", {"rigid", "flexible", "mixed", "unknown"}),
    }
    for slot, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) != len(_KEYED_FIELDS):
            raise CosmosSemanticError("Cosmos coarse keyed line has missing or extra fields")
        fields: dict[str, str] = {}
        for key, part in zip(_KEYED_FIELDS, parts):
            prefix = key + "="
            if not part.startswith(prefix) or part.count("=") != 1:
                raise CosmosSemanticError("Cosmos coarse keyed field order or identity changed")
            fields[key] = _tagged_value(part[len(prefix):], f"items[{slot}].{key}", max_length=240 if key == "U" else 120, allow_placeholder=key == "I")
        if fields["I"] != str(slot):
            raise CosmosSemanticError("Cosmos coarse output changed request-local identity")
        source_index = source_indices[slot]
        for key, (hand_key, typed_field, allowed) in enum_fields.items():
            raw_value = fields[key]
            if raw_value not in allowed:
                anomalies.append({
                    "kind": "unsupported_enum",
                    "raw_field": key,
                    "typed_path": f"{hand_key}.{typed_field}",
                    "raw_value": raw_value,
                    "normalized_value": "unknown",
                    "request_local_slot": slot,
                    "source_frame_idx": source_index,
                })
                fields[key] = "unknown"
        if len(fields["U"].split()) > 18:
            raise CosmosSemanticError("Cosmos understanding exceeds eighteen words")
        left = {"in_frame": fields["LI"], "contact": fields["LC"], "object": fields["LO"], "rigidity": fields["LR"], "assembly": fields["LA"], "contact_location": fields["LL"]}
        right = {"in_frame": fields["RI"], "contact": fields["RC"], "object": fields["RO"], "rigidity": fields["RR"], "assembly": fields["RA"], "contact_location": fields["RL"]}
        rows.append({"image_index": image_offset + slot, "frame_idx": source_index, "time_sec": source_index / fps, "left_hand": left, "right_hand": right, "understanding": fields["U"]})
    return CosmosCoarseParse(tuple(rows), tuple(anomalies))


def parse_coarse_output(text: str, indices: Sequence[int], fps: float, *, image_offset: int) -> tuple[dict[str, object], ...]:
    """Compatibility view for callers that only need typed normalized rows."""
    return parse_coarse_output_with_anomalies(text, indices, fps, image_offset=image_offset).rows


def _signature(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[hand][field] for hand in HAND_KEYS for field in HAND_FIELDS)  # type: ignore[index]


def parse_boundary_output_with_anomalies(text: str, indices: Sequence[int]) -> CosmosBoundaryParse:
    candidate = _tagged_text(text, "boundary")
    if "\n" in candidate or "\r" in candidate:
        raise CosmosSemanticError("Cosmos boundary output must contain exactly one keyed line")
    match = _BOUNDARY_KEYED.fullmatch(candidate)
    if match is None:
        raise CosmosSemanticError("Cosmos boundary output must match exact S/C/E keyed grammar")
    slot = int(match.group("slot"))
    if not 0 <= slot < len(indices):
        raise CosmosSemanticError("Cosmos boundary must select one request-local submitted slot")
    confidence = _strict_finite_number(float(match.group("confidence")), "Cosmos boundary confidence")
    if not 0 <= confidence <= 1:
        raise CosmosSemanticError("Cosmos boundary confidence must be in [0,1]")
    evidence = _tagged_value(match.group("evidence"), "Cosmos boundary evidence", max_length=240)
    if len(evidence.split()) > 20:
        raise CosmosSemanticError("Cosmos boundary evidence exceeds twenty words")
    return CosmosBoundaryParse({"change_frame_idx": indices[slot], "request_local_frame_idx": slot, "confidence": confidence, "evidence": evidence, "sampled_source_indices": list(indices)})


def parse_boundary_output(text: str, indices: Sequence[int]) -> dict[str, object]:
    """Compatibility view for the boundary record; grammar has no enum fields."""
    return dict(parse_boundary_output_with_anomalies(text, indices).record)


def _hand_action(hand: Mapping[str, object]) -> str:
    if hand.get("in_frame") == "no":
        return "no_action"
    if hand.get("contact") == "yes":
        return "contacting_or_operating_object"
    if hand.get("in_frame") == "yes":
        return "visible_no_contact"
    return "unknown"


def _caption(row: Mapping[str, object]) -> str:
    """Create the stable product caption from typed per-hand state."""
    parts: list[str] = []
    for label, key in (("Left", "left_hand"), ("right", "right_hand")):
        hand = row[key]
        if not isinstance(hand, Mapping):
            raise CosmosSemanticError(f"{key} is unavailable for caption construction")
        obj = str(hand.get("object") or "unknown").strip()
        parts.append(f"{label} hand {_hand_action(hand)} {obj}")
    return "; ".join(parts) + "."


def _expanded_coarse_row(row: Mapping[str, object]) -> dict[str, object]:
    expanded: dict[str, object] = {
        "image_index": int(row["image_index"]),
        "frame_idx": int(row["frame_idx"]),
        "time_sec": float(row["time_sec"]),
        "understanding": str(row.get("understanding") or ""),
    }
    for hand_key in HAND_KEYS:
        hand = row[hand_key]
        if not isinstance(hand, Mapping):
            raise CosmosSemanticError(f"{hand_key} is unavailable for expanded review output")
        for field in HAND_FIELDS:
            expanded[f"{hand_key}_{field}"] = str(hand[field])
    anomalies = row.get("semantic_anomalies")
    if anomalies:
        expanded["semantic_anomalies"] = [dict(item) for item in anomalies if isinstance(item, Mapping)]
    return expanded


def _compact_label(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "image_index": int(row["image_index"]),
        "frame_idx": int(row["frame_idx"]),
        "time_sec": float(row["time_sec"]),
        "left_hand": dict(row["left_hand"]),  # type: ignore[arg-type]
        "right_hand": dict(row["right_hand"]),  # type: ignore[arg-type]
    }


def _coarse_response_family(text: str, output: CosmosOutput, *, media_count: int) -> dict[str, object]:
    """Describe the tagged wire family for audit; parsing remains authoritative."""
    candidate = _WHOLE_TEXT_FENCE.fullmatch(text)
    lines = (candidate.group("body") if candidate else text).splitlines()
    record: dict[str, object] = {
        "root": "tagged_lines",
        "envelope": "whole_text_fence" if candidate else "bare_text",
        "cardinality": f"lines:{len(lines)}",
        "identity": "unavailable",
        "schema": "unavailable",
        "finish": output.finish_reason,
        "tokens": {"prompt": output.prompt_tokens, "completion": output.completion_tokens, "total": output.total_tokens},
        "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    try:
        parsed = parse_coarse_output(text, tuple(range(media_count)), 1.0, image_offset=0)
    except CosmosSemanticError as exc:
        record["schema"] = f"rejected:{exc}"
    else:
        record["identity"] = "exact_request_local_slots"
        record["schema"] = "exact_keyed_tagged_schema"
        record["cardinality"] = f"lines:{len(parsed)}"
    return record


def _boundary_response_family(text: str, output: CosmosOutput) -> dict[str, object]:
    candidate = _WHOLE_TEXT_FENCE.fullmatch(text)
    record = {
        "root": "tagged_line",
        "envelope": "whole_text_fence" if candidate else "bare_text",
        "cardinality": "one_line" if "\n" not in (candidate.group("body") if candidate else text) else "multiple_lines",
        "identity": "unavailable",
        "schema": "unavailable",
        "finish": output.finish_reason,
        "tokens": {"prompt": output.prompt_tokens, "completion": output.completion_tokens, "total": output.total_tokens},
        "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    try:
        parse_boundary_output(text, tuple(range(COARSE_REQUEST_MEDIA_COUNT)))
    except CosmosSemanticError as exc:
        record["schema"] = f"rejected:{exc}"
    else:
        record["identity"] = "exact_request_local_slot"
        record["schema"] = "exact_keyed_tagged_schema"
    return record


def _attempt_record(
    request: AlgorithmRequest[CosmosInput],
    output: CosmosOutput,
    *,
    phase: str,
    repair_count: int,
    validation_error: str | None,
) -> dict[str, object]:
    family = (_coarse_response_family(output.text, output, media_count=len(request.input.source_frame_indices))
              if phase == "coarse" else _boundary_response_family(output.text, output))
    return {
        "scope": request.input.ownership.scope,
        "phase": phase,
        "repair_count": repair_count,
        "validation_error": validation_error,
        "prompt_sha256": hashlib.sha256(request.input.prompt.encode("utf-8")).hexdigest(),
        "source_frame_indices": list(request.input.source_frame_indices),
        "response_family": family,
    }


def _repair_error_text(error: CosmosSemanticError) -> str:
    # The repair asks about the local validator's observed failure, not model prose.
    return str(error)[:480]


def build_segments(coarse: Sequence[Mapping[str, object]], boundaries: Mapping[int, Mapping[str, object]], frame_count: int, fps: float) -> tuple[Mapping[str, object], ...]:
    if not coarse:
        raise CosmosSemanticError("Cosmos coarse state is empty")
    starts = [0]
    states = [coarse[0]]
    for index in range(1, len(coarse)):
        if _signature(coarse[index - 1]) == _signature(coarse[index]):
            continue
        boundary = boundaries.get(index)
        if boundary is None:
            raise CosmosSemanticError("changed coarse state has no boundary refinement")
        frame = boundary["change_frame_idx"]
        if type(frame) is not int or frame <= starts[-1] or frame >= frame_count:
            raise CosmosSemanticError("boundary refinement does not create positive contiguous segments")
        starts.append(frame)
        states.append(coarse[index])
    segments: list[Mapping[str, object]] = []
    for start, end, state in zip(starts, [*starts[1:], frame_count], states):
        interval_anomalies = [
            dict(anomaly)
            for sample in coarse
            for anomaly in sample.get("semantic_anomalies", ())
            if isinstance(anomaly, Mapping) and start <= int(anomaly["source_frame_idx"]) < end
        ]
        segments.append({
            "start_frame": start,
            "end_frame": end,
            "start_sec": start / fps,
            "end_sec": end / fps,
            "left_hand": dict(state["left_hand"]),  # type: ignore[arg-type]
            "right_hand": dict(state["right_hand"]),  # type: ignore[arg-type]
            "understanding": str(state.get("understanding") or ""),
            "source": "cosmos_gallery_until_boundary" if end < frame_count else "cosmos_gallery_after_last_boundary",
            "semantic_anomalies": interval_anomalies,
        })
    return tuple(segments)


def build_semantic_rows(coarse: Sequence[Mapping[str, object]], boundaries: Mapping[int, Mapping[str, object]], frame_count: int, fps: float) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for index, segment in enumerate(build_segments(coarse, boundaries, frame_count, fps)):
        start, end = int(segment["start_frame"]), int(segment["end_frame"])
        interval_anomalies = list(segment.get("semantic_anomalies", ()))
        per_hand: dict[str, dict[str, object]] = {}
        for side, hand_key in (("left", "left_hand"), ("right", "right_hand")):
            hand = segment[hand_key]
            if not isinstance(hand, Mapping):
                raise CosmosSemanticError(f"{hand_key} is unavailable for semantic row construction")
            per_hand[side] = {
                "action_state": _hand_action(hand),
                "in_frame": hand["in_frame"],
                "contact": hand["contact"],
                "object": hand["object"],
                "object_rigidity": hand["rigidity"],
                "object_assembly": hand["assembly"],
                "contact_location": hand["contact_location"],
            }
        rows.append({
            "clip_id": f"cosmos_segment_{index:04d}",
            "start_frame": start,
            "end_frame": end,
            "start_s": start / fps,
            "end_s": end / fps,
            "duration_s": (end - start) / fps,
            "caption": _caption(segment),
            "evidence_frames": [start, max(start, end - 1)],
            "grounding_status": "cosmos_gallery_boundary_video_understanding",
            "provenance": "Cosmos video-understanding output from sampled gallery frames and boundary refinement; object/contact words are semantic annotations, not object-pose or nonpenetration proof.",
            "source": "cosmos_gallery_boundary",
            "claim_scope": "semantic_only_not_physical_evidence",
            "per_hand": per_hand,
            "semantic_anomalies": interval_anomalies,
            "status": "completed_with_anomalies" if interval_anomalies else "completed",
        })
    validate_semantic_coverage(rows, frame_count)
    return tuple(rows)


def validate_semantic_coverage(rows: Sequence[Mapping[str, object]], frame_count: int) -> None:
    if frame_count <= 0 or not rows:
        raise CosmosSemanticError("semantic rows must cover a positive source timeline")
    cursor = 0
    for row in rows:
        start, end = row.get("start_frame"), row.get("end_frame")
        if type(start) is not int or type(end) is not int or start != cursor or end <= start or end > frame_count:
            raise CosmosSemanticError("semantic rows must be ordered, positive, contiguous, and bounded")
        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise CosmosSemanticError("semantic row caption must be non-empty")
        if row.get("claim_scope") != "semantic_only_not_physical_evidence":
            raise CosmosSemanticError("semantic evidence claim scope is invalid")
        cursor = end
    if cursor != frame_count:
        raise CosmosSemanticError("semantic rows do not cover [0,N)")


def run_cosmos_semantics(
    client: StageClientLike,
    source: FrameSourceLike,
    *,
    case_id: str,
    item_id: str,
    revision: str,
    gallery_width: int = 960,
) -> CosmosSemanticResult:
    """Run strict Cosmos semantics with exactly one prompt-corrected parse repair.

    A repair is only entered after a typed, complete Cosmos response fails the
    local JSON/schema validator. It preserves the original media and changes
    ownership scope and prompt, so it cannot be a same-prompt retry.
    """
    timeline = source.timeline
    sampled = sample_source_indices(timeline.frame_count, timeline.fps)
    outputs: list[CosmosOutput] = []
    attempts: list[dict[str, object]] = []
    repair_count = 0

    def execute_with_one_repair(
        request: AlgorithmRequest[CosmosInput],
        *,
        phase: str,
        parse: Any,
        repair_prompt: str,
    ) -> object:
        nonlocal repair_count
        result = client.execute(request)
        if not isinstance(result.output, CosmosOutput):
            raise CosmosSemanticError(f"Cosmos {phase} backend returned the wrong typed output", attempts=attempts)
        output = result.output
        outputs.append(output)
        try:
            _require_complete_output(output)
        except CosmosSemanticError as finish_error:
            attempts.append(_attempt_record(request, output, phase=phase, repair_count=0, validation_error=_repair_error_text(finish_error)))
            raise CosmosSemanticError(str(finish_error), attempts=attempts) from finish_error
        try:
            parsed = parse(output.text)
        except CosmosSemanticError as original_error:
            attempts.append(_attempt_record(request, output, phase=phase, repair_count=0, validation_error=_repair_error_text(original_error)))
            repair_count += 1
            repaired = _request(
                source,
                request.input.source_frame_indices,
                prompt=repair_prompt.replace("{validation_error}", _repair_error_text(original_error)),
                case_id=case_id,
                item_id=item_id,
                revision=revision,
                scope=f"{request.input.ownership.scope.removeprefix('cosmos3.reason:')}:repair:1",
                gallery_width=gallery_width,
                max_tokens=request.input.generation.max_tokens,
            )
            repaired_result = client.execute(repaired)
            if not isinstance(repaired_result.output, CosmosOutput):
                raise CosmosSemanticError(f"Cosmos {phase} repair backend returned the wrong typed output", attempts=attempts)
            repaired_output = repaired_result.output
            outputs.append(repaired_output)
            try:
                _require_complete_output(repaired_output)
            except CosmosSemanticError as finish_error:
                attempts.append(_attempt_record(repaired, repaired_output, phase=phase, repair_count=1, validation_error=_repair_error_text(finish_error)))
                raise CosmosSemanticError(
                    f"Cosmos {phase} repair failed after one correction: {finish_error}",
                    attempts=attempts,
                ) from finish_error
            try:
                parsed = parse(repaired_output.text)
            except CosmosSemanticError as repair_error:
                attempts.append(_attempt_record(repaired, repaired_output, phase=phase, repair_count=1, validation_error=_repair_error_text(repair_error)))
                raise CosmosSemanticError(
                    f"Cosmos {phase} repair failed after one correction: {repair_error}",
                    attempts=attempts,
                ) from repair_error
            attempts.append(_attempt_record(repaired, repaired_output, phase=phase, repair_count=1, validation_error=None))
            return parsed, repaired.input.ownership.scope
        attempts.append(_attempt_record(request, output, phase=phase, repair_count=0, validation_error=None))
        return parsed, request.input.ownership.scope

    coarse: list[dict[str, object]] = []
    anomalies: list[Mapping[str, object]] = []
    coarse_groups = group_bounded(sampled)
    for group_index, group in enumerate(coarse_groups):
        request_indices, request_offset, retain_from = _complete_coarse_group(sampled, group_index, group)
        prompt = _coarse_prompt(COARSE_PROMPT, len(request_indices))
        request = _request(source, request_indices, prompt=prompt, case_id=case_id, item_id=item_id, revision=revision, scope=f"coarse:{group_index:04d}", gallery_width=gallery_width, max_tokens=4096)
        try:
            validated, response_scope = execute_with_one_repair(
                request,
                phase="coarse",
                parse=lambda text: parse_coarse_output_with_anomalies(text, request_indices, timeline.fps, image_offset=request_offset),
                repair_prompt=_coarse_prompt(COARSE_REPAIR_PROMPT, len(request_indices)),
            )
        except CosmosSemanticError as exc:
            if not exc.attempts:
                raise CosmosSemanticError(str(exc), attempts=attempts) from exc
            raise
        if not isinstance(validated, CosmosCoarseParse):
            raise CosmosSemanticError("Cosmos coarse parser returned an invalid result", attempts=attempts)
        response_anomalies = tuple(
            {**dict(anomaly), "request_scope": response_scope}
            for anomaly in validated.anomalies
        )
        anomalies.extend(response_anomalies)
        for row in validated.rows[retain_from:]:
            bound_anomalies = tuple(
                anomaly for anomaly in response_anomalies
                if int(anomaly["source_frame_idx"]) == int(row["frame_idx"])
            )
            coarse.append({**row, "semantic_anomalies": bound_anomalies})
    if tuple(int(row["frame_idx"]) for row in coarse) != sampled:
        raise CosmosSemanticError("coarse Cosmos results do not preserve the full sample grid", attempts=attempts)

    boundaries: dict[int, Mapping[str, object]] = {}
    refinements: list[Mapping[str, object]] = []
    for index in range(1, len(coarse)):
        if _signature(coarse[index - 1]) == _signature(coarse[index]):
            continue
        indices = refinement_indices(int(coarse[index - 1]["frame_idx"]), int(coarse[index]["frame_idx"]))
        prompt = (BOUNDARY_PROMPT
                  .replace("__PREVIOUS_STATE__", json.dumps(coarse[index - 1], separators=(",", ":"), ensure_ascii=True))
                  .replace("__NEXT_STATE__", json.dumps(coarse[index], separators=(",", ":"), ensure_ascii=True))
                  .replace("__IMAGE_GRID__", _grid(indices, timeline.fps)))
        request = _request(source, indices, prompt=prompt, case_id=case_id, item_id=item_id, revision=revision, scope=f"boundary:{index:04d}", gallery_width=gallery_width, max_tokens=512)
        boundary_repair = BOUNDARY_REPAIR_PROMPT.replace("{last_slot}", str(len(indices) - 1)).replace("__IMAGE_GRID__", _grid(indices, timeline.fps))
        try:
            parsed, response_scope = execute_with_one_repair(
                request,
                phase="boundary",
                parse=lambda text: parse_boundary_output_with_anomalies(text, indices),
                repair_prompt=boundary_repair,
            )
        except CosmosSemanticError as exc:
            if not exc.attempts:
                raise CosmosSemanticError(str(exc), attempts=attempts) from exc
            raise
        if not isinstance(parsed, CosmosBoundaryParse):
            raise CosmosSemanticError("Cosmos boundary parser returned an invalid result", attempts=attempts)
        # Boundary grammar is S/C/E only. Keep an explicit empty anomaly list in
        # review artifacts so future compat changes cannot discard enum values.
        boundary_anomalies = tuple(
            {**dict(anomaly), "request_scope": response_scope}
            for anomaly in parsed.anomalies
        )
        anomalies.extend(boundary_anomalies)
        boundaries[index] = parsed.record
        confidence_score = float(parsed.record["confidence"])
        refinements.append({
            "change_index": len(refinements) + 1,
            "coarse_next_index": index,
            "prev_image_index": int(coarse[index - 1]["image_index"]),
            "next_image_index": int(coarse[index]["image_index"]),
            "range_frame_start": int(coarse[index - 1]["frame_idx"]),
            "range_frame_end": int(coarse[index]["frame_idx"]),
            "range_time_start": float(coarse[index - 1]["time_sec"]),
            "range_time_end": float(coarse[index]["time_sec"]),
            "change_frame_idx": int(parsed.record["change_frame_idx"]),
            "change_time_sec": int(parsed.record["change_frame_idx"]) / timeline.fps,
            "confidence": "high" if confidence_score >= 0.8 else "medium" if confidence_score >= 0.5 else "low",
            "confidence_score": confidence_score,
            "evidence": str(parsed.record["evidence"]),
            "previous_label": _compact_label(coarse[index - 1]),
            "next_label": _compact_label(coarse[index]),
            "sampled_source_indices": list(parsed.record["sampled_source_indices"]),
            "request_scope": response_scope,
            "typed_enum_fields": [],
            "semantic_anomalies": [dict(item) for item in boundary_anomalies],
        })

    segments = build_segments(coarse, boundaries, timeline.frame_count, timeline.fps)
    rows = build_semantic_rows(coarse, boundaries, timeline.frame_count, timeline.fps)
    return CosmosSemanticResult(
        rows=rows,
        request_count=len(outputs),
        coarse_request_count=len(coarse_groups),
        refinement_request_count=len(refinements),
        repair_request_count=repair_count,
        sampled_source_indices=sampled,
        outputs=tuple(outputs),
        refinements=tuple(refinements),
        attempts=tuple(attempts),
        anomalies=tuple(anomalies),
        coarse_rows=tuple(_expanded_coarse_row(row) for row in coarse),
        segments=segments,
    )


__all__ = [
    "CosmosBoundaryParse", "CosmosCoarseParse", "CosmosSemanticError", "CosmosSemanticResult", "COARSE_REPAIR_PROMPT", "build_segments", "build_semantic_rows", "group_bounded",
    "parse_boundary_output", "parse_boundary_output_with_anomalies", "parse_coarse_output", "parse_coarse_output_with_anomalies", "refinement_indices", "run_cosmos_semantics",
    "sample_source_indices", "validate_semantic_coverage",
]
