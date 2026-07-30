"""Loopback-only dispatcher for the horizontally replicated UniDepth and DROID lanes.

The dispatcher owns DROID's *logical* session affinity, rather than changing the
model deployments: a successful create binds its server-issued session id to one
replica, and later pushes/finalization are sent only to that replica.  UniDepth is
stateless, so healthy replicas receive deterministic round-robin traffic.

The module deliberately forwards already-received request bytes unchanged.  It
only inspects the compact JSON metadata required to find a DROID session id; binary
tensor parts are never decoded or reconstructed here.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from ego_annotation.serving.binary_envelope import (
    BinaryEnvelopeError,
    content_type_is_binary_envelope,
    parse_binary_envelope_body,
)


DROID_REPLICAS = tuple(f"http://127.0.0.1:{port}" for port in (29002, 28012, 28022, 28007, 28017, 28027))
UNIDEPTH_REPLICAS = tuple(f"http://127.0.0.1:{port}" for port in (29000, 28005))
PUBLIC_DROID_PORT = 28002
PUBLIC_UNIDEPTH_PORT = 28000
LEASE_DB_PATH = "/tmp/ego_lane_dispatcher_leases.sqlite3"
SESSION_TTL_S = 3600.0
# Each DROID infer request owns one entire stateful session.  Six replicas at
# the default therefore admit exactly six concurrent infer requests.
DROID_MAX_SESSIONS_PER_REPLICA = int(os.environ.get("EGO_DROID_MAX_SESSIONS", "1"))
UNIDEPTH_REPLICA_CAPACITY = 8
MAX_BODY_BYTES = 1024 * 1024 * 1024
BACKEND_TIMEOUT_S = 30.0
DROID_INFER_BACKEND_TIMEOUT_S = 86400.0
HEALTH_TIMEOUT_S = 1.0
DROID_RECONCILE_INTERVAL_S = 15.0

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
})
_BOUNDARY_RE = re.compile(r"(?:^|;)\s*boundary=(?:\"([^\"]+)\"|([^;\s]+))", re.IGNORECASE)
_NAME_RE = re.compile(r"(?:^|;)\s*name=\"?metadata\"?(?:;|$)", re.IGNORECASE)


class MetadataError(ValueError):
    """The request/response framing did not contain usable JSON metadata."""


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class LeaseRegistry:
    """SQLite-backed DROID session affinity and active-session counts.

    Each operation opens its own SQLite connection, which is necessary because
    ``ThreadingHTTPServer`` handlers run concurrently.  The two tables below are the
    entire durable state.  Every affinity mutation and its matching count mutation is
    enclosed in one ``BEGIN IMMEDIATE`` transaction.
    """

    def __init__(
        self,
        db_path: str = LEASE_DB_PATH,
        *,
        ttl_s: float = SESSION_TTL_S,
        max_sessions_per_lane: int = DROID_MAX_SESSIONS_PER_REPLICA,
    ) -> None:
        self.db_path = db_path
        self.ttl_s = ttl_s
        self.max_sessions_per_lane = max_sessions_per_lane
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS droid_session_affinity (
                    session_id TEXT PRIMARY KEY,
                    replica_url TEXT NOT NULL,
                    created_monotonic REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS replica_inflight (
                    replica_url TEXT PRIMARY KEY,
                    active_sessions INTEGER NOT NULL DEFAULT 0
                )"""
            )
            for replica_url in DROID_REPLICAS:
                connection.execute(
                    "INSERT OR IGNORE INTO replica_inflight(replica_url, active_sessions) VALUES (?, 0)",
                    (replica_url,),
                )

    def ensure_replicas(self, replicas: Sequence[str]) -> None:
        """Initialize count rows for configured replicas without changing live counts."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for replica_url in replicas:
                    connection.execute(
                        "INSERT OR IGNORE INTO replica_inflight(replica_url, active_sessions) VALUES (?, 0)",
                        (replica_url,),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def pick_replica_for_create(self, healthy_replicas: Sequence[str]) -> str | None:
        """Return the healthy least-active lane below its session limit.

        This is selection only: no reservation is written before a backend confirms a
        real session id.  The dispatcher serializes create/response/bind locally so
        the no-reservation contract cannot oversubscribe this process.
        """
        if not healthy_replicas:
            return None
        with self._connect() as connection:
            placeholders = ",".join("?" for _ in healthy_replicas)
            rows = connection.execute(
                f"SELECT replica_url, active_sessions FROM replica_inflight "
                f"WHERE replica_url IN ({placeholders})", tuple(healthy_replicas),
            ).fetchall()
        counts = {str(url): int(active) for url, active in rows}
        candidates = [
            (counts.get(replica, 0), position, replica)
            for position, replica in enumerate(healthy_replicas)
            if counts.get(replica, 0) < self.max_sessions_per_lane
        ]
        return min(candidates)[2] if candidates else None

    def bind_session(self, session_id: str, replica_url: str, *, created_monotonic: float | None = None) -> bool:
        """Atomically bind a newly created session and increment its lane count."""
        if not session_id:
            raise ValueError("session_id must be non-empty")
        created = time.monotonic() if created_monotonic is None else created_monotonic
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT replica_url FROM droid_session_affinity WHERE session_id = ?", (session_id,),
                ).fetchone()
                if existing is not None:
                    connection.execute("ROLLBACK")
                    return str(existing[0]) == replica_url
                row = connection.execute(
                    "SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (replica_url,),
                ).fetchone()
                active = 0 if row is None else int(row[0])
                if active >= self.max_sessions_per_lane:
                    connection.execute("ROLLBACK")
                    return False
                connection.execute(
                    "INSERT INTO droid_session_affinity(session_id, replica_url, created_monotonic) VALUES (?, ?, ?)",
                    (session_id, replica_url, created),
                )
                if row is None:
                    connection.execute(
                        "INSERT INTO replica_inflight(replica_url, active_sessions) VALUES (?, 1)", (replica_url,),
                    )
                else:
                    connection.execute(
                        "UPDATE replica_inflight SET active_sessions = active_sessions + 1 WHERE replica_url = ?",
                        (replica_url,),
                    )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def reserve_replica_for_infer(self, healthy_replicas: Sequence[str]) -> str | None:
        """Atomically reserve one self-contained infer slot on a healthy replica.

        Infer owns no server-visible session id. Its temporary reservation shares
        ``replica_inflight`` with legacy create/push/finalize sessions, but writes
        no affinity and is returned by the request's unconditional ``finally``.
        """
        if not healthy_replicas:
            return None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in healthy_replicas)
                rows = connection.execute(
                    f"SELECT replica_url, active_sessions FROM replica_inflight "
                    f"WHERE replica_url IN ({placeholders})", tuple(healthy_replicas),
                ).fetchall()
                counts = {str(replica_url): int(active) for replica_url, active in rows}
                candidates = [
                    (counts.get(replica, 0), position, replica)
                    for position, replica in enumerate(healthy_replicas)
                    if counts.get(replica, 0) < self.max_sessions_per_lane
                ]
                if not candidates:
                    connection.execute("COMMIT")
                    return None
                replica = min(candidates)[2]
                row = connection.execute(
                    "SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (replica,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO replica_inflight(replica_url, active_sessions) VALUES (?, 1)", (replica,),
                    )
                else:
                    connection.execute(
                        "UPDATE replica_inflight SET active_sessions = active_sessions + 1 WHERE replica_url = ?",
                        (replica,),
                    )
                connection.execute("COMMIT")
                return replica
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def release_replica_infer(self, replica_url: str) -> None:
        """Return exactly one temporary infer slot without touching affinity."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (replica_url,),
                ).fetchone()
                if row is None or int(row[0]) <= 0:
                    raise RuntimeError(f"unbalanced DROID infer release for {replica_url}")
                connection.execute(
                    "UPDATE replica_inflight SET active_sessions = active_sessions - 1 WHERE replica_url = ?",
                    (replica_url,),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def lookup_session(self, session_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT replica_url FROM droid_session_affinity WHERE session_id = ?", (session_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def release_session(self, session_id: str) -> bool:
        """Atomically remove affinity and decrement exactly its owning lane."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT replica_url FROM droid_session_affinity WHERE session_id = ?", (session_id,),
                ).fetchone()
                if row is None:
                    connection.execute("ROLLBACK")
                    return False
                replica_url = str(row[0])
                connection.execute("DELETE FROM droid_session_affinity WHERE session_id = ?", (session_id,))
                connection.execute(
                    "UPDATE replica_inflight SET active_sessions = "
                    "CASE WHEN active_sessions > 0 THEN active_sessions - 1 ELSE 0 END "
                    "WHERE replica_url = ?",
                    (replica_url,),
                )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def reconcile_active_sessions(self, reported_counts: Mapping[str, int]) -> None:
        """Set dispatcher occupancy from each replica's authoritative status.

        Affinity rows remain durable so existing push/finalize requests still route
        to their original replica. Their aggregate is not authoritative: an idle
        reaper can retire a backend session without a client finalize. Replacing
        only ``active_sessions`` therefore restores admission capacity while
        preserving the caller-visible sticky mapping until its next terminal route.
        """
        normalized: dict[str, int] = {}
        for replica_url, active_sessions in reported_counts.items():
            if isinstance(active_sessions, bool) or not isinstance(active_sessions, int) or active_sessions < 0:
                raise ValueError("reported DROID active_sessions must be a non-negative integer")
            normalized[str(replica_url)] = active_sessions
        if not normalized:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for replica_url, active_sessions in normalized.items():
                    connection.execute(
                        "INSERT OR IGNORE INTO replica_inflight(replica_url, active_sessions) VALUES (?, 0)",
                        (replica_url,),
                    )
                    connection.execute(
                        "UPDATE replica_inflight SET active_sessions = ? WHERE replica_url = ?",
                        (active_sessions, replica_url),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def cleanup_expired(self, *, now_monotonic: float | None = None) -> int:
        """Remove expired affinity rows and their lane counts in one transaction."""
        cutoff = (time.monotonic() if now_monotonic is None else now_monotonic) - self.ttl_s
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    "SELECT replica_url, COUNT(*) FROM droid_session_affinity "
                    "WHERE created_monotonic < ? GROUP BY replica_url", (cutoff,),
                ).fetchall()
                if expired:
                    connection.execute("DELETE FROM droid_session_affinity WHERE created_monotonic < ?", (cutoff,))
                    for replica_url, count in expired:
                        connection.execute(
                            "UPDATE replica_inflight SET active_sessions = "
                            "CASE WHEN active_sessions > ? THEN active_sessions - ? ELSE 0 END "
                            "WHERE replica_url = ?",
                            (int(count), int(count), str(replica_url)),
                        )
                connection.execute("COMMIT")
                return sum(int(count) for _replica, count in expired)
            except Exception:
                connection.execute("ROLLBACK")
                raise


def _metadata_from_multipart(body: bytes, content_type: str) -> Mapping[str, Any]:
    match = _BOUNDARY_RE.search(content_type)
    boundary_text = (match.group(1) or match.group(2)) if match else None
    if not boundary_text:
        raise MetadataError("multipart request has no boundary")
    try:
        boundary = boundary_text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MetadataError("multipart boundary is not ASCII") from exc
    marker = b"--" + boundary
    position = body.find(marker)
    while position >= 0:
        part_start = position + len(marker)
        if body[part_start:part_start + 2] == b"--":
            break
        if body[part_start:part_start + 2] != b"\r\n":
            position = body.find(marker, part_start)
            continue
        header_start = part_start + 2
        header_end = body.find(b"\r\n\r\n", header_start)
        if header_end < 0:
            break
        headers = body[header_start:header_end].decode("latin-1")
        payload_start = header_end + 4
        next_marker = body.find(b"\r\n" + marker, payload_start)
        if next_marker < 0:
            break
        disposition = next((line for line in headers.split("\r\n") if line.lower().startswith("content-disposition:")), "")
        if _NAME_RE.search(disposition.split(":", 1)[-1]):
            try:
                metadata = json.loads(body[payload_start:next_marker].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MetadataError("multipart metadata is not valid JSON") from exc
            if isinstance(metadata, Mapping):
                return metadata
            raise MetadataError("multipart metadata must be an object")
        position = body.find(marker, next_marker + 2)
    raise MetadataError("multipart request has no metadata part")


def extract_metadata(body: bytes, content_type: str | None) -> Mapping[str, Any]:
    """Extract only request metadata from JSON, multipart, or binary-envelope framing."""
    normalized = content_type or ""
    if content_type_is_binary_envelope(normalized):
        try:
            envelope = parse_binary_envelope_body(body)
        except BinaryEnvelopeError as exc:
            raise MetadataError(str(exc)) from exc
        metadata_part = next((part for part in envelope.parts if part.name == "metadata"), None)
        if metadata_part is None or metadata_part.dtype != "application/json" or metadata_part.shape:
            raise MetadataError("binary envelope has no valid metadata part")
        try:
            metadata = json.loads(metadata_part.data.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetadataError("binary envelope metadata is not valid JSON") from exc
        if not isinstance(metadata, Mapping):
            raise MetadataError("binary envelope metadata must be an object")
        return metadata
    if "multipart/form-data" in normalized.lower():
        return _metadata_from_multipart(body, normalized)
    try:
        metadata = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataError("JSON metadata is invalid") from exc
    if not isinstance(metadata, Mapping):
        raise MetadataError("JSON metadata must be an object")
    return metadata


def _session_id(metadata: Mapping[str, Any]) -> str | None:
    """Find the session id in direct or generic gateway-wrapped metadata."""
    direct = metadata.get("session_id")
    if isinstance(direct, str) and direct:
        return direct
    for key in ("metadata", "result"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping):
            found = _session_id(nested)
            if found is not None:
                return found
    return None


def _filtered_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    connection_tokens: set[str] = set()
    copied = list(headers)
    for key, value in copied:
        if key.lower() == "connection":
            connection_tokens.update(token.strip().lower() for token in value.split(","))
    blocked = _HOP_BY_HOP | connection_tokens | {"host", "content-length"}
    return [(key, value) for key, value in copied if key.lower() not in blocked]


class LaneDispatcher:
    """Shared state and routing policy for both public listener ports."""

    def __init__(
        self,
        *,
        registry: LeaseRegistry | None = None,
        droid_replicas: Sequence[str] = DROID_REPLICAS,
        unidepth_replicas: Sequence[str] = UNIDEPTH_REPLICAS,
        unidepth_replica_capacity: int = UNIDEPTH_REPLICA_CAPACITY,
        backend_timeout_s: float = BACKEND_TIMEOUT_S,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ) -> None:
        if unidepth_replica_capacity <= 0:
            raise ValueError("unidepth_replica_capacity must be positive")
        self.registry = registry or LeaseRegistry()
        self.droid_replicas = tuple(droid_replicas)
        self.unidepth_replicas = tuple(unidepth_replicas)
        self.registry.ensure_replicas(self.droid_replicas)
        self.unidepth_replica_capacity = unidepth_replica_capacity
        self.backend_timeout_s = backend_timeout_s
        self.health_timeout_s = health_timeout_s
        self._create_lock = threading.Lock()
        self._unidepth_condition = threading.Condition()
        self._unidepth_active = {replica: 0 for replica in self.unidepth_replicas}
        self._unidepth_waiters = 0

    def _is_healthy(self, replica_url: str) -> bool:
        parsed = urlsplit(replica_url)
        if parsed.scheme != "http" or not parsed.hostname:
            return False
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=self.health_timeout_s)
        try:
            connection.request("GET", "/status", headers={"Connection": "close"})
            response = connection.getresponse()
            response.read()
            return response.status < 500
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def _healthy_replicas(self, replicas: Sequence[str]) -> tuple[str, ...]:
        return tuple(replica for replica in replicas if self._is_healthy(replica))

    def _droid_active_sessions(self, replica_url: str) -> int | None:
        """Read the backend-owned active session count from its typed status."""
        parsed = urlsplit(replica_url)
        if parsed.scheme != "http" or not parsed.hostname:
            return None
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=self.health_timeout_s)
        try:
            connection.request("GET", "/status", headers={"Connection": "close"})
            response = connection.getresponse()
            body = response.read(MAX_BODY_BYTES + 1)
            if response.status >= 500 or len(body) > MAX_BODY_BYTES:
                return None
            payload = json.loads(body.decode("utf-8"))
            status = payload.get("status") if isinstance(payload, Mapping) else None
            active_sessions = status.get("active_sessions") if isinstance(status, Mapping) else None
            if isinstance(active_sessions, bool) or not isinstance(active_sessions, int) or active_sessions < 0:
                return None
            return active_sessions
        except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError):
            return None
        finally:
            connection.close()

    def reconcile_droid_sessions(self) -> None:
        """Return capacity released by a backend idle reaper to the dispatcher."""
        # Serialize create -> backend response -> DB bind against a status snapshot;
        # otherwise a just-created backend session could be counted by both paths.
        with self._create_lock:
            reported = {
                replica: active_sessions
                for replica in self.droid_replicas
                if (active_sessions := self._droid_active_sessions(replica)) is not None
            }
            self.registry.reconcile_active_sessions(reported)

    def _forward(
        self, replica_url: str, path: str, body: bytes, headers: Iterable[tuple[str, str]], *, timeout_s: float | None = None,
    ) -> UpstreamResponse:
        parsed = urlsplit(replica_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise OSError(f"invalid HTTP replica URL: {replica_url}")
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=self.backend_timeout_s if timeout_s is None else timeout_s,
        )
        try:
            connection.putrequest("POST", path, skip_host=True, skip_accept_encoding=True)
            for key, value in _filtered_headers(headers):
                connection.putheader(key, value)
            connection.putheader("Host", parsed.netloc)
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body)
            response = connection.getresponse()
            response_body = response.read(MAX_BODY_BYTES + 1)
            if len(response_body) > MAX_BODY_BYTES:
                raise OSError("upstream response exceeds body limit")
            return UpstreamResponse(response.status, tuple(response.getheaders()), response_body)
        finally:
            connection.close()

    def _droid_create(self, path: str, body: bytes, headers: Iterable[tuple[str, str]]) -> UpstreamResponse | None:
        with self._create_lock:
            self.registry.cleanup_expired()
            replica = self.registry.pick_replica_for_create(self._healthy_replicas(self.droid_replicas))
            if replica is None:
                return None
            try:
                response = self._forward(replica, path, body, headers)
            except (OSError, http.client.HTTPException):
                return None
            if response.status == 200:
                try:
                    session_id = _session_id(extract_metadata(response.body, _header_value(response.headers, "content-type")))
                except MetadataError:
                    return response
                if session_id is not None and not self.registry.bind_session(session_id, replica):
                    return None
            return response

    def _droid_sticky(
        self, path: str, body: bytes, headers: Iterable[tuple[str, str]], *, finalize: bool,
    ) -> tuple[UpstreamResponse | None, str | None]:
        try:
            session_id = _session_id(extract_metadata(body, _header_value(headers, "content-type")))
        except MetadataError:
            return None, "invalid_metadata"
        if session_id is None:
            return None, "missing_session_id"
        replica = self.registry.lookup_session(session_id)
        if replica is None:
            return None, "unknown_session"
        try:
            response = self._forward(replica, path, body, headers)
        except (OSError, http.client.HTTPException):
            return None, "upstream_unavailable"
        if finalize and response.status == 200:
            self.registry.release_session(session_id)
        return response, None

    def _droid_infer(
        self, path: str, body: bytes, headers: Iterable[tuple[str, str]],
    ) -> tuple[UpstreamResponse | None, str | None]:
        """Route self-contained infer by bounded least-active occupancy.

        The slot is released once the upstream request ends, before relaying its
        response. Thus either upstream failure or a downstream disconnect cannot
        strand dispatcher capacity.
        """
        self.registry.cleanup_expired()
        healthy = self._healthy_replicas(self.droid_replicas)
        if not healthy:
            return None, "droid_infer_unavailable"
        replica = self.registry.reserve_replica_for_infer(healthy)
        if replica is None:
            return None, "droid_infer_capacity_exhausted"
        try:
            return self._forward(
                replica, path, body, headers, timeout_s=DROID_INFER_BACKEND_TIMEOUT_S,
            ), None
        except (OSError, http.client.HTTPException):
            return None, "droid_infer_unavailable"
        finally:
            self.registry.release_replica_infer(replica)

    def _reserve_unidepth(self, *, excluded: frozenset[str] = frozenset()) -> str | None:
        """Reserve the first healthy ordered replica with available request capacity.

        Reservations are process-local because both public listeners belong to one
        dispatcher process.  A blocked waiter periodically reprobes health: upstream
        health changes cannot notify this process, while releases notify immediately.
        """
        while True:
            healthy = tuple(
                replica for replica in self._healthy_replicas(self.unidepth_replicas)
                if replica not in excluded
            )
            if not healthy:
                return None
            with self._unidepth_condition:
                for replica in healthy:
                    if self._unidepth_active[replica] < self.unidepth_replica_capacity:
                        self._unidepth_active[replica] += 1
                        return replica
                self._unidepth_waiters += 1
                try:
                    self._unidepth_condition.wait(timeout=self.health_timeout_s)
                finally:
                    self._unidepth_waiters -= 1

    def _release_unidepth(self, replica: str) -> None:
        with self._unidepth_condition:
            active = self._unidepth_active.get(replica)
            if active is None or active <= 0:
                raise RuntimeError(f"unbalanced UniDepth release for {replica}")
            self._unidepth_active[replica] = active - 1
            self._unidepth_condition.notify_all()

    def _unidepth_attempt(
        self, replica: str, path: str, body: bytes, headers: Iterable[tuple[str, str]],
    ) -> tuple[UpstreamResponse | None, bool]:
        """Forward one reserved request and always release its reservation once."""
        try:
            return self._forward(replica, path, body, headers), False
        except (OSError, http.client.HTTPException):
            return None, True
        finally:
            self._release_unidepth(replica)

    def _unidepth(self, path: str, body: bytes, headers: Iterable[tuple[str, str]]) -> UpstreamResponse | None:
        primary = self._reserve_unidepth()
        if primary is None:
            return None
        response, connection_failed = self._unidepth_attempt(primary, path, body, headers)
        should_retry = connection_failed or (response is not None and response.status in (429, 503))
        if not should_retry:
            return response
        alternate = self._reserve_unidepth(excluded=frozenset((primary,)))
        if alternate is None:
            return response
        retry_response, _retry_connection_failed = self._unidepth_attempt(alternate, path, body, headers)
        return retry_response

    def dispatch(self, path: str, body: bytes, headers: Iterable[tuple[str, str]]) -> tuple[UpstreamResponse | None, int | None, Mapping[str, Any] | None]:
        clean_path = urlsplit(path).path
        if clean_path == "/droid.create_session":
            response = self._droid_create(path, body, headers)
            if response is None:
                return None, 429, {"error": "droid_capacity_exhausted", "retryable": True}
            return response, None, None
        if clean_path == "/droid.push_frame":
            response, error = self._droid_sticky(path, body, headers, finalize=False)
            if response is not None:
                return response, None, None
            return None, 400 if error in {"invalid_metadata", "missing_session_id"} else 404 if error == "unknown_session" else 503, {
                "error": error, "retryable": error == "upstream_unavailable",
            }
        if clean_path == "/droid.finalize":
            response, error = self._droid_sticky(path, body, headers, finalize=True)
            if response is not None:
                return response, None, None
            return None, 400 if error in {"invalid_metadata", "missing_session_id"} else 404 if error == "unknown_session" else 503, {
                "error": error, "retryable": error == "upstream_unavailable",
            }
        if clean_path in {"/droid.infer", "/infer"}:
            response, error = self._droid_infer(path, body, headers)
            if response is not None:
                return response, None, None
            if error == "droid_infer_capacity_exhausted":
                return None, 429, {"error": error, "retryable": True}
            return None, 503, {"error": "droid_infer_unavailable", "retryable": True}
        if clean_path.startswith("/unidepth."):
            response = self._unidepth(path, body, headers)
            if response is None:
                return None, 503, {"error": "unidepth_unavailable", "retryable": True}
            return response, None, None
        return None, 404, {"error": "unknown_operation", "retryable": False}


def _header_value(headers: Iterable[tuple[str, str]], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers:
        if key.lower() == lowered:
            return value
    return None


class _DispatcherHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "DispatcherHTTPServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if urlsplit(self.path).path in {"/health", "/-/healthz"}:
            payload = json.dumps(
                {"status": "ok", "service": self.server.service_kind}, separators=(",", ":"),
            ).encode("utf-8")
            self._send(200, (("Content-Type", "application/json"),), payload)
            return
        self._send_json(404, {"error": "unknown_operation", "retryable": False})

    def do_POST(self) -> None:
        clean_path = urlsplit(self.path).path
        allowed = (
            clean_path.startswith("/droid.") or clean_path == "/infer"
            if self.server.service_kind == "droid"
            else clean_path.startswith("/unidepth.")
        )
        if not allowed:
            self._send_json(404, {"error": "wrong_public_lane", "retryable": False})
            return
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length) if content_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(411, {"error": "content_length_required", "retryable": False})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request_too_large", "retryable": False})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json(400, {"error": "truncated_request_body", "retryable": False})
            return
        response, error_status, error_payload = self.server.dispatcher.dispatch(
            self.path, body, tuple(self.headers.items()),
        )
        if response is not None:
            self._send(response.status, response.headers, response.body)
            return
        assert error_status is not None and error_payload is not None
        self._send_json(error_status, error_payload)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        self._send(status, (("Content-Type", "application/json"),), json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    def _send(self, status: int, headers: Iterable[tuple[str, str]], body: bytes) -> None:
        self.send_response(status)
        for key, value in _filtered_headers(headers):
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class DispatcherHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], dispatcher: LaneDispatcher, service_kind: str) -> None:
        self.dispatcher = dispatcher
        self.service_kind = service_kind
        super().__init__(address, _DispatcherHandler)


class DispatcherProcess:
    """Owns both public ``ThreadingHTTPServer`` listeners in one process."""

    def __init__(
        self,
        dispatcher: LaneDispatcher,
        *,
        bind_host: str = "127.0.0.1",
        droid_port: int = PUBLIC_DROID_PORT,
        unidepth_port: int = PUBLIC_UNIDEPTH_PORT,
        cleanup_interval_s: float | None = None,
        reconcile_interval_s: float = DROID_RECONCILE_INTERVAL_S,
    ) -> None:
        self.dispatcher = dispatcher
        self.droid_server = DispatcherHTTPServer((bind_host, droid_port), dispatcher, "droid")
        try:
            self.unidepth_server = DispatcherHTTPServer((bind_host, unidepth_port), dispatcher, "unidepth")
        except Exception:
            self.droid_server.server_close()
            raise
        default_interval = min(60.0, max(1.0, dispatcher.registry.ttl_s / 4.0))
        self.cleanup_interval_s = default_interval if cleanup_interval_s is None else cleanup_interval_s
        if reconcile_interval_s <= 0:
            raise ValueError("reconcile_interval_s must be positive")
        self.reconcile_interval_s = reconcile_interval_s
        self._stop_cleanup = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def droid_address(self) -> tuple[str, int]:
        return self.droid_server.server_address[:2]

    @property
    def unidepth_address(self) -> tuple[str, int]:
        return self.unidepth_server.server_address[:2]

    def start(self) -> None:
        if self._threads:
            return
        for server, name in ((self.droid_server, "droid"), (self.unidepth_server, "unidepth")):
            thread = threading.Thread(target=server.serve_forever, name=f"lane-dispatcher-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="lane-dispatcher-ttl", daemon=True,
        )
        cleanup_thread.start()
        self._threads.append(cleanup_thread)
        # No reconciliation thread: infer reservations are dispatcher-owned until
        # their upstream request returns. Sampling backend active_sessions in that
        # interval could erase a reservation before the adapter has created its
        # private session. B-track infer has no idle reaper, so no capacity return
        # depends on polling backend status.

    def _cleanup_loop(self) -> None:
        while not self._stop_cleanup.wait(self.cleanup_interval_s):
            try:
                self.dispatcher.registry.cleanup_expired()
            except sqlite3.Error:
                # A transient busy/IO error cannot terminate serving; the next
                # bounded interval retries cleanup while request transactions keep
                # enforcing affinity and capacity.
                continue

    def _reconcile_loop(self) -> None:
        while not self._stop_cleanup.wait(self.reconcile_interval_s):
            try:
                self.dispatcher.reconcile_droid_sessions()
            except (sqlite3.Error, ValueError):
                # A malformed/unavailable replica status cannot stop public lanes;
                # the next bounded interval re-reads only backend-owned counters.
                continue

    def shutdown(self) -> None:
        self._stop_cleanup.set()
        for server in (self.droid_server, self.unidepth_server):
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()


def run_dispatcher(
    *,
    bind_host: str = "127.0.0.1",
    droid_port: int = PUBLIC_DROID_PORT,
    unidepth_port: int = PUBLIC_UNIDEPTH_PORT,
    db_path: str = LEASE_DB_PATH,
    unidepth_replica_capacity: int = UNIDEPTH_REPLICA_CAPACITY,
) -> None:
    process = DispatcherProcess(
        LaneDispatcher(
            registry=LeaseRegistry(db_path),
            unidepth_replica_capacity=unidepth_replica_capacity,
        ),
        bind_host=bind_host,
        droid_port=droid_port,
        unidepth_port=unidepth_port,
    )
    process.start()
    stopped = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stopped.set()

    old_term = signal.signal(signal.SIGTERM, _stop)
    old_int = signal.signal(signal.SIGINT, _stop)
    try:
        stopped.wait()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        process.shutdown()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dispatch local UniDepth and DROID requests across replica lanes")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--droid-port", type=int, default=PUBLIC_DROID_PORT)
    parser.add_argument("--unidepth-port", type=int, default=PUBLIC_UNIDEPTH_PORT)
    parser.add_argument("--db-path", default=LEASE_DB_PATH)
    parser.add_argument(
        "--unidepth-replica-capacity", type=_positive_int, default=UNIDEPTH_REPLICA_CAPACITY,
    )
    args = parser.parse_args(argv)
    run_dispatcher(
        bind_host=args.bind_host,
        droid_port=args.droid_port,
        unidepth_port=args.unidepth_port,
        db_path=args.db_path,
        unidepth_replica_capacity=args.unidepth_replica_capacity,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BACKEND_TIMEOUT_S", "DROID_INFER_BACKEND_TIMEOUT_S", "DROID_MAX_SESSIONS_PER_REPLICA", "DROID_RECONCILE_INTERVAL_S", "DROID_REPLICAS", "DispatcherHTTPServer",
    "DispatcherProcess", "HEALTH_TIMEOUT_S", "LEASE_DB_PATH", "LaneDispatcher", "LeaseRegistry", "MAX_BODY_BYTES",
    "PUBLIC_DROID_PORT", "PUBLIC_UNIDEPTH_PORT", "SESSION_TTL_S", "UNIDEPTH_REPLICA_CAPACITY", "UNIDEPTH_REPLICAS",
    "extract_metadata", "main",
    "run_dispatcher",
]
