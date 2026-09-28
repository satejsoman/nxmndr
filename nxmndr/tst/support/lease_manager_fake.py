# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Minimal in-test implementation of the frozen chunk 1a lease API.

Chunk 1 (server.py) codes against the lease API frozen in the rebuild contracts
(CONTRACTS.md section 3). Chunk 1a owns the real implementation in
``nxmndr/server/managers.py``. Until it merges, ``lease_api_shim.install()`` puts
these names into ``nxmndr.server.managers`` so the server and its tests run.
When the real names exist the shim does nothing and this module is unused.

This is not the production cache. It implements the documented semantics that the
server relies on (LRU of unpinned records, pins from session and execution
leases, reservations, coalesced loads, tombstones, TTL expiry, aux resources,
dispose exactly once, shutdown drain) with one global lock, and nothing more.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("tst.support.lease_manager_fake")

# Set by lease_api_shim.install() to the pre-rebuild ModelManager class, which
# knows how to build records from model specs (used by the default loader).
LEGACY_MODEL_MANAGER = None

CLOSE_REASONS = ("closed", "cancelled", "expired", "failed", "disconnected", "shutdown")


# ------------------------------------------------------------------ errors


class ModelCacheError(RuntimeError):
    pass


class CacheExhaustedError(ModelCacheError):
    retryable = True


class ModelInUseError(ModelCacheError):
    pass


class UnknownModelError(ModelCacheError, KeyError):
    pass


class UnknownSessionError(ModelCacheError, KeyError):
    pass


class SessionConflictError(ModelCacheError):
    pass


class SessionClosedError(ModelCacheError):
    pass


class ModelLoadError(ModelCacheError):
    pass


class CacheShutdownError(ModelCacheError):
    pass


# ------------------------------------------------------------------ keys and records


@dataclass(frozen=True)
class ModelCacheKey:
    format: str
    source: str
    revision: str
    checksum: str
    model_class: str
    task: str
    load_options: Tuple[Tuple[str, str], ...]
    auth_scope: str


_FORMAT_NAMES = {0: "", 1: "pytorch", 2: "onnx", 3: "huggingface", 4: "torchhub"}


def cache_key_from_spec(spec, *, artifact_sha256: str = "") -> ModelCacheKey:
    from nxmndr.inference import inference_pb2

    try:
        task = inference_pb2.TaskType.Name(spec.task).lower() if spec.task else ""
    except ValueError:
        task = ""
    token = getattr(spec, "token", "") or ""
    return ModelCacheKey(
        format=_FORMAT_NAMES.get(int(spec.format), ""),
        source=f"sha256:{artifact_sha256}" if artifact_sha256 else str(spec.source or ""),
        revision=str(spec.version or ""),
        checksum=str(spec.checksum or ""),
        model_class=str(spec.model_class or ""),
        task=task,
        load_options=(),
        auth_scope=hashlib.sha256(token.encode()).hexdigest()[:16] if token else "",
    )


@dataclass
class ModelRecord:
    model: Optional[object]
    backend: str
    metadata: Dict[str, object]
    spec: Optional[object] = None
    device_models: Dict[str, object] = field(default_factory=dict)
    model_id: str = ""
    key: Optional[ModelCacheKey] = None
    aux: Dict[str, object] = field(default_factory=dict, repr=False)
    temp_artifacts: List[str] = field(default_factory=list, repr=False)
    dispose_count: int = field(default=0, repr=False)

    def dispose(self) -> None:
        if self.dispose_count:
            return
        self.dispose_count += 1
        for resource in list(self.aux.values()):
            for hook in ("dispose", "close"):
                fn = getattr(resource, hook, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("aux resource %s failed", hook)
                    break
        self.aux.clear()
        for path in self.temp_artifacts:
            Path(path).unlink(missing_ok=True)
        self.temp_artifacts.clear()
        self.device_models = {}
        self.model = None


@dataclass(frozen=True)
class LoadResult:
    model_id: str
    cache_hit: bool


@dataclass(frozen=True)
class SessionLease:
    session_id: str
    model_id: str
    cache_hit: bool
    reused: bool


@dataclass(frozen=True)
class CacheStats:
    capacity: int
    resident: int
    reserved: int
    pinned: int


class ExecutionLease:
    def __init__(self, manager: "ModelManager", lease_id: str, record: ModelRecord, session_id: str):
        self._manager = manager
        self.lease_id = lease_id
        self.model_id = record.model_id
        self.session_id = session_id
        self.record = record
        self._released = False

    def model_for_device(self, device_id: str) -> object:
        return self.record.device_models.get(device_id, self.record.model)

    def resource(self, name: str, factory: Callable[[], object]) -> object:
        return self._manager._resource(self.record, name, factory)

    def release(self) -> None:
        self._manager._release(self)

    def __enter__(self) -> "ExecutionLease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ------------------------------------------------------------------ manager


def _default_loader(provider):
    def _load(model_spec, key, metadata, device_plan):
        legacy = LEGACY_MODEL_MANAGER(provider)
        legacy_id = legacy.load_spec(model_spec, metadata=metadata, device_plan=device_plan)
        rec = legacy.get(legacy_id)
        meta = dict(rec.metadata)
        meta.pop("model_id", None)
        return ModelRecord(
            model=rec.model,
            backend=rec.backend,
            metadata=meta,
            spec=rec.spec,
            device_models=dict(rec.device_models or {}),
        )

    return _load


@dataclass
class _Session:
    model_id: str
    key: Optional[ModelCacheKey]
    last_activity: float
    live_executions: int = 0


class _Pending:
    def __init__(self):
        self.done = threading.Event()
        self.model_id = ""
        self.error: Optional[BaseException] = None


class ModelManager:
    def __init__(
        self,
        provider,
        *,
        capacity: int = 10,
        session_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        loader=None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._provider = provider
        self._capacity = int(capacity)
        self._ttl = float(session_ttl_s)
        self._clock = clock
        self._loader = loader or _default_loader(provider)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._records: "OrderedDict[str, ModelRecord]" = OrderedDict()  # LRU first
        self._by_key: Dict[ModelCacheKey, str] = {}
        self._loading: Dict[ModelCacheKey, _Pending] = {}
        self._reserved = 0
        self._pins: Dict[str, int] = {}
        self._sessions: Dict[str, _Session] = {}
        self._tombstones: Dict[str, Tuple[str, float]] = {}
        self._leases: Dict[str, ExecutionLease] = {}
        self._aux_pending: Dict[Tuple[str, str], threading.Event] = {}
        self._lease_ids = itertools.count(1)
        self._shutdown = False

    # -- helpers (lock held) --
    def _check_open_locked(self) -> None:
        if self._shutdown:
            raise CacheShutdownError("model cache is shut down")

    def _tombstone_locked(self, session_id: str) -> Optional[str]:
        entry = self._tombstones.get(session_id)
        if entry is None:
            return None
        reason, expires = entry
        if self._clock() >= expires:
            del self._tombstones[session_id]
            return None
        return reason

    def _pick_victim_locked(self) -> Optional[str]:
        for mid in self._records:
            if self._pins.get(mid, 0) == 0:
                return mid
        return None

    def _remove_locked(self, model_id: str) -> ModelRecord:
        rec = self._records.pop(model_id)
        self._by_key.pop(rec.key, None)
        self._pins.pop(model_id, None)
        return rec

    def _dispose(self, rec: Optional[ModelRecord]) -> None:
        if rec is not None:
            rec.dispose()

    # -- acquire with load --
    def _acquire(self, model_spec, key, metadata, device_plan, *, overwrite=False, pin_session=None):
        """Return (model_id, cache_hit). With pin_session the pin happens under the lock."""

        while True:
            victim = None
            with self._lock:
                self._check_open_locked()
                mid = self._by_key.get(key)
                if mid is not None and overwrite:
                    if self._pins.get(mid, 0):
                        raise ModelInUseError(f"model {mid} is in use")
                    victim = self._remove_locked(mid)
                    mid = None
                    overwrite = False
                if mid is not None:
                    self._records.move_to_end(mid)
                    if pin_session is not None:
                        self._pin_session_locked(pin_session, mid, key)
                    return mid, True
                pending = self._loading.get(key)
                if pending is None:
                    # An overwrite already freed a slot, so at most one victim per load.
                    if len(self._records) + self._reserved >= self._capacity:
                        victim_id = self._pick_victim_locked()
                        if victim_id is None:
                            raise CacheExhaustedError(
                                f"all {self._capacity} cache slots are pinned or reserved"
                            )
                        victim = self._remove_locked(victim_id)
                    self._reserved += 1
                    pending = _Pending()
                    self._loading[key] = pending
                    owner = True
                else:
                    owner = False
            if not owner:
                pending.done.wait()
                if pending.error is not None:
                    raise ModelLoadError(str(pending.error)) from pending.error
                continue  # re-check: hit (or evicted meanwhile, then load again)
            self._dispose(victim)
            try:
                record = self._loader(model_spec, key, dict(metadata or {}), device_plan)
            except BaseException as exc:
                with self._lock:
                    self._reserved -= 1
                    del self._loading[key]
                    pending.error = exc
                    pending.done.set()
                raise ModelLoadError(f"model load failed: {exc}") from exc
            with self._lock:
                model_id = uuid.uuid4().hex
                record.model_id = model_id
                record.key = key
                self._records[model_id] = record
                self._by_key[key] = model_id
                self._pins[model_id] = 0
                self._reserved -= 1
                del self._loading[key]
                pending.model_id = model_id
                pending.done.set()  # release coalesced waiters before anything can raise
                if pin_session is not None:
                    self._pin_session_locked(pin_session, model_id, key)
                return model_id, False

    def _pin_session_locked(self, session_id, model_id, key) -> None:
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.model_id != model_id:
                raise SessionConflictError(f"session {session_id} is bound to another model")
            return
        self._pins[model_id] = self._pins.get(model_id, 0) + 1
        self._sessions[session_id] = _Session(model_id, key, self._clock())

    # -- public API --
    def load(self, model_spec, *, key, metadata=None, device_plan=None, overwrite=False) -> LoadResult:
        mid, hit = self._acquire(model_spec, key, metadata, device_plan, overwrite=overwrite)
        return LoadResult(mid, hit)

    def open_session(
        self, session_id, *, model_id="", model_spec=None, key=None, metadata=None, device_plan=None
    ) -> SessionLease:
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            self._check_open_locked()
            if self._tombstone_locked(session_id) is not None:
                raise SessionClosedError(f"session {session_id} is closed")
            existing = self._sessions.get(session_id)
            if existing is not None:
                same = (model_id and existing.model_id == model_id) or (
                    key is not None and existing.key == key
                )
                if not same:
                    raise SessionConflictError(f"session {session_id} is bound to another model")
                return SessionLease(session_id, existing.model_id, True, True)
            if model_id:
                if model_id not in self._records:
                    raise UnknownModelError(model_id)
                self._records.move_to_end(model_id)
                self._pin_session_locked(session_id, model_id, self._records[model_id].key)
                return SessionLease(session_id, model_id, True, False)
        if key is None:
            raise ValueError("open_session needs model_id or key")
        mid, hit = self._acquire(model_spec, key, metadata, device_plan, pin_session=session_id)
        return SessionLease(session_id, mid, hit, False)

    def close_session(self, session_id, *, reason) -> bool:
        if reason not in CLOSE_REASONS:
            raise ValueError(f"unknown close reason {reason!r}")
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is None:
                return False
            self._pins[sess.model_id] = self._pins.get(sess.model_id, 1) - 1
            self._tombstones[session_id] = (reason, self._clock() + self._ttl)
            self._cond.notify_all()
            return True

    def touch_session(self, session_id) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.last_activity = self._clock()

    def session_state(self, session_id) -> str:
        with self._lock:
            if session_id in self._sessions:
                return "open"
            reason = self._tombstone_locked(session_id)
            return reason if reason is not None else "unknown"

    def expire_sessions(self) -> List[str]:
        expired = []
        with self._lock:
            now = self._clock()
            for sid, sess in list(self._sessions.items()):
                if sess.live_executions == 0 and now - sess.last_activity > self._ttl:
                    expired.append(sid)
        for sid in expired:
            self.close_session(sid, reason="expired")
        return expired

    def acquire_execution(self, *, model_id="", session_id="") -> ExecutionLease:
        if bool(model_id) == bool(session_id):
            raise ValueError("pass exactly one of model_id or session_id")
        with self._lock:
            self._check_open_locked()
            if session_id:
                sess = self._sessions.get(session_id)
                if sess is None:
                    if self._tombstone_locked(session_id) is not None:
                        raise SessionClosedError(f"session {session_id} is closed")
                    raise UnknownSessionError(session_id)
                model_id = sess.model_id
                sess.live_executions += 1
                sess.last_activity = self._clock()
            record = self._records.get(model_id)
            if record is None:
                raise UnknownModelError(model_id)
            self._records.move_to_end(model_id)
            self._pins[model_id] = self._pins.get(model_id, 0) + 1
            lease = ExecutionLease(self, f"lease-{next(self._lease_ids)}", record, session_id)
            self._leases[lease.lease_id] = lease
            return lease

    def _release(self, lease: ExecutionLease) -> None:
        with self._lock:
            if lease._released:
                return
            lease._released = True
            self._leases.pop(lease.lease_id, None)
            if lease.model_id in self._pins:
                self._pins[lease.model_id] -= 1
            sess = self._sessions.get(lease.session_id) if lease.session_id else None
            if sess is not None:
                sess.live_executions -= 1
                sess.last_activity = self._clock()
            self._cond.notify_all()

    def _resource(self, record: ModelRecord, name: str, factory):
        slot = (record.model_id, name)
        while True:
            with self._lock:
                if name in record.aux:
                    return record.aux[name]
                waiter = self._aux_pending.get(slot)
                if waiter is None:
                    waiter = threading.Event()
                    self._aux_pending[slot] = waiter
                    owner = True
                else:
                    owner = False
            if not owner:
                waiter.wait()
                continue
            try:
                value = factory()
            except BaseException:
                with self._lock:
                    del self._aux_pending[slot]
                waiter.set()
                raise
            with self._lock:
                record.aux[name] = value
                del self._aux_pending[slot]
            waiter.set()
            return value

    def unload(self, model_id) -> bool:
        with self._lock:
            if model_id not in self._records:
                return False
            if self._pins.get(model_id, 0):
                raise ModelInUseError(f"model {model_id} is in use")
            rec = self._remove_locked(model_id)
        self._dispose(rec)
        return True

    def get(self, model_id):
        with self._lock:
            return self._records.get(model_id)

    def list_models(self):
        with self._lock:
            return dict(self._records)

    def pin_count(self, model_id) -> int:
        with self._lock:
            return self._pins.get(model_id, 0)

    def stats(self) -> CacheStats:
        with self._lock:
            pinned = sum(1 for mid in self._records if self._pins.get(mid, 0))
            return CacheStats(self._capacity, len(self._records), self._reserved, pinned)

    def register_temp_artifact(self, path, *, model_id="") -> None:
        with self._lock:
            rec = self._records.get(model_id) if model_id else None
            if rec is not None:
                rec.temp_artifacts.append(str(path))

    def shutdown(self, *, drain_timeout_s=None) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            open_ids = list(self._sessions)
        for sid in open_ids:
            self.close_session(sid, reason="shutdown")
        deadline = None if drain_timeout_s is None else time.monotonic() + drain_timeout_s
        with self._lock:
            while self._leases:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    logger.warning("disposing %d records still leased", len(self._leases))
                    break
                self._cond.wait(remaining)
            records = list(self._records.values())
            self._records.clear()
            self._by_key.clear()
        for rec in records:
            self._dispose(rec)


__all__ = [
    "ModelCacheError",
    "CacheExhaustedError",
    "ModelInUseError",
    "UnknownModelError",
    "UnknownSessionError",
    "SessionConflictError",
    "SessionClosedError",
    "ModelLoadError",
    "CacheShutdownError",
    "ModelCacheKey",
    "cache_key_from_spec",
    "ModelRecord",
    "LoadResult",
    "SessionLease",
    "ExecutionLease",
    "CacheStats",
    "ModelManager",
]
