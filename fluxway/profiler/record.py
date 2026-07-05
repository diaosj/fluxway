from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import blake2b
import time
from collections.abc import Mapping
from typing import TypedDict

import numpy as np

from fluxway.profiler.errors import (
    CausalClockViolation,
    MissingTraceMetadata,
    ProfilerBufferExhausted,
)


RECORD_WIDTH = 16
PAYLOAD_START_COLUMN = 8
MAX_PAYLOAD_FIELDS = RECORD_WIDTH - PAYLOAD_START_COLUMN
EMPTY_TRACE_PAYLOAD: tuple[int, ...] = ()


class TraceEventKind(IntEnum):
    INSTANT = 1
    SLICE_BEGIN = 2
    SLICE_END = 3
    COUNTER = 4
    FLOW_START = 5
    FLOW_END = 6


class TraceClockSource(IntEnum):
    FLUXWAY_RAY = 1
    AXON_NATIVE = 2


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    names_by_id: Mapping[int, str]
    payload_fields_by_name_id: Mapping[int, tuple[str, ...]]

    def name_for(self, ident: int) -> str:
        try:
            return self.names_by_id[ident]
        except KeyError as exc:
            raise MissingTraceMetadata(metadata_kind="name", ident=ident) from exc

    def payload_fields_for(self, name_id: int) -> tuple[str, ...]:
        try:
            return self.payload_fields_by_name_id[name_id]
        except KeyError as exc:
            raise MissingTraceMetadata(
                metadata_kind="payload_fields",
                ident=name_id,
            ) from exc


class ProfilerSnapshot(TypedDict):
    records: np.ndarray
    metadata: TraceMetadata
    process_name: str
    monotonic_to_epoch_offset_ns: int
    clock_uncertainty_ns: int


@dataclass(frozen=True, slots=True)
class TraceCarrier:
    run_id_hash: int
    flow_id: int
    task_id: int
    parent_task_id: int
    sender_local_ts_ns: int
    sender_hlc_physical_ns: int
    sender_hlc_logical: int
    receiver_hlc_physical_ns: int = 0
    receiver_hlc_logical: int = 0
    clock_source: int = int(TraceClockSource.FLUXWAY_RAY)

    @classmethod
    def root(cls, run_id: str) -> TraceCarrier:
        now = time.perf_counter_ns()
        return cls(
            run_id_hash=stable_u64(run_id),
            flow_id=0,
            task_id=stable_u64(run_id, "root"),
            parent_task_id=0,
            sender_local_ts_ns=now,
            sender_hlc_physical_ns=now,
            sender_hlc_logical=0,
        )

    def child(self, *parts: object) -> TraceCarrier:
        now = time.perf_counter_ns()
        task_id = stable_u64(self.run_id_hash, self.task_id, *parts)
        return TraceCarrier(
            run_id_hash=self.run_id_hash,
            flow_id=self.flow_id,
            task_id=task_id,
            parent_task_id=self.task_id,
            sender_local_ts_ns=now,
            sender_hlc_physical_ns=max(now, self.sender_hlc_physical_ns),
            sender_hlc_logical=self.sender_hlc_logical + 1,
            clock_source=self.clock_source,
        )


def stable_u64(*parts: object) -> int:
    h = blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little", signed=False) & ((1 << 63) - 1)


def merge_hlc(
    local_physical_ns: int, local_logical: int, carrier: TraceCarrier
) -> tuple[int, int]:
    remote_physical = carrier.sender_hlc_physical_ns
    remote_logical = carrier.sender_hlc_logical
    physical = max(local_physical_ns, remote_physical)
    if physical == local_physical_ns and physical == remote_physical:
        logical = max(local_logical, remote_logical) + 1
    elif physical == local_physical_ns:
        logical = local_logical + 1
    else:
        logical = remote_logical + 1
    if (physical, logical) <= (remote_physical, remote_logical):
        raise CausalClockViolation(
            sender_hlc_physical_ns=remote_physical,
            sender_hlc_logical=remote_logical,
            receiver_hlc_physical_ns=physical,
            receiver_hlc_logical=logical,
        )
    return physical, logical


class TraceArena:
    def __init__(self, buffer_bytes: int, *, record_width: int = RECORD_WIDTH) -> None:
        if buffer_bytes <= 0:
            raise ValueError("buffer_bytes must be positive.")
        if record_width <= 0:
            raise ValueError("record_width must be positive.")
        record_bytes = np.dtype(np.int64).itemsize * record_width
        capacity = buffer_bytes // record_bytes
        if capacity <= 0:
            raise ValueError(
                f"buffer_bytes={buffer_bytes} cannot hold one profiler record."
            )
        self._records = np.zeros((capacity, record_width), dtype=np.int64)
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._records.shape[0]

    @property
    def size(self) -> int:
        return self._size

    def append(
        self,
        *,
        kind: TraceEventKind,
        ts_ns: int,
        dur_ns: int,
        track_id: int,
        name_id: int,
        flow_id: int = 0,
        task_id: int = 0,
        parent_task_id: int = 0,
        payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD,
    ) -> int:
        if len(payload) > MAX_PAYLOAD_FIELDS:
            raise ValueError(
                f"trace payload has {len(payload)} fields; max={MAX_PAYLOAD_FIELDS}"
            )
        index = self._size
        if index >= self.capacity:
            raise ProfilerBufferExhausted(
                capacity=self.capacity,
                attempted_index=index,
            )
        row = self._records[index]
        row[:PAYLOAD_START_COLUMN] = (
            kind,
            ts_ns,
            dur_ns,
            track_id,
            name_id,
            flow_id,
            task_id,
            parent_task_id,
        )
        if payload:
            row[PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(payload)] = payload
        self._size = index + 1
        return index

    def snapshot(self) -> np.ndarray:
        return self._records[: self._size].copy()
