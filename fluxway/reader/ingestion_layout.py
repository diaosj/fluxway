"""Source-neutral execution of one compiled ingestion worker layout."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from fluxway.common.ingestion import (
    CompiledSegment,
    EncodedWorkerLayoutDescriptor,
    SourceLayoutCompletion,
    IngestionLayoutActivation,
    WorkerLayoutDescriptor,
    validate_canonical_work_intervals,
)


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ScheduledSegmentRead:
    storage_locator: str
    locality_key: str
    cardinality_metric: str
    logical_start: int
    logical_end: int
    segments: tuple[CompiledSegment, ...]

    def __post_init__(self) -> None:
        _canonical_string(self.storage_locator, "scheduled read storage locator")
        _canonical_string(self.locality_key, "scheduled read locality key")
        _canonical_string(self.cardinality_metric, "scheduled read cardinality metric")
        _nonnegative_integer(self.logical_start, "scheduled read logical start")
        _nonnegative_integer(self.logical_end, "scheduled read logical end")
        segments = self.segments
        if not isinstance(segments, tuple):
            raise TypeError("scheduled read segments must be a tuple")
        if not segments:
            raise ValueError("scheduled read requires compiled segments")
        if self.logical_end <= self.logical_start:
            raise ValueError("scheduled read logical interval must be non-empty")
        if any(segment.locality_key != self.locality_key for segment in segments):
            raise ValueError("scheduled read contains another locality key")
        if any(segment.storage_locator != self.storage_locator for segment in segments):
            raise ValueError("scheduled read contains another storage locator")
        if any(
            segment.cardinality_metric != self.cardinality_metric
            for segment in segments
        ):
            raise ValueError("scheduled read contains another cardinality metric")
        if segments[0].logical_start != self.logical_start:
            raise ValueError("scheduled read start does not match its first segment")
        if segments[-1].logical_end != self.logical_end:
            raise ValueError("scheduled read end does not match its last segment")
        for left, right in pairwise(segments):
            if left.logical_end != right.logical_start:
                raise ValueError("scheduled read segments must be contiguous")

    @property
    def cardinality(self) -> int:
        return self.logical_end - self.logical_start


@dataclass(frozen=True, slots=True)
class SegmentReadChunk:
    value: Any
    cardinality: int
    physical_bytes: int
    object_opened: bool

    def __post_init__(self) -> None:
        if self.value is None:
            raise ValueError("segment read chunk value must not be None")
        if (
            _nonnegative_integer(self.cardinality, "segment read chunk cardinality")
            == 0
        ):
            raise ValueError("segment read chunk cardinality must be positive")
        _nonnegative_integer(self.physical_bytes, "segment read chunk physical bytes")
        if not isinstance(self.object_opened, bool):
            raise TypeError("segment read chunk object_opened must be a boolean")


@runtime_checkable
class IngestionReadSession(Protocol):
    def read(
        self, request: ScheduledSegmentRead, *, batch_size: int
    ) -> Iterator[SegmentReadChunk]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class IngestionReadMetrics:
    objects_opened: int
    spans_read: int
    logical_units_read: int
    physical_bytes_read: int
    duplicate_work_ids: int


class IngestionReadStream(Iterator[Any]):
    def __init__(
        self,
        encoded_descriptor: EncodedWorkerLayoutDescriptor,
        reader: IngestionReadSession,
        schedule: tuple[ScheduledSegmentRead, ...],
        batch_size: int,
    ) -> None:
        self._encoded_descriptor = encoded_descriptor
        self._reader = reader
        self._schedule = iter(schedule)
        self._batch_size = batch_size
        self._current: Iterator[SegmentReadChunk] | None = None
        self._current_expected = 0
        self._current_actual = 0
        self._current_open_events = 0
        self._objects_opened = 0
        self._logical_units_read = 0
        self._physical_bytes_read = 0
        self._complete = False
        self._closed = False

    def __iter__(self) -> IngestionReadStream:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return self._next_value()
        except BaseException:
            self.close()
            raise

    def _next_value(self) -> Any:
        while True:
            if self._current is None:
                try:
                    request = next(self._schedule)
                except StopIteration as error:
                    if (
                        self._logical_units_read
                        != self._encoded_descriptor.expected_cardinality
                    ):
                        raise ValueError(
                            "ingestion reader cardinality mismatch: "
                            f"expected={self._encoded_descriptor.expected_cardinality} "
                            f"actual={self._logical_units_read}"
                        ) from error
                    self._complete = True
                    self.close()
                    raise
                try:
                    chunks = self._reader.read(request, batch_size=self._batch_size)
                except BaseException:
                    self.close()
                    raise
                if not isinstance(chunks, Iterator):
                    raise TypeError("ingestion segment reader must return an iterator")
                self._current = chunks
                self._current_expected = request.cardinality
                self._current_actual = 0
                self._current_open_events = 0
            try:
                chunk = next(self._current)
            except StopIteration as error:
                if self._current_open_events > 1:
                    raise ValueError(
                        "ingestion segment reader must report at most one physical "
                        "object open per scheduled read"
                    ) from error
                if self._current_actual != self._current_expected:
                    raise ValueError(
                        "ingestion segment reader cardinality mismatch: "
                        f"expected={self._current_expected} "
                        f"actual={self._current_actual}"
                    ) from error
                self._current = None
                continue
            if not isinstance(chunk, SegmentReadChunk):
                raise TypeError("ingestion segment reader must yield SegmentReadChunk")
            if chunk.object_opened:
                self._current_open_events += 1
                self._objects_opened += 1
                if self._current_open_events > 1:
                    raise ValueError(
                        "ingestion segment reader must report at most one physical "
                        "object open per scheduled read"
                    )
            self._current_actual += chunk.cardinality
            if self._current_actual > self._current_expected:
                raise ValueError(
                    "ingestion segment reader cardinality mismatch: "
                    f"expected={self._current_expected} "
                    f"actual={self._current_actual}"
                )
            self._logical_units_read += chunk.cardinality
            self._physical_bytes_read += chunk.physical_bytes
            return chunk.value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._reader.close()

    def __enter__(self) -> IngestionReadStream:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    @property
    def metrics(self) -> IngestionReadMetrics:
        if not self._complete:
            raise RuntimeError(
                "ingestion read metrics are unavailable before stream exhaustion"
            )
        return IngestionReadMetrics(
            objects_opened=self._objects_opened,
            spans_read=self._encoded_descriptor.segment_count,
            logical_units_read=self._logical_units_read,
            physical_bytes_read=self._physical_bytes_read,
            duplicate_work_ids=0,
        )


def _next_stream_value(stream: IngestionReadStream) -> tuple[bool, Any]:
    try:
        return True, next(stream)
    except StopIteration:
        return False, None


class IngestionLayoutSession:
    def __init__(
        self,
        activation: IngestionLayoutActivation,
        reader: IngestionReadSession,
        *,
        batch_size: int,
    ) -> None:
        if not isinstance(activation, IngestionLayoutActivation):
            raise TypeError("ingestion read session requires layout activation")
        self._activation = activation
        self._stream = read_ingestion_layout(
            activation.descriptor,
            reader,
            batch_size=batch_size,
        )
        self._lock = asyncio.Lock()
        self._acknowledgement: SourceLayoutCompletion | None = None

    async def read(self) -> Any | None:
        async with self._lock:
            if self._acknowledgement is not None:
                return None
            has_value, value = await asyncio.to_thread(_next_stream_value, self._stream)
            if has_value:
                return value
            metrics = self._stream.metrics
            descriptor = self._activation.descriptor
            self._acknowledgement = SourceLayoutCompletion(
                epoch=self._activation.epoch,
                worker_id=descriptor.worker_id,
                layout_digest=self._activation.layout_digest,
                consumed_cardinality=metrics.logical_units_read,
                final_segment_cursor=descriptor.final_segment_cursor,
                terminal_observed=self._activation.terminal,
            )
            return None

    async def close(self) -> None:
        await asyncio.to_thread(self._stream.close)

    @property
    def acknowledgement(self) -> SourceLayoutCompletion:
        if self._acknowledgement is None:
            raise RuntimeError(
                "ingestion acknowledgement is unavailable before exhaustion"
            )
        return self._acknowledgement


def read_ingestion_layout(
    encoded_descriptor: EncodedWorkerLayoutDescriptor,
    reader: IngestionReadSession,
    *,
    batch_size: int,
) -> IngestionReadStream:
    if not isinstance(encoded_descriptor, EncodedWorkerLayoutDescriptor):
        raise TypeError(
            "ingestion layout reader requires EncodedWorkerLayoutDescriptor"
        )
    if not isinstance(reader, IngestionReadSession):
        raise TypeError("ingestion layout reader requires IngestionReadSession")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("ingestion layout reader batch_size must be positive")
    descriptor = WorkerLayoutDescriptor.decode(encoded_descriptor)
    segments = descriptor.segments
    _validate_canonical_work(segments, descriptor.source_identity)
    schedule = _compile_schedule(segments)
    return IngestionReadStream(encoded_descriptor, reader, schedule, batch_size)


def _validate_canonical_work(
    segments: tuple[CompiledSegment, ...], source_identity: str
) -> None:
    validate_canonical_work_intervals(
        tuple(segment.canonical_work_interval(source_identity) for segment in segments)
    )


def _compile_schedule(
    segments: tuple[CompiledSegment, ...],
) -> tuple[ScheduledSegmentRead, ...]:
    if not segments:
        return ()
    grouped: list[list[CompiledSegment]] = [[segments[0]]]
    for segment in segments[1:]:
        previous = grouped[-1][-1]
        if (
            segment.storage_locator == previous.storage_locator
            and segment.locality_key == previous.locality_key
            and segment.cardinality_metric == previous.cardinality_metric
            and segment.logical_start == previous.logical_end
        ):
            grouped[-1].append(segment)
        else:
            grouped.append([segment])
    return tuple(
        ScheduledSegmentRead(
            storage_locator=group[0].storage_locator,
            locality_key=group[0].locality_key,
            cardinality_metric=group[0].cardinality_metric,
            logical_start=group[0].logical_start,
            logical_end=group[-1].logical_end,
            segments=tuple(group),
        )
        for group in grouped
    )


__all__ = [
    "IngestionReadMetrics",
    "IngestionLayoutSession",
    "IngestionReadStream",
    "IngestionReadSession",
    "ScheduledSegmentRead",
    "SegmentReadChunk",
    "read_ingestion_layout",
]
