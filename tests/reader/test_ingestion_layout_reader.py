from __future__ import annotations

import asyncio
import json

# pyrefly: ignore[missing-import]
import msgpack
import pytest

from fluxway.common.metadata import pack_metadata, unpack_metadata

from fluxway.common.ingestion import (
    CompiledSegment,
    EncodedWorkerLayoutDescriptor,
    IngestionLayoutActivation,
    WorkerLayoutDescriptor,
)
from fluxway.reader.ingestion_layout import (
    IngestionLayoutSession,
    IngestionReadSession,
    SegmentReadChunk,
    read_ingestion_layout,
)


def encoded_descriptor(
    segments, *, expected_cardinality: int
) -> EncodedWorkerLayoutDescriptor:
    compiled_segments = tuple(
        CompiledSegment(worker_id="worker-0", **item) for item in segments
    )
    return WorkerLayoutDescriptor(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        layout_digest="compiled-layout",
        worker_id="worker-0",
        segments=compiled_segments,
    ).encode()


def segment(
    unit_id: str,
    storage_locator: str,
    locality_key: str,
    start: int,
    end: int,
):
    return {
        "step": 0,
        "assignment_id": unit_id,
        "unit_id": unit_id,
        "revision_identity": "revision",
        "storage_locator": storage_locator,
        "locality_key": locality_key,
        "cardinality_metric": "rows",
        "logical_start": start,
        "logical_end": end,
        "opaque_token": "parquet",
    }


class RecordingReader:
    def __init__(self) -> None:
        self.requests = []

    def read(self, request, *, batch_size):
        self.requests.append((request, batch_size))
        yield SegmentReadChunk(
            f"{request.storage_locator}:{request.logical_start}:{request.logical_end}",
            request.cardinality,
            11,
            object_opened=True,
        )

    def close(self) -> None:
        return None


def test_layout_reader_coalesces_adjacent_spans_and_reports_counters() -> None:
    reader = RecordingReader()
    assert isinstance(reader, IngestionReadSession)
    stream = read_ingestion_layout(
        encoded_descriptor(
            [
                segment("a-0", "a.parquet", "node-0", 0, 2),
                segment("a-1", "a.parquet", "node-0", 2, 5),
                segment("b-0", "b.parquet", "node-0", 0, 1),
            ],
            expected_cardinality=6,
        ),
        reader,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="before stream exhaustion"):
        _ = stream.metrics
    assert list(stream) == ["a.parquet:0:5", "b.parquet:0:1"]
    assert [
        (request.storage_locator, request.locality_key, batch_size)
        for request, batch_size in reader.requests
    ] == [
        ("a.parquet", "node-0", 2),
        ("b.parquet", "node-0", 2),
    ]
    assert stream.metrics.objects_opened == 2
    assert stream.metrics.spans_read == 3
    assert stream.metrics.logical_units_read == 6
    assert stream.metrics.physical_bytes_read == 22
    assert stream.metrics.duplicate_work_ids == 0


def test_layout_reader_rejects_duplicate_canonical_work_before_io() -> None:
    reader = RecordingReader()

    with pytest.raises(ValueError, match="duplicate canonical work"):
        read_ingestion_layout(
            encoded_descriptor(
                [
                    segment("a-0", "a.parquet", "node-0", 0, 2),
                    segment("a-1", "a.parquet", "node-0", 1, 3),
                ],
                expected_cardinality=4,
            ),
            reader,
            batch_size=2,
        )
    assert reader.requests == []


def test_layout_reader_rejects_reader_cardinality_mismatch() -> None:
    class ShortReader:
        def read(self, request, *, batch_size):
            yield SegmentReadChunk(
                "short", request.cardinality - 1, 1, object_opened=True
            )

        def close(self) -> None:
            return None

    stream = read_ingestion_layout(
        encoded_descriptor(
            [segment("a-0", "a.parquet", "node-0", 0, 2)],
            expected_cardinality=2,
        ),
        ShortReader(),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="cardinality mismatch"):
        list(stream)


def test_layout_reader_rejects_unknown_descriptor_fields() -> None:
    value = encoded_descriptor(
        [segment("a-0", "a.parquet", "node-0", 0, 1)],
        expected_cardinality=1,
    )
    decoded = unpack_metadata(value.encoded)
    decoded["unexpected"] = True
    invalid = EncodedWorkerLayoutDescriptor(
        value.worker_id,
        pack_metadata(decoded),
        value.expected_cardinality,
        value.segment_count,
        value.final_segment_cursor,
    )

    with pytest.raises(ValueError, match="fields do not match"):
        read_ingestion_layout(invalid, RecordingReader(), batch_size=1)


def test_worker_layout_descriptor_is_the_single_wire_codec() -> None:
    encoded = encoded_descriptor(
        [segment("a-0", "a.parquet", "node-0", 0, 2)],
        expected_cardinality=2,
    )

    descriptor = WorkerLayoutDescriptor.decode(encoded)

    assert descriptor.encode() == encoded
    assert descriptor.segments[0].storage_locator == "a.parquet"
    assert WorkerLayoutDescriptor.FIELDS.isdisjoint(
        {"batch", "data", "tensor", "values"}
    )


@pytest.mark.asyncio
async def test_ingestion_read_session_keeps_blocking_reads_off_the_event_loop() -> None:
    descriptor = encoded_descriptor(
        [segment("a-0", "a.parquet", "node-0", 0, 2)],
        expected_cardinality=2,
    )
    activation = IngestionLayoutActivation(
        epoch=0,
        start_position=0,
        end_position=1,
        layout_digest="aggregate-digest",
        terminal=True,
        descriptor=descriptor,
    )
    ticks = 0

    class BlockingReader:
        def read(self, request, *, batch_size):
            import time

            time.sleep(0.05)
            yield SegmentReadChunk([1, 2], 2, 8, True)

        def close(self) -> None:
            return None

    async def ticker():
        nonlocal ticks
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1

    session = IngestionLayoutSession(activation, BlockingReader(), batch_size=2)
    value, _ = await asyncio.gather(session.read(), ticker())

    assert value == [1, 2]
    assert ticks == 3
    assert await session.read() is None
    assert session.acknowledgement.consumed_cardinality == 2
    assert session.acknowledgement.worker_id == "worker-0"


@pytest.mark.parametrize("encoding", ["json", "unsorted-msgpack"])
def test_worker_layout_descriptor_rejects_noncanonical_metadata(encoding) -> None:
    value = encoded_descriptor(
        [segment("a-0", "a.parquet", "node-0", 0, 2)],
        expected_cardinality=2,
    )
    decoded = unpack_metadata(value.encoded)
    payload = (
        json.dumps(decoded).encode()
        if encoding == "json"
        else msgpack.packb(dict(reversed(tuple(decoded.items()))), use_bin_type=True)
    )
    noncanonical = EncodedWorkerLayoutDescriptor(
        value.worker_id,
        payload,
        value.expected_cardinality,
        value.segment_count,
        value.final_segment_cursor,
    )

    with pytest.raises(ValueError, match="not canonical MessagePack"):
        WorkerLayoutDescriptor.decode(noncanonical)


def test_layout_reader_handles_large_intervals_without_row_expansion() -> None:
    cardinality = 10**12
    stream = read_ingestion_layout(
        encoded_descriptor(
            [segment("a-0", "a.parquet", "node-0", 0, cardinality)],
            expected_cardinality=cardinality,
        ),
        RecordingReader(),
        batch_size=1,
    )

    assert list(stream) == [f"a.parquet:0:{cardinality}"]
    assert stream.metrics.logical_units_read == cardinality


def test_layout_reader_accepts_reused_physical_object_without_open_event() -> None:
    class ReusingReader:
        def read(self, request, *, batch_size):
            yield SegmentReadChunk("all", request.cardinality, 1, object_opened=False)

        def close(self) -> None:
            return None

    stream = read_ingestion_layout(
        encoded_descriptor(
            [segment("a-0", "a.parquet", "node-0", 0, 2)],
            expected_cardinality=2,
        ),
        ReusingReader(),
        batch_size=2,
    )

    assert list(stream) == ["all"]
    assert stream.metrics.objects_opened == 0


def test_layout_reader_rejects_multiple_open_events_per_request() -> None:
    class InvalidOpenReader:
        def read(self, request, *, batch_size):
            yield SegmentReadChunk("first", 1, 1, object_opened=True)
            yield SegmentReadChunk("second", 1, 1, object_opened=True)

        def close(self) -> None:
            return None

    stream = read_ingestion_layout(
        encoded_descriptor(
            [segment("a-0", "a.parquet", "node-0", 0, 2)],
            expected_cardinality=2,
        ),
        InvalidOpenReader(),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="at most one physical object"):
        list(stream)


def test_layout_reader_closes_session_on_exhaustion_and_failure() -> None:
    class ClosingReader:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.closed = 0

        def read(self, request, *, batch_size):
            if self.fail:
                raise RuntimeError("read failure")
            yield SegmentReadChunk("all", request.cardinality, 1, object_opened=True)

        def close(self) -> None:
            self.closed += 1

    descriptor = encoded_descriptor(
        [segment("a-0", "a.parquet", "node-0", 0, 2)],
        expected_cardinality=2,
    )
    complete = ClosingReader(fail=False)
    assert list(read_ingestion_layout(descriptor, complete, batch_size=2)) == ["all"]
    assert complete.closed == 1

    failed = ClosingReader(fail=True)
    with pytest.raises(RuntimeError, match="read failure"):
        list(read_ingestion_layout(descriptor, failed, batch_size=2))
    assert failed.closed == 1
