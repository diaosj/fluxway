"""Wire protocol and serialization descriptors for ingestion layout distribution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from fluxway.common.ingestion.segment import CompiledSegment
from fluxway.common.metadata import metadata_digest, pack_metadata, unpack_metadata


def _canonical_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: int, name: str) -> int:
    _nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _digest_layout(
    epoch: IngestionLayoutEpoch,
    descriptors: tuple[EncodedWorkerLayoutDescriptor, ...],
) -> str:
    return metadata_digest(
        {
            "epoch": epoch.epoch,
            "start_position": epoch.start_position,
            "end_position": epoch.end_position,
            "assigned_cardinality": epoch.assigned_cardinality,
            "layout_id": epoch.layout_id,
            "worker_ids": epoch.worker_ids,
            "terminal": epoch.terminal,
            "descriptor_digests": [
                {
                    "worker_id": descriptor.worker_id,
                    "digest": descriptor.descriptor_digest,
                }
                for descriptor in descriptors
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class IngestionLayoutEpoch:
    epoch: int
    start_position: int
    end_position: int
    assigned_cardinality: int
    layout_id: str
    worker_ids: tuple[str, ...]
    terminal: bool

    def __post_init__(self) -> None:
        _nonnegative_int(self.epoch, "ingestion layout epoch")
        _nonnegative_int(self.start_position, "ingestion layout start position")
        _nonnegative_int(self.end_position, "ingestion layout end position")
        _nonnegative_int(
            self.assigned_cardinality, "ingestion layout assigned cardinality"
        )
        _canonical_id(self.layout_id, "ingestion layout id")
        if not isinstance(self.terminal, bool):
            raise TypeError("ingestion layout terminal must be a boolean")
        worker_ids = self.worker_ids
        if not isinstance(worker_ids, tuple):
            raise TypeError("ingestion layout worker ids must be a tuple")
        if not worker_ids:
            raise ValueError("ingestion layout requires logical source workers")
        if any(not isinstance(worker_id, str) for worker_id in worker_ids):
            raise TypeError("ingestion layout worker ids must be strings")
        for worker_id in worker_ids:
            _canonical_id(worker_id, "ingestion layout worker id")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("ingestion layout worker ids must be unique")
        if self.end_position < self.start_position:
            raise ValueError(
                "ingestion layout end position precedes its start position"
            )
        if self.end_position == self.start_position and not self.terminal:
            raise ValueError("zero-length ingestion layout must be terminal")
        if (self.end_position == self.start_position) != (
            self.assigned_cardinality == 0
        ):
            raise ValueError(
                "ingestion layout execution positions and assigned cardinality "
                "must both be empty or both be non-empty"
            )


@dataclass(frozen=True, slots=True)
class EncodedWorkerLayoutDescriptor:
    worker_id: str
    encoded: bytes
    expected_cardinality: int
    segment_count: int
    final_segment_cursor: str

    def __post_init__(self) -> None:
        _canonical_id(self.worker_id, "worker layout descriptor worker id")
        if not isinstance(self.encoded, bytes) or not self.encoded:
            raise ValueError("worker layout descriptor must be non-empty bytes")
        _nonnegative_int(
            self.expected_cardinality, "ingestion layout expected cardinality"
        )
        _nonnegative_int(self.segment_count, "ingestion layout segment count")
        _canonical_id(
            self.final_segment_cursor, "ingestion layout final segment cursor"
        )

    @property
    def descriptor_digest(self) -> str:
        return hashlib.sha256(self.encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class IngestionLayoutActivation:
    epoch: int
    start_position: int
    end_position: int
    layout_digest: str
    terminal: bool
    descriptor: EncodedWorkerLayoutDescriptor

    def __post_init__(self) -> None:
        _nonnegative_int(self.epoch, "ingestion activation epoch")
        _nonnegative_int(self.start_position, "ingestion activation start position")
        _nonnegative_int(self.end_position, "ingestion activation end position")
        if self.end_position < self.start_position:
            raise ValueError("ingestion activation end position precedes its start")
        _canonical_id(self.layout_digest, "ingestion activation layout digest")
        if not isinstance(self.terminal, bool):
            raise TypeError("ingestion activation terminal must be a boolean")
        if not isinstance(self.descriptor, EncodedWorkerLayoutDescriptor):
            raise TypeError(
                "ingestion activation requires EncodedWorkerLayoutDescriptor"
            )
        if (
            self.end_position == self.start_position
            and self.descriptor.expected_cardinality != 0
        ):
            raise ValueError("empty ingestion activation cannot carry work")


@dataclass(frozen=True, slots=True)
class WorkerLayoutDescriptor:
    source_identity: str
    view_identity: str
    schema_digest: str
    lineage_digest: str
    layout_digest: str
    worker_id: str
    segments: tuple[CompiledSegment, ...]
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
            "layout_digest",
            "worker_id",
            "expected_cardinality",
            "segment_count",
            "segments",
        }
    )
    SEGMENT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "step",
            "assignment_id",
            "unit_id",
            "revision_identity",
            "locality_key",
            "physical_identity",
            "cardinality_metric",
            "logical_start",
            "logical_end",
            "opaque_token",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
            "layout_digest",
            "worker_id",
        ):
            _canonical_id(getattr(self, name), f"worker layout {name}")
        segments = self.segments
        if not isinstance(segments, tuple):
            raise TypeError("worker layout segments must be a tuple")
        if any(not isinstance(segment, CompiledSegment) for segment in segments):
            raise TypeError("worker layout requires CompiledSegment values")
        if any(segment.worker_id != self.worker_id for segment in segments):
            raise ValueError("worker layout segment belongs to another worker")

    @property
    def expected_cardinality(self) -> int:
        return sum(segment.cardinality for segment in self.segments)

    def encode(self) -> EncodedWorkerLayoutDescriptor:
        encoded = pack_metadata(
            {
                "source_identity": self.source_identity,
                "view_identity": self.view_identity,
                "schema_digest": self.schema_digest,
                "lineage_digest": self.lineage_digest,
                "layout_digest": self.layout_digest,
                "worker_id": self.worker_id,
                "expected_cardinality": self.expected_cardinality,
                "segment_count": len(self.segments),
                "segments": [item.to_mapping() for item in self.segments],
            },
        )
        return EncodedWorkerLayoutDescriptor(
            worker_id=self.worker_id,
            encoded=encoded,
            expected_cardinality=self.expected_cardinality,
            segment_count=len(self.segments),
            final_segment_cursor=(
                "empty" if not self.segments else str(self.segments[-1].logical_end)
            ),
        )

    @classmethod
    def decode(
        cls, encoded_descriptor: EncodedWorkerLayoutDescriptor
    ) -> WorkerLayoutDescriptor:
        if not isinstance(encoded_descriptor, EncodedWorkerLayoutDescriptor):
            raise TypeError(
                "worker layout codec requires EncodedWorkerLayoutDescriptor"
            )
        try:
            decoded = unpack_metadata(encoded_descriptor.encoded)
        except ValueError as error:
            raise ValueError(
                "worker layout descriptor is not canonical MessagePack"
            ) from error
        if not isinstance(decoded, Mapping) or set(decoded) != cls.FIELDS:
            raise ValueError("worker layout descriptor fields do not match protocol")
        for name in (
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
            "layout_digest",
            "worker_id",
        ):
            _canonical_id(decoded[name], f"worker layout descriptor {name}")
        if decoded["worker_id"] != encoded_descriptor.worker_id:
            raise ValueError("worker layout descriptor worker does not match header")
        for name, expected in (
            ("expected_cardinality", encoded_descriptor.expected_cardinality),
            ("segment_count", encoded_descriptor.segment_count),
        ):
            if (
                _nonnegative_int(decoded[name], f"worker layout descriptor {name}")
                != expected
            ):
                raise ValueError(
                    f"worker layout descriptor {name} does not match header"
                )
        encoded_segments = decoded["segments"]
        if not isinstance(encoded_segments, list):
            raise TypeError("worker layout descriptor segments must be a list")
        if len(encoded_segments) != encoded_descriptor.segment_count:
            raise ValueError(
                "worker layout descriptor segment_count does not match segments"
            )
        segments = []
        for item in encoded_segments:
            if not isinstance(item, Mapping) or set(item) != cls.SEGMENT_FIELDS:
                raise ValueError(
                    "ingestion layout segment fields do not match protocol"
                )
            segments.append(
                CompiledSegment(worker_id=encoded_descriptor.worker_id, **item)
            )
        descriptor = cls(
            source_identity=decoded["source_identity"],
            view_identity=decoded["view_identity"],
            schema_digest=decoded["schema_digest"],
            lineage_digest=decoded["lineage_digest"],
            layout_digest=decoded["layout_digest"],
            worker_id=decoded["worker_id"],
            segments=tuple(segments),
        )
        if descriptor.expected_cardinality != encoded_descriptor.expected_cardinality:
            raise ValueError(
                "ingestion layout segment cardinality does not match header: "
                f"expected={encoded_descriptor.expected_cardinality} "
                f"actual={descriptor.expected_cardinality}"
            )
        expected_cursor = (
            "empty"
            if not descriptor.segments
            else str(descriptor.segments[-1].logical_end)
        )
        if encoded_descriptor.final_segment_cursor != expected_cursor:
            raise ValueError(
                "ingestion layout final segment cursor does not match descriptor"
            )
        return descriptor


@dataclass(frozen=True, slots=True)
class SourceLayoutActivation:
    epoch: IngestionLayoutEpoch
    descriptors: tuple[EncodedWorkerLayoutDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, IngestionLayoutEpoch):
            raise TypeError("source layout activation requires an IngestionLayoutEpoch")
        descriptors = self.descriptors
        if not isinstance(descriptors, tuple):
            raise TypeError("source layout descriptors must be a tuple")
        if any(
            not isinstance(descriptor, EncodedWorkerLayoutDescriptor)
            for descriptor in descriptors
        ):
            raise TypeError("source layout activation requires encoded descriptors")
        if (
            tuple(descriptor.worker_id for descriptor in descriptors)
            != self.epoch.worker_ids
        ):
            raise ValueError(
                "ingestion layout descriptor workers must match the epoch worker order"
            )
        decoded = self._decode_descriptors(descriptors)
        if any(
            descriptor.layout_digest != self.epoch.layout_id for descriptor in decoded
        ):
            raise ValueError(
                "worker layout descriptor digest does not match the epoch layout id"
            )
        identity = (
            decoded[0].source_identity,
            decoded[0].view_identity,
            decoded[0].schema_digest,
            decoded[0].lineage_digest,
        )
        if any(
            (
                descriptor.source_identity,
                descriptor.view_identity,
                descriptor.schema_digest,
                descriptor.lineage_digest,
            )
            != identity
            for descriptor in decoded[1:]
        ):
            raise ValueError("worker layout descriptor identities do not match")
        if (
            sum(descriptor.expected_cardinality for descriptor in descriptors)
            != self.epoch.assigned_cardinality
        ):
            raise ValueError(
                "ingestion layout descriptor cardinalities do not match epoch interval"
            )

    @staticmethod
    def _decode_descriptors(
        descriptors: tuple[EncodedWorkerLayoutDescriptor, ...],
    ) -> tuple[WorkerLayoutDescriptor, ...]:
        return tuple(
            WorkerLayoutDescriptor.decode(descriptor) for descriptor in descriptors
        )

    @property
    def decoded_worker_layouts(self) -> tuple[WorkerLayoutDescriptor, ...]:
        return self._decode_descriptors(self.descriptors)

    def worker_layout(self, worker_id: str) -> WorkerLayoutDescriptor:
        _canonical_id(worker_id, "source layout activation worker id")
        for layout in self.decoded_worker_layouts:
            if layout.worker_id == worker_id:
                return layout
        raise KeyError(f"source layout activation worker is not present: {worker_id!r}")

    @property
    def layout_digest(self) -> str:
        return _digest_layout(self.epoch, self.descriptors)

    @property
    def descriptor_bytes(self) -> int:
        return sum(len(descriptor.encoded) for descriptor in self.descriptors)


@dataclass(frozen=True, slots=True)
class SourceLayoutCompletion:
    epoch: int
    worker_id: str
    layout_digest: str
    consumed_cardinality: int
    final_segment_cursor: str
    terminal_observed: bool

    def __post_init__(self) -> None:
        _nonnegative_int(self.epoch, "ingestion layout acknowledgement epoch")
        _canonical_id(self.worker_id, "ingestion layout acknowledgement worker id")
        _canonical_id(self.layout_digest, "ingestion layout acknowledgement digest")
        _nonnegative_int(
            self.consumed_cardinality,
            "ingestion layout acknowledgement cardinality",
        )
        _canonical_id(
            self.final_segment_cursor,
            "ingestion layout acknowledgement final segment cursor",
        )
        if not isinstance(self.terminal_observed, bool):
            raise TypeError(
                "ingestion layout acknowledgement terminal must be a boolean"
            )


__all__ = [
    "EncodedWorkerLayoutDescriptor",
    "SourceLayoutCompletion",
    "IngestionLayoutActivation",
    "SourceLayoutActivation",
    "IngestionLayoutEpoch",
    "WorkerLayoutDescriptor",
]
