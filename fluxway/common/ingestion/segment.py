"""Canonical work intervals and compiled ingestion segments."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fluxway.common.metadata import metadata_digest


SplitPolicy = Literal["AtomicItem", "SplittableSpan"]


@dataclass(frozen=True, slots=True)
class LayoutUnitDescriptor:
    unit_id: str
    revision_identity: str
    cardinality: int
    cardinality_metric: str
    split_policy: SplitPolicy
    locality_key: str
    physical_identity: str
    logical_start: int
    opaque_token: str | bytes

    def __post_init__(self) -> None:
        _id(self.unit_id, "layout unit id")
        _id(self.revision_identity, "layout unit revision identity")
        _count(self.cardinality, "layout unit cardinality", positive=True)
        _id(self.cardinality_metric, "layout unit cardinality metric")
        if self.split_policy not in {"AtomicItem", "SplittableSpan"}:
            raise ValueError("layout unit split policy is unsupported")
        if self.split_policy == "AtomicItem" and self.cardinality != 1:
            raise ValueError("AtomicItem cardinality must be one")
        _id(self.locality_key, "layout unit locality key")
        _id(self.physical_identity, "layout unit physical identity")
        _count(self.logical_start, "layout unit logical start")
        _opaque_token(self.opaque_token)

    def canonical_work_interval(self, source_identity: str) -> CanonicalWorkInterval:
        return CanonicalWorkInterval(
            source_identity=source_identity,
            revision_identity=self.revision_identity,
            physical_identity=self.physical_identity,
            cardinality_metric=self.cardinality_metric,
            logical_start=self.logical_start,
            logical_end=self.logical_start + self.cardinality,
        )


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _opaque_token(value: str | bytes) -> None:
    if isinstance(value, bytes):
        if not value:
            raise ValueError("opaque metadata must not be empty")
        return
    _id(value, "opaque token")


def _count(value: int, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if positive and value == 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class CanonicalWorkInterval:
    source_identity: str
    revision_identity: str
    physical_identity: str
    cardinality_metric: str
    logical_start: int
    logical_end: int

    def __post_init__(self) -> None:
        _id(self.source_identity, "canonical interval source identity")
        _id(self.revision_identity, "canonical interval revision identity")
        _id(self.physical_identity, "canonical interval physical identity")
        _id(self.cardinality_metric, "canonical interval cardinality metric")
        _count(self.logical_start, "canonical interval logical start")
        _count(self.logical_end, "canonical interval logical end", positive=True)
        if self.logical_end <= self.logical_start:
            raise ValueError("canonical work interval must be non-empty")

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.source_identity,
            self.revision_identity,
            self.physical_identity,
            self.cardinality_metric,
        )


@dataclass(frozen=True, slots=True)
class SourceWorkKey:
    epoch: int
    step: int
    assignment_id: str

    def __post_init__(self) -> None:
        _count(self.epoch, "source work epoch")
        _count(self.step, "source work step")
        _id(self.assignment_id, "source assignment id")

    @classmethod
    def for_provenance(
        cls,
        *,
        epoch: int,
        step: int,
        source_identity: str,
        schema_digest: str,
        descriptor: LayoutUnitDescriptor,
        global_start: int,
        global_batch_size: int,
    ) -> SourceWorkKey:
        identity = metadata_digest(
            {
                "epoch": epoch,
                "step": step,
                "source_identity": source_identity,
                "schema_digest": schema_digest,
                "descriptor": asdict(descriptor),
                "global_start": global_start,
                "global_batch_size": global_batch_size,
            }
        )
        return cls(epoch=epoch, step=step, assignment_id=identity)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceWorkAssignment:
    epoch: int
    step: int
    source_identity: str
    schema_digest: str
    descriptor: LayoutUnitDescriptor
    global_start: int
    global_batch_size: int
    key: SourceWorkKey

    def __post_init__(self) -> None:
        _count(self.epoch, "source assignment epoch")
        _count(self.step, "source assignment step")
        _id(self.source_identity, "source assignment source")
        _id(self.schema_digest, "source assignment schema")
        if not isinstance(self.descriptor, LayoutUnitDescriptor):
            raise TypeError("source assignment requires LayoutUnitDescriptor")
        _count(self.global_start, "source assignment global start")
        _count(
            self.global_batch_size, "source assignment global batch size", positive=True
        )
        if (
            not self.step * self.global_batch_size
            <= self.global_start
            < self.global_start + self.descriptor.cardinality
            <= (self.step + 1) * self.global_batch_size
        ):
            raise ValueError(
                "source assignment provenance is outside its logical batch"
            )
        if not isinstance(self.key, SourceWorkKey):
            raise TypeError("source assignment key must be SourceWorkKey")
        if (self.key.epoch, self.key.step) != (self.epoch, self.step):
            raise ValueError(
                "source assignment key epoch/step does not match provenance"
            )

    @classmethod
    def create(
        cls,
        *,
        epoch: int,
        step: int,
        source_identity: str,
        schema_digest: str,
        descriptor: LayoutUnitDescriptor,
        global_start: int,
        global_batch_size: int,
        key: SourceWorkKey | None = None,
    ) -> SourceWorkAssignment:
        if key is not None:
            expected_key = SourceWorkKey.for_provenance(
                epoch=epoch,
                step=step,
                source_identity=source_identity,
                schema_digest=schema_digest,
                descriptor=descriptor,
                global_start=global_start,
                global_batch_size=global_batch_size,
            )
            if key != expected_key:
                raise ValueError("source assignment key does not match provenance")
            resolved_key = key
        else:
            resolved_key = SourceWorkKey.for_provenance(
                epoch=epoch,
                step=step,
                source_identity=source_identity,
                schema_digest=schema_digest,
                descriptor=descriptor,
                global_start=global_start,
                global_batch_size=global_batch_size,
            )
        return cls(
            epoch=epoch,
            step=step,
            source_identity=source_identity,
            schema_digest=schema_digest,
            descriptor=descriptor,
            global_start=global_start,
            global_batch_size=global_batch_size,
            key=resolved_key,
        )

    @property
    def interval(self) -> CanonicalWorkInterval:
        return self.descriptor.canonical_work_interval(self.source_identity)


@dataclass(frozen=True, slots=True)
class SourceWorkItem:
    assignment: SourceWorkAssignment
    logical_start: int
    logical_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, SourceWorkAssignment):
            raise TypeError("source work item requires SourceWorkAssignment")
        _count(self.logical_start, "source work start")
        _count(self.logical_end, "source work end", positive=True)
        interval = self.assignment.interval
        if not (
            interval.logical_start
            <= self.logical_start
            < self.logical_end
            <= interval.logical_end
        ):
            raise ValueError("source work interval is outside its assignment")

    @property
    def key(self) -> SourceWorkKey:
        return self.assignment.key

    @property
    def global_start(self) -> int:
        return (
            self.assignment.global_start
            + self.logical_start
            - self.assignment.descriptor.logical_start
        )

    @property
    def cardinality(self) -> int:
        return self.logical_end - self.logical_start


@dataclass(frozen=True, slots=True)
class SourceWorkLease:
    item: SourceWorkItem
    lease_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.item, SourceWorkItem):
            raise TypeError("source work lease requires SourceWorkItem")
        _id(self.lease_id, "source lease id")


@dataclass(frozen=True, slots=True)
class SourceWorkAcknowledgement:
    key: SourceWorkKey
    lease_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceWorkKey):
            raise TypeError("source acknowledgement requires SourceWorkKey")
        _id(self.lease_id, "source acknowledgement lease id")


def validate_canonical_work_intervals(
    intervals: tuple[CanonicalWorkInterval, ...],
) -> None:
    ordered = sorted(
        intervals,
        key=lambda interval: (
            interval.identity,
            interval.logical_start,
            interval.logical_end,
        ),
    )
    for left, right in itertools.pairwise(ordered):
        if left.identity == right.identity and right.logical_start < left.logical_end:
            raise ValueError("duplicate canonical work intervals overlap")


@dataclass(frozen=True, slots=True)
class CompiledSegment:
    step: int
    assignment_id: str
    worker_id: str
    unit_id: str
    revision_identity: str
    locality_key: str
    physical_identity: str
    cardinality_metric: str
    logical_start: int
    logical_end: int
    opaque_token: str | bytes

    def __post_init__(self) -> None:
        _count(self.step, "compiled segment step")
        _id(self.assignment_id, "compiled segment assignment")
        _id(self.worker_id, "compiled segment worker id")
        _id(self.unit_id, "compiled segment unit id")
        _id(self.revision_identity, "compiled segment revision identity")
        _id(self.locality_key, "compiled segment locality key")
        _id(self.physical_identity, "compiled segment physical identity")
        _id(self.cardinality_metric, "compiled segment cardinality metric")
        _count(self.logical_start, "compiled segment logical start")
        _count(self.logical_end, "compiled segment logical end", positive=True)
        if self.logical_end <= self.logical_start:
            raise ValueError("compiled segment logical interval must be non-empty")
        _opaque_token(self.opaque_token)

    @property
    def cardinality(self) -> int:
        return self.logical_end - self.logical_start

    def canonical_work_interval(self, source_identity: str) -> CanonicalWorkInterval:
        return CanonicalWorkInterval(
            source_identity=source_identity,
            revision_identity=self.revision_identity,
            physical_identity=self.physical_identity,
            cardinality_metric=self.cardinality_metric,
            logical_start=self.logical_start,
            logical_end=self.logical_end,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "assignment_id": self.assignment_id,
            "unit_id": self.unit_id,
            "revision_identity": self.revision_identity,
            "locality_key": self.locality_key,
            "physical_identity": self.physical_identity,
            "cardinality_metric": self.cardinality_metric,
            "logical_start": self.logical_start,
            "logical_end": self.logical_end,
            "opaque_token": self.opaque_token,
        }


__all__ = [
    "CanonicalWorkInterval",
    "CompiledSegment",
    "LayoutUnitDescriptor",
    "SourceWorkAssignment",
    "SourceWorkKey",
    "validate_canonical_work_intervals",
]
