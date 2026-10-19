"""Immutable, bounded source-capture protocol shared with adapters."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from fluxway.common.ingestion.segment import LayoutUnitDescriptor

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")


@dataclass(frozen=True, slots=True)
class OpaqueSourceCursor:
    payload: bytes
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("source cursor payload must be bytes")
        if not isinstance(self.digest, str) or not _DIGEST_PATTERN.fullmatch(
            self.digest
        ):
            raise ValueError("source cursor digest must be a lowercase SHA-256 hex")
        expected = hashlib.sha256(self.payload).hexdigest()
        if self.digest != expected:
            raise ValueError("source cursor digest does not match its payload")

    @classmethod
    def from_payload(cls, payload: bytes) -> OpaqueSourceCursor:
        if not isinstance(payload, bytes):
            raise TypeError("source cursor payload must be bytes")
        return cls(payload, hashlib.sha256(payload).hexdigest())

    def to_checkpoint(self) -> dict[str, str]:
        return {"payload": self.payload.hex(), "digest": self.digest}

    @classmethod
    def from_checkpoint(cls, value: object) -> OpaqueSourceCursor:
        if not isinstance(value, Mapping) or set(value) != {"payload", "digest"}:
            raise ValueError("opaque source cursor checkpoint fields do not match")
        payload = value["payload"]
        if not isinstance(payload, str) or not payload or len(payload) % 2:
            raise ValueError("opaque source cursor checkpoint payload is invalid")
        try:
            decoded = bytes.fromhex(payload)
        except ValueError as error:
            raise ValueError(
                "opaque source cursor checkpoint payload is not hexadecimal"
            ) from error
        return cls(payload=decoded, digest=value["digest"])


@dataclass(frozen=True, slots=True)
class SourceScanBudget:
    max_records: int
    max_units: int

    def __post_init__(self) -> None:
        for name in (
            "max_records",
            "max_units",
        ):
            _positive_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class SourceBindingIdentity:
    """Stable identity of one bound source object, independent of scan view."""

    configuration_identity: str
    source_identity: str
    mode_identity: str
    native_identity: str

    def __post_init__(self) -> None:
        for name in (
            "configuration_identity",
            "source_identity",
            "mode_identity",
            "native_identity",
        ):
            _identity(getattr(self, name), f"source binding identity {name}")

    def to_checkpoint(self) -> dict[str, str]:
        return {
            "configuration_identity": self.configuration_identity,
            "source_identity": self.source_identity,
            "mode_identity": self.mode_identity,
            "native_identity": self.native_identity,
        }

    @classmethod
    def from_checkpoint(cls, value: object) -> SourceBindingIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "configuration_identity",
            "source_identity",
            "mode_identity",
            "native_identity",
        }:
            raise ValueError("source binding identity checkpoint fields do not match")
        return cls(
            configuration_identity=value["configuration_identity"],
            source_identity=value["source_identity"],
            mode_identity=value["mode_identity"],
            native_identity=value["native_identity"],
        )


class SourceScanStatus(StrEnum):
    READY = "ready"
    PROGRESS = "progress"
    IDLE = "idle"
    TERMINAL = "terminal"

    @classmethod
    def from_scan(
        cls, *, records: int, exhausted: bool, finite: bool
    ) -> SourceScanStatus:
        _nonnegative_int(records, "source scan records")
        if not isinstance(exhausted, bool) or not isinstance(finite, bool):
            raise TypeError("source scan boundaries must be boolean")
        if exhausted and finite:
            return cls.TERMINAL
        if records:
            return cls.READY
        return cls.IDLE if exhausted else cls.PROGRESS


@dataclass(frozen=True, slots=True)
class SourceScanIdentity:
    """The immutable identity of one ordered source view."""

    source_identity: str
    view_identity: str
    schema_digest: str
    lineage_digest: str

    def __post_init__(self) -> None:
        for name in (
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
        ):
            _identity(getattr(self, name), f"source scan identity {name}")

    def to_checkpoint(self) -> dict[str, str]:
        return {
            "source_identity": self.source_identity,
            "view_identity": self.view_identity,
            "schema_digest": self.schema_digest,
            "lineage_digest": self.lineage_digest,
        }

    @classmethod
    def from_checkpoint(cls, value: object) -> SourceScanIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
        }:
            raise ValueError("source scan identity checkpoint fields do not match")
        return cls(
            source_identity=value["source_identity"],
            view_identity=value["view_identity"],
            schema_digest=value["schema_digest"],
            lineage_digest=value["lineage_digest"],
        )


@dataclass(frozen=True, slots=True)
class SourceScanResult:
    source_identity: str
    view_identity: str
    schema_digest: str
    lineage_digest: str
    units: tuple[LayoutUnitDescriptor, ...]
    unit_cursors: tuple[OpaqueSourceCursor, ...]
    records: int
    next_cursor: OpaqueSourceCursor
    status: SourceScanStatus

    def __post_init__(self) -> None:
        SourceScanIdentity(
            source_identity=self.source_identity,
            view_identity=self.view_identity,
            schema_digest=self.schema_digest,
            lineage_digest=self.lineage_digest,
        )
        if not isinstance(self.units, tuple):
            raise TypeError("source scan result units must be a tuple")
        if any(not isinstance(unit, LayoutUnitDescriptor) for unit in self.units):
            raise TypeError("source scan result requires LayoutUnitDescriptor values")
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("source scan result unit ids must be unique")
        if not isinstance(self.unit_cursors, tuple) or any(
            not isinstance(cursor, OpaqueSourceCursor) for cursor in self.unit_cursors
        ):
            raise TypeError("source scan result unit cursors must be opaque")
        if len(self.unit_cursors) != len(self.units):
            raise ValueError("source scan result unit cursors must match its units")
        _nonnegative_int(self.records, "source scan result records")
        if not isinstance(self.next_cursor, OpaqueSourceCursor):
            raise TypeError("source scan result next_cursor must be OpaqueSourceCursor")
        if not isinstance(self.status, SourceScanStatus):
            raise TypeError("source scan result status must be SourceScanStatus")
        if self.records != sum(unit.cardinality for unit in self.units):
            raise ValueError("source scan result record count does not match its units")
        if self.status in (SourceScanStatus.IDLE, SourceScanStatus.PROGRESS) and (
            self.units or self.records
        ):
            raise ValueError(
                f"{self.status.name} source scan result cannot contain units or records"
            )
        if self.status is SourceScanStatus.READY and (
            not self.units or not self.records
        ):
            raise ValueError("READY source scan result must contain records")

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def record_count(self) -> int:
        return self.records

    @property
    def identity(self) -> SourceScanIdentity:
        return SourceScanIdentity(
            source_identity=self.source_identity,
            view_identity=self.view_identity,
            schema_digest=self.schema_digest,
            lineage_digest=self.lineage_digest,
        )

    def cursor_at(
        self,
        consumed_records: int,
        advance: Callable[[OpaqueSourceCursor, int], OpaqueSourceCursor],
    ) -> OpaqueSourceCursor:
        _nonnegative_int(consumed_records, "consumed records")
        if consumed_records > self.records:
            raise ValueError("consumed records exceed scan result")
        if consumed_records == self.records:
            return self.next_cursor
        remaining = consumed_records
        for unit, cursor in zip(self.units, self.unit_cursors, strict=True):
            if remaining < unit.cardinality:
                result = cursor if remaining == 0 else advance(cursor, remaining)
                if not isinstance(result, OpaqueSourceCursor):
                    raise TypeError(
                        "source cursor advancement must return an opaque cursor"
                    )
                return result
            remaining -= unit.cardinality
        raise AssertionError("scan result cardinality does not cover consumed records")

    def validate_transition(self, requested_cursor: OpaqueSourceCursor) -> None:
        if not isinstance(requested_cursor, OpaqueSourceCursor):
            raise TypeError("requested source cursor must be OpaqueSourceCursor")
        if (
            self.status in (SourceScanStatus.READY, SourceScanStatus.PROGRESS)
            and self.next_cursor == requested_cursor
        ):
            raise ValueError(
                f"{self.status.name} source scan result must advance the cursor"
            )

    def validate_budget(self, budget: SourceScanBudget) -> None:
        if not isinstance(budget, SourceScanBudget):
            raise TypeError("source scan result budget must be SourceScanBudget")
        measurements = (
            (self.records, budget.max_records, "record budget"),
            (self.unit_count, budget.max_units, "unit budget"),
        )
        for observed, limit, name in measurements:
            if observed > limit:
                raise ValueError(
                    f"source scan result exceeded {name}: observed={observed} limit={limit}"
                )


__all__ = [
    "OpaqueSourceCursor",
    "SourceBindingIdentity",
    "SourceScanBudget",
    "SourceScanIdentity",
    "SourceScanResult",
    "SourceScanStatus",
]
