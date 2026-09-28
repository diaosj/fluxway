"""Source-neutral adapter contract for algebraic ingestion planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


from fluxway.control_plane.ingestion.compiler import (
    TailLayoutMode,
    LayoutCompileRequest,
    LayoutUnitDescriptor,
    AssignmentPolicy,
    ReaderTopologyDescriptor,
)


from fluxway.control_plane.ingestion.cursor import CatalogCursor


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogViewObservation:
    """Immutable metadata captured by a source adapter for one planning window."""

    source_identity: str
    view_identity: str
    schema_digest: str
    lineage_digest: str
    units: tuple[LayoutUnitDescriptor, ...]
    terminal: bool
    cursor: CatalogCursor

    def __post_init__(self) -> None:
        _identity(self.source_identity, "catalog observation source identity")
        _identity(self.view_identity, "catalog observation view identity")
        _identity(self.schema_digest, "catalog observation schema digest")
        _identity(self.lineage_digest, "catalog observation lineage digest")
        if not isinstance(self.units, tuple):
            raise TypeError("catalog observation units must be a tuple")
        units = self.units
        if any(not isinstance(unit, LayoutUnitDescriptor) for unit in units):
            raise TypeError("catalog observation requires LayoutUnitDescriptor values")
        if len({unit.unit_id for unit in units}) != len(units):
            raise ValueError("catalog observation unit ids must be unique")
        if not isinstance(self.terminal, bool):
            raise TypeError("catalog observation terminal must be a boolean")
        if not isinstance(self.cursor, CatalogCursor):
            raise TypeError("catalog observation cursor must be a CatalogCursor")

    @classmethod
    def create(
        cls,
        *,
        cursor: CatalogCursor | Mapping[str, Any],
        **kwargs: Any,
    ) -> CatalogViewObservation:
        cursor_val = (
            cursor if isinstance(cursor, CatalogCursor) else CatalogCursor(cursor)
        )
        return cls(cursor=cursor_val, **kwargs)

    def layout_request(
        self,
        *,
        topology: ReaderTopologyDescriptor,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
    ) -> LayoutCompileRequest:
        mode = TailLayoutMode.from_value(tail_layout_mode)
        if self.terminal != (mode is not TailLayoutMode.RETAIN_PENDING):
            raise ValueError("catalog terminal state disagrees with tail layout mode")
        return LayoutCompileRequest(
            source_identity=self.source_identity,
            view_identity=self.view_identity,
            schema_digest=self.schema_digest,
            lineage_digest=self.lineage_digest,
            units=self.units,
            topology=topology,
            tail_layout_mode=mode,
            assignment_policy=assignment_policy,
        )


@runtime_checkable
class IngestionSourceAdapter(Protocol):
    """Control-plane metadata adapter shared by all source kinds."""

    def capture_view(
        self, cursor: Mapping[str, Any], *, cardinality_limit: int
    ) -> CatalogViewObservation: ...

    def encode_cursor(self, cursor: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def decode_cursor(self, state: Mapping[str, Any]) -> Mapping[str, Any]: ...


class InMemoryIngestionSourceAdapter:
    """Reference adapter used by contract and component integration tests."""

    def __init__(self, observations: tuple[CatalogViewObservation, ...]) -> None:
        observations = tuple(observations)
        if not observations or any(
            not isinstance(item, CatalogViewObservation) for item in observations
        ):
            raise TypeError("in-memory adapter requires CatalogViewObservation values")
        if any(
            item.source_identity != observations[0].source_identity
            for item in observations
        ):
            raise ValueError(
                "in-memory adapter observations must share source identity"
            )
        self._observations = observations

    def capture_view(
        self, cursor: Mapping[str, Any], *, cardinality_limit: int
    ) -> CatalogViewObservation:
        if not isinstance(cursor, Mapping):
            raise TypeError("in-memory adapter cursor must be a mapping")
        if (
            not isinstance(cardinality_limit, int)
            or isinstance(cardinality_limit, bool)
            or cardinality_limit <= 0
        ):
            raise ValueError("in-memory adapter cardinality_limit must be positive")
        index = cursor.get("index", 0)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(self._observations)
        ):
            raise ValueError("in-memory adapter cursor index is outside observations")
        observation = self._observations[index]
        if sum(unit.cardinality for unit in observation.units) > cardinality_limit:
            raise ValueError("in-memory adapter observation exceeds cardinality_limit")
        return observation

    def encode_cursor(self, cursor: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(cursor, Mapping):
            raise TypeError("in-memory adapter cursor must be a mapping")
        return dict(cursor)

    def decode_cursor(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.encode_cursor(state)


__all__ = [
    "CatalogCursor",
    "CatalogViewObservation",
    "InMemoryIngestionSourceAdapter",
    "IngestionSourceAdapter",
]
