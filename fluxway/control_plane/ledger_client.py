"""Typed client protocol for logical source assignments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ray.actor import ActorHandle

from fluxway.common.ingestion.segment import (
    CompiledSegment,
    SourceWorkAcknowledgement,
    SourceWorkLease,
)


@runtime_checkable
class SourceWorkClient(Protocol):
    async def acquire_source_work(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]: ...

    async def acknowledge_source_work(
        self, acknowledgements: Sequence[SourceWorkAcknowledgement]
    ) -> None: ...


class ControlPlaneClient:
    def __init__(self, handle: Any) -> None:
        if isinstance(handle, ActorHandle):
            self._acquire_source_work = handle.acquire_source_work.remote
            self._acknowledge_source_work = handle.acknowledge_source_work.remote
        elif isinstance(handle, SourceWorkClient):
            self._acquire_source_work = handle.acquire_source_work
            self._acknowledge_source_work = handle.acknowledge_source_work
        else:
            raise TypeError(
                "control-plane handle must be a Ray actor or SourceWorkClient"
            )

    async def acquire_source_work(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]:
        result = await self._acquire_source_work(epoch, tuple(segments))
        if not isinstance(result, tuple) or any(
            not isinstance(value, SourceWorkLease) for value in result
        ):
            raise TypeError("controller returned invalid source work leases")
        return result

    async def acknowledge_source_work(
        self, acknowledgements: Sequence[SourceWorkAcknowledgement]
    ) -> None:
        values = tuple(acknowledgements)
        if any(not isinstance(value, SourceWorkAcknowledgement) for value in values):
            raise TypeError("source work acknowledgements must be typed")
        await self._acknowledge_source_work(values)
