"""Public interface for topology-independent ordered control streams."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True, init=False)
class ControlRecord:
    stream_id: str
    position: int
    _payload: dict[str, Any]

    def __init__(
        self,
        *,
        stream_id: str,
        position: int,
        payload: Mapping[str, Any],
    ) -> None:
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("control stream_id must be non-empty")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("control position must be a non-negative integer")
        if not isinstance(payload, Mapping):
            raise TypeError("control payload must be a mapping")
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "_payload", copy.deepcopy(dict(payload)))

    @property
    def payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    @property
    def key(self) -> tuple[str, int]:
        return self.stream_id, self.position

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ControlRecord:
        if not isinstance(value, Mapping):
            raise TypeError("serialized control record must be a mapping")
        expected = {"stream_id", "position", "payload"}
        if set(value) != expected:
            raise ValueError(
                "serialized control record fields do not match: "
                f"expected={sorted(expected)} actual={sorted(value)}"
            )
        return cls(
            stream_id=value["stream_id"],
            position=value["position"],
            payload=value["payload"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "position": self.position,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class StreamProgress:
    durable_cut: int
    published_end: int
    terminal_end: int | None


@runtime_checkable
class ControlLedgerClient(Protocol):
    async def append(self, records: Sequence[ControlRecord]) -> None: ...
    async def try_append(self, records: Sequence[ControlRecord]) -> bool: ...
    async def receive(self, stream_id: str, position: int) -> ControlRecord | None: ...
    async def close(self, stream_id: str) -> None: ...
    async def progress(self, stream_id: str) -> StreamProgress: ...
