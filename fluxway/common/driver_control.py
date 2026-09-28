"""Typed control requests shared by Driver components and workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fluxway.common.ingestion import IngestionLayoutActivation


@runtime_checkable
class DriverControlRequest(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def operator_name(self) -> str: ...


@runtime_checkable
class WorkerControlTransport(Protocol):
    async def dispatch(self, request: DriverControlRequest) -> object: ...


@dataclass(frozen=True, slots=True)
class DriverComponentCapabilities:
    worker_control: WorkerControlTransport

    def __post_init__(self) -> None:
        if not isinstance(self.worker_control, WorkerControlTransport):
            raise TypeError(
                "driver worker_control must implement WorkerControlTransport"
            )


@dataclass(frozen=True, slots=True)
class IngestionLayoutActivationRequest:
    rank: int
    operator_name: str
    activation: IngestionLayoutActivation

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 0
        ):
            raise ValueError("activation request rank must be a non-negative integer")
        if not isinstance(self.operator_name, str) or not self.operator_name:
            raise ValueError("activation request operator_name must be non-empty")
        if not isinstance(self.activation, IngestionLayoutActivation):
            raise TypeError("activation request requires IngestionLayoutActivation")


__all__ = [
    "DriverComponentCapabilities",
    "DriverControlRequest",
    "IngestionLayoutActivationRequest",
    "WorkerControlTransport",
]
