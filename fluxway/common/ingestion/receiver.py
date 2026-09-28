"""Worker-side receiver interface for ingestion layouts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fluxway.common.ingestion.protocol import IngestionLayoutActivation


@runtime_checkable
class IngestionLayoutReceiver(Protocol):
    async def activate_ingestion_layout(
        self, activation: IngestionLayoutActivation
    ) -> None: ...


def ingestion_worker_id(spmd_id: str, operator_name: str, rank: int) -> str:
    for value, name in ((spmd_id, "spmd_id"), (operator_name, "operator_name")):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"ingestion worker {name} must be a canonical string")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("ingestion worker rank must be a non-negative integer")
    return f"{spmd_id}:{operator_name}:worker:{rank}"


__all__ = [
    "IngestionLayoutReceiver",
    "ingestion_worker_id",
]
