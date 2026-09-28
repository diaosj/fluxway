"""Controller-owned logical source work runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fluxway.common.ingestion import (
    CompiledSegment,
    SourceWorkAcknowledgement,
    SourceWorkLease,
    WorkerLayoutDescriptor,
)

from fluxway.common.types.checkpoint import CheckpointCommitPermit, CheckpointCut
from fluxway.common.types.configs import SourceControlConfig
from fluxway.control_plane.checkpoint_transaction import PreparedCheckpointTransaction
from fluxway.control_plane.logical_work_ledger import (
    LogicalWorkEpoch,
    LogicalWorkLedger,
)


class SourceControlRuntime:
    """Typed logical-work facade used by Controller and source Workers."""

    def __init__(self, config: SourceControlConfig) -> None:
        if not isinstance(config, SourceControlConfig):
            raise TypeError("SourceControlRuntime requires a SourceControlConfig")
        self._config = config
        self._work_ledger = LogicalWorkLedger(
            max_in_flight_steps=config.max_in_flight_steps
        )

    @property
    def config(self) -> SourceControlConfig:
        return self._config

    @property
    def planning_limits(self) -> tuple[int, int]:
        return (
            self._config.max_planning_units,
            self._config.max_planning_descriptor_bytes,
        )

    @property
    def work_ledger(self) -> LogicalWorkLedger:
        return self._work_ledger

    async def publish_work_epoch(
        self, epoch: LogicalWorkEpoch, bindings: Sequence[WorkerLayoutDescriptor]
    ) -> None:
        await self._work_ledger.publish(epoch, bindings)

    async def acquire_work(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]:
        return await self._work_ledger.acquire_available(epoch, segments)

    async def acknowledge_work(
        self, acknowledgements: Sequence[SourceWorkAcknowledgement]
    ) -> None:
        await self._work_ledger.acknowledge(acknowledgements)

    async def candidate_work_cut(self) -> int:
        return await self._work_ledger.candidate_cut()

    async def wait_for_work_cut(self, target: int) -> int:
        return await self._work_ledger.wait_for_candidate_cut(target)

    def logical_work_checkpoint_state(
        self, *, committed_cut: int | None = None
    ) -> dict[str, Any]:
        return self._work_ledger.checkpoint_state(committed_cut=committed_cut)

    def restore_logical_work_state(self, state: Mapping[str, Any]) -> None:
        self._work_ledger.restore_state(state)

    def prepare_checkpoint(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        state_digest: str,
        state: Mapping[str, Any],
    ) -> PreparedCheckpointTransaction[dict[str, Any]]:
        return self._work_ledger.prepare_checkpoint(
            checkpoint_id, permit, cut, state_digest, state
        )

    def apply_prepared(
        self, prepared_state: PreparedCheckpointTransaction[dict[str, Any]]
    ) -> None:
        self._work_ledger.apply_prepared(prepared_state)

    def abort_prepared(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._work_ledger.abort_prepared(checkpoint_id, permit)


__all__ = ["SourceControlRuntime"]
