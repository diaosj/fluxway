"""Manifest-inline checkpoint component for small pure state."""

from __future__ import annotations

import copy
import inspect
import msgpack
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointComponentSpec,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointRestoreSpec,
    CheckpointSaveScope,
    CheckpointSaveSpec,
    PartitionRecord,
    WorkerSaveResult,
    RestoreTopologyMode,
    TopologyMismatchError,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentRestore,
    CheckpointComponentSave,
)
from fluxway.control_plane.checkpointer.repository import validate_checkpoint_value


StateSave = Callable[[Any, SaveSession], Any]
StateRestore = Callable[[Any, Any, RestoreSession], Any]
PartitionSelector = Callable[
    [Sequence[PartitionRecord], RestoreSession], PartitionRecord
]


def _async_callable(
    function: Callable[..., Any], name: str
) -> Callable[..., Awaitable[Any]]:
    if not callable(function):
        raise TypeError(f"{name} must be callable")
    if inspect.iscoroutinefunction(function):
        return function

    async def invoke(*args: Any) -> Any:
        result = function(*args)
        if inspect.isawaitable(result):
            raise TypeError(f"{name} returned an awaitable but is not async def")
        return result

    return invoke


class StateCheckpointComponent(
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentSave,
    CheckpointComponentRestore,
):
    def __init__(
        self,
        component_id: str,
        save_state: StateSave,
        restore_state: StateRestore,
        *,
        restore_topology: RestoreTopologyMode = RestoreTopologyMode.EXACT_TOPOLOGY,
        partition_selector: PartitionSelector | None = None,
    ) -> None:
        self._descriptor = CheckpointComponentSpec(
            component_id=component_id,
            format_version=1,
            checkpoint_objects=(),
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            restore=CheckpointRestoreSpec(topology=restore_topology),
        )
        self._save_state = _async_callable(save_state, "save_state")
        self._restore_state = _async_callable(restore_state, "restore_state")
        if restore_topology is RestoreTopologyMode.LOGICAL_PARTITION_REMAP:
            if partition_selector is None:
                raise ValueError(
                    "logical partition remap requires an explicit partition_selector"
                )
        elif partition_selector is not None:
            raise ValueError(
                "partition_selector is only valid with logical partition remap"
            )
        self._partition_selector = partition_selector
        self._runtime: CheckpointRuntime | None = None

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(
            component=self,
            component_lifecycle=self,
            save=self,
            restore=self,
        )

    async def open(self, runtime: CheckpointRuntime) -> None:
        if self._runtime is not None:
            raise RuntimeError(
                f"inline component {self.descriptor.component_id!r} is open"
            )
        self._runtime = runtime

    async def prepare_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)
        state = copy.deepcopy(await self._save_state(session.instance, session))
        validate_checkpoint_value(state, "inline_checkpoint_state")
        return WorkerSaveResult(
            record=CheckpointObjectRecord(
                payload=msgpack.packb(state, use_bin_type=True)
            )
        )

    async def finish_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)
        partitions: list[PartitionRecord] = []
        for rank, prepared in sorted(session.worker_results.items()):
            if prepared.record is None:
                raise ValueError(f"inline checkpoint rank {rank} has no record")
            if prepared.commit_metadata:
                raise ValueError(
                    f"inline checkpoint rank {rank} returned commit_save metadata"
                )
            partitions.append(
                PartitionRecord(
                    partition_key=f"rank/{rank}",
                    source_rank=rank,
                    record=prepared.record,
                )
            )
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            format_version=self.descriptor.format_version,
            partitions=tuple(partitions),
        )

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def publish_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker_runtime(session)
        publication = session.component_checkpoint
        if publication.component_id != self.descriptor.component_id:
            raise ValueError("inline publication belongs to another component")

        restore_spec = self.descriptor.restore
        if restore_spec is None:
            raise RuntimeError(
                f"inline component {self.descriptor.component_id!r} has no restore specification"
            )

        current = session.current_topology or session.source_topology
        if (
            restore_spec.topology is RestoreTopologyMode.EXACT_TOPOLOGY
            and current != session.source_topology
        ):
            raise TopologyMismatchError(
                f"inline component {self.descriptor.component_id!r} requires exact "
                "source and current topology"
            )
        if restore_spec.topology is RestoreTopologyMode.GLOBAL_RESHARD:
            if not publication.partitions:
                raise TopologyMismatchError(
                    f"inline checkpoint component {self.descriptor.component_id!r} "
                    "requires a shared partition for global reshard"
                )
            selected = publication.partitions[0]
            if any(
                partition.record.payload != selected.record.payload
                for partition in publication.partitions[1:]
            ):
                raise TopologyMismatchError(
                    f"inline checkpoint component {self.descriptor.component_id!r} "
                    "has rank-dependent state and cannot be globally resharded"
                )
        elif restore_spec.topology is RestoreTopologyMode.LOGICAL_PARTITION_REMAP:
            selector = self._partition_selector
            if selector is None:
                raise RuntimeError("logical partition remap selector is not configured")
            selected = selector(publication.partitions, session)
        else:
            partition_key = f"rank/{session.rank}"
            matches = [
                partition
                for partition in publication.partitions
                if partition.partition_key == partition_key
            ]
            if len(matches) != 1:
                raise KeyError(
                    f"inline checkpoint partition not found: {partition_key!r}"
                )
            selected = matches[0]
        state = copy.deepcopy(msgpack.unpackb(selected.record.payload, raw=False))
        await self._restore_state(session.instance, state, session)

    async def close(self) -> None:
        if self._runtime is None:
            raise RuntimeError(
                f"inline component {self.descriptor.component_id!r} is not open"
            )
        self._runtime = None

    def _require_worker_runtime(
        self, context: SaveSession | RestoreSession
    ) -> CheckpointRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(
                f"inline component {self.descriptor.component_id!r} is not open"
            )
        if runtime.role is not CheckpointRuntimeRole.WORKER:
            raise RuntimeError("inline state operations require a worker runtime")
        if context.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint context belongs to another component")
        return runtime

    @staticmethod
    def _require_permit(context: SaveSession, permit) -> None:
        if permit is None or permit != context.commit_permit:
            raise PermissionError(
                "checkpoint commit_save permit does not match context"
            )
