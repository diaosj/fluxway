"""Manifest-inline checkpoint backend for small pure state."""

from __future__ import annotations

import copy
import inspect
import msgpack
from collections.abc import Awaitable, Callable
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointCapability,
    ComponentCheckpoint,
    CheckpointObjectRecord,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    ComponentSpec,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CommitScope,
    PartitionRecord,
    PrepareScope,
    WorkerSaveResult,
    RestoreTopologyMode,
    TopologyMismatchError,
)
from fluxway.control_plane.checkpointer.backend import (
    CheckpointComponentRegistration,
    CheckpointComponent,
)
from fluxway.control_plane.checkpointer.repository import validate_checkpoint_value


StateCapture = Callable[[Any, SaveSession], Any]
StateRestore = Callable[[Any, Any, RestoreSession], Any]


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


class StateCheckpointComponent(CheckpointComponent):
    def __init__(
        self,
        component_id: str,
        capture_state: StateCapture,
        restore_state: StateRestore,
    ) -> None:
        self._descriptor = ComponentSpec(
            component_id=component_id,
            format_version=1,
            prepare_scope=PrepareScope.LEADER,
            commit_scope=CommitScope.LEADER,
            checkpoint_objects=(),
            restore_topology=RestoreTopologyMode.EXACT_TOPOLOGY,
            capabilities=frozenset(
                {CheckpointCapability.PERSISTENCE, CheckpointCapability.LIFECYCLE}
            ),
            supports_eager_capture=False,
        )
        self._capture_state = _async_callable(capture_state, "capture_state")
        self._restore_state = _async_callable(restore_state, "restore_state")
        self._runtime: CheckpointRuntime | None = None

    @property
    def descriptor(self) -> ComponentSpec:
        return self._descriptor

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(
            descriptor=self.descriptor,
            component=self,
        )

    async def open(self, runtime: CheckpointRuntime) -> None:
        if self._runtime is not None:
            raise RuntimeError(
                f"inline backend {self.descriptor.component_id!r} is open"
            )
        self._runtime = runtime

    async def start_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)
        state = copy.deepcopy(await self._capture_state(session.instance, session))
        validate_checkpoint_value(state, "inline_checkpoint_state")
        return WorkerSaveResult(
            record=CheckpointObjectRecord(
                payload=msgpack.packb(state, use_bin_type=True)
            )
        )

    async def finish_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def commit_save(self, session: CommitSaveSession) -> ComponentCheckpoint:
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
        return ComponentCheckpoint(
            component_id=self.descriptor.component_id,
            format_version=self.descriptor.format_version,
            partitions=tuple(partitions),
        )

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def on_checkpoint_available(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker_runtime(session)

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker_runtime(session)
        publication = session.component_checkpoint
        if publication.component_id != self.descriptor.component_id:
            raise ValueError("inline publication belongs to another backend")
        current = session.current_topology or session.source_topology
        if current != session.source_topology:
            raise TopologyMismatchError(
                f"inline backend {self.descriptor.component_id!r} requires exact "
                "source and current topology"
            )
        partition_key = f"rank/{session.rank}"
        matches = [
            partition
            for partition in publication.partitions
            if partition.partition_key == partition_key
        ]
        if len(matches) != 1:
            raise KeyError(f"inline checkpoint partition not found: {partition_key!r}")
        state = copy.deepcopy(msgpack.unpackb(matches[0].record.payload, raw=False))
        await self._restore_state(session.instance, state, session)

    async def close(self) -> None:
        if self._runtime is None:
            raise RuntimeError(
                f"inline backend {self.descriptor.component_id!r} is not open"
            )
        self._runtime = None

    def _require_worker_runtime(
        self, context: SaveSession | RestoreSession
    ) -> CheckpointRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(
                f"inline backend {self.descriptor.component_id!r} is not open"
            )
        if runtime.role is not CheckpointRuntimeRole.WORKER:
            raise RuntimeError("inline state operations require a worker runtime")
        if context.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint context belongs to another backend")
        return runtime

    @staticmethod
    def _require_permit(context: SaveSession, permit) -> None:
        if permit is None or permit != context.commit_permit:
            raise PermissionError(
                "checkpoint commit_save permit does not match context"
            )
