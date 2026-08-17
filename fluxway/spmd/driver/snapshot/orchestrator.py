from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from fluxway.common.types import (
    CheckpointComponentPublication,
    CheckpointCommitPermit,
    CheckpointCut,
    DriverEvent,
    DriverState,
    WorkerSaveResult,
    SpmdCheckpoint,
    TopologySnapshot,
    DrainEvent,
    DrainEventKind,
)
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.components import DriverComponentRegistry
from fluxway.spmd.driver.state_machine import DriverStateMachine

logger = get_logger(__name__)


class SnapshotOrchestrator:
    def __init__(
        self,
        spmd_id: str,
        executor: Any,
        synchronizer: Any,
        controller_handle: Any | None,
        state_machine: DriverStateMachine,
        profiler_runtime: Any,
        component_registry: DriverComponentRegistry,
        notify_component_state: Callable[[DriverState], Any],
    ) -> None:
        self.spmd_id = spmd_id
        self.executor = executor
        self.synchronizer = synchronizer
        self.controller_handle = controller_handle
        self.state_machine = state_machine
        self._profiler_runtime = profiler_runtime
        self._component_registry = component_registry
        self._notify_component_state = notify_component_state
        self._pending_checkpoint_id: str | None = None
        self._pending_attempt_id: str | None = None
        self._pending_permit: CheckpointCommitPermit | None = None
        self._pending_cut: CheckpointCut | None = None
        self._pending_repository_root: str | None = None
        self._pending_snapshot: SpmdCheckpoint | None = None
        self._published = False
        self._rollback_task: asyncio.Task[None] | None = None
        self._drained_component_notification: asyncio.Task[None] | None = None

    async def request_drain(
        self,
        step_count: int,
        terminal_cut: int | None = None,
        reason: str = "",
    ) -> None:
        del reason
        entered_draining = self.state_machine.admit_drain()
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_drain(
                target_step=step_count,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._request_drain_unprofiled(
                    step_count, terminal_cut, entered_draining
                )
            return
        await self._request_drain_unprofiled(step_count, terminal_cut, entered_draining)

    async def _request_drain_unprofiled(
        self, step_count: int, terminal_cut: int | None, entered_draining: bool
    ) -> None:
        if entered_draining:
            await self._notify_component_state(DriverState.DRAINING)
        if terminal_cut is not None:
            self.synchronizer.apply_boundary_event(
                DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, terminal_cut)
            )
        await self.executor.set_drain_boundary(step_count, terminal_cut)
        if not self.synchronizer.is_draining:
            self.synchronizer.drain_to_step_count(
                step_count=step_count, on_drained=self._on_drain_complete
            )
            return
        current = self.synchronizer.drain_step_count
        if current is None:
            raise RuntimeError("active synchronizer drain is missing its target")
        if step_count > current:
            return
        if step_count == current:
            return
        self.synchronizer.shorten_drain_to_step_count(
            step_count, on_drained=self._on_drain_complete
        )

    async def request_snapshot(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        repository_root: str,
    ) -> None:
        self.state_machine.require_transition(DriverEvent.SNAPSHOT_REQUESTED)
        notification = self._drained_component_notification
        if notification is not None:
            await notification
            self._drained_component_notification = None
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_snapshot(
                snapshot_id=checkpoint_id,
                target_step=cut.logical_step,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._request_snapshot_unprofiled(
                    checkpoint_id,
                    attempt_id,
                    permit,
                    cut,
                    repository_root,
                )
            return
        await self._request_snapshot_unprofiled(
            checkpoint_id, attempt_id, permit, cut, repository_root
        )

    async def _request_snapshot_unprofiled(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        repository_root: str,
    ) -> None:
        self._pending_checkpoint_id = checkpoint_id
        self._pending_attempt_id = attempt_id
        self._pending_permit = permit
        self._pending_cut = cut
        self._pending_repository_root = repository_root
        self._pending_snapshot = None
        self._published = False
        self._rollback_task = None
        await self._persist_worker_states()

    def _on_drain_complete(self) -> None:
        if self.state_machine.state is DriverState.DRAINING:
            self.state_machine.require_transition(DriverEvent.DRAIN_COMPLETE)
        elif self.state_machine.state is not DriverState.DRAINED:
            raise RuntimeError(
                "drain completion requires DRAINING or DRAINED Driver state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        self._drained_component_notification = self._schedule_component_state(
            DriverState.DRAINED
        )
        if self.controller_handle is not None:
            self.controller_handle.report_driver_drained.remote(self.spmd_id)

    async def _persist_worker_states(self) -> None:
        checkpoint_id = self._pending_checkpoint_id
        attempt_id = self._pending_attempt_id
        permit = self._pending_permit
        cut = self._pending_cut
        repository_root = self._pending_repository_root
        if (
            checkpoint_id is None
            or attempt_id is None
            or permit is None
            or cut is None
            or repository_root is None
        ):
            raise RuntimeError("Driver checkpoint transaction is incomplete")
        if self.controller_handle is None:
            raise RuntimeError("Driver checkpoint requires a Controller handle")
        try:
            await self.executor.checkpoint_prepare_save(
                repository_root, checkpoint_id, attempt_id, permit, cut
            )
            save_results = await self.executor.checkpoint_save()
            if not save_results:
                raise RuntimeError(
                    f"checkpoint save returned no workers for {self.spmd_id!r}"
                )
            await self.executor.checkpoint_finish_save()
            runtime_state = self._canonical_runtime_state(save_results)
            component_records = self._gather_component_records(save_results)
            committed = await self.executor.checkpoint_commit_save(component_records)
            publications = self._canonical_publications(committed)
            runtime_state["driver_components"] = (
                self._component_registry.checkpoint_state()
            )
            snapshot = SpmdCheckpoint(
                source_topology=self._current_topology(),
                runtime_state=runtime_state,
                components=publications,
            )
            self._pending_snapshot = snapshot
        except BaseException as primary_error:
            errors: list[BaseException] = [primary_error]
            try:
                await self._rollback_save_once()
            except BaseException as rollback_error:
                errors.append(rollback_error)
            failure = (
                errors[0]
                if len(errors) == 1
                else BaseExceptionGroup("Driver checkpoint rollback failed", errors)
            )
            logger.error(
                "checkpoint lifecycle failed spmd_id=%s error=%r",
                self.spmd_id,
                failure,
            )
            self._fail_snapshot_state()
            return
        self.controller_handle.receive_snapshot_metadata.remote(
            self.spmd_id, checkpoint_id, snapshot
        )

    async def publish_save(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._require_pending_snapshot(checkpoint_id)
        if permit != self._pending_permit:
            raise PermissionError("snapshot publication permit is stale")
        if self._published:
            raise RuntimeError("Driver checkpoint transaction is already published")
        await self.executor.checkpoint_publish_save()
        self._published = True

    async def rollback_save(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._require_pending_snapshot(checkpoint_id)
        if permit != self._pending_permit:
            raise PermissionError("snapshot rollback_save permit is stale")
        if self._published:
            raise RuntimeError(
                "published checkpoint checkpoint_objects cannot be rolled back"
            )
        await self._rollback_save_once()
        self._reset_checkpoint_transaction(preserve_checkpoint_id=True)

    def _require_pending_snapshot(self, checkpoint_id: str) -> None:
        if checkpoint_id != self._pending_checkpoint_id:
            raise ValueError(
                f"snapshot transaction does not match Driver state: {checkpoint_id!r}"
            )
        if self._pending_snapshot is None:
            raise RuntimeError("snapshot transaction has no committed publication")

    def _reset_checkpoint_transaction(
        self, *, preserve_checkpoint_id: bool = False
    ) -> None:
        if not preserve_checkpoint_id:
            self._pending_checkpoint_id = None
        self._pending_attempt_id = None
        self._pending_permit = None
        self._pending_cut = None
        self._pending_repository_root = None
        self._pending_snapshot = None
        self._published = False

    def _schedule_component_state(self, state: DriverState) -> asyncio.Task[None]:
        task = asyncio.create_task(self._notify_component_state(state))
        task.add_done_callback(self._consume_component_notification)
        return task

    @staticmethod
    def _consume_component_notification(task: asyncio.Task[Any]) -> None:
        error = task.exception()
        if error is not None:
            logger.error("driver component state notification failed: %r", error)

    def _fail_snapshot_state(self) -> None:
        self.synchronizer.clear_drain()
        self.executor.clear_drain_boundary()
        if (
            self.controller_handle is not None
            and self._pending_checkpoint_id is not None
        ):
            self.controller_handle.report_snapshot_failed.remote(
                self.spmd_id, self._pending_checkpoint_id
            )
        if self.state_machine.state is DriverState.SNAPSHOTTING:
            self.state_machine.require_transition(DriverEvent.SNAPSHOT_FAILED)

    async def _rollback_save_once(self) -> None:
        task = self._rollback_task
        if task is None:
            task = asyncio.create_task(self.executor.checkpoint_rollback_save())
            self._rollback_task = task
        await asyncio.shield(task)

    async def fail_snapshot(self, checkpoint_id: str) -> None:
        if checkpoint_id != self._pending_checkpoint_id:
            raise ValueError(
                f"snapshot transaction does not match Driver state: {checkpoint_id!r}"
            )
        if not self._published:
            await self._rollback_save_once()
        self._fail_snapshot_state()
        self._reset_checkpoint_transaction()

    async def resume(self) -> None:
        if not self.state_machine.can_transition(
            self.state_machine.state, DriverEvent.RESUME
        ):
            raise RuntimeError(
                "resume requires SNAPSHOTTING or DRAINED state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        if self._pending_snapshot is not None and not self._published:
            raise RuntimeError("Driver cannot resume before checkpoint publication")
        self.synchronizer.clear_drain()
        self.executor.clear_drain_boundary()
        self.state_machine.require_transition(DriverEvent.RESUME)
        await self._notify_component_state(DriverState.RUNNING)
        self._reset_checkpoint_transaction()

    async def restore_from_manifest(
        self,
        checkpoint_id: str,
        cut: CheckpointCut,
        snapshot: SpmdCheckpoint,
        repository_root: str,
    ) -> None:
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_restore(
                checkpoint_id=checkpoint_id,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._restore_from_manifest_unprofiled(
                    checkpoint_id, cut, snapshot, repository_root
                )
            return
        await self._restore_from_manifest_unprofiled(
            checkpoint_id, cut, snapshot, repository_root
        )

    async def _restore_from_manifest_unprofiled(
        self,
        checkpoint_id: str,
        cut: CheckpointCut,
        snapshot: SpmdCheckpoint,
        repository_root: str,
    ) -> None:
        self.state_machine.require_transition(DriverEvent.RESTORE_REQUESTED)
        await self._notify_component_state(DriverState.RESTORING)
        try:
            runtime_state = dict(snapshot.runtime_state)
            driver_components = runtime_state.pop("driver_components", None)
            if not isinstance(driver_components, Mapping):
                raise ValueError("SPMD snapshot is missing driver component state")
            self._component_registry.restore_state(driver_components)
            self.synchronizer.load_spmd_state(runtime_state)
            await self.executor.checkpoint_restore(
                repository_root,
                checkpoint_id,
                cut,
                snapshot.source_topology,
                self._current_topology(),
                runtime_state,
                snapshot.components,
            )
        except BaseException:
            if self.state_machine.state is DriverState.RESTORING:
                self.state_machine.require_transition(DriverEvent.RESTORE_FAILED)
            raise
        self.state_machine.require_transition(DriverEvent.RESTORE_COMPLETE)
        await self._notify_component_state(DriverState.READY)

    @staticmethod
    def _canonical_runtime_state(
        save_results: list[tuple[int, dict[str, Any], Mapping[str, WorkerSaveResult]]],
    ) -> dict[str, Any]:
        rank, first, _ = save_results[0]
        for candidate_rank, candidate, _ in save_results[1:]:
            if candidate != first:
                raise ValueError(
                    "worker runtime checkpoint states disagree: "
                    f"rank={rank} state={first!r} rank={candidate_rank} "
                    f"state={candidate!r}"
                )
        return dict(first)

    @staticmethod
    def _gather_component_records(
        save_results: list[tuple[int, dict[str, Any], Mapping[str, WorkerSaveResult]]],
    ) -> dict[str, dict[int, WorkerSaveResult]]:
        expected = set(save_results[0][2])
        gathered: dict[str, dict[int, WorkerSaveResult]] = {
            component_id: {} for component_id in expected
        }
        for rank, _, records in save_results:
            if set(records) != expected:
                raise ValueError(
                    "worker checkpoint component sets disagree: "
                    f"expected={sorted(expected)} rank={rank} "
                    f"actual={sorted(records)}"
                )
            for component_id, record in records.items():
                gathered[component_id][rank] = record
        return gathered

    @staticmethod
    def _canonical_publications(
        committed: Mapping[int, Mapping[str, CheckpointComponentPublication]],
    ) -> dict[str, CheckpointComponentPublication]:
        publications: dict[str, CheckpointComponentPublication] = {}
        for rank, rank_publications in committed.items():
            for component_id, publication in rank_publications.items():
                previous = publications.get(component_id)
                if previous is not None and previous != publication:
                    raise ValueError(
                        "checkpoint component publications disagree across ranks: "
                        f"component={component_id!r} rank={rank}"
                    )
                publications[component_id] = publication
        return publications

    def _current_topology(self) -> TopologySnapshot:
        view = self.executor.config.require_graph_view()
        mesh = view.mesh
        if mesh is None:
            raise RuntimeError(
                f"checkpoint requires device mesh for spmd_id={self.spmd_id!r}"
            )
        return TopologySnapshot(
            world_size=view.world_size,
            mesh_axis_names=tuple(mesh.axis_names),
            mesh_axis_shapes=tuple(mesh.shape),
            partition_scheme_version="fluxway-device-mesh-v1",
        )
