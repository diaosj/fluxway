from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fluxway.common.types import (
    ComponentCheckpoint,
    CheckpointCaptureMode,
    CheckpointCommitPermit,
    CheckpointCut,
    DriverEvent,
    DriverState,
    WorkerSaveResult,
    SpmdCheckpoint,
    TopologySnapshot,
)
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.components import DriverComponentRegistry

logger = get_logger(__name__)


class SnapshotOrchestrator:
    def __init__(
        self,
        spmd_id: str,
        executor: Any,
        synchronizer: Any,
        controller_handle: Any | None,
        get_driver_state: Callable[[], DriverState],
        trigger_state_event: Callable[[DriverEvent], Any],
        profiler_runtime: Any,
        component_registry: DriverComponentRegistry,
        notify_component_state: Callable[[DriverState], Any],
    ) -> None:
        self.spmd_id = spmd_id
        self.executor = executor
        self.synchronizer = synchronizer
        self.controller_handle = controller_handle
        self.get_driver_state = get_driver_state
        self.trigger_state_event = trigger_state_event
        self._profiler_runtime = profiler_runtime
        self._component_registry = component_registry
        self._notify_component_state = notify_component_state
        self._pending_checkpoint_id: str | None = None
        self._pending_attempt_id: str | None = None
        self._pending_permit: CheckpointCommitPermit | None = None
        self._pending_drain_step_count: int | None = None
        self._pending_cut: CheckpointCut | None = None
        self._pending_repository_root: str | None = None
        self._pending_snapshot: SpmdCheckpoint | None = None
        self._published = False

    async def request_drain(self, step_count: int, reason: str = "") -> None:
        if self.get_driver_state() != DriverState.RUNNING:
            raise RuntimeError(
                f"request_drain requires RUNNING state, got "
                f"{self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_drain(
                target_step=step_count,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._request_drain_unprofiled(step_count)
            return
        await self._request_drain_unprofiled(step_count)

    async def _request_drain_unprofiled(self, step_count: int) -> None:
        self._pending_drain_step_count = step_count
        await self._notify_component_state(DriverState.DRAINING)
        self.trigger_state_event(DriverEvent.DRAIN_REQUESTED)
        await self.executor.set_drain_step_count(step_count)
        self.synchronizer.drain_to_step_count(
            step_count=step_count, on_drained=self._on_drain_complete
        )

    async def request_terminal_drain(self, step_count: int) -> None:
        state = self.get_driver_state()
        if state is DriverState.RUNNING:
            await self._request_drain_unprofiled(step_count)
            return
        if state is DriverState.DRAINING:
            current = self._pending_drain_step_count
            if current is None:
                raise RuntimeError("active Driver drain is missing its step count")
            if step_count > current:
                raise ValueError(
                    "terminal drain cannot extend active step count: "
                    f"current={current} terminal={step_count}"
                )
            self._pending_drain_step_count = step_count
            await self.executor.set_drain_step_count(step_count)
            self.synchronizer.shorten_drain_to_step_count(step_count)
            return
        if state is DriverState.DRAINED:
            current = self._pending_drain_step_count
            if current is None or step_count > current:
                raise RuntimeError(
                    "terminal step is not covered by the completed Driver drain"
                )
            return
        raise RuntimeError(
            "terminal drain requires RUNNING, DRAINING, or DRAINED Driver state, "
            f"got {state}"
        )

    async def request_snapshot(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        repository_root: str,
        capture_mode: CheckpointCaptureMode,
    ) -> None:
        if self.get_driver_state() != DriverState.DRAINED:
            raise RuntimeError(
                f"request_snapshot requires DRAINED state, got "
                f"{self.get_driver_state()} spmd_id={self.spmd_id}"
            )
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
                    capture_mode,
                )
            return
        await self._request_snapshot_unprofiled(
            checkpoint_id, attempt_id, permit, cut, repository_root, capture_mode
        )

    async def _request_snapshot_unprofiled(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        repository_root: str,
        capture_mode: CheckpointCaptureMode,
    ) -> None:
        await self._notify_component_state(DriverState.DRAINED)
        self.trigger_state_event(DriverEvent.SNAPSHOT_REQUESTED)
        self._pending_checkpoint_id = checkpoint_id
        self._pending_attempt_id = attempt_id
        self._pending_permit = permit
        self._pending_cut = cut
        self._pending_repository_root = repository_root
        self._pending_snapshot = None
        self._published = False
        await self._persist_worker_states(capture_mode)

    def _on_drain_complete(self) -> None:
        if self.get_driver_state() != DriverState.DRAINING:
            raise RuntimeError(
                f"_on_drain_complete requires DRAINING state, got "
                f"{self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        self.trigger_state_event(DriverEvent.DRAIN_COMPLETE)
        if self.controller_handle is not None:
            self.controller_handle.report_driver_drained.remote(self.spmd_id)

    async def _persist_worker_states(self, capture_mode: CheckpointCaptureMode) -> None:
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
            await self.executor.checkpoint_prepare(
                repository_root, checkpoint_id, attempt_id, permit, cut, capture_mode
            )
            captures = await self.executor.checkpoint_capture()
            if not captures:
                raise RuntimeError(
                    f"checkpoint save returned no workers for {self.spmd_id!r}"
                )
            await self.executor.checkpoint_finalize()
            runtime_state = self._canonical_runtime_state(captures)
            gathered = self._gather_backend_records(captures)
            committed = await self.executor.checkpoint_commit(gathered)
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
                await self.executor.checkpoint_abort()
            except BaseException as abort_error:
                errors.append(abort_error)
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
            self._clear_drain_and_fail()
            return
        self.controller_handle.receive_snapshot_metadata.remote(
            self.spmd_id, checkpoint_id, snapshot
        )

    async def mark_available(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._require_pending_snapshot(checkpoint_id)
        if permit != self._pending_permit:
            raise PermissionError("snapshot publication permit is stale")
        if self._published:
            raise RuntimeError("Driver checkpoint transaction is already published")
        await self.executor.checkpoint_mark_published()
        self._published = True

    async def abort_checkpoint_objects(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._require_pending_snapshot(checkpoint_id)
        if permit != self._pending_permit:
            raise PermissionError("snapshot rollback_save permit is stale")
        if self._published:
            raise RuntimeError(
                "published checkpoint checkpoint_objects cannot be aborted"
            )
        await self.executor.checkpoint_abort()
        self._pending_snapshot = None
        self._pending_attempt_id = None
        self._pending_permit = None

    def _require_pending_snapshot(self, checkpoint_id: str) -> None:
        if checkpoint_id != self._pending_checkpoint_id:
            raise ValueError(
                f"snapshot transaction does not match Driver state: {checkpoint_id!r}"
            )
        if self._pending_snapshot is None:
            raise RuntimeError("snapshot transaction has no committed publication")

    def _clear_drain_and_fail(self) -> None:
        self.synchronizer.clear_drain()
        self.executor.clear_drain_step_count()
        if (
            self.controller_handle is not None
            and self._pending_checkpoint_id is not None
        ):
            self.controller_handle.report_snapshot_failed.remote(
                self.spmd_id, self._pending_checkpoint_id
            )
        self.trigger_state_event(DriverEvent.SNAPSHOT_FAILED)

    async def resume(self) -> None:
        if self.get_driver_state() not in (
            DriverState.SNAPSHOTTING,
            DriverState.DRAINED,
        ):
            raise RuntimeError(
                "resume requires SNAPSHOTTING or DRAINED state, "
                f"got {self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        if self._pending_snapshot is not None and not self._published:
            raise RuntimeError("Driver cannot resume before checkpoint publication")
        self.synchronizer.clear_drain()
        self.executor.clear_drain_step_count()
        await self._notify_component_state(DriverState.RUNNING)
        self._pending_snapshot = None
        self._pending_attempt_id = None
        self._pending_permit = None
        self._pending_cut = None
        self._pending_repository_root = None
        self._published = False
        self.trigger_state_event(DriverEvent.SNAPSHOT_COMPLETE)

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
        await self._notify_component_state(DriverState.DRAINING)
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
        await self._notify_component_state(DriverState.RUNNING)

    @staticmethod
    def _canonical_runtime_state(
        captures: list[tuple[int, dict[str, Any], Mapping[str, WorkerSaveResult]]],
    ) -> dict[str, Any]:
        rank, first, _ = captures[0]
        for candidate_rank, candidate, _ in captures[1:]:
            if candidate != first:
                raise ValueError(
                    "worker runtime checkpoint states disagree: "
                    f"rank={rank} state={first!r} rank={candidate_rank} "
                    f"state={candidate!r}"
                )
        return dict(first)

    @staticmethod
    def _gather_backend_records(
        captures: list[tuple[int, dict[str, Any], Mapping[str, WorkerSaveResult]]],
    ) -> dict[str, dict[int, WorkerSaveResult]]:
        expected = set(captures[0][2])
        gathered: dict[str, dict[int, WorkerSaveResult]] = {
            component_id: {} for component_id in expected
        }
        for rank, _, records in captures:
            if set(records) != expected:
                raise ValueError(
                    "worker checkpoint backend sets disagree: "
                    f"expected={sorted(expected)} rank={rank} "
                    f"actual={sorted(records)}"
                )
            for component_id, record in records.items():
                gathered[component_id][rank] = record
        return gathered

    @staticmethod
    def _canonical_publications(
        committed: Mapping[int, Mapping[str, ComponentCheckpoint]],
    ) -> dict[str, ComponentCheckpoint]:
        publications: dict[str, ComponentCheckpoint] = {}
        for rank, rank_publications in committed.items():
            for component_id, publication in rank_publications.items():
                previous = publications.get(component_id)
                if previous is not None and previous != publication:
                    raise ValueError(
                        "checkpoint backend publications disagree across ranks: "
                        f"backend={component_id!r} rank={rank}"
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
