import pathlib
from typing import Any
from collections.abc import Callable

from fluxway.common.types import DriverEvent, DriverState
from fluxway.control_plane.components import DriverComponentRegistry
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.checkpointer.protocol import validate_checkpoint_state

logger = get_logger(__name__)


class SnapshotOrchestrator:
    """Manages snapshot and checkpoint orchestration for the SPMD Driver."""

    def __init__(
        self,
        spmd_id: str,
        executor: Any,
        synchronizer: Any,
        checkpoint_backend: Any,
        controller_handle: Any | None,
        get_driver_state: Callable[[], DriverState],
        trigger_state_event: Callable[[DriverEvent], Any],
        profiler_runtime: Any,
        component_registry: DriverComponentRegistry,
        notify_component_state: Callable[[DriverState], Any],
    ):
        self.spmd_id = spmd_id
        self.executor = executor
        self.synchronizer = synchronizer
        self._checkpoint_backend = checkpoint_backend
        self.controller_handle = controller_handle
        self.get_driver_state = get_driver_state
        self.trigger_state_event = trigger_state_event
        self._profiler_runtime = profiler_runtime
        self._component_registry = component_registry
        self._notify_component_state = notify_component_state

        self._pending_snapshot_id: str | None = None
        self._pending_target_step_count: int | None = None
        self._pending_ckpt_dir: str | None = None
        self._pending_artifact_metadata: dict[str, dict[int, dict]] | None = None
        self._artifacts_committed = False

    async def request_drain(self, target_count: int, reason: str = "") -> None:
        if self.get_driver_state() != DriverState.RUNNING:
            raise RuntimeError(
                f"request_drain requires RUNNING state, "
                f"got {self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_drain(
                target_step=target_count,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._request_drain_unprofiled(target_count)
            return
        await self._request_drain_unprofiled(target_count)

    async def _request_drain_unprofiled(self, target_count: int) -> None:
        self._pending_target_step_count = target_count
        await self._notify_component_state(DriverState.DRAINING)
        self.trigger_state_event(DriverEvent.DRAIN_REQUESTED)
        await self.executor.set_drain_target_count(target_count)
        self.synchronizer.drain_to_step_count(
            target_count=target_count, on_drained=self._on_drain_complete
        )

    async def request_snapshot(
        self, snapshot_id: str, target_count: int, ckpt_dir: str
    ) -> None:
        if self.get_driver_state() != DriverState.DRAINED:
            raise RuntimeError(
                f"request_snapshot requires DRAINED state, "
                f"got {self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_snapshot(
                snapshot_id=snapshot_id,
                target_step=target_count,
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._request_snapshot_unprofiled(
                    snapshot_id,
                    target_count,
                    ckpt_dir,
                )
            return
        await self._request_snapshot_unprofiled(snapshot_id, target_count, ckpt_dir)

    async def request_terminal_drain(self, target_count: int) -> None:
        state = self.get_driver_state()
        if state is DriverState.RUNNING:
            await self._request_drain_unprofiled(target_count)
            return
        if state is DriverState.DRAINING:
            current = self._pending_target_step_count
            if current is None:
                raise RuntimeError("active Driver drain is missing its target")
            if target_count > current:
                raise ValueError(
                    f"terminal drain cannot extend active target: "
                    f"current={current} terminal={target_count}"
                )
            self._pending_target_step_count = target_count
            await self.executor.set_drain_target_count(target_count)
            self.synchronizer.shorten_drain_to_step_count(target_count)
            return
        if state is DriverState.DRAINED:
            current = self._pending_target_step_count
            if current is None or target_count > current:
                raise RuntimeError(
                    "terminal target is not covered by the completed Driver drain"
                )
            return
        raise RuntimeError(
            f"terminal drain requires RUNNING, DRAINING, or DRAINED Driver state, "
            f"got {state}"
        )

    async def _request_snapshot_unprofiled(
        self, snapshot_id: str, target_count: int, ckpt_dir: str
    ) -> None:
        await self._notify_component_state(DriverState.DRAINED)
        self.trigger_state_event(DriverEvent.SNAPSHOT_REQUESTED)
        self._pending_snapshot_id = snapshot_id
        self._pending_target_step_count = target_count
        self._pending_ckpt_dir = ckpt_dir
        self._pending_artifact_metadata = None
        self._artifacts_committed = False
        await self._persist_worker_states()

    def _on_drain_complete(self) -> None:
        if self.get_driver_state() != DriverState.DRAINING:
            raise RuntimeError(
                f"_on_drain_complete requires DRAINING state, "
                f"got {self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        self.trigger_state_event(DriverEvent.DRAIN_COMPLETE)
        if self.controller_handle is not None:
            self.controller_handle.report_driver_drained.remote(self.spmd_id)

    async def _persist_worker_states(self) -> None:
        if self._pending_ckpt_dir is None or self._pending_target_step_count is None:
            logger.error(
                f"Failed to persist worker states: pending_ckpt_dir or pending_target_step_count is not set. spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return
        spmd_dir = (
            pathlib.Path(self._pending_ckpt_dir)
            / str(self._pending_target_step_count)
            / self.spmd_id
        )
        try:
            force_write = getattr(self._checkpoint_backend, "_force_write", False)
            if spmd_dir.exists() and any(spmd_dir.iterdir()):
                if not force_write:
                    raise FileExistsError(
                        f"Checkpoint directory {spmd_dir} already exists and is not empty. "
                        "Set force_write=True in CheckpointConfig to overwrite."
                    )
                else:
                    import shutil

                    shutil.rmtree(spmd_dir, ignore_errors=True)

            spmd_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Failed to create checkpoint directory {spmd_dir} error={e} spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return

        if self.controller_handle is None:
            logger.error(
                f"controller_handle is None, cannot report snapshot metadata "
                f"spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return

        try:
            await self.executor.prepare_checkpoint_artifacts(
                self._pending_ckpt_dir, self._pending_target_step_count
            )
            results = await self.executor.save_state_to_disk(
                self._pending_ckpt_dir, self._pending_target_step_count
            )

            from collections import defaultdict
            import copy

            gathered_metadata: dict[str, dict[int, dict]] = defaultdict(dict)
            ranks = []
            for rank, spmd_state, rank_storage, op_states in results:
                ranks.append(rank)
                for op_name, op_state in op_states.items():
                    gathered_metadata[op_name][rank] = op_state

            if not results:
                raise ValueError(
                    f"No results from save_state_to_disk spmd_id={self.spmd_id}"
                )

            canonical_spmd_state = results[0][1]
            shared_operator_states = {}
            for op_name, per_rank_states in gathered_metadata.items():
                if ranks[0] not in per_rank_states:
                    continue
                candidate_state = copy.deepcopy(per_rank_states[ranks[0]])
                try:
                    validate_checkpoint_state(
                        candidate_state,
                        path=(
                            f"manifest.{self.spmd_id}."
                            f"spmd_state.operator_states.{op_name}"
                        ),
                    )
                except TypeError as exc:
                    logger.debug(
                        "Operator state remains rank-local checkpoint storage "
                        "op=%s spmd_id=%s reason=%s",
                        op_name,
                        self.spmd_id,
                        exc,
                    )
                    continue
                shared_operator_states[op_name] = candidate_state
            canonical_spmd_state = dict(canonical_spmd_state)
            canonical_spmd_state["operator_states"] = shared_operator_states
            canonical_spmd_state["driver_components"] = (
                self._component_registry.checkpoint_state()
            )
            self._pending_artifact_metadata = dict(gathered_metadata)

            rank_metadata: dict[int, tuple[dict, dict]] = {
                rank: (canonical_spmd_state, rank_storage)
                for rank, _, rank_storage, _ in results
            }

        except Exception as error:
            errors = [error]
            try:
                await self.executor.abort_checkpoint_artifacts(
                    self._pending_ckpt_dir,
                    self._pending_target_step_count,
                )
            except Exception as abort_error:
                errors.append(abort_error)
            failure = (
                errors[0]
                if len(errors) == 1
                else ExceptionGroup("snapshot persistence rollback failed", errors)
            )
            logger.error(
                "Snapshot lifecycle error for spmd_id=%s: %r",
                self.spmd_id,
                failure,
            )
            self._clear_drain_and_fail()
            return

        self.controller_handle.receive_snapshot_metadata.remote(
            self.spmd_id, self._pending_snapshot_id, rank_metadata
        )

    async def commit_artifacts(self, snapshot_id: str) -> None:
        if snapshot_id != self._pending_snapshot_id:
            raise ValueError(
                f"snapshot transaction does not match Driver state: {snapshot_id!r}"
            )
        if (
            self._pending_ckpt_dir is None
            or self._pending_target_step_count is None
            or self._pending_artifact_metadata is None
        ):
            raise RuntimeError("snapshot transaction is missing prepared artifacts")
        if self._artifacts_committed:
            return
        await self.executor.commit_checkpoint_artifacts(
            self._pending_ckpt_dir,
            self._pending_target_step_count,
            self._pending_artifact_metadata,
        )
        self._artifacts_committed = True

    async def abort_artifacts(self, snapshot_id: str) -> None:
        if snapshot_id != self._pending_snapshot_id:
            raise ValueError(
                f"snapshot transaction does not match Driver state: {snapshot_id!r}"
            )
        if self._pending_ckpt_dir is None or self._pending_target_step_count is None:
            raise RuntimeError("snapshot transaction is missing artifact location")
        await self.executor.abort_checkpoint_artifacts(
            self._pending_ckpt_dir,
            self._pending_target_step_count,
        )
        self._pending_artifact_metadata = None
        self._artifacts_committed = False

    def _clear_drain_and_fail(self) -> None:
        self.synchronizer.clear_drain()
        self.executor.clear_drain_target_count()
        if self.controller_handle and self._pending_snapshot_id:
            self.controller_handle.report_snapshot_failed.remote(
                self.spmd_id, self._pending_snapshot_id
            )
        self.trigger_state_event(DriverEvent.SNAPSHOT_FAILED)

    async def resume(self) -> None:
        if self.get_driver_state() not in (
            DriverState.SNAPSHOTTING,
            DriverState.DRAINED,
        ):
            raise RuntimeError(
                f"resume requires SNAPSHOTTING or DRAINED state, "
                f"got {self.get_driver_state()} spmd_id={self.spmd_id}"
            )
        self.synchronizer.clear_drain()
        self.executor.clear_drain_target_count()
        await self._notify_component_state(DriverState.RUNNING)
        self._pending_artifact_metadata = None
        self._artifacts_committed = False
        self.trigger_state_event(DriverEvent.SNAPSHOT_COMPLETE)

    async def restore_from_disk(
        self,
        spmd_state: dict[str, Any],
        rank_storage: dict[int, dict[str, Any]],
        step: int,
    ) -> None:
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_restore(
                snapshot_id="restore_from_disk",
                worker_count=len(self.executor.get_worker_handles()),
            ):
                await self._restore_from_disk_unprofiled(spmd_state, rank_storage, step)
            return
        await self._restore_from_disk_unprofiled(spmd_state, rank_storage, step)

    async def _restore_from_disk_unprofiled(
        self,
        spmd_state: dict[str, Any],
        rank_storage: dict[int, dict[str, Any]],
        step: int,
    ) -> None:
        await self._notify_component_state(DriverState.DRAINING)
        if not isinstance(spmd_state, dict):
            raise TypeError("SPMD checkpoint state must be a dictionary")
        driver_components = spmd_state.get("driver_components")
        if not isinstance(driver_components, dict):
            raise ValueError("SPMD checkpoint state is missing driver_components")
        self._component_registry.restore_state(driver_components)
        worker_spmd_state = dict(spmd_state)
        worker_spmd_state.pop("driver_components", None)
        self.synchronizer.load_spmd_state(worker_spmd_state)
        await self.executor.load_state_from_disk(worker_spmd_state, rank_storage, step)
        await self._notify_component_state(DriverState.RUNNING)
