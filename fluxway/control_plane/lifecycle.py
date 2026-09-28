from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from fluxway.common.types import (
    ControllerEvent,
    ControllerState,
    DrainEvent,
    DrainEventKind,
    DrainPhase,
    ExecutionFailure,
    SaveCheckpointRequest,
)
from fluxway.common.types.drain import DrainAdmission
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.errors import ExecutionFailedError

logger = get_logger(__name__)

if TYPE_CHECKING:
    from fluxway.control_plane.controller import ControllerActor


class ExecutionLifecycle:
    def __init__(self, controller: ControllerActor) -> None:
        self._controller = controller
        self._delivery: tuple[DrainAdmission, asyncio.Task[None]] | None = None
        self._operations: set[asyncio.Task[None]] = set()
        self._terminal_cleanup: asyncio.Task[None] | None = None
        self._startup: asyncio.Task[int | None] | None = None

    async def transition(self, event: ControllerEvent) -> None:
        if not await self._controller.state_machine.trigger_event(event):
            raise RuntimeError(f"execution lifecycle rejected {event}")

    async def enter_state(self, state: ControllerState) -> None:
        if state in (
            ControllerState.COMPLETED,
            ControllerState.FAILED,
            ControllerState.CANCELLED,
        ):
            return
        if state in (
            ControllerState.COMPLETING,
            ControllerState.FAILING,
            ControllerState.CANCELLING,
        ):
            self.begin_terminal_cleanup(state)
            return
        controller = self._controller
        try:
            if controller._profiler_runtime.enabled:
                async with (
                    controller._profiler_runtime.recorder.controller_fsm_entry_callback(
                        state=state.name
                    )
                ):
                    await self._enter_state(state)
            else:
                await self._enter_state(state)
        except Exception as error:
            if state not in (ControllerState.COMPLETED, ControllerState.FAILED):
                await self.fail(error, state.name)
            raise

    async def _enter_state(self, state: ControllerState) -> None:
        controller = self._controller
        await controller._notify_control_plane_components(state)
        if state is ControllerState.TOPOLOGY_REGISTERING:
            await controller._initiate_topology_registration()
        elif state is ControllerState.READY:
            controller._ready_event.set()
        elif state is ControllerState.RUNNING:
            if controller._heartbeat_manager is not None:
                await controller._heartbeat_manager.start_monitoring_async()
            if (
                controller.async_step_manager.is_all_completed()
                and controller._drain_strategy is None
            ):
                await self.transition(ControllerEvent.ALL_DRIVERS_COMPLETED)

    async def start(self, resume: str | int | None) -> int | None:
        if (
            self._terminal_cleanup is not None
            or self._controller.state_machine.is_completed()
        ):
            raise RuntimeError("execution startup requires a live execution")
        if self._startup is not None:
            raise RuntimeError("execution startup is already registered")
        self._startup = asyncio.create_task(self._run_startup(resume))
        self._startup.add_done_callback(self._startup_finished)
        try:
            return await asyncio.shield(self._startup)
        except asyncio.CancelledError:
            failures = self._controller.state_machine.get_failures()
            if self._startup.cancelled() and failures:
                raise ExecutionFailedError(failures) from None
            raise

    def _startup_finished(self, startup: asyncio.Task[int | None]) -> None:
        if not startup.cancelled():
            startup.exception()

    async def _run_startup(self, resume: str | int | None) -> int | None:
        try:
            return await self._start(resume)
        except Exception as error:
            await self.fail(error, "STARTUP")
            raise

    async def _start(self, resume: str | int | None) -> int | None:
        controller = self._controller
        await controller.initialize()
        await controller._ready_event.wait()
        if controller.state_machine.state is not ControllerState.READY:
            raise RuntimeError("execution did not reach READY")
        await self.transition(ControllerEvent.START)
        restored_step = None
        if resume is not None:
            restored_step = await self.restore(resume)
        if controller._graph.requires_source_control_runtime:
            await controller.activate_source_layout_from_descriptor()
        self.require_state(ControllerState.ACTIVATING)
        await self.fanout("activate_execution")
        self.require_state(ControllerState.ACTIVATING)
        if controller._drain_boundary.observed_cut != 0:
            await self.fanout("start_source_operators")
        await self.transition(ControllerEvent.ACTIVATION_COMPLETE)
        await self.admit(initial=True)
        return restored_step

    async def restore(self, reference: object) -> int | None:
        controller = self._controller
        if controller.state_machine.state not in (
            ControllerState.READY,
            ControllerState.ACTIVATING,
        ):
            raise RuntimeError("checkpoint restore requires pre-execution readiness")
        if controller._checkpointer is None:
            raise ValueError("Cannot restore without a checkpoint_config")
        return await controller._checkpointer.restore(reference)

    async def resume(self) -> None:
        try:
            if self.completion_reached():
                await self.transition(ControllerEvent.ALL_DRIVERS_COMPLETED)
                return
            await self.transition(ControllerEvent.RESUME)
            await self.fanout("resume")
            await self.transition(ControllerEvent.ACTIVATION_COMPLETE)
            await self.admit(initial=False)
        except Exception as error:
            await self.fail(error, "RESUME")
            raise

    def completion_reached(self) -> bool:
        controller = self._controller
        boundary = controller._drain_boundary
        max_steps = controller.async_step_manager._config.max_steps
        return boundary.phase is DrainPhase.COMPLETED or (
            max_steps is not None and (boundary.committed_cut or 0) >= max_steps
        )

    async def drained(self) -> None:
        controller = self._controller
        boundary = controller._drain_boundary
        target = boundary.admitted_cut
        if target is None:
            raise RuntimeError("drain completion is missing its admitted target")
        if boundary.phase is not DrainPhase.DRAINED or boundary.drained_cut != target:
            return
        admission = controller._admission
        if controller._source_control is not None:
            await controller._source_control.wait_for_work_cut(target)
        if controller._admission != admission:
            return
        await controller._reduce_drain_event(
            DrainEvent(DrainEventKind.SNAPSHOT_BEGIN, target)
        )
        if controller._checkpointer is not None and target != boundary.committed_cut:
            controller._checkpoint_task = asyncio.create_task(self.checkpoint(target))
        else:
            await self.commit(target)

    async def checkpoint(self, target: int) -> None:
        controller = self._controller
        try:
            checkpointer = controller._checkpointer
            if checkpointer is None:
                raise RuntimeError(
                    "checkpoint lifecycle requires a configured checkpointer"
                )
            request = SaveCheckpointRequest.for_step(target)
            if controller._profiler_runtime.enabled:
                async with controller._profiler_runtime.recorder.controller_checkpoint(
                    target_step=target
                ):
                    await checkpointer.save(request)
            else:
                await checkpointer.save(request)
            await self.commit(target)
        except Exception as error:
            await self.fail(error, "CHECKPOINT")
            raise

    async def commit(self, target: int) -> None:
        await self._controller._reduce_drain_event(
            DrainEvent(DrainEventKind.CHECKPOINT_COMMITTED, target)
        )
        await self.resume()

    def require_state(self, expected: ControllerState) -> None:
        actual = self._controller.state_machine.state
        if actual is not expected:
            raise RuntimeError(f"execution requires {expected}, got {actual}")

    async def fail(self, error: Exception, phase: str) -> None:
        machine = self._controller.state_machine
        if machine.state is not ControllerState.FAILED:
            await machine.trigger_event(
                ControllerEvent.DRIVER_ERROR,
                spmd_id="controller",
                failure=ExecutionFailure.from_exception(
                    spmd_id="controller",
                    rank=None,
                    phase=phase,
                    exception=error,
                ),
            )

    async def admit(self, *, initial: bool) -> None:
        controller = self._controller
        if controller._drain_boundary.phase is DrainPhase.DRAINING:
            await self.deliver_current()
        elif controller._drain_strategy is not None:
            if initial:
                await controller._issue_initial_drain()
            else:
                target = controller._drain_strategy.get_next_drain_target(
                    controller._drain_boundary.committed_cut or 0
                )
                if target is not None:
                    await controller._schedule_drain(target)

    async def deliver_current(self) -> None:
        controller = self._controller
        if not controller.state_machine.is_running():
            return
        admission = controller._admission
        if admission is None:
            return
        if self._delivery is None or self._delivery[0] != admission:
            try:
                operation = controller.spmd_manager.enqueue_driver_calls(
                    "request_drain", admission
                )
            except Exception as error:
                await self.fail(error, "ADMISSION")
                raise
            supervised = asyncio.create_task(self._supervise_admission(operation))
            self._operations.add(supervised)
            supervised.add_done_callback(self._operation_finished)
            self._delivery = (admission, supervised)
        await asyncio.shield(self._delivery[1])

    async def _supervise_admission(self, operation: asyncio.Future) -> None:
        try:
            await operation
        except Exception as error:
            await self.fail(error, "ADMISSION")
            raise

    def _operation_finished(self, operation: asyncio.Task[None]) -> None:
        self._operations.remove(operation)
        if not operation.cancelled():
            operation.exception()

    def begin_terminal_cleanup(self, state: ControllerState) -> None:
        if state not in (
            ControllerState.COMPLETING,
            ControllerState.FAILING,
            ControllerState.CANCELLING,
        ):
            raise ValueError("terminal cleanup requires a terminal transition state")
        if self._terminal_cleanup is not None:
            return
        cleanup = asyncio.create_task(self._run_terminal_cleanup(state))
        self._terminal_cleanup = cleanup
        cleanup.add_done_callback(self._terminal_cleanup_finished)

    async def cancel(self) -> None:
        machine = self._controller.state_machine
        if self._terminal_cleanup is None and not machine.is_completed():
            await self.transition(ControllerEvent.CANCEL)
        if self._terminal_cleanup is not None:
            await asyncio.shield(self._terminal_cleanup)
        if machine.state is ControllerState.FAILED:
            await self._controller._wait_for_completion_unprofiled(None)

    async def _run_terminal_cleanup(self, state: ControllerState) -> None:
        controller = self._controller
        errors: list[Exception] = []
        if state is not ControllerState.COMPLETING and self._startup is not None:
            if not self._startup.done():
                self._startup.cancel()
            await asyncio.gather(self._startup, return_exceptions=True)
        try:
            await controller._notify_control_plane_components(state)
        except Exception as error:
            logger.exception("Terminal component notification failed")
            errors.append(error)
        try:
            await controller._finalize_runtime(
                export_profiler_trace=state is ControllerState.COMPLETING
            )
        except Exception as error:
            logger.exception("Terminal resource finalization failed")
            errors.append(error)
        if controller._spmd_manager is not None:
            try:
                await controller._spmd_manager.close_driver_calls()
            except Exception as error:
                errors.append(error)
        await asyncio.gather(*tuple(self._operations), return_exceptions=True)
        outcome = {
            ControllerState.FAILING: ControllerState.FAILED,
            ControllerState.CANCELLING: ControllerState.CANCELLED,
            ControllerState.COMPLETING: ControllerState.COMPLETED,
        }[controller.state_machine.state]
        if errors:
            outcome = ControllerState.FAILED
        try:
            await controller._notify_control_plane_components(outcome)
        except Exception as error:
            errors.append(error)
        if errors:
            error = (
                errors[0]
                if len(errors) == 1
                else ExceptionGroup("terminal cleanup failed", errors)
            )
            await controller.state_machine.trigger_event(
                ControllerEvent.CLEANUP_FAILED,
                failure=ExecutionFailure.from_exception(
                    spmd_id="controller",
                    rank=None,
                    phase="CONTROLLER_CLEANUP",
                    exception=error,
                ),
            )
            return
        await controller.state_machine.trigger_event(ControllerEvent.CLEANUP_FINISHED)

    def _terminal_cleanup_finished(self, cleanup: asyncio.Task[None]) -> None:
        if not cleanup.cancelled():
            cleanup.exception()

    async def fanout(
        self, method: Literal["activate_execution", "start_source_operators", "resume"]
    ) -> None:
        await self._controller.spmd_manager.enqueue_driver_calls(method)
