"""
ControllerStateMachine for managing Controller lifecycle.

State transitions (event-driven initialization):
- INITIALIZING -> PENDING (on INIT_COMPLETE: resources allocated, SPMDs started)
- PENDING -> TOPOLOGY_REGISTERING (on ALL_DRIVERS_READY: all drivers report workers_ready)
    -> triggers topology registration
- TOPOLOGY_REGISTERING -> RUNNING (on ALL_SPMDS_TOPOLOGY_READY: all workers running)
    -> starts heartbeat timeout monitoring
- PENDING / TOPOLOGY_REGISTERING / RUNNING -> FAILING (on DRIVER_ERROR)
- FAILING -> FAILED
- RUNNING -> COMPLETING -> COMPLETED

Supports async callbacks for Ray Actor Controller.
"""

import asyncio
from typing import TYPE_CHECKING

from fluxway.common.types import ControllerState, ControllerEvent
from fluxway.common.utils.logging import get_logger

if TYPE_CHECKING:
    from fluxway.control_plane.controller import ControllerActor

logger = get_logger(__name__)


class ControllerStateMachine:
    """
    State machine for Controller lifecycle management.

    Handles state transitions based on events from heartbeat manager
    and driver reports. Supports async callbacks for Ray Actor.
    """

    # State transition table: (current_state, event) -> next_state
    _TRANSITIONS = {
        # --- Event-driven initialization flow ---
        (
            ControllerState.INITIALIZING,
            ControllerEvent.INIT_COMPLETE,
        ): ControllerState.PENDING,
        (
            ControllerState.PENDING,
            ControllerEvent.ALL_DRIVERS_READY,
        ): ControllerState.TOPOLOGY_REGISTERING,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.ALL_SPMDS_TOPOLOGY_READY,
        ): ControllerState.RUNNING,
        # --- Normal operation ---
        (
            ControllerState.RUNNING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.RUNNING,
            ControllerEvent.ALL_DRIVERS_COMPLETED,
        ): ControllerState.COMPLETING,
        # # If task run super fast, RUNNING may not be reached, and ALL_DRIVERS_COMPLETED may be received.
        # (
        #     ControllerState.PENDING,
        #     ControllerEvent.ALL_DRIVERS_COMPLETED,
        # ): ControllerState.COMPLETED,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.ALL_DRIVERS_COMPLETED,
        ): ControllerState.COMPLETING,
        (
            ControllerState.COMPLETING,
            ControllerEvent.CLEANUP_FINISHED,
        ): ControllerState.COMPLETED,
        (
            ControllerState.FAILING,
            ControllerEvent.CLEANUP_FINISHED,
        ): ControllerState.FAILED,
        (
            ControllerState.COMPLETING,
            ControllerEvent.CLEANUP_FAILED,
        ): ControllerState.FAILED,
        (
            ControllerState.FAILING,
            ControllerEvent.CLEANUP_FAILED,
        ): ControllerState.FAILED,
        # Errors during init phases
        (
            ControllerState.PENDING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
    }

    def __init__(self, controller: "ControllerActor"):  # type: ignore
        """
        Initialize ControllerStateMachine.

        Args:
            controller: Reference to the Controller instance
        """
        self._controller = controller
        self._state = ControllerState.INITIALIZING
        self._lock = asyncio.Lock()

        # Track error information
        self._error_spmds: dict[str, str] = {}  # spmd_id -> error_message

    @property
    def state(self) -> ControllerState:
        """Get current state (sync access for simple reads)."""
        return self._state

    async def trigger_event(self, event: ControllerEvent, **kwargs) -> bool:
        """
        Trigger a state transition event.

        Args:
            event: The event to trigger
            **kwargs: Additional event data
                - spmd_id: SPMD ID (for DRIVER_ERROR)
                - error: Error message (for DRIVER_ERROR)

        Returns:
            True if transition occurred, False otherwise
        """
        async with self._lock:
            current = self._state
            key = (current, event)

            if key not in self._TRANSITIONS:
                logger.warning(
                    f"Invalid state transition: {current} + {event}. Ignoring event."
                )
                return False
            next_state = self._TRANSITIONS[key]

            # Handle event-specific data
            if event in (ControllerEvent.DRIVER_ERROR, ControllerEvent.CLEANUP_FAILED):
                spmd_id = kwargs.get(
                    "spmd_id",
                    "cleanup" if event == ControllerEvent.CLEANUP_FAILED else "unknown",
                )
                error = kwargs.get("error", "Unknown error")
                self._error_spmds[spmd_id] = error
                logger.error(f"Controller error: {spmd_id} - {error}")

            # Perform state transition
            old_state = self._state
            self._state = next_state
            profiler_runtime = getattr(self._controller, "_profiler_runtime", None)
            if profiler_runtime is not None and profiler_runtime.enabled:
                profiler_runtime.recorder.controller_fsm_transition(
                    from_state=old_state.name,
                    event=event.name,
                    to_state=next_state.name,
                )
            logger.debug(
                f"State transition: {old_state} -> {next_state} (event: {event})"
            )

        # Execute entry callback (outside lock to prevent deadlock)
        await self._execute_entry_callback(next_state)

        return True

    async def _execute_entry_callback(self, state: ControllerState) -> None:
        """Execute the entry callback for a state."""
        callbacks = {
            ControllerState.PENDING: self._on_enter_pending,
            ControllerState.TOPOLOGY_REGISTERING: self._on_enter_topology_registering,
            ControllerState.RUNNING: self._on_enter_running,
            ControllerState.COMPLETING: self._on_enter_completing,
            ControllerState.FAILING: self._on_enter_failing,
            ControllerState.COMPLETED: self._on_enter_completed,
            ControllerState.FAILED: self._on_enter_failed,
        }

        callback = callbacks.get(state)
        if callback:
            try:
                profiler_runtime = getattr(self._controller, "_profiler_runtime", None)
                if profiler_runtime is not None and profiler_runtime.enabled:
                    async with profiler_runtime.recorder.controller_fsm_entry_callback(
                        state=state.name,
                    ):
                        await callback()
                else:
                    await callback()
            except Exception as e:
                logger.error(f"Error in entry callback for {state}: {e}")
                if state in (ControllerState.COMPLETING, ControllerState.FAILING) and (
                    self._state == state
                ):
                    await self.trigger_event(
                        ControllerEvent.CLEANUP_FAILED,
                        error=str(e),
                    )
                    return
                raise

    def get_state(self) -> ControllerState:
        """Get current state (alias for property)."""
        return self.state

    def is_running(self) -> bool:
        """Check if controller is in RUNNING state."""
        return self.state == ControllerState.RUNNING

    def is_completed(self) -> bool:
        """Check if controller has completed."""
        return self.state in (ControllerState.COMPLETED, ControllerState.FAILED)

    def get_error_spmds(self) -> dict[str, str]:
        """Get SPMDs currently in error state."""
        return dict(self._error_spmds)

    # =========================================================================
    # State entry callbacks (async)
    # =========================================================================

    async def _on_enter_pending(self) -> None:
        """Called when entering PENDING state.

        Resources are allocated and SPMDs are started (fire-and-forget).
        Waiting for all drivers to report workers_ready via heartbeat.
        """
        logger.debug(
            "Controller entering PENDING state, waiting for all drivers to report workers_ready"
        )

    async def _on_enter_topology_registering(self) -> None:
        """Called when entering TOPOLOGY_REGISTERING state.

        All drivers have reported workers_ready.
        Initiates topology registration on all SPMDs (fire-and-forget).
        Workers will auto-start after handler registration.
        """
        logger.debug(
            "Controller entering TOPOLOGY_REGISTERING state, initiating topology registration"
        )

        # Initiate topology registration (fire-and-forget, no await needed)
        try:
            await self._controller._initiate_topology_registration()
        except Exception as e:
            logger.error(f"Failed to initiate topology registration: {e}")

    async def _on_enter_running(self) -> None:
        """Called when entering RUNNING state.

        All workers are running after topology registration.
        Starts heartbeat timeout monitoring.
        """
        logger.debug("Controller entering RUNNING state")

        # Start heartbeat timeout monitoring now that system is fully initialized
        try:
            if getattr(self._controller, "_heartbeat_manager", None):
                await self._controller._heartbeat_manager.start_monitoring_async()
                logger.debug("Heartbeat timeout monitoring started")
            # If Drain strategy is configured, issue the first drain target to trigger execution
            has_trigger = getattr(self._controller, "_issue_initial_drain", None)
            if has_trigger:
                await self._controller._issue_initial_drain()
            if getattr(self._controller, "_start_sources_when_running", False):
                self._controller._start_sources_when_running = False
                start_sources = getattr(
                    self._controller, "start_source_operators", None
                )
                if start_sources:
                    await start_sources()
        except Exception as e:
            logger.error(
                f"Failed to start heartbeat monitoring or issue initial drain: {e}"
            )

        # Prevent discarded events: Check if operations were already completed
        # during restoration (PENDING state).  However, when a drain strategy is
        # active, the drain→checkpoint→resume pipeline will detect max_steps and
        # trigger ALL_DRIVERS_COMPLETED itself after the final checkpoint is
        # persisted.  Firing the event here would race with that pipeline —
        # cleanup would release placement groups while saves are in-flight.
        if self._controller._async_step_manager.is_all_completed():
            has_drain = self._controller._drain_strategy is not None
            if has_drain:
                logger.debug(
                    "All operators completed upon entering RUNNING, but drain "
                    "strategy is active — deferring completion to the "
                    "drain→checkpoint→resume pipeline"
                )
            else:
                logger.debug(
                    "wait_for_completion detected all operators completed "
                    "upon entering RUNNING, transitioning"
                )
                await self.trigger_event(ControllerEvent.ALL_DRIVERS_COMPLETED)

    async def _on_enter_completing(self) -> None:
        """
        Called when entering COMPLETING state.

        Reclaims all SPMD resources and performs cleanup.
        """
        logger.debug("Controller entering COMPLETING state")

        cleanup_error = None
        try:
            await self._controller._finalize_runtime(
                export_profiler_trace=True,
            )
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs: {e}")
            cleanup_error = e
        if cleanup_error is not None:
            await self.trigger_event(
                ControllerEvent.CLEANUP_FAILED,
                error=str(cleanup_error),
            )
            return
        await self.trigger_event(ControllerEvent.CLEANUP_FINISHED)

    async def _on_enter_failing(self) -> None:
        """
        Called when entering FAILING state.

        Handles unrecoverable failures.
        """
        logger.error("Controller entering FAILING state")

        errors = dict(self._error_spmds)
        logger.error(f"Failed SPMDs: {errors}")

        # Attempt to cleanup resources even on failure
        cleanup_error = None
        try:
            await self._controller._finalize_runtime(
                export_profiler_trace=False,
            )
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs during failure handling: {e}")
            cleanup_error = e
        if cleanup_error is not None:
            await self.trigger_event(
                ControllerEvent.CLEANUP_FAILED,
                error=str(cleanup_error),
            )
            return
        await self.trigger_event(ControllerEvent.CLEANUP_FINISHED)

    async def _on_enter_completed(self) -> None:
        """Called when entering COMPLETED state."""
        logger.debug("Controller entering COMPLETED state")

    async def _on_enter_failed(self) -> None:
        """Called when entering FAILED state."""
        logger.error("Controller entering FAILED state")

    async def reset(self) -> None:
        """Reset state machine to initial state."""
        async with self._lock:
            self._state = ControllerState.INITIALIZING
            self._error_spmds.clear()
        logger.debug("Controller state machine reset")
