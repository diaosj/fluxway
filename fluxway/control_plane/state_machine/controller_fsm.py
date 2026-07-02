"""
ControllerStateMachine for managing Controller lifecycle.

State transitions (event-driven initialization):
- INITIALIZING -> PENDING (on INIT_COMPLETE: resources allocated, SPMDs started)
- PENDING -> TOPOLOGY_REGISTERING (on ALL_DRIVERS_READY: all drivers report workers_ready)
    -> triggers topology registration
- TOPOLOGY_REGISTERING -> RUNNING (on ALL_SPMDS_TOPOLOGY_READY: all workers running)
    -> starts heartbeat timeout monitoring
- RUNNING -> BLOCKED (on DRIVER_ERROR, blocks all SPMDs, triggers restart)
- BLOCKED -> RUNNING (on DRIVER_RECOVERED)
- BLOCKED -> FAILING -> FAILED
- RUNNING -> COMPLETING -> COMPLETED

Supports async callbacks for Ray Actor Controller.
"""

import asyncio
import re
from typing import TYPE_CHECKING

from fluxway.common.types import ControllerState, ControllerEvent
from fluxway.common.types.configs import ErrorPolicy
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
        ): ControllerState.BLOCKED,
        (
            ControllerState.BLOCKED,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
        (
            ControllerState.BLOCKED,
            ControllerEvent.DRIVER_RECOVERED,
        ): ControllerState.RUNNING,
        (
            ControllerState.BLOCKED,
            ControllerEvent.RECOVERY_FAILED,
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
        # Recovery from BLOCKED
        (
            ControllerState.BLOCKED,
            ControllerEvent.ALL_DRIVERS_READY,
        ): ControllerState.RUNNING,
        # Errors during init phases
        (
            ControllerState.PENDING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
    }

    def __init__(
        self,
        controller: "ControllerActor",
        blocked_timeout: float = 300.0,
        error_policy: ErrorPolicy | None = None,
    ):  # type: ignore
        """
        Initialize ControllerStateMachine.

        Args:
            controller: Reference to the Controller instance
            blocked_timeout: Max seconds to stay in BLOCKED before auto-failing
            error_policy: Error classification policy for fail-fast decisions
        """
        self._controller = controller
        self._state = ControllerState.INITIALIZING
        self._lock = asyncio.Lock()

        # Track error information
        self._error_spmds: dict[str, str] = {}  # spmd_id -> error_message
        self._restarting_spmds: set = set()

        self._blocked_timeout_secs = blocked_timeout
        self._blocked_timeout_task: asyncio.Task | None = None
        self._error_policy = error_policy or ErrorPolicy()

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
                - spmd_id: SPMD ID (for DRIVER_ERROR, DRIVER_RECOVERED)
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
            if event == ControllerEvent.DRIVER_ERROR:
                spmd_id = kwargs.get("spmd_id", "unknown")
                error = kwargs.get("error", "Unknown error")
                self._error_spmds[spmd_id] = error
                logger.error(f"Driver error: {spmd_id} - {error}")

            elif event == ControllerEvent.DRIVER_RECOVERED:
                spmd_id = kwargs.get("spmd_id")
                if spmd_id and spmd_id in self._error_spmds:
                    del self._error_spmds[spmd_id]
                if spmd_id and spmd_id in self._restarting_spmds:
                    self._restarting_spmds.remove(spmd_id)
                logger.debug(f"Driver recovered: {spmd_id}")

                if current == ControllerState.BLOCKED and self._error_spmds:
                    # Do not transition if there are still errors
                    return True

            # Perform state transition
            old_state = self._state
            self._state = next_state
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
            ControllerState.BLOCKED: self._on_enter_blocked,
            ControllerState.COMPLETING: self._on_enter_completing,
            ControllerState.FAILING: self._on_enter_failing,
            ControllerState.COMPLETED: self._on_enter_completed,
            ControllerState.FAILED: self._on_enter_failed,
        }

        callback = callbacks.get(state)
        if callback:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in entry callback for {state}: {e}")

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
        self._cancel_blocked_timeout()
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

    async def _on_enter_blocked(self) -> None:
        """
        Called when entering BLOCKED state.

        Blocks all SPMDs, then either initiates recovery or triggers
        RECOVERY_FAILED based on the fail_fast config of error SPMDs.
        """
        logger.warning("Controller entering BLOCKED state")

        # Block all SPMDs
        try:
            await self._controller.block_all_spmds()
        except Exception as e:
            logger.error(f"Failed to block all SPMDs: {e}")

        # Check if any failed node should fail-fast
        for spmd_id, error_msg in self._error_spmds.items():
            if self._should_fail_fast(spmd_id, error_msg):
                logger.error(
                    f"SPMD {spmd_id} error is non-recoverable "
                    f"(fail_fast or no matching recoverable_pattern), "
                    f"skipping recovery"
                )
                await self.trigger_event(ControllerEvent.RECOVERY_FAILED)
                return

        # Initiate recovery for error SPMDs
        error_spmds = list(self._error_spmds.keys())
        recovery_failed = False

        for spmd_id in error_spmds:
            if spmd_id in self._restarting_spmds:
                continue
            self._restarting_spmds.add(spmd_id)
            try:
                success = await self._controller.restart_spmd(spmd_id)
                if not success:
                    raise RuntimeError(f"restart_spmd returned failure for {spmd_id}")
            except Exception as e:
                logger.error(f"Failed to restart SPMD {spmd_id}: {e}")
                self._restarting_spmds.discard(spmd_id)
                recovery_failed = True

        if recovery_failed:
            await self.trigger_event(ControllerEvent.RECOVERY_FAILED)
            return

        self._start_blocked_timeout()

    async def _on_enter_completing(self) -> None:
        """
        Called when entering COMPLETING state.

        Reclaims all SPMD resources and performs cleanup.
        """
        logger.debug("Controller entering COMPLETING state")

        try:
            if hasattr(self._controller, "stop_checkpointer"):
                await self._controller.stop_checkpointer()
            await self._controller._cleanup_all_spmds()
            release_fn = getattr(
                self._controller, "_release_stopped_spmd_handles", None
            )
            if release_fn is not None:
                release_fn()
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs: {e}")
        await self.trigger_event(ControllerEvent.CLEANUP_FINISHED)

    async def _on_enter_failing(self) -> None:
        """
        Called when entering FAILING state.

        Handles unrecoverable failures.
        """
        self._cancel_blocked_timeout()
        logger.error("Controller entering FAILING state")

        errors = dict(self._error_spmds)
        logger.error(f"Failed SPMDs: {errors}")

        # Attempt to cleanup resources even on failure
        try:
            await self._controller._cleanup_all_spmds()
            release_fn = getattr(
                self._controller, "_release_stopped_spmd_handles", None
            )
            if release_fn is not None:
                release_fn()
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs during failure handling: {e}")
        await self.trigger_event(ControllerEvent.CLEANUP_FINISHED)

    async def _on_enter_completed(self) -> None:
        """Called when entering COMPLETED state."""
        logger.debug("Controller entering COMPLETED state")

    async def _on_enter_failed(self) -> None:
        """Called when entering FAILED state."""
        logger.error("Controller entering FAILED state")

    def _should_fail_fast(self, spmd_id: str, error_message: str) -> bool:
        for pattern in self._error_policy.recoverable_patterns:
            if re.search(pattern, error_message):
                return False
        node = self._controller._graph.nodes.get(spmd_id)
        if node:
            return node.spmd_config.fail_fast
        return True

    def _start_blocked_timeout(self) -> None:
        self._cancel_blocked_timeout()
        self._blocked_timeout_task = asyncio.get_event_loop().create_task(
            self._blocked_timeout_watchdog()
        )

    def _cancel_blocked_timeout(self) -> None:
        if self._blocked_timeout_task and not self._blocked_timeout_task.done():
            self._blocked_timeout_task.cancel()
        self._blocked_timeout_task = None

    async def _blocked_timeout_watchdog(self) -> None:
        try:
            await asyncio.sleep(self._blocked_timeout_secs)
            if self._state == ControllerState.BLOCKED:
                logger.error(
                    f"BLOCKED state timed out after {self._blocked_timeout_secs}s, "
                    f"triggering RECOVERY_FAILED"
                )
                await self.trigger_event(ControllerEvent.RECOVERY_FAILED)
        except asyncio.CancelledError:
            pass

    async def reset(self) -> None:
        """Reset state machine to initial state."""
        self._cancel_blocked_timeout()
        async with self._lock:
            self._state = ControllerState.INITIALIZING
            self._error_spmds.clear()
            self._restarting_spmds.clear()
        logger.debug("Controller state machine reset")
