"""
DriverStateMachine for managing SPMD Driver lifecycle.

State transitions:
- INITIALIZING -> READY (on INIT_COMPLETE)
- INITIALIZING -> ERROR (on INIT_FAILED)
- READY -> RUNNING (on START)
- RUNNING -> BLOCKED (on BLOCK_REQUESTED)
- RUNNING -> ERROR (on WORKER_ERROR)
- BLOCKED -> RUNNING (on UNBLOCK_REQUESTED)
- RUNNING -> COMPLETED (on ALL_WORKERS_COMPLETED or COMPLETE)
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from fluxway.common.types import DriverEvent, DriverState, ExecutionFailure

logger = logging.getLogger(__name__)


class DriverStateMachine:
    """
    State machine for SPMD Driver lifecycle management.

    Handles state transitions based on events from workers and controller.
    """

    # State transition table: (current_state, event) -> next_state
    _TRANSITIONS = {
        (DriverState.INITIALIZING, DriverEvent.INIT_COMPLETE): DriverState.READY,
        (DriverState.INITIALIZING, DriverEvent.INIT_FAILED): DriverState.ERROR,
        (DriverState.INITIALIZING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.INITIALIZING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.READY, DriverEvent.START): DriverState.ACTIVATING,
        (DriverState.ACTIVATING, DriverEvent.ACTIVATION_COMPLETE): DriverState.RUNNING,
        (DriverState.ACTIVATING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.ACTIVATING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.READY, DriverEvent.RESTORE_REQUESTED): DriverState.RESTORING,
        (DriverState.RESTORING, DriverEvent.RESTORE_COMPLETE): DriverState.READY,
        (DriverState.RESTORING, DriverEvent.RESTORE_FAILED): DriverState.ERROR,
        (DriverState.RESTORING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.RESTORING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.READY, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.READY, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.RUNNING, DriverEvent.BLOCK_REQUESTED): DriverState.BLOCKED,
        (DriverState.RUNNING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.RUNNING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.BLOCKED, DriverEvent.UNBLOCK_REQUESTED): DriverState.RUNNING,
        (DriverState.BLOCKED, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.BLOCKED, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.RUNNING, DriverEvent.ALL_WORKERS_COMPLETED): DriverState.COMPLETED,
        (DriverState.RUNNING, DriverEvent.COMPLETE): DriverState.COMPLETED,
        # Drain → Drained → Snapshot → Running
        (DriverState.RUNNING, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
        (DriverState.DRAINING, DriverEvent.DRAIN_COMPLETE): DriverState.DRAINED,
        (DriverState.DRAINING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.DRAINING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        (DriverState.DRAINED, DriverEvent.SNAPSHOT_REQUESTED): DriverState.SNAPSHOTTING,
        (DriverState.SNAPSHOTTING, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
        (DriverState.DRAINED, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.DRAINED, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
        # Allow resuming even if we didn't snapshot
        (DriverState.DRAINED, DriverEvent.RESUME): DriverState.RUNNING,
        (DriverState.SNAPSHOTTING, DriverEvent.RESUME): DriverState.RUNNING,
        (DriverState.SNAPSHOTTING, DriverEvent.SNAPSHOT_FAILED): DriverState.ERROR,
        (DriverState.SNAPSHOTTING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.SNAPSHOTTING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
    }

    _TRANSITIONS.update(
        {
            (state, DriverEvent.STOP): DriverState.STOPPING
            for state in DriverState
            if state not in (DriverState.STOPPING, DriverState.STOPPED)
        }
    )
    _TRANSITIONS[DriverState.STOPPING, DriverEvent.STOPPED] = DriverState.STOPPED

    def __init__(self, driver: "Any"):
        """
        Initialize DriverStateMachine.

        Args:
            driver: Reference to the SPMDDriver instance
        """
        self._driver = driver
        self._state = DriverState.INITIALIZING
        self._lock = threading.Lock()

        self._worker_failures: dict[int, ExecutionFailure] = {}
        self._failure: ExecutionFailure | None = None

        # Callbacks for state entry
        self._entry_callbacks: dict[DriverState, Callable[[], None]] = {
            DriverState.READY: self._on_enter_ready,
            DriverState.RUNNING: self._on_enter_running,
            DriverState.BLOCKED: self._on_enter_blocked,
            DriverState.DRAINING: self._on_enter_draining,
            DriverState.DRAINED: self._on_enter_drained,
            DriverState.SNAPSHOTTING: self._on_enter_snapshotting,
            DriverState.COMPLETED: self._on_enter_completed,
            DriverState.ERROR: self._on_enter_error,
        }

    @property
    def state(self) -> DriverState:
        """Get current state."""
        with self._lock:
            return self._state

    @classmethod
    def can_transition(cls, state: DriverState, event: DriverEvent) -> bool:
        return (state, event) in cls._TRANSITIONS

    def trigger(self, event: DriverEvent, **kwargs) -> bool:
        """
        Trigger a state transition event.

        Args:
            event: The event to trigger
            **kwargs: Additional event data
                - failure: Structured execution failure

        Returns:
            True if transition occurred, False otherwise
        """
        with self._lock:
            current = self._state
            key = (current, event)

            # Guard: BLOCK/UNBLOCK must not interrupt drain or snapshot
            if event in (
                DriverEvent.BLOCK_REQUESTED,
                DriverEvent.UNBLOCK_REQUESTED,
            ) and current in (
                DriverState.DRAINING,
                DriverState.DRAINED,
                DriverState.SNAPSHOTTING,
            ):
                logger.warning(
                    f"Ignoring {event} in {current} state spmd_id={self._driver.spmd_id}"
                )
                return False

            if not self.can_transition(current, event):
                logger.debug(f"No transition for: {current} + {event}. Ignoring event.")
                return False

            next_state = self._TRANSITIONS[key]

            if next_state is DriverState.ERROR:
                failure = self._failure_for_event(event, kwargs)
                if event is DriverEvent.WORKER_ERROR:
                    worker_rank = failure.rank
                    if worker_rank is None:
                        raise ValueError("WORKER_ERROR failure requires a worker rank.")
                else:
                    worker_rank = None
                if self._failure is None:
                    self._failure = failure
                if worker_rank is not None:
                    self._worker_failures[worker_rank] = failure
                logger.error(
                    "Driver failure in SPMD %s: %s",
                    self._driver.spmd_id,
                    failure,
                )

            # Perform state transition
            old_state = self._state
            self._state = next_state
            logger.debug(
                f"SPMD {self._driver.spmd_id} state transition: "
                f"{old_state} -> {next_state} (event: {event})"
            )

        # Execute entry callback (outside lock to prevent deadlock)
        entry_callback = self._entry_callbacks.get(next_state)
        if entry_callback:
            entry_callback()

        return True

    def require_transition(self, event: DriverEvent, **kwargs) -> DriverState:
        if not self.trigger(event, **kwargs):
            raise RuntimeError(
                f"Driver event {event.name} is invalid in {self.state.name} state "
                f"spmd_id={self._driver.spmd_id}"
            )
        return self.state

    def admit_drain(self) -> bool:
        """Admit a periodic or terminal drain through one lifecycle path.

        Returns whether this call entered DRAINING. Existing DRAINING/DRAINED
        states are already owned by the same drain cycle and are handled by the
        synchronizer target reducer.
        """
        state = self.state
        if state in (DriverState.RUNNING, DriverState.SNAPSHOTTING):
            self.require_transition(DriverEvent.DRAIN_REQUESTED)
            return True
        if state in (DriverState.DRAINING, DriverState.DRAINED):
            return False
        raise RuntimeError(
            "drain requires RUNNING, DRAINING, or DRAINED Driver state, "
            f"got {state} spmd_id={self._driver.spmd_id}"
        )

    def _failure_for_event(
        self,
        event: DriverEvent,
        event_data: dict[str, Any],
    ) -> ExecutionFailure:
        failure = event_data.get("failure")
        if failure is not None:
            if not isinstance(failure, ExecutionFailure):
                raise TypeError(
                    f"{event.name} failure must be ExecutionFailure, "
                    f"got {type(failure).__name__}."
                )
            if failure.spmd_id != self._driver.spmd_id:
                raise ValueError(
                    f"{event.name} failure SPMD {failure.spmd_id!r} does not match "
                    f"driver {self._driver.spmd_id!r}."
                )
            return failure
        if event is DriverEvent.WORKER_ERROR:
            raise TypeError("WORKER_ERROR requires failure=ExecutionFailure(...).")
        if event is DriverEvent.COMPONENT_ERROR:
            raise TypeError("COMPONENT_ERROR requires failure=ExecutionFailure(...).")
        if event is DriverEvent.INIT_FAILED:
            return ExecutionFailure(
                spmd_id=self._driver.spmd_id,
                rank=None,
                op_name=None,
                step=None,
                phase="DRIVER_INITIALIZATION",
                exception_type="DriverInitializationFailed",
                message="SPMD driver initialization failed.",
                traceback=None,
            )
        if event is DriverEvent.SNAPSHOT_FAILED:
            return ExecutionFailure(
                spmd_id=self._driver.spmd_id,
                rank=None,
                op_name=None,
                step=None,
                phase="SNAPSHOT",
                exception_type="SnapshotFailed",
                message="SPMD snapshot failed.",
                traceback=None,
            )
        if event is DriverEvent.RESTORE_FAILED:
            return ExecutionFailure(
                spmd_id=self._driver.spmd_id,
                rank=None,
                op_name=None,
                step=None,
                phase="RESTORE",
                exception_type="RestoreFailed",
                message="SPMD restore failed.",
                traceback=None,
            )
        raise ValueError(f"Driver event {event.name} cannot enter ERROR.")

    def get_state(self) -> DriverState:
        """Get current state (alias for property)."""
        return self.state

    def is_running(self) -> bool:
        """Check if driver is in RUNNING state."""
        return self.state == DriverState.RUNNING

    def is_ready(self) -> bool:
        """Check if driver is in READY state."""
        return self.state == DriverState.READY

    def is_error(self) -> bool:
        """Check if driver is in ERROR state."""
        return self.state == DriverState.ERROR

    def is_completed(self) -> bool:
        """Check if driver has completed."""
        return self.state == DriverState.COMPLETED

    @property
    def failure(self) -> ExecutionFailure | None:
        with self._lock:
            return self._failure

    def get_worker_failures(self) -> dict[int, ExecutionFailure]:
        with self._lock:
            return dict(self._worker_failures)

    # =========================================================================
    # State entry callbacks
    # =========================================================================

    def _on_enter_ready(self) -> None:
        """Called when entering READY state.

        Note: Heartbeat is already started in initialize() (before READY),
        so we do NOT call _start_heartbeat() here.
        """
        logger.debug(f"SPMD {self._driver.spmd_id} is ready")

    def _on_enter_running(self) -> None:
        """Called when entering RUNNING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is running")

    def _on_enter_blocked(self) -> None:
        """Called when entering BLOCKED state."""
        logger.warning(f"SPMD {self._driver.spmd_id} is blocked")
        # Block all workers
        self._driver._block_all_workers()

    def _on_enter_draining(self) -> None:
        """Called when entering DRAINING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is draining to target step")

    def _on_enter_drained(self) -> None:
        """Called when entering DRAINED state."""
        logger.debug(f"SPMD {self._driver.spmd_id} has reached its drain target")

    def _on_enter_snapshotting(self) -> None:
        """Called when entering SNAPSHOTTING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is snapshotting operator states")

    def _on_enter_completed(self) -> None:
        """Called when entering COMPLETED state."""
        logger.debug(f"SPMD {self._driver.spmd_id} completed")
        # Report completion to controller
        self._driver._report_completion()

    def _on_enter_error(self) -> None:
        """Called when entering ERROR state."""
        logger.error(f"SPMD {self._driver.spmd_id} entered ERROR state")
        self._driver._schedule_driver_component_state(DriverState.ERROR)
        failure = self.failure
        if failure is None:
            raise RuntimeError("ERROR state requires an ExecutionFailure.")
        self._driver._report_error(failure)

    def reset(self) -> None:
        """Reset state machine to initial state."""
        with self._lock:
            self._state = DriverState.INITIALIZING
            self._worker_failures.clear()
            self._failure = None
        logger.debug(f"SPMD {self._driver.spmd_id} state machine reset")
