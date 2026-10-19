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
from typing import Any

from fluxway.common.types import DriverEvent, DriverState, ExecutionFailure
from fluxway.spmd.driver.state_machine.driver_transitions import (
    DriverEffect,
    plan_driver_transition,
)

logger = logging.getLogger(__name__)


class DriverStateMachine:
    """
    State machine for SPMD Driver lifecycle management.

    Applies pure lifecycle transitions and executes their local effects.
    """

    def __init__(self, driver: "Any"):
        """
        Initialize DriverStateMachine.

        Args:
            driver: Reference to the SPMDDriver instance
        """
        self._driver = driver
        self._state = DriverState.INITIALIZING
        self._state_revision = 0
        self._lock = threading.Lock()

        self._worker_failures: dict[int, ExecutionFailure] = {}
        self._failure: ExecutionFailure | None = None

    @property
    def state(self) -> DriverState:
        """Get current state."""
        with self._lock:
            return self._state

    @property
    def state_revision(self) -> int:
        with self._lock:
            return self._state_revision

    @classmethod
    def can_transition(cls, state: DriverState, event: DriverEvent) -> bool:
        return plan_driver_transition(state, event) is not None

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
        invalid_reason: str | None = None
        failure_to_report: ExecutionFailure | None = None
        with self._lock:
            current = self._state
            if event in (
                DriverEvent.BLOCK_REQUESTED,
                DriverEvent.UNBLOCK_REQUESTED,
            ) and current in (
                DriverState.DRAINING,
                DriverState.DRAINED,
                DriverState.SNAPSHOTTING,
            ):
                invalid_reason = (
                    f"Ignoring {event} in {current} state "
                    f"spmd_id={self._driver.spmd_id}"
                )
                transition = None
            else:
                transition = plan_driver_transition(current, event)
                if transition is None:
                    invalid_reason = f"No transition for: {current} + {event}"

            if invalid_reason is not None:
                pass
            else:
                next_state = transition.target_state
                if next_state is DriverState.ERROR:
                    failure = self._failure_for_event(event, kwargs)
                    if event is DriverEvent.WORKER_ERROR:
                        worker_rank = failure.rank
                        if worker_rank is None:
                            raise ValueError(
                                "WORKER_ERROR failure requires a worker rank."
                            )
                    else:
                        worker_rank = None
                    if self._failure is None:
                        self._failure = failure
                    if worker_rank is not None:
                        self._worker_failures[worker_rank] = failure
                    failure_to_report = failure
                old_state = self._state
                self._state = next_state
                self._state_revision += 1
                state_revision = self._state_revision

        if invalid_reason is not None:
            logger.debug("%s", invalid_reason)
            return False
        logger.debug(
            "SPMD %s state transition: %s -> %s (event: %s)",
            self._driver.spmd_id,
            old_state,
            next_state,
            event,
        )
        if failure_to_report is not None:
            logger.error(
                "Driver failure in SPMD %s: %s",
                self._driver.spmd_id,
                failure_to_report,
            )

        self._apply_effect(transition.effect, state_revision=state_revision)

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

    def _apply_effect(self, effect: DriverEffect, *, state_revision: int) -> None:
        if effect is DriverEffect.NONE:
            return
        if effect is DriverEffect.BLOCK_WORKERS:
            submit = getattr(self._driver, "_submit_driver_effect", None)
            if not callable(submit):
                raise RuntimeError("Driver effect sink is not configured")
            submit(
                self._driver._block_all_workers,
                effect_name=effect.value,
                state_revision=state_revision,
            )
            return
        if effect is DriverEffect.REPORT_COMPLETION:
            submit = getattr(self._driver, "_submit_driver_effect", None)
            if not callable(submit):
                raise RuntimeError("Driver effect sink is not configured")
            submit(
                self._driver._report_completion,
                effect_name=effect.value,
                state_revision=state_revision,
            )
            return
        if effect is DriverEffect.REPORT_ERROR:
            failure = self.failure
            if failure is None:
                raise RuntimeError("ERROR state requires an ExecutionFailure.")
            self._driver._schedule_driver_component_state(DriverState.ERROR)
            submit = getattr(self._driver, "_submit_driver_effect", None)
            if not callable(submit):
                raise RuntimeError("Driver effect sink is not configured")
            submit(
                self._driver._report_error,
                args=(failure,),
                effect_name=effect.value,
                state_revision=state_revision,
            )
            return
        raise RuntimeError(f"unsupported Driver effect: {effect}")

    def reset(self) -> None:
        """Reset state machine to initial state."""
        with self._lock:
            self._state = DriverState.INITIALIZING
            self._state_revision = 0
            self._worker_failures.clear()
            self._failure = None
        logger.debug(f"SPMD {self._driver.spmd_id} state machine reset")
