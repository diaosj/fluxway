"""Local lifecycle transitions coordinated by ExecutionLifecycle."""

import asyncio
from typing import TYPE_CHECKING

from fluxway.common.types import ControllerEvent, ControllerState, ExecutionFailure
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.state_machine.controller_transitions import (
    ControllerEffect,
    plan_controller_transition,
)

if TYPE_CHECKING:
    from fluxway.control_plane.controller import ControllerActor

logger = get_logger(__name__)


class ControllerStateMachine:
    """
    State machine for Controller lifecycle management.

    Handles state transitions based on events from heartbeat manager
    and driver reports. Supports async callbacks for Ray Actor.
    """

    def __init__(self, controller: "ControllerActor"):  # type: ignore
        """
        Initialize ControllerStateMachine.

        Args:
            controller: Reference to the Controller instance
        """
        self._controller = controller
        self._state = ControllerState.INITIALIZING
        self._state_revision = 0
        self._lock = asyncio.Lock()
        self._terminal_event = asyncio.Event()

        self._failures: dict[str, ExecutionFailure] = {}

    @property
    def state(self) -> ControllerState:
        """Get current state (sync access for simple reads)."""
        return self._state

    @property
    def state_revision(self) -> int:
        """Monotonic version of the committed Controller state."""
        return self._state_revision

    async def trigger_event(self, event: ControllerEvent, **kwargs) -> bool:
        """
        Trigger a state transition event.

        Args:
            event: The event to trigger
            **kwargs: Additional event data
                - spmd_id: SPMD ID (for DRIVER_ERROR)
                - failure: Structured execution failure

        Returns:
            True if transition occurred, False otherwise
        """
        transition = None
        failure_to_log: ExecutionFailure | None = None
        async with self._lock:
            current = self._state
            repeated_failure = (
                current is ControllerState.FAILING
                and event is ControllerEvent.DRIVER_ERROR
            )
            if repeated_failure:
                self._record_driver_failure(kwargs)
                failure_to_log = kwargs["failure"]
            else:
                transition = plan_controller_transition(current, event)
            if not repeated_failure and transition is None:
                logger.warning(
                    "Invalid state transition: %s + %s. Ignoring event.",
                    current,
                    event,
                )
                return False
            if repeated_failure:
                next_state = current
            else:
                next_state = transition.target_state

            if not repeated_failure and event == ControllerEvent.DRIVER_ERROR:
                self._record_driver_failure(kwargs)
                failure_to_log = kwargs["failure"]
            elif not repeated_failure and event == ControllerEvent.CLEANUP_FAILED:
                self._record_reclaim_failure(kwargs)
                failure_to_log = kwargs["failure"]

            # Perform state transition
            old_state = self._state
            self._state = next_state
            self._state_revision += 1
            if next_state in (
                ControllerState.COMPLETED,
                ControllerState.FAILED,
                ControllerState.CANCELLED,
            ):
                self._terminal_event.set()
        if transition is None:
            if failure_to_log is not None:
                logger.error("Controller error: %s", failure_to_log)
            return True

        profiler_runtime = getattr(self._controller, "_profiler_runtime", None)
        if profiler_runtime is not None and profiler_runtime.enabled:
            profiler_runtime.recorder.controller_fsm_transition(
                from_state=old_state.name,
                event=event.name,
                to_state=next_state.name,
            )
        logger.debug(
            "State transition: %s -> %s (event: %s)", old_state, next_state, event
        )
        if failure_to_log is not None:
            logger.error("Controller error: %s", failure_to_log)

        if transition.effect is ControllerEffect.BEGIN_TERMINAL_CLEANUP:
            self._controller._execution_lifecycle.begin_terminal_cleanup(next_state)
        else:
            await self._controller._execution_lifecycle.enter_state(next_state)
        return True

    def get_state(self) -> ControllerState:
        """Get current state (alias for property)."""
        return self.state

    def is_running(self) -> bool:
        """Check if controller is in RUNNING state."""
        return self.state == ControllerState.RUNNING

    def is_completed(self) -> bool:
        """Check if controller has completed."""
        return self.state in (
            ControllerState.COMPLETED,
            ControllerState.FAILED,
            ControllerState.CANCELLED,
        )

    async def wait_until_terminal(
        self, timeout: float | None = None
    ) -> ControllerState:
        if not self.is_completed():
            if timeout is None:
                await self._terminal_event.wait()
            else:
                await asyncio.wait_for(self._terminal_event.wait(), timeout=timeout)
        state = self.state
        if state not in (
            ControllerState.COMPLETED,
            ControllerState.FAILED,
            ControllerState.CANCELLED,
        ):
            raise RuntimeError(f"Terminal event set in non-terminal state: {state}")
        return state

    def get_failures(self) -> dict[str, ExecutionFailure]:
        return dict(self._failures)

    def _record_driver_failure(self, event_data: dict) -> None:
        spmd_id = event_data.get("spmd_id")
        failure = event_data.get("failure")
        if not isinstance(spmd_id, str) or not spmd_id:
            raise TypeError("DRIVER_ERROR requires a non-empty spmd_id.")
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("DRIVER_ERROR requires an ExecutionFailure.")
        if spmd_id != failure.spmd_id:
            raise ValueError(
                "DRIVER_ERROR spmd_id must match failure.spmd_id: "
                f"{spmd_id!r} != {failure.spmd_id!r}"
            )
        self._failures[spmd_id] = failure

    def _record_reclaim_failure(self, event_data: dict) -> None:
        failure = event_data.get("failure")
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("CLEANUP_FAILED requires an ExecutionFailure.")
        if failure.spmd_id != "controller" or failure.phase != "CONTROLLER_CLEANUP":
            raise ValueError(
                "CLEANUP_FAILED requires controller/CONTROLLER_CLEANUP identity."
            )
        self._failures[failure.spmd_id] = failure

    async def reset(self) -> None:
        """Reset state machine to initial state."""
        async with self._lock:
            self._state = ControllerState.INITIALIZING
            self._state_revision = 0
            self._failures.clear()
            self._terminal_event.clear()
        logger.debug("Controller state machine reset")
