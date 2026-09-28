"""Local lifecycle transitions coordinated by ExecutionLifecycle."""

import asyncio
from typing import TYPE_CHECKING, ClassVar

from fluxway.common.types import ControllerEvent, ControllerState, ExecutionFailure
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
    _TRANSITIONS: ClassVar[
        dict[tuple[ControllerState, ControllerEvent], ControllerState]
    ] = {
        # --- Event-driven initialization flow ---
        (
            ControllerState.INITIALIZING,
            ControllerEvent.INIT_COMPLETE,
        ): ControllerState.PENDING,
        (
            ControllerState.INITIALIZING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.PENDING,
            ControllerEvent.ALL_DRIVERS_READY,
        ): ControllerState.TOPOLOGY_REGISTERING,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.ALL_SPMDS_TOPOLOGY_READY,
        ): ControllerState.READY,
        (
            ControllerState.READY,
            ControllerEvent.START,
        ): ControllerState.ACTIVATING,
        (
            ControllerState.ACTIVATING,
            ControllerEvent.ACTIVATION_COMPLETE,
        ): ControllerState.RUNNING,
        (ControllerState.RUNNING, ControllerEvent.RESUME): ControllerState.RESUMING,
        (
            ControllerState.RESUMING,
            ControllerEvent.ACTIVATION_COMPLETE,
        ): ControllerState.RUNNING,
        (
            ControllerState.ACTIVATING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.RESUMING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.COMPLETING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        # --- Normal operation ---
        (
            ControllerState.RUNNING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
        (
            ControllerState.RUNNING,
            ControllerEvent.ALL_DRIVERS_COMPLETED,
        ): ControllerState.COMPLETING,
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
        (
            ControllerState.READY,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.FAILING,
    }

    _TRANSITIONS.update(
        {
            (state, ControllerEvent.CANCEL): ControllerState.CANCELLING
            for state in (
                ControllerState.INITIALIZING,
                ControllerState.PENDING,
                ControllerState.TOPOLOGY_REGISTERING,
                ControllerState.READY,
                ControllerState.ACTIVATING,
                ControllerState.RESUMING,
                ControllerState.RUNNING,
            )
        }
    )
    _TRANSITIONS.update(
        {
            (
                ControllerState.CANCELLING,
                ControllerEvent.CLEANUP_FINISHED,
            ): ControllerState.CANCELLED,
            (
                ControllerState.CANCELLING,
                ControllerEvent.CLEANUP_FAILED,
            ): ControllerState.FAILED,
            (
                ControllerState.CANCELLING,
                ControllerEvent.DRIVER_ERROR,
            ): ControllerState.FAILING,
        }
    )

    def __init__(self, controller: "ControllerActor"):  # type: ignore
        """
        Initialize ControllerStateMachine.

        Args:
            controller: Reference to the Controller instance
        """
        self._controller = controller
        self._state = ControllerState.INITIALIZING
        self._lock = asyncio.Lock()
        self._terminal_event = asyncio.Event()

        self._failures: dict[str, ExecutionFailure] = {}

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
                - failure: Structured execution failure

        Returns:
            True if transition occurred, False otherwise
        """
        async with self._lock:
            current = self._state
            if (
                current is ControllerState.FAILING
                and event is ControllerEvent.DRIVER_ERROR
            ):
                self._record_driver_failure(kwargs)
                return True
            key = (current, event)

            if key not in self._TRANSITIONS:
                logger.warning(
                    f"Invalid state transition: {current} + {event}. Ignoring event."
                )
                return False
            next_state = self._TRANSITIONS[key]

            if event == ControllerEvent.DRIVER_ERROR:
                self._record_driver_failure(kwargs)
            elif event == ControllerEvent.CLEANUP_FAILED:
                self._record_reclaim_failure(kwargs)

            # Perform state transition
            old_state = self._state
            self._state = next_state
            if next_state in (
                ControllerState.COMPLETED,
                ControllerState.FAILED,
                ControllerState.CANCELLED,
            ):
                self._terminal_event.set()
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
        logger.error(f"Controller error: {failure}")

    def _record_reclaim_failure(self, event_data: dict) -> None:
        failure = event_data.get("failure")
        if not isinstance(failure, ExecutionFailure):
            raise TypeError("CLEANUP_FAILED requires an ExecutionFailure.")
        if failure.spmd_id != "controller" or failure.phase != "CONTROLLER_CLEANUP":
            raise ValueError(
                "CLEANUP_FAILED requires controller/CONTROLLER_CLEANUP identity."
            )
        self._failures[failure.spmd_id] = failure
        logger.error(f"Controller reclaim error: {failure}")

    async def reset(self) -> None:
        """Reset state machine to initial state."""
        async with self._lock:
            self._state = ControllerState.INITIALIZING
            self._failures.clear()
            self._terminal_event.clear()
        logger.debug("Controller state machine reset")
