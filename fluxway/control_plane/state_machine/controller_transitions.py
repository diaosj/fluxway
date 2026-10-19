"""Pure Controller lifecycle transition relation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fluxway.common.types import ControllerEvent, ControllerState


class ControllerEffect(StrEnum):
    ENTER_STATE = "enter_state"
    BEGIN_TERMINAL_CLEANUP = "begin_terminal_cleanup"


@dataclass(frozen=True, slots=True)
class ControllerTransitionPlan:
    source_state: ControllerState
    event: ControllerEvent
    target_state: ControllerState
    effect: ControllerEffect


_TRANSITIONS: dict[tuple[ControllerState, ControllerEvent], ControllerState] = {
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
    (ControllerState.READY, ControllerEvent.START): ControllerState.ACTIVATING,
    (
        ControllerState.ACTIVATING,
        ControllerEvent.ACTIVATION_COMPLETE,
    ): ControllerState.RUNNING,
    (ControllerState.RUNNING, ControllerEvent.RESUME): ControllerState.RESUMING,
    (
        ControllerState.RESUMING,
        ControllerEvent.ACTIVATION_COMPLETE,
    ): ControllerState.RUNNING,
    (ControllerState.ACTIVATING, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
    (ControllerState.RESUMING, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
    (ControllerState.COMPLETING, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
    (ControllerState.RUNNING, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
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
    (ControllerState.FAILING, ControllerEvent.CLEANUP_FINISHED): ControllerState.FAILED,
    (
        ControllerState.CANCELLING,
        ControllerEvent.CLEANUP_FINISHED,
    ): ControllerState.CANCELLED,
    (
        ControllerState.COMPLETING,
        ControllerEvent.CLEANUP_FAILED,
    ): ControllerState.FAILED,
    (ControllerState.FAILING, ControllerEvent.CLEANUP_FAILED): ControllerState.FAILED,
    (
        ControllerState.CANCELLING,
        ControllerEvent.CLEANUP_FAILED,
    ): ControllerState.FAILED,
    (ControllerState.PENDING, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
    (
        ControllerState.TOPOLOGY_REGISTERING,
        ControllerEvent.DRIVER_ERROR,
    ): ControllerState.FAILING,
    (ControllerState.READY, ControllerEvent.DRIVER_ERROR): ControllerState.FAILING,
}

for _state in (
    ControllerState.INITIALIZING,
    ControllerState.PENDING,
    ControllerState.TOPOLOGY_REGISTERING,
    ControllerState.READY,
    ControllerState.ACTIVATING,
    ControllerState.RESUMING,
    ControllerState.RUNNING,
):
    _TRANSITIONS[(_state, ControllerEvent.CANCEL)] = ControllerState.CANCELLING


def plan_controller_transition(
    state: ControllerState, event: ControllerEvent
) -> ControllerTransitionPlan | None:
    next_state = _TRANSITIONS.get((state, event))
    if next_state is None:
        return None
    effect = (
        ControllerEffect.BEGIN_TERMINAL_CLEANUP
        if next_state
        in (
            ControllerState.COMPLETING,
            ControllerState.FAILING,
            ControllerState.CANCELLING,
        )
        else ControllerEffect.ENTER_STATE
    )
    return ControllerTransitionPlan(state, event, next_state, effect)
