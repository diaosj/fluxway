"""Pure Driver lifecycle transition relation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fluxway.common.types import DriverEvent, DriverState


class DriverEffect(StrEnum):
    NONE = "none"
    BLOCK_WORKERS = "block_workers"
    REPORT_COMPLETION = "report_completion"
    REPORT_ERROR = "report_error"


@dataclass(frozen=True, slots=True)
class DriverTransitionPlan:
    source_state: DriverState
    event: DriverEvent
    target_state: DriverState
    effect: DriverEffect


_TRANSITIONS: dict[tuple[DriverState, DriverEvent], DriverState] = {
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
    (DriverState.RUNNING, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
    (DriverState.DRAINING, DriverEvent.DRAIN_COMPLETE): DriverState.DRAINED,
    (DriverState.DRAINING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
    (DriverState.DRAINING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
    (DriverState.DRAINED, DriverEvent.SNAPSHOT_REQUESTED): DriverState.SNAPSHOTTING,
    (DriverState.SNAPSHOTTING, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
    (DriverState.DRAINED, DriverEvent.WORKER_ERROR): DriverState.ERROR,
    (DriverState.DRAINED, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
    (DriverState.DRAINED, DriverEvent.RESUME): DriverState.RUNNING,
    (DriverState.SNAPSHOTTING, DriverEvent.RESUME): DriverState.RUNNING,
    (DriverState.SNAPSHOTTING, DriverEvent.SNAPSHOT_FAILED): DriverState.ERROR,
    (DriverState.SNAPSHOTTING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
    (DriverState.SNAPSHOTTING, DriverEvent.COMPONENT_ERROR): DriverState.ERROR,
}

for _state in DriverState:
    if _state not in (DriverState.STOPPING, DriverState.STOPPED):
        _TRANSITIONS[(_state, DriverEvent.STOP)] = DriverState.STOPPING
_TRANSITIONS[(DriverState.STOPPING, DriverEvent.STOPPED)] = DriverState.STOPPED


def plan_driver_transition(
    state: DriverState, event: DriverEvent
) -> DriverTransitionPlan | None:
    next_state = _TRANSITIONS.get((state, event))
    if next_state is None:
        return None
    effect = {
        DriverState.BLOCKED: DriverEffect.BLOCK_WORKERS,
        DriverState.COMPLETED: DriverEffect.REPORT_COMPLETION,
        DriverState.ERROR: DriverEffect.REPORT_ERROR,
    }.get(next_state, DriverEffect.NONE)
    return DriverTransitionPlan(state, event, next_state, effect)
