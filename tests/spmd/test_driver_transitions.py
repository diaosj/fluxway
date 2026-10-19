from fluxway.common.types import DriverEvent, DriverState
from fluxway.spmd.driver.state_machine.driver_transitions import (
    DriverEffect,
    plan_driver_transition,
)


def test_block_transition_has_explicit_worker_effect() -> None:
    transition = plan_driver_transition(
        DriverState.RUNNING, DriverEvent.BLOCK_REQUESTED
    )

    assert transition is not None
    assert transition.target_state is DriverState.BLOCKED
    assert transition.effect is DriverEffect.BLOCK_WORKERS


def test_invalid_driver_transition_has_no_effect() -> None:
    assert plan_driver_transition(DriverState.STOPPED, DriverEvent.START) is None
