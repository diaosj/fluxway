from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.types import ControllerEvent, ControllerState
from fluxway.control_plane.state_machine.controller_transitions import (
    ControllerEffect,
    plan_controller_transition,
)
from fluxway.control_plane.state_machine.controller_fsm import ControllerStateMachine


def test_resume_transition_does_not_encode_runtime_side_effects() -> None:
    transition = plan_controller_transition(
        ControllerState.RUNNING, ControllerEvent.RESUME
    )

    assert transition is not None
    assert transition.target_state is ControllerState.RESUMING
    assert transition.effect is ControllerEffect.ENTER_STATE


def test_terminal_transition_has_cleanup_effect() -> None:
    transition = plan_controller_transition(
        ControllerState.RUNNING, ControllerEvent.ALL_DRIVERS_COMPLETED
    )

    assert transition is not None
    assert transition.target_state is ControllerState.COMPLETING
    assert transition.effect is ControllerEffect.BEGIN_TERMINAL_CLEANUP


def test_terminal_states_are_absorbing_except_cleanup_completion() -> None:
    for state in (
        ControllerState.COMPLETED,
        ControllerState.FAILED,
        ControllerState.CANCELLED,
    ):
        for event in ControllerEvent:
            assert plan_controller_transition(state, event) is None


@pytest.mark.asyncio
async def test_transition_lock_is_not_held_during_enter_effect() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Lifecycle:
        async def enter_state(self, state: ControllerState) -> None:
            entered.set()
            await release.wait()

        def begin_terminal_cleanup(self, state: ControllerState) -> None:
            release.set()

    controller = SimpleNamespace(
        _profiler_runtime=SimpleNamespace(enabled=False),
        _execution_lifecycle=Lifecycle(),
    )
    machine = ControllerStateMachine(controller)
    machine._state = ControllerState.READY
    first = asyncio.create_task(machine.trigger_event(ControllerEvent.START))
    await asyncio.wait_for(entered.wait(), timeout=1)

    await asyncio.wait_for(machine.trigger_event(ControllerEvent.CANCEL), timeout=1)
    release.set()
    assert await first is True


@pytest.mark.asyncio
async def test_state_revision_increases_only_for_committed_transitions() -> None:
    controller = SimpleNamespace(
        _profiler_runtime=SimpleNamespace(enabled=False),
        _execution_lifecycle=SimpleNamespace(
            enter_state=lambda state: asyncio.sleep(0)
        ),
    )
    machine = ControllerStateMachine(controller)
    assert machine.state_revision == 0
    assert await machine.trigger_event(ControllerEvent.INIT_COMPLETE)
    assert machine.state_revision == 1
    assert not await machine.trigger_event(ControllerEvent.INIT_COMPLETE)
    assert machine.state_revision == 1
