from __future__ import annotations

import asyncio

import pytest

from fluxway.common.types import ControllerState
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.source_progress import (
    SourceProgressState,
    SourceProgressSupervisor,
)


@pytest.mark.asyncio
async def test_controller_running_resume_reuses_one_generation() -> None:
    controller_type = ControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_type)
    notifications: list[ControllerState] = []

    async def progress() -> None:
        await asyncio.sleep(0)

    supervisor = SourceProgressSupervisor(progress)
    controller._source_progress_supervisor = supervisor

    class Components:
        async def notify_state(self, state: ControllerState) -> None:
            notifications.append(state)

    controller._control_plane_components = Components()
    await controller._notify_control_plane_components(ControllerState.RUNNING)
    await controller._notify_control_plane_components(ControllerState.RUNNING)

    assert supervisor.generation == 1
    assert supervisor.state is SourceProgressState.RUNNING
    assert notifications == [ControllerState.RUNNING, ControllerState.RUNNING]

    await controller._notify_control_plane_components(ControllerState.FAILING)
    assert supervisor.state is SourceProgressState.STOPPED


@pytest.mark.asyncio
async def test_start_is_idempotent_for_live_generation() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def tick() -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    supervisor = SourceProgressSupervisor(tick)
    await supervisor.start()
    await supervisor.start()
    assert supervisor.state is SourceProgressState.RUNNING
    assert supervisor.generation == 1

    supervisor.wake()
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert calls == 1
    release.set()
    await supervisor.stop()
    assert supervisor.state is SourceProgressState.STOPPED


@pytest.mark.asyncio
async def test_failed_generation_is_observable_and_cannot_restart() -> None:
    failure = ValueError("planner failed")

    async def tick() -> None:
        raise failure

    supervisor = SourceProgressSupervisor(tick)
    await supervisor.start()
    await asyncio.sleep(0.01)

    with pytest.raises(RuntimeError, match="source progress supervisor failed"):
        await supervisor.raise_if_failed()
    assert supervisor.state is SourceProgressState.FAILED

    with pytest.raises(RuntimeError, match="source progress supervisor failed"):
        await supervisor.start()


@pytest.mark.asyncio
async def test_stop_is_idempotent_after_successful_generation() -> None:
    supervisor = SourceProgressSupervisor(lambda: asyncio.sleep(0))
    await supervisor.start()
    await supervisor.stop()
    await supervisor.stop()
    assert supervisor.state is SourceProgressState.STOPPED
