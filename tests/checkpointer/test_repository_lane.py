import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from fluxway.common.types import CheckpointAssembly, SaveCheckpointRequest


@pytest.mark.asyncio
async def test_repository_fencing_progresses_with_default_executor_occupied(
    checkpointer_factory, tmp_path, monkeypatch
):
    owner = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    owner.set_default_executor(pool)
    entered = owner.create_future()
    release = Event()

    def occupy():
        owner.call_soon_threadsafe(entered.set_result, None)
        release.wait()

    blocked = asyncio.create_task(asyncio.to_thread(occupy))
    checkpointer = await checkpointer_factory(
        object(), str(tmp_path / "repo"), interval_steps=1
    )
    original = checkpointer._repository.incomplete_transactions
    observed = []

    def inspect():
        observed.append((get_ident(), asyncio.get_running_loop()))
        return original()

    monkeypatch.setattr(checkpointer._repository, "incomplete_transactions", inspect)
    try:
        await entered
        await asyncio.wait_for(checkpointer._fence_incomplete_transactions(), 2)
        assert observed[0][0] != get_ident()
        assert observed[0][1] is not owner
    finally:
        release.set()
        await blocked


@pytest.mark.asyncio
async def test_checkpoint_state_transitions_wake_owner_loop_waiter(
    checkpointer_factory, tmp_path
):
    owner = asyncio.get_running_loop()
    owner_thread = get_ident()
    released = asyncio.Event()
    request = SaveCheckpointRequest.for_step(1)
    transitions = []

    def prepare(*args):
        assert get_ident() == owner_thread
        assert asyncio.get_running_loop() is owner
        transitions.append("prepare")
        return request.cut

    def apply(*args):
        assert get_ident() == owner_thread
        assert asyncio.get_running_loop() is owner
        transitions.append("apply")
        released.set()

    controller = SimpleNamespace(
        request_snapshot=AsyncMock(
            return_value=CheckpointAssembly(request.cut, {}, {})
        ),
        publish_snapshot=AsyncMock(),
        rollback_snapshot=AsyncMock(),
        prepare_checkpoint_transaction=prepare,
        apply_prepared_checkpoint_transaction=apply,
        abort_checkpoint_transaction=Mock(),
    )
    checkpointer = await checkpointer_factory(
        controller, str(tmp_path / "repo"), interval_steps=1
    )
    waiting = asyncio.create_task(released.wait())
    try:
        await checkpointer.save(request)
        await asyncio.wait_for(waiting, 2)
        assert transitions == ["prepare", "apply"]
        assert await checkpointer.list_steps() == (1,)
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
