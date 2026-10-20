import asyncio
from types import SimpleNamespace

import pytest

from fluxway.control_plane.spmd_manager.manager import SPMDHandle, SPMDManager


def manager_with(*drivers):
    manager = SPMDManager()
    for index, driver in enumerate(drivers):
        handle = SPMDHandle(f"driver{index}")
        handle.set_driver_handle(driver)
        manager.register_spmd(handle.spmd_id, handle)
    return manager


@pytest.mark.asyncio
async def test_all_rpc_calls_are_enqueued_before_first_await():
    calls = []
    release = asyncio.Event()

    def remote(rank):
        def enqueue(value):
            calls.append((rank, value))

            async def finish():
                await release.wait()
                return rank

            return asyncio.create_task(finish())

        return enqueue

    manager = manager_with(
        *(
            SimpleNamespace(invoke=SimpleNamespace(remote=remote(rank)))
            for rank in range(4)
        )
    )
    operation = manager.enqueue_driver_calls("invoke", 17)
    assert calls == [(rank, 17) for rank in range(4)]
    assert not operation.done()
    release.set()
    assert await operation == list(range(4))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    ["missing_driver", "missing_method", "noncallable_remote"],
)
async def test_invalid_participant_prevents_partial_rpc_submission(invalid):
    calls = []

    def enqueue():
        calls.append(True)
        return asyncio.get_running_loop().create_future()

    driver = SimpleNamespace(invoke=SimpleNamespace(remote=enqueue))
    manager = manager_with(driver, driver)
    handle = manager._spmd_handles["driver1"]
    if invalid == "missing_driver":
        handle.set_driver_handle(None)
    elif invalid == "missing_method":
        handle.set_driver_handle(SimpleNamespace())
    else:
        handle.set_driver_handle(SimpleNamespace(invoke=SimpleNamespace(remote=None)))
    assert callable(getattr(manager, "enqueue_driver_calls", None))
    with pytest.raises((RuntimeError, AttributeError, TypeError)):
        manager.enqueue_driver_calls("invoke")
    assert calls == []


@pytest.mark.asyncio
async def test_remote_failure_is_raised_without_waiting_for_slow_peer():
    release = asyncio.Event()

    async def slow():
        await release.wait()

    async def fail():
        raise RuntimeError("remote failed")

    slow_task = asyncio.create_task(slow())
    manager = manager_with(
        SimpleNamespace(invoke=SimpleNamespace(remote=lambda: slow_task)),
        SimpleNamespace(
            invoke=SimpleNamespace(remote=lambda: asyncio.create_task(fail()))
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="remote failed"):
            await asyncio.wait_for(manager.enqueue_driver_calls("invoke"), 2)
        assert not slow_task.done()
    finally:
        release.set()
        await slow_task


@pytest.mark.asyncio
async def test_empty_registry_is_rejected():
    with pytest.raises(RuntimeError, match="participants"):
        SPMDManager().enqueue_driver_calls("invoke")


@pytest.mark.asyncio
async def test_submission_failure_remains_an_exception_and_observes_prior_calls():
    release = asyncio.Event()
    finished = asyncio.Event()

    async def peer():
        await release.wait()
        finished.set()
        raise ValueError("peer also failed")

    prior = asyncio.create_task(peer())

    def fail_submission():
        raise RuntimeError("serialization failed")

    manager = manager_with(
        SimpleNamespace(invoke=SimpleNamespace(remote=lambda: prior)),
        SimpleNamespace(invoke=SimpleNamespace(remote=fail_submission)),
    )
    with pytest.raises(RuntimeError, match="serialization failed"):
        await manager.enqueue_driver_calls("invoke")
    assert not prior.done()
    release.set()
    await finished.wait()
    await asyncio.sleep(0)
    assert prior.done()
    assert prior._log_traceback is False


def test_topology_is_copied_before_driver_registration():
    manager = SPMDManager()
    topology = {"reader": ["input"], "input": []}
    manager.establish_topology(topology)
    topology["reader"].clear()
    assert manager.get_successors("reader") == ["input"]
    assert manager.get_predecessors("input") == ["reader"]


def test_topology_replacement_after_registration_fails_before_mutation():
    manager = SPMDManager()
    manager.establish_topology({"reader": ["input"], "input": []})
    manager.register_spmd("reader", SPMDHandle("reader"))
    with pytest.raises(RuntimeError, match="before Driver registration"):
        manager.establish_topology({"reader": []})
    assert manager.get_successors("reader") == ["input"]
    assert manager.get_predecessors("input") == ["reader"]
