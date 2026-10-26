import asyncio

import pytest

from fluxway.common.async_operations import AsyncOperationSupervisor


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_cancellation", [False, True])
async def test_observed_operation_cancellation_is_not_reported_again(
    prior_cancellation,
):
    supervisor = AsyncOperationSupervisor()

    async def caller():
        if prior_cancellation:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
            assert task.cancelling() > 0

        async def operation():
            raise asyncio.CancelledError("operation cancelled itself")

        with pytest.raises(asyncio.CancelledError):
            await supervisor.run(operation)
        supervisor.raise_unobserved()
        assert await supervisor.drain() == ()

    await asyncio.create_task(caller())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [ValueError, asyncio.CancelledError])
async def test_abandoned_failure_remains_owned_until_drain(failure_type):
    supervisor = AsyncOperationSupervisor()
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        raise failure_type("late failure")

    caller = asyncio.create_task(supervisor.run(operation))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    supervisor.stop_accepting()
    draining = asyncio.create_task(supervisor.drain())
    await asyncio.sleep(0)
    assert not draining.done()
    release.set()
    failures = await draining
    assert len(failures) == 1
    assert isinstance(failures[0], failure_type)
    if failure_type is ValueError:
        assert str(failures[0]) == "late failure"
    with pytest.raises(RuntimeError, match="closing"):
        await supervisor.run(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [None, ValueError, asyncio.CancelledError])
async def test_caller_cancellation_at_completion_keeps_failure_owned(failure_type):
    supervisor = AsyncOperationSupervisor()

    async def operation():
        asyncio.get_running_loop().call_soon(caller.cancel)
        if failure_type is not None:
            raise failure_type("completed failure")
        return "completed"

    caller = asyncio.create_task(supervisor.run(operation))
    with pytest.raises(asyncio.CancelledError):
        await caller
    failures = await supervisor.drain()
    if failure_type is None:
        assert failures == ()
        supervisor.raise_unobserved()
    else:
        assert len(failures) == 1
        assert isinstance(failures[0], failure_type)
        with pytest.raises(BaseExceptionGroup) as caught:
            supervisor.raise_unobserved()
        assert caught.value.exceptions == failures


@pytest.mark.asyncio
@pytest.mark.parametrize("child_task", [False, True])
async def test_owned_operation_cannot_stop_or_drain_its_owner(child_task):
    supervisor = AsyncOperationSupervisor()

    async def operation():
        if child_task:
            await asyncio.create_task(supervisor.drain())
        else:
            supervisor.stop_accepting()

    with pytest.raises(RuntimeError, match="cannot close its supervisor"):
        await asyncio.wait_for(supervisor.run(operation), timeout=2)
    assert supervisor.accepting
    assert await supervisor.drain() == ()


@pytest.mark.asyncio
async def test_owner_preserves_native_async_concurrency_and_observed_errors():
    supervisor = AsyncOperationSupervisor()
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def operation(index):
        entered[index].set()
        await release.wait()
        raise ValueError(str(index))

    callers = [
        asyncio.create_task(supervisor.run(lambda index=i: operation(index)))
        for i in range(2)
    ]
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in entered)), timeout=2
    )
    release.set()
    failures = await asyncio.gather(*callers, return_exceptions=True)
    assert [str(error) for error in failures] == ["0", "1"]
    assert await supervisor.drain() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
@pytest.mark.parametrize("failure_type", [None, ValueError, asyncio.CancelledError])
async def test_concurrent_drain_waits_for_result_observation(
    cancel_waiter, failure_type
):
    supervisor = AsyncOperationSupervisor()
    completed = asyncio.Event()

    async def operation():
        completed.set()
        if cancel_waiter:
            asyncio.get_running_loop().call_soon(caller.cancel)
        if failure_type is not None:
            raise failure_type("completed failure")
        return "completed"

    async def drain_at_completion():
        await completed.wait()
        await asyncio.sleep(0)
        return await supervisor.drain()

    draining = asyncio.create_task(drain_at_completion())
    caller = asyncio.create_task(supervisor.run(operation))
    result, failures = await asyncio.wait_for(
        asyncio.gather(caller, draining, return_exceptions=True), timeout=2
    )
    if cancel_waiter:
        assert isinstance(result, asyncio.CancelledError)
    elif failure_type is not None:
        assert isinstance(result, failure_type)
    else:
        assert result == "completed"
    if cancel_waiter and failure_type is not None:
        assert len(failures) == 1
        assert isinstance(failures[0], failure_type)
    else:
        assert failures == ()
    assert await supervisor.drain() == failures


@pytest.mark.asyncio
async def test_cancelled_drain_does_not_cancel_owned_work_or_settlement():
    supervisor = AsyncOperationSupervisor()
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        raise ValueError("late failure")

    caller = asyncio.create_task(supervisor.run(operation))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    draining = asyncio.create_task(supervisor.drain())
    await asyncio.sleep(0)
    draining.cancel()
    with pytest.raises(asyncio.CancelledError):
        await draining
    release.set()
    failures = await asyncio.wait_for(supervisor.drain(), timeout=2)
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert str(failures[0]) == "late failure"
