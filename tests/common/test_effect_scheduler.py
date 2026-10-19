from __future__ import annotations

import asyncio

import pytest

from fluxway.common.effect_scheduler import (
    BoundedLifecycleEffectScheduler,
    LifecycleEffectExecutionMode,
    LifecycleEffectQueueFullError,
    LifecycleEffectSchedulerClosedError,
    LifecycleEffectSubmission,
)


@pytest.mark.asyncio
async def test_same_ordering_key_is_strictly_serial() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=8, worker_count=3)
    await scheduler.start()
    active = 0
    maximum = 0
    order: list[int] = []
    gate = asyncio.Event()

    async def operation(value: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        order.append(value)
        if value == 0:
            await gate.wait()
        active -= 1
        return value

    first = scheduler.submit(
        LifecycleEffectSubmission(operation, args=(0,), ordering_key="controller")
    )
    second = scheduler.submit(
        LifecycleEffectSubmission(operation, args=(1,), ordering_key="controller")
    )
    await asyncio.sleep(0)
    assert order == [0]
    gate.set()
    assert (await first).value == 0
    assert (await second).value == 1
    assert maximum == 1
    assert order == [0, 1]
    await scheduler.close()


@pytest.mark.asyncio
async def test_different_ordering_keys_run_concurrently() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=8, worker_count=2)
    await scheduler.start()
    entered = asyncio.Barrier(2)

    async def operation() -> None:
        await entered.wait()

    first = scheduler.submit(
        LifecycleEffectSubmission(operation, ordering_key="controller")
    )
    second = scheduler.submit(
        LifecycleEffectSubmission(operation, ordering_key="driver")
    )
    assert (await first).error is None
    assert (await second).error is None
    await scheduler.close()


@pytest.mark.asyncio
async def test_queue_full_is_reported_without_waiting() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=1, worker_count=1)
    await scheduler.start()
    release = asyncio.Event()

    async def operation() -> None:
        await release.wait()

    first = scheduler.submit(LifecycleEffectSubmission(operation, ordering_key="a"))
    await asyncio.sleep(0)
    second = scheduler.submit(LifecycleEffectSubmission(operation, ordering_key="b"))
    with pytest.raises(LifecycleEffectQueueFullError):
        scheduler.submit(LifecycleEffectSubmission(operation, ordering_key="c"))
    release.set()
    await first
    await second
    await scheduler.close()


def _thread_operation() -> str:
    return "thread"


def _process_operation(value: int) -> int:
    return value * value


@pytest.mark.asyncio
async def test_thread_and_process_lanes_execute_outside_event_loop() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=8, worker_count=2)
    await scheduler.start()
    thread_outcome = await scheduler.submit(
        LifecycleEffectSubmission(
            _thread_operation, mode=LifecycleEffectExecutionMode.THREAD
        )
    )
    process_outcome = await scheduler.submit(
        LifecycleEffectSubmission(
            _process_operation,
            args=(7,),
            mode=LifecycleEffectExecutionMode.PROCESS,
        )
    )
    assert thread_outcome.value == "thread"
    assert process_outcome.value == 49
    assert thread_outcome.error is None
    assert process_outcome.error is None
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_cancel_accepted_effect() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=4, worker_count=1)
    await scheduler.start()
    completed = asyncio.Event()

    async def operation() -> str:
        await asyncio.sleep(0.01)
        completed.set()
        return "done"

    wait = scheduler.submit(LifecycleEffectSubmission(operation))
    wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait
    await asyncio.wait_for(completed.wait(), timeout=1)
    await scheduler.close()


@pytest.mark.asyncio
async def test_close_rejects_new_submissions_and_is_idempotent() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=2, worker_count=1)
    await scheduler.start()
    await scheduler.close()
    await scheduler.close()
    with pytest.raises(LifecycleEffectSchedulerClosedError):
        scheduler.submit(LifecycleEffectSubmission(asyncio.sleep))


@pytest.mark.asyncio
async def test_concurrent_close_calls_share_one_shutdown() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=2, worker_count=2)
    await scheduler.start()
    await asyncio.gather(scheduler.close(), scheduler.close(), scheduler.close())
    assert not scheduler.started


@pytest.mark.asyncio
async def test_concurrent_start_calls_create_one_worker_set() -> None:
    scheduler = BoundedLifecycleEffectScheduler(queue_size=2, worker_count=2)
    await asyncio.gather(scheduler.start(), scheduler.start(), scheduler.start())
    assert scheduler.started
    await scheduler.close()
