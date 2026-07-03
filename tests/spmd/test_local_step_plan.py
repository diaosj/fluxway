import asyncio

import pytest

from fluxway.spmd.worker.local_plan import (
    LocalStepNodeKind,
    LocalStepPlan,
    LocalStepPlanExecutor,
    MappingCallableRegistry,
)


@pytest.mark.asyncio
async def test_single_call_plan_returns_result():
    executor = LocalStepPlanExecutor()
    plan = LocalStepPlan.single_call("op")
    registry = MappingCallableRegistry({"op": lambda: 7})

    assert await executor.execute(plan, registry) == 7


@pytest.mark.asyncio
async def test_same_resource_nodes_are_serialized():
    executor = LocalStepPlanExecutor()
    events: list[str] = []

    async def first():
        events.append("first:start")
        await asyncio.sleep(0.01)
        events.append("first:end")
        return 1

    async def second():
        events.append("second:start")
        await asyncio.sleep(0.01)
        events.append("second:end")
        return 2

    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.CALL), int(LocalStepNodeKind.CALL)),
        callable_ids=("first", "second"),
        resource_ids=(3, 3),
        dependency_offsets=(0, 0, 0),
        dependencies=(),
        result_node=1,
    )
    registry = MappingCallableRegistry({"first": first, "second": second})

    assert await executor.execute(plan, registry) == 2
    assert events in (
        ["first:start", "first:end", "second:start", "second:end"],
        ["second:start", "second:end", "first:start", "first:end"],
    )


@pytest.mark.asyncio
async def test_different_resources_can_overlap():
    executor = LocalStepPlanExecutor()
    started = asyncio.Event()
    release = asyncio.Event()
    overlap_seen = False

    async def first():
        nonlocal overlap_seen
        started.set()
        await release.wait()
        return 1

    async def second():
        nonlocal overlap_seen
        await started.wait()
        overlap_seen = True
        release.set()
        return 2

    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.CALL), int(LocalStepNodeKind.CALL)),
        callable_ids=("first", "second"),
        resource_ids=(1, 2),
        dependency_offsets=(0, 0, 0),
        dependencies=(),
        result_node=1,
    )
    registry = MappingCallableRegistry({"first": first, "second": second})

    assert await executor.execute(plan, registry) == 2
    assert overlap_seen


@pytest.mark.asyncio
async def test_unsupported_node_kind_raises():
    executor = LocalStepPlanExecutor()
    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.SEND),),
        callable_ids=("",),
        resource_ids=(-1,),
        dependency_offsets=(0, 0),
        dependencies=(),
        result_node=0,
    )

    with pytest.raises(NotImplementedError):
        await executor.execute(plan, MappingCallableRegistry({}))


@pytest.mark.asyncio
async def test_drain_waits_for_active_plan():
    executor = LocalStepPlanExecutor()
    release = asyncio.Event()
    finished = False

    async def slow_call():
        nonlocal finished
        await release.wait()
        finished = True
        return 1

    task = asyncio.create_task(
        executor.execute(
            LocalStepPlan.single_call("slow"),
            MappingCallableRegistry({"slow": slow_call}),
        )
    )
    await asyncio.sleep(0)
    drain_task = asyncio.create_task(executor.drain())
    await asyncio.sleep(0.01)
    assert not drain_task.done()
    release.set()
    await task
    await drain_task
    assert finished
