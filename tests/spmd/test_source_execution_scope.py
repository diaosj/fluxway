import asyncio
from unittest.mock import AsyncMock

import pytest

from fluxway.common.ingestion import (
    IngestionLayoutActivation,
    SourceWorkAcknowledgement,
    WorkerLayoutDescriptor,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.spmd.worker.source_scope import SourceExecutionScope
from fluxway.spmd.worker.source_scope import SourceExhaustedError


def activation(compiled):
    return IngestionLayoutActivation(
        epoch=compiled.work_epoch.epoch,
        start_position=compiled.work_epoch.start_step,
        end_position=compiled.work_epoch.end_step,
        layout_digest=compiled.layout_digest,
        terminal=compiled.work_epoch.terminal,
        descriptor=compiled.encoded_worker_layouts()[0],
    )


@pytest.mark.asyncio
async def test_successive_windows_share_authority_and_keep_prior_grants(source_layout):
    first = source_layout(demands=(1,), cardinality=2, terminal=False)
    second = source_layout(
        demands=(1,), cardinality=2, revision="next", epoch=1, start_step=2
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    for compiled in (first, second):
        await ledger.publish(
            compiled.work_epoch,
            tuple(
                map(WorkerLayoutDescriptor.decode, compiled.encoded_worker_layouts())
            ),
        )
    scope = SourceExecutionScope(
        acquire=ledger.acquire_available,
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 2,
    )
    install = AsyncMock()
    scope.require_unactivated()
    await scope.activate("read", activation(first), install)
    await scope.activate("read", activation(first), install)
    install.assert_awaited_once()
    with pytest.raises(RuntimeError, match="fresh"):
        scope.require_unactivated()
    leases = await scope.take("read", 0)
    await scope.activate("read", activation(second), install)
    for step in range(4):
        if step:
            leases = await scope.take("read", step)
        assert scope.finish("read", step) == leases
        await ledger.acknowledge(
            tuple(SourceWorkAcknowledgement(v.item.key, v.lease_id) for v in leases)
        )
        await ledger.commit_cut(step + 1)
    assert await ledger.candidate_cut() == 4


@pytest.mark.asyncio
async def test_activation_failure_never_publishes_partial_window(source_layout):
    compiled = source_layout(demands=(1,), cardinality=2)
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    await ledger.publish(
        compiled.work_epoch,
        tuple(map(WorkerLayoutDescriptor.decode, compiled.encoded_worker_layouts())),
    )
    scope = SourceExecutionScope(
        acquire=ledger.acquire_available,
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    with pytest.raises(RuntimeError, match="install failed"):
        await scope.activate(
            "read",
            activation(compiled),
            AsyncMock(side_effect=RuntimeError("install failed")),
        )
    scope.require_unactivated()
    with pytest.raises(RuntimeError, match="install failed"):
        await scope.take("read", 0)


@pytest.mark.asyncio
async def test_unfinished_activation_is_not_restore_ready(source_layout):
    compiled = source_layout(demands=(1,), cardinality=2)
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    await ledger.publish(
        compiled.work_epoch,
        tuple(map(WorkerLayoutDescriptor.decode, compiled.encoded_worker_layouts())),
    )
    scope = SourceExecutionScope(
        acquire=ledger.acquire_available,
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def install(value):
        entered.set()
        await release.wait()

    task = asyncio.create_task(scope.activate("read", activation(compiled), install))
    await entered.wait()
    with pytest.raises(RuntimeError, match="fresh"):
        scope.require_unactivated()
    waiting = asyncio.create_task(scope.take("read", 0))
    await asyncio.sleep(0)
    assert not waiting.done()
    release.set()
    await task
    leases = await waiting
    assert scope.finish("read", 0) == leases


@pytest.mark.asyncio
async def test_cancelled_activation_waiter_does_not_abandon_installation(source_layout):
    compiled = source_layout(demands=(1,), cardinality=2)
    scope = SourceExecutionScope(
        acquire=AsyncMock(),
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def install(value):
        calls.append(value)
        entered.set()
        await release.wait()

    value = activation(compiled)
    caller = asyncio.create_task(scope.activate("read", value, install))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    second = asyncio.create_task(scope.activate("read", value, install))
    release.set()
    await second
    assert calls == [value]
    assert not scope._installing
    assert scope._windows["read"][0].activation == value


@pytest.mark.asyncio
async def test_take_waits_for_next_contiguous_activation(source_layout):
    first = source_layout(demands=(1,), cardinality=1, terminal=False)
    second = source_layout(
        demands=(1,), cardinality=1, revision="next", epoch=1, start_step=1
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    for compiled in (first, second):
        await ledger.publish(
            compiled.work_epoch,
            tuple(
                map(WorkerLayoutDescriptor.decode, compiled.encoded_worker_layouts())
            ),
        )
    scope = SourceExecutionScope(
        acquire=ledger.acquire_available,
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    install = AsyncMock()
    await scope.activate("read", activation(first), install)

    waiting = asyncio.create_task(scope.take("read", 1))
    await asyncio.sleep(0)
    assert not waiting.done()

    await scope.activate("read", activation(second), install)
    leases = await asyncio.wait_for(waiting, timeout=1)
    assert leases[0].item.key.step == 1


@pytest.mark.asyncio
async def test_terminal_activation_returns_typed_exhaustion(source_layout):
    compiled = source_layout(demands=(1,), cardinality=1, terminal=True)
    scope = SourceExecutionScope(
        acquire=AsyncMock(),
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    await scope.activate("read", activation(compiled), AsyncMock())
    with pytest.raises(SourceExhaustedError, match="beyond terminal end"):
        await scope.take("read", 1)


@pytest.mark.asyncio
async def test_cancel_wakes_source_waiters(source_layout):
    compiled = source_layout(demands=(1,), cardinality=1, terminal=False)
    scope = SourceExecutionScope(
        acquire=AsyncMock(),
        spawn=asyncio.create_task,
        completed=lambda op: 0,
        depth=lambda op: 1,
    )
    await scope.activate("read", activation(compiled), AsyncMock())
    waiting = asyncio.create_task(scope.take("read", 1))
    await asyncio.sleep(0)
    await scope.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
