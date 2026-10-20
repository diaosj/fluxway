import asyncio
from dataclasses import replace

import pytest

from fluxway.common.ingestion import SourceWorkAcknowledgement, WorkerLayoutDescriptor
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    TailLayoutMode,
    LayoutCompileRequest,
    AssignmentPolicy,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
)
from fluxway.common.ingestion import LayoutUnitDescriptor
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.spmd.worker.source_work import SourceWorkRuntime


async def runtime_for(*, credit=8, depth=4, completed=0):
    compiled = AlgebraicLayoutCompiler().compile(
        LayoutCompileRequest(
            source_identity="source",
            view_identity="v",
            schema_digest="schema",
            lineage_digest="v",
            units=(
                LayoutUnitDescriptor(
                    "u", "v", 8, "rows", "SplittableSpan", "n", "f", 0, b"meta"
                ),
            ),
            topology=ReaderTopologyDescriptor(
                1, (ReaderGroupDemand("w", (RankDemand("w", 1),)),)
            ),
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        )
    )
    binding = WorkerLayoutDescriptor.decode(compiled.encoded_worker_layouts()[0])
    ledger = LogicalWorkLedger(max_in_flight_steps=credit)
    await ledger.publish(compiled.work_epoch, (binding,))
    calls = []

    async def acquire(epoch, segments):
        calls.append(tuple(segment.step for segment in segments))
        return await ledger.acquire_available(epoch, segments)

    runtime = SourceWorkRuntime(
        epoch=0,
        segments_by_step={s.step: (s,) for s in binding.segments},
        batch_steps=depth,
        acquire=acquire,
        spawn=asyncio.create_task,
        completed=lambda: completed,
    )
    return runtime, ledger, calls


async def acknowledge(ledger, leases):
    await ledger.acknowledge(
        tuple(SourceWorkAcknowledgement(v.item.key, v.lease_id) for v in leases)
    )


@pytest.mark.asyncio
async def test_pipeline_steps_share_one_grant_batch():
    runtime, ledger, calls = await runtime_for()
    values = await asyncio.gather(*(runtime.take(step) for step in (3, 0, 2, 1)))
    assert calls == [(0, 1, 2, 3)]
    assert [v[0].item.key.step for v in values] == [3, 0, 2, 1]
    await acknowledge(ledger, tuple(v[0] for v in values))
    assert await ledger.candidate_cut() == 4
    with pytest.raises(RuntimeError, match="consumed"):
        await runtime.take(0)


@pytest.mark.asyncio
async def test_future_credit_wait_does_not_block_available_prefix():
    runtime, ledger, calls = await runtime_for(credit=2)
    future = asyncio.create_task(runtime.take(3))
    first = await asyncio.wait_for(runtime.take(0), 2)
    second = await asyncio.wait_for(runtime.take(1), 2)
    assert not future.done()
    await acknowledge(ledger, first + second)
    await ledger.commit_cut(2)
    third = await asyncio.wait_for(runtime.take(2), 2)
    fourth = await asyncio.wait_for(future, 2)
    assert calls == [(0, 1, 2, 3), (2, 3)]
    await acknowledge(ledger, third + fourth)
    assert await ledger.candidate_cut() == 4


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_grant_for_next_local_consumer():
    runtime, _, calls = await runtime_for()
    original = runtime._acquire
    entered, release = asyncio.Event(), asyncio.Event()

    async def acquire(epoch, segments):
        entered.set()
        await release.wait()
        return await original(epoch, segments)

    runtime._acquire = acquire
    waiter = asyncio.create_task(runtime.take(0))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert (await runtime.take(0))[0].item.key.step == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_partial_restore_skips_prior_chunk_steps():
    runtime, _, calls = await runtime_for(completed=3)
    assert (await runtime.take(3))[0].item.key.step == 3
    assert calls == [(3,)]


@pytest.mark.asyncio
async def test_invalid_reply_never_enters_cache():
    runtime, _, _ = await runtime_for()
    original = runtime._acquire

    async def acquire(epoch, segments):
        leases = await original(epoch, segments)
        return (replace(leases[0], item=leases[-1].item),)

    runtime._acquire = acquire
    with pytest.raises(ValueError, match="prefix"):
        await runtime.take(0)
    assert not runtime._cache
