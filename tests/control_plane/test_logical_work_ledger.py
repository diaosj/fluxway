import asyncio
from dataclasses import replace

import pytest

from fluxway.common.ingestion import WorkerLayoutDescriptor
from fluxway.common.metadata import metadata_digest
from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.common.ingestion import SourceWorkAcknowledgement
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger


def bindings(compiled):
    return tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )


def segments(compiled, step):
    return tuple(
        segment
        for layout in compiled.worker_layouts
        for segment in layout.segments
        if segment.step == step
    )


def ack(lease):
    return SourceWorkAcknowledgement(lease.item.key, lease.lease_id)


@pytest.mark.asyncio
async def test_frontier_requires_every_interval_and_contiguous_steps(source_layout):
    compiled = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    await ledger.publish(compiled.work_epoch, bindings(compiled))
    first = await ledger.acquire(0, segments(compiled, 0))
    later = await ledger.acquire(0, segments(compiled, 1))
    assert first[0].item.key == first[1].item.key
    assert first[0].lease_id != first[1].lease_id
    await ledger.acknowledge(tuple(map(ack, later)))
    await ledger.acknowledge((ack(first[1]), ack(first[1])))
    assert await ledger.candidate_cut() == 0
    await ledger.acknowledge((ack(first[0]),))
    assert await ledger.candidate_cut() == 2
    assert ledger.committed_cut == 0


@pytest.mark.asyncio
async def test_acquire_and_ack_batches_are_atomic(source_layout):
    compiled = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    await ledger.publish(compiled.work_epoch, bindings(compiled))
    step = segments(compiled, 0)
    first = (await ledger.acquire(0, step[:1]))[0]
    with pytest.raises(RuntimeError, match="leased"):
        await ledger.acquire(0, step[::-1])
    second = (await ledger.acquire(0, step[1:]))[0]
    with pytest.raises(RuntimeError, match="stale"):
        await ledger.acknowledge(
            (ack(first), replace(ack(second), lease_id="old-attempt"))
        )
    await ledger.acknowledge((ack(second),))
    assert await ledger.candidate_cut() == 0
    wrong_key = replace(first.item.key, assignment_id="wrong-assignment")
    with pytest.raises(ValueError, match="conflicts"):
        await ledger.acknowledge((replace(ack(first), key=wrong_key),))
    await ledger.acknowledge((ack(first),))
    assert await ledger.candidate_cut() == 1


@pytest.mark.asyncio
async def test_publication_rejects_incomplete_and_overlapping_bindings(source_layout):
    compiled = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    bound = bindings(compiled)
    with pytest.raises(ValueError, match="cover|missing"):
        await ledger.publish(compiled.work_epoch, bound[:1])
    with pytest.raises(ValueError, match="overlapping"):
        await ledger.publish(
            compiled.work_epoch,
            (
                bound[0],
                replace(
                    bound[1],
                    segments=tuple(
                        replace(segment, worker_id=bound[1].worker_id)
                        for segment in bound[0].segments
                    ),
                ),
            ),
        )
    await ledger.publish(compiled.work_epoch, bound)


@pytest.mark.asyncio
async def test_cancelled_window_wait_does_not_block_ack_or_checkpoint(source_layout):
    compiled = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=1)
    await ledger.publish(compiled.work_epoch, bindings(compiled))
    waiting = asyncio.create_task(ledger.acquire(0, segments(compiled, 1)))
    await asyncio.sleep(0)
    assert not waiting.done()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    leases = await ledger.acquire(0, segments(compiled, 0))
    await ledger.acknowledge(tuple(map(ack, leases)))
    waiting = asyncio.create_task(ledger.acquire(0, segments(compiled, 1)))
    await asyncio.sleep(0)
    assert not waiting.done()
    await ledger.commit_cut(1)
    assert len(await asyncio.wait_for(waiting, 1)) == 2


@pytest.mark.asyncio
async def test_checkpoint_restore_fences_attempt_and_keeps_canonical_plan(
    source_layout,
):
    original = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=3)
    await ledger.publish(original.work_epoch, bindings(original))
    old = await ledger.acquire(0, segments(original, 1))
    first = await ledger.acquire(0, segments(original, 0))
    with pytest.raises(ValueError):
        ledger.checkpoint_state(committed_cut=1)
    await ledger.acknowledge(tuple(map(ack, first)))
    image = ledger.checkpoint_state(committed_cut=1)
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    cut = CheckpointCut.for_position("cut", 1)
    prepared = ledger.prepare_checkpoint(
        "checkpoint", permit, cut, metadata_digest(image), image
    )
    assert ledger.committed_cut == 0
    ledger.abort_prepared("checkpoint", permit)
    with pytest.raises(RuntimeError, match="prepare"):
        ledger.apply_prepared(prepared)
    prepared = ledger.prepare_checkpoint(
        "checkpoint", permit, cut, metadata_digest(image), image
    )
    ledger.apply_prepared(prepared)
    ledger.apply_prepared(prepared)
    assert ledger.committed_cut == 1
    assert all(lease.item.key.step >= 1 for lease in ledger._leases.values())
    assert not ledger._acknowledged
    rebound = source_layout(demands=(1, 1, 2))
    assert rebound.work_epoch == original.work_epoch
    restored = LogicalWorkLedger(max_in_flight_steps=3)
    await restored.publish(rebound.work_epoch, bindings(rebound))
    restored.restore_state(image)
    replay = await restored.acquire(0, segments(rebound, 1))
    assert sorted(
        row
        for lease in replay
        for row in range(lease.item.logical_start, lease.item.logical_end)
    ) == [4, 5, 6, 7]
    with pytest.raises(RuntimeError, match="stale"):
        await restored.acknowledge((ack(old[0]),))
    with pytest.raises(RuntimeError, match="committed cut"):
        await restored.acquire(0, segments(rebound, 0))
    changed = source_layout(revision="different")
    another = LogicalWorkLedger(max_in_flight_steps=3)
    await another.publish(changed.work_epoch, bindings(changed))
    with pytest.raises(ValueError, match="immutable plans"):
        another.restore_state(image)


@pytest.mark.asyncio
async def test_contiguous_revision_windows_advance_incrementally(source_layout):
    first = source_layout(cardinality=4, terminal=False)
    second = source_layout(cardinality=4, revision="append", epoch=1, start_step=1)
    ledger = LogicalWorkLedger(max_in_flight_steps=3)
    await ledger.publish(first.work_epoch, bindings(first))
    await ledger.publish(second.work_epoch, bindings(second))
    for layout in (second, first):
        leases = await ledger.acquire(
            layout.work_epoch.epoch, segments(layout, layout.work_epoch.start_step)
        )
        await ledger.acknowledge(tuple(map(ack, leases)))
    assert await ledger.candidate_cut() == 2
