import asyncio
from dataclasses import replace

import pytest

from fluxway.common.ingestion import SourceWorkAcknowledgement, WorkerLayoutDescriptor
from fluxway.common.metadata import metadata_digest
from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.control_plane.ingestion.compiler import AlgebraicLayoutCompiler
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
@pytest.mark.parametrize("field", ["view_identity", "lineage_digest"])
async def test_publication_rejects_incomplete_source_identity(
    source_layout, field: str
):
    compiled = source_layout()
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    bound = list(bindings(compiled))
    bound[0] = replace(bound[0], **{field: f"changed-{field}"})
    with pytest.raises(ValueError, match="source, view, schema, or lineage"):
        await ledger.publish(compiled.work_epoch, tuple(bound))


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
    assert len(await asyncio.wait_for(waiting, 1)) == 2
    await ledger.commit_cut(1)


@pytest.mark.asyncio
async def test_checkpoint_restore_fences_attempt_and_accepts_new_plan(
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
    seed = source_layout(demands=(1, 1, 2), cardinality=8)
    rebound = AlgebraicLayoutCompiler().compile(
        replace(seed.request, units=(replace(seed.request.units[0], logical_start=4),)),
        epoch=0,
        start_step=1,
    )
    restored = LogicalWorkLedger(max_in_flight_steps=3)
    restored.restore_state(image)
    await restored.publish(rebound.work_epoch, bindings(rebound))
    replay = await restored.acquire(0, segments(rebound, 1))
    assert sorted(
        row
        for lease in replay
        for row in range(lease.item.logical_start, lease.item.logical_end)
    ) == [4, 5, 6, 7]
    with pytest.raises(RuntimeError, match="stale"):
        await restored.acknowledge((ack(old[0]),))
    another = LogicalWorkLedger(max_in_flight_steps=3)
    another.restore_state(image)
    with pytest.raises(ValueError, match="contiguous"):
        await another.publish(original.work_epoch, bindings(original))


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


@pytest.mark.asyncio
async def test_durable_cut_reclaims_old_epochs_and_keeps_one_anchor(
    source_layout,
) -> None:
    layouts = tuple(
        source_layout(cardinality=4, epoch=epoch, start_step=epoch, terminal=False)
        for epoch in range(3)
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    for layout in layouts:
        await ledger.publish(layout.work_epoch, bindings(layout))
        leases = await ledger.acquire(
            layout.work_epoch.epoch, segments(layout, layout.work_epoch.start_step)
        )
        await ledger.acknowledge(tuple(map(ack, leases)))
    await ledger.commit_cut(2)
    assert tuple(ledger._epochs) == (2,)
    assert all(key[0] == 2 for key in ledger._bindings)
    with pytest.raises(RuntimeError, match="reclaimed"):
        await ledger.acquire(0, segments(layouts[0], 0))


@pytest.mark.asyncio
async def test_checkpoint_restarts_epoch_number_at_durable_cut(source_layout) -> None:
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    compiled = []
    for epoch in range(3):
        value = source_layout(
            demands=(1,),
            cardinality=1,
            epoch=epoch,
            start_step=epoch,
            terminal=False,
        )
        compiled.append(value)
        await ledger.publish(value.work_epoch, bindings(value))
        leases = await ledger.acquire(epoch, segments(value, epoch))
        await ledger.acknowledge(tuple(map(ack, leases)))
    await ledger.commit_cut(2)
    image = ledger.checkpoint_state(committed_cut=2)
    assert image == {"committed_cut": 2}

    restored = LogicalWorkLedger(max_in_flight_steps=4)
    restored.restore_state(image)
    rebuilt = source_layout(
        demands=(1,), cardinality=1, epoch=0, start_step=2, terminal=False
    )
    await restored.publish(rebuilt.work_epoch, bindings(rebuilt))
    assert restored.committed_cut == 2
    with pytest.raises(RuntimeError, match="reclaimed"):
        await restored.acquire(1, segments(compiled[1], 1))
