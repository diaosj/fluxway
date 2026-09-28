import asyncio

import pytest

from fluxway.common.ingestion import SourceWorkAcknowledgement, WorkerLayoutDescriptor
from fluxway.common.types.configs import SourceControlConfig
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.control_plane.source_control import SourceControlRuntime


@pytest.mark.asyncio
async def test_drain_wait_is_released_by_contiguous_logical_acknowledgements(
    source_layout,
) -> None:
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await ledger.publish(compiled.work_epoch, bindings)
    waiting = asyncio.create_task(ledger.wait_for_candidate_cut(1))
    leases = await ledger.acquire(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await ledger.acknowledge(
        (SourceWorkAcknowledgement(leases[1].item.key, leases[1].lease_id),)
    )
    await asyncio.sleep(0)
    assert not waiting.done()
    await ledger.acknowledge(
        (SourceWorkAcknowledgement(leases[0].item.key, leases[0].lease_id),)
    )
    assert await waiting == 1


@pytest.mark.asyncio
async def test_restore_preserves_committed_cut_and_allows_resharded_identity(
    source_layout,
) -> None:
    first = LogicalWorkLedger(max_in_flight_steps=4)
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await first.publish(compiled.work_epoch, bindings)
    leases = await first.acquire(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await first.acknowledge(
        tuple(SourceWorkAcknowledgement(l.item.key, l.lease_id) for l in leases)
    )
    state = first.checkpoint_state(committed_cut=1)
    restored = LogicalWorkLedger(max_in_flight_steps=4)
    rebound = source_layout(cardinality=4, demands=(1, 1, 2))
    await restored.publish(
        rebound.work_epoch,
        tuple(
            WorkerLayoutDescriptor.decode(value)
            for value in rebound.encoded_worker_layouts()
        ),
    )
    restored.restore_state(state)
    assert restored.committed_cut == 1
    assert rebound.work_epoch == compiled.work_epoch


def test_source_runtime_has_no_ordered_stream_state() -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=4))
    assert set(runtime.logical_work_checkpoint_state()) == {
        "version",
        "committed_cut",
        "epochs",
    }
