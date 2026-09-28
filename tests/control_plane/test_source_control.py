import asyncio

import pytest

from fluxway.common.ingestion import WorkerLayoutDescriptor
from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.common.types.configs import SourceControlConfig
from fluxway.common.ingestion import SourceWorkAcknowledgement
from fluxway.control_plane.source_control import SourceControlRuntime


@pytest.mark.asyncio
async def test_runtime_publishes_and_acknowledges_logical_work(source_layout) -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await runtime.publish_work_epoch(compiled.work_epoch, bindings)
    leases = await runtime.acquire_work(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await runtime.acknowledge_work(
        tuple(
            SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
            for lease in leases
        )
    )
    assert await runtime.candidate_work_cut() == 1


@pytest.mark.asyncio
async def test_runtime_waits_for_logical_cut_without_polling(source_layout) -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await runtime.publish_work_epoch(compiled.work_epoch, bindings)
    waiting = asyncio.create_task(runtime.wait_for_work_cut(1))
    await asyncio.sleep(0)
    assert not waiting.done()
    leases = await runtime.acquire_work(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await runtime.acknowledge_work(
        tuple(
            SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
            for lease in leases
        )
    )
    assert await waiting == 1


def test_runtime_checkpoint_is_logical_work_state() -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    state = runtime.logical_work_checkpoint_state()
    assert set(state) == {"version", "committed_cut", "epochs"}


def test_runtime_checkpoint_transaction_uses_logical_cut() -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    state = runtime.logical_work_checkpoint_state()
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    with pytest.raises(ValueError, match="digest"):
        runtime.prepare_checkpoint(
            "checkpoint", permit, CheckpointCut.for_position("cut", 0), "bad", state
        )
