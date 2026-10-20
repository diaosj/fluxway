import asyncio
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.manifest import (
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)


@pytest.mark.asyncio
async def test_completed_step_releases_frame_payload_without_manual_cleanup():
    payload = np.arange(4, dtype=np.int64)
    payload_ref = weakref.ref(payload)
    builder = TransportFrameBuilder(
        step=0,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
    )
    entry = builder.add_tensor_transfer_entry(
        transfer_code=11,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        coherence_epoch=0,
    )
    builder.add_payload(entry, PayloadRole.VALUES, payload, payload.size)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 0, {11: 1}, [])
    await mailbox.deposit_transport_frame("op", "source", builder.build())
    events: list[str] = []
    clear_step = mailbox.clear_step

    async def release_input(_op_name: str, _step: int) -> None:
        events.append("coherence")

    async def clear_mailbox(_op_name: str, _step: int) -> None:
        events.append("mailbox")
        await clear_step(_op_name, _step)

    raw_worker = SPMDWorker.__ray_actor_class__
    worker = SimpleNamespace(
        _finished_plan_steps={
            "op": {0: ExecutionResult(status=ExecutionStatus.SUCCESS)}
        },
        _completed_step_count={"op": 0},
        _coherence_input_grants={},
        _coherence_alias_dependencies={},
        _coherence_input_tensors={},
        _coherence_input_bindings={},
        _local_input_origins={},
        _barriers_declared={("op", 0): True},
        _clear_direct_local_step=lambda _op, _step: None,
        _return_input_coherence=release_input,
        _spawn_background_task=asyncio.create_task,
        _mailbox=SimpleNamespace(clear_step=clear_mailbox),
        _advance_step=lambda _op, *, result: asyncio.sleep(0),
        _source_operator_names=(),
        _source_control_operator_names=(),
    )
    await raw_worker._commit_finished_steps(worker, "op")
    await asyncio.sleep(0)

    del builder, payload
    gc.collect()
    assert payload_ref() is None
    assert await mailbox.get_tensor_shard_views("op", 11, 0) == {}
    assert "coherence" in events and "mailbox" in events
