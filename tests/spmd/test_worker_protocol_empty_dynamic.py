from typing import cast
from types import SimpleNamespace

import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.assembly import compile_assemble_plan
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.manifest import EmptyKind, PayloadRole, TransportFrameBuilder
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager


class FakeOpManager:
    def _resolve_input_spec(self, op_name, tuuid):
        return None

    def get_predecessor_topology(self, op_name):
        return {"Input": SimpleNamespace(tuuid_sharding={52: object()})}

    def _resolve_producer_spec(self, tuuid, provider_map=None):
        return None


@pytest.mark.asyncio
async def test_dynamic_assemble_accepts_complete_protocol_empty_arrivals():
    mailbox = WorkerMailbox()
    mailbox.declare_step("lookup", 2, {52: 3}, [])

    for rank in range(3):
        builder = TransportFrameBuilder(
            step=2,
            sender_spmd_code=0,
            sender_rank=rank,
            sender_world_size=3,
        )
        edge = builder.add_edge(
            tuuid=52,
            meta_state=0,
            empty_kind=EmptyKind.PROTOCOL_EMPTY,
            min_rows=0,
            expected_count=3,
            global_shape=(768,),
            source_step=2,
        )
        builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int64), 0)
        await mailbox.deposit_transport_frame("lookup", "Input", builder.build())

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    WorkerClass = cls.modified_class if cls else SPMDWorker  # type: ignore[reportAttributeAccessIssue]
    worker = WorkerClass.__new__(WorkerClass)  # type: ignore[reportArgumentType]
    worker.rank = 0
    worker.spmd_id = "Embedding_geek_id"
    worker._mailbox = mailbox
    worker._op_manager = cast(OperatorManager, FakeOpManager())

    shards = await mailbox.get_tensor_shard_views("lookup", 52, 2)
    arrivals = await mailbox.get_tensor_arrivals("lookup", 52, 2)
    result = worker._assemble_to_dynamic_tensor(
        "lookup",
        52,
        2,
        shards,
        arrivals,
        worker.op_manager.get_predecessor_topology("lookup"),
        assemble_plan=compile_assemble_plan(
            op_manager=worker.op_manager,
            op_name="lookup",
            tuuid=52,
        ),
    )

    assert shards == {}
    assert result.flat_values.shape == (0,)
    assert result.flat_values.dtype == np.int64
    assert result.global_shape == (768,)
    assert result.receipt is not None
    assert result.receipt.world_size == 3
    assert result.receipt.materialize(0).shape == (0,)
