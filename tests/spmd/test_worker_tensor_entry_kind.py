from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from fluxway.common.allocation import AllocationRegistry
from fluxway.common.coherence import CoherenceDirectory
from fluxway.common.compiled_graph import (
    CompiledLogicalTensor,
    CompiledTensorTransfer,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
)
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.spmd.worker.assembly import compile_assemble_plan
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.manifest import (
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)


_TEST_SHARDING = ShardingSpec.chunk(PartitionSpec())


class FakeOpManager:
    def __init__(self):
        self._operator_spec = SimpleNamespace(
            runtime_policy=SimpleNamespace(mode="sync"),
            tensor_access={"value": "readonly"},
            input_param_map={"value": ("value",)},
            input_treedefs={},
        )

    def get_predecessor_route(self, op_name, transfer_code):
        assert op_name == "lookup"
        assert transfer_code == 52
        transfer = CompiledTensorTransfer(
            spec=TensorTransfer(OperatorPort("output"), OperatorPort("input")),
            transfer_code=52,
            dependency_code=1,
            logical_tuuid=52,
            source_sharding=_TEST_SHARDING,
            target_sharding=_TEST_SHARDING,
        )
        return SimpleNamespace(
            dependency=SimpleNamespace(code=1),
            transfer=transfer,
            logical_tensor=CompiledLogicalTensor(
                logical_tuuid=52,
                producer=OperatorKey("Input", "emit"),
                producer_port="output",
                producer_sharding=_TEST_SHARDING,
                effective_sharding=_TEST_SHARDING,
            ),
            spmd_id="Input",
            local_only=False,
        )

    def get_transfer_source_layout(self, op_name, transfer_code):
        return SimpleNamespace(is_content_addressable=True)

    def get_transfer_target_layout(self, op_name, transfer_code):
        return SimpleNamespace(is_content_addressable=True)

    def get_operator_spec(self, op_name):
        assert op_name == "lookup"
        return self._operator_spec

    def get_static_args(self):
        return {}

    def get_operator_input_types(self, op_name):
        assert op_name == "lookup"
        return {"value": object}

    def get_input_transfer_codes(self, op_name):
        assert op_name == "lookup"
        return {"lookup.value": 52}

    def get_expected_control_arg_names(self, op_name):
        assert op_name == "lookup"
        return []


async def _mailbox_with_kinds(kinds):
    mailbox = WorkerMailbox()
    mailbox.declare_step("lookup", 2, {52: len(kinds)}, [])
    for rank, tensor_kind in enumerate(kinds):
        builder = TransportFrameBuilder(
            step=2,
            sender_spmd_code=0,
            sender_rank=rank,
            sender_world_size=len(kinds),
        )
        entry_index = builder.add_tensor_transfer_entry(
            transfer_code=52,
            layout_state=0,
            tensor_kind=tensor_kind,
            expected_count=len(kinds),
            global_shape=(768,),
            coherence_epoch=2,
        )
        builder.add_payload(
            entry_index,
            PayloadRole.VALUES,
            np.empty((0,), dtype=np.float32),
            0,
        )
        await mailbox.deposit_transport_frame("lookup", "Input", builder.build())
    return mailbox


def _worker(mailbox):
    metadata = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_class = metadata.modified_class if metadata else SPMDWorker
    worker = worker_class.__new__(worker_class)
    worker.rank = 0
    worker.spmd_id = "Target"
    worker._mailbox = mailbox
    worker._op_manager = cast(OperatorManager, FakeOpManager())
    worker._profiler_runtime = SimpleNamespace(enabled=False)  # type: ignore[assignment]
    assemble_plan = compile_assemble_plan(
        op_manager=worker.op_manager,
        op_name="lookup",
        transfer_code=52,
    )
    worker._assemble_plan_cache = {("lookup", 52): assemble_plan}
    worker._coherence_input_grants = {}
    from fluxway.common.coherence import CoherenceSubject, ExecutionToken, GrantReady

    subject = CoherenceSubject(0, 0, 52)
    worker._coherence_directory = CoherenceDirectory()
    worker._coherence_directory.accept_parent_read(subject, 2)
    read = worker._coherence_directory.request_read(subject, 2, ExecutionToken(2))
    assert isinstance(read, GrantReady)
    from fluxway.spmd.worker.worker import _InputCoherenceGrant

    grant = _InputCoherenceGrant(
        parent_spmd_id="Input",
        parent_rank=0,
        subject=subject,
        route_code=52,
        epoch=2,
        local_access_token=read.token,
    )
    worker._coherence_input_grants[("lookup", 2)] = {grant.identity: grant}
    worker._coherence_input_bindings = {}
    worker._local_input_origins = {}
    worker._coherence_input_intents = {}
    worker._coherence_input_tensors = {}
    worker._coherence_directory = worker._coherence_directory
    worker._local_spmd_code = 1
    worker._allocation_registry = AllocationRegistry()
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tensor_kind",
    [TensorEntryKind.TENSOR, TensorEntryKind.ROUTING_MISS],
)
async def test_business_empty_tensor_reaches_operator_input(tensor_kind):
    worker = _worker(await _mailbox_with_kinds([tensor_kind] * 3))

    prepared = await worker._assemble_input("lookup", 2)

    tensor = prepared.value["value"]
    assert tensor.flat_values.shape == (0,)
    assert tensor.flat_values.dtype == np.float32
    assert tensor.global_shape == (768,)
    assert tensor.receipt is not None
    assert tensor.receipt.world_size == 3


@pytest.mark.asyncio
async def test_completed_without_tensor_bypasses_assembly():
    worker = _worker(
        await _mailbox_with_kinds([TensorEntryKind.COMPLETED_WITHOUT_TENSOR] * 3)
    )

    async def fail_assembly(*_args, **_kwargs):
        raise AssertionError("completed-without-tensor must bypass assembly")

    worker._assemble_tensor = fail_assembly
    result = await worker._assemble_input("lookup", 2)

    assert isinstance(result, ExecutionResult)
    assert result.status is ExecutionStatus.COMPLETED_WITHOUT_TENSOR


@pytest.mark.asyncio
async def test_routing_miss_from_productive_rank_prevents_false_completion():
    worker = _worker(
        await _mailbox_with_kinds(
            [
                TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
                TensorEntryKind.ROUTING_MISS,
                TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
            ]
        )
    )

    prepared = await worker._assemble_input("lookup", 2)

    assert prepared.value["value"].flat_values.shape == (0,)


@pytest.mark.asyncio
async def test_routing_miss_protocol_error_names_canonical_logical_tensor():
    worker = _worker(await _mailbox_with_kinds([TensorEntryKind.ROUTING_MISS]))

    with pytest.raises(ValueError, match="Input.emit:output"):
        worker._routing_miss_tensor(
            "lookup",
            52,
            2,
            {
                ("Input", 0): {
                    "expected_count": 1,
                    "tensor_kind": TensorEntryKind.ROUTING_MISS,
                    "global_shape": (768,),
                    "payload_schema": ({"dtype": np.dtype(np.float32)},),
                    "sender_world_size": 1,
                },
                ("Input", 1): {
                    "expected_count": 2,
                    "tensor_kind": TensorEntryKind.ROUTING_MISS,
                    "global_shape": (768,),
                    "payload_schema": ({"dtype": np.dtype(np.float32)},),
                    "sender_world_size": 2,
                },
            },
        )
