from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.rpc.protocol import TransportFrameEnvelope
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.manifest import (
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)
from tests.compiled_graph_fixtures import (
    compile_test_graph,
    spmd_config_from_graph,
)


class _IntegrationOperators:
    def produce_one(self) -> dict[str, Any]:
        return {"x": np.asarray([1.0])}

    def produce_two(self) -> dict[str, Any]:
        return {"x": np.asarray([1.0]), "y": np.asarray([2.0])}

    def test_op(self, x: Any) -> Any:
        return x

    def compute(self, x: Any, y: Any) -> tuple[Any, Any]:
        return x, y

    def control_target(self) -> None:
        return None


def _worker_class():
    metadata = getattr(SPMDWorker, "__ray_metadata__", None)
    return metadata.modified_class if metadata else SPMDWorker


def _single_transfer_plan(mesh: DeviceMesh):
    sharding = ShardingSpec.chunk(PartitionSpec())
    return compile_test_graph(
        name="worker-single-transfer",
        spmds={
            "source": (
                mesh,
                {
                    "produce_one": OperatorSpec(
                        name="produce_one",
                        owner=_IntegrationOperators,
                        is_source=True,
                        output_arguments={"data": ("x",)},
                        output_sharding={"x": sharding},
                    )
                },
            ),
            "target": (
                mesh,
                {
                    "test_op": OperatorSpec(
                        name="test_op",
                        owner=_IntegrationOperators,
                        input_arguments={"data": ("x",)},
                        input_sharding={"x": sharding},
                    )
                },
            ),
        },
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce_one"),
                target=OperatorKey("target", "test_op"),
                transfers=(TensorTransfer(OperatorPort("x"), OperatorPort("x")),),
            ),
        ),
    )


def _bundled_transfer_plan(mesh: DeviceMesh):
    sharding = ShardingSpec.chunk(PartitionSpec("data"))
    return compile_test_graph(
        name="worker-bundled-transfer",
        spmds={
            "source": (
                mesh,
                {
                    "produce_two": OperatorSpec(
                        name="produce_two",
                        owner=_IntegrationOperators,
                        is_source=True,
                        output_arguments={"data": ("x", "y")},
                        output_sharding={"x": sharding, "y": sharding},
                    )
                },
            ),
            "target": (
                mesh,
                {
                    "compute": OperatorSpec(
                        name="compute",
                        owner=_IntegrationOperators,
                        input_arguments={"data": ("x", "y")},
                        input_sharding={"x": sharding, "y": sharding},
                    )
                },
            ),
        },
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce_two"),
                target=OperatorKey("target", "compute"),
                transfers=(
                    TensorTransfer(OperatorPort("x"), OperatorPort("x")),
                    TensorTransfer(OperatorPort("y"), OperatorPort("y")),
                ),
            ),
        ),
    )


def _control_config(mesh: DeviceMesh):
    plan = compile_test_graph(
        name="worker-control",
        spmds={
            "target": (
                mesh,
                {
                    "control_target": OperatorSpec(
                        name="control_target",
                        owner=_IntegrationOperators,
                    )
                },
            )
        },
        dependencies=(),
    )
    return spmd_config_from_graph(plan, "target")


@pytest.mark.asyncio
async def test_worker_receive_transport_frame_dispatches_to_mailbox():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(1,))
    plan = _single_transfer_plan(mesh)
    config = spmd_config_from_graph(plan, "target")
    worker = _worker_class()(config, rank=0, spmd_id="target")
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    worker.register_spmd_code_map(1, {1: "source"})
    transfer_code = plan.dependencies[0].transfers[0].transfer_code

    test_data = np.array([1.0, 2.0])
    builder = TransportFrameBuilder(
        step=0,
        sender_spmd_code=1,
        sender_rank=0,
        sender_world_size=1,
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=transfer_code,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=0,
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, test_data, 2)
    frame = builder.build()

    await worker.receive_transport_frames(
        (
            TransportFrameEnvelope(
                header=frame.header,
                payload_segments=frame.payload_segments,
            ),
        )
    )

    stored = await worker._mailbox.get_tensor_shard_views(
        "test_op",
        transfer_code,
        0,
    )
    assert ("source", 0) in stored
    np.testing.assert_array_equal(stored[("source", 0)][0].flat_values, test_data)


@pytest.mark.asyncio
async def test_worker_receive_bundled_frame_dispatches_all_transfers_to_mailbox():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    plan = _bundled_transfer_plan(mesh)
    config = spmd_config_from_graph(plan, "target")
    worker = _worker_class()(config, rank=0, spmd_id="target")
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    worker.register_spmd_code_map(3, {3: "source"})
    input_codes = worker.op_manager.get_input_transfer_codes("compute")

    first = np.array([1, 2])
    second = np.array([3, 4])
    builder = TransportFrameBuilder(
        step=1,
        sender_spmd_code=3,
        sender_rank=0,
        sender_world_size=2,
    )
    for transfer_code, payload in (
        (input_codes["compute.x"], first),
        (input_codes["compute.y"], second),
    ):
        entry_index = builder.add_tensor_transfer_entry(
            transfer_code=transfer_code,
            meta_state=0,
            tensor_kind=TensorEntryKind.TENSOR,
            expected_count=1,
            source_step=1,
        )
        builder.add_payload(entry_index, PayloadRole.VALUES, payload, len(payload))
    frame = builder.build()

    await worker.receive_transport_frames(
        (
            TransportFrameEnvelope(
                header=frame.header,
                payload_segments=frame.payload_segments,
            ),
        )
    )

    first_stored = await worker._mailbox.get_tensor_shard_views(
        "compute",
        input_codes["compute.x"],
        1,
    )
    second_stored = await worker._mailbox.get_tensor_shard_views(
        "compute",
        input_codes["compute.y"],
        1,
    )
    np.testing.assert_array_equal(first_stored[("source", 0)][0].flat_values, first)
    np.testing.assert_array_equal(second_stored[("source", 0)][0].flat_values, second)


@pytest.mark.asyncio
async def test_worker_receive_control_dispatches_to_mailbox():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(1,))
    config = _control_config(mesh)
    worker = _worker_class()(config, rank=0, spmd_id="target")
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    worker._mailbox.deposit_control_arguments = AsyncMock()
    worker._ensure_step_declared = lambda *args, **kwargs: None

    await worker.receive_control(
        op_name="control_target",
        step=3,
        names=["lr"],
        values=[0.01],
    )

    worker._mailbox.deposit_control_arguments.assert_awaited_once_with(
        "control_target",
        step=3,
        names=["lr"],
        values=[0.01],
    )
