import sys
import os
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.getcwd())

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import (
    ArgumentType,
    OperatorMeta,
    SPMDConfig,
    SPMDTopology,
    SPMDTopologyInfo,
)
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.manifest import EmptyKind, PayloadRole, TransportFrameBuilder


def _make_test_config(mesh: DeviceMesh, spmd_id: str = "") -> SPMDConfig:
    """Create a minimal SPMDConfig with a no-op 'test_op' operator registered."""
    meta = OperatorMeta(name="test_op", input_arguments=None, output_arguments=None)

    def dummy_fn():
        pass

    return SPMDConfig(
        mesh=mesh,
        spmd_id=spmd_id,
        operators={"test_op": "test_op"},
        operator_metas={"test_op": meta},
    )


@pytest.fixture
def mock_mesh():
    return DeviceMesh(axis_names=("data",), device_shape=(1,))


@pytest.fixture
def worker_config(mock_mesh):
    return _make_test_config(mock_mesh, spmd_id="test_spmd")


@pytest.mark.asyncio
async def test_worker_receive_transport_frame_dispatches_to_mailbox(worker_config):
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0, spmd_id="test_spmd")
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    worker.register_spmd_code_map(1, {1: "src_spmd"})

    worker._op_manager.get_operators_for_tuuid = MagicMock(return_value=["test_op"])
    worker._op_manager.get_expected_tensor_shards = MagicMock(return_value={42: 1})
    worker._op_manager.get_expected_control_arg_names = MagicMock(return_value=[])
    worker._op_manager.get_operator_metadata = MagicMock(return_value=None)

    test_data = np.array([1.0, 2.0])
    builder = TransportFrameBuilder(
        step=0, sender_spmd_code=1, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=42,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )
    builder.add_payload(edge, PayloadRole.VALUES, test_data, 2)
    frame = builder.build()
    await worker.receive_transport_frame(frame.header, *frame.payload_segments)

    stored = await worker._mailbox.get_tensor_shard_views("test_op", 42, 0)
    assert ("src_spmd", 0) in stored
    np.testing.assert_array_equal(stored[("src_spmd", 0)][0].flat_values, test_data)


@pytest.mark.asyncio
async def test_worker_receive_bundled_frame_dispatches_all_edges_to_mailbox():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    input_spec = ShardingSpec.chunk(PartitionSpec("data"))
    meta = OperatorMeta(
        name="compute",
        input_arguments={ArgumentType.DATA.value: ["x", "y"]},
        input_sharding={"x": input_spec, "y": input_spec},
        output_arguments=None,
    )
    config = SPMDConfig(
        mesh=mesh,
        operators={"compute": "compute"},
        operator_metas={"compute": meta},
        topology={
            "compute": SPMDTopology(
                predecessors={
                    "s0": SPMDTopologyInfo(
                        spmd_id="s0",
                        world_size=2,
                        mesh=mesh,
                        tuuid_sharding={998: input_spec, 999: input_spec},
                        tuuid_map={"x": 998, "y": 999},
                    )
                }
            )
        },
    )
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    worker.register_spmd_code_map(3, {3: "s0"})

    first = np.array([1, 2])
    second = np.array([3, 4])
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=3, sender_rank=0, sender_world_size=2
    )
    first_edge = builder.add_edge(
        tuuid=998,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )
    builder.add_payload(first_edge, PayloadRole.VALUES, first, 2)
    second_edge = builder.add_edge(
        tuuid=999,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )
    builder.add_payload(second_edge, PayloadRole.VALUES, second, 2)
    frame = builder.build()

    await worker.receive_transport_frame(frame.header, *frame.payload_segments)

    first_stored = await worker._mailbox.get_tensor_shard_views("compute", 998, 1)
    second_stored = await worker._mailbox.get_tensor_shard_views("compute", 999, 1)
    np.testing.assert_array_equal(first_stored[("s0", 0)][0].flat_values, first)
    np.testing.assert_array_equal(second_stored[("s0", 0)][0].flat_values, second)


@pytest.mark.asyncio
async def test_worker_receive_control_dispatches_to_mailbox(worker_config):
    """Verify that receive_control correctly calls deposit_control_arguments on the mailbox."""
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0, spmd_id="test_spmd")
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)

    worker._mailbox.deposit_control_arguments = AsyncMock()
    worker._ensure_step_declared = lambda *a, **kw: None

    await worker.receive_control(op_name="test_op", step=3, names=["lr"], values=[0.01])

    worker._mailbox.deposit_control_arguments.assert_called_once()
    _, kwargs = worker._mailbox.deposit_control_arguments.call_args
    assert kwargs.get("names") == [
        "lr"
    ] or worker._mailbox.deposit_control_arguments.call_args[0][2:4] == (["lr"], [0.01])
