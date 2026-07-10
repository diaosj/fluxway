import types
from typing import Any

import numpy as np
import pytest

from fluxway.common.decorators import operator, spmd
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.control_plane.graph.builder import Builder
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.rpc.protocol import LocalWorkerDataPlane, RayActorDataPlane
from fluxway.spmd.worker.data_plane import CompiledSendPlan, CompiledSuccessorRoute
from fluxway.spmd.worker.mailbox import WorkerMailbox
from fluxway.spmd.worker.data_plane_executor import _validate_local_only_batch
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.transport.communication import TransportFrameBatch

_LOCAL_SPEC = {"type": "static", "axis": []}


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class _LocalOnlyTransportSPMD:
    @operator(name="produce", output_sharding={"x": _LOCAL_SPEC})
    def produce(self) -> dict:
        return {"x": 1}

    @operator(name="consume", input_sharding={"x": _LOCAL_SPEC})
    def consume(self, x: Any):
        return x


async def _receive_transport_frames(_frames):
    return ()


async def _receive_control(_op_name, _step, _names, _values):
    return None


def _local_only_config():
    builder = Builder()
    builder.add_node("A", _LocalOnlyTransportSPMD)
    builder.add_func_edge(
        "A",
        "produce",
        "A",
        "consume",
        source_output="x",
        target_input="x",
        local_only=True,
    )
    return builder.build("local_only_transport").nodes["A"].spmd_config


def test_local_only_successor_uses_direct_worker_data_plane():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(config, rank=0, spmd_id="A")
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=0,
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=ProfilerRuntimeFactory.from_config(
            None,
            process_name="worker/test/rank:0",
        ),
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code,
        stub,
    )
    stub._local_only_successor_handlers = types.MethodType(
        raw_cls._local_only_successor_handlers,
        stub,
    )
    stub._operator_send_context = types.MethodType(
        raw_cls._operator_send_context,
        stub,
    )

    send_context = stub._operator_send_context("produce")

    handler = send_context.successor_handlers[0][0]
    assert isinstance(handler, LocalWorkerDataPlane)
    assert not isinstance(handler, RayActorDataPlane)
    assert send_context.compiled_plan.successor_routes[0].local_only is True


def test_local_only_batch_rejects_non_local_rank():
    route = CompiledSuccessorRoute(
        context_index=0,
        spmd_id="A",
        effective_rank=0,
        tensor_routes=(),
        local_only=True,
    )
    plan = CompiledSendPlan(
        op_name="produce",
        rank=0,
        sender_spmd_code=3,
        sender_world_size=2,
        successor_routes=(route,),
        output_count=1,
        tuuid_to_name=None,
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=1,
        batch_seq=0,
        frames=(),
    )

    with pytest.raises(ValueError, match="same-rank routing"):
        _validate_local_only_batch(route, plan, batch)


@pytest.mark.asyncio
async def test_local_only_direct_deposit_preserves_transport_tensor_receipt():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(config, rank=1, spmd_id="A")
    mailbox = WorkerMailbox()
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=1,
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=ProfilerRuntimeFactory.from_config(
            None,
            process_name="worker/test/rank:1",
        ),
        _mailbox=mailbox,
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code,
        stub,
    )
    stub._local_only_successor_handlers = types.MethodType(
        raw_cls._local_only_successor_handlers,
        stub,
    )
    stub._operator_send_context = types.MethodType(
        raw_cls._operator_send_context,
        stub,
    )
    stub._ensure_step_declared = types.MethodType(
        raw_cls._ensure_step_declared,
        stub,
    )
    stub._deposit_local_only_outputs = types.MethodType(
        raw_cls._deposit_local_only_outputs,
        stub,
    )

    send_context = stub._operator_send_context("produce")
    original = TransportTensor(
        np.asarray([1, 2, 3], dtype=np.int32),
        global_shape=(3,),
        receipt=SymmetryReceipt(7, 4),
    )
    await stub._deposit_local_only_outputs(
        [original],
        send_context,
        {},
        0,
    )

    tuuid = op_manager.get_expected_tuuids("consume")[0]
    shards = await mailbox.get_tensor_shard_views("consume", tuuid, 0)
    stored = shards[("A", 1)][0]
    assert isinstance(stored, TransportTensor)
    assert stored.receipt is not None
    assert stored.receipt.world_size == 7
    assert int(stored.receipt.sender_rank_tags) == 4


@pytest.mark.asyncio
async def test_send_output_all_local_only_returns_no_background_tasks_with_profiler():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(config, rank=1, spmd_id="A")
    mailbox = WorkerMailbox()
    profiler_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(
            trace_dir="/tmp/fluxway_test_send_output_all_local_only",
            run_id="profiler_enabled",
            buffer_bytes_per_process=1024 * 1024,
        ),
        process_name="worker/test/rank:1",
    )
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=1,
        config=config,
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=profiler_runtime,
        _mailbox=profiler_runtime.worker_mailbox(mailbox),
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        _data_plane_batch_max_frames=0,
        _data_plane_batch_flush_us=0,
        _data_plane_executor=None,
        _delivery_runtime=types.SimpleNamespace(submit=None),
        _driver_handle=None,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
    )
    for name in (
        "_require_local_spmd_code",
        "_local_only_successor_handlers",
        "_operator_send_context",
        "_send_output",
        "_send_step_data_plane_frames_background",
        "_deposit_local_only_outputs",
        "_non_local_send_plan",
        "_ensure_step_declared",
    ):
        setattr(stub, name, types.MethodType(getattr(raw_cls, name), stub))

    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        data={"x": TransportTensor.from_tensor(np.asarray([1, 2, 3], dtype=np.int32))},
    )

    send_tasks = await stub._send_output(
        "produce",
        result,
        exec_step=0,
        step_tuuid_cache={},
    )

    tuuid = op_manager.get_expected_tuuids("consume")[0]
    shards = await mailbox.get_tensor_shard_views("consume", tuuid, 0)
    assert send_tasks == ()
    assert isinstance(shards[("A", 1)][0], TransportTensor)
