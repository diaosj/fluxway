import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_step_frame_groups,
    compile_send_plan,
)
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionStatus,
    InterleavedDispatcher,
)
from fluxway.spmd.worker.worker import SPMDWorker
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _compile_interleaved_compiled_graph(
    *,
    source_mesh: DeviceMesh,
    target_mesh: DeviceMesh,
    source_sharding: ShardingSpec,
    target_sharding: ShardingSpec,
    interleave_unit_size: int,
):
    loader = OperatorSpec(
        name="loader",
        is_source=True,
        is_interleaved=True,
        interleave_unit_size=interleave_unit_size,
        output_arguments={"data": ("t1",)},
        output_sharding={"t1": source_sharding},
    )
    processor = OperatorSpec(
        name="processor",
        input_arguments={"data": ("t1",)},
        input_sharding={"t1": target_sharding},
    )
    return compile_graph(
        name="interleaved-source",
        spmds=(
            compiled_spmd_spec("loader", source_mesh, {"loader": loader}),
            compiled_spmd_spec("processor", target_mesh, {"processor": processor}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("loader", "loader"),
                target=OperatorKey("processor", "processor"),
                transfers=(TensorTransfer(OperatorPort("t1"), OperatorPort("t1")),),
            ),
        ),
    )


def _build_interleaved_frames(
    *,
    compiled_graph,
    rank: int,
    interleave_unit_size: int,
    exec_step: int,
) -> None:
    source_plan = compiled_graph.spmd("loader")
    compiled_plan = compile_send_plan(
        graph_view=source_plan,
        source_op_id="loader",
        rank=rank,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=source_plan.outgoing,
                effective_rank=rank % interleave_unit_size,
            ),
        ),
        sender_spmd_code=1,
        interleave_unit_size=interleave_unit_size,
    )
    build_step_frame_groups(
        compiled_plan=compiled_plan,
        flat_results=[np.ones((10, 10))],
        step_transfer_cache={},
        exec_step=exec_step,
    )


def test_multi_worker_interleaved_unit_mesh_inheritance():
    source_mesh = DeviceMesh(
        axis_names=("data", "mp"),
        device_shape=(2, 2),
        device_ids=[0, 1, 2, 3],
    )
    target_mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    compiled_graph = _compile_interleaved_compiled_graph(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_sharding=ShardingSpec.chunk(PartitionSpec("data", "mp")),
        target_sharding=ShardingSpec.chunk(PartitionSpec()),
        interleave_unit_size=2,
    )

    for exec_step, ranks, expected_device_ids in (
        (0, (0, 1), [0, 1]),
        (1, (2, 3), [2, 3]),
    ):
        with patch(
            "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
        ) as distributor_cls:
            distributor = distributor_cls.return_value
            distributor.build_transport_frame_args.return_value = []
            for rank in ranks:
                _build_interleaved_frames(
                    compiled_graph=compiled_graph,
                    rank=rank,
                    interleave_unit_size=2,
                    exec_step=exec_step,
                )

        assert distributor.plan_and_shard.call_count == 2
        assert distributor.build_transport_frame_args.call_count == 2
        assert [
            call.kwargs["sender_rank"]
            for call in distributor.build_transport_frame_args.call_args_list
        ] == list(ranks)
        for call in distributor.plan_and_shard.call_args_list:
            source_spec = call.args[1]
            assert source_spec.mesh.shape == (1, 2)
            assert source_spec.mesh.axis_names == ("data", "mp")
            assert list(source_spec.mesh.device_ids.flatten()) == expected_device_ids


def test_single_worker_interleaved_unit():
    source_mesh = DeviceMesh(axis_names=("data",), device_shape=(4,))
    target_mesh = DeviceMesh(axis_names=("data",), device_shape=(1,))
    compiled_graph = _compile_interleaved_compiled_graph(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_sharding=ShardingSpec.chunk(PartitionSpec("data")),
        target_sharding=ShardingSpec.chunk(PartitionSpec("data")),
        interleave_unit_size=1,
    )

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        _build_interleaved_frames(
            compiled_graph=compiled_graph,
            rank=0,
            interleave_unit_size=1,
            exec_step=0,
        )

    distributor.plan_and_shard.assert_called_once()
    distributor.build_transport_frame_args.assert_called_once()
    assert distributor.build_transport_frame_args.call_args.kwargs["sender_rank"] == 0
    source_spec = distributor.plan_and_shard.call_args.args[1]
    assert source_spec.mesh.shape == (1,)
    assert list(source_spec.mesh.device_ids.flatten()) == [0]


def test_interleaved_dependency_uses_active_unit_size():
    compiled_graph = _compile_interleaved_compiled_graph(
        source_mesh=DeviceMesh(("data",), (4,)),
        target_mesh=DeviceMesh(("data",), (1,)),
        source_sharding=ShardingSpec.chunk(PartitionSpec("data")),
        target_sharding=ShardingSpec.chunk(PartitionSpec("data")),
        interleave_unit_size=2,
    )
    dependency = compiled_graph.dependencies[0]

    assert dependency.source_world_size == 4
    assert dependency.source_participating_count == 2


def test_interleaved_source_requires_complete_units():
    with pytest.raises(
        ValueError,
        match=(
            "interleaved source loader.loader mesh size 3 must be divisible by "
            "interleave_unit_size 2"
        ),
    ):
        _compile_interleaved_compiled_graph(
            source_mesh=DeviceMesh(("data",), (3,)),
            target_mesh=DeviceMesh(("data",), (1,)),
            source_sharding=ShardingSpec.chunk(PartitionSpec("data")),
            target_sharding=ShardingSpec.chunk(PartitionSpec("data")),
            interleave_unit_size=2,
        )


def test_interleaved_source_unit_contains_complete_trailing_mesh():
    with pytest.raises(
        ValueError,
        match=(
            "interleaved source loader.loader interleave_unit_size 1 must be "
            "divisible by trailing mesh size 2"
        ),
    ):
        _compile_interleaved_compiled_graph(
            source_mesh=DeviceMesh(("data", "model"), (2, 2)),
            target_mesh=DeviceMesh(("data",), (1,)),
            source_sharding=ShardingSpec.chunk(PartitionSpec("data", "model")),
            target_sharding=ShardingSpec.chunk(PartitionSpec("data")),
            interleave_unit_size=1,
        )


def test_interleaved_source_balances_complete_units_across_steps():
    dispatcher = InterleavedDispatcher(2)

    owners = [
        [rank for rank in range(6) if dispatcher.should_execute(rank, 6, step)]
        for step in range(9)
    ]

    assert owners == [
        [0, 1],
        [2, 3],
        [4, 5],
        [0, 1],
        [2, 3],
        [4, 5],
        [0, 1],
        [2, 3],
        [4, 5],
    ]


@pytest.mark.asyncio
async def test_inactive_interleaved_rank_does_not_send():
    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    worker.rank = 2
    worker._world_size = 4
    worker._completed_step_count = {"loader": 0}
    worker._mailbox = WorkerMailbox()
    worker._op_manager = SimpleNamespace(
        get_dispatcher=lambda _name: InterleavedDispatcher(2)
    )
    worker._drain_cond = asyncio.Condition()
    worker._drain_step_count = None
    worker._barriers_declared = {("loader", 0): True}
    worker._operator_send_context = lambda _name: (_ for _ in ()).throw(
        AssertionError("inactive rank must not access a send plan")
    )

    result = await worker_class._execute_operator_prepare_input(worker, "loader", 0)

    assert result.status is ExecutionStatus.INACTIVE
    assert not worker_class._result_requires_send(worker, result)
    assert (
        await worker_class._send_output(
            worker,
            "loader",
            result,
            exec_step=0,
            step_transfer_cache={},
        )
        == ()
    )
