import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import (
    SPMDConfig,
    OperatorMeta,
    EdgeShardingInfo,
    ArgumentType,
    SPMDTopology,
    SPMDTopologyInfo,
)
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec
from fluxway.common.mesh import DeviceMesh
from fluxway.common.decorators import spmd, operator as fw_operator
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_step_frame_groups,
    compile_send_plan,
)


def _build_frame_groups_with_plan(
    *,
    op_name,
    rank,
    mesh,
    flat_results,
    output_uuids,
    output_sharding,
    successor_contexts,
    tuuid_to_name,
    step_tuuid_cache,
    exec_step,
    sender_spmd_code,
    sender_world_size,
    interleave_unit_size,
):
    compiled_plan = compile_send_plan(
        op_name=op_name,
        rank=rank,
        mesh=mesh,
        output_uuids=tuple(output_uuids),
        output_sharding=output_sharding,
        successor_contexts=tuple(successor_contexts),
        tuuid_to_name=tuuid_to_name,
        sender_spmd_code=sender_spmd_code,
        sender_world_size=sender_world_size,
        interleave_unit_size=interleave_unit_size,
    )
    return build_step_frame_groups(
        compiled_plan=compiled_plan,
        flat_results=flat_results,
        step_tuuid_cache=step_tuuid_cache,
        exec_step=exec_step,
    )


@spmd(mesh_axis_names=("data", "mp"), mesh_axis_shapes=(2, 2))
class _LoaderOp:
    @fw_operator(name="loader", is_source=True)
    def loader(self):
        return {"t1": np.ones((10, 10))}


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(2,))
class _ProcessorOp:
    @fw_operator(name="processor")
    def processor(self, t1: np.ndarray | None = None):
        pass


def _meta_to_topology(meta: OperatorMeta) -> dict[str, SPMDTopology]:
    """Helper for unit tests to derive topology from OperatorMeta."""
    top_dict = {}
    top = SPMDTopology()
    for pred_id, edges in meta.input_pre_sharding.items():
        if edges:
            e = edges[0]
            top.predecessors[pred_id] = SPMDTopologyInfo(
                spmd_id=pred_id,
                tuuid_sharding=e.tuuid_sharding,
                tuuid_map=e.tuuid_map,
                is_interleaved=e.is_interleaved,
                interleave_unit_size=e.interleave_unit_size,
                mesh=e.mesh,
                world_size=e.mesh.size if getattr(e, "mesh", None) else 0,
            )
    for succ_id, edges in meta.output_post_sharding.items():
        if edges:
            e = edges[0]
            top.successors[succ_id] = SPMDTopologyInfo(
                spmd_id=succ_id,
                tuuid_sharding=e.tuuid_sharding,
                tuuid_map=e.tuuid_map,
                is_interleaved=e.is_interleaved,
                interleave_unit_size=e.interleave_unit_size,
                mesh=e.mesh,
                world_size=e.mesh.size if getattr(e, "mesh", None) else 0,
            )
    top_dict[meta.name] = top
    return top_dict


@pytest.mark.asyncio
async def test_multi_worker_interleaved_unit_mesh_inheritance():
    """
    Verify that interleaved units correctly inherit the sub-mesh structure.
    Loader (4 ranks, mesh (2,2)) -> Processor (2 ranks)
    """
    loader_mesh = DeviceMesh(
        axis_names=("data", "mp"), device_shape=(2, 2), device_ids=[0, 1, 2, 3]
    )
    processor_mesh = DeviceMesh(
        axis_names=("data",), device_shape=(2,), device_ids=[0, 1]
    )

    sharding = ShardingSpec.chunk(PartitionSpec("data", "mp"))
    edge_info = EdgeShardingInfo(
        is_interleaved=True,
        interleave_unit_size=2,
        tuuid_sharding={1001: sharding},
        tuuid_map={"t1": 1001},
        mesh=loader_mesh,
    )

    loader_meta = OperatorMeta(
        name="loader",
        is_source=True,
        is_interleaved=True,
        interleave_unit_size=2,
        output_arguments={ArgumentType.DATA.value: ["t1"]},
        output_uuids=[1001],
        output_sharding={1001: sharding},
        output_post_sharding={"processor": [edge_info]},
        owner_class_module=_LoaderOp.__module__,
        owner_class_qualname=_LoaderOp.__qualname__,
    )

    processor_meta = OperatorMeta(
        name="processor",
        input_arguments={ArgumentType.DATA.value: ["t1"]},
        input_sharding={"t1": ShardingSpec.chunk(PartitionSpec())},
        input_pre_sharding={"loader": [edge_info]},
        owner_class_module=_ProcessorOp.__module__,
        owner_class_qualname=_ProcessorOp.__qualname__,
    )

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    # Instantiate Loader Workers
    loader_workers = []
    for rank in range(4):
        config = SPMDConfig(
            mesh=loader_mesh,
            spmd_id="loader",
            operators={"loader": "loader"},
            operator_metas={"loader": loader_meta, "processor": processor_meta},
            topology=_meta_to_topology(loader_meta),
            active_operator_names=("loader", "loader", "processor"),
        )
        w = cls(config, rank=rank)
        w.setup_and_start(local_rank=rank, local_world_size=4, node_rank=0, num_nodes=1)
        w.register_spmd_code_map(1, {1: "loader", 2: "processor"})
        w._successor_handlers = {"processor": {0: AsyncMock(), 1: AsyncMock()}}
        loader_workers.append(w)

    # Instantiate Processor Workers (for shard count check)
    processor_workers = []
    for rank in range(2):
        config = SPMDConfig(
            mesh=processor_mesh,
            spmd_id="processor",
            operators={"processor": "processor"},
            operator_metas={"loader": loader_meta, "processor": processor_meta},
            topology=_meta_to_topology(processor_meta),
            active_operator_names=("processor", "loader", "processor"),
        )
        w = cls(config, rank=rank)
        w.setup_and_start(local_rank=rank, local_world_size=2, node_rank=0, num_nodes=1)
        processor_workers.append(w)

    assert processor_workers[0]._op_manager is not None
    shards_dict = processor_workers[0]._op_manager.get_expected_tensor_shards(
        "processor"
    )
    assert shards_dict[1001] == 2, (
        f"Target processor should expect 2 shards, got {shards_dict}"
    )

    # 2. BUILD STEP 0: Unit 0 (Ranks 0, 1) active
    with patch("fluxway.spmd.worker.data_plane.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.build_transport_frame_args.return_value = []

        for w in loader_workers[:2]:
            succ_info = w.op_manager.get_successor_info("loader", "processor")
            assert succ_info is not None
            _build_frame_groups_with_plan(
                op_name="loader",
                rank=w.rank,
                mesh=w._mesh,
                flat_results=[np.ones((10, 10))],
                output_uuids=[1001],
                output_sharding={1001: sharding},
                successor_contexts=[
                    SuccessorRouteContext(
                        context_index=0,
                        spmd_id="processor",
                        info=succ_info,
                        effective_rank=w.rank % edge_info.interleave_unit_size,
                    )
                ],
                tuuid_to_name={1001: "t1"},
                step_tuuid_cache={},
                exec_step=0,
                sender_spmd_code=1,
                sender_world_size=4,
                interleave_unit_size=loader_meta.interleave_unit_size,
            )

        assert mock_dist.plan_and_shard.call_count == 2
        assert mock_dist.build_transport_frame_args.call_count == 2
        assert [
            call.kwargs["sender_rank"]
            for call in mock_dist.build_transport_frame_args.call_args_list
        ] == [0, 1]
        found_u0 = False
        for call in mock_dist.plan_and_shard.call_args_list:
            source_spec = call.args[1]
            if list(source_spec.mesh.device_ids.flatten()) == [0, 1]:
                assert source_spec.mesh.shape == (1, 2)
                assert source_spec.mesh.axis_names == ("data", "mp")
                found_u0 = True
        assert found_u0, "Unit 0 sub-mesh distribution not found"

    # 3. BUILD STEP 1: Unit 1 (Ranks 2, 3) active
    with patch("fluxway.spmd.worker.data_plane.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.build_transport_frame_args.return_value = []

        for w in loader_workers[2:]:
            succ_info = w.op_manager.get_successor_info("loader", "processor")
            assert succ_info is not None
            _build_frame_groups_with_plan(
                op_name="loader",
                rank=w.rank,
                mesh=w._mesh,
                flat_results=[np.ones((10, 10))],
                output_uuids=[1001],
                output_sharding={1001: sharding},
                successor_contexts=[
                    SuccessorRouteContext(
                        context_index=0,
                        spmd_id="processor",
                        info=succ_info,
                        effective_rank=w.rank % edge_info.interleave_unit_size,
                    )
                ],
                tuuid_to_name={1001: "t1"},
                step_tuuid_cache={},
                exec_step=1,
                sender_spmd_code=1,
                sender_world_size=4,
                interleave_unit_size=loader_meta.interleave_unit_size,
            )

        assert mock_dist.plan_and_shard.call_count == 2
        assert mock_dist.build_transport_frame_args.call_count == 2
        assert [
            call.kwargs["sender_rank"]
            for call in mock_dist.build_transport_frame_args.call_args_list
        ] == [2, 3]
        found_u1 = False
        for call in mock_dist.plan_and_shard.call_args_list:
            source_spec = call.args[1]
            if list(source_spec.mesh.device_ids.flatten()) == [2, 3]:
                found_u1 = True
        assert found_u1, "Unit 1 sub-mesh distribution not found"


@pytest.mark.asyncio
async def test_single_worker_interleaved_unit():
    loader_mesh = DeviceMesh(axis_names=("data",), device_shape=(4,))
    sharding = ShardingSpec.chunk(PartitionSpec("data"))
    edge = EdgeShardingInfo(
        is_interleaved=True,
        interleave_unit_size=4,
        tuuid_sharding={2001: sharding},
        tuuid_map={"t1": 2001},
        mesh=loader_mesh,
    )
    loader_meta = OperatorMeta(
        name="loader",
        is_source=True,
        is_interleaved=True,
        interleave_unit_size=1,
        output_arguments={ArgumentType.DATA.value: ["t1"]},
        output_uuids=[2001],
        output_sharding={2001: sharding},
        output_post_sharding={"processor": [edge]},
        owner_class_module=_LoaderOp.__module__,
        owner_class_qualname=_LoaderOp.__qualname__,
    )

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    loader_workers = []
    for rank in range(4):
        config = SPMDConfig(
            mesh=loader_mesh,
            spmd_id="loader",
            operators={"loader": "loader"},
            operator_metas={"loader": loader_meta},
            topology=_meta_to_topology(loader_meta),
            active_operator_names=("loader", "loader"),
        )
        w = cls(config, rank=rank)
        w.setup_and_start(local_rank=rank, local_world_size=4, node_rank=0, num_nodes=1)
        w.register_spmd_code_map(1, {1: "loader", 2: "processor"})
        w._successor_handlers = {"processor": {0: AsyncMock()}}
        loader_workers.append(w)

    with patch("fluxway.spmd.worker.data_plane.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.build_transport_frame_args.return_value = []

        w = loader_workers[0]
        succ_info = w.op_manager.get_successor_info("loader", "processor")
        assert succ_info is not None
        _build_frame_groups_with_plan(
            op_name="loader",
            rank=w.rank,
            mesh=w._mesh,
            flat_results=[np.ones((10, 10))],
            output_uuids=[2001],
            output_sharding={2001: sharding},
            successor_contexts=[
                SuccessorRouteContext(
                    context_index=0,
                    spmd_id="processor",
                    info=succ_info,
                    effective_rank=w.rank % edge.interleave_unit_size,
                )
            ],
            tuuid_to_name={2001: "t1"},
            step_tuuid_cache={},
            exec_step=0,
            sender_spmd_code=1,
            sender_world_size=4,
            interleave_unit_size=loader_meta.interleave_unit_size,
        )
        assert mock_dist.plan_and_shard.call_count == 1
        assert mock_dist.build_transport_frame_args.call_count == 1
        assert mock_dist.build_transport_frame_args.call_args.kwargs["sender_rank"] == 0

        source_spec = mock_dist.plan_and_shard.call_args.args[1]
        assert source_spec.mesh.shape == (1,)
        assert list(source_spec.mesh.device_ids.flatten()) == [0]
