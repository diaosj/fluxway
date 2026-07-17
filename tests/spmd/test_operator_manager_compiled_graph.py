from typing import Any

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
from fluxway.spmd.worker.assembly import compile_assemble_plan
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager
from tests.compiled_graph_fixtures import (
    compile_test_graph,
    spmd_config_from_graph,
)


_REPLICATED = ShardingSpec.chunk(PartitionSpec())
_PARTITIONED = ShardingSpec.chunk(PartitionSpec("data"))


class _Source:
    def produce(self) -> dict[str, Any]:
        return {"value": 1, "basis": 1, "mirror_value": 1}

    def before(self) -> None:
        return None


class _Target:
    def consume(self, mirror: Any, replica: Any, shard: Any) -> None:
        del mirror, replica, shard

    def after(self) -> None:
        return None


class _ReferentialSource:
    def upstream(self) -> dict[str, Any]:
        return {"ids": [1]}

    def lookup(self, unique_ids: Any) -> dict[str, Any]:
        return {"embedding": unique_ids}


class _ReferentialTarget:
    def consume(self, embedding: Any) -> None:
        del embedding


_SOURCE_OPERATORS = {
    "before": OperatorSpec(name="before", owner=_Source),
    "produce": OperatorSpec(
        name="produce",
        owner=_Source,
        is_source=True,
        output_arguments={"data": ("value", "basis", "mirror_value")},
        output_sharding={
            "value": _PARTITIONED,
            "basis": _PARTITIONED,
            "mirror_value": ShardingSpec.follow("produce.basis"),
        },
    ),
}
_TARGET_OPERATORS = {
    "after": OperatorSpec(name="after", owner=_Target),
    "consume": OperatorSpec(
        name="consume",
        owner=_Target,
        input_arguments={"data": ("mirror", "replica", "shard")},
        input_sharding={
            "mirror": ShardingSpec.follow("consume.replica"),
            "replica": _REPLICATED,
            "shard": _PARTITIONED,
        },
    ),
}


def _enabled(result: Any) -> bool:
    return result is not None


def _plan():
    tensor_dependency = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("target", "consume"),
        transfers=(
            TensorTransfer(
                OperatorPort("value"),
                OperatorPort("shard"),
            ),
            TensorTransfer(
                OperatorPort("value"),
                OperatorPort("replica"),
            ),
            TensorTransfer(
                OperatorPort("mirror_value"),
                OperatorPort("mirror"),
            ),
        ),
        condition=_enabled,
    )
    ordering_dependency = DependencySpec(
        source=OperatorKey("source", "before"),
        target=OperatorKey("target", "after"),
    )
    plan = compile_test_graph(
        name="operator-routes",
        spmds={
            "source": (DeviceMesh(("data",), (2,)), _SOURCE_OPERATORS),
            "target": (DeviceMesh(("data",), (2,)), _TARGET_OPERATORS),
        },
        dependencies=(tensor_dependency, ordering_dependency),
    )
    return plan, tensor_dependency, ordering_dependency


def _manager(plan, spmd_id: str) -> OperatorManager:
    return OperatorManager(
        spmd_config_from_graph(plan, spmd_id),
        rank=0,
        spmd_id=spmd_id,
        static_args={},
    )


def _cross_spmd_referential_plan(
    effective_sharding: ShardingSpec | None = None,
):
    dynamic = effective_sharding or ShardingSpec.hash(PartitionSpec("data"))
    source_operators = {
        "upstream": OperatorSpec(
            name="upstream",
            owner=_ReferentialSource,
            is_source=True,
            output_arguments={"data": ("ids",)},
            output_sharding={"ids": dynamic},
        ),
        "lookup": OperatorSpec(
            name="lookup",
            owner=_ReferentialSource,
            input_arguments={"data": ("unique_ids",)},
            output_arguments={"data": ("embedding",)},
            input_sharding={"unique_ids": dynamic},
            output_sharding={"embedding": ShardingSpec.follow("lookup.unique_ids")},
        ),
    }
    target_operators = {
        "consume": OperatorSpec(
            name="consume",
            owner=_ReferentialTarget,
            input_arguments={"data": ("embedding",)},
            input_sharding={"embedding": dynamic},
        )
    }
    plan = compile_test_graph(
        name="cross-spmd-referential-sharding",
        spmds={
            "source": (DeviceMesh(("data",), (2,)), source_operators),
            "target": (DeviceMesh(("data",), (2,)), target_operators),
        },
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "upstream"),
                target=OperatorKey("source", "lookup"),
                transfers=(
                    TensorTransfer(
                        OperatorPort("ids"),
                        OperatorPort("unique_ids"),
                    ),
                ),
                local_only=True,
            ),
            DependencySpec(
                source=OperatorKey("source", "lookup"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(
                        OperatorPort("embedding"),
                        OperatorPort("embedding"),
                    ),
                ),
            ),
        ),
    )
    return plan, dynamic


def test_operator_manager_uses_transfer_codes_as_exact_input_streams():
    plan, tensor_dependency, ordering_dependency = _plan()
    source = _manager(plan, "source")
    target = _manager(plan, "target")

    source_routes = source.get_successor_routes("produce")
    target_routes = target.get_predecessor_routes("consume")
    assert tuple(route.transfer.transfer_code for route in source_routes) == (1, 2, 3)
    assert tuple(route.transfer.transfer_code for route in target_routes) == (1, 2, 3)
    assert all(route.dependency.spec is tensor_dependency for route in source_routes)
    assert all(route.condition is _enabled for route in source_routes)
    assert all(
        route.transfer
        is plan.spmd(route.spmd_id).transfer(route.transfer.transfer_code)
        for route in (*source_routes, *target_routes)
    )
    assert source_routes[0].logical_tuuid == 3
    assert source_routes[1].logical_tuuid == 1
    assert source_routes[2].logical_tuuid == 1
    assert not hasattr(source_routes[0], "tuuid_map")
    assert not hasattr(source_routes[0], "tuuid_sharding")

    input_codes = target.get_input_transfer_codes("consume")
    assert input_codes == {
        "consume.mirror": 1,
        "consume.replica": 2,
        "consume.shard": 3,
    }
    assert source.get_input_transfer_codes("produce") == {}
    assert target.get_transfer_target_op(1) == "consume"
    assert target.get_transfer_target_op(2) == "consume"
    assert target.get_transfer_target_op(3) == "consume"
    assert target.get_expected_transfer_codes("consume") == [1, 2, 3]

    input_sharding = target.get_input_sharding("consume")
    assert input_sharding == {
        1: plan.spmd("target").transfer(1).target_sharding,
        2: _REPLICATED,
        3: _PARTITIONED,
    }
    assert target.get_tensor_input_sharding_spec(2) is _REPLICATED
    assert target.get_tensor_input_sharding_spec(3) is _PARTITIONED
    assert target.get_sender_world_size(1) == 2
    assert target.get_sender_world_size(2) == 2
    assert target.get_sender_world_size(3) == 2

    expected = target.get_expected_transfer_shards("consume")
    assert expected == {1: 2, 2: 2, 3: 1}
    assert target.resolve_tensor_producers("consume", 1, 0) == [0, 1]
    assert target.resolve_tensor_producers("consume", 2, 0) == [0, 1]
    assert target.resolve_tensor_producers("consume", 3, 0) == [0]

    tensor_compiled = next(
        item for item in plan.dependencies if item.spec is tensor_dependency
    )
    ordering_compiled = next(
        item for item in plan.dependencies if item.spec is ordering_dependency
    )
    assert source.get_outgoing_tensor_dependencies("produce") == (tensor_compiled,)
    assert tensor_compiled.spec is tensor_dependency
    assert source.get_outgoing_ordering_dependencies("before") == (ordering_compiled,)
    assert ordering_compiled.spec is ordering_dependency
    assert target.get_expected_ordering_counts("after") == {1: 2}

    with pytest.raises(ValueError, match="no incoming transfer code 99"):
        target.get_tensor_input_sharding_spec(99)
    with pytest.raises(ValueError, match="no target operator for transfer code 99"):
        target.get_transfer_target_op(99)


def test_spmd_graph_view_carries_canonical_referential_sharding_closure():
    plan, dynamic = _cross_spmd_referential_plan()
    transfer = next(
        transfer
        for dependency in plan.dependencies
        if dependency.spec.target.spmd_id == "target"
        for transfer in dependency.transfers
    )

    logical_tensor = plan.logical_tensor(transfer.logical_tuuid)
    reference_tuuid = logical_tensor.producer_sharding.routing.tuuid

    assert transfer.source_sharding.is_referential
    assert reference_tuuid == 2
    assert logical_tensor.producer == OperatorKey("source", "lookup")
    assert logical_tensor.producer_port == "embedding"
    assert logical_tensor.effective_sharding is dynamic
    assert plan.spmd("target").logical_tensor(transfer.logical_tuuid) is logical_tensor
    assert plan.spmd("target").logical_tensor(reference_tuuid) is plan.logical_tensor(2)
    assert (
        plan.spmd("target").operator("consume").effective_input_sharding["embedding"]
        is dynamic
    )
    assert not hasattr(transfer, "effective_source_sharding")
    assert not hasattr(transfer, "effective_target_sharding")


def test_operator_manager_consumes_compiled_referential_sharding_closure():
    plan, dynamic = _cross_spmd_referential_plan()

    target = _manager(plan, "target")
    transfer_code = target.get_expected_transfer_codes("consume")[0]

    assert target.get_transfer_source_layout("consume", transfer_code).spec is dynamic
    assert target.get_transfer_target_layout("consume", transfer_code).spec is dynamic
    assert target.resolve_tensor_producers("consume", transfer_code, 0) == [0, 1]


def test_referential_transfer_barrier_keeps_all_source_participants():
    replicated = ShardingSpec.chunk(PartitionSpec())
    plan, _ = _cross_spmd_referential_plan(replicated)

    target = _manager(plan, "target")
    transfer_code = target.get_expected_transfer_codes("consume")[0]

    assert target.resolve_tensor_producers("consume", transfer_code, 0) == [0]
    assert target.get_expected_transfer_shards("consume") == {transfer_code: 2}


def test_assemble_plan_is_exactly_bound_to_one_transfer():
    plan, _, _ = _plan()
    target = _manager(plan, "target")

    mirror_plan = compile_assemble_plan(
        op_manager=target,
        op_name="consume",
        transfer_code=1,
    )
    replica_plan = compile_assemble_plan(
        op_manager=target,
        op_name="consume",
        transfer_code=2,
    )
    shard_plan = compile_assemble_plan(
        op_manager=target,
        op_name="consume",
        transfer_code=3,
    )

    tensor_dependency = next(item for item in plan.dependencies if item.transfers)
    assert mirror_plan.dependency is tensor_dependency
    assert mirror_plan.transfer is target._plan.transfer(1)
    assert mirror_plan.source_spmd_id == "source"
    assert mirror_plan.source_layout.spec is _PARTITIONED
    assert mirror_plan.target_layout.spec is _REPLICATED
    assert replica_plan.source_layout.spec is _PARTITIONED
    assert replica_plan.target_layout.spec is _REPLICATED
    assert shard_plan.source_layout.spec is _PARTITIONED
    assert shard_plan.target_layout.spec is _PARTITIONED
    assert not hasattr(mirror_plan, "tuuid")
    assert not hasattr(mirror_plan, "predecessor_topology")

    shards = {
        ("source", 0): ["rank-0"],
        ("source", 1): ["rank-1"],
        ("other", 0): ["other"],
    }

    def flatten(values, _):
        return [item for rank_values in values.values() for item in rank_values]

    assert replica_plan.filter_shards(
        op_manager=target,
        worker_rank=0,
        step=0,
        shards_dict=shards,
        flatten_shards=flatten,
    ) == ["rank-0", "rank-1"]
    assert shard_plan.filter_shards(
        op_manager=target,
        worker_rank=0,
        step=0,
        shards_dict=shards,
        flatten_shards=flatten,
    ) == ["rank-0"]
