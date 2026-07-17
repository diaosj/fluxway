from unittest.mock import patch

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    CompiledDependency,
    DependencySpec,
    GraphCompiler,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.sharding import P, PartitionSpec, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_completed_without_tensor_step_frame_groups,
    build_step_frame_groups,
    compile_send_plan,
    prepare_route_tensor,
)
from fluxway.transport.communication import TransportRoutingMode
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _dependency(
    source_op: str,
    target_op: str,
    source_port: str,
    target_port: str,
) -> DependencySpec:
    return DependencySpec(
        source=OperatorKey("source", source_op),
        target=OperatorKey("target", target_op),
        transfers=(
            TensorTransfer(
                source=OperatorPort(source_port),
                target=OperatorPort(target_port),
            ),
        ),
    )


def test_sender_plan_uses_append_only_graph_view_generation():
    replicated = ShardingSpec.chunk(PartitionSpec())
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("value",)},
        input_sharding={"value": replicated},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (1,))
    compiler = GraphCompiler(graph_id="eager-append-test")
    initial = compiler.compile(
        name="eager-append",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": produce}),
            compiled_spmd_spec("target", target_mesh, {}),
        ),
        dependencies=(),
    )
    dependency = _dependency("produce", "consume", "value", "value")
    extended = compiler.compile(
        name="eager-append",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": produce}),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(dependency,),
    )
    source_view = extended.spmd("source")
    send_plan = compile_send_plan(
        graph_view=source_view,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_view.outgoing, effective_rank=0),
        ),
        sender_spmd_code=1,
        interleave_unit_size=None,
    )

    assert initial.generation == 0
    assert extended.generation == 1
    assert initial.spmd("source").operator("produce").output_uuids == (1,)
    assert source_view.operator("produce").output_uuids == (1,)
    assert send_plan.successor_routes[0].tensor_routes[0].logical_tuuid == 1


def _compile_fanout_plan(
    *,
    second_target_sharding: ShardingSpec,
):
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    target_a = OperatorSpec(
        name="consume_a",
        input_arguments={"data": ("left",)},
        input_sharding={"left": replicated},
    )
    target_b = OperatorSpec(
        name="consume_b",
        input_arguments={"data": ("right",)},
        input_sharding={"right": second_target_sharding},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (1,))
    plan = compile_graph(
        name="sender-transfer-routes",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": source}),
            compiled_spmd_spec(
                "target",
                target_mesh,
                {"consume_a": target_a, "consume_b": target_b},
            ),
        ),
        dependencies=(
            _dependency("produce", "consume_a", "value", "left"),
            _dependency("produce", "consume_b", "value", "right"),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=source_plan.outgoing,
                effective_rank=0,
            ),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    return compiled, source_plan.outgoing


def test_compile_send_plan_preserves_each_canonical_transfer_route():
    partitioned = ShardingSpec.chunk(PartitionSpec("data"))
    compiled, dependencies = _compile_fanout_plan(second_target_sharding=partitioned)

    routes = compiled.successor_routes[0].tensor_routes
    expected_transfers = tuple(
        transfer for dependency in dependencies for transfer in dependency.transfers
    )
    assert [route.transfer_code for route in routes] == [
        transfer.transfer_code for transfer in expected_transfers
    ]
    assert [route.dependency_code for route in routes] == [
        dependency.code for dependency in dependencies
    ]
    assert {route.logical_tuuid for route in routes} == {1}
    assert [route.target_op for route in routes] == ["consume_a", "consume_b"]
    assert [route.target_port for route in routes] == ["left", "right"]
    assert [route.output_index for route in routes] == [0, 0]
    assert all(route.routing_mode is TransportRoutingMode.ALGEBRAIC for route in routes)
    assert routes[0].target_spec != routes[1].target_spec


def test_sender_batches_shared_logical_tensor_by_transfer_code_without_rewrapping():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)
    raw = np.arange(4, dtype=np.float32)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[raw],
            step_transfer_cache={},
            exec_step=5,
        )

    distributor.plan_and_shard.assert_called_once()
    call = distributor.plan_and_shard.call_args
    tensors = call.args[0]
    assert len(tensors) == 2
    assert tensors[0] is tensors[1]
    assert call.kwargs["transfer_codes"] == [1, 2]


def test_sender_batches_multiple_outputs_without_payload_copy():
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("first", "second")},
        output_sharding={"first": replicated, "second": replicated},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("left", "right")},
        input_sharding={"left": replicated, "right": replicated},
    )
    mesh = DeviceMesh(("data",), (1,))
    plan = compile_graph(
        name="sender-multiple-outputs",
        spmds=(
            compiled_spmd_spec("source", mesh, {"produce": source}),
            compiled_spmd_spec("target", mesh, {"consume": target}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("first"), OperatorPort("left")),
                    TensorTransfer(OperatorPort("second"), OperatorPort("right")),
                ),
            ),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_plan.outgoing, effective_rank=0),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    first = np.arange(4, dtype=np.float32)
    second = np.arange(4, 8, dtype=np.float32)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[first, second],
            step_transfer_cache={},
            exec_step=5,
        )

    call = distributor.plan_and_shard.call_args
    tensors = call.args[0]
    assert [
        route.output_index for route in compiled.successor_routes[0].tensor_routes
    ] == [
        0,
        1,
    ]
    assert call.kwargs["transfer_codes"] == [1, 2]
    assert tensors[0].flat_values is first
    assert tensors[1].flat_values is second


def test_sender_separates_content_routed_outputs_with_distinct_values():
    dynamic = ShardingSpec.hash(P("data"))
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("first", "second")},
        output_sharding={"first": dynamic, "second": dynamic},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("left", "right")},
        input_sharding={"left": dynamic, "right": dynamic},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (2,))
    plan = compile_graph(
        name="sender-distinct-content-routing",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": source}),
            compiled_spmd_spec("target", target_mesh, {"consume": target}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("first"), OperatorPort("left")),
                    TensorTransfer(OperatorPort("second"), OperatorPort("right")),
                ),
            ),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_plan.outgoing, effective_rank=0),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    first = np.array([0, 2], dtype=np.int64)
    second = np.array([1, 3], dtype=np.int64)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[first, second],
            step_transfer_cache={},
            exec_step=5,
        )

    assert distributor.plan_and_shard.call_count == 2
    routed_values = [
        call.args[0][0].flat_values
        for call in distributor.plan_and_shard.call_args_list
    ]
    assert routed_values == [first, second]


def test_completed_without_tensor_rows_use_transfer_codes():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_completed_without_tensor_step_frame_groups(
            compiled_plan=compiled,
            empty_values=None,
            exec_step=9,
        )

    assert [
        call.kwargs["transfer_code"]
        for call in distributor.plan_completed_without_tensor.call_args_list
    ] == [1, 2]


def test_declared_transfer_rejects_none_leaf_instead_of_silent_drop():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)

    with pytest.raises(ValueError, match="Declared transfer code 1"):
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[None],
            step_transfer_cache={},
            exec_step=9,
        )


def _compile_mirror_send_plan(*, bind_mirror_input: bool = True):
    dynamic = ShardingSpec.hash(P("data"))
    mirror = ShardingSpec.follow("lookup.unique_ids")
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids",)},
        output_sharding={"ids": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("unique_ids",)},
        output_arguments={"data": ("embedding",)},
        input_sharding={"unique_ids": dynamic},
        output_sharding={"embedding": mirror},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding",)},
        input_sharding={"embedding": dynamic},
    )
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    incoming = DependencySpec(
        source=OperatorKey("source", "upstream"),
        target=OperatorKey("source", "lookup"),
        transfers=(
            TensorTransfer(
                source=OperatorPort("ids"),
                target=OperatorPort("unique_ids"),
            ),
        ),
        local_only=True,
    )
    plan = compile_graph(
        name="sender-mirror-route",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"lookup": lookup, "upstream": upstream},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": target}),
        ),
        dependencies=(
            *((incoming,) if bind_mirror_input else ()),
            _dependency("lookup", "consume", "embedding", "embedding"),
        ),
    )
    source_plan = plan.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_plan.outgoing
        if dependency.source_operator.key.op_id == "lookup"
    )
    input_transfer_code = source_plan.incoming[0].transfers[0].transfer_code
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=outgoing,
                effective_rank=0,
            ),
        ),
        sender_spmd_code=4,
        interleave_unit_size=None,
    )
    return compiled, input_transfer_code


@pytest.mark.parametrize(
    ("source_world_size", "target_world_size", "receipt_world_size", "valid"),
    ((4, 2, 2, True), (2, 3, 2, False)),
)
def test_receipt_rank_domain_comes_from_runtime_lineage(
    source_world_size, target_world_size, receipt_world_size, valid
):
    dynamic = ShardingSpec.hash(P("data"))
    source_mesh = DeviceMesh(("data",), (source_world_size,))
    target_mesh = DeviceMesh(("data",), (target_world_size,))
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids",)},
        output_sharding={"ids": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("ids",)},
        output_arguments={"data": ("embedding",)},
        input_sharding={"ids": dynamic},
        output_sharding={"embedding": ShardingSpec.follow("lookup.ids")},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding",)},
        input_sharding={"embedding": dynamic},
    )
    graph = compile_graph(
        name="receipt-rank-domain-mismatch",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"upstream": upstream, "lookup": lookup},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "upstream"),
                target=OperatorKey("source", "lookup"),
                transfers=(TensorTransfer(OperatorPort("ids"), OperatorPort("ids")),),
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
    source_view = graph.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_view.outgoing
        if dependency.spec.source.op_id == "lookup"
    )

    compiled = compile_send_plan(
        graph_view=source_view,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(SuccessorRouteContext(0, outgoing, effective_rank=0),),
        sender_spmd_code=1,
        interleave_unit_size=None,
    )
    route = compiled.successor_routes[0].tensor_routes[0]
    assert route.receipt_routing is not None
    assert route.receipt_routing.rank_count == target_world_size

    input_transfer_code = source_view.incoming[0].transfers[0].transfer_code
    reference = TransportTensor(
        np.ones((2,), dtype=np.int64),
        receipt=SymmetryReceipt(receipt_world_size, np.array([0, 1], dtype=np.int32)),
    )

    def build():
        return build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[np.ones((2, 8), dtype=np.float32)],
            step_transfer_cache={input_transfer_code: reference},
            exec_step=0,
        )

    if valid:
        assert build()
    else:
        with pytest.raises(ValueError, match="receipt world size does not match"):
            build()


def test_mirror_route_uses_exact_input_transfer_code_cache_key():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    assert route.receipt_routing is not None
    assert route.receipt_routing.input_transfer_code == input_transfer_code
    assert route.receipt_routing.reference_path == "lookup.unique_ids"
    assert route.receipt_routing.rank_count == 2
    assert route.routing_mode is TransportRoutingMode.RECEIPT
    assert route.source_spec.spec.is_dynamic
    assert not route.source_spec.spec.is_referential
    assert route.target_spec.spec.is_dynamic
    assert not route.target_spec.spec.is_referential
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        origin_indices=ExplicitIndex(np.array([20, 10], dtype=np.int64)),
        global_shape=(64,),
        receipt=SymmetryReceipt(2, np.array([1, 0], dtype=np.int32)),
    )
    output = TransportTensor(np.ones((2, 8), dtype=np.float32), global_shape=(2, 8))

    inherited = prepare_route_tensor(
        compiled_plan=compiled,
        route=route,
        raw_leaf=output,
        step_transfer_cache={input_transfer_code: source_ref},
        exec_step=3,
        wrapped_results={},
    )

    assert inherited is not None
    assert inherited.receipt is source_ref.receipt
    assert inherited.origin_indices is source_ref.origin_indices


def test_mirror_route_rejects_unknown_path_without_logical_tuuid_fallback():
    with pytest.raises(ValueError, match="lookup.unique_ids"):
        _compile_mirror_send_plan(bind_mirror_input=False)


def test_mirror_route_rejects_input_without_required_receipt():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        global_shape=(64,),
    )

    with pytest.raises(ValueError, match="requires a SymmetryReceipt"):
        prepare_route_tensor(
            compiled_plan=compiled,
            route=route,
            raw_leaf=np.ones((2, 8), dtype=np.float32),
            step_transfer_cache={input_transfer_code: source_ref},
            exec_step=3,
            wrapped_results={},
        )


def test_mirror_route_preserves_transformed_tensor_lineage():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        origin_indices=ExplicitIndex(np.array([0, 1], dtype=np.int64)),
        receipt=SymmetryReceipt(2, np.array([0, 1], dtype=np.int32)),
    )
    embedding = TransportTensor(
        np.array([[10.0], [20.0]], dtype=np.float32),
        origin_indices=source_ref.origin_indices,
        receipt=source_ref.receipt,
    )
    transformed = embedding.gather(np.array([1, 0], dtype=np.int32))

    inherited = prepare_route_tensor(
        compiled_plan=compiled,
        route=route,
        raw_leaf=transformed,
        step_transfer_cache={input_transfer_code: source_ref},
        exec_step=3,
        wrapped_results={},
    )

    np.testing.assert_array_equal(
        inherited.receipt.sender_rank_tags,
        np.array([1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        inherited.origin_indices_materialized,
        np.array([1, 0], dtype=np.int64),
    )


def test_sender_separates_mirror_outputs_with_distinct_receipts():
    dynamic = ShardingSpec.hash(P("data"))
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids_a", "ids_b")},
        output_sharding={"ids_a": dynamic, "ids_b": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("ids_a", "ids_b")},
        output_arguments={"data": ("embedding_a", "embedding_b")},
        input_sharding={"ids_a": dynamic, "ids_b": dynamic},
        output_sharding={
            "embedding_a": ShardingSpec.follow("lookup.ids_a"),
            "embedding_b": ShardingSpec.follow("lookup.ids_b"),
        },
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding_a", "embedding_b")},
        input_sharding={"embedding_a": dynamic, "embedding_b": dynamic},
    )
    graph = compile_graph(
        name="sender-distinct-receipts",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"upstream": upstream, "lookup": lookup},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "upstream"),
                target=OperatorKey("source", "lookup"),
                transfers=(
                    TensorTransfer(OperatorPort("ids_a"), OperatorPort("ids_a")),
                    TensorTransfer(OperatorPort("ids_b"), OperatorPort("ids_b")),
                ),
                local_only=True,
            ),
            DependencySpec(
                source=OperatorKey("source", "lookup"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(
                        OperatorPort("embedding_a"),
                        OperatorPort("embedding_a"),
                    ),
                    TensorTransfer(
                        OperatorPort("embedding_b"),
                        OperatorPort("embedding_b"),
                    ),
                ),
            ),
        ),
    )
    source_view = graph.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_view.outgoing
        if dependency.spec.source.op_id == "lookup"
    )
    compiled = compile_send_plan(
        graph_view=source_view,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(SuccessorRouteContext(0, outgoing, effective_rank=0),),
        sender_spmd_code=4,
        interleave_unit_size=None,
    )
    shared_origin = ExplicitIndex(np.array([0, 1], dtype=np.int64))
    input_transfers = source_view.incoming[0].transfers
    step_transfer_cache = {
        input_transfers[0].transfer_code: TransportTensor(
            np.array([2, 4], dtype=np.int64),
            origin_indices=shared_origin,
            receipt=SymmetryReceipt(2, 0),
        ),
        input_transfers[1].transfer_code: TransportTensor(
            np.array([3, 5], dtype=np.int64),
            origin_indices=shared_origin,
            receipt=SymmetryReceipt(2, 1),
        ),
    }

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[
                np.ones((2, 4), dtype=np.float32),
                np.full((2, 4), 2.0, dtype=np.float32),
            ],
            step_transfer_cache=step_transfer_cache,
            exec_step=5,
        )

    assert distributor.plan_and_shard.call_count == 2
    receipts = [
        call.args[0][0].receipt for call in distributor.plan_and_shard.call_args_list
    ]
    assert receipts == [
        step_transfer_cache[input_transfers[0].transfer_code].receipt,
        step_transfer_cache[input_transfers[1].transfer_code].receipt,
    ]


def test_successor_context_rejects_mixed_peer_dependencies():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, dependencies = _compile_fanout_plan(second_target_sharding=replicated)
    foreign = CompiledDependency(
        code=99,
        spec=DependencySpec(
            source=dependencies[0].spec.source,
            target=OperatorKey("foreign", "consume"),
        ),
        source_operator=dependencies[0].source_operator,
        target_operator=dependencies[0].target_operator,
        transfers=(),
        source_world_size=1,
        target_world_size=1,
        source_mesh=dependencies[0].source_mesh,
        target_mesh=dependencies[0].target_mesh,
    )

    with pytest.raises(ValueError, match="target SPMD"):
        SuccessorRouteContext(
            context_index=compiled.successor_routes[0].context_index,
            dependencies=(dependencies[0], foreign),
            effective_rank=0,
        )
