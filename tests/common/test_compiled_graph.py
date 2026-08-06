from dataclasses import FrozenInstanceError

import pytest
from ray import cloudpickle

from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.coherence import TensorAccessIntent
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _replicated() -> ShardingSpec:
    return ShardingSpec.chunk(PartitionSpec())


def test_remote_donate_input_is_rejected_at_compile_time():
    replicated = _replicated()
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("item",)},
        input_sharding={"item": replicated},
        tensor_access={"item": TensorAccessIntent.DONATE},
    )
    with pytest.raises(ValueError, match="requires a local_only dependency"):
        compile_graph(
            name="remote-donate",
            spmds=(
                compiled_spmd_spec(
                    "source", DeviceMesh(("data",), (1,)), {"produce": produce}
                ),
                compiled_spmd_spec(
                    "target", DeviceMesh(("data",), (1,)), {"consume": consume}
                ),
            ),
            dependencies=(
                DependencySpec(
                    source=OperatorKey("source", "produce"),
                    target=OperatorKey("target", "consume"),
                    transfers=(
                        TensorTransfer(OperatorPort("value"), OperatorPort("item")),
                    ),
                ),
            ),
        )


def test_compiled_graph_compiler_is_mode_neutral():
    replicated = _replicated()
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("item",)},
        input_sharding={"item": replicated},
    )
    threshold = 3

    def should_send(value):
        return value > threshold

    dependency = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("target", "consume"),
        transfers=(
            TensorTransfer(
                source=OperatorPort("value"),
                target=OperatorPort("item"),
            ),
        ),
        condition=should_send,
    )
    graph_plan = compile_graph(
        name="graph",
        spmds=(
            compiled_spmd_spec(
                spmd_id="source",
                mesh=DeviceMesh(("data",), (2,)),
                operators={"produce": produce},
            ),
            compiled_spmd_spec(
                spmd_id="target",
                mesh=DeviceMesh(("data",), (4,)),
                operators={"consume": consume},
            ),
        ),
        dependencies=(dependency,),
    )
    dependency_free_plan = compile_graph(
        name="dependency-free",
        spmds=(
            compiled_spmd_spec(
                spmd_id="source",
                mesh=DeviceMesh(("data",), (1,)),
                operators={"produce": produce},
            ),
        ),
        dependencies=(),
    )

    graph_spmd = graph_plan.spmd("source")
    graph_operator = graph_spmd.operator("produce")
    dependency_free_operator = dependency_free_plan.spmd("source").operator("produce")
    assert graph_spmd.graph_id == graph_plan.graph_id
    assert graph_operator.spec is produce
    assert dependency_free_operator.spec is produce
    assert graph_operator.output_uuids == (1,)
    assert graph_operator.function_id == dependency_free_operator.function_id
    assert len(graph_operator.input_signature_id) == 64
    assert len(graph_operator.output_signature_id) == 64

    compiled_dependency = graph_plan.dependencies[0]
    assert compiled_dependency.code == 1
    assert compiled_dependency.spec is dependency
    assert compiled_dependency.source_operator is graph_operator
    assert compiled_dependency.target_operator is graph_plan.spmd("target").operator(
        "consume"
    )
    compiled_transfer = compiled_dependency.transfers[0]
    assert compiled_transfer.transfer_code == 1
    assert compiled_transfer.dependency_code == compiled_dependency.code
    assert compiled_transfer.logical_tuuid == 1
    assert graph_plan.spmd("source").transfer(1) is compiled_transfer
    assert graph_plan.spmd("target").transfer(1) is compiled_transfer
    assert compiled_dependency.source_world_size == 2
    assert compiled_dependency.target_world_size == 4
    assert (
        compiled_dependency.spec.condition is not None
        and compiled_dependency.spec.condition(4) is True
    )
    assert (
        compiled_dependency.spec.condition is not None
        and compiled_dependency.spec.condition(3) is False
    )
    assert graph_plan.spmd("source").outgoing == (compiled_dependency,)
    assert graph_plan.spmd("target").incoming == (compiled_dependency,)

    with pytest.raises(FrozenInstanceError):
        graph_operator.output_uuids = ()  # type: ignore[misc]

    restored = cloudpickle.loads(cloudpickle.dumps(graph_plan))
    restored_operator = restored.spmd("source").operator("produce")
    restored_dependency = restored.dependencies[0]
    restored_transfer = restored_dependency.transfers[0]
    assert restored_operator.spec == produce
    assert restored_dependency.source_operator is restored_operator
    assert restored_dependency.spec.condition(4) is True
    assert restored.spmd("source").outgoing[0] is restored_dependency
    assert restored.spmd("source").transfer(1) is restored_transfer
    assert restored.spmd("target").transfer(1) is restored_transfer


def test_transfer_codes_are_plan_wide_canonical_wire_identities():
    replicated = _replicated()
    partitioned = ShardingSpec.chunk(PartitionSpec("data"))
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    consume_a = OperatorSpec(
        name="consume_a",
        input_arguments={"data": ("replica", "shard")},
        input_sharding={"replica": replicated, "shard": partitioned},
    )
    consume_b = OperatorSpec(
        name="consume_b",
        input_arguments={"data": ("item",)},
        input_sharding={"item": replicated},
    )
    spmds = (
        compiled_spmd_spec(
            spmd_id="source",
            mesh=DeviceMesh(("data",), (2,)),
            operators={"produce": produce},
        ),
        compiled_spmd_spec(
            spmd_id="target",
            mesh=DeviceMesh(("data",), (2,)),
            operators={"consume_a": consume_a, "consume_b": consume_b},
        ),
    )
    first = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("target", "consume_a"),
        transfers=(
            TensorTransfer(OperatorPort("value"), OperatorPort("shard")),
            TensorTransfer(OperatorPort("value"), OperatorPort("replica")),
        ),
    )
    second = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("target", "consume_b"),
        transfers=(TensorTransfer(OperatorPort("value"), OperatorPort("item")),),
    )

    plan = compile_graph(
        name="wire-identity",
        spmds=spmds,
        dependencies=(second, first),
    )
    reordered = compile_graph(
        name="wire-identity",
        spmds=spmds,
        dependencies=(
            DependencySpec(
                source=first.source,
                target=first.target,
                transfers=tuple(reversed(first.transfers)),
            ),
            second,
        ),
    )

    def codes(compiled_graph):
        return {
            (
                dependency.key,
                transfer.spec.source.name,
                transfer.spec.target.name,
            ): transfer.transfer_code
            for dependency in compiled_graph.dependencies
            for transfer in dependency.transfers
        }

    transfers = tuple(
        transfer
        for dependency in plan.dependencies
        for transfer in dependency.transfers
    )
    assert codes(plan) == codes(reordered)
    assert {transfer.transfer_code for transfer in transfers} == {1, 2, 3}
    assert {transfer.logical_tuuid for transfer in transfers} == {1}
    assert all(
        transfer.dependency_code == dependency.code
        for dependency in plan.dependencies
        for transfer in dependency.transfers
    )
    first_transfers = plan.dependencies[0].transfers
    assert {
        transfer.spec.target.name: transfer.target_sharding
        for transfer in first_transfers
    } == {"replica": replicated, "shard": partitioned}

    for transfer in transfers:
        assert plan.spmd("source").transfer(transfer.transfer_code) is transfer
        assert plan.spmd("target").transfer(transfer.transfer_code) is transfer

    with pytest.raises(ValueError, match="no transfer code 4"):
        plan.spmd("source").transfer(4)


def test_compiler_keeps_exact_operator_dependencies_for_one_target_spmd():
    before = OperatorSpec(name="before")
    after_a = OperatorSpec(name="after_a")
    after_b = OperatorSpec(name="after_b")
    plan = compile_graph(
        name="ordering",
        spmds=(
            compiled_spmd_spec(
                spmd_id="source",
                mesh=DeviceMesh(("data",), (1,)),
                operators={"before": before},
            ),
            compiled_spmd_spec(
                spmd_id="target",
                mesh=DeviceMesh(("data",), (1,)),
                operators={"after_a": after_a, "after_b": after_b},
            ),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "before"),
                target=OperatorKey("target", "after_a"),
            ),
            DependencySpec(
                source=OperatorKey("source", "before"),
                target=OperatorKey("target", "after_b"),
            ),
        ),
    )

    assert tuple(item.spec.target.op_id for item in plan.spmd("source").outgoing) == (
        "after_a",
        "after_b",
    )
    assert tuple(item.code for item in plan.spmd("target").incoming) == (1, 2)


def test_compiler_resolves_referential_sharding_without_rewriting_spec():
    replicated = _replicated()
    mirrored = ShardingSpec.follow("produce.value")
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value", "mirror")},
        output_sharding={"value": replicated, "mirror": mirrored},
    )
    plan = compile_graph(
        name="mirror",
        spmds=(
            compiled_spmd_spec(
                spmd_id="source",
                mesh=DeviceMesh(("data",), (1,)),
                operators={"produce": produce},
            ),
        ),
        dependencies=(),
    )

    compiled = plan.spmd("source").operator("produce")
    from fluxway.common.types.sharding import ReferentialRouting

    assert isinstance(produce.output_sharding["mirror"].routing, ReferentialRouting)
    assert produce.output_sharding["mirror"].routing.tuuid is None
    assert isinstance(compiled.output_sharding["mirror"].routing, ReferentialRouting)
    assert compiled.output_sharding["mirror"].routing.tuuid == 1
    assert compiled.output_uuids == (1, 2)


def test_compiler_reports_referential_cycle_with_business_port_names():
    cycle = OperatorSpec(
        name="produce",
        output_arguments={"data": ("left", "right")},
        output_sharding={
            "left": ShardingSpec.follow("produce.right"),
            "right": ShardingSpec.follow("produce.left"),
        },
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("item",)},
        input_sharding={"item": _replicated()},
    )

    with pytest.raises(
        ValueError,
        match=(r"source\.produce:left.*source\.produce:right.*source\.produce:left"),
    ):
        compile_graph(
            name="referential-cycle",
            spmds=(
                compiled_spmd_spec(
                    spmd_id="source",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"produce": cycle},
                ),
                compiled_spmd_spec(
                    spmd_id="target",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"consume": consume},
                ),
            ),
            dependencies=(
                DependencySpec(
                    source=OperatorKey("source", "produce"),
                    target=OperatorKey("target", "consume"),
                    transfers=(
                        TensorTransfer(
                            OperatorPort("left"),
                            OperatorPort("item"),
                        ),
                    ),
                ),
            ),
        )


def test_compiler_rejects_unimplemented_reference_dimension_map_by_port_name():
    replicated = _replicated()
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("basis", "mirror")},
        output_sharding={
            "basis": replicated,
            "mirror": ShardingSpec.follow(
                "produce.basis",
                dim_map={1: 0},
            ),
        },
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("item",)},
        input_sharding={"item": replicated},
    )

    with pytest.raises(
        ValueError,
        match=r"source\.produce:mirror uses unsupported dimension map",
    ):
        compile_graph(
            name="referential-dimension-map",
            spmds=(
                compiled_spmd_spec(
                    spmd_id="source",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"produce": produce},
                ),
                compiled_spmd_spec(
                    spmd_id="target",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"consume": consume},
                ),
            ),
            dependencies=(
                DependencySpec(
                    source=OperatorKey("source", "produce"),
                    target=OperatorKey("target", "consume"),
                    transfers=(
                        TensorTransfer(
                            OperatorPort("mirror"),
                            OperatorPort("item"),
                        ),
                    ),
                ),
            ),
        )


def test_compiled_graph_rejects_unknown_operator_dependency():
    operator = OperatorSpec(name="produce")
    dependency = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("missing", "consume"),
    )

    with pytest.raises(ValueError, match="unknown target operator"):
        compile_graph(
            name="invalid",
            spmds=(
                compiled_spmd_spec(
                    spmd_id="source",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"produce": operator},
                ),
            ),
            dependencies=(dependency,),
        )


def test_compiled_graph_rejects_operator_mapping_key_mismatch():
    declaration = OperatorSpec(name="declared")

    with pytest.raises(ValueError, match="does not match OperatorSpec.name"):
        compile_graph(
            name="invalid-operator-key",
            spmds=(
                compiled_spmd_spec(
                    spmd_id="source",
                    mesh=DeviceMesh(("data",), (1,)),
                    operators={"alias": declaration},
                ),
            ),
            dependencies=(),
        )


def test_dependency_rejects_duplicate_tensor_transfer():
    transfer = TensorTransfer(
        source=OperatorPort("value"),
        target=OperatorPort("item"),
    )

    with pytest.raises(ValueError, match="duplicate transfers"):
        DependencySpec(
            source=OperatorKey("source", "produce"),
            target=OperatorKey("target", "consume"),
            transfers=(transfer, transfer),
        )


def test_compiled_graph_rejects_missing_or_nonreciprocal_symmetric_dependency():
    before = OperatorSpec(name="before")
    after = OperatorSpec(name="after")
    spmds = (
        compiled_spmd_spec(
            spmd_id="source",
            mesh=DeviceMesh(("data",), (1,)),
            operators={"before": before},
        ),
        compiled_spmd_spec(
            spmd_id="target",
            mesh=DeviceMesh(("data",), (1,)),
            operators={"after": after},
        ),
    )
    forward_key = ("source", "before", "target", "after")
    reverse_key = ("target", "after", "source", "before")
    forward = DependencySpec(
        source=OperatorKey("source", "before"),
        target=OperatorKey("target", "after"),
        symmetric_to=reverse_key,
    )

    with pytest.raises(ValueError, match="unknown symmetric dependency"):
        compile_graph(
            name="missing-symmetric",
            spmds=spmds,
            dependencies=(forward,),
        )

    reverse = DependencySpec(
        source=OperatorKey("target", "after"),
        target=OperatorKey("source", "before"),
    )
    with pytest.raises(ValueError, match="must reference each other"):
        compile_graph(
            name="nonreciprocal-symmetric",
            spmds=spmds,
            dependencies=(forward, reverse),
        )

    reciprocal = DependencySpec(
        source=reverse.source,
        target=reverse.target,
        symmetric_to=forward_key,
    )
    plan = compile_graph(
        name="reciprocal-symmetric",
        spmds=spmds,
        dependencies=(forward, reciprocal),
    )
    assert {dependency.spec.key for dependency in plan.dependencies} == {
        forward_key,
        reverse_key,
    }
