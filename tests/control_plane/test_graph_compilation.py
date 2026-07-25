from typing import Any

import pytest
from pydantic import ValidationError
from ray import cloudpickle

import fluxway as fw
from fluxway.common.decorators import operator, spmd


_REP = {"type": "static", "axis": []}


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(2,),
)
class _PlanSPMD:
    @operator(name="produce", output_sharding={"value": _REP})
    def produce(self) -> dict[str, Any]:
        return {"value": 1}

    @operator(name="consume", input_sharding={"item": _REP})
    def consume(self, item: Any) -> None:
        del item

    @operator(name="consume_pair", input_sharding={"first": _REP, "second": _REP})
    def consume_pair(self, first: Any, second: Any) -> None:
        del first, second

    @operator(
        name="loop",
        input_sharding={"token": _REP},
        output_sharding={"next": _REP},
    )
    def loop(self, token: Any) -> dict[str, Any]:
        return {"next": token}


def _condition(result: Any) -> bool:
    return result is not None


def _build_graph():
    builder = fw.Builder()
    builder.add_node("source", _PlanSPMD)
    builder.add_node("target", _PlanSPMD)
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
        condition_fn=_condition,
    )
    return builder.build("compiled-graph")


def test_builder_compiles_one_ray_native_compiled_graph():
    graph = _build_graph()
    plan = graph.require_compiled_graph()
    dependency = plan.dependencies[0]

    assert plan.name == "compiled-graph"
    assert graph.dependency_specs[0] is dependency.spec
    assert dependency.spec.condition is _condition
    assert dependency.transfers[0].logical_tuuid == 1
    assert graph.nodes["source"].spmd_config.graph_view is plan.spmd("source")
    assert graph.nodes["target"].spmd_config.graph_view is plan.spmd("target")
    assert (
        plan.spmd("source").operator("produce").spec
        is graph.nodes["source"].spmd_config.operator_specs["produce"]
    )

    operator_fields = type(
        graph.nodes["source"].spmd_config.operator_specs["produce"]
    ).model_fields
    assert "spmd_id" not in operator_fields
    assert "graph_facts" not in operator_fields
    assert "output_uuids" not in operator_fields


def test_built_graph_keeps_plan_identity_after_ray_serialization():
    graph = cloudpickle.loads(cloudpickle.dumps(_build_graph()))
    plan = graph.compiled_graph

    assert graph.nodes["source"].spmd_config.graph_view is plan.spmd("source")
    assert graph.nodes["target"].spmd_config.graph_view is plan.spmd("target")
    assert plan.spmd("source").outgoing[0] is plan.dependencies[0]
    assert plan.dependencies[0].spec.condition(object()) is True


def test_rebuilding_builder_keeps_existing_compiled_graph_isolated():
    builder = fw.Builder()
    builder.add_node("source", _PlanSPMD)
    builder.add_node("target", _PlanSPMD)
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
    )

    first = builder.build("first")
    first_plan = first.compiled_graph
    second = builder.build("second")
    second_plan = second.compiled_graph

    assert first_plan is not None
    assert second_plan is not None
    assert first_plan.name == "first"
    assert second_plan.name == "second"
    assert first_plan.graph_id != second_plan.graph_id
    assert first.nodes["source"].spmd_config.graph_view is first_plan.spmd("source")
    assert second.nodes["source"].spmd_config.graph_view is second_plan.spmd("source")
    with pytest.raises(ValidationError, match="Field is frozen"):
        first.nodes["source"].spmd_config.graph_view = second_plan.spmd("source")


def test_builder_preserves_dependency_attributes_in_compiled_graph():
    builder = fw.Builder()
    builder.add_node("source", _PlanSPMD)
    builder.add_node("target", _PlanSPMD)
    forward_key = ("source", "produce", "target", "consume")
    reverse_key = ("target", "loop", "source", "loop")
    builder.add_func_edge(
        *forward_key,
        tensor_map={"value": "item"},
        condition_fn=_condition,
        symmetric_to=reverse_key,
    )
    builder.add_func_edge(
        *reverse_key,
        tensor_map={"next": "token"},
        symmetric_to=forward_key,
    )
    builder.add_func_edge(
        "target",
        "loop",
        "target",
        "loop",
        tensor_map={"next": "token"},
        local_only=True,
    )

    plan = builder.build("dependency-attributes").compiled_graph
    assert plan is not None
    dependencies = {
        dependency.spec.key: dependency.spec for dependency in plan.dependencies
    }
    assert dependencies[forward_key].condition is _condition
    assert dependencies[forward_key].symmetric_to == reverse_key
    assert dependencies[forward_key].local_only is False
    assert dependencies[reverse_key].symmetric_to == forward_key
    local_key = ("target", "loop", "target", "loop")
    assert dependencies[local_key].condition is None
    assert dependencies[local_key].symmetric_to is None
    assert dependencies[local_key].local_only is True


def test_builder_merges_repeated_bindings_into_one_dependency():
    builder = fw.Builder()
    builder.add_node("source", _PlanSPMD)
    builder.add_node("target", _PlanSPMD)
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume_pair",
        tensor_map={"value": "first"},
    )
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume_pair",
        tensor_map={"value": "second"},
    )

    graph = builder.build("repeated-bindings")
    plan = graph.compiled_graph
    assert plan is not None
    assert len(graph.dependency_specs) == len(plan.dependencies) == 1
    assert graph.dependency_specs[0] is plan.dependencies[0].spec
    assert tuple(
        (transfer.spec.source.name, transfer.spec.target.name)
        for transfer in plan.dependencies[0].transfers
    ) == (("value", "first"), ("value", "second"))


def test_builder_distinguishes_inferred_and_zero_transfer_dependencies():
    inferred = fw.Builder()
    inferred.add_node("source", _PlanSPMD)
    inferred.add_node("target", _PlanSPMD)
    inferred.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map=None,
    )

    ordering = fw.Builder()
    ordering.add_node("source", _PlanSPMD)
    ordering.add_node("target", _PlanSPMD)
    ordering.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={},
    )

    assert (
        len(
            inferred.build("inferred")
            .require_compiled_graph()
            .dependencies[0]
            .transfers
        )
        == 1
    )
    assert (
        ordering.build("ordering").require_compiled_graph().dependencies[0].transfers
        == ()
    )
