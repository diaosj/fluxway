from typing import Any

import fluxway as fw
from fluxway.common.bind import bind_spmd


_REP = {"type": "static", "axis": []}


class _BaseOperators:
    def produce(self) -> dict[str, Any]:
        return {"value": 1}

    def consume(self, item: Any) -> None:
        del item


class _InheritedOperators(_BaseOperators):
    pass


def test_bound_inherited_operator_identity_uses_compiled_metadata():
    bound_spmd = bind_spmd(
        _InheritedOperators,
        resource_profile=fw.ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("data",),
        mesh_axis_shapes=(1,),
        produce={"output_sharding": {"value": _REP}},
        consume={"input_sharding": {"item": _REP}},
    )
    builder = fw.Builder()
    builder.add_node("source", bound_spmd)
    builder.add_node("target", bound_spmd)
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
    )

    graph = builder.build("bound-compiled-graph")
    plan = graph.compiled_graph
    assert plan is not None
    assert all(node.spmd_class is bound_spmd for node in graph.nodes.values())
    functions = {
        operator.key.op_id: operator.function_id
        for spmd in plan.spmds
        for operator in spmd.operators
    }
    assert functions["produce"].endswith("._InheritedOperators.produce")
    assert functions["consume"].endswith("._InheritedOperators.consume")
