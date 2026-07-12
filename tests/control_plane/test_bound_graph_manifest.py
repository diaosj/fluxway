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


def test_bound_inherited_operator_identity_uses_resolved_metadata():
    bound_spmd = bind_spmd(
        _InheritedOperators,
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

    manifest = fw.graph_manifest(builder.build("bound-manifest"))
    assert all(
        spmd.class_id == f"{bound_spmd.__module__}.{bound_spmd.__qualname__}"
        for spmd in manifest.spmds
    )
    functions = {
        operator.op_id: operator.function_id
        for spmd in manifest.spmds
        for operator in spmd.operators
    }
    assert functions["produce"].endswith("._BaseOperators.produce")
    assert functions["consume"].endswith("._BaseOperators.consume")
