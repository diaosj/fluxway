from typing import Any

import pytest

from fluxway.common.decorators import operator, spmd
from fluxway.control_plane.graph.builder import Builder
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.scheduler import build_topology
from fluxway.resources import ResourceProfile

_LOCAL_SPEC = {"type": "static", "axis": []}


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
)
class _LocalOnlySPMD:
    @operator(name="produce", output_sharding={"x": _LOCAL_SPEC})
    def produce(self) -> dict:
        return {"x": 1}

    @operator(
        name="consume",
        input_sharding={"x": _LOCAL_SPEC},
        output_sharding={"y": _LOCAL_SPEC},
    )
    def consume(self, x: Any) -> dict:
        return {"y": x}

    @operator(name="consume_remote", input_sharding={"x": _LOCAL_SPEC})
    def consume_remote(self, x: Any) -> None:
        del x


def test_functional_graph_local_edge_marks_metadata_and_skips_global_topology():
    with FunctionalGraph("local_only") as graph:
        node = graph.node("A", _LocalOnlySPMD)
        produced = node.produce()
        node.consume(x=produced.x, local=True)
        task_graph = graph.build()

    assert len(task_graph.dependency_specs) == 1
    edge = task_graph.dependency_specs[0]
    assert edge.local_only is True

    config = task_graph.nodes["A"].spmd_config
    plan = config.require_graph_view()
    assert plan.outgoing[0] is plan.incoming[0]
    assert plan.outgoing[0].spec is edge
    assert plan.outgoing[0].spec.local_only is True

    assert build_topology(task_graph.require_compiled_graph())["A"] == []


def test_builder_rejects_cross_node_local_only_edges():
    builder = Builder()
    builder.add_node("A", _LocalOnlySPMD)
    builder.add_node("B", _LocalOnlySPMD)

    with pytest.raises(ValueError, match="source_node == target_node"):
        builder.add_func_edge(
            "A",
            "produce",
            "B",
            "consume",
            source_output="x",
            target_input="x",
            local_only=True,
        )


def test_builder_preserves_independent_locality_for_exact_dependencies():
    builder = Builder()
    builder.add_node("A", _LocalOnlySPMD)
    builder.add_func_edge(
        "A",
        "produce",
        "A",
        "consume",
        source_output="x",
        target_input="x",
        local_only=True,
    )

    builder.add_func_edge(
        "A",
        "produce",
        "A",
        "consume_remote",
        source_output="x",
        target_input="x",
    )

    plan = builder.build("mixed-locality").require_compiled_graph()
    assert tuple(dependency.spec.local_only for dependency in plan.dependencies) == (
        True,
        False,
    )
