from typing import Any

import pytest

from fluxway.common.decorators import operator, spmd
from fluxway.control_plane.graph.builder import Builder
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.scheduler.scheduler import Scheduler

_LOCAL_SPEC = {"type": "static", "axis": []}


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
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
        task_graph = graph.compile()

    assert len(task_graph.edges) == 1
    edge = task_graph.edges[0]
    assert edge.local_only is True

    config = task_graph.nodes["A"].spmd_config
    successor = config.topology["produce"].successors["A"]
    predecessor = config.topology["consume"].predecessors["A"]
    assert successor.local_only is True
    assert predecessor.local_only is True

    produce_meta = config.operator_metas["produce"]
    consume_meta = config.operator_metas["consume"]
    assert produce_meta.output_post_sharding["A"][0].local_only is True
    assert consume_meta.input_pre_sharding["A"][0].local_only is True
    assert Scheduler()._build_topology(task_graph)["A"] == []


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


def test_builder_rejects_mixed_local_and_remote_same_self_successor():
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

    with pytest.raises(ValueError, match="Cannot mix local_only and remote edges"):
        builder.add_func_edge(
            "A",
            "produce",
            "A",
            "consume_remote",
            source_output="x",
            target_input="x",
        )
