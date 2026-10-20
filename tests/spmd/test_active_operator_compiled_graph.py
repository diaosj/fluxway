from __future__ import annotations

import pytest

from fluxway import OperatorSpec
from fluxway.common.compiled_graph import (
    SPMDGraphView,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import AsyncStepConfig, Graph, Node, SPMDConfig
from fluxway.control_plane.async_manager.manager import AsyncStepManager
from fluxway.spmd.driver.driver import SPMDDriver
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _compiled_graph(
    spmd_id: str,
    *operator_ids: str,
) -> tuple[DeviceMesh, SPMDGraphView]:
    mesh = DeviceMesh(axis_names=("worker",), device_shape=(2,))
    compiled_graph = compile_graph(
        name="active-operator-test",
        spmds=(
            compiled_spmd_spec(
                spmd_id=spmd_id,
                mesh=mesh,
                operators={name: OperatorSpec(name=name) for name in operator_ids},
            ),
        ),
        dependencies=(),
    )
    return mesh, compiled_graph.spmd(spmd_id)


def _active_graph(spmd_id: str, *operator_ids: str) -> Graph:
    mesh = DeviceMesh(axis_names=("worker",), device_shape=(2,))
    compiled_graph = compile_graph(
        name="active-operator-test",
        spmds=(
            compiled_spmd_spec(
                spmd_id=spmd_id,
                mesh=mesh,
                operators={name: OperatorSpec(name=name) for name in operator_ids},
            ),
        ),
        dependencies=(),
    )
    config = SPMDConfig(
        spmd_id=spmd_id,
        mesh=mesh,
        graph_view=compiled_graph.spmd(spmd_id),
        operator_specs={"not-active": OperatorSpec(name="not-active")},
    )
    return Graph(
        name="active-operator-test",
        nodes={
            spmd_id: Node(
                name=spmd_id,
                spmd_class=object,
                spmd_config=config,
            )
        },
        compiled_graph=compiled_graph,
    )


def _config(spmd_id: str, *operator_ids: str) -> SPMDConfig:
    mesh, plan = _compiled_graph(spmd_id, *operator_ids)
    return SPMDConfig(
        spmd_id=spmd_id,
        mesh=mesh,
        graph_view=plan,
        operator_specs={"not-active": OperatorSpec(name="not-active")},
    )


def test_spmd_compiled_graph_caches_canonical_operator_ids() -> None:
    _, plan = _compiled_graph("train", "update", "forward")

    assert plan.operator_ids == ("forward", "update")
    assert plan.operator_ids is plan.operator_ids


def test_driver_registers_only_compiled_graph_operators() -> None:
    driver = SPMDDriver.__ray_actor_class__(  # type: ignore[attr-defined]
        spmd_id="train",
        config=_config("train", "update", "forward"),
        placement_group=None,
        worker_bundle_indices=[0, 1],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a", "node-b"],
    )

    drained: list[bool] = []
    driver.spmd_synchronizer.drain_to_step_count(1, lambda: drained.append(True))
    for rank in range(2):
        driver.spmd_synchronizer.report_worker_op_completed(rank, "not-active", 1)
        driver.spmd_synchronizer.report_worker_op_completed(rank, "forward", 1)
    assert drained == []

    for rank in range(2):
        driver.spmd_synchronizer.report_worker_op_completed(rank, "update", 1)
    assert drained == [True]


def test_async_manager_completes_only_compiled_graph_operators() -> None:
    graph = _active_graph("train", "update", "forward")
    manager = AsyncStepManager(graph, AsyncStepConfig(max_steps=2))

    manager.report_op_completed("train", "not-active", 100)
    manager.report_op_completed("train", "forward", 2)
    assert manager.is_all_completed() is False

    manager.report_op_completed("train", "update", 2)
    assert manager.is_all_completed() is True


def test_active_operator_consumers_reject_missing_compiled_graph() -> None:
    config = SPMDConfig(spmd_id="train")
    with pytest.raises(RuntimeError, match="has no graph view"):
        SPMDDriver.__ray_actor_class__(  # type: ignore[attr-defined]
            spmd_id="train",
            config=config,
            placement_group=None,
            worker_bundle_indices=[0],
            controller_handle=None,
            graph_topology={},
            worker_resource={"CPU": 0},
            expected_node_ids=["node-a"],
        )

    graph = Graph(name="missing-plan-test", nodes={})
    with pytest.raises(RuntimeError, match="has no compiled graph"):
        AsyncStepManager(graph, AsyncStepConfig(max_steps=1))
