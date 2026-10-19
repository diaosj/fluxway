import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from ray import cloudpickle

from fluxway.common.mesh import DeviceMesh
from fluxway.control_plane.scheduler import Scheduler
from fluxway.placement import (
    AFFINITY_FIRST,
    ClusterNode,
    ClusterSnapshot,
    PlacementPlanner,
    PlacementRecipe,
    PlacementRuntimeConfig,
    ResourceProfile,
    ResourceScale,
    subseteq,
    support,
    workers,
)

from .test_planner import graph_node, placement_graph, placement_inventory


class AcceleratorSPMD:
    pass


class AffineSPMD:
    pass


class MutatedRuntimeSPMD:
    pass


def _compiled_affinity_graph():
    gib = 1 << 30
    accelerator = ResourceProfile(
        base={
            "CPU": 1,
            "accelerator:H100": 1,
            "accelerator_memory": 8 * gib,
            "memory": gib,
        }
    )
    affine = ResourceProfile(
        base={"CPU": 1, "memory": gib},
        scale=ResourceScale(
            step={"CPU": Decimal("0.5"), "memory": gib},
            max_steps=2,
            weight=1,
        ),
    )
    graph = placement_graph(
        {
            "accelerator": graph_node("accelerator", AcceleratorSPMD, 1, accelerator),
            "affine": graph_node("affine", AffineSPMD, 1, affine),
        },
        name="compiled-affinity",
    )
    recipe = PlacementRecipe(
        recipe_id="compiled-affinity",
        constraints=(
            subseteq(
                support(workers(AffineSPMD)),
                support(workers(AcceleratorSPMD)),
            ),
        ),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="accelerator-node",
                resources={
                    "CPU": 4,
                    "accelerator:H100": 1,
                    "accelerator_memory": 8 * gib,
                    "memory": 4 * gib,
                },
            ),
            ClusterNode(
                node_id="cpu-node",
                resources={"CPU": 4, "memory": 4 * gib},
            ),
        )
    )
    return graph, recipe, snapshot, accelerator, affine


def test_compiled_spmd_declaration_is_the_graph_view_single_source() -> None:
    graph, _, _, accelerator, affine = _compiled_affinity_graph()
    compiled = graph.require_compiled_graph()

    assert compiled.declarations == tuple(view.declaration for view in compiled.spmds)
    assert compiled.spmd("accelerator").declaration is compiled.declarations[0]
    assert compiled.spmd("affine").declaration is compiled.declarations[1]
    assert compiled.spmd("accelerator").resource_profile is accelerator
    assert compiled.spmd("affine").resource_profile is affine
    assert compiled.spmd("affine").mesh is compiled.spmd("affine").declaration.mesh
    assert compiled.spmd("affine").mesh is not graph.nodes["affine"].spmd_config.mesh
    mesh = compiled.spmd("affine").mesh
    assert mesh.is_frozen is True

    with pytest.raises(ValueError, match="read-only"):
        mesh.device_ids[0] = 99
    with pytest.raises(AttributeError, match="frozen"):
        mesh._shape = (9,)
    with pytest.raises(TypeError):
        mesh.axis_map["other"] = 1  # type: ignore[index]

    restored = cloudpickle.loads(cloudpickle.dumps(compiled))
    restored_mesh = restored.spmd("affine").mesh
    assert restored_mesh.is_frozen is True
    assert restored.spmd("affine").declaration is restored.declarations[1]
    with pytest.raises(ValueError, match="read-only"):
        restored_mesh.device_ids[0] = 99

    with pytest.raises(ValidationError, match="frozen"):
        graph.compiled_graph = None


def test_affinity_resolve_attest_and_snapshot_ignore_runtime_node_mutation(
    monkeypatch,
) -> None:
    graph, recipe, snapshot, _, affine = _compiled_affinity_graph()
    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    before = placement_inventory(graph)

    for node in graph.nodes.values():
        node.spmd_class = MutatedRuntimeSPMD
        node.resource_profile = ResourceProfile(base={"TPU": 99})
        node.spmd_config.mesh = DeviceMesh(("mutated",), (4,))

    after = placement_inventory(graph)
    assert after.declarations == before.declarations
    assert after.spmd_ids_for(workers(AffineSPMD).spmd) == ("affine",)
    assert affine.minimum == {"CPU": Decimal(1), "memory": Decimal(1 << 30)}
    assert affine.maximum == {
        "CPU": Decimal(2),
        "memory": Decimal(3 * (1 << 30)),
    }
    assert scheduler._placement_snapshot(snapshot) == snapshot

    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)

    async def create_job_pg(*, plan, snapshot, timeout_s):
        return SimpleNamespace(id="compiled-placement")

    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)
    asyncio.run(scheduler.allocate_resources())

    plan = scheduler.resolved_plan
    assert plan is not None
    assert plan.allocated_quanta["affine"] == 2
    assert plan.node_ids_for("accelerator") == {"accelerator-node"}
    assert plan.node_ids_for("affine") == {"accelerator-node"}
    assert {
        assignment.spmd_id: assignment.resources for assignment in plan.assignments
    } == {
        "accelerator": {
            "CPU": Decimal(1),
            "accelerator:H100": Decimal(1),
            "accelerator_memory": Decimal(8 * (1 << 30)),
            "memory": Decimal(1 << 30),
        },
        "affine": {
            "CPU": Decimal(2),
            "memory": Decimal(3 * (1 << 30)),
        },
    }

    attested = PlacementPlanner(solve_timeout_s=10).attest(
        after,
        recipe,
        snapshot,
        plan,
        {
            "accelerator": ("accelerator-node",),
            "affine": ("accelerator-node",),
        },
    )
    assert all(result.satisfied for result in attested.constraint_results)
