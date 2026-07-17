from decimal import Decimal

from fluxway.placement import (
    AFFINITY_FIRST_V1,
    ClusterNode,
    ClusterSnapshot,
    PlacementPlanner,
    PlacementRecipe,
    ResourceProfile,
    ResourceScale,
    allocated_quanta,
    lexicographic,
    maximize,
    workers,
)

from .test_planner import SPMD0, graph_node, placement_graph, placement_inventory


def test_ray_component_limits_are_accounted_but_excluded_from_actor_bundles() -> None:
    profile = ResourceProfile(
        base={"CPU": 1, "memory": 1024},
        scale=ResourceScale(step={"CPU": 1}, max_steps=3),
        ray_component_limits={"CPU": 2, "memory": 2048},
    )
    graph = placement_graph({"metrics": graph_node("metrics", SPMD0, 1, profile)})
    recipe = PlacementRecipe(
        recipe_id="ray-component-limits",
        version="1.0.0",
        constraints=(),
        objective=lexicographic(maximize(allocated_quanta(workers(SPMD0)))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="cpu",
                resources={"CPU": 4, "memory": 4096},
            ),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.allocated_quanta == {"metrics": 1}
    assert plan.assignments[0].resources == {
        "CPU": Decimal(2),
        "memory": Decimal(1024),
    }
    assert plan.assignments[0].ray_component_limits == {
        "CPU": Decimal(2),
        "memory": Decimal(2048),
    }
    assert plan.ray_component_limit_resources() == {
        "cpu": {"CPU": Decimal(2), "memory": Decimal(2048)}
    }
    assert plan.bundle_resources() == {
        "cpu": {"CPU": Decimal(2), "memory": Decimal(1024)}
    }
    assert plan.unused_resources == {"CPU": Decimal(0), "memory": Decimal(1024)}


def test_ray_component_limits_is_per_worker_and_immutable() -> None:
    profile = ResourceProfile(base={"CPU": 1}, ray_component_limits={"CPU": "0.5"})
    graph = placement_graph({"metrics": graph_node("metrics", SPMD0, 2, profile)})
    recipe = PlacementRecipe(
        recipe_id="ray-component-limits-workers",
        version="1.0.0",
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 3}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert [assignment.ray_component_limits for assignment in plan.assignments] == [
        {"CPU": Decimal("0.5")},
        {"CPU": Decimal("0.5")},
    ]
    assert plan.ray_component_limit_resources() == {"cpu": {"CPU": Decimal(1)}}
    assert plan.bundle_resources() == {"cpu": {"CPU": Decimal(2)}}
    assert profile.ray_component_limits == {"CPU": Decimal("0.5")}
