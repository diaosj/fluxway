import pytest
from collections.abc import Mapping
from decimal import Decimal

from fluxway.common.mesh import DeviceMesh
from fluxway.common.compiled_graph import SPMDGraphSpec, compile_graph
from fluxway.common.types import (
    Graph,
    Node,
    SPMDConfig,
)
from fluxway.placement.inventory import PlacementInventory
from fluxway.placement import (
    AFFINITY_FIRST,
    affinity_constraints,
    affinity_coverage,
    co_location_mass,
    all_workers,
    any_of,
    ClusterNode,
    ClusterSnapshot,
    PlacementContractError,
    PlacementInfeasibleError,
    PlacementPlanner,
    PlacementRecipe,
    ResourceProfile,
    ResourceScale,
    allocated_quanta,
    cardinality,
    disjoint,
    equal,
    less_equal,
    rank,
    resource_capacity,
    resource_fragmentation,
    resource_usage,
    lexicographic,
    max_min,
    maximize,
    maximize_majorization,
    minimize,
    negate,
    subseteq,
    support,
    target_occupancy,
    weighted_dominant_share,
    workers,
)


class SPMD0: ...


class SPMD1: ...


class SPMD2: ...


class SPMD3: ...


class SPMD4: ...


class SPMD5: ...


def graph_node(
    spmd_id: str,
    spmd_class: type,
    world_size: int,
    resource_profile: ResourceProfile | None = None,
) -> Node:
    return Node(
        name=spmd_id,
        spmd_config=SPMDConfig(
            spmd_id=spmd_id,
            mesh=DeviceMesh(("data",), device_shape=(world_size,)),
        ),
        spmd_class=spmd_class,
        resource_profile=resource_profile or ResourceProfile(base={"CPU": 1}),
    )


def placement_graph(
    nodes: Mapping[str, Node],
    *,
    name: str = "placement-test",
) -> Graph:
    graph_nodes = dict(nodes)
    compiled_graph = compile_graph(
        name=name,
        spmds=tuple(
            SPMDGraphSpec.create(
                spmd_id=spmd_id,
                spmd_class=node.spmd_class,
                mesh=node.spmd_config.mesh,
                resource_profile=node.resource_profile,  # type: ignore[arg-type]
                operators={},
            )
            for spmd_id, node in sorted(graph_nodes.items())
        ),
        dependencies=(),
    )
    return Graph(name=name, nodes=graph_nodes, compiled_graph=compiled_graph)


def placement_inventory(graph: Graph) -> PlacementInventory:
    return PlacementInventory.from_compiled_graph(graph.require_compiled_graph())


def scenario(s4_world_size: int) -> tuple[Graph, PlacementRecipe, ClusterSnapshot]:
    graph_nodes = {
        "s0": graph_node("s0", SPMD0, 4, ResourceProfile(base={"GPU": 1, "CPU": 1})),
        "s1": graph_node("s1", SPMD1, 12, ResourceProfile(base={"GPU": 1, "CPU": 1})),
        "s2": graph_node(
            "s2",
            SPMD2,
            1,
            ResourceProfile(
                base={"CPU": 1},
                scale=ResourceScale(step={"CPU": 1}, max_steps=100, weight=1),
            ),
        ),
        "s3": graph_node(
            "s3",
            SPMD3,
            1,
            ResourceProfile(
                base={"CPU": 1},
                scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=1),
            ),
        ),
        "s4": graph_node("s4", SPMD4, s4_world_size),
        "s5": graph_node("s5", SPMD5, 2),
    }
    graph = placement_graph(graph_nodes, name="placement")
    s0 = workers(SPMD0)
    s1 = workers(SPMD1)
    s3 = workers(SPMD3)
    s4 = workers(SPMD4)
    s5 = workers(SPMD5)
    recipe = PlacementRecipe(
        recipe_id="six-spmd",
        constraints=(
            subseteq(support(s3), support(s0)),
            subseteq(support(s4), support(s1)),
            subseteq(support(rank(s5, 0)), support(s0)),
            subseteq(support(rank(s5, 1)), support(s1)),
        ),
        objective=AFFINITY_FIRST,
    )
    nodes = (
        ClusterNode(node_id="gpu-0", resources={"CPU": 16, "GPU": 8}),
        ClusterNode(node_id="gpu-1", resources={"CPU": 16, "GPU": 8}),
        *(
            ClusterNode(node_id=f"cpu-{index}", resources={"CPU": 16})
            for index in range(4)
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(nodes)
    return graph, recipe, snapshot


@pytest.mark.parametrize("s4_world_size", [1, 2])
def test_planner_resolves_six_spmd_affinity(s4_world_size: int) -> None:
    graph, recipe, snapshot = scenario(s4_world_size)

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert sorted(plan.worker_counts_by_node("s1"), reverse=True) == [8, 4]
    assert plan.worker_counts_by_node("s0") == [4]
    assert len(plan.node_ids_for("s3")) == 1
    assert plan.node_ids_for("s3") == plan.node_ids_for("s0")
    assert len(plan.node_ids_for("s4")) == s4_world_size
    assert plan.allocated_quanta["s3"] == 2
    assert plan.allocated_quanta["s2"] == 15
    assert plan.recipe == recipe
    assert all(result.solve_duration > 0 for result in plan.objective_results)
    assert plan.objective_results[0].optimum_value == (2,)
    assert plan.objective_results[0].value_scale == 76 - s4_world_size
    assert plan.objective_results[1].optimum_value == (15,)
    assert plan.objective_results[1].value_scale == 74 - s4_world_size
    assert isinstance(plan.objective_results[3].optimum_value, tuple)
    assert tuple(sorted(plan.objective_results[3].optimum_value, reverse=True)) == (
        plan.objective_results[3].optimum_value
    )


def test_planner_places_single_s4_worker_on_dense_s1_node() -> None:
    graph, recipe, snapshot = scenario(1)

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    s4_node = next(iter(plan.node_ids_for("s4")))
    assert plan.worker_count("s1", s4_node) == 8


def test_resource_profile_applies_to_multiple_instances_of_same_class() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", SPMD2, 1),
            "right": graph_node("right", SPMD2, 1),
        }
    )
    recipe = PlacementRecipe(
        recipe_id="shared-profile",
        constraints=(equal(cardinality(support(workers(SPMD2))), 1),),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 2}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert {assignment.spmd_id for assignment in plan.assignments} == {
        "left",
        "right",
    }
    assert plan.support["left"] == plan.support["right"]


def test_planner_rejects_nonexclusive_requested_resources() -> None:
    graph = placement_graph(
        nodes={"gpu": graph_node("gpu", SPMD0, 8, ResourceProfile(base={"GPU": 1}))}
    )
    recipe = PlacementRecipe(
        recipe_id="exclusive-cluster",
        constraints=(),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot(
        nodes=(
            ClusterNode(node_id="gpu-0", resources={"GPU": 8}),
            ClusterNode(node_id="gpu-1", resources={"GPU": 8}),
        ),
        available={"GPU": 8},
    )

    with pytest.raises(PlacementContractError, match="exclusive"):
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )


def test_planner_compiles_resource_and_set_constraints() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", SPMD0, 1),
            "right": graph_node("right", SPMD1, 1),
        }
    )
    left = workers(SPMD0)
    right = workers(SPMD1)
    recipe = PlacementRecipe(
        recipe_id="constraint-algebra",
        constraints=(
            disjoint(support(left), support(right)),
            equal(cardinality(support(left)), 1),
            less_equal(
                resource_usage(left, "CPU"),
                resource_capacity(support(left), "CPU"),
            ),
            equal(allocated_quanta(left), 0),
        ),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="cpu-0", resources={"CPU": 1}),
            ClusterNode(node_id="cpu-1", resources={"CPU": 1}),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.node_ids_for("left").isdisjoint(plan.node_ids_for("right"))


def test_planner_minimizes_recipe_resource_quantum_fragmentation() -> None:
    graph = placement_graph(
        nodes={
            "fractional": graph_node(
                "fractional",
                SPMD0,
                2,
                ResourceProfile(base={"CPU": Decimal("0.6")}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="fragmentation",
        constraints=(),
        objective=lexicographic(minimize(resource_fragmentation())),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="cpu-0", resources={"CPU": 1}),
            ClusterNode(node_id="cpu-1", resources={"CPU": 1}),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == 4
    assert plan.objective_results[0].value_scale == 3
    assert plan.resource_fragmentation == {
        "cpu-0": {"CPU": Decimal("0.4")},
        "cpu-1": {"CPU": Decimal("0.4")},
    }


def test_fragmentation_preserves_unusable_multidimensional_resources() -> None:
    graph = placement_graph(
        nodes={
            "worker": graph_node(
                "worker", SPMD0, 1, ResourceProfile(base={"CPU": 2, "GPU": 1})
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="multidimensional-fragmentation",
        constraints=(),
        objective=lexicographic(minimize(resource_fragmentation())),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="mixed", resources={"CPU": 2, "GPU": 2}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == 1
    assert plan.resource_fragmentation == {
        "mixed": {"CPU": Decimal("0"), "GPU": Decimal("1")}
    }


def test_planner_accepts_resource_scalar_as_custom_objective() -> None:
    graph = placement_graph(
        nodes={
            "worker": graph_node(
                "worker",
                SPMD0,
                1,
                ResourceProfile(base={"CPU": Decimal("1.5")}),
            )
        }
    )
    worker = workers(SPMD0)
    recipe = PlacementRecipe(
        recipe_id="resource-objective",
        constraints=(),
        objective=lexicographic(minimize(resource_usage(worker, "CPU"))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 2}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == 3


def test_fragmentation_normalizes_memory_by_allocation_quantum() -> None:
    gib = 1 << 30
    graph = placement_graph(
        nodes={
            "worker": graph_node(
                "worker",
                SPMD0,
                1,
                ResourceProfile(base={"GPU": 1, "memory": gib}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="memory-fragmentation",
        constraints=(),
        objective=lexicographic(minimize(resource_fragmentation())),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="mixed",
                resources={"GPU": 1, "memory": 2 * gib},
            ),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == 1
    assert plan.objective_results[0].value_scale == 1
    assert plan.resource_fragmentation == {
        "mixed": {"GPU": Decimal(0), "memory": Decimal(gib)}
    }


def test_weighted_drf_matches_finite_leximin_oracle() -> None:
    graph = placement_graph(
        nodes={
            "weighted": graph_node(
                "weighted",
                SPMD0,
                1,
                ResourceProfile(
                    base={},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=3, weight=2),
                ),
            ),
            "ordinary": graph_node(
                "ordinary",
                SPMD1,
                1,
                ResourceProfile(
                    base={},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=3, weight=1),
                ),
            ),
        }
    )
    recipe = PlacementRecipe(
        recipe_id="weighted-drf-oracle",
        constraints=(),
        objective=lexicographic(max_min(weighted_dominant_share(all_workers()))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 3}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.allocated_quanta == {"ordinary": 1, "weighted": 2}
    assert plan.objective_results[0].optimum_value == (2, 2)
    assert plan.objective_results[0].value_scale == 6


def test_majorization_optimum_is_locked_before_later_objectives() -> None:
    graph = placement_graph(
        nodes={
            "source": graph_node("source", SPMD0, 1),
            "target": graph_node("target", SPMD1, 4),
        }
    )
    source = workers(SPMD0)
    target = workers(SPMD1)
    relations = affinity_constraints()
    recipe = PlacementRecipe(
        recipe_id="majorization-oracle",
        constraints=(subseteq(support(source), support(target)),),
        objective=lexicographic(
            maximize_majorization(target_occupancy(relations)),
            maximize(affinity_coverage(relations)),
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="cpu-0", resources={"CPU": 3}),
            ClusterNode(node_id="cpu-1", resources={"CPU": 3}),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == (3, 1)
    assert sorted(plan.occupancy["target"].values(), reverse=True) == [3, 1]


def test_affinity_graph_collects_relations_nested_in_boolean_constraints() -> None:
    graph = placement_graph(
        nodes={
            "elastic": graph_node(
                "elastic",
                SPMD0,
                1,
                ResourceProfile(
                    base={"CPU": 1},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=1),
                ),
            ),
            "target": graph_node("target", SPMD1, 1),
        }
    )
    elastic = workers(SPMD0)
    target = workers(SPMD1)
    relation = subseteq(support(elastic), support(target))
    recipe = PlacementRecipe(
        recipe_id="nested-affinity",
        constraints=(any_of(relation, negate(negate(relation))),),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 4}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == (2,)
    assert plan.objective_results[1].optimum_value == ()


def test_negated_subset_is_not_a_positive_affinity_relation() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node(
                "left",
                SPMD0,
                1,
                ResourceProfile(
                    base={"CPU": 1},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=1),
                ),
            ),
            "anchor": graph_node("anchor", SPMD1, 1),
            "right": graph_node(
                "right",
                SPMD2,
                1,
                ResourceProfile(
                    base={"CPU": 1},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=1),
                ),
            ),
        }
    )
    recipe = PlacementRecipe(
        recipe_id="negative-relation",
        constraints=(
            negate(
                subseteq(
                    support(workers(SPMD0)),
                    support(workers(SPMD1)),
                )
            ),
        ),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="cpu-0", resources={"CPU": 3}),
            ClusterNode(node_id="cpu-1", resources={"CPU": 2}),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.allocated_quanta == {"anchor": 0, "left": 1, "right": 1}
    assert plan.objective_results[0].optimum_value == ()
    assert plan.objective_results[1].optimum_value == (1, 1)


def test_affinity_scalar_constraints_are_compiled_and_attested() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", SPMD0, 2),
            "right": graph_node("right", SPMD1, 1),
        }
    )
    relation = subseteq(support(workers(SPMD0)), support(workers(SPMD1)))
    recipe = PlacementRecipe(
        recipe_id="affinity-scalar-constraints",
        constraints=(
            relation,
            equal(affinity_coverage(affinity_constraints()), 1),
            equal(co_location_mass(affinity_constraints()), 2),
        ),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="node-0", resources={"CPU": 3}),
            ClusterNode(node_id="node-1", resources={"CPU": 3}),
        )
    )
    planner = PlacementPlanner(solve_timeout_s=10)

    inventory = placement_inventory(graph)
    plan = planner.resolve(inventory, recipe, snapshot)
    actual_node_ids = {
        spmd_id: tuple(
            assignment.node_id
            for assignment in plan.assignments
            if assignment.spmd_id == spmd_id
        )
        for spmd_id in graph.nodes
    }
    attested = planner.attest(inventory, recipe, snapshot, plan, actual_node_ids)

    assert all(result.satisfied for result in plan.constraint_results)
    assert all(result.satisfied for result in attested.constraint_results)


def test_resource_fragmentation_constraint_uses_exact_normalized_value() -> None:
    graph = placement_graph(
        nodes={
            "fragment": graph_node(
                "fragment",
                SPMD0,
                1,
                ResourceProfile(base={"CPU": 2, "GPU": 1}),
            )
        }
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 2, "GPU": 2}),)
    )

    def recipe(expected: int) -> PlacementRecipe:
        return PlacementRecipe(
            recipe_id=f"fragmentation-equals-{expected}",
            constraints=(equal(resource_fragmentation(), expected),),
            objective=lexicographic(minimize(resource_fragmentation())),
        )

    planner = PlacementPlanner(solve_timeout_s=10)
    with pytest.raises(PlacementInfeasibleError):
        planner.resolve(placement_inventory(graph), recipe(0), snapshot)

    plan = planner.resolve(placement_inventory(graph), recipe(1), snapshot)

    assert plan.resource_fragmentation_score == 1
    assert plan.resource_fragmentation_scale == 1
    assert plan.constraint_results[0].satisfied is True


@pytest.mark.parametrize(
    ("resource", "worker_quantity", "node_quantity"),
    [("CPU", 2, 1), ("GPU", 1, 0)],
)
def test_infeasible_error_reports_resource_deficit(
    resource: str, worker_quantity: int, node_quantity: int
) -> None:
    graph = placement_graph(
        nodes={
            "worker": graph_node(
                "worker",
                SPMD0,
                1,
                ResourceProfile(base={resource: worker_quantity}),
            )
        }
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={resource: node_quantity}),)
    )
    recipe = PlacementRecipe(
        recipe_id="resource-deficit",
        constraints=(),
        objective=AFFINITY_FIRST,
    )

    with pytest.raises(PlacementInfeasibleError) as error:
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )

    message = str(error.value)
    assert "cause=minimum resource capacity deficit" in message
    assert f"{resource}={worker_quantity - node_quantity}" in message
    assert "hint=increase cluster resources" in message
    assert "workers=worker[count=1" in message


def test_infeasible_error_reports_single_worker_node_limit() -> None:
    graph = placement_graph(
        nodes={
            "worker": graph_node("worker", SPMD0, 2, ResourceProfile(base={"CPU": 3}))
        }
    )
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(node_id=f"node-{index}", resources={"CPU": 2})
            for index in range(3)
        )
    )
    recipe = PlacementRecipe(
        recipe_id="single-worker-limit",
        constraints=(),
        objective=AFFINITY_FIRST,
    )

    with pytest.raises(PlacementInfeasibleError) as error:
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )

    message = str(error.value)
    assert "deficits=none" in message
    assert "single_worker_limits=CPU(worker=3,node=2)" in message
    assert "cause=hard placement constraint or scaled-resource infeasibility" in message


def test_infeasible_error_identifies_hard_constraint_when_resources_fit() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", SPMD0, 1, ResourceProfile(base={"CPU": 1})),
            "right": graph_node("right", SPMD1, 1, ResourceProfile(base={"CPU": 1})),
        }
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 2}),)
    )
    recipe = PlacementRecipe(
        recipe_id="hard-constraint",
        constraints=(disjoint(support(workers(SPMD0)), support(workers(SPMD1))),),
        objective=AFFINITY_FIRST,
    )

    with pytest.raises(PlacementInfeasibleError) as error:
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )

    message = str(error.value)
    assert "deficits=none" in message
    assert "single_worker_limits=none" in message
    assert "cause=hard placement constraint or scaled-resource infeasibility" in message
    assert "hint=review affinity/disjoint/equality constraints" in message
