from decimal import Decimal
from itertools import product
from typing import cast

import pytest
from ortools.sat.python import cp_model

from fluxway.common.types import Graph
from fluxway.placement import (
    AFFINITY_FIRST,
    ClusterNode,
    ClusterSnapshot,
    affinity_constraints,
    affinity_coverage,
    PlacementInfeasibleError,
    PlacementContractError,
    PlacementPlanner,
    PlacementRecipe,
    PlacementSolveTimeoutError,
    ResourceProfile,
    ResourceScale,
    NodeSetExpr,
    all_of,
    all_workers,
    allocated_quanta,
    any_of,
    cardinality,
    co_location_mass,
    difference,
    disjoint,
    equal,
    instance,
    intersection,
    less_equal,
    lexicographic,
    minimize,
    maximize,
    maximize_majorization,
    negate,
    occupancy,
    rank,
    ranks,
    resource_capacity,
    resource_fragmentation,
    resource_usage,
    subseteq,
    support,
    target_occupancy,
    union,
    workers,
    weighted_dominant_share,
    max_min,
)

from .test_planner import graph_node, placement_graph, placement_inventory


class LeftSPMD: ...


class RightSPMD: ...


class PropertySPMD: ...


class NodeZeroAnchor: ...


class NodeOneAnchor: ...


class FragmentationSPMD: ...


def test_constraint_algebra_matches_exhaustive_assignment_oracle() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", LeftSPMD, 1),
            "right": graph_node("right", RightSPMD, 1),
            "node-zero": graph_node(
                "node-zero",
                NodeZeroAnchor,
                1,
                ResourceProfile(base={"anchor:zero": 1}),
            ),
            "node-one": graph_node(
                "node-one",
                NodeOneAnchor,
                1,
                ResourceProfile(base={"anchor:one": 1}),
            ),
        }
    )
    left = union(instance("left"), ranks(workers(LeftSPMD), [0]))
    right = rank(workers(RightSPMD), 0)
    left_nodes = support(occupancy(left))
    right_nodes = support(right)
    anchor_nodes = (
        support(workers(NodeZeroAnchor)),
        support(workers(NodeOneAnchor)),
    )
    all_nodes = cast(NodeSetExpr, union(left_nodes, right_nodes))
    cases = (
        (
            subseteq(left_nodes, right_nodes),
            lambda a, b: {a} <= {b},
        ),
        (
            disjoint(left_nodes, right_nodes),
            lambda a, b: {a}.isdisjoint({b}),
        ),
        (
            all_of(
                subseteq(left_nodes, right_nodes),
                disjoint(left_nodes, right_nodes),
            ),
            lambda a, b: {a} <= {b} and {a}.isdisjoint({b}),
        ),
        (
            negate(
                any_of(
                    equal(left_nodes, right_nodes),
                    disjoint(left_nodes, right_nodes),
                )
            ),
            lambda a, b: not ({a} == {b} or {a}.isdisjoint({b})),
        ),
        (
            less_equal(cardinality(all_nodes), 1),
            lambda a, b: len({a, b}) <= 1,
        ),
        (
            equal(
                intersection(left_nodes, right_nodes),
                difference(all_nodes, union(left_nodes, right_nodes)),
            ),
            lambda a, b: {a}.isdisjoint({b}),
        ),
        (
            all_of(
                less_equal(
                    resource_usage(all_workers(), "CPU"),
                    resource_capacity(all_nodes, "CPU"),
                ),
                equal(allocated_quanta(all_workers()), 0),
            ),
            lambda _a, _b: True,
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="node-0",
                resources={"CPU": 2, "anchor:zero": 1},
            ),
            ClusterNode(
                node_id="node-1",
                resources={"CPU": 2, "anchor:one": 1},
            ),
        )
    )
    for case_index, (constraint, oracle) in enumerate(cases):
        for left_node, right_node in product(range(2), repeat=2):
            recipe = PlacementRecipe(
                recipe_id=(f"constraint-oracle-{case_index}-{left_node}-{right_node}"),
                constraints=(
                    equal(left_nodes, anchor_nodes[left_node]),
                    equal(right_nodes, anchor_nodes[right_node]),
                    constraint,
                ),
                objective=lexicographic(minimize(cardinality(support(all_workers())))),
            )
            try:
                PlacementPlanner(solve_timeout_s=10).resolve(
                    placement_inventory(graph), recipe, snapshot
                )
            except PlacementInfeasibleError:
                solver_value = False
            else:
                solver_value = True
            assert solver_value is oracle(left_node, right_node)


def test_affinity_objectives_include_positive_equal_relations() -> None:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", LeftSPMD, 1),
            "right": graph_node("right", RightSPMD, 1),
            "node-zero": graph_node(
                "node-zero",
                NodeZeroAnchor,
                1,
                ResourceProfile(base={"anchor:zero": 1}),
            ),
            "node-one": graph_node(
                "node-one",
                NodeOneAnchor,
                1,
                ResourceProfile(base={"anchor:one": 1}),
            ),
        }
    )
    left_nodes = support(workers(LeftSPMD))
    right_nodes = support(workers(RightSPMD))
    anchors = (
        support(workers(NodeZeroAnchor)),
        support(workers(NodeOneAnchor)),
    )
    relations = affinity_constraints()
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="node-0",
                resources={"CPU": 2, "anchor:zero": 1},
            ),
            ClusterNode(
                node_id="node-1",
                resources={"CPU": 2, "anchor:one": 1},
            ),
        )
    )
    for left_node, right_node in product(range(2), repeat=2):
        recipe = PlacementRecipe(
            recipe_id=f"affinity-oracle-{left_node}-{right_node}",
            constraints=(
                equal(left_nodes, anchors[left_node]),
                equal(right_nodes, anchors[right_node]),
            ),
            objective=lexicographic(
                maximize(affinity_coverage(relations)),
                maximize(co_location_mass(relations)),
            ),
        )

        plan = PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )

        assert plan.objective_results[0].optimum_value == 2
        assert plan.objective_results[1].optimum_value == 2


@pytest.mark.parametrize(
    ("capacity", "expected_fragmentation"),
    (
        ({"CPU": 2, "GPU": 2}, Decimal("1")),
        ({"CPU": 4, "GPU": 3}, Decimal("1")),
        ({"CPU": 5, "GPU": 1}, Decimal("3")),
    ),
)
def test_fragmentation_matches_finite_multidimensional_packing_oracle(
    capacity: dict[str, int], expected_fragmentation: Decimal
) -> None:
    graph = placement_graph(
        nodes={
            "fragment": graph_node(
                "fragment",
                FragmentationSPMD,
                1,
                ResourceProfile(base={"CPU": 2, "GPU": 1}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id=f"fragmentation-oracle-{capacity['CPU']}-{capacity['GPU']}",
        constraints=(),
        objective=lexicographic(minimize(resource_fragmentation())),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources=capacity),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert sum(plan.resource_fragmentation["node"].values()) == (expected_fragmentation)


@pytest.mark.parametrize(
    ("worker_count", "node_count"),
    ((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)),
)
def test_assignment_is_total_and_preserves_resources(
    worker_count: int, node_count: int
) -> None:
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                worker_count,
                ResourceProfile(base={"CPU": Decimal("0.5")}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id=f"resource-property-{worker_count}-{node_count}",
        constraints=(),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(node_id=f"node-{index}", resources={"CPU": 1})
            for index in range(node_count)
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert {
        (assignment.spmd_id, assignment.rank) for assignment in plan.assignments
    } == {("property", rank_index) for rank_index in range(worker_count)}
    for node in snapshot.nodes:
        used = sum(
            (
                assignment.resources.get("CPU", Decimal(0))
                for assignment in plan.assignments
                if assignment.node_id == node.node_id
            ),
            Decimal(0),
        )
        assert used <= node.resources["CPU"]
    assert (
        sum(
            (
                assignment.resources.get("CPU", Decimal(0))
                for assignment in plan.assignments
            ),
            Decimal(0),
        )
        <= snapshot.available["CPU"]
    )


def test_solver_call_count_does_not_scale_with_homogeneous_workers(
    monkeypatch,
) -> None:
    worker_count = 64
    graph = placement_graph(
        nodes={"property": graph_node("property", PropertySPMD, worker_count)}
    )
    recipe = PlacementRecipe(
        recipe_id="solver-call-bound",
        constraints=(),
        objective=AFFINITY_FIRST,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": worker_count}),)
    )
    solve = cp_model.CpSolver.solve
    solve_count = 0

    def counted_solve(self, model, solution_callback=None):
        nonlocal solve_count
        solve_count += 1
        return solve(self, model, solution_callback)

    monkeypatch.setattr(cp_model.CpSolver, "solve", counted_solve)

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert len(plan.assignments) == worker_count
    assert solve_count <= 8


def test_lexicographic_program_uses_one_decreasing_solve_budget(
    monkeypatch,
) -> None:
    graph = placement_graph(nodes={"property": graph_node("property", PropertySPMD, 1)})
    recipe = PlacementRecipe(
        recipe_id="single-deadline",
        constraints=(),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1}),)
    )
    elapsed = 0.0
    budgets: list[float] = []
    solve = cp_model.CpSolver.solve

    def monotonic() -> float:
        return elapsed

    def timed_solve(self, model, solution_callback=None):
        nonlocal elapsed
        budgets.append(self.parameters.max_time_in_seconds)
        elapsed += 0.1
        return solve(self, model, solution_callback)

    monkeypatch.setattr("fluxway.placement.planner.time.monotonic", monotonic)
    monkeypatch.setattr(cp_model.CpSolver, "solve", timed_solve)

    PlacementPlanner(solve_timeout_s=0.5).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert budgets == pytest.approx([0.5, 0.4])


def test_lexicographic_program_stops_when_shared_deadline_expires(
    monkeypatch,
) -> None:
    graph = placement_graph(nodes={"property": graph_node("property", PropertySPMD, 1)})
    recipe = PlacementRecipe(
        recipe_id="expired-deadline",
        constraints=(),
        objective=lexicographic(
            minimize(cardinality(support(all_workers()))),
            minimize(cardinality(support(all_workers()))),
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1}),)
    )
    elapsed = 0.0
    solve_count = 0
    solve = cp_model.CpSolver.solve

    def monotonic() -> float:
        return elapsed

    def timed_solve(self, model, solution_callback=None):
        nonlocal elapsed, solve_count
        solve_count += 1
        elapsed += 0.3
        return solve(self, model, solution_callback)

    monkeypatch.setattr("fluxway.placement.planner.time.monotonic", monotonic)
    monkeypatch.setattr(cp_model.CpSolver, "solve", timed_solve)

    with pytest.raises(PlacementSolveTimeoutError, match="deadline"):
        PlacementPlanner(solve_timeout_s=0.5).resolve(
            placement_inventory(graph), recipe, snapshot
        )

    assert solve_count == 2


def test_feasible_incumbent_without_optimality_is_a_timeout(monkeypatch) -> None:
    graph = placement_graph(nodes={"property": graph_node("property", PropertySPMD, 1)})
    recipe = PlacementRecipe(
        recipe_id="feasible-incumbent",
        constraints=(),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1}),)
    )
    solve = cp_model.CpSolver.solve
    solve_count = 0

    def feasible_after_base(self, model, solution_callback=None):
        nonlocal solve_count
        solve_count += 1
        if solve_count == 1:
            return solve(self, model, solution_callback)
        return cp_model.FEASIBLE

    monkeypatch.setattr(cp_model.CpSolver, "solve", feasible_after_base)

    with pytest.raises(PlacementSolveTimeoutError, match="timed out"):
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )


def _majorization_complexity_case() -> tuple[Graph, PlacementRecipe, ClusterSnapshot]:
    graph = placement_graph(
        nodes={
            "left": graph_node("left", LeftSPMD, 1),
            "right": graph_node("right", RightSPMD, 1),
        }
    )
    relation = subseteq(support(workers(LeftSPMD)), support(workers(RightSPMD)))
    recipe = PlacementRecipe(
        recipe_id="complexity-admission",
        constraints=(relation,),
        objective=lexicographic(
            maximize_majorization(target_occupancy(affinity_constraints()))
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(node_id=f"node-{index}", resources={"CPU": 2})
            for index in range(4)
        )
    )
    return graph, recipe, snapshot


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    (
        ("max_objective_components", 3, "objective components 4 exceed 3"),
        ("max_sorting_comparators", 5, "sorting comparators 6 exceed 5"),
        ("max_lexicographic_solves", 3, "lexicographic solves 4 exceed 3"),
    ),
)
def test_complexity_is_rejected_before_model_construction(
    monkeypatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    graph, recipe, snapshot = _majorization_complexity_case()
    limits = {
        "max_objective_components": 100,
        "max_sorting_comparators": 100,
        "max_lexicographic_solves": 100,
        limit_name: limit,
    }
    planner = PlacementPlanner(solve_timeout_s=10, **limits)  # type: ignore

    def model_construction_must_not_start(*args, **kwargs):
        raise AssertionError("CP-SAT model construction started before admission")

    monkeypatch.setattr(planner, "_compile", model_construction_must_not_start)

    with pytest.raises(PlacementContractError, match=message):
        planner.resolve(placement_inventory(graph), recipe, snapshot)


def test_batcher_comparator_count_matches_power_of_two_boundaries() -> None:
    assert PlacementPlanner.sorting_comparator_count(0) == 0
    assert PlacementPlanner.sorting_comparator_count(1) == 0
    assert PlacementPlanner.sorting_comparator_count(4) == 6
    assert PlacementPlanner.sorting_comparator_count(1024) == 28160
    assert PlacementPlanner.sorting_comparator_count(1025) == 67584


def test_model_size_is_rejected_before_assignment_matrix_construction(
    monkeypatch,
) -> None:
    instance_count = 20
    node_count = 20
    graph = placement_graph(
        nodes={
            f"instance-{index}": graph_node(f"instance-{index}", PropertySPMD, 1)
            for index in range(instance_count)
        }
    )
    recipe = PlacementRecipe(
        recipe_id="model-variable-admission",
        constraints=(),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(node_id=f"node-{index}", resources={"CPU": instance_count})
            for index in range(node_count)
        )
    )
    planner = PlacementPlanner(solve_timeout_s=10, max_model_variables=1000)

    def model_construction_must_not_start(*args, **kwargs):
        raise AssertionError("CP-SAT model construction started before admission")

    monkeypatch.setattr(planner, "_compile", model_construction_must_not_start)

    with pytest.raises(PlacementContractError, match="model variables"):
        planner.resolve(placement_inventory(graph), recipe, snapshot)


def test_fractional_resource_scalar_uses_physical_quantity() -> None:
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                1,
                ResourceProfile(base={"CPU": Decimal("0.1")}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="fractional-resource-scalar",
        constraints=(equal(resource_usage(all_workers(), "CPU"), 1),),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1}),)
    )

    with pytest.raises(PlacementInfeasibleError):
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )


def test_scalar_comparison_rejects_incompatible_resource_dimensions() -> None:
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                1,
                ResourceProfile(base={"CPU": 1, "GPU": 1}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="incompatible-resource-dimensions",
        constraints=(
            equal(
                resource_usage(all_workers(), "CPU"),
                resource_usage(all_workers(), "GPU"),
            ),
        ),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1, "GPU": 1}),)
    )

    with pytest.raises(PlacementContractError, match="incompatible dimensions"):
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )


def test_max_min_accepts_occupancy_vector() -> None:
    graph = placement_graph(nodes={"property": graph_node("property", PropertySPMD, 2)})
    recipe = PlacementRecipe(
        recipe_id="max-min-occupancy",
        constraints=(),
        objective=lexicographic(max_min(occupancy(all_workers()))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(node_id="node-0", resources={"CPU": 2}),
            ClusterNode(node_id="node-1", resources={"CPU": 2}),
        )
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.objective_results[0].optimum_value == (1, 1)


def test_majorization_accepts_weighted_dominant_share_vector() -> None:
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                1,
                ResourceProfile(
                    base={"CPU": 1},
                    scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=1),
                ),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="majorize-weighted-share",
        constraints=(),
        objective=lexicographic(
            maximize_majorization(weighted_dominant_share(all_workers()))
        ),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 3}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.allocated_quanta["property"] == 2


def test_resource_integerization_supports_pib_scale_memory() -> None:
    pib = 1 << 50
    tib = 1 << 40
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                1,
                ResourceProfile(base={"memory": tib}),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="pib-memory",
        constraints=(),
        objective=lexicographic(minimize(cardinality(support(all_workers())))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"memory": pib}),)
    )

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    assert plan.assignments[0].resources["memory"] == Decimal(tib)


def test_drf_coefficients_reject_int64_overflow() -> None:
    graph = placement_graph(
        nodes={
            "property": graph_node(
                "property",
                PropertySPMD,
                1,
                ResourceProfile(
                    base={"CPU": 1},
                    scale=ResourceScale(
                        step={"CPU": 1},
                        max_steps=100,
                        weight=Decimal("1e-19"),
                    ),
                ),
            )
        }
    )
    selected = workers(PropertySPMD)
    recipe = PlacementRecipe(
        recipe_id="drf-overflow",
        constraints=(),
        objective=lexicographic(max_min(weighted_dominant_share(selected))),
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="node", resources={"CPU": 1000}),)
    )

    with pytest.raises(PlacementContractError, match="int64"):
        PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )
