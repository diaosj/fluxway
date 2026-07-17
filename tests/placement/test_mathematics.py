from decimal import Decimal

import pytest
from pydantic import ValidationError
import fluxway

from fluxway.placement import (
    AFFINITY_FIRST_V1,
    PlacementRecipe,
    ClusterNode,
    ClusterSnapshot,
    ResourceProfile,
    ResourceScale,
    cardinality,
    disjoint,
    equal,
    lexicographic,
    minimize,
    rank,
    resource_fragmentation,
    subseteq,
    support,
    workers,
    weighted_dominant_share,
)


class SPMD0: ...


class SPMD1: ...


class SPMD5: ...


def test_recipe_preserves_direct_spmd_class_references() -> None:
    s0 = workers(SPMD0)
    s1 = workers(SPMD1)
    s5 = workers(SPMD5)
    constraints = (
        subseteq(support(rank(s5, 0)), support(s0)),
        subseteq(support(rank(s5, 1)), support(s1)),
        equal(cardinality(support(s0)), 1),
    )
    recipe = PlacementRecipe(
        recipe_id="formal-placement",
        version="1.0.0",
        constraints=constraints,
        objective=lexicographic(minimize(cardinality(support(s0)))),
    )

    assert constraints[0].left.workers.source.spmd.spmd_class is SPMD5
    assert constraints[0].right.workers.spmd.spmd_class is SPMD0
    assert hash(recipe) == hash(recipe)


def test_resource_vectors_reject_inexact_float() -> None:
    with pytest.raises(ValidationError, match="Decimal"):
        ResourceProfile(base={"CPU": 0.5})


def test_cluster_resources_share_resource_profile_validation() -> None:
    with pytest.raises(ValidationError):
        ClusterNode(node_id="node", resources={"CPU": True})


def test_resource_fragmentation_is_a_hard_constraint_scalar() -> None:
    constraint = equal(resource_fragmentation(), 0)

    assert constraint.left.kind == "resource_fragmentation"


def test_weighted_dominant_share_is_not_a_hard_constraint_scalar() -> None:
    with pytest.raises((TypeError, ValidationError)):
        equal(weighted_dominant_share(workers(SPMD0)), 0)  # type: ignore[arg-type]


def test_ast_is_frozen() -> None:
    expression = support(workers(SPMD0))

    with pytest.raises(ValidationError):
        expression.kind = "changed"  # type: ignore[misc]


def test_resource_vectors_are_deeply_frozen() -> None:
    profile = ResourceProfile(
        base={"CPU": 1},
        scale=ResourceScale(step={"CPU": 1}, max_steps=1, weight=1),
    )

    with pytest.raises(TypeError):
        profile.base["CPU"] = Decimal("2")  # type: ignore[index]
    assert profile.scale is not None
    with pytest.raises(TypeError):
        profile.scale.step["CPU"] = Decimal("2")  # type: ignore[index]

    assert profile.base["CPU"] == 1


def test_cluster_snapshot_is_canonical_under_node_permutation() -> None:
    left = ClusterNode(node_id="left", resources={"CPU": 1})
    right = ClusterNode(node_id="right", resources={"CPU": 1})

    forward = ClusterSnapshot.from_nodes((left, right))
    reverse = ClusterSnapshot.from_nodes((right, left))

    assert forward == reverse


def test_standard_objective_is_a_typed_ast() -> None:
    assert [objective.kind for objective in AFFINITY_FIRST_V1.objectives] == [
        "max_min",
        "max_min",
        "minimize",
        "maximize_majorization",
        "maximize",
        "maximize",
        "minimize",
    ]
    assert AFFINITY_FIRST_V1.objectives[0].expression.kind == (
        "weighted_dominant_share"
    )
    assert AFFINITY_FIRST_V1.objectives[-1].expression.kind == (
        "resource_fragmentation"
    )


def test_formal_placement_language_is_exported_at_package_root() -> None:
    assert fluxway.PlacementRecipe is PlacementRecipe
    assert fluxway.support is support
    assert fluxway.subseteq is subseteq
    assert fluxway.AFFINITY_FIRST_V1 is AFFINITY_FIRST_V1


def test_recipe_preserves_expert_constraint_order() -> None:
    s0 = support(workers(SPMD0))
    s1 = support(workers(SPMD1))
    first = subseteq(s0, s1)
    second = disjoint(s0, s1)
    common = {
        "recipe_id": "canonical",
        "version": "1.0.0",
        "objective": AFFINITY_FIRST_V1,
    }
    left = PlacementRecipe(**common, constraints=(first, second, first))
    assert left.constraints == (first, second, first)
