from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from fluxway.resources import ResourceVector, normalize_resource_vector

from .mathematics import (
    ConstraintNode,
    PlacementModel,
    PlacementRecipe,
)


class PlacementError(RuntimeError):
    pass


class PlacementContractError(PlacementError):
    pass


class PlacementInfeasibleError(PlacementError):
    pass


class PlacementSolveTimeoutError(PlacementError):
    pass


class PlacementMaterializationError(PlacementError):
    pass


class PlacementGroupTimeoutError(PlacementMaterializationError):
    pass


class PlacementAttestationError(PlacementMaterializationError):
    pass


DEFAULT_MAX_OBJECTIVE_COMPONENTS = 1024
DEFAULT_MAX_SORTING_COMPARATORS = 32768
DEFAULT_MAX_LEXICOGRAPHIC_SOLVES = 2048
DEFAULT_MAX_MODEL_VARIABLES = 500_000
DEFAULT_MAX_MODEL_CONSTRAINTS = 1_000_000


class ClusterNode(PlacementModel):
    node_id: str = Field(min_length=1)
    resources: dict[str, Decimal]

    @field_validator("resources", mode="before")
    @classmethod
    def validate_resources(cls, value: Any) -> dict[str, Decimal]:
        return normalize_resource_vector(value)


class ClusterSnapshot(PlacementModel):
    nodes: tuple[ClusterNode, ...]
    available: dict[str, Decimal]

    @field_validator("available", mode="before")
    @classmethod
    def validate_available(cls, value: Any) -> dict[str, Decimal]:
        return normalize_resource_vector(value)

    @field_validator("nodes")
    @classmethod
    def canonicalize_nodes(
        cls, value: tuple[ClusterNode, ...]
    ) -> tuple[ClusterNode, ...]:
        return tuple(sorted(value, key=lambda node: node.node_id))

    @model_validator(mode="after")
    def validate_nodes(self) -> ClusterSnapshot:
        node_ids = [node.node_id for node in self.nodes]
        if not self.nodes:
            raise ValueError("cluster snapshot must contain at least one node")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("cluster node IDs must be unique")
        return self

    @classmethod
    def from_nodes(cls, nodes: Iterable[ClusterNode]) -> ClusterSnapshot:
        node_tuple = tuple(nodes)
        available: defaultdict[str, Decimal] = defaultdict(Decimal)
        for node in node_tuple:
            for resource, quantity in node.resources.items():
                available[resource] += quantity
        return cls(nodes=node_tuple, available=dict(available))


class PlacementRuntimeConfig(PlacementModel):
    solve_timeout_s: float = Field(default=30, gt=0)
    pg_timeout_s: float = Field(default=300, gt=0)
    driver_resources: ResourceVector = Field(default_factory=dict)
    max_objective_components: int = Field(
        default=DEFAULT_MAX_OBJECTIVE_COMPONENTS, gt=0
    )
    max_sorting_comparators: int = Field(default=DEFAULT_MAX_SORTING_COMPARATORS, gt=0)
    max_lexicographic_solves: int = Field(
        default=DEFAULT_MAX_LEXICOGRAPHIC_SOLVES, gt=0
    )
    max_model_variables: int = Field(default=DEFAULT_MAX_MODEL_VARIABLES, gt=0)
    max_model_constraints: int = Field(default=DEFAULT_MAX_MODEL_CONSTRAINTS, gt=0)

    @field_validator("driver_resources", mode="before")
    @classmethod
    def validate_driver_resources(cls, value: Any) -> dict[str, Decimal]:
        return normalize_resource_vector(value)


class WorkerAssignment(PlacementModel):
    spmd_id: str
    rank: int
    node_id: str
    resources: dict[str, Decimal]
    ray_component_limits: dict[str, Decimal] = Field(default_factory=dict)
    driver_resources: ResourceVector = Field(default_factory=dict)


class ObjectiveResult(PlacementModel):
    index: int
    kind: str
    optimum_value: int | tuple[int, ...]
    value_scale: int = 1
    solve_duration: float


class ConstraintResult(PlacementModel):
    index: int
    constraint: ConstraintNode
    satisfied: bool


class ResolvedPlacementPlan(PlacementModel):
    recipe_id: str
    recipe_version: str
    recipe: PlacementRecipe
    assignments: tuple[WorkerAssignment, ...]
    allocated_quanta: dict[str, int]
    objective_results: tuple[ObjectiveResult, ...]
    constraint_results: tuple[ConstraintResult, ...]
    occupancy: dict[str, dict[str, int]]
    support: dict[str, tuple[str, ...]]
    resource_fragmentation: dict[str, dict[str, Decimal]]
    resource_fragmentation_score: int
    resource_fragmentation_scale: int = Field(gt=0)
    unused_resources: dict[str, Decimal]

    def node_ids_for(self, spmd_id: str) -> frozenset[str]:
        return frozenset(
            assignment.node_id
            for assignment in self.assignments
            if assignment.spmd_id == spmd_id
        )

    def worker_count(self, spmd_id: str, node_id: str) -> int:
        return sum(
            assignment.spmd_id == spmd_id and assignment.node_id == node_id
            for assignment in self.assignments
        )

    def worker_counts_by_node(self, spmd_id: str) -> list[int]:
        return [
            self.worker_count(spmd_id, node_id)
            for node_id in sorted(self.node_ids_for(spmd_id))
        ]

    def bundle_resources(self) -> dict[str, dict[str, Decimal]]:
        bundles: dict[str, defaultdict[str, Decimal]] = {}
        for assignment in self.assignments:
            resources = bundles.setdefault(assignment.node_id, defaultdict(Decimal))
            for resource, quantity in assignment.resources.items():
                resources[resource] += quantity
            for resource, quantity in assignment.driver_resources.items():
                resources[resource] += quantity
        return {
            node_id: dict(sorted(resources.items()))
            for node_id, resources in sorted(bundles.items())
        }

    def ray_component_limit_resources(self) -> dict[str, dict[str, Decimal]]:
        limits: dict[str, defaultdict[str, Decimal]] = {}
        for assignment in self.assignments:
            resources = limits.setdefault(assignment.node_id, defaultdict(Decimal))
            for resource, quantity in assignment.ray_component_limits.items():
                resources[resource] += quantity
        return {
            node_id: dict(sorted(resources.items()))
            for node_id, resources in sorted(limits.items())
            if any(quantity > 0 for quantity in resources.values())
        }


class AllocationReport(ResolvedPlacementPlan):
    status: Literal["PLANNED", "PG_READY", "READY", "FAILED"]
    capacity: ClusterSnapshot
    attested_node_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    pg_id: str | None = None
    error: str | None = None

    @classmethod
    def from_plan(
        cls,
        plan: ResolvedPlacementPlan,
        capacity: ClusterSnapshot,
        *,
        status: Literal["PLANNED", "PG_READY", "READY", "FAILED"],
        attested_node_ids: Mapping[str, tuple[str, ...]] | None = None,
        pg_id: str | None = None,
        error: str | None = None,
    ) -> AllocationReport:
        return cls(
            **plan.model_dump(),
            status=status,
            capacity=capacity,
            attested_node_ids=dict(attested_node_ids or {}),
            pg_id=pg_id,
            error=error,
        )

    @classmethod
    def from_failure(
        cls,
        recipe: PlacementRecipe,
        capacity: ClusterSnapshot,
        error: Exception,
    ) -> AllocationReport:
        return cls(
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.version,
            recipe=recipe,
            assignments=(),
            allocated_quanta={},
            objective_results=(),
            constraint_results=(),
            occupancy={},
            support={},
            resource_fragmentation={},
            resource_fragmentation_score=0,
            resource_fragmentation_scale=1,
            unused_resources=dict(capacity.available),
            status="FAILED",
            capacity=capacity,
            error=str(error),
        )
