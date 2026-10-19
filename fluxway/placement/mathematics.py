from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    cast,
    overload,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PlacementModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )


@runtime_checkable
class SPMDRef(Protocol):
    @property
    def spmd_id(self) -> str: ...

    @property
    def spmd_class(self) -> type: ...


class SPMDClassRef(PlacementModel):
    spmd_class: type

    @classmethod
    def from_type(cls, value: type) -> SPMDClassRef:
        return cls(spmd_class=value)

    @property
    def module(self) -> str:
        return self.spmd_class.__module__

    @property
    def qualname(self) -> str:
        return self.spmd_class.__qualname__


class WorkerSetExpr(PlacementModel):
    kind: Any


class WorkersExpr(WorkerSetExpr):
    kind: Literal["workers"] = "workers"
    spmd: SPMDClassRef


class InstanceExpr(WorkerSetExpr):
    kind: Literal["instance"] = "instance"
    spmd_id: str


class AllWorkersExpr(WorkerSetExpr):
    kind: Literal["all_workers"] = "all_workers"


class AffinityConstraintSetExpr(PlacementModel):
    kind: Literal["affinity_constraints"] = "affinity_constraints"


class IncidentSPMDsExpr(WorkerSetExpr):
    kind: Literal["incident_spmds"] = "incident_spmds"
    relations: AffinityConstraintSetExpr


class IsolatedSPMDsExpr(WorkerSetExpr):
    kind: Literal["isolated_spmds"] = "isolated_spmds"
    relations: AffinityConstraintSetExpr


class RankExpr(WorkerSetExpr):
    kind: Literal["rank"] = "rank"
    source: WorkerSet
    index: int = Field(ge=0)


class RanksExpr(WorkerSetExpr):
    kind: Literal["ranks"] = "ranks"
    source: WorkerSet
    indices: tuple[int, ...]

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("ranks requires at least one index")
        if any(index < 0 for index in value):
            raise ValueError("rank indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("rank indices must be unique")
        return tuple(sorted(value))


class WorkerUnionExpr(WorkerSetExpr):
    kind: Literal["worker_union"] = "worker_union"
    items: tuple[WorkerSet, ...]


type WorkerSet = Annotated[
    WorkersExpr
    | InstanceExpr
    | AllWorkersExpr
    | IncidentSPMDsExpr
    | IsolatedSPMDsExpr
    | RankExpr
    | RanksExpr
    | WorkerUnionExpr,
    Field(discriminator="kind"),
]


class VectorExpr(PlacementModel):
    kind: Any


class OccupancyExpr(VectorExpr):
    kind: Literal["occupancy"] = "occupancy"
    workers: WorkerSet


class TargetOccupancyExpr(VectorExpr):
    kind: Literal["target_occupancy"] = "target_occupancy"
    relations: AffinityConstraintSetExpr


class WeightedDominantShareExpr(VectorExpr):
    kind: Literal["weighted_dominant_share"] = "weighted_dominant_share"
    workers: WorkerSet


type Vector = Annotated[
    OccupancyExpr | TargetOccupancyExpr | WeightedDominantShareExpr,
    Field(discriminator="kind"),
]


class NodeSetExpr(PlacementModel):
    kind: Any


class SupportExpr(NodeSetExpr):
    kind: Literal["support"] = "support"
    workers: WorkerSet


class NodeUnionExpr(NodeSetExpr):
    kind: Literal["node_union"] = "node_union"
    items: tuple[NodeSet, ...]


class NodeIntersectionExpr(NodeSetExpr):
    kind: Literal["node_intersection"] = "node_intersection"
    items: tuple[NodeSet, ...]


class NodeDifferenceExpr(NodeSetExpr):
    kind: Literal["node_difference"] = "node_difference"
    left: NodeSet
    right: NodeSet


type NodeSet = Annotated[
    SupportExpr | NodeUnionExpr | NodeIntersectionExpr | NodeDifferenceExpr,
    Field(discriminator="kind"),
]


class ScalarExpr(PlacementModel):
    kind: Any


class IntegerExpr(ScalarExpr):
    kind: Literal["integer"] = "integer"
    value: int


class CardinalityExpr(ScalarExpr):
    kind: Literal["cardinality"] = "cardinality"
    nodes: NodeSet


class ResourceUsageExpr(ScalarExpr):
    kind: Literal["resource_usage"] = "resource_usage"
    workers: WorkerSet
    resource: str


class ResourceCapacityExpr(ScalarExpr):
    kind: Literal["resource_capacity"] = "resource_capacity"
    nodes: NodeSet
    resource: str


class AllocatedQuantaExpr(ScalarExpr):
    kind: Literal["allocated_quanta"] = "allocated_quanta"
    workers: WorkerSet


class AffinityCoverageExpr(ScalarExpr):
    kind: Literal["affinity_coverage"] = "affinity_coverage"
    relations: AffinityConstraintSetExpr


class CoLocationMassExpr(ScalarExpr):
    kind: Literal["co_location_mass"] = "co_location_mass"
    relations: AffinityConstraintSetExpr


class ResourceFragmentationExpr(ScalarExpr):
    kind: Literal["resource_fragmentation"] = "resource_fragmentation"


type Scalar = Annotated[
    IntegerExpr
    | CardinalityExpr
    | ResourceUsageExpr
    | ResourceCapacityExpr
    | AllocatedQuantaExpr
    | AffinityCoverageExpr
    | CoLocationMassExpr
    | ResourceFragmentationExpr,
    Field(discriminator="kind"),
]


class Constraint(PlacementModel):
    kind: Any


class SubsetEqConstraint(Constraint):
    kind: Literal["subseteq"] = "subseteq"
    left: NodeSet
    right: NodeSet


class DisjointConstraint(Constraint):
    kind: Literal["disjoint"] = "disjoint"
    left: NodeSet
    right: NodeSet


class EqualConstraint(Constraint):
    kind: Literal["equal"] = "equal"
    left: NodeSet | Scalar
    right: NodeSet | Scalar

    @model_validator(mode="after")
    def validate_operand_types(self) -> EqualConstraint:
        if isinstance(self.left, NodeSetExpr) != isinstance(self.right, NodeSetExpr):
            raise ValueError("equal operands must have the same mathematical type")
        return self


class LessEqualConstraint(Constraint):
    kind: Literal["less_equal"] = "less_equal"
    left: Scalar
    right: Scalar


class AllOfConstraint(Constraint):
    kind: Literal["all_of"] = "all_of"
    items: tuple[ConstraintNode, ...]


class AnyOfConstraint(Constraint):
    kind: Literal["any_of"] = "any_of"
    items: tuple[ConstraintNode, ...]


class NegateConstraint(Constraint):
    kind: Literal["negate"] = "negate"
    item: ConstraintNode


type ConstraintNode = Annotated[
    SubsetEqConstraint
    | DisjointConstraint
    | EqualConstraint
    | LessEqualConstraint
    | AllOfConstraint
    | AnyOfConstraint
    | NegateConstraint,
    Field(discriminator="kind"),
]


class Objective(PlacementModel):
    kind: Any


class MinimizeObjective(Objective):
    kind: Literal["minimize"] = "minimize"
    expression: Scalar


class MaximizeObjective(Objective):
    kind: Literal["maximize"] = "maximize"
    expression: Scalar


class MaxMinObjective(Objective):
    kind: Literal["max_min"] = "max_min"
    expression: Vector


class MaximizeMajorizationObjective(Objective):
    kind: Literal["maximize_majorization"] = "maximize_majorization"
    expression: Vector


type ObjectiveNode = Annotated[
    MinimizeObjective
    | MaximizeObjective
    | MaxMinObjective
    | MaximizeMajorizationObjective,
    Field(discriminator="kind"),
]


class LexicographicObjective(PlacementModel):
    kind: Literal["lexicographic"] = "lexicographic"
    objectives: tuple[ObjectiveNode, ...]

    @field_validator("objectives")
    @classmethod
    def validate_objectives(
        cls, value: tuple[ObjectiveNode, ...]
    ) -> tuple[ObjectiveNode, ...]:
        if not value:
            raise ValueError("lexicographic requires at least one objective")
        return value


class PlacementRecipe(PlacementModel):
    recipe_id: str = Field(min_length=1)
    constraints: tuple[ConstraintNode, ...]
    objective: LexicographicObjective


def workers(spmd: type | SPMDRef) -> WorkersExpr:
    spmd_class = spmd.spmd_class if isinstance(spmd, SPMDRef) else spmd
    if not isinstance(spmd_class, type):
        raise TypeError("workers expects an SPMD class or reference")
    return WorkersExpr(spmd=SPMDClassRef.from_type(spmd_class))


def instance(spmd: str | SPMDRef) -> InstanceExpr:
    spmd_id = spmd.spmd_id if isinstance(spmd, SPMDRef) else spmd
    return InstanceExpr(spmd_id=spmd_id)


type WorkerSetSource = WorkerSet | type | SPMDRef | Iterable[WorkerSet | type | SPMDRef]


def _worker_set(source: WorkerSetSource) -> WorkerSet:
    if isinstance(source, WorkerSetExpr):
        return cast(WorkerSet, source)
    if isinstance(source, SPMDRef):
        return instance(source)
    if isinstance(source, type):
        return workers(source)
    if hasattr(source, "__iter__") and not isinstance(source, (str, bytes, Mapping)):
        items = tuple(cast(Iterable, source))
        if not items:
            raise ValueError("worker-set reference collection must not be empty")
        return WorkerUnionExpr(items=tuple(_worker_set(item) for item in items))
    raise TypeError(
        "expected a worker-set expression, SPMD class, SPMD reference, "
        "or a non-empty iterable of them"
    )


def all_workers() -> AllWorkersExpr:
    return AllWorkersExpr()


def affinity_constraints() -> AffinityConstraintSetExpr:
    return AffinityConstraintSetExpr()


def incident_spmds(relations: AffinityConstraintSetExpr) -> IncidentSPMDsExpr:
    return IncidentSPMDsExpr(relations=relations)


def isolated_spmds(relations: AffinityConstraintSetExpr) -> IsolatedSPMDsExpr:
    return IsolatedSPMDsExpr(relations=relations)


def rank(source: WorkerSetSource, index: int) -> RankExpr:
    return RankExpr(source=cast(WorkerSet, _worker_set(source)), index=index)


def ranks(source: WorkerSetSource, indices: Sequence[int]) -> RanksExpr:
    return RanksExpr(
        source=cast(WorkerSet, _worker_set(source)), indices=tuple(indices)
    )


def occupancy(source: WorkerSetSource) -> OccupancyExpr:
    return OccupancyExpr(workers=cast(WorkerSet, _worker_set(source)))


def target_occupancy(relations: AffinityConstraintSetExpr) -> TargetOccupancyExpr:
    return TargetOccupancyExpr(relations=relations)


def support(
    source: WorkerSetSource | OccupancyExpr,
) -> SupportExpr:
    workers_expr = (
        source.workers if isinstance(source, OccupancyExpr) else _worker_set(source)
    )
    return SupportExpr(workers=cast(WorkerSet, workers_expr))


@overload
def union(*items: WorkerSetExpr | type | SPMDRef) -> WorkerUnionExpr: ...


@overload
def union(*items: NodeSetExpr) -> NodeUnionExpr: ...


def union(
    *items: WorkerSetExpr | NodeSetExpr | type | SPMDRef,
) -> WorkerUnionExpr | NodeUnionExpr:
    if not items:
        raise ValueError("union requires at least one expression")
    if all(isinstance(item, (WorkerSetExpr, type, SPMDRef)) for item in items):
        return WorkerUnionExpr(
            items=cast(
                tuple[WorkerSet, ...],
                tuple(_worker_set(cast(WorkerSetSource, item)) for item in items),
            )
        )
    if all(isinstance(item, NodeSetExpr) for item in items):
        return NodeUnionExpr(items=tuple(items))  # type: ignore[arg-type]
    raise TypeError("union expressions must have the same mathematical type")


def intersection(*items: NodeSetExpr) -> NodeIntersectionExpr:
    if not items:
        raise ValueError("intersection requires at least one expression")
    return NodeIntersectionExpr(items=cast(tuple[NodeSet, ...], tuple(items)))


def difference(left: NodeSetExpr, right: NodeSetExpr) -> NodeDifferenceExpr:
    return NodeDifferenceExpr(left=cast(NodeSet, left), right=cast(NodeSet, right))


def cardinality(source: NodeSetExpr) -> CardinalityExpr:
    return CardinalityExpr(nodes=cast(NodeSet, source))


def resource_usage(source: WorkerSetSource, resource: str) -> ResourceUsageExpr:
    return ResourceUsageExpr(
        workers=cast(WorkerSet, _worker_set(source)), resource=resource
    )


def resource_capacity(source: NodeSetExpr, resource: str) -> ResourceCapacityExpr:
    return ResourceCapacityExpr(nodes=cast(NodeSet, source), resource=resource)


def allocated_quanta(source: WorkerSetSource) -> AllocatedQuantaExpr:
    return AllocatedQuantaExpr(workers=cast(WorkerSet, _worker_set(source)))


def weighted_dominant_share(
    source: WorkerSetSource,
) -> WeightedDominantShareExpr:
    return WeightedDominantShareExpr(workers=cast(WorkerSet, _worker_set(source)))


def affinity_coverage(relations: AffinityConstraintSetExpr) -> AffinityCoverageExpr:
    return AffinityCoverageExpr(relations=relations)


def co_location_mass(relations: AffinityConstraintSetExpr) -> CoLocationMassExpr:
    return CoLocationMassExpr(relations=relations)


def resource_fragmentation() -> ResourceFragmentationExpr:
    return ResourceFragmentationExpr()


def _scalar(value: ScalarExpr | int) -> ScalarExpr:
    return value if isinstance(value, ScalarExpr) else IntegerExpr(value=value)


def subseteq(left: NodeSetExpr, right: NodeSetExpr) -> SubsetEqConstraint:
    return SubsetEqConstraint(left=cast(NodeSet, left), right=cast(NodeSet, right))


def disjoint(left: NodeSetExpr, right: NodeSetExpr) -> DisjointConstraint:
    return DisjointConstraint(left=cast(NodeSet, left), right=cast(NodeSet, right))


def equal(
    left: NodeSetExpr | ScalarExpr | int,
    right: NodeSetExpr | ScalarExpr | int,
    *others: NodeSetExpr | ScalarExpr | int,
) -> EqualConstraint | AllOfConstraint:
    constraints = tuple(_binary_equal(left, operand) for operand in (right, *others))
    if len(constraints) == 1:
        return constraints[0]
    return AllOfConstraint(items=constraints)


def _binary_equal(
    left: NodeSetExpr | ScalarExpr | int,
    right: NodeSetExpr | ScalarExpr | int,
) -> EqualConstraint:
    normalized_left = _scalar(left) if isinstance(left, (ScalarExpr, int)) else left
    normalized_right = _scalar(right) if isinstance(right, (ScalarExpr, int)) else right
    return EqualConstraint(
        left=cast(NodeSet | Scalar, normalized_left),
        right=cast(NodeSet | Scalar, normalized_right),
    )


def less_equal(left: ScalarExpr | int, right: ScalarExpr | int) -> LessEqualConstraint:
    return LessEqualConstraint(
        left=cast(Scalar, _scalar(left)),
        right=cast(Scalar, _scalar(right)),
    )


def all_of(*items: Constraint) -> AllOfConstraint:
    return AllOfConstraint(items=cast(tuple[ConstraintNode, ...], tuple(items)))


def any_of(*items: Constraint) -> AnyOfConstraint:
    return AnyOfConstraint(items=cast(tuple[ConstraintNode, ...], tuple(items)))


def negate(item: Constraint) -> NegateConstraint:
    return NegateConstraint(item=cast(ConstraintNode, item))


def minimize(expression: ScalarExpr) -> MinimizeObjective:
    return MinimizeObjective(expression=cast(Scalar, expression))


def maximize(expression: ScalarExpr) -> MaximizeObjective:
    return MaximizeObjective(expression=cast(Scalar, expression))


def max_min(expression: VectorExpr) -> MaxMinObjective:
    return MaxMinObjective(expression=cast(Vector, expression))


def maximize_majorization(
    expression: VectorExpr,
) -> MaximizeMajorizationObjective:
    return MaximizeMajorizationObjective(expression=cast(Vector, expression))


def lexicographic(*objectives: Objective) -> LexicographicObjective:
    return LexicographicObjective(
        objectives=cast(tuple[ObjectiveNode, ...], tuple(objectives))
    )


_AFFINITY_RELATIONS = affinity_constraints()
AFFINITY_FIRST = lexicographic(
    max_min(weighted_dominant_share(incident_spmds(_AFFINITY_RELATIONS))),
    max_min(weighted_dominant_share(isolated_spmds(_AFFINITY_RELATIONS))),
    minimize(cardinality(support(all_workers()))),
    maximize_majorization(target_occupancy(_AFFINITY_RELATIONS)),
    maximize(affinity_coverage(_AFFINITY_RELATIONS)),
    maximize(co_location_mass(_AFFINITY_RELATIONS)),
    minimize(resource_fragmentation()),
)


RankExpr.model_rebuild()
RanksExpr.model_rebuild()
WorkerUnionExpr.model_rebuild()
OccupancyExpr.model_rebuild()
TargetOccupancyExpr.model_rebuild()
SupportExpr.model_rebuild()
NodeUnionExpr.model_rebuild()
NodeIntersectionExpr.model_rebuild()
NodeDifferenceExpr.model_rebuild()
CardinalityExpr.model_rebuild()
ResourceUsageExpr.model_rebuild()
ResourceCapacityExpr.model_rebuild()
AllocatedQuantaExpr.model_rebuild()
WeightedDominantShareExpr.model_rebuild()
AffinityCoverageExpr.model_rebuild()
CoLocationMassExpr.model_rebuild()
ResourceFragmentationExpr.model_rebuild()
SubsetEqConstraint.model_rebuild()
DisjointConstraint.model_rebuild()
EqualConstraint.model_rebuild()
LessEqualConstraint.model_rebuild()
AllOfConstraint.model_rebuild()
AnyOfConstraint.model_rebuild()
NegateConstraint.model_rebuild()
MinimizeObjective.model_rebuild()
MaximizeObjective.model_rebuild()
MaxMinObjective.model_rebuild()
MaximizeMajorizationObjective.model_rebuild()
LexicographicObjective.model_rebuild()
PlacementRecipe.model_rebuild()
