from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast, overload

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)


class PlacementModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class SPMDClassRef(PlacementModel):
    module: str
    qualname: str

    @classmethod
    def from_type(cls, value: type) -> SPMDClassRef:
        return cls(module=value.__module__, qualname=value.__qualname__)


class WorkerSetExpr(PlacementModel):
    kind: str


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
    kind: str


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
    kind: str


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
    kind: str


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
    kind: str


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
    kind: str


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


def _resource_vector(value: Any) -> dict[str, Decimal]:
    if not isinstance(value, Mapping):
        raise ValueError("resource vector must be a mapping")
    result: dict[str, Decimal] = {}
    for resource, quantity in value.items():
        if not isinstance(resource, str) or not resource:
            raise ValueError("resource names must be non-empty strings")
        if isinstance(quantity, float):
            raise ValueError("resource quantities must use int or Decimal, not float")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, str, Decimal)):
            raise ValueError("resource quantities must use int or Decimal")
        decimal = Decimal(quantity)
        if decimal < 0:
            raise ValueError("resource quantities must be non-negative")
        result[resource] = decimal
    return dict(sorted(result.items()))


def _freeze_resource_vector(
    value: Mapping[str, Decimal],
) -> Mapping[str, Decimal]:
    return MappingProxyType(dict(value))


type ResourceVector = Annotated[
    Mapping[str, Decimal],
    AfterValidator(_freeze_resource_vector),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]


class ResourceScale(PlacementModel):
    step: ResourceVector
    max_steps: int = Field(ge=0)
    weight: Decimal = Field(gt=Decimal(0))

    @field_validator("step", mode="before")
    @classmethod
    def validate_step(cls, value: Any) -> dict[str, Decimal]:
        result = _resource_vector(value)
        if not result or all(quantity == 0 for quantity in result.values()):
            raise ValueError("scale step must contain a positive resource quantity")
        return result

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, value: Any) -> Decimal:
        if isinstance(value, float):
            raise ValueError("weight must use int or Decimal, not float")
        return Decimal(value)


class ResourceProfile(PlacementModel):
    base: ResourceVector
    scale: ResourceScale | None = None

    @field_validator("base", mode="before")
    @classmethod
    def validate_base(cls, value: Any) -> dict[str, Decimal]:
        return _resource_vector(value)


class ResourceProfileBinding(PlacementModel):
    spmd: SPMDClassRef
    profile: ResourceProfile


_COMMUTATIVE_ITEMS = {
    "worker_union",
    "node_union",
    "node_intersection",
    "all_of",
    "any_of",
}
_COMMUTATIVE_BINARY = {"disjoint", "equal"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _deduplicated_ordered(values: Sequence[Any]) -> list[Any]:
    by_json = {_canonical_json(value): value for value in values}
    return [by_json[key] for key in sorted(by_json)]


def _normalize_ast_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_ast_json(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    normalized = {key: _normalize_ast_json(item) for key, item in value.items()}
    kind = normalized.get("kind")
    if kind in _COMMUTATIVE_ITEMS:
        items = normalized.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{kind} AST must contain an items list")
        flattened = []
        for item in items:
            if isinstance(item, Mapping) and item.get("kind") == kind:
                nested = item.get("items")
                if not isinstance(nested, list):
                    raise ValueError(f"nested {kind} AST must contain an items list")
                flattened.extend(nested)
            else:
                flattened.append(item)
        normalized["items"] = _deduplicated_ordered(flattened)
    if kind in _COMMUTATIVE_BINARY:
        operands = _deduplicated_ordered([normalized["left"], normalized["right"]])
        if len(operands) == 2:
            normalized["left"], normalized["right"] = operands
    return normalized


class PlacementRecipe(PlacementModel):
    recipe_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    profiles: tuple[ResourceProfileBinding, ...]
    constraints: tuple[ConstraintNode, ...]
    objective: LexicographicObjective

    @model_validator(mode="before")
    @classmethod
    def normalize_profiles(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        profiles = value.get("profiles")
        if not isinstance(profiles, Mapping):
            return value
        bindings = []
        for spmd, profile in profiles.items():
            if not isinstance(spmd, type):
                raise ValueError("profile keys must be SPMD classes")
            bindings.append(
                ResourceProfileBinding(
                    spmd=SPMDClassRef.from_type(spmd), profile=profile
                )
            )
        normalized = dict(value)
        normalized["profiles"] = tuple(
            sorted(
                bindings,
                key=lambda binding: (binding.spmd.module, binding.spmd.qualname),
            )
        )
        return normalized

    @model_validator(mode="after")
    def validate_profile_uniqueness(self) -> PlacementRecipe:
        refs = [
            (binding.spmd.module, binding.spmd.qualname) for binding in self.profiles
        ]
        if len(refs) != len(set(refs)):
            raise ValueError("each SPMD class must have exactly one resource profile")
        return self

    @property
    def normalized(self) -> PlacementRecipe:
        return PlacementRecipe.model_validate(self._normalized_payload())

    def _normalized_payload(self) -> dict[str, Any]:
        payload = cast(
            dict[str, Any], _normalize_ast_json(self.model_dump(mode="json"))
        )
        constraints = payload.get("constraints")
        if not isinstance(constraints, list):
            raise ValueError("PlacementRecipe constraints must serialize as a list")
        payload["constraints"] = _deduplicated_ordered(constraints)
        profiles = payload.get("profiles")
        if not isinstance(profiles, list):
            raise ValueError("PlacementRecipe profiles must serialize as a list")
        payload["profiles"] = sorted(
            profiles,
            key=lambda binding: (
                binding["spmd"]["module"],
                binding["spmd"]["qualname"],
            ),
        )
        return payload

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self._normalized_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workers(spmd: type) -> WorkersExpr:
    if not isinstance(spmd, type):
        raise TypeError("workers expects an SPMD class")
    return WorkersExpr(spmd=SPMDClassRef.from_type(spmd))


def instance(spmd_id: str) -> InstanceExpr:
    return InstanceExpr(spmd_id=spmd_id)


def all_workers() -> AllWorkersExpr:
    return AllWorkersExpr()


def affinity_constraints() -> AffinityConstraintSetExpr:
    return AffinityConstraintSetExpr()


def incident_spmds(relations: AffinityConstraintSetExpr) -> IncidentSPMDsExpr:
    return IncidentSPMDsExpr(relations=relations)


def isolated_spmds(relations: AffinityConstraintSetExpr) -> IsolatedSPMDsExpr:
    return IsolatedSPMDsExpr(relations=relations)


def rank(source: WorkerSetExpr, index: int) -> RankExpr:
    return RankExpr(source=cast(WorkerSet, source), index=index)


def ranks(source: WorkerSetExpr, indices: Sequence[int]) -> RanksExpr:
    return RanksExpr(source=cast(WorkerSet, source), indices=tuple(indices))


def occupancy(source: WorkerSetExpr) -> OccupancyExpr:
    return OccupancyExpr(workers=cast(WorkerSet, source))


def target_occupancy(relations: AffinityConstraintSetExpr) -> TargetOccupancyExpr:
    return TargetOccupancyExpr(relations=relations)


def support(source: WorkerSetExpr | OccupancyExpr) -> SupportExpr:
    workers_expr = source.workers if isinstance(source, OccupancyExpr) else source
    return SupportExpr(workers=cast(WorkerSet, workers_expr))


@overload
def union(*items: WorkerSetExpr) -> WorkerUnionExpr: ...


@overload
def union(*items: NodeSetExpr) -> NodeUnionExpr: ...


def union(*items: WorkerSetExpr | NodeSetExpr) -> WorkerUnionExpr | NodeUnionExpr:
    if not items:
        raise ValueError("union requires at least one expression")
    if all(isinstance(item, WorkerSetExpr) for item in items):
        return WorkerUnionExpr(items=tuple(items))  # type: ignore[arg-type]
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


def resource_usage(source: WorkerSetExpr, resource: str) -> ResourceUsageExpr:
    return ResourceUsageExpr(workers=cast(WorkerSet, source), resource=resource)


def resource_capacity(source: NodeSetExpr, resource: str) -> ResourceCapacityExpr:
    return ResourceCapacityExpr(nodes=cast(NodeSet, source), resource=resource)


def allocated_quanta(source: WorkerSetExpr) -> AllocatedQuantaExpr:
    return AllocatedQuantaExpr(workers=cast(WorkerSet, source))


def weighted_dominant_share(source: WorkerSetExpr) -> WeightedDominantShareExpr:
    return WeightedDominantShareExpr(workers=cast(WorkerSet, source))


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
AFFINITY_FIRST_V1 = lexicographic(
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
