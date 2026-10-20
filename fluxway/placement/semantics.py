from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from functools import singledispatchmethod
from typing import TYPE_CHECKING, cast

from .inventory import PlacementInventory

from .mathematics import (
    AffinityCoverageExpr,
    AllocatedQuantaExpr,
    AllOfConstraint,
    AllWorkersExpr,
    AnyOfConstraint,
    CardinalityExpr,
    CoLocationMassExpr,
    Constraint,
    DisjointConstraint,
    EqualConstraint,
    IncidentSPMDsExpr,
    InstanceExpr,
    IntegerExpr,
    IsolatedSPMDsExpr,
    LessEqualConstraint,
    NegateConstraint,
    NodeDifferenceExpr,
    NodeIntersectionExpr,
    NodeSetExpr,
    NodeUnionExpr,
    RankExpr,
    RanksExpr,
    ResourceCapacityExpr,
    ResourceFragmentationExpr,
    ResourceUsageExpr,
    ScalarExpr,
    SubsetEqConstraint,
    SupportExpr,
    WorkerSetExpr,
    WorkerUnionExpr,
    WorkersExpr,
)
from .models import (
    ClusterSnapshot,
    PlacementContractError,
    WorkerAssignment,
)

if TYPE_CHECKING:
    from .compiler import _Worker


class _ExpressionResolver:
    def __init__(
        self,
        inventory: PlacementInventory,
        workers: tuple[_Worker, ...],
        constraints: tuple[Constraint, ...],
    ):
        self._workers = workers
        self._constraints = constraints
        self.spmd_ids = inventory.spmd_ids
        self._class_to_spmd: defaultdict[type, list[str]] = defaultdict(list)
        for declaration in inventory.declarations:
            self._class_to_spmd[declaration.spmd_class].append(declaration.spmd_id)

    @singledispatchmethod
    def worker_indices(self, expression: WorkerSetExpr) -> tuple[int, ...]:
        raise PlacementContractError(
            f"unsupported worker-set expression {expression.kind!r}"
        )

    @worker_indices.register
    def _(self, expression: WorkersExpr) -> tuple[int, ...]:
        key = expression.spmd.spmd_class
        if key not in self._class_to_spmd:
            raise PlacementContractError(
                f"SPMD class {key.__module__}.{key.__qualname__} is not in the registry"
            )
        spmd_ids = self._class_to_spmd[key]
        return tuple(
            worker.index for worker in self._workers if worker.spmd_id in spmd_ids
        )

    @worker_indices.register
    def _(self, expression: InstanceExpr) -> tuple[int, ...]:
        if expression.spmd_id not in self.spmd_ids:
            raise PlacementContractError(
                f"SPMD instance {expression.spmd_id!r} is not in the Graph"
            )
        return tuple(
            worker.index
            for worker in self._workers
            if worker.spmd_id == expression.spmd_id
        )

    @worker_indices.register
    def _(self, expression: RankExpr) -> tuple[int, ...]:
        source = self.worker_indices(expression.source)
        selected = tuple(
            index for index in source if self._workers[index].rank == expression.index
        )
        if not selected:
            raise PlacementContractError(
                f"rank {expression.index} does not exist in the worker set"
            )
        return selected

    @worker_indices.register
    def _(self, expression: RanksExpr) -> tuple[int, ...]:
        source = self.worker_indices(expression.source)
        selected = tuple(
            index for index in source if self._workers[index].rank in expression.indices
        )
        selected_values = {self._workers[index].rank for index in selected}
        if selected_values != set(expression.indices):
            raise PlacementContractError(f"ranks {expression.indices} do not all exist")
        return selected

    @worker_indices.register
    def _(self, expression: WorkerUnionExpr) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    index
                    for item in expression.items
                    for index in self.worker_indices(item)
                }
            )
        )

    @worker_indices.register
    def _(self, expression: AllWorkersExpr) -> tuple[int, ...]:
        return tuple(worker.index for worker in self._workers)

    @worker_indices.register
    def _(self, expression: IncidentSPMDsExpr) -> tuple[int, ...]:
        incident = self.incident_spmd_ids()
        return tuple(
            worker.index for worker in self._workers if worker.spmd_id in incident
        )

    @worker_indices.register
    def _(self, expression: IsolatedSPMDsExpr) -> tuple[int, ...]:
        incident = self.incident_spmd_ids()
        return tuple(
            worker.index for worker in self._workers if worker.spmd_id not in incident
        )

    def worker(self, index: int) -> _Worker:
        return self._workers[index]

    def incident_spmd_ids(self) -> set[str]:
        result: set[str] = set()
        for constraint in self.support_affinity_constraints():
            for expression in (constraint.left, constraint.right):
                assert isinstance(expression, SupportExpr)
                result.update(
                    self._workers[index].spmd_id
                    for index in self.worker_indices(expression.workers)
                )
        return result

    def scalable_spmd_ids(self, expression: WorkerSetExpr) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self._workers[index].spmd_id
                    for index in self.worker_indices(expression)
                    if self._workers[index].profile.scale is not None
                }
            )
        )

    def support_affinity_constraints(
        self,
    ) -> tuple[SubsetEqConstraint | EqualConstraint, ...]:
        relations = self.affinity_constraints()
        for relation in relations:
            if not isinstance(relation.left, SupportExpr) or not isinstance(
                relation.right, SupportExpr
            ):
                raise PlacementContractError(
                    "affinity objectives require support-based affinity constraints"
                )
        return relations

    def affinity_target_worker_sets(self) -> tuple[WorkerSetExpr, ...]:
        targets = {
            repr(cast(SupportExpr, relation.right).workers): cast(
                SupportExpr, relation.right
            ).workers
            for relation in self.support_affinity_constraints()
        }
        return tuple(targets[key] for key in sorted(targets))

    def affinity_constraints(
        self,
    ) -> tuple[SubsetEqConstraint | EqualConstraint, ...]:
        analyzer = _PositiveAffinityAnalyzer()
        relations = {
            key: relation
            for constraint in self._constraints
            for key, relation in analyzer.guaranteed(constraint, True).items()
        }
        return tuple(relations[key] for key in sorted(relations))


class _PositiveAffinityAnalyzer:
    def _key(self, constraint: Constraint) -> str:
        return repr(constraint)

    @singledispatchmethod
    def guaranteed(
        self,
        constraint: Constraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        return {}

    @guaranteed.register
    def _(
        self,
        constraint: SubsetEqConstraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        if (
            positive
            and isinstance(constraint.left, SupportExpr)
            and isinstance(constraint.right, SupportExpr)
        ):
            return {self._key(constraint): constraint}
        return {}

    @guaranteed.register
    def _(
        self,
        constraint: EqualConstraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        if (
            positive
            and isinstance(constraint.left, SupportExpr)
            and isinstance(constraint.right, SupportExpr)
        ):
            return {self._key(constraint): constraint}
        return {}

    @guaranteed.register
    def _(
        self,
        constraint: NegateConstraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        return self.guaranteed(constraint.item, not positive)

    def _combine(
        self,
        constraints: tuple[Constraint, ...],
        positive: bool,
        conjunction: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        children = [self.guaranteed(item, positive) for item in constraints]
        if conjunction:
            return {
                key: relation for child in children for key, relation in child.items()
            }
        if not children:
            return {}
        common = set(children[0]).intersection(*(set(child) for child in children[1:]))
        return {key: children[0][key] for key in common}

    @guaranteed.register
    def _(
        self,
        constraint: AllOfConstraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        return self._combine(constraint.items, positive, positive)

    @guaranteed.register
    def _(
        self,
        constraint: AnyOfConstraint,
        positive: bool,
    ) -> dict[str, SubsetEqConstraint | EqualConstraint]:
        return self._combine(constraint.items, positive, not positive)


class _AssignmentEvaluator:
    def __init__(
        self,
        resolver: _ExpressionResolver,
        assignments: tuple[WorkerAssignment, ...],
        allocated_quanta: Mapping[str, int],
        snapshot: ClusterSnapshot,
        resource_fragmentation_score: int,
        resource_fragmentation_scale: int,
    ):
        self._resolver = resolver
        self._allocated_quanta = allocated_quanta
        self._snapshot = snapshot
        self._resource_fragmentation_score = resource_fragmentation_score
        self._resource_fragmentation_scale = resource_fragmentation_scale
        self._assignment_by_worker = {
            (assignment.spmd_id, assignment.rank): assignment
            for assignment in assignments
        }

    @singledispatchmethod
    def nodes(self, expression: NodeSetExpr) -> frozenset[str]:
        raise PlacementContractError(
            f"unsupported node-set expression {expression.kind!r}"
        )

    @nodes.register
    def _(self, expression: SupportExpr) -> frozenset[str]:
        return frozenset(
            self._assignment(index).node_id
            for index in self._resolver.worker_indices(expression.workers)
        )

    @nodes.register
    def _(self, expression: NodeUnionExpr) -> frozenset[str]:
        return frozenset().union(*(self.nodes(item) for item in expression.items))

    @nodes.register
    def _(self, expression: NodeIntersectionExpr) -> frozenset[str]:
        iterator = iter(expression.items)
        result = self.nodes(next(iterator))
        for item in iterator:
            result = result.intersection(self.nodes(item))
        return result

    @nodes.register
    def _(self, expression: NodeDifferenceExpr) -> frozenset[str]:
        return self.nodes(expression.left).difference(self.nodes(expression.right))

    @singledispatchmethod
    def scalar(self, expression: ScalarExpr) -> int | Decimal:
        raise PlacementContractError(
            f"scalar {expression.kind!r} cannot be attested as a hard constraint"
        )

    @scalar.register
    def _(self, expression: IntegerExpr) -> int | Decimal:
        return expression.value

    @scalar.register
    def _(self, expression: CardinalityExpr) -> int | Decimal:
        return len(self.nodes(expression.nodes))

    @scalar.register
    def _(self, expression: ResourceUsageExpr) -> int | Decimal:
        return sum(
            (
                self._assignment(index).resources.get(expression.resource, Decimal(0))
                + self._assignment(index).ray_component_limits.get(
                    expression.resource, Decimal(0)
                )
                for index in self._resolver.worker_indices(expression.workers)
            ),
            Decimal(0),
        )

    @scalar.register
    def _(self, expression: ResourceCapacityExpr) -> int | Decimal:
        selected = self.nodes(expression.nodes)
        return sum(
            (
                node.resources.get(expression.resource, Decimal(0))
                for node in self._snapshot.nodes
                if node.node_id in selected
            ),
            Decimal(0),
        )

    @scalar.register
    def _(self, expression: AllocatedQuantaExpr) -> int | Decimal:
        spmd_ids = {
            self._resolver.worker(index).spmd_id
            for index in self._resolver.worker_indices(expression.workers)
        }
        return sum(self._allocated_quanta[spmd_id] for spmd_id in spmd_ids)

    @scalar.register
    def _(self, expression: AffinityCoverageExpr) -> int | Decimal:
        coverage = 0
        for relation in self._resolver.support_affinity_constraints():
            left = self.nodes(relation.left)
            right = self.nodes(relation.right)
            coverage += len(left.intersection(right))
        return coverage

    @scalar.register
    def _(self, expression: CoLocationMassExpr) -> int | Decimal:
        mass = 0
        for relation in self._resolver.support_affinity_constraints():
            for node in self._snapshot.nodes:
                mass += self._occupancy(
                    cast(SupportExpr, relation.left).workers, node.node_id
                ) * (
                    self._occupancy(
                        cast(SupportExpr, relation.right).workers, node.node_id
                    )
                )
        return mass

    @scalar.register
    def _(self, expression: ResourceFragmentationExpr) -> int | Decimal:
        return Decimal(self._resource_fragmentation_score) / Decimal(
            self._resource_fragmentation_scale
        )

    @singledispatchmethod
    def constraint(self, constraint: Constraint) -> bool:
        raise PlacementContractError(
            f"unsupported constraint AST node {constraint.kind!r}"
        )

    @constraint.register
    def _(self, constraint: SubsetEqConstraint) -> bool:
        return self.nodes(constraint.left).issubset(self.nodes(constraint.right))

    @constraint.register
    def _(self, constraint: DisjointConstraint) -> bool:
        return self.nodes(constraint.left).isdisjoint(self.nodes(constraint.right))

    @constraint.register
    def _(self, constraint: EqualConstraint) -> bool:
        if isinstance(constraint.left, NodeSetExpr):
            if not isinstance(constraint.right, NodeSetExpr):
                raise PlacementContractError(
                    "equal operand types changed after validation"
                )
            return self.nodes(constraint.left) == self.nodes(constraint.right)
        if not isinstance(constraint.right, ScalarExpr):
            raise PlacementContractError("equal operand types changed after validation")
        return self.scalar(constraint.left) == self.scalar(constraint.right)

    @constraint.register
    def _(self, constraint: LessEqualConstraint) -> bool:
        return self.scalar(constraint.left) <= self.scalar(constraint.right)

    @constraint.register
    def _(self, constraint: AllOfConstraint) -> bool:
        return all(self.constraint(item) for item in constraint.items)

    @constraint.register
    def _(self, constraint: AnyOfConstraint) -> bool:
        return any(self.constraint(item) for item in constraint.items)

    @constraint.register
    def _(self, constraint: NegateConstraint) -> bool:
        return not self.constraint(constraint.item)

    def _assignment(self, worker_index: int) -> WorkerAssignment:
        worker = self._resolver.worker(worker_index)
        return self._assignment_by_worker[(worker.spmd_id, worker.rank)]

    def _occupancy(self, expression: WorkerSetExpr, node_id: str) -> int:
        return sum(
            self._assignment(index).node_id == node_id
            for index in self._resolver.worker_indices(expression)
        )
