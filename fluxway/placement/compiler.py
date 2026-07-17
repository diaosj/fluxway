from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import singledispatchmethod

from ortools.sat.python import cp_model
from fluxway.resources import ResourceProfile

from .mathematics import (
    AffinityCoverageExpr,
    AllocatedQuantaExpr,
    AllOfConstraint,
    AnyOfConstraint,
    CardinalityExpr,
    CoLocationMassExpr,
    Constraint,
    DisjointConstraint,
    EqualConstraint,
    IntegerExpr,
    LessEqualConstraint,
    NegateConstraint,
    NodeDifferenceExpr,
    NodeIntersectionExpr,
    NodeSetExpr,
    NodeUnionExpr,
    OccupancyExpr,
    PlacementModel,
    ResourceCapacityExpr,
    ResourceFragmentationExpr,
    ResourceUsageExpr,
    ScalarExpr,
    SubsetEqConstraint,
    SupportExpr,
    TargetOccupancyExpr,
    VectorExpr,
    WeightedDominantShareExpr,
    WorkerSetExpr,
)
from .models import ClusterSnapshot, PlacementContractError
from .semantics import _ExpressionResolver


RAY_RESOURCE_SCALE = 10_000
CP_SAT_INT_MAX = (1 << 63) - 1


def _ray_units(value: Decimal) -> int:
    scaled = value * RAY_RESOURCE_SCALE
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise PlacementContractError(
            f"resource quantity {value} exceeds Ray's 0.0001 precision"
        )
    return int(integral)


def _checked_int64(value: int, label: str) -> int:
    if not -CP_SAT_INT_MAX <= value <= CP_SAT_INT_MAX:
        raise PlacementContractError(
            f"{label} exceeds the CP-SAT int64 domain: {value}"
        )
    return value


def _checked_product(*values: int, label: str) -> int:
    result = math.prod(values)
    return _checked_int64(result, label)


@dataclass(frozen=True)
class _Worker:
    index: int
    spmd_id: str
    rank: int
    profile: ResourceProfile
    driver_resources: Mapping[str, Decimal]


@dataclass(frozen=True)
class _PlacementUnit:
    index: int
    worker_indices: tuple[int, ...]
    spmd_id: str
    profile: ResourceProfile
    driver_resources: Mapping[str, Decimal]

    @property
    def size(self) -> int:
        return len(self.worker_indices)


@dataclass(frozen=True)
class _ResourceUnits:
    quantum_ray_units: Mapping[str, int]

    @classmethod
    def derive(
        cls,
        workers: tuple[_Worker, ...],
        snapshot: ClusterSnapshot,
        resource_names: Iterable[str],
    ) -> _ResourceUnits:
        quantities: defaultdict[str, list[int]] = defaultdict(list)

        def collect(vector: Mapping[str, Decimal]) -> None:
            for resource, quantity in vector.items():
                ray_units = _ray_units(quantity)
                if ray_units > 0:
                    quantities[resource].append(ray_units)

        collect(snapshot.available)
        for node in snapshot.nodes:
            collect(node.resources)
        for worker in workers:
            collect(worker.profile.base)
            collect(worker.profile.ray_component_limits)
            collect(worker.driver_resources)
            if worker.profile.scale is not None:
                collect(worker.profile.scale.step)
        quanta = {
            resource: math.gcd(*quantities[resource]) if quantities[resource] else 1
            for resource in resource_names
        }
        return cls(quantum_ray_units=quanta)

    def units(self, resource: str, quantity: Decimal) -> int:
        ray_units = _ray_units(quantity)
        quantum = self.quantum_ray_units[resource]
        if ray_units % quantum:
            raise PlacementContractError(
                f"resource {resource!r} quantity {quantity} is not divisible by "
                f"its model quantum"
            )
        return _checked_int64(ray_units // quantum, f"resource {resource!r} quantity")

    def quantity(self, resource: str, units: int) -> Decimal:
        ray_units = units * self.quantum_ray_units[resource]
        return Decimal(ray_units) / Decimal(RAY_RESOURCE_SCALE)

    def physical_fraction(self, resource: str) -> tuple[int, int]:
        quantum = self.quantum_ray_units[resource]
        divisor = math.gcd(quantum, RAY_RESOURCE_SCALE)
        return quantum // divisor, RAY_RESOURCE_SCALE // divisor


@dataclass
class _CompiledModel:
    model: cp_model.CpModel
    assignments: dict[tuple[int, int], cp_model.IntVar]
    workers: tuple[_Worker, ...]
    units: tuple[_PlacementUnit, ...]
    worker_to_unit: dict[int, int]
    resource_units: _ResourceUnits
    worker_set_cache: dict[PlacementModel, tuple[int, ...]]
    occupancy_cache: dict[tuple[PlacementModel, int], cp_model.LinearExprT]
    support_cache: dict[tuple[PlacementModel, int], cp_model.IntVar]
    allocated_quanta: dict[str, cp_model.IntVar]
    node_usage: dict[tuple[int, str], cp_model.LinearExprT]
    fragmentation: dict[tuple[int, str], cp_model.IntVar]
    fragmentation_weights: dict[str, int]
    fragmentation_scale: int


@dataclass(frozen=True)
class _VectorObjective:
    components: tuple[cp_model.LinearExprT, ...]
    upper_bounds: tuple[int, ...]
    value_scale: int = 1


@dataclass(frozen=True)
class _ScaledScalar:
    expression: cp_model.LinearExprT
    scale: int
    dimension: str | None


class _ConstraintCompiler:
    def __init__(
        self,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        snapshot: ClusterSnapshot,
    ):
        self._compiled = compiled
        self._resolver = resolver
        self._snapshot = snapshot
        self._literal_index = 0
        self._scalar_index = 0

    def _key(self, expression: PlacementModel) -> PlacementModel:
        return expression

    def worker_indices(self, expression: WorkerSetExpr) -> tuple[int, ...]:
        key = self._key(expression)
        if key not in self._compiled.worker_set_cache:
            self._compiled.worker_set_cache[key] = self._resolver.worker_indices(
                expression
            )
        return self._compiled.worker_set_cache[key]

    def occupancy(
        self, expression: WorkerSetExpr, node_index: int
    ) -> cp_model.LinearExprT:
        key = (self._key(expression), node_index)
        if key not in self._compiled.occupancy_cache:
            selected_workers = frozenset(self.worker_indices(expression))
            selected_units = {
                self._compiled.worker_to_unit[worker_index]
                for worker_index in selected_workers
            }
            for unit_index in selected_units:
                unit_workers = frozenset(
                    self._compiled.units[unit_index].worker_indices
                )
                if not unit_workers.issubset(selected_workers):
                    raise PlacementContractError(
                        f"worker-set expression {key} selects part of placement "
                        f"unit {unit_index}"
                    )
            self._compiled.occupancy_cache[key] = sum(
                self._compiled.assignments[(unit_index, node_index)]
                for unit_index in selected_units
            )
        return self._compiled.occupancy_cache[key]

    def support(self, expression: NodeSetExpr, node_index: int) -> cp_model.IntVar:
        key = (self._key(expression), node_index)
        if key in self._compiled.support_cache:
            return self._compiled.support_cache[key]
        result = self._new_literal("support")
        self._compile_support(expression, node_index, result)
        self._compiled.support_cache[key] = result
        return result

    @singledispatchmethod
    def _compile_support(
        self,
        expression: NodeSetExpr,
        node_index: int,
        result: cp_model.IntVar,
    ) -> None:
        raise PlacementContractError(
            f"unsupported node-set expression {expression.kind!r}"
        )

    @_compile_support.register
    def _(
        self,
        expression: SupportExpr,
        node_index: int,
        result: cp_model.IntVar,
    ) -> None:
        occupancy = self.occupancy(expression.workers, node_index)
        size = len(self.worker_indices(expression.workers))
        self._compiled.model.add(occupancy >= result)
        self._compiled.model.add(occupancy <= size * result)

    @_compile_support.register
    def _(
        self,
        expression: NodeUnionExpr,
        node_index: int,
        result: cp_model.IntVar,
    ) -> None:
        items = [self.support(item, node_index) for item in expression.items]
        self._compiled.model.add_bool_or(items).only_enforce_if(result)
        self._compiled.model.add_bool_and(
            [item.Not() for item in items]
        ).only_enforce_if(result.Not())

    @_compile_support.register
    def _(
        self,
        expression: NodeIntersectionExpr,
        node_index: int,
        result: cp_model.IntVar,
    ) -> None:
        items = [self.support(item, node_index) for item in expression.items]
        self._compiled.model.add_bool_and(items).only_enforce_if(result)
        self._compiled.model.add_bool_or(
            [item.Not() for item in items]
        ).only_enforce_if(result.Not())

    @_compile_support.register
    def _(
        self,
        expression: NodeDifferenceExpr,
        node_index: int,
        result: cp_model.IntVar,
    ) -> None:
        left = self.support(expression.left, node_index)
        right = self.support(expression.right, node_index)
        self._compiled.model.add_bool_and([left, right.Not()]).only_enforce_if(result)
        self._compiled.model.add_bool_or([left.Not(), right]).only_enforce_if(
            result.Not()
        )

    @singledispatchmethod
    def scalar(self, expression: ScalarExpr) -> cp_model.LinearExprT:
        raise PlacementContractError(
            f"scalar {expression.kind!r} is not valid in a hard constraint"
        )

    @scalar.register(IntegerExpr)
    def _(self, expression: IntegerExpr) -> cp_model.LinearExprT:
        return expression.value

    @scalar.register(CardinalityExpr)
    def _(self, expression: CardinalityExpr) -> cp_model.LinearExprT:
        return sum(
            self.support(expression.nodes, node_index)
            for node_index in range(len(self._snapshot.nodes))
        )

    @scalar.register(ResourceUsageExpr)
    def _(self, expression: ResourceUsageExpr) -> cp_model.LinearExprT:
        terms: list[cp_model.LinearExprT] = []
        for worker_index in self.worker_indices(expression.workers):
            worker = self._compiled.workers[worker_index]
            base = self._compiled.resource_units.units(
                expression.resource,
                worker.profile.placement_base.get(expression.resource, Decimal(0)),
            )
            scale = worker.profile.scale
            term: cp_model.LinearExprT = base
            if scale is not None:
                term = base + (
                    self._compiled.resource_units.units(
                        expression.resource,
                        scale.step.get(expression.resource, Decimal(0)),
                    )
                    * self._compiled.allocated_quanta[worker.spmd_id]
                )
            terms.append(term)
        return sum(terms)

    @scalar.register(ResourceCapacityExpr)
    def _(self, expression: ResourceCapacityExpr) -> cp_model.LinearExprT:
        return sum(
            self._compiled.resource_units.units(
                expression.resource,
                self._snapshot.nodes[node_index].resources.get(
                    expression.resource, Decimal(0)
                ),
            )
            * self.support(expression.nodes, node_index)
            for node_index in range(len(self._snapshot.nodes))
        )

    @scalar.register(AllocatedQuantaExpr)
    def _(self, expression: AllocatedQuantaExpr) -> cp_model.LinearExprT:
        spmd_ids = {
            self._compiled.workers[worker_index].spmd_id
            for worker_index in self.worker_indices(expression.workers)
        }
        return sum(self._compiled.allocated_quanta[spmd_id] for spmd_id in spmd_ids)

    @scalar.register(ResourceFragmentationExpr)
    def _(self, expression: ResourceFragmentationExpr) -> cp_model.LinearExprT:
        return sum(
            remainder * self._compiled.fragmentation_weights[resource]
            for (_, resource), remainder in self._compiled.fragmentation.items()
        )

    @scalar.register(AffinityCoverageExpr)
    def _(self, expression: AffinityCoverageExpr) -> cp_model.LinearExprT:
        coverage = []
        for relation in self._resolver.support_affinity_constraints():
            left_expr = relation.left
            right_expr = relation.right
            assert isinstance(left_expr, SupportExpr)
            assert isinstance(right_expr, SupportExpr)
            for node_index in range(len(self._snapshot.nodes)):
                left = self.support(left_expr, node_index)
                right = self.support(right_expr, node_index)
                both = self._compiled.model.new_bool_var(
                    self._new_scalar_name("affinity_coverage")
                )
                self._compiled.model.add_bool_and([left, right]).only_enforce_if(both)
                self._compiled.model.add_bool_or(
                    [left.Not(), right.Not()]
                ).only_enforce_if(both.Not())
                coverage.append(both)
        return sum(coverage)

    @scalar.register(CoLocationMassExpr)
    def _(self, expression: CoLocationMassExpr) -> cp_model.LinearExprT:
        products = []
        for relation in self._resolver.support_affinity_constraints():
            left_expr = relation.left
            right_expr = relation.right
            assert isinstance(left_expr, SupportExpr)
            assert isinstance(right_expr, SupportExpr)
            left_size = len(self._resolver.worker_indices(left_expr.workers))
            right_size = len(self._resolver.worker_indices(right_expr.workers))
            for node_index in range(len(self._snapshot.nodes)):
                left_occupancy = self.occupancy(left_expr.workers, node_index)
                right_occupancy = self.occupancy(right_expr.workers, node_index)
                product = self._compiled.model.new_int_var(
                    0,
                    _checked_product(
                        left_size,
                        right_size,
                        label="co-location mass upper bound",
                    ),
                    self._new_scalar_name("co_location_mass"),
                )
                self._compiled.model.add_multiplication_equality(
                    product, [left_occupancy, right_occupancy]
                )
                products.append(product)
        return sum(products)

    @singledispatchmethod
    def scaled_scalar(self, expression: ScalarExpr) -> _ScaledScalar:
        dimensions: dict[type[ScalarExpr], str | None] = {
            IntegerExpr: None,
            CardinalityExpr: "node_count",
            AllocatedQuantaExpr: "allocated_quanta",
            AffinityCoverageExpr: "node_count",
            CoLocationMassExpr: "worker_pair_count",
        }
        try:
            dimension = dimensions[type(expression)]
        except KeyError as error:
            raise PlacementContractError(
                f"scalar {expression.kind!r} has no physical dimension"
            ) from error
        return _ScaledScalar(self.scalar(expression), 1, dimension)

    @scaled_scalar.register(ResourceUsageExpr)
    @scaled_scalar.register(ResourceCapacityExpr)
    def _(self, expression: ResourceUsageExpr | ResourceCapacityExpr) -> _ScaledScalar:
        multiplier, scale = self._compiled.resource_units.physical_fraction(
            expression.resource
        )
        return _ScaledScalar(
            self.scalar(expression) * multiplier,
            scale,
            f"resource:{expression.resource}",
        )

    @scaled_scalar.register(ResourceFragmentationExpr)
    def _(self, expression: ResourceFragmentationExpr) -> _ScaledScalar:
        return _ScaledScalar(
            self.scalar(expression),
            self._compiled.fragmentation_scale,
            "resource_fragmentation",
        )

    def _comparison_expressions(
        self,
        left: ScalarExpr,
        right: ScalarExpr,
    ) -> tuple[cp_model.LinearExprT, cp_model.LinearExprT]:
        left_scalar = self.scaled_scalar(left)
        right_scalar = self.scaled_scalar(right)
        if (
            left_scalar.dimension is not None
            and right_scalar.dimension is not None
            and left_scalar.dimension != right_scalar.dimension
        ):
            raise PlacementContractError(
                "scalar comparison has incompatible dimensions: "
                f"{left_scalar.dimension!r} and {right_scalar.dimension!r}"
            )
        return (
            left_scalar.expression * right_scalar.scale,
            right_scalar.expression * left_scalar.scale,
        )

    @singledispatchmethod
    def constraint_literal(self, constraint: Constraint) -> cp_model.IntVar:
        raise PlacementContractError(
            f"unsupported constraint AST node {constraint.kind!r}"
        )

    @constraint_literal.register
    def _(self, constraint: SubsetEqConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        items = []
        for node_index in range(len(self._snapshot.nodes)):
            left = self.support(constraint.left, node_index)
            right = self.support(constraint.right, node_index)
            item = self._new_literal("subseteq")
            self._compiled.model.add_bool_or([left.Not(), right]).only_enforce_if(item)
            self._compiled.model.add_bool_and([left, right.Not()]).only_enforce_if(
                item.Not()
            )
            items.append(item)
        self._reify_all(result, items)
        return result

    @constraint_literal.register
    def _(self, constraint: DisjointConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        items = []
        for node_index in range(len(self._snapshot.nodes)):
            left = self.support(constraint.left, node_index)
            right = self.support(constraint.right, node_index)
            item = self._new_literal("disjoint")
            self._compiled.model.add_bool_or([left.Not(), right.Not()]).only_enforce_if(
                item
            )
            self._compiled.model.add_bool_and([left, right]).only_enforce_if(item.Not())
            items.append(item)
        self._reify_all(result, items)
        return result

    @constraint_literal.register
    def _(self, constraint: EqualConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        if isinstance(constraint.left, NodeSetExpr):
            if not isinstance(constraint.right, NodeSetExpr):
                raise PlacementContractError(
                    "equal operand types changed after validation"
                )
            items = []
            for node_index in range(len(self._snapshot.nodes)):
                left = self.support(constraint.left, node_index)
                right = self.support(constraint.right, node_index)
                item = self._new_literal("equal_set")
                self._compiled.model.add(left == right).only_enforce_if(item)
                self._compiled.model.add(left != right).only_enforce_if(item.Not())
                items.append(item)
            self._reify_all(result, items)
            return result
        if not isinstance(constraint.right, ScalarExpr):
            raise PlacementContractError("equal operand types changed after validation")
        left_scaled, right_scaled = self._comparison_expressions(
            constraint.left, constraint.right
        )
        self._compiled.model.add(left_scaled == right_scaled).only_enforce_if(result)
        self._compiled.model.add(left_scaled != right_scaled).only_enforce_if(
            result.Not()
        )
        return result

    @constraint_literal.register
    def _(self, constraint: LessEqualConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        left_scaled, right_scaled = self._comparison_expressions(
            constraint.left, constraint.right
        )
        self._compiled.model.add(left_scaled <= right_scaled).only_enforce_if(result)
        self._compiled.model.add(left_scaled > right_scaled).only_enforce_if(
            result.Not()
        )
        return result

    @constraint_literal.register
    def _(self, constraint: AllOfConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        self._reify_all(
            result, [self.constraint_literal(item) for item in constraint.items]
        )
        return result

    @constraint_literal.register
    def _(self, constraint: AnyOfConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        self._reify_any(
            result, [self.constraint_literal(item) for item in constraint.items]
        )
        return result

    @constraint_literal.register
    def _(self, constraint: NegateConstraint) -> cp_model.IntVar:
        result = self._new_literal("constraint")
        item = self.constraint_literal(constraint.item)
        self._compiled.model.add(result + item == 1)
        return result

    def _new_literal(self, prefix: str) -> cp_model.IntVar:
        self._literal_index += 1
        return self._compiled.model.new_bool_var(f"{prefix}_{self._literal_index}")

    def _new_scalar_name(self, prefix: str) -> str:
        self._scalar_index += 1
        return f"{prefix}_{self._scalar_index}"

    def _reify_all(self, result: cp_model.IntVar, items: list[cp_model.IntVar]) -> None:
        if not items:
            self._compiled.model.add(result == 1)
            return
        self._compiled.model.add_bool_and(items).only_enforce_if(result)
        self._compiled.model.add_bool_or(
            [item.Not() for item in items]
        ).only_enforce_if(result.Not())

    def _reify_any(self, result: cp_model.IntVar, items: list[cp_model.IntVar]) -> None:
        if not items:
            self._compiled.model.add(result == 0)
            return
        self._compiled.model.add_bool_or(items).only_enforce_if(result)
        self._compiled.model.add_bool_and(
            [item.Not() for item in items]
        ).only_enforce_if(result.Not())


class _ObjectiveCompiler:
    def __init__(
        self,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        snapshot: ClusterSnapshot,
    ):
        self._compiled = compiled
        self._resolver = resolver
        self._snapshot = snapshot
        self._constraints = _ConstraintCompiler(compiled, resolver, snapshot)
        self._product_index = 0

    def scalar(self, expression: ScalarExpr) -> cp_model.LinearExprT:
        return self._constraints.scalar(expression)

    def scaled_scalar(self, expression: ScalarExpr) -> _ScaledScalar:
        return self._constraints.scaled_scalar(expression)

    @singledispatchmethod
    def vector(
        self,
        expression: VectorExpr,
        solver: cp_model.CpSolver,
        fixed_spmds: set[str],
    ) -> _VectorObjective:
        raise PlacementContractError(
            f"unsupported vector objective expression {expression.kind!r}"
        )

    @vector.register(WeightedDominantShareExpr)
    def _(
        self,
        expression: WeightedDominantShareExpr,
        solver: cp_model.CpSolver,
        fixed_spmds: set[str],
    ) -> _VectorObjective:
        return self._weighted_dominant_shares(expression, solver, fixed_spmds)

    @vector.register(OccupancyExpr)
    def _(
        self,
        expression: OccupancyExpr,
        solver: cp_model.CpSolver,
        fixed_spmds: set[str],
    ) -> _VectorObjective:
        size = len(self._resolver.worker_indices(expression.workers))
        return _VectorObjective(
            components=tuple(
                self._constraints.occupancy(expression.workers, node_index)
                for node_index in range(len(self._snapshot.nodes))
            ),
            upper_bounds=(size,) * len(self._snapshot.nodes),
        )

    @vector.register(TargetOccupancyExpr)
    def _(
        self,
        expression: TargetOccupancyExpr,
        solver: cp_model.CpSolver,
        fixed_spmds: set[str],
    ) -> _VectorObjective:
        components = []
        upper_bounds = []
        for target in self._resolver.affinity_target_worker_sets():
            size = len(self._resolver.worker_indices(target))
            for node_index in range(len(self._snapshot.nodes)):
                components.append(self._constraints.occupancy(target, node_index))
                upper_bounds.append(size)
        return _VectorObjective(
            components=tuple(components), upper_bounds=tuple(upper_bounds)
        )

    def _weighted_dominant_shares(
        self,
        expression: WeightedDominantShareExpr,
        solver: cp_model.CpSolver,
        fixed_spmds: set[str],
    ) -> _VectorObjective:
        selected_spmds = self._resolver.scalable_spmd_ids(expression.workers)
        base_usage: defaultdict[str, int] = defaultdict(int)
        for worker in self._compiled.workers:
            for resource, quantity in worker.profile.placement_base.items():
                base_usage[resource] += self._compiled.resource_units.units(
                    resource, quantity
                )
            if worker.spmd_id in fixed_spmds and worker.profile.scale is not None:
                quanta = solver.value(self._compiled.allocated_quanta[worker.spmd_id])
                for resource, quantity in worker.profile.scale.step.items():
                    base_usage[resource] += (
                        self._compiled.resource_units.units(resource, quantity) * quanta
                    )
        remaining = {
            resource: self._compiled.resource_units.units(resource, quantity)
            - base_usage.get(resource, 0)
            for resource, quantity in self._snapshot.available.items()
        }
        ratios: dict[str, tuple[Fraction, ...]] = {}
        max_steps: dict[str, int] = {}
        for spmd_id in selected_spmds:
            spmd_workers = [
                worker for worker in self._compiled.workers if worker.spmd_id == spmd_id
            ]
            scale = spmd_workers[0].profile.scale
            if scale is None:
                raise PlacementContractError(
                    f"SPMD {spmd_id!r} lost its scale during objective compilation"
                )
            shares = []
            for resource, quantity in scale.step.items():
                capacity = remaining.get(resource, 0)
                if capacity <= 0:
                    continue
                shares.append(
                    Fraction(
                        self._compiled.resource_units.units(resource, quantity)
                        * len(spmd_workers),
                        capacity,
                    )
                    / Fraction(scale.weight)
                )
            ratios[spmd_id] = tuple(shares)
            max_steps[spmd_id] = scale.max_steps
        denominator = (
            _checked_int64(
                math.lcm(
                    *(
                        ratio.denominator
                        for shares in ratios.values()
                        for ratio in shares
                    )
                ),
                "weighted DRF common denominator",
            )
            if any(ratios.values())
            else 1
        )
        components: list[cp_model.LinearExprT] = []
        upper_bounds = []
        for spmd_id in selected_spmds:
            coefficients = [
                _checked_int64(
                    int(ratio * denominator),
                    f"weighted DRF coefficient for {spmd_id!r}",
                )
                for ratio in ratios[spmd_id]
            ]
            if not coefficients:
                components.append(0)
                upper_bounds.append(0)
                continue
            upper = _checked_product(
                max(coefficients),
                max_steps[spmd_id],
                label=f"weighted DRF upper bound for {spmd_id!r}",
            )
            share = self._compiled.model.new_int_var(
                0, upper, self._product_name("weighted_dominant_share")
            )
            self._compiled.model.add_max_equality(
                share,
                [
                    coefficient * self._compiled.allocated_quanta[spmd_id]
                    for coefficient in coefficients
                ],
            )
            components.append(share)
            upper_bounds.append(upper)
        return _VectorObjective(
            components=tuple(components),
            upper_bounds=tuple(upper_bounds),
            value_scale=denominator,
        )

    def _product_name(self, prefix: str) -> str:
        self._product_index += 1
        return f"{prefix}_{self._product_index}"
