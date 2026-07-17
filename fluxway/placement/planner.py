from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import singledispatchmethod
from typing import Any, cast

from ortools.sat.python import cp_model

from fluxway.resources import normalize_resource_vector

from .compiler import (
    _CompiledModel,
    _ConstraintCompiler,
    _ObjectiveCompiler,
    _PlacementUnit,
    _ResourceUnits,
    _VectorObjective,
    _Worker,
    _checked_int64,
    _checked_product,
)
from .mathematics import (
    AffinityCoverageExpr,
    CoLocationMassExpr,
    MaxMinObjective,
    MaximizeMajorizationObjective,
    MaximizeObjective,
    MinimizeObjective,
    Objective,
    OccupancyExpr,
    PlacementModel,
    PlacementRecipe,
    RankExpr,
    RanksExpr,
    TargetOccupancyExpr,
    VectorExpr,
    WeightedDominantShareExpr,
)
from .models import (
    ClusterSnapshot,
    ConstraintResult,
    ObjectiveResult,
    PlacementAttestationError,
    PlacementContractError,
    PlacementError,
    PlacementInfeasibleError,
    PlacementSolveTimeoutError,
    ResolvedPlacementPlan,
    WorkerAssignment,
    DEFAULT_MAX_LEXICOGRAPHIC_SOLVES,
    DEFAULT_MAX_MODEL_CONSTRAINTS,
    DEFAULT_MAX_MODEL_VARIABLES,
    DEFAULT_MAX_OBJECTIVE_COMPONENTS,
    DEFAULT_MAX_SORTING_COMPARATORS,
)
from .inventory import PlacementInventory
from .semantics import _AssignmentEvaluator, _ExpressionResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ObjectiveComplexity:
    components: int
    sorting_comparators: int
    lexicographic_solves: int
    auxiliary_variables: int = 0
    auxiliary_constraints: int = 0


class PlacementPlanner:
    def __init__(
        self,
        *,
        solve_timeout_s: float,
        driver_resources: Mapping[str, Decimal] | None = None,
        max_objective_components: int = DEFAULT_MAX_OBJECTIVE_COMPONENTS,
        max_sorting_comparators: int = DEFAULT_MAX_SORTING_COMPARATORS,
        max_lexicographic_solves: int = DEFAULT_MAX_LEXICOGRAPHIC_SOLVES,
        max_model_variables: int = DEFAULT_MAX_MODEL_VARIABLES,
        max_model_constraints: int = DEFAULT_MAX_MODEL_CONSTRAINTS,
    ):
        if solve_timeout_s <= 0:
            raise ValueError("solve_timeout_s must be positive")
        self._solve_timeout_s = solve_timeout_s
        self._driver_resources = normalize_resource_vector(driver_resources or {})
        limits = {
            "max_objective_components": max_objective_components,
            "max_sorting_comparators": max_sorting_comparators,
            "max_lexicographic_solves": max_lexicographic_solves,
            "max_model_variables": max_model_variables,
            "max_model_constraints": max_model_constraints,
        }
        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._max_objective_components = max_objective_components
        self._max_sorting_comparators = max_sorting_comparators
        self._max_lexicographic_solves = max_lexicographic_solves
        self._max_model_variables = max_model_variables
        self._max_model_constraints = max_model_constraints

    @staticmethod
    def sorting_comparator_count(component_count: int) -> int:
        if component_count < 0:
            raise ValueError("component_count must be non-negative")
        if component_count < 2:
            return 0
        padded_size = 1 << (component_count - 1).bit_length()
        levels = padded_size.bit_length() - 1
        return padded_size * levels * (levels + 1) // 4

    def _require_complexity(
        self,
        recipe: PlacementRecipe,
        resolver: _ExpressionResolver,
        workers: tuple[_Worker, ...],
        units: tuple[_PlacementUnit, ...],
        snapshot: ClusterSnapshot,
    ) -> None:
        estimates = tuple(
            self._objective_complexity(
                objective,
                resolver,
                len(snapshot.nodes),
            )
            for objective in recipe.objective.objectives
        )
        complexity = _ObjectiveComplexity(
            components=sum(estimate.components for estimate in estimates),
            sorting_comparators=sum(
                estimate.sorting_comparators for estimate in estimates
            ),
            lexicographic_solves=sum(
                estimate.lexicographic_solves for estimate in estimates
            ),
            auxiliary_variables=sum(
                estimate.auxiliary_variables for estimate in estimates
            ),
            auxiliary_constraints=sum(
                estimate.auxiliary_constraints for estimate in estimates
            ),
        )
        model_variables, model_constraints = self._model_size_estimate(
            recipe, workers, units, snapshot, complexity
        )
        logger.info(
            "placement.program.complexity objective_components=%s "
            "sorting_comparators=%s lexicographic_solves=%s "
            "model_variables=%s model_constraints=%s",
            complexity.components,
            complexity.sorting_comparators,
            complexity.lexicographic_solves,
            model_variables,
            model_constraints,
        )
        limits = (
            (
                "objective components",
                complexity.components,
                self._max_objective_components,
            ),
            (
                "sorting comparators",
                complexity.sorting_comparators,
                self._max_sorting_comparators,
            ),
            (
                "lexicographic solves",
                complexity.lexicographic_solves,
                self._max_lexicographic_solves,
            ),
            ("model variables", model_variables, self._max_model_variables),
            ("model constraints", model_constraints, self._max_model_constraints),
        )
        for label, estimate, limit in limits:
            if estimate > limit:
                raise PlacementContractError(
                    f"placement {label} {estimate} exceed {limit} for recipe "
                    f"{recipe.recipe_id!r}"
                )

    @singledispatchmethod
    def _objective_complexity(
        self,
        objective: Objective,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> _ObjectiveComplexity:
        raise PlacementContractError(
            f"unsupported objective AST node {objective.kind!r}"
        )

    @_objective_complexity.register(MinimizeObjective)
    @_objective_complexity.register(MaximizeObjective)
    def _(
        self,
        objective: MinimizeObjective | MaximizeObjective,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> _ObjectiveComplexity:
        variables, constraints = self._scalar_auxiliary_size(
            objective.expression, resolver, node_count
        )
        return _ObjectiveComplexity(1, 0, 1, variables, constraints)

    @_objective_complexity.register(MaxMinObjective)
    def _(
        self,
        objective: MaxMinObjective,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> _ObjectiveComplexity:
        component_count = self._vector_component_count(
            objective.expression, resolver, node_count
        )
        variables, constraints = self._vector_auxiliary_size(
            objective.expression, resolver, node_count
        )
        return _ObjectiveComplexity(
            component_count,
            self.sorting_comparator_count(component_count),
            max(1, component_count),
            variables,
            constraints,
        )

    @_objective_complexity.register(MaximizeMajorizationObjective)
    def _(
        self,
        objective: MaximizeMajorizationObjective,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> _ObjectiveComplexity:
        component_count = self._vector_component_count(
            objective.expression, resolver, node_count
        )
        variables, constraints = self._vector_auxiliary_size(
            objective.expression, resolver, node_count
        )
        return _ObjectiveComplexity(
            component_count,
            self.sorting_comparator_count(component_count),
            max(1, component_count),
            variables,
            constraints,
        )

    @singledispatchmethod
    def _scalar_auxiliary_size(
        self,
        expression: PlacementModel,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> tuple[int, int]:
        return 0, 0

    @_scalar_auxiliary_size.register(AffinityCoverageExpr)
    def _(
        self,
        expression: AffinityCoverageExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> tuple[int, int]:
        size = len(resolver.support_affinity_constraints()) * node_count
        return size, 2 * size

    @_scalar_auxiliary_size.register(CoLocationMassExpr)
    def _(
        self,
        expression: CoLocationMassExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> tuple[int, int]:
        size = len(resolver.support_affinity_constraints()) * node_count
        return size, size

    @singledispatchmethod
    def _vector_auxiliary_size(
        self,
        expression: VectorExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> tuple[int, int]:
        return 0, 0

    @_vector_auxiliary_size.register(WeightedDominantShareExpr)
    def _(
        self,
        expression: WeightedDominantShareExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> tuple[int, int]:
        size = len(resolver.scalable_spmd_ids(expression.workers))
        return size, size

    @singledispatchmethod
    def _vector_component_count(
        self,
        expression: VectorExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> int:
        raise PlacementContractError(
            f"unsupported vector objective expression {expression.kind!r}"
        )

    @_vector_component_count.register(OccupancyExpr)
    def _(
        self,
        expression: OccupancyExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> int:
        return node_count

    @_vector_component_count.register(TargetOccupancyExpr)
    def _(
        self,
        expression: TargetOccupancyExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> int:
        return len(resolver.affinity_target_worker_sets()) * node_count

    @_vector_component_count.register(WeightedDominantShareExpr)
    def _(
        self,
        expression: WeightedDominantShareExpr,
        resolver: _ExpressionResolver,
        node_count: int,
    ) -> int:
        return len(resolver.scalable_spmd_ids(expression.workers))

    def _model_size_estimate(
        self,
        recipe: PlacementRecipe,
        workers: tuple[_Worker, ...],
        units: tuple[_PlacementUnit, ...],
        snapshot: ClusterSnapshot,
        objective: _ObjectiveComplexity,
    ) -> tuple[int, int]:
        node_count = len(snapshot.nodes)
        unit_count = len(units)
        spmd_count = len({worker.spmd_id for worker in workers})
        elastic_unit_count = sum(unit.profile.scale is not None for unit in units)
        symmetry_pair_count = len(self._node_symmetry_pairs(snapshot))
        resource_count = len(self._resource_names(workers, snapshot))
        allocation_vector_count = self._allocation_vector_count(workers)
        ast_nodes = self._ast_node_count(recipe)

        variables = (
            spmd_count
            + unit_count * node_count
            + symmetry_pair_count * (4 * unit_count + 1)
            + elastic_unit_count * node_count
            + node_count * (allocation_vector_count + resource_count)
            + ast_nodes * node_count
            + objective.auxiliary_variables
            + 2 * objective.sorting_comparators
        )
        constraints = (
            unit_count
            + symmetry_pair_count * (8 * unit_count + 2)
            + elastic_unit_count * node_count
            + node_count * resource_count
            + resource_count
            + 2 * node_count * resource_count
            + 2 * ast_nodes * node_count
            + objective.auxiliary_constraints
            + 2 * objective.sorting_comparators
            + objective.components
        )
        return variables, constraints

    def _ast_node_count(self, value: Any) -> int:
        if isinstance(value, PlacementModel):
            return 1 + sum(
                self._ast_node_count(getattr(value, field_name))
                for field_name in type(value).model_fields
            )
        if isinstance(value, Mapping):
            return sum(self._ast_node_count(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(self._ast_node_count(item) for item in value)
        return 0

    def _node_symmetry_pairs(
        self, snapshot: ClusterSnapshot
    ) -> tuple[tuple[int, int], ...]:
        nodes_by_shape: defaultdict[tuple[tuple[str, Decimal], ...], list[int]] = (
            defaultdict(list)
        )
        for node_index, node in enumerate(snapshot.nodes):
            shape = tuple(
                (resource, quantity)
                for resource, quantity in node.resources.items()
                if not resource.startswith("node:")
            )
            nodes_by_shape[shape].append(node_index)
        return tuple(
            pair
            for node_indices in nodes_by_shape.values()
            for pair in zip(node_indices, node_indices[1:], strict=False)
        )

    def _resource_names(
        self,
        workers: tuple[_Worker, ...],
        snapshot: ClusterSnapshot,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(snapshot.available)
                | {
                    resource
                    for worker in workers
                    for resource in (
                        set(worker.profile.base)
                        | set(worker.profile.ray_component_limits)
                        | set(worker.driver_resources)
                        | (
                            set(worker.profile.scale.step)
                            if worker.profile.scale is not None
                            else set()
                        )
                    )
                }
            )
        )

    def _allocation_vector_count(self, workers: tuple[_Worker, ...]) -> int:
        vectors: set[tuple[tuple[str, Decimal], ...]] = set()
        for worker in workers:
            candidates = [
                worker.profile.base,
                worker.profile.ray_component_limits,
                worker.driver_resources,
            ]
            if worker.profile.scale is not None:
                candidates.append(worker.profile.scale.step)
            for candidate in candidates:
                key = tuple(
                    sorted(
                        (resource, quantity)
                        for resource, quantity in candidate.items()
                        if quantity > 0
                    )
                )
                if key:
                    vectors.add(key)
        return len(vectors)

    def resolve(
        self,
        inventory: PlacementInventory,
        recipe: PlacementRecipe,
        snapshot: ClusterSnapshot,
    ) -> ResolvedPlacementPlan:
        deadline = time.monotonic() + self._solve_timeout_s
        logger.info(
            "placement.recipe.selected recipe_id=%s version=%s",
            recipe.recipe_id,
            recipe.version,
        )
        logger.info(
            "placement.capacity.snapshot nodes=%s available=%s",
            len(snapshot.nodes),
            snapshot.available,
        )
        self._require_exclusive_resources(inventory, snapshot)
        workers = self._expand_workers(inventory)
        resolver = _ExpressionResolver(inventory, workers, recipe.constraints)
        logger.info(
            "placement.problem.created constraints=%s workers=%s recipe=%r",
            len(recipe.constraints),
            len(workers),
            recipe,
        )
        units = self._placement_units(workers, resolver, recipe)
        self._require_complexity(recipe, resolver, workers, units, snapshot)
        compiled = self._compile(workers, units, resolver, recipe, snapshot)
        for index, constraint in enumerate(recipe.constraints):
            logger.info(
                "placement.constraint.validated index=%s kind=%s",
                index,
                constraint.kind,
            )
        self._require_feasible(compiled, recipe, deadline)
        solver, objective_results = self._solve_objectives(
            compiled, resolver, recipe, snapshot, deadline
        )
        allocated_quanta = {
            spmd_id: solver.value(variable)
            for spmd_id, variable in compiled.allocated_quanta.items()
        }
        assignments = self._materialize(compiled, solver, snapshot, allocated_quanta)
        unused = self._unused_resources(assignments, snapshot)
        occupancy, support = self._placement_sets(assignments)
        fragmentation_score = self._fragmentation_score(compiled, solver)
        constraint_results = self._constraint_results(
            recipe,
            resolver,
            assignments,
            allocated_quanta,
            snapshot,
            fragmentation_score,
            compiled.fragmentation_scale,
        )
        fragmentation = self._fragmentation_report(compiled, solver, snapshot)
        plan = ResolvedPlacementPlan(
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.version,
            recipe=recipe,
            assignments=assignments,
            allocated_quanta=allocated_quanta,
            objective_results=objective_results,
            constraint_results=constraint_results,
            occupancy=occupancy,
            support=support,
            resource_fragmentation=fragmentation,
            resource_fragmentation_score=fragmentation_score,
            resource_fragmentation_scale=compiled.fragmentation_scale,
            unused_resources=unused,
        )
        logger.info(
            "placement.solution.resolved bundles=%s workers=%s",
            len(plan.bundle_resources()),
            len(plan.assignments),
        )
        return plan

    def _require_exclusive_resources(
        self,
        inventory: PlacementInventory,
        snapshot: ClusterSnapshot,
    ) -> None:
        requested = {
            resource
            for resource, quantity in self._driver_resources.items()
            if quantity > 0
        }
        requested.update(inventory.resource_names)
        cluster_resources = {
            resource for node in snapshot.nodes for resource in node.resources
        }
        missing = sorted(requested - cluster_resources)
        if missing:
            raise PlacementContractError(
                f"Ray cluster does not expose requested resources: {missing}"
            )
        nonexclusive = {}
        for resource in sorted(requested):
            total = sum(
                (node.resources.get(resource, Decimal(0)) for node in snapshot.nodes),
                Decimal(0),
            )
            available = snapshot.available.get(resource, Decimal(0))
            if available != total:
                nonexclusive[resource] = {
                    "available": available,
                    "total": total,
                }
        if nonexclusive:
            raise PlacementContractError(
                "exact placement requires exclusive access to requested Ray "
                f"resources: {nonexclusive}"
            )

    def attest(
        self,
        inventory: PlacementInventory,
        recipe: PlacementRecipe,
        snapshot: ClusterSnapshot,
        plan: ResolvedPlacementPlan,
        actual_node_ids: Mapping[str, tuple[str, ...]],
    ) -> ResolvedPlacementPlan:
        if plan.recipe != recipe:
            raise PlacementAttestationError(
                "resolved plan and attestation Recipe differ"
            )
        workers = self._expand_workers(inventory)
        resolver = _ExpressionResolver(inventory, workers, recipe.constraints)
        expected_spmd_ids = {worker.spmd_id for worker in workers}
        if set(actual_node_ids) != expected_spmd_ids:
            raise PlacementAttestationError(
                "attestation SPMD IDs differ from the resolved plan: "
                f"expected={sorted(expected_spmd_ids)} "
                f"actual={sorted(actual_node_ids)}"
            )
        assignment_by_worker = {
            (assignment.spmd_id, assignment.rank): assignment
            for assignment in plan.assignments
        }
        attested_assignments = []
        for worker in workers:
            actual_for_spmd = actual_node_ids[worker.spmd_id]
            if len(actual_for_spmd) != sum(
                candidate.spmd_id == worker.spmd_id for candidate in workers
            ):
                raise PlacementAttestationError(
                    f"SPMD {worker.spmd_id!r} attested {len(actual_for_spmd)} "
                    "node IDs for a different worker count"
                )
            planned = assignment_by_worker[(worker.spmd_id, worker.rank)]
            actual_node_id = actual_for_spmd[worker.rank]
            if actual_node_id != planned.node_id:
                raise PlacementAttestationError(
                    f"placement attestation failed spmd_id={worker.spmd_id} "
                    f"rank={worker.rank} expected={planned.node_id!r} "
                    f"actual={actual_node_id!r}"
                )
            attested_assignments.append(
                planned.model_copy(update={"node_id": actual_node_id})
            )
        assignments = tuple(attested_assignments)
        occupancy, support = self._placement_sets(assignments)
        constraint_results = self._constraint_results(
            recipe,
            resolver,
            assignments,
            plan.allocated_quanta,
            snapshot,
            plan.resource_fragmentation_score,
            plan.resource_fragmentation_scale,
        )
        return cast(
            ResolvedPlacementPlan,
            plan.model_copy(
                update={
                    "assignments": assignments,
                    "constraint_results": constraint_results,
                    "occupancy": occupancy,
                    "support": support,
                }
            ),
        )

    def _expand_workers(self, inventory: PlacementInventory) -> tuple[_Worker, ...]:
        workers: list[_Worker] = []
        for declaration in inventory.declarations:
            for rank in range(declaration.world_size):
                workers.append(
                    _Worker(
                        index=len(workers),
                        spmd_id=declaration.spmd_id,
                        rank=rank,
                        profile=declaration.resource_profile,
                        driver_resources=(self._driver_resources if rank == 0 else {}),
                    )
                )
        return tuple(workers)

    def _worker_resources(
        self, worker: _Worker, units: Mapping[str, int]
    ) -> dict[str, Decimal]:
        return dict(worker.profile.resources_at(units[worker.spmd_id]))

    def _placement_resources(
        self, worker: _Worker, units: Mapping[str, int]
    ) -> dict[str, Decimal]:
        resources = dict(worker.profile.placement_resources_at(units[worker.spmd_id]))
        for resource, quantity in worker.driver_resources.items():
            resources[resource] = resources.get(resource, Decimal(0)) + quantity
        return resources

    def _require_feasible(
        self,
        compiled: _CompiledModel,
        recipe: PlacementRecipe,
        deadline: float,
    ) -> None:
        solver = self._solver()
        self._apply_remaining_solve_budget(solver, deadline, "feasibility")
        status = solver.solve(compiled.model)
        if status == cp_model.OPTIMAL:
            return
        if status == cp_model.INFEASIBLE:
            raise PlacementInfeasibleError(
                f"placement base resources are infeasible for recipe {recipe.recipe_id}"
            )
        if status == cp_model.UNKNOWN:
            raise PlacementSolveTimeoutError(
                f"placement feasibility solve timed out for recipe {recipe.recipe_id}"
            )
        raise PlacementError(
            f"placement feasibility solver returned {solver.status_name(status)}"
        )

    def _placement_units(
        self,
        workers: tuple[_Worker, ...],
        resolver: _ExpressionResolver,
        recipe: PlacementRecipe,
    ) -> tuple[_PlacementUnit, ...]:
        sensitive: set[int] = {
            worker.index for worker in workers if worker.driver_resources
        }

        def collect(value: Any) -> None:
            if isinstance(value, (RankExpr, RanksExpr)):
                sensitive.update(resolver.worker_indices(value))
            if isinstance(value, PlacementModel):
                for field_name in type(value).model_fields:
                    collect(getattr(value, field_name))
            elif isinstance(value, Mapping):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(recipe)
        raw_units: list[tuple[int, ...]] = []
        for spmd_id in sorted({worker.spmd_id for worker in workers}):
            spmd_workers = tuple(
                worker.index for worker in workers if worker.spmd_id == spmd_id
            )
            raw_units.extend((index,) for index in spmd_workers if index in sensitive)
            cohort = tuple(index for index in spmd_workers if index not in sensitive)
            if cohort:
                raw_units.append(cohort)
        ordered = sorted(raw_units, key=lambda indices: indices[0])
        return tuple(
            _PlacementUnit(
                index=unit_index,
                worker_indices=worker_indices,
                spmd_id=workers[worker_indices[0]].spmd_id,
                profile=workers[worker_indices[0]].profile,
                driver_resources=workers[worker_indices[0]].driver_resources,
            )
            for unit_index, worker_indices in enumerate(ordered)
        )

    def _compile(
        self,
        workers: tuple[_Worker, ...],
        units: tuple[_PlacementUnit, ...],
        resolver: _ExpressionResolver,
        recipe: PlacementRecipe,
        snapshot: ClusterSnapshot,
    ) -> _CompiledModel:
        model = cp_model.CpModel()
        worker_to_unit = {
            worker_index: unit.index
            for unit in units
            for worker_index in unit.worker_indices
        }
        spmd_profiles = {worker.spmd_id: worker.profile for worker in workers}
        allocated_quanta = {
            spmd_id: model.new_int_var(
                0,
                profile.scale.max_steps if profile.scale is not None else 0,
                f"quanta_{spmd_id}",
            )
            for spmd_id, profile in sorted(spmd_profiles.items())
        }
        assignments = {
            (unit.index, node_index): model.new_int_var(
                0,
                unit.size,
                f"x_{unit.index}_{node_index}",
            )
            for unit in units
            for node_index in range(len(snapshot.nodes))
        }
        model.add_decision_strategy(
            [assignments[key] for key in sorted(assignments)],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )
        for unit in units:
            model.add(
                sum(
                    assignments[(unit.index, node_index)]
                    for node_index in range(len(snapshot.nodes))
                )
                == unit.size
            )
        for pair_index, (left_node, right_node) in enumerate(
            self._node_symmetry_pairs(snapshot)
        ):
            self._add_lexicographic_greater_equal(
                model,
                tuple(assignments[(unit.index, left_node)] for unit in units),
                tuple(assignments[(unit.index, right_node)] for unit in units),
                f"node_symmetry_{pair_index}",
            )
        resource_names = self._resource_names(workers, snapshot)
        resource_units = _ResourceUnits.derive(workers, snapshot, resource_names)
        scale_products: dict[tuple[int, int], cp_model.IntVar] = {}
        for unit in units:
            if unit.profile.scale is None:
                continue
            for node_index in range(len(snapshot.nodes)):
                product = model.new_int_var(
                    0,
                    _checked_product(
                        unit.profile.scale.max_steps,
                        unit.size,
                        label=f"elastic placement unit {unit.index}",
                    ),
                    f"quanta_on_node_{unit.index}_{node_index}",
                )
                model.add_multiplication_equality(
                    product,
                    [
                        allocated_quanta[unit.spmd_id],
                        assignments[(unit.index, node_index)],
                    ],
                )
                scale_products[(unit.index, node_index)] = product
        node_usage: dict[tuple[int, str], cp_model.LinearExprT] = {}
        for node_index, node in enumerate(snapshot.nodes):
            for resource in resource_names:
                usage_upper = sum(
                    resource_units.units(
                        resource,
                        unit.profile.placement_base.get(resource, Decimal(0)),
                    )
                    * unit.size
                    + resource_units.units(
                        resource,
                        unit.driver_resources.get(resource, Decimal(0)),
                    )
                    * unit.size
                    + (
                        resource_units.units(
                            resource,
                            unit.profile.scale.step.get(resource, Decimal(0)),
                        )
                        * unit.profile.scale.max_steps
                        * unit.size
                        if unit.profile.scale is not None
                        else 0
                    )
                    for unit in units
                )
                _checked_int64(
                    usage_upper,
                    f"node {node.node_id!r} {resource!r} usage upper bound",
                )
                usage = sum(
                    resource_units.units(
                        resource,
                        unit.profile.placement_base.get(resource, Decimal(0)),
                    )
                    * assignments[(unit.index, node_index)]
                    + resource_units.units(
                        resource,
                        unit.driver_resources.get(resource, Decimal(0)),
                    )
                    * assignments[(unit.index, node_index)]
                    + (
                        resource_units.units(
                            resource, unit.profile.scale.step.get(resource, Decimal(0))
                        )
                        * scale_products[(unit.index, node_index)]
                        if unit.profile.scale is not None
                        else 0
                    )
                    for unit in units
                )
                node_usage[(node_index, resource)] = usage
                model.add(
                    usage
                    <= resource_units.units(
                        resource, node.resources.get(resource, Decimal(0))
                    )
                )
        for resource in resource_names:
            aggregate_upper = sum(
                resource_units.units(
                    resource,
                    worker.profile.placement_base.get(resource, Decimal(0)),
                )
                + resource_units.units(
                    resource,
                    worker.driver_resources.get(resource, Decimal(0)),
                )
                + (
                    resource_units.units(
                        resource,
                        worker.profile.scale.step.get(resource, Decimal(0)),
                    )
                    * worker.profile.scale.max_steps
                    if worker.profile.scale is not None
                    else 0
                )
                for worker in workers
            )
            _checked_int64(
                aggregate_upper,
                f"aggregate {resource!r} usage upper bound",
            )
            aggregate_usage = sum(
                resource_units.units(
                    resource,
                    worker.profile.placement_base.get(resource, Decimal(0)),
                )
                + resource_units.units(
                    resource,
                    worker.driver_resources.get(resource, Decimal(0)),
                )
                + (
                    resource_units.units(
                        resource,
                        worker.profile.scale.step.get(resource, Decimal(0)),
                    )
                    * allocated_quanta[worker.spmd_id]
                    if worker.profile.scale is not None
                    else 0
                )
                for worker in workers
            )
            model.add(
                aggregate_usage
                <= resource_units.units(
                    resource, snapshot.available.get(resource, Decimal(0))
                )
            )
        unit_keys: set[tuple[tuple[str, int], ...]] = set()
        for profile in spmd_profiles.values():
            vectors = [profile.placement_base]
            if profile.scale is not None:
                vectors.append(profile.scale.step)
            for vector in vectors:
                allocation_key = tuple(
                    sorted(
                        (resource, resource_units.units(resource, quantity))
                        for resource, quantity in vector.items()
                        if quantity > 0
                    )
                )
                if allocation_key:
                    unit_keys.add(allocation_key)
        for worker in workers:
            allocation_key = tuple(
                sorted(
                    (resource, resource_units.units(resource, quantity))
                    for resource, quantity in worker.driver_resources.items()
                    if quantity > 0
                )
            )
            if allocation_key:
                unit_keys.add(allocation_key)
        allocation_units = tuple(dict(unit) for unit in sorted(unit_keys))
        fragmentation_quanta = {
            resource: math.gcd(
                *(unit[resource] for unit in allocation_units if resource in unit)
            )
            for resource in sorted(
                {resource for unit in allocation_units for resource in unit}
            )
        }
        fragmentation_scale = (
            _checked_int64(
                math.lcm(*fragmentation_quanta.values()),
                "resource fragmentation scale",
            )
            if fragmentation_quanta
            else 1
        )
        fragmentation_weights = {
            resource: fragmentation_scale // quantum
            for resource, quantum in fragmentation_quanta.items()
        }
        fragmentation: dict[tuple[int, str], cp_model.IntVar] = {}
        for node_index, node in enumerate(snapshot.nodes):
            pack_counts = []
            for unit_index, unit_vector in enumerate(allocation_units):
                upper = min(
                    resource_units.units(
                        resource, node.resources.get(resource, Decimal(0))
                    )
                    // quantity
                    for resource, quantity in unit_vector.items()
                )
                pack_counts.append(
                    model.new_int_var(
                        0, upper, f"fragmentation_pack_{node_index}_{unit_index}"
                    )
                )
            fragmentation_resources = sorted(
                {resource for unit in allocation_units for resource in unit}
            )
            for resource in fragmentation_resources:
                fragmentation_upper = _checked_product(
                    resource_units.units(
                        resource, node.resources.get(resource, Decimal(0))
                    ),
                    fragmentation_weights[resource],
                    label=(
                        f"node {node.node_id!r} {resource!r} fragmentation upper bound"
                    ),
                )
                packed = cp_model.LinearExpr.weighted_sum(
                    pack_counts,
                    [unit.get(resource, 0) for unit in allocation_units],
                )
                residual = -node_usage[(node_index, resource)] + resource_units.units(
                    resource, node.resources.get(resource, Decimal(0))
                )
                model.add(packed <= residual)
                remainder = model.new_int_var(
                    0,
                    fragmentation_upper // fragmentation_weights[resource],
                    f"fragmentation_{node_index}_{resource}",
                )
                model.add(remainder + packed == residual)
                fragmentation[(node_index, resource)] = remainder
        compiled = _CompiledModel(
            model=model,
            assignments=assignments,
            workers=workers,
            units=units,
            worker_to_unit=worker_to_unit,
            resource_units=resource_units,
            worker_set_cache={},
            occupancy_cache={},
            support_cache={},
            allocated_quanta=allocated_quanta,
            node_usage=node_usage,
            fragmentation=fragmentation,
            fragmentation_weights=fragmentation_weights,
            fragmentation_scale=fragmentation_scale,
        )
        constraint_compiler = _ConstraintCompiler(compiled, resolver, snapshot)
        for constraint in recipe.constraints:
            literal = constraint_compiler.constraint_literal(constraint)
            model.add(literal == 1)
        return compiled

    def _add_lexicographic_greater_equal(
        self,
        model: cp_model.CpModel,
        left: tuple[cp_model.LinearExprT, ...],
        right: tuple[cp_model.LinearExprT, ...],
        prefix: str,
    ) -> None:
        equal_prefix = model.new_bool_var(f"{prefix}_equal_0")
        model.add(equal_prefix == 1)
        greater_terms = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            equal = model.new_bool_var(f"{prefix}_item_{index}_equal")
            model.add(left_item == right_item).only_enforce_if(equal)
            model.add(left_item != right_item).only_enforce_if(equal.Not())
            greater = model.new_bool_var(f"{prefix}_item_{index}_greater")
            model.add(left_item > right_item).only_enforce_if(greater)
            model.add(left_item <= right_item).only_enforce_if(greater.Not())
            first_greater = model.new_bool_var(f"{prefix}_item_{index}_first_greater")
            model.add_bool_and([equal_prefix, greater]).only_enforce_if(first_greater)
            model.add_bool_or([equal_prefix.Not(), greater.Not()]).only_enforce_if(
                first_greater.Not()
            )
            greater_terms.append(first_greater)
            next_equal_prefix = model.new_bool_var(f"{prefix}_equal_{index + 1}")
            model.add_bool_and([equal_prefix, equal]).only_enforce_if(next_equal_prefix)
            model.add_bool_or([equal_prefix.Not(), equal.Not()]).only_enforce_if(
                next_equal_prefix.Not()
            )
            equal_prefix = next_equal_prefix
        model.add_bool_or([equal_prefix, *greater_terms])

    def _solve_objectives(
        self,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        recipe: PlacementRecipe,
        snapshot: ClusterSnapshot,
        deadline: float,
    ) -> tuple[cp_model.CpSolver, tuple[ObjectiveResult, ...]]:
        objective_compiler = _ObjectiveCompiler(compiled, resolver, snapshot)
        results: list[ObjectiveResult] = []
        solver = self._solver()
        fixed_elastic_spmds: set[str] = set()
        for index, objective in enumerate(recipe.objective.objectives):
            logger.info(
                "placement.objective.started index=%s kind=%s objective_ast=%s "
                "incident_spmds=%s isolated_spmds=%s",
                index,
                objective.kind,
                repr(objective),
                sorted(resolver.incident_spmd_ids()),
                sorted(set(resolver.spmd_ids) - resolver.incident_spmd_ids()),
            )
            started = time.monotonic()
            optimum, value_scale = self._solve_objective(
                objective,
                objective_compiler,
                compiled,
                resolver,
                solver,
                fixed_elastic_spmds,
                recipe,
                index,
                deadline,
            )
            duration = time.monotonic() - started
            results.append(
                ObjectiveResult(
                    index=index,
                    kind=objective.kind,
                    optimum_value=optimum,
                    value_scale=value_scale,
                    solve_duration=duration,
                )
            )
            logger.info(
                "placement.objective.completed index=%s kind=%s optimum_value=%s "
                "value_scale=%s solve_duration=%s remaining_resources=%s",
                index,
                objective.kind,
                optimum,
                value_scale,
                duration,
                self._remaining_resources(compiled, solver, snapshot),
            )
        return solver, tuple(results)

    @singledispatchmethod
    def _solve_objective(
        self,
        objective: Objective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        solver: cp_model.CpSolver,
        fixed_elastic_spmds: set[str],
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
    ) -> tuple[int | tuple[int, ...], int]:
        raise PlacementContractError(
            f"unsupported objective AST node {objective.kind!r}"
        )

    @_solve_objective.register(MaxMinObjective)
    def _(
        self,
        objective: MaxMinObjective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        solver: cp_model.CpSolver,
        fixed_elastic_spmds: set[str],
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
    ) -> tuple[int | tuple[int, ...], int]:
        vector = objective_compiler.vector(
            objective.expression, solver, fixed_elastic_spmds
        )
        ordered = self._sort_vector(compiled, vector, f"objective_{index}_leximin")
        optimum = self._optimize_vector(
            compiled,
            solver,
            ordered,
            maximize=True,
            recipe=recipe,
            objective_index=index,
            deadline=deadline,
        )
        self._record_fixed_weighted_drf_spmds(
            objective.expression, resolver, fixed_elastic_spmds
        )
        return optimum, vector.value_scale

    @_solve_objective.register(MaximizeMajorizationObjective)
    def _(
        self,
        objective: MaximizeMajorizationObjective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        solver: cp_model.CpSolver,
        fixed_elastic_spmds: set[str],
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
    ) -> tuple[int | tuple[int, ...], int]:
        vector = objective_compiler.vector(
            objective.expression, solver, fixed_elastic_spmds
        )
        ascending = self._sort_vector(
            compiled, vector, f"objective_{index}_majorization"
        )
        optimum = self._optimize_vector(
            compiled,
            solver,
            tuple(reversed(ascending)),
            maximize=True,
            recipe=recipe,
            objective_index=index,
            deadline=deadline,
        )
        self._record_fixed_weighted_drf_spmds(
            objective.expression, resolver, fixed_elastic_spmds
        )
        return optimum, 1

    def _record_fixed_weighted_drf_spmds(
        self,
        expression: VectorExpr,
        resolver: _ExpressionResolver,
        fixed_elastic_spmds: set[str],
    ) -> None:
        if isinstance(expression, WeightedDominantShareExpr):
            fixed_elastic_spmds.update(resolver.scalable_spmd_ids(expression.workers))

    def _solve_scalar_objective(
        self,
        objective: MinimizeObjective | MaximizeObjective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
        maximize: bool,
    ) -> tuple[int | tuple[int, ...], int]:
        scalar = objective_compiler.scaled_scalar(objective.expression)
        optimum = self._optimize_scalar(
            compiled,
            solver,
            scalar.expression,
            maximize=maximize,
            recipe=recipe,
            objective_index=index,
            deadline=deadline,
        )
        return optimum, scalar.scale

    @_solve_objective.register(MinimizeObjective)
    def _(
        self,
        objective: MinimizeObjective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        solver: cp_model.CpSolver,
        fixed_elastic_spmds: set[str],
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
    ) -> tuple[int | tuple[int, ...], int]:
        return self._solve_scalar_objective(
            objective,
            objective_compiler,
            compiled,
            solver,
            recipe,
            index,
            deadline,
            False,
        )

    @_solve_objective.register(MaximizeObjective)
    def _(
        self,
        objective: MaximizeObjective,
        objective_compiler: _ObjectiveCompiler,
        compiled: _CompiledModel,
        resolver: _ExpressionResolver,
        solver: cp_model.CpSolver,
        fixed_elastic_spmds: set[str],
        recipe: PlacementRecipe,
        index: int,
        deadline: float,
    ) -> tuple[int | tuple[int, ...], int]:
        return self._solve_scalar_objective(
            objective,
            objective_compiler,
            compiled,
            solver,
            recipe,
            index,
            deadline,
            True,
        )

    def _sort_vector(
        self,
        compiled: _CompiledModel,
        vector: _VectorObjective,
        prefix: str,
    ) -> tuple[cp_model.LinearExprT, ...]:
        size = len(vector.components)
        if size < 2:
            return vector.components
        padded_size = 1 << (size - 1).bit_length()
        padding = padded_size - size
        items: list[tuple[cp_model.LinearExprT, int]] = [(0, 0) for _ in range(padding)]
        items.extend(zip(vector.components, vector.upper_bounds, strict=True))
        comparator_index = 0

        def compare_swap(
            left: tuple[cp_model.LinearExprT, int],
            right: tuple[cp_model.LinearExprT, int],
            ascending: bool,
        ) -> tuple[
            tuple[cp_model.LinearExprT, int],
            tuple[cp_model.LinearExprT, int],
        ]:
            nonlocal comparator_index
            low = compiled.model.new_int_var(
                0,
                min(left[1], right[1]),
                f"{prefix}_{comparator_index}_low",
            )
            high = compiled.model.new_int_var(
                0,
                max(left[1], right[1]),
                f"{prefix}_{comparator_index}_high",
            )
            comparator_index += 1
            compiled.model.add_min_equality(low, [left[0], right[0]])
            compiled.model.add_max_equality(high, [left[0], right[0]])
            ordered = ((low, min(left[1], right[1])), (high, max(left[1], right[1])))
            return ordered if ascending else (ordered[1], ordered[0])

        def merge(
            values: list[tuple[cp_model.LinearExprT, int]],
            ascending: bool,
        ) -> list[tuple[cp_model.LinearExprT, int]]:
            if len(values) < 2:
                return values
            half = len(values) // 2
            compared = list(values)
            for index in range(half):
                compared[index], compared[index + half] = compare_swap(
                    values[index], values[index + half], ascending
                )
            return merge(compared[:half], ascending) + merge(compared[half:], ascending)

        def sort(
            values: list[tuple[cp_model.LinearExprT, int]],
            ascending: bool,
        ) -> list[tuple[cp_model.LinearExprT, int]]:
            if len(values) < 2:
                return values
            half = len(values) // 2
            bitonic = sort(values[:half], True) + sort(values[half:], False)
            return merge(bitonic, ascending)

        ordered = sort(items, True)[padding:]
        return tuple(component for component, _ in ordered)

    def _optimize_vector(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        expressions: tuple[cp_model.LinearExprT, ...],
        *,
        maximize: bool,
        recipe: PlacementRecipe,
        objective_index: int,
        deadline: float,
    ) -> tuple[int, ...]:
        if not expressions:
            self._solve(compiled, solver, recipe, objective_index, deadline)
            return ()
        return tuple(
            self._optimize_scalar(
                compiled,
                solver,
                expression,
                maximize=maximize,
                recipe=recipe,
                objective_index=objective_index,
                deadline=deadline,
            )
            for expression in expressions
        )

    def _optimize_scalar(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        expression: cp_model.LinearExprT,
        *,
        maximize: bool,
        recipe: PlacementRecipe,
        objective_index: int | str,
        deadline: float,
    ) -> int:
        if maximize:
            compiled.model.maximize(expression)
        else:
            compiled.model.minimize(expression)
        self._solve(compiled, solver, recipe, objective_index, deadline)
        optimum = int(round(solver.objective_value))
        compiled.model.add(expression == optimum)
        compiled.model.clear_objective()
        return optimum

    def _solve(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        recipe: PlacementRecipe,
        objective_index: int | str,
        deadline: float,
    ) -> None:
        self._apply_remaining_solve_budget(
            solver, deadline, f"objective {objective_index}"
        )
        status = solver.solve(compiled.model)
        if status in (cp_model.UNKNOWN, cp_model.FEASIBLE):
            raise PlacementSolveTimeoutError(
                f"objective {objective_index} timed out for recipe {recipe.recipe_id}"
            )
        if status == cp_model.INFEASIBLE:
            raise PlacementInfeasibleError(
                f"objective {objective_index} made recipe {recipe.recipe_id} infeasible"
            )
        if status != cp_model.OPTIMAL:
            raise PlacementError(
                f"objective {objective_index} returned {solver.status_name(status)}"
            )

    def _remaining_resources(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        snapshot: ClusterSnapshot,
    ) -> dict[str, Decimal]:
        quanta = {
            spmd_id: solver.value(variable)
            for spmd_id, variable in compiled.allocated_quanta.items()
        }
        used: defaultdict[str, Decimal] = defaultdict(Decimal)
        for worker in compiled.workers:
            for resource, quantity in self._placement_resources(worker, quanta).items():
                used[resource] += quantity
        return {
            resource: quantity - used.get(resource, Decimal(0))
            for resource, quantity in snapshot.available.items()
        }

    def _solver(self) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        return solver

    def _apply_remaining_solve_budget(
        self,
        solver: cp_model.CpSolver,
        deadline: float,
        stage: str,
    ) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlacementSolveTimeoutError(
                f"placement solve deadline expired before {stage}"
            )
        solver.parameters.max_time_in_seconds = remaining

    def _materialize(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        snapshot: ClusterSnapshot,
        allocated_quanta: Mapping[str, int],
    ) -> tuple[WorkerAssignment, ...]:
        assignments: list[WorkerAssignment] = []
        for unit in compiled.units:
            worker_indices = iter(unit.worker_indices)
            selected_nodes: list[int] = []
            for node_index in range(len(snapshot.nodes)):
                count = solver.value(compiled.assignments[(unit.index, node_index)])
                selected_nodes.extend([node_index] * count)
            if len(selected_nodes) != unit.size:
                raise PlacementError(
                    f"placement unit {unit.index} has assignments {selected_nodes}"
                )
            for worker_index, node_index in zip(
                worker_indices, selected_nodes, strict=True
            ):
                worker = compiled.workers[worker_index]
                assignments.append(
                    WorkerAssignment(
                        spmd_id=worker.spmd_id,
                        rank=worker.rank,
                        node_id=snapshot.nodes[node_index].node_id,
                        resources=self._worker_resources(worker, allocated_quanta),
                        ray_component_limits=dict(worker.profile.ray_component_limits),
                        driver_resources=worker.driver_resources,
                    )
                )
        return tuple(sorted(assignments, key=lambda item: (item.spmd_id, item.rank)))

    def _placement_sets(
        self, assignments: tuple[WorkerAssignment, ...]
    ) -> tuple[dict[str, dict[str, int]], dict[str, tuple[str, ...]]]:
        counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for assignment in assignments:
            counts[assignment.spmd_id][assignment.node_id] += 1
        occupancy = {
            spmd_id: dict(sorted(node_counts.items()))
            for spmd_id, node_counts in sorted(counts.items())
        }
        support = {
            spmd_id: tuple(node_counts) for spmd_id, node_counts in occupancy.items()
        }
        return occupancy, support

    def _constraint_results(
        self,
        recipe: PlacementRecipe,
        resolver: _ExpressionResolver,
        assignments: tuple[WorkerAssignment, ...],
        allocated_quanta: Mapping[str, int],
        snapshot: ClusterSnapshot,
        resource_fragmentation_score: int,
        resource_fragmentation_scale: int,
    ) -> tuple[ConstraintResult, ...]:
        evaluator = _AssignmentEvaluator(
            resolver,
            assignments,
            allocated_quanta,
            snapshot,
            resource_fragmentation_score,
            resource_fragmentation_scale,
        )
        results = tuple(
            ConstraintResult(
                index=index,
                constraint=constraint,
                satisfied=evaluator.constraint(constraint),
            )
            for index, constraint in enumerate(recipe.constraints)
        )
        failed = [result.index for result in results if not result.satisfied]
        if failed:
            raise PlacementError(
                f"resolved placement failed constraints at indices {failed}"
            )
        return results

    def _fragmentation_report(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
        snapshot: ClusterSnapshot,
    ) -> dict[str, dict[str, Decimal]]:
        result: defaultdict[str, dict[str, Decimal]] = defaultdict(dict)
        for (node_index, resource), variable in sorted(compiled.fragmentation.items()):
            result[snapshot.nodes[node_index].node_id][resource] = (
                compiled.resource_units.quantity(resource, solver.value(variable))
            )
        return dict(result)

    def _fragmentation_score(
        self,
        compiled: _CompiledModel,
        solver: cp_model.CpSolver,
    ) -> int:
        return sum(
            solver.value(remainder) * compiled.fragmentation_weights[resource]
            for (_, resource), remainder in compiled.fragmentation.items()
        )

    def _unused_resources(
        self,
        assignments: tuple[WorkerAssignment, ...],
        snapshot: ClusterSnapshot,
    ) -> dict[str, Decimal]:
        used: defaultdict[str, Decimal] = defaultdict(Decimal)
        for assignment in assignments:
            for resource, quantity in assignment.resources.items():
                used[resource] += quantity
            for resource, quantity in assignment.ray_component_limits.items():
                used[resource] += quantity
            for resource, quantity in assignment.driver_resources.items():
                used[resource] += quantity
        return {
            resource: quantity - used.get(resource, Decimal(0))
            for resource, quantity in snapshot.available.items()
        }
