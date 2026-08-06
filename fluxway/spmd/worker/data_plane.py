"""Pure sender-side data-plane frame construction."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from collections import defaultdict
from dataclasses import dataclass, field, replace
import inspect
import logging
from types import MappingProxyType
from typing import Any

from fluxway.common.compiled_graph import (
    CompiledDependency,
    CompiledOperator,
    CompiledTensorTransfer,
    SPMDGraphView,
)
from fluxway.common.coherence import CoherenceAction
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_policy import ArgumentType
from fluxway.common.types.sharding import (
    PhysicalLayout,
    ReferentialRouting,
    ShardingSpec,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.transport.communication import (
    AsyncDataDistributor,
    TransportFrameTemplateCache,
)
from fluxway.transport.routing import (
    SelectedRoutingSource,
    TransportRoutingMode,
    select_routing_source,
)
from fluxway.transport.kernels import FrameTableCapacity
from fluxway.transport.manifest import (
    DependencyDecision,
    FrameFlag,
    TransportFrameBuilder,
    TransportFrameHeader,
)
from fluxway.profiler.recorder import TraceRecorder

logger = logging.getLogger(__name__)


FrameArg = tuple[int, TransportFrameHeader, tuple[Any, ...]]
StepFrameGroup = tuple[int, FrameArg]
type DependencyDecisions = Mapping[int, DependencyDecision] | None


def build_coherence_frame(
    *,
    step: int,
    sender_spmd_code: int,
    sender_rank: int,
    sender_world_size: int,
    epoch: int,
    action: CoherenceAction,
    route_codes: tuple[int, ...],
) -> TransportFrameHeader:
    if not route_codes:
        raise ValueError("coherence frame requires at least one route code.")
    builder = TransportFrameBuilder(
        step=step,
        sender_spmd_code=sender_spmd_code,
        sender_rank=sender_rank,
        sender_world_size=sender_world_size,
        flags=FrameFlag.COHERENCE,
        capacity=FrameTableCapacity(entry_count=len(route_codes)),
    )
    for route_code in route_codes:
        builder.add_coherence_entry(
            route_code=route_code,
            coherence_epoch=epoch,
            coherence_action=action,
        )
    return builder.build().header


class DependencyDecisionEvaluationError(TypeError):
    pass


@dataclass(frozen=True, slots=True)
class SuccessorRouteContext:
    context_index: int
    dependencies: tuple[CompiledDependency, ...]
    effective_rank: int

    def __post_init__(self) -> None:
        if self.context_index < 0:
            raise ValueError("Successor route context_index must be non-negative.")
        if self.effective_rank < 0:
            raise ValueError("Successor route effective_rank must be non-negative.")
        if not self.dependencies:
            raise ValueError("Successor route context requires dependencies.")
        dependency_codes = {dependency.code for dependency in self.dependencies}
        if len(dependency_codes) != len(self.dependencies):
            raise ValueError("Successor route context has duplicate dependencies.")
        target_spmd_ids = {
            dependency.spec.target.spmd_id for dependency in self.dependencies
        }
        if len(target_spmd_ids) != 1:
            raise ValueError(
                "Successor route context dependencies must share one target SPMD."
            )
        localities = {dependency.spec.local_only for dependency in self.dependencies}
        if len(localities) != 1:
            raise ValueError(
                "Successor route context dependencies must share one locality."
            )
        source_operators = {
            dependency.source_operator.key for dependency in self.dependencies
        }
        if len(source_operators) != 1:
            raise ValueError(
                "Successor route context dependencies must share one source operator."
            )
        target_world_sizes = {
            dependency.target_world_size for dependency in self.dependencies
        }
        if len(target_world_sizes) != 1:
            raise ValueError(
                "Successor route context dependencies disagree on target world size."
            )
        target_meshes = {dependency.target_mesh for dependency in self.dependencies}
        if len(target_meshes) != 1:
            raise ValueError(
                "Successor route context dependencies disagree on target mesh."
            )

    @property
    def spmd_id(self) -> str:
        return self.dependencies[0].spec.target.spmd_id

    @property
    def local_only(self) -> bool:
        return self.dependencies[0].spec.local_only

    @property
    def target_world_size(self) -> int:
        return self.dependencies[0].target_world_size

    @property
    def target_mesh(self) -> DeviceMesh:
        return self.dependencies[0].target_mesh


@dataclass(frozen=True, slots=True)
class CompiledReceiptRouting:
    input_transfer_code: int
    reference_path: str
    rank_count: int

    def __post_init__(self) -> None:
        if self.input_transfer_code <= 0:
            raise ValueError("Receipt input transfer code must be positive.")
        if not self.reference_path:
            raise ValueError("Receipt reference path must be non-empty.")
        if self.rank_count <= 0:
            raise ValueError("Receipt rank count must be positive.")


@dataclass(frozen=True, slots=True)
class CompiledTensorRoute:
    output_index: int
    transfer_code: int
    dependency_code: int
    logical_tuuid: int
    target_spmd_id: str
    target_op: str
    target_port: str
    source_spec: PhysicalLayout
    target_spec: PhysicalLayout
    receipt_routing: CompiledReceiptRouting | None
    tensor_name: str
    empty_targets: tuple[tuple[int, int], ...]
    child_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.child_slots) != self.target_spec.mesh.size:
            raise ValueError(
                "Compiled tensor route child_slots must cover the target rank domain."
            )
        if any(slot < 0 for slot in self.child_slots):
            raise ValueError("Compiled tensor route child slots must be non-negative.")
        if len(set(self.child_slots)) != len(self.child_slots):
            raise ValueError("Compiled tensor route child slots must be injective.")
        if (
            self.receipt_routing is not None
            and self.receipt_routing.rank_count != self.target_spec.mesh.size
        ):
            raise ValueError(
                "Compiled receipt rank domain does not match target mesh size: "
                f"receipt={self.receipt_routing.rank_count} "
                f"target={self.target_spec.mesh.size}."
            )

    @property
    def routing_mode(self) -> TransportRoutingMode:
        if self.receipt_routing is not None:
            return TransportRoutingMode.RECEIPT
        return TransportRoutingMode.ALGEBRAIC


@dataclass(frozen=True, slots=True)
class CompiledOrderingRoute:
    dependency_code: int
    target_op: str
    target_ranks: tuple[int, ...]
    expected_count: int


@dataclass(frozen=True, slots=True)
class CompiledDependencyDecisionRoute:
    dependency_code: int
    target_op: str
    target_ranks: tuple[int, ...]
    expected_count: int


@dataclass(frozen=True, slots=True)
class CompiledDependencyCondition:
    dependency_code: int
    condition: Callable[[Any], bool | Awaitable[bool]]

    def __post_init__(self) -> None:
        if self.dependency_code <= 0:
            raise ValueError("Dependency condition code must be positive.")
        if not callable(self.condition):
            raise TypeError("Dependency condition must be callable.")


@dataclass(frozen=True, slots=True)
class CompiledSuccessorRoute:
    context_index: int
    spmd_id: str
    effective_rank: int
    tensor_routes: tuple[CompiledTensorRoute, ...]
    ordering_routes: tuple[CompiledOrderingRoute, ...] = ()
    dependency_decision_routes: tuple[CompiledDependencyDecisionRoute, ...] = ()
    local_only: bool = False


@dataclass(frozen=True, slots=True)
class CompiledCoherenceChild:
    logical_tuuid: int
    child_slot: int
    target_spmd_id: str
    target_rank: int
    transfer_code: int


@dataclass(frozen=True, slots=True)
class CompiledSendPlan:
    op_name: str
    source_spmd_id: str
    rank: int
    sender_spmd_code: int
    sender_world_size: int
    successor_routes: tuple[CompiledSuccessorRoute, ...]
    output_count: int
    transfer_code_to_name: dict[int, str]
    dependency_conditions: tuple[CompiledDependencyCondition, ...] = ()
    frame_template_cache: TransportFrameTemplateCache = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _tensor_routes_by_code: Mapping[int, CompiledTensorRoute] = field(
        init=False,
        compare=False,
        repr=False,
    )
    _coherence_children_by_slot: Mapping[tuple[int, int], CompiledCoherenceChild] = (
        field(
            init=False,
            compare=False,
            repr=False,
        )
    )

    def __post_init__(self) -> None:
        tensor_routes = tuple(
            route
            for context in self.successor_routes
            for route in context.tensor_routes
        )
        routes_by_code = {route.transfer_code: route for route in tensor_routes}
        if len(routes_by_code) != len(tensor_routes):
            raise ValueError("Compiled send plan has duplicate tensor transfer codes.")
        endpoint_by_slot: dict[int, dict[int, tuple[str, int]]] = {}
        slot_by_endpoint: dict[int, dict[tuple[str, int], int]] = {}
        route_codes_by_endpoint: dict[tuple[int, str, int], list[int]] = {}
        for route in tensor_routes:
            for target_rank, child_slot in enumerate(route.child_slots):
                endpoint = (route.target_spmd_id, target_rank)
                logical_slots = endpoint_by_slot.setdefault(route.logical_tuuid, {})
                existing_endpoint = logical_slots.get(child_slot)
                if existing_endpoint is not None and existing_endpoint != endpoint:
                    raise ValueError(
                        "Compiled coherence child slot maps multiple endpoints: "
                        f"logical_tuuid={route.logical_tuuid} slot={child_slot}."
                    )
                logical_endpoints = slot_by_endpoint.setdefault(route.logical_tuuid, {})
                existing_slot = logical_endpoints.get(endpoint)
                if existing_slot is not None and existing_slot != child_slot:
                    raise ValueError(
                        "Compiled coherence endpoint maps to multiple child slots: "
                        f"logical_tuuid={route.logical_tuuid} endpoint={endpoint}."
                    )
                logical_slots[child_slot] = endpoint
                logical_endpoints[endpoint] = child_slot
                route_codes_by_endpoint.setdefault(
                    (route.logical_tuuid, route.target_spmd_id, target_rank),
                    [],
                ).append(route.transfer_code)
        object.__setattr__(
            self,
            "_tensor_routes_by_code",
            MappingProxyType(routes_by_code),
        )
        coherence_children = {
            (logical_tuuid, child_slot): CompiledCoherenceChild(
                logical_tuuid=logical_tuuid,
                child_slot=child_slot,
                target_spmd_id=target_spmd_id,
                target_rank=target_rank,
                transfer_code=min(
                    route_codes_by_endpoint[
                        (logical_tuuid, target_spmd_id, target_rank)
                    ]
                ),
            )
            for logical_tuuid, slots in endpoint_by_slot.items()
            for child_slot, (target_spmd_id, target_rank) in slots.items()
        }
        object.__setattr__(
            self,
            "_coherence_children_by_slot",
            MappingProxyType(coherence_children),
        )
        condition_codes = tuple(
            condition.dependency_code for condition in self.dependency_conditions
        )
        if len(set(condition_codes)) != len(condition_codes):
            raise ValueError("Compiled send plan has duplicate dependency conditions.")
        decision_route_codes = tuple(
            route.dependency_code
            for context in self.successor_routes
            for route in context.dependency_decision_routes
        )
        if len(set(decision_route_codes)) != len(decision_route_codes):
            raise ValueError(
                "Compiled send plan has duplicate dependency decision routes."
            )
        missing_route_codes = set(condition_codes).difference(decision_route_codes)
        if missing_route_codes:
            raise ValueError(
                "Compiled send plan has conditions without decision routes: "
                f"{sorted(missing_route_codes)}."
            )

    def tensor_route(self, transfer_code: int) -> CompiledTensorRoute:
        try:
            return self._tensor_routes_by_code[transfer_code]
        except KeyError as exc:
            raise ValueError(
                f"Compiled send plan has no tensor transfer code {transfer_code}."
            ) from exc

    def coherence_logical_tuuid(self, transfer_code: int) -> int:
        return self.tensor_route(transfer_code).logical_tuuid

    def coherence_child_slot(self, transfer_code: int, target_rank: int) -> int:
        route = self.tensor_route(transfer_code)
        if target_rank < 0 or target_rank >= len(route.child_slots):
            raise ValueError(
                f"target_rank={target_rank} is outside transfer "
                f"{transfer_code} rank domain {len(route.child_slots)}."
            )
        return route.child_slots[target_rank]

    def coherence_child(
        self, logical_tuuid: int, child_slot: int
    ) -> CompiledCoherenceChild:
        try:
            return self._coherence_children_by_slot[(logical_tuuid, child_slot)]
        except KeyError as exc:
            raise ValueError(
                "Compiled send plan has no coherence child for "
                f"logical_tuuid={logical_tuuid} child_slot={child_slot}."
            ) from exc

    def project_successors(
        self,
        context_indices: tuple[int, ...],
    ) -> CompiledSendPlan:
        if len(set(context_indices)) != len(context_indices):
            raise ValueError("Send plan projection requires unique context indices.")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self.successor_routes)
            for index in context_indices
        ):
            raise ValueError(
                "Send plan projection contains an invalid successor context index."
            )
        if context_indices == tuple(range(len(self.successor_routes))):
            return self
        successor_routes = tuple(
            replace(self.successor_routes[index], context_index=projected_index)
            for projected_index, index in enumerate(context_indices)
        )
        dependency_codes = {
            route.dependency_code
            for context in successor_routes
            for route in context.dependency_decision_routes
        }
        transfer_codes = {
            route.transfer_code
            for context in successor_routes
            for route in context.tensor_routes
        }
        return replace(
            self,
            successor_routes=successor_routes,
            transfer_code_to_name={
                code: name
                for code, name in self.transfer_code_to_name.items()
                if code in transfer_codes
            },
            dependency_conditions=tuple(
                condition
                for condition in self.dependency_conditions
                if condition.dependency_code in dependency_codes
            ),
        )


def compile_send_plan(
    *,
    graph_view: SPMDGraphView,
    source_op_id: str,
    rank: int,
    successor_contexts: tuple[SuccessorRouteContext, ...],
    sender_spmd_code: int,
    interleave_unit_size: int | None,
) -> CompiledSendPlan:
    source_operator = graph_view.operator(source_op_id)
    source_mesh = graph_view.mesh
    sender_world_size = graph_view.world_size
    if rank < 0 or rank >= sender_world_size:
        raise ValueError(
            f"Sender rank {rank} is outside world size {sender_world_size}."
        )
    if sender_world_size != source_mesh.size:
        raise ValueError(
            f"Sender world size {sender_world_size} does not match source mesh "
            f"size {source_mesh.size}."
        )
    if sender_spmd_code < 0:
        raise ValueError("sender_spmd_code must be non-negative.")
    coherence_child_slots = _compile_coherence_child_slots(successor_contexts)
    input_bindings = _index_input_bindings(
        source_operator=source_operator,
        incoming_dependencies=graph_view.incoming,
    )
    output_names = tuple(source_operator.get_arguments(True, ArgumentType.DATA))
    output_uuids = source_operator.output_uuids
    if len(output_names) != len(output_uuids):
        raise ValueError(
            f"Compiled operator {source_operator.key!r} has {len(output_names)} "
            f"data output names but {len(output_uuids)} logical TUUIDs."
        )
    output_index_by_name = {name: index for index, name in enumerate(output_names)}
    if len(output_index_by_name) != len(output_names):
        raise ValueError(
            f"Compiled operator {source_operator.key!r} has duplicate output names."
        )
    successor_routes: list[CompiledSuccessorRoute] = []
    transfer_code_to_name: dict[int, str] = {}
    dependency_codes: set[int] = set()
    dependency_conditions: list[CompiledDependencyCondition] = []
    for context in successor_contexts:
        tensor_routes: list[CompiledTensorRoute] = []
        ordering_routes: list[CompiledOrderingRoute] = []
        dependency_decision_routes: list[CompiledDependencyDecisionRoute] = []
        target_world_size = context.target_world_size
        if target_world_size <= 0:
            raise ValueError(
                f"[SPMDWorker] Invalid successor world_size={target_world_size} "
                f"for op={source_operator.key.op_id} successor={context.spmd_id}."
            )
        for dependency in context.dependencies:
            if dependency.code in dependency_codes:
                raise ValueError(
                    f"Duplicate dependency code {dependency.code} in sender plan "
                    f"for {source_operator.key!r}."
                )
            dependency_codes.add(dependency.code)
            if dependency.source_operator.key != source_operator.key:
                raise ValueError(
                    f"Dependency {dependency.key!r} does not originate from "
                    f"operator {source_operator.key.op_id!r}."
                )
            if dependency.source_world_size != sender_world_size:
                raise ValueError(
                    f"Dependency {dependency.key!r} source world size "
                    f"{dependency.source_world_size} does not match sender world size "
                    f"{sender_world_size}."
                )
            if dependency.source_mesh != source_mesh:
                raise ValueError(
                    f"Dependency {dependency.key!r} source mesh does not match "
                    "compile_send_plan source_mesh."
                )
            target_ranks = (
                (rank,) if context.local_only else tuple(range(target_world_size))
            )
            condition = dependency.spec.condition
            if condition is not None:
                dependency_conditions.append(
                    CompiledDependencyCondition(
                        dependency_code=dependency.code,
                        condition=condition,
                    )
                )
            dependency_decision_routes.append(
                CompiledDependencyDecisionRoute(
                    dependency_code=dependency.code,
                    target_op=dependency.spec.target.op_id,
                    target_ranks=target_ranks,
                    expected_count=(dependency.source_participating_count),
                )
            )
            if dependency.transfers:
                for transfer in dependency.transfers:
                    route = _compile_tensor_route(
                        dependency=dependency,
                        transfer=transfer,
                        graph_view=graph_view,
                        source_operator=source_operator,
                        source_mesh=source_mesh,
                        target_mesh=context.target_mesh,
                        output_uuids=output_uuids,
                        output_index_by_name=output_index_by_name,
                        input_bindings=input_bindings,
                        rank=rank,
                        effective_rank=context.effective_rank,
                        interleave_unit_size=interleave_unit_size,
                        coherence_child_slots=coherence_child_slots,
                    )
                    existing_name = transfer_code_to_name.get(route.transfer_code)
                    if existing_name is not None:
                        raise ValueError(
                            f"Duplicate transfer code {route.transfer_code} in sender "
                            f"plan for {source_operator.key!r}."
                        )
                    transfer_code_to_name[route.transfer_code] = route.tensor_name
                    tensor_routes.append(route)
                continue
            if dependency.spec.condition is None:
                ordering_routes.append(
                    CompiledOrderingRoute(
                        dependency_code=dependency.code,
                        target_op=dependency.spec.target.op_id,
                        target_ranks=target_ranks,
                        expected_count=(dependency.source_participating_count),
                    )
                )
        successor_routes.append(
            CompiledSuccessorRoute(
                context_index=context.context_index,
                spmd_id=context.spmd_id,
                effective_rank=context.effective_rank,
                tensor_routes=tuple(tensor_routes),
                ordering_routes=tuple(ordering_routes),
                dependency_decision_routes=tuple(dependency_decision_routes),
                local_only=context.local_only,
            )
        )
    return CompiledSendPlan(
        op_name=source_operator.key.op_id,
        source_spmd_id=source_operator.key.spmd_id,
        rank=rank,
        sender_spmd_code=sender_spmd_code,
        sender_world_size=sender_world_size,
        successor_routes=tuple(successor_routes),
        output_count=len(output_uuids),
        transfer_code_to_name=transfer_code_to_name,
        dependency_conditions=tuple(dependency_conditions),
    )


def _compile_coherence_child_slots(
    successor_contexts: tuple[SuccessorRouteContext, ...],
) -> dict[tuple[int, str, int], int]:
    endpoints_by_logical_tuuid: dict[int, set[tuple[str, int]]] = {}
    for context in successor_contexts:
        for dependency in context.dependencies:
            for transfer in dependency.transfers:
                endpoints = endpoints_by_logical_tuuid.setdefault(
                    transfer.logical_tuuid,
                    set(),
                )
                endpoints.update(
                    (context.spmd_id, target_rank)
                    for target_rank in range(context.target_world_size)
                )
    return {
        (logical_tuuid, target_spmd_id, target_rank): child_slot
        for logical_tuuid, endpoints in endpoints_by_logical_tuuid.items()
        for child_slot, (target_spmd_id, target_rank) in enumerate(sorted(endpoints))
    }


async def evaluate_dependency_decisions(
    compiled_plan: CompiledSendPlan,
    result: Any,
) -> DependencyDecisions:
    conditions = compiled_plan.dependency_conditions
    if not conditions:
        return None
    values: list[Any] = [None] * len(conditions)
    pending: list[tuple[int, asyncio.Task[tuple[bool, Any]]]] = []
    try:
        for index, compiled_condition in enumerate(conditions):
            value = compiled_condition.condition(result)
            if type(value) is bool:
                values[index] = value
            elif inspect.isawaitable(value):
                pending.append(
                    (
                        index,
                        asyncio.create_task(_capture_condition_result(value)),
                    )
                )
            else:
                _dependency_decision_from_value(
                    compiled_condition.dependency_code,
                    value,
                )
    except BaseException:
        for _, task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(
                *(task for _, task in pending),
                return_exceptions=True,
            )
        raise

    if pending:
        outcomes = await asyncio.gather(*(task for _, task in pending))
        for (index, _), (completed, value) in zip(pending, outcomes, strict=True):
            if not completed:
                raise value
            values[index] = value

    return {
        compiled_condition.dependency_code: _dependency_decision_from_value(
            compiled_condition.dependency_code,
            values[index],
        )
        for index, compiled_condition in enumerate(conditions)
    }


async def _capture_condition_result(value: Any) -> tuple[bool, Any]:
    try:
        return True, await value
    except BaseException as error:
        return False, error


def _dependency_decision_from_value(
    dependency_code: int,
    value: Any,
) -> DependencyDecision:
    if type(value) is not bool:
        raise DependencyDecisionEvaluationError(
            "Condition for dependency_code="
            f"{dependency_code} must return exact bool, got "
            f"{type(value).__name__}."
        )
    return DependencyDecision.PROCEED if value else DependencyDecision.SKIP


def skipped_dependency_decisions(
    compiled_plan: CompiledSendPlan,
) -> dict[int, DependencyDecision]:
    return {
        route.dependency_code: DependencyDecision.SKIP
        for context in compiled_plan.successor_routes
        for route in context.dependency_decision_routes
    }


type _InputBinding = tuple[CompiledDependency, CompiledTensorTransfer]


def _index_input_bindings(
    *,
    source_operator: CompiledOperator,
    incoming_dependencies: tuple[CompiledDependency, ...],
) -> dict[str, _InputBinding]:
    allowed_paths = {
        f"{source_operator.key.op_id}.{name}"
        for name in source_operator.get_arguments(False, ArgumentType.DATA)
    }
    bindings: dict[str, _InputBinding] = {}
    codes: set[int] = set()
    for dependency in incoming_dependencies:
        if dependency.target_operator.key != source_operator.key:
            continue
        for transfer in dependency.transfers:
            path = f"{source_operator.key.op_id}.{transfer.spec.target.name}"
            if path not in allowed_paths:
                raise ValueError(
                    f"Input transfer path {path!r} is not an exact input path of "
                    f"operator {source_operator.key.op_id!r}; expected one of "
                    f"{sorted(allowed_paths)!r}."
                )
            if path in bindings:
                raise ValueError(f"Duplicate input transfer path {path!r}.")
            if transfer.transfer_code in codes:
                raise ValueError(
                    f"Duplicate input transfer code {transfer.transfer_code}."
                )
            bindings[path] = (dependency, transfer)
            codes.add(transfer.transfer_code)
    return bindings


def _compile_tensor_route(
    *,
    dependency: CompiledDependency,
    transfer: CompiledTensorTransfer,
    graph_view: SPMDGraphView,
    source_operator: CompiledOperator,
    source_mesh: DeviceMesh,
    target_mesh: DeviceMesh,
    output_uuids: tuple[int, ...],
    output_index_by_name: Mapping[str, int],
    input_bindings: Mapping[str, _InputBinding],
    rank: int,
    effective_rank: int,
    interleave_unit_size: int | None,
    coherence_child_slots: Mapping[tuple[int, str, int], int],
) -> CompiledTensorRoute:
    source_port = transfer.spec.source.name
    try:
        output_index = output_index_by_name[source_port]
    except KeyError as exc:
        raise ValueError(
            f"Transfer code {transfer.transfer_code} references unknown source output "
            f"{source_port!r} on operator {source_operator.key!r}."
        ) from exc
    if output_uuids[output_index] != transfer.logical_tuuid:
        raise ValueError(
            f"Transfer code {transfer.transfer_code} logical TUUID "
            f"{transfer.logical_tuuid} does not match output {source_port!r} TUUID "
            f"{output_uuids[output_index]}."
        )
    source_spec = _bind_source_spec(
        source_logical=graph_view.logical_tensor(
            transfer.logical_tuuid
        ).require_effective_sharding(),
        mesh=source_mesh,
        rank=rank,
        interleave_unit_size=interleave_unit_size,
    )
    target_spec = dependency.target_operator.effective_input_sharding[
        transfer.spec.target.name
    ].bind(target_mesh)
    empty_targets = tuple(
        (
            target_rank,
            len(source_spec.resolve_producers(target_spec, target_rank)),
        )
        for target_rank in source_spec.resolve_consumers(target_spec, effective_rank)
    )
    for target_rank, expected_count in empty_targets:
        if expected_count <= 0:
            raise ValueError(
                f"[SPMDWorker] Compiled invalid expected_count={expected_count} "
                f"target_rank={target_rank} op={source_operator.key.op_id} "
                f"successor={dependency.spec.target.spmd_id} "
                f"transfer_code={transfer.transfer_code}."
            )
    receipt_routing = _compile_receipt_routing(
        transfer=transfer,
        input_bindings=input_bindings,
        target_rank_count=dependency.target_world_size,
    )
    target_op = dependency.spec.target.op_id
    target_port = transfer.spec.target.name
    tensor_name = (
        f"{source_operator.key.spmd_id}.{source_operator.key.op_id}.{source_port}"
        f"->{dependency.spec.target.spmd_id}.{target_op}.{target_port}"
    )
    return CompiledTensorRoute(
        output_index=output_index,
        transfer_code=transfer.transfer_code,
        dependency_code=transfer.dependency_code,
        logical_tuuid=transfer.logical_tuuid,
        target_spmd_id=dependency.spec.target.spmd_id,
        target_op=target_op,
        target_port=target_port,
        source_spec=source_spec,
        target_spec=target_spec,
        receipt_routing=receipt_routing,
        tensor_name=tensor_name,
        empty_targets=empty_targets,
        child_slots=tuple(
            coherence_child_slots[
                (
                    transfer.logical_tuuid,
                    dependency.spec.target.spmd_id,
                    target_rank,
                )
            ]
            for target_rank in range(dependency.target_world_size)
        ),
    )


def _compile_receipt_routing(
    *,
    transfer: CompiledTensorTransfer,
    input_bindings: Mapping[str, _InputBinding],
    target_rank_count: int,
) -> CompiledReceiptRouting | None:
    routing = transfer.source_sharding.routing
    if not transfer.uses_receipt_routing:
        return None
    assert isinstance(routing, ReferentialRouting)
    if routing.tuuid is None:
        raise ValueError(
            f"Transfer code {transfer.transfer_code} has unresolved referential "
            f"routing path {routing.path!r}."
        )
    try:
        _, input_transfer = input_bindings[routing.path]
    except KeyError as exc:
        raise ValueError(
            f"Transfer code {transfer.transfer_code} referential routing path "
            f"{routing.path!r} has no exact input binding."
        ) from exc
    return CompiledReceiptRouting(
        input_transfer_code=input_transfer.transfer_code,
        reference_path=routing.path,
        rank_count=target_rank_count,
    )


def _bind_source_spec(
    *,
    source_logical: ShardingSpec,
    mesh: Any,
    rank: int,
    interleave_unit_size: int | None,
) -> PhysicalLayout:
    source_spec = source_logical.bind(mesh)
    if interleave_unit_size is None:
        return source_spec
    if isinstance(interleave_unit_size, bool) or not isinstance(
        interleave_unit_size, int
    ):
        raise TypeError("interleave_unit_size must be an integer or None.")
    if interleave_unit_size <= 0:
        raise ValueError("interleave_unit_size must be positive.")
    if interleave_unit_size > source_spec.mesh.size:
        raise ValueError(
            f"interleave_unit_size {interleave_unit_size} exceeds source mesh size "
            f"{source_spec.mesh.size}."
        )
    unit_idx = rank // interleave_unit_size
    unit_ranks = list(
        range(
            unit_idx * interleave_unit_size,
            (unit_idx + 1) * interleave_unit_size,
        )
    )
    base_mesh = source_spec.mesh
    other_dims_size = 1
    for size in base_mesh.shape[1:]:
        other_dims_size *= size
    if interleave_unit_size % other_dims_size != 0:
        raise ValueError(
            f"interleave_unit_size {interleave_unit_size} is not divisible by "
            f"the non-leading mesh size {other_dims_size}."
        )
    if unit_ranks[-1] >= base_mesh.size:
        raise ValueError(
            f"Rank {rank} selects an interleave unit outside source mesh size "
            f"{base_mesh.size}."
        )
    first_axis_size = interleave_unit_size // other_dims_size
    return source_logical.bind(
        DeviceMesh(
            device_shape=(first_axis_size,) + base_mesh.shape[1:],
            axis_names=base_mesh.axis_names,
            device_ids=unit_ranks,
        )
    )


def _validate_condition_decisions(
    compiled_plan: CompiledSendPlan,
    dependency_decisions: DependencyDecisions,
) -> None:
    conditions = compiled_plan.dependency_conditions
    if dependency_decisions is None:
        if not conditions:
            return
        raise ValueError(
            f"Operator {compiled_plan.op_name!r} requires dependency decisions."
        )
    if not conditions:
        raise ValueError(
            f"Operator {compiled_plan.op_name!r} has no condition decisions; use None."
        )
    if len(dependency_decisions) != len(conditions) or any(
        condition.dependency_code not in dependency_decisions
        for condition in conditions
    ):
        raise ValueError(
            f"Operator {compiled_plan.op_name!r} requires condition decision codes "
            f"{[condition.dependency_code for condition in conditions]}, got "
            f"{sorted(dependency_decisions)}."
        )


def build_step_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    flat_results: list[Any],
    step_transfer_cache: Mapping[int, Any],
    exec_step: int,
    dependency_decisions: DependencyDecisions = None,
    profiler_recorder: TraceRecorder | None = None,
) -> list[StepFrameGroup]:
    _validate_condition_decisions(compiled_plan, dependency_decisions)
    frame_groups: list[StepFrameGroup] = []
    wrapped_results: dict[int, TransportTensor] = {}

    for context in compiled_plan.successor_routes:
        if not context.local_only:
            frame_groups.extend(
                _build_causal_frame_groups(
                    compiled_plan=compiled_plan,
                    context=context,
                    exec_step=exec_step,
                    dependency_decisions=dependency_decisions,
                )
            )
        batches = build_successor_batches(
            compiled_plan=compiled_plan,
            flat_results=flat_results,
            context=context,
            step_transfer_cache=step_transfer_cache,
            exec_step=exec_step,
            wrapped_results=wrapped_results,
            dependency_decisions=dependency_decisions,
        )
        if not batches:
            continue

        distributor = AsyncDataDistributor(context.effective_rank)

        def plan_distribution(
            distributor: AsyncDataDistributor,
            batches: dict[
                Any,
                list[tuple[int, TransportTensor, SelectedRoutingSource]],
            ],
        ) -> None:
            for (source_spec, target_spec, _), batch in batches.items():
                batch_transfer_codes = [item[0] for item in batch]
                batch_transport_tensors = [item[1] for item in batch]
                routing_source = batch[0][2]

                if logger.isEnabledFor(logging.DEBUG):
                    for transfer_code, leaf_val, _ in batch:
                        name = compiled_plan.transfer_code_to_name[transfer_code]
                        logger.debug(
                            "Distributor plan name=%s transfer_code=%s global_shape=%s "
                            "origin_indices=%s routing_indices=%s rank=%s",
                            name,
                            transfer_code,
                            leaf_val.global_shape,
                            leaf_val.origin_indices,
                            leaf_val.routing_indices,
                            compiled_plan.rank,
                        )

                distributor.plan_and_shard(
                    batch_transport_tensors,
                    source_spec,
                    target_spec,
                    routing_source=routing_source,
                    transfer_codes=batch_transfer_codes,
                    transfer_code_to_name=compiled_plan.transfer_code_to_name,
                )

        if profiler_recorder is None:
            plan_distribution(distributor, batches)
        else:
            with profiler_recorder.transport_distribute_sync(
                op_name=compiled_plan.op_name,
                rank=compiled_plan.rank,
                step=exec_step,
                target_spmd_id=context.spmd_id,
            ):
                plan_distribution(distributor, batches)

        for frame_arg in distributor.build_transport_frame_args(
            step=exec_step,
            sender_spmd_code=compiled_plan.sender_spmd_code,
            sender_world_size=compiled_plan.sender_world_size,
            sender_rank=compiled_plan.rank,
            template_cache=compiled_plan.frame_template_cache,
        ):
            if profiler_recorder is not None:
                target_rank, _header, payload_segments = frame_arg
                profiler_recorder.data_plane_tensor_split(
                    op_name=compiled_plan.op_name,
                    rank=compiled_plan.rank,
                    step=exec_step,
                    target_rank=target_rank,
                    payload_count=len(payload_segments),
                )
            frame_groups.append((context.context_index, frame_arg))

    return frame_groups


def build_completed_without_tensor_step_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    empty_values: Any,
    exec_step: int,
    dependency_decisions: DependencyDecisions = None,
) -> list[StepFrameGroup]:
    _validate_condition_decisions(compiled_plan, dependency_decisions)
    frame_groups: list[StepFrameGroup] = []
    for context in compiled_plan.successor_routes:
        if not context.local_only:
            frame_groups.extend(
                _build_causal_frame_groups(
                    compiled_plan=compiled_plan,
                    context=context,
                    exec_step=exec_step,
                    dependency_decisions=dependency_decisions,
                )
            )
        distributor = AsyncDataDistributor(context.effective_rank)
        for route in context.tensor_routes:
            if (
                dependency_decisions is not None
                and dependency_decisions.get(route.dependency_code)
                is DependencyDecision.SKIP
            ):
                continue
            for target_rank, expected_count in route.empty_targets:
                distributor.plan_completed_without_tensor(
                    target_rank=target_rank,
                    expected_count=expected_count,
                    transfer_code=route.transfer_code,
                    payload=empty_values,
                    source_step=exec_step,
                )
        for frame_arg in distributor.build_transport_frame_args(
            step=exec_step,
            sender_spmd_code=compiled_plan.sender_spmd_code,
            sender_world_size=compiled_plan.sender_world_size,
            sender_rank=compiled_plan.rank,
            template_cache=compiled_plan.frame_template_cache,
        ):
            frame_groups.append((context.context_index, frame_arg))
    return frame_groups


def build_dependency_decision_only_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    dependency_decisions: Mapping[int, DependencyDecision],
    exec_step: int,
) -> list[StepFrameGroup]:
    if not dependency_decisions:
        raise ValueError("Dependency-decision-only frames require decisions.")
    route_count = sum(
        len(context.dependency_decision_routes)
        for context in compiled_plan.successor_routes
    )
    if len(dependency_decisions) != route_count or any(
        route.dependency_code not in dependency_decisions
        for context in compiled_plan.successor_routes
        for route in context.dependency_decision_routes
    ):
        raise ValueError(
            f"Operator {compiled_plan.op_name!r} decision-only frames require all "
            "dependency decision routes."
        )
    frame_groups: list[StepFrameGroup] = []
    for context in compiled_plan.successor_routes:
        if context.local_only:
            continue
        frame_groups.extend(
            _build_causal_frame_groups(
                compiled_plan=compiled_plan,
                context=context,
                exec_step=exec_step,
                dependency_decisions=dependency_decisions,
            )
        )
    return frame_groups


def _build_causal_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    context: CompiledSuccessorRoute,
    exec_step: int,
    dependency_decisions: DependencyDecisions,
) -> list[StepFrameGroup]:
    return [
        *_build_ordering_frame_groups(
            compiled_plan=compiled_plan,
            context=context,
            exec_step=exec_step,
            dependency_decisions=dependency_decisions,
        ),
        *_build_dependency_decision_frame_groups(
            compiled_plan=compiled_plan,
            context=context,
            exec_step=exec_step,
            dependency_decisions=dependency_decisions,
        ),
    ]


def _build_dependency_decision_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    context: CompiledSuccessorRoute,
    exec_step: int,
    dependency_decisions: DependencyDecisions,
) -> list[StepFrameGroup]:
    if not dependency_decisions:
        return []
    routes_by_target: dict[int, list[CompiledDependencyDecisionRoute]] = defaultdict(
        list
    )
    for route in context.dependency_decision_routes:
        if route.dependency_code not in dependency_decisions:
            continue
        for target_rank in route.target_ranks:
            routes_by_target[target_rank].append(route)

    frame_groups: list[StepFrameGroup] = []
    for target_rank in sorted(routes_by_target):
        routes = routes_by_target[target_rank]
        builder = TransportFrameBuilder(
            step=exec_step,
            sender_spmd_code=compiled_plan.sender_spmd_code,
            sender_rank=compiled_plan.rank,
            sender_world_size=compiled_plan.sender_world_size,
            flags=FrameFlag.DEPENDENCY_DECISION,
            capacity=FrameTableCapacity(entry_count=len(routes)),
        )
        for route in routes:
            builder.add_dependency_decision_entry(
                dependency_code=route.dependency_code,
                decision=dependency_decisions[route.dependency_code],
                expected_count=route.expected_count,
            )
        frame = builder.build()
        frame_groups.append(
            (
                context.context_index,
                (target_rank, frame.header, frame.payload_segments),
            )
        )
    return frame_groups


def _build_ordering_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    context: CompiledSuccessorRoute,
    exec_step: int,
    dependency_decisions: DependencyDecisions,
) -> list[StepFrameGroup]:
    routes_by_target: dict[int, list[CompiledOrderingRoute]] = defaultdict(list)
    for route in context.ordering_routes:
        if (
            dependency_decisions is not None
            and route.dependency_code in dependency_decisions
        ):
            continue
        for target_rank in route.target_ranks:
            routes_by_target[target_rank].append(route)

    frame_groups: list[StepFrameGroup] = []
    for target_rank in sorted(routes_by_target):
        routes = routes_by_target[target_rank]
        builder = TransportFrameBuilder(
            step=exec_step,
            sender_spmd_code=compiled_plan.sender_spmd_code,
            sender_rank=compiled_plan.rank,
            sender_world_size=compiled_plan.sender_world_size,
            flags=FrameFlag.ORDERING_COMPLETION,
            capacity=FrameTableCapacity(entry_count=len(routes)),
        )
        for route in routes:
            builder.add_ordering_completion_entry(
                dependency_code=route.dependency_code,
                expected_count=route.expected_count,
            )
        frame = builder.build()
        frame_groups.append(
            (
                context.context_index,
                (target_rank, frame.header, frame.payload_segments),
            )
        )
    return frame_groups


def build_successor_batches(
    *,
    compiled_plan: CompiledSendPlan,
    flat_results: list[Any],
    context: CompiledSuccessorRoute,
    step_transfer_cache: Mapping[int, Any],
    exec_step: int,
    wrapped_results: dict[int, TransportTensor],
    dependency_decisions: DependencyDecisions,
) -> dict[Any, list[tuple[int, TransportTensor, SelectedRoutingSource]]]:
    batches: dict[
        Any,
        list[tuple[int, TransportTensor, SelectedRoutingSource]],
    ] = defaultdict(list)
    for route in context.tensor_routes:
        if (
            dependency_decisions is not None
            and dependency_decisions.get(route.dependency_code)
            is DependencyDecision.SKIP
        ):
            continue
        if route.output_index >= len(flat_results):
            raise ValueError(
                f"Operator {compiled_plan.op_name!r} produced {len(flat_results)} "
                f"flat outputs, but transfer code {route.transfer_code} requires "
                f"output index {route.output_index}."
            )
        raw_leaf = flat_results[route.output_index]
        leaf_tensor = prepare_route_tensor(
            compiled_plan=compiled_plan,
            route=route,
            raw_leaf=raw_leaf,
            step_transfer_cache=step_transfer_cache,
            exec_step=exec_step,
            wrapped_results=wrapped_results,
        )

        routing_source = select_routing_source(
            tensor=leaf_tensor,
            source=route.source_spec,
            target=route.target_spec,
            mode=route.routing_mode,
            expected_receipt_world_size=route.target_spec.mesh.size,
        )
        batches[
            (
                route.source_spec,
                route.target_spec,
                routing_source.batch_key,
            )
        ].append((route.transfer_code, leaf_tensor, routing_source))
    return batches


def prepare_route_tensor(
    *,
    compiled_plan: CompiledSendPlan,
    route: CompiledTensorRoute,
    raw_leaf: Any,
    step_transfer_cache: Mapping[int, Any],
    exec_step: int,
    wrapped_results: dict[int, TransportTensor],
) -> TransportTensor:
    if raw_leaf is None:
        raise ValueError(
            f"Declared transfer code {route.transfer_code} from operator "
            f"{compiled_plan.op_name!r} produced a None leaf at output index "
            f"{route.output_index}. Whole-step empty propagation must use "
            "build_completed_without_tensor_step_frame_groups()."
        )
    leaf_tensor = wrapped_results.get(route.logical_tuuid)
    if leaf_tensor is None:
        if isinstance(raw_leaf, TransportTensor):
            leaf_tensor = raw_leaf
        else:
            leaf_tensor = TransportTensor.from_tensor(raw_leaf)
        ensure_transport_source_step(
            op_name=compiled_plan.op_name,
            tensor_name=route.tensor_name,
            tensor=leaf_tensor,
            exec_step=exec_step,
        )
        if route.receipt_routing is not None:
            leaf_tensor = inherit_mirror_metadata(
                op_name=compiled_plan.op_name,
                mirror_transfer_code=route.receipt_routing.input_transfer_code,
                leaf_tensor=leaf_tensor,
                mirror_reference_path=route.receipt_routing.reference_path,
                step_transfer_cache=step_transfer_cache,
                exec_step=exec_step,
                spmd_id=compiled_plan.source_spmd_id,
                rank=compiled_plan.rank,
                tensor_name=route.tensor_name,
            )
        wrapped_results[route.logical_tuuid] = leaf_tensor
    else:
        ensure_transport_source_step(
            op_name=compiled_plan.op_name,
            tensor_name=route.tensor_name,
            tensor=leaf_tensor,
            exec_step=exec_step,
        )
    if route.target_spec.is_content_addressable and (
        leaf_tensor.routing_indices is None
        and leaf_tensor.receipt is None
        and leaf_tensor.flat_values is None
    ):
        raise ValueError(
            f"[Fluxway Routing Error] Operator '{compiled_plan.op_name}' "
            f"output '{route.tensor_name}' for transfer code "
            f"{route.transfer_code} is dynamically routed, but its "
            "TransportTensor lacks routing_indices, receipt, and flat_values."
        )
    return leaf_tensor


def ensure_transport_source_step(
    *,
    op_name: str,
    tensor_name: str,
    tensor: TransportTensor,
    exec_step: int,
) -> None:
    source_step = tensor.source_step
    if source_step is None:
        tensor.source_step = exec_step
        return
    if source_step != exec_step:
        raise ValueError(
            f"[Fluxway SourceStep Error] Operator '{op_name}' output "
            f"'{tensor_name}' has source_step={source_step}, but exec_step={exec_step}. "
            "Operator outputs must belong to the step that produced them."
        )


def inherit_mirror_metadata(
    *,
    op_name: str,
    mirror_transfer_code: int,
    leaf_tensor: TransportTensor,
    mirror_reference_path: str,
    step_transfer_cache: Mapping[int, Any],
    exec_step: int,
    spmd_id: str,
    rank: int,
    tensor_name: str,
) -> TransportTensor:
    source_ref = step_transfer_cache.get(mirror_transfer_code)
    if source_ref is None:
        raise ValueError(
            f"[Fluxway Mirror Error] Mirror resolution failed for output '{tensor_name}' "
            f"in operator '{op_name}'. It is configured to follow the sharding of "
            f"'{mirror_reference_path}' (transfer code {mirror_transfer_code}), but this mirror "
            "source was not "
            f"found in the SPMD {spmd_id} rank {rank} step {exec_step} input cache. "
            f"Verify that '{mirror_reference_path}' is properly declared as an input to '{op_name}' "
            "in the graph configuration."
        )
    if source_ref.receipt is None:
        raise ValueError(
            f"[Fluxway Mirror Error] Referential output '{tensor_name}' in operator "
            f"'{op_name}' requires a SymmetryReceipt from input transfer code "
            f"{mirror_transfer_code}, but the cached input has no receipt."
        )

    leaf_receipt = leaf_tensor.receipt
    if leaf_receipt is None:
        if leaf_tensor.origin_indices is not None:
            raise ValueError(
                f"[Fluxway Mirror Error] Referential output '{tensor_name}' in "
                f"operator '{op_name}' carries origin_indices without a receipt. "
                "Transform referential outputs through TransportTensor operations "
                "so receipt and origin lineage remain aligned."
            )
        if leaf_tensor.routing_entity_count != source_ref.routing_entity_count:
            raise ValueError(
                f"[Fluxway Mirror Error] Referential output '{tensor_name}' in "
                f"operator '{op_name}' has {leaf_tensor.routing_entity_count} routing "
                f"entities, but reference '{mirror_reference_path}' has "
                f"{source_ref.routing_entity_count}. Return a TransportTensor carrying "
                "the transformed receipt and origin lineage."
            )
        inherited_receipt = source_ref.receipt
        inherited_origin_indices = source_ref.origin_indices
    else:
        if leaf_receipt.world_size != source_ref.receipt.world_size:
            raise ValueError(
                f"[Fluxway Mirror Error] Referential output '{tensor_name}' receipt "
                f"world size {leaf_receipt.world_size} does not match reference "
                f"world size {source_ref.receipt.world_size}."
            )
        if source_ref.origin_indices is not None and leaf_tensor.origin_indices is None:
            raise ValueError(
                f"[Fluxway Mirror Error] Referential output '{tensor_name}' preserves "
                "a receipt but drops the reference origin lineage."
            )
        inherited_receipt = leaf_receipt
        inherited_origin_indices = leaf_tensor.origin_indices

    inherited_global_shape = leaf_tensor.global_shape
    if inherited_global_shape is None and source_ref.global_shape is not None:
        actual_shape = leaf_tensor.shape
        src_axes = source_ref.routing_axes
        dst_axes = leaf_tensor.routing_axes
        if len(src_axes) != len(dst_axes):
            raise ValueError(
                f"[Fluxway Mirror Error] Mirror shape alignment mismatch in operator "
                f"'{op_name}'. Source routing axes {src_axes} (length {len(src_axes)}) "
                f"does not match destination routing axes {dst_axes}."
            )
        new_shape = list(actual_shape)
        for idx, out_axis in enumerate(dst_axes):
            if out_axis >= len(new_shape):
                raise ValueError(
                    f"[Fluxway Mirror Error] Destination routing axis {out_axis} "
                    f"is out of bounds for shape {actual_shape} in operator '{op_name}'."
                )
            src_axis = src_axes[idx]
            if src_axis >= len(source_ref.global_shape):
                raise ValueError(
                    f"[Fluxway Mirror Error] Source routing axis {src_axis} is out "
                    f"of bounds for global shape {source_ref.global_shape} in operator "
                    f"'{op_name}'."
                )
            new_shape[out_axis] = source_ref.global_shape[src_axis]
        inherited_global_shape = tuple(new_shape)

    return leaf_tensor.with_lineage(
        origin_indices=inherited_origin_indices,
        receipt=inherited_receipt,
        global_shape=inherited_global_shape,
    )
