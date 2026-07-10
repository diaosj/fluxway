"""Pure sender-side data-plane frame construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import Any

from fluxway.common.mesh import DeviceMesh
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
from fluxway.transport.manifest import TransportFrameHeader
from fluxway.profiler.recorder import TraceRecorder

logger = logging.getLogger(__name__)


FrameArg = tuple[int, TransportFrameHeader, tuple[Any, ...]]
StepFrameGroup = tuple[int, FrameArg]


@dataclass(frozen=True, slots=True)
class SuccessorRouteContext:
    context_index: int
    spmd_id: str
    info: Any
    effective_rank: int
    local_only: bool = False


@dataclass(frozen=True, slots=True)
class CompiledTensorRoute:
    output_index: int
    tuuid: int
    source_spec: Any
    target_spec: Any
    tensor_name: str
    empty_targets: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CompiledSuccessorRoute:
    context_index: int
    spmd_id: str
    effective_rank: int
    tensor_routes: tuple[CompiledTensorRoute, ...]
    local_only: bool = False


@dataclass(frozen=True, slots=True)
class CompiledSendPlan:
    op_name: str
    rank: int
    sender_spmd_code: int
    sender_world_size: int
    successor_routes: tuple[CompiledSuccessorRoute, ...]
    output_count: int
    tuuid_to_name: dict[int, str] | None
    frame_template_cache: TransportFrameTemplateCache = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


def compile_send_plan(
    *,
    op_name: str,
    rank: int,
    mesh: Any,
    output_uuids: tuple[int, ...],
    output_sharding: dict[int, ShardingSpec],
    successor_contexts: tuple[SuccessorRouteContext, ...],
    tuuid_to_name: dict[int, str] | None,
    sender_spmd_code: int,
    sender_world_size: int,
    interleave_unit_size: int | None,
) -> CompiledSendPlan:
    successor_routes: list[CompiledSuccessorRoute] = []
    for context in successor_contexts:
        tensor_routes: list[CompiledTensorRoute] = []
        succ_info = context.info
        for output_index, tuuid in enumerate(output_uuids):
            if tuuid not in succ_info.tuuid_sharding:
                continue
            source_logical = output_sharding.get(tuuid)
            target_logical = succ_info.tuuid_sharding.get(tuuid)
            if source_logical is None or target_logical is None:
                raise ValueError(
                    f"[SPMDWorker] Missing sharding for output tuuid={tuuid} "
                    f"op={op_name} successor={context.spmd_id} "
                    f"source_has={source_logical is not None} "
                    f"target_has={target_logical is not None}."
                )
            source_spec = _bind_source_spec(
                source_logical=source_logical,
                mesh=mesh,
                rank=rank,
                interleave_unit_size=interleave_unit_size,
            )
            target_spec = target_logical.bind(succ_info.mesh or mesh)
            empty_targets = tuple(
                (
                    target_rank,
                    len(source_spec.resolve_producers(target_spec, target_rank)),
                )
                for target_rank in source_spec.resolve_consumers(
                    target_spec,
                    context.effective_rank,
                )
            )
            for target_rank, expected_count in empty_targets:
                if expected_count <= 0:
                    raise ValueError(
                        f"[SPMDWorker] Compiled invalid expected_count={expected_count} "
                        f"target_rank={target_rank} op={op_name} successor={context.spmd_id} "
                        f"tuuid={tuuid}."
                    )
            tensor_routes.append(
                CompiledTensorRoute(
                    output_index=output_index,
                    tuuid=tuuid,
                    source_spec=source_spec,
                    target_spec=target_spec,
                    tensor_name=(
                        tuuid_to_name.get(tuuid, f"TUUID:{tuuid}")
                        if tuuid_to_name
                        else f"TUUID:{tuuid}"
                    ),
                    empty_targets=empty_targets,
                )
            )
        successor_routes.append(
            CompiledSuccessorRoute(
                context_index=context.context_index,
                spmd_id=context.spmd_id,
                effective_rank=context.effective_rank,
                tensor_routes=tuple(tensor_routes),
                local_only=context.local_only,
            )
        )
    return CompiledSendPlan(
        op_name=op_name,
        rank=rank,
        sender_spmd_code=sender_spmd_code,
        sender_world_size=sender_world_size,
        successor_routes=tuple(successor_routes),
        output_count=len(output_uuids),
        tuuid_to_name=tuuid_to_name,
    )


def _bind_source_spec(
    *,
    source_logical: ShardingSpec,
    mesh: Any,
    rank: int,
    interleave_unit_size: int | None,
) -> Any:
    source_spec = source_logical.bind(mesh)
    if not isinstance(interleave_unit_size, int):
        return source_spec
    unit_idx = rank // interleave_unit_size
    unit_ranks = list(
        range(
            unit_idx * interleave_unit_size,
            (unit_idx + 1) * interleave_unit_size,
        )
    )
    if source_spec.mesh is None:
        return source_spec
    base_mesh = source_spec.mesh
    other_dims_size = 1
    for size in base_mesh.shape[1:]:
        other_dims_size *= size
    if interleave_unit_size % other_dims_size != 0:
        return source_spec
    first_axis_size = interleave_unit_size // other_dims_size
    return source_logical.bind(
        DeviceMesh(
            device_shape=(first_axis_size,) + base_mesh.shape[1:],
            axis_names=base_mesh.axis_names,
            device_ids=unit_ranks,
        )
    )


def build_step_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    flat_results: list[Any],
    step_tuuid_cache: dict[int, Any],
    exec_step: int,
    profiler_recorder: TraceRecorder | None = None,
) -> list[StepFrameGroup]:
    frame_groups: list[StepFrameGroup] = []
    wrapped_results: dict[int, TransportTensor] = {}

    for context in compiled_plan.successor_routes:
        batches = build_successor_batches(
            compiled_plan=compiled_plan,
            flat_results=flat_results,
            context=context,
            step_tuuid_cache=step_tuuid_cache,
            exec_step=exec_step,
            wrapped_results=wrapped_results,
        )
        if not batches:
            continue

        distributor = AsyncDataDistributor(context.effective_rank)

        def plan_distribution(
            distributor: AsyncDataDistributor,
            batches: dict[Any, list[tuple[int, TransportTensor]]],
        ) -> None:
            for (source_spec, target_spec, _), batch in batches.items():
                batch_target_uuids = [item[0] for item in batch]
                batch_transport_tensors = [item[1] for item in batch]

                if logger.isEnabledFor(logging.DEBUG):
                    for leaf_tuuid, leaf_val in batch:
                        name = (
                            compiled_plan.tuuid_to_name.get(leaf_tuuid, "unknown")
                            if compiled_plan.tuuid_to_name
                            else "unknown"
                        )
                        logger.debug(
                            "Distributor plan name=%s tuuid=%s global_shape=%s "
                            "origin_indices=%s routing_indices=%s rank=%s",
                            name,
                            leaf_tuuid,
                            leaf_val.global_shape,
                            leaf_val.origin_indices,
                            leaf_val.routing_indices,
                            compiled_plan.rank,
                        )

                distributor.plan_and_shard(
                    batch_transport_tensors,
                    source_spec,
                    target_spec,
                    tuuid=batch_target_uuids,
                    tuuid_to_name=compiled_plan.tuuid_to_name,
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


def build_empty_step_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    empty_values: Any,
    exec_step: int,
) -> list[StepFrameGroup]:
    frame_groups: list[StepFrameGroup] = []
    for context in compiled_plan.successor_routes:
        distributor = AsyncDataDistributor(context.effective_rank)
        for route in context.tensor_routes:
            for target_rank, expected_count in route.empty_targets:
                distributor.plan_protocol_empty(
                    target_rank=target_rank,
                    expected_count=expected_count,
                    tuuid=route.tuuid,
                    payload=empty_values,
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


def build_successor_batches(
    *,
    compiled_plan: CompiledSendPlan,
    flat_results: list[Any],
    context: CompiledSuccessorRoute,
    step_tuuid_cache: dict[int, Any],
    exec_step: int,
    wrapped_results: dict[int, TransportTensor],
) -> dict[Any, list[tuple[int, TransportTensor]]]:
    batches: dict[Any, list[tuple[int, TransportTensor]]] = defaultdict(list)
    for route in context.tensor_routes:
        if route.output_index >= len(flat_results):
            continue
        raw_leaf = flat_results[route.output_index]
        if raw_leaf is None:
            continue

        leaf_tensor = wrapped_results.get(route.output_index)
        if leaf_tensor is None:
            if isinstance(raw_leaf, TransportTensor):
                leaf_tensor = raw_leaf
            else:
                leaf_tensor = TransportTensor.from_tensor(raw_leaf)
            wrapped_results[route.output_index] = leaf_tensor

        if isinstance(route.source_spec, PhysicalLayout) and isinstance(
            route.source_spec.spec.routing, ReferentialRouting
        ):
            leaf_tensor = inherit_mirror_metadata(
                op_name=compiled_plan.op_name,
                tuuid=route.tuuid,
                leaf_tensor=leaf_tensor,
                source_spec=route.source_spec,
                tuuid_to_name=compiled_plan.tuuid_to_name,
                step_tuuid_cache=step_tuuid_cache,
                exec_step=exec_step,
                spmd_id=context.spmd_id,
                rank=compiled_plan.rank,
            )

        if route.target_spec is not None and route.target_spec.is_content_addressable:
            if (
                leaf_tensor.routing_indices is None
                and leaf_tensor.receipt is None
                and leaf_tensor.flat_values is None
            ):
                raise ValueError(
                    f"[Fluxway Routing Error] Operator '{compiled_plan.op_name}' "
                    f"output '{route.tensor_name}' "
                    "is destined for a dynamic routing edge, but its TransportTensor "
                    "lacks routing metadata (routing_indices, receipt, or flat_values). "
                    "If you returned a TransportTensor manually, ensure you provide explicit routing metadata."
                )

        if leaf_tensor.routing_indices is not None:
            routing_id = id(leaf_tensor.routing_indices)
        elif leaf_tensor.origin_indices is not None:
            routing_id = id(leaf_tensor.origin_indices)
        elif leaf_tensor.receipt is not None:
            routing_id = id(leaf_tensor.receipt)
        else:
            routing_id = 0
        batches[(route.source_spec, route.target_spec, routing_id)].append(
            (route.tuuid, leaf_tensor)
        )
    return batches


def inherit_mirror_metadata(
    *,
    op_name: str,
    tuuid: int,
    leaf_tensor: TransportTensor,
    source_spec: PhysicalLayout,
    tuuid_to_name: dict[int, str] | None,
    step_tuuid_cache: dict[int, Any],
    exec_step: int,
    spmd_id: str,
    rank: int,
) -> TransportTensor:
    if not source_spec.is_referential:
        return leaf_tensor
    mirror_tuuid = getattr(source_spec.spec.routing, "tuuid", None)
    if mirror_tuuid is None:
        return leaf_tensor

    source_ref = step_tuuid_cache.get(mirror_tuuid)
    if source_ref is None:
        output_name = (
            tuuid_to_name.get(tuuid, f"TUUID:{tuuid}")
            if tuuid_to_name
            else f"TUUID:{tuuid}"
        )
        source_path = getattr(source_spec.spec.routing, "path", f"TUUID:{mirror_tuuid}")
        raise ValueError(
            f"[Fluxway Mirror Error] Mirror resolution failed for output '{output_name}' "
            f"in operator '{op_name}'. It is configured to follow the sharding of "
            f"'{source_path}' (TUUID {mirror_tuuid}), but this mirror source was not "
            f"found in the SPMD {spmd_id} rank {rank} step {exec_step} input cache. "
            f"Verify that '{source_path}' is properly declared as an input to '{op_name}' "
            "in the graph configuration."
        )
    if not source_ref.receipt:
        return leaf_tensor

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

    return TransportTensor(
        leaf_tensor.flat_values,
        nested_splits=leaf_tensor.nested_splits,
        routing_indices=leaf_tensor.routing_indices,
        global_shape=inherited_global_shape,
        origin_indices=leaf_tensor.origin_indices or source_ref.origin_indices,
        receipt=source_ref.receipt,
    )
