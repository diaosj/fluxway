"""Data-plane backend executor boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import ray

from fluxway.common.types.sharding import ShardingSpec
from fluxway.rpc.protocol import AxonDataPlane, RayActorDataPlane
from fluxway.spmd.worker.data_plane import (
    FrameArg,
    SuccessorRouteContext,
    build_step_frame_groups,
)
from fluxway.transport.manifest import TransportFrameHeader


# pyrefly: ignore [no-matching-overload]
@ray.remote
def _ray_send_step_frames(
    op_name: str,
    rank: int,
    mesh: Any,
    flat_results: list[Any],
    output_uuids: list[int],
    output_sharding: dict[int, ShardingSpec],
    successor_contexts: list[SuccessorRouteContext],
    tuuid_to_name: dict[int, str] | None,
    step_tuuid_cache: dict[int, Any],
    exec_step: int,
    sender_spmd_code: int,
    sender_world_size: int,
    interleave_unit_size: int | None,
    successor_handlers: list[dict[int, Any]],
    source_spmd_id: str,
    max_in_flight_frames: int,
) -> list[bool]:
    return asyncio.run(
        build_and_send_step_frames(
            op_name=op_name,
            rank=rank,
            mesh=mesh,
            flat_results=flat_results,
            output_uuids=output_uuids,
            output_sharding=output_sharding,
            successor_contexts=successor_contexts,
            tuuid_to_name=tuuid_to_name,
            step_tuuid_cache=step_tuuid_cache,
            exec_step=exec_step,
            sender_spmd_code=sender_spmd_code,
            sender_world_size=sender_world_size,
            interleave_unit_size=interleave_unit_size,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            max_in_flight_frames=max_in_flight_frames,
        )
    )


def _object_requires_cuda_visible(obj: Any, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)
    if hasattr(obj, "__cuda_array_interface__"):
        return True
    dlpack_device = getattr(obj, "__dlpack_device__", None)
    if callable(dlpack_device):
        device = dlpack_device()
        if isinstance(device, tuple) and device and int(device[0]) == 2:
            return True
    if isinstance(obj, dict):
        return any(_object_requires_cuda_visible(value, seen) for value in obj.values())
    if isinstance(obj, list | tuple):
        return any(_object_requires_cuda_visible(value, seen) for value in obj)
    for attr in (
        "flat_values",
        "nested_splits",
        "routing_indices",
        "origin_indices",
        "receipt",
        "sender_rank_tags",
    ):
        if hasattr(obj, attr):
            if _object_requires_cuda_visible(getattr(obj, attr), seen):
                return True
    return False


async def build_and_send_step_frames(
    *,
    op_name: str,
    rank: int,
    mesh: Any,
    flat_results: list[Any],
    output_uuids: list[int],
    output_sharding: dict[int, ShardingSpec],
    successor_contexts: list[SuccessorRouteContext],
    tuuid_to_name: dict[int, str] | None,
    step_tuuid_cache: dict[int, Any],
    exec_step: int,
    sender_spmd_code: int,
    sender_world_size: int,
    interleave_unit_size: int | None,
    successor_handlers: list[dict[int, Any]],
    source_spmd_id: str,
    max_in_flight_frames: int,
    build_in_thread: bool = False,
) -> list[bool]:
    if len(successor_handlers) != len(successor_contexts):
        raise ValueError(
            "successor_handlers must align with successor_contexts, "
            f"got handlers={len(successor_handlers)} contexts={len(successor_contexts)}."
        )
    if max_in_flight_frames <= 0:
        raise ValueError(
            f"max_in_flight_frames must be positive, got {max_in_flight_frames}."
        )
    build_kwargs = dict(
        op_name=op_name,
        rank=rank,
        mesh=mesh,
        flat_results=flat_results,
        output_uuids=output_uuids,
        output_sharding=output_sharding,
        successor_contexts=successor_contexts,
        tuuid_to_name=tuuid_to_name,
        step_tuuid_cache=step_tuuid_cache,
        exec_step=exec_step,
        sender_spmd_code=sender_spmd_code,
        sender_world_size=sender_world_size,
        interleave_unit_size=interleave_unit_size,
    )
    if build_in_thread:
        frame_groups = await asyncio.to_thread(build_step_frame_groups, **build_kwargs)
    else:
        frame_groups = build_step_frame_groups(**build_kwargs)
    if not frame_groups:
        return []
    send_semaphore = asyncio.Semaphore(max_in_flight_frames)
    return await asyncio.gather(
        *(
            _send_transport_frame_to_handler(
                successor_handlers[context_index],
                frame_arg,
                source_spmd_id=source_spmd_id,
                source_rank=rank,
                output_step=exec_step,
                send_semaphore=send_semaphore,
            )
            for context_index, frame_arg in frame_groups
        )
    )


async def _send_transport_frame_to_handler(
    handlers: dict[int, Any],
    frame_arg: FrameArg,
    *,
    source_spmd_id: str,
    source_rank: int,
    output_step: int,
    send_semaphore: asyncio.Semaphore,
) -> bool:
    (
        target_rank,
        header,
        payload_segments,
    ) = frame_arg
    if not isinstance(header, TransportFrameHeader):
        raise TypeError(
            f"frame_arg header must be TransportFrameHeader, got {type(header).__name__}."
        )
    handler = handlers.get(target_rank)
    if handler is None:
        raise ValueError(
            "[SPMDWorker] Missing successor handler for transport frame: "
            f"source={source_spmd_id}:{source_rank} target_rank={target_rank} "
            f"step={output_step}."
        )

    await send_semaphore.acquire()
    try:
        await handler.receive_transport_frame(header, *payload_segments)
    finally:
        send_semaphore.release()
    return True


def _validate_successor_handlers(
    successor_handlers: list[dict[int, Any]],
    *,
    expected_type: type,
    executor_name: str,
) -> None:
    for context_index, handlers in enumerate(successor_handlers):
        for target_rank, handler in handlers.items():
            if not isinstance(handler, expected_type):
                raise TypeError(
                    f"{executor_name} requires {expected_type.__name__} "
                    f"handlers, got {type(handler).__name__} for "
                    f"context_index={context_index} target_rank={target_rank}."
                )


class RayDataPlaneBackendExecutor:
    async def send_step_frames(
        self,
        *,
        op_name: str,
        rank: int,
        mesh: Any,
        flat_results: list[Any],
        output_uuids: list[int],
        output_sharding: dict[int, ShardingSpec],
        successor_contexts: list[SuccessorRouteContext],
        tuuid_to_name: dict[int, str] | None,
        step_tuuid_cache: dict[int, Any],
        exec_step: int,
        sender_spmd_code: int,
        sender_world_size: int,
        interleave_unit_size: int | None,
        successor_handlers: list[dict[int, Any]],
        source_spmd_id: str,
        max_in_flight_frames: int,
    ) -> list[bool]:
        if max_in_flight_frames <= 0:
            raise ValueError(
                f"max_in_flight_frames must be positive, got {max_in_flight_frames}."
            )
        _validate_successor_handlers(
            successor_handlers,
            expected_type=RayActorDataPlane,
            executor_name="Ray fused data-plane executor",
        )
        if _object_requires_cuda_visible((flat_results, step_tuuid_cache)):
            return await build_and_send_step_frames(
                op_name=op_name,
                rank=rank,
                mesh=mesh,
                flat_results=flat_results,
                output_uuids=output_uuids,
                output_sharding=output_sharding,
                successor_contexts=successor_contexts,
                tuuid_to_name=tuuid_to_name,
                step_tuuid_cache=step_tuuid_cache,
                exec_step=exec_step,
                sender_spmd_code=sender_spmd_code,
                sender_world_size=sender_world_size,
                interleave_unit_size=interleave_unit_size,
                successor_handlers=successor_handlers,
                source_spmd_id=source_spmd_id,
                max_in_flight_frames=max_in_flight_frames,
                build_in_thread=True,
            )
        ray_task = _ray_send_step_frames
        ref = ray_task.remote(
            op_name,
            rank,
            mesh,
            flat_results,
            output_uuids,
            output_sharding,
            successor_contexts,
            tuuid_to_name,
            step_tuuid_cache,
            exec_step,
            sender_spmd_code,
            sender_world_size,
            interleave_unit_size,
            successor_handlers,
            source_spmd_id,
            max_in_flight_frames,
        )
        return await ref

    async def send_transport_frame(
        self,
        handlers: dict[int, Any],
        frame_arg: FrameArg,
        *,
        source_spmd_id: str,
        source_rank: int,
        output_step: int,
        send_semaphore: asyncio.Semaphore,
    ) -> bool:
        return await _send_transport_frame_to_handler(
            handlers,
            frame_arg,
            source_spmd_id=source_spmd_id,
            source_rank=source_rank,
            output_step=output_step,
            send_semaphore=send_semaphore,
        )


class AxonDataPlaneBackendExecutor:
    async def send_step_frames(
        self,
        *,
        op_name: str,
        rank: int,
        mesh: Any,
        flat_results: list[Any],
        output_uuids: list[int],
        output_sharding: dict[int, ShardingSpec],
        successor_contexts: list[SuccessorRouteContext],
        tuuid_to_name: dict[int, str] | None,
        step_tuuid_cache: dict[int, Any],
        exec_step: int,
        sender_spmd_code: int,
        sender_world_size: int,
        interleave_unit_size: int | None,
        successor_handlers: list[dict[int, Any]],
        source_spmd_id: str,
        max_in_flight_frames: int,
    ) -> list[bool]:
        _validate_successor_handlers(
            successor_handlers,
            expected_type=AxonDataPlane,
            executor_name="Axon fused data-plane executor",
        )
        return await build_and_send_step_frames(
            op_name=op_name,
            rank=rank,
            mesh=mesh,
            flat_results=flat_results,
            output_uuids=output_uuids,
            output_sharding=output_sharding,
            successor_contexts=successor_contexts,
            tuuid_to_name=tuuid_to_name,
            step_tuuid_cache=step_tuuid_cache,
            exec_step=exec_step,
            sender_spmd_code=sender_spmd_code,
            sender_world_size=sender_world_size,
            interleave_unit_size=interleave_unit_size,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            max_in_flight_frames=max_in_flight_frames,
        )
