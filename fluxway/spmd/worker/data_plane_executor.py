"""Data-plane frame build and batch enqueue boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import heapq
from typing import Any

from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    GrantPending,
    GrantReady,
    GrantRejected,
    GrantToken,
)
from fluxway.rpc.protocol import DataPlaneCallable, TransportFrameEnvelope
from fluxway.spmd.worker.data_plane import (
    CompiledSendPlan,
    DependencyDecisions,
    StepFrameGroup,
    build_dependency_decision_only_frame_groups,
    build_completed_without_tensor_step_frame_groups,
    build_step_frame_groups,
)
from fluxway.transport.communication import TransportBatchPlan, TransportFrameBatch
from fluxway.profiler.recorder import TraceRecorder
from fluxway.spmd.worker.runtime import ScheduleKey

SubmitDelivery = Callable[[Awaitable[Any]], Awaitable[asyncio.Task[Any]]]


@dataclass(slots=True)
class _QueuedBatch:
    frames: tuple[TransportFrameEnvelope, ...]
    future: asyncio.Future[Any]
    compiled_plan: CompiledSendPlan
    output_step: int
    target_spmd_id: str
    target_rank: int
    batch_seq: int
    profiler_recorder: TraceRecorder | None
    schedule_key: ScheduleKey


class _TargetBatchQueue:
    def __init__(self, *, max_frames: int, flush_us: int) -> None:
        if max_frames <= 0:
            raise ValueError(f"batch_max_frames must be positive, got {max_frames}.")
        if flush_us < 0:
            raise ValueError(f"batch_flush_us must be non-negative, got {flush_us}.")
        self._max_frames = max_frames
        self._flush_s = flush_us / 1_000_000.0
        self._items: list[tuple[ScheduleKey, int, _QueuedBatch]] = []
        self._sequence = 0
        self._flush_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        *,
        handler: DataPlaneCallable,
        frames: tuple[TransportFrameEnvelope, ...],
        compiled_plan: CompiledSendPlan,
        output_step: int,
        target_spmd_id: str,
        target_rank: int,
        batch_seq: int,
        profiler_recorder: TraceRecorder | None,
        schedule_key: ScheduleKey,
    ) -> Any:
        if not frames:
            raise ValueError("transport frame batch must contain at least one frame.")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        async with self._lock:
            item = _QueuedBatch(
                frames=frames,
                future=future,
                compiled_plan=compiled_plan,
                output_step=output_step,
                target_spmd_id=target_spmd_id,
                target_rank=target_rank,
                batch_seq=batch_seq,
                profiler_recorder=profiler_recorder,
                schedule_key=schedule_key,
            )
            heapq.heappush(
                self._items,
                (schedule_key, self._sequence, item),
            )
            self._sequence += 1
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_later(handler))
        return await future

    async def _flush_later(self, handler: DataPlaneCallable) -> None:
        if self._flush_s > 0:
            await asyncio.sleep(self._flush_s)
        while True:
            async with self._lock:
                if not self._items:
                    return
                selected: list[_QueuedBatch] = []
                frame_count = 0
                while self._items:
                    _next_schedule_key, _next_sequence, next_item = self._items[0]
                    next_frame_count = len(next_item.frames)
                    if selected and frame_count + next_frame_count > self._max_frames:
                        break
                    _schedule_key, _sequence, item = heapq.heappop(self._items)
                    selected.append(item)
                    frame_count += next_frame_count
                    if frame_count >= self._max_frames:
                        break
            await self._flush_selected(handler, selected)

    async def _flush_selected(
        self,
        handler: DataPlaneCallable,
        selected: list[_QueuedBatch],
    ) -> None:
        frames = tuple(frame for item in selected for frame in item.frames)
        first = selected[0]
        try:
            result = await _send_transport_frame_batch_to_handler(
                handler,
                frames,
                compiled_plan=first.compiled_plan,
                output_step=first.output_step,
                target_rank=first.target_rank,
                target_spmd_id=first.target_spmd_id,
                batch_seq=first.batch_seq,
                profiler_recorder=first.profiler_recorder,
            )
        except Exception as exc:
            for item in selected:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for item in selected:
            if not item.future.done():
                item.future.set_result(result)


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


async def _build_and_send_frame_groups(
    *,
    compiled_plan: CompiledSendPlan,
    frame_group_builder: Callable[[TraceRecorder | None], list[StepFrameGroup]],
    exec_step: int,
    successor_handlers: tuple[dict[int, Any], ...],
    source_spmd_id: str,
    batch_max_frames: int,
    submit_delivery: SubmitDelivery,
    batch_delivery_factory: Callable[..., Awaitable[Any]] | None = None,
    profiler_recorder: TraceRecorder | None = None,
    delivery_schedule_key: ScheduleKey,
    coherence_directory: CoherenceDirectory,
) -> list[asyncio.Task[Any]]:
    if len(successor_handlers) != len(compiled_plan.successor_routes):
        raise ValueError(
            "successor_handlers must align with compiled successor routes, "
            f"got handlers={len(successor_handlers)} "
            f"routes={len(compiled_plan.successor_routes)}."
        )
    if batch_max_frames <= 0:
        raise ValueError(f"batch_max_frames must be positive, got {batch_max_frames}.")

    def build_frame_groups_sync() -> list[Any]:
        if profiler_recorder is None:
            return frame_group_builder(None)
        with profiler_recorder.data_plane_frame_build_sync(
            op_name=compiled_plan.op_name,
            rank=compiled_plan.rank,
            step=exec_step,
            successor_count=len(compiled_plan.successor_routes),
            output_count=compiled_plan.output_count,
        ):
            return frame_group_builder(profiler_recorder)

    frame_groups = await asyncio.to_thread(build_frame_groups_sync)
    if not frame_groups:
        return []

    batches = TransportBatchPlan.from_frame_groups(frame_groups).batches(
        batch_max_frames=batch_max_frames,
    )
    delivery_tasks: list[asyncio.Task[Any]] = []
    if profiler_recorder is None:
        for batch in batches:
            route = compiled_plan.successor_routes[batch.context_index]
            target_spmd_id = route.spmd_id
            _validate_local_only_batch(route, compiled_plan, batch)
            handler = _handler_for_batch(
                batch,
                successor_handlers=successor_handlers,
                source_spmd_id=source_spmd_id,
                source_rank=compiled_plan.rank,
                output_step=exec_step,
            )
            frames = _batch_envelopes(batch)
            delegated_tokens = await _delegate_batch_reads(
                coherence_directory,
                compiled_plan,
                batch,
            )
            delivery = (
                batch_delivery_factory(
                    handler=handler,
                    frames=frames,
                    compiled_plan=compiled_plan,
                    output_step=exec_step,
                    target_spmd_id=target_spmd_id,
                    target_rank=batch.target_rank,
                    batch_seq=batch.batch_seq,
                    profiler_recorder=None,
                    schedule_key=delivery_schedule_key,
                )
                if batch_delivery_factory is not None
                else _send_transport_frame_batch_to_handler(
                    handler,
                    frames,
                    compiled_plan=compiled_plan,
                    output_step=exec_step,
                    target_spmd_id=target_spmd_id,
                    target_rank=batch.target_rank,
                    batch_seq=batch.batch_seq,
                )
            )
            delivery_tasks.append(
                await _submit_delegated_batch(
                    directory=coherence_directory,
                    delegated_tokens=delegated_tokens,
                    delivery=delivery,
                    submit_delivery=submit_delivery,
                )
            )
        return delivery_tasks

    async with profiler_recorder.data_plane_frame_submit(
        source_rank=compiled_plan.rank,
        output_step=exec_step,
        frame_count=len(frame_groups),
    ):
        for batch in batches:
            route = compiled_plan.successor_routes[batch.context_index]
            target_spmd_id = route.spmd_id
            _validate_local_only_batch(route, compiled_plan, batch)
            handler = _handler_for_batch(
                batch,
                successor_handlers=successor_handlers,
                source_spmd_id=source_spmd_id,
                source_rank=compiled_plan.rank,
                output_step=exec_step,
            )
            frames = _batch_envelopes(batch)
            delegated_tokens = await _delegate_batch_reads(
                coherence_directory,
                compiled_plan,
                batch,
            )
            async with profiler_recorder.data_plane_enqueue_batch(
                op_name=compiled_plan.op_name,
                step=exec_step,
                source_spmd_code=compiled_plan.sender_spmd_code,
                source_rank=compiled_plan.rank,
                target_spmd_id=target_spmd_id,
                target_rank=batch.target_rank,
                batch_seq=batch.batch_seq,
                frame_count=len(frames),
            ):
                delivery = (
                    batch_delivery_factory(
                        handler=handler,
                        frames=frames,
                        compiled_plan=compiled_plan,
                        output_step=exec_step,
                        target_spmd_id=target_spmd_id,
                        target_rank=batch.target_rank,
                        batch_seq=batch.batch_seq,
                        profiler_recorder=profiler_recorder,
                        schedule_key=delivery_schedule_key,
                    )
                    if batch_delivery_factory is not None
                    else _send_transport_frame_batch_to_handler(
                        handler,
                        frames,
                        compiled_plan=compiled_plan,
                        output_step=exec_step,
                        target_spmd_id=target_spmd_id,
                        target_rank=batch.target_rank,
                        batch_seq=batch.batch_seq,
                        profiler_recorder=profiler_recorder,
                    )
                )
                delivery_tasks.append(
                    await _submit_delegated_batch(
                        directory=coherence_directory,
                        delegated_tokens=delegated_tokens,
                        delivery=delivery,
                        submit_delivery=submit_delivery,
                    )
                )
    return delivery_tasks


def _delegated_reads(
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
) -> set[tuple[CoherenceSubject, int, int]]:
    return {
        (
            CoherenceSubject(
                compiled_plan.sender_spmd_code,
                compiled_plan.rank,
                compiled_plan.coherence_logical_tuuid(item.transfer_code),
            ),
            item.epoch,
            compiled_plan.coherence_child_slot(
                item.transfer_code,
                batch.target_rank,
            ),
        )
        for item in batch.delegations
    }


async def _delegate_batch_reads(
    directory: CoherenceDirectory,
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
) -> tuple[GrantToken, ...]:
    tokens: list[GrantToken] = []
    try:
        for subject, epoch, child_slot in _delegated_reads(compiled_plan, batch):
            result = directory.delegate_local_payload(subject, epoch, child_slot)
            if isinstance(result, GrantReady):
                tokens.append(directory.register_child_delivery(result.token))
                continue
            if isinstance(result, GrantPending):
                token = await result.ready
                tokens.append(directory.register_child_delivery(token))
                continue
            if isinstance(result, GrantRejected):
                raise result.error
            raise TypeError(f"unsupported grant result={type(result).__name__}.")
    except BaseException:
        _complete_delegations(directory, tokens, delivered=False)
        raise
    return tuple(tokens)


def _complete_delegations(
    directory: CoherenceDirectory,
    tokens: list[GrantToken] | tuple[GrantToken, ...],
    *,
    delivered: bool,
) -> None:
    for token in reversed(tokens):
        directory.complete_child_delivery(token, delivered=delivered)


async def _submit_delegated_batch(
    *,
    directory: CoherenceDirectory,
    delegated_tokens: tuple[GrantToken, ...],
    delivery: Awaitable[Any],
    submit_delivery: SubmitDelivery,
) -> asyncio.Task[Any]:
    async def deliver_and_complete() -> Any:
        try:
            result = await delivery
        except BaseException:
            _complete_delegations(directory, delegated_tokens, delivered=False)
            raise
        _complete_delegations(directory, delegated_tokens, delivered=True)
        return result

    try:
        return await submit_delivery(deliver_and_complete())
    except BaseException:
        _complete_delegations(directory, delegated_tokens, delivered=False)
        raise


def _handler_for_batch(
    batch: TransportFrameBatch,
    *,
    successor_handlers: tuple[dict[int, Any], ...],
    source_spmd_id: str,
    source_rank: int,
    output_step: int,
) -> DataPlaneCallable:
    handlers = successor_handlers[batch.context_index]
    handler = handlers.get(batch.target_rank)
    if handler is None:
        raise ValueError(
            "[SPMDWorker] Missing successor handler for transport frame: "
            f"source={source_spmd_id}:{source_rank} target_rank={batch.target_rank} "
            f"step={output_step}."
        )
    effective_handler = getattr(handler, "delegate", handler)
    if not isinstance(effective_handler, DataPlaneCallable):
        raise TypeError(
            "Batching data-plane executor requires DataPlaneCallable handlers, "
            f"got {type(handler).__name__} for context_index={batch.context_index} "
            f"target_rank={batch.target_rank}."
        )
    return handler


def _batch_envelopes(
    batch: TransportFrameBatch,
) -> tuple[TransportFrameEnvelope, ...]:
    return tuple(
        TransportFrameEnvelope(header=header, payload_segments=payload_segments)
        for header, payload_segments in batch.frames
    )


def _validate_local_only_batch(
    route: Any,
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
) -> None:
    if not getattr(route, "local_only", False):
        return
    if batch.target_rank != compiled_plan.rank:
        raise ValueError(
            "local_only transport requires same-rank routing, "
            f"op={compiled_plan.op_name} source_rank={compiled_plan.rank} "
            f"target_spmd={route.spmd_id} target_rank={batch.target_rank}."
        )


async def _send_transport_frame_batch_to_handler(
    handler: DataPlaneCallable,
    frames: tuple[TransportFrameEnvelope, ...],
    *,
    compiled_plan: CompiledSendPlan,
    output_step: int,
    target_spmd_id: str,
    target_rank: int,
    batch_seq: int,
    profiler_recorder: TraceRecorder | None = None,
) -> Any:
    if not frames:
        raise ValueError("transport frame batch must contain at least one frame.")
    if profiler_recorder is None:
        return await handler.receive_transport_frames(frames)
    async with profiler_recorder.data_plane_batch_rpc(
        op_name=compiled_plan.op_name,
        step=output_step,
        source_spmd_code=compiled_plan.sender_spmd_code,
        source_rank=compiled_plan.rank,
        target_spmd_id=target_spmd_id,
        target_rank=target_rank,
        batch_seq=batch_seq,
        frame_count=len(frames),
    ):
        return await handler.receive_transport_frames(frames)


class BatchingDataPlaneExecutor:
    def __init__(self, coherence_directory: CoherenceDirectory) -> None:
        self._queues: dict[tuple[int, int], _TargetBatchQueue] = {}
        self._coherence_directory = coherence_directory

    async def send_tensor_frames(
        self,
        *,
        compiled_plan: CompiledSendPlan,
        flat_results: list[Any],
        step_transfer_cache: dict[int, Any],
        exec_step: int,
        successor_handlers: tuple[dict[int, Any], ...],
        source_spmd_id: str,
        batch_max_frames: int,
        batch_flush_us: int,
        submit_delivery: SubmitDelivery,
        dependency_decisions: DependencyDecisions = None,
        profiler_recorder: TraceRecorder | None = None,
        delivery_schedule_key: ScheduleKey,
    ) -> list[asyncio.Task[Any]]:
        def build_frame_groups(recorder: TraceRecorder | None) -> list[StepFrameGroup]:
            return build_step_frame_groups(
                compiled_plan=compiled_plan,
                flat_results=flat_results,
                step_transfer_cache=step_transfer_cache,
                exec_step=exec_step,
                dependency_decisions=dependency_decisions,
                profiler_recorder=recorder,
            )

        return await self._send_frame_groups(
            compiled_plan=compiled_plan,
            frame_group_builder=build_frame_groups,
            exec_step=exec_step,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            batch_max_frames=batch_max_frames,
            batch_flush_us=batch_flush_us,
            submit_delivery=submit_delivery,
            profiler_recorder=profiler_recorder,
            delivery_schedule_key=delivery_schedule_key,
        )

    async def send_completed_without_tensor_frames(
        self,
        *,
        compiled_plan: CompiledSendPlan,
        empty_values: Any,
        exec_step: int,
        successor_handlers: tuple[dict[int, Any], ...],
        source_spmd_id: str,
        batch_max_frames: int,
        batch_flush_us: int,
        submit_delivery: SubmitDelivery,
        dependency_decisions: DependencyDecisions = None,
        profiler_recorder: TraceRecorder | None = None,
        delivery_schedule_key: ScheduleKey,
    ) -> list[asyncio.Task[Any]]:
        def build_frame_groups(_recorder: TraceRecorder | None) -> list[StepFrameGroup]:
            return build_completed_without_tensor_step_frame_groups(
                compiled_plan=compiled_plan,
                empty_values=empty_values,
                exec_step=exec_step,
                dependency_decisions=dependency_decisions,
            )

        return await self._send_frame_groups(
            compiled_plan=compiled_plan,
            frame_group_builder=build_frame_groups,
            exec_step=exec_step,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            batch_max_frames=batch_max_frames,
            batch_flush_us=batch_flush_us,
            submit_delivery=submit_delivery,
            profiler_recorder=profiler_recorder,
            delivery_schedule_key=delivery_schedule_key,
        )

    async def send_dependency_decision_frames(
        self,
        *,
        compiled_plan: CompiledSendPlan,
        exec_step: int,
        successor_handlers: tuple[dict[int, Any], ...],
        source_spmd_id: str,
        batch_max_frames: int,
        batch_flush_us: int,
        submit_delivery: SubmitDelivery,
        dependency_decisions: DependencyDecisions,
        profiler_recorder: TraceRecorder | None = None,
        delivery_schedule_key: ScheduleKey,
    ) -> list[asyncio.Task[Any]]:
        def build_frame_groups(_recorder: TraceRecorder | None) -> list[StepFrameGroup]:
            assert dependency_decisions is not None
            return build_dependency_decision_only_frame_groups(
                compiled_plan=compiled_plan,
                dependency_decisions=dependency_decisions,
                exec_step=exec_step,
            )

        return await self._send_frame_groups(
            compiled_plan=compiled_plan,
            frame_group_builder=build_frame_groups,
            exec_step=exec_step,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            batch_max_frames=batch_max_frames,
            batch_flush_us=batch_flush_us,
            submit_delivery=submit_delivery,
            profiler_recorder=profiler_recorder,
            delivery_schedule_key=delivery_schedule_key,
        )

    async def _send_frame_groups(
        self,
        *,
        compiled_plan: CompiledSendPlan,
        frame_group_builder: Callable[[TraceRecorder | None], list[StepFrameGroup]],
        exec_step: int,
        successor_handlers: tuple[dict[int, Any], ...],
        source_spmd_id: str,
        batch_max_frames: int,
        batch_flush_us: int,
        submit_delivery: SubmitDelivery,
        profiler_recorder: TraceRecorder | None,
        delivery_schedule_key: ScheduleKey,
    ) -> list[asyncio.Task[Any]]:
        return await _build_and_send_frame_groups(
            compiled_plan=compiled_plan,
            frame_group_builder=frame_group_builder,
            exec_step=exec_step,
            successor_handlers=successor_handlers,
            source_spmd_id=source_spmd_id,
            batch_max_frames=batch_max_frames,
            submit_delivery=submit_delivery,
            batch_delivery_factory=lambda **kwargs: self._enqueue_batch_delivery(
                batch_max_frames=batch_max_frames,
                batch_flush_us=batch_flush_us,
                **kwargs,
            ),
            profiler_recorder=profiler_recorder,
            delivery_schedule_key=delivery_schedule_key,
            coherence_directory=self._coherence_directory,
        )

    async def _enqueue_batch_delivery(
        self,
        *,
        handler: DataPlaneCallable,
        frames: tuple[TransportFrameEnvelope, ...],
        compiled_plan: CompiledSendPlan,
        output_step: int,
        target_spmd_id: str,
        target_rank: int,
        batch_seq: int,
        profiler_recorder: TraceRecorder | None,
        batch_max_frames: int,
        batch_flush_us: int,
        schedule_key: ScheduleKey,
    ) -> Any:
        key = (id(handler), target_rank)
        queue = self._queues.get(key)
        if queue is None:
            queue = _TargetBatchQueue(
                max_frames=batch_max_frames,
                flush_us=batch_flush_us,
            )
            self._queues[key] = queue
        return await queue.enqueue(
            handler=handler,
            frames=frames,
            compiled_plan=compiled_plan,
            output_step=output_step,
            target_spmd_id=target_spmd_id,
            target_rank=target_rank,
            batch_seq=batch_seq,
            profiler_recorder=profiler_recorder,
            schedule_key=schedule_key,
        )
