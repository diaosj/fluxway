"""Data-plane frame build and batch enqueue boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
import heapq
from typing import Any

from fluxway.rpc.protocol import (
    DataPlaneCallable,
    TransportFrameEnvelope,
)
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
BatchEnqueue = Callable[[TransportFrameBatch], Awaitable[Any]]
BatchSubmit = Callable[
    [tuple[TransportFrameBatch, ...], BatchEnqueue],
    Awaitable[Any],
]


@dataclass(slots=True)
class _QueuedBatch:
    handler: DataPlaneCallable
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
                handler=handler,
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
                self._flush_task = asyncio.create_task(self._flush_later())
                self._flush_task.add_done_callback(self._on_flush_done)
        try:
            return await future
        except asyncio.CancelledError:
            async with self._lock:
                self._items = [
                    item_record
                    for item_record in self._items
                    if item_record[2] is not item
                ]
                heapq.heapify(self._items)
            if not future.done():
                future.cancel()
            raise

    def _on_flush_done(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            return
        pending = [item for _key, _sequence, item in self._items]
        self._items.clear()
        for item in pending:
            if not item.future.done():
                item.future.cancel()

    async def _flush_later(self) -> None:
        selected: list[_QueuedBatch] = []
        try:
            if self._flush_s > 0:
                await asyncio.sleep(self._flush_s)
            while True:
                async with self._lock:
                    if not self._items:
                        return
                    selected = []
                    frame_count = 0
                    while self._items:
                        _next_schedule_key, _next_sequence, next_item = self._items[0]
                        next_frame_count = len(next_item.frames)
                        if (
                            selected
                            and frame_count + next_frame_count > self._max_frames
                        ):
                            break
                        if selected and next_item.handler is not selected[0].handler:
                            break
                        _schedule_key, _sequence, item = heapq.heappop(self._items)
                        selected.append(item)
                        frame_count += next_frame_count
                        if frame_count >= self._max_frames:
                            break
                await self._flush_selected(selected)
                selected = []
        except BaseException as error:
            pending = selected + [item for _key, _seq, item in self._items]
            self._items.clear()
            for item in pending:
                if item.future.done():
                    continue
                if isinstance(error, asyncio.CancelledError):
                    item.future.cancel()
                else:
                    item.future.set_exception(error)
            raise

    async def _flush_selected(
        self,
        selected: list[_QueuedBatch],
    ) -> None:
        frames = tuple(frame for item in selected for frame in item.frames)
        first = selected[0]
        try:
            result = await _send_transport_frame_batch_to_handler(
                first.handler,
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
    batch_delivery_factory: Callable[..., Awaitable[Any]],
    batch_submit: BatchSubmit,
    profiler_recorder: TraceRecorder | None = None,
    delivery_schedule_key: ScheduleKey,
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

    async def enqueue_batch(batch: TransportFrameBatch) -> Any:
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

        async def send_batch() -> Any:
            async def send() -> Any:
                return await batch_delivery_factory(
                    handler=handler,
                    frames=frames,
                    compiled_plan=compiled_plan,
                    output_step=exec_step,
                    context_index=batch.context_index,
                    target_spmd_id=target_spmd_id,
                    target_rank=batch.target_rank,
                    batch_seq=batch.batch_seq,
                    profiler_recorder=profiler_recorder,
                    schedule_key=delivery_schedule_key,
                )

            task = await submit_delivery(send())
            return await task

        if profiler_recorder is None:
            return await send_batch()
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
            return await send_batch()

    publication = batch_submit(tuple(batches), enqueue_batch)
    publication_task = asyncio.ensure_future(publication)
    if profiler_recorder is None:
        return [publication_task]
    try:
        async with profiler_recorder.data_plane_frame_submit(
            source_rank=compiled_plan.rank,
            output_step=exec_step,
            frame_count=len(frame_groups),
        ):
            return [publication_task]
    except BaseException:
        await _cancel_delivery_tasks((publication_task,))
        raise


async def _cancel_delivery_tasks(tasks: Iterable[asyncio.Task[Any]]) -> None:
    pending = tuple(task for task in tasks if not task.done())
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


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
    def __init__(self) -> None:
        self._queues: dict[tuple[str, int, str, int, str, int], _TargetBatchQueue] = {}

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
        batch_submit: BatchSubmit,
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
            batch_submit=batch_submit,
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
        batch_submit: BatchSubmit,
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
            batch_submit=batch_submit,
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
        batch_submit: BatchSubmit,
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
            batch_submit=batch_submit,
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
        batch_submit: BatchSubmit,
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
            batch_submit=batch_submit,
        )

    async def _enqueue_batch_delivery(
        self,
        *,
        handler: DataPlaneCallable,
        frames: tuple[TransportFrameEnvelope, ...],
        compiled_plan: CompiledSendPlan,
        output_step: int,
        context_index: int,
        target_spmd_id: str,
        target_rank: int,
        batch_seq: int,
        profiler_recorder: TraceRecorder | None,
        batch_max_frames: int,
        batch_flush_us: int,
        schedule_key: ScheduleKey,
    ) -> Any:
        key = (
            compiled_plan.source_spmd_id,
            compiled_plan.rank,
            compiled_plan.op_name,
            context_index,
            target_spmd_id,
            target_rank,
        )
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
