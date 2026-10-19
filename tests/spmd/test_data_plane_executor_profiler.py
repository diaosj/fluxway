import asyncio
import gc
import threading
import types
import weakref

import numpy as np
import pytest
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.rpc.protocol import (
    AxonDataPlane,
    DataPlaneCallable,
    TransportFrameEnvelope,
)
from fluxway.spmd.worker import data_plane_executor as executor_module
from fluxway.spmd.worker.data_plane import CompiledSendPlan, CompiledSuccessorRoute
from fluxway.spmd.worker.data_plane_executor import (
    BatchingDataPlaneExecutor,
    _TargetBatchQueue,
)
from fluxway.transport.manifest import (
    ENTRY_WIDTH,
    PAYLOAD_WIDTH,
    PayloadRole,
    TRANSPORT_ABI_TAG,
    TensorEntryKind,
    TransportFrameBuilder,
    TransportFrameHeader,
)


class _Handler(DataPlaneCallable):
    def __init__(self) -> None:
        self.calls = 0

    async def receive_transport_frames(self, frames):
        self.calls += 1
        return True

    async def receive_control(self, op_name, step, names, values):
        return True


def _empty_header(step: int = 0) -> TransportFrameHeader:
    return TransportFrameHeader(
        frame_i32=np.array(
            [TRANSPORT_ABI_TAG, step, 0, 0, 1, 0, 0, 0],
            dtype=np.int32,
        ),
        entry_i32=np.zeros((0, ENTRY_WIDTH), dtype=np.int32),
        payload_i32=np.zeros((0, PAYLOAD_WIDTH), dtype=np.int32),
        shape_i32=np.zeros((0,), dtype=np.int32),
        global_shape_i64=np.zeros((0,), dtype=np.int64),
        compact_i64=np.zeros((0,), dtype=np.int64),
    )


def _compiled_plan() -> CompiledSendPlan:
    return CompiledSendPlan.create(
        op_name="op",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(),
        output_count=1,
        transfer_code_to_name={},
    )


@pytest.mark.asyncio
async def test_target_batch_queue_drains_enqueue_during_inflight_flush():
    class BlockingHandler(DataPlaneCallable):
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def receive_transport_frames(self, frames):
            self.calls.append(len(frames))
            self.entered.set()
            await self.release.wait()
            return len(self.calls)

        async def receive_control(self, op_name, step, names, values):
            return True

    queue = _TargetBatchQueue(max_frames=1, flush_us=0)
    handler = BlockingHandler()
    frames = (TransportFrameEnvelope(header=_empty_header(), payload_segments=()),)
    plan = _compiled_plan()

    first = asyncio.create_task(
        queue.enqueue(
            handler=handler,
            frames=frames,
            compiled_plan=plan,
            output_step=1,
            target_spmd_id="succ",
            target_rank=0,
            batch_seq=0,
            profiler_recorder=None,
            schedule_key=(0, 0, 1, "op"),
        )
    )
    await handler.entered.wait()
    second = asyncio.create_task(
        queue.enqueue(
            handler=handler,
            frames=frames,
            compiled_plan=plan,
            output_step=0,
            target_spmd_id="succ",
            target_rank=0,
            batch_seq=1,
            profiler_recorder=None,
            schedule_key=(0, 0, 0, "op"),
        )
    )
    await asyncio.sleep(0.01)
    handler.release.set()

    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)
    assert handler.calls == [1, 1]


@pytest.mark.asyncio
async def test_target_batch_queue_cancels_waiters_when_flush_is_cancelled():
    class BlockingHandler(DataPlaneCallable):
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def receive_transport_frames(self, _frames):
            self.entered.set()
            await asyncio.Event().wait()

        async def receive_control(self, _op_name, _step, _names, _values):
            return True

    queue = _TargetBatchQueue(max_frames=1, flush_us=0)
    handler = BlockingHandler()
    frames = (TransportFrameEnvelope(header=_empty_header(), payload_segments=()),)
    enqueue = asyncio.create_task(
        queue.enqueue(
            handler=handler,
            frames=frames,
            compiled_plan=_compiled_plan(),
            output_step=0,
            target_spmd_id="succ",
            target_rank=0,
            batch_seq=0,
            profiler_recorder=None,
            schedule_key=(0, 0, 0, "op"),
        )
    )
    await handler.entered.wait()
    queue._flush_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(enqueue), timeout=1.0)


@pytest.mark.asyncio
async def test_profiler_builds_frames_in_background_thread_and_records_split_spans(
    monkeypatch,
    tmp_path,
):
    event_loop_thread = threading.get_ident()
    build_threads: list[int] = []

    def fake_build_step_frame_groups(**kwargs):
        build_threads.append(threading.get_ident())
        header = TransportFrameHeader(
            frame_i32=np.array(
                [TRANSPORT_ABI_TAG, kwargs["exec_step"], 0, 0, 1, 0, 0, 0],
                dtype=np.int32,
            ),
            entry_i32=np.zeros((0, ENTRY_WIDTH), dtype=np.int32),
            payload_i32=np.zeros((0, PAYLOAD_WIDTH), dtype=np.int32),
            shape_i32=np.zeros((0,), dtype=np.int32),
            global_shape_i64=np.zeros((0,), dtype=np.int64),
            compact_i64=np.zeros((0,), dtype=np.int64),
        )
        return [(0, (0, header, ()))]

    monkeypatch.setattr(
        executor_module,
        "build_step_frame_groups",
        fake_build_step_frame_groups,
    )
    runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    handler = _Handler()
    compiled_plan = CompiledSendPlan.create(
        op_name="op",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="succ",
                effective_rank=0,
                tensor_routes=(),
            ),
        ),
        output_count=0,
        transfer_code_to_name={},
    )
    delivery_tasks = []

    async def submit_delivery(coro):
        task = asyncio.create_task(coro)
        delivery_tasks.append(task)
        return task

    executor = BatchingDataPlaneExecutor()
    result = await executor.send_tensor_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=3,
        successor_handlers=({0: handler},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=0,
        submit_delivery=submit_delivery,
        delivery_schedule_key=(0, 0, 3, "op"),
        profiler_recorder=runtime.recorder,
        batch_submit=lambda batches, enqueue: asyncio.gather(
            *(enqueue(batch) for batch in batches)
        ),
    )

    await asyncio.gather(*result)
    assert delivery_tasks
    await asyncio.gather(*delivery_tasks)
    assert handler.calls == 1
    assert build_threads and build_threads[0] != event_loop_thread
    names = [
        runtime.recorder.metadata.name_for(int(row[4]))
        for row in runtime.recorder.arena.snapshot()
    ]
    assert "data_plane.frame_build" in names
    assert "data_plane.frame_submit" in names
    assert "data_plane.enqueue_batch" in names
    assert "data_plane.batch_rpc" in names
    assert "data_plane.build_frame_groups" not in names
    assert "data_plane.transport_rpc" not in names
    assert "data_plane.send_transport_frame" not in names


@pytest.mark.asyncio
async def test_batching_executor_coalesces_same_target_frames_for_axon_flat_batch(
    monkeypatch,
):
    def fake_build_step_frame_groups(**kwargs):
        step = kwargs["exec_step"]
        payload = np.array([step], dtype=np.float32)
        builder = TransportFrameBuilder(
            step=step,
            sender_spmd_code=7,
            sender_rank=0,
            sender_world_size=1,
        )
        entry_index = builder.add_tensor_transfer_entry(
            transfer_code=1,
            layout_state=0,
            tensor_kind=TensorEntryKind.TENSOR,
            expected_count=1,
            coherence_epoch=step,
            global_shape=(1,),
        )
        builder.add_payload(entry_index, PayloadRole.VALUES, payload, valid_count=1)
        frame = builder.build()
        return [(0, (0, frame.header, frame.payload_segments))]

    monkeypatch.setattr(
        executor_module,
        "build_step_frame_groups",
        fake_build_step_frame_groups,
    )

    class AxonHandle:
        payload_device_class = "cpu"
        completion_mode = "receiver_success"

        def __init__(self):
            self.calls = []

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            self.calls.append((frames, payload_device_class, sender_spmd_code))
            return len(frames)

        async def invoke_control(self, op_name, step, names, values, sender_spmd_code):
            return True

    compiled_plan = CompiledSendPlan.create(
        op_name="op",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=7,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="succ",
                effective_rank=0,
                tensor_routes=(
                    types.SimpleNamespace(
                        transfer_code=1,
                        logical_tuuid=1,
                        target_spmd_id="succ",
                        child_slots=(0,),
                    ),
                ),
            ),
        ),
        output_count=1,
        transfer_code_to_name={1: "value"},
    )
    handle = AxonHandle()
    axon = AxonDataPlane(handle, sender_spmd_code=7)
    executor = BatchingDataPlaneExecutor()
    delivery_tasks = []

    async def submit_delivery(coro):
        task = asyncio.create_task(coro)
        delivery_tasks.append(task)
        return task

    first = await executor.send_tensor_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=4,
        successor_handlers=({0: axon},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=10_000,
        submit_delivery=submit_delivery,
        delivery_schedule_key=(0, 0, 4, "op"),
        batch_submit=lambda batches, enqueue: asyncio.gather(
            *(enqueue(batch) for batch in batches)
        ),
    )
    second = await executor.send_tensor_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=4,
        successor_handlers=({0: axon},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=10_000,
        submit_delivery=submit_delivery,
        delivery_schedule_key=(0, 0, 4, "op"),
        batch_submit=lambda batches, enqueue: asyncio.gather(
            *(enqueue(batch) for batch in batches)
        ),
    )
    await asyncio.gather(*first, *second)

    assert len(handle.calls) == 1
    frames = handle.calls[0][0]
    assert handle.calls[0][1] == "cpu"
    assert handle.calls[0][2] == 7
    assert all(isinstance(frame, TransportFrameEnvelope) for frame in frames)
    assert [int(frame.header.frame_i32[1]) for frame in frames] == [4, 4]
    assert [float(frame.payload_segments[0][0]) for frame in frames] == [4.0, 4.0]


@pytest.mark.asyncio
async def test_executor_submits_one_complete_target_fanout_publication(monkeypatch):
    def fake_build_step_frame_groups(**kwargs):
        header = _empty_header(kwargs["exec_step"])
        return [(0, (target_rank, header, ())) for target_rank in (0, 1)]

    monkeypatch.setattr(
        executor_module,
        "build_step_frame_groups",
        fake_build_step_frame_groups,
    )
    compiled_plan = CompiledSendPlan.create(
        op_name="op",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="sink",
                effective_rank=0,
                tensor_routes=(),
            ),
        ),
        output_count=0,
        transfer_code_to_name={},
    )
    handlers = {target_rank: _Handler() for target_rank in (0, 1)}
    submitted: list[tuple[int, ...]] = []

    async def submit_delivery(coro):
        return asyncio.create_task(coro)

    async def submit_publication(batches, enqueue):
        submitted.append(tuple(batch.target_rank for batch in batches))
        return await asyncio.gather(*(enqueue(batch) for batch in batches))

    result = await BatchingDataPlaneExecutor().send_tensor_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=5,
        successor_handlers=(handlers,),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=0,
        submit_delivery=submit_delivery,
        delivery_schedule_key=(0, 0, 5, "op"),
        batch_submit=submit_publication,
    )

    assert len(result) == 1
    await asyncio.gather(*result)
    assert submitted == [(0, 1)]
    assert [handlers[target_rank].calls for target_rank in (0, 1)] == [1, 1]


@pytest.mark.asyncio
async def test_executor_drops_producer_frame_reference_after_publication(monkeypatch):
    payload_reference: weakref.ReferenceType[np.ndarray] | None = None

    def fake_build_step_frame_groups(**kwargs):
        nonlocal payload_reference
        payload = np.arange(64, dtype=np.int64)
        payload_reference = weakref.ref(payload)
        return [(0, (0, _empty_header(kwargs["exec_step"]), (payload,)))]

    monkeypatch.setattr(
        executor_module,
        "build_step_frame_groups",
        fake_build_step_frame_groups,
    )
    compiled_plan = CompiledSendPlan.create(
        op_name="op",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="sink",
                effective_rank=0,
                tensor_routes=(),
            ),
        ),
        output_count=0,
        transfer_code_to_name={},
    )
    handler = _Handler()

    async def submit_delivery(coro):
        return asyncio.create_task(coro)

    async def submit_publication(batches, enqueue):
        return await asyncio.gather(*(enqueue(batch) for batch in batches))

    result = await BatchingDataPlaneExecutor().send_tensor_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=6,
        successor_handlers=({0: handler},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=0,
        submit_delivery=submit_delivery,
        delivery_schedule_key=(0, 0, 6, "op"),
        batch_submit=submit_publication,
    )
    await asyncio.gather(*result)
    gc.collect()
    assert payload_reference is not None
    assert payload_reference() is None
