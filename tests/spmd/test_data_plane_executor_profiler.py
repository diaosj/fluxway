import asyncio
import threading

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
    build_and_send_step_frames,
)
from fluxway.transport.manifest import TransportFrameHeader


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
        frame_i32=np.array([1, step, 0, 0, 1, 0, 0, 0], dtype=np.int32),
        edge_i32=np.zeros((0, 11), dtype=np.int32),
        payload_i32=np.zeros((0, 8), dtype=np.int32),
        shape_i32=np.zeros((0,), dtype=np.int32),
        global_shape_i64=np.zeros((0,), dtype=np.int64),
        compact_i64=np.zeros((0,), dtype=np.int64),
    )


def _compiled_plan() -> CompiledSendPlan:
    return CompiledSendPlan(
        op_name="op",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(),
        output_count=1,
        tuuid_to_name=None,
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
        )
    )
    await asyncio.sleep(0.01)
    handler.release.set()

    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)
    assert handler.calls == [1, 1]


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
                [1, kwargs["exec_step"], 0, 0, 1, 0, 0, 0], dtype=np.int32
            ),
            edge_i32=np.zeros((0, 11), dtype=np.int32),
            payload_i32=np.zeros((0, 8), dtype=np.int32),
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
    compiled_plan = CompiledSendPlan(
        op_name="op",
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
        tuuid_to_name=None,
    )
    delivery_tasks = []

    async def submit_delivery(coro):
        task = asyncio.create_task(coro)
        delivery_tasks.append(task)
        return task

    result = await build_and_send_step_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_tuuid_cache={},
        exec_step=3,
        successor_handlers=({0: handler},),
        source_spmd_id="source",
        batch_max_frames=8,
        submit_delivery=submit_delivery,
        profiler_recorder=runtime.recorder,
    )

    assert result == delivery_tasks
    await asyncio.gather(*result)
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
        frame_i32 = np.array([1, step, 7, 0, 1, 0, 1, 0], dtype=np.int32)
        payload_i32 = np.zeros((1, 8), dtype=np.int32)
        payload_i32[0] = np.array([0, 6, 1, 0, 1, 1, 0, 0], dtype=np.int32)
        header = TransportFrameHeader(
            frame_i32=frame_i32,
            edge_i32=np.zeros((0, 11), dtype=np.int32),
            payload_i32=payload_i32,
            shape_i32=np.array([1], dtype=np.int32),
            global_shape_i64=np.zeros((0,), dtype=np.int64),
            compact_i64=np.zeros((0,), dtype=np.int64),
        )
        return [(0, (0, header, (np.array([step], dtype=np.float32),)))]

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

    compiled_plan = CompiledSendPlan(
        op_name="op",
        rank=0,
        sender_spmd_code=7,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="succ",
                effective_rank=0,
                tensor_routes=(),
            ),
        ),
        output_count=1,
        tuuid_to_name=None,
    )
    handle = AxonHandle()
    axon = AxonDataPlane(handle, sender_spmd_code=7)
    executor = BatchingDataPlaneExecutor()
    delivery_tasks = []

    async def submit_delivery(coro):
        task = asyncio.create_task(coro)
        delivery_tasks.append(task)
        return task

    first = await executor.send_step_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_tuuid_cache={},
        exec_step=4,
        successor_handlers=({0: axon},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=10_000,
        submit_delivery=submit_delivery,
    )
    second = await executor.send_step_frames(
        compiled_plan=compiled_plan,
        flat_results=[],
        step_tuuid_cache={},
        exec_step=5,
        successor_handlers=({0: axon},),
        source_spmd_id="source",
        batch_max_frames=8,
        batch_flush_us=10_000,
        submit_delivery=submit_delivery,
    )

    assert first + second == delivery_tasks
    await asyncio.gather(*delivery_tasks)

    assert len(handle.calls) == 1
    frames, payload_device_class, sender_spmd_code = handle.calls[0]
    assert payload_device_class == "cpu"
    assert sender_spmd_code == 7
    assert isinstance(frames, tuple)
    assert len(frames) == 2
    assert all(isinstance(frame, TransportFrameEnvelope) for frame in frames)
    assert [int(frame.header.frame_i32[1]) for frame in frames] == [4, 5]
    assert [float(frame.payload_segments[0][0]) for frame in frames] == [4.0, 5.0]
