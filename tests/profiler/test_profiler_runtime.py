from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fluxway.profiler import (
    CausalClockViolation,
    MissingAxonNativeHlc,
    MissingTraceMetadata,
    ProfilerBufferExhausted,
    ProfilerConfig,
    ProfilerDiagnostic,
    ProfilerRuntimeFactory,
)
from fluxway.profiler.clock import AxonNativeHlcBridge, RayFluxwayHlcBridge
from fluxway.profiler.hierarchy import (
    TRACE_ROOT_TRACK_NAME,
    TraceTrackHierarchy,
)
from fluxway.profiler.perfetto_writer import PerfettoTraceWriter
from fluxway.profiler.record import (
    PAYLOAD_START_COLUMN,
    TraceArena,
    TraceCarrier,
    TraceEventKind,
)
from fluxway.profiler.recorder import (
    LOCAL_STEP_WAIT_INPUTS,
    WORKER_RUN_LOOP,
    TraceRecorder,
)
from fluxway.profiler.filtering import filter_steady_state_records
from fluxway.rpc.protocol import DataPlaneCallable, TransportFrameEnvelope
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.user_callable_executor import (
    UserCallableExecutor,
    UserCallableSpec,
)
from fluxway.transport.manifest import TransportFrameHeader


def test_disabled_factory_returns_original_objects():
    factory = ProfilerRuntimeFactory.from_config(
        None, process_name="worker/spmd/rank:0"
    )
    executor = object()
    mailbox = object()
    handlers = {0: object()}

    assert factory.enabled is False
    assert factory.data_plane_backend_executor(executor) is executor
    assert factory.local_step_plan_executor(executor) is executor
    assert factory.user_callable_executor(executor) is executor
    assert factory.worker_mailbox(mailbox) is mailbox
    assert factory.wrap_data_plane_handlers("remote", handlers) is handlers


def test_enabled_factory_wraps_without_replacing_delegate_behavior(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    executor = object()
    mailbox = object()
    handlers = {0: _RayLikeDataPlane()}

    assert factory.enabled is True
    assert factory.data_plane_backend_executor(executor) is not executor
    assert factory.worker_mailbox(mailbox) is not mailbox
    wrapped_handlers = factory.wrap_data_plane_handlers("remote", handlers)
    assert wrapped_handlers is not handlers
    assert wrapped_handlers[0] is not handlers[0]


def test_tracing_mailbox_declare_step_forwards_exact_barrier_metadata(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    mailbox = _MailboxLike()
    wrapped = factory.worker_mailbox(mailbox)

    wrapped.declare_step("op", 3, {11: 1}, ["state"])

    assert mailbox.declared == {
        "op_name": "op",
        "step": 3,
        "expected_tensors": {11: 1},
        "expected_states": ["state"],
        "barrier_config": None,
    }


def test_heartbeat_diagnostics_are_opt_in(tmp_path):
    default_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="default"),
        process_name="controller",
    )
    diagnostic_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(
            trace_dir=str(tmp_path),
            run_id="diagnostic",
            diagnostics=frozenset({ProfilerDiagnostic.HEARTBEAT}),
        ),
        process_name="controller",
    )

    assert default_runtime.heartbeat_observer() is None
    assert default_runtime.heartbeat_diagnostics_enabled() is False
    assert diagnostic_runtime.heartbeat_observer() is not None
    assert diagnostic_runtime.heartbeat_diagnostics_enabled() is True


@pytest.mark.asyncio
async def test_user_callable_execute_records_step_function_and_input_names(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    async def fn() -> int:
        return 11

    result = await wrapped.execute(
        UserCallableSpec(
            callable_id="lookup",
            fn=fn,
            is_stateful=False,
            resource_id=1,
            operator_name="lookup",
            operator_step=9,
            debug_tensor_names=("ids", "weights"),
            debug_control_names=("lr",),
            debug_static_names=("training",),
        )
    )
    await runtime.drain()
    runtime.close()
    assert result == 11

    snapshot = factory.recorder.snapshot()
    metadata = snapshot["metadata"]
    execute_name_id = next(
        ident
        for ident, name in metadata.names_by_id.items()
        if name == "user_callable.execute"
    )
    row = next(row for row in snapshot["records"] if int(row[4]) == execute_name_id)
    fields = metadata.payload_fields_for(execute_name_id)
    payload = row[PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)]
    values = dict(zip(fields, payload, strict=True))

    assert metadata.name_for(int(values["function_id"])) == "lookup"
    assert metadata.name_for(int(values["op_id"])) == "lookup"
    assert int(values["step"]) == 9
    assert metadata.name_for(int(values["tensor_names_id"])) == "ids,weights"
    assert metadata.name_for(int(values["control_names_id"])) == "lr"
    assert metadata.name_for(int(values["static_names_id"])) == "training"
    assert metadata.name_for(int(values["lane_id"])) == "inline_async"


def test_profiler_config_rejects_negative_skip_initial_steps(tmp_path):
    with pytest.raises(ValueError, match="skip_initial_steps"):
        ProfilerConfig(
            trace_dir=str(tmp_path),
            run_id="bad",
            skip_initial_steps=-1,
        )


@pytest.mark.asyncio
async def test_trace_recorder_skips_step_level_records_below_threshold():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
        skip_initial_steps=2,
    )

    recorder.worker_scheduler_instant(spec=WORKER_RUN_LOOP)
    async with recorder.local_step_node(
        spec=LOCAL_STEP_WAIT_INPUTS,
        op_name="op",
        step=1,
    ):
        pass
    async with recorder.local_step_node(
        spec=LOCAL_STEP_WAIT_INPUTS,
        op_name="op",
        step=2,
    ):
        pass

    names = [
        recorder.metadata.name_for(int(row[4])) for row in recorder.arena.snapshot()
    ]
    assert names.count("worker.run_loop") == 1
    assert names.count("local_step.wait_inputs") == 1


@pytest.mark.asyncio
async def test_steady_state_filter_removes_lifecycle_records():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="driver/spmd/driver",
    )
    async with recorder.driver_initialize(worker_count=1):
        pass
    async with recorder.driver_drain(target_step=3, worker_count=1):
        pass

    filtered = filter_steady_state_records(recorder.arena.snapshot(), recorder.metadata)
    names = [recorder.metadata.name_for(int(row[4])) for row in filtered]

    assert names == ["driver.drain"]


def test_trace_arena_fixed_capacity_exhaustion():
    arena = TraceArena(buffer_bytes=16 * np.dtype(np.int64).itemsize)
    arena.append(
        kind=TraceEventKind.INSTANT,
        ts_ns=1,
        dur_ns=0,
        track_id=1,
        name_id=1,
        payload=(7,),
    )
    with pytest.raises(ProfilerBufferExhausted):
        arena.append(
            kind=TraceEventKind.INSTANT,
            ts_ns=2,
            dur_ns=0,
            track_id=1,
            name_id=1,
        )


def test_profiler_buffer_exhausted_carries_structured_context():
    arena = TraceArena(buffer_bytes=16 * np.dtype(np.int64).itemsize)
    arena.append(
        kind=TraceEventKind.INSTANT,
        ts_ns=1,
        dur_ns=0,
        track_id=1,
        name_id=1,
    )

    with pytest.raises(ProfilerBufferExhausted) as exc_info:
        arena.append(
            kind=TraceEventKind.INSTANT,
            ts_ns=2,
            dur_ns=0,
            track_id=1,
            name_id=1,
        )

    error = exc_info.value
    assert error.code == "profiler_buffer_exhausted"
    assert error.context == {"capacity": 1, "attempted_index": 1}
    assert error.as_dict()["code"] == "profiler_buffer_exhausted"
    assert "capacity=1" in str(error)


def test_causal_clock_violation_carries_sender_receiver_hlc():
    error = CausalClockViolation(
        sender_hlc_physical_ns=10,
        sender_hlc_logical=2,
        receiver_hlc_physical_ns=10,
        receiver_hlc_logical=2,
    )

    assert error.code == "causal_clock_violation"
    assert error.context == {
        "sender_hlc_physical_ns": 10,
        "sender_hlc_logical": 2,
        "receiver_hlc_physical_ns": 10,
        "receiver_hlc_logical": 2,
    }


def test_ray_hlc_bridge_merges_sender_carrier():
    carrier = TraceCarrier.root("run").child("send")
    received = RayFluxwayHlcBridge().receive(carrier)

    assert received.receiver_hlc_physical_ns >= carrier.sender_hlc_physical_ns
    assert (
        received.receiver_hlc_physical_ns,
        received.receiver_hlc_logical,
    ) > (carrier.sender_hlc_physical_ns, carrier.sender_hlc_logical)


def test_axon_native_hlc_bridge_uses_native_metadata():
    carrier = TraceCarrier.root("run").child("send")
    result = _AxonResult(native_hlc_physical_ns=123, native_hlc_logical=4)
    received = AxonNativeHlcBridge().receive_result(carrier, result)

    assert received.receiver_hlc_physical_ns == 123
    assert received.receiver_hlc_logical == 4


def test_axon_native_hlc_bridge_rejects_missing_metadata():
    carrier = TraceCarrier.root("run").child("send")
    with pytest.raises(MissingAxonNativeHlc) as exc_info:
        AxonNativeHlcBridge().receive_result(carrier, object())
    assert exc_info.value.code == "missing_axon_native_hlc"
    assert exc_info.value.context == {
        "result_type": "object",
        "has_metadata": False,
    }


@pytest.mark.asyncio
async def test_tracing_data_plane_callable_delegates_and_records(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    delegate = _RayLikeDataPlane()
    wrapped = factory.wrap_data_plane_handlers("remote", {3: delegate})[3]
    header = TransportFrameHeader(
        frame_i32=np.array([1, 4, 0, 0, 1, 0, 0, 0], dtype=np.int32),
        edge_i32=np.zeros((0, 11), dtype=np.int32),
        payload_i32=np.zeros((0, 8), dtype=np.int32),
        shape_i32=np.zeros((0,), dtype=np.int32),
        global_shape_i64=np.zeros((0,), dtype=np.int64),
        compact_i64=np.zeros((0,), dtype=np.int64),
    )

    result = await wrapped.receive_transport_frames(
        (TransportFrameEnvelope(header=header, payload_segments=(np.ones(2),)),)
    )

    assert result is True
    assert delegate.transport_calls == 1
    assert factory.recorder.arena.size >= 2


def test_recorder_metadata_contains_semantic_payload_schema():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    recorder.data_plane_hlc_receive(
        carrier=recorder.root_carrier,
        receiver_hlc_physical_ns=123,
        receiver_hlc_logical=4,
        clock_source=1,
    )

    metadata = recorder.metadata
    event_id = recorder.intern("data_plane.hlc_receive")

    assert metadata.name_for(event_id) == "data_plane.hlc_receive"
    assert metadata.payload_fields_for(event_id) == (
        "receiver_hlc_physical_ns",
        "receiver_hlc_logical",
        "clock_source",
    )
    with pytest.raises(MissingTraceMetadata):
        metadata.name_for(1)


def test_trace_hierarchy_models_fluxway_abstractions():
    worker = TraceTrackHierarchy.from_track_name(
        "worker/InputProcessor/rank:0/transport/send"
    )
    driver = TraceTrackHierarchy.from_track_name(
        "driver/InputProcessor/driver/checkpoint"
    )
    controller = TraceTrackHierarchy.from_track_name("controller/controller/fsm")

    assert worker.groups == (
        "SPMDs",
        "InputProcessor",
        "workers",
        "rank:0",
        "transport",
    )
    assert worker.leaf_name == "send"
    assert driver.groups == ("SPMDs", "InputProcessor", "driver")
    assert driver.leaf_name == "checkpoint"
    assert controller.groups == ("Control Plane", "controller")
    assert controller.leaf_name == "fsm"
    assert worker.to_layout(leaf_uuid=7).nodes[0].name == TRACE_ROOT_TRACK_NAME
    with pytest.raises(ValueError):
        TraceTrackHierarchy.from_track_name("unknown/path")


def test_perfetto_writer_dependency_contract(tmp_path):
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    recorder.data_plane_hlc_receive(
        carrier=recorder.root_carrier,
        receiver_hlc_physical_ns=123,
        receiver_hlc_logical=4,
        clock_source=1,
    )
    if importlib.util.find_spec("perfetto") is None:
        with pytest.raises(ImportError, match="perfetto==0.57.1"):
            PerfettoTraceWriter()
        return

    path = Path(tmp_path) / "trace.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
    )
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.asyncio
async def test_perfetto_writer_decodes_interned_debug_annotations(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    # pyrefly: ignore [missing-import]
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    async with recorder.user_callable_execute(
        callable_id="lookup_callable",
        function_name="lookup",
        op_name="lookup",
        step=4,
        tensor_names="ids,weights",
        control_names="lr",
        static_names="training",
        lane="inline_async",
    ):
        pass

    path = Path(tmp_path) / "trace.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
    )

    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())
    annotations = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.name != "user_callable.execute":
            continue
        for annotation in event.debug_annotations:
            if annotation.HasField("string_value"):
                annotations[annotation.name] = annotation.string_value
            elif annotation.HasField("int_value"):
                annotations[annotation.name] = int(annotation.int_value)
        break

    assert annotations["callable_id"] == "lookup_callable"
    assert annotations["function_id"] == "lookup"
    assert annotations["op_id"] == "lookup"
    assert annotations["step"] == 4
    assert annotations["tensor_names_id"] == "ids,weights"
    assert annotations["control_names_id"] == "lr"
    assert annotations["static_names_id"] == "training"
    assert annotations["lane_id"] == "inline_async"


@dataclass
class _AxonResult:
    native_hlc_physical_ns: int
    native_hlc_logical: int


class _RayLikeDataPlane(DataPlaneCallable):
    def __init__(self) -> None:
        self.transport_calls = 0
        self.control_calls = 0

    async def receive_transport_frames(self, frames):
        self.transport_calls += 1
        return True

    async def receive_control(self, op_name, step, names, values):
        self.control_calls += 1
        return True


class _MailboxLike:
    def __init__(self) -> None:
        self.declared = None

    def declare_step(
        self,
        op_name,
        step,
        expected_tensors,
        expected_states,
        barrier_config=None,
    ):
        self.declared = {
            "op_name": op_name,
            "step": step,
            "expected_tensors": expected_tensors,
            "expected_states": expected_states,
            "barrier_config": barrier_config,
        }
