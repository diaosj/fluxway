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
from fluxway.profiler.record import TraceArena, TraceCarrier, TraceEventKind
from fluxway.profiler.recorder import TraceRecorder
from fluxway.rpc.protocol import DataPlaneCallable


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

    result = await wrapped.receive_transport_frame("header", np.ones(2))

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


@dataclass
class _AxonResult:
    native_hlc_physical_ns: int
    native_hlc_logical: int


class _RayLikeDataPlane(DataPlaneCallable):
    def __init__(self) -> None:
        self.transport_calls = 0
        self.control_calls = 0

    async def receive_transport_frame(self, header, *payload_iovs):
        self.transport_calls += 1
        return True

    async def receive_control(self, op_name, step, names, values):
        self.control_calls += 1
        return True
