import asyncio
from typing import Any
import contextvars
import threading
import time
from types import SimpleNamespace

import pytest

import fluxway as fw
from fluxway.common.compiled_graph import (
    SPMDGraphView,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import SPMDConfig
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.profiler.record import PAYLOAD_START_COLUMN
from fluxway.spmd.worker.user_callable_executor import (
    USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    SPMDObjectScope,
    UserCallableExecutor,
    UserCallableLane,
    UserCallableSpec,
)
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.worker import SPMDWorker
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _spmd_graph_view(
    operator: OperatorSpec,
    *,
    spmd_id: str = "test",
) -> SPMDGraphView:
    return compile_graph(
        name="user-callable-context",
        spmds=(
            compiled_spmd_spec(
                spmd_id=spmd_id,
                mesh=DeviceMesh(("data",), (1,)),
                operators={operator.name: operator},
            ),
        ),
        dependencies=(),
    ).spmd(spmd_id)


def _trace_fields(op_id: str) -> dict[str, Any]:
    graph_view = _spmd_graph_view(OperatorSpec(name=op_id), spmd_id="trace")
    return {
        "graph_id": graph_view.graph_id,
        "compiled_operator": graph_view.operator(op_id),
    }


def test_worker_callable_spec_uses_exact_compiled_operator():
    operator = OperatorSpec(name="run")
    graph_view = _spmd_graph_view(operator)
    config = SPMDConfig(
        spmd_id="test",
        graph_view=graph_view,
        operator_specs={"run": operator},
    )
    raw_worker = SPMDWorker.__ray_metadata__.modified_class
    worker = object.__new__(raw_worker)
    worker.spmd_id = "test"
    worker.rank = 0
    worker.config = config
    worker._op_manager = SimpleNamespace(
        get_operator_spec=lambda _: operator,
        operator_is_coroutine_function=lambda _: False,
        execute=lambda *args, **kwargs: None,
    )

    spec = raw_worker._operator_user_callable_spec(worker, "run", 3, {})

    assert spec.graph_id == graph_view.graph_id
    assert spec.compiled_operator is graph_view.operator("run")
    assert spec.callable_id == "run"
    assert spec.operator_step == 3


def _make_executor(*, max_workers: int = 1) -> UserCallableExecutor:
    return UserCallableExecutor(
        worker_process_runtime=WorkerProcessCallRuntime(max_workers=max_workers),
    )


def _recorded_event_names(factory: ProfilerRuntimeFactory) -> list[str]:
    snapshot = factory.recorder.snapshot()
    metadata = snapshot["metadata"]
    return [metadata.name_for(int(row[4])) for row in snapshot["records"]]


def _recorded_execution_modes(factory: ProfilerRuntimeFactory) -> list[str]:
    snapshot = factory.recorder.snapshot()
    metadata = snapshot["metadata"]
    name_id = next(
        ident
        for ident, name in metadata.names_by_id.items()
        if name == "user_callable.execute"
    )
    fields = metadata.payload_fields_for(name_id)
    mode_index = fields.index("execution_mode")
    return [
        metadata.name_for(int(row[PAYLOAD_START_COLUMN + mode_index]))
        for row in snapshot["records"]
        if int(row[4]) == name_id
    ]


def test_process_affine_stateful_coroutine_uses_coroutine_async_lane():
    class Trainer:
        async def run(self) -> None:
            return None

    operator = OperatorSpec(
        name="run",
        owner=Trainer,
        requires_worker_process=True,
    )
    graph_view = _spmd_graph_view(operator)
    config = SPMDConfig(
        spmd_id="test",
        graph_view=graph_view,
        operator_specs={"run": operator},
    )
    raw_worker = SPMDWorker.__ray_metadata__.modified_class
    worker = object.__new__(raw_worker)
    worker.spmd_id = "test"
    worker.rank = 0
    worker.config = config
    worker._op_manager = SimpleNamespace(
        get_operator_spec=lambda _: operator,
        operator_is_coroutine_function=lambda _: True,
    )
    spec = raw_worker._operator_user_callable_spec(worker, "run", 0, {})
    executor = _make_executor()

    assert "is_stateful" not in UserCallableSpec.__dataclass_fields__
    assert "process_affine" not in UserCallableSpec.__dataclass_fields__
    assert executor.select(spec).lane is UserCallableLane.COROUTINE_ASYNC
    executor._worker_process_runtime.close()


def test_regular_function_uses_worker_process_lane_despite_coroutine_adapter():
    async def worker_generated_adapter() -> None:
        return None

    spec = UserCallableSpec(
        callable_id="step",
        fn=worker_generated_adapter,
        resource_id=0,
        operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        operator_is_coroutine_function=False,
    )
    executor = _make_executor()

    assert executor.select(spec).lane is UserCallableLane.WORKER_PROCESS


def test_coroutine_function_uses_coroutine_async_lane():
    async def coroutine_function() -> None:
        return None

    spec = UserCallableSpec(
        callable_id="step",
        fn=coroutine_function,
        resource_id=0,
        operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        operator_is_coroutine_function=True,
    )
    executor = _make_executor()

    assert executor.select(spec).lane is UserCallableLane.COROUTINE_ASYNC


def test_regular_function_pipeline_depth_sets_worker_process_capacity():
    raw_worker = SPMDWorker.__ray_metadata__.modified_class
    worker = object.__new__(raw_worker)
    worker._operator_specs = {
        "regular": SimpleNamespace(
            name="regular",
            owner_class_qualname="",
            runtime_policy=SimpleNamespace(pipeline_depth=4),
        )
    }

    assert raw_worker._worker_process_call_max_workers(worker) == 4


@pytest.mark.asyncio
async def test_user_callable_executor_shutdown_waits_for_worker_process_calls():
    executor = _make_executor()
    started = threading.Event()
    release = threading.Event()

    def call() -> None:
        started.set()
        release.wait(timeout=1)

    call_task = asyncio.create_task(
        executor.execute(
            UserCallableSpec(
                callable_id="step",
                fn=call,
                resource_id=0,
                operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
                operator_is_coroutine_function=False,
            )
        )
    )
    while not started.is_set():
        await asyncio.sleep(0)
    shutdown_task = asyncio.create_task(executor.shutdown())
    await asyncio.sleep(0)

    assert not shutdown_task.done()

    release.set()
    await asyncio.gather(call_task, shutdown_task)
    executor._worker_process_runtime.close()


@pytest.mark.asyncio
async def test_worker_process_failure_propagates_to_caller():
    executor = _make_executor()

    def fail() -> None:
        raise RuntimeError("worker call failed")

    task = asyncio.create_task(
        executor.execute(
            UserCallableSpec(
                callable_id="step",
                fn=fail,
                resource_id=0,
                operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
                operator_is_coroutine_function=False,
            )
        )
    )
    with pytest.raises(RuntimeError, match="worker call failed"):
        await task

    await executor.shutdown()
    executor._worker_process_runtime.close()


@pytest.mark.asyncio
async def test_worker_process_runtime_runs_regular_function_without_blocking_event_loop():
    runtime = WorkerProcessCallRuntime(max_workers=1)
    stop = asyncio.Event()
    gaps: list[float] = []

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    def long_call() -> str:
        time.sleep(0.15)
        return "done"

    heartbeat_task = asyncio.create_task(heartbeat())
    result = await runtime.submit(
        long_call,
        callable_id="call",
        resource_id=1,
        execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    )
    stop.set()
    await heartbeat_task
    runtime.close()

    assert result == "done"
    assert gaps
    assert max(gaps) < 0.08


@pytest.mark.asyncio
async def test_worker_process_runtime_completes_awaitable_inside_thread():
    runtime = WorkerProcessCallRuntime(max_workers=1)

    async def inner() -> int:
        await asyncio.sleep(0)
        return 7

    result = await runtime.submit(
        lambda: inner(),
        callable_id="call",
        resource_id=1,
        execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    )
    runtime.close()

    assert result == 7


@pytest.mark.asyncio
async def test_worker_process_runtime_preserves_contextvars():
    runtime = WorkerProcessCallRuntime(max_workers=1)
    token_var: contextvars.ContextVar[str] = contextvars.ContextVar("token")
    token = token_var.set("rank-context")
    try:
        result = await runtime.submit(
            token_var.get,
            callable_id="call",
            resource_id=1,
            execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        )
    finally:
        token_var.reset(token)
        runtime.close()

    assert result == "rank-context"


@pytest.mark.asyncio
async def test_worker_process_runtime_propagates_exception():
    runtime = WorkerProcessCallRuntime(max_workers=1)

    def fail() -> None:
        raise RuntimeError("call failed")

    with pytest.raises(RuntimeError, match="call failed"):
        await runtime.submit(
            fail,
            callable_id="call",
            resource_id=1,
            execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        )
    runtime.close()


@pytest.mark.asyncio
async def test_worker_process_runtime_parallel_when_thread_safe_resource_allows_it():
    runtime = WorkerProcessCallRuntime(max_workers=2)
    release = asyncio.Event()
    started = 0

    def blocking_call() -> int:
        nonlocal started
        started += 1
        while not release.is_set():
            time.sleep(0.001)
        return started

    first = asyncio.create_task(
        runtime.submit(
            blocking_call,
            callable_id="call",
            resource_id=-1,
            execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        )
    )
    second = asyncio.create_task(
        runtime.submit(
            blocking_call,
            callable_id="call",
            resource_id=-1,
            execution_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
        )
    )
    deadline = time.perf_counter() + 1.0
    while started < 2 and time.perf_counter() < deadline:
        await asyncio.sleep(0.005)
    release.set()
    await asyncio.gather(first, second)
    runtime.close()

    assert started == 2


@pytest.mark.asyncio
async def test_process_affine_worker_process_runtime_isolates_operator_lanes():
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    release = threading.Event()
    started: list[str] = []

    def blocking(name: str):
        def call() -> str:
            started.append(name)
            while not release.is_set():
                time.sleep(0.001)
            return name

        return call

    first = asyncio.create_task(
        executor.execute(
            UserCallableSpec(
                callable_id="train_core",
                fn=blocking("compute"),
                resource_id=-1,
                operator_lane="compute",
                operator_is_coroutine_function=False,
            )
        )
    )
    second = asyncio.create_task(
        executor.execute(
            UserCallableSpec(
                callable_id="prepare_dense_inputs",
                fn=blocking("data"),
                resource_id=-1,
                operator_lane="data",
                operator_is_coroutine_function=False,
            )
        )
    )

    try:
        deadline = time.perf_counter() + 1.0
        while len(started) < 2 and time.perf_counter() < deadline:
            await asyncio.sleep(0.001)
        release.set()
        assert sorted(started) == ["compute", "data"]
        assert sorted(runtime.execution_lane_names()) == [
            USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            "compute",
            "data",
        ]
        assert sorted(await asyncio.gather(first, second)) == ["compute", "data"]
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        runtime.close()


@pytest.mark.asyncio
async def test_user_profiler_context_propagates_to_worker_local_lanes(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    async def coroutine_async() -> int:
        with fw.profiler.scope("coroutine_async", lane="coroutine_async"):
            return 1

    def regular_function() -> int:
        with fw.profiler.scope("regular_function", lane="worker_process"):
            return 2

    def worker_process() -> int:
        with fw.profiler.scope("worker_process", lane="worker_process"):
            return 3

    await wrapped.execute(
        UserCallableSpec(
            callable_id="coroutine_async",
            fn=coroutine_async,
            resource_id=1,
            operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            operator_is_coroutine_function=True,
            operator_step=1,
            object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            **_trace_fields("coroutine_async"),
        )
    )
    await wrapped.execute(
        UserCallableSpec(
            callable_id="regular_function",
            fn=regular_function,
            resource_id=1,
            operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            operator_is_coroutine_function=False,
            operator_step=1,
            object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            **_trace_fields("regular_function"),
        )
    )
    await wrapped.execute(
        UserCallableSpec(
            callable_id="worker_process",
            fn=worker_process,
            resource_id=1,
            operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            operator_is_coroutine_function=False,
            operator_step=1,
            object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            **_trace_fields("worker_process"),
        )
    )

    runtime.close()

    names = _recorded_event_names(factory)
    assert "user.coroutine_async" in names
    assert "user.regular_function" in names
    assert "user.worker_process" in names


@pytest.mark.asyncio
async def test_profiler_selects_lane_once(monkeypatch, tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)
    original_select = executor.select
    selection_count = 0

    def count_selection(spec):
        nonlocal selection_count
        selection_count += 1
        return original_select(spec)

    monkeypatch.setattr(executor, "select", count_selection)

    await wrapped.execute(
        UserCallableSpec(
            callable_id="regular_function",
            fn=lambda: 4,
            resource_id=1,
            operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            operator_is_coroutine_function=False,
            operator_step=1,
            object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            **_trace_fields("regular_function"),
        )
    )

    runtime.close()
    assert selection_count == 1


@pytest.mark.asyncio
async def test_profiler_preserves_worker_process_lane_isolation(tmp_path):
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    runtime = WorkerProcessCallRuntime(max_workers=1, profiler_runtime=factory)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    wrapped = factory.user_callable_executor(executor)
    release = threading.Event()
    started: list[str] = []

    def blocking(name: str):
        def call() -> str:
            started.append(name)
            release.wait(timeout=2)
            with fw.profiler.scope(name, lane=name):
                return name

        return call

    tasks = [
        asyncio.create_task(
            wrapped.execute(
                UserCallableSpec(
                    callable_id=name,
                    fn=blocking(name),
                    resource_id=-1,
                    operator_lane=lane,
                    operator_is_coroutine_function=False,
                    operator_step=1,
                    object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
                    **_trace_fields(name),
                )
            )
        )
        for name, lane in (("compute", "compute"), ("data", "data"))
    ]

    try:
        deadline = time.perf_counter() + 0.5
        while len(started) < 2 and time.perf_counter() < deadline:
            await asyncio.sleep(0.001)
        assert sorted(started) == ["compute", "data"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        runtime.close()

    names = _recorded_event_names(factory)
    assert names.count("worker_process_call.thread_run") == 2
    assert "user.compute" in names
    assert "user.data" in names
    assert _recorded_execution_modes(factory) == ["worker_process", "worker_process"]
