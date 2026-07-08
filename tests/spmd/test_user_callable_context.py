import inspect
import asyncio
import contextvars
import time

import pytest

import fluxway as fw
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.spmd.worker.user_callable_executor import (
    UserCallableExecutor,
    UserCallableLane,
    UserCallableSpec,
    _RayOperatorExecutionActor,
)
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime


def _make_executor(*, max_workers: int = 1) -> UserCallableExecutor:
    return UserCallableExecutor(
        worker_process_runtime=WorkerProcessCallRuntime(max_workers=max_workers),
    )


def _recorded_event_names(factory: ProfilerRuntimeFactory) -> list[str]:
    snapshot = factory.recorder.snapshot()
    metadata = snapshot["metadata"]
    return [metadata.name_for(int(row[4])) for row in snapshot["records"]]


def test_ray_operator_execution_actor_accepts_rank0_address_context():
    assert "rank0_address" in UserCallableSpec.__dataclass_fields__

    cls = getattr(_RayOperatorExecutionActor, "__ray_metadata__", None)
    actor_cls = cls.modified_class if cls else _RayOperatorExecutionActor  # type: ignore
    params = inspect.signature(actor_cls.__init__).parameters
    assert "rank0_address" in params


def test_process_affine_stateful_operator_uses_worker_process_lane():
    assert "process_affine" in UserCallableSpec.__dataclass_fields__

    def fn():
        return None

    spec = UserCallableSpec(
        callable_id="step",
        fn=fn,
        is_stateful=True,
        resource_id=0,
        operator_name="step",
        process_affine=True,
    )

    executor = _make_executor()
    assert executor._lane_for(spec) is UserCallableLane.WORKER_PROCESS


def test_stateful_ops_with_same_owner_share_ray_actor_key():
    class Table:
        def lookup(self):
            return None

        def update(self):
            return None

    executor = _make_executor()
    owner = (Table.__module__, Table.__qualname__, Table)
    lookup = UserCallableSpec(
        callable_id="lookup",
        fn=lambda: None,
        is_stateful=True,
        resource_id=0,
        operator_name="lookup",
        operator_spmd_id="Embedding",
        operator_owner_classes=(owner,),
    )
    update = UserCallableSpec(
        callable_id="update",
        fn=lambda: None,
        is_stateful=True,
        resource_id=0,
        operator_name="update",
        operator_spmd_id="Embedding",
        operator_owner_classes=(owner,),
    )

    assert executor._ray_actor_key(lookup) == executor._ray_actor_key(update)


@pytest.mark.asyncio
async def test_worker_process_runtime_runs_sync_call_without_blocking_event_loop():
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
    result = await runtime.submit(long_call, callable_id="call", resource_id=1)
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

    result = await runtime.submit(lambda: inner(), callable_id="call", resource_id=1)
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
        await runtime.submit(fail, callable_id="call", resource_id=1)
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
        runtime.submit(blocking_call, callable_id="call", resource_id=-1)
    )
    second = asyncio.create_task(
        runtime.submit(blocking_call, callable_id="call", resource_id=-1)
    )
    deadline = time.perf_counter() + 1.0
    while started < 2 and time.perf_counter() < deadline:
        await asyncio.sleep(0.005)
    release.set()
    await asyncio.gather(first, second)
    runtime.close()

    assert started == 2


@pytest.mark.asyncio
async def test_user_profiler_context_propagates_to_worker_local_lanes(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    async def inline_async() -> int:
        with fw.profiler.scope("inline_async", lane="inline_async"):
            return 1

    def inline_sync() -> int:
        with fw.profiler.scope("inline_sync", lane="inline_sync"):
            return 2

    def worker_process() -> int:
        with fw.profiler.scope("worker_process", lane="worker_process"):
            return 3

    await wrapped.execute(
        UserCallableSpec(
            callable_id="inline_async",
            fn=inline_async,
            is_stateful=False,
            resource_id=1,
            operator_name="inline_async",
            operator_step=1,
            operator_rank=0,
        )
    )
    await wrapped.execute(
        UserCallableSpec(
            callable_id="inline_sync",
            fn=inline_sync,
            is_stateful=False,
            resource_id=1,
            operator_name="inline_sync",
            operator_step=1,
            operator_rank=0,
        )
    )
    await wrapped.execute(
        UserCallableSpec(
            callable_id="worker_process",
            fn=worker_process,
            is_stateful=False,
            resource_id=1,
            operator_name="worker_process",
            operator_step=1,
            operator_rank=0,
            process_affine=True,
        )
    )

    runtime.close()

    names = _recorded_event_names(factory)
    assert "user.inline_async" in names
    assert "user.inline_sync" in names
    assert "user.worker_process" in names


@pytest.mark.asyncio
async def test_user_profiler_context_propagates_to_local_free_thread_lane(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    executor._free_threading_enabled = True
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    def local_free_thread() -> int:
        with fw.profiler.scope("local_free_thread", lane="local_free_thread"):
            return 4

    await wrapped.execute(
        UserCallableSpec(
            callable_id="local_free_thread",
            fn=local_free_thread,
            is_stateful=False,
            resource_id=1,
            operator_name="local_free_thread",
            operator_step=1,
            operator_rank=0,
        )
    )

    runtime.close()

    assert "user.local_free_thread" in _recorded_event_names(factory)
