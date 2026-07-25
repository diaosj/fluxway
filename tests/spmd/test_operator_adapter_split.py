import inspect

import pytest

from fluxway.common import asyncio_runtime
from fluxway.common.compiled_graph import (
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import SPMDConfig
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.user_callable_executor import UserCallableExecutor
from fluxway.spmd.worker.worker import SPMDWorker
from tests.compiled_graph_fixtures import compiled_spmd_spec


class _Trainer:
    pass


def _configured_worker(*, coroutine_function: bool, manager):
    operator = OperatorSpec(
        name="run",
        owner=_Trainer,
        requires_worker_process=True,
    )
    mesh = DeviceMesh(("data",), (1,))
    plan = compile_graph(
        name="adapter-split",
        spmds=(
            compiled_spmd_spec(
                spmd_id="test",
                mesh=mesh,
                operators={"run": operator},
            ),
        ),
        dependencies=(),
    )
    manager.get_operator_spec = lambda _: operator
    manager.operator_is_coroutine_function = lambda _: coroutine_function
    raw_worker = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(raw_worker)
    worker.spmd_id = "test"
    worker.rank = 0
    worker.config = SPMDConfig(
        spmd_id="test",
        mesh=mesh,
        operator_specs={"run": operator},
        graph_view=plan.spmd("test"),
    )
    worker._op_manager = manager
    return worker, raw_worker


@pytest.mark.asyncio
async def test_regular_operator_adapter_avoids_asyncio_run(monkeypatch):
    calls = []

    class Manager:
        def execute(self, op_name, step, *args, **kwargs):
            calls.append((op_name, step, args, kwargs))
            return 7

        async def execute_async(self, *args, **kwargs):
            raise AssertionError("regular operator used coroutine execution")

    worker, raw_worker = _configured_worker(
        coroutine_function=False,
        manager=Manager(),
    )
    spec = raw_worker._operator_user_callable_spec(worker, "run", 3, {"x": 1})
    assert not inspect.iscoroutinefunction(spec.fn)

    def forbidden_run(coroutine):
        coroutine.close()
        raise AssertionError("regular operator created an asyncio event loop")

    monkeypatch.setattr(asyncio_runtime, "run", forbidden_run)
    runtime = WorkerProcessCallRuntime(max_workers=1)
    try:
        executor = UserCallableExecutor(worker_process_runtime=runtime)
        assert await executor.execute(spec) == 7
    finally:
        runtime.close()

    assert calls == [("run", 3, (), {"x": 1})]


@pytest.mark.asyncio
async def test_coroutine_operator_adapter_uses_async_entrypoint():
    calls = []

    class Manager:
        def execute(self, *args, **kwargs):
            raise AssertionError("coroutine operator used regular execution")

        async def execute_async(self, op_name, step, *args, **kwargs):
            calls.append((op_name, step, args, kwargs))
            return 11

    worker, raw_worker = _configured_worker(
        coroutine_function=True,
        manager=Manager(),
    )
    spec = raw_worker._operator_user_callable_spec(worker, "run", 4, 2)
    assert inspect.iscoroutinefunction(spec.fn)

    runtime = WorkerProcessCallRuntime(max_workers=1)
    try:
        executor = UserCallableExecutor(worker_process_runtime=runtime)
        assert await executor.execute(spec) == 11
    finally:
        runtime.close()

    assert calls == [("run", 4, (2,), {})]
