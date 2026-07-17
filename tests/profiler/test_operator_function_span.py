import asyncio

from fluxway.profiler.record import RECORD_WIDTH, TraceArena
from fluxway.profiler.recorder import TraceRecorder
from fluxway.profiler.user_api import activate_user_profiler
from fluxway.spmd.worker.operator_manager.dispatcher import StandardDispatcher
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager


def _bare_manager(name: str, fn, *, coroutine_function: bool) -> OperatorManager:
    manager = object.__new__(OperatorManager)
    manager._operators = {name: fn}
    manager._operator_is_coroutine_function = {name: coroutine_function}
    manager._dispatchers = {name: StandardDispatcher()}
    manager._static_args = {}
    manager._rank = 0
    manager._world_size = 1
    manager._local_rank = 0
    manager._local_world_size = 1
    manager._node_rank = 0
    manager._num_nodes = 1
    manager.is_source_operator = lambda _: False
    return manager


def _function_spans(recorder: TraceRecorder):
    snapshot = recorder.snapshot()
    metadata = snapshot["metadata"]
    return [
        row
        for row in snapshot["records"]
        if metadata.name_for(int(row[4])) == "user_callable.function_run"
    ]


def test_regular_operator_records_function_span() -> None:
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=16 * RECORD_WIDTH * 8),
        run_id="function-span",
        process_name="worker/test/rank:0",
    )
    manager = _bare_manager(
        "regular",
        lambda value: value + 1,
        coroutine_function=False,
    )
    with activate_user_profiler(
        recorder=recorder,
        op_name="regular",
        step=3,
        rank=0,
        execution_mode="worker_process",
    ):
        assert manager.execute("regular", 3, 4) == 5

    rows = _function_spans(recorder)
    assert len(rows) == 1
    assert int(rows[0][2]) > 0


def test_coroutine_operator_span_includes_await() -> None:
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=16 * RECORD_WIDTH * 8),
        run_id="function-span",
        process_name="worker/test/rank:0",
    )

    async def coroutine() -> int:
        await asyncio.sleep(0.02)
        return 7

    manager = _bare_manager("coroutine", coroutine, coroutine_function=True)

    async def run() -> None:
        with activate_user_profiler(
            recorder=recorder,
            op_name="coroutine",
            step=4,
            rank=0,
            execution_mode="coroutine_async",
        ):
            assert await manager.execute_async("coroutine", 4) == 7

    asyncio.run(run())
    rows = _function_spans(recorder)
    assert len(rows) == 1
    assert int(rows[0][2]) >= 15_000_000
