import asyncio
from types import SimpleNamespace
import types
from unittest.mock import MagicMock

import pytest

from fluxway.common.types import ExecutionFailure, WorkerStatus
from fluxway.spmd.worker.worker import SPMDWorker, _EXECUTION_STOP_ITEM
from fluxway.spmd.worker.admission import StepAdmissionState


def _raw_worker_class():
    return SPMDWorker.__ray_actor_class__  # type: ignore[attr-defined]


def _driver():
    receive_worker_failure = MagicMock()
    receive_worker_heartbeat = MagicMock()
    return SimpleNamespace(
        receive_worker_failure=SimpleNamespace(remote=receive_worker_failure),
        receive_worker_heartbeat=SimpleNamespace(remote=receive_worker_heartbeat),
    )


def _worker():
    raw_class = _raw_worker_class()
    worker = SimpleNamespace(
        spmd_id="training",
        rank=2,
        _failure=None,
        _running=True,
        _driver_handle=_driver(),
        _execution_queues={
            "compute": asyncio.PriorityQueue(),
            "communication": asyncio.PriorityQueue(),
        },
        _plan_tasks=set(),
        _send_tasks={},
        _background_tasks=set(),
        _heartbeat_task=None,
        _op_runnable={"op": asyncio.Event()},
    )
    worker._op_runnable["op"].set()
    worker._record_failure = types.MethodType(raw_class._record_failure, worker)
    worker._send_heartbeat = types.MethodType(raw_class._send_heartbeat, worker)
    return worker


async def _pending_task():
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_first_failure_stops_execution_and_reports_one_structured_fact():
    worker = _worker()
    first = RuntimeError("first failure")
    second = ValueError("second failure")

    worker._record_failure(
        first,
        phase="LOCAL_STEP_PLAN",
        op_name="lookup",
        step=4,
    )
    latched = worker._failure
    worker._record_failure(second, phase="BACKGROUND")

    assert isinstance(latched, ExecutionFailure)
    assert worker._failure is latched
    assert worker._running is False
    assert latched.op_name == "lookup"
    assert latched.step == 4
    assert latched.phase == "LOCAL_STEP_PLAN"
    assert latched.exception_type == "RuntimeError"
    worker._driver_handle.receive_worker_failure.remote.assert_called_once_with(
        worker.rank,
        latched,
    )
    assert [queue.get_nowait() for queue in worker._execution_queues.values()] == [
        _EXECUTION_STOP_ITEM,
        _EXECUTION_STOP_ITEM,
    ]


@pytest.mark.asyncio
async def test_failure_cancels_execution_tasks_but_preserves_heartbeat():
    worker = _worker()
    plan_task = asyncio.create_task(_pending_task())
    send_task = asyncio.create_task(_pending_task())
    background_task = asyncio.create_task(_pending_task())
    heartbeat_task = asyncio.create_task(_pending_task())
    worker._plan_tasks.add(plan_task)
    worker._send_tasks[send_task] = ("op", 0)
    worker._background_tasks.add(background_task)
    worker._heartbeat_task = heartbeat_task

    worker._record_failure(RuntimeError("operator failed"), phase="SCHEDULER")
    await asyncio.sleep(0)

    assert plan_task.cancelled()
    assert send_task.cancelled()
    assert background_task.cancelled()
    assert not heartbeat_task.cancelled()

    heartbeat_task.cancel()
    await asyncio.gather(heartbeat_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failure_heartbeat_carries_the_latched_execution_failure():
    worker = _worker()
    worker._record_failure(
        RuntimeError("send failed"), phase="SEND", op_name="op", step=1
    )

    await worker._send_heartbeat()

    call = worker._driver_handle.receive_worker_heartbeat.remote.call_args
    assert call.args[0] == worker.rank
    heartbeat = call.args[1]
    assert heartbeat.status is WorkerStatus.ERROR
    assert heartbeat.failure is worker._failure
    assert not hasattr(heartbeat, "error_message")


@pytest.mark.asyncio
async def test_failed_worker_does_not_admit_pending_pipeline_steps():
    worker = SimpleNamespace(
        _running=False,
        _failure=ExecutionFailure(
            spmd_id="training",
            rank=2,
            op_name="op",
            step=0,
            phase="LOCAL_STEP_PLAN",
            exception_type="RuntimeError",
            message="failed",
            traceback=None,
        ),
        _ready_pending_pipeline_steps={"op": [1]},
        _completed_step_count={"op": 0},
        _scheduled_step_count={"op": 0},
        _in_op_step_queue=set(),
        _pending_blocked={"op": []},
        _pending_future_steps={"op": []},
        _mailbox_ready_steps={"op": set()},
        _running_plan_steps={"op": set()},
        _finished_plan_steps={"op": {}},
        _can_start_local_plan=MagicMock(return_value=True),
        _start_local_step_plan=MagicMock(),
        _async_lag_limit_reached=MagicMock(return_value=False),
        _phase_order_step_blocked=MagicMock(return_value=False),
        _pending_async_lag_steps={"op": []},
        _pending_phase_order_steps={"op": []},
    )
    admission = StepAdmissionState(
        completed=worker._completed_step_count,
        scheduled=worker._scheduled_step_count,
        queue_membership=worker._in_op_step_queue,
        pending_blocked=worker._pending_blocked,
        pending_future=worker._pending_future_steps,
        pending_async_lag=worker._pending_async_lag_steps,
        pending_phase_order=worker._pending_phase_order_steps,
        ready_pipeline=worker._ready_pending_pipeline_steps,
        mailbox_ready=worker._mailbox_ready_steps,
        running=worker._running_plan_steps,
        finished=worker._finished_plan_steps,
        is_periodic=lambda _op: False,
        is_runnable=lambda _op: True,
        hold_prepare_token=lambda _op, _step: False,
        async_lag_blocked=worker._async_lag_limit_reached,
        phase_order_blocked=worker._phase_order_step_blocked,
        can_start=worker._can_start_local_plan,
    )

    await admission.start_pending(
        "op",
        running=lambda: worker._running,
        source_exhausted=lambda _op: False,
        start_plan=worker._start_local_step_plan,
    )

    worker._start_local_step_plan.assert_not_called()


def test_failed_worker_does_not_spawn_work_for_late_mailbox_readiness():
    raw_class = _raw_worker_class()
    worker = SimpleNamespace(
        _running=False,
        spmd_id="training",
        rank=2,
        op_manager=SimpleNamespace(is_periodic_trigger=lambda _op_name: True),
        _should_start_prepare_on_ready=MagicMock(return_value=True),
        _start_prepare_and_enqueue_token=MagicMock(),
        _spawn_background_task=MagicMock(),
    )

    raw_class._on_step_ready(worker, "op", 3)

    worker._should_start_prepare_on_ready.assert_not_called()
    worker._spawn_background_task.assert_not_called()
