from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from fluxway.common.types import ExecutionWaitState, WorkerHeartbeat, WorkerStatus
from fluxway.spmd.worker.lifecycle import WorkerExecutionLifecycle, WorkerPhase
from fluxway.spmd.worker.worker import (
    SPMDWorker,
    _classify_execution_wait_state,
)


def _worker() -> SPMDWorker:
    worker = object.__new__(SPMDWorker.__ray_metadata__.modified_class)
    worker._failure = None
    worker.spmd_id = "test"
    worker.rank = 0
    worker._execution_lifecycle = WorkerExecutionLifecycle()
    worker._execution_lifecycle._phase = WorkerPhase.ACTIVE
    worker.config = SimpleNamespace(
        execution_progress_warning_s=1.0,
        heartbeat_timeout_s=999.0,
    )
    worker._last_progress_timestamp = time.time() - 2.0
    worker._last_stall_warning_timestamp = None
    worker._execution_diagnostic_snapshot = lambda: {"stall_reason": "operator_plan"}
    worker._execution_generation = 3
    worker._execution_epoch = 4
    worker._execution_sequence = 0
    worker._profiler_runtime = SimpleNamespace(enabled=False)
    return worker


def test_progress_warning_threshold_is_separate_from_liveness_timeout(caplog) -> None:
    worker = _worker()
    worker._warn_if_stalled()
    assert "no committed progress" in caplog.text
    assert worker.is_running()

    worker.config.execution_progress_warning_s = 999.0
    worker.config.heartbeat_timeout_s = 1.0
    worker._last_stall_warning_timestamp = None
    caplog.clear()
    worker._warn_if_stalled()
    assert "no committed progress" not in caplog.text


def test_execution_phase_is_opaque_and_sequence_is_monotonic() -> None:
    worker = _worker()
    recorder = MagicMock()
    worker._profiler_runtime = SimpleNamespace(enabled=True, recorder=recorder)

    worker._record_execution_phase("operator.invocation.begin", op_name="train", step=7)
    worker._record_execution_phase("operator.invocation.end", op_name="train", step=7)

    calls = recorder.execution_phase.call_args_list
    assert [call.kwargs["sequence"] for call in calls] == [1, 2]
    assert [call.kwargs["phase"] for call in calls] == [
        "operator.invocation.begin",
        "operator.invocation.end",
    ]
    assert calls[0].kwargs["execution_generation"] == 3


def test_wait_classification_separates_active_operator_from_empty_barrier() -> None:
    assert (
        _classify_execution_wait_state(
            execution_phase="operator.invocation.begin",
            execution_phase_active=True,
            active_plan_steps=True,
            pending_plan_tasks=True,
            pending_send_tasks=False,
            pending_input_prepare_tasks=False,
            queued_work=False,
            pending_admission=False,
        )
        is ExecutionWaitState.OPERATOR_EXECUTION_ACTIVE
    )
    assert (
        _classify_execution_wait_state(
            execution_phase=None,
            execution_phase_active=False,
            active_plan_steps=False,
            pending_plan_tasks=False,
            pending_send_tasks=False,
            pending_input_prepare_tasks=False,
            queued_work=False,
            pending_admission=False,
        )
        is ExecutionWaitState.COMPLETION_BARRIER_OR_DRIVER_WAIT
    )


def test_execution_phase_activity_does_not_reset_committed_progress() -> None:
    worker = _worker()
    committed_timestamp = worker._last_progress_timestamp
    worker._record_execution_phase(
        "operator.invocation.begin",
        op_name="train",
        step=7,
    )
    assert worker._last_progress_timestamp == committed_timestamp
    assert worker._execution_phase == "operator.invocation.begin"
    assert worker._execution_phase_active is True


def test_worker_heartbeat_carries_only_generic_execution_metadata() -> None:
    heartbeat = WorkerHeartbeat(
        rank=0,
        timestamp=1.0,
        status=WorkerStatus.RUNNING,
        execution_generation=3,
        execution_operator="train",
        execution_step=7,
        execution_epoch=4,
        execution_sequence=2,
        execution_in_flight=True,
    )

    assert heartbeat.execution_operator == "train"
    assert heartbeat.execution_step == 7
    assert heartbeat.execution_in_flight is True
    assert heartbeat.execution_wait_state is ExecutionWaitState.IDLE
    assert "completed_step_count" not in heartbeat.as_dict()
