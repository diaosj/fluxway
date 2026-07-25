from dataclasses import FrozenInstanceError
import pickle

import pytest
from pydantic import ValidationError
from ray import cloudpickle

import fluxway
from fluxway.common.types import (
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    ExecutionFailure,
    WorkerHeartbeat,
    WorkerStatus,
)
from fluxway.control_plane import ExecutionFailedError


def _raise_user_error() -> None:
    raise RuntimeError("user async failure")


def test_execution_failure_api_is_available_at_fluxway_top_level() -> None:
    assert fluxway.ExecutionFailure is ExecutionFailure
    assert fluxway.ExecutionFailedError is ExecutionFailedError


def test_execution_failure_captures_structured_exception_context() -> None:
    failure: ExecutionFailure | None = None
    try:
        _raise_user_error()
    except RuntimeError as error:
        failure = ExecutionFailure.from_exception(
            spmd_id="dense",
            rank=2,
            op_name="train",
            step=7,
            phase="LOCAL_STEP_PLAN",
            exception=error,
        )

    assert failure is not None
    assert failure.spmd_id == "dense"
    assert failure.rank == 2
    assert failure.op_name == "train"
    assert failure.step == 7
    assert failure.phase == "LOCAL_STEP_PLAN"
    assert failure.exception_type == "RuntimeError"
    assert failure.message == "user async failure"
    assert failure.traceback is not None
    assert "_raise_user_error" in failure.traceback
    assert str(failure) == (
        "dense rank=2 op=train step=7 phase=LOCAL_STEP_PLAN "
        "RuntimeError: user async failure"
    )


def test_execution_failure_is_frozen_and_ray_serializable() -> None:
    failure = ExecutionFailure(
        spmd_id="dense",
        rank=None,
        op_name=None,
        step=None,
        phase="HEARTBEAT",
        exception_type="WorkerHeartbeatTimeout",
        message="worker 0 heartbeat timed out",
        traceback=None,
    )

    with pytest.raises(FrozenInstanceError):
        failure.message = "changed"  # type: ignore[misc]

    assert pickle.loads(pickle.dumps(failure)) == failure
    assert cloudpickle.loads(cloudpickle.dumps(failure)) == failure


def test_heartbeat_uses_failure_as_the_only_error_fact() -> None:
    failure = ExecutionFailure(
        spmd_id="dense",
        rank=0,
        op_name="train",
        step=3,
        phase="SEND",
        exception_type="RuntimeError",
        message="delivery failed",
        traceback=None,
    )
    worker = WorkerHeartbeat(
        rank=0,
        timestamp=1.0,
        status=WorkerStatus.ERROR,
        failure=failure,
    )
    driver = DriverHeartbeat(
        spmd_id="dense",
        timestamp=1.0,
        status=DriverStatus.ERROR,
        state=DriverState.ERROR,
        failure=failure,
        num_workers=1,
        healthy_num_workers=0,
    )

    assert worker.failure is failure
    assert driver.failure is failure
    assert not hasattr(worker, "error_message")
    assert not hasattr(driver, "error_message")


@pytest.mark.parametrize(
    "heartbeat",
    (
        lambda failure: WorkerHeartbeat(
            rank=0,
            timestamp=1.0,
            status=WorkerStatus.ERROR,
            failure=failure,
        ),
        lambda failure: DriverHeartbeat(
            spmd_id="dense",
            timestamp=1.0,
            status=DriverStatus.ERROR,
            state=DriverState.ERROR,
            failure=failure,
        ),
    ),
)
def test_error_heartbeat_requires_failure(heartbeat) -> None:
    with pytest.raises(ValidationError):
        heartbeat(None)


@pytest.mark.parametrize(
    "heartbeat",
    (
        lambda failure: WorkerHeartbeat(
            rank=0,
            timestamp=1.0,
            status=WorkerStatus.RUNNING,
            failure=failure,
        ),
        lambda failure: DriverHeartbeat(
            spmd_id="dense",
            timestamp=1.0,
            status=DriverStatus.HEALTHY,
            state=DriverState.RUNNING,
            failure=failure,
        ),
    ),
)
def test_non_error_heartbeat_rejects_failure(heartbeat) -> None:
    failure = ExecutionFailure(
        spmd_id="dense",
        rank=0,
        op_name=None,
        step=None,
        phase="HEARTBEAT",
        exception_type="RuntimeError",
        message="failed",
        traceback=None,
    )
    with pytest.raises(ValidationError):
        heartbeat(failure)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("spmd_id", ""),
        ("rank", -1),
        ("op_name", ""),
        ("step", -1),
        ("phase", ""),
        ("exception_type", ""),
    ),
)
def test_execution_failure_rejects_invalid_identity(field: str, value: object) -> None:
    values: dict[str, object] = {
        "spmd_id": "dense",
        "rank": 0,
        "op_name": "train",
        "step": 3,
        "phase": "CALL",
        "exception_type": "RuntimeError",
        "message": "failed",
        "traceback": None,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        ExecutionFailure(**values)  # type: ignore[arg-type]
