import asyncio
import pickle
import time

import pytest
from ray import cloudpickle

from fluxway.common.types import (
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    ExecutionFailure,
    HeartbeatConfig,
)
from fluxway.control_plane import ExecutionFailedError
from fluxway.control_plane.heartbeat import HeartbeatManager


def _driver_failure() -> ExecutionFailure:
    return ExecutionFailure(
        spmd_id="dense",
        rank=None,
        op_name=None,
        step=None,
        phase="DRIVER",
        exception_type="RuntimeError",
        message="driver failed",
        traceback=None,
    )


@pytest.mark.asyncio
async def test_heartbeat_manager_forwards_driver_failure() -> None:
    received: list[tuple[str, ExecutionFailure]] = []
    manager = HeartbeatManager(
        on_driver_error=lambda spmd_id, failure: received.append((spmd_id, failure))
    )
    manager.register_spmd("dense")
    failure = _driver_failure()

    await manager.receive_heartbeat(
        DriverHeartbeat(
            spmd_id="dense",
            timestamp=time.time(),
            status=DriverStatus.ERROR,
            state=DriverState.ERROR,
            failure=failure,
        )
    )

    assert received == [("dense", failure)]
    assert received[0][1] is failure


@pytest.mark.asyncio
async def test_heartbeat_manager_propagates_error_callback_failure() -> None:
    async def reject_failure(spmd_id: str, failure: ExecutionFailure) -> None:
        raise RuntimeError(f"{spmd_id}: {failure.message}")

    manager = HeartbeatManager(on_driver_error=reject_failure)
    manager.register_spmd("dense")

    with pytest.raises(RuntimeError, match="dense: driver failed"):
        await manager.receive_heartbeat(
            DriverHeartbeat(
                spmd_id="dense",
                timestamp=time.time(),
                status=DriverStatus.ERROR,
                state=DriverState.ERROR,
                failure=_driver_failure(),
            )
        )


@pytest.mark.asyncio
async def test_heartbeat_manager_rejects_missing_error_failure_invariant() -> None:
    manager = HeartbeatManager()
    manager.register_spmd("dense")
    heartbeat = DriverHeartbeat.model_construct(
        spmd_id="dense",
        timestamp=time.time(),
        status=DriverStatus.ERROR,
        state=DriverState.ERROR,
        failure=None,
        num_workers=0,
        healthy_num_workers=0,
        workers_ready=False,
        workers_running=False,
    )

    with pytest.raises(RuntimeError, match="ERROR invariant violated"):
        await manager.receive_heartbeat(heartbeat)


@pytest.mark.asyncio
async def test_heartbeat_timeout_reports_structured_failure() -> None:
    received: list[tuple[str, ExecutionFailure]] = []
    received_event = asyncio.Event()

    async def receive_failure(spmd_id: str, failure: ExecutionFailure) -> None:
        received.append((spmd_id, failure))
        received_event.set()

    manager = HeartbeatManager(
        config=HeartbeatConfig(
            heartbeat_interval=0.001,
            heartbeat_timeout=0.001,
            max_missed_heartbeats=1,
        ),
        on_driver_error=receive_failure,
    )
    manager.register_spmd("dense")

    await manager.start_monitoring_async()
    try:
        await asyncio.wait_for(received_event.wait(), timeout=0.2)
    finally:
        await manager.stop_monitoring_async()

    assert len(received) == 1
    spmd_id, failure = received[0]
    assert spmd_id == "dense"
    assert failure.spmd_id == "dense"
    assert failure.rank is None
    assert failure.op_name is None
    assert failure.step is None
    assert failure.phase == "DRIVER_HEARTBEAT"
    assert failure.exception_type == "DriverHeartbeatTimeout"
    assert failure.message == "Heartbeat timeout (missed 1)"
    assert failure.traceback is None


def test_execution_failed_error_is_public_and_carries_failures() -> None:
    failure = _driver_failure()
    error = ExecutionFailedError({"dense": failure})

    assert error.failures == {"dense": failure}
    assert "dense phase=DRIVER RuntimeError: driver failed" in str(error)
    assert pickle.loads(pickle.dumps(error)).failures == {"dense": failure}
    assert cloudpickle.loads(cloudpickle.dumps(error)).failures == {"dense": failure}
