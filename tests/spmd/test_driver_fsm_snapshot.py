"""
Unit tests for DriverStateMachine snapshot-related transitions.

Proves that DRAINING and SNAPSHOTTING states are correctly entered/exited,
and that BLOCK/UNBLOCK events are rejected during drain/snapshot.
"""

from unittest.mock import MagicMock

from fluxway.common.types import DriverEvent, DriverState, ExecutionFailure
from fluxway.spmd.driver.state_machine.driver_fsm import DriverStateMachine


def _make_fsm() -> tuple[DriverStateMachine, MagicMock]:
    driver = MagicMock()
    driver.spmd_id = "test_spmd"
    fsm = DriverStateMachine(driver)
    # Advance to RUNNING so snapshot path is reachable
    fsm.trigger(DriverEvent.INIT_COMPLETE)  # INITIALIZING → READY
    fsm.trigger(DriverEvent.START)  # READY → RUNNING
    assert fsm.state == DriverState.RUNNING
    return fsm, driver


def _worker_failure() -> ExecutionFailure:
    return ExecutionFailure(
        spmd_id="test_spmd",
        rank=0,
        op_name=None,
        step=None,
        phase="WORKER_EXECUTION",
        exception_type="RuntimeError",
        message="crash",
        traceback=None,
    )


class TestSnapshotHappyPath:
    def test_running_to_draining(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        assert fsm.state == DriverState.DRAINING

    def test_draining_to_drained(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        assert fsm.state == DriverState.DRAINED

    def test_drained_to_snapshotting(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        assert fsm.state == DriverState.SNAPSHOTTING

    def test_snapshotting_to_running(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        fsm.trigger(DriverEvent.SNAPSHOT_COMPLETE)
        assert fsm.state == DriverState.RUNNING

    def test_drained_to_running_no_snapshot(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_COMPLETE)
        assert fsm.state == DriverState.RUNNING


class TestSnapshotErrorPaths:
    def test_running_worker_error_fails_collective(self):
        fsm, driver = _make_fsm()
        failure = _worker_failure()
        fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)
        assert fsm.state == DriverState.ERROR
        driver._report_error.assert_called_once_with(failure)
        driver._recover_worker.assert_not_called()

    def test_blocked_worker_error_fails_collective(self):
        fsm, driver = _make_fsm()
        fsm.trigger(DriverEvent.BLOCK_REQUESTED)
        failure = _worker_failure()
        fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)
        assert fsm.state == DriverState.ERROR
        driver._report_error.assert_called_once_with(failure)
        driver._recover_worker.assert_not_called()

    def test_draining_worker_error_goes_to_error(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        failure = _worker_failure()
        fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)
        assert fsm.state == DriverState.ERROR
        assert fsm.failure is failure

    def test_snapshotting_worker_error_goes_to_error(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        failure = _worker_failure()
        fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)
        assert fsm.state == DriverState.ERROR
        assert fsm.failure is failure

    def test_snapshotting_failure_goes_to_error(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        fsm.trigger(DriverEvent.SNAPSHOT_FAILED)
        assert fsm.state == DriverState.ERROR
        assert fsm.failure is not None
        assert fsm.failure.phase == "SNAPSHOT"
        assert fsm.failure.exception_type == "SnapshotFailed"


class TestBlockGuard:
    def test_block_requested_ignored_during_draining(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        result = fsm.trigger(DriverEvent.BLOCK_REQUESTED)
        assert result is False
        assert fsm.state == DriverState.DRAINING

    def test_unblock_requested_ignored_during_draining(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        result = fsm.trigger(DriverEvent.UNBLOCK_REQUESTED)
        assert result is False
        assert fsm.state == DriverState.DRAINING

    def test_block_requested_ignored_during_snapshotting(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        result = fsm.trigger(DriverEvent.BLOCK_REQUESTED)
        assert result is False
        assert fsm.state == DriverState.SNAPSHOTTING

    def test_unblock_requested_ignored_during_snapshotting(self):
        fsm, _ = _make_fsm()
        fsm.trigger(DriverEvent.DRAIN_REQUESTED)
        fsm.trigger(DriverEvent.DRAIN_COMPLETE)
        fsm.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        result = fsm.trigger(DriverEvent.UNBLOCK_REQUESTED)
        assert result is False
        assert fsm.state == DriverState.SNAPSHOTTING
