"""Tests for checkpoint deadline, state machine transitions, permit fencing, and watchdog."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock
import pytest
from fluxway.control_plane.checkpointer.repository import CheckpointRepository

from fluxway.common.types import (
    CheckpointDeadline,
    CheckpointCut,
    CheckpointAssembly,
    SaveCheckpointRequest,
    WorkerHeartbeat,
    WorkerStatus,
    DriverState,
    HeartbeatConfig,
)
from fluxway.spmd.driver.heartbeat import HeartbeatManager


@pytest.mark.asyncio
async def test_checkpoint_deadline_bounds_execution():
    deadline = CheckpointDeadline.start(0.05)
    assert not deadline.is_expired()
    assert deadline.remaining_s() > 0

    with pytest.raises(TimeoutError):
        await deadline.run_phase(asyncio.sleep(0.1))

    assert deadline.is_expired()


@pytest.mark.asyncio
async def test_checkpointer_save_deadline_aborts_on_hang(
    checkpointer_factory, tmp_path
):
    controller = type("Controller", (), {})()
    cut = CheckpointCut.for_step("test_cut", 5)

    async def hanging_snapshot(*args, **kwargs):
        await asyncio.sleep(1.0)
        return CheckpointAssembly(cut, {}, {})

    controller.request_snapshot = AsyncMock(side_effect=hanging_snapshot)
    controller.abort_checkpoint_transaction = Mock()
    controller.rollback_snapshot = AsyncMock()

    checkpointer = await checkpointer_factory(
        controller,
        str(tmp_path / "repo"),
        interval_steps=1,
        snapshot_timeout_s=0.05,
    )

    with pytest.raises(TimeoutError):
        await checkpointer.save(SaveCheckpointRequest.for_step(5))

    # Staging phase timed out -> snapshot was never assembled
    assert controller.rollback_snapshot.await_count == 0
    assert not (tmp_path / "repo" / "manifests" / "5").exists()


@pytest.mark.asyncio
async def test_checkpointer_post_decision_failure_preserves_committed_manifest(
    checkpointer_factory,
    tmp_path,
):
    req = SaveCheckpointRequest.for_step(10)
    controller = type("Controller", (), {})()
    controller.request_snapshot = AsyncMock(
        return_value=CheckpointAssembly(req.cut, {}, {})
    )
    controller.prepare_checkpoint_transaction = Mock(return_value={})
    controller.apply_prepared_checkpoint_transaction = Mock()
    controller.abort_checkpoint_transaction = Mock()
    controller.rollback_snapshot = AsyncMock()

    # Publish fails after transaction decided
    async def failing_publish(*args, **kwargs):
        raise RuntimeError("simulated publish RPC failure")

    controller.publish_snapshot = AsyncMock(side_effect=failing_publish)

    checkpointer = await checkpointer_factory(
        controller,
        str(tmp_path / "repo"),
        interval_steps=1,
    )

    with pytest.raises(RuntimeError, match="simulated publish RPC failure"):
        await checkpointer.save(req)

    # Because decision was made, controller.abort_checkpoint_transaction and rollback_snapshot MUST NOT be called
    assert controller.abort_checkpoint_transaction.call_count == 0
    assert controller.rollback_snapshot.await_count == 0

    incomplete = CheckpointRepository(tmp_path / "repo").incomplete_transactions()
    assert len(incomplete) == 1
    checkpoint_id = incomplete[0].checkpoint_id
    assert incomplete[0].status.value == "commit_decided"
    assert CheckpointRepository(tmp_path / "repo").latest() is None

    # A restarted checkpointer fences the incomplete physical-write attempt.
    assert await checkpointer.restore_latest() is None
    fenced = CheckpointRepository(tmp_path / "repo").transaction(checkpoint_id)
    assert fenced is not None
    assert fenced.status.value == "abandoned"
    assert not (tmp_path / "repo" / "manifests" / checkpoint_id).exists()


def test_heartbeat_watchdog_detects_checkpoint_stalled_worker():
    failures = []
    now = time.time()
    worker_times = {0: now}
    worker_hbs = {
        0: WorkerHeartbeat(
            rank=0,
            timestamp=now,
            status=WorkerStatus.RUNNING,
            checkpoint_id="ckpt_1",
            checkpoint_progress_timestamp=now - 700.0,
            checkpoint_completed_units=1,
        )
    }

    hb_mgr = HeartbeatManager(
        spmd_id="worker_spmd",
        num_workers=1,
        heartbeat_config=HeartbeatConfig(
            heartbeat_interval=1.0, heartbeat_timeout=600.0
        ),
        profile_heartbeat=False,
        get_driver_state=lambda: DriverState.SNAPSHOTTING,
        get_driver_failure=lambda: None,
        get_workers_ready=lambda: True,
        get_topology_ready=lambda: True,
        get_worker_heartbeat_times=lambda: worker_times,
        controller_handle=Mock(),
        trigger_state_event=Mock(),
        receive_worker_failure=lambda rank, fail: failures.append((rank, fail)),
        get_worker_heartbeats=lambda: worker_hbs,
        inactivity_timeout_s=600.0,
    )

    hb_mgr._send_heartbeat()
    assert len(failures) == 1
    rank, failure = failures[0]
    assert rank == 0
    assert failure.exception_type == "WorkerCheckpointInactivityTimeout"


def test_heartbeat_watchdog_ignores_idle_worker_during_snapshotting():
    failures = []
    now = time.time()
    worker_times = {0: now}
    worker_hbs = {
        0: WorkerHeartbeat(
            rank=0,
            timestamp=now,
            status=WorkerStatus.RUNNING,
            checkpoint_id=None,
            checkpoint_progress_timestamp=0.0,
        )
    }

    hb_mgr = HeartbeatManager(
        spmd_id="worker_spmd",
        num_workers=1,
        heartbeat_config=HeartbeatConfig(
            heartbeat_interval=1.0, heartbeat_timeout=600.0
        ),
        profile_heartbeat=False,
        get_driver_state=lambda: DriverState.SNAPSHOTTING,
        get_driver_failure=lambda: None,
        get_workers_ready=lambda: True,
        get_topology_ready=lambda: True,
        get_worker_heartbeat_times=lambda: worker_times,
        controller_handle=Mock(),
        trigger_state_event=Mock(),
        receive_worker_failure=lambda rank, fail: failures.append((rank, fail)),
        get_worker_heartbeats=lambda: worker_hbs,
        inactivity_timeout_s=600.0,
    )

    hb_mgr._send_heartbeat()
    assert len(failures) == 0


def test_heartbeat_watchdog_detects_dead_worker_during_snapshotting():
    failures = []
    now = time.time()
    worker_times = {0: now - 650.0}

    hb_mgr = HeartbeatManager(
        spmd_id="worker_spmd",
        num_workers=1,
        heartbeat_config=HeartbeatConfig(
            heartbeat_interval=1.0, heartbeat_timeout=600.0
        ),
        profile_heartbeat=False,
        get_driver_state=lambda: DriverState.SNAPSHOTTING,
        get_driver_failure=lambda: None,
        get_workers_ready=lambda: True,
        get_topology_ready=lambda: True,
        get_worker_heartbeat_times=lambda: worker_times,
        controller_handle=Mock(),
        trigger_state_event=Mock(),
        receive_worker_failure=lambda rank, fail: failures.append((rank, fail)),
        get_worker_heartbeats=lambda: {},
        inactivity_timeout_s=600.0,
    )

    hb_mgr._send_heartbeat()
    assert len(failures) == 1
    rank, failure = failures[0]
    assert rank == 0
    assert failure.exception_type == "WorkerHeartbeatTimeout"
    assert failure.phase == "CHECKPOINT"
