import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import SPMDConfig, OperatorMeta
from fluxway.spmd.driver.driver import SPMDDriver


@pytest.mark.asyncio
async def test_driver_stop_waits_for_worker_shutdown():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=2),
            operator_metas={"compute": OperatorMeta(name="compute")},
            active_operator_names=("compute",),
        ),
        placement_group=object(),
        worker_bundle_indices=[1, 2],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a", "node-a"],
        checkpoint_backend=None,
    )

    worker0 = MagicMock()
    worker0.stop.remote = AsyncMock(return_value=None)
    worker0.set_driver_handle.remote = AsyncMock(return_value=None)
    worker1 = MagicMock()
    worker1.stop.remote = AsyncMock(return_value=None)
    worker1.set_driver_handle.remote = AsyncMock(return_value=None)

    driver._worker_handles = {0: worker0, 1: worker1}
    driver._stop_heartbeat = AsyncMock()

    await driver.stop()

    driver._stop_heartbeat.assert_called_once_with()
    worker0.stop.remote.assert_awaited_once_with()
    worker1.stop.remote.assert_awaited_once_with()
    worker0.set_driver_handle.remote.assert_awaited_once_with(None)
    worker1.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_propagates_worker_shutdown_failure_after_detach():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=1),
            operator_metas={"step": OperatorMeta(name="step")},
            active_operator_names=("step",),
        ),
        placement_group=object(),
        worker_bundle_indices=[0],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
        checkpoint_backend=None,
    )
    worker = MagicMock()
    worker.stop.remote = AsyncMock(side_effect=RuntimeError("drain failed"))
    worker.set_driver_handle.remote = AsyncMock(return_value=None)
    driver._worker_handles = {0: worker}
    driver._stop_heartbeat = AsyncMock()

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert str(raised.value.exceptions[0]) == "drain failed"
    worker.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_reports_submission_and_detach_failures():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=2),
            operator_metas={"step": OperatorMeta(name="step")},
            active_operator_names=("step",),
        ),
        placement_group=object(),
        worker_bundle_indices=[0, 1],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a", "node-b"],
        checkpoint_backend=None,
    )
    failed = MagicMock()
    failed.stop.remote = MagicMock(side_effect=RuntimeError("submit failed"))
    failed.set_driver_handle.remote = AsyncMock(
        side_effect=RuntimeError("detach failed")
    )
    healthy = MagicMock()
    healthy.stop.remote = AsyncMock(return_value=None)
    healthy.set_driver_handle.remote = AsyncMock(return_value=None)
    driver._worker_handles = {0: failed, 1: healthy}
    driver._stop_heartbeat = AsyncMock()

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert {str(error) for error in raised.value.exceptions} == {
        "submit failed",
        "detach failed",
    }
    healthy.stop.remote.assert_awaited_once_with()
    healthy.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_shuts_down_workers_when_heartbeat_stop_fails():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=1),
            operator_metas={"step": OperatorMeta(name="step")},
            active_operator_names=("step",),
        ),
        placement_group=object(),
        worker_bundle_indices=[0],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
        checkpoint_backend=None,
    )
    worker = MagicMock()
    worker.stop.remote = AsyncMock(return_value=None)
    worker.set_driver_handle.remote = AsyncMock(return_value=None)
    driver._worker_handles = {0: worker}
    driver._stop_heartbeat = AsyncMock(
        side_effect=RuntimeError("heartbeat stop failed")
    )

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert str(raised.value.exceptions[0]) == "heartbeat stop failed"
    worker.stop.remote.assert_awaited_once_with()
    worker.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


@pytest.mark.asyncio
async def test_stop_heartbeat_joins_background_thread():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=1),
            operator_metas={"compute": OperatorMeta(name="compute")},
            active_operator_names=("compute",),
        ),
        placement_group=object(),
        worker_bundle_indices=[1],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
        checkpoint_backend=None,
    )

    join_mock = MagicMock()
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    fake_thread.join = join_mock

    driver._heartbeat_running = True
    driver._heartbeat_thread = fake_thread

    await driver._stop_heartbeat()

    join_mock.assert_called_once()
    assert driver._heartbeat_thread is None


def test_driver_ignores_worker_callbacks_while_stopping():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(
            mesh=SimpleNamespace(size=1),
            operator_metas={"compute": OperatorMeta(name="compute")},
            active_operator_names=("compute",),
        ),
        placement_group=object(),
        worker_bundle_indices=[1],
        controller_handle=MagicMock(),
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
        checkpoint_backend=None,
    )

    heartbeat = MagicMock()

    driver._stopping = True
    driver.spmd_synchronizer.report_worker_op_completed = MagicMock()

    driver.receive_worker_heartbeat(0, heartbeat)
    driver.report_worker_op_completed(0, "step", 1)
    driver.routing_control_argument_to_controller("target", "op", ["x"], 1, [1])

    assert driver._worker_heartbeats == {}
    assert driver._worker_heartbeat_times == {}
    driver.spmd_synchronizer.report_worker_op_completed.assert_not_called()
    driver.controller_handle.receive_control_argument.remote.assert_not_called()
