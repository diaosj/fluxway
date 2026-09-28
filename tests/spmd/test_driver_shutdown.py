import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway import OperatorSpec
from fluxway.common.compiled_graph import compile_graph
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import SPMDConfig, DriverEvent
from fluxway.spmd.driver.driver import SPMDDriver
from fluxway.spmd.driver.executor import ray_executor
from tests.compiled_graph_fixtures import compiled_spmd_spec


@pytest.fixture(autouse=True)
def terminate_worker_actors(monkeypatch):
    terminate = MagicMock()
    monkeypatch.setattr(ray_executor.ray, "kill", terminate)
    return terminate


def _config(spmd_id: str, operator_id: str, world_size: int) -> SPMDConfig:
    mesh = DeviceMesh(axis_names=("worker",), device_shape=(world_size,))
    operator = OperatorSpec(name=operator_id)
    compiled_graph = compile_graph(
        name="driver-shutdown-test",
        spmds=(
            compiled_spmd_spec(
                spmd_id=spmd_id,
                mesh=mesh,
                operators={operator_id: operator},
            ),
        ),
        dependencies=(),
    ).spmd(spmd_id)
    return SPMDConfig(
        spmd_id=spmd_id,
        graph_view=compiled_graph,
        mesh=mesh,
        operator_specs={operator_id: operator},
    )


@pytest.mark.asyncio
async def test_driver_stop_waits_for_worker_shutdown():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=_config("dense", "compute", 2),
        placement_group=object(),
        worker_bundle_indices=[1, 2],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a", "node-a"],
    )

    worker0 = MagicMock()
    worker0.stop.remote = AsyncMock(return_value=None)
    worker0.set_driver_handle.remote = AsyncMock(return_value=None)
    worker1 = MagicMock()
    worker1.stop.remote = AsyncMock(return_value=None)
    worker1.set_driver_handle.remote = AsyncMock(return_value=None)

    driver.executor._worker_handles = {0: worker0, 1: worker1}
    driver.heartbeat_manager.stop = AsyncMock()

    await driver.stop()

    driver.heartbeat_manager.stop.assert_awaited_once_with()
    worker0.stop.remote.assert_awaited_once_with()
    worker1.stop.remote.assert_awaited_once_with()
    worker0.set_driver_handle.remote.assert_awaited_once_with(None)
    worker1.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver.executor._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_propagates_worker_shutdown_failure_after_detach():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=_config("metrics", "step", 1),
        placement_group=object(),
        worker_bundle_indices=[0],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )
    worker = MagicMock()
    worker.stop.remote = AsyncMock(side_effect=RuntimeError("drain failed"))
    worker.set_driver_handle.remote = AsyncMock(return_value=None)
    driver.executor._worker_handles = {0: worker}
    driver.heartbeat_manager.stop = AsyncMock()

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert str(raised.value.exceptions[0]) == "drain failed"
    worker.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver.executor._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_reports_submission_and_detach_failures():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=_config("metrics", "step", 2),
        placement_group=object(),
        worker_bundle_indices=[0, 1],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a", "node-b"],
    )
    failed = MagicMock()
    failed.stop.remote = MagicMock(side_effect=RuntimeError("submit failed"))
    failed.set_driver_handle.remote = AsyncMock(
        side_effect=RuntimeError("detach failed")
    )
    healthy = MagicMock()
    healthy.stop.remote = AsyncMock(return_value=None)
    healthy.set_driver_handle.remote = AsyncMock(return_value=None)
    driver.executor._worker_handles = {0: failed, 1: healthy}
    driver.heartbeat_manager.stop = AsyncMock()

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert {str(error) for error in raised.value.exceptions} == {
        "submit failed",
        "detach failed",
    }
    healthy.stop.remote.assert_awaited_once_with()
    healthy.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver.executor._worker_handles == {}


@pytest.mark.asyncio
async def test_driver_stop_shuts_down_workers_when_heartbeat_stop_fails():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="metrics",
        config=_config("metrics", "step", 1),
        placement_group=object(),
        worker_bundle_indices=[0],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )
    worker = MagicMock()
    worker.stop.remote = AsyncMock(return_value=None)
    worker.set_driver_handle.remote = AsyncMock(return_value=None)
    driver.executor._worker_handles = {0: worker}
    driver.heartbeat_manager.stop = AsyncMock(
        side_effect=RuntimeError("heartbeat stop failed")
    )

    with pytest.raises(ExceptionGroup, match="SPMD .* shutdown failed") as raised:
        await driver.stop()

    assert str(raised.value.exceptions[0]) == "heartbeat stop failed"
    worker.stop.remote.assert_awaited_once_with()
    worker.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver.executor._worker_handles == {}


@pytest.mark.asyncio
async def test_stop_heartbeat_joins_background_thread():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=_config("dense", "compute", 1),
        placement_group=object(),
        worker_bundle_indices=[1],
        controller_handle=None,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )

    join_mock = MagicMock()
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    fake_thread.join = join_mock

    driver.heartbeat_manager._heartbeat_running = True
    driver.heartbeat_manager._heartbeat_thread = fake_thread

    await driver.heartbeat_manager.stop()

    join_mock.assert_called_once()
    assert driver.heartbeat_manager._heartbeat_thread is None


def test_driver_ignores_worker_callbacks_while_stopping():
    # pyrefly: ignore [missing-attribute]
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=_config("dense", "compute", 1),
        placement_group=object(),
        worker_bundle_indices=[1],
        controller_handle=MagicMock(),
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )

    heartbeat = MagicMock()

    driver.state_machine.require_transition(DriverEvent.STOP)
    driver.spmd_synchronizer.report_worker_op_completed = MagicMock()

    driver.receive_worker_heartbeat(0, heartbeat)
    driver.report_worker_op_completed(0, "step", 1)
    driver.routing_control_argument_to_controller("target", "op", ["x"], 1, [1])

    assert driver._worker_heartbeats == {}
    assert driver._worker_heartbeat_times == {}
    driver.spmd_synchronizer.report_worker_op_completed.assert_not_called()
    driver.controller_handle.receive_control_argument.remote.assert_not_called()
