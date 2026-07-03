from unittest.mock import MagicMock

import pytest

from fluxway.common.types import DriverResourceSpec, ResourceSpec
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.scheduler import PGManager


def test_controller_exposes_resource_allocation_summary():
    graph_stub = MagicMock()
    graph_stub.nodes = {}
    graph_stub.edges = []
    graph_stub.source_nodes = []

    ctrl = ControllerActor.__ray_actor_class__(graph=graph_stub)  # type: ignore[attr-defined]
    ctrl._scheduler.pg_manager._spmd_to_resource["dense"] = ResourceSpec(
        num_cpus=2, memory_gb=1.0
    )
    ctrl._scheduler.pg_manager._spmd_to_num_workers["dense"] = 6
    ctrl._scheduler.pg_manager._spmd_to_driver_resource["dense"] = DriverResourceSpec(
        num_cpus=1, memory_gb=1.0
    )

    summary = ctrl.get_resource_allocation_summary()

    assert summary["dense"]["bundle_count"] == 7
    assert summary["dense"]["worker_count"] == 6
    assert summary["dense"]["total_memory_gb"] == 7.0
    assert summary["dense"]["total_cpu"] == 13.0


def test_pg_manager_resource_allocation_summary():
    pg_manager = PGManager()
    pg_manager._spmd_to_resource["summary_spmd"] = ResourceSpec(
        num_cpus=1, memory_gb=1.5
    )
    pg_manager._spmd_to_num_workers["summary_spmd"] = 3
    pg_manager._spmd_to_driver_resource["summary_spmd"] = DriverResourceSpec(
        num_cpus=1, memory_gb=0.5
    )

    summary = pg_manager.get_resource_allocation_summary("summary_spmd")

    assert summary is not None
    assert summary["spmd_id"] == "summary_spmd"
    assert summary["worker_count"] == 3
    assert summary["bundle_count"] == 4
    assert summary["driver_bundle"]["memory_gb"] == 0.5
    assert summary["worker_bundle"]["memory_gb"] == 1.5
    assert summary["total_memory_gb"] == pytest.approx(5.0)
    assert summary["total_cpu"] == pytest.approx(4.0)
