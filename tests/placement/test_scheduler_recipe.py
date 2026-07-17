import asyncio
from decimal import Decimal
from unittest.mock import Mock
from types import SimpleNamespace
import pytest

from fluxway.placement import (
    ClusterSnapshot,
    PlacementContractError,
    PlacementRuntimeConfig,
)
from fluxway.control_plane.scheduler import Scheduler

from .test_planner import scenario


def test_scheduler_requires_graph_and_recipe() -> None:
    with pytest.raises(TypeError):
        Scheduler()


def test_scheduler_allocates_recipe_as_one_job_pg(monkeypatch) -> None:
    graph, recipe, snapshot = scenario(2)
    calls = []

    async def create_job_pg(*, plan, snapshot, timeout_s):
        calls.append((plan, snapshot, timeout_s))
        return SimpleNamespace(id="job-pg")

    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)
    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)

    allocated = asyncio.run(scheduler.allocate_resources())

    assert allocated is None
    assert scheduler.resolved_plan is calls[0][0]
    assert calls[0][1] is snapshot
    assert calls[0][2] == 7
    assert len(calls) == 1
    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.pg_id == "job-pg"


def test_scheduler_converts_ray_snapshot_floats_to_decimal(monkeypatch) -> None:
    graph, recipe, _ = scenario(1)
    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.scheduler.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-0",
                "Resources": {"CPU": 8.0, "node:10.0.0.1": 1.0},
            }
        ],
    )
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.scheduler.ray.available_resources",
        lambda: {"CPU": 7.5, "node:10.0.0.1": 1.0},
    )

    snapshot = scheduler._capture_snapshot()

    assert snapshot.nodes[0].resources["CPU"] == Decimal("8.0")
    assert snapshot.available["CPU"] == Decimal("7.5")


def test_scheduler_marks_changed_resource_snapshot_as_failed(monkeypatch) -> None:
    graph, recipe, snapshot = scenario(1)
    changed = ClusterSnapshot(
        nodes=snapshot.nodes,
        available={**snapshot.available, "CPU": snapshot.available["CPU"] - 1},
    )
    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    snapshots = iter((snapshot, changed))
    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: next(snapshots))

    with pytest.raises(PlacementContractError, match="resources changed"):
        asyncio.run(scheduler.allocate_resources())

    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.status == "FAILED"
    assert scheduler.allocation_report.error is not None
    assert "resources changed" in scheduler.allocation_report.error


def test_scheduler_ignores_unrequested_resource_changes(monkeypatch) -> None:
    graph, recipe, snapshot = scenario(1)
    snapshot_with_object_store = ClusterSnapshot(
        nodes=snapshot.nodes,
        available={**snapshot.available, "object_store_memory": Decimal(100)},
    )
    changed_object_store = ClusterSnapshot(
        nodes=snapshot.nodes,
        available={**snapshot.available, "object_store_memory": Decimal(99)},
    )
    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    snapshots = iter((snapshot_with_object_store, changed_object_store))
    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: next(snapshots))

    async def create_job_pg(*, plan, snapshot, timeout_s):
        return SimpleNamespace(id="job-pg")

    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)

    allocated = asyncio.run(scheduler.allocate_resources())

    assert allocated is None
    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.status == "PG_READY"


def test_scheduler_releases_the_job_placement_group() -> None:
    graph, recipe, snapshot = scenario(1)
    scheduler = Scheduler(
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    scheduler.pg_manager.release_job_pg = Mock()
    scheduler._resolved_plan = Mock()
    scheduler._cluster_snapshot = snapshot

    scheduler.release_resources()

    scheduler.pg_manager.release_job_pg.assert_called_once_with()
    assert scheduler.resolved_plan is None
    assert scheduler.allocation_report is None
