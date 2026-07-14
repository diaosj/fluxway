import asyncio
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import Graph
from fluxway.placement import (
    AFFINITY_FIRST_V1,
    ClusterNode,
    ClusterSnapshot,
    PlacementContractError,
    PlacementInfeasibleError,
    PlacementPlanner,
    PlacementRecipe,
    PlacementRuntimeConfig,
    ResourceProfile,
    SPMDWorkerGroup,
    rank,
    subseteq,
    support,
    workers,
)
from fluxway.control_plane.scheduler import Scheduler

from .test_planner import scenario


def test_planner_logs_each_constraint_once(caplog) -> None:
    graph, recipe, snapshot = scenario(1)
    from fluxway.placement import PlacementPlanner

    with caplog.at_level(logging.INFO):
        PlacementPlanner(solve_timeout_s=10).resolve(graph, recipe, snapshot)

    messages = [record.getMessage() for record in caplog.records]
    validated = [
        message
        for message in messages
        if message.startswith("placement.constraint.validated")
    ]
    assert len(validated) == len(recipe.constraints)
    event_order = [
        "placement.recipe.selected",
        "placement.capacity.snapshot",
        "placement.program.normalized",
        "placement.constraint.validated",
        "placement.objective.started",
        "placement.objective.completed",
        "placement.solution.resolved",
    ]
    positions = [
        next(
            index for index, message in enumerate(messages) if message.startswith(event)
        )
        for event in event_order
    ]
    assert positions == sorted(positions)


def test_scheduler_exposes_pg_ready_allocation_report(monkeypatch) -> None:
    graph, recipe, snapshot = scenario(1)
    fake_pg = type("FakePG", (), {"id": "pg-1"})()

    async def create_job_pg(*, plan, snapshot, timeout_s):
        return fake_pg

    scheduler = Scheduler(
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)
    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)

    asyncio.run(scheduler.allocate_resources(graph))

    report = scheduler.allocation_report
    assert report is not None
    assert report.status == "PG_READY"
    assert report.pg_id == "pg-1"
    assert report.recipe_digest == recipe.digest
    assert report.capacity == snapshot
    assert report.normalized_recipe == recipe.normalized
    assert [result.satisfied for result in report.constraint_results] == [
        True,
    ] * len(recipe.constraints)
    assert report.occupancy["s1"] in (
        {"gpu-0": 8, "gpu-1": 4},
        {"gpu-0": 4, "gpu-1": 8},
    )
    assert set(report.support["s4"]).issubset(report.support["s1"])


def test_scheduler_marks_report_ready_after_all_workers_attest(
    monkeypatch, caplog
) -> None:
    graph, recipe, snapshot = scenario(1)
    spmd_manager = AsyncMock()
    scheduler = Scheduler(
        spmd_manager=spmd_manager,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )

    async def create_job_pg(*, plan, snapshot, timeout_s):
        return type("FakePG", (), {"id": "pg-ready"})()

    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)
    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)
    with caplog.at_level(logging.INFO):
        asyncio.run(scheduler.allocate_resources(graph))
        assert scheduler.resolved_plan is not None
        groups = {}
        for spmd_id, node in graph.nodes.items():
            assignments = sorted(
                (
                    assignment
                    for assignment in scheduler.resolved_plan.assignments
                    if assignment.spmd_id == spmd_id
                ),
                key=lambda assignment: assignment.rank,
            )
            groups[spmd_id] = SPMDWorkerGroup(
                spmd_id=spmd_id,
                placement_group=object(),
                rank_to_bundle=(0,) * node.spmd_config.num_workers,
                expected_node_ids=tuple(
                    assignment.node_id for assignment in assignments
                ),
                worker_resources={"CPU": Decimal("1")},
            )

        async def start_spmd(**kwargs):
            return type(
                "AttestedHandle",
                (),
                {"attested_node_ids": kwargs["worker_group"].expected_node_ids},
            )()

        spmd_manager.start_spmd.side_effect = start_spmd
        monkeypatch.setattr(
            scheduler.pg_manager,
            "get_worker_group",
            lambda spmd_id: groups[spmd_id],
        )
        asyncio.run(scheduler.init_spmds(graph))

    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.status == "READY"
    assert scheduler.allocation_report.attested_node_ids == {
        spmd_id: group.expected_node_ids for spmd_id, group in groups.items()
    }
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("placement.summary")
    ]
    assert len(summaries) == 1
    assert "status=READY" in summaries[0]


def test_invalid_rank_is_not_logged_as_validated(caplog) -> None:
    from .test_planner import SPMD0, graph_node

    placement_graph = Graph(nodes={"worker": graph_node("worker", SPMD0, 1)})
    worker = workers(SPMD0)
    recipe = PlacementRecipe(
        recipe_id="invalid-rank",
        version="1.0.0",
        profiles={SPMD0: ResourceProfile(base={"CPU": 1})},
        constraints=(subseteq(support(rank(worker, 1)), support(worker)),),
        objective=AFFINITY_FIRST_V1,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 1}),)
    )

    with caplog.at_level(logging.INFO), pytest.raises(PlacementContractError):
        PlacementPlanner(solve_timeout_s=10).resolve(placement_graph, recipe, snapshot)

    assert not any(
        record.getMessage().startswith("placement.constraint.validated")
        for record in caplog.records
    )


def test_infeasible_scheduler_run_records_one_failed_summary(
    monkeypatch, caplog
) -> None:
    from .test_planner import SPMD0, graph_node

    graph = Graph(nodes={"worker": graph_node("worker", SPMD0, 1)})
    recipe = PlacementRecipe(
        recipe_id="infeasible-report",
        version="1.0.0",
        profiles={SPMD0: ResourceProfile(base={"CPU": 2})},
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="cpu", resources={"CPU": 1}),)
    )
    scheduler = Scheduler(
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=7),
    )
    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)

    with caplog.at_level(logging.INFO), pytest.raises(PlacementInfeasibleError):
        asyncio.run(scheduler.allocate_resources(graph))

    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.status == "FAILED"
    assert scheduler.allocation_report.error is not None
    summaries = [
        record
        for record in caplog.records
        if record.getMessage().startswith("placement.summary")
    ]
    assert len(summaries) == 1


def test_topology_failure_releases_job_pg_and_records_one_failed_summary(
    monkeypatch, caplog
) -> None:
    graph, recipe, snapshot = scenario(1)
    spmd_manager = MagicMock()
    spmd_manager.establish_topology.side_effect = ValueError("topology failed")
    scheduler = Scheduler(
        spmd_manager=spmd_manager,
        placement_recipe=recipe,
        placement_runtime=PlacementRuntimeConfig(solve_timeout_s=10),
    )
    fake_pg = type("FakePG", (), {"id": "pg-topology"})()
    removed = []

    async def create_job_pg(*, plan, snapshot, timeout_s):
        scheduler.pg_manager._job_pg = fake_pg
        return fake_pg

    monkeypatch.setattr(scheduler, "_capture_snapshot", lambda: snapshot)
    monkeypatch.setattr(scheduler.pg_manager, "create_job_pg", create_job_pg)
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.remove_placement_group",
        removed.append,
    )

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(ValueError, match="topology failed"),
    ):
        asyncio.run(scheduler.schedule(graph))

    assert scheduler.allocation_report is not None
    assert scheduler.allocation_report.status == "FAILED"
    assert scheduler.allocation_report.error == "topology failed"
    assert scheduler.pg_manager.has_job_pg is False
    assert removed == [fake_pg]
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("placement.summary")
    ]
    assert len(summaries) == 1
    assert "status=FAILED" in summaries[0]
