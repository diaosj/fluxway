from __future__ import annotations

from types import SimpleNamespace

import pytest

from fluxway.common.ingestion import (
    SourceScanBudget,
    SourceScanStatus,
    SourceWorkAcknowledgement,
    WorkerLayoutDescriptor,
)
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    AssignmentPolicy,
    LayoutCompileRequest,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.example.recommendation.source_adapter import (
    RecommendationSourceAdapter,
    RecommendationSourceDescriptor,
    RecommendationSourcePlanner,
)


@pytest.mark.asyncio
async def test_recommendation_planner_publishes_logical_work():
    planner = RecommendationSourcePlanner(
        source_identity="recommendation", revision_id=7
    )
    descriptors = planner.plan_units(0, 3)
    ledger = LogicalWorkLedger(max_in_flight_steps=3)
    compiled = AlgebraicLayoutCompiler().compile(
        LayoutCompileRequest(
            "recommendation",
            "view",
            "schema",
            descriptors,
            ReaderTopologyDescriptor(
                3, (ReaderGroupDemand("worker", (RankDemand("rank", 3),)),)
            ),
            TailLayoutMode.REJECT_REMAINDER,
            "lineage",
            assignment_policy=AssignmentPolicy.default(),
        )
    )
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await ledger.publish(compiled.work_epoch, bindings)
    leases = await ledger.acquire(0, bindings[0].segments)
    await ledger.acknowledge(
        tuple(
            SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
            for lease in leases
        )
    )
    assert await ledger.candidate_cut() == 1


def test_recommendation_source_adapter_uses_canonical_planner():
    adapter = RecommendationSourceAdapter(
        source_identity="recommendation", revision_id=7
    )
    scan_result = adapter.scan_next(
        adapter.cursor(start=3, end=5, finite=True),
        SourceScanBudget(4, 1),
    )
    assert len(scan_result.units) == 1
    assert scan_result.units[0].logical_start == 3
    assert scan_result.units[0].cardinality == 2
    assert all(
        unit.revision_identity == "recommendation:7" for unit in scan_result.units
    )
    assert scan_result.status is SourceScanStatus.TERMINAL


@pytest.mark.parametrize("finite", [True, False])
@pytest.mark.parametrize("limit", [1, 2, 3, 8])
def test_recommendation_preserves_boundary_across_partial_scans(finite, limit):
    adapter = RecommendationSourceAdapter()
    cursor = adapter.cursor(start=3, end=7, finite=finite)
    rows = []
    for _ in range(6):
        result = adapter.scan_next(cursor, SourceScanBudget(limit, 1))
        result.validate_transition(cursor)
        rows.extend(
            row
            for unit in result.units
            for row in range(unit.logical_start, unit.logical_start + unit.cardinality)
        )
        cursor = result.next_cursor
        if result.status in (SourceScanStatus.IDLE, SourceScanStatus.TERMINAL):
            break
    else:
        pytest.fail("source did not reach its declared boundary")
    assert rows == [3, 4, 5, 6]
    assert result.status is (
        SourceScanStatus.TERMINAL if finite else SourceScanStatus.IDLE
    )
    repeated = adapter.scan_next(cursor, SourceScanBudget(limit, 1))
    assert repeated.status is result.status
    assert repeated.records == 0
    assert repeated.next_cursor == cursor


@pytest.mark.parametrize("finite", [True, False])
def test_empty_recommendation_range_has_a_stable_boundary(finite):
    adapter = RecommendationSourceAdapter()
    cursor = adapter.cursor(start=3, end=3, finite=finite)
    result = adapter.scan_next(cursor, SourceScanBudget(2, 1))
    assert result.status is (
        SourceScanStatus.TERMINAL if finite else SourceScanStatus.IDLE
    )
    assert result.next_cursor == cursor


def test_recommendation_descriptor_builds_binding_with_cursor_owned_boundary():
    cycle = ((0,),)
    graph_view = SimpleNamespace(
        operator=lambda _: SimpleNamespace(
            execution_layout=SimpleNamespace(cycle=cycle)
        )
    )
    config = SimpleNamespace(
        num_workers=1,
        static_args={"global_batch_size": 4, "num_samples": 3, "num_epochs": 2},
        require_graph_view=lambda: graph_view,
    )

    binding = RecommendationSourceDescriptor().build_source_scan_binding(
        spmd_id="train",
        operator_name="read",
        config=config,
    )

    assert binding.identity.source_identity == "recommendation"
    assert binding.identity.mode_identity == "finite"
    initial_cursor = binding.initial_cursor_factory()
    result = binding.adapter.scan_next(initial_cursor, SourceScanBudget(16, 1))
    assert result.records == 6
    assert result.status is SourceScanStatus.TERMINAL
