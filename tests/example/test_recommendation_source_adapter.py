from __future__ import annotations

import pytest

from fluxway.common.ingestion import SourceWorkAcknowledgement
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.common.ingestion import WorkerLayoutDescriptor
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    TailLayoutMode,
    LayoutCompileRequest,
    ReaderTopologyDescriptor,
    ReaderGroupDemand,
    RankDemand,
    AssignmentPolicy,
)
from fluxway.example.recommendation.source_adapter import (
    RecommendationSourceAdapter,
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
    view = adapter.capture_view({"start": 3, "end": 5}, cardinality_limit=4)
    assert len(view.units) == 1
    assert view.units[0].logical_start == 3
    assert view.units[0].cardinality == 2
    assert all(unit.revision_identity == "recommendation:7" for unit in view.units)
    assert view.cursor == {"start": 3, "end": 5}
