from __future__ import annotations

from dataclasses import replace

import pytest

from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
    SourceWorkAcknowledgement,
)
from fluxway.common.types import CheckpointCut
from fluxway.common.ingestion import SourceBindingIdentity
from fluxway.common.types.configs import SourceControlConfig
from fluxway.control_plane.components import ControlPlaneComponentRegistry
from fluxway.control_plane.ingestion import (
    AssignmentPolicy,
    IngestionLayoutPlanningComponent,
    TailLayoutMode,
)
from tests.control_plane.source_scan_fixtures import InMemorySourceScanAdapter

from fluxway.control_plane.source_control import SourceControlRuntime
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger


def _segments(compiled, step):
    return tuple(
        segment
        for layout in compiled.worker_layouts
        for segment in layout.segments
        if segment.step == step
    )


def _fixed_initial_cursor(cursor):
    return lambda: cursor


@pytest.mark.asyncio
@pytest.mark.parametrize("consumed_steps", [0, 1, 3])
async def test_source_checkpoint_replans_from_consumed_cut_across_topology_change(
    source_layout, consumed_steps
):
    original = source_layout(demands=(2, 2), cardinality=14, terminal=False)
    topology = original.request.topology
    planner = IngestionLayoutPlanningComponent("ingestion-planning", topology=topology)
    control = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=4))
    registry = ControlPlaneComponentRegistry()
    registry.register(planner)
    registry.register(control)
    start = InMemorySourceScanAdapter.cursor(0)
    end = InMemorySourceScanAdapter.cursor(14)
    result = SourceScanResult(
        source_identity=original.request.source_identity,
        view_identity=original.request.view_identity,
        schema_digest=original.request.schema_digest,
        lineage_digest=original.request.lineage_digest,
        units=(replace(original.request.units[0], cardinality=14),),
        unit_cursors=tuple(
            OpaqueSourceCursor.from_payload(unit.unit_id.encode())
            for unit in (replace(original.request.units[0], cardinality=14),)
        ),
        records=14,
        next_cursor=end,
        status=SourceScanStatus.READY,
    )
    restored_planner = IngestionLayoutPlanningComponent(
        "ingestion-planning",
        topology=source_layout(
            demands=(4,), cardinality=14, terminal=False
        ).request.topology,
    )
    restored_control = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=4))
    restored_registry = ControlPlaneComponentRegistry()
    restored_registry.register(restored_planner)
    restored_registry.register(restored_control)
    try:
        await planner.plan_source_scan(
            InMemorySourceScanAdapter((result,)),
            _fixed_initial_cursor(start),
            control.work_ledger,
            start,
            budget=SourceScanBudget(14, 1),
            topology=topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
            max_steps=3,
        )
        compiled = planner.windows[0]
        old_leases = []
        for step in range(3):
            leases = await control.work_ledger.acquire(0, _segments(compiled, step))
            old_leases.extend(leases)
            await control.work_ledger.acknowledge(
                tuple(
                    SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
                    for lease in leases
                )
            )
        await control.work_ledger.commit_cut(consumed_steps)
        cut = CheckpointCut.for_position("source-cut", consumed_steps)
        checkpoint = registry.checkpoint_state(cut)
        restored_registry.restore_state(checkpoint, cut)
        assert restored_planner.windows == ()
        assert restored_planner._pending_units == ()
        assert restored_control.work_ledger.epochs == ()
        assert restored_control.work_ledger.committed_cut == consumed_steps
        assert await restored_control.work_ledger.candidate_cut() == consumed_steps
        bookmark_state = planner.checkpoint_state(cut)
        assert set(bookmark_state) == {
            "cut_position",
            "cursor",
            "initial_cursor",
            "binding_identity",
            "source_identity",
            "view_identity",
            "schema_digest",
            "lineage_digest",
            "global_cardinality",
            "tail_layout_mode",
        }
        assert (
            bookmark_state["initial_cursor"]
            == InMemorySourceScanAdapter.cursor(0).to_checkpoint()
        )
        assert bookmark_state["binding_identity"] is None
        assert restored_planner.resume_cursor == InMemorySourceScanAdapter.cursor(
            consumed_steps * 4
        )
        await restored_planner.plan_source_scan(
            InMemorySourceScanAdapter((result,)),
            _fixed_initial_cursor(restored_planner.resume_cursor),
            restored_control.work_ledger,
            restored_planner.resume_cursor,
            budget=SourceScanBudget(14, 1),
            topology=restored_planner._topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=consumed_steps,
            max_steps=1,
        )
        while (
            await restored_planner.plan_next_window(
                restored_control.work_ledger,
                max_steps=1,
            )
            is not None
        ):
            pass
        replay_rows = []
        for window in restored_planner.windows:
            for step in range(
                max(consumed_steps, window.work_epoch.start_step),
                window.work_epoch.end_step,
            ):
                leases = await restored_control.work_ledger.acquire(
                    window.work_epoch.epoch, _segments(window, step)
                )
                replay_rows.extend(
                    row
                    for lease in leases
                    for row in range(lease.item.logical_start, lease.item.logical_end)
                )
        pending_rows = [
            row
            for unit in restored_planner._pending_units
            for row in range(unit.logical_start, unit.logical_start + unit.cardinality)
        ]
        assert sorted(replay_rows + pending_rows) == list(range(consumed_steps * 4, 14))
        for lease in old_leases:
            with pytest.raises(RuntimeError, match="stale"):
                await restored_control.work_ledger.acknowledge(
                    (SourceWorkAcknowledgement(lease.item.key, lease.lease_id),)
                )
    finally:
        await planner.close()
        await restored_planner.close()


@pytest.mark.asyncio
async def test_planner_restore_rejects_missing_initial_cursor(source_layout):
    request = source_layout(demands=(1,), cardinality=2, terminal=False).request
    result = SourceScanResult(
        source_identity=request.source_identity,
        view_identity=request.view_identity,
        schema_digest=request.schema_digest,
        lineage_digest=request.lineage_digest,
        units=request.units,
        unit_cursors=(InMemorySourceScanAdapter.cursor(0),),
        records=request.cardinality,
        next_cursor=InMemorySourceScanAdapter.cursor(request.cardinality),
        status=SourceScanStatus.READY,
    )
    planner = IngestionLayoutPlanningComponent(
        "initial-cursor", topology=request.topology
    )
    restored = IngestionLayoutPlanningComponent(
        "initial-cursor", topology=request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    cut = CheckpointCut.for_position("cut", 1)
    try:
        await planner.plan_source_scan(
            InMemorySourceScanAdapter((result,)),
            _fixed_initial_cursor(InMemorySourceScanAdapter.cursor(0)),
            ledger,
            InMemorySourceScanAdapter.cursor(0),
            budget=SourceScanBudget(2, 1),
            topology=request.topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
        state = dict(planner.checkpoint_state(cut))
        state["initial_cursor"] = None
        with pytest.raises(ValueError, match="initial cursor"):
            restored.restore_state(state, cut)
    finally:
        await planner.close()
        await restored.close()


@pytest.mark.asyncio
async def test_planner_restore_rejects_changed_binding_identity(source_layout):
    request = source_layout(demands=(1,), cardinality=2, terminal=False).request
    result = SourceScanResult(
        source_identity=request.source_identity,
        view_identity=request.view_identity,
        schema_digest=request.schema_digest,
        lineage_digest=request.lineage_digest,
        units=request.units,
        unit_cursors=(InMemorySourceScanAdapter.cursor(0),),
        records=request.cardinality,
        next_cursor=InMemorySourceScanAdapter.cursor(request.cardinality),
        status=SourceScanStatus.READY,
    )
    identity = SourceBindingIdentity("config-a", "source", "finite", "native-a")
    planner = IngestionLayoutPlanningComponent("binding-a", topology=request.topology)
    restored = IngestionLayoutPlanningComponent("binding-b", topology=request.topology)
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    cut = CheckpointCut.for_position("cut", 1)
    try:
        planner.bind_source_identity(identity)
        await planner.plan_source_scan(
            InMemorySourceScanAdapter((result,)),
            _fixed_initial_cursor(InMemorySourceScanAdapter.cursor(0)),
            ledger,
            InMemorySourceScanAdapter.cursor(0),
            budget=SourceScanBudget(2, 1),
            topology=request.topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
        state = planner.checkpoint_state(cut)
        restored.bind_source_identity(
            SourceBindingIdentity("config-b", "source", "finite", "native-b")
        )
        with pytest.raises(ValueError, match="binding identity"):
            restored.restore_state(state, cut)
    finally:
        await planner.close()
        await restored.close()


@pytest.mark.asyncio
async def test_planner_restore_installs_binding_identity_without_scan_view(
    source_layout,
):
    request = source_layout(demands=(1,), cardinality=2, terminal=False).request
    identity = SourceBindingIdentity("config", "source", "continuous", "native")
    planner = IngestionLayoutPlanningComponent(
        "binding-empty", topology=request.topology
    )
    restored = IngestionLayoutPlanningComponent(
        "binding-empty", topology=request.topology
    )
    cut = CheckpointCut.for_position("cut", 0)
    try:
        planner.bind_source_identity(identity)
        state = planner.checkpoint_state(cut)
        assert state["cursor"] is None
        assert state["binding_identity"] == identity.to_checkpoint()

        restored.restore_state(state, cut)

        assert restored._binding_identity == identity
        assert restored.checkpoint_state(cut) == state
    finally:
        await planner.close()
        await restored.close()


@pytest.mark.asyncio
async def test_planner_restore_rejects_unbound_checkpoint_for_live_binding(
    source_layout,
):
    request = source_layout(demands=(1,), cardinality=2, terminal=False).request
    planner = IngestionLayoutPlanningComponent(
        "binding-null", topology=request.topology
    )
    restored = IngestionLayoutPlanningComponent(
        "binding-null", topology=request.topology
    )
    cut = CheckpointCut.for_position("cut", 0)
    try:
        state = planner.checkpoint_state(cut)
        restored.bind_source_identity(
            SourceBindingIdentity("config", "source", "continuous", "native")
        )
        with pytest.raises(ValueError, match="binding identity"):
            restored.restore_state(state, cut)
    finally:
        await planner.close()
        await restored.close()
