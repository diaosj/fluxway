from __future__ import annotations

from dataclasses import replace

import pytest

from fluxway.common.ingestion import (
    SourceWorkAssignment,
    SourceWorkItem,
    SourceWorkKind,
    source_work_global_valid_count,
)
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    IngestionLayoutPlanningComponent,
    TailLayoutMode,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from tests.control_plane.test_ingestion_compiler import request


def test_global_valid_count_is_part_of_assignment_identity():
    compiled = AlgebraicLayoutCompiler().compile(
        request(
            cardinality=5,
            worker_demands=(8,),
            tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
        )
    )
    original = compiled.work_epoch.assignments[0]
    changed = SourceWorkAssignment.create(
        epoch=original.epoch,
        step=original.step,
        source_identity=original.source_identity,
        view_identity=original.view_identity,
        schema_digest=original.schema_digest,
        lineage_digest=original.lineage_digest,
        descriptor=original.descriptor,
        global_start=original.global_start,
        global_batch_size=original.global_batch_size,
        global_valid_count=6,
    )
    assert changed.key != original.key


@pytest.mark.parametrize("rows", [5, 8])
def test_padding_classification_is_explicit_and_survives_request_codec(rows):
    from fluxway.control_plane.ingestion.compiler import (
        _request_from_dict,
        _request_to_dict,
    )
    from fluxway.common.metadata import pack_metadata, unpack_metadata

    source_request = request(
        cardinality=rows,
        worker_demands=(8,),
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
    )
    source_request = replace(
        source_request,
        units=(
            replace(
                source_request.units[0],
                storage_locator="__VIRTUAL_PAD__",
                revision_identity=source_request.view_identity,
                opaque_token="pad:virtual",
            ),
        ),
    )
    restored = _request_from_dict(
        unpack_metadata(pack_metadata(_request_to_dict(source_request)))
    )
    assert restored == source_request
    assert not restored.units[0].is_synthetic_padding
    compiled = AlgebraicLayoutCompiler().compile(restored)
    assert {value.global_valid_count for value in compiled.work_epoch.assignments} == {
        rows
    }
    assert (
        compiled.work_epoch.assignments[0].descriptor.work_kind
        is SourceWorkKind.PHYSICAL
    )
    if rows < 8:
        assert (
            compiled.work_epoch.assignments[-1].descriptor.work_kind
            is SourceWorkKind.SYNTHETIC_PADDING
        )
    assert (
        sum(
            interval.logical_end - interval.logical_start
            for interval in compiled.canonical_work_intervals
        )
        == rows
    )


def test_assignment_validity_is_mandatory():
    compiled = AlgebraicLayoutCompiler().compile(
        request(cardinality=8, worker_demands=(8,))
    )
    original = compiled.work_epoch.assignments[0]
    values = {name: getattr(original, name) for name in original.__dataclass_fields__}
    values.pop("key")
    values.pop("global_valid_count")
    with pytest.raises(TypeError, match="global_valid_count"):
        SourceWorkAssignment.create(**values)


def test_shared_work_batch_rejects_conflicting_count_and_provenance():
    compiler = AlgebraicLayoutCompiler()
    source_request = request(
        cardinality=5,
        worker_demands=(8,),
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
    )
    compiled = compiler.compile(source_request)
    values = tuple(
        SourceWorkItem(
            value,
            value.descriptor.logical_start,
            value.descriptor.logical_start + value.descriptor.cardinality,
        )
        for value in compiled.work_epoch.assignments
    )
    assert source_work_global_valid_count(values) == 5
    original = values[0].assignment
    changed = SourceWorkAssignment.create(
        epoch=original.epoch,
        step=original.step,
        source_identity=original.source_identity,
        view_identity=original.view_identity,
        schema_digest=original.schema_digest,
        lineage_digest=original.lineage_digest,
        descriptor=original.descriptor,
        global_start=original.global_start,
        global_batch_size=original.global_batch_size,
        global_valid_count=6,
    )
    with pytest.raises(
        ValueError, match="conflicting batch provenance or global valid count"
    ):
        source_work_global_valid_count(
            (replace(values[0], assignment=changed), values[1])
        )
    other_step = compiler.compile(source_request, start_step=1).work_epoch.assignments[
        0
    ]
    other = SourceWorkItem(
        other_step,
        other_step.descriptor.logical_start,
        other_step.descriptor.logical_start + other_step.descriptor.cardinality,
    )
    with pytest.raises(ValueError, match="conflicting batch provenance"):
        source_work_global_valid_count((values[0], other))


def test_logical_epoch_rejects_counts_inconsistent_with_real_cardinality():
    compiled = AlgebraicLayoutCompiler().compile(
        request(
            cardinality=5,
            worker_demands=(8,),
            tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
        )
    )
    with pytest.raises(ValueError, match="global valid count"):
        replace(
            compiled.work_epoch,
            assignments=tuple(
                replace(value, global_valid_count=6)
                for value in compiled.work_epoch.assignments
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("rows", "limit"), [(9, 1), (17, 2)])
async def test_direct_bounded_planner_publishes_terminal_remainder(rows, limit):
    source_request = request(
        cardinality=rows,
        worker_demands=(8,),
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
    )
    planner = IngestionLayoutPlanningComponent(
        "direct-tail", topology=source_request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    try:
        first = await planner.plan_and_publish(
            ledger, source_request, epoch=0, start_position=0, max_steps=limit
        )
        assert not first.epoch.terminal
        assert first.epoch.end_position == limit
        tail = await planner.plan_next_window(ledger, max_steps=limit)
        assert tail is not None
        assert tail.epoch.terminal
        assert tail.epoch.start_position == limit
        assert tail.epoch.end_position == limit + 1
        assert {item.global_valid_count for item in ledger.epochs[-1].assignments} == {
            1
        }
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_direct_retained_tail_waits_for_more_source_work():
    source_request = request(
        cardinality=9,
        worker_demands=(8,),
        tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
    )
    planner = IngestionLayoutPlanningComponent(
        "retain-tail", topology=source_request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    try:
        first = await planner.plan_and_publish(
            ledger, source_request, epoch=0, start_position=0, max_steps=1
        )
        assert not first.epoch.terminal
        assert await planner.plan_next_window(ledger, max_steps=1) is None
        assert len(ledger.epochs) == 1
    finally:
        await planner.close()
