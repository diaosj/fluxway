from __future__ import annotations

from dataclasses import replace

import pytest

from fluxway.common.ingestion import LayoutUnitDescriptor
from fluxway.control_plane.ingestion.compiler import (
    AlgebraicLayoutCompiler,
    AssignmentPolicy,
    LayoutCompileRequest,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
)
from fluxway.control_plane.ingestion.planning_component import (
    _split_units_at_cardinality,
)


def _file_units(fragment_rows):
    units = []
    for file_index, row_count in enumerate((8, 6, 13, 3)):
        identity = f"file-{file_index}"
        for row in range(0, row_count, fragment_rows):
            units.append(
                LayoutUnitDescriptor(
                    unit_id=f"{identity}:{row}",
                    revision_identity=f"commit-{file_index // 2}",
                    cardinality=min(fragment_rows, row_count - row),
                    cardinality_metric="rows",
                    split_policy="SplittableSpan",
                    locality_key=identity,
                    storage_locator=identity,
                    logical_start=row,
                    opaque_token=identity,
                )
            )
    return tuple(units)


def _topology(demands):
    return ReaderTopologyDescriptor(
        global_cardinality=sum(demands),
        groups=tuple(
            ReaderGroupDemand(
                worker_id=f"reader-{index}",
                ranks=(RankDemand(f"rank-{index}", demand),),
                node_id=f"node-{index % 2}",
            )
            for index, demand in enumerate(demands)
        ),
    )


def _global_records(compiled):
    records = {}
    for assignment in compiled.work_epoch.assignments:
        unit = assignment.descriptor
        for offset in range(unit.cardinality):
            position = assignment.global_start + offset
            assert position not in records
            records[position] = (unit.storage_locator, unit.logical_start + offset)
    assigned_rows = sorted(records.values())
    worker_rows = sorted(
        (segment.storage_locator, row)
        for layout in compiled.worker_layouts
        for segment in layout.segments
        for row in range(segment.logical_start, segment.logical_end)
    )
    assert worker_rows == assigned_rows
    return records


@pytest.mark.parametrize("consumed_steps", [0, 1, 2, 3, 6, 7])
@pytest.mark.parametrize("window_steps", [1, 2, 5])
@pytest.mark.parametrize("demands", [(4,), (2, 2), (1, 1, 1, 1)])
@pytest.mark.parametrize("fragment_rows", [1, 5, 13])
@pytest.mark.parametrize("affinity", [False, True])
def test_replanning_preserves_step_records_and_tail(
    consumed_steps, window_steps, demands, fragment_rows, affinity
):
    compiler = AlgebraicLayoutCompiler()
    request = LayoutCompileRequest(
        source_identity="table",
        view_identity="fixed-view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=_file_units(13),
        topology=_topology((2, 2)),
        tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
    )
    original = _global_records(compiler.compile(request))
    expected_rows = [
        (f"file-{index}", row)
        for index, count in enumerate((8, 6, 13, 3))
        for row in range(count)
    ]
    assert original == dict(enumerate(expected_rows[:28]))
    batch = request.topology.global_cardinality
    _, remaining = _split_units_at_cardinality(
        _file_units(fragment_rows), consumed_steps * batch
    )
    position = consumed_steps
    epoch = 0
    replay = {}
    while sum(unit.cardinality for unit in remaining) >= batch:
        steps = min(window_steps, sum(unit.cardinality for unit in remaining) // batch)
        window, remaining = _split_units_at_cardinality(remaining, steps * batch)
        compiled = compiler.compile(
            replace(
                request,
                units=window,
                topology=_topology(demands),
                assignment_policy=AssignmentPolicy(
                    affinity_contiguous_chunks=affinity,
                    affinity_same_file=affinity,
                    node_affinity_hint=affinity,
                ),
            ),
            epoch=epoch,
            start_step=position,
        )
        records = _global_records(compiled)
        assert not replay.keys() & records.keys()
        replay.update(records)
        position += steps
        epoch += 1
    assert replay == {
        index: value
        for index, value in original.items()
        if index >= consumed_steps * batch
    }
    assert [
        (unit.storage_locator, row)
        for unit in remaining
        for row in range(unit.logical_start, unit.logical_start + unit.cardinality)
    ] == [("file-3", 1), ("file-3", 2)]
