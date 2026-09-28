from __future__ import annotations


import pytest

from fluxway.common.types import CheckpointCut
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    TailLayoutMode,
    IngestionLayoutPlanningComponent,
    LayoutCompileRequest,
    LayoutUnitDescriptor,
    AssignmentPolicy,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
)
from fluxway.control_plane.ingestion.compiler import (
    _request_from_dict,
    _request_to_dict,
)


def request(
    *,
    cardinality: int = 10,
    metric: str = "objects",
    worker_demands: tuple[int, ...] = (5, 5),
    tail_layout_mode: TailLayoutMode | str = TailLayoutMode.REJECT_REMAINDER,
    assignment_policy: AssignmentPolicy = AssignmentPolicy.default(),
) -> LayoutCompileRequest:
    groups = tuple(
        ReaderGroupDemand(
            f"worker-{index}",
            (RankDemand(f"rank-{index}", demand),),
        )
        for index, demand in enumerate(worker_demands)
    )
    return LayoutCompileRequest(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(
            LayoutUnitDescriptor(
                "unit",
                "revision",
                cardinality,
                metric,
                "SplittableSpan",
                "node",
                "object",
                0,
                "opaque",
            ),
        ),
        topology=ReaderTopologyDescriptor(sum(worker_demands), groups),
        tail_layout_mode=tail_layout_mode,
        assignment_policy=assignment_policy,
    )


@pytest.mark.parametrize(
    "metric", ["objects", "documents", "samples", "frames", "tokens", "chunks"]
)
def test_compiler_is_cardinality_metric_neutral(metric: str) -> None:
    compiled = AlgebraicLayoutCompiler().compile(request(metric=metric))

    assert compiled.cardinality == 10
    assert {
        segment.cardinality_metric
        for layout in compiled.worker_layouts
        for segment in layout.segments
    } == {metric}


def test_layout_request_preserves_policy_input_without_post_init_mutation() -> None:
    string_request = request(tail_layout_mode="reject_remainder")
    enum_request = request(tail_layout_mode=TailLayoutMode.REJECT_REMAINDER)
    assert string_request.tail_layout_mode == "reject_remainder"
    assert enum_request.tail_layout_mode is TailLayoutMode.REJECT_REMAINDER


def test_node_affinity_hint_is_optional_and_binds_attested_topology() -> None:
    logical_topology = ReaderTopologyDescriptor(
        4,
        (
            ReaderGroupDemand("worker-0", (RankDemand("rank-0", 2),)),
            ReaderGroupDemand("worker-1", (RankDemand("rank-1", 2),)),
        ),
    )
    logical_request = request(
        cardinality=4,
        worker_demands=(2, 2),
        assignment_policy=AssignmentPolicy.default(),
    )
    assert logical_request.topology == logical_request.topology
    compiled = AlgebraicLayoutCompiler().compile(logical_request)
    assert {
        segment.worker_id
        for layout in compiled.worker_layouts
        for segment in layout.segments
    } == {"worker-0", "worker-1"}

    bound_topology = logical_topology.bind_node_ids(("node-a", "node-b"))
    hinted = request(
        cardinality=4,
        worker_demands=(2, 2),
        assignment_policy=AssignmentPolicy(node_affinity_hint=True),
    )
    hinted = LayoutCompileRequest(
        source_identity=hinted.source_identity,
        view_identity=hinted.view_identity,
        schema_digest=hinted.schema_digest,
        lineage_digest=hinted.lineage_digest,
        units=hinted.units,
        topology=bound_topology,
        tail_layout_mode=hinted.tail_layout_mode,
        assignment_policy=hinted.assignment_policy,
    )
    assert AlgebraicLayoutCompiler().compile(hinted).layout_digest

    with pytest.raises(ValueError, match="complete active node identities"):
        AlgebraicLayoutCompiler().compile(
            LayoutCompileRequest(
                source_identity=hinted.source_identity,
                view_identity=hinted.view_identity,
                schema_digest=hinted.schema_digest,
                lineage_digest=hinted.lineage_digest,
                units=hinted.units,
                topology=logical_topology,
                tail_layout_mode=hinted.tail_layout_mode,
                assignment_policy=hinted.assignment_policy,
            )
        )


def test_layout_request_codec_uses_canonical_tail_layout_mode_member() -> None:
    encoded_from_string = _request_to_dict(request(tail_layout_mode="reject_remainder"))
    encoded_from_enum = _request_to_dict(
        request(tail_layout_mode=TailLayoutMode.REJECT_REMAINDER)
    )
    assert encoded_from_string["tail_layout_mode"] is TailLayoutMode.REJECT_REMAINDER
    assert encoded_from_enum["tail_layout_mode"] is TailLayoutMode.REJECT_REMAINDER

    restored = _request_from_dict(encoded_from_string)
    assert restored.tail_layout_mode is TailLayoutMode.REJECT_REMAINDER


def test_layout_epoch_separates_execution_positions_from_source_cardinality() -> None:
    compiled = AlgebraicLayoutCompiler().compile(
        request(cardinality=12, worker_demands=(8, 4))
    )

    envelope = compiled.to_source_layout_activation(
        epoch=0,
        start_position=0,
        end_position=1,
        terminal=True,
    )

    assert envelope.epoch.start_position == 0
    assert envelope.epoch.end_position == 1
    assert envelope.epoch.assigned_cardinality == 12
    assert tuple(
        descriptor.expected_cardinality for descriptor in envelope.descriptors
    ) == (8, 4)


def test_layout_epoch_requires_logical_step_interval() -> None:
    compiled = AlgebraicLayoutCompiler().compile(
        request(cardinality=4, worker_demands=(1, 1))
    )

    with pytest.raises(ValueError, match="step_count"):
        compiled.to_source_layout_activation(
            epoch=0,
            start_position=7,
            end_position=8,
            terminal=True,
        )


def test_compiler_is_deterministic_and_preserves_work_across_worker_resize() -> None:
    two_workers = AlgebraicLayoutCompiler().compile(request(worker_demands=(5, 5)))
    three_workers = AlgebraicLayoutCompiler().compile(request(worker_demands=(4, 3, 3)))

    assert two_workers.layout_digest != three_workers.layout_digest
    assert _logical_ordinals(two_workers) == set(range(10))
    assert _logical_ordinals(three_workers) == set(range(10))
    assert AlgebraicLayoutCompiler().compile(request()) == two_workers


def test_compiler_rejects_atomic_cardinality_and_unimplemented_policies() -> None:
    with pytest.raises(ValueError, match="AtomicItem cardinality must be one"):
        LayoutUnitDescriptor(
            "unit",
            "revision",
            2,
            "objects",
            "AtomicItem",
            "node",
            "object",
            0,
            "opaque",
        )


def test_compiler_rejects_duplicate_canonical_work_identity() -> None:
    duplicate_units = (
        LayoutUnitDescriptor(
            "unit-0",
            "revision",
            1,
            "objects",
            "SplittableSpan",
            "node",
            "object",
            0,
            "opaque-0",
        ),
        LayoutUnitDescriptor(
            "unit-1",
            "revision",
            1,
            "objects",
            "SplittableSpan",
            "node",
            "object",
            0,
            "opaque-1",
        ),
    )
    duplicate_request = LayoutCompileRequest(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=duplicate_units,
        topology=ReaderTopologyDescriptor(
            1,
            (ReaderGroupDemand("worker-0", (RankDemand("rank-0", 1),)),),
        ),
        tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
        assignment_policy=AssignmentPolicy.default(),
    )

    with pytest.raises(ValueError, match="canonical work intervals overlap"):
        AlgebraicLayoutCompiler().compile(duplicate_request)


def test_ordered_prefix_balances_each_logical_step() -> None:
    units = tuple(
        LayoutUnitDescriptor(
            f"{file_id}-{row_group}",
            "revision",
            row_count,
            "rows",
            "SplittableSpan",
            file_id,
            file_id,
            sum(groups[:row_group]),
            f"{file_id}:{row_group}",
        )
        for file_id, groups in {
            "A": (13_421, 18_902, 11_200),
            "B": (317,),
            "C": (4_500, 7_890),
            "D": (1_450,),
        }.items()
        for row_group, row_count in enumerate(groups)
    )
    compiled = AlgebraicLayoutCompiler().compile(
        LayoutCompileRequest(
            source_identity="source",
            view_identity="view",
            schema_digest="schema",
            lineage_digest="lineage",
            units=units,
            topology=ReaderTopologyDescriptor(
                64,
                (
                    ReaderGroupDemand("worker-0", (RankDemand("rank-0", 32),)),
                    ReaderGroupDemand("worker-1", (RankDemand("rank-1", 32),)),
                ),
            ),
            tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        )
    )

    assert compiled.cardinality == 57_664
    assert compiled.discarded_cardinality == 16
    for step in range(compiled.step_count):
        for layout in compiled.worker_layouts:
            assert (
                sum(
                    segment.cardinality
                    for segment in layout.segments
                    if segment.step == step
                )
                == 32
            )


def test_default_policy_reuses_physical_reader_owners_across_cycle() -> None:
    topology = ReaderTopologyDescriptor(
        8,
        tuple(
            ReaderGroupDemand(f"worker-{index}", (RankDemand(f"rank-{index}", 4),))
            for index in range(2)
        ),
        execution_cycle=((0, 1), (1, 0)),
    )
    compiled = AlgebraicLayoutCompiler().compile(
        LayoutCompileRequest(
            source_identity="source",
            view_identity="view-owner",
            schema_digest="schema",
            lineage_digest="lineage",
            units=(
                LayoutUnitDescriptor(
                    "file",
                    "rev",
                    16,
                    "rows",
                    "SplittableSpan",
                    "local",
                    "file",
                    0,
                    "file",
                ),
            ),
            topology=topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        )
    )
    for step in range(2):
        owners = {
            segment.worker_id
            for layout in compiled.worker_layouts
            for segment in layout.segments
            if segment.step == step
        }
        assert owners == {"worker-0", "worker-1"}
    assert compiled.work_epoch.digest


def test_policy_forbidding_splits_rejects_splittable_unit() -> None:
    topology = ReaderTopologyDescriptor(
        8,
        tuple(
            ReaderGroupDemand(f"worker-{index}", (RankDemand(f"rank-{index}", 4),))
            for index in range(2)
        ),
    )
    with pytest.raises(ValueError, match="forbids splitting"):
        AlgebraicLayoutCompiler().compile(
            LayoutCompileRequest(
                source_identity="source",
                view_identity="view-no-split",
                schema_digest="schema",
                lineage_digest="lineage",
                units=(
                    LayoutUnitDescriptor(
                        "file",
                        "rev",
                        8,
                        "rows",
                        "SplittableSpan",
                        "local",
                        "file",
                        0,
                        "file",
                    ),
                ),
                topology=topology,
                tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
                assignment_policy=AssignmentPolicy(allow_unit_splitting=False),
            )
        )


def test_planning_component_checkpoint_round_trip() -> None:
    component = IngestionLayoutPlanningComponent("planner")
    compiled = component.accept(request())
    cut = CheckpointCut.for_position("cut", 0)
    state = component.checkpoint_state(cut)
    restored = IngestionLayoutPlanningComponent("planner", topology=request().topology)

    restored.restore_state(state, cut)

    assert restored.layout_digest == compiled.layout_digest
    assert restored.accept(request()) == compiled


def _logical_ordinals(compiled) -> set[int]:
    return {
        ordinal
        for interval in compiled.canonical_work_intervals
        for ordinal in range(interval.logical_start, interval.logical_end)
    }


def test_virtual_pad_tail_mode_cannot_discard_remainder() -> None:
    pad_unit = LayoutUnitDescriptor(
        unit_id="unit-pad",
        revision_identity="rev",
        cardinality=4,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key="local",
        physical_identity="__VIRTUAL_PAD__",
        logical_start=0,
        opaque_token="pad:virtual",
    )
    topology = ReaderTopologyDescriptor(
        4,
        (ReaderGroupDemand("worker-0", (RankDemand("rank-0", 4),)),),
    )
    with pytest.raises(
        ValueError,
        match="layout cannot discard remainder",
    ):
        LayoutCompileRequest(
            source_identity="source",
            view_identity="view",
            schema_digest="schema",
            lineage_digest="lineage",
            units=(pad_unit,),
            topology=topology,
            tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        )

    req_require = LayoutCompileRequest(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(pad_unit,),
        topology=topology,
        tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
        assignment_policy=AssignmentPolicy.default(),
    )
    compiled = AlgebraicLayoutCompiler().compile(req_require)
    assert compiled.step_count == 1
    assert compiled.discarded_cardinality == 0


def test_synthesize_padding_tail_mode_adds_virtual_tail() -> None:
    real_unit = LayoutUnitDescriptor(
        unit_id="unit-real",
        revision_identity="rev",
        cardinality=5,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key="local",
        physical_identity="file.parquet",
        logical_start=0,
        opaque_token="real",
    )
    topology = ReaderTopologyDescriptor(
        8,
        (ReaderGroupDemand("worker-0", (RankDemand("rank-0", 8),)),),
    )
    req = LayoutCompileRequest(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(real_unit,),
        topology=topology,
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
        assignment_policy=AssignmentPolicy.default(),
    )
    compiled = AlgebraicLayoutCompiler().compile(req)
    assert compiled.step_count == 1
    assert compiled.discarded_cardinality == 0
    worker_segments = compiled.worker_layouts[0].segments
    assert len(worker_segments) == 2
    assert worker_segments[0].physical_identity == "file.parquet"
    assert worker_segments[0].cardinality == 5
    assert worker_segments[1].physical_identity == "__VIRTUAL_PAD__"
    assert worker_segments[1].cardinality == 3
