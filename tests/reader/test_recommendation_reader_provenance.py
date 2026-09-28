import asyncio

import pytest

from fluxway.common.decorators import _inject_context_properties, operator_context
from fluxway.common.ingestion import (
    LayoutUnitDescriptor,
    SourceWorkAssignment,
    SourceWorkItem,
)
from fluxway.common.types.configs import OperatorContext
from fluxway.example.recommendation.reader import RecommendationReader


def _static_args() -> dict[str, int]:
    return {
        "batch_size": 4,
        "num_samples": 100,
        "vocab_size_user": 10,
        "vocab_size_item": 20,
        "max_seq_len": 5,
    }


def _work(
    *, step: int, global_start: int, global_batch_size: int
) -> tuple[SourceWorkItem, ...]:
    descriptor = LayoutUnitDescriptor(
        unit_id=f"unit-{step}-{global_start}",
        revision_identity="recommendation:test",
        cardinality=4,
        cardinality_metric="samples",
        split_policy="SplittableSpan",
        locality_key="recommendation",
        physical_identity="recommendation:test",
        logical_start=global_start,
        opaque_token=b"test",
    )
    assignment = SourceWorkAssignment.create(
        epoch=0,
        step=step,
        source_identity="recommendation",
        schema_digest="recommendation-schema-v1",
        descriptor=descriptor,
        global_start=global_start,
        global_batch_size=global_batch_size,
    )
    return (SourceWorkItem(assignment, global_start, global_start + 4),)


def _reader(static_args: dict[str, int]) -> RecommendationReader:
    _inject_context_properties(RecommendationReader)
    reader = object.__new__(RecommendationReader)
    reader.static_args = static_args
    RecommendationReader.__init__(reader)
    return reader


def _read(reader: RecommendationReader, context: OperatorContext):
    with operator_context(
        static_args=reader.static_args,
        rank=context.physical_rank,
        world_size=context.physical_world_size,
        operator_ctx=context,
    ):
        return asyncio.run(RecommendationReader.__dict__["read"].func(reader))


def test_recommendation_reader_requires_operator_context():
    reader = _reader(_static_args())
    with pytest.raises(
        RuntimeError,
        match="RecommendationReader.read requires an active OperatorContext",
    ):
        asyncio.run(RecommendationReader.__dict__["read"].func(reader))


def test_recommendation_reader_provenance_comes_from_canonical_work():
    reader = _reader(_static_args())
    context = OperatorContext(
        physical_rank=1,
        physical_world_size=4,
        logical_rank=1,
        logical_world_size=4,
        step=3,
        source_work_items=_work(step=3, global_start=52, global_batch_size=16),
    )
    batch = _read(reader, context)
    assert batch is not None
    origin = batch.active.user_id.origin_indices
    assert origin.materialize().tolist() == [52, 53, 54, 55]
    assert batch.active.user_id.global_shape == (16, 1)


def test_recommendation_reader_interleaved_provenance_comes_from_canonical_work():
    reader = _reader(_static_args())
    context = OperatorContext(
        physical_rank=2,
        physical_world_size=4,
        logical_rank=0,
        logical_world_size=2,
        step=1,
        is_interleaved=True,
        source_work_items=_work(step=1, global_start=8, global_batch_size=8),
    )
    batch = _read(reader, context)
    assert batch is not None
    assert batch.active.user_id.origin_indices.materialize().tolist() == [8, 9, 10, 11]
    assert batch.active.user_id.global_shape == (8, 1)
    assert batch.active.hist_click_item_seq.global_shape == (8, 5)
    assert batch.active.hist_click_item_seq.nested_splits
    assert batch.active.hist_click_item_seq.flat_values.ndim == 1
    assert batch.passive.target_target_id.nested_splits
    assert batch.labels.is_click.global_shape == (8,)
