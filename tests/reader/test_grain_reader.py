"""
Unit tests for GrainReader.

GrainReader.read() is a @operator-decorated staticmethod backed by a
per-owner instance within the worker-scoped operator container.
Unit tests that need to exercise different rank configurations must
instantiate GrainReader directly and call next_item() / _create_data_loader()
rather than going through the staticmethod wrapper.
"""

import numpy as np
import pytest

try:
    import grain.python as grain
except ImportError:
    grain = None

from dataclasses import dataclass

from fluxway.common.decorators import data_type
from fluxway.common.decorators import spmd_source
from fluxway.reader.grain_reader import GrainReader, StridedIndexSampler
from fluxway.spmd.worker.operator_manager.dispatcher import ExecutionDispatcher
from fluxway.common.execution_layout import compile_execution_layout
from fluxway.common.mesh import DeviceMesh


@data_type()
@dataclass
class SimpleBatch:
    features: np.ndarray


class MockDataSource:
    def __init__(self, size: int):
        self._data = [{"features": np.array([i])} for i in range(size)]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]


def add_one_to_features(record):
    return {"features": record["features"] + 1}


def _make_reader(data_source, rank: int, world_size: int, **kwargs) -> GrainReader:
    """
    Build a GrainReader with its iterator initialised for the given rank.

    Uses the same path as the production worker: injects static_args / rank /
    world_size as instance attributes, then calls _create_data_loader directly.
    """
    reader = GrainReader()
    reader.static_args = {
        "data_source": data_source,
        "shuffle": False,
        "num_epochs": 1,
        "output_data_type": SimpleBatch,
        **kwargs,
    }
    reader.rank = rank
    reader.world_size = world_size
    reader._create_data_loader(reader.static_args, rank, world_size)
    return reader


def _drain(reader: GrainReader):
    results = []
    while True:
        batch = reader.next_item()
        if batch is None:
            break
        results.append(batch)
    return results


@spmd_source(
    mesh_axis_names=("worker",),
    mesh_axis_shapes=(2,),
    interleaved=True,
    interleave_unit_size=2,
)
class _BatchedInterleavedReader(GrainReader):
    def records_per_position(self) -> int:
        return 3


def _records_before_checkpoint(
    *, world_size: int, checkpoint_position: int
) -> set[int]:
    records: set[int] = set()
    for rank in range(world_size):
        reader = object.__new__(_BatchedInterleavedReader)
        reader.rank = rank
        reader.world_size = world_size
        sampler = StridedIndexSampler(
            range(100),
            shard_index=rank,
            shard_count=world_size,
            block_size=reader.records_per_position(),
            drop_remainder=False,
        )
        records.update(
            sampler[index]
            for index in range(reader._local_items_before(checkpoint_position))
        )
    return records


def test_virtual_interleave_checkpoint_reshards_record_blocks() -> None:
    saved_records = _records_before_checkpoint(
        world_size=4,
        checkpoint_position=3,
    )
    restored_records = _records_before_checkpoint(
        world_size=6,
        checkpoint_position=3,
    )

    assert saved_records == set(range(18))
    assert restored_records == saved_records


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
class TestGrainReaderSharding:
    """Verify that GrainReader partitions data correctly across ranks."""

    def test_two_shards_cover_all_records(self):
        ds = MockDataSource(10)
        rank0 = _drain(_make_reader(ds, rank=0, world_size=2))
        rank1 = _drain(_make_reader(ds, rank=1, world_size=2))

        assert len(rank0) == 5
        assert len(rank1) == 5
        assert all(isinstance(b, SimpleBatch) for b in rank0 + rank1)

        ids0 = {int(b.features[0]) for b in rank0}
        ids1 = {int(b.features[0]) for b in rank1}
        assert ids0.isdisjoint(ids1), "Shards must not overlap"
        assert ids0 | ids1 == set(range(10)), "Shards must cover all records"

    def test_single_shard_reads_all_records(self):
        ds = MockDataSource(4)
        batches = _drain(_make_reader(ds, rank=0, world_size=1))

        assert len(batches) == 4
        assert all(isinstance(b, SimpleBatch) for b in batches)

    def test_interleaved_readers_follow_global_batch_order(self):
        ds = MockDataSource(10)
        readers = [_make_reader(ds, rank=rank, world_size=2) for rank in range(2)]
        dispatcher = ExecutionDispatcher(
            compile_execution_layout(DeviceMesh(("worker",), (2,)), unit_size=1)
        )

        actual = []
        for step in range(10):
            owners = [
                rank for rank in range(2) if dispatcher.should_execute(rank, 2, step)
            ]
            assert len(owners) == 1
            batch = readers[owners[0]].next_item()
            assert batch is not None
            actual.append(int(batch.features[0]))

        assert actual == list(range(10))

    def test_interleaved_units_read_consecutive_logical_shards(self):
        ds = MockDataSource(12)
        readers = [_make_reader(ds, rank=rank, world_size=4) for rank in range(4)]
        dispatcher = ExecutionDispatcher(
            compile_execution_layout(DeviceMesh(("worker",), (4,)), unit_size=2)
        )

        actual = []
        for step in range(6):
            owners = [
                rank for rank in range(4) if dispatcher.should_execute(rank, 4, step)
            ]
            assert len(owners) == 2
            for owner in owners:
                batch = readers[owner].next_item()
                assert batch is not None
                actual.append(int(batch.features[0]))

        assert actual == list(range(12))

    def test_interleaved_num_logical_shards_mismatch_causes_duplicate_shards(self):
        ds = MockDataSource(20)
        # When num_logical_shards=2 < physical_world_size=4:
        # rank 0 -> shard 0, rank 1 -> shard 1, rank 2 -> shard 0, rank 3 -> shard 1
        readers = [
            _make_reader(ds, rank=rank, world_size=4, num_logical_shards=2)
            for rank in range(4)
        ]
        dispatcher = ExecutionDispatcher(
            compile_execution_layout(DeviceMesh(("worker",), (4,)), unit_size=2)
        )

        # Step 0 (unit 0: ranks 0, 1) reads shard 0 and 1
        owners_step0 = [
            rank for rank in range(4) if dispatcher.should_execute(rank, 4, 0)
        ]
        items_step0 = [int(readers[r].next_item().features[0]) for r in owners_step0]

        # Step 1 (unit 1: ranks 2, 3) reads shard 0 and 1 again
        owners_step1 = [
            rank for rank in range(4) if dispatcher.should_execute(rank, 4, 1)
        ]
        items_step1 = [int(readers[r].next_item().features[0]) for r in owners_step1]

        # Demonstrates that misaligned num_logical_shards produces duplicate samples
        assert items_step0 == items_step1 == [0, 1]

    def test_strided_reader_checkpoint_restores_local_cursor(self):
        ds = MockDataSource(10)
        reader = _make_reader(ds, rank=0, world_size=2)
        assert [int(reader.next_item().features[0]) for _ in range(2)] == [0, 2]
        from fluxway.common.types import (
            SaveSession,
            RestoreSession,
            CheckpointCommitPermit,
            CheckpointCut,
            TopologySnapshot,
        )

        topology = TopologySnapshot(2, ("data",), (2,), "v1")
        save_session = SaveSession(
            instance=reader,
            attempt_id="tx",
            checkpoint_id="checkpoint",
            commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
            cut=CheckpointCut.for_step("cut", 2),
            spmd_id="reader",
            rank=0,
            source_topology=topology,
            component_id="reader",
        )
        state = reader.save_state(save_session)

        restored = _make_reader(ds, rank=0, world_size=2)
        restored.restore_checkpoint_state(
            state,
            RestoreSession(
                instance=restored,
                checkpoint_id=save_session.checkpoint_id,
                cut=save_session.cut,
                spmd_id=save_session.spmd_id,
                rank=save_session.rank,
                saved_topology=topology,
                restore_topology=topology,
                component_id=save_session.component_id,
                component_checkpoint=None,
            ),
        )
        batch = restored.next_item()

        assert batch is not None
        assert int(batch.features[0]) == 4

    def test_strided_reader_drop_remainder_equalizes_shard_lengths(self):
        ds = MockDataSource(5)
        rank0 = _drain(_make_reader(ds, rank=0, world_size=2, drop_remainder=True))
        rank1 = _drain(_make_reader(ds, rank=1, world_size=2, drop_remainder=True))

        assert [int(batch.features[0]) for batch in rank0] == [0, 2]
        assert [int(batch.features[0]) for batch in rank1] == [1, 3]

    def test_callable_transformations_preserve_record_map_semantics(self):
        ds = MockDataSource(4)
        reader = _make_reader(
            ds,
            rank=0,
            world_size=1,
            transformations=[add_one_to_features],
        )

        assert [int(batch.features[0]) for batch in _drain(reader)] == [1, 2, 3, 4]


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
class TestGrainReaderOutputType:
    """Verify that output_data_type wraps raw dicts into the target dataclass."""

    def test_output_type_conversion(self):
        ds = MockDataSource(4)
        reader = _make_reader(ds, rank=0, world_size=1)
        batch = reader.next_item()

        assert batch is not None
        assert isinstance(batch, SimpleBatch)
        np.testing.assert_array_equal(batch.features, np.array([0]))
