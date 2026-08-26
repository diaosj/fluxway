import asyncio

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointComponentSpec,
    CheckpointCut,
    CheckpointSaveScope,
    CheckpointSaveSpec,
    TopologySnapshot,
    WorkerSaveResult,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponent,
    CheckpointComponentFinalize,
    CheckpointComponentRegistration,
    CheckpointComponentSave,
)
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


class _Finalizer(CheckpointComponent, CheckpointComponentFinalize):
    def __init__(self, events, fail_rank=None):
        self._events = events
        self._fail_rank = fail_rank
        self._descriptor = CheckpointComponentSpec("algorithm-finalize")

    @property
    def descriptor(self):
        return self._descriptor

    async def finalize_save(self, session):
        self._events.append(("finalize", session.rank))
        if session.rank == self._fail_rank:
            raise RuntimeError("finalizer failed")

    def registration(self):
        return CheckpointComponentRegistration(component=self, finalize=self)


class _LeaderStorage(CheckpointComponent, CheckpointComponentSave):
    def __init__(self, events):
        self._events = events
        self._descriptor = CheckpointComponentSpec(
            "storage",
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
        )

    @property
    def descriptor(self):
        return self._descriptor

    async def prepare_save(self, session):
        self._events.append(("prepare", session.rank))

    async def save(self, session):
        self._events.append(("save", session.rank))
        return WorkerSaveResult()

    async def finish_save(self, session):
        self._events.append(("finish", session.rank))

    async def commit_save(self, session):
        raise AssertionError("not exercised")

    async def rollback_save(self, session):
        self._events.append(("rollback", session.rank))

    async def publish_save(self, session):
        self._events.append(("publish", session.rank))

    def registration(self):
        return CheckpointComponentRegistration(component=self, save=self)


def _coordinator(rank, registrations):
    return WorkerCheckpointCoordinator(
        components=registrations,
        operator_instance=object(),
        repository_root="/tmp/checkpoint-finalize-test",
        spmd_id="test",
        rank=rank,
        source_topology=TopologySnapshot(2, ("dp",), (2,), "test-v1"),
    )


async def _prepare(coordinator):
    await coordinator.prepare_save(
        "checkpoint",
        "attempt",
        CheckpointCommitPermit("attempt", 0, b"permit"),
        CheckpointCut.for_step("cut", 4),
    )


@pytest.mark.asyncio
async def test_finalizers_run_on_every_rank_before_leader_storage_prepare():
    events = []
    finalizer = _Finalizer(events)
    storage = _LeaderStorage(events)
    workers = tuple(
        _coordinator(rank, (finalizer.registration(), storage.registration()))
        for rank in range(2)
    )

    await asyncio.gather(*(_prepare(worker) for worker in workers))

    assert sorted(event for event in events if event[0] == "finalize") == [
        ("finalize", 0),
        ("finalize", 1),
    ]
    assert [event for event in events if event[0] == "prepare"] == [("prepare", 0)]
    assert not [event for event in events if event[0] == "save"]

    await asyncio.gather(*(worker.save() for worker in workers))
    assert [event for event in events if event[0] == "save"] == [
        ("save", 0),
        ("save", 1),
    ]


@pytest.mark.asyncio
async def test_finalizer_failure_skips_storage_prepare_and_clears_transaction():
    events = []
    finalizer = _Finalizer(events, fail_rank=0)
    storage = _LeaderStorage(events)
    worker = _coordinator(0, (finalizer.registration(), storage.registration()))

    with pytest.raises(RuntimeError, match="finalizer failed"):
        await _prepare(worker)

    assert events == [("finalize", 0)]
    assert worker.state is None
