import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointObjectCleanedUpError,
    CheckpointAssembly,
    SaveCheckpointRequest,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
    CheckpointRetentionPolicy,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
    CheckpointObjectCleanup,
)
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.pickle_component import (
    PickleCheckpointComponent,
)
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


class _Owner:
    def __init__(self) -> None:
        self.state = {"value": 1}


class _FailOnceCleanuper(CheckpointObjectCleanup):
    def __init__(self, delegate: CheckpointObjectCleanup) -> None:
        self._delegate = delegate
        self._remaining_failures = 1

    async def cleanup(self, reservation) -> None:
        if self._remaining_failures:
            self._remaining_failures -= 1
            raise OSError("injected physical cleanup failure")
        await self._delegate.cleanup(reservation)


def _component(retain_last: int = 1) -> PickleCheckpointComponent:
    return PickleCheckpointComponent(
        "model",
        lambda owner, context: owner.state,
        lambda owner, state, context: setattr(owner, "state", state),
        retention=CheckpointRetentionPolicy(retain_last=retain_last),
    )


class _Controller:
    def __init__(self, owner, topology) -> None:
        self.owner = owner
        self.topology = topology
        self._active = {}

    async def request_snapshot(
        self,
        checkpoint_id,
        attempt_id,
        permit,
        cut,
        repository_dir,
    ):
        coordinator = WorkerCheckpointCoordinator(
            (_component().registration(),),
            self.owner,
            repository_dir,
            "train",
            0,
            self.topology,
        )
        await coordinator.prepare_save(checkpoint_id, attempt_id, permit, cut)
        record = (await coordinator.save())["model"]
        await coordinator.finish_save()
        components = await coordinator.commit_save({"model": {0: record}})
        self._active[checkpoint_id] = coordinator
        return CheckpointAssembly(
            cut=cut,
            framework_state={},
            spmds={
                "train": SpmdCheckpoint(
                    source_topology=self.topology,
                    runtime_state={},
                    components=components,
                )
            },
        )

    async def publish_snapshot(self, checkpoint_id, permit):
        coordinator = self._active.pop(checkpoint_id)
        await coordinator.publish_save()
        await coordinator.close()

    async def rollback_snapshot(self, checkpoint_id, permit):
        coordinator = self._active.pop(checkpoint_id, None)
        if coordinator is not None:
            await coordinator.rollback_active()
            await coordinator.close()

    async def commit_checkpoint_state(self, step, framework_state):
        return None

    async def restore_snapshot(self, checkpoint_id, manifest, repository_dir):
        raise AssertionError("cleaned_up checkpoint reached the restore controller")


class _MultiRankController:
    def __init__(self, owners, topology) -> None:
        self.owners = owners
        self.topology = topology
        self._active = {}

    async def request_snapshot(
        self,
        checkpoint_id,
        attempt_id,
        permit,
        cut,
        repository_dir,
    ):
        coordinators = tuple(
            WorkerCheckpointCoordinator(
                (_component().registration(),),
                owner,
                repository_dir,
                "train",
                rank,
                self.topology,
            )
            for rank, owner in enumerate(self.owners)
        )
        for coordinator in coordinators:
            await coordinator.prepare_save(checkpoint_id, attempt_id, permit, cut)
        save_results = await asyncio.gather(
            *(coordinator.save() for coordinator in coordinators)
        )
        await asyncio.gather(
            *(coordinator.finish_save() for coordinator in coordinators)
        )
        gathered = {
            "model": {
                rank: rank_results["model"]
                for rank, rank_results in enumerate(save_results)
            }
        }
        publications = await asyncio.gather(
            *(coordinator.commit_save(gathered) for coordinator in coordinators)
        )
        self._active[checkpoint_id] = coordinators
        return CheckpointAssembly(
            cut=cut,
            framework_state={},
            spmds={
                "train": SpmdCheckpoint(
                    source_topology=self.topology,
                    runtime_state={},
                    components=publications[0],
                )
            },
        )

    async def publish_snapshot(self, checkpoint_id, permit):
        coordinators = self._active.pop(checkpoint_id)
        await asyncio.gather(
            *(coordinator.publish_save() for coordinator in coordinators)
        )
        await asyncio.gather(*(coordinator.close() for coordinator in coordinators))

    async def rollback_snapshot(self, checkpoint_id, permit):
        coordinators = self._active.pop(checkpoint_id, ())
        await asyncio.gather(
            *(coordinator.rollback_active() for coordinator in coordinators)
        )
        await asyncio.gather(*(coordinator.close() for coordinator in coordinators))

    async def commit_checkpoint_state(self, step, framework_state):
        return None


def _checkpoint_object_path(manifest) -> Path:
    checkpoint_object = (
        manifest.spmds["train"]
        .components["model"]
        .partitions[0]
        .record.checkpoint_objects[0]
    )
    parsed = urlparse(checkpoint_object.storage_locator.decode("utf-8"))
    if parsed.scheme != "file":
        raise AssertionError(
            "PickleCheckpointComponent returned a non-file storage_locator"
        )
    return Path(unquote(parsed.path))


def _checkpoint_object_paths(manifest) -> tuple[Path, ...]:
    publication = manifest.spmds["train"].components["model"]
    return tuple(
        Path(
            unquote(
                urlparse(
                    partition.record.checkpoint_objects[0].storage_locator.decode()
                ).path
            )
        )
        for partition in publication.partitions
    )


@pytest.mark.asyncio
async def test_retain_last_automatically_cleanups_old_physical_checkpoint_object(
    tmp_path,
):
    topology = TopologySnapshot(1, (), (), "v1")
    owner = _Owner()
    controller = _Controller(owner, topology)
    control_component = _component()
    checkpointer = FluxwayCheckpointer(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((control_component.registration(),))
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    first_path = _checkpoint_object_path(first)
    assert first_path.is_dir()
    owner.state = {"value": 2}
    second = await checkpointer.save(SaveCheckpointRequest.for_step(2))

    assert not first_path.exists()
    assert _checkpoint_object_path(second).is_dir()
    with pytest.raises(CheckpointObjectCleanedUpError):
        await checkpointer.restore(StepReference(1))
    await checkpointer.close()


@pytest.mark.asyncio
async def test_retain_last_counts_checkpoint_object_versions_instead_of_rank_checkpoint_objects(
    tmp_path,
):
    topology = TopologySnapshot(2, ("data",), (2,), "v1")
    owners = (_Owner(), _Owner())
    controller = _MultiRankController(owners, topology)
    control_component = _component()
    checkpointer = FluxwayCheckpointer(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((control_component.registration(),))
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    first_paths = _checkpoint_object_paths(first)
    assert len(first_paths) == topology.world_size
    assert all(path.is_dir() for path in first_paths)
    for owner in owners:
        owner.state = {"value": 2}
    second = await checkpointer.save(SaveCheckpointRequest.for_step(2))

    assert all(not path.exists() for path in first_paths)
    assert all(path.is_dir() for path in _checkpoint_object_paths(second))
    await checkpointer.close()


@pytest.mark.asyncio
async def test_multirank_reservation_write_failure_rolls_back_every_claim(
    tmp_path, monkeypatch
):
    topology = TopologySnapshot(2, ("data",), (2,), "v1")
    owners = (_Owner(), _Owner())
    controller = _MultiRankController(owners, topology)
    control_component = _component()
    checkpointer = FluxwayCheckpointer(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((control_component.registration(),))
        },
    )
    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    checkpoint_objects = tuple(
        partition.record.checkpoint_objects[0]
        for partition in first.spmds["train"].components["model"].partitions
    )
    original_write = control_component._write_lifecycle
    reservation_writes = 0

    def fail_second_reservation_write(checkpoint_object, state):
        nonlocal reservation_writes
        claim = state["claim"]
        if isinstance(claim, dict) and claim["status"] == "reserved":
            reservation_writes += 1
            if reservation_writes == 2:
                raise OSError("injected lifecycle write failure")
        original_write(checkpoint_object, state)

    monkeypatch.setattr(
        control_component, "_write_lifecycle", fail_second_reservation_write
    )
    with pytest.raises(OSError, match="injected lifecycle write failure"):
        await checkpointer.save(SaveCheckpointRequest.for_step(2))

    leases = []
    for rank, checkpoint_object in enumerate(checkpoint_objects):
        leases.append(
            await checkpointer.hold_checkpoint_object(
                "train", checkpoint_object, "reader", 60, f"request-{rank}"
            )
        )
    for lease in leases:
        await checkpointer.release_checkpoint_hold("train", lease)
    await checkpointer.close()


@pytest.mark.asyncio
async def test_failed_cleanup_is_automatically_repaired_before_next_save(tmp_path):
    topology = TopologySnapshot(2, ("data",), (2,), "v1")
    owners = (_Owner(), _Owner())
    controller = _MultiRankController(owners, topology)
    control_component = _component()
    registration = control_component.registration()
    fail_once = _FailOnceCleanuper(control_component)
    control_registration = CheckpointComponentRegistration(
        component=registration.component,
        component_lifecycle=registration.component_lifecycle,
        save=registration.save,
        restore=registration.restore,
        object_lifecycle=registration.object_lifecycle,
        object_cleanup=fail_once,
    )
    checkpointer = FluxwayCheckpointer(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((control_registration,))
        },
    )
    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    first_paths = _checkpoint_object_paths(first)
    for owner in owners:
        owner.state = {"value": 2}

    with pytest.raises(OSError, match="injected physical cleanup failure"):
        await checkpointer.save(SaveCheckpointRequest.for_step(2))
    second = checkpointer.latest()
    assert second is not None
    assert len(checkpointer.repository.list_pending_cleanups()) == 1
    assert all(path.is_dir() for path in first_paths)
    second_paths = _checkpoint_object_paths(second)
    await checkpointer.close()

    for owner in owners:
        owner.state = {"value": 3}
    next_controller = _MultiRankController(owners, topology)
    next_component = _component()
    next_checkpointer = FluxwayCheckpointer(
        next_controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((next_component.registration(),))
        },
    )
    third = await next_checkpointer.save(SaveCheckpointRequest.for_step(3))
    assert next_checkpointer.repository.list_pending_cleanups() == ()
    assert all(not path.exists() for path in first_paths)
    assert all(not path.exists() for path in second_paths)
    assert all(path.is_dir() for path in _checkpoint_object_paths(third))
    await next_checkpointer.close()
