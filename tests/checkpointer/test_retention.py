import inspect
from types import SimpleNamespace

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCut,
    CheckpointObjectRef,
    CheckpointObjectRecord,
    CheckpointObjectSpec,
    CheckpointCommitPermit,
    CheckpointManifest,
    CheckpointComponentPublication,
    CheckpointComponentSpec,
    CleanupReservation,
    CleanupReservationBinding,
    CheckpointCleanupIntent,
    CheckpointRetentionPolicy,
    SpmdCheckpoint,
    TopologySnapshot,
)
from fluxway.common.types import DrainBoundary
from fluxway.control_plane.checkpointer.retention import (
    CheckpointRetentionCoordinator,
)
from fluxway.control_plane.checkpointer._filesystem_component import (
    FilesystemCheckpointComponent,
)
from fluxway.control_plane.checkpointer.orbax_component import OrbaxCheckpointComponent
from fluxway.control_plane.checkpointer.pickle_component import (
    WorkerLocalCheckpointComponent,
)
from fluxway.control_plane.checkpointer.safetensors_component import (
    SafeTensorsCheckpointComponent,
)
from fluxway.integrations.sparse_embedding.checkpoint import (
    SparseEmbeddingCheckpointComponent,
)


class _CleanupLifecycle:
    def __init__(self, *, publish_error: BaseException | None = None) -> None:
        self.publish_error = publish_error
        self.rolled_back: list[CleanupReservation] = []

    async def commit_cleanup(self, reservation, permit) -> None:
        if self.publish_error is not None:
            raise self.publish_error

    async def rollback_cleanup(self, reservation, permit) -> None:
        self.rolled_back.append(reservation)


class _Registry:
    def __init__(self, registrations) -> None:
        self._registrations = registrations

    def require(self, component_id):
        return self._registrations[component_id]


class _ReservingLifecycle:
    def __init__(self) -> None:
        self.requests = []

    async def reserve_cleanup(self, request):
        self.requests.append(request)
        return CleanupReservation(
            request.permit.attempt_id,
            request.checkpoint_object_id,
            request.ordered_candidates[: request.required_count],
            request.permit.coordinator_generation,
        )

    async def commit_cleanup(self, reservation, permit) -> None:
        return None

    async def rollback_cleanup(self, reservation, permit) -> None:
        return None


def _multi_object_manifest(sequence: int) -> CheckpointManifest:
    objects = tuple(
        CheckpointObjectRef(
            component_id="dense",
            checkpoint_object_id=checkpoint_object_id,
            version_id=f"{checkpoint_object_id}-{sequence}",
            storage_locator=f"opaque:{checkpoint_object_id}:{sequence}".encode(),
            format_version=1,
        )
        for checkpoint_object_id in ("model_state", "optimizer_state")
    )
    return CheckpointManifest(
        schema_version=1,
        checkpoint_id=f"checkpoint-{sequence}",
        attempt_id=f"attempt-{sequence}",
        coordinator_generation=1,
        cut=CheckpointCut.for_step(f"cut-{sequence}", sequence),
        checkpoint_sequence=sequence,
        references=(),
        framework_state={},
        spmds={
            "train": SpmdCheckpoint(
                source_topology=TopologySnapshot(1, (), (), "v1"),
                runtime_state={
                    "drain_boundary": DrainBoundary(committed_cut=sequence).to_dict()
                },
                components={
                    "dense": CheckpointComponentPublication(
                        component_id="dense",
                        format_version=1,
                        shared_records=(
                            CheckpointObjectRecord(checkpoint_objects=objects),
                        ),
                    )
                },
            )
        },
    )


@pytest.mark.asyncio
async def test_retention_publication_failure_rolls_back_every_reservation(io_lane):
    first = _CleanupLifecycle()
    second = _CleanupLifecycle(
        publish_error=RuntimeError("injected publication failure")
    )
    registry = _Registry(
        {
            "first": SimpleNamespace(object_lifecycle=first),
            "second": SimpleNamespace(object_lifecycle=second),
        }
    )
    coordinator = CheckpointRetentionCoordinator(
        {"train": registry},
        SimpleNamespace(),
        CheckpointRetentionPolicy(retain_last=1),
        io_lane=io_lane,
    )
    permit = CheckpointCommitPermit("attempt", 1, b"token")
    reservations = tuple(
        CleanupReservationBinding(
            artifact_id="artifact",
            spmd_id="train",
            component_id=component_id,
            reservation=CleanupReservation(
                "attempt",
                "state",
                (
                    CheckpointObjectRef(
                        component_id,
                        "state",
                        "v1",
                        f"opaque:{component_id}".encode(),
                        1,
                    ),
                ),
                1,
            ),
        )
        for component_id in ("first", "second")
    )

    with pytest.raises(RuntimeError, match="injected publication failure"):
        await coordinator.publish(reservations, permit)

    assert first.rolled_back == [reservations[0].reservation]
    assert second.rolled_back == [reservations[1].reservation]


@pytest.mark.asyncio
async def test_artifact_retention_reserves_each_required_object(io_lane):
    lifecycle = _ReservingLifecycle()
    descriptor = CheckpointComponentSpec(
        component_id="dense",
        save=SimpleNamespace(),
        restore=SimpleNamespace(),
        checkpoint_objects=(
            CheckpointObjectSpec("model_state"),
            CheckpointObjectSpec("optimizer_state"),
        ),
    )
    registration = SimpleNamespace(
        descriptor=descriptor,
        object_lifecycle=lifecycle,
    )
    repository = SimpleNamespace(cleaned_checkpoint_objects=lambda _: frozenset())
    coordinator = CheckpointRetentionCoordinator(
        {"train": _Registry({"dense": registration})},
        repository,
        CheckpointRetentionPolicy(retain_last=1),
        io_lane=io_lane,
    )
    permit = CheckpointCommitPermit("attempt-2", 1, b"token")

    reservations = await coordinator.reserve(
        permit,
        _multi_object_manifest(2),
        (_multi_object_manifest(1),),
    )

    assert {request.checkpoint_object_id for request in lifecycle.requests} == {
        "model_state",
        "optimizer_state",
    }
    assert all(request.required_count == 1 for request in lifecycle.requests)
    assert {binding.reservation.checkpoint_object_id for binding in reservations} == {
        "model_state",
        "optimizer_state",
    }


def test_cleanup_intent_groups_bindings_by_complete_artifact():
    bindings = (
        CleanupReservationBinding(
            artifact_id="artifact-a",
            spmd_id="reader",
            component_id="state",
            reservation=CleanupReservation("attempt", "state", (), 1),
        ),
        CleanupReservationBinding(
            artifact_id="artifact-a",
            spmd_id="embedding",
            component_id="state",
            reservation=CleanupReservation("attempt", "state", (), 1),
        ),
    )
    intents = CheckpointRetentionCoordinator.cleanup_intents(bindings)
    assert intents == (CheckpointCleanupIntent("artifact-a", bindings),)


def test_retention_has_single_artifact_level_owner():
    assert not hasattr(CheckpointObjectSpec, "retention")
    for component_type in (
        FilesystemCheckpointComponent,
        OrbaxCheckpointComponent,
        WorkerLocalCheckpointComponent,
        SafeTensorsCheckpointComponent,
        SparseEmbeddingCheckpointComponent,
    ):
        parameters = inspect.signature(component_type.__init__).parameters
        assert "retention" not in parameters
        assert "retain_last" not in parameters


@pytest.mark.asyncio
async def test_component_retention_domain_resolves_through_single_coordinator(io_lane):
    coordinator = CheckpointRetentionCoordinator(
        {},
        SimpleNamespace(),
        CheckpointRetentionPolicy(retain_last=5),
        {
            "short": CheckpointRetentionPolicy(retain_last=1),
            "archive": CheckpointRetentionPolicy(retain_forever=True),
        },
        io_lane=io_lane,
    )
    short = SimpleNamespace(descriptor=SimpleNamespace(retention_domain="short"))
    archive = SimpleNamespace(descriptor=SimpleNamespace(retention_domain="archive"))
    default = SimpleNamespace(descriptor=SimpleNamespace(retention_domain=None))

    assert coordinator._policy_for(short).retain_last == 1
    assert coordinator._policy_for(archive).retain_forever
    assert coordinator._policy_for(default).retain_last == 5

    missing = SimpleNamespace(descriptor=SimpleNamespace(retention_domain="missing"))
    with pytest.raises(ValueError, match="retention domain is not configured"):
        coordinator._policy_for(missing)


@pytest.mark.asyncio
async def test_artifact_limit_uses_strictest_component_domain_constraint(io_lane):
    descriptor_a = CheckpointComponentSpec(
        "dense",
        retention_domain="short",
        save=SimpleNamespace(),
        restore=SimpleNamespace(),
        checkpoint_objects=(CheckpointObjectSpec("state"),),
    )
    descriptor_b = CheckpointComponentSpec(
        "sparse",
        retention_domain="long",
        save=SimpleNamespace(),
        restore=SimpleNamespace(),
        checkpoint_objects=(CheckpointObjectSpec("state"),),
    )
    registrations = {
        "dense": SimpleNamespace(descriptor=descriptor_a),
        "sparse": SimpleNamespace(descriptor=descriptor_b),
    }
    coordinator = CheckpointRetentionCoordinator(
        {"train": _Registry(registrations)},
        SimpleNamespace(),
        CheckpointRetentionPolicy(retain_last=5),
        {
            "short": CheckpointRetentionPolicy(retain_last=1),
            "long": CheckpointRetentionPolicy(retain_last=3),
        },
        io_lane=io_lane,
    )
    manifest = SimpleNamespace(
        spmds={"train": SimpleNamespace(components={"dense": None, "sparse": None})}
    )
    assert coordinator._artifact_retention_limit(manifest) == 3
