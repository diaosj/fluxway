from types import SimpleNamespace

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCapability,
    CheckpointCut,
    CheckpointObjectRef,
    CheckpointObjectRecord,
    CheckpointObjectSpec,
    CheckpointCommitPermit,
    CheckpointManifest,
    ComponentCheckpoint,
    ComponentSpec,
    CleanupReservation,
    KeepPolicy,
    SpmdCheckpoint,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.keep import (
    CheckpointCleanupCoordinator,
    BackendCleanupReservation,
)


class _Keep:
    def __init__(self, *, publish_error: BaseException | None = None) -> None:
        self.publish_error = publish_error
        self.aborted: list[CleanupReservation] = []

    async def commit_cleanup(self, reservation, permit) -> None:
        if self.publish_error is not None:
            raise self.publish_error

    async def rollback_cleanup(self, reservation, permit) -> None:
        self.aborted.append(reservation)


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
        schema_version=4,
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
                runtime_state={},
                components={
                    "dense": ComponentCheckpoint(
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
async def test_keep_publication_failure_aborts_every_reservation():
    first = _Keep()
    second = _Keep(publish_error=RuntimeError("injected publication failure"))
    registry = _Registry(
        {
            "first": SimpleNamespace(object_lifecycle=first),
            "second": SimpleNamespace(object_lifecycle=second),
        }
    )
    coordinator = CheckpointCleanupCoordinator({"train": registry}, SimpleNamespace())
    permit = CheckpointCommitPermit("attempt", 1, b"token")
    reservations = tuple(
        BackendCleanupReservation(
            "train",
            component_id,
            CleanupReservation(
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

    assert first.aborted == [reservations[0].reservation]
    assert second.aborted == [reservations[1].reservation]


@pytest.mark.asyncio
async def test_keep_last_is_independent_for_each_checkpoint_object():
    lifecycle = _ReservingLifecycle()
    descriptor = ComponentSpec(
        component_id="dense",
        capabilities=frozenset(
            {
                CheckpointCapability.PERSISTENCE,
                CheckpointCapability.LIFECYCLE,
                CheckpointCapability.RETENTION,
                CheckpointCapability.RECLAIMER,
            }
        ),
        checkpoint_objects=(
            CheckpointObjectSpec("model_state", KeepPolicy(keep_last=1)),
            CheckpointObjectSpec("optimizer_state", KeepPolicy(keep_last=1)),
        ),
    )
    registration = SimpleNamespace(
        descriptor=descriptor,
        object_lifecycle=lifecycle,
    )
    repository = SimpleNamespace(is_checkpoint_object_cleaned_up=lambda *_: False)
    coordinator = CheckpointCleanupCoordinator(
        {"train": _Registry({"dense": registration})}, repository
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
