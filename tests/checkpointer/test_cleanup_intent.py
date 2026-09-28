import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCleanupIntent,
    CleanupReservation,
    CleanupReservationBinding,
)
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.control_plane.checkpointer.directory import CheckpointDirectory


def _intent(artifact_id: str) -> CheckpointCleanupIntent:
    binding = CleanupReservationBinding(
        artifact_id=artifact_id,
        spmd_id="spmd",
        component_id="component",
        reservation=CleanupReservation("attempt", "object", (), 1),
    )
    return CheckpointCleanupIntent(artifact_id, (binding,))


@pytest.mark.asyncio
async def test_maintenance_lane_deduplicates_and_serializes_artifacts():
    release = asyncio.Event()
    started: list[str] = []
    active = 0
    maximum = 0

    async def cleanup(intents):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        started.extend(intent.artifact_id for intent in intents)
        await release.wait()
        active -= 1

    checkpointer = FluxwayCheckpointer.__new__(FluxwayCheckpointer)
    checkpointer._cleanup_lane_lock = asyncio.Lock()
    checkpointer._cleanup_tasks = set()
    checkpointer._cleanup_failures = []
    checkpointer._cleanup_artifacts_inflight = set()
    checkpointer._retention = SimpleNamespace(cleanup=cleanup)

    checkpointer._schedule_cleanup_intents((_intent("a"), _intent("a")))
    checkpointer._schedule_cleanup_intents((_intent("b"),))
    await asyncio.sleep(0.01)
    assert started == ["a"]
    assert maximum == 1

    release.set()
    await asyncio.gather(*tuple(checkpointer._cleanup_tasks))
    assert started == ["a", "b"]


def test_cleanup_intent_completion_is_idempotent_after_crash_repair(tmp_path):
    root = tmp_path / "repository"
    CheckpointDirectory.open(root)
    repository = CheckpointRepository(root)
    current = _intent("artifact-a")
    repository._atomic_write(
        repository._cleanup_intent_path(current),
        __import__("msgpack").packb(
            {
                "artifact_id": current.artifact_id,
                "reservations": [
                    {
                        "artifact_id": binding.artifact_id,
                        "spmd_id": binding.spmd_id,
                        "component_id": binding.component_id,
                        "reservation": {
                            "attempt_id": binding.reservation.attempt_id,
                            "checkpoint_object_id": binding.reservation.checkpoint_object_id,
                            "claimed": [],
                            "generation": binding.reservation.generation,
                        },
                    }
                    for binding in current.reservations
                ],
            },
            use_bin_type=True,
        ),
    )
    repository.complete_cleanup_intent(current)
    repository.complete_cleanup_intent(current)
    assert repository.list_cleanup_intents() == ()
