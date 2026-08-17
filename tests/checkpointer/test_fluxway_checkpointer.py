from unittest.mock import AsyncMock

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCut,
    CheckpointAssembly,
    SaveCheckpointRequest,
    StepReference,
)
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectoryOwnershipError,
)


def test_checkpointer_claims_a_new_repository_directory(tmp_path):
    root = tmp_path / "repository"
    FluxwayCheckpointer(object(), str(root), interval_steps=1)
    assert (root / CHECKPOINT_ROOT_MARKER).is_file()


def test_checkpointer_rejects_an_unowned_directory(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    with pytest.raises(CheckpointDirectoryOwnershipError):
        FluxwayCheckpointer(object(), str(root), interval_steps=1)


@pytest.mark.asyncio
async def test_save_publishes_manifest_then_finalizes_framework_state(tmp_path):
    cut = CheckpointCut.for_step("transaction", 4)
    controller = type("Controller", (), {})()
    controller.request_snapshot = AsyncMock(
        return_value=CheckpointAssembly(cut, {}, {})
    )
    controller.publish_snapshot = AsyncMock()
    controller.commit_checkpoint_state = AsyncMock()
    controller.rollback_snapshot = AsyncMock()
    checkpointer = FluxwayCheckpointer(
        controller, str(tmp_path / "repository"), interval_steps=1
    )

    manifest = await checkpointer.save(
        SaveCheckpointRequest(
            attempt_id="attempt", cut=cut, references=(StepReference(4),)
        )
    )

    assert checkpointer.repository.resolve(StepReference(4)) == manifest
    published_snapshot, permit = controller.publish_snapshot.await_args.args
    assert published_snapshot == manifest.checkpoint_id
    assert permit.attempt_id == "attempt"
    controller.commit_checkpoint_state.assert_awaited_once_with(4, {})
