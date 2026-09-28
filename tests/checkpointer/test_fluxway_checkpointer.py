import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCut,
    CheckpointAssembly,
    SaveCheckpointRequest,
    StepReference,
)
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.control_plane.checkpointer.directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectoryOwnershipError,
)


@pytest.mark.asyncio
async def test_checkpointer_claims_a_new_repository_directory(
    checkpointer_factory, tmp_path
):
    root = tmp_path / "repository"
    await checkpointer_factory(object(), str(root), interval_steps=1)
    assert (root / CHECKPOINT_ROOT_MARKER).is_file()


@pytest.mark.asyncio
async def test_checkpointer_rejects_an_unowned_directory(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    with pytest.raises(CheckpointDirectoryOwnershipError):
        await FluxwayCheckpointer.create(object(), str(root), interval_steps=1)


@pytest.mark.asyncio
async def test_concurrent_saves_are_serialized(checkpointer_factory, tmp_path):
    checkpointer = await checkpointer_factory(
        object(), str(tmp_path / "repository"), interval_steps=1
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def delayed_save(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return request

    checkpointer._save = delayed_save
    request = SaveCheckpointRequest(
        "attempt", CheckpointCut.for_step("cut", 1), references=(StepReference(1),)
    )
    first = asyncio.create_task(checkpointer.save(request))
    await entered.wait()
    second = asyncio.create_task(checkpointer.save(request))
    await asyncio.sleep(0)
    assert calls == 1

    release.set()
    assert await asyncio.gather(first, second) == [request, request]
    assert calls == 2


@pytest.mark.asyncio
async def test_save_publishes_manifest_then_finalizes_framework_state(
    checkpointer_factory, tmp_path
):
    events = []
    cut = CheckpointCut.for_step("transaction", 4)
    controller = type("Controller", (), {})()
    controller.request_snapshot = AsyncMock(
        side_effect=lambda *args, **kwargs: (
            events.append("assembly"),
            CheckpointAssembly(cut, {}, {}),
        )[1]
    )
    controller.publish_snapshot = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("publish")
    )
    controller.prepare_checkpoint_transaction = Mock(
        side_effect=lambda *args, **kwargs: (events.append("prepare"), {})[1]
    )
    controller.apply_prepared_checkpoint_transaction = Mock(
        side_effect=lambda *args, **kwargs: events.append("apply")
    )
    controller.abort_checkpoint_transaction = Mock()
    controller.rollback_snapshot = AsyncMock()
    checkpointer = await checkpointer_factory(
        controller, str(tmp_path / "repository"), interval_steps=1
    )

    manifest = await checkpointer.save(
        SaveCheckpointRequest(
            attempt_id="attempt", cut=cut, references=(StepReference(4),)
        )
    )

    assert (
        CheckpointRepository(tmp_path / "repository").resolve(StepReference(4))
        == manifest
    )
    tmp_args = controller.publish_snapshot.await_args
    assert tmp_args is not None
    published_snapshot, permit = tmp_args.args[:2]
    assert published_snapshot == manifest.checkpoint_id
    assert permit.attempt_id == "attempt"
    controller.apply_prepared_checkpoint_transaction.assert_called_once()
    assert events == ["assembly", "prepare", "publish", "apply"]


@pytest.mark.asyncio
async def test_post_publication_commit_failure_is_not_restorable(
    checkpointer_factory, tmp_path
):
    events = []
    cut = CheckpointCut.for_step("transaction", 4)
    controller = type("Controller", (), {})()
    controller.request_snapshot = AsyncMock(
        side_effect=lambda *args, **kwargs: (
            events.append("assembly"),
            CheckpointAssembly(cut, {}, {}),
        )[1]
    )
    controller.publish_snapshot = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("publish")
    )
    controller.prepare_checkpoint_transaction = Mock(
        side_effect=lambda *args, **kwargs: (events.append("prepare"), {})[1]
    )

    def fail_apply(*args):
        events.append("apply")
        raise RuntimeError("injected post-publication apply failure")

    controller.apply_prepared_checkpoint_transaction = Mock(side_effect=fail_apply)
    controller.abort_checkpoint_transaction = Mock()
    controller.rollback_snapshot = AsyncMock()

    checkpointer = await checkpointer_factory(
        controller, str(tmp_path / "repository"), interval_steps=1
    )

    with pytest.raises(RuntimeError, match="post-publication apply failure"):
        await checkpointer.save(
            SaveCheckpointRequest(
                attempt_id="attempt", cut=cut, references=(StepReference(4),)
            )
        )

    assert CheckpointRepository(tmp_path / "repository").latest() is None
    incomplete = CheckpointRepository(tmp_path / "repository").incomplete_transactions()
    assert len(incomplete) == 1
    checkpoint_id = incomplete[0].checkpoint_id
    assert await checkpointer.restore_latest() is None
    record = CheckpointRepository(tmp_path / "repository").transaction(checkpoint_id)
    assert record is not None
    assert record.status.value == "abandoned"
    assert events == ["assembly", "prepare", "publish", "apply"]
    controller.abort_checkpoint_transaction.assert_not_called()
    controller.rollback_snapshot.assert_not_awaited()
