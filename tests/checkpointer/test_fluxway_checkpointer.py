"""
Unit tests for FluxwayCheckpointer payload persistence.

Covers strict pure-data payloads and checkpoint lifecycle operations.
"""

import msgpack
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectoryOwnershipError,
)


@pytest.fixture
def checkpointer(tmp_path):
    """Return a FluxwayCheckpointer using a temporary ckpt_dir."""
    mock_controller = MagicMock()
    return FluxwayCheckpointer(
        controller=mock_controller,
        ckpt_dir=str(tmp_path / "ckpts"),
        interval_steps=1,
        max_to_keep=10,
    )


def test_checkpointer_claims_new_checkpoint_root(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"

    FluxwayCheckpointer(
        controller=MagicMock(),
        ckpt_dir=str(checkpoint_root),
        interval_steps=1,
    )

    assert (checkpoint_root / CHECKPOINT_ROOT_MARKER).is_file()


def test_checkpointer_rejects_existing_unowned_directory(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    with pytest.raises(CheckpointDirectoryOwnershipError, match="not owned"):
        FluxwayCheckpointer(
            controller=MagicMock(),
            ckpt_dir=str(checkpoint_root),
            interval_steps=1,
        )


class TestPayloadPersistence:
    def test_persist_and_load_plain_dict(self, checkpointer, tmp_path):
        payload = {
            "__fluxway_control_plane__": {},
            "spmd_0": {
                "spmd_state": {"completed_step_count": 3},
                "rank_storage": {"0": {"orbax_final_path": "/ckpt/0"}},
            },
        }
        checkpointer._persist(step=5, payload=payload)
        loaded = checkpointer._load(step=5)
        assert loaded == payload

    def test_persist_and_load_with_custom_encoder(self, checkpointer, tmp_path):
        payload = {
            "__fluxway_control_plane__": {},
            "spmd_0": {
                "spmd_state": {
                    "some_hash": b"\xab\xcd\xef",
                    "some_set": {"a", "b", "c"},
                },
            },
        }
        checkpointer._persist(step=6, payload=payload)
        loaded = checkpointer._load(step=6)

        # Verify custom types are perfectly preserved
        assert isinstance(loaded["spmd_0"]["spmd_state"]["some_hash"], bytes)
        assert loaded["spmd_0"]["spmd_state"]["some_hash"] == b"\xab\xcd\xef"
        # msgpack coerces our set to a list via the hook, so we verify it's a list
        assert isinstance(loaded["spmd_0"]["spmd_state"]["some_set"], list)
        assert set(loaded["spmd_0"]["spmd_state"]["some_set"]) == {"a", "b", "c"}

    @pytest.mark.parametrize("runtime_object", [len, MagicMock])
    def test_payload_rejects_runtime_objects(self, checkpointer, runtime_object):
        payload = {"spmd_0": {"spmd_state": {"value": runtime_object}}}
        with pytest.raises(TypeError, match="unsupported checkpoint state type"):
            checkpointer._persist(step=10, payload=payload)


class TestCheckpointLifecycle:
    @pytest.mark.asyncio
    async def test_save_finalizes_control_progress_after_payload_persist(
        self, tmp_path
    ):
        controller = MagicMock()
        control_state = {"control_ledger": {"durable_cuts": {}}}
        payload = {
            "schema_version": "1.0",
            "spmd_0": {},
            "__fluxway_control_plane__": control_state,
        }
        controller.request_snapshot = AsyncMock(return_value=payload)
        controller.commit_snapshot_artifacts = AsyncMock()
        controller.abort_snapshot_artifacts = AsyncMock()
        controller.finalize_snapshot = AsyncMock()
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(tmp_path / "ordered"),
            interval_steps=1,
        )

        await checkpointer.save(3)

        assert checkpointer._load(3) == payload
        controller.finalize_snapshot.assert_awaited_once_with(3, control_state)

    @pytest.mark.asyncio
    async def test_save_does_not_finalize_when_payload_persist_fails(self, tmp_path):
        controller = MagicMock()
        control_state = {"control_ledger": {"durable_cuts": {}}}
        controller.request_snapshot = AsyncMock(
            return_value={
                "spmd_0": {"state": object()},
                "__fluxway_control_plane__": control_state,
            }
        )
        controller.commit_snapshot_artifacts = AsyncMock()
        controller.abort_snapshot_artifacts = AsyncMock()
        controller.finalize_snapshot = AsyncMock()
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(tmp_path / "failed"),
            interval_steps=1,
        )

        with pytest.raises(TypeError, match="unsupported checkpoint state type"):
            await checkpointer.save(3)

        controller.finalize_snapshot.assert_not_awaited()
        controller.commit_snapshot_artifacts.assert_not_awaited()
        controller.abort_snapshot_artifacts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_preserves_primary_and_abort_failures(self, tmp_path):
        controller = MagicMock()
        controller.request_snapshot = AsyncMock(
            return_value={
                "spmd_0": {},
                "__fluxway_control_plane__": {"control_ledger": {}},
            }
        )
        controller.commit_snapshot_artifacts = AsyncMock()
        controller.abort_snapshot_artifacts = AsyncMock(
            side_effect=RuntimeError("abort failed")
        )
        controller.finalize_snapshot = AsyncMock()
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(tmp_path / "grouped-failure"),
            interval_steps=1,
        )
        checkpointer._prepare = MagicMock(side_effect=ValueError("prepare failed"))

        with pytest.raises(BaseExceptionGroup) as captured:
            await checkpointer.save(3)

        messages = {str(error) for error in captured.value.exceptions}
        assert messages == {"prepare failed", "abort failed"}
        controller.finalize_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_snapshot_failure_aborts_distributed_artifacts(
        self, tmp_path
    ):
        controller = MagicMock()
        controller.request_snapshot = AsyncMock(
            side_effect=RuntimeError("driver save failed")
        )
        controller.abort_snapshot_artifacts = AsyncMock()
        controller.finalize_snapshot = AsyncMock()
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(tmp_path / "request-failure"),
            interval_steps=1,
        )

        with pytest.raises(RuntimeError, match="driver save failed"):
            await checkpointer.save(3)

        controller.abort_snapshot_artifacts.assert_awaited_once()
        controller.finalize_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_write_failure_preserves_published_checkpoint(self, tmp_path):
        checkpoint_root = tmp_path / "overwrite"
        controller = MagicMock()
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(checkpoint_root),
            interval_steps=1,
            force_write=True,
        )
        old_orbax = checkpoint_root / "1" / "orbax"
        old_orbax.mkdir(parents=True)
        (old_orbax / "value").write_text("old")
        new_orbax = checkpoint_root / "1" / "orbax.tmp"
        new_orbax.mkdir()
        (new_orbax / "value").write_text("new")

        payload = {
            "spmd_0": {
                "rank_storage": {
                    "0": {
                        "orbax_tmp_path": str(new_orbax),
                        "orbax_final_path": str(old_orbax),
                    }
                }
            },
            "__fluxway_control_plane__": {"control_ledger": {}},
        }
        controller.request_snapshot = AsyncMock(return_value=payload)
        controller.commit_snapshot_artifacts = AsyncMock(
            side_effect=RuntimeError("commit failed")
        )
        controller.abort_snapshot_artifacts = AsyncMock()
        controller.finalize_snapshot = AsyncMock()
        old_payload = {
            "schema_version": "1.0",
            "spmd_0": {"version": 1},
            "__fluxway_control_plane__": {"control_ledger": {}},
        }
        with open(checkpoint_root / "1" / "metadata.msgpack", "wb") as file:
            msgpack.pack(old_payload, file)

        with pytest.raises(RuntimeError, match="commit failed"):
            await checkpointer.save(1)

        assert checkpointer._load(1) == old_payload
        assert (old_orbax / "value").read_text() == "old"
        controller.abort_snapshot_artifacts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalize_failure_keeps_publish_and_closes_local_transaction(
        self, tmp_path
    ):
        controller = MagicMock()
        payload = {
            "schema_version": "1.0",
            "spmd_0": {},
            "__fluxway_control_plane__": {"control_ledger": {}},
        }
        controller.request_snapshot = AsyncMock(return_value=payload)
        controller.commit_snapshot_artifacts = AsyncMock()
        controller.abort_snapshot_artifacts = AsyncMock()
        controller.finalize_snapshot = AsyncMock(
            side_effect=RuntimeError("finalize failed")
        )
        checkpointer = FluxwayCheckpointer(
            controller=controller,
            ckpt_dir=str(tmp_path / "finalize-failure"),
            interval_steps=1,
        )

        with pytest.raises(RuntimeError, match="finalize failed"):
            await checkpointer.save(1)

        assert checkpointer._load(1) == payload
        assert checkpointer._prepared_steps == {}
        assert checkpointer._published_steps == {}
        controller.abort_snapshot_artifacts.assert_not_awaited()

    def test_list_steps_and_latest_step(self, checkpointer, tmp_path):
        for step in [1, 3, 5]:
            checkpointer._persist(step=step, payload={"spmd_0": {}})

        assert checkpointer.list_steps() == [1, 3, 5]
        assert checkpointer.latest_step() == 5

    def test_load_missing_step_raises(self, checkpointer):
        with pytest.raises(FileNotFoundError, match="payload not found"):
            checkpointer._load(step=999)

    @pytest.mark.asyncio
    async def test_restore_latest_returns_none_when_checkpoint_root_is_empty(
        self, checkpointer
    ):
        assert await checkpointer.restore("latest") is None

    def test_evict_old_checkpoints(self, checkpointer, tmp_path):
        checkpointer._max_to_keep = 2
        for step in [1, 2, 3, 4]:
            checkpointer._persist(step=step, payload={"spmd_0": {}})

        # Only the latest 2 should remain
        assert checkpointer.list_steps() == [3, 4]

    def test_force_write_overwrite_existing(self, checkpointer, tmp_path):
        payload_v1 = {
            "schema_version": "1.0",
            "__fluxway_control_plane__": {},
            "spmd_0": {"version": 1},
        }
        payload_v2 = {
            "schema_version": "1.0",
            "__fluxway_control_plane__": {},
            "spmd_0": {"version": 2},
        }
        checkpointer._persist(step=1, payload=payload_v1)

        # Without force_write, should raise
        with pytest.raises(FileExistsError):
            checkpointer._persist(step=1, payload=payload_v2)

        # With force_write, should succeed
        checkpointer._force_write = True
        checkpointer._persist(step=1, payload=payload_v2)
        loaded = checkpointer._load(step=1)
        assert loaded["spmd_0"]["version"] == 2
