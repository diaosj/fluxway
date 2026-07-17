"""
Unit tests for FluxwayCheckpointer manifest persistence.

Covers:
 - _persist / _load round-trip with plain dicts
 - _persist / _load with Python type objects (static_args containing dataclasses)
 - _persist / _load with callable objects
 - list_steps / latest_step / evict old checkpoints
"""

from dataclasses import make_dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectoryOwnershipError,
)

# Module-level dynamic dataclass for pickle compatibility
DynamicBatch = make_dataclass("DynamicBatch", [("x", int), ("y", float)])


def my_activation(x: Any) -> Any:
    return x * 2


def custom_fn(x: Any) -> Any:
    return x + 42


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


class TestManifestPersistence:
    def test_persist_and_load_plain_dict(self, checkpointer, tmp_path):
        manifest = {
            "spmd_0": {
                "spmd_state": {"completed_step_count": 3},
                "rank_storage": {"0": {"orbax_final_path": "/ckpt/0"}},
            }
        }
        checkpointer._persist(step=5, manifest=manifest)
        loaded = checkpointer._load(step=5)
        assert loaded == manifest

    def test_manifest_with_type_object(self, checkpointer, tmp_path):
        """Test that manifest containing Python type objects can be serialized."""
        manifest = {
            "spmd_0": {
                "spmd_state": {
                    "completed_step_count": 5,
                    "static_args": {
                        "UserBatch": DynamicBatch,
                        "dataset_path": "/data/train",
                        "batch_size": 64,
                    },
                },
                "rank_storage": {"0": {"orbax_final_path": "/ckpt/0"}},
            }
        }
        checkpointer._persist(step=10, manifest=manifest)
        loaded = checkpointer._load(step=10)

        assert loaded["spmd_0"]["spmd_state"]["completed_step_count"] == 5
        assert (
            loaded["spmd_0"]["spmd_state"]["static_args"]["dataset_path"]
            == "/data/train"
        )
        assert loaded["spmd_0"]["spmd_state"]["static_args"]["batch_size"] == 64
        # Verify the type object is correctly restored
        RestoredBatch = loaded["spmd_0"]["spmd_state"]["static_args"]["UserBatch"]
        assert RestoredBatch is DynamicBatch
        instance = RestoredBatch(x=1, y=2.0)
        assert instance.x == 1
        assert instance.y == 2.0

    def test_manifest_with_callable(self, checkpointer, tmp_path):
        """Test that manifest containing callable objects can be serialized."""
        manifest = {
            "spmd_0": {
                "spmd_state": {
                    "completed_step_count": 7,
                    "static_args": {
                        "activation": my_activation,
                        "scale": 1.5,
                    },
                },
                "rank_storage": {"0": {"orbax_final_path": "/ckpt/0"}},
            }
        }
        checkpointer._persist(step=20, manifest=manifest)
        loaded = checkpointer._load(step=20)

        restored_callable = loaded["spmd_0"]["spmd_state"]["static_args"]["activation"]
        assert restored_callable(5) == 10
        assert loaded["spmd_0"]["spmd_state"]["static_args"]["scale"] == 1.5

    def test_manifest_with_builtin_callable(self, checkpointer, tmp_path):
        """Test that manifest containing built-in callable can be serialized."""
        manifest = {
            "spmd_0": {
                "spmd_state": {
                    "completed_step_count": 8,
                    "static_args": {
                        "transform": len,
                    },
                },
                "rank_storage": {"0": {"orbax_final_path": "/ckpt/0"}},
            }
        }
        checkpointer._persist(step=30, manifest=manifest)
        loaded = checkpointer._load(step=30)

        restored = loaded["spmd_0"]["spmd_state"]["static_args"]["transform"]
        assert restored([1, 2, 3]) == 3

    def test_checkpoint_save_load_roundtrip(self, checkpointer, tmp_path):
        """Test full save-load roundtrip with complex static_args."""
        manifest = {
            "spmd_0": {
                "spmd_state": {
                    "completed_step_count": 100,
                    "static_args": {
                        "UserBatch": DynamicBatch,
                        "fn": custom_fn,
                        "config": {"lr": 0.01, "epochs": 10},
                    },
                },
                "rank_storage": {
                    "0": {"orbax_final_path": "/ckpt/spmd_0/0"},
                    "1": {"orbax_final_path": "/ckpt/spmd_0/1"},
                },
            },
            "spmd_1": {
                "spmd_state": {
                    "completed_step_count": 100,
                    "static_args": {
                        "batch_size": 64,
                    },
                },
                "rank_storage": {
                    "0": {"orbax_final_path": "/ckpt/spmd_1/0"},
                },
            },
        }
        checkpointer._persist(step=50, manifest=manifest)
        loaded = checkpointer._load(step=50)

        assert loaded == manifest
        # Ensure type object is usable
        Batch = loaded["spmd_0"]["spmd_state"]["static_args"]["UserBatch"]
        assert Batch is DynamicBatch
        # Ensure callable is usable
        fn = loaded["spmd_0"]["spmd_state"]["static_args"]["fn"]
        assert fn(0) == 42


class TestCheckpointLifecycle:
    def test_list_steps_and_latest_step(self, checkpointer, tmp_path):
        for step in [1, 3, 5]:
            checkpointer._persist(step=step, manifest={"spmd_0": {}})

        assert checkpointer.list_steps() == [1, 3, 5]
        assert checkpointer.latest_step() == 5

    def test_load_missing_step_raises(self, checkpointer):
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            checkpointer._load(step=999)

    @pytest.mark.asyncio
    async def test_restore_latest_returns_none_when_checkpoint_root_is_empty(
        self, checkpointer
    ):
        assert await checkpointer.restore("latest") is None

    def test_evict_old_checkpoints(self, checkpointer, tmp_path):
        checkpointer._max_to_keep = 2
        for step in [1, 2, 3, 4]:
            checkpointer._persist(step=step, manifest={"spmd_0": {}})

        # Only the latest 2 should remain
        assert checkpointer.list_steps() == [3, 4]

    def test_force_write_overwrite_existing(self, checkpointer, tmp_path):
        manifest_v1 = {"spmd_0": {"version": 1}}
        manifest_v2 = {"spmd_0": {"version": 2}}
        checkpointer._persist(step=1, manifest=manifest_v1)

        # Without force_write, should raise
        with pytest.raises(FileExistsError):
            checkpointer._persist(step=1, manifest=manifest_v2)

        # With force_write, should succeed
        checkpointer._force_write = True
        checkpointer._persist(step=1, manifest=manifest_v2)
        loaded = checkpointer._load(step=1)
        assert loaded["spmd_0"]["version"] == 2
