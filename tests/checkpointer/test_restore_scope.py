from __future__ import annotations

import pytest

from fluxway.common.types.configs import CheckpointConfig


def test_checkpoint_config_declares_restore_scope_and_bindings() -> None:
    config = CheckpointConfig(
        repository_dir="/tmp/checkpoint-scope-test",
        interval_steps=1,
        restore_scope=("RecsysBackboneTrain", "Embedding_geek_id"),
        restore_spmd_bindings={"RecsysBackboneTrain": "RecsysBackboneEval"},
    )
    assert config.restore_scope == (
        "RecsysBackboneTrain",
        "Embedding_geek_id",
    )
    assert config.restore_spmd_bindings == {"RecsysBackboneTrain": "RecsysBackboneEval"}


@pytest.mark.asyncio
async def test_restore_scope_remaps_only_declared_manifest_spmds(
    checkpointer_factory, tmp_path
) -> None:
    checkpointer = await checkpointer_factory(
        object(),
        str(tmp_path / "repository"),
        interval_steps=1,
        restore_scope=("source",),
        restore_spmd_bindings={"source": "target"},
    )
    assert checkpointer.restore_scope == ("source",)
    assert checkpointer.restore_spmd_bindings == {"source": "target"}
