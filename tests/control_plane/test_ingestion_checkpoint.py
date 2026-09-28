from __future__ import annotations


import pytest

from fluxway.common.metadata import metadata_digest, pack_metadata, unpack_metadata

from fluxway.common.types import CheckpointCommitPermit, CheckpointCut, ControllerState
from fluxway.control_plane.components import ControlPlaneComponentRegistry
from fluxway.control_plane.ingestion import (
    CatalogCheckpointImage,
    IngestionCatalogCheckpointComponent,
)


def image(position: int = 4) -> CatalogCheckpointImage:
    return CatalogCheckpointImage.create(
        source_identity="source",
        catalog_view_identity="view-1",
        schema_digest="schema",
        lineage_digest="lineage-1",
        layout_id="layout-1",
        layout_digest="digest-1",
        end_position=position,
        cursor={"snapshot_id": 7},
    )


def transaction(component, cut, permit):
    state = component.checkpoint_state(cut)
    digest = metadata_digest(state)
    return component.prepare_checkpoint("checkpoint", permit, cut, digest)


@pytest.mark.asyncio
async def test_catalog_checkpoint_apply_is_synchronous_and_idempotent() -> None:
    component = IngestionCatalogCheckpointComponent("catalog")
    await component.on_state(ControllerState.RUNNING)
    component.stage(image())
    cut = CheckpointCut.for_position("cut", 4)
    prepared = transaction(
        component, cut, CheckpointCommitPermit("attempt", 0, b"token")
    )
    component.apply_prepared(prepared)
    assert component.committed_image == image()
    component.apply_prepared(prepared)
    assert component.committed_image == image()


def test_catalog_checkpoint_abort_is_fenced() -> None:
    component = IngestionCatalogCheckpointComponent("catalog")
    component.stage(image())
    cut = CheckpointCut.for_position("cut", 4)
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    transaction(component, cut, permit)
    with pytest.raises(ValueError, match="abort does not match prepare"):
        component.abort_prepared(
            "checkpoint", CheckpointCommitPermit("stale", 0, b"stale")
        )
    component.abort_prepared("checkpoint", permit)
    assert component.committed_image is None


def test_registry_prepare_and_apply_use_prepared_tokens() -> None:
    component = IngestionCatalogCheckpointComponent("catalog")
    component.stage(image())
    registry = ControlPlaneComponentRegistry()
    registry.register_transactional(component)
    cut = CheckpointCut.for_position("cut", 4)
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    states = registry.checkpoint_state(cut)
    prepared = registry.prepare_checkpoint_transaction(
        "checkpoint", permit, cut, states
    )
    registry.apply_prepared_checkpoint_transaction(
        "checkpoint", permit, cut, states, prepared
    )
    assert component.committed_image == image()


def test_catalog_checkpoint_rejects_regression_and_future_cut() -> None:
    component = IngestionCatalogCheckpointComponent("catalog")
    component.stage(image())
    with pytest.raises(ValueError, match="position regressed"):
        component.stage(image(3))
    with pytest.raises(ValueError, match="exceeds requested cut"):
        component.checkpoint_state(CheckpointCut.for_position("old", 2))


def test_binary_cursor_survives_component_transaction_and_restore() -> None:
    snapshot = CatalogCheckpointImage.create(
        source_identity="source",
        catalog_view_identity="view-1",
        schema_digest="schema",
        lineage_digest="lineage-1",
        layout_id="layout-1",
        layout_digest="digest-1",
        end_position=4,
        cursor={"revision": 7, "metadata": b"\x00\xff"},
    )
    component = IngestionCatalogCheckpointComponent("catalog")
    registry = ControlPlaneComponentRegistry()
    registry.register_transactional(component)
    component.stage(snapshot)
    cut = CheckpointCut.for_position("cut", 4)
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    states = unpack_metadata(pack_metadata(registry.checkpoint_state(cut)))
    prepared = registry.prepare_checkpoint_transaction(
        "checkpoint", permit, cut, states
    )
    assert component.committed_image is None
    registry.apply_prepared_checkpoint_transaction(
        "checkpoint", permit, cut, states, prepared
    )
    assert component.committed_image == snapshot
    restored = IngestionCatalogCheckpointComponent("catalog")
    restored_registry = ControlPlaneComponentRegistry()
    restored_registry.register_transactional(restored)
    restored_registry.restore_state(states, cut)
    assert restored.committed_image == snapshot
