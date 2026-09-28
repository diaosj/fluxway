"""Source-neutral catalog progress participant for the Controller checkpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fluxway.common.metadata import metadata_digest

from fluxway.common.types import CheckpointCommitPermit, CheckpointCut, ControllerState
from fluxway.control_plane.checkpoint_transaction import (
    CheckpointTransactionFingerprint,
    PreparedCheckpointTransaction,
    PreparedStateStore,
)
from fluxway.control_plane.components import ComponentState
from fluxway.control_plane.ingestion.cursor import CatalogCursor


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _position(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogCheckpointImage:
    source_identity: str
    catalog_view_identity: str
    schema_digest: str
    lineage_digest: str
    layout_id: str
    layout_digest: str
    end_position: int
    cursor: CatalogCursor

    def __post_init__(self) -> None:
        _identity(self.source_identity, "catalog checkpoint source identity")
        _identity(self.catalog_view_identity, "catalog checkpoint view identity")
        _identity(self.schema_digest, "catalog checkpoint schema digest")
        _identity(self.lineage_digest, "catalog checkpoint lineage digest")
        _identity(self.layout_id, "catalog checkpoint layout id")
        _identity(self.layout_digest, "catalog checkpoint layout digest")
        _position(self.end_position, "catalog checkpoint end position")
        if not isinstance(self.cursor, CatalogCursor):
            raise TypeError("catalog checkpoint cursor must be a CatalogCursor")

    def __deepcopy__(self, memo: dict[int, Any]) -> CatalogCheckpointImage:
        copied = type(self).create(
            source_identity=self.source_identity,
            catalog_view_identity=self.catalog_view_identity,
            schema_digest=self.schema_digest,
            lineage_digest=self.lineage_digest,
            layout_id=self.layout_id,
            layout_digest=self.layout_digest,
            end_position=self.end_position,
            cursor=self.cursor,
        )
        memo[id(self)] = copied
        return copied

    @classmethod
    def create(
        cls,
        *,
        cursor: CatalogCursor | Mapping[str, Any],
        **kwargs: Any,
    ) -> CatalogCheckpointImage:
        cursor_val = (
            cursor if isinstance(cursor, CatalogCursor) else CatalogCursor(cursor)
        )
        return cls(cursor=cursor_val, **kwargs)

    @property
    def image_digest(self) -> str:
        return metadata_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_identity": self.source_identity,
            "catalog_view_identity": self.catalog_view_identity,
            "schema_digest": self.schema_digest,
            "lineage_digest": self.lineage_digest,
            "layout_id": self.layout_id,
            "layout_digest": self.layout_digest,
            "end_position": self.end_position,
            "cursor": self.cursor.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CatalogCheckpointImage:
        if not isinstance(value, Mapping):
            raise TypeError("catalog checkpoint image must be a mapping")
        expected = {
            "source_identity",
            "catalog_view_identity",
            "schema_digest",
            "lineage_digest",
            "layout_id",
            "layout_digest",
            "end_position",
            "cursor",
        }
        if set(value) != expected:
            raise ValueError("catalog checkpoint image fields do not match")
        return cls.create(
            source_identity=value["source_identity"],
            catalog_view_identity=value["catalog_view_identity"],
            schema_digest=value["schema_digest"],
            lineage_digest=value["lineage_digest"],
            layout_id=value["layout_id"],
            layout_digest=value["layout_digest"],
            end_position=value["end_position"],
            cursor=value["cursor"],
        )


class IngestionCatalogCheckpointComponent:
    def __init__(self, component_id: str) -> None:
        self.component_id = _identity(component_id, "catalog checkpoint component id")
        self._staged_image: CatalogCheckpointImage | None = None
        self._committed_image: CatalogCheckpointImage | None = None
        self._checkpoint_transactions = PreparedStateStore[
            CatalogCheckpointImage | None
        ]()
        self._lock = asyncio.Lock()

    @property
    def committed_image(self) -> CatalogCheckpointImage | None:
        return self._committed_image

    @property
    def staged_image(self) -> CatalogCheckpointImage | None:
        return self._staged_image

    async def on_state(self, state: ComponentState) -> None:
        if not isinstance(state, ControllerState):
            raise TypeError("catalog checkpoint component requires ControllerState")

    def stage(self, image: CatalogCheckpointImage) -> None:
        if not isinstance(image, CatalogCheckpointImage):
            raise TypeError("catalog checkpoint stage requires CatalogCheckpointImage")
        current = self._staged_image or self._committed_image
        if current is not None:
            if image == current:
                return
            if image.end_position < current.end_position:
                raise ValueError("catalog checkpoint position regressed")
            if image.end_position == current.end_position:
                raise ValueError(
                    "catalog checkpoint image conflicts at the same position"
                )
        self._staged_image = image

    def checkpoint_state(self, cut: CheckpointCut) -> dict[str, Any]:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("catalog checkpoint requires CheckpointCut")
        if cut.position is None:
            raise ValueError("catalog checkpoint requires a global position")
        image = self._staged_image or self._committed_image
        if image is not None and image.end_position > cut.position:
            raise ValueError("catalog checkpoint image exceeds requested cut")
        return {
            "version": 1,
            "image": None if image is None else image.to_dict(),
            "image_digest": None if image is None else image.image_digest,
        }

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("catalog checkpoint restore requires CheckpointCut")
        if cut.position is None:
            raise ValueError("catalog checkpoint restore requires a global position")
        if not isinstance(state, Mapping) or set(state) != {
            "version",
            "image",
            "image_digest",
        }:
            raise ValueError("catalog checkpoint state fields do not match")
        if state["version"] != 1:
            raise ValueError("unsupported catalog checkpoint state version")
        serialized = state["image"]
        if serialized is None:
            if state["image_digest"] is not None:
                raise ValueError("empty catalog checkpoint image has a digest")
            self._staged_image = None
            self._committed_image = None
            return
        image = CatalogCheckpointImage.from_mapping(serialized)
        if state["image_digest"] != image.image_digest:
            raise ValueError("catalog checkpoint image digest is invalid")
        if image.end_position > cut.position:
            raise ValueError("catalog checkpoint image exceeds restore cut")
        self._staged_image = None
        self._committed_image = image

    async def _commit_prepared_image(
        self, image: CatalogCheckpointImage | None, position: int
    ) -> None:
        async with self._lock:
            self._commit_image(image, position)

    def _commit_image(
        self, image: CatalogCheckpointImage | None, position: int
    ) -> None:
        if image is None:
            return
        if image.end_position > position:
            raise ValueError("catalog checkpoint commit precedes staged image")
        if self._committed_image is not None:
            if image == self._committed_image:
                if self._staged_image == image:
                    self._staged_image = None
                return
            if image.end_position <= self._committed_image.end_position:
                raise ValueError("catalog checkpoint commit regressed")
        self._committed_image = image
        if self._staged_image == image:
            self._staged_image = None

    def prepare_checkpoint(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        state_digest: str,
    ) -> PreparedCheckpointTransaction[CatalogCheckpointImage | None]:
        _identity(checkpoint_id, "catalog checkpoint id")
        if not isinstance(permit, CheckpointCommitPermit):
            raise TypeError(
                "catalog checkpoint prepare requires CheckpointCommitPermit"
            )
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("catalog checkpoint prepare requires positioned cut")
        _identity(state_digest, "catalog checkpoint state digest")
        state = self.checkpoint_state(cut)
        actual_digest = metadata_digest(state)
        if actual_digest != state_digest:
            raise ValueError("catalog checkpoint prepare state digest is invalid")
        serialized_image = state["image"]
        prepared_image = (
            None
            if serialized_image is None
            else CatalogCheckpointImage.from_mapping(serialized_image)
        )
        fingerprint = CheckpointTransactionFingerprint.create(
            permit=permit, cut=cut, state_digest=state_digest
        )
        self._checkpoint_transactions.prepare(
            checkpoint_id, fingerprint, prepared_image
        )
        return self._checkpoint_transactions.prepared(checkpoint_id)

    def apply_prepared(
        self,
        prepared_state: PreparedCheckpointTransaction[CatalogCheckpointImage | None],
    ) -> None:
        if not isinstance(prepared_state, PreparedCheckpointTransaction):
            raise TypeError("catalog checkpoint prepared state has invalid type")

        def apply(image: CatalogCheckpointImage | None) -> None:
            if image is not None:
                self._commit_image(image, image.end_position)

        self._checkpoint_transactions.apply(
            prepared_state.checkpoint_id,
            prepared_state.fingerprint,
            apply,
        )

    def abort_prepared(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        _identity(checkpoint_id, "catalog checkpoint id")
        if not isinstance(permit, CheckpointCommitPermit):
            raise TypeError("catalog checkpoint abort requires CheckpointCommitPermit")
        self._checkpoint_transactions.abort(checkpoint_id, permit)


__all__ = [
    "CatalogCheckpointImage",
    "IngestionCatalogCheckpointComponent",
]
