"""Trusted Python state persisted by a component-owned filesystem checkpoint_object."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ray import cloudpickle

from fluxway.common.types.checkpoint import (
    CheckpointObjectRecord,
    SaveSession,
    RestoreSession,
    RestoreTopologyMode,
)
from fluxway.control_plane.checkpointer._filesystem_component import (
    FilesystemCheckpointComponent,
)


class WorkerLocalCheckpointComponent(FilesystemCheckpointComponent):
    def __init__(
        self,
        component_id: str,
        save_state,
        restore_state,
        retention_domain: str | None = None,
    ) -> None:
        super().__init__(
            component_id,
            save_state,
            restore_state,
            restore_topology=RestoreTopologyMode.EXACT_TOPOLOGY,
            retention_domain=retention_domain,
        )

    async def _write_state(
        self, state: Any, directory: Path, context: SaveSession
    ) -> Mapping[str, Any]:
        path = directory / "state.pkl"

        def write() -> None:
            with path.open("xb") as stream:
                cloudpickle.dump(state, stream)

        await asyncio.to_thread(write)
        return {"file": "state.pkl"}

    async def _read_state(
        self, record: CheckpointObjectRecord, context: RestoreSession
    ) -> Any:
        if len(record.checkpoint_objects) != 1:
            raise ValueError("Pickle record must contain exactly one checkpoint_object")
        path = (
            self._file_path(record.checkpoint_objects[0].storage_locator) / "state.pkl"
        )

        def read() -> Any:
            with path.open("rb") as stream:
                return cloudpickle.load(stream)

        return await asyncio.to_thread(read)

    async def _finish_save_io(self, context: SaveSession) -> None:
        return None

    async def _close_resources(self) -> None:
        return None
