"""Per-worker pickle backend for user-defined Python checkpoint state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ray import cloudpickle

from fluxway.control_plane.checkpointer.backend import WorkerCheckpointBackend


class PickleBackend(WorkerCheckpointBackend, backend_name="pickle"):
    """Persist arbitrary user state inside a worker-owned checkpoint artifact.

    Pickle is deliberately limited to rank-local operator state. Controller
    manifests remain strict pure metadata and never contain this payload.
    Restoring a checkpoint executes its pickle by design, so checkpoints must
    come from a trusted owner.
    """

    def __init__(self, force_write: bool = False):
        self._force_write = force_write

    async def save(
        self,
        state: dict[str, Any],
        ckpt_dir: str,
        spmd_id: str,
        rank: int,
        step: int,
    ) -> dict[str, Any]:
        path = Path(ckpt_dir) / str(step) / spmd_id / f"worker_{rank}" / "state.pkl"
        if path.exists() and not self._force_write:
            raise FileExistsError(f"Checkpoint file already exists: {path}")

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as stream:
                cloudpickle.dump(state, stream)
            temporary.replace(path)

        await asyncio.to_thread(write)
        return {"ckpt_dir": str(path.parent), "pickle_path": str(path)}

    async def load(
        self, metadata: dict[str, Any], restore_layout: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        path = Path(metadata["pickle_path"])

        def read() -> dict[str, Any]:
            with path.open("rb") as stream:
                state = cloudpickle.load(stream)
            if not isinstance(state, dict):
                raise TypeError("pickle checkpoint root must be a state dictionary")
            return state

        return await asyncio.to_thread(read)

    def finalize(self) -> None:
        return None
