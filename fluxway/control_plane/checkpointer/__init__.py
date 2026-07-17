from .checkpoint import FluxwayCheckpointer
from .backend import WorkerCheckpointBackend
from .directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectory,
    CheckpointDirectoryOwnershipError,
)
from .safetensors import SafeTensorsBackend

try:
    from .orbax import OrbaxBackend  # type: ignore
except ModuleNotFoundError as exc:
    _ORBAX_IMPORT_ERROR = exc

    class OrbaxBackend(WorkerCheckpointBackend, backend_name="orbax"):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "OrbaxBackend requires optional jax/orbax dependencies."
            ) from _ORBAX_IMPORT_ERROR

        async def save(
            self,
            state: dict,
            ckpt_dir: str,
            spmd_id: str,
            rank: int,
            step: int,
        ) -> dict:
            raise NotImplementedError()

        async def load(self, metadata: dict, target_state: dict | None = None) -> dict:
            raise NotImplementedError()

        def finalize(self) -> None:
            raise NotImplementedError()


__all__ = [
    "FluxwayCheckpointer",
    "CHECKPOINT_ROOT_MARKER",
    "CheckpointDirectory",
    "CheckpointDirectoryOwnershipError",
    "WorkerCheckpointBackend",
    "SafeTensorsBackend",
    "OrbaxBackend",
]
