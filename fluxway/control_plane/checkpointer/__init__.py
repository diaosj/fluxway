from .checkpoint import FluxwayCheckpointer
from .backend import WorkerCheckpointBackend
from .directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectory,
    CheckpointDirectoryOwnershipError,
)
from .safetensors import SafeTensorsBackend
from .pickle_backend import PickleBackend
from fluxway.common.types.operator import (
    CheckpointArtifactAborter,
    CheckpointArtifactCommitter,
    CheckpointArtifactPreparer,
    CheckpointContext,
    CheckpointParticipant,
    CheckpointMetadataProvider,
    CheckpointRestoreLayoutProvider,
)
from .protocol import (
    CheckpointMetadata,
    CheckpointState,
    abort_participant_checkpoint_artifacts,
    commit_participant_checkpoint_artifacts,
    restore_participant_state,
    save_participant_state,
    participant_checkpoint_metadata,
    prepare_participant_checkpoint_artifacts,
    validate_checkpoint_state,
)

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

        async def load(self, metadata: dict, restore_layout: dict | None = None) -> dict:
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
    "PickleBackend",
    "OrbaxBackend",
    "CheckpointParticipant",
    "CheckpointContext",
    "CheckpointArtifactPreparer",
    "CheckpointRestoreLayoutProvider",
    "CheckpointMetadataProvider",
    "CheckpointArtifactCommitter",
    "CheckpointArtifactAborter",
    "CheckpointState",
    "CheckpointMetadata",
    "prepare_participant_checkpoint_artifacts",
    "commit_participant_checkpoint_artifacts",
    "abort_participant_checkpoint_artifacts",
    "save_participant_state",
    "restore_participant_state",
    "participant_checkpoint_metadata",
    "validate_checkpoint_state",
]
