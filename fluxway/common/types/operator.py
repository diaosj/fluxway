from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Protocol, Any, runtime_checkable


@dataclass(frozen=True, slots=True)
class CheckpointContext:
    root_dir: str
    step: int
    spmd_id: str
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if not self.root_dir or not self.spmd_id:
            raise ValueError("checkpoint context requires root_dir and spmd_id")
        if self.step < 0 or self.rank < 0 or self.world_size <= 0:
            raise ValueError("checkpoint context contains invalid progress values")
        if self.rank >= self.world_size:
            raise ValueError("checkpoint rank must be smaller than world_size")

    @property
    def spmd_dir(self) -> str:
        return str(Path(self.root_dir) / str(self.step) / self.spmd_id)


@runtime_checkable
class CheckpointParticipant(Protocol):
    """Explicit state boundary shared by source, dense, and sparse SPMDs."""

    def save_state(self, context: CheckpointContext) -> Any: ...

    def restore_state(self, state: Any, context: CheckpointContext) -> None: ...


@runtime_checkable
class CheckpointRestoreLayoutProvider(Protocol):
    """Provides the live pytree/sharding layout used by a backend to restore."""

    def checkpoint_restore_layout(self, context: CheckpointContext) -> Any: ...


@runtime_checkable
class CheckpointArtifactPreparer(Protocol):
    def prepare_checkpoint_artifacts(self, context: CheckpointContext) -> None: ...


@runtime_checkable
class CheckpointMetadataProvider(Protocol):
    def checkpoint_metadata(
        self, saved_state: Any, context: CheckpointContext
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class CheckpointArtifactCommitter(Protocol):
    def commit_checkpoint_artifacts(
        self,
        gathered_metadata: Mapping[int, Mapping[str, Any]],
        context: CheckpointContext,
    ) -> None: ...


@runtime_checkable
class CheckpointArtifactAborter(Protocol):
    def abort_checkpoint_artifacts(self, context: CheckpointContext) -> None: ...
