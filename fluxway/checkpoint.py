from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointHold,
    CheckpointIdReference,
    CheckpointManifest,
    CheckpointObjectRef,
    CheckpointObjectSpec,
    CheckpointRuntime,
    CommitAuthority,
    CommitSaveSession,
    ComponentCheckpoint,
    ComponentSpec,
    KeepPolicy,
    NameReference,
    RestoreSession,
    SaveCheckpointRequest,
    SaveSession,
    StepReference,
    WorkerSaveResult,
)
from fluxway.control_plane.checkpointer.backend import (
    CheckpointComponent,
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
    CheckpointObjectCleaner,
    CheckpointObjectLifecycle,
)
from fluxway.control_plane.checkpointer.inline import StateCheckpointComponent

StateSave = Callable[[Any, SaveSession], Any]
StateRestore = Callable[[Any, Any, RestoreSession], Any]


def state_checkpoint_component(
    component_id: str,
    save: StateSave,
    restore: StateRestore,
) -> StateCheckpointComponent:
    return StateCheckpointComponent(component_id, save, restore)


def register_checkpoint_component(
    component: CheckpointComponent,
    *,
    object_lifecycle: CheckpointObjectLifecycle | None = None,
    cleaner: CheckpointObjectCleaner | None = None,
) -> CheckpointComponentRegistration:
    return CheckpointComponentRegistration(
        descriptor=component.descriptor,
        component=component,
        object_lifecycle=object_lifecycle,
        cleaner=cleaner,
    )


__all__ = [
    "CheckpointComponent",
    "CheckpointComponentRegistration",
    "CheckpointComponentRegistry",
    "CheckpointHold",
    "CheckpointObjectCleaner",
    "CheckpointObjectLifecycle",
    "CheckpointIdReference",
    "CheckpointManifest",
    "CheckpointObjectRef",
    "CheckpointObjectSpec",
    "CheckpointRuntime",
    "CommitAuthority",
    "CommitSaveSession",
    "ComponentCheckpoint",
    "ComponentSpec",
    "KeepPolicy",
    "NameReference",
    "RestoreSession",
    "SaveCheckpointRequest",
    "SaveSession",
    "StateCheckpointComponent",
    "StepReference",
    "WorkerSaveResult",
    "state_checkpoint_component",
    "register_checkpoint_component",
]
