"""Common checkpoint participant protocol for every Fluxway SPMD operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fluxway.common.types.operator import CheckpointContext


CheckpointState = Any
CheckpointMetadata = Mapping[str, Any]


def participant_checkpoint_metadata(
    participant: Any, saved_state: Any, context: CheckpointContext
) -> CheckpointMetadata | None:
    export = getattr(participant, "checkpoint_metadata", None)
    if not callable(export):
        return None
    state = export(saved_state, context)
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint participant metadata must be a mapping")
    validate_checkpoint_state(state)
    return state


def prepare_participant_checkpoint_artifacts(
    participant: Any, context: CheckpointContext
) -> bool:
    begin = getattr(participant, "prepare_checkpoint_artifacts", None)
    if not callable(begin):
        return False
    begin(context)
    return True


def save_participant_state(participant: Any, context: CheckpointContext) -> Any:
    """Invoke a participant without coupling the worker to a concrete class."""

    save = getattr(participant, "save_state", None)
    if not callable(save):
        return None
    return save(context)


def restore_participant_state(
    participant: Any, state: CheckpointState, context: CheckpointContext
) -> None:
    """Invoke a participant's restore hook with one stable calling convention."""

    restore = getattr(participant, "restore_state", None)
    if not callable(restore):
        raise TypeError(
            f"checkpoint participant {type(participant).__name__} does not "
            "implement restore_state"
        )
    restore(state, context)


def commit_participant_checkpoint_artifacts(
    participant: Any,
    context: CheckpointContext,
    gathered_metadata: Mapping[int, CheckpointMetadata],
) -> None:
    """Run the optional cross-rank commit phase for custom participants."""

    commit = getattr(participant, "commit_checkpoint_artifacts", None)
    if not callable(commit):
        return
    commit(gathered_metadata, context)


def abort_participant_checkpoint_artifacts(
    participant: Any,
    context: CheckpointContext,
) -> None:
    abort = getattr(participant, "abort_checkpoint_artifacts", None)
    if callable(abort):
        abort(context)


def validate_checkpoint_state(value: Any, path: str = "checkpoint_state") -> None:
    """Reject runtime/payload objects before they can enter a checkpoint."""

    if value is None or isinstance(value, str | int | float | bool | bytes):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key: {key!r}")
            validate_checkpoint_state(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple | set):
        for index, item in enumerate(value):
            validate_checkpoint_state(item, f"{path}[{index}]")
        return
    module = type(value).__module__
    if module.startswith(("numpy", "pyarrow", "jax", "ray")):
        raise TypeError(
            f"{path} contains unsupported checkpoint payload type "
            f"{type(value).__name__}; persist source metadata instead"
        )
    raise TypeError(
        f"{path} contains unsupported checkpoint state type {type(value).__name__}"
    )
