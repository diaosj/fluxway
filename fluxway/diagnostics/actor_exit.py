from __future__ import annotations

from collections.abc import Callable

from ray._private.event.event_logger import get_event_logger
from ray._private.worker import global_worker
from ray.exceptions import ActorDiedError
from ray.core.generated.event_pb2 import Event

_MAX_ERROR_MESSAGE_LENGTH = 4096
ActorExitEmitter = Callable[..., None]


def report_actor_exit(
    error: ActorDiedError,
    *,
    spmd_id: str,
    emit: ActorExitEmitter,
) -> None:
    if not isinstance(error, ActorDiedError):
        raise TypeError(f"error must be ActorDiedError, got {type(error).__name__}")
    if not spmd_id:
        raise ValueError("spmd_id must be non-empty")
    message = str(error)
    if len(message) > _MAX_ERROR_MESSAGE_LENGTH:
        message = message[:_MAX_ERROR_MESSAGE_LENGTH]
    emit(
        "Fluxway Ray actor exited unexpectedly",
        diagnostic_kind="actor_exit",
        spmd_id=spmd_id,
        exception_type=type(error).__name__,
        actor_id=error.actor_id,
        error_message=message,
    )


def report_actor_exit_to_dashboard(error: ActorDiedError, *, spmd_id: str) -> None:
    logs_dir = global_worker.node.get_logs_dir_path()
    event_logger = get_event_logger(Event.SourceType.CORE_WORKER, logs_dir)
    report_actor_exit(error, spmd_id=spmd_id, emit=event_logger.error)


__all__ = ["report_actor_exit", "report_actor_exit_to_dashboard"]
