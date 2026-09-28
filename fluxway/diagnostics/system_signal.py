from __future__ import annotations

import os
import selectors
import signal
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FrameType
from typing import Any

import ray
from ray._private.event.event_logger import get_event_logger
from ray._private.ray_constants import WORKER_PROCESS_SETUP_HOOK_ENV_VAR
from ray._private.worker import global_worker
from ray.core.generated.event_pb2 import Event


_SIGNALS: tuple[int, ...] = tuple(
    int(signal_number)
    for signal_number in (
        signal.SIGTERM,
        signal.SIGINT,
        signal.SIGHUP,
        signal.SIGQUIT,
    )
    if hasattr(signal, signal.Signals(signal_number).name)
)
_RAY_HANDLER_PATCH_MARKER = "_fluxway_ray_signal_handler_patch"
_DIRECT_HANDLER_MARKER = "_fluxway_system_signal_reporter_installed"
_EMITTER_MARKER = "_fluxway_system_signal_emitter"
_EVENT_QUEUE_CAPACITY = 64
_EVENT_WAKE_BYTE = b"\x00"
_USER_SETUP_HOOK_ENV = "FLUXWAY_USER_WORKER_PROCESS_SETUP_HOOK"
_BOOTSTRAP_HOOK_PATH = "fluxway.diagnostics.system_signal.worker_process_setup_hook"


@dataclass
class _SignalMetadata:
    spmd_id: str = ""
    rank: int | None = None


_metadata = _SignalMetadata()


@dataclass(frozen=True)
class _SignalEvent:
    logger: Any
    fields: dict[str, Any]


class _SignalEventEmitter:
    def __init__(self) -> None:
        self._events: deque[_SignalEvent] = deque(maxlen=_EVENT_QUEUE_CAPACITY)
        self._dropped_events = 0
        self._delivered_events = 0
        self._in_flight = 0
        self._queue_lock = threading.Lock()
        self._read_fd, self._write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self._thread = threading.Thread(
            target=self._run,
            name="fluxway-system-signal-emitter",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def delivered_events(self) -> int:
        return self._delivered_events

    def publish(self, event: _SignalEvent) -> None:
        with self._queue_lock:
            if len(self._events) + self._in_flight >= _EVENT_QUEUE_CAPACITY:
                self._dropped_events += 1
                return
            self._events.append(event)
        try:
            os.write(self._write_fd, _EVENT_WAKE_BYTE)
        except BlockingIOError:
            pass

    def _run(self) -> None:
        selector = selectors.DefaultSelector()
        selector.register(self._read_fd, selectors.EVENT_READ)
        while True:
            selector.select(timeout=1.0)
            while True:
                try:
                    os.read(self._read_fd, 4096)
                except BlockingIOError:
                    break
                with self._queue_lock:
                    if not self._events:
                        break
                    event = self._events.popleft()
                    self._in_flight += 1
                try:
                    event.logger.error(
                        "Fluxway worker received system signal", **event.fields
                    )
                    self._delivered_events += 1
                finally:
                    with self._queue_lock:
                        self._in_flight -= 1


def _event_emitter() -> _SignalEventEmitter:
    emitter = getattr(global_worker, _EMITTER_MARKER, None)
    if emitter is None:
        emitter = _SignalEventEmitter()
        setattr(global_worker, _EMITTER_MARKER, emitter)
    return emitter


def _handler_name(handler: Any) -> str:
    if handler is signal.SIG_DFL:
        return "SIG_DFL"
    if handler is signal.SIG_IGN:
        return "SIG_IGN"
    if isinstance(handler, Callable):
        return getattr(handler, "__qualname__", repr(handler))
    return repr(handler)


def _set_metadata(*, spmd_id: str, rank: int) -> None:
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    _metadata.spmd_id = spmd_id
    _metadata.rank = rank


def _event_context() -> tuple[Any, dict[str, Any]]:
    assert global_worker.node is not None
    logs_dir = global_worker.node.get_logs_dir_path()
    logger = get_event_logger(Event.SourceType.CORE_WORKER, logs_dir)
    context = ray.get_runtime_context()
    return logger, {
        "diagnostic_kind": "system_signal",
        "job_id": context.get_job_id(),
        "actor_id": context.get_actor_id(),
        "worker_id": context.get_worker_id(),
        "pid": str(os.getpid()),
        "node_id": context.get_node_id(),
    }


def _delegate(signum: int, frame: FrameType | None, previous: Any) -> None:
    signal_name = signal.Signals(signum).name
    if callable(previous):
        previous(signum, frame)
    elif previous is signal.SIG_DFL:
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)
    elif previous is signal.SIG_IGN:
        return
    else:
        raise RuntimeError(
            f"unsupported previous handler for {signal_name}: {_handler_name(previous)}"
        )


def _build_reporter(
    emitter: _SignalEventEmitter,
    base_fields: dict[str, Any],
    previous_handlers: Mapping[int, Any],
    logger: Any,
) -> Callable[[int, FrameType | None], None]:
    def report(signum: int, frame: FrameType | None) -> None:
        signal_name = signal.Signals(signum).name
        fields = {
            **base_fields,
            "spmd_id": _metadata.spmd_id,
            "rank": "" if _metadata.rank is None else str(_metadata.rank),
            "signal_number": str(signum),
            "signal_name": signal_name,
        }
        try:
            emitter.publish(_SignalEvent(logger=logger, fields=fields))
        finally:
            _delegate(signum, frame, previous_handlers[signum])

    return report


def _install_handlers(previous_handlers: Mapping[int, Any]) -> None:
    logger, base_fields = _event_context()
    emitter = _event_emitter()
    reporter = _build_reporter(emitter, base_fields, previous_handlers, logger)
    for signum in _SIGNALS:
        signal.signal(signum, reporter)


def register_system_signal_metadata(*, spmd_id: str, rank: int) -> None:
    """Publish actor identity for a process-level signal adapter."""

    _set_metadata(spmd_id=spmd_id, rank=rank)


def install_system_signal_bootstrap() -> None:
    """Decorate Ray's main-thread SIGTERM installation during worker bootstrap."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "system signal bootstrap must run on the process main thread"
        )

    import ray._private.utils as ray_utils

    if getattr(ray_utils, _RAY_HANDLER_PATCH_MARKER, False):
        return

    original_set_sigterm_handler = ray_utils.set_sigterm_handler
    previous_non_ray_handlers: dict[int, Any] = {
        signum: signal.getsignal(signum)
        for signum in _SIGNALS
        if signum != int(signal.SIGTERM)
    }

    def set_sigterm_handler(
        sigterm_handler: Callable[[int, FrameType | None], None],
    ) -> None:
        previous_handlers: dict[int, Any] = {
            **previous_non_ray_handlers,
            int(signal.SIGTERM): sigterm_handler,
        }
        logger, base_fields = _event_context()
        emitter = _event_emitter()
        reporter = _build_reporter(emitter, base_fields, previous_handlers, logger)
        original_set_sigterm_handler(reporter)
        for signum in _SIGNALS:
            if signum != int(signal.SIGTERM):
                signal.signal(signum, reporter)

    ray_utils.set_sigterm_handler = set_sigterm_handler
    setattr(ray_utils, _RAY_HANDLER_PATCH_MARKER, True)


def compose_worker_process_setup_hook(user_hook: Any | None) -> str:
    """Return the importable worker bootstrap hook path."""

    if user_hook is not None and not isinstance(user_hook, str):
        raise TypeError(
            "worker_process_setup_hook must be a module path string or None"
        )
    return _BOOTSTRAP_HOOK_PATH


def prepare_worker_runtime_env(
    runtime_env: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the worker runtime environment for Fluxway signal diagnostics."""

    prepared = dict(runtime_env or {})
    raw_env_vars = prepared.get("env_vars", {})
    if not isinstance(raw_env_vars, Mapping):
        raise TypeError("runtime_env.env_vars must be a mapping")
    env_vars = dict(raw_env_vars)
    if WORKER_PROCESS_SETUP_HOOK_ENV_VAR in env_vars:
        raise ValueError(
            "runtime_env.env_vars must not define Ray's worker setup hook directly"
        )
    user_hook = prepared.pop("worker_process_setup_hook", None)
    env_vars[WORKER_PROCESS_SETUP_HOOK_ENV_VAR] = compose_worker_process_setup_hook(
        user_hook
    )
    if user_hook is not None:
        env_vars[_USER_SETUP_HOOK_ENV] = user_hook
    prepared["env_vars"] = env_vars
    return prepared


def worker_process_setup_hook() -> None:
    """Run the Fluxway bootstrap and the optional importable user hook."""

    user_hook = os.environ.get(_USER_SETUP_HOOK_ENV)
    if user_hook:
        from ray._common.utils import load_class

        loaded_hook = load_class(user_hook)
        if not callable(loaded_hook):
            raise TypeError(f"loaded worker setup hook is not callable: {user_hook}")
        loaded_hook()
    install_system_signal_bootstrap()


def install_system_signal_reporter(*, spmd_id: str, rank: int) -> None:
    """Install the direct main-thread adapter or publish async-actor metadata."""

    register_system_signal_metadata(spmd_id=spmd_id, rank=rank)
    if threading.current_thread() is not threading.main_thread():
        return

    if getattr(global_worker, _DIRECT_HANDLER_MARKER, False):
        return

    previous_handlers: dict[int, Any] = {
        signum: signal.getsignal(signum) for signum in _SIGNALS
    }
    _install_handlers(previous_handlers)
    setattr(global_worker, _DIRECT_HANDLER_MARKER, True)


__all__ = [
    "compose_worker_process_setup_hook",
    "install_system_signal_bootstrap",
    "install_system_signal_reporter",
    "register_system_signal_metadata",
    "prepare_worker_runtime_env",
    "worker_process_setup_hook",
]
