from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import json
from typing import Any

from fluxway.profiler.recorder import TraceRecorder


_ACTIVE_USER_PROFILER: contextvars.ContextVar[UserProfilerHandle | None] = (
    contextvars.ContextVar("fluxway_active_user_profiler", default=None)
)
_RESERVED_ATTR_KEYS = frozenset({"op_id", "step", "rank", "execution_mode", "attrs_id"})


class _NoopScope:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        return False


_NOOP_SCOPE = _NoopScope()


class UserProfilerHandle:
    __slots__ = ("_recorder", "_op_name", "_step", "_rank", "_execution_mode")

    def __init__(
        self,
        *,
        recorder: TraceRecorder,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
    ) -> None:
        self._recorder = recorder
        self._op_name = op_name
        self._step = step
        self._rank = rank
        self._execution_mode = execution_mode

    @contextmanager
    def scope(self, name: str, attrs: dict[str, Any]) -> Generator[None, None, None]:
        event_name = _event_name(name)
        attrs_json = _attrs_json(attrs)
        with self._recorder.user_custom_scope(
            name=event_name,
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
            attrs_json=attrs_json,
        ):
            yield

    def event(self, name: str, attrs: dict[str, Any]) -> None:
        self._recorder.user_custom_event(
            name=_event_name(name),
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
            attrs_json=_attrs_json(attrs),
        )


def enabled() -> bool:
    return _ACTIVE_USER_PROFILER.get() is not None


def scope(name: str, **attrs: Any) -> Any:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None:
        return _NOOP_SCOPE
    return handle.scope(name, attrs)


def event(name: str, **attrs: Any) -> None:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None:
        return
    handle.event(name, attrs)


@contextmanager
def activate_user_profiler(
    *,
    recorder: TraceRecorder,
    op_name: str,
    step: int,
    rank: int,
    execution_mode: str,
) -> Generator[None, None, None]:
    handle = UserProfilerHandle(
        recorder=recorder,
        op_name=op_name,
        step=step,
        rank=rank,
        execution_mode=execution_mode,
    )
    token = _ACTIVE_USER_PROFILER.set(handle)
    try:
        yield
    finally:
        _ACTIVE_USER_PROFILER.reset(token)


def _event_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("profiler scope/event name must be a string.")
    if not name:
        raise ValueError("profiler scope/event name must be non-empty.")
    if name.startswith("user."):
        raise ValueError("profiler scope/event name must not include the user. prefix.")
    if any(ord(char) < 32 for char in name):
        raise ValueError("profiler scope/event name must not contain control chars.")
    return f"user.{name}"


def _attrs_json(attrs: dict[str, Any]) -> str:
    for key, value in attrs.items():
        _validate_attr_key(key)
        _validate_attr_value(key, value)
    return json.dumps(attrs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_attr_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError("profiler attr keys must be strings.")
    if not key:
        raise ValueError("profiler attr keys must be non-empty.")
    if key in _RESERVED_ATTR_KEYS:
        raise ValueError(f"profiler attr key {key!r} is reserved.")
    if any(ord(char) < 32 for char in key):
        raise ValueError("profiler attr keys must not contain control chars.")


def _validate_attr_value(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, str | int | float):
        return
    raise TypeError(
        f"profiler attr {key!r} must be str, int, float, bool, or None; "
        f"got {type(value).__name__}."
    )
