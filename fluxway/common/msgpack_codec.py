"""Typed boundary for the untyped MessagePack dependency."""

from __future__ import annotations

from typing import Any, cast

import msgpack as _msgpack  # type: ignore[import-untyped]


def packb(value: Any, **kwargs: Any) -> bytes:
    return cast(bytes, _msgpack.packb(value, **kwargs))


def unpackb(
    payload: bytes | bytearray | memoryview,
    **kwargs: Any,
) -> Any:
    return _msgpack.unpackb(payload, **kwargs)
