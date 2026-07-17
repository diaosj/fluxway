from __future__ import annotations

from operator import index
from typing import Any, Protocol, cast


class DLPackTensor(Protocol):
    @property
    def dtype(self) -> Any: ...

    @property
    def shape(self) -> Any: ...

    def __dlpack__(self, *args: Any, **kwargs: Any) -> object: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


def require_dlpack_tensor(value: Any, *, name: str) -> DLPackTensor:
    dlpack = getattr(value, "__dlpack__", None)
    dlpack_device = getattr(value, "__dlpack_device__", None)
    if not callable(dlpack) or not callable(dlpack_device):
        raise TypeError(
            f"{name} must be a DLPack tensor implementing callable __dlpack__ and "
            f"__dlpack_device__, got {type(value).__name__}."
        )
    if not hasattr(value, "dtype") or not hasattr(value, "shape"):
        raise TypeError(
            f"{name} must expose tensor dtype and shape metadata, "
            f"got {type(value).__name__}."
        )
    return cast(DLPackTensor, value)


def dlpack_tensor_shape(value: Any, *, name: str) -> tuple[int, ...]:
    tensor = require_dlpack_tensor(value, name=name)
    try:
        shape = tuple(index(dimension) for dimension in tensor.shape)
    except TypeError as exc:
        raise TypeError(
            f"{name} shape must be an iterable of integers, got {tensor.shape!r}."
        ) from exc
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"{name} shape dimensions must be non-negative, got {shape!r}."
        )
    return shape


def dlpack_device(value: Any, *, name: str) -> tuple[int, int]:
    tensor = require_dlpack_tensor(value, name=name)
    device = tensor.__dlpack_device__()
    if not isinstance(device, tuple) or len(device) != 2:
        raise TypeError(
            f"{name} __dlpack_device__ must return "
            f"(device_type, device_id), got {device!r}."
        )
    try:
        device_type = index(device[0])
        device_id = index(device[1])
    except TypeError as exc:
        raise TypeError(
            f"{name} __dlpack_device__ must return two integers, got {device!r}."
        ) from exc
    if device_id < 0:
        raise TypeError(f"DLPack device id must be non-negative, got {device_id}.")
    return device_type, device_id
