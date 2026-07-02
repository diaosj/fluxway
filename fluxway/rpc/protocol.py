"""
Fluxway Data Plane RPC Protocol.
Used for high-performance peer-to-peer data transmission (Tensors).
Control plane operations (scheduling, heartbeats) still use native Ray.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from fluxway.common.tensor_backend import ensure_cpu_serializable


class DataPlaneCallable(ABC):
    """
    Abstract base class providing a decoupled interface for data transmission.
    Allows replacing Ray's .remote() with other RPC libraries without changing worker logic.
    """

    @abstractmethod
    def __init__(self, handle: Any, sender_spmd_code: int):
        pass

    @abstractmethod
    async def receive_transport_frame(
        self,
        frame_i32: Any,
        edge_i32: Any,
        payload_i32: Any,
        shape_i32: Any,
        global_shape_i64: Any,
        compact_i64: Any,
        *payload_segments: Any,
    ) -> Any:
        """
        Data Plane: flat transport manifest frame.
        """
        pass

    @abstractmethod
    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        """
        Data Plane: Method for state data transmission.
        """
        pass


class RayActorDataPlane(DataPlaneCallable):
    """
    Ray-specific implementation of DataPlaneCallable.
    """

    def __init__(self, handle: Any, sender_spmd_code: int):
        self._handle = handle
        if not isinstance(sender_spmd_code, int) or sender_spmd_code < 0:
            raise ValueError(
                f"sender_spmd_code must be a non-negative int, got {sender_spmd_code!r}."
            )
        self._sender_spmd_code = sender_spmd_code

    async def receive_transport_frame(
        self,
        frame_i32: Any,
        edge_i32: Any,
        payload_i32: Any,
        shape_i32: Any,
        global_shape_i64: Any,
        compact_i64: Any,
        *payload_segments: Any,
    ) -> Any:
        cpu_segments = tuple(ensure_cpu_serializable(p) for p in payload_segments)
        return await self._handle.receive_transport_frame.remote(
            frame_i32,
            edge_i32,
            payload_i32,
            shape_i32,
            global_shape_i64,
            compact_i64,
            *cpu_segments,
        )

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        return await self._handle.receive_control.remote(op_name, step, names, values)

    def __repr__(self):
        return f"RayActorDataPlane(handle={self._handle})"


class DataPlaneRegistry:
    """Registry for data plane implementations."""

    _registry: ClassVar[dict[str, type[DataPlaneCallable]]] = {}

    @classmethod
    def register(cls, name: str, impl_cls: type[DataPlaneCallable]) -> None:
        cls._registry[name] = impl_cls

    @classmethod
    def create(
        cls, transport_type: str, handle: Any, sender_spmd_code: int
    ) -> DataPlaneCallable:
        impl_cls = cls._registry.get(transport_type)
        if impl_cls is None:
            raise ValueError(f"Unknown transport_type: {transport_type}")
        return impl_cls(handle, sender_spmd_code=sender_spmd_code)


# Register default Ray implementation
DataPlaneRegistry.register("ray", RayActorDataPlane)
