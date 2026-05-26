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
    async def receive_shards(
        self,
        tuuids: list[int],
        step: int,
        sender_id: str,
        sender_rank: int,
        payloads: list[Any],
        counts: list[int],
        global_shapes: list[tuple[int, ...] | None] | None = None,
        meta_tensor_state_list: list[int] | None = None,
        slice_compact_desc_params: list[list[int] | None] | None = None,
    ) -> Any:
        """
        Data Plane: Primary method for tensor shard transmission.
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

    def __init__(self, handle: Any):
        self._handle = handle

    async def receive_shards(
        self,
        tuuids: list[int],
        step: int,
        sender_id: str,
        sender_rank: int,
        payloads: list[Any],
        counts: list[int],
        global_shapes: list[tuple[int, ...] | None] | None = None,
        meta_tensor_state_list: list[int] | None = None,
        slice_compact_desc_params: list[list[int] | None] | None = None,
    ) -> Any:
        cpu_payloads = [ensure_cpu_serializable(p) for p in payloads]
        return await self._handle.receive_shards.remote(
            tuuids=tuuids,
            step=step,
            sender_id=sender_id,
            sender_rank=sender_rank,
            payloads=cpu_payloads,
            counts=counts,
            global_shapes=global_shapes,
            meta_tensor_state_list=meta_tensor_state_list,
            slice_compact_desc_params=slice_compact_desc_params,
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
    def create(cls, transport_type: str, handle: Any) -> DataPlaneCallable:
        impl_cls = cls._registry.get(transport_type)
        if impl_cls is None:
            raise ValueError(f"Unknown transport_type: {transport_type}")
        return impl_cls(handle)


# Register default Ray implementation
DataPlaneRegistry.register("ray", RayActorDataPlane)
