"""Execution-side physical resource registration and handle resolution."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Generator
from typing import Any

from pydantic import ConfigDict, field_validator
from fluxway.common.schema import FluxwayBaseSchema


class ResourceHandle(FluxwayBaseSchema):
    """Immutable handle pointing to an execution-side physical resource."""

    model_config = ConfigDict(frozen=True)

    namespace: str
    resource_id: str

    @field_validator("namespace", "resource_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("resource identifier must be a non-empty string")
        return value


class ResourceRegistry:
    """Thread-safe container for physical execution resources."""

    def __init__(self) -> None:
        self._resources: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    def register(self, namespace: str, resource_id: str, resource: Any) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise TypeError("namespace must be a non-empty string")
        if not isinstance(resource_id, str) or not resource_id:
            raise TypeError("resource_id must be a non-empty string")
        with self._lock:
            key = (namespace, resource_id)
            if key in self._resources:
                raise RuntimeError(
                    f"Resource {resource_id!r} in namespace {namespace!r} is already registered."
                )
            self._resources[key] = resource

    def resolve(self, namespace: str, resource_id: str) -> Any:
        if not isinstance(namespace, str) or not namespace:
            raise TypeError("namespace must be a non-empty string")
        if not isinstance(resource_id, str) or not resource_id:
            raise TypeError("resource_id must be a non-empty string")
        with self._lock:
            key = (namespace, resource_id)
            if key not in self._resources:
                raise RuntimeError(
                    f"Resource {resource_id!r} in namespace {namespace!r} not found. "
                    "Ensure the Operator has initialized and registered it."
                )
            return self._resources[key]

    def unregister(self, namespace: str, resource_id: str) -> Any:
        if not isinstance(namespace, str) or not namespace:
            raise TypeError("namespace must be a non-empty string")
        if not isinstance(resource_id, str) or not resource_id:
            raise TypeError("resource_id must be a non-empty string")
        with self._lock:
            key = (namespace, resource_id)
            if key not in self._resources:
                raise RuntimeError(
                    f"Resource {resource_id!r} in namespace {namespace!r} not found."
                )
            return self._resources.pop(key)

    def contains(self, namespace: str, resource_id: str) -> bool:
        with self._lock:
            return (namespace, resource_id) in self._resources

    def clear(self) -> None:
        with self._lock:
            self._resources.clear()


_ACTIVE_REGISTRY: contextvars.ContextVar[ResourceRegistry | None] = (
    contextvars.ContextVar("active_resource_registry", default=None)
)


def get_active_resource_registry() -> ResourceRegistry:
    registry = _ACTIVE_REGISTRY.get()
    if registry is None:
        raise RuntimeError(
            "No active ResourceRegistry bound in the current execution context. "
            "Ensure bind_resource_registry() or set_active_resource_registry() is called."
        )
    return registry


def set_active_resource_registry(registry: ResourceRegistry) -> contextvars.Token:
    if not isinstance(registry, ResourceRegistry):
        raise TypeError("registry must be an instance of ResourceRegistry")
    return _ACTIVE_REGISTRY.set(registry)


def reset_active_resource_registry(token: contextvars.Token | None = None) -> None:
    if token is not None:
        _ACTIVE_REGISTRY.reset(token)
    else:
        _ACTIVE_REGISTRY.set(None)


@contextlib.contextmanager
def bind_resource_registry(
    registry: ResourceRegistry,
) -> Generator[ResourceRegistry]:
    if not isinstance(registry, ResourceRegistry):
        raise TypeError("registry must be an instance of ResourceRegistry")
    token = _ACTIVE_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_REGISTRY.reset(token)


__all__ = [
    "ResourceHandle",
    "ResourceRegistry",
    "get_active_resource_registry",
    "set_active_resource_registry",
    "reset_active_resource_registry",
    "bind_resource_registry",
]
