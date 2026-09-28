"""Worker-scoped operator instance ownership."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class OperatorInstanceContainer:
    """Own one lazily-created instance for each operator owner class."""

    def __init__(self) -> None:
        self._instances: dict[type, Any] = {}
        self._lock = threading.Lock()

    def get(self, owner_cls: type) -> Any | None:
        if not isinstance(owner_cls, type):
            raise TypeError("operator owner must be a class")
        with self._lock:
            return self._instances.get(owner_cls)

    def require(self, owner_cls: type) -> Any:
        instance = self.get(owner_cls)
        if instance is None:
            raise RuntimeError(
                f"operator owner {owner_cls.__module__}.{owner_cls.__qualname__} "
                "has not been instantiated"
            )
        return instance

    def get_or_create(self, owner_cls: type, factory: Callable[[], Any]) -> Any:
        if not isinstance(owner_cls, type):
            raise TypeError("operator owner must be a class")
        if not callable(factory):
            raise TypeError("operator instance factory must be callable")
        with self._lock:
            instance = self._instances.get(owner_cls)
            if instance is None:
                instance = factory()
                self._instances[owner_cls] = instance
            return instance


__all__ = ["OperatorInstanceContainer"]
