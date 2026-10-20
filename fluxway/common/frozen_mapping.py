from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class FrozenMap[K, V](Mapping[K, V]):
    __slots__ = ("_items", "_values")

    def __init__(self, value: Mapping[K, V]) -> None:
        self._items = tuple(sorted(value.items(), key=lambda item: repr(item[0])))
        self._values = dict(self._items)

    def __getitem__(self, key: K) -> V:
        return self._values[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMap[K, V]:
        return self

    def __reduce__(self):
        return FrozenMap, (dict(self._items),)

    def __hash__(self) -> int:
        return hash(self._items)


def freeze_mapping[K, V](value: Mapping[K, V]) -> Mapping[K, V]:
    if isinstance(value, FrozenMap):
        return value
    return FrozenMap(value)
