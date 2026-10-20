from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TransferRelation:
    source_count: int
    producers_by_target: tuple[tuple[int, ...], ...]
    consumers_by_source: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.source_count) is not int or self.source_count <= 0:
            raise ValueError("transfer source count must be positive")
        if not isinstance(self.producers_by_target, tuple):
            raise TypeError("transfer producers_by_target must be a tuple")
        if not self.producers_by_target:
            raise ValueError("transfer requires target coverage")
        for target, producers in enumerate(self.producers_by_target):
            if not isinstance(producers, tuple):
                raise TypeError(f"transfer target {target} producers must be a tuple")
            if not producers or len(set(producers)) != len(producers):
                raise ValueError("transfer target requires unique nonempty producers")
            for producer in producers:
                if type(producer) is not int or not 0 <= producer < self.source_count:
                    raise ValueError("transfer producer is outside source rank domain")
        if not isinstance(self.consumers_by_source, tuple):
            raise TypeError("transfer consumers_by_source must be a tuple")
        if len(self.consumers_by_source) != self.source_count:
            raise ValueError("transfer reverse index must cover every source rank")
        if any(not isinstance(targets, tuple) for targets in self.consumers_by_source):
            raise TypeError("transfer reverse index entries must be tuples")
        expected = _reverse_index(self.source_count, self.producers_by_target)
        if self.consumers_by_source != expected:
            raise ValueError("transfer reverse index does not match producers")

    @classmethod
    def from_producers(
        cls,
        source_count: int,
        producers_by_target: tuple[tuple[int, ...], ...],
    ) -> "TransferRelation":
        consumers = _reverse_index(source_count, producers_by_target)
        return cls(source_count, producers_by_target, consumers)


def _reverse_index(
    source_count: int, producers_by_target: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    if type(source_count) is not int or source_count <= 0:
        raise ValueError("transfer source count must be positive")
    if not isinstance(producers_by_target, tuple):
        raise TypeError("transfer producers_by_target must be a tuple")
    if not producers_by_target:
        raise ValueError("transfer requires target coverage")
    consumers: list[list[int]] = [[] for _ in range(source_count)]
    for target, producers in enumerate(producers_by_target):
        if not isinstance(producers, tuple):
            raise TypeError(f"transfer target {target} producers must be a tuple")
        if not producers or len(set(producers)) != len(producers):
            raise ValueError("transfer target requires unique nonempty producers")
        for producer in producers:
            if type(producer) is not int or not 0 <= producer < source_count:
                raise ValueError("transfer producer is outside source rank domain")
            consumers[producer].append(target)
    return tuple(tuple(targets) for targets in consumers)


@runtime_checkable
class UnitRouting(Protocol):
    def bind(self, source_count: int, target_count: int) -> TransferRelation: ...


@dataclass(frozen=True, slots=True)
class WholeUnitRouting:
    producer_by_target: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.producer_by_target, tuple):
            raise TypeError("whole-unit producer_by_target must be a tuple")

    def bind(self, source_count: int, target_count: int) -> TransferRelation:
        if len(self.producer_by_target) != target_count:
            raise ValueError("whole-unit assignment must cover the target rank domain")
        relation = TransferRelation.from_producers(
            source_count, tuple((p,) for p in self.producer_by_target)
        )
        if any(not targets for targets in relation.consumers_by_source):
            raise ValueError(
                "whole-unit assignment requires complete producer coverage"
            )
        return relation
