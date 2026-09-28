from __future__ import annotations

from fluxway.common.decorators import operator, operator_context
from fluxway.common.operator_instances import OperatorInstanceContainer


class SharedOwner:
    def __init__(self, offset: int = 0) -> None:
        self.init_rank = self.rank
        self.offset = offset
        self.calls = 0

    @operator(name="first")
    def first(self, value: int) -> int:
        self.calls += 1
        return value + self.offset

    @operator(name="second")
    def second(self, value: int) -> int:
        self.calls += 1
        return value + self.offset


def test_context_container_preserves_owner_sharing_and_rank_injection() -> None:
    container = OperatorInstanceContainer()
    with operator_context(
        static_args={"offset": 10},
        rank=2,
        world_size=4,
        instance_container=container,
    ):
        assert SharedOwner.first(5) == 15
        assert SharedOwner.second(7) == 17
    instance = container.require(SharedOwner)
    assert instance.init_rank == 2
    assert instance.calls == 2


def test_context_containers_isolate_same_owner_class() -> None:
    first = OperatorInstanceContainer()
    second = OperatorInstanceContainer()
    with operator_context(
        static_args={"offset": 100}, rank=0, world_size=2, instance_container=first
    ):
        assert SharedOwner.first(1) == 101
    with operator_context(
        static_args={"offset": 200}, rank=1, world_size=2, instance_container=second
    ):
        assert SharedOwner.first(1) == 201
    assert first.require(SharedOwner) is not second.require(SharedOwner)
