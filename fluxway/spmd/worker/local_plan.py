"""Per-worker flat execution plan runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol

from fluxway.spmd.worker.user_callable_executor import (
    UserCallableExecutor,
    UserCallableSpec,
)


class LocalStepNodeKind(IntEnum):
    CALL = 1
    SEND = 2
    WAIT = 3
    BARRIER = 4


@dataclass(frozen=True, slots=True)
class LocalStepPlan:
    node_kinds: tuple[int, ...]
    callable_ids: tuple[str, ...]
    resource_ids: tuple[int, ...]
    dependency_offsets: tuple[int, ...]
    dependencies: tuple[int, ...]
    result_node: int

    @classmethod
    def single_call(cls, callable_id: str, *, resource_id: int = 0) -> LocalStepPlan:
        if not callable_id:
            raise ValueError("callable_id must be non-empty.")
        return cls(
            node_kinds=(int(LocalStepNodeKind.CALL),),
            callable_ids=(callable_id,),
            resource_ids=(resource_id,),
            dependency_offsets=(0, 0),
            dependencies=(),
            result_node=0,
        )

    def validate(self) -> None:
        node_count = len(self.node_kinds)
        if node_count == 0:
            raise ValueError("LocalStepPlan must contain at least one node.")
        if len(self.callable_ids) != node_count:
            raise ValueError("callable_ids length must match node_kinds length.")
        if len(self.resource_ids) != node_count:
            raise ValueError("resource_ids length must match node_kinds length.")
        if len(self.dependency_offsets) != node_count + 1:
            raise ValueError("dependency_offsets length must be node_count + 1.")
        if self.dependency_offsets[0] != 0:
            raise ValueError("dependency_offsets must start at 0.")
        if self.dependency_offsets[-1] != len(self.dependencies):
            raise ValueError("dependency_offsets must end at dependency count.")
        previous = 0
        for offset in self.dependency_offsets:
            if offset < previous:
                raise ValueError("dependency_offsets must be monotonically increasing.")
            previous = offset
        if self.result_node < 0 or self.result_node >= node_count:
            raise ValueError(
                f"result_node={self.result_node} is outside node count {node_count}."
            )
        for index, raw_kind in enumerate(self.node_kinds):
            try:
                LocalStepNodeKind(raw_kind)
            except ValueError as exc:
                raise ValueError(
                    f"node {index} has unsupported kind {raw_kind}."
                ) from exc
        for dep in self.dependencies:
            if dep < 0 or dep >= node_count:
                raise ValueError(
                    f"dependency {dep} is outside node count {node_count}."
                )

    def dependencies_for(self, node_index: int) -> tuple[int, ...]:
        begin = self.dependency_offsets[node_index]
        end = self.dependency_offsets[node_index + 1]
        return self.dependencies[begin:end]


class LocalCallableRegistry(Protocol):
    def spec_for(self, callable_id: str, resource_id: int) -> UserCallableSpec: ...


class MappingCallableRegistry:
    def __init__(self, callables: Mapping[str, Any]):
        self._callables = dict(callables)

    def spec_for(self, callable_id: str, resource_id: int) -> UserCallableSpec:
        fn = self._callables.get(callable_id)
        if fn is None:
            raise KeyError(f"Unknown local callable_id {callable_id!r}.")
        return UserCallableSpec(
            callable_id=callable_id,
            fn=fn,
            is_stateful=False,
            resource_id=resource_id,
        )


class LocalStepPlanExecutor:
    def __init__(
        self, user_callable_executor: UserCallableExecutor | None = None
    ) -> None:
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._user_callable_executor = user_callable_executor or UserCallableExecutor()

    async def execute(
        self, plan: LocalStepPlan, registry: LocalCallableRegistry
    ) -> Any:
        plan.validate()
        node_count = len(plan.node_kinds)
        done_events = tuple(asyncio.Event() for _ in range(node_count))
        node_results: list[Any] = [None] * node_count
        resource_locks: dict[int, asyncio.Lock] = {
            resource_id: asyncio.Lock()
            for resource_id in set(plan.resource_ids)
            if resource_id >= 0
        }

        async def run_node(node_index: int) -> None:
            for dependency in plan.dependencies_for(node_index):
                await done_events[dependency].wait()
            raw_kind = plan.node_kinds[node_index]
            kind = LocalStepNodeKind(raw_kind)
            if kind is not LocalStepNodeKind.CALL:
                raise NotImplementedError(
                    f"LocalStepPlan node kind {kind.name} is not implemented yet."
                )
            resource_id = plan.resource_ids[node_index]
            lock = resource_locks.get(resource_id)
            spec = registry.spec_for(plan.callable_ids[node_index], resource_id)
            if lock is None:
                node_results[node_index] = await self._user_callable_executor.execute(
                    spec
                )
            else:
                async with lock:
                    node_results[
                        node_index
                    ] = await self._user_callable_executor.execute(spec)
            done_events[node_index].set()

        tasks = [asyncio.create_task(run_node(index)) for index in range(node_count)]
        self._active_tasks.update(tasks)
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            for task in tasks:
                self._active_tasks.discard(task)
        return node_results[plan.result_node]

    async def drain_user_callables(self) -> None:
        await self._user_callable_executor.drain()

    @property
    def user_callable_executor(self) -> UserCallableExecutor:
        return self._user_callable_executor

    async def drain(self) -> None:
        while True:
            tasks = [task for task in self._active_tasks if not task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks)
