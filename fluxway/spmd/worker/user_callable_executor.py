"""User callable execution lanes for worker-local DAG nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fluxway.spmd.worker.runtime import (
    WORKER_PROCESS_DEFAULT_EXECUTION_LANE,
    WorkerProcessCallRuntime,
)

if TYPE_CHECKING:
    from fluxway.common.compiled_graph import CompiledOperator


USER_CALLABLE_DEFAULT_OPERATOR_LANE = WORKER_PROCESS_DEFAULT_EXECUTION_LANE


class UserCallableLane(StrEnum):
    COROUTINE_ASYNC = "coroutine_async"
    WORKER_PROCESS = "worker_process"


@dataclass(frozen=True, slots=True)
class SPMDObjectScope:
    spmd_id: str
    rank: int

    def __post_init__(self) -> None:
        if not self.spmd_id:
            raise ValueError("SPMD object scope requires a non-empty spmd_id")
        if self.rank < 0:
            raise ValueError("SPMD object scope rank must be non-negative")


@dataclass(frozen=True, slots=True)
class UserCallableSpec:
    callable_id: str
    fn: Callable[..., Any]
    resource_id: int
    operator_lane: str
    operator_is_coroutine_function: bool
    operator_step: int | None = None
    object_scope: SPMDObjectScope | None = None
    graph_id: str | None = None
    compiled_operator: CompiledOperator | None = None


@dataclass(frozen=True, slots=True)
class SelectedUserCallable:
    spec: UserCallableSpec
    lane: UserCallableLane


class UserCallableExecutor:
    def __init__(
        self,
        *,
        worker_process_runtime: WorkerProcessCallRuntime,
    ) -> None:
        self._worker_process_runtime = worker_process_runtime

    async def execute(self, spec: UserCallableSpec) -> Any:
        return await self.execute_selected(self.select(spec))

    def select(self, spec: UserCallableSpec) -> SelectedUserCallable:
        if spec.operator_is_coroutine_function:
            return SelectedUserCallable(
                spec=spec,
                lane=UserCallableLane.COROUTINE_ASYNC,
            )
        if not spec.operator_lane:
            raise ValueError(
                f"worker-process callable {spec.callable_id!r} requires operator_lane."
            )
        return SelectedUserCallable(spec=spec, lane=UserCallableLane.WORKER_PROCESS)

    async def execute_selected(self, selected: SelectedUserCallable) -> Any:
        if selected.lane is UserCallableLane.COROUTINE_ASYNC:
            return await selected.spec.fn()
        if selected.lane is UserCallableLane.WORKER_PROCESS:
            return await self._execute_worker_process(selected.spec)
        raise ValueError(f"unsupported user callable lane {selected.lane!r}.")

    async def drain(self) -> None:
        await self._worker_process_runtime.drain()

    async def shutdown(self) -> None:
        await self.drain()

    async def _execute_worker_process(self, spec: UserCallableSpec) -> Any:
        return await self._worker_process_runtime.submit(
            spec.fn,
            callable_id=spec.callable_id,
            resource_id=spec.resource_id,
            execution_lane=spec.operator_lane,
        )
