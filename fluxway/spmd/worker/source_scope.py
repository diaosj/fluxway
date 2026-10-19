from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from fluxway.common.ingestion import (
    CompiledSegment,
    IngestionLayoutActivation,
    SourceWorkLease,
    WorkerLayoutDescriptor,
)
from fluxway.spmd.worker.source_work import SourceWorkRuntime


@dataclass(frozen=True, slots=True)
class _SourceWindow:
    activation: IngestionLayoutActivation
    runtime: SourceWorkRuntime


class SourceExhaustedError(RuntimeError):
    """Raised when a source step is beyond a published terminal activation."""


class SourceExecutionScope:
    def __init__(
        self,
        *,
        acquire: Callable[
            [int, tuple[CompiledSegment, ...]], Awaitable[tuple[SourceWorkLease, ...]]
        ],
        spawn: Callable[[Coroutine[Any, Any, None]], asyncio.Task],
        completed: Callable[[str], int],
        depth: Callable[[str], int],
    ) -> None:
        self._acquire = acquire
        self._spawn = spawn
        self._completed = completed
        self._depth = depth
        self._windows: dict[str, dict[int, _SourceWindow]] = {}
        self._steps: dict[str, dict[int, _SourceWindow]] = {}
        self._executing: dict[tuple[str, int], tuple[SourceWorkLease, ...]] = {}
        self._taking: set[tuple[str, int]] = set()
        self._installing: dict[str, tuple[IngestionLayoutActivation, asyncio.Task]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self._terminal_end: dict[str, int] = {}
        self._failures: dict[str, BaseException] = {}
        self._cancelled = False

    def _condition(self, operator: str) -> asyncio.Condition:
        return self._conditions.setdefault(operator, asyncio.Condition())

    def require_unactivated(self) -> None:
        if self._windows or self._installing:
            raise RuntimeError("source restore requires a fresh, unactivated Worker")

    async def activate(
        self,
        operator: str,
        activation: IngestionLayoutActivation,
        install: Callable[[IngestionLayoutActivation], Awaitable[None]],
    ) -> None:
        condition = self._condition(operator)
        async with condition:
            if self._cancelled:
                raise asyncio.CancelledError
            windows = self._windows.get(operator, {})
            prior = windows.get(activation.epoch)
            if prior is not None:
                if prior.activation != activation:
                    raise ValueError(
                        "source activation replay conflicts with installed window"
                    )
                return
            pending = self._installing.get(operator)
            if pending is not None:
                if pending[0] != activation:
                    raise ValueError(
                        "source activation conflicts with pending installation"
                    )
                task = pending[1]
            else:
                if windows:
                    previous = windows[max(windows)].activation
                    if (
                        previous.terminal
                        or activation.epoch != previous.epoch + 1
                        or activation.start_position != previous.end_position
                    ):
                        raise ValueError(
                            "source windows must be contiguous and cannot follow terminal"
                        )
                binding = WorkerLayoutDescriptor.decode(activation.descriptor)
                by_step: dict[int, list[CompiledSegment]] = {}
                for segment in binding.segments:
                    if (
                        not activation.start_position
                        <= segment.step
                        < activation.end_position
                    ):
                        raise ValueError(
                            "source segment lies outside its activation window"
                        )
                    by_step.setdefault(segment.step, []).append(segment)
                installed_steps = self._steps.get(operator, {})
                if installed_steps.keys() & by_step.keys():
                    raise ValueError("source windows overlap active steps")
                runtime = SourceWorkRuntime(
                    epoch=activation.epoch,
                    segments_by_step={
                        step: tuple(segments) for step, segments in by_step.items()
                    },
                    batch_steps=self._depth(operator),
                    acquire=self._acquire,
                    spawn=self._spawn,
                    completed=lambda: self._completed(operator),
                )
                task = self._spawn(
                    self._install(
                        operator, activation, install, runtime, tuple(by_step)
                    )
                )
                self._installing[operator] = (activation, task)
        await asyncio.shield(task)

    async def _install(
        self,
        operator: str,
        activation: IngestionLayoutActivation,
        install: Callable[[IngestionLayoutActivation], Awaitable[None]],
        runtime: SourceWorkRuntime,
        steps: tuple[int, ...],
    ) -> None:
        try:
            await install(activation)
            window = _SourceWindow(activation, runtime)
            condition = self._condition(operator)
            async with condition:
                self._windows.setdefault(operator, {})[activation.epoch] = window
                self._steps.setdefault(operator, {}).update(
                    (step, window) for step in steps
                )
                if activation.terminal:
                    self._terminal_end[operator] = activation.end_position
                self._installing.pop(operator, None)
                condition.notify_all()
        except BaseException as error:
            condition = self._condition(operator)
            async with condition:
                self._installing.pop(operator, None)
                self._failures[operator] = error
                condition.notify_all()
            raise

    async def take(self, operator: str, step: int) -> tuple[SourceWorkLease, ...]:
        key = (operator, step)
        condition = self._condition(operator)
        async with condition:
            while True:
                failure = self._failures.get(operator)
                if failure is not None:
                    raise failure
                if self._cancelled:
                    raise asyncio.CancelledError
                window = self._steps.get(operator, {}).get(step)
                if window is not None:
                    if key in self._executing or key in self._taking:
                        raise RuntimeError("source work is already executing")
                    self._taking.add(key)
                    break
                terminal_end = self._terminal_end.get(operator)
                if terminal_end is not None:
                    if step >= terminal_end:
                        raise SourceExhaustedError(
                            f"source step {step} is beyond terminal end {terminal_end}"
                        )
                    raise RuntimeError(
                        f"terminal source activation is missing step {step}"
                    )
                await condition.wait()
        try:
            leases = await window.runtime.take(step)
        except BaseException:
            async with condition:
                self._taking.discard(key)
                condition.notify_all()
            raise
        async with condition:
            self._taking.discard(key)
            self._executing[key] = leases
        return leases

    def finish(self, operator: str, step: int) -> tuple[SourceWorkLease, ...]:
        return self._executing.pop((operator, step))

    async def cancel(self) -> None:
        self._cancelled = True
        for condition in tuple(self._conditions.values()):
            async with condition:
                condition.notify_all()

    async def fail(self, operator: str, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("source scope failure must be an exception")
        condition = self._condition(operator)
        async with condition:
            self._failures[operator] = error
            condition.notify_all()

    async def fail_all(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("source scope failure must be an exception")
        for operator in tuple(self._conditions):
            await self.fail(operator, error)
