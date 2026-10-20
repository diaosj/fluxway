"""Bounded execution lanes for lifecycle effects.

The scheduler owns execution capacity only.  It never owns a state-machine
state and never interprets an effect result; the caller remains responsible
for revision checks and state transitions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
import inspect
import multiprocessing
from typing import Any


class LifecycleEffectExecutionMode(StrEnum):
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"


@dataclass(frozen=True, slots=True)
class LifecycleEffectSubmission:
    operation: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] | None = None
    ordering_key: str = "default"
    state_revision: int = 0
    mode: LifecycleEffectExecutionMode = LifecycleEffectExecutionMode.ASYNC

    def __post_init__(self) -> None:
        if not callable(self.operation):
            raise TypeError("effect operation must be callable")
        if not isinstance(self.args, tuple):
            raise TypeError("effect args must be a tuple")
        if self.kwargs is not None and not isinstance(self.kwargs, Mapping):
            raise TypeError("effect kwargs must be a mapping or None")
        if not isinstance(self.ordering_key, str) or not self.ordering_key:
            raise ValueError("effect ordering_key must be non-empty")
        if not isinstance(self.state_revision, int) or self.state_revision < 0:
            raise ValueError("effect state_revision must be a non-negative integer")
        if not isinstance(self.mode, LifecycleEffectExecutionMode):
            raise TypeError("effect mode must be LifecycleEffectExecutionMode")


@dataclass(frozen=True, slots=True)
class LifecycleEffectOutcome:
    ordering_key: str
    state_revision: int
    value: Any = None
    error: BaseException | None = None


class LifecycleEffectQueueFullError(RuntimeError):
    """Raised when the bounded effect queue has no capacity."""


class LifecycleEffectSchedulerClosedError(RuntimeError):
    """Raised when an effect is submitted after scheduler shutdown starts."""


@dataclass(slots=True)
class _QueuedLifecycleEffect:
    submission: LifecycleEffectSubmission
    outcome: asyncio.Future[LifecycleEffectOutcome]


class BoundedLifecycleEffectScheduler:
    """Run accepted effects with bounded, explicit execution capacity."""

    def __init__(
        self,
        *,
        queue_size: int = 64,
        worker_count: int = 4,
        thread_count: int = 4,
        process_count: int = 1,
    ) -> None:
        for name, value in (
            ("queue_size", queue_size),
            ("worker_count", worker_count),
            ("thread_count", thread_count),
            ("process_count", process_count),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._queue: asyncio.Queue[_QueuedLifecycleEffect] = asyncio.Queue(
            maxsize=queue_size
        )
        self._worker_count = worker_count
        self._thread_count = thread_count
        self._process_count = process_count
        self._workers: tuple[asyncio.Task[None], ...] = ()
        self._thread_executor: ThreadPoolExecutor | None = None
        self._process_executor: ProcessPoolExecutor | None = None
        self._process_executor_lock = asyncio.Lock()
        self._ordering_locks: dict[str, asyncio.Lock] = {}
        self._start_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._closed = False

    @property
    def queue_size(self) -> int:
        return self._queue.maxsize

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def started(self) -> bool:
        return self._started and not self._closing and not self._closed

    async def start(self) -> None:
        async with self._start_lock:
            if self._closed or self._closing:
                raise LifecycleEffectSchedulerClosedError("effect scheduler is closing")
            if self._started:
                return
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self._thread_count,
                thread_name_prefix="fluxway-effect",
            )
            self._process_executor = None
            self._workers = tuple(
                asyncio.create_task(self._worker_loop(index))
                for index in range(self._worker_count)
            )
            self._started = True

    def submit(
        self, submission: LifecycleEffectSubmission
    ) -> asyncio.Future[LifecycleEffectOutcome]:
        if not isinstance(submission, LifecycleEffectSubmission):
            raise TypeError("effect scheduler accepts LifecycleEffectSubmission")
        if not self._started:
            raise RuntimeError("effect scheduler must be started before submit")
        if self._closing or self._closed:
            raise LifecycleEffectSchedulerClosedError("effect scheduler is closed")
        loop = asyncio.get_running_loop()
        owner_future: asyncio.Future[LifecycleEffectOutcome] = loop.create_future()
        try:
            self._queue.put_nowait(_QueuedLifecycleEffect(submission, owner_future))
        except asyncio.QueueFull as error:
            raise LifecycleEffectQueueFullError(
                f"effect queue is full (capacity={self._queue.maxsize})"
            ) from error
        # Shielding makes cancellation of the caller's wait independent from
        # execution of the already accepted owner effect.
        return asyncio.shield(owner_future)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                return
            self._closing = True
            await self._queue.join()
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = ()
            await self._shutdown_executor(self._thread_executor)
            await self._shutdown_executor(self._process_executor)
            self._thread_executor = None
            self._process_executor = None
            self._closed = True

    async def _worker_loop(self, index: int) -> None:
        del index
        while True:
            item = await self._queue.get()
            try:
                lock = self._ordering_locks.setdefault(
                    item.submission.ordering_key, asyncio.Lock()
                )
                async with lock:
                    result = await self._execute(item.submission)
                if not item.outcome.cancelled():
                    item.outcome.set_result(result)
            except BaseException as error:
                if not item.outcome.cancelled():
                    item.outcome.set_result(
                        LifecycleEffectOutcome(
                            ordering_key=item.submission.ordering_key,
                            state_revision=item.submission.state_revision,
                            error=error,
                        )
                    )
            finally:
                self._queue.task_done()

    async def _execute(
        self, submission: LifecycleEffectSubmission
    ) -> LifecycleEffectOutcome:
        kwargs = dict(submission.kwargs or {})
        try:
            if submission.mode is LifecycleEffectExecutionMode.ASYNC:
                value = submission.operation(*submission.args, **kwargs)
                if not inspect.isawaitable(value):
                    raise TypeError("async effect operation must return an awaitable")
                value = await value
            elif submission.mode is LifecycleEffectExecutionMode.THREAD:
                value = await self._run_executor(
                    self._thread_executor, submission.operation, submission.args, kwargs
                )
            elif submission.mode is LifecycleEffectExecutionMode.PROCESS:
                async with self._process_executor_lock:
                    if self._process_executor is None:
                        self._process_executor = await asyncio.to_thread(
                            ProcessPoolExecutor,
                            max_workers=self._process_count,
                            mp_context=multiprocessing.get_context("spawn"),
                        )
                value = await self._run_executor(
                    self._process_executor,
                    submission.operation,
                    submission.args,
                    kwargs,
                )
            else:
                raise RuntimeError(f"unsupported effect mode: {submission.mode}")
        except BaseException as error:
            return LifecycleEffectOutcome(
                ordering_key=submission.ordering_key,
                state_revision=submission.state_revision,
                error=error,
            )
        return LifecycleEffectOutcome(
            ordering_key=submission.ordering_key,
            state_revision=submission.state_revision,
            value=value,
        )

    async def _run_executor(
        self,
        executor: Executor | None,
        operation: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if executor is None:
            raise RuntimeError("effect executor is not started")
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                executor, _invoke_with_kwargs, operation, args, dict(kwargs)
            )
        return await loop.run_in_executor(executor, operation, *args)

    @staticmethod
    async def _shutdown_executor(executor: Executor | None) -> None:
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)


def _invoke_with_kwargs(
    operation: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    return operation(*args, **kwargs)


__all__ = [
    "BoundedLifecycleEffectScheduler",
    "LifecycleEffectExecutionMode",
    "LifecycleEffectOutcome",
    "LifecycleEffectQueueFullError",
    "LifecycleEffectSchedulerClosedError",
    "LifecycleEffectSubmission",
]
