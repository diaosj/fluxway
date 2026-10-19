"""Controller-owned source progress supervision."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum


class SourceProgressState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class SourceProgressSupervisor:
    """Own one source-window progress task per execution generation."""

    def __init__(
        self,
        progress: Callable[[], Awaitable[None]],
        *,
        on_failure: Callable[[BaseException], Awaitable[None]] | None = None,
    ) -> None:
        if not callable(progress):
            raise TypeError("progress must be callable")
        if on_failure is not None and not callable(on_failure):
            raise TypeError("on_failure must be callable when provided")
        self._progress = progress
        self._on_failure = on_failure
        self._state = SourceProgressState.STOPPED
        self._generation = 0
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._failure_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> SourceProgressState:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        failure: BaseException | None = None
        async with self._lock:
            if self._state is SourceProgressState.RUNNING:
                return
            if self._state is SourceProgressState.STARTING:
                return
            if self._state is SourceProgressState.STOPPING:
                raise RuntimeError("source progress supervisor is stopping")
            if self._state is SourceProgressState.FAILED:
                failure = self._failure
            else:
                self._generation += 1
                self._failure = None
                self._wake.set()
                self._state = SourceProgressState.STARTING
                self._task = asyncio.create_task(self._run(self._generation))
                self._state = SourceProgressState.RUNNING
        if failure is not None:
            await self._join_failed_task()
            await self._join_failure_callback()
            raise RuntimeError("source progress supervisor failed") from failure

    async def stop(self) -> None:
        failure: BaseException | None = None
        async with self._lock:
            if self._state is SourceProgressState.STOPPED:
                return
            if self._state is SourceProgressState.FAILED:
                failure = self._failure
                task = None
            else:
                task = self._task
            if failure is None and task is None:
                self._state = SourceProgressState.STOPPED
                return
            if failure is None:
                assert task is not None
                self._state = SourceProgressState.STOPPING
                self._wake.set()
                if not task.done():
                    task.cancel()
        if failure is not None:
            await self._join_failed_task()
            await self._join_failure_callback()
            raise RuntimeError("source progress supervisor failed") from failure
        assert task is not None
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            async with self._lock:
                self._record_failure_locked(error)
            await self.raise_if_failed()
        async with self._lock:
            if self._failure is None:
                self._state = SourceProgressState.STOPPED
                self._task = None
            else:
                self._state = SourceProgressState.FAILED
        await self._join_failure_callback()
        await self.raise_if_failed()

    async def raise_if_failed(self) -> None:
        async with self._lock:
            failure = self._failure
        if failure is None:
            return
        await self._join_failed_task()
        await self._join_failure_callback()
        raise RuntimeError("source progress supervisor failed") from failure

    async def _run(self, generation: int) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                if generation != self._generation:
                    raise RuntimeError("source progress supervisor generation changed")
                await self._progress()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            async with self._lock:
                self._record_failure_locked(error)
            raise

    def _record_failure_locked(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        self._state = SourceProgressState.FAILED
        if self._on_failure is not None:
            self._failure_task = asyncio.ensure_future(self._on_failure(error))

    async def _join_failure_callback(self) -> None:
        task = self._failure_task
        if task is None:
            return
        self._failure_task = None
        await task

    async def _join_failed_task(self) -> None:
        task = self._task
        if task is None:
            return
        if not task.done():
            await asyncio.wait((task,))
        task.exception()
