from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import Future
from threading import Thread
from typing import Any


class IOLane:
    def __init__(self, name: str, *, capacity: int) -> None:
        if not name or capacity < 1:
            raise ValueError("I/O lane requires a name and positive capacity")
        self._name = name
        self._capacity = asyncio.Semaphore(capacity)
        self._owner: asyncio.AbstractEventLoop | None = None
        self._ready: asyncio.Future[asyncio.AbstractEventLoop] | None = None
        self._exited: asyncio.Future[None] | None = None
        self._thread: Thread | None = None
        self._operations: set[asyncio.Task[Any]] = set()
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._unobserved: dict[asyncio.Task[Any], BaseException] = {}
        self._close_task: asyncio.Task[None] | None = None

    def _bind_owner(self) -> asyncio.AbstractEventLoop:
        owner = asyncio.get_running_loop()
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise RuntimeError("I/O lane cannot be shared across owner event loops")
        return owner

    def _start(self) -> asyncio.Future[asyncio.AbstractEventLoop]:
        owner = self._bind_owner()
        if self._ready is None:
            self._ready = owner.create_future()
            self._exited = owner.create_future()
            self._thread = Thread(target=self._run, name=self._name)
            self._thread.start()
        return self._ready

    def _run(self) -> None:
        assert self._owner is not None
        assert self._ready is not None
        assert self._exited is not None
        try:
            with asyncio.Runner() as runner:
                loop = runner.get_loop()
                self._owner.call_soon_threadsafe(self._ready.set_result, loop)
                loop.run_forever()
        except BaseException as error:
            self._owner.call_soon_threadsafe(self._thread_failed, error)
        else:
            self._owner.call_soon_threadsafe(self._exited.set_result, None)

    def _thread_failed(self, error: BaseException) -> None:
        assert self._ready is not None
        assert self._exited is not None
        if not self._ready.done():
            self._ready.set_exception(error)
        self._exited.set_exception(error)

    async def call(self, operation: Callable[..., Any], /, *args, **kwargs) -> Any:
        self._bind_owner()
        if self._close_task is not None:
            raise RuntimeError("I/O lane is closing")
        if self._unobserved:
            raise BaseExceptionGroup(
                "unobserved I/O lane failure", list(self._unobserved.values())
            )
        await self._capacity.acquire()
        if self._close_task is not None:
            self._capacity.release()
            raise RuntimeError("I/O lane is closing")
        if self._unobserved:
            self._capacity.release()
            raise BaseExceptionGroup(
                "unobserved I/O lane failure", list(self._unobserved.values())
            )
        task = asyncio.create_task(self._dispatch(operation, args, kwargs))
        self._operations.add(task)
        task.add_done_callback(self._settled)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._record_abandoned_failure(task)
            else:
                self._abandoned.add(task)
            raise

    async def _dispatch(self, operation, args, kwargs):
        lane = await asyncio.shield(self._start())

        async def invoke():
            result = operation(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        submitted: Future[Any] = asyncio.run_coroutine_threadsafe(invoke(), lane)
        return await asyncio.wrap_future(submitted)

    def _settled(self, task: asyncio.Task[Any]) -> None:
        self._operations.remove(task)
        self._capacity.release()
        if task in self._abandoned:
            self._abandoned.remove(task)
            self._record_abandoned_failure(task)
        elif not task.cancelled():
            task.exception()

    def _record_abandoned_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            self._unobserved[task] = asyncio.CancelledError()
        elif (error := task.exception()) is not None:
            self._unobserved[task] = error

    async def close(self) -> None:
        self._bind_owner()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._observe_close)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _observe_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _close(self) -> None:
        if self._operations:
            await asyncio.gather(*tuple(self._operations), return_exceptions=True)
        if self._ready is not None:
            try:
                lane = await asyncio.shield(self._ready)
                lane.call_soon_threadsafe(lane.stop)
            finally:
                assert self._exited is not None
                try:
                    await asyncio.shield(self._exited)
                finally:
                    await self._join_thread()
        if self._unobserved:
            raise BaseExceptionGroup(
                "unobserved I/O lane failure", list(self._unobserved.values())
            )

    async def _join_thread(self) -> None:
        owner = self._bind_owner()
        joined = owner.create_future()
        assert self._thread is not None

        def join():
            self._thread.join()
            owner.call_soon_threadsafe(joined.set_result, None)

        Thread(target=join, name=f"{self._name}-join").start()
        await joined
