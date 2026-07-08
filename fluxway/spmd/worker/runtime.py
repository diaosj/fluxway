"""Worker-local lifecycle runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
import inspect
import threading
from typing import Any

from fluxway.profiler.recorder import WORKER_OUTBOUND_TASK


class WorkerProcessCallRuntime:
    def __init__(
        self,
        *,
        max_workers: int,
        profiler_runtime: Any | None = None,
        thread_name_prefix: str = "fluxway-worker-call",
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"worker process call max_workers must be positive, got {max_workers}."
            )
        self._profiler_runtime = profiler_runtime
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._futures: set[Future[Any]] = set()
        self._futures_lock = threading.Lock()
        self._closed = False

    async def submit(
        self,
        fn: Callable[[], Any],
        *,
        callable_id: str,
        resource_id: int,
    ) -> Any:
        if self._closed:
            raise RuntimeError("WorkerProcessCallRuntime is closed.")
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        profiler_runtime = self._profiler_runtime
        start_future: asyncio.Future[None] | None = None
        if profiler_runtime is not None and profiler_runtime.enabled:
            start_future = loop.create_future()

        def run_in_worker_thread() -> Any:
            if start_future is not None:
                loop.call_soon_threadsafe(self._mark_thread_started, start_future)
            recorder = (
                profiler_runtime.recorder
                if profiler_runtime is not None and profiler_runtime.enabled
                else None
            )

            def invoke() -> Any:
                result = fn()
                if inspect.iscoroutine(result):
                    return asyncio.run(result)
                return result

            if recorder is None:
                return context.run(invoke)
            with recorder.worker_process_thread_run(
                callable_id=callable_id,
                resource_id=resource_id,
            ):
                return context.run(invoke)

        recorder = (
            profiler_runtime.recorder
            if profiler_runtime is not None and profiler_runtime.enabled
            else None
        )
        if recorder is None:
            future = self._executor.submit(run_in_worker_thread)
        else:
            with recorder.worker_process_submit(
                callable_id=callable_id,
                resource_id=resource_id,
                pending_call_count=self.pending_count() + 1,
            ):
                future = self._executor.submit(run_in_worker_thread)
        self._add_future(future)

        def on_done(done_future: Future[Any]) -> None:
            self._discard_future(done_future)
            if start_future is not None and not start_future.done():
                loop.call_soon_threadsafe(self._mark_thread_started, start_future)

        future.add_done_callback(on_done)

        try:
            if recorder is not None and start_future is not None:
                async with recorder.worker_process_queue_wait(
                    callable_id=callable_id,
                    resource_id=resource_id,
                    pending_call_count=self.pending_count(),
                ):
                    await start_future
                async with recorder.worker_process_future_wait(
                    callable_id=callable_id,
                    resource_id=resource_id,
                    pending_call_count=self.pending_count(),
                ):
                    return await asyncio.wrap_future(future)
            return await asyncio.wrap_future(future)
        finally:
            if future.done():
                self._discard_future(future)

    @staticmethod
    def _mark_thread_started(start_future: asyncio.Future[None]) -> None:
        if not start_future.done():
            start_future.set_result(None)

    def _add_future(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.add(future)

    def _discard_future(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _pending_futures(self) -> list[Future[Any]]:
        with self._futures_lock:
            return [future for future in self._futures if not future.done()]

    def pending_count(self) -> int:
        return len(self._pending_futures())

    async def drain(self) -> None:
        while True:
            futures = self._pending_futures()
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
            )

    async def cancel_pending(self) -> None:
        futures = self._pending_futures()
        for future in futures:
            future.cancel()
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )
        with self._futures_lock:
            self._futures.clear()

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


class LocalInputPrepareRuntime:
    def __init__(
        self,
        *,
        background_tasks: set[asyncio.Task],
        on_task_done: Any,
        profiler_runtime: Any | None = None,
        on_prepare_ready: Callable[[str, int], None] | None = None,
        max_workers: int = 1,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"input prepare max_workers must be positive, got {max_workers}."
            )
        self._background_tasks = background_tasks
        self._on_task_done = on_task_done
        self._profiler_runtime = profiler_runtime
        self._on_prepare_ready = on_prepare_ready
        self._tasks: dict[tuple[str, int], asyncio.Task[Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="fluxway-input-prepare",
        )
        self._thread_futures: set[Future[Any]] = set()
        self._thread_futures_lock = threading.Lock()

    def start(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> asyncio.Task[Any]:
        key = (op_name, step)
        task = self._tasks.get(key)
        if task is not None:
            return task
        awaitable = awaitable_factory()
        if self._profiler_runtime is not None and self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder

            async def traced_prepare() -> Any:
                async with recorder.prepare_run(op_name=op_name, step=step):
                    return await awaitable

            coro = traced_prepare()
            with recorder.prepare_register(
                op_name=op_name,
                step=step,
                pending_prepare_count=len(self._tasks) + 1,
            ):
                task = asyncio.create_task(coro)
        else:
            task = asyncio.create_task(awaitable)
        self._tasks[key] = task
        self._background_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._on_task_done(done_task)
            if (
                not done_task.cancelled()
                and done_task.exception() is None
                and self._on_prepare_ready is not None
            ):
                self._on_prepare_ready(op_name, step)

        task.add_done_callback(_done)
        return task

    async def run_assemble(
        self,
        op_name: str,
        step: int,
        fn: Callable[[], Any],
    ) -> Any:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        profiler_runtime = self._profiler_runtime

        def run_in_prepare_thread() -> Any:
            recorder = (
                profiler_runtime.recorder
                if profiler_runtime is not None and profiler_runtime.enabled
                else None
            )
            if recorder is None:
                return context.run(fn)
            with recorder.prepare_assemble_thread(op_name=op_name, step=step):
                return context.run(fn)

        future = self._executor.submit(run_in_prepare_thread)
        self._add_thread_future(future)
        future.add_done_callback(self._discard_thread_future)
        try:
            return await asyncio.wrap_future(future, loop=loop)
        finally:
            if future.done():
                self._discard_thread_future(future)

    def _add_thread_future(self, future: Future[Any]) -> None:
        with self._thread_futures_lock:
            self._thread_futures.add(future)

    def _discard_thread_future(self, future: Future[Any]) -> None:
        with self._thread_futures_lock:
            self._thread_futures.discard(future)

    def _pending_thread_futures(self) -> list[Future[Any]]:
        with self._thread_futures_lock:
            return [future for future in self._thread_futures if not future.done()]

    async def consume(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        key = (op_name, step)
        task = self.start(op_name, step, awaitable_factory)
        try:
            if self._profiler_runtime is not None and self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.prepare_consume(
                    op_name=op_name,
                    step=step,
                ):
                    return await task
            return await task
        finally:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    async def await_ready(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        task = self.start(op_name, step, awaitable_factory)
        return await task

    async def drain(self) -> None:
        while True:
            tasks = [task for task in self._tasks.values() if not task.done()]
            thread_futures = self._pending_thread_futures()
            if not tasks and not thread_futures:
                return
            if self._profiler_runtime is not None and self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.prepare_fence(
                    pending_prepare_count=len(tasks) + len(thread_futures),
                ):
                    await self._await_prepare_work(tasks, thread_futures)
            else:
                await self._await_prepare_work(tasks, thread_futures)

    async def _await_prepare_work(
        self,
        tasks: list[asyncio.Task[Any]],
        thread_futures: list[Future[Any]],
    ) -> None:
        awaitables: list[Awaitable[Any]] = list(tasks)
        awaitables.extend(asyncio.wrap_future(future) for future in thread_futures)
        if awaitables:
            await asyncio.gather(*awaitables)

    def is_pending(self, op_name: str, step: int) -> bool:
        task = self._tasks.get((op_name, step))
        return task is not None and not task.done()

    def is_ready(self, op_name: str, step: int) -> bool:
        task = self._tasks.get((op_name, step))
        return (
            task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        )

    def pending_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    async def cancel_pending(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        thread_futures = self._pending_thread_futures()
        for task in tasks:
            task.cancel()
        for future in thread_futures:
            future.cancel()
        if tasks or thread_futures:
            await asyncio.gather(
                *tasks,
                *(asyncio.wrap_future(future) for future in thread_futures),
                return_exceptions=True,
            )
        self._tasks.clear()
        with self._thread_futures_lock:
            self._thread_futures.clear()

    async def cancel_matching(
        self,
        predicate: Callable[[str, int], bool],
    ) -> None:
        selected = [
            (key, task)
            for key, task in self._tasks.items()
            if not task.done() and predicate(key[0], key[1])
        ]
        for _key, task in selected:
            task.cancel()
        if selected:
            await asyncio.gather(
                *(task for _key, task in selected),
                return_exceptions=True,
            )
        for key, task in selected:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class OutboundDeliveryRuntime:
    def __init__(
        self,
        *,
        max_in_flight: int,
        profiler_runtime: Any,
        background_tasks: set[asyncio.Task],
        on_task_done: Any,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError(
                f"data_plane_max_in_flight must be positive, got {max_in_flight}."
            )
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._profiler_runtime = profiler_runtime
        self._background_tasks = background_tasks
        self._on_task_done = on_task_done
        self._tasks: set[asyncio.Task] = set()

    async def submit(self, awaitable: Coroutine[Any, Any, Any]) -> asyncio.Task:
        original_coro = awaitable
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            task_name = getattr(
                getattr(original_coro, "cr_code", None),
                "co_name",
                type(original_coro).__name__,
            )

            async def traced_coro() -> Any:
                async with recorder.worker_scheduler_span(
                    spec=WORKER_OUTBOUND_TASK,
                    payload=(recorder.intern(task_name),),
                ):
                    return await self._run_delivery(original_coro)

            coro = traced_coro()
        else:
            coro = self._run_delivery(original_coro)
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.delivery_register_token(
                pending_delivery_count=len(self._tasks) + 1,
            ):
                task = asyncio.create_task(coro)
                self._background_tasks.add(task)
                self._tasks.add(task)
        else:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            self._tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._tasks.discard(done_task)
            if self._profiler_runtime.enabled and not done_task.cancelled():
                exc = done_task.exception()
                if exc is not None:
                    with self._profiler_runtime.recorder.delivery_poison(
                        error=type(exc).__name__,
                    ):
                        pass
            self._on_task_done(done_task)

        task.add_done_callback(_done)
        return task

    async def _run_delivery(self, awaitable: Coroutine[Any, Any, Any]) -> Any:
        acquired = False
        started = False
        try:
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.delivery_semaphore_wait(
                    pending_delivery_count=self.pending_count(),
                ):
                    await self._semaphore.acquire()
            else:
                await self._semaphore.acquire()
            acquired = True
            started = True
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.delivery_run(
                    pending_delivery_count=self.pending_count(),
                ):
                    return await awaitable
            return await awaitable
        finally:
            if acquired:
                self._semaphore.release()
            if not started:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()

    def pending_count(self) -> int:
        return sum(1 for task in self._tasks if not task.done())

    async def drain(self, fence_id: str) -> None:
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        while True:
            tasks = [task for task in self._tasks if not task.done()]
            if not tasks:
                return
            if recorder is None:
                await asyncio.gather(*tasks)
            else:
                async with recorder.delivery_fence(
                    fence_id=fence_id,
                    pending_delivery_count=len(tasks),
                ):
                    await asyncio.gather(*tasks)

    async def cancel_pending(self) -> None:
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
