"""Supervise accepted operations so they remain owned when callers stop waiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

_Result = TypeVar("_Result")
_active_supervisors: ContextVar[tuple[AsyncOperationSupervisor, ...]] = ContextVar(
    "active_operation_supervisors", default=()
)


class AsyncOperationSupervisor:
    """Supervises in-flight asynchronous operations with custody and drain guarantees.

    Rationale:
    Standard asyncio.shield leaves abandoned tasks unmonitored on caller timeout,
    leaking unobserved exceptions and risking shutdown corruption. This supervisor
    holds custody of accepted tasks, captures late failures, and guarantees safe
    drainage before resource disposal.
    """

    def __init__(self) -> None:
        self._accepting = True
        self._settlements: dict[asyncio.Task[Any], asyncio.Future[None]] = {}
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._unobserved: dict[asyncio.Task[Any], BaseException] = {}

    @property
    def accepting(self) -> bool:
        return self._accepting

    def ensure_not_active(self) -> None:
        """Prevent self-await deadlocks during drain.

        If a running task or its child task calls drain(), drain() would await
        the calling task itself, causing a permanent circular wait.
        """
        if self in _active_supervisors.get():
            raise RuntimeError("a supervised operation cannot close its supervisor")

    def raise_unobserved(self) -> None:
        """Raise accumulated failures from abandoned tasks."""
        if self._unobserved:
            raise BaseExceptionGroup(
                "unobserved supervised operation failure",
                list(self._unobserved.values()),
            )

    def ensure_admissible(
        self, closing_message: str = "operation supervisor is closing"
    ) -> None:
        """Check acceptance and propagate any accumulated unobserved failures."""
        if not self._accepting:
            raise RuntimeError(closing_message)
        self.raise_unobserved()

    async def run_with_capacity(
        self,
        capacity: asyncio.Semaphore,
        operation: Callable[[], Awaitable[_Result]],
        *,
        closing_message: str = "operation supervisor is closing",
    ) -> _Result:
        """Acquire bounded capacity and supervise the operation, releasing capacity only when custody settles."""
        self.ensure_admissible(closing_message)
        await capacity.acquire()
        if not self._accepting:
            capacity.release()
            raise RuntimeError(closing_message)
        try:
            self.raise_unobserved()
        except BaseException:
            capacity.release()
            raise
        return await self.run(
            operation,
            on_settled=capacity.release,
            closing_message=closing_message,
        )

    async def run(
        self,
        operation: Callable[[], Awaitable[_Result]],
        *,
        on_settled: Callable[[], None] | None = None,
        closing_message: str = "operation supervisor is closing",
    ) -> _Result:
        """Admit and supervise an operation to completion.

        Uses asyncio.wait to decouple caller cancellation from the underlying task:
        - If the caller cancels, the underlying task continues in background custody.
        - If the underlying task raises CancelledError, it returns via task.result()
          directly to the caller without false-positive unobserved error recording.

        on_settled runs on the owner loop after result custody is resolved.
        """
        self.ensure_admissible(closing_message)
        task = asyncio.create_task(self._invoke(operation))
        settled = asyncio.get_running_loop().create_future()
        self._settlements[task] = settled
        if on_settled is not None:
            settled.add_done_callback(lambda _: on_settled())
        task.add_done_callback(self._completed)
        try:
            # Wait without shielding: caller cancellation cancels the wait, not the task.
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            if task.done():
                self._record_failure(task)
                self._settle(task)
            else:
                self._abandoned.add(task)
            raise
        try:
            return task.result()
        finally:
            self._settle(task)

    async def _invoke(self, operation: Callable[[], Awaitable[_Result]]) -> _Result:
        # Track active supervisor in ContextVar to detect reentrant drain deadlocks.
        token = _active_supervisors.set((*_active_supervisors.get(), self))
        try:
            return await operation()
        finally:
            _active_supervisors.reset(token)

    def stop_accepting(self) -> None:
        """Stop admitting new tasks, ensuring caller is not a managed task."""
        self.ensure_not_active()
        self._accepting = False

    async def drain(self) -> tuple[BaseException, ...]:
        """Join execution and caller observation before reporting abandoned failures."""
        self.stop_accepting()
        if self._settlements:
            await asyncio.shield(asyncio.gather(*tuple(self._settlements.values())))
        return tuple(self._unobserved.values())

    def _completed(self, task: asyncio.Task[Any]) -> None:
        if task in self._abandoned:
            self._abandoned.remove(task)
            self._record_failure(task)
            self._settle(task)
        elif not task.cancelled():
            task.exception()

    def _settle(self, task: asyncio.Task[Any]) -> None:
        self._settlements.pop(task).set_result(None)

    def _record_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            self._unobserved[task] = asyncio.CancelledError()
        elif (error := task.exception()) is not None:
            self._unobserved[task] = error
