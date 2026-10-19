from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum


class WorkerPhase(StrEnum):
    INITIALIZING = "initializing"
    READY = "ready"
    RESTORING = "restoring"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class WorkerExecutionLifecycle:
    def __init__(self) -> None:
        self._phase = WorkerPhase.INITIALIZING
        self._activation = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def phase(self) -> WorkerPhase:
        return self._phase

    @property
    def active(self) -> bool:
        return self._phase is WorkerPhase.ACTIVE

    async def wait_for_activation(self) -> None:
        await self._activation.wait()

    def topology_ready(self) -> None:
        if self._phase is WorkerPhase.INITIALIZING:
            self._phase = WorkerPhase.READY
        elif self._phase is not WorkerPhase.READY:
            raise RuntimeError(
                f"topology registration requires preparation, got {self._phase}"
            )

    def require_restore(self) -> None:
        if self._phase not in (
            WorkerPhase.INITIALIZING,
            WorkerPhase.READY,
            WorkerPhase.RESTORING,
        ):
            raise RuntimeError(
                "checkpoint restore requires a fresh Worker before execution"
            )

    def begin_restore(self) -> None:
        if self._phase is not WorkerPhase.READY:
            raise RuntimeError(
                f"restore requires topology readiness, got {self._phase}"
            )
        self._phase = WorkerPhase.RESTORING

    def finish_restore(self) -> None:
        if self._phase is not WorkerPhase.RESTORING:
            raise RuntimeError(f"restore cannot complete in {self._phase}")
        self._phase = WorkerPhase.READY

    def activate(self) -> bool:
        if self.active:
            return False
        if self._phase is not WorkerPhase.READY:
            raise RuntimeError(f"activation requires READY, got {self._phase}")
        self._phase = WorkerPhase.ACTIVE
        self._activation.set()
        return True

    def fail(self) -> None:
        self._phase = WorkerPhase.FAILED

    async def close(self, release: Callable[[], Awaitable[None]]) -> None:
        if self._close_task is None:
            if self._phase not in (WorkerPhase.FAILED, WorkerPhase.CLOSED):
                self._phase = WorkerPhase.CLOSING
            self._close_task = asyncio.create_task(self._close(release))
            self._close_task.add_done_callback(self._close_finished)
        await asyncio.shield(self._close_task)

    async def _close(self, release: Callable[[], Awaitable[None]]) -> None:
        try:
            await release()
        except BaseException:
            self.fail()
            raise
        self._phase = WorkerPhase.CLOSED

    def _close_finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
