"""Coalesced source pipeline maintenance."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class SourcePipelineMaintenance:
    """Maintains source pipeline depth without owning execution semantics."""

    def __init__(
        self,
        *,
        tasks: dict[str, asyncio.Task],
        spawn: Callable[[Awaitable[None]], asyncio.Task],
        is_source: Callable[[str], bool],
        running: Callable[[], bool],
        exhausted: Callable[[str], bool],
        depth: Callable[[str], int],
        completed: Callable[[str], int],
        max_steps: Callable[[], int | None],
        drain_blocks: Callable[[str, int], bool],
        known_steps: Callable[[str], set[int]],
        enqueue: Callable[[str, int], Awaitable[None]],
    ) -> None:
        self.tasks = tasks
        self._spawn = spawn
        self._is_source = is_source
        self._running = running
        self._exhausted = exhausted
        self._depth = depth
        self._completed = completed
        self._max_steps = max_steps
        self._drain_blocks = drain_blocks
        self._known_steps = known_steps
        self._enqueue = enqueue

    def request(self, op_name: str) -> asyncio.Task:
        existing = self.tasks.get(op_name)
        if existing is not None and not existing.done():
            return existing
        task = self._spawn(self.fill(op_name))
        self.tasks[op_name] = task

        def clear_owner(done: asyncio.Task) -> None:
            if self.tasks.get(op_name) is done:
                self.tasks.pop(op_name, None)

        task.add_done_callback(clear_owner)
        return task

    async def fill(self, op_name: str) -> None:
        if not self._is_source(op_name):
            raise ValueError(
                f"source pipeline maintenance requires a source op, got {op_name!r}."
            )
        if not self._running() or self._exhausted(op_name):
            return
        pipeline_depth = self._depth(op_name)
        max_steps = self._max_steps()
        while self._running():
            completed = self._completed(op_name)
            known = {
                step
                for step in self._known_steps(op_name)
                if step >= completed and (max_steps is None or step < max_steps)
            }
            if len(known) >= pipeline_depth:
                return
            candidate = completed
            while candidate in known:
                candidate += 1
            if max_steps is not None and candidate >= max_steps:
                return
            if self._drain_blocks(op_name, candidate):
                return
            await self._enqueue(op_name, candidate)
