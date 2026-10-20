from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any

from fluxway.common.ingestion import CompiledSegment, SourceWorkLease


class SourceWorkRuntime:
    def __init__(
        self,
        *,
        epoch: int,
        segments_by_step: Mapping[int, tuple[CompiledSegment, ...]],
        batch_steps: int,
        acquire: Callable[
            [int, tuple[CompiledSegment, ...]], Awaitable[tuple[SourceWorkLease, ...]]
        ],
        spawn: Callable[[Coroutine[Any, Any, None]], asyncio.Task],
        completed: Callable[[], int],
    ) -> None:
        if type(batch_steps) is not int or batch_steps <= 0:
            raise ValueError("source batch step count must be positive")
        self._epoch = epoch
        self._segments = dict(segments_by_step)
        self._steps = tuple(sorted(self._segments))
        self._ordinal = {step: index for index, step in enumerate(self._steps)}
        self._batch_steps = batch_steps
        self._acquire = acquire
        self._spawn = spawn
        self._completed = completed
        self._next_by_chunk: dict[int, int] = {}
        self._pending: dict[int, asyncio.Task] = {}
        self._cache: dict[int, tuple[SourceWorkLease, ...]] = {}

    async def take(self, step: int) -> tuple[SourceWorkLease, ...]:
        index = self._ordinal[step]
        chunk = index // self._batch_steps
        end = min((chunk + 1) * self._batch_steps, len(self._steps))
        while True:
            if step in self._cache:
                return self._cache.pop(step)
            start = max(
                self._next_by_chunk.get(chunk, chunk * self._batch_steps),
                bisect_left(self._steps, self._completed()),
            )
            if index < start:
                raise RuntimeError(
                    "source step is already consumed or below restored progress"
                )
            task = self._pending.get(chunk)
            if task is None:
                task = self._spawn(self._fetch(chunk, start, end))
                self._pending[chunk] = task
            try:
                await asyncio.shield(task)
            finally:
                if task.done() and self._pending.get(chunk) is task:
                    del self._pending[chunk]

    async def _fetch(self, chunk: int, start: int, end: int) -> None:
        steps = self._steps[start:end]
        requested = tuple(segment for step in steps for segment in self._segments[step])
        leases = await self._acquire(self._epoch, requested)
        by_step: dict[int, list[SourceWorkLease]] = {}
        ids: set[str] = set()
        for lease in leases:
            if lease.item.key.epoch != self._epoch or lease.lease_id in ids:
                raise ValueError(
                    "source lease reply has conflicting epoch or duplicate lease"
                )
            ids.add(lease.lease_id)
            by_step.setdefault(lease.item.key.step, []).append(lease)
        accepted = tuple(sorted(by_step))
        if not accepted or accepted != steps[: len(accepted)]:
            raise ValueError("source lease reply must contain a nonempty step prefix")
        for step, values in by_step.items():
            expected = {
                (s.assignment_id, s.logical_start, s.logical_end)
                for s in self._segments[step]
            }
            actual = [
                (v.item.key.assignment_id, v.item.logical_start, v.item.logical_end)
                for v in values
            ]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("source lease reply does not cover the requested step")
        self._cache.update((step, tuple(values)) for step, values in by_step.items())
        self._next_by_chunk[chunk] = start + len(accepted)
