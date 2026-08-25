"""Worker-side ownership and admission of logical execution steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Awaitable, Callable
from typing import Any


class AdmissionAction(StrEnum):
    DISPATCH = "dispatch"
    DROP = "drop"
    DEFER = "defer"
    HOLD_PREPARE = "hold_prepare"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    action: AdmissionAction
    reason: str


class StepAdmissionState:
    """Owns pending lifecycle membership and applies worker admission policy."""

    def __init__(
        self,
        *,
        completed: dict[str, int],
        scheduled: dict[str, int],
        queue_membership: set[tuple[str, int]],
        pending_blocked: dict[str, list[int]],
        pending_future: dict[str, list[int]],
        pending_async_lag: dict[str, list[int]],
        pending_phase_order: dict[str, list[int]],
        ready_pipeline: dict[str, list[int]],
        mailbox_ready: dict[str, set[int]],
        running: dict[str, set[int]],
        finished: dict[str, dict[int, Any]],
        is_periodic: Callable[[str], bool],
        is_runnable: Callable[[str], bool],
        hold_prepare_token: Callable[[str, int], bool],
        async_lag_blocked: Callable[[str, int], bool],
        phase_order_blocked: Callable[[str, int], bool],
        can_start: Callable[[str], bool],
    ) -> None:
        self.completed = completed
        self.scheduled = scheduled
        self.queue_membership = queue_membership
        self.pending_blocked = pending_blocked
        self.pending_future = pending_future
        self.pending_async_lag = pending_async_lag
        self.pending_phase_order = pending_phase_order
        self.ready_pipeline = ready_pipeline
        self.mailbox_ready = mailbox_ready
        self.running = running
        self.finished = finished
        self._is_periodic = is_periodic
        self._is_runnable = is_runnable
        self._hold_prepare_token = hold_prepare_token
        self._async_lag_blocked = async_lag_blocked
        self._phase_order_blocked = phase_order_blocked
        self._can_start = can_start

    @staticmethod
    def _append_once(values: list[int], step: int) -> None:
        if step not in values:
            values.append(step)

    def _defer(
        self, op_name: str, step: int, values: list[int], reason: str
    ) -> AdmissionDecision:
        self.queue_membership.discard((op_name, step))
        self._append_once(values, step)
        return AdmissionDecision(AdmissionAction.DEFER, reason)

    def evaluate(self, op_name: str, step: int) -> AdmissionDecision:
        if not self._is_runnable(op_name):
            return self._defer(op_name, step, self.pending_blocked[op_name], "blocked")

        if not self._is_periodic(op_name):
            completed = self.completed[op_name]
            if step < completed:
                self.queue_membership.discard((op_name, step))
                return AdmissionDecision(AdmissionAction.DROP, "stale")
            scheduled = self.scheduled[op_name]
            if step < scheduled:
                self.queue_membership.discard((op_name, step))
                return AdmissionDecision(AdmissionAction.DROP, "scheduled")
            if step > scheduled:
                return self._defer(
                    op_name, step, self.pending_future[op_name], "future"
                )

        if self._hold_prepare_token(op_name, step):
            return AdmissionDecision(AdmissionAction.HOLD_PREPARE, "prepare_pending")
        if self._async_lag_blocked(op_name, step):
            return self._defer(
                op_name, step, self.pending_async_lag[op_name], "async_lag"
            )
        if self._phase_order_blocked(op_name, step):
            return self._defer(
                op_name, step, self.pending_phase_order[op_name], "phase_order"
            )
        if not self._can_start(op_name):
            return self._defer(
                op_name, step, self.ready_pipeline[op_name], "pipeline_capacity"
            )
        self.queue_membership.discard((op_name, step))
        return AdmissionDecision(AdmissionAction.DISPATCH, "ready")

    def known_steps(self, op_name: str) -> set[int]:
        known = set(self.pending_blocked.get(op_name, ()))
        known.update(self.pending_async_lag.get(op_name, ()))
        known.update(self.pending_phase_order.get(op_name, ()))
        known.update(self.mailbox_ready.get(op_name, ()))
        known.update(self.running.get(op_name, ()))
        known.update(self.finished.get(op_name, ()))
        known.update(self.ready_pipeline.get(op_name, ()))
        known.update(self.pending_future.get(op_name, ()))
        known.update(
            step for queued_op, step in self.queue_membership if queued_op == op_name
        )
        completed = self.completed.get(op_name, 0)
        scheduled = self.scheduled.get(op_name, completed)
        known.update(range(completed, scheduled))
        return known

    def move_future(self, op_name: str) -> None:
        if self._is_periodic(op_name):
            return
        pending = self.pending_future[op_name]
        while self.scheduled[op_name] in pending:
            step = self.scheduled[op_name]
            pending.remove(step)
            self._append_once(self.ready_pipeline[op_name], step)

    async def start_pending(
        self,
        op_name: str,
        *,
        running: Callable[[], bool],
        source_exhausted: Callable[[str], bool],
        start_plan: Callable[[str, int], None],
    ) -> None:
        if not running() or source_exhausted(op_name):
            return
        self.move_future(op_name)
        pending = self.ready_pipeline[op_name]
        while running() and pending and self._can_start(op_name):
            step = pending.pop(0)
            if self._async_lag_blocked(op_name, step):
                self._append_once(self.pending_async_lag[op_name], step)
                continue
            if self._phase_order_blocked(op_name, step):
                self._append_once(self.pending_phase_order[op_name], step)
                continue
            start_plan(op_name, step)
            self.move_future(op_name)

    async def _promote_pending(
        self,
        pending_by_op: dict[str, list[int]],
        blocked: Callable[[str, int], bool],
        enqueue: Callable[[str, int], Awaitable[None]],
    ) -> None:
        for op_name, pending in pending_by_op.items():
            if not pending:
                continue
            pending_by_op[op_name] = []
            for step in sorted(set(pending)):
                if blocked(op_name, step):
                    self._append_once(pending_by_op[op_name], step)
                else:
                    await enqueue(op_name, step)

    async def release_async_lag(
        self, enqueue: Callable[[str, int], Awaitable[None]]
    ) -> None:
        await self._promote_pending(
            self.pending_async_lag,
            self._async_lag_blocked,
            enqueue,
        )

    async def release_phase_order(
        self, enqueue: Callable[[str, int], Awaitable[None]]
    ) -> None:
        await self._promote_pending(
            self.pending_phase_order,
            self._phase_order_blocked,
            enqueue,
        )

    def discard_at_or_after(self, op_name: str, terminal_step: int) -> None:
        for pending in (
            self.pending_blocked[op_name],
            self.pending_future[op_name],
            self.ready_pipeline[op_name],
            self.pending_async_lag[op_name],
            self.pending_phase_order[op_name],
        ):
            pending[:] = [step for step in pending if step < terminal_step]
        self.mailbox_ready[op_name].difference_update(
            step for step in self.mailbox_ready[op_name] if step >= terminal_step
        )
        self.queue_membership.difference_update(
            (op_name, step)
            for queued_op, step in self.queue_membership
            if queued_op == op_name and step >= terminal_step
        )
