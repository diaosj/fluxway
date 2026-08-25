"""Ordered coherence payload retirement for asynchronous frame construction."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
import logging
from enum import StrEnum
from typing import Any, TypeVar

from fluxway.common.coherence import CoherenceStateTransitionError, CoherenceInstance

PayloadT = TypeVar("PayloadT")
_MISSING = object()
logger = logging.getLogger(__name__)
RetirePayload = Callable[[tuple[CoherenceInstance, ...], PayloadT], Awaitable[Any]]
AdmitInstance = Callable[[CoherenceInstance], Awaitable[None]]


@dataclass(slots=True)
class _Intent:
    instance: CoherenceInstance
    payloads: deque[_Ticket[Any]] = field(default_factory=deque)
    active_payloads: int = 0
    admitted: bool = False
    sealed: bool = False
    admission_task: asyncio.Task[None] | None = None

    @property
    def is_completed(self) -> bool:
        return self.sealed and self.active_payloads == 0

    def next_ticket(self) -> _Ticket[Any] | None:
        return next(iter(self.payloads), None)

    def owns_next_ticket(self, ticket: _Ticket[Any]) -> bool:
        return self.next_ticket() is ticket

    def remove_next_ticket(self, ticket: _Ticket[Any]) -> None:
        if not self.owns_next_ticket(ticket):
            raise CoherenceStateTransitionError(
                "coherence intent next ticket changed during retirement."
            )
        self.payloads.popleft()


class _TicketState(StrEnum):
    READY = "ready"
    ADMITTING = "admitting"
    DELIVERING = "delivering"
    RETIRED = "retired"
    ABORTED = "aborted"


@dataclass(slots=True)
class _Ticket[PayloadT]:
    instances: tuple[CoherenceInstance, ...]
    payload: PayloadT
    future: asyncio.Future[Any] | None
    intents: tuple[_Intent, ...] = field(default_factory=tuple)
    state: _TicketState = _TicketState.READY
    inline_claimed: bool = False
    direct_executing: bool = False
    result: Any = _MISSING
    error: BaseException | None = None


class _SubjectLane:
    __slots__ = ("_intents",)

    def __init__(self) -> None:
        self._intents: deque[_Intent] = deque()

    def append(self, intent: _Intent) -> None:
        self._intents.append(intent)

    def __bool__(self) -> bool:
        return bool(self._intents)

    def active_intents(self) -> tuple[_Intent, ...]:
        return tuple(self._intents)

    def next_intent_to_admit(self) -> _Intent | None:
        return next(iter(self._intents), None)

    def owns_next_intent(self, intent: _Intent) -> bool:
        return self.next_intent_to_admit() is intent

    def next_ticket_to_retire(self) -> _Ticket[Any] | None:
        intent = self.next_intent_to_admit()
        return intent.next_ticket() if intent is not None else None

    def discard_completed_prefix(self) -> tuple[_Intent, ...]:
        discarded: list[_Intent] = []
        while (intent := self.next_intent_to_admit()) is not None:
            if not intent.is_completed:
                break
            discarded.append(intent)
            self._intents.popleft()
        return tuple(discarded)

    def retire_next_intent(self, intent: _Intent) -> None:
        if not self.owns_next_intent(intent):
            raise CoherenceStateTransitionError(
                "coherence ROB next intent changed during retirement."
            )
        if not intent.is_completed:
            raise CoherenceStateTransitionError(
                "coherence ROB head intent is not complete."
            )
        self._intents.popleft()


class CoherenceSequencer[PayloadT]:
    """Event-loop-owned per-subject reorder buffer."""

    def __init__(
        self,
        retire_payload: RetirePayload[PayloadT],
        admit_instance: AdmitInstance,
    ) -> None:
        self._retire_payload = retire_payload
        self._admit_instance = admit_instance
        self._lanes: dict[object, _SubjectLane] = {}
        self._intent_index: dict[CoherenceInstance, _Intent] = {}
        self._last_issued_instance: dict[object, CoherenceInstance] = {}
        self._last_retired_instance: dict[object, CoherenceInstance] = {}
        self._tickets: dict[int, _Ticket[Any]] = {}
        self._drain_task: asyncio.Task[None] | None = None
        self._retire_tasks: dict[int, asyncio.Task[None]] = {}
        self._fenced_error: BaseException | None = None

    def issue_intent(self, instance: CoherenceInstance) -> None:
        self._ensure_open()
        lane = self._lanes.setdefault(instance.subject, _SubjectLane())
        last_instance = self._last_issued_instance.get(instance.subject)
        if last_instance is not None and instance.epoch <= last_instance.epoch:
            raise CoherenceStateTransitionError(
                "coherence intents must be issued in strictly increasing order: "
                f"subject={instance.subject} epoch={instance.epoch} "
                f"last={last_instance.epoch}."
            )
        intent = _Intent(instance)
        lane.append(intent)
        self._intent_index[instance] = intent
        self._last_issued_instance[instance.subject] = instance

    def commit_payload(
        self,
        instances: Iterable[CoherenceInstance],
        payload: PayloadT,
    ) -> Coroutine[Any, Any, Any]:
        self._ensure_open()
        ordered_values = tuple(instances)
        if not ordered_values:
            raise ValueError("coherence payload must reference at least one instance.")
        seen_instances: set[CoherenceInstance] = set()
        for instance in ordered_values:
            if instance in seen_instances:
                raise CoherenceStateTransitionError(
                    f"coherence payload references an instance more than once: {instance}."
                )
            seen_instances.add(instance)
        ordered = ordered_values
        subjects: dict[object, CoherenceInstance] = {}
        for instance in ordered:
            previous = subjects.setdefault(instance.subject, instance)
            if previous != instance:
                raise CoherenceStateTransitionError(
                    "one coherence payload cannot span multiple epochs of one subject: "
                    f"subject={instance.subject} epochs={previous.epoch},{instance.epoch}."
                )
        ticket = _Ticket(ordered, payload, None)
        intents: list[_Intent] = []
        try:
            for instance in ordered:
                intent = self._intent_index.get(instance)
                if intent is None:
                    state = (
                        "retired"
                        if self._last_retired_instance.get(instance.subject) == instance
                        else "never-issued"
                    )
                    active = tuple(self._intent_index)
                    last_issued = self._last_issued_instance.get(instance.subject)
                    raise CoherenceStateTransitionError(
                        "coherence intent is not declared: "
                        f"instance={instance} state={state}; "
                        f"last_issued={last_issued!r} active_intents={active!r}."
                    )
                if intent.sealed:
                    raise CoherenceStateTransitionError(
                        f"coherence intent is already sealed: {instance}."
                    )
                intents.append(intent)
            ticket.intents = tuple(intents)
            for intent in intents:
                intent.payloads.append(ticket)
                intent.active_payloads += 1
            self._tickets[id(ticket)] = ticket
            if self._claim_inline_retirement(ticket):
                ticket.inline_claimed = True
        except BaseException:
            self._abort_ticket(ticket)
            raise
        return self._await_ticket(ticket)

    async def _await_ticket(self, ticket: _Ticket[Any]) -> Any:
        if (
            self._fenced_error is not None
            and ticket.future is None
            and ticket.state is not _TicketState.RETIRED
        ):
            raise self._fenced_error
        if ticket.state is _TicketState.RETIRED:
            return ticket.result if ticket.result is not _MISSING else None
        if ticket.state is _TicketState.ABORTED:
            if ticket.error is not None:
                raise ticket.error
            if self._fenced_error is not None:
                raise self._fenced_error
            raise RuntimeError("coherence payload aborted")

        if ticket.future is None and not ticket.inline_claimed:
            if self._claim_inline_retirement(ticket):
                ticket.inline_claimed = True
            else:
                ticket.future = asyncio.get_running_loop().create_future()

        if ticket.inline_claimed:
            ticket.inline_claimed = False
            ticket.direct_executing = True
            current_task = asyncio.current_task()
            if current_task is None:
                raise RuntimeError("coherence retirement requires an active task.")
            self._retire_tasks[id(ticket)] = current_task
            try:
                result = await self._admit_and_retire(ticket, direct=True)
                self._schedule_drain()
                return result
            except asyncio.CancelledError:
                self._abort_ticket(ticket, asyncio.CancelledError())
                if ticket.state is not _TicketState.DELIVERING:
                    self._schedule_drain()
                raise
            except BaseException as error:
                self._fence_after_failure(error)
                raise
            finally:
                ticket.direct_executing = False
                self._retire_tasks.pop(id(ticket), None)

        try:
            future = ticket.future
            if future is None:
                raise RuntimeError("blocked coherence ticket has no completion future.")
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._abort_ticket(ticket, asyncio.CancelledError())
            if ticket.state is not _TicketState.DELIVERING:
                self._schedule_drain()
            raise

    def _is_ticket_ready(self, ticket: _Ticket[Any]) -> bool:
        if ticket.state is not _TicketState.READY:
            return False
        for intent in ticket.intents:
            lane = self._lanes.get(intent.instance.subject)
            if (
                lane is None
                or not lane.owns_next_intent(intent)
                or not intent.sealed
                or intent.next_ticket() is not ticket
            ):
                return False
        return True

    def _claim_inline_retirement(self, ticket: _Ticket[Any]) -> bool:
        if not self._is_ticket_ready(ticket):
            return False
        ticket.state = _TicketState.ADMITTING
        return True

    def close_intent(self, instance: CoherenceInstance) -> None:
        self._ensure_open()
        intent = self._intent_index.get(instance)
        if intent is None:
            raise CoherenceStateTransitionError(
                f"coherence intent is not declared: {instance}."
            )
        if intent.sealed:
            raise CoherenceStateTransitionError(
                f"coherence intent is already sealed: {instance}."
            )
        intent.sealed = True
        self._schedule_drain()

    def abort_intent(self, instance: CoherenceInstance, error: BaseException) -> None:
        if self._fenced_error is not None:
            return
        if instance not in self._intent_index:
            raise CoherenceStateTransitionError(
                f"coherence intent is not declared: {instance}."
            )
        self.fence(error)

    def fence(self, error: BaseException | None = None) -> None:
        if self._fenced_error is not None:
            return
        self._fenced_error = error or RuntimeError("coherence sequencer fenced")
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
        current_task = asyncio.current_task()
        for task in tuple(self._retire_tasks.values()):
            if task is not current_task and not task.done():
                task.cancel()
        for lane in self._lanes.values():
            for intent in lane.active_intents():
                if (
                    intent.admission_task is not None
                    and not intent.admission_task.done()
                ):
                    intent.admission_task.cancel()
        self._abort_all(self._fenced_error)
        self._lanes.clear()
        self._intent_index.clear()

    def _ensure_open(self) -> None:
        if self._fenced_error is not None:
            raise self._fenced_error

    def _schedule_drain(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._lanes and self._fenced_error is None:
            self._discard_completed_intents()
            if not self._lanes:
                return
            ready_tickets = self._next_ready_tickets()
            if not ready_tickets:
                if self._all_active_intents_sealed():
                    cycle = self._dependency_cycle()
                    if cycle:
                        error = CoherenceStateTransitionError(
                            "coherence payload dependency cycle detected: "
                            f"tickets={cycle}."
                        )
                        self._fence_after_failure(error)
                    elif not any(
                        not task.done() for task in self._retire_tasks.values()
                    ):
                        error = CoherenceStateTransitionError(
                            "coherence reorder buffer is sealed but has no ready ticket: "
                            f"pending={self._pending_snapshot()}."
                        )
                        self._fence_after_failure(error)
                return
            for ticket in ready_tickets:
                self._start_retirement(ticket)

    def _start_retirement(self, ticket: _Ticket[Any]) -> None:
        if ticket.state is not _TicketState.READY or id(ticket) not in self._tickets:
            return
        ticket.state = _TicketState.ADMITTING
        task = asyncio.create_task(self._admit_and_retire(ticket))
        self._retire_tasks[id(ticket)] = task

        def done(done_task: asyncio.Task[None]) -> None:
            self._retire_tasks.pop(id(ticket), None)
            if done_task.cancelled():
                if ticket.state is not _TicketState.ABORTED:
                    self._abort_all(asyncio.CancelledError())
                return
            error = done_task.exception()
            if error is not None:
                self._fence_after_failure(error)
                return
            self._schedule_drain()

        task.add_done_callback(done)

    async def _admit_and_retire(
        self, ticket: _Ticket[Any], *, direct: bool = False
    ) -> Any:
        admission_tasks: list[asyncio.Future[None]] = []
        for intent in ticket.intents:
            if intent.admitted:
                continue
            if direct and intent.admission_task is None:
                await self._admit_intent(intent)
                continue
            if intent.admission_task is None:
                intent.admission_task = asyncio.create_task(self._admit_intent(intent))
            admission_tasks.append(asyncio.shield(intent.admission_task))
        if admission_tasks:
            await asyncio.gather(*admission_tasks)
        return await self._retire_ticket(ticket)

    async def _admit_intent(self, intent: _Intent) -> None:
        await self._admit_instance(intent.instance)
        intent.admitted = True

    async def _retire_ticket(self, ticket: _Ticket[Any]) -> Any:
        ticket.state = _TicketState.DELIVERING
        try:
            result = await self._retire_payload(ticket.instances, ticket.payload)
        except BaseException as error:
            try:
                self._finish_retired_ticket(ticket)
            except BaseException as cleanup_error:
                self._fence_after_failure(cleanup_error)
                raise cleanup_error from error
            future = ticket.future
            if future is not None and not future.cancelled() and not future.done():
                future.set_exception(error)
            raise
        try:
            self._finish_retired_ticket(ticket, result)
        except BaseException as cleanup_error:
            self._fence_after_failure(cleanup_error)
            raise
        return result

    def _finish_retired_ticket(
        self, ticket: _Ticket[Any], result: Any = _MISSING
    ) -> None:
        if ticket.state is _TicketState.RETIRED:
            return
        if ticket.state is not _TicketState.ABORTED:
            ticket.state = _TicketState.RETIRED
        if result is not _MISSING:
            ticket.result = result
        for intent in ticket.intents:
            lane = self._lanes.get(intent.instance.subject)
            if lane is None or not lane.owns_next_intent(intent):
                if self._fenced_error is not None:
                    return
                raise CoherenceStateTransitionError(
                    "coherence ROB current intent changed during retirement."
                )
            if not intent.owns_next_ticket(ticket):
                if self._fenced_error is not None:
                    return
                raise CoherenceStateTransitionError(
                    "coherence ROB ticket changed during retirement: "
                    f"ticket_id={id(ticket)} "
                    f"instance={intent.instance!r} "
                    f"expected_ticket_id={id(ticket)} "
                    f"actual_head_ticket_id="
                    f"{id(intent.next_ticket()) if intent.next_ticket() is not None else None}."
                )
            intent.remove_next_ticket(ticket)
            intent.active_payloads -= 1
            if intent.active_payloads < 0:
                raise CoherenceStateTransitionError(
                    f"coherence payload count underflow for {intent.instance}."
                )
            if intent.is_completed:
                self._intent_index.pop(intent.instance, None)
                self._last_retired_instance[intent.instance.subject] = intent.instance
                lane.retire_next_intent(intent)
                if not lane:
                    del self._lanes[intent.instance.subject]
        self._tickets.pop(id(ticket), None)
        future = ticket.future
        if result is not _MISSING and future is not None and not future.done():
            future.set_result(result)

    def _discard_completed_intents(self) -> None:
        for subject, lane in tuple(self._lanes.items()):
            for intent in lane.discard_completed_prefix():
                self._intent_index.pop(intent.instance, None)
            if not lane:
                del self._lanes[subject]

    def _next_ready_tickets(self) -> tuple[_Ticket[PayloadT], ...]:
        candidates: dict[int, _Ticket[PayloadT]] = {}
        for lane in self._lanes.values():
            ticket = lane.next_ticket_to_retire()
            if ticket is not None and self._is_ticket_ready(ticket):
                candidates[id(ticket)] = ticket
        return tuple(candidates.values())

    def _dependency_cycle(self) -> tuple[int, ...]:
        active_tickets = {
            id(ticket): ticket
            for lane in self._lanes.values()
            for intent in lane.active_intents()
            for ticket in intent.payloads
            if ticket.state not in (_TicketState.ABORTED, _TicketState.RETIRED)
        }
        if not active_tickets:
            return ()

        edges: dict[int, set[int]] = {ticket_id: set() for ticket_id in active_tickets}
        for ticket_id, ticket in active_tickets.items():
            for instance in ticket.instances:
                lane = self._lanes.get(instance.subject)
                if lane is None:
                    continue
                current_intent = lane.next_intent_to_admit()
                if current_intent is None:
                    continue
                intent = self._intent_index[instance]
                if current_intent is intent:
                    continue
                for blocker in current_intent.payloads:
                    blocker_id = id(blocker)
                    if (
                        blocker_id != ticket_id
                        and blocker.state
                        not in (_TicketState.ABORTED, _TicketState.RETIRED)
                        and blocker_id in self._tickets
                    ):
                        edges[ticket_id].add(blocker_id)

        visiting: set[int] = set()
        visited: set[int] = set()
        path: list[int] = []

        def visit(ticket_id: int) -> tuple[int, ...] | None:
            if ticket_id in visiting:
                start = path.index(ticket_id)
                return tuple(path[start:] + [ticket_id])
            if ticket_id in visited:
                return None
            visiting.add(ticket_id)
            path.append(ticket_id)
            for dependency in edges[ticket_id]:
                cycle = visit(dependency)
                if cycle:
                    return cycle
            path.pop()
            visiting.remove(ticket_id)
            visited.add(ticket_id)
            return None

        for ticket_id in edges:
            cycle = visit(ticket_id)
            if cycle:
                return cycle
        return ()

    def _all_active_intents_sealed(self) -> bool:
        return all(
            intent.sealed
            for lane in self._lanes.values()
            for intent in lane.active_intents()
        )

    def _pending_snapshot(self) -> tuple[dict[str, Any], ...]:
        pending: list[dict[str, Any]] = []
        for subject, lane in self._lanes.items():
            for intent in lane.active_intents():
                ticket = intent.next_ticket()
                pending.append(
                    {
                        "subject": repr(subject),
                        "instance": repr(intent.instance),
                        "sealed": intent.sealed,
                        "active_payloads": intent.active_payloads,
                        "head_ticket_id": id(ticket) if ticket is not None else None,
                        "head_ticket_state": ticket.state
                        if ticket is not None
                        else None,
                    }
                )
        return tuple(pending)

    def _abort_ticket(
        self,
        ticket: _Ticket[Any],
        error: BaseException | None = None,
    ) -> None:
        if ticket.state is _TicketState.RETIRED:
            if ticket.future is not None and not ticket.future.done():
                ticket.future.set_exception(
                    error or RuntimeError("retired coherence payload has no result")
                )
            return
        previous_state = ticket.state
        retire_task = self._retire_tasks.get(id(ticket))
        ticket.state = _TicketState.ABORTED
        abort_error = error or RuntimeError("coherence payload aborted")
        ticket.error = abort_error
        if previous_state is _TicketState.DELIVERING:
            future = ticket.future
            if future is not None and not future.done():
                if (
                    ticket.inline_claimed is False
                    and retire_task is asyncio.current_task()
                ):
                    future.cancel()
                elif isinstance(abort_error, asyncio.CancelledError):
                    future.cancel()
                else:
                    future.set_exception(abort_error)
            self._tickets.pop(id(ticket), None)
            return
        if (
            retire_task is not None
            and retire_task is not asyncio.current_task()
            and not retire_task.done()
        ):
            retire_task.cancel()
        for intent in ticket.intents:
            if ticket in intent.payloads:
                intent.payloads.remove(ticket)
                intent.active_payloads -= 1
            if (
                intent.active_payloads == 0
                and intent.admission_task is not None
                and not intent.admission_task.done()
            ):
                intent.admission_task.cancel()
        future = ticket.future
        if future is not None and not future.done():
            if ticket.direct_executing or isinstance(
                abort_error, asyncio.CancelledError
            ):
                future.cancel()
            else:
                future.set_exception(abort_error)
        self._tickets.pop(id(ticket), None)

    def _abort_all(self, error: BaseException) -> None:
        for ticket in tuple(self._tickets.values()):
            self._abort_ticket(ticket, error)
        self._tickets.clear()
        self._lanes.clear()
        self._intent_index.clear()
        self._retire_tasks.clear()

    def _fence_after_failure(self, error: BaseException) -> None:
        if self._fenced_error is None:
            self._fenced_error = error
        for lane in self._lanes.values():
            for intent in lane.active_intents():
                if (
                    intent.admission_task is not None
                    and not intent.admission_task.done()
                ):
                    intent.admission_task.cancel()
        self._abort_all(self._fenced_error)
