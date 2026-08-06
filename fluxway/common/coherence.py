from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
import typing
from typing import NewType


class CoherenceError(RuntimeError):
    pass


class StaleEpochError(CoherenceError):
    pass


class CoherenceBackpressureError(CoherenceError):
    pass


class CoherenceStateTransitionError(CoherenceError):
    pass


class UnsupportedRemoteWritebackError(CoherenceError):
    pass


class RetirementInvariantError(CoherenceError):
    pass


class UnsupportedAllocationHandoffError(CoherenceError):
    pass


class CoherenceState(IntEnum):
    INVALID = 0
    EXCLUSIVE = 1
    SHARED = 2
    MODIFIED = 3


class CoherenceAction(IntEnum):
    NONE = 0
    BUS_RD = 1
    RFO = 2
    SNOOP_INVALIDATE = 4
    GRANT_RETURN = 5
    OWNERSHIP_GRANT = 7
    OWNERSHIP_CANCEL = 8


class ParentAuthority(IntEnum):
    NONE = 0
    SHARED = 1
    OWNERSHIP = 2


class ExecutionFenceState(IntEnum):
    OPEN = 0
    CLOSING = 1
    FENCED = 2


class GrantKind(IntEnum):
    READ = 1
    WRITE = 2
    DELEGATED_READ = 3


class TensorAccessIntent(StrEnum):
    READONLY = "readonly"
    WRITABLE = "writable"
    DONATE = "donate"


class CoherenceEvent(IntEnum):
    PrRd = 1
    PrWr = 2
    TxBusRd = 3
    RxBusRd = 4
    PublishLocal = 5
    TxGrantReturn = 6
    TxOwnershipGrant = 7
    RxOwnershipGrant = 8
    TxOwnershipCancel = 9
    RxSnoopInval = 10
    AckSnoopInval = 11


@dataclass(slots=True)
class TransitionResult:
    status: str
    message: str = ""
    action: typing.Callable[[CoherenceEntryState], None] | None = None


ExecutionToken = NewType("ExecutionToken", int)


@dataclass(frozen=True, slots=True)
class CoherenceSubject:
    owner_spmd_code: int
    owner_rank: int
    logical_tuuid: int

    def __post_init__(self) -> None:
        if type(self.owner_spmd_code) is not int:
            raise TypeError(
                f"owner_spmd_code must be an int, got {type(self.owner_spmd_code).__name__}."
            )
        if type(self.owner_rank) is not int:
            raise TypeError(
                f"owner_rank must be an int, got {type(self.owner_rank).__name__}."
            )
        if type(self.logical_tuuid) is not int:
            raise TypeError(
                f"logical_tuuid must be an int, got {type(self.logical_tuuid).__name__}."
            )
        if self.owner_spmd_code < 0:
            raise ValueError("owner_spmd_code must be non-negative.")
        if self.owner_rank < 0:
            raise ValueError("owner_rank must be non-negative.")
        if self.logical_tuuid <= 0:
            raise ValueError("logical_tuuid must be positive.")


@dataclass(frozen=True, slots=True)
class TensorInstance:
    subject: CoherenceSubject
    epoch: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int:
            raise TypeError("TensorInstance.epoch must be an int.")
        if self.epoch < 0:
            raise ValueError("TensorInstance.epoch must be non-negative.")


@dataclass(frozen=True, slots=True)
class GrantToken:
    key: CoherenceSubject
    epoch: int
    kind: GrantKind
    execution: ExecutionToken
    new_grant: bool = True
    grant_id: int = 0


@dataclass(frozen=True, slots=True)
class GrantReady:
    token: GrantToken


@dataclass(frozen=True, slots=True)
class GrantPending:
    ready: asyncio.Future[GrantToken]
    reason: str
    requires_parent_ownership: bool = False
    invalidate_children: int = 0


@dataclass(frozen=True, slots=True)
class GrantRejected:
    error: CoherenceError


GrantResult = GrantReady | GrantPending | GrantRejected


@dataclass(slots=True)
class _PendingRequest:
    epoch: int
    kind: GrantKind
    execution: ExecutionToken
    future: asyncio.Future[GrantToken]
    publishes_local_payload: bool = False


@dataclass(slots=True)
class _PendingDeliveryWaiter:
    child_bitmap: int
    future: asyncio.Future[None]


@dataclass(slots=True)
class CoherenceEntryState:
    epoch: int
    state: CoherenceState = CoherenceState.INVALID
    child_share_bitmap: int = 0
    child_delivery_committed_bitmap: int = 0
    child_return_pending_bitmap: int = 0
    child_delivery_inflight: dict[int, int] = field(default_factory=dict)
    local_reader_count: int = 0
    local_read_grants: set[int] = field(default_factory=set)
    local_writer: ExecutionToken | None = None
    local_writer_grant_id: int | None = None
    ownership_child_id: int | None = None
    parent_authority: ParentAuthority = ParentAuthority.NONE
    revoking: bool = False
    pending_readers: deque[_PendingRequest] = field(default_factory=deque)
    pending_writers: deque[_PendingRequest] = field(default_factory=deque)
    pending_quiescence: deque[asyncio.Future[None]] = field(default_factory=deque)
    pending_delivery_waiters: deque[_PendingDeliveryWaiter] = field(
        default_factory=deque
    )


class CoherenceDirectory:
    """Endpoint-local, event-loop-owned directory for direct coherence edges."""

    def _evaluate_transition(
        self, event: CoherenceEvent, entry: CoherenceEntryState, **kwargs
    ) -> TransitionResult:
        if event == CoherenceEvent.PrRd:
            execution = kwargs.get("execution")
            if entry.ownership_child_id is not None:
                return TransitionResult("PENDING", "child owns data")
            if entry.state is CoherenceState.INVALID:
                return TransitionResult("ERROR", "cannot read invalid data.")
            if entry.revoking:
                return TransitionResult(
                    "ERROR", "cannot read while invalidation is active."
                )
            if entry.local_writer is not None:
                if execution == entry.local_writer:
                    return TransitionResult(
                        "ERROR", "active writer cannot acquire a separate read grant."
                    )
                return TransitionResult("PENDING", "writer active")
            if self._has_pending_writer(entry):
                return TransitionResult("PENDING", "ownership request pending")

            return TransitionResult(
                "READY",
                action=lambda e: setattr(e, "state", CoherenceState.SHARED)
                if e.state == CoherenceState.EXCLUSIVE
                else None,
            )

        elif event == CoherenceEvent.PrWr:
            if entry.state is CoherenceState.INVALID:
                return TransitionResult("ERROR", "cannot write invalid data.")
            if (
                entry.local_writer is not None
                or entry.ownership_child_id is not None
                or entry.local_reader_count
                or entry.child_share_bitmap
                or entry.child_delivery_inflight
            ):
                return TransitionResult("PENDING", "read grants active")
            if entry.parent_authority is ParentAuthority.SHARED:
                return TransitionResult("PENDING", "parent ownership required")

            return TransitionResult(
                "READY", action=lambda e: setattr(e, "state", CoherenceState.MODIFIED)
            )

        elif event == CoherenceEvent.TxBusRd:
            child_id = kwargs.get("child_id")
            assert child_id is not None, "child_id is required"
            publishes_local_payload = kwargs.get("publishes_local_payload", False)
            if self._try_tx_bus_rd_ready(
                entry,
                child_id,
                publishes_local_payload=publishes_local_payload,
            ):
                return TransitionResult("READY")
            return self._tx_bus_rd_blocker(
                entry,
                publishes_local_payload=publishes_local_payload,
            )

        elif event == CoherenceEvent.RxBusRd:
            if (
                entry.local_writer is not None
                or entry.ownership_child_id is not None
                or entry.state is CoherenceState.MODIFIED
            ):
                return TransitionResult(
                    "ERROR_BACKPRESSURE",
                    "cannot accept parent read grant while ownership is active.",
                )
            if entry.parent_authority is not ParentAuthority.NONE:
                return TransitionResult("ERROR", "parent authority is already active.")
            if entry.state is not CoherenceState.INVALID:
                return TransitionResult(
                    "ERROR", "BUS_RD requires an invalid destination entry."
                )

            def action(e):
                e.parent_authority = ParentAuthority.SHARED
                e.state = CoherenceState.SHARED

            return TransitionResult("READY", action=action)

        elif event == CoherenceEvent.PublishLocal:
            if entry.parent_authority is not ParentAuthority.NONE:
                return TransitionResult(
                    "ERROR", "local payload publication cannot replace a parent grant."
                )
            if entry.state is not CoherenceState.INVALID:
                return TransitionResult(
                    "ERROR", "local payload publication requires an invalid entry."
                )
            if not self._is_locally_quiescent(entry):
                return TransitionResult(
                    "ERROR_BACKPRESSURE",
                    "local payload publication requires a quiescent entry.",
                )

            return TransitionResult(
                "READY", action=lambda e: setattr(e, "state", CoherenceState.EXCLUSIVE)
            )

        elif event == CoherenceEvent.TxGrantReturn:
            if entry.parent_authority is ParentAuthority.NONE:
                return TransitionResult("ERROR", "has no parent grant to return.")
            if not self._is_locally_quiescent(entry):
                return TransitionResult(
                    "ERROR_BACKPRESSURE", "still has active local or child grants."
                )

            def action(e):
                e.parent_authority = ParentAuthority.NONE
                e.revoking = False
                e.state = CoherenceState.INVALID

            return TransitionResult("READY", action=action)

        elif event == CoherenceEvent.TxOwnershipGrant:
            child_id = kwargs.get("child_id")
            assert child_id is not None, "child_id is required"
            if (
                entry.local_reader_count
                or entry.child_share_bitmap
                or entry.child_delivery_inflight
            ):
                return TransitionResult(
                    "ERROR_BACKPRESSURE", "shared grants are still active."
                )
            if entry.ownership_child_id is not None:
                return TransitionResult("ERROR", "ownership is already delegated.")

            def action(e):
                e.local_writer = None
                e.local_writer_grant_id = None
                e.ownership_child_id = child_id
                e.state = CoherenceState.INVALID

            return TransitionResult("READY", action=action)

        elif event == CoherenceEvent.RxOwnershipGrant:
            if entry.parent_authority is not ParentAuthority.SHARED:
                return TransitionResult(
                    "ERROR", "ownership requires shared parent authority."
                )
            if (
                entry.local_reader_count
                or entry.child_share_bitmap
                or entry.ownership_child_id is not None
            ):
                return TransitionResult(
                    "ERROR_BACKPRESSURE", "local or child grants are still active."
                )

            return TransitionResult(
                "READY",
                action=lambda e: setattr(
                    e, "parent_authority", ParentAuthority.OWNERSHIP
                ),
            )

        elif event == CoherenceEvent.TxOwnershipCancel:
            if entry.parent_authority is not ParentAuthority.OWNERSHIP:
                return TransitionResult("ERROR", "no parent ownership to cancel.")
            if entry.local_writer is not None:
                return TransitionResult(
                    "ERROR", "ownership cannot be cancelled after write access starts."
                )

            def action(e):
                e.parent_authority = ParentAuthority.NONE
                e.state = CoherenceState.INVALID

            return TransitionResult("READY", action=action)

        elif event == CoherenceEvent.RxSnoopInval:
            if entry.revoking:
                return TransitionResult("ERROR", "invalidation is already active.")
            if (
                entry.parent_authority is ParentAuthority.OWNERSHIP
                or entry.state is CoherenceState.MODIFIED
                or entry.local_writer is not None
                or entry.ownership_child_id is not None
            ):
                return TransitionResult(
                    "ERROR",
                    "modified ownership cannot be invalidated without writeback.",
                )

            return TransitionResult(
                "READY", action=lambda e: setattr(e, "revoking", True)
            )

        elif event == CoherenceEvent.AckSnoopInval:
            if not entry.revoking:
                return TransitionResult("ERROR", "invalidation was not started.")
            if (
                entry.local_reader_count
                or entry.local_writer is not None
                or entry.ownership_child_id is not None
                or entry.child_share_bitmap
                or entry.child_delivery_inflight
            ):
                return TransitionResult(
                    "ERROR_BACKPRESSURE", "local or child grants are still active."
                )

            return TransitionResult(
                "READY", action=lambda e: setattr(e, "state", CoherenceState.INVALID)
            )

        return TransitionResult("ERROR", f"Unknown event {event}")

    def __init__(self) -> None:
        self._entries: dict[CoherenceSubject, CoherenceEntryState] = {}
        self._next_grant_id = 1
        self._active_delivery_receipts: dict[
            int, tuple[CoherenceSubject, int, int]
        ] = {}

    def register(self, key: CoherenceSubject, epoch: int) -> CoherenceEntryState:
        self._require_subject(key)
        self._require_epoch(epoch)
        entry = self._entries.get(key)
        if entry is None:
            entry = CoherenceEntryState(epoch=epoch)
            self._entries[key] = entry
            return entry
        if epoch < entry.epoch:
            raise StaleEpochError(
                f"coherence key={key} received stale epoch={epoch}; current={entry.epoch}."
            )
        if epoch > entry.epoch:
            if not self._can_advance_epoch(entry):
                raise CoherenceBackpressureError(
                    f"coherence key={key} cannot advance epoch while grants are active."
                )
            self._advance_epoch(entry, epoch)
        return entry

    def fence(self) -> None:
        for entry in self._entries.values():
            for request in (*entry.pending_readers, *entry.pending_writers):
                if not request.future.done():
                    request.future.cancel()
            for future in entry.pending_quiescence:
                if not future.done():
                    future.cancel()
            for waiter in entry.pending_delivery_waiters:
                if not waiter.future.done():
                    waiter.future.cancel()
        self._active_delivery_receipts.clear()
        self._entries.clear()

    def request_read(
        self, key: CoherenceSubject, epoch: int, execution: ExecutionToken
    ) -> GrantResult:
        entry = self.register(key, epoch)
        result = self._evaluate_transition(
            CoherenceEvent.PrRd, entry, execution=execution
        )
        if result.status == "PENDING":
            return self._pending(
                entry, epoch, GrantKind.READ, execution, result.message
            )
        if result.status == "ERROR":
            return GrantRejected(CoherenceStateTransitionError(result.message))

        if result.action:
            result.action(entry)
        token = self._acquire_read(entry, key, epoch, execution)
        return GrantReady(token)

    def request_write(
        self, key: CoherenceSubject, epoch: int, execution: ExecutionToken
    ) -> GrantResult:
        self._require_subject(key)
        self._require_epoch(epoch)
        entry = self._entries.get(key)
        if entry is None:
            entry = CoherenceEntryState(epoch=epoch)
            self._entries[key] = entry
        elif epoch < entry.epoch:
            raise StaleEpochError(
                f"coherence key={key} received stale epoch={epoch}; current={entry.epoch}."
            )
        elif epoch > entry.epoch:
            if not self._can_advance_epoch(entry):
                return self._pending(
                    entry,
                    epoch,
                    GrantKind.WRITE,
                    execution,
                    "current epoch grants active",
                )
            self._advance_epoch(entry, epoch)

        result = self._evaluate_transition(CoherenceEvent.PrWr, entry)
        if result.status == "PENDING":
            return self._pending(
                entry, epoch, GrantKind.WRITE, execution, result.message
            )
        if result.status == "ERROR":
            return GrantRejected(CoherenceStateTransitionError(result.message))

        if result.action:
            result.action(entry)
        return GrantReady(self._acquire_write(entry, key, epoch, execution))

    def delegate_read(
        self, key: CoherenceSubject, epoch: int, child_id: int
    ) -> GrantResult:
        return self._delegate_read(
            key,
            epoch,
            child_id,
            publishes_local_payload=False,
        )

    def delegate_local_payload(
        self,
        key: CoherenceSubject,
        epoch: int,
        child_id: int,
    ) -> GrantResult:
        return self._delegate_read(
            key,
            epoch,
            child_id,
            publishes_local_payload=True,
        )

    def _delegate_read(
        self,
        key: CoherenceSubject,
        epoch: int,
        child_id: int,
        *,
        publishes_local_payload: bool,
    ) -> GrantResult:
        self._require_subject(key)
        if child_id < 0:
            raise ValueError(f"child_id must be non-negative, got {child_id}.")
        self._require_epoch(epoch)
        entry = self._entries.get(key)
        if entry is None:
            entry = CoherenceEntryState(epoch=epoch)
            self._entries[key] = entry
        elif epoch < entry.epoch:
            raise StaleEpochError(
                f"coherence key={key} received stale epoch={epoch}; current={entry.epoch}."
            )
        elif epoch > entry.epoch:
            if not self._can_advance_epoch(entry):
                return self._pending(
                    entry,
                    epoch,
                    GrantKind.DELEGATED_READ,
                    ExecutionToken(child_id),
                    "current epoch grants active",
                    publishes_local_payload=publishes_local_payload,
                )
            self._advance_epoch(entry, epoch)

        mask = 1 << child_id
        new_grant = not bool(entry.child_share_bitmap & mask)
        if self._try_tx_bus_rd_ready(
            entry,
            child_id,
            publishes_local_payload=publishes_local_payload,
        ):
            return GrantReady(
                GrantToken(
                    key,
                    epoch,
                    GrantKind.DELEGATED_READ,
                    ExecutionToken(child_id),
                    new_grant=new_grant,
                )
            )
        result = self._tx_bus_rd_blocker(
            entry,
            publishes_local_payload=publishes_local_payload,
        )
        if result.status == "PENDING":
            return self._pending(
                entry,
                epoch,
                GrantKind.DELEGATED_READ,
                ExecutionToken(child_id),
                result.message,
                publishes_local_payload=publishes_local_payload,
            )
        if result.status == "ERROR":
            return GrantRejected(CoherenceStateTransitionError(result.message))

        raise CoherenceStateTransitionError(
            f"unexpected BUS_RD transition status={result.status}."
        )

    @classmethod
    def _try_tx_bus_rd_ready(
        cls,
        entry: CoherenceEntryState,
        child_id: int,
        *,
        publishes_local_payload: bool,
    ) -> bool:
        if (
            entry.revoking
            or entry.local_writer is not None
            or entry.ownership_child_id is not None
            or cls._has_pending_writer(entry)
            or (
                publishes_local_payload
                and entry.parent_authority is not ParentAuthority.NONE
            )
            or (not publishes_local_payload and entry.state is CoherenceState.INVALID)
            or entry.state is CoherenceState.MODIFIED
        ):
            return False
        if entry.state is CoherenceState.INVALID:
            entry.state = CoherenceState.EXCLUSIVE
        mask = 1 << child_id
        if not entry.child_share_bitmap & mask:
            entry.child_share_bitmap |= mask
            if entry.state is CoherenceState.EXCLUSIVE:
                entry.state = CoherenceState.SHARED
        return True

    @classmethod
    def _tx_bus_rd_blocker(
        cls,
        entry: CoherenceEntryState,
        *,
        publishes_local_payload: bool,
    ) -> TransitionResult:
        if entry.revoking:
            return TransitionResult(
                "ERROR", "cannot delegate while invalidation is active."
            )
        if entry.local_writer is not None:
            return TransitionResult("PENDING", "local writer active")
        if entry.ownership_child_id is not None:
            return TransitionResult("PENDING", "child owns data")
        if cls._has_pending_writer(entry):
            return TransitionResult("PENDING", "ownership request pending")
        if (
            publishes_local_payload
            and entry.parent_authority is not ParentAuthority.NONE
        ):
            return TransitionResult(
                "ERROR",
                "local payload publication cannot replace a parent grant.",
            )
        if entry.state is CoherenceState.INVALID:
            return TransitionResult("ERROR", "cannot delegate invalid data.")
        if entry.state is CoherenceState.MODIFIED:
            return TransitionResult(
                "ERROR", "cannot delegate modified ownership as shared data."
            )
        raise CoherenceStateTransitionError("BUS_RD blocker is not blocking.")

    def accept_parent_read(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self.register(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.RxBusRd, entry)
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)

    def publish_local_payload(
        self,
        key: CoherenceSubject,
        epoch: int,
    ) -> CoherenceEntryState:
        entry = self.register(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.PublishLocal, entry)
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)
        return entry

    def return_parent(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.TxGrantReturn, entry)
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(f"coherence key={key} {result.message}")
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(f"coherence key={key} {result.message}")
        if result.action:
            result.action(entry)
        self._wake_pending(entry, key)

    def parent_return_ready(self, key: CoherenceSubject, epoch: int) -> bool:
        entry = self._require_current(key, epoch)
        return (
            entry.parent_authority is not ParentAuthority.NONE
            and self._is_locally_quiescent(entry)
        )

    def has_parent_ownership(self, key: CoherenceSubject, epoch: int) -> bool:
        return (
            self._require_current(key, epoch).parent_authority
            is ParentAuthority.OWNERSHIP
        )

    def require_retirement_quiescence(self, instance: TensorInstance) -> None:
        entry = self._require_current(instance.subject, instance.epoch)
        if not self._can_advance_epoch(entry):
            raise RetirementInvariantError(
                f"tensor instance is not retirement-quiescent: {instance}."
            )

    def require_local_write(
        self,
        instance: TensorInstance,
        execution: ExecutionToken,
    ) -> GrantToken:
        entry = self._require_current(instance.subject, instance.epoch)
        if (
            entry.state is not CoherenceState.MODIFIED
            or entry.local_writer != execution
            or entry.local_writer_grant_id is None
        ):
            raise UnsupportedAllocationHandoffError(
                "allocation handoff requires the current execution write grant."
            )
        return GrantToken(
            instance.subject,
            instance.epoch,
            GrantKind.WRITE,
            execution,
            grant_id=entry.local_writer_grant_id,
        )

    def handoff_local_write(
        self,
        old: TensorInstance,
        new: TensorInstance,
        execution: ExecutionToken,
    ) -> None:
        self.require_local_handoff(old, new, execution)
        new_entry = self.register(new.subject, new.epoch)
        old_entry = self._require_current(old.subject, old.epoch)
        old_entry.local_writer = None
        old_entry.local_writer_grant_id = None
        old_entry.state = CoherenceState.INVALID
        new_entry.state = CoherenceState.EXCLUSIVE
        self._wake_pending(old_entry, old.subject)

    def require_local_handoff(
        self,
        old: TensorInstance,
        new: TensorInstance,
        execution: ExecutionToken,
    ) -> None:
        self.require_local_write(old, execution)
        if new.subject == old.subject:
            raise UnsupportedAllocationHandoffError(
                "allocation handoff requires a new SSA subject."
            )
        new_entry = self._entries.get(new.subject)
        if new_entry is not None and (
            new_entry.epoch != new.epoch
            or new_entry.state is not CoherenceState.INVALID
            or not self._can_advance_epoch(new_entry)
        ):
            raise UnsupportedAllocationHandoffError(
                "allocation handoff destination must be invalid and quiescent."
            )

    def grant_is_active(self, token: GrantToken) -> bool:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is GrantKind.READ:
            return token.grant_id in entry.local_read_grants
        if token.kind is GrantKind.WRITE:
            return (
                entry.local_writer == token.execution
                and entry.local_writer_grant_id == token.grant_id
            )
        raise CoherenceStateTransitionError(
            f"unsupported local grant kind={token.kind}."
        )

    async def wait_until_quiescent(
        self,
        key: CoherenceSubject,
        epoch: int,
    ) -> None:
        entry = self.register(key, epoch)
        if self._is_locally_quiescent(entry):
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry.pending_quiescence.append(future)
        try:
            await future
        finally:
            if not future.done():
                future.cancel()
            if future in entry.pending_quiescence:
                entry.pending_quiescence.remove(future)

    def return_child(self, key: CoherenceSubject, epoch: int, child_id: int) -> None:
        entry = self._require_current(key, epoch)
        mask = 1 << child_id
        if not entry.child_share_bitmap & mask:
            raise CoherenceStateTransitionError(
                f"child_id={child_id} has no active share grant for key={key}."
            )
        if entry.child_return_pending_bitmap & mask:
            raise CoherenceStateTransitionError(
                f"child_id={child_id} already has a pending share return."
            )
        if entry.child_delivery_inflight.get(child_id, 0):
            entry.child_return_pending_bitmap |= mask
            return
        entry.child_share_bitmap &= ~mask
        entry.child_delivery_committed_bitmap &= ~mask
        self._settle_shared_state(entry)
        if entry.pending_readers or entry.pending_writers or entry.pending_quiescence:
            self._wake_pending(entry, key)

    def register_child_delivery(self, token: GrantToken) -> GrantToken:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is not GrantKind.DELEGATED_READ:
            raise CoherenceStateTransitionError(
                "child delivery requires a delegated read grant."
            )
        child_id = int(token.execution)
        if not entry.child_share_bitmap & (1 << child_id):
            raise CoherenceStateTransitionError(
                f"child_id={child_id} has no share grant for delivery."
            )
        receipt_id = self._next_grant_id
        self._next_grant_id += 1
        entry.child_delivery_inflight[child_id] = (
            entry.child_delivery_inflight.get(child_id, 0) + 1
        )
        self._active_delivery_receipts[receipt_id] = (
            token.key,
            token.epoch,
            child_id,
        )
        return GrantToken(
            token.key,
            token.epoch,
            token.kind,
            token.execution,
            new_grant=token.new_grant,
            grant_id=receipt_id,
        )

    def complete_child_delivery(
        self,
        token: GrantToken,
        *,
        delivered: bool,
    ) -> None:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is not GrantKind.DELEGATED_READ or token.grant_id <= 0:
            raise CoherenceStateTransitionError(
                "child delivery completion requires a delivery receipt."
            )
        child_id = int(token.execution)
        receipt = self._active_delivery_receipts.pop(token.grant_id, None)
        if receipt != (token.key, token.epoch, child_id):
            raise CoherenceStateTransitionError(
                f"child delivery receipt={token.grant_id} is not active."
            )
        inflight = entry.child_delivery_inflight.get(child_id, 0)
        if inflight <= 0:
            raise CoherenceStateTransitionError(
                "child delivery inflight count underflow."
            )
        if inflight == 1:
            del entry.child_delivery_inflight[child_id]
        else:
            entry.child_delivery_inflight[child_id] = inflight - 1
        mask = 1 << child_id
        sharing_changed = False
        if delivered and entry.child_share_bitmap & mask:
            entry.child_delivery_committed_bitmap |= mask
            entry.child_return_pending_bitmap &= ~mask
        if child_id not in entry.child_delivery_inflight:
            if entry.child_return_pending_bitmap & mask:
                entry.child_return_pending_bitmap &= ~mask
                entry.child_share_bitmap &= ~mask
                entry.child_delivery_committed_bitmap &= ~mask
                sharing_changed = True
            elif (
                not delivered
                and entry.child_share_bitmap & mask
                and not entry.child_delivery_committed_bitmap & mask
            ):
                entry.child_share_bitmap &= ~mask
                sharing_changed = True
        if entry.pending_delivery_waiters:
            self._wake_delivery_waiters(entry)
        if sharing_changed:
            self._settle_shared_state(entry)
            self._wake_pending(entry, token.key)

    async def wait_until_deliveries_complete(
        self,
        key: CoherenceSubject,
        epoch: int,
        child_bitmap: int,
    ) -> None:
        entry = self._require_current(key, epoch)
        if child_bitmap < 0:
            raise ValueError("child_bitmap must be non-negative.")
        if not self._delivery_receipts_overlap(entry, child_bitmap):
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiter = _PendingDeliveryWaiter(child_bitmap, future)
        entry.pending_delivery_waiters.append(waiter)
        try:
            await future
        finally:
            if not future.done():
                future.cancel()
            if waiter in entry.pending_delivery_waiters:
                entry.pending_delivery_waiters.remove(waiter)

    def delegate_ownership(self, token: GrantToken, child_id: int) -> None:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is not GrantKind.WRITE:
            raise CoherenceStateTransitionError("ownership requires a write grant.")
        if (
            entry.local_writer != token.execution
            or entry.local_writer_grant_id != token.grant_id
        ):
            raise CoherenceStateTransitionError("write grant is not active.")

        result = self._evaluate_transition(
            CoherenceEvent.TxOwnershipGrant, entry, child_id=child_id
        )
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)

        if result.action:
            result.action(entry)

    def accept_parent_ownership(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.RxOwnershipGrant, entry)
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)
        self._wake_pending(entry, key)

    def cancel_parent_ownership(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.TxOwnershipCancel, entry)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)
        self._wake_pending(entry, key)

    def return_ownership_child(
        self,
        key: CoherenceSubject,
        epoch: int,
        child_id: int,
        *,
        data_valid: bool = False,
    ) -> None:
        entry = self._require_current(key, epoch)
        if entry.ownership_child_id != child_id:
            raise CoherenceStateTransitionError(
                f"child_id={child_id} does not own coherence key={key}."
            )
        entry.ownership_child_id = None
        entry.state = CoherenceState.EXCLUSIVE if data_valid else CoherenceState.INVALID
        self._wake_pending(entry, key)

    def invalidate(self, key: CoherenceSubject, epoch: int) -> None:
        self.begin_invalidation(key, epoch)
        self.finish_invalidation(key, epoch)

    def begin_invalidation(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.RxSnoopInval, entry)
        if result.status == "ERROR" and "writeback" in result.message:
            raise UnsupportedRemoteWritebackError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)

    def finish_invalidation(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.AckSnoopInval, entry)
        if result.status == "ERROR_BACKPRESSURE":
            raise CoherenceBackpressureError(result.message)
        if result.status == "ERROR":
            raise CoherenceStateTransitionError(result.message)
        if result.action:
            result.action(entry)

    def release(self, token: GrantToken) -> None:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is GrantKind.READ:
            if token.grant_id not in entry.local_read_grants:
                raise CoherenceStateTransitionError("read grant was already released.")
            entry.local_read_grants.remove(token.grant_id)
            entry.local_reader_count -= 1
            self._settle_shared_state(entry)
        elif token.kind is GrantKind.WRITE:
            if (
                entry.local_writer != token.execution
                or entry.local_writer_grant_id != token.grant_id
            ):
                raise CoherenceStateTransitionError(
                    "write grant does not belong to execution."
                )
            entry.local_writer = None
            entry.local_writer_grant_id = None
            if entry.parent_authority is ParentAuthority.OWNERSHIP:
                entry.state = CoherenceState.MODIFIED
            elif entry.child_share_bitmap:
                entry.state = CoherenceState.SHARED
            else:
                entry.state = CoherenceState.EXCLUSIVE
        else:
            raise CoherenceStateTransitionError(f"unsupported grant kind={token.kind}.")
        self._wake_pending(entry, token.key)

    def _pending(
        self,
        entry: CoherenceEntryState,
        epoch: int,
        kind: GrantKind,
        execution: ExecutionToken,
        reason: str,
        *,
        publishes_local_payload: bool = False,
    ) -> GrantPending:
        future: asyncio.Future[GrantToken] = asyncio.get_running_loop().create_future()
        request = _PendingRequest(
            epoch,
            kind,
            execution,
            future,
            publishes_local_payload,
        )
        if kind in (GrantKind.READ, GrantKind.DELEGATED_READ):
            entry.pending_readers.append(request)
        else:
            entry.pending_writers.append(request)
        return GrantPending(
            future,
            reason,
            requires_parent_ownership=(
                kind is GrantKind.WRITE
                and entry.parent_authority is ParentAuthority.SHARED
            ),
            invalidate_children=(
                entry.child_share_bitmap if kind is GrantKind.WRITE else 0
            ),
        )

    def _wake_pending(self, entry: CoherenceEntryState, key: CoherenceSubject) -> None:
        if entry.local_writer is not None or entry.ownership_child_id is not None:
            return
        if (
            entry.local_reader_count == 0
            and entry.child_share_bitmap == 0
            and not entry.child_delivery_inflight
        ):
            while entry.pending_writers:
                request = entry.pending_writers.popleft()
                if request.future.done():
                    continue
                if request.epoch < entry.epoch:
                    request.future.set_exception(
                        StaleEpochError(
                            f"coherence key={key} pending epoch={request.epoch}; "
                            f"current={entry.epoch}."
                        )
                    )
                    continue
                if request.epoch > entry.epoch:
                    if not self._can_advance_epoch(entry):
                        entry.pending_writers.appendleft(request)
                        return
                    self._advance_epoch(entry, request.epoch)
                if (
                    request.kind is GrantKind.DELEGATED_READ
                    and request.publishes_local_payload
                    and entry.state is CoherenceState.INVALID
                ):
                    if entry.parent_authority is not ParentAuthority.NONE:
                        request.future.set_exception(
                            CoherenceStateTransitionError(
                                "local payload publication cannot replace a parent grant."
                            )
                        )
                        continue
                    entry.state = CoherenceState.EXCLUSIVE
                if entry.state is CoherenceState.INVALID:
                    request.future.set_exception(
                        CoherenceStateTransitionError(
                            "pending write grant became invalid before acquisition."
                        )
                    )
                    continue
                if entry.parent_authority is ParentAuthority.SHARED:
                    entry.pending_writers.appendleft(request)
                    return
                request.future.set_result(
                    self._acquire_write(
                        entry,
                        key,
                        request.epoch,
                        request.execution,
                    )
                )
                return
        while entry.pending_readers and entry.local_writer is None:
            request = entry.pending_readers.popleft()
            if request.future.done():
                continue
            if request.epoch < entry.epoch:
                request.future.set_exception(
                    StaleEpochError(
                        f"coherence key={key} pending epoch={request.epoch}; "
                        f"current={entry.epoch}."
                    )
                )
                continue
            if request.epoch > entry.epoch:
                if not self._can_advance_epoch(entry):
                    entry.pending_readers.appendleft(request)
                    return
                self._advance_epoch(entry, request.epoch)
            if (
                request.kind is GrantKind.DELEGATED_READ
                and request.publishes_local_payload
                and entry.state is CoherenceState.INVALID
            ):
                if entry.parent_authority is not ParentAuthority.NONE:
                    request.future.set_exception(
                        CoherenceStateTransitionError(
                            "local payload publication cannot replace parent authority."
                        )
                    )
                    continue
                entry.state = CoherenceState.EXCLUSIVE
            if entry.state is CoherenceState.INVALID:
                request.future.set_exception(
                    CoherenceStateTransitionError(
                        "pending read grant became invalid before acquisition."
                    )
                )
                continue
            if request.kind is GrantKind.READ:
                token = self._acquire_read(
                    entry,
                    key,
                    request.epoch,
                    request.execution,
                )
            elif request.kind is GrantKind.DELEGATED_READ:
                entry.child_share_bitmap |= 1 << int(request.execution)
                token = GrantToken(
                    key,
                    request.epoch,
                    request.kind,
                    request.execution,
                )
            else:
                request.future.set_exception(
                    CoherenceStateTransitionError(
                        f"unsupported pending reader kind={request.kind}."
                    )
                )
                continue
            entry.state = CoherenceState.SHARED
            request.future.set_result(token)
        if self._is_locally_quiescent(entry):
            while entry.pending_quiescence:
                future = entry.pending_quiescence.popleft()
                if not future.done():
                    future.set_result(None)

    @staticmethod
    def _has_pending_writer(entry: CoherenceEntryState) -> bool:
        while entry.pending_writers and entry.pending_writers[0].future.done():
            entry.pending_writers.popleft()
        return bool(entry.pending_writers)

    @classmethod
    def _wake_delivery_waiters(cls, entry: CoherenceEntryState) -> None:
        if not entry.pending_delivery_waiters:
            return
        remaining = deque()
        for waiter in entry.pending_delivery_waiters:
            if waiter.future.done():
                continue
            if cls._delivery_receipts_overlap(entry, waiter.child_bitmap):
                remaining.append(waiter)
                continue
            waiter.future.set_result(None)
        entry.pending_delivery_waiters = remaining

    @staticmethod
    def _delivery_receipts_overlap(
        entry: CoherenceEntryState,
        child_bitmap: int,
    ) -> bool:
        return any(
            child_bitmap & (1 << child_id) for child_id in entry.child_delivery_inflight
        )

    def _require_current(
        self, key: CoherenceSubject, epoch: int
    ) -> CoherenceEntryState:
        entry = self._entries.get(key)
        if entry is None:
            raise CoherenceStateTransitionError(f"unknown coherence key={key}.")
        if epoch != entry.epoch:
            raise StaleEpochError(
                f"coherence key={key} received epoch={epoch}; current={entry.epoch}."
            )
        return entry

    @classmethod
    def _is_locally_quiescent(cls, entry: CoherenceEntryState) -> bool:
        return (
            entry.local_reader_count == 0
            and entry.local_writer is None
            and entry.ownership_child_id is None
            and entry.child_share_bitmap == 0
            and not entry.child_delivery_inflight
            and not cls._has_live_pending(entry)
        )

    @classmethod
    def _can_advance_epoch(cls, entry: CoherenceEntryState) -> bool:
        return (
            cls._is_locally_quiescent(entry)
            and entry.parent_authority is ParentAuthority.NONE
        )

    @staticmethod
    def _advance_epoch(entry: CoherenceEntryState, epoch: int) -> None:
        entry.epoch = epoch
        entry.state = CoherenceState.INVALID
        entry.revoking = False

    def _acquire_read(
        self,
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        epoch: int,
        execution: ExecutionToken,
    ) -> GrantToken:
        grant_id = self._next_grant_id
        self._next_grant_id += 1
        entry.local_read_grants.add(grant_id)
        entry.local_reader_count += 1
        return GrantToken(
            key,
            epoch,
            GrantKind.READ,
            execution,
            grant_id=grant_id,
        )

    def _acquire_write(
        self,
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        epoch: int,
        execution: ExecutionToken,
    ) -> GrantToken:
        grant_id = self._next_grant_id
        self._next_grant_id += 1
        entry.local_writer = execution
        entry.local_writer_grant_id = grant_id
        entry.state = CoherenceState.MODIFIED
        return GrantToken(
            key,
            epoch,
            GrantKind.WRITE,
            execution,
            grant_id=grant_id,
        )

    @staticmethod
    def _settle_shared_state(entry: CoherenceEntryState) -> None:
        if (
            entry.state is CoherenceState.SHARED
            and entry.local_reader_count == 0
            and entry.child_share_bitmap == 0
            and not entry.child_delivery_inflight
        ):
            entry.state = (
                CoherenceState.SHARED
                if entry.parent_authority is ParentAuthority.SHARED
                else (
                    CoherenceState.MODIFIED
                    if entry.parent_authority is ParentAuthority.OWNERSHIP
                    else CoherenceState.EXCLUSIVE
                )
            )

    @staticmethod
    def _has_live_pending(entry: CoherenceEntryState) -> bool:
        for queue in (entry.pending_writers, entry.pending_readers):
            while queue and queue[0].future.done():
                queue.popleft()
            if queue:
                return True
        return False

    @staticmethod
    def _require_epoch(epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}.")

    @staticmethod
    def _require_subject(subject: CoherenceSubject) -> None:
        if not isinstance(subject, CoherenceSubject):
            raise TypeError(
                "coherence subject must be CoherenceSubject, "
                f"got {type(subject).__name__}."
            )
