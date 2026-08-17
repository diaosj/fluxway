from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import ClassVar, NewType


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


class InvalidationResult(IntEnum):
    STARTED = 1
    SATISFIED_BY_PARENT_RETURN = 2


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


class TransitionStatus(StrEnum):
    READY = "ready"
    PENDING = "pending"
    ERROR = "error"
    ERROR_BACKPRESSURE = "error_backpressure"


class PendingDisposition(IntEnum):
    BLOCKED = 1
    GRANTED = 2
    REJECTED = 3


ExecutionToken = NewType("ExecutionToken", int)


@dataclass(frozen=True, slots=True)
class CoherenceTransitionContext:
    execution: ExecutionToken | None = None
    child_id: int | None = None
    publishes_local_payload: bool = False


@dataclass(frozen=True, slots=True)
class TransitionResult:
    status: TransitionStatus
    message: str = ""
    action: Callable[[CoherenceEntryState, CoherenceTransitionContext], None] | None = (
        None
    )

    def commit(
        self,
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext = CoherenceTransitionContext(),
    ) -> None:
        if self.status is not TransitionStatus.READY or self.action is None:
            raise CoherenceStateTransitionError(
                f"cannot commit transition status={self.status}."
            )
        self.action(entry, context)


@dataclass(frozen=True, slots=True, order=True)
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


@dataclass(frozen=True, slots=True, order=True)
class CoherenceInstance:
    subject: CoherenceSubject
    epoch: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int:
            raise TypeError("CoherenceInstance.epoch must be an int.")
        if self.epoch < 0:
            raise ValueError("CoherenceInstance.epoch must be non-negative.")


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
    request_id: int
    reason: str
    requires_parent_ownership: bool = False
    invalidate_children: int = 0


@dataclass(frozen=True, slots=True)
class GrantRejected:
    error: CoherenceError


GrantResult = GrantReady | GrantPending | GrantRejected


@dataclass(frozen=True, slots=True)
class ParentReturnToken:
    key: CoherenceSubject
    epoch: int
    transaction_id: int


@dataclass(slots=True)
class ParentGrantState:
    authority: ParentAuthority = ParentAuthority.NONE
    return_transaction: ParentReturnToken | None = None
    return_tombstone_epoch: int | None = None

    def begin_return(self, token: ParentReturnToken) -> None:
        if self.return_transaction is not None:
            raise CoherenceStateTransitionError(
                "parent return transaction is already active."
            )
        if self.authority not in (
            ParentAuthority.SHARED,
            ParentAuthority.OWNERSHIP,
        ):
            raise CoherenceStateTransitionError("parent grant is not returnable.")
        self.authority = ParentAuthority.NONE
        self.return_transaction = token
        self.return_tombstone_epoch = token.epoch

    def observe_return(self, key: CoherenceSubject) -> None:
        if self.return_transaction is None:
            raise CoherenceStateTransitionError(
                f"coherence key={key} has no active parent return transaction."
            )
        self.return_transaction = None

    def complete_return(self, token: ParentReturnToken) -> None:
        if self.return_transaction != token:
            raise CoherenceStateTransitionError(
                f"parent return id={token.transaction_id} is not bound to entry."
            )
        self.return_transaction = None

    def accept_shared(self, epoch: int) -> None:
        if self.authority is not ParentAuthority.NONE:
            raise CoherenceStateTransitionError("parent authority is already active.")
        if self.return_transaction is not None:
            raise CoherenceStateTransitionError(
                "parent return transaction is still active."
            )
        self.authority = ParentAuthority.SHARED
        if self.return_tombstone_epoch == epoch:
            self.return_tombstone_epoch = None

    def coalesces_invalidation(self, epoch: int) -> bool:
        return self.return_tombstone_epoch == epoch

    @property
    def return_in_flight(self) -> bool:
        return self.return_transaction is not None


@dataclass(slots=True)
class _PendingRequest:
    request_id: int
    epoch: int
    kind: GrantKind
    execution: ExecutionToken
    publishes_local_payload: bool = False
    result: GrantReady | GrantRejected | None = None


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
    parent_grant: ParentGrantState = field(default_factory=ParentGrantState)
    revoking: bool = False
    pending_readers: deque[_PendingRequest] = field(default_factory=deque)
    pending_writers: deque[_PendingRequest] = field(default_factory=deque)


@dataclass(frozen=True, slots=True)
class TransitionGuard:
    predicate: Callable[[CoherenceEntryState, CoherenceTransitionContext], bool]
    result: TransitionResult


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    guards: tuple[TransitionGuard, ...]
    ready: TransitionResult
    validate: Callable[[CoherenceTransitionContext], None] | None = None


class CoherenceDirectory:
    """Endpoint-local, event-loop-owned directory for direct coherence edges."""

    _transition_specs_cache: ClassVar[
        Mapping[CoherenceEvent, TransitionSpec] | None
    ] = None

    def __init__(self) -> None:
        self._entries: dict[CoherenceSubject, CoherenceEntryState] = {}
        self._next_grant_id = 1
        self._next_request_id = 1
        self._pending_requests: dict[int, _PendingRequest] = {}
        self._progress_listeners: list[Callable[[], None]] = []
        self._next_parent_return_transaction_id = 1
        self._parent_return_transactions: dict[int, ParentReturnToken] = {}
        self._active_delivery_receipts: dict[
            int, tuple[CoherenceSubject, int, int]
        ] = {}
        transition_specs = type(self)._transition_specs_cache
        if transition_specs is None:
            transition_specs = self._create_transition_specs()
            type(self)._transition_specs_cache = transition_specs
        self._transition_specs = transition_specs
        self._pending_events = {
            GrantKind.READ: CoherenceEvent.PrRd,
            GrantKind.WRITE: CoherenceEvent.PrWr,
            GrantKind.DELEGATED_READ: CoherenceEvent.TxBusRd,
        }
        self._pending_grant_factories: dict[
            GrantKind,
            Callable[
                [CoherenceEntryState, CoherenceSubject, _PendingRequest], GrantToken
            ],
        ] = {
            GrantKind.READ: self._complete_pending_read,
            GrantKind.WRITE: self._complete_pending_write,
            GrantKind.DELEGATED_READ: self._complete_pending_delegated_read,
        }
        registered_grant_kinds = frozenset(self._pending_events)
        if registered_grant_kinds != frozenset(GrantKind):
            missing = frozenset(GrantKind) - registered_grant_kinds
            raise CoherenceStateTransitionError(
                f"incomplete pending event mapping: missing={missing}."
            )
        if frozenset(self._pending_grant_factories) != registered_grant_kinds:
            raise CoherenceStateTransitionError(
                "pending event and grant factory mappings do not cover the same kinds."
            )

    def _evaluate_transition(
        self,
        event: CoherenceEvent,
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext = CoherenceTransitionContext(),
    ) -> TransitionResult:
        try:
            spec = self._transition_specs[event]
        except KeyError as exc:
            raise CoherenceStateTransitionError(
                f"unregistered coherence event={event!r}."
            ) from exc
        if spec.validate is not None:
            spec.validate(context)
        for guard in spec.guards:
            if guard.predicate(entry, context):
                return guard.result
        return spec.ready

    @classmethod
    def _create_transition_specs(cls) -> Mapping[CoherenceEvent, TransitionSpec]:
        def guard(
            predicate: Callable[
                [CoherenceEntryState, CoherenceTransitionContext], bool
            ],
            status: TransitionStatus,
            message: str,
        ) -> TransitionGuard:
            return TransitionGuard(predicate, TransitionResult(status, message))

        def ready(
            action: Callable[[CoherenceEntryState, CoherenceTransitionContext], None],
        ) -> TransitionResult:
            return TransitionResult(TransitionStatus.READY, action=action)

        set_modified = cls._set_state_action(CoherenceState.MODIFIED)
        set_exclusive = cls._set_state_action(CoherenceState.EXCLUSIVE)
        set_invalid = cls._set_state_action(CoherenceState.INVALID)
        specs = {
            CoherenceEvent.PrRd: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.ownership_child_id is not None,
                        TransitionStatus.PENDING,
                        "child owns data",
                    ),
                    guard(
                        lambda entry, _: entry.state is CoherenceState.INVALID,
                        TransitionStatus.ERROR,
                        "cannot read invalid data.",
                    ),
                    guard(
                        lambda entry, _: entry.revoking,
                        TransitionStatus.ERROR,
                        "cannot read while invalidation is active.",
                    ),
                    guard(
                        lambda entry, context: entry.local_writer is not None
                        and context.execution == entry.local_writer,
                        TransitionStatus.ERROR,
                        "active writer cannot acquire a separate read grant.",
                    ),
                    guard(
                        lambda entry, _: entry.local_writer is not None,
                        TransitionStatus.PENDING,
                        "writer active",
                    ),
                    guard(
                        lambda entry, _: cls._has_pending_writer(entry),
                        TransitionStatus.PENDING,
                        "ownership request pending",
                    ),
                ),
                ready(cls._share_if_exclusive),
            ),
            CoherenceEvent.PrWr: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.state is CoherenceState.INVALID,
                        TransitionStatus.ERROR,
                        "cannot write invalid data.",
                    ),
                    guard(
                        lambda entry, _: entry.local_writer is not None
                        or entry.ownership_child_id is not None
                        or bool(entry.local_reader_count)
                        or bool(entry.child_share_bitmap)
                        or bool(entry.child_delivery_inflight),
                        TransitionStatus.PENDING,
                        "read grants active",
                    ),
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is ParentAuthority.SHARED,
                        TransitionStatus.PENDING,
                        "parent ownership required",
                    ),
                ),
                ready(set_modified),
            ),
            CoherenceEvent.TxBusRd: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.revoking,
                        TransitionStatus.ERROR,
                        "cannot delegate while invalidation is active.",
                    ),
                    guard(
                        lambda entry, _: entry.local_writer is not None,
                        TransitionStatus.PENDING,
                        "local writer active",
                    ),
                    guard(
                        lambda entry, _: entry.ownership_child_id is not None,
                        TransitionStatus.PENDING,
                        "child owns data",
                    ),
                    guard(
                        lambda entry, _: cls._has_pending_writer(entry),
                        TransitionStatus.PENDING,
                        "ownership request pending",
                    ),
                    guard(
                        lambda entry, context: context.publishes_local_payload
                        and entry.parent_grant.authority is not ParentAuthority.NONE,
                        TransitionStatus.ERROR,
                        "local payload publication cannot replace a parent grant.",
                    ),
                    guard(
                        lambda entry, context: not context.publishes_local_payload
                        and entry.state is CoherenceState.INVALID,
                        TransitionStatus.ERROR,
                        "cannot delegate invalid data.",
                    ),
                    guard(
                        lambda entry, _: entry.state is CoherenceState.MODIFIED,
                        TransitionStatus.ERROR,
                        "cannot delegate modified ownership as shared data.",
                    ),
                ),
                ready(cls._delegate_shared_read),
                cls._require_child_context,
            ),
            CoherenceEvent.RxBusRd: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.local_writer is not None
                        or entry.ownership_child_id is not None
                        or entry.state is CoherenceState.MODIFIED,
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "cannot accept parent read grant while ownership is active.",
                    ),
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is not ParentAuthority.NONE
                        or entry.parent_grant.return_in_flight,
                        TransitionStatus.ERROR,
                        "parent authority is already active.",
                    ),
                    guard(
                        lambda entry, _: entry.state is not CoherenceState.INVALID,
                        TransitionStatus.ERROR,
                        "BUS_RD requires an invalid destination entry.",
                    ),
                ),
                ready(cls._accept_shared_parent_grant),
            ),
            CoherenceEvent.PublishLocal: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is not ParentAuthority.NONE,
                        TransitionStatus.ERROR,
                        "local payload publication cannot replace a parent grant.",
                    ),
                    guard(
                        lambda entry, _: entry.state is not CoherenceState.INVALID,
                        TransitionStatus.ERROR,
                        "local payload publication requires an invalid entry.",
                    ),
                    guard(
                        lambda entry, _: not cls._is_locally_quiescent(entry),
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "local payload publication requires a quiescent entry.",
                    ),
                ),
                ready(set_exclusive),
            ),
            CoherenceEvent.TxGrantReturn: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        not in (ParentAuthority.SHARED, ParentAuthority.OWNERSHIP),
                        TransitionStatus.ERROR,
                        "has no parent grant to return.",
                    ),
                    guard(
                        lambda entry, _: not cls._is_locally_quiescent(entry),
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "still has active local or child grants.",
                    ),
                ),
                ready(set_invalid),
            ),
            CoherenceEvent.TxOwnershipGrant: TransitionSpec(
                (
                    guard(
                        lambda entry, _: bool(entry.local_reader_count)
                        or bool(entry.child_share_bitmap)
                        or bool(entry.child_delivery_inflight),
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "shared grants are still active.",
                    ),
                    guard(
                        lambda entry, _: entry.ownership_child_id is not None,
                        TransitionStatus.ERROR,
                        "ownership is already delegated.",
                    ),
                ),
                ready(cls._delegate_ownership),
                cls._require_child_context,
            ),
            CoherenceEvent.RxOwnershipGrant: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is not ParentAuthority.SHARED,
                        TransitionStatus.ERROR,
                        "ownership requires shared parent authority.",
                    ),
                    guard(
                        lambda entry, _: bool(entry.local_reader_count)
                        or bool(entry.child_share_bitmap)
                        or entry.ownership_child_id is not None,
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "local or child grants are still active.",
                    ),
                ),
                ready(cls._accept_parent_ownership),
            ),
            CoherenceEvent.TxOwnershipCancel: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is not ParentAuthority.OWNERSHIP,
                        TransitionStatus.ERROR,
                        "no parent ownership to cancel.",
                    ),
                    guard(
                        lambda entry, _: entry.local_writer is not None,
                        TransitionStatus.ERROR,
                        "ownership cannot be cancelled after write access starts.",
                    ),
                ),
                ready(cls._cancel_parent_ownership),
            ),
            CoherenceEvent.RxSnoopInval: TransitionSpec(
                (
                    guard(
                        lambda entry, _: entry.revoking,
                        TransitionStatus.ERROR,
                        "invalidation is already active.",
                    ),
                    guard(
                        lambda entry, _: entry.parent_grant.authority
                        is ParentAuthority.OWNERSHIP
                        or entry.state is CoherenceState.MODIFIED
                        or entry.local_writer is not None
                        or entry.ownership_child_id is not None,
                        TransitionStatus.ERROR,
                        "modified ownership cannot be invalidated without writeback.",
                    ),
                ),
                ready(cls._start_invalidation),
            ),
            CoherenceEvent.AckSnoopInval: TransitionSpec(
                (
                    guard(
                        lambda entry, _: not entry.revoking,
                        TransitionStatus.ERROR,
                        "invalidation was not started.",
                    ),
                    guard(
                        lambda entry, _: bool(entry.local_reader_count)
                        or entry.local_writer is not None
                        or entry.ownership_child_id is not None
                        or bool(entry.child_share_bitmap)
                        or bool(entry.child_delivery_inflight),
                        TransitionStatus.ERROR_BACKPRESSURE,
                        "local or child grants are still active.",
                    ),
                ),
                ready(set_invalid),
            ),
        }
        registered_events = frozenset(specs)
        protocol_events = frozenset(CoherenceEvent)
        if registered_events != protocol_events:
            missing = protocol_events - registered_events
            extra = registered_events - protocol_events
            raise CoherenceStateTransitionError(
                f"incomplete coherence state machine: missing={missing}, extra={extra}."
            )
        return MappingProxyType(specs)

    @staticmethod
    def _set_state_action(
        state: CoherenceState,
    ) -> Callable[[CoherenceEntryState, CoherenceTransitionContext], None]:
        def set_state(
            entry: CoherenceEntryState,
            context: CoherenceTransitionContext,
        ) -> None:
            entry.state = state

        return set_state

    @staticmethod
    def _share_if_exclusive(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        if entry.state is CoherenceState.EXCLUSIVE:
            entry.state = CoherenceState.SHARED

    @staticmethod
    def _delegate_shared_read(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        child_id = context.child_id
        if child_id is None:
            raise CoherenceStateTransitionError("TxBusRd requires child_id.")
        if entry.state is CoherenceState.INVALID:
            entry.state = CoherenceState.EXCLUSIVE
        child_mask = 1 << child_id
        if not entry.child_share_bitmap & child_mask:
            entry.child_share_bitmap |= child_mask
            if entry.state is CoherenceState.EXCLUSIVE:
                entry.state = CoherenceState.SHARED

    @staticmethod
    def _accept_shared_parent_grant(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        entry.parent_grant.accept_shared(entry.epoch)
        entry.state = CoherenceState.SHARED

    @staticmethod
    def _delegate_ownership(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        child_id = context.child_id
        if child_id is None:
            raise CoherenceStateTransitionError("TxOwnershipGrant requires child_id.")
        entry.local_writer = None
        entry.local_writer_grant_id = None
        entry.ownership_child_id = child_id
        entry.state = CoherenceState.INVALID

    @staticmethod
    def _accept_parent_ownership(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        entry.parent_grant.authority = ParentAuthority.OWNERSHIP

    @staticmethod
    def _cancel_parent_ownership(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        entry.parent_grant.authority = ParentAuthority.NONE
        entry.state = CoherenceState.INVALID

    @staticmethod
    def _start_invalidation(
        entry: CoherenceEntryState,
        context: CoherenceTransitionContext,
    ) -> None:
        entry.revoking = True

    @staticmethod
    def _require_child_context(context: CoherenceTransitionContext) -> None:
        if context.child_id is None:
            raise CoherenceStateTransitionError(
                "coherence transition requires child_id."
            )

    def admit(self, instance: CoherenceInstance) -> CoherenceEntryState:
        """Synchronously admit one canonical coherence instance."""
        self._require_subject(instance.subject)
        self._require_epoch(instance.epoch)
        entry = self._entries.get(instance.subject)
        if entry is None:
            entry = CoherenceEntryState(epoch=instance.epoch)
            self._entries[instance.subject] = entry
            return entry
        if instance.epoch < entry.epoch:
            raise StaleEpochError(
                f"coherence key={instance.subject} received stale epoch={instance.epoch}; "
                f"current={entry.epoch}."
            )
        if instance.epoch > entry.epoch:
            if not self._can_advance_epoch(entry):
                raise CoherenceBackpressureError(
                    f"coherence key={instance.subject} cannot advance epoch while "
                    "grants are active."
                )
            self._advance_epoch(entry, instance.epoch)
        return entry

    def _admit_request(
        self, instance: CoherenceInstance
    ) -> tuple[CoherenceEntryState, bool]:
        try:
            return self.admit(instance), True
        except CoherenceBackpressureError:
            entry = self._entries.get(instance.subject)
            if entry is None:
                raise
            return entry, False

    def add_progress_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        if listener in self._progress_listeners:
            raise CoherenceStateTransitionError(
                "progress listener is already registered."
            )
        self._progress_listeners.append(listener)

        def remove() -> None:
            if listener in self._progress_listeners:
                self._progress_listeners.remove(listener)

        return remove

    def _signal_progress(self) -> None:
        for listener in tuple(self._progress_listeners):
            listener()

    def can_admit(self, instance: CoherenceInstance) -> bool:
        self._require_subject(instance.subject)
        self._require_epoch(instance.epoch)
        entry = self._entries.get(instance.subject)
        if entry is None:
            return True
        if instance.epoch < entry.epoch:
            raise StaleEpochError(
                f"coherence key={instance.subject} received stale epoch={instance.epoch}; "
                f"current={entry.epoch}."
            )
        return instance.epoch == entry.epoch or self._can_advance_epoch(entry)

    def take_pending_result(self, request_id: int) -> GrantReady | GrantRejected | None:
        request = self._pending_requests.get(request_id)
        if request is None:
            raise CoherenceStateTransitionError(
                f"unknown coherence pending request id={request_id}."
            )
        if request.result is None:
            return None
        result = request.result
        del self._pending_requests[request_id]
        return result

    def pending_result(self, request_id: int) -> GrantReady | GrantRejected | None:
        request = self._pending_requests.get(request_id)
        if request is None:
            raise CoherenceStateTransitionError(
                f"unknown coherence pending request id={request_id}."
            )
        return request.result

    def cancel_pending_request(self, request_id: int) -> None:
        request = self._pending_requests.pop(request_id, None)
        if request is None:
            raise CoherenceStateTransitionError(
                f"unknown coherence pending request id={request_id}."
            )
        request.result = GrantRejected(
            CoherenceStateTransitionError(
                f"coherence pending request id={request_id} was cancelled."
            )
        )
        for entry in self._entries.values():
            for queue in (entry.pending_readers, entry.pending_writers):
                try:
                    queue.remove(request)
                except ValueError:
                    pass
        self._signal_progress()

    def is_quiescent(self, instance: CoherenceInstance) -> bool:
        entry = self._require_current(instance.subject, instance.epoch)
        return self._is_locally_quiescent(entry)

    def deliveries_complete(
        self,
        instance: CoherenceInstance,
        child_bitmap: int,
    ) -> bool:
        if child_bitmap < 0:
            raise ValueError("child_bitmap must be non-negative.")
        entry = self._require_current(instance.subject, instance.epoch)
        return not self._delivery_receipts_overlap(entry, child_bitmap)

    def fence(self) -> None:
        self._pending_requests.clear()
        self._active_delivery_receipts.clear()
        self._parent_return_transactions.clear()
        self._entries.clear()
        self._signal_progress()

    def request_read(
        self, key: CoherenceSubject, epoch: int, execution: ExecutionToken
    ) -> GrantResult:
        entry = self.admit(CoherenceInstance(key, epoch))
        context = CoherenceTransitionContext(execution=execution)
        result = self._evaluate_transition(
            CoherenceEvent.PrRd,
            entry,
            context,
        )
        if result.status is TransitionStatus.PENDING:
            return self._pending(
                entry, epoch, GrantKind.READ, execution, result.message
            )
        if result.status is TransitionStatus.ERROR:
            return GrantRejected(CoherenceStateTransitionError(result.message))

        result.commit(entry, context)
        token = self._acquire_read(entry, key, epoch, execution)
        return GrantReady(token)

    def request_write(
        self, key: CoherenceSubject, epoch: int, execution: ExecutionToken
    ) -> GrantResult:
        entry, ready = self._admit_request(CoherenceInstance(key, epoch))
        if not ready:
            return self._pending(
                entry,
                epoch,
                GrantKind.WRITE,
                execution,
                "current epoch grants active",
            )

        result = self._evaluate_transition(CoherenceEvent.PrWr, entry)
        if result.status is TransitionStatus.PENDING:
            return self._pending(
                entry, epoch, GrantKind.WRITE, execution, result.message
            )
        if result.status is TransitionStatus.ERROR:
            return GrantRejected(CoherenceStateTransitionError(result.message))

        result.commit(entry)
        token = self._acquire_write(entry, key, epoch, execution)
        return GrantReady(token)

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
        entry, ready = self._admit_request(CoherenceInstance(key, epoch))
        if not ready:
            return self._pending(
                entry,
                epoch,
                GrantKind.DELEGATED_READ,
                ExecutionToken(child_id),
                "current epoch grants active",
                publishes_local_payload=publishes_local_payload,
            )

        mask = 1 << child_id
        new_grant = not bool(entry.child_share_bitmap & mask)
        context = CoherenceTransitionContext(
            child_id=child_id,
            publishes_local_payload=publishes_local_payload,
        )
        result = self._evaluate_transition(
            CoherenceEvent.TxBusRd,
            entry,
            context,
        )
        if result.status is TransitionStatus.PENDING:
            return self._pending(
                entry,
                epoch,
                GrantKind.DELEGATED_READ,
                ExecutionToken(child_id),
                result.message,
                publishes_local_payload=publishes_local_payload,
            )
        if result.status is TransitionStatus.ERROR:
            return GrantRejected(CoherenceStateTransitionError(result.message))
        if result.status is not TransitionStatus.READY:
            raise CoherenceStateTransitionError(
                f"unexpected BUS_RD transition status={result.status}."
            )
        result.commit(entry, context)
        token = GrantToken(
            key,
            epoch,
            GrantKind.DELEGATED_READ,
            ExecutionToken(child_id),
            new_grant=new_grant,
        )
        return GrantReady(token)

    def accept_parent_read(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._entries.get(key)
        if entry is not None and epoch > entry.epoch:
            if entry.parent_grant.return_in_flight:
                entry.parent_grant.observe_return(key)
                self.admit(CoherenceInstance(key, epoch))
            else:
                entry = self.admit(CoherenceInstance(key, epoch))
        else:
            entry = self.admit(CoherenceInstance(key, epoch))
        result = self._evaluate_transition(CoherenceEvent.RxBusRd, entry)
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)

    def publish_local_payload(
        self,
        key: CoherenceSubject,
        epoch: int,
    ) -> CoherenceEntryState:
        entry = self.admit(CoherenceInstance(key, epoch))
        result = self._evaluate_transition(CoherenceEvent.PublishLocal, entry)
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)
        return entry

    def begin_parent_return(
        self, key: CoherenceSubject, epoch: int
    ) -> ParentReturnToken:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.TxGrantReturn, entry)
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(f"coherence key={key} {result.message}")
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(f"coherence key={key} {result.message}")
        result.commit(entry)
        transaction_id = self._next_parent_return_transaction_id
        self._next_parent_return_transaction_id += 1
        token = ParentReturnToken(key, epoch, transaction_id)
        entry.parent_grant.begin_return(token)
        self._parent_return_transactions[transaction_id] = token
        return token

    def complete_parent_return(self, token: ParentReturnToken) -> None:
        active = self._parent_return_transactions.pop(token.transaction_id, None)
        if active != token:
            raise CoherenceStateTransitionError(
                "parent return transaction "
                f"id={token.transaction_id} is already completed or inactive."
            )
        entry = self._entries.get(token.key)
        if entry is None:
            raise CoherenceStateTransitionError(
                f"parent return id={token.transaction_id} has no directory entry."
            )
        if entry.epoch > token.epoch:
            return
        if entry.epoch < token.epoch:
            raise StaleEpochError(
                f"coherence key={token.key} completed future epoch={token.epoch}; "
                f"current={entry.epoch}."
            )
        entry.parent_grant.complete_return(token)
        entry.revoking = False
        self._wake_pending(entry, token.key)

    def parent_return_ready(self, key: CoherenceSubject, epoch: int) -> bool:
        entry = self._require_current(key, epoch)
        return entry.parent_grant.authority in (
            ParentAuthority.SHARED,
            ParentAuthority.OWNERSHIP,
        ) and self._is_locally_quiescent(entry)

    def has_parent_ownership(self, key: CoherenceSubject, epoch: int) -> bool:
        return (
            self._require_current(key, epoch).parent_grant.authority
            is ParentAuthority.OWNERSHIP
        )

    def require_current_entry_retirement(self, instance: CoherenceInstance) -> None:
        entry = self._entries.get(instance.subject)
        if entry is None:
            raise RetirementInvariantError(
                f"tensor instance has no current coherence entry: {instance}."
            )
        if instance.epoch != entry.epoch:
            raise RetirementInvariantError(
                f"tensor instance is not the current coherence epoch: {instance}; "
                f"current={entry.epoch}."
            )
        if not self._can_advance_epoch(entry):
            raise RetirementInvariantError(
                f"tensor instance is not retirement-quiescent: {instance}."
            )

    @staticmethod
    def require_returned_instance(
        instance: CoherenceInstance,
        returned_instances: frozenset[CoherenceInstance],
    ) -> None:
        if instance not in returned_instances:
            raise RetirementInvariantError(
                f"tensor instance lacks a completed parent-return proof: {instance}."
            )

    def require_local_write(
        self,
        instance: CoherenceInstance,
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
        old: CoherenceInstance,
        new: CoherenceInstance,
        execution: ExecutionToken,
    ) -> None:
        self.require_local_handoff(old, new, execution)
        new_entry = self.admit(new)
        old_entry = self._require_current(old.subject, old.epoch)
        old_entry.local_writer = None
        old_entry.local_writer_grant_id = None
        old_entry.state = CoherenceState.INVALID
        new_entry.state = CoherenceState.EXCLUSIVE
        self._wake_pending(old_entry, old.subject)

    def require_local_handoff(
        self,
        old: CoherenceInstance,
        new: CoherenceInstance,
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
            self._signal_progress()
            return
        entry.child_share_bitmap &= ~mask
        entry.child_delivery_committed_bitmap &= ~mask
        self._settle_shared_state(entry)
        if entry.pending_readers or entry.pending_writers:
            self._wake_pending(entry, key)
        else:
            self._signal_progress()

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
        if sharing_changed:
            self._settle_shared_state(entry)
            self._wake_pending(entry, token.key)
        else:
            self._signal_progress()

    def delegate_ownership(self, token: GrantToken, child_id: int) -> None:
        entry = self._require_current(token.key, token.epoch)
        if token.kind is not GrantKind.WRITE:
            raise CoherenceStateTransitionError("ownership requires a write grant.")
        if (
            entry.local_writer != token.execution
            or entry.local_writer_grant_id != token.grant_id
        ):
            raise CoherenceStateTransitionError("write grant is not active.")

        context = CoherenceTransitionContext(child_id=child_id)
        result = self._evaluate_transition(
            CoherenceEvent.TxOwnershipGrant,
            entry,
            context,
        )
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)

        result.commit(entry, context)

    def accept_parent_ownership(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.RxOwnershipGrant, entry)
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)
        self._wake_pending(entry, key)

    def cancel_parent_ownership(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.TxOwnershipCancel, entry)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)
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

    def invalidate(self, key: CoherenceSubject, epoch: int) -> InvalidationResult:
        result = self.begin_invalidation(key, epoch)
        if result is InvalidationResult.STARTED:
            self.finish_invalidation(key, epoch)
        return result

    def begin_invalidation(
        self, key: CoherenceSubject, epoch: int
    ) -> InvalidationResult:
        entry = self._entries.get(key)
        if entry is not None and entry.parent_grant.coalesces_invalidation(epoch):
            return InvalidationResult.SATISFIED_BY_PARENT_RETURN
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.RxSnoopInval, entry)
        if result.status is TransitionStatus.ERROR and "writeback" in result.message:
            raise UnsupportedRemoteWritebackError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)
        return InvalidationResult.STARTED

    def finish_invalidation(self, key: CoherenceSubject, epoch: int) -> None:
        entry = self._require_current(key, epoch)
        result = self._evaluate_transition(CoherenceEvent.AckSnoopInval, entry)
        if result.status is TransitionStatus.ERROR_BACKPRESSURE:
            raise CoherenceBackpressureError(result.message)
        if result.status is TransitionStatus.ERROR:
            raise CoherenceStateTransitionError(result.message)
        result.commit(entry)

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
            if entry.parent_grant.authority is ParentAuthority.OWNERSHIP:
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
        request_id = self._next_request_id
        self._next_request_id += 1
        request = _PendingRequest(
            request_id,
            epoch,
            kind,
            execution,
            publishes_local_payload,
        )
        self._pending_requests[request_id] = request
        if kind in (GrantKind.READ, GrantKind.DELEGATED_READ):
            entry.pending_readers.append(request)
        else:
            entry.pending_writers.append(request)
        return GrantPending(
            request_id,
            reason,
            requires_parent_ownership=(
                kind is GrantKind.WRITE
                and entry.parent_grant.authority is ParentAuthority.SHARED
            ),
            invalidate_children=(
                entry.child_share_bitmap if kind is GrantKind.WRITE else 0
            ),
        )

    def _wake_pending(self, entry: CoherenceEntryState, key: CoherenceSubject) -> None:
        self._signal_progress()
        if entry.local_writer is not None or entry.ownership_child_id is not None:
            return
        if (
            entry.local_reader_count == 0
            and entry.child_share_bitmap == 0
            and not entry.child_delivery_inflight
        ):
            while entry.pending_writers:
                request = entry.pending_writers.popleft()
                if request.result is not None:
                    continue
                if request.epoch < entry.epoch:
                    request.result = GrantRejected(
                        StaleEpochError(
                            f"coherence key={key} pending epoch={request.epoch}; current={entry.epoch}."
                        )
                    )
                    continue
                if request.epoch > entry.epoch:
                    if not self._can_advance_epoch(entry):
                        entry.pending_writers.appendleft(request)
                        return
                    self.admit(CoherenceInstance(key, request.epoch))
                disposition = self._resume_pending_request(entry, key, request)
                if disposition is PendingDisposition.BLOCKED:
                    entry.pending_writers.appendleft(request)
                    return
                if disposition is PendingDisposition.GRANTED:
                    return
        while entry.pending_readers and entry.local_writer is None:
            request = entry.pending_readers.popleft()
            if request.result is not None:
                continue
            if request.epoch < entry.epoch:
                request.result = GrantRejected(
                    StaleEpochError(
                        f"coherence key={key} pending epoch={request.epoch}; current={entry.epoch}."
                    )
                )
                continue
            if request.epoch > entry.epoch:
                if not self._can_advance_epoch(entry):
                    entry.pending_readers.appendleft(request)
                    return
                self.admit(CoherenceInstance(key, request.epoch))
            disposition = self._resume_pending_request(entry, key, request)
            if disposition is PendingDisposition.BLOCKED:
                entry.pending_readers.appendleft(request)
                return
        self._signal_progress()

    def _resume_pending_request(
        self,
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        request: _PendingRequest,
    ) -> PendingDisposition:
        event = self._pending_events[request.kind]
        context = CoherenceTransitionContext(
            execution=request.execution,
            child_id=int(request.execution),
            publishes_local_payload=request.publishes_local_payload,
        )
        result = self._evaluate_transition(
            event,
            entry,
            context,
        )
        match result.status:
            case TransitionStatus.PENDING:
                return PendingDisposition.BLOCKED
            case TransitionStatus.ERROR:
                request.result = GrantRejected(
                    CoherenceStateTransitionError(
                        f"pending {request.kind.name.lower()} grant became invalid before acquisition: {result.message}"
                    )
                )
                return PendingDisposition.REJECTED
            case TransitionStatus.ERROR_BACKPRESSURE:
                request.result = GrantRejected(
                    CoherenceBackpressureError(result.message)
                )
                return PendingDisposition.REJECTED
            case TransitionStatus.READY:
                result.commit(entry, context)
                token = self._pending_grant_factories[request.kind](entry, key, request)
                request.result = GrantReady(token)
                return PendingDisposition.GRANTED
        raise CoherenceStateTransitionError(
            f"unhandled transition status={result.status}."
        )

    def _complete_pending_read(
        self,
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        request: _PendingRequest,
    ) -> GrantToken:
        return self._acquire_read(entry, key, request.epoch, request.execution)

    def _complete_pending_write(
        self,
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        request: _PendingRequest,
    ) -> GrantToken:
        return self._acquire_write(entry, key, request.epoch, request.execution)

    @staticmethod
    def _complete_pending_delegated_read(
        entry: CoherenceEntryState,
        key: CoherenceSubject,
        request: _PendingRequest,
    ) -> GrantToken:
        return GrantToken(
            key,
            request.epoch,
            request.kind,
            request.execution,
        )

    @staticmethod
    def _has_pending_writer(entry: CoherenceEntryState) -> bool:
        while entry.pending_writers and entry.pending_writers[0].result is not None:
            entry.pending_writers.popleft()
        return bool(entry.pending_writers)

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
            and entry.parent_grant.authority is ParentAuthority.NONE
            and not entry.parent_grant.return_in_flight
        )

    @staticmethod
    def _advance_epoch(entry: CoherenceEntryState, epoch: int) -> None:
        if entry.parent_grant.return_in_flight:
            raise CoherenceBackpressureError(
                "cannot advance epoch while a parent return transaction is active."
            )
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
                if entry.parent_grant.authority is ParentAuthority.SHARED
                else (
                    CoherenceState.MODIFIED
                    if entry.parent_grant.authority is ParentAuthority.OWNERSHIP
                    else CoherenceState.EXCLUSIVE
                )
            )

    @staticmethod
    def _has_live_pending(entry: CoherenceEntryState) -> bool:
        for queue in (entry.pending_writers, entry.pending_readers):
            while queue and queue[0].result is not None:
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
