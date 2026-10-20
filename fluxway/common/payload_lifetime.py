"""Explicit ownership-transfer lifecycle for transport payloads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from fluxway.common.coherence import CoherenceInstance, CoherenceStateTransitionError


@dataclass(frozen=True, slots=True)
class ReceiverAcceptanceReceipt:
    publication_id: int
    instance: CoherenceInstance | None
    child_slot: int
    context_index: int
    target_rank: int
    batch_seq: int

    def __post_init__(self) -> None:
        if type(self.publication_id) is not int or self.publication_id <= 0:
            raise ValueError("publication_id must be a positive int.")
        if type(self.child_slot) is not int or self.child_slot < 0:
            raise ValueError("child_slot must be a non-negative int.")
        if type(self.context_index) is not int or self.context_index < 0:
            raise ValueError("context_index must be a non-negative int.")
        if type(self.target_rank) is not int or self.target_rank < 0:
            raise ValueError("target_rank must be a non-negative int.")
        if type(self.batch_seq) is not int or self.batch_seq < 0:
            raise ValueError("batch_seq must be a non-negative int.")


class PayloadRecycler(Protocol):
    def release(self, handle: PayloadHandle) -> None: ...


class PayloadHandle:
    __slots__ = ("_owner", "_released", "token")

    def __init__(self, token: object, owner: Any) -> None:
        self.token = token
        self._owner = owner
        self._released = False

    @property
    def owner(self) -> Any:
        if self._released:
            raise CoherenceStateTransitionError("payload handle was already released.")
        return self._owner

    @property
    def released(self) -> bool:
        return self._released

    def _drop_owner(self) -> None:
        if self._released:
            raise CoherenceStateTransitionError("payload handle was already released.")
        self._owner = None
        self._released = True


class ReferencePayloadRecycler:
    """Ownership-transfer recycler: drops the final strong reference exactly once."""

    def release(self, handle: PayloadHandle) -> None:
        handle._drop_owner()


@dataclass(frozen=True, slots=True)
class PayloadReleaseProof:
    publication_id: int
    accepted_children: frozenset[tuple[CoherenceInstance | None, int, int, int, int]]
    local_leases: int
    transport_retained: bool


class PayloadLifetime:
    """Single owner for transport retention, local leases, and physical release."""

    __slots__ = (
        "publication_id",
        "_expected_children",
        "_accepted_children",
        "_handles",
        "_recycler",
        "_local_leases",
        "_transport_retained",
        "_fenced_error",
        "_released",
    )

    def __init__(
        self,
        publication_id: int,
        handles: Iterable[PayloadHandle],
        expected_children: Iterable[
            tuple[CoherenceInstance | None, int, int, int, int]
        ],
        recycler: PayloadRecycler | None = None,
    ) -> None:
        if type(publication_id) is not int or publication_id <= 0:
            raise ValueError("publication_id must be a positive int.")
        expected = frozenset(expected_children)
        if len(expected) == 0:
            raise ValueError("payload lifetime requires at least one child.")
        if any(
            type(context_index) is not int
            or context_index < 0
            or type(target_rank) is not int
            or target_rank < 0
            or type(batch_seq) is not int
            or batch_seq < 0
            for _instance, _child_slot, context_index, target_rank, batch_seq in expected
        ):
            raise ValueError("payload child keys must be non-negative integers.")
        self.publication_id = publication_id
        self._expected_children = expected
        self._accepted_children: set[
            tuple[CoherenceInstance | None, int, int, int, int]
        ] = set()
        self._handles = tuple(handles)
        self._recycler = recycler or ReferencePayloadRecycler()
        self._local_leases = 0
        self._transport_retained = True
        self._fenced_error: BaseException | None = None
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def accepted_children(
        self,
    ) -> frozenset[tuple[CoherenceInstance | None, int, int, int, int]]:
        return frozenset(self._accepted_children)

    def retain_local_lease(self) -> None:
        self._ensure_open()
        self._local_leases += 1

    def release_local_lease(self) -> None:
        if self._local_leases <= 0:
            raise CoherenceStateTransitionError("payload local lease count underflow.")
        self._local_leases -= 1
        self._try_release()

    def accept(self, receipt: ReceiverAcceptanceReceipt) -> None:
        self._ensure_open()
        if receipt.publication_id != self.publication_id:
            raise CoherenceStateTransitionError(
                "receiver receipt publication mismatch."
            )
        key = (
            receipt.instance,
            receipt.child_slot,
            receipt.context_index,
            receipt.target_rank,
            receipt.batch_seq,
        )
        if key not in self._expected_children:
            raise CoherenceStateTransitionError(
                "receiver receipt child is not declared."
            )
        if key in self._accepted_children:
            raise CoherenceStateTransitionError("receiver receipt was delivered twice.")
        self._accepted_children.add(key)
        self._try_release()

    def accept_child(
        self,
        context_index: int,
        target_rank: int,
        batch_seq: int,
        child_slot: int,
        instance: CoherenceInstance | None = None,
    ) -> ReceiverAcceptanceReceipt:
        self._ensure_open()
        receipt = ReceiverAcceptanceReceipt(
            self.publication_id,
            instance,
            child_slot,
            context_index,
            target_rank,
            batch_seq,
        )
        key = (instance, child_slot, context_index, target_rank, batch_seq)
        if key not in self._expected_children:
            raise CoherenceStateTransitionError(
                "receiver child operation is not declared."
            )
        if key in self._accepted_children:
            raise CoherenceStateTransitionError(
                "receiver child operation was completed twice."
            )
        self._accepted_children.add(key)
        self._try_release()
        return receipt

    def release_transport_retention(self) -> PayloadReleaseProof:
        self._ensure_open()
        if not self._transport_retained:
            raise CoherenceStateTransitionError(
                "payload transport retention was already released."
            )
        if self._accepted_children != self._expected_children:
            missing = self._expected_children - self._accepted_children
            raise CoherenceStateTransitionError(
                f"payload release requires receiver acceptance for {sorted(missing)!r}."
            )
        self._transport_retained = False
        self._try_release()
        return self.release_proof()

    def fence(self, error: BaseException) -> None:
        if self._released:
            raise CoherenceStateTransitionError("cannot fence a released payload.")
        if self._fenced_error is None:
            self._fenced_error = error

    def release_proof(self) -> PayloadReleaseProof:
        return PayloadReleaseProof(
            self.publication_id,
            frozenset(self._accepted_children),
            self._local_leases,
            self._transport_retained,
        )

    def _try_release(self) -> None:
        if self._released or self._fenced_error is not None:
            return
        if self._transport_retained or self._local_leases:
            return
        if self._accepted_children != self._expected_children:
            return
        try:
            for handle in self._handles:
                self._recycler.release(handle)
        except BaseException as error:
            self._fenced_error = error
            raise
        self._released = True

    def _ensure_open(self) -> None:
        if self._fenced_error is not None:
            raise self._fenced_error
        if self._released:
            raise CoherenceStateTransitionError(
                "payload lifetime was already released."
            )


def payload_handle(owner: Any) -> PayloadHandle:
    return PayloadHandle(object(), owner)
