import pytest

from fluxway.common.coherence import (
    CoherenceInstance,
    CoherenceStateTransitionError,
    CoherenceSubject,
)
from fluxway.common.payload_lifetime import (
    PayloadHandle,
    PayloadLifetime,
    ReceiverAcceptanceReceipt,
)


class _Recycler:
    def __init__(self) -> None:
        self.handles: list[PayloadHandle] = []

    def release(self, handle: PayloadHandle) -> None:
        self.handles.append(handle)
        handle._drop_owner()


def _receipt(publication_id: int, child_slot: int, context: int, rank: int, seq: int):
    return ReceiverAcceptanceReceipt(
        publication_id,
        CoherenceInstance(CoherenceSubject(1, 0, 1), 0),
        child_slot,
        context,
        rank,
        seq,
    )


def test_payload_lifetime_releases_once_after_acceptance_and_local_lease() -> None:
    recycler = _Recycler()
    handle = PayloadHandle("payload", object())
    instance = CoherenceInstance(CoherenceSubject(1, 0, 1), 0)
    lifetime = PayloadLifetime(
        1,
        (handle,),
        ((instance, 4, 0, 0, 2), (instance, 5, 1, 1, 2)),
        recycler,
    )
    lifetime.retain_local_lease()
    lifetime.accept(_receipt(1, 4, 0, 0, 2))
    lifetime.accept(_receipt(1, 5, 1, 1, 2))
    lifetime.release_transport_retention()
    assert recycler.handles == []
    with pytest.raises(CoherenceStateTransitionError, match="already released"):
        lifetime.release_transport_retention()
    lifetime.release_local_lease()
    assert recycler.handles == [handle]
    assert handle.released


def test_payload_lifetime_rejects_duplicate_and_missing_children() -> None:
    recycler = _Recycler()
    lifetime = PayloadLifetime(
        2,
        (PayloadHandle("payload", object()),),
        ((CoherenceInstance(CoherenceSubject(1, 0, 1), 0), 8, 0, 0, 1),),
        recycler,
    )
    lifetime.accept(_receipt(2, 8, 0, 0, 1))
    with pytest.raises(CoherenceStateTransitionError, match="twice"):
        lifetime.accept(_receipt(2, 8, 0, 0, 1))
    other = PayloadLifetime(
        3,
        (PayloadHandle("other", object()),),
        ((CoherenceInstance(CoherenceSubject(1, 0, 1), 0), 8, 0, 0, 1),),
        recycler,
    )
    with pytest.raises(
        CoherenceStateTransitionError, match="requires receiver acceptance"
    ):
        other.release_transport_retention()


def test_payload_lifetime_fence_prevents_reuse() -> None:
    recycler = _Recycler()
    lifetime = PayloadLifetime(
        4,
        (PayloadHandle("fenced", object()),),
        ((None, 3, 0, 0, 1),),
        recycler,
    )
    lifetime.fence(RuntimeError("receiver failed"))
    with pytest.raises(RuntimeError, match="receiver failed"):
        lifetime.accept_child(0, 0, 1, 3)
    assert recycler.handles == []
