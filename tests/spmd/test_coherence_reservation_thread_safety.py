"""Regression tests for coherent reservation ownership transfer."""

import threading
import asyncio
import numpy as np

import pytest

from fluxway.common.coherence import (
    CoherenceInstance,
    CoherenceStateTransitionError,
    CoherenceSubject,
)
from fluxway.spmd.worker.coherence_sequencer import (
    CoherenceSequencer,
    ConsumerLeaseReservationRef,
)
from fluxway.common.types.tensor import TransportTensor


async def noop_retire(_instances, _payload):
    return None


async def noop_admit(_instance):
    return None


async def _reservation(
    instances: tuple[CoherenceInstance, ...] | None = None,
) -> tuple[
    CoherenceSequencer,
    ConsumerLeaseReservationRef,
    asyncio.Event,
    asyncio.Task,
]:
    ready = asyncio.Event()
    finish = asyncio.Event()
    sequencer = CoherenceSequencer(
        retire_payload=noop_retire,
        admit_instance=noop_admit,
    )
    if instances is None:
        instances = (
            CoherenceInstance(
                subject=CoherenceSubject(
                    owner_spmd_code=0, owner_rank=0, logical_tuuid=1
                ),
                epoch=0,
            ),
        )
    for instance in instances:
        sequencer.issue_intent(instance)

    reservation: ConsumerLeaseReservationRef | None = None

    async def retire(instances, _payload):
        nonlocal reservation
        reservation = sequencer.reserve_consumer_lease(instances)
        ready.set()
        await finish.wait()

    sequencer._retire_payload = retire
    delivery = asyncio.create_task(sequencer.commit_payload(instances, 42))
    for instance in instances:
        sequencer.close_intent(instance)
    await ready.wait()
    assert reservation is not None
    return sequencer, reservation, finish, delivery


@pytest.mark.asyncio
async def test_unclaimed_cancel_releases_reservation() -> None:
    sequencer, reservation, finish, delivery = await _reservation()

    reservation.cancel()

    assert not reservation.claimed
    assert not sequencer._consumer_lease_reservations
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_multi_instance_cleanup_releases_every_binding() -> None:
    instances = tuple(
        CoherenceInstance(
            subject=CoherenceSubject(
                owner_spmd_code=0, owner_rank=0, logical_tuuid=index
            ),
            epoch=0,
        )
        for index in (1, 2)
    )
    sequencer, reservation, finish, delivery = await _reservation(instances)
    bindings = tuple(
        control.bindings for control in sequencer._consumer_lease_reservations.values()
    )

    reservation.release_for_cleanup()

    assert len(sequencer._consumer_lease_reservations) == 0
    assert all(
        ticket.active_consumer_leases == 0
        for ticket, _intent in (
            binding for control_bindings in bindings for binding in control_bindings
        )
    )
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_claimed_reservation_requires_lease_owner() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    lease = reservation.claim()

    assert reservation.claimed
    with pytest.raises(CoherenceStateTransitionError, match="claimed"):
        reservation.cancel()

    lease.release()
    assert not sequencer._consumer_lease_reservations
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_claimed_lease_can_be_transferred_to_interrupted_cleanup() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    lease = reservation.claim()

    reservation.release_for_cleanup()

    reservation.release_for_cleanup()
    with pytest.raises(CoherenceStateTransitionError, match="already released"):
        lease.release()
    with pytest.raises(CoherenceStateTransitionError, match="already released"):
        reservation.claim()
    assert not sequencer._consumer_lease_reservations
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_tensor_cleanup_releases_claimed_reservation_lease() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    tensor = TransportTensor(np.arange(2, dtype=np.int64))
    tensor._consumer_lease_reservation_ref = reservation

    tensor._cleanup_consumer_lease_reservation_ref()

    assert tensor._consumer_lease_reservation_ref is None
    assert not sequencer._consumer_lease_reservations
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_tensor_acquire_transfers_reservation_before_cleanup() -> None:
    _sequencer, reservation, finish, delivery = await _reservation()
    tensor = TransportTensor(np.arange(2, dtype=np.int64))
    tensor._consumer_lease_reservation_ref = reservation

    lease = tensor.acquire_consumer_lease()

    assert tensor._consumer_lease_reservation_ref is None
    lease.release()
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_fence_releases_pending_reservation_without_leaking_lease() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    ticket, _intent = sequencer._consumer_lease_reservations[
        next(iter(sequencer._consumer_lease_reservations))
    ].bindings[0]

    assert ticket.active_consumer_leases == 1
    sequencer.fence()

    assert ticket.active_consumer_leases == 0
    assert not sequencer._consumer_lease_reservations
    reservation.release_for_cleanup()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery


@pytest.mark.asyncio
async def test_fence_and_tensor_cleanup_have_one_protocol_release() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    ticket, _intent = sequencer._consumer_lease_reservations[
        next(iter(sequencer._consumer_lease_reservations))
    ].bindings[0]
    tensor = TransportTensor(np.arange(2, dtype=np.int64))
    tensor._consumer_lease_reservation_ref = reservation

    sequencer.fence()
    tensor._cleanup_consumer_lease_reservation_ref()

    assert ticket.active_consumer_leases == 0
    assert tensor._consumer_lease_reservation_ref is None
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery


@pytest.mark.asyncio
async def test_cancel_after_cleanup_is_a_released_state_error() -> None:
    _sequencer, reservation, finish, delivery = await _reservation()

    reservation.release_for_cleanup()

    with pytest.raises(CoherenceStateTransitionError, match="already released"):
        reservation.cancel()
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_tensor_cleanup_from_worker_thread_releases_on_owner_loop() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    ticket, _intent = sequencer._consumer_lease_reservations[
        next(iter(sequencer._consumer_lease_reservations))
    ].bindings[0]
    tensor = TransportTensor(np.arange(2, dtype=np.int64))
    tensor._consumer_lease_reservation_ref = reservation

    thread = threading.Thread(target=tensor._cleanup_consumer_lease_reservation_ref)
    thread.start()
    thread.join()
    await asyncio.sleep(0)

    assert ticket.active_consumer_leases == 0
    assert tensor._consumer_lease_reservation_ref is None
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_concurrent_claim_and_cancel_has_one_owner() -> None:
    _sequencer, reservation, finish, delivery = await _reservation()
    outcomes: list[str] = []

    def claim() -> None:
        try:
            lease = reservation.claim()
        except CoherenceStateTransitionError:
            outcomes.append("claim_error")
        else:
            outcomes.append("claim")
            lease.release()

    def cancel() -> None:
        try:
            reservation.cancel()
        except CoherenceStateTransitionError:
            outcomes.append("cancel_error")
        else:
            outcomes.append("cancel")

    threads = (threading.Thread(target=claim), threading.Thread(target=cancel))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) in (
        ["cancel", "claim_error"],
        ["cancel_error", "claim"],
    )
    finish.set()
    await delivery


@pytest.mark.asyncio
async def test_concurrent_claim_and_fence_has_one_owner() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    ticket, _intent = sequencer._consumer_lease_reservations[
        next(iter(sequencer._consumer_lease_reservations))
    ].bindings[0]
    claim_started = threading.Event()
    outcomes: list[str] = []

    def claim() -> None:
        claim_started.set()
        try:
            reservation.claim()
        except CoherenceStateTransitionError:
            outcomes.append("claim_error")
        else:
            outcomes.append("claim")

    thread = threading.Thread(target=claim)
    thread.start()
    assert claim_started.wait(timeout=1)
    sequencer.fence()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert outcomes in (["claim"], ["claim_error"])
    assert ticket.active_consumer_leases == 0
    assert not sequencer._consumer_lease_reservations
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery


@pytest.mark.asyncio
async def test_concurrent_consumer_lease_release_has_one_owner() -> None:
    sequencer, reservation, finish, delivery = await _reservation()
    lease = reservation.claim()
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def release() -> None:
        barrier.wait()
        try:
            lease.release()
        except CoherenceStateTransitionError:
            outcomes.append("release_error")
        else:
            outcomes.append("release")

    threads = (threading.Thread(target=release), threading.Thread(target=release))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    await asyncio.sleep(0)

    assert sorted(outcomes) == ["release", "release_error"]
    finish.set()
    await delivery
