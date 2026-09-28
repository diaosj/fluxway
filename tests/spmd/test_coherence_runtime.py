import asyncio
import gc
import weakref

import numpy as np
import pytest

from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    ExecutionToken,
    GrantReady,
    ParentAuthority,
    CoherenceInstance,
)
from fluxway.spmd.worker.coherence_runtime import (
    CoherenceBatchRetireRuntime,
    ParentGrantReturnRuntime,
    ParentReturnRoute,
)
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer
from fluxway.spmd.worker.data_plane import InstancePublication, PublicationBatch
from fluxway.rpc.protocol import ReceiverIngressCompletion
from fluxway.transport.communication import (
    TransportDelegationDescriptor,
    TransportFrameBatch,
)


def _entry(directory: CoherenceDirectory, key: CoherenceSubject, epoch: int):
    return directory.admit(CoherenceInstance(key, epoch))


def _shared_child(directory: CoherenceDirectory, key: CoherenceSubject) -> None:
    directory.accept_parent_read(key, 7)
    local = directory.request_read(key, 7, ExecutionToken(1))
    assert isinstance(local, GrantReady)
    directory.release(local.token)


def _plan(instance: CoherenceInstance):
    return type(
        "Plan",
        (),
        {
            "coherence_instance": staticmethod(lambda transfer_code, epoch: instance),
            "coherence_child_slot": staticmethod(
                lambda transfer_code, target_rank: target_rank
            ),
            "describe_publication": staticmethod(
                lambda batches: PublicationBatch(
                    tuple(batches),
                    (InstancePublication(instance, (0, 1)),),
                )
            ),
        },
    )()


@pytest.mark.asyncio
async def test_parent_return_runtime_stages_local_transitions_before_send() -> None:
    directory = CoherenceDirectory()
    routes = (
        ParentReturnRoute(CoherenceSubject(3, 0, 19), 11, 7, "parent", 0),
        ParentReturnRoute(CoherenceSubject(3, 0, 20), 12, 7, "parent", 0),
    )
    for route in routes:
        _shared_child(directory, route.subject)
    observed: list[tuple[ParentReturnRoute, ...]] = []

    async def send(group: tuple[ParentReturnRoute, ...]) -> None:
        observed.append(group)
        for route in group:
            assert (
                _entry(directory, route.subject, 7).parent_grant.return_transaction
                is not None
            )

    runtime = ParentGrantReturnRuntime(directory, send)
    await runtime.return_parent_grants(routes)

    assert observed == [routes]
    for route in routes:
        assert (
            _entry(directory, route.subject, 7).parent_grant.authority
            is ParentAuthority.NONE
        )


@pytest.mark.asyncio
async def test_parent_return_send_failure_remains_fail_stop() -> None:
    directory = CoherenceDirectory()
    route = ParentReturnRoute(
        CoherenceSubject(3, 0, 19),
        11,
        7,
        "parent",
        0,
    )
    _shared_child(directory, route.subject)

    async def fail(_group: tuple[ParentReturnRoute, ...]) -> None:
        raise RuntimeError("return transport failed")

    runtime = ParentGrantReturnRuntime(directory, fail)
    with pytest.raises(RuntimeError, match="return transport failed"):
        await runtime.return_parent_grants((route,))

    assert (
        _entry(directory, route.subject, 7).parent_grant.return_transaction is not None
    )


@pytest.mark.asyncio
async def test_parent_return_runtime_accepts_reentrant_next_epoch() -> None:
    parent = CoherenceDirectory()
    child = CoherenceDirectory()
    subject = CoherenceSubject(3, 0, 19)
    route = ParentReturnRoute(subject, 11, 7, "parent", 0)
    parent.publish_local_payload(subject, 7)
    delegated = parent.delegate_read(subject, 7, child_id=0)
    assert isinstance(delegated, GrantReady)
    _shared_child(child, subject)
    ready = parent.delegate_local_payload(subject, 8, child_id=0)
    assert isinstance(ready, GrantReady)
    assert ready.token.epoch == 8

    async def send(_group: tuple[ParentReturnRoute, ...]) -> None:
        parent.return_child(subject, 7, child_id=0)
        child.accept_parent_read(subject, 8)

    runtime = ParentGrantReturnRuntime(child, send)
    retired = await runtime.return_parent_grants((route,))

    assert _entry(child, subject, 8).parent_grant.authority is ParentAuthority.SHARED
    assert retired == frozenset((CoherenceInstance(subject, 7),))


@pytest.mark.asyncio
async def test_one_instance_fanout_dispatches_all_children_without_sibling_wait() -> (
    None
):
    directory = CoherenceDirectory()
    subject = CoherenceSubject(3, 0, 31)
    instance = CoherenceInstance(subject, 7)
    directory.publish_local_payload(subject, 7)

    async def retire(_instances, payload):
        return await payload()

    async def admit(candidate):
        directory.admit(candidate)

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(instance)
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan = _plan(instance)
    batches = tuple(
        TransportFrameBatch(
            context_index=0,
            target_rank=target_rank,
            batch_seq=0,
            frames=((object(), ()),),
            delegations=(TransportDelegationDescriptor(1, 7),),
        )
        for target_rank in (0, 1)
    )
    entered: list[int] = []
    releases = {target_rank: asyncio.Event() for target_rank in (0, 1)}

    async def enqueue(batch):
        entered.append(batch.target_rank)
        await releases[batch.target_rank].wait()
        return ReceiverIngressCompletion(batch.target_rank)

    task = asyncio.create_task(runtime.retire_publication(plan, batches, enqueue))
    sequencer.close_intent(instance)
    for _ in range(20):
        if len(entered) == 2:
            break
        await asyncio.sleep(0)
    assert entered == [0, 1]
    releases[0].set()
    releases[1].set()
    assert await task == (0, 1)


@pytest.mark.asyncio
async def test_remote_publication_releases_producer_payload_after_all_acceptances() -> (
    None
):
    directory = CoherenceDirectory()
    subject = CoherenceSubject(3, 0, 32)
    instance = CoherenceInstance(subject, 7)
    directory.publish_local_payload(subject, 7)

    async def retire(_instances, payload):
        return await payload()

    async def admit(candidate):
        directory.admit(candidate)

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(instance)
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan = _plan(instance)
    batches = tuple(
        TransportFrameBatch(
            context_index=0,
            target_rank=target_rank,
            batch_seq=0,
            frames=((object(), ()),),
            delegations=(TransportDelegationDescriptor(1, 7),),
        )
        for target_rank in (0, 1)
    )

    async def publish() -> weakref.ReferenceType[np.ndarray]:
        payload = np.arange(128, dtype=np.int64)
        reference = weakref.ref(payload)

        async def enqueue(batch):
            assert payload[0] == 0
            return ReceiverIngressCompletion(batch.target_rank)

        task = asyncio.create_task(runtime.retire_publication(plan, batches, enqueue))
        sequencer.close_intent(instance)
        assert await task == (0, 1)
        return reference

    reference = await publish()
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
async def test_failed_publication_cancels_sibling_and_releases_all_child_receipts() -> (
    None
):
    directory = CoherenceDirectory()
    subject = CoherenceSubject(3, 0, 33)
    instance = CoherenceInstance(subject, 7)
    directory.publish_local_payload(subject, 7)

    async def retire(_instances, payload):
        return await payload()

    async def admit(candidate):
        directory.admit(candidate)

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(instance)
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan = _plan(instance)
    batches = tuple(
        TransportFrameBatch(
            context_index=0,
            target_rank=target_rank,
            batch_seq=0,
            frames=((object(), ()),),
            delegations=(TransportDelegationDescriptor(1, 7),),
        )
        for target_rank in (0, 1)
    )
    sibling_entered = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def enqueue(batch):
        if batch.target_rank == 0:
            await sibling_entered.wait()
            raise RuntimeError("target delivery failed")
        sibling_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    task = asyncio.create_task(runtime.retire_publication(plan, batches, enqueue))
    sequencer.close_intent(instance)
    with pytest.raises(RuntimeError, match="target delivery failed"):
        await task
    await asyncio.wait_for(sibling_cancelled.wait(), timeout=1.0)
    assert _entry(directory, subject, 7).child_share_bitmap == 0
