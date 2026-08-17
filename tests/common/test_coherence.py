import asyncio
from typing import cast

import pytest

from fluxway.common.coherence import (
    CoherenceBackpressureError,
    CoherenceDirectory,
    CoherenceEvent,
    InvalidationResult,
    ParentReturnToken,
    CoherenceStateTransitionError,
    CoherenceSubject,
    CoherenceInstance,
    CoherenceState,
    ExecutionToken,
    GrantPending,
    GrantReady,
    GrantRejected,
    StaleEpochError,
    TransitionStatus,
)
from fluxway.spmd.worker.coherence_runtime import CoherenceAdmissionRuntime


def _entry(directory: CoherenceDirectory, key: CoherenceSubject, epoch: int):
    return directory.admit(CoherenceInstance(key, epoch))


def _subject() -> CoherenceSubject:
    return CoherenceSubject(1, 2, 3)


def _runtime(directory: CoherenceDirectory) -> CoherenceAdmissionRuntime:
    return CoherenceAdmissionRuntime(directory)


def _publish(
    directory: CoherenceDirectory, key: CoherenceSubject, epoch: int = 7
) -> None:
    directory.publish_local_payload(key, epoch)


def test_transition_specs_are_complete_and_reused() -> None:
    first = CoherenceDirectory()
    second = CoherenceDirectory()

    assert set(first._transition_specs) == set(CoherenceEvent)
    assert first._transition_specs is second._transition_specs
    assert all(
        spec.ready.status is TransitionStatus.READY
        for spec in first._transition_specs.values()
    )


def test_directory_rejects_route_shaped_tuple_subject() -> None:
    directory = CoherenceDirectory()

    with pytest.raises(TypeError, match="CoherenceSubject"):
        _entry(directory, cast(CoherenceSubject, (1, 2, 3)), 7)


@pytest.mark.parametrize("operation", ("write", "delegate"))
def test_all_acquisition_paths_reject_unstructured_subject(operation) -> None:
    directory = CoherenceDirectory()
    key = cast(CoherenceSubject, (1, 2, 3))

    with pytest.raises(TypeError, match="CoherenceSubject"):
        if operation == "write":
            directory.request_write(key, 7, ExecutionToken(1))
        else:
            directory.delegate_read(key, 7, child_id=0)


def test_local_read_grant_releases_to_exclusive() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)

    result = directory.request_read(key, 7, ExecutionToken(11))

    assert isinstance(result, GrantReady)
    assert _entry(directory, key, 7).state is CoherenceState.SHARED

    directory.release(result.token)

    assert _entry(directory, key, 7).local_reader_count == 0
    assert _entry(directory, key, 7).state is CoherenceState.EXCLUSIVE


def test_duplicate_read_release_cannot_consume_another_reader() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    first = directory.request_read(key, 7, ExecutionToken(1))
    second = directory.request_read(key, 7, ExecutionToken(2))
    assert isinstance(first, GrantReady)
    assert isinstance(second, GrantReady)
    directory.release(first.token)

    with pytest.raises(CoherenceStateTransitionError, match="already released"):
        directory.release(first.token)

    assert _entry(directory, key, 7).local_reader_count == 1
    directory.release(second.token)


def test_child_bitmap_blocks_epoch_advance() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    directory.delegate_read(key, 7, child_id=4)

    with pytest.raises(CoherenceBackpressureError):
        _entry(directory, key, 8)

    directory.return_child(key, 7, child_id=4)
    advanced = _entry(directory, key, 8)

    assert advanced.epoch == 8
    assert advanced.child_share_bitmap == 0


def test_stale_epoch_is_rejected() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _entry(directory, key, 7)

    with pytest.raises(StaleEpochError):
        _entry(directory, key, 6)


def test_epoch_advance_does_not_restore_invalid_data() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _entry(directory, key, 7)
    directory.invalidate(key, 7)

    advanced = _entry(directory, key, 8)

    assert advanced.state is CoherenceState.INVALID
    assert isinstance(
        directory.request_read(key, 8, ExecutionToken(1)),
        GrantRejected,
    )


def test_new_parent_payload_restores_validity_after_epoch_advance() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _entry(directory, key, 7)
    directory.invalidate(key, 7)

    directory.accept_parent_read(key, 8)

    assert _entry(directory, key, 8).state is CoherenceState.SHARED


def test_local_payload_publication_restores_validity_explicitly() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _entry(directory, key, 7)
    directory.invalidate(key, 7)
    assert _entry(directory, key, 8).state is CoherenceState.INVALID

    published = directory.publish_local_payload(key, 8)

    assert published.state is CoherenceState.EXCLUSIVE
    assert isinstance(directory.delegate_read(key, 8, child_id=0), GrantReady)


def test_local_payload_publication_cannot_replace_parent_authority() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    directory.accept_parent_read(key, 7)

    with pytest.raises(CoherenceStateTransitionError, match="parent grant"):
        directory.publish_local_payload(key, 7)


@pytest.mark.asyncio
async def test_future_local_payload_delegation_suspends_until_prior_epoch_returns() -> (
    None
):
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    directory.delegate_read(key, 7, child_id=0)

    future = directory.delegate_local_payload(key, 8, child_id=0)

    assert isinstance(future, GrantPending)
    directory.return_child(key, 7, child_id=0)
    token = await _runtime(directory).await_grant(future)
    assert token.epoch == 8
    assert _entry(directory, key, 8).state is CoherenceState.SHARED


@pytest.mark.asyncio
async def test_pending_writer_resumes_without_blocking_thread() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    reader = directory.request_read(key, 7, ExecutionToken(11))
    assert isinstance(reader, GrantReady)

    writer = directory.request_write(key, 7, ExecutionToken(12))
    assert isinstance(writer, GrantPending)
    assert directory.pending_result(writer.request_id) is None

    directory.release(reader.token)
    token = await asyncio.wait_for(_runtime(directory).await_grant(writer), timeout=0.1)

    assert token.execution == ExecutionToken(12)
    assert _entry(directory, key, 7).state is CoherenceState.MODIFIED


def test_modified_parent_ownership_cannot_be_delegated_as_shared() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    directory.accept_parent_read(key, 7)
    directory.accept_parent_ownership(key, 7)
    writer = directory.request_write(key, 7, ExecutionToken(12))
    assert isinstance(writer, GrantReady)
    directory.release(writer.token)

    result = directory.delegate_read(key, 7, child_id=0)

    assert isinstance(result, GrantRejected)
    assert "modified ownership" in str(result.error)
    entry = _entry(directory, key, 7)
    assert entry.state is CoherenceState.MODIFIED
    assert entry.child_share_bitmap == 0


@pytest.mark.asyncio
async def test_pending_writer_backpressures_later_readers() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    first_reader = directory.request_read(key, 7, ExecutionToken(1))
    assert isinstance(first_reader, GrantReady)
    writer = directory.request_write(key, 7, ExecutionToken(2))
    assert isinstance(writer, GrantPending)

    later_reader = directory.request_read(key, 7, ExecutionToken(3))

    assert isinstance(later_reader, GrantPending)
    directory.release(first_reader.token)
    write_token = await _runtime(directory).await_grant(writer)
    assert directory.pending_result(later_reader.request_id) is None
    directory.release(write_token)
    assert directory.pending_result(later_reader.request_id) is not None


@pytest.mark.asyncio
async def test_inflight_delivery_invalidates_early_child_return() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    first = directory.delegate_read(key, 7, child_id=0)
    assert isinstance(first, GrantReady)
    first_receipt = directory.register_child_delivery(first.token)
    directory.complete_child_delivery(first_receipt, delivered=True)
    second = directory.delegate_read(key, 7, child_id=0)
    assert isinstance(second, GrantReady)
    second_receipt = directory.register_child_delivery(second.token)

    directory.return_child(key, 7, child_id=0)
    writer = directory.request_write(key, 7, ExecutionToken(1))

    assert isinstance(writer, GrantPending)
    assert _entry(directory, key, 7).child_share_bitmap == 1
    directory.complete_child_delivery(second_receipt, delivered=True)
    assert _entry(directory, key, 7).child_share_bitmap == 1
    assert directory.pending_result(writer.request_id) is None
    directory.return_child(key, 7, child_id=0)
    assert directory.pending_result(writer.request_id) is not None


@pytest.mark.asyncio
async def test_failed_inflight_delivery_applies_pending_child_return() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    grant = directory.delegate_read(key, 7, child_id=0)
    assert isinstance(grant, GrantReady)
    receipt = directory.register_child_delivery(grant.token)
    directory.return_child(key, 7, child_id=0)
    writer = directory.request_write(key, 7, ExecutionToken(1))
    assert isinstance(writer, GrantPending)

    directory.complete_child_delivery(receipt, delivered=False)

    assert _entry(directory, key, 7).child_share_bitmap == 0
    assert directory.pending_result(writer.request_id) is not None


@pytest.mark.asyncio
async def test_future_epoch_delegation_requires_new_payload_publication() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    current = directory.delegate_read(key, 7, child_id=4)
    assert isinstance(current, GrantReady)

    future = directory.delegate_read(key, 8, child_id=4)

    assert isinstance(future, GrantPending)
    assert directory.pending_result(future.request_id) is None

    directory.return_child(key, 7, child_id=4)
    with pytest.raises(CoherenceStateTransitionError, match="became invalid"):
        await asyncio.wait_for(_runtime(directory).await_grant(future), timeout=0.1)

    assert _entry(directory, key, 8).state is CoherenceState.INVALID


@pytest.mark.asyncio
async def test_parent_return_transaction_allows_reentrant_next_epoch() -> None:
    parent = CoherenceDirectory()
    child = CoherenceDirectory()
    key = _subject()
    parent.publish_local_payload(key, 7)
    delegated = parent.delegate_read(key, 7, child_id=4)
    assert isinstance(delegated, GrantReady)
    child.accept_parent_read(key, 7)
    local = child.request_read(key, 7, ExecutionToken(11))
    assert isinstance(local, GrantReady)
    child.release(local.token)

    pending = parent.delegate_local_payload(key, 8, child_id=4)
    assert isinstance(pending, GrantPending)
    returning = child.begin_parent_return(key, 7)
    assert isinstance(returning, ParentReturnToken)
    assert _entry(child, key, 7).parent_grant.return_transaction == returning

    parent.return_child(key, 7, child_id=4)
    child.accept_parent_read(key, 8)
    child.complete_parent_return(returning)

    token = await _runtime(parent).await_grant(pending)
    assert token.epoch == 8
    assert _entry(child, key, 8).parent_grant.authority.name == "SHARED"
    with pytest.raises(CoherenceStateTransitionError, match="already completed"):
        child.complete_parent_return(returning)


@pytest.mark.asyncio
async def test_crossing_invalidation_is_coalesced_by_parent_return() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    directory.accept_parent_read(key, 7)
    local = directory.request_read(key, 7, ExecutionToken(11))
    assert isinstance(local, GrantReady)
    directory.release(local.token)
    returning = directory.begin_parent_return(key, 7)

    assert (
        directory.begin_invalidation(key, 7)
        is InvalidationResult.SATISFIED_BY_PARENT_RETURN
    )
    directory.complete_parent_return(returning)
    assert _entry(directory, key, 7).parent_grant.authority.name == "NONE"


def test_same_epoch_regrant_clears_parent_return_tombstone() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    directory.accept_parent_read(key, 7)
    returning = directory.begin_parent_return(key, 7)
    directory.complete_parent_return(returning)

    directory.accept_parent_read(key, 7)

    assert directory.begin_invalidation(key, 7) is InvalidationResult.STARTED


def test_duplicate_child_delegation_is_idempotent_and_not_new() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)

    first = directory.delegate_read(key, 7, child_id=4)
    duplicate = directory.delegate_read(key, 7, child_id=4)

    assert isinstance(first, GrantReady)
    assert isinstance(duplicate, GrantReady)
    assert first.token.new_grant
    assert not duplicate.token.new_grant


def test_invalidation_never_discards_active_child_bitmap() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    directory.delegate_read(key, 7, child_id=4)

    with pytest.raises(CoherenceBackpressureError):
        directory.invalidate(key, 7)

    assert _entry(directory, key, 7).child_share_bitmap == 1 << 4


def test_parent_return_requires_all_local_consumers() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    directory.accept_parent_read(key, 7)
    first = directory.request_read(key, 7, ExecutionToken(11))
    second = directory.request_read(key, 7, ExecutionToken(12))
    assert isinstance(first, GrantReady)
    assert isinstance(second, GrantReady)

    directory.release(first.token)

    assert not directory.parent_return_ready(key, 7)

    directory.release(second.token)

    assert directory.parent_return_ready(key, 7)


@pytest.mark.asyncio
async def test_quiescence_future_resolves_without_polling() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    directory.delegate_read(key, 7, child_id=4)
    waiter = asyncio.create_task(
        _runtime(directory).await_quiescent(CoherenceInstance(key, 7))
    )
    await asyncio.sleep(0)

    assert not waiter.done()

    directory.return_child(key, 7, child_id=4)
    await asyncio.wait_for(waiter, timeout=0.1)


@pytest.mark.asyncio
async def test_remote_ownership_waits_for_other_direct_children() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    directory.delegate_read(key, 7, child_id=4)
    directory.delegate_read(key, 7, child_id=5)
    directory.return_child(key, 7, child_id=4)

    ownership = directory.request_write(key, 7, ExecutionToken(4))

    assert isinstance(ownership, GrantPending)
    directory.return_child(key, 7, child_id=5)
    token = await asyncio.wait_for(
        _runtime(directory).await_grant(ownership), timeout=0.1
    )
    directory.delegate_ownership(token, child_id=4)

    entry = _entry(directory, key, 7)
    assert entry.ownership_child_id == 4
    assert entry.local_writer is None
    assert entry.state is CoherenceState.INVALID

    directory.return_ownership_child(key, 7, child_id=4)

    entry = _entry(directory, key, 7)
    assert entry.ownership_child_id is None
    assert entry.state is CoherenceState.INVALID


def test_clean_ownership_return_restores_exclusive_state() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    ownership = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(ownership, GrantReady)
    directory.delegate_ownership(ownership.token, child_id=4)

    directory.return_ownership_child(
        key,
        7,
        child_id=4,
        data_valid=True,
    )

    assert _entry(directory, key, 7).state is CoherenceState.EXCLUSIVE


def test_parent_read_is_rejected_while_child_owns_data() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    writer = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(writer, GrantReady)
    directory.delegate_ownership(writer.token, child_id=0)

    with pytest.raises(CoherenceBackpressureError, match="ownership"):
        directory.accept_parent_read(key, 7)


@pytest.mark.asyncio
async def test_dirty_ownership_return_rejects_queued_reader() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    writer = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(writer, GrantReady)
    reader = directory.request_read(key, 7, ExecutionToken(5))
    assert isinstance(reader, GrantPending)
    directory.delegate_ownership(writer.token, child_id=4)

    directory.return_ownership_child(key, 7, child_id=4, data_valid=False)

    with pytest.raises(CoherenceStateTransitionError, match="became invalid"):
        await _runtime(directory).await_grant(reader)


@pytest.mark.asyncio
async def test_dirty_ownership_return_rejects_queued_writer() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    writer = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(writer, GrantReady)
    pending = directory.request_write(key, 7, ExecutionToken(5))
    assert isinstance(pending, GrantPending)
    directory.delegate_ownership(writer.token, child_id=4)

    directory.return_ownership_child(key, 7, child_id=4, data_valid=False)

    with pytest.raises(CoherenceStateTransitionError, match="became invalid"):
        await _runtime(directory).await_grant(pending)


@pytest.mark.asyncio
async def test_cancelled_pending_writer_does_not_block_successor() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    reader = directory.request_read(key, 7, ExecutionToken(1))
    assert isinstance(reader, GrantReady)
    cancelled = directory.request_write(key, 7, ExecutionToken(2))
    successor = directory.request_write(key, 7, ExecutionToken(3))
    assert isinstance(cancelled, GrantPending)
    assert isinstance(successor, GrantPending)
    directory.cancel_pending_request(cancelled.request_id)

    directory.release(reader.token)

    token = await _runtime(directory).await_grant(successor)
    assert token.execution == ExecutionToken(3)


def test_invalid_entry_rejects_same_epoch_child_delegation() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _entry(directory, key, 7)
    directory.invalidate(key, 7)

    result = directory.delegate_read(key, 7, child_id=4)

    assert isinstance(result, GrantRejected)
    assert _entry(directory, key, 7).child_share_bitmap == 0


def test_active_writer_cannot_acquire_read_grant_with_same_execution() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    writer = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(writer, GrantReady)

    result = directory.request_read(key, 7, ExecutionToken(4))

    assert isinstance(result, GrantRejected)
    entry = _entry(directory, key, 7)
    assert entry.local_writer == ExecutionToken(4)
    assert entry.local_reader_count == 0


@pytest.mark.asyncio
async def test_active_writer_cannot_be_reacquired_by_equal_numeric_token() -> None:
    directory = CoherenceDirectory()
    key = _subject()
    _publish(directory, key)
    writer = directory.request_write(key, 7, ExecutionToken(4))
    assert isinstance(writer, GrantReady)

    colliding = directory.request_write(key, 7, ExecutionToken(4))

    assert isinstance(colliding, GrantPending)
    directory.release(writer.token)
    successor = await _runtime(directory).await_grant(colliding)

    with pytest.raises(CoherenceStateTransitionError, match="does not belong"):
        directory.release(writer.token)

    assert _entry(directory, key, 7).local_writer_grant_id == successor.grant_id
    directory.release(successor)
