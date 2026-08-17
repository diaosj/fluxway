import pytest

from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    ExecutionToken,
    GrantPending,
    GrantReady,
    ParentAuthority,
    CoherenceInstance,
)
from fluxway.spmd.worker.coherence_runtime import (
    CoherenceAdmissionRuntime,
    ParentGrantReturnRuntime,
    ParentReturnRoute,
)


def _entry(directory: CoherenceDirectory, key: CoherenceSubject, epoch: int):
    return directory.admit(CoherenceInstance(key, epoch))


def _shared_child(directory: CoherenceDirectory, key: CoherenceSubject) -> None:
    directory.accept_parent_read(key, 7)
    local = directory.request_read(key, 7, ExecutionToken(1))
    assert isinstance(local, GrantReady)
    directory.release(local.token)


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
    pending = parent.delegate_local_payload(subject, 8, child_id=0)
    assert isinstance(pending, GrantPending)

    async def send(_group: tuple[ParentReturnRoute, ...]) -> None:
        parent.return_child(subject, 7, child_id=0)
        child.accept_parent_read(subject, 8)

    runtime = ParentGrantReturnRuntime(child, send)
    retired = await runtime.return_parent_grants((route,))

    token = await CoherenceAdmissionRuntime(parent).await_grant(pending)
    assert token.epoch == 8
    assert _entry(child, subject, 8).parent_grant.authority is ParentAuthority.SHARED
    assert retired == frozenset((CoherenceInstance(subject, 7),))
