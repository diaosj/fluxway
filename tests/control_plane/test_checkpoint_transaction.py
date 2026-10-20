from __future__ import annotations

import pytest

from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.control_plane.checkpoint_transaction import (
    CheckpointTransactionFingerprint,
    PreparedStateStore,
)


def fingerprint(
    position: int, attempt: str = "attempt"
) -> CheckpointTransactionFingerprint:
    return CheckpointTransactionFingerprint.create(
        permit=CheckpointCommitPermit(
            f"{attempt}-{position}", 0, f"token-{position}".encode()
        ),
        cut=CheckpointCut.for_position(f"cut-{position}", position),
        state_digest=f"digest-{position}",
    )


def test_store_retains_only_latest_commit() -> None:
    store = PreparedStateStore[int]()
    effects: list[int] = []

    def record(value: int) -> None:
        effects.append(value)

    first = fingerprint(1)
    second = fingerprint(2)
    store.prepare("checkpoint-1", first, 1)
    store.apply("checkpoint-1", first, record)
    store.prepare("checkpoint-2", second, 2)
    store.apply("checkpoint-2", second, record)

    with pytest.raises(RuntimeError, match="requires prepare"):
        store.apply("checkpoint-1", first, record)
    assert effects == [1, 2]


def test_store_rejects_prepared_checkpoint_older_than_latest_commit() -> None:
    store = PreparedStateStore[int]()
    effects: list[int] = []

    def record(value: int) -> None:
        effects.append(value)

    first = fingerprint(1)
    second = fingerprint(2)
    store.prepare("checkpoint-1", first, 1)
    store.prepare("checkpoint-2", second, 2)
    store.apply("checkpoint-2", second, record)

    with pytest.raises(ValueError, match="is not newer than committed transaction"):
        store.apply("checkpoint-1", first, record)
    assert effects == [2]


def test_store_keeps_prepare_when_commit_effect_fails() -> None:
    store = PreparedStateStore[int]()
    transaction = fingerprint(1)
    store.prepare("checkpoint", transaction, 1)

    def fail(_value: int) -> None:
        raise RuntimeError("injected effect failure")

    with pytest.raises(RuntimeError, match="injected effect failure"):
        store.apply("checkpoint", transaction, fail)

    committed: list[int] = []

    def record(value: int) -> None:
        committed.append(value)

    store.apply("checkpoint", transaction, record)
    assert committed == [1]


def test_store_requires_strictly_increasing_position_on_apply() -> None:
    store = PreparedStateStore[int]()
    effects: list[int] = []

    def record(value: int) -> None:
        effects.append(value)

    first = fingerprint(1, "first")
    second = fingerprint(1, "second")
    store.prepare("checkpoint-1", first, 1)
    store.apply("checkpoint-1", first, record)

    # prepare allows same position, but apply requires strictly newer position
    store.prepare("checkpoint-2", second, 2)
    with pytest.raises(ValueError, match="is not newer than committed transaction"):
        store.apply("checkpoint-2", second, record)


def test_store_deeply_freezes_cut_mappings() -> None:
    stream_cuts = {"source": {"offsets": [1, 2]}}
    cut = CheckpointCut("cut", 1, stream_cuts=stream_cuts)
    transaction = CheckpointTransactionFingerprint.create(
        permit=CheckpointCommitPermit("attempt", 0, b"token"),
        cut=cut,
        state_digest="digest",
    )
    store = PreparedStateStore[int]()
    store.prepare("checkpoint", transaction, 1)

    stream_cuts["source"]["offsets"].append(3)
    conflicting = CheckpointTransactionFingerprint.create(
        permit=CheckpointCommitPermit("attempt", 0, b"token"),
        cut=cut,
        state_digest="digest",
    )

    def record(_value: int) -> None:
        return None

    with pytest.raises(ValueError, match="does not match prepare"):
        store.apply("checkpoint", conflicting, record)


def test_store_abort_removes_prepared_transaction() -> None:
    store = PreparedStateStore[int]()
    first = fingerprint(1)
    store.prepare("checkpoint-1", first, 100)
    assert store.prepared("checkpoint-1").value == 100

    store.abort("checkpoint-1", first.permit)
    with pytest.raises(RuntimeError, match="requires prepare"):
        store.prepared("checkpoint-1")
