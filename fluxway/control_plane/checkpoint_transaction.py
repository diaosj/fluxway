from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from fluxway.common.types import CheckpointCommitPermit, CheckpointCut


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointTransactionFingerprint:
    permit: CheckpointCommitPermit
    cut: CheckpointCut
    state_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.permit, CheckpointCommitPermit):
            raise TypeError(
                "checkpoint fingerprint permit must be CheckpointCommitPermit"
            )
        if not isinstance(self.cut, CheckpointCut):
            raise TypeError("checkpoint fingerprint cut must be CheckpointCut")
        if not isinstance(self.state_digest, str) or not self.state_digest:
            raise ValueError("checkpoint fingerprint state_digest must be non-empty")

    @classmethod
    def create(
        cls,
        *,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        state_digest: str,
    ) -> CheckpointTransactionFingerprint:
        return cls(
            permit=permit,
            cut=cut.frozen_copy(),
            state_digest=state_digest,
        )


@dataclass(frozen=True, slots=True)
class PreparedCheckpointTransaction[PreparedValue]:
    checkpoint_id: str
    fingerprint: CheckpointTransactionFingerprint
    value: PreparedValue
    prepare_ordinal: int


class PreparedStateStore[PreparedValue]:
    def __init__(self) -> None:
        self._prepared: dict[str, PreparedCheckpointTransaction[PreparedValue]] = {}
        self._latest_committed: (
            tuple[str, CheckpointTransactionFingerprint, int] | None
        ) = None
        self._next_prepare_ordinal = 0

    def prepare(
        self,
        checkpoint_id: str,
        fingerprint: CheckpointTransactionFingerprint,
        value: PreparedValue,
    ) -> None:
        committed = self._latest_committed
        if committed is not None and committed[0] == checkpoint_id:
            if committed[1] != fingerprint:
                raise ValueError(
                    "checkpoint transaction prepare conflicts with committed transaction"
                )
            return
        if committed is not None and self._position(fingerprint) < self._position(
            committed[1]
        ):
            raise ValueError(
                "checkpoint transaction prepare is not newer than committed transaction"
            )
        prepared = self._prepared.get(checkpoint_id)
        if prepared is not None:
            if prepared.fingerprint != fingerprint or prepared.value != value:
                raise ValueError("checkpoint transaction prepare conflicts")
            return
        self._prepared[checkpoint_id] = PreparedCheckpointTransaction(
            checkpoint_id, fingerprint, value, self._next_prepare_ordinal
        )
        self._next_prepare_ordinal += 1

    def prepared(
        self, checkpoint_id: str
    ) -> PreparedCheckpointTransaction[PreparedValue]:
        prepared = self._prepared.get(checkpoint_id)
        if prepared is None:
            committed = self._latest_committed
            if committed is not None and committed[0] == checkpoint_id:
                raise RuntimeError("checkpoint transaction is already applied")
            raise RuntimeError("checkpoint transaction requires prepare")
        return PreparedCheckpointTransaction(
            prepared.checkpoint_id,
            prepared.fingerprint,
            copy.deepcopy(prepared.value),
            prepared.prepare_ordinal,
        )

    def apply(
        self,
        checkpoint_id: str,
        fingerprint: CheckpointTransactionFingerprint,
        effect: Callable[[PreparedValue], None],
    ) -> None:
        """Apply a validated image synchronously and record it as committed."""
        committed = self._latest_committed
        if committed is not None and committed[0] == checkpoint_id:
            if committed[1] != fingerprint:
                raise ValueError(
                    "checkpoint transaction apply does not match committed transaction"
                )
            return
        prepared = self.prepared(checkpoint_id)
        if prepared.fingerprint != fingerprint:
            raise ValueError("checkpoint transaction apply does not match prepare")
        if committed is not None and self._position(fingerprint) <= self._position(
            committed[1]
        ):
            raise ValueError(
                "checkpoint transaction apply is not newer than committed transaction"
            )
        effect(prepared.value)
        self._prepared.pop(checkpoint_id)
        self._latest_committed = (checkpoint_id, fingerprint, prepared.prepare_ordinal)

    def abort(self, checkpoint_id: str, permit: CheckpointCommitPermit) -> None:
        prepared = self._prepared.get(checkpoint_id)
        if prepared is None:
            return
        if prepared.fingerprint.permit != permit:
            raise ValueError("checkpoint transaction abort does not match prepare")
        self._prepared.pop(checkpoint_id)

    @staticmethod
    def _position(fingerprint: CheckpointTransactionFingerprint) -> int:
        position = fingerprint.cut.position
        if position is None:
            raise ValueError("checkpoint transaction requires positioned cut")
        return position
