"""Canonical source plans, transient interval leases, and committed cuts."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fluxway.common.ingestion import (
    SourceWorkAcknowledgement,
    SourceWorkItem,
    SourceWorkLease,
    WorkerLayoutDescriptor,
)
from fluxway.common.ingestion.segment import (
    CompiledSegment,
    SourceWorkAssignment,
    SourceWorkKey,
    _count,
    validate_canonical_work_intervals,
)
from fluxway.common.metadata import metadata_digest
from fluxway.common.types.checkpoint import CheckpointCommitPermit, CheckpointCut
from fluxway.control_plane.checkpoint_transaction import (
    CheckpointTransactionFingerprint,
    PreparedCheckpointTransaction,
    PreparedStateStore,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LogicalWorkEpoch:
    epoch: int
    start_step: int
    end_step: int
    assignments: tuple[SourceWorkAssignment, ...]
    terminal: bool
    digest: str

    def __post_init__(self) -> None:
        for value in (self.epoch, self.start_step, self.end_step):
            _count(value, "logical epoch boundary")
        if self.end_step < self.start_step:
            raise ValueError("logical epoch boundaries are inverted")
        if not isinstance(self.terminal, bool):
            raise TypeError("logical epoch terminal must be boolean")
        if self.end_step == self.start_step and not self.terminal:
            raise ValueError("empty nonterminal observations must not publish an epoch")
        if not isinstance(self.assignments, tuple):
            raise TypeError("logical epoch assignments must be a tuple")
        values = self.assignments
        if any(not isinstance(value, SourceWorkAssignment) for value in values):
            raise TypeError("logical epoch requires SourceWorkAssignment values")
        if any(
            value.epoch != self.epoch
            or not self.start_step <= value.step < self.end_step
            for value in values
        ):
            raise ValueError("source assignment is outside its epoch")
        if len({value.key for value in values}) != len(values):
            raise ValueError("duplicate source assignment")
        if {value.step for value in values} != set(
            range(self.start_step, self.end_step)
        ):
            raise ValueError("logical epoch must cover every step")
        validate_canonical_work_intervals(tuple(value.interval for value in values))
        if not self.digest:
            raise ValueError("logical epoch digest must be non-empty")

    @classmethod
    def create(
        cls,
        *,
        epoch: int,
        start_step: int,
        end_step: int,
        assignments: tuple[SourceWorkAssignment, ...],
        terminal: bool,
        digest: str | None = None,
    ) -> LogicalWorkEpoch:
        if not isinstance(assignments, tuple):
            raise TypeError("logical epoch assignments must be a tuple")
        resolved_digest = (
            digest
            if digest is not None
            else metadata_digest(
                {
                    "epoch": epoch,
                    "start_step": start_step,
                    "end_step": end_step,
                    "terminal": terminal,
                    "assignments": [value.key.assignment_id for value in assignments],
                }
            )
        )
        return cls(
            epoch=epoch,
            start_step=start_step,
            end_step=end_step,
            assignments=assignments,
            terminal=terminal,
            digest=resolved_digest,
        )


@dataclass(frozen=True, slots=True)
class PreparedLogicalWorkPublication:
    epoch: LogicalWorkEpoch
    additions: tuple[tuple[tuple[int, CompiledSegment], SourceWorkItem], ...]
    generation: int


class LogicalWorkLedger:
    def __init__(self, *, max_in_flight_steps: int) -> None:
        _count(max_in_flight_steps, "max_in_flight_steps", positive=True)
        self._max_in_flight_steps = max_in_flight_steps
        self._epochs: dict[int, LogicalWorkEpoch] = {}
        self._bindings: dict[tuple[int, CompiledSegment], SourceWorkItem] = {}
        self._leased: set[tuple[int, CompiledSegment]] = set()
        self._leases: dict[str, SourceWorkLease] = {}
        self._acknowledged: dict[str, SourceWorkAcknowledgement] = {}
        self._remaining: dict[int, int] = {}
        self._candidate_cut = 0
        self._committed_cut = 0
        self._changed = asyncio.Event()
        self._lock = asyncio.Lock()
        self._checkpoint_transactions = PreparedStateStore[dict[str, Any]]()
        self._publication_generation = 0

    @property
    def committed_cut(self) -> int:
        return self._committed_cut

    def _notify_changed(self) -> None:
        changed = self._changed
        self._changed = asyncio.Event()
        changed.set()

    def _prepare_publication(
        self, epoch: LogicalWorkEpoch, bindings: Sequence[WorkerLayoutDescriptor]
    ) -> PreparedLogicalWorkPublication:
        if not isinstance(epoch, LogicalWorkEpoch):
            raise TypeError("logical work publication requires LogicalWorkEpoch")
        values = tuple(bindings)
        if not values or any(
            not isinstance(value, WorkerLayoutDescriptor) for value in values
        ):
            raise TypeError("logical work publication requires worker layouts")
        if len({value.worker_id for value in values}) != len(values):
            raise ValueError("logical work publication has duplicate workers")
        assignments = {value.key: value for value in epoch.assignments}
        coverage: dict[SourceWorkKey, list[CompiledSegment]] = defaultdict(list)
        additions: dict[tuple[int, CompiledSegment], SourceWorkItem] = {}
        for binding in values:
            for segment in binding.segments:
                key = SourceWorkKey(epoch.epoch, segment.step, segment.assignment_id)
                try:
                    assignment = assignments[key]
                except KeyError as error:
                    raise ValueError(
                        "worker binding references unknown assignment"
                    ) from error
                descriptor = assignment.descriptor
                if (binding.source_identity, binding.schema_digest) != (
                    assignment.source_identity,
                    assignment.schema_digest,
                ):
                    raise ValueError(
                        "worker binding source or schema differs from assignment"
                    )
                if (
                    segment.unit_id,
                    segment.revision_identity,
                    segment.locality_key,
                    segment.physical_identity,
                    segment.cardinality_metric,
                    segment.opaque_token,
                ) != (
                    descriptor.unit_id,
                    descriptor.revision_identity,
                    descriptor.locality_key,
                    descriptor.physical_identity,
                    descriptor.cardinality_metric,
                    descriptor.opaque_token,
                ):
                    raise ValueError(
                        "worker segment descriptor differs from assignment"
                    )
                item = SourceWorkItem(
                    assignment, segment.logical_start, segment.logical_end
                )
                identity = (epoch.epoch, segment)
                if identity in additions:
                    raise ValueError("duplicate worker segment")
                additions[identity] = item
                coverage[key].append(segment)
        for key, assignment in assignments.items():
            cursor = assignment.interval.logical_start
            for segment in sorted(coverage[key], key=lambda value: value.logical_start):
                if segment.logical_start != cursor:
                    raise ValueError(
                        "assignment binding has overlapping or missing intervals"
                    )
                cursor = segment.logical_end
            if cursor != assignment.interval.logical_end:
                raise ValueError("assignment binding does not cover its interval")
        prior = self._epochs.get(epoch.epoch)
        if prior is not None:
            existing = {
                key: value
                for key, value in self._bindings.items()
                if key[0] == epoch.epoch
            }
            if prior != epoch or existing != additions:
                raise ValueError("logical epoch conflicts with prior publication")
            return PreparedLogicalWorkPublication(
                epoch, tuple(additions.items()), self._publication_generation
            )
        previous = self._epochs[max(self._epochs)] if self._epochs else None
        if epoch.epoch != len(self._epochs) or epoch.start_step != (
            0 if previous is None else previous.end_step
        ):
            raise ValueError("logical epochs and step windows must be contiguous")
        if previous is not None and previous.terminal:
            raise ValueError("cannot publish after a terminal epoch")
        return PreparedLogicalWorkPublication(
            epoch, tuple(additions.items()), self._publication_generation
        )

    def _apply_publication(self, prepared: PreparedLogicalWorkPublication) -> None:
        if prepared.generation != self._publication_generation:
            raise RuntimeError("logical work publication is stale")
        if prepared.epoch.epoch in self._epochs:
            return
        additions = dict(prepared.additions)
        self._epochs[prepared.epoch.epoch] = prepared.epoch
        self._bindings.update(additions)
        for item in additions.values():
            self._remaining[item.key.step] = self._remaining.get(item.key.step, 0) + 1
        self._publication_generation += 1
        self._notify_changed()

    def prepare_publication(
        self, epoch: LogicalWorkEpoch, bindings: Sequence[WorkerLayoutDescriptor]
    ) -> PreparedLogicalWorkPublication:
        """Validate a complete epoch without mutating ledger state."""
        return self._prepare_publication(epoch, bindings)

    def apply_publication(self, prepared: PreparedLogicalWorkPublication) -> None:
        if not isinstance(prepared, PreparedLogicalWorkPublication):
            raise TypeError("logical work publication requires a prepared publication")
        self._apply_publication(prepared)

    async def publish(
        self, epoch: LogicalWorkEpoch, bindings: Sequence[WorkerLayoutDescriptor]
    ) -> None:
        async with self._lock:
            prepared = self._prepare_publication(epoch, bindings)
            self._apply_publication(prepared)

    async def acquire(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]:
        return await self._acquire(epoch, segments, available_prefix=False)

    async def acquire_available(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]:
        return await self._acquire(epoch, segments, available_prefix=True)

    async def _acquire(
        self, epoch: int, segments: Sequence[CompiledSegment], *, available_prefix: bool
    ) -> tuple[SourceWorkLease, ...]:
        _count(epoch, "source acquisition epoch")
        values = tuple(segments)
        if not values or any(
            not isinstance(value, CompiledSegment) for value in values
        ):
            raise TypeError(
                "source acquisition requires nonempty CompiledSegment values"
            )
        identities = tuple((epoch, value) for value in values)
        if len(set(identities)) != len(identities):
            raise ValueError("source acquisition contains duplicate segments")
        if (
            not available_prefix
            and max(value.step for value in values)
            - min(value.step for value in values)
            >= self._max_in_flight_steps
        ):
            raise ValueError("source acquisition batch exceeds the in-flight window")
        while True:
            async with self._lock:
                items = tuple(self._bindings[key] for key in identities)
                if any(item.key.step < self._committed_cut for item in items):
                    raise RuntimeError("source work is below the committed cut")
                if any(key in self._leased for key in identities):
                    raise RuntimeError("source interval is already leased")
                limit = self._committed_cut + self._max_in_flight_steps
                eligible = tuple(
                    (key, item)
                    for key, item in zip(identities, items, strict=True)
                    if item.key.step < limit
                )
                if eligible and (available_prefix or len(eligible) == len(items)):
                    leases = tuple(
                        SourceWorkLease(item, uuid4().hex) for _, item in eligible
                    )
                    self._leased.update(key for key, _ in eligible)
                    self._leases.update((lease.lease_id, lease) for lease in leases)
                    return leases
                changed = self._changed
            await changed.wait()

    async def acknowledge(
        self, acknowledgements: Sequence[SourceWorkAcknowledgement]
    ) -> None:
        values = tuple(acknowledgements)
        if any(not isinstance(value, SourceWorkAcknowledgement) for value in values):
            raise TypeError("source acknowledgements must be typed")
        async with self._lock:
            additions: dict[str, SourceWorkAcknowledgement] = {}
            for value in values:
                lease = self._leases.get(value.lease_id)
                if lease is None:
                    raise RuntimeError(
                        "source acknowledgement has a stale or unknown lease"
                    )
                if lease.item.key != value.key:
                    raise ValueError("source acknowledgement conflicts with its lease")
                prior = self._acknowledged.get(value.lease_id)
                if prior is not None and prior != value:
                    raise ValueError("source acknowledgement conflicts")
                if prior is None:
                    additions[value.lease_id] = value
            self._acknowledged.update(additions)
            for value in additions.values():
                self._remaining[value.key.step] -= 1
            while self._remaining.get(self._candidate_cut) == 0:
                self._candidate_cut += 1
            if additions:
                self._notify_changed()

    async def candidate_cut(self) -> int:
        return self._candidate_cut

    async def wait_for_candidate_cut(self, target: int) -> int:
        _count(target, "source candidate target")
        while self._candidate_cut < target:
            await self._changed.wait()
        return self._candidate_cut

    def _validate_cut(self, cut: int) -> None:
        _count(cut, "source committed cut")
        if not self._committed_cut <= cut <= self._candidate_cut:
            raise ValueError(
                "source cut is outside the committed and candidate frontier"
            )

    async def commit_cut(self, cut: int) -> None:
        self._commit_cut(cut)

    def _commit_cut(self, cut: int) -> None:
        self._validate_cut(cut)
        self._committed_cut = cut
        for lease_id, lease in tuple(self._leases.items()):
            if lease.item.key.step < cut:
                self._leases.pop(lease_id)
                self._acknowledged.pop(lease_id, None)
        self._leased = {key for key in self._leased if key[1].step >= cut}
        self._remaining = {
            step: count for step, count in self._remaining.items() if step >= cut
        }
        self._notify_changed()

    def checkpoint_state(self, *, committed_cut: int | None = None) -> dict[str, Any]:
        target = self._committed_cut if committed_cut is None else committed_cut
        self._validate_cut(target)
        return {
            "version": 3,
            "committed_cut": target,
            "epochs": [[epoch.epoch, epoch.digest] for epoch in self._epochs.values()],
        }

    def prepare_checkpoint(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        state_digest: str,
        state: Mapping[str, Any],
    ) -> PreparedCheckpointTransaction[dict[str, Any]]:
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("source checkpoint requires a positioned cut")
        expected = self.checkpoint_state(committed_cut=cut.position)
        if not isinstance(state, Mapping) or dict(state) != expected:
            raise ValueError(
                "source checkpoint state does not match current plan and cut"
            )
        if metadata_digest(expected) != state_digest:
            raise ValueError("source checkpoint digest does not match state")
        fingerprint = CheckpointTransactionFingerprint.create(
            permit=permit, cut=cut, state_digest=state_digest
        )
        self._checkpoint_transactions.prepare(checkpoint_id, fingerprint, expected)
        return self._checkpoint_transactions.prepared(checkpoint_id)

    def apply_prepared(
        self, prepared: PreparedCheckpointTransaction[dict[str, Any]]
    ) -> None:
        def commit(image: dict[str, Any]) -> None:
            self._commit_cut(image["committed_cut"])

        self._checkpoint_transactions.apply(
            prepared.checkpoint_id, prepared.fingerprint, commit
        )

    def abort_prepared(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self._checkpoint_transactions.abort(checkpoint_id, permit)

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "version",
            "committed_cut",
            "epochs",
        }:
            raise ValueError("source checkpoint fields do not match")
        cut = state["committed_cut"]
        _count(cut, "source restore cut")
        if state["version"] != 3 or state["epochs"] != [
            [epoch.epoch, epoch.digest] for epoch in self._epochs.values()
        ]:
            raise ValueError("source checkpoint does not match rebuilt immutable plans")
        if cut > max((epoch.end_step for epoch in self._epochs.values()), default=0):
            raise ValueError("source restore cut exceeds published plans")
        self._remaining.clear()
        for item in self._bindings.values():
            if item.key.step >= cut:
                self._remaining[item.key.step] = (
                    self._remaining.get(item.key.step, 0) + 1
                )
        self._committed_cut = self._candidate_cut = cut
        self._leases.clear()
        self._leased.clear()
        self._acknowledged.clear()
        self._checkpoint_transactions = PreparedStateStore()
        self._notify_changed()
