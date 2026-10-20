"""
Drain mechanism strategy and types.

Drain is an execution barrier that limits how many steps a worker can process
ahead of the control plane. It natively separates the execution control from
the persistence mechanism (like Checkpointing).
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class DrainPhase(StrEnum):
    OPEN = "open"
    OBSERVED = "observed"
    DRAINING = "draining"
    DRAINED = "drained"
    SNAPSHOTTING = "snapshotting"
    COMPLETED = "completed"


class DrainEventKind(StrEnum):
    PERIODIC_REQUEST = "periodic_request"
    TERMINAL_OBSERVATION = "terminal_observation"
    DRAIN_COMPLETE = "drain_complete"
    SNAPSHOT_BEGIN = "snapshot_begin"
    CHECKPOINT_COMMITTED = "checkpoint_committed"
    RESTORE = "restore"
    LATE_COMPLETION = "late_completion"


@dataclass(frozen=True, slots=True)
class DrainEvent:
    kind: DrainEventKind
    cut: int | None = None

    def __post_init__(self) -> None:
        if self.cut is not None and (
            isinstance(self.cut, bool) or not isinstance(self.cut, int)
        ):
            raise TypeError("drain event cut must be an integer or None")
        if self.cut is not None and self.cut < 0:
            raise ValueError(f"drain event cut must be non-negative, got {self.cut}")


@dataclass(frozen=True, slots=True)
class DrainAdmission:
    revision: int
    target: int
    terminal_cut: int | None = None

    def __post_init__(self) -> None:
        for name in ("revision", "target"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"admission {name} must be a non-negative integer")
        if self.terminal_cut is not None:
            if type(self.terminal_cut) is not int or self.terminal_cut < 0:
                raise ValueError(
                    "admission terminal cut must be a non-negative integer"
                )
            if self.target > self.terminal_cut:
                raise ValueError("admission target exceeds terminal cut")

    def supersedes(self, previous: "DrainAdmission | None") -> bool:
        if previous is None:
            return True
        if self.revision < previous.revision:
            raise ValueError("stale admission revision")
        if self.revision == previous.revision and self != previous:
            raise ValueError("conflicting admission at the same revision")
        return self != previous


@dataclass(frozen=True, slots=True)
class DrainBoundary:
    phase: DrainPhase = DrainPhase.OPEN
    observed_cut: int | None = None
    admitted_cut: int | None = None
    drained_cut: int | None = None
    committed_cut: int | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        for name in (
            "observed_cut",
            "admitted_cut",
            "drained_cut",
            "committed_cut",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("drain boundary revision must be an integer")
        if self.revision < 0:
            raise ValueError("drain boundary revision must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "observed_cut": self.observed_cut,
            "admitted_cut": self.admitted_cut,
            "drained_cut": self.drained_cut,
            "committed_cut": self.committed_cut,
            "revision": self.revision,
        }

    @classmethod
    def from_committed_cut(cls, committed_cut: int) -> "DrainBoundary":
        """Start a fresh execution epoch at a durable checkpoint position."""
        if isinstance(committed_cut, bool) or not isinstance(committed_cut, int):
            raise TypeError("committed cut must be an integer")
        if committed_cut < 0:
            raise ValueError(f"committed cut must be non-negative, got {committed_cut}")
        return cls(
            phase=DrainPhase.OPEN,
            drained_cut=committed_cut,
            committed_cut=committed_cut,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DrainBoundary":
        if not isinstance(value, Mapping):
            raise TypeError("drain boundary state must be a mapping")
        expected = {
            "phase",
            "observed_cut",
            "admitted_cut",
            "drained_cut",
            "committed_cut",
            "revision",
        }
        if set(value) != expected:
            raise ValueError(
                "drain boundary fields do not match protocol: "
                f"expected={sorted(expected)} actual={sorted(value)}"
            )
        try:
            phase = DrainPhase(value["phase"])
        except ValueError as error:
            raise ValueError(
                f"unknown drain boundary phase {value['phase']!r}"
            ) from error
        return cls(
            phase=phase,
            observed_cut=value["observed_cut"],
            admitted_cut=value["admitted_cut"],
            drained_cut=value["drained_cut"],
            committed_cut=value["committed_cut"],
            revision=value["revision"],
        )


@dataclass(frozen=True, slots=True)
class DrainTransition:
    boundary: DrainBoundary
    dispatch_target: int | None = None
    clamped_completion: int | None = None
    terminal_completed: bool = False


def reduce_drain_boundary(
    boundary: DrainBoundary, event: DrainEvent
) -> DrainTransition:
    """Apply one serialized drain event and return the resulting command."""

    if event.kind is DrainEventKind.LATE_COMPLETION:
        if event.cut is None:
            raise ValueError("late completion requires a cut")
        cap = boundary.observed_cut
        if cap is None:
            cap = boundary.admitted_cut
        return DrainTransition(
            boundary=boundary,
            clamped_completion=(event.cut if cap is None else min(event.cut, cap)),
        )

    if event.kind in (
        DrainEventKind.PERIODIC_REQUEST,
        DrainEventKind.TERMINAL_OBSERVATION,
    ):
        if event.cut is None:
            raise ValueError(f"{event.kind} requires a cut")
        if (
            event.kind is DrainEventKind.PERIODIC_REQUEST
            and boundary.committed_cut is not None
            and event.cut < boundary.committed_cut
        ):
            raise ValueError(
                "periodic drain would move the durable boundary backwards: "
                f"request={event.cut} committed={boundary.committed_cut}"
            )
        observed = boundary.observed_cut
        if event.kind is DrainEventKind.TERMINAL_OBSERVATION:
            if (
                boundary.committed_cut is not None
                and event.cut < boundary.committed_cut
            ):
                raise ValueError(
                    "terminal observation would roll back a published checkpoint: "
                    f"observation={event.cut} committed={boundary.committed_cut}"
                )
            observed = event.cut if observed is None else min(observed, event.cut)
        target = event.cut if observed is None else min(event.cut, observed)
        current = boundary.admitted_cut
        if current is not None and boundary.phase in (
            DrainPhase.DRAINING,
            DrainPhase.DRAINED,
            DrainPhase.SNAPSHOTTING,
        ):
            target = min(current, target)
        changed = current != target or observed != boundary.observed_cut
        phase = boundary.phase
        drained_cut = boundary.drained_cut
        if event.kind is DrainEventKind.TERMINAL_OBSERVATION:
            if target != current:
                drained_cut = None
            if (
                boundary.committed_cut is not None
                and observed == boundary.committed_cut
                and target == boundary.committed_cut
                and boundary.phase is DrainPhase.OPEN
            ):
                phase = DrainPhase.COMPLETED
            elif phase is DrainPhase.OPEN:
                phase = DrainPhase.OBSERVED
        if target != current:
            phase = DrainPhase.DRAINING
        next_boundary = replace(
            boundary,
            phase=phase,
            observed_cut=observed,
            admitted_cut=target,
            drained_cut=drained_cut,
            revision=boundary.revision + int(changed),
        )
        return DrainTransition(
            boundary=next_boundary,
            dispatch_target=target if changed else None,
        )

    if event.kind is DrainEventKind.DRAIN_COMPLETE:
        if event.cut is None:
            raise ValueError("drain completion requires a cut")
        if boundary.admitted_cut is None or event.cut > boundary.admitted_cut:
            raise ValueError(
                "drain completion exceeds admitted target: "
                f"completed={event.cut} admitted={boundary.admitted_cut}"
            )
        if (
            boundary.phase
            in (DrainPhase.DRAINED, DrainPhase.SNAPSHOTTING, DrainPhase.COMPLETED)
            and boundary.drained_cut == event.cut
        ):
            return DrainTransition(boundary=boundary)
        terminal = boundary.observed_cut is not None
        next_boundary = replace(
            boundary,
            phase=DrainPhase.DRAINED,
            drained_cut=event.cut,
            revision=boundary.revision + 1,
        )
        return DrainTransition(boundary=next_boundary, terminal_completed=terminal)

    if event.kind is DrainEventKind.SNAPSHOT_BEGIN:
        if event.cut is None:
            raise ValueError("snapshot begin requires a cut")
        if boundary.drained_cut is None or boundary.drained_cut < event.cut:
            raise ValueError(
                "snapshot begins before drain completion: "
                f"snapshot={event.cut} drained={boundary.drained_cut}"
            )
        if (
            boundary.phase is DrainPhase.SNAPSHOTTING
            and boundary.admitted_cut == event.cut
        ):
            return DrainTransition(boundary=boundary)
        return DrainTransition(
            boundary=replace(
                boundary,
                phase=DrainPhase.SNAPSHOTTING,
                revision=boundary.revision + 1,
            )
        )

    if event.kind is DrainEventKind.CHECKPOINT_COMMITTED:
        if event.cut is None:
            raise ValueError("checkpoint commit requires a cut")
        if boundary.drained_cut is None or boundary.drained_cut < event.cut:
            raise ValueError(
                "checkpoint commit exceeds drained boundary: "
                f"commit={event.cut} drained={boundary.drained_cut}"
            )
        if boundary.committed_cut is not None and event.cut < boundary.committed_cut:
            raise ValueError(
                "checkpoint commit would move durable cut backwards: "
                f"commit={event.cut} committed={boundary.committed_cut}"
            )
        if boundary.committed_cut == event.cut:
            return DrainTransition(boundary=boundary)
        terminal = boundary.observed_cut is not None
        next_boundary = replace(
            boundary,
            phase=DrainPhase.COMPLETED if terminal else DrainPhase.OPEN,
            committed_cut=event.cut,
            revision=boundary.revision + 1,
        )
        return DrainTransition(
            boundary=next_boundary,
            terminal_completed=terminal,
        )

    if event.kind is DrainEventKind.RESTORE:
        if event.cut is None:
            raise ValueError("restore requires a cut")
        if boundary.committed_cut is not None and event.cut < boundary.committed_cut:
            raise ValueError(
                "restore would move durable cut backwards: "
                f"restore={event.cut} committed={boundary.committed_cut}"
            )
        if (
            boundary.committed_cut == event.cut
            and boundary.admitted_cut == event.cut
            and boundary.drained_cut == event.cut
        ):
            return DrainTransition(boundary=boundary)
        return DrainTransition(
            boundary=replace(
                boundary,
                phase=DrainPhase.COMPLETED
                if boundary.observed_cut is not None
                else DrainPhase.OPEN,
                committed_cut=event.cut,
                drained_cut=event.cut,
                admitted_cut=event.cut,
                revision=boundary.revision + 1,
            )
        )

    raise AssertionError(f"unsupported drain event {event.kind!r}")


class DrainStrategy(ABC):
    """
    Abstract strategy for calculating when to drain the pipeline.
    """

    @abstractmethod
    def get_initial_drain_target(self, baseline: int = 0) -> int | None:
        """Get the step target for the very first drain ceiling."""
        pass

    @abstractmethod
    def get_next_drain_target(self, current_completed_step: int) -> int | None:
        """Given the current completed step, calculate the next drain target."""
        pass


class GraphModeDrainStrategy(DrainStrategy):
    """
    Graph mode: drain periodically based on a fixed interval of steps.
    """

    def __init__(self, interval_steps: int):
        if interval_steps < 1:
            raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")
        self.interval_steps = interval_steps

    def get_initial_drain_target(self, baseline: int = 0) -> int | None:
        return baseline + self.interval_steps

    def get_next_drain_target(self, current_completed_step: int) -> int | None:
        return current_completed_step + self.interval_steps
