"""Worker-local checkpoint activity observability."""

from __future__ import annotations

from dataclasses import dataclass
import time

from fluxway.common.types.configs import CheckpointActivityPhase


@dataclass(frozen=True, slots=True)
class CheckpointActivitySnapshot:
    phase: CheckpointActivityPhase
    progress_timestamp: float
    checkpoint_id: str | None
    completed_units: int


class CheckpointActivity:
    def __init__(self) -> None:
        self._phase = CheckpointActivityPhase.IDLE
        self._progress_timestamp = time.time()
        self._checkpoint_id: str | None = None
        self._completed_units = 0

    def begin(self, checkpoint_id: str) -> None:
        if not checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        if self._phase not in {
            CheckpointActivityPhase.IDLE,
            CheckpointActivityPhase.PUBLISHED,
            CheckpointActivityPhase.FAILED,
        }:
            raise RuntimeError(f"checkpoint activity is already active: {self._phase}")
        self._checkpoint_id = checkpoint_id
        self._completed_units = 0
        self._phase = CheckpointActivityPhase.CUTTING

    def phase(self, phase: CheckpointActivityPhase) -> None:
        if not isinstance(phase, CheckpointActivityPhase):
            raise TypeError("checkpoint activity phase is invalid")
        if self._checkpoint_id is None:
            raise RuntimeError("checkpoint activity has not begun")
        allowed = {
            CheckpointActivityPhase.CUTTING: {
                CheckpointActivityPhase.CAPTURING,
                CheckpointActivityPhase.FAILED,
            },
            CheckpointActivityPhase.CAPTURING: {
                CheckpointActivityPhase.STAGING,
                CheckpointActivityPhase.FAILED,
            },
            CheckpointActivityPhase.STAGING: {
                CheckpointActivityPhase.COMMITTING,
                CheckpointActivityPhase.FAILED,
            },
            CheckpointActivityPhase.COMMITTING: {
                CheckpointActivityPhase.PUBLISHED,
                CheckpointActivityPhase.FAILED,
            },
        }
        if phase not in allowed.get(self._phase, set()):
            raise RuntimeError(
                f"invalid checkpoint activity transition: {self._phase} -> {phase}"
            )
        self._phase = phase
        self._progress_timestamp = time.time()

    def unit_completed(self) -> None:
        if self._checkpoint_id is None:
            raise RuntimeError("checkpoint activity has not begun")
        if self._phase in {
            CheckpointActivityPhase.IDLE,
            CheckpointActivityPhase.PUBLISHED,
            CheckpointActivityPhase.FAILED,
        }:
            raise RuntimeError(
                f"checkpoint unit cannot complete in phase {self._phase}"
            )
        self._completed_units += 1
        self._progress_timestamp = time.time()

    def published(self) -> None:
        if self._checkpoint_id is None:
            raise RuntimeError("checkpoint activity has not begun")
        self._phase = CheckpointActivityPhase.PUBLISHED
        self._progress_timestamp = time.time()

    def failed(self) -> None:
        if self._checkpoint_id is None:
            raise RuntimeError("checkpoint activity has not begun")
        self._phase = CheckpointActivityPhase.FAILED
        self._progress_timestamp = time.time()

    def snapshot(self) -> CheckpointActivitySnapshot:
        return CheckpointActivitySnapshot(
            phase=self._phase,
            progress_timestamp=self._progress_timestamp,
            checkpoint_id=self._checkpoint_id,
            completed_units=self._completed_units,
        )
