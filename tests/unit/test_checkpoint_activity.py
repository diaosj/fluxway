from __future__ import annotations

import time

import pytest

from fluxway.spmd.worker.checkpoint_activity import (
    CheckpointActivity,
    CheckpointActivityPhase,
)


def test_checkpoint_activity_tracks_progress_without_training_progress() -> None:
    activity = CheckpointActivity()
    activity.begin("step-40")
    assert activity.snapshot().phase is CheckpointActivityPhase.CUTTING
    activity.phase(CheckpointActivityPhase.CAPTURING)
    before = activity.snapshot().progress_timestamp
    time.sleep(0.001)
    activity.unit_completed()
    snapshot = activity.snapshot()
    assert snapshot.phase is CheckpointActivityPhase.CAPTURING
    assert snapshot.completed_units == 1
    assert snapshot.progress_timestamp > before


def test_checkpoint_activity_rejects_unit_outside_active_phase() -> None:
    activity = CheckpointActivity()
    with pytest.raises(RuntimeError, match="has not begun"):
        activity.unit_completed()


def test_checkpoint_activity_failure_is_terminal_until_next_begin() -> None:
    activity = CheckpointActivity()
    activity.begin("step-80")
    activity.failed()
    assert activity.snapshot().phase is CheckpointActivityPhase.FAILED
    activity.begin("step-100")
    assert activity.snapshot().phase is CheckpointActivityPhase.CUTTING
