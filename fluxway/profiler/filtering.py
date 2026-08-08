from __future__ import annotations

import numpy as np

from fluxway.profiler.record import TraceMetadata


STEADY_STATE_EVENT_NAMES = frozenset(
    {
        "controller.initialize",
        "controller.completion",
        "controller.reclaim",
        "driver.initialize",
        "driver.start_workers",
        "spmd_manager.placement_group_create",
        "spmd_manager.driver_start",
        "spmd_manager.driver_stop",
    }
)


def filter_steady_state_records(
    records: np.ndarray,
    metadata: TraceMetadata,
) -> np.ndarray:
    if records.ndim != 2:
        raise ValueError(f"records must be a 2D array, got ndim={records.ndim}.")
    if len(records) == 0:
        return records
    retention = np.ones((len(records),), dtype=bool)
    for index, row in enumerate(records):
        track_name = metadata.name_for(int(row[3]))
        event_name = metadata.name_for(int(row[4]))
        if "/lifecycle" in track_name or event_name in STEADY_STATE_EVENT_NAMES:
            retention[index] = False
    return records[retention]
