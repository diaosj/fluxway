from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from fluxway.profiler.record import ProfilerSnapshot, TraceMetadata


def merge_profiler_snapshots(
    snapshots: Iterable[ProfilerSnapshot | None],
    *,
    process_name: str,
) -> ProfilerSnapshot | None:
    present = [snapshot for snapshot in snapshots if snapshot is not None]
    if not present:
        return None

    records: list[np.ndarray] = []
    names_by_id: dict[int, str] = {}
    payload_fields_by_name_id: dict[int, tuple[str, ...]] = {}
    clock_uncertainty_ns = 0

    for snapshot in present:
        metadata = snapshot["metadata"]
        _merge_names(names_by_id, metadata.names_by_id)
        _merge_payload_fields(
            payload_fields_by_name_id,
            metadata.payload_fields_by_name_id,
        )
        records_array = snapshot["records"].copy()
        records_array[:, 1] += snapshot["monotonic_to_epoch_offset_ns"]
        records.append(records_array)
        clock_uncertainty_ns = max(
            clock_uncertainty_ns,
            snapshot["clock_uncertainty_ns"],
        )

    merged_records = np.vstack(records)
    return {
        "records": merged_records,
        "metadata": TraceMetadata(
            names_by_id=names_by_id,
            payload_fields_by_name_id=payload_fields_by_name_id,
        ),
        "process_name": process_name,
        "monotonic_to_epoch_offset_ns": 0,
        "clock_uncertainty_ns": clock_uncertainty_ns,
    }


def _merge_names(target: dict[int, str], incoming: Mapping[int, str]) -> None:
    for ident, name in incoming.items():
        existing = target.get(ident)
        if existing is not None and existing != name:
            raise ValueError(
                f"profiler metadata name id conflict for id={ident}: "
                f"{existing!r} != {name!r}."
            )
        target[ident] = name


def _merge_payload_fields(
    target: dict[int, tuple[str, ...]],
    incoming: Mapping[int, tuple[str, ...]],
) -> None:
    for ident, fields in incoming.items():
        existing = target.get(ident)
        if existing is not None and existing != fields:
            raise ValueError(
                f"profiler payload schema conflict for name_id={ident}: "
                f"{existing!r} != {fields!r}."
            )
        target[ident] = fields
