"""Bounded replay metadata for topology-independent ordered control streams."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from fluxway.control_plane.ledger_client import ControlRecord, StreamProgress


ControlKey = tuple[str, int]
AppendPlan = tuple[
    dict[ControlKey, ControlRecord],
    dict[str, int],
    set[str],
    dict[str, int],
]


class ControlLedger:
    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("control ledger capacity must be an integer")
        if capacity <= 0:
            raise ValueError("control ledger capacity must be positive")
        self._capacity = capacity
        self._streams: set[str] = set()
        self._durable_cuts: dict[str, int] = {}
        self._published_ends: dict[str, int] = {}
        self._records: dict[ControlKey, ControlRecord] = {}
        self._retained_counts: dict[str, int] = {}
        self._received: set[ControlKey] = set()
        self._terminal_ends: dict[str, int] = {}
        self._sealed_streams: set[str] = set()
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._streams))

    @property
    def terminal_ends(self) -> dict[str, int]:
        return dict(self._terminal_ends)

    @staticmethod
    def _validate_stream_id(stream_id: str) -> None:
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("control stream_id must be non-empty")

    @staticmethod
    def _validate_position(position: int, *, name: str = "position") -> None:
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError(f"control {name} must be an integer")
        if position < 0:
            raise ValueError(f"control {name} must be non-negative")

    def _durable_cut(self, stream_id: str) -> int:
        return self._durable_cuts.get(stream_id, 0)

    def _published_end(self, stream_id: str) -> int:
        return self._published_ends.get(stream_id, self._durable_cut(stream_id))

    def _plan_append(self, records: Sequence[ControlRecord]) -> AppendPlan:
        new_records: dict[ControlKey, ControlRecord] = {}
        published_ends: dict[str, int] = {}
        streams: set[str] = set()
        retained_increments: dict[str, int] = {}
        for record in records:
            if not isinstance(record, ControlRecord):
                raise TypeError("control append accepts only ControlRecord values")
            stream_id = record.stream_id
            position = record.position
            if stream_id in self._terminal_ends:
                raise RuntimeError(f"control stream is closed: {stream_id}")
            durable_cut = self._durable_cut(stream_id)
            published_end = published_ends.get(
                stream_id, self._published_end(stream_id)
            )
            if position < durable_cut:
                raise RuntimeError(
                    "control record precedes durable cut: "
                    f"stream_id={stream_id!r} position={position} cut={durable_cut}"
                )
            key = record.key
            if position < published_end:
                existing = new_records.get(key, self._records.get(key))
                if existing != record:
                    raise ValueError(
                        "control position has conflicting payload: "
                        f"stream_id={stream_id!r} position={position}"
                    )
                continue
            if position != published_end:
                raise ValueError(
                    "control append would create a gap: "
                    f"stream_id={stream_id!r} expected={published_end} "
                    f"actual={position}"
                )
            new_records[key] = record
            published_ends[stream_id] = published_end + 1
            streams.add(stream_id)
            retained_increments[stream_id] = retained_increments.get(stream_id, 0) + 1
        return new_records, published_ends, streams, retained_increments

    async def append(self, records: Sequence[ControlRecord]) -> None:
        records = tuple(records)
        async with self._condition:
            while True:
                candidate = self._append_candidate(records)
                if candidate is not None:
                    self._commit_append(candidate)
                    return
                await self._condition.wait()

    async def try_append(self, records: Sequence[ControlRecord]) -> bool:
        records = tuple(records)
        async with self._condition:
            candidate = self._append_candidate(records)
            if candidate is None:
                return False
            self._commit_append(candidate)
            return True

    def _append_candidate(self, records: Sequence[ControlRecord]) -> AppendPlan | None:
        candidate = self._plan_append(records)
        _new_records, _published_ends, _streams, retained_increments = candidate
        oversized = {
            stream_id: count
            for stream_id, count in retained_increments.items()
            if count > self._capacity
        }
        if oversized:
            raise ValueError(
                "control append batch exceeds ledger capacity: "
                f"records={oversized} capacity_per_stream={self._capacity}"
            )
        if any(
            self._retained_counts.get(stream_id, 0) + increment > self._capacity
            for stream_id, increment in retained_increments.items()
        ):
            return None
        return candidate

    def _commit_append(
        self,
        candidate: AppendPlan,
    ) -> None:
        records, published_ends, streams, retained_increments = candidate
        self._records.update(records)
        self._published_ends.update(published_ends)
        self._streams.update(streams)
        for stream_id, increment in retained_increments.items():
            self._retained_counts[stream_id] = (
                self._retained_counts.get(stream_id, 0) + increment
            )
        self._condition.notify_all()

    async def receive(self, stream_id: str, position: int) -> ControlRecord | None:
        self._validate_stream_id(stream_id)
        self._validate_position(position)
        async with self._condition:
            self._streams.add(stream_id)
            while True:
                durable_cut = self._durable_cut(stream_id)
                if position < durable_cut:
                    raise RuntimeError(
                        "control receive precedes durable cut: "
                        f"stream_id={stream_id!r} position={position} cut={durable_cut}"
                    )
                key = (stream_id, position)
                record = self._records.get(key)
                if record is not None:
                    self._received.add(key)
                    return record
                published_end = self._published_end(stream_id)
                if position < published_end:
                    raise RuntimeError(
                        "control ledger retained interval is not contiguous: "
                        f"stream_id={stream_id!r} position={position}"
                    )
                terminal_end = self._terminal_ends.get(stream_id)
                if (
                    terminal_end is not None
                    and stream_id not in self._sealed_streams
                    and position >= terminal_end
                ):
                    return None
                await self._condition.wait()

    async def seal(self, stream_id: str) -> None:
        self._validate_stream_id(stream_id)
        async with self._condition:
            self._seal_locked(stream_id)
            self._sealed_streams.add(stream_id)

    def _seal_locked(self, stream_id: str) -> None:
        self._streams.add(stream_id)
        published_end = self._published_end(stream_id)
        terminal_end = self._terminal_ends.get(stream_id)
        if terminal_end is not None and terminal_end != published_end:
            raise RuntimeError(
                "control terminal end conflicts with published end: "
                f"stream_id={stream_id!r} terminal={terminal_end} "
                f"published={published_end}"
            )
        self._terminal_ends[stream_id] = published_end

    async def progress(self, stream_id: str) -> StreamProgress:
        self._validate_stream_id(stream_id)
        async with self._condition:
            self._streams.add(stream_id)
            return StreamProgress(
                durable_cut=self._durable_cut(stream_id),
                published_end=self._published_end(stream_id),
                terminal_end=self._terminal_ends.get(stream_id),
            )

    def checkpoint_state(self, *, cuts: Mapping[str, int]) -> dict[str, Any]:
        if not isinstance(cuts, Mapping):
            raise TypeError("control checkpoint cuts must be a mapping")
        if set(cuts) != self._streams:
            raise ValueError(
                "control checkpoint cuts do not match streams: "
                f"expected={sorted(self._streams)} actual={sorted(cuts)}"
            )
        normalized_cuts: dict[str, int] = {}
        for stream_id, cut in cuts.items():
            self._validate_stream_id(stream_id)
            self._validate_position(cut, name="checkpoint cut")
            durable_cut = self._durable_cut(stream_id)
            published_end = self._published_end(stream_id)
            if not durable_cut <= cut <= published_end:
                raise ValueError(
                    "control checkpoint cut is outside the published interval: "
                    f"stream_id={stream_id!r} durable={durable_cut} "
                    f"cut={cut} published={published_end}"
                )
            missing = [
                position
                for position in range(durable_cut, cut)
                if (stream_id, position) not in self._received
            ]
            if missing:
                raise RuntimeError(
                    "control checkpoint contains positions that were not received: "
                    f"stream_id={stream_id!r} positions={missing}"
                )
            normalized_cuts[stream_id] = cut

        replay = [
            record.to_dict()
            for key, record in sorted(self._records.items())
            if key[1] >= normalized_cuts[key[0]]
        ]
        return {
            "durable_cuts": dict(sorted(normalized_cuts.items())),
            "published_ends": {
                stream_id: self._published_end(stream_id)
                for stream_id in sorted(self._streams)
            },
            "records": replay,
            "terminal_ends": dict(sorted(self._terminal_ends.items())),
            "sealed_streams": sorted(self._sealed_streams),
        }

    def _validated_checkpoint(
        self, state: Mapping[str, Any]
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[ControlKey, ControlRecord],
        dict[str, int],
        set[str],
    ]:
        if not isinstance(state, Mapping):
            raise TypeError("control checkpoint state must be a mapping")
        expected_fields = {
            "durable_cuts",
            "published_ends",
            "records",
            "terminal_ends",
            "sealed_streams",
        }
        if set(state) != expected_fields:
            raise ValueError(
                "control checkpoint fields do not match: "
                f"expected={sorted(expected_fields)} actual={sorted(state)}"
            )
        durable_cuts = state["durable_cuts"]
        published_ends = state["published_ends"]
        terminal_ends = state["terminal_ends"]
        serialized_records = state["records"]
        sealed_streams = state["sealed_streams"]
        if not isinstance(durable_cuts, Mapping) or not isinstance(
            published_ends, Mapping
        ):
            raise TypeError("control checkpoint cuts and ends must be mappings")
        if set(durable_cuts) != set(published_ends):
            raise ValueError(
                "control checkpoint cuts and ends identify different streams"
            )
        normalized_cuts: dict[str, int] = {}
        normalized_ends: dict[str, int] = {}
        for stream_id in durable_cuts:
            self._validate_stream_id(stream_id)
            cut = durable_cuts[stream_id]
            end = published_ends[stream_id]
            self._validate_position(cut, name="durable cut")
            self._validate_position(end, name="published end")
            if cut > end:
                raise ValueError("control durable cut exceeds published end")
            normalized_cuts[stream_id] = cut
            normalized_ends[stream_id] = end
        if not isinstance(terminal_ends, Mapping):
            raise TypeError("control checkpoint terminal ends must be a mapping")
        normalized_terminal: dict[str, int] = {}
        for stream_id, terminal_end in terminal_ends.items():
            if stream_id not in normalized_ends:
                raise ValueError(
                    "control terminal stream is absent from published ends"
                )
            self._validate_position(terminal_end, name="terminal end")
            if terminal_end != normalized_ends[stream_id]:
                raise ValueError("control terminal end must equal published end")
            normalized_terminal[stream_id] = terminal_end
        if not isinstance(sealed_streams, list) or any(
            not isinstance(stream_id, str) for stream_id in sealed_streams
        ):
            raise TypeError("control checkpoint sealed_streams must be a string list")
        normalized_sealed = set(sealed_streams)
        if len(normalized_sealed) != len(sealed_streams):
            raise ValueError("control checkpoint contains duplicate sealed streams")
        if not normalized_sealed <= set(normalized_terminal):
            raise ValueError("sealed control streams must be terminal")
        if not isinstance(serialized_records, list):
            raise TypeError("control checkpoint records must be a list")
        records: dict[ControlKey, ControlRecord] = {}
        for serialized in serialized_records:
            record = ControlRecord.from_mapping(serialized)
            if record.stream_id not in normalized_cuts:
                raise ValueError("control record references an unknown stream")
            cut = normalized_cuts[record.stream_id]
            end = normalized_ends[record.stream_id]
            if not cut <= record.position < end:
                raise ValueError("control record is outside the replay interval")
            if record.key in records:
                raise ValueError("control checkpoint contains a duplicate position")
            records[record.key] = record
        expected_keys = {
            (stream_id, position)
            for stream_id, cut in normalized_cuts.items()
            for position in range(cut, normalized_ends[stream_id])
        }
        if set(records) != expected_keys:
            raise ValueError("control checkpoint replay interval is not contiguous")
        retained_by_stream: dict[str, int] = {}
        for stream_id, _position in records:
            retained_by_stream[stream_id] = retained_by_stream.get(stream_id, 0) + 1
        if any(count > self._capacity for count in retained_by_stream.values()):
            raise ValueError("control checkpoint exceeds per-stream ledger capacity")
        return (
            normalized_cuts,
            normalized_ends,
            records,
            normalized_terminal,
            normalized_sealed,
        )

    async def commit_checkpoint(self, state: Mapping[str, Any]) -> None:
        cuts, ends, records, terminal, sealed = self._validated_checkpoint(state)
        async with self._condition:
            current_ends = {
                stream_id: self._published_end(stream_id) for stream_id in self._streams
            }
            if (
                ends != current_ends
                or terminal != self._terminal_ends
                or sealed != self._sealed_streams
            ):
                raise RuntimeError(
                    "control ledger changed after checkpoint preparation"
                )
            expected_records = {
                key: record
                for key, record in self._records.items()
                if key[1] >= cuts[key[0]]
            }
            if records != expected_records:
                raise RuntimeError("control replay records changed after preparation")
            self._durable_cuts = cuts
            self._records = records
            self._retained_counts = {}
            for stream_id, _position in records:
                self._retained_counts[stream_id] = (
                    self._retained_counts.get(stream_id, 0) + 1
                )
            self._received = {key for key in self._received if key[1] >= cuts[key[0]]}
            self._condition.notify_all()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if self._streams or self._records or self._received or self._terminal_ends:
            raise RuntimeError("cannot restore a non-empty control ledger")
        cuts, ends, records, terminal, sealed = self._validated_checkpoint(state)
        self._streams = set(cuts)
        self._durable_cuts = cuts
        self._published_ends = ends
        self._records = records
        for stream_id, _position in records:
            self._retained_counts[stream_id] = (
                self._retained_counts.get(stream_id, 0) + 1
            )
        self._terminal_ends = terminal
        self._sealed_streams = sealed
