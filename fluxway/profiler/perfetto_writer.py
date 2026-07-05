from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fluxway.profiler.hierarchy import TraceTrackHierarchy
from fluxway.profiler.record import (
    MAX_PAYLOAD_FIELDS,
    PAYLOAD_START_COLUMN,
    TraceEventKind,
    TraceMetadata,
)


FLUXWAY_TRACE_PACKET_SEQUENCE_ID = 1


class PerfettoTraceWriter:
    def __init__(self) -> None:
        try:
            # pyrefly: ignore [missing-import]
            from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

            # pyrefly: ignore [missing-import]
            from perfetto.trace_builder.proto_builder import StreamingTraceProtoBuilder
        except ImportError as exc:
            raise ImportError(
                "PerfettoTraceWriter requires the optional dependency "
                "`perfetto==0.57.1`."
            ) from exc
        self._proto = proto
        self._builder_cls = StreamingTraceProtoBuilder

    def write(
        self,
        records: np.ndarray,
        path: str | Path,
        *,
        metadata: TraceMetadata,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            builder = self._builder_cls(f)
            emitted_tracks: set[int] = set()
            emitted_descriptors: set[int] = set()
            for row in records:
                self._write_record(
                    builder,
                    row,
                    metadata,
                    emitted_tracks,
                    emitted_descriptors,
                )

    def _write_record(
        self,
        builder: Any,
        row: np.ndarray,
        metadata: TraceMetadata,
        emitted_tracks: set[int],
        emitted_descriptors: set[int],
    ) -> None:
        proto = self._proto
        kind = int(row[0])
        track_uuid = int(row[3])
        name_id = int(row[4])
        if track_uuid not in emitted_tracks:
            self._write_hierarchical_track_descriptor(
                builder,
                track_uuid=track_uuid,
                track_name=metadata.name_for(track_uuid),
                emitted_descriptors=emitted_descriptors,
            )
            emitted_tracks.add(track_uuid)
        payload_fields = metadata.payload_fields_for(name_id)
        payload_values = row[
            PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(payload_fields)
        ]
        if kind == int(TraceEventKind.SLICE_BEGIN):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_SLICE_BEGIN,
                flow_id=int(row[5]),
                payload_fields=payload_fields,
                payload_values=payload_values,
            )
            self._write_track_event(
                builder,
                ts_ns=int(row[1]) + int(row[2]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_SLICE_END,
                flow_id=0,
                payload_fields=payload_fields,
                payload_values=payload_values,
            )
            return
        if kind == int(TraceEventKind.FLOW_END):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_INSTANT,
                terminating_flow_id=int(row[5]),
                payload_fields=payload_fields,
                payload_values=payload_values,
            )
            return
        if kind == int(TraceEventKind.COUNTER):
            self._write_track_event(
                builder,
                ts_ns=int(row[1]),
                track_uuid=track_uuid,
                name=metadata.name_for(name_id),
                event_type=proto.TrackEvent.TYPE_COUNTER,
                counter_value=int(row[PAYLOAD_START_COLUMN]),
                payload_fields=payload_fields,
                payload_values=payload_values,
            )
            return
        self._write_track_event(
            builder,
            ts_ns=int(row[1]),
            track_uuid=track_uuid,
            name=metadata.name_for(name_id),
            event_type=proto.TrackEvent.TYPE_INSTANT,
            flow_id=int(row[5]),
            payload_fields=payload_fields,
            payload_values=payload_values,
        )

    def _write_track_descriptor(
        self,
        builder: Any,
        *,
        track_uuid: int,
        name: str,
        parent_uuid: int = 0,
        sibling_order_rank: int = 0,
    ) -> None:
        packet = builder.create_packet()
        packet.trusted_packet_sequence_id = FLUXWAY_TRACE_PACKET_SEQUENCE_ID
        packet.track_descriptor.uuid = track_uuid
        packet.track_descriptor.parent_uuid = parent_uuid
        packet.track_descriptor.name = name
        packet.track_descriptor.sibling_order_rank = sibling_order_rank
        builder.write_packet(packet)

    def _write_hierarchical_track_descriptor(
        self,
        builder: Any,
        *,
        track_uuid: int,
        track_name: str,
        emitted_descriptors: set[int],
    ) -> None:
        layout = TraceTrackHierarchy.from_track_name(track_name).to_layout(
            leaf_uuid=track_uuid
        )
        for node in layout.nodes:
            if node.uuid in emitted_descriptors:
                continue
            self._write_track_descriptor(
                builder,
                track_uuid=node.uuid,
                name=node.name,
                parent_uuid=node.parent_uuid,
                sibling_order_rank=node.sibling_order_rank,
            )
            emitted_descriptors.add(node.uuid)

    def _write_track_event(
        self,
        builder: Any,
        *,
        ts_ns: int,
        track_uuid: int,
        name: str,
        event_type: int,
        flow_id: int = 0,
        terminating_flow_id: int = 0,
        counter_value: int | None = None,
        payload_fields: tuple[str, ...],
        payload_values: np.ndarray,
    ) -> None:
        if len(payload_fields) > MAX_PAYLOAD_FIELDS:
            raise ValueError(
                f"trace payload has {len(payload_fields)} fields; max={MAX_PAYLOAD_FIELDS}"
            )
        packet = builder.create_packet()
        packet.trusted_packet_sequence_id = FLUXWAY_TRACE_PACKET_SEQUENCE_ID
        packet.timestamp = ts_ns
        event = packet.track_event
        event.track_uuid = track_uuid
        event.name = name
        event.type = event_type
        if counter_value is not None:
            event.counter_value = counter_value
        if flow_id:
            event.flow_ids.append(flow_id)
        if terminating_flow_id:
            event.terminating_flow_ids.append(terminating_flow_id)
        for name, value in zip(payload_fields, payload_values, strict=True):
            annotation = event.debug_annotations.add()
            annotation.name = name
            annotation.int_value = int(value)
        builder.write_packet(packet)
