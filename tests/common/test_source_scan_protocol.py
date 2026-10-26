from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from fluxway.common.ingestion.segment import LayoutUnitDescriptor
from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)


def test_source_binding_identity_checkpoint_round_trip_is_strict() -> None:
    identity = SourceBindingIdentity("config", "source", "finite", "native")
    assert SourceBindingIdentity.from_checkpoint(identity.to_checkpoint()) == identity
    with pytest.raises(ValueError, match="fields do not match"):
        SourceBindingIdentity.from_checkpoint(
            {**identity.to_checkpoint(), "extra": "field"}
        )


def _unit() -> LayoutUnitDescriptor:
    return LayoutUnitDescriptor(
        unit_id="range-0",
        revision_identity="snapshot-1",
        cardinality=2,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key="file-a",
        storage_locator="file-a",
        logical_start=0,
        opaque_token=b"range-token",
    )


@pytest.mark.parametrize(
    ("records", "exhausted", "finite", "expected"),
    [
        (0, False, False, SourceScanStatus.PROGRESS),
        (0, False, True, SourceScanStatus.PROGRESS),
        (0, True, False, SourceScanStatus.IDLE),
        (0, True, True, SourceScanStatus.TERMINAL),
        (2, False, False, SourceScanStatus.READY),
        (2, False, True, SourceScanStatus.READY),
        (2, True, False, SourceScanStatus.READY),
        (2, True, True, SourceScanStatus.TERMINAL),
    ],
)
def test_scan_state_covers_all_boundary_facts(records, exhausted, finite, expected):
    assert (
        SourceScanStatus.from_scan(records=records, exhausted=exhausted, finite=finite)
        is expected
    )


def test_opaque_cursor_validates_canonical_digest() -> None:
    payload = b"cursor"
    cursor = OpaqueSourceCursor(
        payload=payload,
        digest=hashlib.sha256(payload).hexdigest(),
    )

    assert cursor.payload == payload
    assert cursor.digest == hashlib.sha256(payload).hexdigest()

    with pytest.raises(ValueError, match="digest"):
        OpaqueSourceCursor(payload=payload, digest="wrong")


def test_opaque_cursor_checkpoint_round_trip_is_canonical() -> None:
    cursor = OpaqueSourceCursor.from_payload(b"cursor")

    restored = OpaqueSourceCursor.from_checkpoint(cursor.to_checkpoint())

    assert restored == cursor
    assert set(cursor.to_checkpoint()) == {"payload", "digest"}
    with pytest.raises(ValueError, match="fields do not match"):
        OpaqueSourceCursor.from_checkpoint(
            {"payload": cursor.payload.hex(), "digest": cursor.digest, "version": 1}
        )
    with pytest.raises(ValueError, match="does not match"):
        OpaqueSourceCursor.from_checkpoint(
            {"payload": cursor.payload.hex(), "digest": "0" * 64}
        )


def test_source_scan_budget_rejects_nonpositive_and_noninteger_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        SourceScanBudget(
            max_records=0,
            max_units=1,
        )

    with pytest.raises(TypeError, match="integer"):
        SourceScanBudget(
            max_records=True,
            max_units=1,
        )


def test_ready_scan_result_contains_bounded_ordered_records() -> None:
    cursor = OpaqueSourceCursor.from_payload(b"next")
    scan_result = SourceScanResult(
        source_identity="source-a",
        view_identity="snapshot-1",
        schema_digest="schema-digest",
        lineage_digest="lineage-digest",
        units=(_unit(),),
        unit_cursors=tuple(
            OpaqueSourceCursor.from_payload(unit.unit_id.encode())
            for unit in (_unit(),)
        ),
        records=2,
        next_cursor=cursor,
        status=SourceScanStatus.READY,
    )

    scan_result.validate_budget(
        SourceScanBudget(
            max_records=2,
            max_units=1,
        )
    )
    assert scan_result.unit_count == 1
    assert scan_result.record_count == 2

    with pytest.raises(ValueError, match="record budget"):
        scan_result.validate_budget(
            SourceScanBudget(
                max_records=1,
                max_units=1,
            )
        )


def test_idle_scan_result_cannot_contain_records_and_may_advance_cursor() -> None:
    cursor = OpaqueSourceCursor.from_payload(b"same")
    scan_result = SourceScanResult(
        source_identity="source-a",
        view_identity="snapshot-1",
        schema_digest="schema-digest",
        lineage_digest="lineage-digest",
        units=(),
        unit_cursors=(),
        records=0,
        next_cursor=cursor,
        status=SourceScanStatus.IDLE,
    )
    scan_result.validate_transition(cursor)
    assert scan_result.next_cursor == cursor

    with pytest.raises(ValueError, match="IDLE"):
        SourceScanResult(
            source_identity="source-a",
            view_identity="snapshot-1",
            schema_digest="schema-digest",
            lineage_digest="lineage-digest",
            units=(_unit(),),
            unit_cursors=tuple(
                OpaqueSourceCursor.from_payload(unit.unit_id.encode())
                for unit in (_unit(),)
            ),
            records=2,
            next_cursor=cursor,
            status=SourceScanStatus.IDLE,
        )

    advanced_scan_result = SourceScanResult(
        source_identity="source-a",
        view_identity="snapshot-1",
        schema_digest="schema-digest",
        lineage_digest="lineage-digest",
        units=(),
        unit_cursors=(),
        records=0,
        next_cursor=OpaqueSourceCursor.from_payload(b"advanced"),
        status=SourceScanStatus.IDLE,
    )
    advanced_scan_result.validate_transition(cursor)
    assert advanced_scan_result.next_cursor != cursor


def test_progress_scan_result_requires_a_new_cursor_and_no_records() -> None:
    requested = OpaqueSourceCursor.from_payload(b"start")
    result = SourceScanResult(
        source_identity="source-a",
        view_identity="snapshot-1",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(),
        unit_cursors=(),
        records=0,
        next_cursor=OpaqueSourceCursor.from_payload(b"next"),
        status=SourceScanStatus.PROGRESS,
    )
    result.validate_transition(requested)
    with pytest.raises(ValueError, match="advance"):
        result.validate_transition(result.next_cursor)
    with pytest.raises(ValueError, match="PROGRESS"):
        replace(result, units=(_unit(),), unit_cursors=(requested,), records=2)


def test_scan_result_unit_cardinality_must_match_record_count() -> None:
    with pytest.raises(ValueError, match="record count"):
        SourceScanResult(
            source_identity="source-a",
            view_identity="snapshot-1",
            schema_digest="schema-digest",
            lineage_digest="lineage-digest",
            units=(_unit(),),
            unit_cursors=tuple(
                OpaqueSourceCursor.from_payload(unit.unit_id.encode())
                for unit in (_unit(),)
            ),
            records=1,
            next_cursor=OpaqueSourceCursor.from_payload(b"next"),
            status=SourceScanStatus.READY,
        )


@pytest.mark.parametrize("consumed", range(6))
def test_consumption_cursor_selects_ordered_unit_without_source_io(consumed):
    units = (
        replace(_unit(), unit_id="a", cardinality=2),
        replace(_unit(), unit_id="b", cardinality=3),
    )
    origins = tuple(
        OpaqueSourceCursor.from_payload(value) for value in (b"a:0", b"b:0")
    )
    end = OpaqueSourceCursor.from_payload(b"end")
    result = SourceScanResult(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=units,
        unit_cursors=origins,
        records=5,
        next_cursor=end,
        status=SourceScanStatus.READY,
    )
    calls = []

    def advance(cursor, offset):
        calls.append((cursor, offset))
        file_id, start = cursor.payload.split(b":")
        return OpaqueSourceCursor.from_payload(
            file_id + b":" + str(int(start) + offset).encode()
        )

    expected = (b"a:0", b"a:1", b"b:0", b"b:1", b"b:2", b"end")
    assert result.cursor_at(consumed, advance).payload == expected[consumed]
    assert len(calls) == (0 if consumed in (0, 2, 5) else 1)
    with pytest.raises(ValueError, match="exceed"):
        result.cursor_at(6, advance)
    with pytest.raises(ValueError, match="match"):
        replace(result, unit_cursors=origins[:1])
