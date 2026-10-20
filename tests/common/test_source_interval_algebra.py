from __future__ import annotations

import pytest

from fluxway.common.ingestion import LayoutUnitDescriptor


def _interval(start, count):
    return LayoutUnitDescriptor(
        unit_id="origin",
        revision_identity="snapshot",
        cardinality=count,
        cardinality_metric="rows",
        split_policy="SplittableSpan",
        locality_key="file",
        physical_identity="file",
        logical_start=start,
        opaque_token=b"original-source-range",
    )


@pytest.mark.parametrize("start", [0, 10, 2**32, 2**63])
def test_slice_composition_is_independent_of_window_boundaries(start):
    for size in range(1, 10):
        original = _interval(start, size)
        assert original.slice(0, size) is original
        for offset in range(size):
            for count in range(1, size - offset + 1):
                outer = original.slice(offset, count)
                assert outer.opaque_token is original.opaque_token
                assert outer.physical_identity == original.physical_identity
                for inner_offset in range(count):
                    for inner_count in range(1, count - inner_offset + 1):
                        assert outer.slice(inner_offset, inner_count) == original.slice(
                            offset + inner_offset, inner_count
                        )


@pytest.mark.parametrize("start", [0, 10, 2**63])
def test_slices_partition_the_original_records_exactly(start):
    for size in range(2, 16):
        original = _interval(start, size)
        for split in range(1, size):
            left = original.slice(0, split)
            right = original.slice(split, size - split)
            assert left.logical_start + left.cardinality == right.logical_start
            rows = [
                row
                for unit in (left, right)
                for row in range(
                    unit.logical_start, unit.logical_start + unit.cardinality
                )
            ]
            assert rows == list(range(start, start + size))


@pytest.mark.parametrize(
    "offset,count", [(-1, 1), (0, 0), (8, 1), (7, 2), (True, 1), (0, 1.5)]
)
def test_slice_rejects_invalid_boundaries(offset, count):
    with pytest.raises(ValueError):
        _interval(10, 8).slice(offset, count)
