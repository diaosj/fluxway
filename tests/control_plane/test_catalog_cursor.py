"""Unit tests for CatalogCursor and CatalogViewObservation refactoring."""

from __future__ import annotations

import copy

import pytest

from fluxway.common.ingestion.segment import LayoutUnitDescriptor
from fluxway.control_plane.ingestion import (
    CatalogCheckpointImage,
    CatalogCursor,
    CatalogViewObservation,
)


def _make_unit(unit_id: str = "u-1") -> LayoutUnitDescriptor:
    return LayoutUnitDescriptor(
        unit_id=unit_id,
        revision_identity="rev-1",
        cardinality=1,
        cardinality_metric="rows",
        split_policy="AtomicItem",
        locality_key="loc-1",
        physical_identity="phys-1",
        logical_start=0,
        opaque_token="token-1",
    )


def test_catalog_cursor_validation() -> None:
    with pytest.raises(TypeError, match="catalog cursor must be a mapping"):
        CatalogCursor([1, 2, 3])  # type: ignore

    with pytest.raises(TypeError, match="MessagePack serializable"):
        CatalogCursor({"key": lambda x: x})


def test_catalog_cursor_isolation_and_immutability() -> None:
    source = {"start": 0, "meta": {"part": 1}}
    cursor = CatalogCursor(source)

    # Mutating source does not affect cursor
    source["meta"]["part"] = 99
    assert cursor["meta"]["part"] == 1
    assert cursor["start"] == 0

    # Cannot set attributes or items
    with pytest.raises(TypeError):
        cursor["start"] = 10  # type: ignore

    with pytest.raises(AttributeError):
        cursor.new_attr = 123  # type: ignore


def test_catalog_cursor_mapping_interface() -> None:
    data = {"a": 1, "b": "str", "c": [1, 2]}
    cursor = CatalogCursor(data)

    assert len(cursor) == 3
    assert "a" in cursor
    assert cursor.get("b") == "str"
    assert cursor.get("nonexistent", 42) == 42
    assert set(cursor.keys()) == {"a", "b", "c"}
    assert list(iter(cursor)) == ["a", "b", "c"]
    assert cursor == {"a": 1, "b": "str", "c": [1, 2]}
    assert cursor == CatalogCursor(data)
    assert hash(cursor) == hash(CatalogCursor(data))

    # Test idempotence
    same_cursor = CatalogCursor(cursor)
    assert same_cursor is not cursor
    assert same_cursor == cursor

    # Test deepcopy returns self (safe for immutable objects)
    assert copy.deepcopy(cursor) is cursor

    # Test to_dict returns a fresh mutable copy
    dict_copy = cursor.to_dict()
    assert dict_copy == data
    dict_copy["a"] = 999
    assert cursor["a"] == 1


def test_catalog_view_observation_direct_construction() -> None:
    unit = _make_unit()
    cursor = CatalogCursor({"offset": 10})

    obs = CatalogViewObservation(
        source_identity="source-1",
        view_identity="view-1",
        schema_digest="schema-hash",
        lineage_digest="lineage-hash",
        units=(unit,),
        terminal=False,
        cursor=cursor,
    )
    assert obs.cursor == {"offset": 10}
    assert obs.cursor is cursor

    # Enforces CatalogCursor type: raw dict rejected
    with pytest.raises(TypeError, match="must be a CatalogCursor"):
        CatalogViewObservation(
            source_identity="source-1",
            view_identity="view-1",
            schema_digest="schema-hash",
            lineage_digest="lineage-hash",
            units=(unit,),
            terminal=False,
            cursor={"offset": 10},  # type: ignore
        )


def test_catalog_view_observation_create_accepts_dict_and_cursor() -> None:
    unit = _make_unit()

    obs1 = CatalogViewObservation.create(
        source_identity="source-1",
        view_identity="view-1",
        schema_digest="schema-hash",
        lineage_digest="lineage-hash",
        units=(unit,),
        terminal=False,
        cursor={"offset": 10},
    )
    assert isinstance(obs1.cursor, CatalogCursor)
    assert obs1.cursor["offset"] == 10

    cursor = CatalogCursor({"offset": 10})
    obs2 = CatalogViewObservation.create(
        source_identity="source-1",
        view_identity="view-1",
        schema_digest="schema-hash",
        lineage_digest="lineage-hash",
        units=(unit,),
        terminal=False,
        cursor=cursor,
    )
    assert obs2 == obs1


def test_catalog_checkpoint_image_with_cursor() -> None:
    cursor = CatalogCursor({"snapshot_id": 7})

    image = CatalogCheckpointImage(
        source_identity="source-1",
        catalog_view_identity="view-1",
        schema_digest="schema-hash",
        lineage_digest="lineage-hash",
        layout_id="layout-1",
        layout_digest="digest-1",
        end_position=4,
        cursor=cursor,
    )
    assert image.cursor == {"snapshot_id": 7}
    assert image.to_dict()["cursor"] == {"snapshot_id": 7}

    image_from_dict = CatalogCheckpointImage.create(
        source_identity="source-1",
        catalog_view_identity="view-1",
        schema_digest="schema-hash",
        lineage_digest="lineage-hash",
        layout_id="layout-1",
        layout_digest="digest-1",
        end_position=4,
        cursor={"snapshot_id": 7},
    )
    assert image_from_dict == image
