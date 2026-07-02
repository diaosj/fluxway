import numpy as np
import pytest

from fluxway.transport.manifest import (
    FRAME_VERSION,
    PAYLOAD_DTYPE,
    EmptyKind,
    PayloadRole,
    TransportFrameBuilder,
    legacy_metadata_nbytes_as_i64,
    metadata_nbytes,
)


def test_manifest_uses_i32_metadata_and_preserves_u64_payload_and_i64_shape():
    ids = np.array([2**40, 2**40 + 1], dtype=np.uint64)
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    edge = builder.add_edge(
        tuuid=9,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=4,
        global_shape=(2**35,),
    )
    builder.add_payload(edge, PayloadRole.VALUES, ids, 2)
    frame = builder.build()

    assert frame.frame_i32.dtype == np.int32
    assert frame.edge_i32.dtype == np.int32
    assert frame.payload_i32.dtype == np.int32
    assert frame.shape_i32.dtype == np.int32
    assert frame.global_shape_i64.dtype == np.int64
    assert frame.payload_segments[0].dtype == np.uint64
    assert int(frame.payload_segments[0][0]) == 2**40
    assert int(frame.global_shape_i64[0]) == 2**35
    assert metadata_nbytes(frame) < legacy_metadata_nbytes_as_i64(frame)


def test_manifest_rejects_i32_shape_overflow():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=1,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )

    with pytest.raises(OverflowError):
        builder.add_payload(
            edge,
            PayloadRole.VALUES,
            np.empty((np.iinfo(np.int32).max + 1,), dtype=np.int8),
            0,
        )


def test_manifest_validate_rejects_negative_metadata():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=1,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.arange(1, dtype=np.int32), 1)
    frame = builder.build()
    frame.frame_i32[FRAME_VERSION] = -1

    with pytest.raises(ValueError, match="negative int32 metadata"):
        frame.validate()


def test_manifest_rejects_payload_without_dtype():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=1,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )

    with pytest.raises(TypeError, match="dtype"):
        builder.add_payload(edge, PayloadRole.VALUES, [1, 2], 2)


def test_manifest_validate_rejects_dtype_mismatch():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=1,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.arange(1, dtype=np.int32), 1)
    frame = builder.build()
    frame.payload_i32[0, PAYLOAD_DTYPE] += 1

    with pytest.raises(ValueError, match="dtype_code"):
        frame.validate()
