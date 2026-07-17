import numpy as np
import pytest

from fluxway.transport.manifest import (
    ENTRY_EXPECTED_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_SOURCE_STEP,
    ENTRY_TENSOR_KIND,
    ENTRY_WIDTH,
    FRAME_FLAGS,
    FRAME_VERSION,
    PAYLOAD_DTYPE,
    PROTOCOL_VERSION,
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrameHeader,
    TransportFrameBuilder,
    TransportFrameTemplate,
    legacy_metadata_nbytes_as_i64,
    metadata_nbytes,
)


def test_transport_route_protocol_is_version_four_with_tensor_entry_kind():
    assert PROTOCOL_VERSION == 4
    assert ENTRY_WIDTH == 12
    assert hasattr(TransportFrameBuilder, "add_tensor_transfer_entry")


def test_ordering_frame_is_header_only_and_keeps_entry_width():
    builder = TransportFrameBuilder(
        step=5,
        sender_spmd_code=3,
        sender_rank=2,
        sender_world_size=8,
        flags=FrameFlag.ORDERING_COMPLETION,
    )
    builder.add_ordering_completion_entry(
        dependency_code=17,
        expected_count=8,
        source_step=5,
    )

    frame = builder.build()

    assert int(frame.frame_i32[FRAME_FLAGS]) == int(FrameFlag.ORDERING_COMPLETION)
    assert frame.entry_i32.shape == (1, ENTRY_WIDTH)
    assert int(frame.entry_i32[0, ENTRY_ROUTE_CODE]) == 17
    assert int(frame.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 8
    assert int(frame.entry_i32[0, ENTRY_SOURCE_STEP]) == 5
    assert TensorEntryKind(int(frame.entry_i32[0, ENTRY_TENSOR_KIND])) is (
        TensorEntryKind.NOT_APPLICABLE
    )
    assert frame.payload_i32.shape == (0, 8)
    assert frame.payload_segments == ()

    frame.entry_i32[0, ENTRY_TENSOR_KIND] = int(TensorEntryKind.ROUTING_MISS)
    with pytest.raises(ValueError, match="must use NOT_APPLICABLE"):
        frame.validate()


@pytest.mark.parametrize(
    ("sender_rank", "sender_world_size"),
    ((0, 0), (2, 2)),
)
def test_transport_frame_builder_rejects_invalid_sender_rank_domain(
    sender_rank,
    sender_world_size,
):
    with pytest.raises(ValueError, match="sender rank domain"):
        TransportFrameBuilder(
            step=0,
            sender_spmd_code=0,
            sender_rank=sender_rank,
            sender_world_size=sender_world_size,
        )


def test_manifest_uses_i32_metadata_and_preserves_u64_payload_and_i64_shape():
    ids = np.array([2**40, 2**40 + 1], dtype=np.uint64)
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=9,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=4,
        source_step=1,
        global_shape=(2**35,),
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, ids, 2)
    frame = builder.build()

    assert frame.frame_i32.dtype == np.int32
    assert frame.entry_i32.dtype == np.int32
    assert frame.payload_i32.dtype == np.int32
    assert frame.shape_i32.dtype == np.int32
    assert frame.global_shape_i64.dtype == np.int64
    assert frame.payload_segments[0].dtype == np.uint64
    assert int(np.asarray(frame.payload_segments[0])[0]) == 2**40
    assert int(frame.global_shape_i64[0]) == 2**35
    assert metadata_nbytes(frame) < legacy_metadata_nbytes_as_i64(frame)


def test_transport_frame_header_round_trips_without_payload_copy():
    values = np.array([1.0, 2.0], dtype=np.float32)
    builder = TransportFrameBuilder(
        step=2, sender_spmd_code=1, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=7,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=2,
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, values, 2)
    frame = builder.build()

    rebuilt = frame.header.to_frame(frame.payload_segments)

    assert rebuilt.payload_segments[0] is values
    assert rebuilt.header is not frame.header
    assert np.array_equal(rebuilt.frame_i32, frame.frame_i32)


def test_transport_frame_header_rejects_payload_shape_mismatch():
    values = np.array([1.0, 2.0], dtype=np.float32)
    builder = TransportFrameBuilder(
        step=2, sender_spmd_code=1, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=7,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=2,
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, values, 2)
    header = builder.build().header

    with pytest.raises(ValueError, match="does not match segment shape"):
        header.to_frame((np.array([1.0, 2.0, 3.0], dtype=np.float32),))


def test_transport_frame_template_preserves_route_code_without_payload_copy():
    values = np.array([1.0, 2.0], dtype=np.float32)
    builder = TransportFrameBuilder(
        step=2, sender_spmd_code=1, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=31,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=2,
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, values, 2)
    template = TransportFrameTemplate.from_frame(builder.build())
    next_values = np.array([3.0, 4.0], dtype=np.float32)

    frame = template.instantiate(
        step=3,
        payload_segments=(next_values,),
        payload_valid_counts=(2,),
        global_shapes=(None,),
        source_steps=(3,),
    )

    assert frame.entry_i32.shape == (1, ENTRY_WIDTH)
    assert int(frame.entry_i32[0, ENTRY_ROUTE_CODE]) == 31
    assert frame.payload_segments[0] is next_values


def test_transport_frame_header_rejects_non_numpy_metadata():
    with pytest.raises(TypeError, match="host numpy metadata"):
        TransportFrameHeader(
            frame_i32=[1, 2, 3],  # type: ignore
            entry_i32=np.zeros((0, ENTRY_WIDTH), dtype=np.int32),
            payload_i32=np.zeros((0, 8), dtype=np.int32),
            shape_i32=np.zeros((0,), dtype=np.int32),
            global_shape_i64=np.zeros((0,), dtype=np.int64),
            compact_i64=np.zeros((0,), dtype=np.int64),
        ).validate()


def test_manifest_rejects_i32_shape_overflow():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=1,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=1,
    )

    with pytest.raises(OverflowError):
        builder.add_payload(
            entry_index,
            PayloadRole.VALUES,
            np.empty((np.iinfo(np.int32).max + 1,), dtype=np.int8),
            0,
        )


def test_manifest_validate_rejects_negative_metadata():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=1,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=1,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.arange(1, dtype=np.int32), 1
    )
    frame = builder.build()
    frame.frame_i32[FRAME_VERSION] = -1

    with pytest.raises(ValueError, match="negative int32 metadata"):
        frame.validate()


def test_manifest_rejects_previous_protocol_version_without_fallback():
    builder = TransportFrameBuilder(
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
        flags=FrameFlag.ORDERING_COMPLETION,
    )
    builder.add_ordering_completion_entry(
        dependency_code=1,
        expected_count=1,
        source_step=1,
    )
    frame = builder.build()
    frame.frame_i32[FRAME_VERSION] = 3

    with pytest.raises(ValueError, match="Unsupported transport frame version 3"):
        frame.validate()


def test_manifest_rejects_non_dlpack_payload():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=1,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=1,
    )

    class ArrayMetadataOnly:
        dtype = np.dtype(np.int64)
        shape = (2,)

    for payload in ([1, 2], ArrayMetadataOnly()):
        with pytest.raises(TypeError, match="DLPack tensor"):
            builder.add_payload(entry_index, PayloadRole.VALUES, payload, 2)  # type: ignore[arg-type]


def test_manifest_validate_rejects_dtype_mismatch():
    builder = TransportFrameBuilder(
        step=1, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=1,
        meta_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        source_step=1,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.arange(1, dtype=np.int32), 1
    )
    frame = builder.build()
    frame.payload_i32[0, PAYLOAD_DTYPE] += 1

    with pytest.raises(ValueError, match="dtype_code"):
        frame.validate()


def test_frame_flag_rejects_route_kind_mismatch():
    tensor_builder = TransportFrameBuilder(
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
    )
    with pytest.raises(ValueError, match="ORDERING_COMPLETION frame"):
        tensor_builder.add_ordering_completion_entry(
            dependency_code=1,
            expected_count=1,
            source_step=1,
        )

    ordering_builder = TransportFrameBuilder(
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
        flags=FrameFlag.ORDERING_COMPLETION,
    )
    with pytest.raises(ValueError, match="TENSOR_TRANSFER frame"):
        ordering_builder.add_tensor_transfer_entry(
            transfer_code=1,
            meta_state=0,
            tensor_kind=TensorEntryKind.TENSOR,
            expected_count=1,
            source_step=1,
        )


@pytest.mark.parametrize(
    "tensor_kind",
    [
        TensorEntryKind.ROUTING_MISS,
        TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
    ],
)
def test_absent_tensor_entry_rejects_nonzero_payload_count(tensor_kind):
    builder = TransportFrameBuilder(
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=1,
        meta_state=0,
        tensor_kind=tensor_kind,
        expected_count=1,
        source_step=1,
    )
    builder.add_payload(
        entry_index,
        PayloadRole.VALUES,
        np.arange(1, dtype=np.int32),
        1,
    )

    with pytest.raises(ValueError, match="requires zero payload valid counts"):
        builder.build()
