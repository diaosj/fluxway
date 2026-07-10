import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.manifest import EmptyKind, PayloadRole, TransportFrameBuilder


@pytest.mark.asyncio
async def test_mailbox_deposits_flat_frame_and_preserves_64bit_payload():
    ids = np.array([2**40], dtype=np.uint64)
    origin = np.array([2**33], dtype=np.int64)
    builder = TransportFrameBuilder(
        step=2, sender_spmd_code=0, sender_rank=1, sender_world_size=4
    )
    edge = builder.add_edge(
        tuuid=10,
        meta_state=2,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=4,
        source_step=2,
        global_shape=(2**35,),
    )
    builder.add_payload(edge, PayloadRole.VALUES, ids, 1)
    builder.add_payload(edge, PayloadRole.ORIGIN, origin, 1)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 2, {10: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    stored = mailbox._states["op"][2].tensor_slots[10][("Input", 1)][0]
    assert stored.flat_values.dtype == np.uint64
    assert int(stored.flat_values[0]) == 2**40
    assert stored.global_shape is not None
    assert tuple(stored.global_shape) == (2**35,)


@pytest.mark.asyncio
async def test_mailbox_protocol_empty_satisfies_barrier_without_business_slot():
    builder = TransportFrameBuilder(
        step=3, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=11,
        meta_state=0,
        empty_kind=EmptyKind.PROTOCOL_EMPTY,
        min_rows=0,
        expected_count=1,
        source_step=3,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 3, {11: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 3)]
    assert 11 not in mailbox._states["op"][3].tensor_slots
    arrivals = await mailbox.get_tensor_arrivals("op", 11, 3)
    assert set(arrivals.keys()) == {("Input", 0)}
    assert arrivals[("Input", 0)]["empty_kind"] is EmptyKind.PROTOCOL_EMPTY
    assert arrivals[("Input", 0)]["expected_count"] == 1


@pytest.mark.asyncio
async def test_mailbox_data_empty_allowed_preserves_zero_row_slot():
    builder = TransportFrameBuilder(
        step=7, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=15,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=1,
        source_step=7,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 7, {15: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 7)]
    stored = mailbox._states["op"][7].tensor_slots[15][("Input", 0)][0]
    assert stored.flat_values.shape == (0,)
    assert stored.flat_values.dtype == np.int32


@pytest.mark.asyncio
async def test_mailbox_rejects_data_required_empty_frame():
    builder = TransportFrameBuilder(
        step=4, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=12,
        meta_state=0,
        empty_kind=EmptyKind.DATA_REQUIRED,
        min_rows=1,
        expected_count=1,
        source_step=4,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 4, {12: 1}, [])
    with pytest.raises(ValueError, match="DATA_REQUIRED"):
        await mailbox.deposit_transport_frame("op", "Input", builder.build())


@pytest.mark.asyncio
async def test_mailbox_rejects_duplicate_protocol_empty_frame():
    builder = TransportFrameBuilder(
        step=8, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    edge = builder.add_edge(
        tuuid=16,
        meta_state=0,
        empty_kind=EmptyKind.PROTOCOL_EMPTY,
        min_rows=0,
        expected_count=1,
        source_step=8,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int64), 0)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 8, {16: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    duplicate = TransportFrameBuilder(
        step=8, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    duplicate_edge = duplicate.add_edge(
        tuuid=16,
        meta_state=0,
        empty_kind=EmptyKind.PROTOCOL_EMPTY,
        min_rows=0,
        expected_count=1,
        source_step=8,
    )
    duplicate.add_payload(
        duplicate_edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int64), 0
    )
    with pytest.raises(ValueError, match="DUPLICATE SHARD RECEIVED"):
        await mailbox.deposit_transport_frame("op", "Input", duplicate.build())


@pytest.mark.asyncio
async def test_mailbox_frame_expected_count_configures_barrier():
    builder = TransportFrameBuilder(
        step=5, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    edge = builder.add_edge(
        tuuid=13,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=4,
        source_step=5,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.array([1], dtype=np.int32), 1)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 5, {13: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == []
    assert mailbox._barriers["op"][5].tensor_progress[13] == "1/4"


@pytest.mark.asyncio
async def test_mailbox_frame_expected_count_reconfigures_after_control_arrival():
    builder = TransportFrameBuilder(
        step=6, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    edge = builder.add_edge(
        tuuid=14,
        meta_state=0,
        empty_kind=EmptyKind.DATA_EMPTY_ALLOWED,
        min_rows=0,
        expected_count=4,
        source_step=6,
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.array([1], dtype=np.int32), 1)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 6, {14: 1}, ["state"])
    await mailbox.deposit_control_arguments("op", 6, ["state"], [1])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == []
    assert mailbox._barriers["op"][6].tensor_progress[14] == "1/4"


@pytest.mark.asyncio
async def test_mailbox_zero_copy_writes_into_buffer():
    mailbox = WorkerMailbox()
    buffer = np.zeros((4, 2), dtype=np.float32)

    await mailbox.register_zero_copy_slot(
        "compute", 701, step=0, target_buffers=[buffer], expect_offsets={0: 0}
    )
    await mailbox.deposit_shards(
        "compute",
        [701],
        step=0,
        sender_id="s0",
        rank=0,
        sender_world_size=4,
        payloads=[np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
        counts=[2],
        source_step=0,
        meta_tensor_state_list=[0],
    )

    buffers = await mailbox.get_zero_copy_buffers("compute", 701, step=0)
    assert buffers is not None
    np.testing.assert_array_equal(buffers[0][:2], [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.asyncio
async def test_mailbox_zero_copy_ragged_writes_data_and_lengths():
    mailbox = WorkerMailbox()
    data_buffer = np.zeros(10, dtype=np.float32)
    lengths_buffer = np.zeros(4, dtype=np.int32)
    flat_values = np.array([1, 1, 1, 2, 2], dtype=np.float32)
    lengths = np.array([3, 2], dtype=np.int32)

    await mailbox.register_zero_copy_slot(
        "compute",
        1001,
        step=0,
        target_buffers=[data_buffer, lengths_buffer],
        expect_offsets={0: [0, 0]},
    )
    await mailbox.deposit_shards(
        "compute",
        [1001],
        0,
        "s0",
        0,
        1,
        payloads=[flat_values, lengths],
        counts=[5, 2],
        source_step=0,
        meta_tensor_state_list=[1],
    )

    buffers = await mailbox.get_zero_copy_buffers("compute", 1001, step=0)
    assert buffers is not None
    np.testing.assert_array_equal(buffers[0][:5], flat_values)
    np.testing.assert_array_equal(buffers[1][:2], lengths)


@pytest.mark.asyncio
async def test_mailbox_zero_copy_slots_accept_multiple_tensor_payloads():
    mailbox = WorkerMailbox()
    first_buffer = np.zeros(4, dtype=np.float32)
    second_buffer = np.zeros(4, dtype=np.float32)

    await mailbox.register_zero_copy_slot("op", 101, 0, [first_buffer], {0: 0})
    await mailbox.register_zero_copy_slot("op", 102, 0, [second_buffer], {0: 0})

    first_payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    second_payload = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    await mailbox.deposit_shards(
        "op",
        [101, 102],
        0,
        "Input",
        0,
        1,
        payloads=[first_payload, second_payload],
        counts=[4, 4],
        source_step=0,
        meta_tensor_state_list=[0, 0],
    )

    first_result = await mailbox.get_zero_copy_buffers("op", 101, 0)
    second_result = await mailbox.get_zero_copy_buffers("op", 102, 0)

    assert first_result is not None
    assert second_result is not None
    np.testing.assert_array_equal(first_result[0], first_payload)
    np.testing.assert_array_equal(second_result[0], second_payload)
