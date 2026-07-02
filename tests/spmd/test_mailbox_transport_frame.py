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
        global_shape=(2**35,),
    )
    builder.add_payload(edge, PayloadRole.VALUES, ids, 1)
    builder.add_payload(edge, PayloadRole.ORIGIN, origin, 1)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 2, {10: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    stored = mailbox._tensor_slots["op"][10][2][("Input", 1)][0]
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
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 3, {11: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 3)]
    assert 11 not in mailbox._tensor_slots.get("op", {})
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
    )
    builder.add_payload(edge, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0)

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 7, {15: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 7)]
    stored = mailbox._tensor_slots["op"][15][7][("Input", 0)][0]
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
