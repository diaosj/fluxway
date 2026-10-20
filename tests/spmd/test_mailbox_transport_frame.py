import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import TransportShardView, WorkerMailbox
from fluxway.transport.manifest import (
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)


def _tensor_frame(transfer_code: int, payload: np.ndarray):
    builder = TransportFrameBuilder(
        step=9, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=transfer_code,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        coherence_epoch=9,
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, payload, len(payload))
    return builder.build()


@pytest.mark.asyncio
async def test_mailbox_deposits_flat_frame_and_preserves_64bit_payload():
    ids = np.array([2**40], dtype=np.uint64)
    origin = np.array([2**33], dtype=np.int64)
    builder = TransportFrameBuilder(
        step=2, sender_spmd_code=0, sender_rank=1, sender_world_size=4
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=10,
        layout_state=2,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=4,
        coherence_epoch=2,
        global_shape=(2**35,),
    )
    builder.add_payload(entry_index, PayloadRole.VALUES, ids, 1)
    builder.add_payload(entry_index, PayloadRole.ORIGIN, origin, 1)

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 2, {10: 4}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    stored = mailbox._states["op"][2].transfer_slots[10][("Input", 1)][0]
    assert stored.flat_values.dtype == np.uint64
    assert int(stored.flat_values[0]) == 2**40
    assert stored.global_shape is not None
    assert tuple(stored.global_shape) == (2**35,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tensor_kind",
    [
        TensorEntryKind.ROUTING_MISS,
        TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
    ],
)
async def test_mailbox_absent_tensor_satisfies_barrier_without_business_slot(
    tensor_kind,
):
    builder = TransportFrameBuilder(
        step=3, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=11,
        layout_state=0,
        tensor_kind=tensor_kind,
        expected_count=1,
        coherence_epoch=3,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0
    )

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 3, {11: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 3)]
    assert 11 not in mailbox._states["op"][3].transfer_slots
    arrivals = await mailbox.get_tensor_arrivals("op", 11, 3)
    assert set(arrivals.keys()) == {("Input", 0)}
    assert arrivals[("Input", 0)]["tensor_kind"] is tensor_kind
    assert arrivals[("Input", 0)]["expected_count"] == 1


@pytest.mark.asyncio
async def test_mailbox_tensor_preserves_zero_row_slot():
    builder = TransportFrameBuilder(
        step=7, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=15,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        coherence_epoch=7,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.empty((0,), dtype=np.int32), 0
    )

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 7, {15: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == [("op", 7)]
    stored = mailbox._states["op"][7].transfer_slots[15][("Input", 0)][0]
    assert stored.flat_values.shape == (0,)
    assert stored.flat_values.dtype == np.int32


@pytest.mark.asyncio
async def test_mailbox_rejects_duplicate_routing_miss_frame():
    builder = TransportFrameBuilder(
        step=8, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=16,
        layout_state=0,
        tensor_kind=TensorEntryKind.ROUTING_MISS,
        expected_count=1,
        coherence_epoch=8,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.empty((0,), dtype=np.int64), 0
    )

    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 8, {16: 1}, [])
    await mailbox.deposit_transport_frame("op", "Input", builder.build())

    duplicate = TransportFrameBuilder(
        step=8, sender_spmd_code=0, sender_rank=0, sender_world_size=1
    )
    duplicate_entry_index = duplicate.add_tensor_transfer_entry(
        transfer_code=16,
        layout_state=0,
        tensor_kind=TensorEntryKind.ROUTING_MISS,
        expected_count=1,
        coherence_epoch=8,
    )
    duplicate.add_payload(
        duplicate_entry_index,
        PayloadRole.VALUES,
        np.empty((0,), dtype=np.int64),
        0,
    )
    with pytest.raises(ValueError, match="DUPLICATE SHARD RECEIVED"):
        await mailbox.deposit_transport_frame("op", "Input", duplicate.build())


@pytest.mark.asyncio
async def test_mailbox_rejects_frame_expected_count_mismatch():
    builder = TransportFrameBuilder(
        step=5, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=13,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=4,
        coherence_epoch=5,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.array([1], dtype=np.int32), 1
    )

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 5, {13: 1}, [])
    with pytest.raises(ValueError, match="expected count mismatch"):
        await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == []
    assert mailbox._barriers["op"][5].transfer_progress[13] == "0/1"


@pytest.mark.asyncio
async def test_mailbox_rejects_count_mismatch_after_control_arrival():
    builder = TransportFrameBuilder(
        step=6, sender_spmd_code=0, sender_rank=0, sender_world_size=4
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=14,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=4,
        coherence_epoch=6,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.array([1], dtype=np.int32), 1
    )

    fired = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op, step: fired.append((op, step)))
    mailbox.declare_step("op", 6, {14: 1}, ["state"])
    await mailbox.deposit_control_arguments("op", 6, ["state"], [1])
    with pytest.raises(ValueError, match="expected count mismatch"):
        await mailbox.deposit_transport_frame("op", "Input", builder.build())

    assert fired == []
    assert mailbox._barriers["op"][6].transfer_progress[14] == "0/1"


@pytest.mark.asyncio
async def test_mailbox_separates_transfer_streams_from_the_same_sender():
    payload = np.array([1, 2], dtype=np.int32)
    mailbox = WorkerMailbox()
    mailbox.declare_step("op", 9, expected_transfers={41: 1, 42: 1}, expected_states=[])

    await mailbox.deposit_transport_frame("op", "Input", _tensor_frame(41, payload))

    assert not mailbox.is_step_ready("op", 9)
    assert mailbox._barriers["op"][9].transfer_progress == {41: "1/1", 42: "0/1"}

    await mailbox.deposit_transport_frame("op", "Input", _tensor_frame(42, payload))

    assert mailbox.is_step_ready("op", 9)
    first = await mailbox.get_tensor_shard_views("op", transfer_code=41, step=9)
    second = await mailbox.get_tensor_shard_views("op", transfer_code=42, step=9)
    first_view = first[("Input", 0)][0]
    second_view = second[("Input", 0)][0]
    assert isinstance(first_view, TransportShardView)
    assert isinstance(second_view, TransportShardView)
    assert first_view.payloads[0] is payload
    assert second_view.payloads[0] is payload


@pytest.mark.asyncio
async def test_mailbox_tensor_deposit_rejects_ordering_frame():
    builder = TransportFrameBuilder(
        step=9,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
        flags=FrameFlag.ORDERING_COMPLETION,
    )
    builder.add_ordering_completion_entry(dependency_code=7, expected_count=1)

    with pytest.raises(ValueError, match="FrameFlag.TENSOR_TRANSFER"):
        await WorkerMailbox().deposit_transport_frame("op", "Input", builder.build())
