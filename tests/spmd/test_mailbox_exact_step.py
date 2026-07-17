import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.manifest import PayloadRole
from tests.spmd.mailbox_frame_helpers import TensorFrameEntry, deposit_tensor_frame

TRANSFER_A = 1001
TRANSFER_B = 1002


@pytest.mark.asyncio
async def test_mailbox_does_not_satisfy_future_step_with_prior_tensor():
    mailbox = WorkerMailbox()

    mailbox.declare_step(
        "op", step=1, expected_transfers={TRANSFER_A: 1}, expected_states=[]
    )
    await deposit_tensor_frame(
        mailbox,
        "op",
        "sender0",
        step=1,
        sender_rank=0,
        sender_world_size=1,
        entries=(
            TensorFrameEntry(
                transfer_code=TRANSFER_A,
                payloads=((PayloadRole.VALUES, np.array([1.0]), 1),),
            ),
        ),
    )
    assert mailbox.is_step_ready("op", 1)

    mailbox.declare_step(
        "op", step=2, expected_transfers={TRANSFER_A: 1}, expected_states=[]
    )
    assert not mailbox.is_step_ready("op", 2)
    assert await mailbox.get_tensor_shard_views("op", TRANSFER_A, 2) == {}


@pytest.mark.asyncio
async def test_mailbox_does_not_satisfy_future_step_with_prior_control():
    mailbox = WorkerMailbox()

    mailbox.declare_step("op", step=1, expected_transfers={}, expected_states=["state"])
    await mailbox.deposit_control_arguments("op", 1, ["state"], [{"value": 1}])
    assert mailbox.is_step_ready("op", 1)

    mailbox.declare_step("op", step=2, expected_transfers={}, expected_states=["state"])
    assert not mailbox.is_step_ready("op", 2)
    assert await mailbox.get_control_argument("op", "state", 2) is None


@pytest.mark.asyncio
async def test_clear_step_releases_exact_step_only():
    mailbox = WorkerMailbox()

    mailbox.declare_step("op", step=1, expected_transfers={}, expected_states=["state"])
    mailbox.declare_step("op", step=2, expected_transfers={}, expected_states=["state"])
    await mailbox.deposit_control_arguments("op", 1, ["state"], ["one"])
    await mailbox.deposit_control_arguments("op", 2, ["state"], ["two"])

    await mailbox.clear_step("op", 1)

    assert await mailbox.get_control_argument("op", "state", 1) is None
    assert await mailbox.get_control_argument("op", "state", 2) == "two"


@pytest.mark.asyncio
async def test_mailbox_batch_tensor_snapshot_matches_single_snapshots():
    mailbox = WorkerMailbox()
    mailbox.declare_step(
        "op",
        step=3,
        expected_transfers={TRANSFER_A: 1, TRANSFER_B: 1},
        expected_states=[],
    )
    await deposit_tensor_frame(
        mailbox,
        "op",
        "sender0",
        step=3,
        sender_rank=0,
        sender_world_size=1,
        entries=(
            TensorFrameEntry(
                transfer_code=TRANSFER_A,
                payloads=((PayloadRole.VALUES, np.array([1.0]), 1),),
            ),
            TensorFrameEntry(
                transfer_code=TRANSFER_B,
                payloads=((PayloadRole.VALUES, np.array([2.0]), 1),),
            ),
        ),
    )

    expected = [
        await mailbox.get_tensor_shard_views_and_arrivals("op", transfer_code, 3)
        for transfer_code in (TRANSFER_A, TRANSFER_B)
    ]
    actual = await mailbox.get_tensor_shard_views_and_arrivals_many(
        "op", (TRANSFER_A, TRANSFER_B), 3
    )

    assert actual == expected


@pytest.mark.asyncio
async def test_mailbox_batch_tensor_snapshot_rejects_duplicate_transfer_codes():
    mailbox = WorkerMailbox()
    with pytest.raises(ValueError, match="must be unique"):
        await mailbox.get_tensor_shard_views_and_arrivals_many(
            "op", (TRANSFER_A, TRANSFER_A), 3
        )
