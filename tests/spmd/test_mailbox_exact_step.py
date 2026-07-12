import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox

TUUID_A = 1001
TUUID_B = 1002


@pytest.mark.asyncio
async def test_mailbox_does_not_satisfy_future_step_with_prior_tensor():
    mailbox = WorkerMailbox()

    mailbox.declare_step(
        "op", step=1, expected_tensors={TUUID_A: 1}, expected_states=[]
    )
    await mailbox.deposit_shards(
        "op",
        [TUUID_A],
        1,
        "sender0",
        0,
        1,
        [np.array([1.0])],
        [1],
        1,
    )
    assert mailbox.is_step_ready("op", 1)

    mailbox.declare_step(
        "op", step=2, expected_tensors={TUUID_A: 1}, expected_states=[]
    )
    assert not mailbox.is_step_ready("op", 2)
    assert await mailbox.get_tensor_shard_views("op", TUUID_A, 2) == {}


@pytest.mark.asyncio
async def test_mailbox_does_not_satisfy_future_step_with_prior_control():
    mailbox = WorkerMailbox()

    mailbox.declare_step("op", step=1, expected_tensors={}, expected_states=["state"])
    await mailbox.deposit_control_arguments("op", 1, ["state"], [{"value": 1}])
    assert mailbox.is_step_ready("op", 1)

    mailbox.declare_step("op", step=2, expected_tensors={}, expected_states=["state"])
    assert not mailbox.is_step_ready("op", 2)
    assert await mailbox.get_control_argument("op", "state", 2) is None


@pytest.mark.asyncio
async def test_clear_step_releases_exact_step_only():
    mailbox = WorkerMailbox()

    mailbox.declare_step("op", step=1, expected_tensors={}, expected_states=["state"])
    mailbox.declare_step("op", step=2, expected_tensors={}, expected_states=["state"])
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
        expected_tensors={TUUID_A: 1, TUUID_B: 1},
        expected_states=[],
    )
    await mailbox.deposit_shards(
        "op",
        [TUUID_A, TUUID_B],
        3,
        "sender0",
        0,
        1,
        [np.array([1.0]), np.array([2.0])],
        [1, 1],
        3,
    )

    expected = [
        await mailbox.get_tensor_shard_views_and_arrivals("op", tuuid, 3)
        for tuuid in (TUUID_A, TUUID_B)
    ]
    actual = await mailbox.get_tensor_shard_views_and_arrivals_many(
        "op", (TUUID_A, TUUID_B), 3
    )

    assert actual == expected


@pytest.mark.asyncio
async def test_mailbox_batch_tensor_snapshot_rejects_duplicate_tuuids():
    mailbox = WorkerMailbox()
    with pytest.raises(ValueError, match="must be unique"):
        await mailbox.get_tensor_shard_views_and_arrivals_many(
            "op", (TUUID_A, TUUID_A), 3
        )
