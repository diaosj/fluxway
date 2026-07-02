import numpy as np
import pytest

from fluxway.common.ragged import encode_ragged_level_count
from fluxway.common.tensor_backend import set_backend
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.mark.asyncio
async def test_mailbox_reconstructs_multilevel_ragged_splits():
    mailbox = WorkerMailbox()
    meta_state = encode_ragged_level_count(1, 2)

    # Payload protocol for one 2-level ragged tensor:
    #   flat_values=[11,12,13]
    #   level0 lengths=[2]      row -> two child lists
    #   level1 lengths=[2,1]    child list -> flat values
    await mailbox.deposit_shards(
        op_name="op",
        tuuids=[7],
        step=0,
        sender_id="src",
        rank=0,
        sender_world_size=1,
        payloads=[
            np.array([11, 12, 13], dtype=np.int64),
            np.array([2], dtype=np.int32),
            np.array([2, 1], dtype=np.int32),
        ],
        counts=[3, 1, 2],
        meta_tensor_state_list=[meta_state],
    )

    shards = await mailbox.get_all_tensor_shard_views("op", 0)
    tensor = shards[7][("src", 0)][0]

    assert [s.tolist() for s in tensor.nested_splits] == [[0, 2], [0, 2, 3]]
    assert tensor.flat_values.tolist() == [11, 12, 13]
