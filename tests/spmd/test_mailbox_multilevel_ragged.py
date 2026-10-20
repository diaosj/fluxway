import numpy as np
import pytest

from fluxway.common.ragged import encode_ragged_level_count
from fluxway.common.tensor_backend import set_backend
from fluxway.spmd.worker.mailbox.mailbox import TransportShardView, WorkerMailbox
from fluxway.transport.manifest import PayloadRole
from tests.spmd.mailbox_frame_helpers import TensorFrameEntry, deposit_tensor_frame


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.mark.asyncio
async def test_mailbox_reconstructs_multilevel_ragged_splits():
    mailbox = WorkerMailbox()
    meta_state = encode_ragged_level_count(1, 2)
    flat_values = np.array([11, 12, 13], dtype=np.int64)
    level0_lengths = np.array([2], dtype=np.int32)
    level1_lengths = np.array([2, 1], dtype=np.int32)

    # Payload protocol for one 2-level ragged tensor:
    #   flat_values=[11,12,13]
    #   level0 lengths=[2]      row -> two child lists
    #   level1 lengths=[2,1]    child list -> flat values
    mailbox.declare_step("op", step=0, expected_transfers={7: 1}, expected_states=[])
    await deposit_tensor_frame(
        mailbox,
        "op",
        "src",
        step=0,
        sender_rank=0,
        sender_world_size=1,
        entries=(
            TensorFrameEntry(
                transfer_code=7,
                meta_state=meta_state,
                payloads=(
                    (PayloadRole.VALUES, flat_values, 3),
                    (PayloadRole.RAGGED_LENGTHS, level0_lengths, 1),
                    (PayloadRole.RAGGED_LENGTHS, level1_lengths, 2),
                ),
            ),
        ),
    )

    shards = await mailbox.get_all_tensor_shard_views("op", 0)
    tensor = shards[7][("src", 0)][0]

    assert [s.tolist() for s in tensor.nested_splits] == [[0, 2], [0, 2, 3]]
    assert tensor.flat_values.tolist() == [11, 12, 13]


def test_transport_shard_adopts_ray_numpy_payloads_into_cuda_domain():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device is required")
    set_backend("cupy")
    view = TransportShardView(
        payloads=(
            np.array([11, 12, 13], dtype=np.int64),
            np.array([3], dtype=np.int32),
        ),
        counts=(3, 1),
        meta_state=encode_ragged_level_count(1, 1),
        sender_world_size=1,
        sender_rank=0,
        global_shape=(3,),
        compact_params=None,
    )

    assert isinstance(view.flat_values, cp.ndarray)
    assert all(isinstance(split, cp.ndarray) for split in view.nested_splits)
    assert cp.asnumpy(view.nested_splits[0]).tolist() == [0, 3]
