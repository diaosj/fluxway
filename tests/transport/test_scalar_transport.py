import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.sharding import ShardingConfig
from fluxway.common.types.tensor import TransportTensor
from fluxway.transport.communication import AsyncDataDistributor
from fluxway.transport.manifest import EDGE_EXPECTED_COUNT, FRAME_SENDER_SPMD_CODE


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_scalar_replication_self_delivery_uses_frame_abi():
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(2,))
    source_spec = ShardingConfig(type="static", axis=[]).to_spec().bind(mesh)
    target_spec = ShardingConfig(type="static", axis=[]).to_spec().bind(mesh)
    scalar = TransportTensor.from_tensor(jnp.array(42.0), global_shape=())
    scalar.source_step = 1

    async def capture_for_rank(rank):
        captured = {}

        async def send_fn(
            target_rank,
            header,
            *payload_iovs,
        ):
            captured[target_rank] = (header, payload_iovs)
            return True

        await AsyncDataDistributor(rank).distribute(
            local_data=scalar,
            source=source_spec,
            target=target_spec,
            send_fn=send_fn,
            tuuid=20,
            step=1,
            sender_spmd_code=9,
            sender_rank=rank,
            sender_world_size=mesh.size,
        )
        return captured

    captured0 = asyncio.run(capture_for_rank(0))
    captured1 = asyncio.run(capture_for_rank(1))

    assert set(captured0) == {0}
    assert set(captured1) == {1}
    assert int(captured0[0][0].frame_i32[FRAME_SENDER_SPMD_CODE]) == 9
    assert int(captured1[1][0].frame_i32[FRAME_SENDER_SPMD_CODE]) == 9
    assert int(captured0[0][0].edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
    assert np.asarray(captured0[0][1][0]).shape == ()
