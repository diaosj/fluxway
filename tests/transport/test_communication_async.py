import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.ragged import ragged_level_count
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.common.types.tensor import TransportTensor
from fluxway.transport.communication import AsyncDataDistributor
from fluxway.transport.manifest import (
    EDGE_EMPTY_KIND,
    EDGE_EXPECTED_COUNT,
    EDGE_META_STATE,
    FRAME_EDGE_COUNT,
    FRAME_PAYLOAD_COUNT,
    FRAME_SENDER_SPMD_CODE,
    EmptyKind,
    PAYLOAD_ROLE,
    PayloadRole,
)


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


async def _capture_frame(distributor, **kwargs):
    captured = {}

    async def send_fn(
        target_id,
        header,
        *payload_iovs,
    ):
        captured[target_id] = (header, payload_iovs)
        return True

    result = await distributor.distribute(send_fn=send_fn, **kwargs)
    return result, captured


def test_async_distributor_uses_only_flat_frame_abi():
    mesh = DeviceMesh(axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1]))
    spec = ShardingSpec.chunk(P("x")).bind(mesh)
    data = TransportTensor(jnp.arange(10, dtype=jnp.float32), global_shape=(20,))

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=spec,
            target=spec,
            tuuid=17,
            step=3,
            sender_spmd_code=5,
            sender_rank=0,
            sender_world_size=mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0}
    (
        header,
        segments,
    ) = captured[0]
    assert int(header.frame_i32[FRAME_SENDER_SPMD_CODE]) == 5
    assert int(header.frame_i32[FRAME_EDGE_COUNT]) == 1
    assert int(header.frame_i32[FRAME_PAYLOAD_COUNT]) == 1
    assert header.edge_i32.dtype == np.int32
    assert header.payload_i32.dtype == np.int32
    assert header.shape_i32.dtype == np.int32
    assert header.global_shape_i64.dtype == np.int64
    assert header.compact_i64.dtype == np.int64
    assert len(segments) == 1


def test_async_distributor_batch_metadata_is_encoded_in_frame_rows():
    mesh = DeviceMesh(axis_names=("x",), device_shape=(1,), device_ids=np.array([0]))
    spec = ShardingSpec.chunk(P("x")).bind(mesh)
    dense = TransportTensor(jnp.arange(4, dtype=jnp.float32), global_shape=(4,))
    ragged = TransportTensor(
        jnp.arange(5, dtype=jnp.float32),
        nested_splits=[jnp.array([0, 2, 5], dtype=jnp.int32)],
        global_shape=(2,),
    )
    origin = TransportTensor(
        jnp.arange(3, dtype=jnp.float32),
        origin_indices=jnp.array([7, 8, 9], dtype=jnp.int64),
        global_shape=(10,),
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=[dense, ragged, origin],
            source=spec,
            target=spec,
            tuuid=[21, 22, 23],
            step=4,
            sender_spmd_code=2,
            sender_rank=0,
            sender_world_size=1,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    header, _payload_iovs = captured[0]
    edge_i32 = header.edge_i32
    payload_i32 = header.payload_i32
    assert int(edge_i32[0, EDGE_META_STATE]) == 0
    assert ragged_level_count(int(edge_i32[1, EDGE_META_STATE])) == 1
    assert bool(int(edge_i32[1, EDGE_META_STATE]) & 1)
    assert int(edge_i32[2, EDGE_META_STATE]) == 2
    assert payload_i32[:, PAYLOAD_ROLE].tolist() == [
        int(PayloadRole.VALUES),
        int(PayloadRole.VALUES),
        int(PayloadRole.RAGGED_LENGTHS),
        int(PayloadRole.VALUES),
        int(PayloadRole.ORIGIN),
    ]


def test_async_distributor_nested_ragged_preserves_payload_dtype_and_levels():
    src_mesh = DeviceMesh(
        axis_names=("x",), device_shape=(1,), device_ids=np.array([0])
    )
    dst_mesh = DeviceMesh(
        axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1])
    )
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)
    data = TransportTensor.from_lists(
        [[[2**40, 2**40 + 1], [2**40 + 2]], [[2**40 + 3]]],
        dtype=np.uint64,
        global_shape=(2,),
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            tuuid=91,
            step=7,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1}
    for header, payload_iovs in captured.values():
        assert int(header.frame_i32[FRAME_EDGE_COUNT]) == 1
        assert int(header.frame_i32[FRAME_PAYLOAD_COUNT]) == 3
        assert int(header.edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
        assert ragged_level_count(int(header.edge_i32[0, EDGE_META_STATE])) == 2
        assert header.payload_i32[:, PAYLOAD_ROLE].tolist() == [
            int(PayloadRole.VALUES),
            int(PayloadRole.RAGGED_LENGTHS),
            int(PayloadRole.RAGGED_LENGTHS),
        ]
        assert payload_iovs[0].dtype == np.uint64


def test_transport_frame_static_manifest_emits_only_real_producers():
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)
    data = TransportTensor(jnp.arange(5, dtype=jnp.int32), global_shape=(10,))

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            tuuid=92,
            step=8,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1}
    assert int(captured[0][0].edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
    assert int(captured[1][0].edge_i32[0, EDGE_EXPECTED_COUNT]) == 2


def test_transport_frame_dynamic_manifest_uses_all_sender_protocol_empty():
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("y",))
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.hash(P("y")).bind(dst_mesh)
    data = TransportTensor(jnp.array([0, 1, 4], dtype=jnp.int32), global_shape=(6,))

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            tuuid=93,
            step=9,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1, 2, 3}
    for frame in captured.values():
        edge_i32 = frame[0].edge_i32
        assert int(edge_i32[0, EDGE_EXPECTED_COUNT]) == src_mesh.size
    assert int(captured[2][0].edge_i32[0, EDGE_EMPTY_KIND]) == int(
        EmptyKind.PROTOCOL_EMPTY
    )
    assert int(captured[3][0].edge_i32[0, EDGE_EMPTY_KIND]) == int(
        EmptyKind.PROTOCOL_EMPTY
    )
