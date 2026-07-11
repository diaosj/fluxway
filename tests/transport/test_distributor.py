import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import get_backend, set_backend
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.transport.communication import AsyncDataDistributor
from fluxway.transport.manifest import (
    EDGE_COMPACT_COUNT,
    EDGE_EXPECTED_COUNT,
    PAYLOAD_VALID_COUNT,
)


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


async def _capture(local_rank, local_data, source, target, **kwargs):
    captured = {}

    async def send_fn(
        target_rank,
        header,
        *payload_iovs,
    ):
        captured[target_rank] = (
            header,
            payload_iovs,
        )
        return True

    result = await AsyncDataDistributor(local_rank).distribute(
        local_data=local_data,
        source=source,
        target=target,
        send_fn=send_fn,
        **kwargs,
    )
    return result, captured


def test_colocation_dispatch_uses_two_payload_segments():
    mesh = DeviceMesh(("x",), (2,), [0, 1])
    source = ShardingSpec.chunk(P("x")).bind(mesh)
    target = ShardingSpec.chunk(P("x")).bind(mesh)
    keys = TransportTensor(
        jnp.array([10, 20], dtype=jnp.int32),
        global_shape=(4,),
        source_step=1,
    )
    vals = TransportTensor(
        jnp.array([0.1, 0.2], dtype=jnp.float32),
        global_shape=(4,),
        source_step=1,
    )

    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            [keys, vals],
            source,
            target,
            tuuid=[101, 102],
            step=1,
            sender_spmd_code=6,
            sender_rank=0,
            sender_world_size=mesh.size,
        )
    )

    assert set(captured) == {0}
    (
        header,
        segments,
    ) = captured[0]
    assert int(header.edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
    assert len(segments) == 2
    assert get_backend().to_numpy(segments[0]).tolist() == [10, 20]
    assert get_backend().to_numpy(segments[1]).tolist() == pytest.approx([0.1, 0.2])


def test_mirror_backflow_drops_static_padding_receipt_rows():
    backend = get_backend()
    world_size = 3
    receipt = SymmetryReceipt(
        world_size,
        jnp.array([2, world_size, 0, world_size, 2, 1], dtype=jnp.int32),
    )
    grads = jnp.array(
        [
            [20.0, 20.5],
            [91.0, 91.5],
            [0.0, 0.5],
            [93.0, 93.5],
            [22.0, 22.5],
            [10.0, 10.5],
        ],
        dtype=jnp.float32,
    )

    plan = list(
        AsyncDataDistributor(local_rank=0)._plan_mirror_group(
            receipt=receipt,
            payloads=[grads],
            has_meta_tensor_state_list=[0],
            compact_desc_info=[None],
            global_shapes_list=[tuple(grads.shape)],
            source_steps_list=[0],
        )
    )

    actual = {}
    for ranks, tensors, counts, *_rest in plan:
        rank = int(ranks[0])
        count = int(counts[0]) if counts else 0
        actual[rank] = backend.to_numpy(tensors[0][:count]).tolist() if count else []

    assert actual == {
        0: [[0.0, 0.5]],
        1: [[10.0, 10.5]],
        2: [[20.0, 20.5], [22.0, 22.5]],
    }


def test_content_source_to_static_target_keeps_origin_descriptor_symbolic(monkeypatch):
    backend = get_backend()
    src_mesh = DeviceMesh(("x",), (2,), [0, 1])
    dst_mesh = DeviceMesh(("y",), (2,), [0, 1])
    source = ShardingSpec.hash(P("x")).bind(src_mesh)
    target = ShardingSpec.chunk(P("y")).bind(dst_mesh)
    tt = TransportTensor(
        backend.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=backend.float32()),
        origin_indices=ContinuousIndex.chunk(2, 2),
        global_shape=(4, 2),
        source_step=2,
    )

    def forbidden(self):
        raise AssertionError("ContinuousIndex.materialize should not be called")

    monkeypatch.setattr(ContinuousIndex, "materialize", forbidden)

    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            tt,
            source,
            target,
            tuuid=201,
            step=2,
            sender_spmd_code=7,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )
    )

    non_empty = [
        (target_rank, header.edge_i32[0, EDGE_COMPACT_COUNT], payload_iovs[0].shape[0])
        for target_rank, (header, payload_iovs) in captured.items()
        if payload_iovs[0].shape[0] > 0
    ]
    assert non_empty == [(1, 4, 2)]


def test_mirror_routing_builds_single_expected_frame():
    mesh = DeviceMesh(device_shape=(1,), axis_names=("worker",))
    receipt = SymmetryReceipt(world_size=43, sender_rank_tags=jnp.array([42, 42, 42]))
    data = jnp.array([[1.0], [2.0], [3.0]])
    tensor = TransportTensor(data, receipt=receipt, source_step=1)
    spec = ShardingSpec.follow("source.input").bind(mesh)

    distributor = AsyncDataDistributor(local_rank=0)
    distributor.plan_and_shard([tensor], source=spec, target=spec, tuuid=[123])
    frame = distributor._build_transport_frame_for_target(
        target_id=42,
        groups=distributor._dest_agg[42],
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
    )

    assert int(frame.edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
    assert int(frame.payload_i32[0, PAYLOAD_VALID_COUNT]) == 3
    np.testing.assert_allclose(frame.payload_segments[0], data)
