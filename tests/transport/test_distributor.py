import asyncio

import jax.numpy as jnp
import pytest

from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import get_backend, set_backend
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.transport.communication import AsyncDataDistributor
from fluxway.transport.manifest import EDGE_COMPACT_COUNT, EDGE_EXPECTED_COUNT


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


async def _capture(local_rank, local_data, source, target, **kwargs):
    captured = {}

    async def send_fn(
        target_rank,
        frame_i32,
        edge_i32,
        payload_i32,
        shape_i32,
        global_shape_i64,
        compact_i64,
        *payload_segments,
    ):
        captured[target_rank] = (
            frame_i32,
            edge_i32,
            payload_i32,
            shape_i32,
            global_shape_i64,
            compact_i64,
            payload_segments,
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
    keys = TransportTensor(jnp.array([10, 20], dtype=jnp.int32), global_shape=(4,))
    vals = TransportTensor(jnp.array([0.1, 0.2], dtype=jnp.float32), global_shape=(4,))

    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            [keys, vals],
            source,
            target,
            tuuid=[101, 102],
            step=1,
            sender_spmd_code=6,
            sender_world_size=mesh.size,
        )
    )

    assert set(captured) == {0}
    (
        _frame_i32,
        edge_i32,
        _payload_i32,
        _shape_i32,
        _global_shape_i64,
        _compact_i64,
        segments,
    ) = captured[0]
    assert int(edge_i32[0, EDGE_EXPECTED_COUNT]) == 1
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
            sender_world_size=src_mesh.size,
        )
    )

    non_empty = [
        (target_rank, frame[1][0, EDGE_COMPACT_COUNT], frame[6][0].shape[0])
        for target_rank, frame in captured.items()
        if frame[6][0].shape[0] > 0
    ]
    assert non_empty == [(1, 4, 2)]
