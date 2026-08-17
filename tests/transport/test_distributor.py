import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.coherence import CoherenceInstance, CoherenceSubject
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import (
    HybridBackend,
    get_backend,
    set_backend,
    set_hybrid_device,
)
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.transport.communication import (
    AsyncDataDistributor,
    TransportRoutingMode,
    _pack_payloads_by_plan,
)
from fluxway.transport.manifest import (
    ENTRY_EXPECTED_COUNT,
    ENTRY_STATE_FLAGS,
    PAYLOAD_VALID_COUNT,
)
from fluxway.transport.coherence import entry_layout_state


@pytest.fixture(autouse=True)
def use_hybrid_backend():
    set_backend("hybrid")
    yield


async def _capture(local_rank, local_data, source, target, **kwargs):
    captured = {}
    codes = kwargs.get("transfer_codes")
    codes = tuple(codes) if isinstance(codes, list | tuple) else (int(codes),)
    tensors = [local_data] if isinstance(local_data, TransportTensor) else local_data
    if tensors is not None:
        for code, tensor in zip(codes, tensors, strict=True):
            if not tensor._coherence_instances:
                tensor._bind_coherence_instances(
                    (
                        CoherenceInstance(
                            CoherenceSubject(
                                int(kwargs["sender_spmd_code"]),
                                int(kwargs["sender_rank"]),
                                code,
                            ),
                            int(kwargs.get("step", tensor.source_step or 0)),
                        ),
                    )
                )

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
            transfer_codes=[101, 102],
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
    assert int(header.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 1
    assert len(segments) == 2
    assert get_backend().to_numpy(segments[0]).tolist() == [10, 20]
    assert get_backend().to_numpy(segments[1]).tolist() == pytest.approx([0.1, 0.2])


def test_cuda_pack_preserves_device_plan_domain():
    cp = pytest.importorskip("cupy")
    try:
        backend = HybridBackend(default_device="cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
        return
    payload = cp.asarray([10, 20, 30, 40], dtype=cp.int32)

    flat_buffers, counts, offsets, send_offsets = _pack_payloads_by_plan(
        backend=backend,
        payloads_tuple=(payload,),
        nested_length_list=(None,),
        payload_routing_axes=((0,),),
        static_shapes=(4,),
        sort_indices=cp.asarray([0, 2, 1, 3], dtype=cp.int32),
        send_counts=cp.asarray([2, 2], dtype=cp.int32),
    )

    reordered = flat_buffers[0][0]
    assert all(
        isinstance(value, cp.ndarray)
        for value in (reordered, counts[0], offsets[0], send_offsets)
    )
    cp.testing.assert_array_equal(reordered, cp.asarray([10, 30, 20, 40]))
    cp.testing.assert_array_equal(counts[0], cp.asarray([2, 2]))
    cp.testing.assert_array_equal(offsets[0], cp.asarray([0, 2]))


def test_cuda_ragged_pack_adopts_numpy_receipt_plan_once():
    cp = pytest.importorskip("cupy")
    try:
        backend = HybridBackend(default_device="cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
        return
    payload = cp.asarray([10, 11, 20], dtype=cp.int32)
    row_lengths = cp.asarray([2, 1], dtype=cp.int32)

    flat_buffers, counts, offsets, send_offsets = _pack_payloads_by_plan(
        backend=backend,
        payloads_tuple=(payload,),
        nested_length_list=((row_lengths,),),
        payload_routing_axes=((0,),),
        static_shapes=(2, 3),
        sort_indices=np.asarray([1, 0], dtype=np.int32),
        send_counts=np.asarray([1, 1], dtype=np.int32),
    )

    reordered, reordered_levels = flat_buffers[0]
    assert reordered_levels is not None
    assert all(
        isinstance(value, cp.ndarray)
        for value in (
            reordered,
            reordered_levels[0],
            *counts,
            *offsets,
        )
    )
    assert isinstance(send_offsets, np.ndarray)
    cp.testing.assert_array_equal(reordered, cp.asarray([20, 10, 11]))
    cp.testing.assert_array_equal(reordered_levels[0], cp.asarray([1, 2]))


def test_cuda_frame_retentions_header_on_host_and_all_payloads_on_device():
    cp = pytest.importorskip("cupy")
    try:
        set_hybrid_device("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    source_mesh = DeviceMesh(("source",), (1,), [0])
    target_mesh = DeviceMesh(("target",), (2,), [0, 1])
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)
    tensor = TransportTensor(
        flat_values=cp.asarray([10, 11, 20], dtype=cp.int32),
        nested_splits=[cp.asarray([0, 2, 3], dtype=cp.int32)],
        receipt=SymmetryReceipt(2, 0),
        source_step=0,
    )

    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            tensor,
            source,
            target,
            transfer_codes=17,
            step=0,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )

    assert set(captured) == {0, 1}
    for header, payload_iovs in captured.values():
        assert all(
            isinstance(table, np.ndarray)
            for table in (
                header.frame_i32,
                header.entry_i32,
                header.payload_i32,
                header.shape_i32,
                header.global_shape_i64,
                header.compact_i64,
            )
        )
        assert all(isinstance(payload, cp.ndarray) for payload in payload_iovs)
    assert captured[0][1]


def test_mirror_backflow_drops_static_padding_receipt_rows():
    backend = get_backend()
    world_size = 3
    source_mesh = DeviceMesh(("source",), (1,), [0])
    target_mesh = DeviceMesh(("target",), (world_size,), list(range(world_size)))
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)
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

    tensor = TransportTensor(grads, receipt=receipt, source_step=0)
    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            tensor,
            source,
            target,
            transfer_codes=1,
            step=0,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )

    actual = {
        rank: backend.to_numpy(payloads[0]).tolist()
        for rank, (_header, payloads) in captured.items()
    }

    assert actual == {
        0: [[0.0, 0.5]],
        1: [[10.0, 10.5]],
        2: [[20.0, 20.5], [22.0, 22.5]],
    }


def test_receipt_routing_preserves_complete_ragged_rows():
    source_mesh = DeviceMesh(("source",), (1,), [0])
    target_mesh = DeviceMesh(("target",), (2,), [0, 1])
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)
    tensor = TransportTensor(
        flat_values=jnp.array([10, 11, 20], dtype=jnp.int32),
        nested_splits=[jnp.array([0, 2, 3], dtype=jnp.int32)],
        receipt=SymmetryReceipt(2, jnp.array([0, 1], dtype=jnp.int32)),
        source_step=0,
    )

    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            tensor,
            source,
            target,
            transfer_codes=17,
            step=0,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )

    assert get_backend().to_numpy(captured[0][1][0]).tolist() == [10, 11]
    assert get_backend().to_numpy(captured[0][1][1]).tolist() == [2]
    assert get_backend().to_numpy(captured[1][1][0]).tolist() == [20]
    assert get_backend().to_numpy(captured[1][1][1]).tolist() == [1]


def test_grouped_numpy_receipt_avoids_payload_sorting():
    world_size = 3
    source_mesh = DeviceMesh(("source",), (1,), [0])
    target_mesh = DeviceMesh(("target",), (world_size,), list(range(world_size)))
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)
    tags = np.array([0, 0, 1, 2, 2, world_size], dtype=np.int32)
    values = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    tensor = TransportTensor(
        values,
        receipt=SymmetryReceipt(world_size, tags),
        source_step=4,
    )
    (_result, sort_indices), captured = asyncio.run(
        _capture(
            0,
            tensor,
            source,
            target,
            transfer_codes=1,
            step=4,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )

    assert sort_indices is None
    actual = {
        rank: np.asarray(payloads[0]).tolist()
        for rank, (_header, payloads) in captured.items()
    }
    assert actual == {
        0: [[0.0, 1.0], [2.0, 3.0]],
        1: [[4.0, 5.0]],
        2: [[6.0, 7.0], [8.0, 9.0]],
    }


def test_unordered_numpy_mirror_retentions_general_sorting_contract():
    backend = get_backend()
    world_size = 3
    source_mesh = DeviceMesh(("source",), (1,), [0])
    target_mesh = DeviceMesh(("target",), (world_size,), list(range(world_size)))
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)
    receipt = SymmetryReceipt(
        world_size,
        np.array([2, 0, 2, 1], dtype=np.int32),
    )
    values = jnp.array([[20.0], [0.0], [21.0], [10.0]], dtype=jnp.float32)

    tensor = TransportTensor(values, receipt=receipt, source_step=4)
    (_result, _sort), captured = asyncio.run(
        _capture(
            0,
            tensor,
            source,
            target,
            transfer_codes=1,
            step=4,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )

    actual = {
        rank: backend.to_numpy(payloads[0]).reshape(-1).tolist()
        for rank, (_header, payloads) in captured.items()
    }
    assert actual == {0: [0.0], 1: [10.0], 2: [20.0, 21.0]}


def test_content_source_to_static_target_retentions_origin_descriptor_symbolic(
    monkeypatch,
):
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
            transfer_codes=201,
            step=2,
            sender_spmd_code=7,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )
    )

    non_empty = [
        (
            target_rank,
            4
            if entry_layout_state(int(header.entry_i32[0, ENTRY_STATE_FLAGS])) & 4
            else 0,
            payload_iovs[0].shape[0],
        )
        for target_rank, (header, payload_iovs) in captured.items()
        if payload_iovs[0].shape[0] > 0
    ]
    assert non_empty == [(1, 4, 2)]


def test_mirror_routing_builds_single_expected_frame():
    source_mesh = DeviceMesh(device_shape=(1,), axis_names=("source",))
    target_mesh = DeviceMesh(device_shape=(43,), axis_names=("target",))
    receipt = SymmetryReceipt(world_size=43, sender_rank_tags=jnp.array([42, 42, 42]))
    data = jnp.array([[1.0], [2.0], [3.0]])
    tensor = TransportTensor(data, receipt=receipt, source_step=1)
    tensor._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(1, 0, 123), 1),)
    )
    source = ShardingSpec.chunk(P()).bind(source_mesh)
    target = ShardingSpec.chunk(P("target")).bind(target_mesh)

    distributor = AsyncDataDistributor(local_rank=0)
    asyncio.run(
        distributor.distribute(
            [tensor],
            source=source,
            target=target,
            transfer_codes=[123],
            routing_mode=TransportRoutingMode.RECEIPT,
        )
    )
    frame = distributor._build_transport_frame_for_target(
        target_id=42,
        groups=distributor._dest_agg[42],
        step=1,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
    )

    assert int(frame.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 1
    assert int(frame.payload_i32[0, PAYLOAD_VALID_COUNT]) == 3
    np.testing.assert_allclose(frame.payload_segments[0], data)  # type: ignore[no-matching-overload]
