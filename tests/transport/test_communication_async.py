import asyncio

import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.ragged import ragged_level_count
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.common.types.tensor import TransportTensor
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.communication import AsyncDataDistributor
from fluxway.transport.coherence import entry_layout_state, entry_tensor_kind
from fluxway.transport.manifest import (
    ENTRY_EXPECTED_COUNT,
    ENTRY_GLOBAL_SHAPE_BEGIN,
    ENTRY_GLOBAL_SHAPE_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_AUX,
    ENTRY_STATE_FLAGS,
    FRAME_ENTRY_COUNT,
    FRAME_FLAGS,
    FRAME_PAYLOAD_COUNT,
    FRAME_SENDER_SPMD_CODE,
    FrameFlag,
    PAYLOAD_ROLE,
    PayloadRole,
    TensorEntryKind,
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


def _with_source_step(tensor: TransportTensor, step: int) -> TransportTensor:
    tensor.source_step = step
    return tensor


def test_async_distributor_uses_only_flat_frame_abi():
    mesh = DeviceMesh(axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1]))
    spec = ShardingSpec.chunk(P("x")).bind(mesh)
    data = TransportTensor(
        jnp.arange(10, dtype=jnp.float32),
        global_shape=(20,),
        source_step=3,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=spec,
            target=spec,
            transfer_codes=17,
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
    assert int(header.frame_i32[FRAME_FLAGS]) == int(FrameFlag.TENSOR_TRANSFER)
    assert int(header.entry_i32[0, ENTRY_ROUTE_CODE]) == 17
    assert TensorEntryKind(
        entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
    ) is (TensorEntryKind.TENSOR)
    assert int(header.frame_i32[FRAME_ENTRY_COUNT]) == 1
    assert int(header.frame_i32[FRAME_PAYLOAD_COUNT]) == 1
    assert header.entry_i32.dtype == np.int32
    assert header.payload_i32.dtype == np.int32
    assert header.shape_i32.dtype == np.int32
    assert header.global_shape_i64.dtype == np.int64
    assert header.compact_i64.dtype == np.int64
    assert len(segments) == 1


def test_async_distributor_batch_metadata_is_encoded_in_frame_rows():
    mesh = DeviceMesh(axis_names=("x",), device_shape=(1,), device_ids=np.array([0]))
    spec = ShardingSpec.chunk(P("x")).bind(mesh)
    dense = TransportTensor(
        jnp.arange(2, dtype=jnp.float32),
        global_shape=(2,),
        source_step=4,
    )
    ragged = TransportTensor(
        jnp.arange(5, dtype=jnp.float32),
        nested_splits=[jnp.array([0, 2, 5], dtype=jnp.int32)],
        global_shape=(2,),
        source_step=4,
    )
    origin = TransportTensor(
        jnp.arange(2, dtype=jnp.float32),
        origin_indices=jnp.array([7, 8], dtype=jnp.int32),
        global_shape=(10,),
        source_step=4,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=[dense, ragged, origin],
            source=spec,
            target=spec,
            transfer_codes=[21, 22, 23],
            step=4,
            sender_spmd_code=2,
            sender_rank=0,
            sender_world_size=1,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    header, _payload_iovs = captured[0]
    entry_i32 = header.entry_i32
    payload_i32 = header.payload_i32
    assert entry_i32[:, ENTRY_ROUTE_CODE].tolist() == [21, 22, 23]
    assert entry_layout_state(int(entry_i32[0, ENTRY_STATE_FLAGS])) == 0
    assert (
        ragged_level_count(entry_layout_state(int(entry_i32[1, ENTRY_STATE_FLAGS])))
        == 1
    )
    assert bool(entry_layout_state(int(entry_i32[1, ENTRY_STATE_FLAGS])) & 1)
    assert entry_layout_state(int(entry_i32[2, ENTRY_STATE_FLAGS])) == 2
    assert payload_i32[:, PAYLOAD_ROLE].tolist() == [
        int(PayloadRole.VALUES),
        int(PayloadRole.VALUES),
        int(PayloadRole.RAGGED_LENGTHS),
        int(PayloadRole.VALUES),
        int(PayloadRole.ORIGIN),
    ]


@pytest.mark.parametrize(
    "order",
    (
        (("dp", 237), ("replicated", 238)),
        (("replicated", 238), ("dp", 237)),
    ),
)
def test_late_aggregation_preserves_expected_count_per_transfer(order) -> None:
    source_mesh = DeviceMesh(("dp",), (8,), list(range(8)))
    target_mesh = DeviceMesh(("dp",), (1,), [0])
    distributor = AsyncDataDistributor(local_rank=0)

    for kind, transfer_code in order:
        if kind == "dp":
            source = ShardingSpec.chunk(P("dp")).bind(source_mesh)
            target = ShardingSpec.chunk(P("dp")).bind(target_mesh)
            tensor = TransportTensor(
                np.asarray([10, 11], dtype=np.int64),
                global_shape=(16,),
                source_step=0,
            )
        else:
            source = ShardingSpec.chunk(P()).bind(source_mesh)
            target = ShardingSpec.chunk(P()).bind(target_mesh)
            tensor = TransportTensor(
                np.asarray(16, dtype=np.int32), global_shape=(), source_step=0
            )
        asyncio.run(
            distributor.distribute(
                local_data=tensor,
                source=source,
                target=target,
                transfer_codes=transfer_code,
            )
        )

    expected = {237: 8, 238: 1}
    _target_id, plain_header, plain_payloads = distributor.build_transport_frame_args(
        step=0,
        sender_spmd_code=1,
        sender_rank=0,
        sender_world_size=8,
    )[0]
    plain = plain_header.to_frame(plain_payloads)
    template_cache = {}
    _target_id, template_cold_header, template_cold_payloads = (
        distributor.build_transport_frame_args(
            step=0,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=8,
            template_cache=template_cache,
        )[0]
    )
    template_cold = template_cold_header.to_frame(template_cold_payloads)
    _target_id, template_warm_header, template_warm_payloads = (
        distributor.build_transport_frame_args(
            step=1,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=8,
            template_cache=template_cache,
        )[0]
    )
    template_warm = template_warm_header.to_frame(template_warm_payloads)
    for frame in (plain, template_cold, template_warm):
        assert {
            int(entry[ENTRY_ROUTE_CODE]): int(entry[ENTRY_EXPECTED_COUNT])
            for entry in frame.entry_i32
        } == expected
        entry_by_transfer = {
            int(entry[ENTRY_ROUTE_CODE]): entry for entry in frame.entry_i32
        }
        assert int(entry_by_transfer[237][ENTRY_AUX]) == 0
        assert int(entry_by_transfer[238][ENTRY_AUX]) == 0
        assert entry_layout_state(int(entry_by_transfer[237][ENTRY_STATE_FLAGS])) == 0
        assert entry_layout_state(int(entry_by_transfer[238][ENTRY_STATE_FLAGS])) == 0
        dp_shape_begin = int(entry_by_transfer[237][ENTRY_GLOBAL_SHAPE_BEGIN])
        dp_shape_count = int(entry_by_transfer[237][ENTRY_GLOBAL_SHAPE_COUNT])
        scalar_shape_begin = int(entry_by_transfer[238][ENTRY_GLOBAL_SHAPE_BEGIN])
        scalar_shape_count = int(entry_by_transfer[238][ENTRY_GLOBAL_SHAPE_COUNT])
        assert frame.global_shape_i64[
            dp_shape_begin : dp_shape_begin + dp_shape_count
        ].tolist() == [16]
        assert scalar_shape_count == 0
        assert scalar_shape_begin >= 0

    mailbox = WorkerMailbox()
    mailbox.declare_step("step", 0, expected, [])
    asyncio.run(mailbox.deposit_transport_frame("step", "Input", plain))


def test_async_distributor_nested_ragged_preserves_payload_dtype_and_levels():
    src_mesh = DeviceMesh(
        axis_names=("x",), device_shape=(1,), device_ids=np.array([0])
    )
    dst_mesh = DeviceMesh(
        axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1])
    )
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)
    data = _with_source_step(
        TransportTensor.from_lists(
            [[[2**40, 2**40 + 1], [2**40 + 2]], [[2**40 + 3]]],
            dtype=np.uint64,
            global_shape=(2,),
        ),
        7,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            transfer_codes=91,
            step=7,
            sender_spmd_code=3,
            sender_rank=0,
            sender_world_size=1,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1}
    for header, payload_iovs in captured.values():
        assert int(header.frame_i32[FRAME_ENTRY_COUNT]) == 1
        assert int(header.frame_i32[FRAME_PAYLOAD_COUNT]) == 3
        assert int(header.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 1
        assert (
            ragged_level_count(
                entry_layout_state(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            == 2
        )
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
    data = TransportTensor(
        jnp.arange(5, dtype=jnp.int32),
        global_shape=(10,),
        source_step=8,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            transfer_codes=92,
            step=8,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1}
    assert int(captured[0][0].entry_i32[0, ENTRY_EXPECTED_COUNT]) == 1
    assert int(captured[1][0].entry_i32[0, ENTRY_EXPECTED_COUNT]) == 2


def test_transport_frame_dynamic_manifest_marks_unrouted_targets_as_routing_miss():
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("y",))
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.hash(P("y")).bind(dst_mesh)
    data = TransportTensor(
        jnp.array([0, 1, 4], dtype=jnp.int32),
        global_shape=(6,),
        source_step=9,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            transfer_codes=93,
            step=9,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())
    assert set(captured) == {0, 1, 2, 3}
    for frame in captured.values():
        entry_i32 = frame[0].entry_i32
        assert int(entry_i32[0, ENTRY_EXPECTED_COUNT]) == src_mesh.size
    assert (
        TensorEntryKind(
            entry_tensor_kind(int(captured[2][0].entry_i32[0, ENTRY_STATE_FLAGS]))
        )
        is TensorEntryKind.ROUTING_MISS
    )
    assert (
        TensorEntryKind(
            entry_tensor_kind(int(captured[3][0].entry_i32[0, ENTRY_STATE_FLAGS]))
        )
        is TensorEntryKind.ROUTING_MISS
    )


def test_dynamic_zero_tensor_is_routing_miss_not_completed_without_tensor():
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("y",))
    source_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    target_spec = ShardingSpec.hash(P("y")).bind(dst_mesh)
    data = TransportTensor(
        jnp.empty((0,), dtype=jnp.int32),
        global_shape=(0,),
        source_step=10,
    )

    async def run_test():
        return await _capture_frame(
            AsyncDataDistributor(local_rank=0),
            local_data=data,
            source=source_spec,
            target=target_spec,
            transfer_codes=94,
            step=10,
            sender_spmd_code=1,
            sender_rank=0,
            sender_world_size=src_mesh.size,
        )

    (_results, _sort), captured = asyncio.run(run_test())

    assert set(captured) == {0, 1, 2, 3}
    assert {
        TensorEntryKind(entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS])))
        for header, _ in captured.values()
    } == {TensorEntryKind.ROUTING_MISS}


@pytest.mark.parametrize(
    ("method_name", "expected_kind"),
    [
        ("plan_routing_miss", TensorEntryKind.ROUTING_MISS),
        (
            "plan_completed_without_tensor",
            TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
        ),
    ],
)
def test_absent_tensor_planning_preserves_causal_kind(method_name, expected_kind):
    distributor = AsyncDataDistributor(local_rank=0)
    getattr(distributor, method_name)(
        target_rank=0,
        expected_count=1,
        transfer_code=94,
        payload=np.empty((0,), dtype=np.int32),
        source_step=10,
    )

    target_rank, header, payloads = distributor.build_transport_frame_args(
        step=10,
        sender_spmd_code=1,
        sender_rank=0,
        sender_world_size=1,
    )[0]

    assert target_rank == 0
    assert (
        TensorEntryKind(entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS])))
        is expected_kind
    )
    assert payloads[0].shape == (0,)
