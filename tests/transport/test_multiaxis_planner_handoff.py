import asyncio
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.common.coherence import CoherenceInstance, CoherenceSubject
from fluxway.common.execution_layout import ExecutionGroup
from fluxway.common.logical_partition_binding import LogicalPartitionBinding
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import get_backend, use_backend
from fluxway.common.types.sharding import (
    AlgebraicRouting,
    FunctionPartition,
    IdentityIndices,
    LiteralIndices,
    P,
    RangeIndices,
    RoutingDomain,
    ShardingSpec,
)
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.transport.communication import (
    AsyncDataDistributor,
    TransportRoutingMode,
)
from fluxway.transport.manifest import (
    ENTRY_STATE_FLAGS,
    TensorEntryKind,
    entry_tensor_kind,
)
from fluxway.transport.planner import TransportPlanner


@pytest.fixture(autouse=True)
def use_hybrid_backend():
    with use_backend("hybrid"), jax.default_device(jax.devices("cpu")[0]):
        yield


def test_multiaxis_flatten_routing_domain_derivation_and_plan():
    mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 3))
    layout = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(mesh)

    domain_0 = RoutingDomain.from_layout(
        target=layout,
        axis_sizes=(12,),
        axis_indices=(RangeIndices.contiguous(start=0, length=6),),
    )
    assert len(domain_0.axes) == 1
    assert domain_0.axes[0].mesh_axes == ("dp", "fsdp")
    assert domain_0.axes[0].num_workers == 6

    plan_0 = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_0)
    assert plan_0.target_num_workers == 6
    assert plan_0.send_counts.tolist() == [2, 2, 2, 0, 0, 0]
    assert int(plan_0.sort_indices.shape[0]) == 6

    domain_1 = RoutingDomain.from_layout(
        target=layout,
        axis_sizes=(12,),
        axis_indices=(RangeIndices.contiguous(start=6, length=6),),
    )
    plan_1 = TransportPlanner.compute_plan(src_local_rank=1, domain=domain_1)
    assert plan_1.target_num_workers == 6
    assert plan_1.send_counts.tolist() == [0, 0, 0, 2, 2, 2]
    assert int(plan_1.sort_indices.shape[0]) == 6


def test_multiaxis_flatten_axis_ordering_permutations():
    mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 3))

    layout_dp_fsdp = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(mesh)
    domain_a = RoutingDomain.from_layout(
        target=layout_dp_fsdp,
        axis_sizes=(6,),
        axis_indices=(IdentityIndices(6),),
    )
    plan_a = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_a)
    np.testing.assert_array_equal(
        np.asarray(plan_a.dest_devices).ravel(),
        [0, 1, 2, 3, 4, 5],
    )

    layout_fsdp_dp = ShardingSpec.chunk(P(("fsdp", "dp"))).bind(mesh)
    domain_b = RoutingDomain.from_layout(
        target=layout_fsdp_dp,
        axis_sizes=(6,),
        axis_indices=(IdentityIndices(6),),
    )
    plan_b = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_b)
    np.testing.assert_array_equal(
        np.asarray(plan_b.dest_devices).ravel(),
        [0, 3, 1, 4, 2, 5],
    )


def test_multiaxis_flatten_with_replica_fanout():
    mesh = DeviceMesh(axis_names=("dp", "fsdp", "replica"), device_shape=(2, 2, 2))
    layout = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(mesh)
    assert layout.replica_size == 2
    assert layout.replica_axes is not None
    assert set(layout.replica_axes) == {"replica"}

    domain = RoutingDomain.from_layout(
        target=layout,
        axis_sizes=(4,),
        axis_indices=(IdentityIndices(4),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)
    assert plan.target_num_workers == 8
    assert plan.replication_factor == 2
    assert plan.dest_devices.shape == (4, 2)
    assert plan.sort_indices.shape[0] == 4
    assert plan.send_counts.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


def test_multiaxis_dynamic_function_partition_handoff():
    mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))

    def modulo_route(keys, *, partition_count: int, domain_size: int):
        del domain_size
        return keys % partition_count

    func_partition = FunctionPartition(target=modulo_route)
    layout = ShardingSpec(
        routing=AlgebraicRouting(strategy=func_partition, partition=P(("dp", "fsdp"))),
    ).bind(mesh)

    keys = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    domain = RoutingDomain.from_layout(
        target=layout,
        axis_sizes=(8,),
        axis_indices=(LiteralIndices(length=8, values=keys),),
    )

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)
    assert plan.target_num_workers == 4
    np.testing.assert_array_equal(
        np.asarray(plan.dest_devices).ravel(),
        [0, 1, 2, 3, 0, 1, 2, 3],
    )
    assert plan.send_counts.tolist() == [2, 2, 2, 2]


def test_multiaxis_physical_handoff_scheduling_noncontiguous_ranks():
    target_physical_ranks = (100, 101, 200, 201)
    logical_mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))
    group = ExecutionGroup.create(target_physical_ranks, logical_mesh)

    src_mesh = DeviceMesh(axis_names=("dp",), device_shape=(2,))
    src_layout = ShardingSpec.chunk(P("dp")).bind(src_mesh)
    target_layout = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(logical_mesh)

    delivered_frames: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in target_physical_ranks
    }

    async def run_distribution():
        async def make_send_fn(src_physical_rank: int):
            async def send_fn(target_logical_id: int, header, *payload_iovs):
                target_p_rank = group.physical_rank(target_logical_id)
                delivered_frames[target_p_rank].append(
                    {
                        "from_src": src_physical_rank,
                        "header": header,
                        "payloads": payload_iovs,
                    }
                )
                return True

            return send_fn

        for src_rank in (0, 1):
            data_keys = jnp.arange(src_rank * 8, (src_rank + 1) * 8, dtype=jnp.int32)
            data_vals = jnp.arange(src_rank * 8, (src_rank + 1) * 8, dtype=jnp.float32)

            tt_keys = TransportTensor(data_keys, global_shape=(16,), source_step=1)
            tt_vals = TransportTensor(data_vals, global_shape=(16,), source_step=1)

            tt_keys._bind_coherence_instances(
                (CoherenceInstance(CoherenceSubject(1, src_rank, 101), 1),)
            )
            tt_vals._bind_coherence_instances(
                (CoherenceInstance(CoherenceSubject(1, src_rank, 102), 1),)
            )

            distributor = AsyncDataDistributor(local_rank=src_rank)
            send_fn = await make_send_fn(src_rank)

            await distributor.distribute(
                local_data=[tt_keys, tt_vals],
                source=src_layout,
                target=target_layout,
                send_fn=send_fn,
                transfer_codes=[101, 102],
                step=1,
                sender_spmd_code=1,
                sender_rank=src_rank,
                sender_world_size=2,
            )

    asyncio.run(run_distribution())

    backend = get_backend()
    assert len(delivered_frames[100]) == 1
    assert backend.to_numpy(delivered_frames[100][0]["payloads"][0]).tolist() == [
        0,
        1,
        2,
        3,
    ]

    assert len(delivered_frames[101]) == 1
    assert backend.to_numpy(delivered_frames[101][0]["payloads"][0]).tolist() == [
        4,
        5,
        6,
        7,
    ]

    assert len(delivered_frames[200]) == 1
    assert backend.to_numpy(delivered_frames[200][0]["payloads"][0]).tolist() == [
        8,
        9,
        10,
        11,
    ]

    assert len(delivered_frames[201]) == 1
    assert backend.to_numpy(delivered_frames[201][0]["payloads"][0]).tolist() == [
        12,
        13,
        14,
        15,
    ]


def test_multiaxis_logical_partition_binding_consistency():
    physical_ranks = (10, 20, 30, 40)
    logical_mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))
    group = ExecutionGroup.create(physical_ranks, logical_mesh)

    binding = LogicalPartitionBinding(
        {p: f"partition_table_{p}" for p in range(4)},
        axes=("dp", "fsdp"),
    )

    for l_rank in range(4):
        p_rank = group.physical_rank(l_rank)
        assert group.logical_rank(p_rank) == l_rank
        resolved = binding.resolve(l_rank, logical_mesh=logical_mesh)
        assert resolved == f"partition_table_{l_rank}"


def test_multiaxis_bidirectional_flow_with_symmetry_receipt():
    cpu_mesh = DeviceMesh(axis_names=("cpu",), device_shape=(2,))
    gpu_mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))

    cpu_ranks = (0, 1)

    src_cpu_spec = ShardingSpec.chunk(P("cpu")).bind(cpu_mesh)
    dst_gpu_spec = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(gpu_mesh)

    received_by_cpu = {r: [] for r in cpu_ranks}

    async def run_reverse_handoff():
        async def cpu_send_fn(target_rank, header, *payloads):
            kind = TensorEntryKind(
                entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            received_by_cpu[target_rank].append((kind, payloads))
            return True

        grad_0 = TransportTensor(
            jnp.array([1.1, 1.2], dtype=jnp.float32),
            receipt=SymmetryReceipt(
                world_size=2, sender_rank_tags=jnp.array([0, 0], dtype=jnp.int32)
            ),
            source_step=1,
        )
        grad_0._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(2, 0, 99), 1),)
        )
        dist_gpu0 = AsyncDataDistributor(local_rank=0)
        await dist_gpu0.distribute(
            local_data=[grad_0],
            source=dst_gpu_spec,
            target=src_cpu_spec,
            send_fn=cpu_send_fn,
            transfer_codes=[99],
            step=1,
            sender_spmd_code=2,
            sender_rank=0,
            sender_world_size=4,
            routing_mode=TransportRoutingMode.RECEIPT,
        )

        grad_2 = TransportTensor(
            jnp.array([3.1, 3.2], dtype=jnp.float32),
            receipt=SymmetryReceipt(
                world_size=2, sender_rank_tags=jnp.array([1, 1], dtype=jnp.int32)
            ),
            source_step=1,
        )
        grad_2._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(2, 2, 99), 1),)
        )
        dist_gpu2 = AsyncDataDistributor(local_rank=2)
        await dist_gpu2.distribute(
            local_data=[grad_2],
            source=dst_gpu_spec,
            target=src_cpu_spec,
            send_fn=cpu_send_fn,
            transfer_codes=[99],
            step=1,
            sender_spmd_code=2,
            sender_rank=2,
            sender_world_size=4,
            routing_mode=TransportRoutingMode.RECEIPT,
        )

    asyncio.run(run_reverse_handoff())

    backend = get_backend()
    assert len(received_by_cpu[0]) == 2
    kinds_0 = [k for k, _ in received_by_cpu[0]]
    assert TensorEntryKind.TENSOR in kinds_0
    assert TensorEntryKind.ROUTING_MISS in kinds_0

    data_0 = [p for k, p in received_by_cpu[0] if k == TensorEntryKind.TENSOR][0]
    np.testing.assert_allclose(
        backend.to_numpy(data_0[0]),
        [1.1, 1.2],
        rtol=1e-5,
    )

    assert len(received_by_cpu[1]) == 2
    kinds_1 = [k for k, _ in received_by_cpu[1]]
    assert TensorEntryKind.TENSOR in kinds_1
    assert TensorEntryKind.ROUTING_MISS in kinds_1

    data_1 = [p for k, p in received_by_cpu[1] if k == TensorEntryKind.TENSOR][0]
    np.testing.assert_allclose(
        backend.to_numpy(data_1[0]),
        [3.1, 3.2],
        rtol=1e-5,
    )


def test_multiaxis_e2e_training_flow_rule_embedding_convergence():
    backend = get_backend()
    embedding_dim = 4
    learning_rate = 0.05
    rng = np.random.RandomState(42)

    W_ref = rng.randn(8, embedding_dim).astype(np.float32)
    W_cpu0 = W_ref[0:4].copy()
    W_cpu1 = W_ref[4:8].copy()
    W_dense = rng.randn(embedding_dim, 1).astype(np.float32)

    batch_A = np.array([1, 2, 1, 3, 3, 1, 2], dtype=np.int32)
    batch_B = np.array([4, 6, 1, 2, 5, 2, 4], dtype=np.int32)
    targets_A = np.ones((len(batch_A), 1), dtype=np.float32)
    targets_B = np.full((len(batch_B), 1), 2.0, dtype=np.float32)

    emb_A_ref = W_ref[batch_A]
    pred_A_ref = emb_A_ref @ W_dense
    grad_emb_A_ref = (pred_A_ref - targets_A) @ W_dense.T

    emb_B_ref = W_ref[batch_B]
    pred_B_ref = emb_B_ref @ W_dense
    grad_emb_B_ref = (pred_B_ref - targets_B) @ W_dense.T

    grad_W_ref = np.zeros_like(W_ref)
    np.add.at(grad_W_ref, batch_A, grad_emb_A_ref)
    np.add.at(grad_W_ref, batch_B, grad_emb_B_ref)
    W_ref_updated = W_ref - learning_rate * grad_W_ref

    cpu_ranks = (0, 1)
    cpu_mesh = DeviceMesh(axis_names=("cpu",), device_shape=(2,))
    cpu_spec = ShardingSpec.chunk(P("cpu")).bind(cpu_mesh)

    gpu_ranks = (100, 101, 200, 201)
    gpu_mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))
    gpu_group = ExecutionGroup.create(gpu_ranks, gpu_mesh)

    def route_to_target(keys, *, partition_count: int, domain_size: int):
        del partition_count, domain_size
        return keys

    func_partition = FunctionPartition(target=route_to_target)
    target_layout = ShardingSpec(
        routing=AlgebraicRouting(strategy=func_partition, partition=P(("dp", "fsdp"))),
    ).bind(gpu_mesh)
    gpu_spec = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(gpu_mesh)

    mailboxes = {r: asyncio.Queue() for r in list(cpu_ranks) + list(gpu_ranks)}

    def make_cpu_to_gpu_send_fn(src_cpu_rank: int):
        async def send_fn(target_logical_id: int, header, *payload_iovs):
            target_p_rank = gpu_group.physical_rank(target_logical_id)
            await mailboxes[target_p_rank].put((src_cpu_rank, header, payload_iovs))
            return True

        return send_fn

    def make_gpu_to_cpu_send_fn(src_gpu_p_rank: int):
        async def send_fn(target_cpu_rank: int, header, *payload_iovs):
            await mailboxes[target_cpu_rank].put((src_gpu_p_rank, header, payload_iovs))
            return True

        return send_fn

    unique_A, restore_A = np.unique(batch_A, return_inverse=True)
    unique_B, restore_B = np.unique(batch_B, return_inverse=True)

    emb_A_cpu0 = W_cpu0[unique_A]
    emb_B_cpu0 = W_cpu0[unique_B[:2]]
    emb_B_cpu1 = W_cpu1[unique_B[2:] - 4]

    cpu0_dests = np.array([0, 0, 0, 2, 2], dtype=np.int32)
    cpu0_embs = np.concatenate([emb_A_cpu0, emb_B_cpu0], axis=0)

    cpu1_dests = np.array([2, 2, 2], dtype=np.int32)
    cpu1_embs = emb_B_cpu1

    async def run_cpu_sender(src_rank: int, embs: np.ndarray, dest_tags: np.ndarray):
        tt = TransportTensor(
            jnp.asarray(embs),
            routing_indices=jnp.asarray(dest_tags),
            origin_indices=jnp.asarray(dest_tags),
            global_shape=embs.shape,
            source_step=1,
        )
        tt._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(1, src_rank, 100 + src_rank), 1),)
        )
        distributor = AsyncDataDistributor(local_rank=src_rank)
        send_fn = make_cpu_to_gpu_send_fn(src_rank)
        await distributor.distribute(
            local_data=[tt],
            source=cpu_spec,
            target=target_layout,
            send_fn=send_fn,
            transfer_codes=[100 + src_rank],
            step=1,
            sender_spmd_code=1,
            sender_rank=src_rank,
            sender_world_size=2,
        )

    gpu_received_data = {r: {} for r in gpu_ranks}

    async def run_gpu_receiver(gpu_p_rank: int):
        mailbox = mailboxes[gpu_p_rank]
        for _ in range(2):
            src_cpu_rank, header, payload_iovs = await mailbox.get()
            kind = TensorEntryKind(
                entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            if kind == TensorEntryKind.TENSOR and len(payload_iovs) > 0:
                gpu_received_data[gpu_p_rank][src_cpu_rank] = backend.to_numpy(
                    payload_iovs[0]
                )
            else:
                gpu_received_data[gpu_p_rank][src_cpu_rank] = None

    async def run_simulation():
        await asyncio.gather(
            run_cpu_sender(0, cpu0_embs, cpu0_dests),
            run_cpu_sender(1, cpu1_embs, cpu1_dests),
            run_gpu_receiver(100),
            run_gpu_receiver(101),
            run_gpu_receiver(200),
            run_gpu_receiver(201),
        )

    asyncio.run(run_simulation())

    unique_emb_A_received = gpu_received_data[100][0]
    assert unique_emb_A_received.shape == (3, 4)
    full_emb_A_restored = unique_emb_A_received[restore_A]
    np.testing.assert_allclose(full_emb_A_restored, emb_A_ref, rtol=1e-5)

    pred_A = full_emb_A_restored @ W_dense
    grad_full_emb_A = (pred_A - targets_A) @ W_dense.T
    grad_unique_emb_A = np.zeros_like(unique_emb_A_received)
    np.add.at(grad_unique_emb_A, restore_A, grad_full_emb_A)

    emb_B_from_cpu0 = gpu_received_data[200][0]
    emb_B_from_cpu1 = gpu_received_data[200][1]
    assert emb_B_from_cpu0.shape == (2, 4)
    assert emb_B_from_cpu1.shape == (3, 4)

    unique_emb_B_received = np.concatenate([emb_B_from_cpu0, emb_B_from_cpu1], axis=0)
    full_emb_B_restored = unique_emb_B_received[restore_B]
    np.testing.assert_allclose(full_emb_B_restored, emb_B_ref, rtol=1e-5)

    pred_B = full_emb_B_restored @ W_dense
    grad_full_emb_B = (pred_B - targets_B) @ W_dense.T
    grad_unique_emb_B = np.zeros_like(unique_emb_B_received)
    np.add.at(grad_unique_emb_B, restore_B, grad_full_emb_B)

    receipt_gpu100 = SymmetryReceipt(
        world_size=2, sender_rank_tags=jnp.array([0, 0, 0], dtype=jnp.int32)
    )
    tt_grad_100 = TransportTensor(
        jnp.asarray(grad_unique_emb_A), receipt=receipt_gpu100, source_step=1
    )
    tt_grad_100._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 0, 99), 1),)
    )

    receipt_gpu200 = SymmetryReceipt(
        world_size=2, sender_rank_tags=jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    )
    tt_grad_200 = TransportTensor(
        jnp.asarray(grad_unique_emb_B), receipt=receipt_gpu200, source_step=1
    )
    tt_grad_200._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 2, 99), 1),)
    )

    cpu_received_grads = {0: [], 1: []}

    async def run_gpu_backward_sender(
        logical_rank: int, tt_grad: TransportTensor, p_rank: int
    ):
        distributor = AsyncDataDistributor(local_rank=logical_rank)
        send_fn = make_gpu_to_cpu_send_fn(p_rank)
        await distributor.distribute(
            local_data=[tt_grad],
            source=gpu_spec,
            target=cpu_spec,
            send_fn=send_fn,
            transfer_codes=[99],
            step=1,
            sender_spmd_code=2,
            sender_rank=logical_rank,
            sender_world_size=4,
            routing_mode=TransportRoutingMode.RECEIPT,
        )

    async def run_cpu_gradient_receiver(cpu_rank: int, expected_gpu_senders: int):
        mailbox = mailboxes[cpu_rank]
        for _ in range(expected_gpu_senders):
            sender_p_rank, header, payload_iovs = await mailbox.get()
            kind = TensorEntryKind(
                entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            if kind == TensorEntryKind.TENSOR and len(payload_iovs) > 0:
                cpu_received_grads[cpu_rank].append(backend.to_numpy(payload_iovs[0]))

    async def run_backward_simulation():
        await asyncio.gather(
            run_gpu_backward_sender(0, tt_grad_100, 100),
            run_gpu_backward_sender(2, tt_grad_200, 200),
            run_cpu_gradient_receiver(0, expected_gpu_senders=2),
            run_cpu_gradient_receiver(1, expected_gpu_senders=2),
        )

    asyncio.run(run_backward_simulation())

    grad_cpu0 = np.zeros_like(W_cpu0)
    for chunk in cpu_received_grads[0]:
        if chunk.shape[0] == 3:
            np.add.at(grad_cpu0, unique_A, chunk)
        elif chunk.shape[0] == 2:
            np.add.at(grad_cpu0, unique_B[:2], chunk)
        else:
            raise ValueError(f"Unexpected chunk shape: {chunk.shape}")

    W_cpu0_updated = W_cpu0 - learning_rate * grad_cpu0

    grad_cpu1 = np.zeros_like(W_cpu1)
    for chunk in cpu_received_grads[1]:
        if chunk.shape[0] == 3:
            np.add.at(grad_cpu1, unique_B[2:] - 4, chunk)
        else:
            raise ValueError(f"Unexpected chunk shape: {chunk.shape}")

    W_cpu1_updated = W_cpu1 - learning_rate * grad_cpu1
    W_updated = np.concatenate([W_cpu0_updated, W_cpu1_updated], axis=0)

    np.testing.assert_allclose(
        W_updated,
        W_ref_updated,
        rtol=1e-5,
        atol=1e-5,
    )


def test_multiaxis_all_ranks_active_bidirectional_training_flow():
    backend = get_backend()
    embedding_dim = 4
    learning_rate = 0.05
    rng = np.random.RandomState(42)

    W_u_ref = rng.randn(8, embedding_dim).astype(np.float32)
    W_i_ref = rng.randn(8, embedding_dim).astype(np.float32)

    W_u_cpu0 = W_u_ref[0:4].copy()
    W_u_cpu1 = W_u_ref[4:8].copy()

    W_i_cpu0 = W_i_ref[0:4].copy()
    W_i_cpu1 = W_i_ref[4:8].copy()

    u_batch_A = np.array([1, 2, 1, 3], dtype=np.int32)
    i_batch_A = np.array([4, 5, 4, 6], dtype=np.int32)
    target_A = np.ones((4, 1), dtype=np.float32)

    u_batch_B = np.array([2, 3, 5, 2], dtype=np.int32)
    i_batch_B = np.array([1, 6, 7, 1], dtype=np.int32)
    target_B = np.zeros((4, 1), dtype=np.float32)

    u_emb_A = W_u_ref[u_batch_A]
    i_emb_A = W_i_ref[i_batch_A]
    pred_A = np.sum(u_emb_A * i_emb_A, axis=1, keepdims=True)
    err_A = pred_A - target_A
    grad_u_emb_A = err_A * i_emb_A
    grad_i_emb_A = err_A * u_emb_A

    u_emb_B = W_u_ref[u_batch_B]
    i_emb_B = W_i_ref[i_batch_B]
    pred_B = np.sum(u_emb_B * i_emb_B, axis=1, keepdims=True)
    err_B = pred_B - target_B
    grad_u_emb_B = err_B * i_emb_B
    grad_i_emb_B = err_B * u_emb_B

    grad_W_u_ref = np.zeros_like(W_u_ref)
    np.add.at(grad_W_u_ref, u_batch_A, grad_u_emb_A)
    np.add.at(grad_W_u_ref, u_batch_B, grad_u_emb_B)
    W_u_ref_updated = W_u_ref - learning_rate * grad_W_u_ref

    grad_W_i_ref = np.zeros_like(W_i_ref)
    np.add.at(grad_W_i_ref, i_batch_A, grad_i_emb_A)
    np.add.at(grad_W_i_ref, i_batch_B, grad_i_emb_B)
    W_i_ref_updated = W_i_ref - learning_rate * grad_W_i_ref

    cpu_ranks = (0, 1)
    cpu_mesh = DeviceMesh(axis_names=("cpu",), device_shape=(2,))
    cpu_spec = ShardingSpec.chunk(P("cpu")).bind(cpu_mesh)

    gpu_ranks = (100, 101, 200, 201)
    gpu_mesh = DeviceMesh(axis_names=("dp", "fsdp"), device_shape=(2, 2))
    gpu_group = ExecutionGroup.create(gpu_ranks, gpu_mesh)
    gpu_spec = ShardingSpec.chunk(P(("dp", "fsdp"))).bind(gpu_mesh)

    mailboxes = {r: asyncio.Queue() for r in list(cpu_ranks) + list(gpu_ranks)}

    def make_cpu_to_gpu_send_fn(src_cpu_rank: int):
        async def send_fn(target_logical_id: int, header, *payload_iovs):
            target_p_rank = gpu_group.physical_rank(target_logical_id)
            await mailboxes[target_p_rank].put((src_cpu_rank, header, payload_iovs))
            return True

        return send_fn

    def make_gpu_to_cpu_send_fn(src_gpu_p_rank: int):
        async def send_fn(target_cpu_rank: int, header, *payload_iovs):
            await mailboxes[target_cpu_rank].put((src_gpu_p_rank, header, payload_iovs))
            return True

        return send_fn

    u_uniq_A, u_inv_A = np.unique(u_batch_A, return_inverse=True)
    i_uniq_A, i_inv_A = np.unique(i_batch_A, return_inverse=True)

    u_uniq_B, u_inv_B = np.unique(u_batch_B, return_inverse=True)
    i_uniq_B, i_inv_B = np.unique(i_batch_B, return_inverse=True)

    cpu0_embs = np.concatenate(
        [
            W_u_cpu0[u_uniq_A],
            W_u_cpu0[u_uniq_B[:2]],
            W_i_cpu0[i_uniq_B[:1]],
        ],
        axis=0,
    )
    cpu0_dests = np.array([0, 0, 0, 2, 2, 3], dtype=np.int32)

    cpu1_embs = np.concatenate(
        [
            W_i_cpu1[i_uniq_A - 4],
            W_u_cpu1[u_uniq_B[2:] - 4],
            W_i_cpu1[i_uniq_B[1:] - 4],
        ],
        axis=0,
    )
    cpu1_dests = np.array([1, 1, 1, 2, 3, 3], dtype=np.int32)

    def route_func(keys, *, partition_count: int, domain_size: int):
        del partition_count, domain_size
        return keys

    func_partition = FunctionPartition(target=route_func)
    target_layout = ShardingSpec(
        routing=AlgebraicRouting(strategy=func_partition, partition=P(("dp", "fsdp"))),
    ).bind(gpu_mesh)

    async def run_cpu_sender(src_rank: int, embs: np.ndarray, dest_tags: np.ndarray):
        tt = TransportTensor(
            jnp.asarray(embs),
            routing_indices=jnp.asarray(dest_tags),
            origin_indices=jnp.asarray(dest_tags),
            global_shape=embs.shape,
            source_step=1,
        )
        tt._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(1, src_rank, 100 + src_rank), 1),)
        )
        distributor = AsyncDataDistributor(local_rank=src_rank)
        send_fn = make_cpu_to_gpu_send_fn(src_rank)
        await distributor.distribute(
            local_data=[tt],
            source=cpu_spec,
            target=target_layout,
            send_fn=send_fn,
            transfer_codes=[100 + src_rank],
            step=1,
            sender_spmd_code=1,
            sender_rank=src_rank,
            sender_world_size=2,
        )

    gpu_received = {r: {} for r in gpu_ranks}

    async def run_gpu_receiver(gpu_p_rank: int):
        mailbox = mailboxes[gpu_p_rank]
        for _ in range(2):
            src_cpu_rank, header, payload_iovs = await mailbox.get()
            kind = TensorEntryKind(
                entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            if kind == TensorEntryKind.TENSOR and len(payload_iovs) > 0:
                gpu_received[gpu_p_rank][src_cpu_rank] = backend.to_numpy(
                    payload_iovs[0]
                )
            else:
                gpu_received[gpu_p_rank][src_cpu_rank] = None

    async def run_forward_exchange():
        await asyncio.gather(
            run_cpu_sender(0, cpu0_embs, cpu0_dests),
            run_cpu_sender(1, cpu1_embs, cpu1_dests),
            run_gpu_receiver(100),
            run_gpu_receiver(101),
            run_gpu_receiver(200),
            run_gpu_receiver(201),
        )

    asyncio.run(run_forward_exchange())

    u_emb_A_rec = gpu_received[100][0][u_inv_A]
    np.testing.assert_allclose(u_emb_A_rec, u_emb_A, rtol=1e-5)

    i_emb_A_rec = gpu_received[101][1][i_inv_A]
    np.testing.assert_allclose(i_emb_A_rec, i_emb_A, rtol=1e-5)

    u_emb_B_parts = [gpu_received[200][0], gpu_received[200][1]]
    u_emb_B_uniq = np.concatenate(u_emb_B_parts, axis=0)
    u_emb_B_rec = u_emb_B_uniq[u_inv_B]
    np.testing.assert_allclose(u_emb_B_rec, u_emb_B, rtol=1e-5)

    i_emb_B_parts = [gpu_received[201][0], gpu_received[201][1]]
    i_emb_B_uniq = np.concatenate(i_emb_B_parts, axis=0)
    i_emb_B_rec = i_emb_B_uniq[i_inv_B]
    np.testing.assert_allclose(i_emb_B_rec, i_emb_B, rtol=1e-5)

    grad_u_uniq_A = np.zeros_like(gpu_received[100][0])
    np.add.at(grad_u_uniq_A, u_inv_A, grad_u_emb_A)

    grad_i_uniq_A = np.zeros_like(gpu_received[101][1])
    np.add.at(grad_i_uniq_A, i_inv_A, grad_i_emb_A)

    grad_u_uniq_B = np.zeros_like(u_emb_B_uniq)
    np.add.at(grad_u_uniq_B, u_inv_B, grad_u_emb_B)

    grad_i_uniq_B = np.zeros_like(i_emb_B_uniq)
    np.add.at(grad_i_uniq_B, i_inv_B, grad_i_emb_B)

    receipt_100 = SymmetryReceipt(2, jnp.array([0, 0, 0], dtype=jnp.int32))
    tt_100 = TransportTensor(
        jnp.asarray(grad_u_uniq_A), receipt=receipt_100, source_step=1
    )
    tt_100._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 0, 99), 1),)
    )

    receipt_101 = SymmetryReceipt(2, jnp.array([1, 1, 1], dtype=jnp.int32))
    tt_101 = TransportTensor(
        jnp.asarray(grad_i_uniq_A), receipt=receipt_101, source_step=1
    )
    tt_101._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 1, 99), 1),)
    )

    receipt_200 = SymmetryReceipt(2, jnp.array([0, 0, 1], dtype=jnp.int32))
    tt_200 = TransportTensor(
        jnp.asarray(grad_u_uniq_B), receipt=receipt_200, source_step=1
    )
    tt_200._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 2, 99), 1),)
    )

    receipt_201 = SymmetryReceipt(2, jnp.array([0, 1, 1], dtype=jnp.int32))
    tt_201 = TransportTensor(
        jnp.asarray(grad_i_uniq_B), receipt=receipt_201, source_step=1
    )
    tt_201._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(2, 3, 99), 1),)
    )

    cpu_received_grads = {0: [], 1: []}

    async def run_gpu_backward_sender(
        logical_rank: int, tt_grad: TransportTensor, p_rank: int
    ):
        distributor = AsyncDataDistributor(local_rank=logical_rank)
        send_fn = make_gpu_to_cpu_send_fn(p_rank)
        await distributor.distribute(
            local_data=[tt_grad],
            source=gpu_spec,
            target=cpu_spec,
            send_fn=send_fn,
            transfer_codes=[99],
            step=1,
            sender_spmd_code=2,
            sender_rank=logical_rank,
            sender_world_size=4,
            routing_mode=TransportRoutingMode.RECEIPT,
        )

    async def run_cpu_gradient_receiver(cpu_rank: int, expected_senders: int):
        mailbox = mailboxes[cpu_rank]
        for _ in range(expected_senders):
            sender_p_rank, header, payload_iovs = await mailbox.get()
            kind = TensorEntryKind(
                entry_tensor_kind(int(header.entry_i32[0, ENTRY_STATE_FLAGS]))
            )
            if kind == TensorEntryKind.TENSOR and len(payload_iovs) > 0:
                cpu_received_grads[cpu_rank].append(backend.to_numpy(payload_iovs[0]))

    async def run_backward_exchange():
        await asyncio.gather(
            run_gpu_backward_sender(0, tt_100, 100),
            run_gpu_backward_sender(1, tt_101, 101),
            run_gpu_backward_sender(2, tt_200, 200),
            run_gpu_backward_sender(3, tt_201, 201),
            run_cpu_gradient_receiver(0, expected_senders=4),
            run_cpu_gradient_receiver(1, expected_senders=4),
        )

    asyncio.run(run_backward_exchange())

    grad_W_u_cpu0 = np.zeros_like(W_u_cpu0)
    grad_W_i_cpu0 = np.zeros_like(W_i_cpu0)

    for chunk in cpu_received_grads[0]:
        if chunk.shape[0] == 3:
            np.add.at(grad_W_u_cpu0, u_uniq_A, chunk)
        elif chunk.shape[0] == 2:
            np.add.at(grad_W_u_cpu0, u_uniq_B[:2], chunk)
        elif chunk.shape[0] == 1:
            np.add.at(grad_W_i_cpu0, i_uniq_B[:1], chunk)
        else:
            raise ValueError(f"Unexpected chunk shape: {chunk.shape}")

    W_u_cpu0_updated = W_u_cpu0 - learning_rate * grad_W_u_cpu0
    W_i_cpu0_updated = W_i_cpu0 - learning_rate * grad_W_i_cpu0

    grad_W_u_cpu1 = np.zeros_like(W_u_cpu1)
    grad_W_i_cpu1 = np.zeros_like(W_i_cpu1)

    for chunk in cpu_received_grads[1]:
        if chunk.shape[0] == 3:
            np.add.at(grad_W_i_cpu1, i_uniq_A - 4, chunk)
        elif chunk.shape[0] == 1:
            np.add.at(grad_W_u_cpu1, u_uniq_B[2:] - 4, chunk)
        elif chunk.shape[0] == 2:
            np.add.at(grad_W_i_cpu1, i_uniq_B[1:] - 4, chunk)
        else:
            raise ValueError(f"Unexpected chunk shape: {chunk.shape}")

    W_u_cpu1_updated = W_u_cpu1 - learning_rate * grad_W_u_cpu1
    W_i_cpu1_updated = W_i_cpu1 - learning_rate * grad_W_i_cpu1

    W_u_updated = np.concatenate([W_u_cpu0_updated, W_u_cpu1_updated], axis=0)
    W_i_updated = np.concatenate([W_i_cpu0_updated, W_i_cpu1_updated], axis=0)

    np.testing.assert_allclose(W_u_updated, W_u_ref_updated, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(W_i_updated, W_i_ref_updated, rtol=1e-5, atol=1e-5)
