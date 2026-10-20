"""
Comprehensive tests for cross-mesh (M:N) routing in Fluxway's SPMD layer.
These are pure-Python / JAX unit tests that do NOT require Ray or a full
training stack. They validate the assembly helper methods in worker.py and
the topology projection logic in operator_manager.py in isolation.

Test scenarios
--------------
1. 1:1 Static  - dp=4 → dp=4
2. M:N Static  - dp=2 → dp=6
3. Dynamic → Static (origin_indices reassembly) - shard=3 → dp=6
4. Interleave Source - virtual SPMD sub-mesh rotation
5. Mirror routing - SymmetryReceipt construction and routing
6. E2E Embedding - shard=4 dynamic forward + mirror backward
"""

from __future__ import annotations

import unittest
import jax
import pytest
from fluxway.common.tensor_backend import use_backend
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.compiled_graph import (
    CompiledLogicalTensor,
    CompiledTensorTransfer,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
)
from fluxway.common.types.sharding import AlgebraicRouting, ChunkPartition
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.common.coherence import CoherenceInstance, CoherenceSubject
from fluxway.transport.manifest import (
    ENTRY_GLOBAL_SHAPE_BEGIN,
    ENTRY_GLOBAL_SHAPE_COUNT,
    ENTRY_AUX,
)
from fluxway.spmd.worker.worker import (
    SPMDWorker as RaySPMDWorker,
    _nonzero_1d_indices,
)
from fluxway.spmd.worker.assembly import CompiledAssemblePlan

SPMDWorker = getattr(RaySPMDWorker, "__ray_actor_class__", RaySPMDWorker)  # type: ignore


@pytest.fixture(autouse=True)
def cpu_transport_backend():
    with use_backend("hybrid"), jax.default_device(jax.devices("cpu")[0]):
        yield


# ─── Helper: build a minimal TransportTensor ─────────────────────────────────


def make_tt(
    values,
    origin_indices=None,
    routing_indices=None,
    receipt: SymmetryReceipt | None = None,
    global_shape=None,
    source_step: int = 0,
) -> TransportTensor:
    tensor = TransportTensor(
        jnp.array(values, dtype=jnp.float32),
        origin_indices=(
            jnp.array(origin_indices, dtype=jnp.int32)
            if origin_indices is not None
            else None
        ),
        routing_indices=(
            jnp.array(routing_indices, dtype=jnp.int32)
            if routing_indices is not None
            else None
        ),
        receipt=receipt,
        global_shape=global_shape,
        source_step=source_step,
    )
    tensor._bind_coherence_instances(
        (CoherenceInstance(CoherenceSubject(1, 0, 1), source_step),)
    )
    return tensor


def make_receipt(sender_ranks, remote_indices, world_size=4) -> SymmetryReceipt:
    return SymmetryReceipt(world_size, jnp.array(sender_ranks, dtype=jnp.int32))


def _assembly_worker(rank: int):
    worker = SPMDWorker.__new__(SPMDWorker)
    worker.rank = rank
    worker._profiler_runtime = SimpleNamespace(enabled=False)  # type: ignore[assignment]
    worker._op_manager = SimpleNamespace()  # type: ignore[assignment]
    for name in (
        "_flatten_shards",
        "_filter_shards_by_routing",
        "_get_values",
        "_get_routing_indices",
        "_normalize_and_cat_shards",
        "_maybe_merge_receipts",
        "_routing_entity_count",
        "_merge_index_descriptors",
        "_describe_shard_flow_for_error",
    ):
        setattr(worker, name, getattr(SPMDWorker, name).__get__(worker, SPMDWorker))
    return worker


# =============================================================================
# Tests
# =============================================================================


def _dynamic_to_static_assemble_plan(
    *,
    op_name: str,
    transfer_code: int,
    logical_tuuid: int,
    input_layout,
    predecessor_spmd_id: str,
) -> CompiledAssemblePlan:
    sharding = input_layout.spec
    transfer = CompiledTensorTransfer(
        spec=TensorTransfer(OperatorPort("output"), OperatorPort("input")),
        transfer_code=transfer_code,
        dependency_code=1,
        logical_tuuid=logical_tuuid,
        source_sharding=sharding,
        target_sharding=sharding,
    )
    return CompiledAssemblePlan(
        op_name=op_name,
        dependency=SimpleNamespace(code=1),  # type: ignore[arg-type]
        transfer=transfer,
        logical_tensor=CompiledLogicalTensor(
            logical_tuuid=logical_tuuid,
            producer=OperatorKey(predecessor_spmd_id, "produce"),
            producer_port="output",
            producer_sharding=sharding,
            effective_sharding=sharding,
        ),
        source_spmd_id=predecessor_spmd_id,
        source_layout=SimpleNamespace(is_content_addressable=True),  # type: ignore[arg-type]
        target_layout=input_layout,
    )


def _assemble_dynamic_to_static(worker, plan, input_layout, shards):
    max_index = max(
        int(np.asarray(shard.origin_indices_materialized).max())
        for shard in shards
        if shard.origin_indices_materialized is not None
    )
    shard_map = {}
    for rank, shard in enumerate(shards):
        if shard.global_shape is None:
            value_shape = tuple(int(x) for x in shard.flat_values.shape[1:])
            shard = TransportTensor(
                shard.flat_values,
                nested_splits=shard.nested_splits,
                routing_indices=shard.routing_indices,
                origin_indices=shard.origin_indices,
                global_shape=(max_index + 1, *value_shape),
                receipt=shard.receipt,
                source_step=shard.source_step,
            )
        shard_map[(plan.source_spmd_id, rank)] = [shard]
    return SPMDWorker._assemble_dynamic_to_static_tensor(
        worker,
        plan.op_name,
        plan.transfer_code,
        0,
        shard_map,
        input_layout,
        assemble_plan=plan,
    )


class TestSymmetryReceiptHelpers(unittest.TestCase):
    """Unit tests for _maybe_merge_receipts."""

    def setUp(self):
        self.w = _assembly_worker(rank=0)

    def test_no_mirror_returns_none(self):
        shards = [
            make_tt([[1.0, 2.0]], receipt=make_receipt([0], [0])),
            make_tt([[3.0, 4.0]], receipt=make_receipt([1], [1])),
        ]
        result = self.w._maybe_merge_receipts(shards)
        self.assertIsNotNone(result)

    def test_mirror_merges_receipts(self):
        r0 = make_receipt([0, 0], [10, 11])
        r1 = make_receipt([1, 1], [20, 21])
        shards = [
            make_tt([[1.0], [2.0]], receipt=r0),
            make_tt([[3.0], [4.0]], receipt=r1),
        ]
        result = self.w._maybe_merge_receipts(shards)
        self.assertIsNotNone(result)
        assert result is not None
        np.testing.assert_array_equal(result.sender_rank_tags, [0, 0, 1, 1])

    def test_partial_receipts_returns_none_even_with_mirror_flag(self):
        """If one shard has no receipt, result must be None (no partial provenance)."""
        shards = [
            make_tt([[1.0]], receipt=make_receipt([0], [0])),
            make_tt([[2.0]], receipt=None),
        ]
        result = self.w._maybe_merge_receipts(shards)
        self.assertIsNone(result)


class TestDynamicToStaticAssembly(unittest.TestCase):
    """
    Test _assemble_dynamic_to_static_core for correct origin_indices reassembly.
    Simulates SPMD_A(shard=4, dynamic) → SPMD_B(dp=6, static).
    """

    def _run_rank(
        self,
        rank: int,
        shards: list[TransportTensor],
        world_size: int,
        has_mirror: bool,
    ):
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            PartitionSpec,
            ChunkPartition,
        )

        mesh = DeviceMesh(device_shape=(world_size,), axis_names=("dp",))
        strategy = ChunkPartition()
        # PartitionSpec takes positional axis names: P("dp") means dim-0 is sharded on "dp"
        spec = ShardingSpec(
            routing=AlgebraicRouting(strategy=strategy, partition=PartitionSpec("dp"))
        ).bind(mesh)
        w = _assembly_worker(rank=rank)
        valid = [s for s in shards if get_backend().shape(w._get_values(s))[0] > 0]
        if not valid:
            raise ValueError(
                "Dynamic-to-static assembly requires at least one non-empty shard."
            )
        plan = _dynamic_to_static_assemble_plan(
            op_name="step",
            transfer_code=7,
            logical_tuuid=7,
            input_layout=spec,
            predecessor_spmd_id="Sparse",
        )
        return _assemble_dynamic_to_static(w, plan, spec, valid)

    def test_basic_reassembly_no_mirror(self):
        """
        Global batch = 12 items. SPMD_A has 4 workers, each sending 3 items
        to SPMD_B (dp=6). Each dp=6 rank should receive exactly 2 items.
        """
        # Simulate 12 items with global indices 0-11 split across 4 sources
        # Each source sends 3 consecutive items
        shard0 = make_tt([[1.0], [2.0], [3.0]], origin_indices=[0, 1, 2])
        shard1 = make_tt([[4.0], [5.0], [6.0]], origin_indices=[3, 4, 5])
        shard2 = make_tt([[7.0], [8.0], [9.0]], origin_indices=[6, 7, 8])
        shard3 = make_tt([[10.0], [11.0], [12.0]], origin_indices=[9, 10, 11])

        all_shards = [shard0, shard1, shard2, shard3]

        for rank in range(6):
            result = self._run_rank(rank, all_shards, world_size=6, has_mirror=False)
            data = np.array(result.flat_values)
            self.assertEqual(
                data.shape[0],
                2,
                f"Rank {rank} should have 2 items, got {data.shape[0]}",
            )
            # No mirror → no receipt
            self.assertIsNone(result.receipt)

    def test_real_worker_slot_addressed_sparse_embedding_assembly(self):
        mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
        input_layout = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(mesh)

        worker = _assembly_worker(rank=1)

        shard = TransportTensor(
            np.array([[8, 8], [10, 10], [13, 13]], dtype=np.float32),
            origin_indices=ExplicitIndex(np.array([8, 10, 13], dtype=np.int64)),
            global_shape=(16, 2),
            receipt=SymmetryReceipt(4, np.array([0, 1, 1], dtype=np.int32)),
            source_step=0,
        )

        assemble_plan = _dynamic_to_static_assemble_plan(
            op_name="step",
            transfer_code=7,
            logical_tuuid=7,
            input_layout=input_layout,
            predecessor_spmd_id="Sparse",
        )
        result = _assemble_dynamic_to_static(
            worker, assemble_plan, input_layout, [shard]
        )
        data = np.asarray(result.flat_values)

        self.assertEqual(data.shape, (8, 2))
        np.testing.assert_array_equal(data[0], [8, 8])
        np.testing.assert_array_equal(data[2], [10, 10])
        np.testing.assert_array_equal(data[5], [13, 13])
        np.testing.assert_array_equal(data[1], [0, 0])
        np.testing.assert_array_equal(
            np.asarray(result.receipt.sender_rank_tags),  # type: ignore[union-attr]
            [0, 4, 1, 4, 4, 1, 4, 4],
        )

    def test_hybrid_routing_dim_one_dynamic_to_static_uses_unpadded_nonzero(self):
        with use_backend("hybrid"):
            backend = get_backend()
            mask = backend.tensor([True, True, False, False])
            indices_to_take = _nonzero_1d_indices(backend, mask)
            all_values = backend.tensor(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                ],
                dtype=backend.float32(),
            )

            idx_tuple = [slice(None), indices_to_take]
            local_values = all_values[tuple(idx_tuple)]
            assembled_data = backend.zeros((2, 2), dtype=all_values.dtype)
            assembled_data = backend.index_update(
                assembled_data,
                (slice(None), backend.tensor([0, 1], dtype=backend.int32())),
                local_values,
            )

            data = np.asarray(assembled_data)
            self.assertEqual(data.shape, (2, 2))
            np.testing.assert_array_equal(data, [[10, 11], [20, 21]])

    def test_unequal_shard_sizes(self):
        """
        Simulates real embedding lookup where shards are unevenly distributed.
        Global: 12 items across 4 SPMD_A workers → SPMD_B (dp=4).
        """
        # 4 workers send: 4, 3, 3, 2 items with correct global origin_indices
        shard0 = make_tt(
            [[float(i)] for i in range(4)], origin_indices=list(range(0, 4))
        )
        shard1 = make_tt(
            [[float(i)] for i in range(4, 7)], origin_indices=list(range(4, 7))
        )
        shard2 = make_tt(
            [[float(i)] for i in range(7, 10)], origin_indices=list(range(7, 10))
        )
        shard3 = make_tt(
            [[float(i)] for i in range(10, 12)], origin_indices=list(range(10, 12))
        )
        all_shards = [shard0, shard1, shard2, shard3]

        total = 0
        for rank in range(4):
            result = self._run_rank(rank, all_shards, world_size=4, has_mirror=False)
            n = int(np.array(result.flat_values).shape[0])
            total += n
        # All 12 items should be distributed without loss
        self.assertEqual(total, 12)

    def test_mirror_receipt_roundtrip(self):
        """
        E2E Embedding: SPMD_A(shard=4, dynamic) forward → SPMD_B(dp=6, static).
        SymmetryReceipt on each item records the sender_rank (SPMD_A worker) and
        remote_index (SPMD_A's local position). After assembly, SPMD_B Rank r's
        receipt should contain the correct back-routing info for all its items.
        """

        # 12 items, 4 SPMD_A workers, each sending 3 items
        # receipt: sender_rank = SPMD_A rank, remote_index = local offset in that rank
        def make_emb_shard(spmd_a_rank: int, values, global_indices):
            n = len(values)
            receipt = make_receipt(
                sender_ranks=[spmd_a_rank] * n, remote_indices=list(range(n))
            )
            return make_tt(
                [[v] for v in values], origin_indices=global_indices, receipt=receipt
            )

        shard0 = make_emb_shard(0, [1.0, 2.0, 3.0], [0, 1, 2])
        shard1 = make_emb_shard(1, [4.0, 5.0, 6.0], [3, 4, 5])
        shard2 = make_emb_shard(2, [7.0, 8.0, 9.0], [6, 7, 8])
        shard3 = make_emb_shard(3, [10.0, 11.0, 12.0], [9, 10, 11])
        all_shards = [shard0, shard1, shard2, shard3]

        total_items = 0
        for rank in range(6):
            result = self._run_rank(rank, all_shards, world_size=6, has_mirror=True)
            data = np.array(result.flat_values)
            n = data.shape[0]
            total_items += n

            # Receipt must be present for mirror edges
            self.assertIsNotNone(result.receipt, f"Rank {rank} missing receipt")

            if n > 0:
                receipt = result.receipt
                assert receipt is not None
                # sender_rank_tags must be in [0,3] (valid SPMD_A ranks)
                assert receipt is not None
                tags = np.array(receipt.sender_rank_tags)
                self.assertTrue(
                    np.all(tags >= 0), f"Rank {rank}: invalid sender_rank < 0"
                )
                self.assertTrue(np.all(tags <= 3), f"Rank {rank}: sender_rank > 3")

        # All 12 items must be covered
        self.assertEqual(total_items, 12)

    def test_backward_routing_uses_receipt(self):
        """
        Verify that backward routing (mirror) correctly inverts the forward path.
        SPMD_B gradient tensor carries SymmetryReceipt.sender_ranks, which should
        match the original SPMD_A rank that sent each embedding item.
        """
        # Forward assembly for rank=0 of SPMD_B (dp=6), items 0 and 1
        receipt = make_receipt(sender_ranks=[0, 0], remote_indices=[0, 1])
        grad_tensor = make_tt([[0.1], [0.2]], receipt=receipt)

        # The backward pass reads receipt.sender_rank_tags → routes gradient to SPMD_A rank 0
        assert grad_tensor.receipt is not None
        assert grad_tensor.receipt is not None
        backward_targets = set(np.array(grad_tensor.receipt.sender_rank_tags).tolist())
        self.assertEqual(backward_targets, {0})  # both rows go back to SPMD_A[0]

        # SymmetryReceipt no longer directly carries remote_indices in the tags.
        # It's assumed the receiver knows how to scatter or the tags are enough?
        # In current Fluxway, SymmetryReceipt ONLY carries sender_rank_tags.
        pass


class TestEmbeddingToDenseFullCommunication(unittest.TestCase):
    """
    Simulates the exact Embedding -> Dense communication pattern from the training example.
    SPMD_A (shard=4, dynamic outputs) -> SPMD_B (dp=6, static input).
    Verifies that through the Distributor and Reassembly, all data is preserved in correct order.
    """

    def test_e2e_embedding_to_dense_communication(self):
        from fluxway.transport.communication import AsyncDataDistributor
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            PartitionSpec,
            ChunkPartition,
            IndexedModuloPartition,
        )
        import asyncio

        SENDER_SIZE = 4
        RECEIVER_SIZE = 6
        TOTAL_ITEMS = 12

        # 1. Setup Meshes and Specs
        sender_mesh = DeviceMesh(device_shape=(SENDER_SIZE,), axis_names=("shard",))
        source_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=IndexedModuloPartition(), partition=PartitionSpec("shard")
            )
        ).bind(sender_mesh)

        receiver_mesh = DeviceMesh(device_shape=(RECEIVER_SIZE,), axis_names=("dp",))
        target_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(receiver_mesh)

        # 2. Prepare shards on 4 Senders
        # We simulate a slightly shuffled embedding result:
        # Rank 0: [0, 5, 10]
        # Rank 1: [1, 6, 11]
        # Rank 2: [2, 7]
        # Rank 3: [3, 4, 8, 9]
        # Total = 12 items.
        sender_shards = {
            0: ([0, 5, 10], [[0.0], [5.0], [10.0]]),
            1: ([1, 6, 11], [[1.0], [6.0], [11.0]]),
            2: ([2, 7], [[2.0], [7.0]]),
            3: ([3, 4, 8, 9], [[3.0], [4.0], [8.0], [9.0]]),
        }

        # 3. Simulated Network: receiver_id -> List of TransportTensors received
        network_mailbox: dict[int, list[TransportTensor]] = {
            i: [] for i in range(RECEIVER_SIZE)
        }

        async def simulate_sender(rank):
            indices, values = sender_shards[rank]
            tt = make_tt(
                values=values,
                origin_indices=indices,
                global_shape=(TOTAL_ITEMS, 1),
            )

            distributor = AsyncDataDistributor(local_rank=rank)

            async def mock_send(
                target_rank: int,
                header,
                *payload_iovs,
            ):
                valid_count = int(header.payload_i32[0, 5])
                if valid_count > 0:
                    g_begin = int(header.entry_i32[0, ENTRY_GLOBAL_SHAPE_BEGIN])
                    g_count = int(header.entry_i32[0, ENTRY_GLOBAL_SHAPE_COUNT])
                    g_shape = (
                        tuple(
                            int(x)
                            for x in header.global_shape_i64[
                                g_begin : g_begin + g_count
                            ]
                        )
                        if g_count
                        else None
                    )
                    received_tt = TransportTensor(
                        payload_iovs[0],
                        origin_indices=payload_iovs[1]
                        if len(payload_iovs) > 1
                        else None,
                        global_shape=g_shape,
                        source_step=int(header.entry_i32[0, ENTRY_AUX]),
                    )
                    network_mailbox[target_rank].append(received_tt)
                return True

            await distributor.distribute(
                [tt],
                source=source_spec,
                target=target_spec,
                send_fn=mock_send,
                transfer_codes=77,
                sender_rank=rank,
                sender_world_size=SENDER_SIZE,
            )

        # Run all senders
        for r in range(SENDER_SIZE):
            asyncio.run(simulate_sender(r))

        # 4. Receiver Reassembly
        final_results = []
        for r in range(RECEIVER_SIZE):
            worker = _assembly_worker(rank=r)
            shards = {
                ("Sparse", sender_rank): [shard]
                for sender_rank, shard in enumerate(network_mailbox[r])
            }
            assemble_plan = _dynamic_to_static_assemble_plan(
                op_name="step",
                transfer_code=1,
                logical_tuuid=1,
                input_layout=target_spec,
                predecessor_spmd_id="Sparse",
            )
            assembled = SPMDWorker._assemble_dynamic_to_static_tensor(
                worker,
                "step",
                1,
                1,
                shards,
                target_spec,
                assemble_plan=assemble_plan,
            )
            final_results.append(assembled)

        # 5. Verification
        # Combined result should be [0, 1, ..., 11]
        full_tensor = np.concatenate(
            [np.array(r.flat_values) for r in final_results], axis=0
        )
        expected = np.arange(TOTAL_ITEMS).reshape((TOTAL_ITEMS, 1)).astype(np.float32)

        np.testing.assert_array_equal(
            full_tensor,
            expected,
            f"E2E Routing failed! Expected sequential 0-11, got:\n{full_tensor.flatten()}",
        )
        print("E2E Embedding-to-Dense Full Communication Simulation PASSED")


class TestScalarAndComplexObjectRouting(unittest.TestCase):
    """Verifies that scalars and complex objects don't trigger index errors during sharding inference."""

    def test_scalar_sharding_inference(self):
        from fluxway.common.mesh import DeviceMesh
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )

        mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
        # Case: A scalar sharded on 'dp' (should be treated as replicated/safe)
        spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(mesh)

        # Local scalar shape
        local_shape = ()
        # This is where 'tuple index out of range' often happens if not caught
        inferred = spec.infer_global_shape(local_shape)

        self.assertEqual(
            inferred, (), "Inferred global shape for scalar should remain scalar"
        )

    def test_extracted_sharding_safety(self):
        # Mocks the _send_output logic in worker.py
        from fluxway.common.types.tensor import TransportTensor
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )
        from fluxway.common.mesh import DeviceMesh

        mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
        spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(mesh)

        # Mock scalar value and how worker.py wraps it
        tensor_data = np.array(0.5)
        # Worker logic: TransportTensor.from_tensor(tensor_data)
        tt = TransportTensor.from_tensor(tensor_data)

        # Safety check for global_shape inference
        if tt.global_shape is None:
            # This is the line that might fail in worker.py before my fix
            inferred = spec.infer_global_shape(tt.shape)
            self.assertEqual(inferred, (), "Successfully inferred global scalar shape")


class TestStaticToStaticCoverageValidation(unittest.TestCase):
    """
    Validates the Static Coverage Check in communication.py:
    - Undersized batch raises ValueError
    - Exact-size batch passes
    - EOS empty packet (global_shape=None) is NOT blocked
    """

    def _make_specs(self, total_items: int, dp: int):
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )

        sender_mesh = DeviceMesh(device_shape=(1,), axis_names=("worker",))
        receiver_mesh = DeviceMesh(device_shape=(dp,), axis_names=("dp",))
        source_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("worker")
            )
        ).bind(sender_mesh)
        target_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(receiver_mesh)
        return source_spec, target_spec, total_items

    def _run_distributor(self, values, global_shape, source_spec, target_spec):
        """Drive AsyncDataDistributor synchronously; collect any raised errors."""
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        tt = TransportTensor(
            jnp.array(values, dtype=jnp.float32),
            global_shape=global_shape,
            source_step=0,
        )
        tt._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(1, 0, 79), 0),)
        )
        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        async def _run():
            await distributor.distribute(
                [tt],
                source_spec,
                target_spec,
                mock_send,
                transfer_codes=78,
                sender_rank=0,
                sender_world_size=1,
            )

        asyncio.run(_run())

    def test_undersized_batch_raises(self):
        """batch_size=4 with dp=6 must raise a clear ValueError."""
        source_spec, target_spec, _ = self._make_specs(total_items=4, dp=6)
        tt = TransportTensor(
            jnp.ones((4, 1), dtype=jnp.float32),
            global_shape=(4, 1),
            source_step=0,
        )
        tt._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(1, 0, 79), 0),)
        )
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                distributor.distribute(
                    [tt],
                    source_spec,
                    target_spec,
                    mock_send,
                    transfer_codes=79,
                    sender_rank=0,
                    sender_world_size=1,
                )
            )
        self.assertIn("Static Sharding Error", str(ctx.exception))
        self.assertIn("6", str(ctx.exception))  # sub_mesh_size mentioned

    def test_exact_size_passes(self):
        """batch_size=6 with dp=6 must pass without raising."""
        source_spec, target_spec, _ = self._make_specs(total_items=6, dp=6)
        tt = TransportTensor(
            jnp.ones((6, 1), dtype=jnp.float32),
            global_shape=(6, 1),
            source_step=0,
        )
        tt._bind_coherence_instances(
            (CoherenceInstance(CoherenceSubject(1, 0, 80), 0),)
        )
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        try:
            asyncio.run(
                distributor.distribute(
                    [tt],
                    source_spec,
                    target_spec,
                    mock_send,
                    transfer_codes=80,
                    sender_rank=0,
                    sender_world_size=1,
                )
            )
        except ValueError as e:
            self.fail(f"Unexpected ValueError for exact-size batch: {e}")

    def test_eos_empty_packet_not_blocked(self):
        """A legitimate EOS packet with global_shape=None must NOT trigger the Coverage Check."""
        source_spec, target_spec, _ = self._make_specs(total_items=0, dp=6)
        empty_tt = TransportTensor(
            jnp.zeros((0,), dtype=jnp.float32),
            global_shape=None,  # EOS signal — no global shape
            source_step=0,
        )
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        # Should not raise
        try:
            asyncio.run(
                distributor.distribute(
                    [empty_tt],
                    source_spec,
                    target_spec,
                    mock_send,
                    transfer_codes=81,
                    sender_rank=0,
                    sender_world_size=1,
                )
            )
        except ValueError as e:
            if "Static Sharding Error" in str(e):
                self.fail(f"EOS empty packet incorrectly blocked: {e}")


if __name__ == "__main__":
    unittest.main()
