from typing import Any
import asyncio

import numpy as np
import sparse_embedding as se  # type: ignore

from fluxway.common.tensor_backend import get_backend
from fluxway.common.decorators import spmd, operator
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.types.tensor import TransportTensor
from .data_types import UserBatch, InputNodeOutput, DPFeatures


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    assert val is not None, "[InputSPMD.hash_routing] tensor.flat_values is None"
    assert mesh.size > 0, (
        f"[InputSPMD.hash_routing] mesh.size must be positive, got {mesh.size}"
    )
    if len(val.shape) > 1:
        val = val[:, 0]
    return np.mod(val, mesh.size)


async def _sparse_unique(
    tt: TransportTensor,
    rank: int,
    world_size: int,
) -> tuple[TransportTensor, TransportTensor]:
    """Deduplicate ids via sparse_embedding.async_unique (C++ high-perf).

    Returns ``(unique_tt, inv_tt)`` carrying the *minimum* metadata required
    for correct cross-mesh reassembly:

    * ``unique_tt`` carries a per-row ``ExplicitIndex`` whose value
      ``first_indices[i] * world_size + rank`` is the absolute position
      of ``unique[i]`` in the conceptual *round-robin interleaved* global
      ravel ``global_ravel[k*D + d] = rank_d.ravel()[k]`` (with
      ``D = world_size``).  This expression is the truly global,
      coordination-free address: it occupies exactly the
      ``[0, batch_total * D)`` range with no gaps, and ``origin % D == rank``
      makes inter-rank disjointness a structural property rather than an
      assumption about a magic upper bound.

      ``first_indices`` comes from ``async_unique(return_first_indices=True)``
      and, because ``unique`` is emitted in first-occurrence order, is
      *strictly monotonic* per rank.  Multiplying by ``world_size`` and
      adding ``rank`` preserves monotonicity within each rank, which is
      the only property the destination needs: an argsort over
      origin_indices restores input rank d's local unique sequence
      regardless of how Mirror-back interleaves shards from multiple
      sparse ranks, so dense d's local ``inv`` reconstructs the original
      batch byte-for-byte.
    * ``inv_tt`` keeps ``global_shape`` only; the framework's static-DP
      path handles its placement deterministically, no extra meta required.
    """
    ids_np = tt.flat_values

    # -- Invariants on input layout --
    assert ids_np is not None and ids_np.size > 0, (
        f"[InputSPMD._sparse_unique] ids_np must be non-empty, got "
        f"shape={getattr(ids_np, 'shape', None)}"
    )
    assert isinstance(world_size, int) and world_size >= 1, (
        f"[InputSPMD._sparse_unique] world_size must be a positive int, "
        f"got {world_size!r}"
    )
    assert isinstance(rank, int) and 0 <= rank < world_size, (
        f"[InputSPMD._sparse_unique] rank {rank!r} out of range "
        f"[0, world_size={world_size})"
    )

    unique_dlm, inv_dlm, first_dlm = await se.async_unique(
        ids_np.ravel(),
        return_inverse=True,
        return_first_indices=True,
    )

    backend = get_backend()
    unique_np = backend.from_dlpack(unique_dlm).astype(np.int64)
    inv_1d = backend.from_dlpack(inv_dlm).astype(np.int32).reshape(-1)
    first_local = backend.from_dlpack(first_dlm).astype(np.int64).reshape(-1)

    # -- Invariants on async_unique outputs --
    assert unique_np.ndim == 1, (
        f"[InputSPMD._sparse_unique] unique must be 1D, got shape={unique_np.shape}"
    )
    assert np.unique(unique_np).size == unique_np.size, (
        "[InputSPMD._sparse_unique] async_unique output contains duplicates"
    )
    assert first_local.shape == unique_np.shape, (
        f"[InputSPMD._sparse_unique] first_indices shape {first_local.shape} "
        f"!= unique shape {unique_np.shape}"
    )
    # async_unique emits unique in first-occurrence order, so first_indices
    # must be strictly monotonic — this is the contract that lets dense
    # rank d use argsort(origin_indices) to recover the producer's order.
    if first_local.size > 1:
        assert bool(np.all(np.diff(first_local) > 0)), (
            "[InputSPMD._sparse_unique] first_indices not strictly monotonic; "
            "async_unique's first-occurrence ordering contract was violated"
        )

    # -- Numerical correctness check: unique + inv reconstruction --
    # invariant: original_ids[i] == unique_ids[inv[i]] for all i
    original_ids = ids_np.ravel()
    assert inv_1d.shape[0] == original_ids.shape[0], (
        f"[InputSPMD._sparse_unique] inv length {inv_1d.shape[0]} "
        f"!= original length {original_ids.shape[0]}"
    )
    if inv_1d.size > 0:
        assert int(inv_1d.min()) >= 0 and int(inv_1d.max()) < unique_np.size, (
            f"[InputSPMD._sparse_unique] inv out of range: "
            f"min={int(inv_1d.min())}, max={int(inv_1d.max())}, "
            f"unique_size={unique_np.size}"
        )
    np.testing.assert_array_equal(
        unique_np[inv_1d],
        original_ids,
        err_msg="[InputSPMD] unique+inv reconstruction failed: unique[inv] != original",
    )

    # Numerical correctness check: first_indices points to first occurrence.
    # invariant: ids[first_indices[i]] == unique[i]
    np.testing.assert_array_equal(
        original_ids[first_local],
        unique_np,
        err_msg=(
            "[InputSPMD] first_indices contract violated: ids[first_indices] != unique"
        ),
    )

    # Map per-rank first-occurrence positions into the round-robin
    # interleaved global ravel.  origin_indices[i] is now the *absolute*
    # position of unique[i] in that global ravel; inter-rank disjointness
    # is structural (origin % world_size == rank) rather than reliant on a
    # magic capacity constant.
    origins = first_local * int(world_size) + int(rank)

    unique_tt = TransportTensor(
        flat_values=unique_np,
        origin_indices=ExplicitIndex(origins),
    )
    inv_tt = TransportTensor(
        flat_values=inv_1d.reshape(ids_np.shape),
        global_shape=tt.global_shape,
    )
    return unique_tt, inv_tt


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
)
class InputSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    """Deduplicates features via sparse_embedding.async_unique and routes to embedding tables."""

    def __init__(
        self,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        pass

    @operator(
        name="process",
        input_sharding={"data": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "unique_user_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "unique_item_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
    )
    async def process(self, data: UserBatch) -> InputNodeOutput:
        # -- Invariants on input feature layout --
        assert data is not None, "[InputSPMD.process] data is None"
        assert data.active.user_id.flat_values.ndim >= 1, (
            f"[InputSPMD.process] user_id must be at least 1D, "
            f"got shape={data.active.user_id.flat_values.shape}"
        )
        assert data.passive.target_item_id.flat_values.ndim >= 1, (
            f"[InputSPMD.process] target_item_id must be at least 1D, "
            f"got shape={data.passive.target_item_id.flat_values.shape}"
        )

        user_task = _sparse_unique(
            data.active.user_id,
            rank=self.rank,
            world_size=self.world_size,
        )
        item_task = _sparse_unique(
            data.passive.target_item_id,
            rank=self.rank,
            world_size=self.world_size,
        )

        (u_user, inv_user), (u_item, inv_item) = await asyncio.gather(
            user_task, item_task
        )

        print(
            f"[InputSPMD.process] rank={self.rank} "
            f"unique_users={u_user.flat_values} user_inv={inv_user.flat_values} "
            f"unique_items={u_item.flat_values} item_inv={inv_item.flat_values}"
        )

        return InputNodeOutput(
            unique_user_ids=u_user,
            user_inv=inv_user,
            unique_item_ids=u_item,
            item_inv=inv_item,
            dp_data=DPFeatures(
                labels=data.labels, dense_stats=data.passive.target_item_stats
            ),
        )
