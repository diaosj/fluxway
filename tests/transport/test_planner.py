"""Tests for TransportPlanner: routing math over `RoutingDomain`.

The new public surface is the single static method
`TransportPlanner.compute_plan(src_local_rank, domain)`.  Callers must
construct a `RoutingDomain` (typically via `RoutingDomain.from_layout`)
first.  These tests therefore drive the planner through the same
factory that production code uses.

Strict typing contract: ``axis_indices`` must be a tuple of
``IndicesSpec`` instances.  Bare tensors and ``None`` are rejected by
the factory \u2014 callers must wrap tensors in ``LiteralIndices`` /
``IdentityIndices`` / ``RangeIndices.{contiguous,cyclic,strided}``.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from fluxway.common.index_descriptor import RaggedIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.sharding import (
    IdentityIndices,
    LiteralIndices,
    P,
    RangeIndices,
    RoutingDomain,
    ShardingSpec,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.transport.planner import TransportPlanner


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def _domain(target, axis_sizes, axis_indices):
    return RoutingDomain.from_layout(
        target=target,
        axis_sizes=axis_sizes,
        axis_indices=axis_indices,
    )


# ---- Scenario 1: static -> static (chunk -> chunk, different num_workers) ----


@pytest.mark.parametrize(
    "src_rank, start, expected_counts",
    [
        (0, 0, [25, 25, 0, 0]),  # rank 0 owns [0, 50)  -> dst {0, 1}
        (1, 50, [0, 0, 25, 25]),  # rank 1 owns [50, 100) -> dst {2, 3}
    ],
    ids=["rank0", "rank1"],
)
def test_static_to_static_intersects(src_rank, start, expected_counts):
    """Source: chunk(2). Target: chunk(4). Each src only sends to the dst
    sub-ranks it actually owns rows for (no broadcast)."""
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    domain = _domain(
        dst_layout,
        axis_sizes=(100,),
        axis_indices=(RangeIndices.contiguous(start=start, length=50),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=src_rank, domain=domain)

    assert plan.target_num_workers == 4
    assert plan.send_counts.tolist() == expected_counts


def test_intersects_only_few_to_many_chunk_unaligned():
    """2 src -> 5 dst with non-divisible chunk: each src only sends to the
    dst sub-ranks it actually intersects (no broadcast).

    global=100, src_world=2, dst_world=5, dst chunk_size=20.
      - src_rank=0 owns [0, 50)  -> hits dst {0, 1, 2} (boundary half on dst 2)
      - src_rank=1 owns [50, 100) -> hits dst {2, 3, 4} (boundary half on dst 2)
    Locks the intersects-only invariant: src 0 must NOT send to dst {3, 4};
    src 1 must NOT send to dst {0, 1}.
    """
    dst_mesh = DeviceMesh(device_shape=(5,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # src_rank=0 perspective
    domain0 = _domain(
        dst_layout,
        axis_sizes=(100,),
        axis_indices=(RangeIndices.contiguous(start=0, length=50),),
    )
    plan0 = TransportPlanner.compute_plan(src_local_rank=0, domain=domain0)
    counts0 = plan0.send_counts.tolist()
    assert counts0 == [20, 20, 10, 0, 0], (
        f"src 0 must intersect only dst (0,1,2); got {counts0}"
    )
    # No broadcast: zero outbound traffic to dst {3, 4}.
    assert counts0[3] == 0 and counts0[4] == 0

    # src_rank=1 perspective
    domain1 = _domain(
        dst_layout,
        axis_sizes=(100,),
        axis_indices=(RangeIndices.contiguous(start=50, length=50),),
    )
    plan1 = TransportPlanner.compute_plan(src_local_rank=1, domain=domain1)
    counts1 = plan1.send_counts.tolist()
    assert counts1 == [0, 0, 10, 20, 20], (
        f"src 1 must intersect only dst (2,3,4); got {counts1}"
    )
    assert counts1[0] == 0 and counts1[1] == 0

    # sort_indices length stays at per-rank local count, NOT global N.
    assert int(plan0.sort_indices.shape[0]) == 50
    assert int(plan1.sort_indices.shape[0]) == 50


# ---- Scenario 2: static -> dynamic (hash via RaggedIndex) ----


def test_static_to_dynamic_hash():
    """Source holds 4 rows; target hash partitions them via RaggedIndex."""
    splits = [jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)]
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.hash(P("x")).bind(dst_mesh)

    # axis_size = 4 (global), indices = local row positions [0..3]
    domain = _domain(
        dst_layout,
        axis_sizes=(4,),
        axis_indices=(IdentityIndices(length=4),),
    )
    # Patch the descriptor in (the factory does not propagate per-call
    # descriptors; production layouts already carry it on the routing).
    patched_axes = tuple(
        dataclasses.replace(ax, index_descriptor=RaggedIndex(splits=splits))
        for ax in domain.axes
    )
    domain = dataclasses.replace(domain, axes=patched_axes)

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 3: dynamic -> static (hash -> chunk via ExplicitIndex) ----


def test_dynamic_to_static():
    """Source owns rows whose dest-rank is encoded in ExplicitIndex.

    Total global rows: 6.  ExplicitIndex assigns owner ranks [0,1,0,1,0,1].
    Local rank 0 therefore owns global rows {0, 2, 4}.  The destination
    is chunk(2) over dim_size=6, so chunk_size=3 and rows route as:
      row 0 -> dst 0, row 2 -> dst 0, row 4 -> dst 1.
    """
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(mesh)

    indices = jnp.array([0, 2, 4], dtype=jnp.int32)
    domain = _domain(
        dst_layout,
        axis_sizes=(6,),
        axis_indices=(LiteralIndices(length=3, values=indices),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    counts = plan.send_counts.tolist()
    assert counts == [2, 1]


def test_hash_literal_indices_keep_negative_bit_patterns_valid():
    dst_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_layout = ShardingSpec.hash(P("x")).bind(dst_mesh)

    keys = jnp.array([4, -1, 8], dtype=jnp.int32)
    domain = _domain(
        dst_layout,
        axis_sizes=(3,),
        axis_indices=(LiteralIndices(length=3, values=keys),),
    )

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    assert plan.send_counts.tolist() == [3]


def test_literal_indices_valid_length_masks_only_padding_tail():
    dst_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_layout = ShardingSpec.hash(P("x")).bind(dst_mesh)

    keys = jnp.array([4, -1, 8, -1], dtype=jnp.int32)
    domain = _domain(
        dst_layout,
        axis_sizes=(4,),
        axis_indices=(LiteralIndices(length=4, values=keys, valid_length=3),),
    )

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    assert plan.send_counts.tolist() == [3]


# ---- Scenario 4: dynamic -> dynamic (greedy via RaggedIndex) ----


def test_dynamic_to_dynamic():
    """Source: greedy(2) on RaggedIndex; target hash via same descriptor."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.hash(P("x")).bind(mesh)

    domain = _domain(
        dst_layout,
        axis_sizes=(3,),
        axis_indices=(IdentityIndices(length=2),),
    )
    patched_axes = tuple(
        dataclasses.replace(ax, index_descriptor=RaggedIndex(splits=splits))
        for ax in domain.axes
    )
    domain = dataclasses.replace(domain, axes=patched_axes)

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    counts = plan.send_counts.tolist()
    # rank 0's indices [0, 1] map to splits starts [0, 10] -> hash via
    # IndexedModulo: values % 2 -> [0, 0]
    assert sum(counts) == 2


# ---- Scenario 5: replicate (fan-out across replica mesh axis) ----


def test_replicate_fanout():
    """Source: chunk(2) on x. Target: chunk(2) on x with a replica axis y."""
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # axis_size = global 10, rank 0 owns local 5 rows.
    domain = _domain(
        dst_layout,
        axis_sizes=(10,),
        axis_indices=(RangeIndices.contiguous(start=0, length=5),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    assert plan.replication_factor == 2

    counts = plan.send_counts.tolist()
    assert counts[0] == 5
    assert counts[1] == 0
    assert counts[2] == 0
    assert counts[3] == 0


# ---- Scenario 6: ragged data reorder via plan.sort_indices ----


def test_ragged_data_reorder():
    """Verify that plan.sort_indices correctly reorders ragged TransportTensor."""
    splits = [jnp.array([0, 2, 3, 6, 8], dtype=jnp.int32)]
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    local_data = TransportTensor(values, splits)

    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.cyclic(P("x")).bind(dst_mesh)

    # Single source rank owns all 4 rows: indices = identity over [0, 4).
    domain = _domain(
        dst_layout,
        axis_sizes=(4,),
        axis_indices=(IdentityIndices(length=4),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]

    reordered = local_data.gather(plan.sort_indices)
    reordered_lists = reordered.to_lists()

    assert reordered_lists[0] == [1.0, 2.0]
    assert reordered_lists[1] == [4.0, 5.0, 6.0]
    assert reordered_lists[2] == [3.0]
    assert reordered_lists[3] == [7.0, 8.0]


# ---- iter_device_slices ----


def test_iter_device_slices():
    """Verify iter_device_slices yields correct (device, offset, count) tuples."""
    dst_mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    dst_layout = ShardingSpec.cyclic(P("x")).bind(dst_mesh)

    domain = _domain(
        dst_layout,
        axis_sizes=(6,),
        axis_indices=(IdentityIndices(length=6),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    slices = list(plan.iter_device_slices())

    assert len(slices) == 3
    total = sum(count for _, _, count in slices)
    assert total == 6


# ---- Multi-axis routing (Cartesian over 2D mesh) ----


def test_multi_axis_routing():
    """Verify the planner accepts multi-axis RoutingDomains.

    `P("x", "y")` produces two routed dims (dim 0 sharded by x, dim 1 by y),
    therefore the domain has two `RoutingAxis` entries. The Cartesian
    product of `axis.size` is the total entity count routed by this plan.
    """
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    dst_layout = ShardingSpec.chunk(P("x", "y")).bind(dst_mesh)

    domain = _domain(
        dst_layout,
        axis_sizes=(2, 2),
        axis_indices=(
            IdentityIndices(length=2),
            IdentityIndices(length=2),
        ),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    assert plan.target_num_workers == 4
    assert sum(plan.send_counts.tolist()) == 4


# ---- Strict typing contract ----


def test_no_implicit_none():
    """``from_layout`` must reject bare ``None`` indices (strict typing)."""
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    with pytest.raises(TypeError, match="IndicesSpec"):
        RoutingDomain.from_layout(
            target=dst_layout,
            axis_sizes=(4,),
            axis_indices=(None,),  # type: ignore
        )


def test_no_implicit_bare_tensor():
    """``from_layout`` must reject bare runtime tensors (strict typing)."""
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    with pytest.raises(TypeError, match="IndicesSpec"):
        RoutingDomain.from_layout(
            target=dst_layout,
            axis_sizes=(4,),
            axis_indices=(jnp.arange(0, 4, dtype=jnp.int32),),  # type: ignore
        )


def test_indices_semantics_decoupled_from_axis_size():
    """Same axis.size + same strategy + different indices length -> different
    sort_indices length.  Locks in the orthogonality between
    ``axis.size`` (strategy ``dim_size`` \u2014 GLOBAL) and ``len(indices)``
    (LOCAL per-rank entity count).
    """
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # Same global axis size; two ranks holding different local slices.
    domain_a = _domain(
        dst_layout,
        axis_sizes=(100,),
        axis_indices=(RangeIndices.contiguous(start=0, length=10),),
    )
    domain_b = _domain(
        dst_layout,
        axis_sizes=(100,),
        axis_indices=(RangeIndices.contiguous(start=0, length=40),),
    )

    plan_a = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_a)
    plan_b = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_b)

    assert int(plan_a.sort_indices.shape[0]) == 10
    assert int(plan_b.sort_indices.shape[0]) == 40
    # axis.size unchanged (both layouts share the same dst); only the
    # per-rank slice differs.
    assert domain_a.axes[0].size == domain_b.axes[0].size == 100


def test_range_indices_planner_path_does_not_call_materialize(monkeypatch):
    """RangeIndices is a symbolic routing source; planner must not expand it via _materialize."""
    import fluxway.transport.planner as planner_mod

    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    domain = _domain(
        dst_layout,
        axis_sizes=(16,),
        axis_indices=(RangeIndices.contiguous(start=0, length=8),),
    )

    def forbidden(spec, backend):
        raise AssertionError(f"_materialize called for {type(spec).__name__}")

    monkeypatch.setattr(planner_mod, "_materialize", forbidden)
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    assert plan.send_counts.tolist() == [8, 0]


# ============================================================
# Patch L: zip-mode + ConstantIndices fast path
# ============================================================


def test_zip_mode_const_axis_does_not_inflate_sort_indices():
    """Multi-axis zip + ConstantIndices keeps sort_indices.len == primary length.

    Locks in the Patch L fix: under transport's row-binding
    (``backend.take(data, sort_indices, axes=(0,))``), an auxiliary
    routed dim must collapse to a single per-rank decision via
    ``ConstantIndices`` so the planner does NOT Cartesian-inflate
    sort_indices to ``primary_len * aux_len`` (the original sparse
    embedding bug where ``sort_indices.len = N * D``).
    """
    from fluxway.common.types.sharding import ConstantIndices

    # dst sharded on (dp=2, tp=2). Use isomorphic src so the aux dim
    # collapses to ConstantIndices.  Primary axis carries N=8 routing
    # keys via LiteralIndices.
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("dp", "tp"))
    dst_layout = ShardingSpec.chunk(P("dp", "tp")).bind(dst_mesh)

    primary_keys = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
    domain = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(8, 8),  # global sizes per dim
        axis_indices=(
            LiteralIndices(length=8, values=primary_keys),
            ConstantIndices(value=0, length=8),  # all 8 rows live on tp=0
        ),
        combine="zip",
    )

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    # Critical invariant: sort_indices length == primary entity count,
    # NOT primary_len * aux_len.
    assert int(plan.sort_indices.shape[0]) == 8, (
        f"sort_indices was inflated by aux axis under zip mode; "
        f"got {plan.sort_indices.shape[0]}, expected 8"
    )

    # All 8 rows live on tp=0; the dp routing splits 0..3 -> dp=0 and
    # 4..7 -> dp=1.  Final dst ranks: 0..3 -> rank 0 (dp=0,tp=0),
    # 4..7 -> rank 2 (dp=1,tp=0).
    counts = plan.send_counts.tolist()
    assert counts == [4, 0, 4, 0]
    assert plan.target_num_workers == 4


def test_zip_mode_const_axis_value_routes_correctly():
    """ConstantIndices.value selects the dst sub-rank for the aux dim."""
    from fluxway.common.types.sharding import ConstantIndices

    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("dp", "tp"))
    dst_layout = ShardingSpec.chunk(P("dp", "tp")).bind(dst_mesh)

    primary_keys = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)

    # value=4 maps to tp=1 under chunk(global=8, sub=2) -> chunk_size=4.
    domain_tp1 = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(8, 8),
        axis_indices=(
            LiteralIndices(length=8, values=primary_keys),
            ConstantIndices(value=4, length=8),
        ),
        combine="zip",
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_tp1)

    # Now 0..3 -> rank 1 (dp=0,tp=1), 4..7 -> rank 3 (dp=1,tp=1).
    counts = plan.send_counts.tolist()
    assert counts == [0, 4, 0, 4]


def test_zip_mode_rejects_inconsistent_axis_lengths():
    """Zip-mode invariant: non-constant axes must share length."""
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("dp", "tp"))
    dst_layout = ShardingSpec.chunk(P("dp", "tp")).bind(dst_mesh)

    domain = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(8, 8),
        axis_indices=(
            LiteralIndices(
                length=8,
                values=jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32),
            ),
            RangeIndices.contiguous(start=0, length=4),  # mismatched length!
        ),
        combine="zip",
    )
    with pytest.raises(ValueError, match="non-constant axes to share length"):
        TransportPlanner.compute_plan(src_local_rank=0, domain=domain)


def test_zip_mode_rejects_unfinalised_constant_indices():
    """ConstantIndices(length=-1) must be finalised before reaching planner."""
    from fluxway.common.types.sharding import ConstantIndices

    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_layout = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    domain = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(4,),
        axis_indices=(ConstantIndices(value=0, length=-1),),
        combine="zip",
    )
    with pytest.raises(ValueError, match="placeholder length=-1"):
        TransportPlanner.compute_plan(src_local_rank=0, domain=domain)


def test_cartesian_mode_default_unchanged():
    """Default combine='cartesian' preserves the legacy multi-axis semantics.

    Guards the dual-mode contract: changing the L1 transport callers to
    ``combine='zip'`` must NOT alter behaviour for the planner's general
    routing-engine consumers.
    """
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    dst_layout = ShardingSpec.chunk(P("x", "y")).bind(dst_mesh)

    domain = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(2, 2),
        axis_indices=(
            IdentityIndices(length=2),
            IdentityIndices(length=2),
        ),
    )
    assert domain.combine == "cartesian"
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)
    # Cartesian: 2*2 = 4 entities, one per dst rank.
    assert int(plan.sort_indices.shape[0]) == 4
    assert sum(plan.send_counts.tolist()) == 4


# ============================================================
# K-8: multi-axis replica fan-out (RoutingDomain.replica)
# ============================================================


def test_multi_replica_axes_dest_devices_fanout():
    """Two replica mesh axes both fan out: dest_devices.shape == (N, 4).

    When the dst layout shards on a single mesh axis (``dp``) and leaves
    two further mesh axes (``rep1``, ``rep2``) replicated,
    ``RoutingDomain.from_layout`` must capture them in ``ReplicaSpec``
    and the planner must broadcast ``primary_dest`` over the
    ``replica_size`` Cartesian product without inflating ``sort_indices``.

    Locks the contract:
        - ``domain.replica.size == prod(replica mesh shape)``
        - ``plan.dest_devices.shape == (N, replica_size)``
        - ``plan.sort_indices.shape[0] == N`` (NOT inflated)
        - ``plan.replication_factor == replica_size``
        - ``send_counts`` are accumulated on the (rep1=0, rep2=0)
          coordinate column (the canonical primary destination).
    """
    dst_mesh = DeviceMesh(device_shape=(2, 2, 2), axis_names=("dp", "rep1", "rep2"))
    # P("dp") shards only on dp; rep1/rep2 are auto-collected as
    # replica_axes by ShardingSpec.bind.
    dst_layout = ShardingSpec.chunk(P("dp")).bind(dst_mesh)
    assert set(dst_layout.replica_axes or ()) == {"rep1", "rep2"}
    assert dst_layout.replica_size == 4

    domain = RoutingDomain.from_layout(
        target=dst_layout,
        axis_sizes=(8,),
        axis_indices=(IdentityIndices(length=8),),
    )
    assert domain.replica is not None
    assert set(domain.replica.axes) == {"rep1", "rep2"}
    assert domain.replica_size == 4

    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    # dest_devices fans out to 4 replicas per primary entity.
    assert tuple(plan.dest_devices.shape) == (8, 4)
    assert plan.replication_factor == 4

    # sort_indices stays at primary count -- the replica fan-out lives in
    # the second axis of dest_devices, not in sort_indices.
    assert int(plan.sort_indices.shape[0]) == 8

    # 8 dst ranks total (full mesh size). primary_dest_flat picks
    # dest_devices[:, 0] which corresponds to (rep1=0, rep2=0); under
    # row-major device_ids, dst_rank = dp*4 + rep1*2 + rep2, so
    # entities 0..3 -> dp=0 -> rank 0, entities 4..7 -> dp=1 -> rank 4.
    assert plan.target_num_workers == 8
    counts = plan.send_counts.tolist()
    assert counts == [4, 0, 0, 0, 4, 0, 0, 0], (
        f"send_counts must accumulate on the canonical (rep1=0, rep2=0) "
        f"column; got {counts}"
    )
