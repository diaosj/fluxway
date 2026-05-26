"""Unit tests for the ``SrcSideView`` derivation (Patch K-1).

``derive_src_side_view`` is a pure function consumed by the L1 transport
layer to seal the src-side reasoning that used to leak into the call
stack. These tests pin its three structural responsibilities:

  F: per-routed-dim LOCAL ``IndicesSpec`` derivation (multi-axis).
  G: ``is_sender_active`` owner-gating on src-replicated mesh axes.
  J: ``expected_senders_per_dst`` static feasibility (None for data-driven
     routing, concrete tuple for static Chunk/Cyclic).
"""

import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.sharding import (
    IdentityIndices,
    P,
    RangeIndices,
    ShardingSpec,
)
from fluxway.transport.communication import (
    SrcSideView,
    derive_src_side_view,
)


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


# ------------------------------------------------------------------
# F: per-dim LOCAL IndicesSpec derivation
# ------------------------------------------------------------------


def test_per_dim_indices_single_axis_chunk():
    """Single routing dim, src chunked: each rank owns a contiguous slice."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # rank 0 of a chunk(2) over global=100 owns [0, 50)
    view0 = derive_src_side_view(
        src_spec=src, src_local_rank=0, target=dst, dst_global_shape=(100,)
    )
    assert len(view0.per_dim_indices) == 1
    dim, spec = view0.per_dim_indices[0]
    assert dim == 0
    assert isinstance(spec, RangeIndices)
    assert spec.start == 0
    assert spec.length == 50

    view1 = derive_src_side_view(
        src_spec=src, src_local_rank=1, target=dst, dst_global_shape=(100,)
    )
    _, spec1 = view1.per_dim_indices[0]
    assert spec1.start == 50
    assert spec1.length == 50


def test_per_dim_indices_dp_tp_heterogeneous():
    """Multi-routing-dim: every routed dim gets its own LOCAL slice.

    src=(dp=2, tp=4) chunked on both dims, dst=(dp=4, tp=2). The src
    rank (dp_coord=1, tp_coord=2) owns dim 0 = [50, 100), dim 1 = [50, 75)
    when global = (100, 100). Verifies that the non-primary dim is *not*
    silently fattened to the global extent (the F bug).
    """
    src_mesh = DeviceMesh(device_shape=(2, 4), axis_names=("dp", "tp"))
    dst_mesh = DeviceMesh(device_shape=(4, 2), axis_names=("dp", "tp"))
    src = ShardingSpec.chunk(P("dp", "tp")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("dp", "tp")).bind(dst_mesh)

    # src rank 6 has coord (dp=1, tp=2) on a (2,4) row-major mesh
    src_rank = src_mesh.coordinate_to_device((1, 2))
    view = derive_src_side_view(
        src_spec=src,
        src_local_rank=int(src_rank),
        target=dst,
        dst_global_shape=(100, 100),
    )
    dims = dict(view.per_dim_indices)
    assert set(dims) == {0, 1}
    assert isinstance(dims[0], RangeIndices)
    assert dims[0].start == 50
    assert dims[0].length == 50
    assert isinstance(dims[1], RangeIndices)
    assert dims[1].start == 50
    assert dims[1].length == 25


def test_per_dim_indices_src_replicated_dim_uses_full_extent():
    """When src does not partition a routed dst dim, src holds the full extent."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    # src only partitions dim 0, leaves dim 1 replicated.
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x", "y")).bind(dst_mesh)

    view = derive_src_side_view(
        src_spec=src, src_local_rank=0, target=dst, dst_global_shape=(100, 50)
    )
    dims = dict(view.per_dim_indices)
    # dim 0 partitioned in src AND src.shard_size(0)==dst.shard_size(0)==2
    # -> isomorphic chunk geometry -> ConstantIndices fast path (Patch L).
    # value is the GLOBAL chunk start; length=-1 is the placeholder that
    # ``_build_routing_domain`` finalises to the primary axis length.
    from fluxway.common.types.sharding import ConstantIndices

    assert isinstance(dims[0], ConstantIndices)
    assert dims[0].value == 0  # rank 0 -> shard 0 -> start = 0
    assert dims[0].length == -1  # placeholder, finalised downstream
    # dim 1 NOT partitioned in src -> full extent identity
    assert isinstance(dims[1], IdentityIndices)
    assert dims[1].length == 50


# ------------------------------------------------------------------
# G: owner-gating on src-replicated mesh axes
# ------------------------------------------------------------------


def test_owner_gating_src_fully_partitioned_all_active():
    """Every src rank is the canonical sender when src has no replica axis."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    for r in range(2):
        v = derive_src_side_view(
            src_spec=src, src_local_rank=r, target=dst, dst_global_shape=(100,)
        )
        assert v.is_sender_active is True


def test_owner_gating_src_replicated_y_axis():
    """src=(2,2) partitioned only on x => y is replica axis. Coord y=0 is owner."""
    src_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # mesh row-major: rank id = x_coord*2 + y_coord
    # (x=0,y=0)=0 owner; (x=0,y=1)=1 NOT owner;
    # (x=1,y=0)=2 owner; (x=1,y=1)=3 NOT owner.
    expected = {0: True, 1: False, 2: True, 3: False}
    for r, want in expected.items():
        v = derive_src_side_view(
            src_spec=src, src_local_rank=r, target=dst, dst_global_shape=(100,)
        )
        assert v.is_sender_active is want, f"rank {r}: expected is_sender_active={want}"


# ------------------------------------------------------------------
# J: expected_senders_per_dst static feasibility
# ------------------------------------------------------------------


def test_expected_senders_static_chunk_x_chunk():
    """Static Chunk x Chunk: per-dst sender count is fully derivable."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    view = derive_src_side_view(
        src_spec=src, src_local_rank=0, target=dst, dst_global_shape=(24,)
    )
    n = view.expected_senders_per_dst
    assert n is not None
    # src rank 0 owns [0, 12) -> hits dst 0,1,2 (chunk size 4 each)
    # src rank 1 owns [12,24) -> hits dst 3,4,5
    # therefore each dst has exactly 1 sender.
    assert n == (1, 1, 1, 1, 1, 1)


def test_expected_senders_data_driven_returns_none():
    """Hash routing is data-driven => static count is unknowable."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.hash(P("x")).bind(dst_mesh)

    view = derive_src_side_view(
        src_spec=src, src_local_rank=0, target=dst, dst_global_shape=(100,)
    )
    assert view.expected_senders_per_dst is None


def test_expected_senders_excludes_non_owner_ranks():
    """Non-owner src ranks should not contribute to expected_senders_per_dst."""
    # src=(2,2) partitioned only on x; y is replica. src_world=4 but only
    # ranks (y=0) are owners (2 owners).
    src_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    view = derive_src_side_view(
        src_spec=src, src_local_rank=0, target=dst, dst_global_shape=(100,)
    )
    n = view.expected_senders_per_dst
    assert n is not None
    # Total over all dst should equal 2 (only 2 owner src ranks active).
    # Two owners each cover 2 dst ranks (chunk(2) -> chunk(4)), so the
    # vector should sum to 4 (each owner hits 2 dsts: [0,1] and [2,3]).
    assert sum(n) == 4
    assert n == (1, 1, 1, 1)


# ------------------------------------------------------------------
# Structural validation
# ------------------------------------------------------------------


def test_view_rejects_empty_per_dim_indices():
    with pytest.raises(ValueError, match="non-empty"):
        SrcSideView(
            per_dim_indices=(),
            is_sender_active=True,
            expected_senders_per_dst=None,
        )


def test_view_rejects_duplicate_dims():
    with pytest.raises(ValueError, match="duplicate"):
        SrcSideView(
            per_dim_indices=(
                (0, IdentityIndices(length=10)),
                (0, IdentityIndices(length=20)),
            ),
            is_sender_active=True,
            expected_senders_per_dst=None,
        )


def test_view_rejects_non_indicesspec_value():
    with pytest.raises(TypeError, match="IndicesSpec"):
        SrcSideView(
            per_dim_indices=((0, "not an indicesspec"),),  # type: ignore[arg-type]
            is_sender_active=True,
            expected_senders_per_dst=None,
        )


# ------------------------------------------------------------------
# K-5/K-7: distributor-level integration of SrcSideView (last_src_view
# and expected_n_for_dst). These verify that ``derive_src_side_view`` is
# wired through ``_plan_single_tensor_group`` and that upper layers can
# discover the per-dst expected sender count without re-implementing the
# derivation.
# ------------------------------------------------------------------


def _build_chunked_tensor(local_shape, global_shape):
    import jax.numpy as jnp
    from fluxway.common.types.tensor import TransportTensor

    flat = jnp.zeros(local_shape, dtype=jnp.float32)
    return TransportTensor(flat, global_shape=global_shape)


def test_distributor_last_src_view_populated_after_plan():
    """After ``plan_and_shard`` runs, the distributor exposes the SrcSideView."""
    from fluxway.transport.communication import DataDistributor

    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    tt = _build_chunked_tensor(local_shape=(12,), global_shape=(24,))
    dist = DataDistributor(local_rank=0)
    assert dist.last_src_view is None
    dist.plan_and_shard(local_data=tt, source=src, target=dst, capacity=12)

    view = dist.last_src_view
    assert view is not None
    assert isinstance(view, SrcSideView)
    # Single routed dim, src chunked -> RangeIndices over rank 0's slice.
    dim, spec = view.per_dim_indices[0]
    assert dim == 0
    assert isinstance(spec, RangeIndices)
    assert spec.start == 0
    assert spec.length == 12
    assert view.is_sender_active is True


def test_distributor_expected_n_for_dst_static_chunk():
    """Static chunk-to-chunk: expected_n is the closed-form per-dst count.

    src=chunk(2) over global=24, dst=chunk(6). Each src rank owns 12
    contiguous rows and reaches 3 contiguous dst ranks; therefore each
    dst receives exactly 1 sender.
    """
    from fluxway.transport.communication import DataDistributor

    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    tt = _build_chunked_tensor(local_shape=(12,), global_shape=(24,))
    dist = DataDistributor(local_rank=0)
    dist.plan_and_shard(local_data=tt, source=src, target=dst, capacity=12)

    src_world = src_mesh.size
    for d in range(dst_mesh.size):
        assert dist.expected_n_for_dst(d, src_world) == 1


def test_distributor_expected_n_for_dst_data_driven_falls_back():
    """Hash routing -> expected_senders_per_dst is None -> always-full SPMD.

    The helper must return ``src_world_size`` so the receiver schedules
    an arrival for every src rank (including those sending empty
    packets). Anything else would deadlock the SPMD round.
    """
    import jax.numpy as jnp
    from fluxway.common.types.tensor import TransportTensor
    from fluxway.transport.communication import DataDistributor

    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("y",))
    src = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst = ShardingSpec.hash(P("y")).bind(dst_mesh)

    keys = jnp.arange(0, 6, dtype=jnp.int32)
    tt = TransportTensor(keys, global_shape=(12,))
    dist = DataDistributor(local_rank=0)
    dist.plan_and_shard(local_data=tt, source=src, target=dst, capacity=6)

    view = dist.last_src_view
    assert view is not None
    assert view.expected_senders_per_dst is None

    src_world = src_mesh.size
    for d in range(dst_mesh.size):
        assert dist.expected_n_for_dst(d, src_world) == src_world


def test_distributor_expected_n_for_dst_no_plan_yet():
    """Before any plan: helper returns the always-full fallback."""
    from fluxway.transport.communication import DataDistributor

    dist = DataDistributor(local_rank=0)
    assert dist.last_src_view is None
    # Any positive src_world_size must be returned verbatim.
    assert dist.expected_n_for_dst(0, 4) == 4
    assert dist.expected_n_for_dst(3, 4) == 4
