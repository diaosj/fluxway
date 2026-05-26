"""Resharding regression tests over the strict-typed ``RoutingDomain`` API.

These cases used to call the legacy ``TransportPlanner.plan(src_spec=...,
dst_spec=...)`` entry, which has been removed. The L2 planner now consumes
a pre-built ``RoutingDomain``; resharding semantics are validated by
constructing the domain through ``RoutingDomain.from_layout`` and supplying
the per-rank ``axis_indices`` explicitly. ``IndicesSpec`` subclasses are
mandatory — bare tensors and ``None`` are rejected at the API boundary.
"""

import jax.numpy as jnp
import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import get_backend, set_backend
from fluxway.common.types.sharding import (
    LiteralIndices,
    P,
    RangeIndices,
    RoutingDomain,
    ShardingSpec,
)
from fluxway.transport.planner import TransportPlanner


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_resharding_2_to_6_static_chunk():
    """Reproduce the Mesh 2 -> Mesh 6 data loss/mis-segmentation."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))

    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    # Topology check via ShardingSpec — independent of L2 planner.
    producers_for_dense_0 = src_spec.resolve_producers(dst_spec, target_rank=0)
    assert 0 in producers_for_dense_0
    assert 1 not in producers_for_dense_0  # idx 12 outside dense 0's [0..3]

    producers_for_dense_3 = src_spec.resolve_producers(dst_spec, target_rank=3)
    assert 0 not in producers_for_dense_3
    assert 1 in producers_for_dense_3

    # From source rank 0 (owns global rows [0, 12)):
    #   chunk(6) over dim_size=24 -> chunk_size=4, so [0..3]->dst0,
    #   [4..7]->dst1, [8..11]->dst2, nothing further.
    domain_0 = RoutingDomain.from_layout(
        target=dst_spec,
        axis_sizes=(24,),
        axis_indices=(RangeIndices.contiguous(start=0, length=12),),
    )
    plan_0 = TransportPlanner.compute_plan(src_local_rank=0, domain=domain_0)
    assert plan_0.send_counts.tolist() == [4, 4, 4, 0, 0, 0]

    # From source rank 1 (owns global rows [12, 24)):
    domain_1 = RoutingDomain.from_layout(
        target=dst_spec,
        axis_sizes=(24,),
        axis_indices=(RangeIndices.contiguous(start=12, length=12),),
    )
    plan_1 = TransportPlanner.compute_plan(src_local_rank=1, domain=domain_1)
    assert plan_1.send_counts.tolist() == [0, 0, 0, 4, 4, 4]


def test_identity_collapse_hash():
    """Reproduce the bug where unique_ids are overwritten by rank indices.

    Source has a single rank holding ``[0, 6, 12, 18]``; destination is
    ``hash(P("x"))`` on 6 ranks. All values map to ``0 mod 6`` so they
    must all land on dst rank 0, and the *payload* must remain the IDs
    themselves (not the routing keys).
    """
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))
    dst_spec = ShardingSpec.hash(P("x")).bind(dst_mesh)

    data = jnp.array([0, 6, 12, 18], dtype=jnp.int32)

    # For hash routing, the LiteralIndices values ARE the hash keys.
    # axis_sizes is the global dim length on the producer side (4).
    domain = RoutingDomain.from_layout(
        target=dst_spec,
        axis_sizes=(4,),
        axis_indices=(LiteralIndices(length=4, values=data),),
    )
    plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)

    counts = plan.send_counts.tolist()
    assert counts[0] == 4
    for i in range(1, 6):
        assert counts[i] == 0

    # The plan reorders by the routing keys, but the original payload must
    # survive the reorder (no metadata-vs-payload collapse).
    backend = get_backend()
    reordered_data = backend.take(data, plan.sort_indices)
    assert backend.to_numpy(reordered_data).tolist() == [0, 6, 12, 18]
    assert not all(x == 0 for x in backend.to_numpy(reordered_data).tolist())
