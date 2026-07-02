"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

import dataclasses
from typing import Any
from collections.abc import Callable
import logging
import math

import asyncio
import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.transport.planner import TransportPlanner
from fluxway.common.types.sharding import (
    PhysicalLayout,
    RoutingAxis,
    RoutingDomain,
    ChunkPartition,
    ConstantIndices,
    CyclicPartition,
    IdentityIndices,
    IndicesSpec,
    LiteralIndices,
    RangeIndices,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.execution import compile
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor
from fluxway.common.ragged import encode_ragged_level_count, ragged_level_count
from fluxway.common.tensor_components import (
    index_descriptor_length,
    index_source_to_indices_spec,
    routing_entity_count,
    select_destination_content_index_source,
    select_source_geometry_index_source,
)
from fluxway.transport.manifest import (
    EDGE_EMPTY_KIND,
    EDGE_EXPECTED_COUNT,
    EDGE_TUUID,
    PAYLOAD_VALID_COUNT,
    EmptyKind,
    PayloadRole,
    SenderEmissionMode,
    TransportFrameBuilder,
    TransportPlanManifest,
)

logger = logging.getLogger(__name__)


# ============================================================
# SrcSideView — transport层 own 的 src-side derivation
# ============================================================
#
# Patch K 的核心抽象：把调用栈中各处散落的 src 侧推导逻辑收敛到一个纯
# 函数。函数完全由 src_spec + src_local_rank + dst_partitions 推导，不依赖运
# 行时数据（除非路由本身是数据驱动）。这同时修复：
#
#  - F: 多 routing dim 异构切分下非 primary dim 丢失 LOCAL indices
#  - G: src 侧 replication 发送权未去重（owner gating）
#  - J: 接收侧 N 隐式协议（transport 未 own expected_senders 推导）
#
# 设计不变量：
#  1. always-full 是正确性默认 —— expected_senders_per_dst is None 时指示
#     调用方必须遵循 always-full SPMD（所有 src rank 都参与、含空包）。
#  2. 静态可推导路径是优化 —— expected_senders_per_dst is tuple[int, ...]
#     时，transport 可以跳过零 sender 的 dst 以及非 owner src rank，同时
#     把 N 派发给接收侧 ArrivalBarrier 以消除上层隐式协议。
#  3. eager / graph 不分叉 —— 分叉点恶是路由是否数据驱动。
# ============================================================


@dataclasses.dataclass(frozen=True)
class SrcSideView:
    """Source-side derivation consumed by ``_build_routing_domain``.

    Carries everything L2 needs to know about the *sender's* geometry,
    derived purely from ``src_spec + src_local_rank + dst_partitions``.
    Replaces the loose ``src_geometry_indices``/``src_spec``/``src_local_rank``
    triplet that was previously threaded through the call stack and
    silently dropped (`del src_local_rank, src_spec, ...`) inside
    ``_build_routing_domain``.

    Attributes:
        per_dim_indices: For every dim that is routed on the *destination*
            side, the LOCAL ``IndicesSpec`` describing what this src rank
            owns along that dim. Stored as a sorted tuple of
            ``(routed_dim, IndicesSpec)`` pairs (frozen-dataclass-friendly
            dict replacement). Empty tuple is illegal — callers must
            short-circuit fully-replicated dst before reaching the
            planner.
        is_sender_active: ``False`` when this src rank lies on a
            non-canonical coordinate of *some* mesh axis on which the src
            layout is replicated (i.e. some other src rank holds the
            same data and is the canonical sender). When ``False`` the
            distributor must still participate in the SPMD round but only
            transmits empty packets.
        expected_senders_per_dst: For each dst rank ``d`` (indexed by
            ``dst_world_size``), the number of src ranks that will send
            *non-empty* payload to ``d``. ``None`` indicates the routing
            graph contains at least one data-driven dim (Hash / Indexed),
            so this count cannot be computed without running data
            through the routing function — callers must fall back to
            ``always-full SPMD`` and configure ``ArrivalBarrier`` with
            ``N = src_world_size``.
    """

    per_dim_indices: tuple[tuple[int, IndicesSpec], ...]
    is_sender_active: bool
    expected_senders_per_dst: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not self.per_dim_indices:
            raise ValueError(
                "SrcSideView.per_dim_indices must be non-empty; fully "
                "replicated targets should be filtered out before reaching L2."
            )
        seen: set[int] = set()
        for dim, spec in self.per_dim_indices:
            if not isinstance(dim, int) or dim < 0:
                raise TypeError(
                    f"per_dim_indices key must be a non-negative int dim; got {dim!r}"
                )
            if dim in seen:
                raise ValueError(f"per_dim_indices contains duplicate dim={dim}")
            seen.add(dim)
            if not isinstance(spec, IndicesSpec):
                raise TypeError(
                    f"per_dim_indices[{dim}] must be IndicesSpec; got "
                    f"{type(spec).__name__}"
                )
        if self.expected_senders_per_dst is not None:
            for n in self.expected_senders_per_dst:
                if not isinstance(n, int) or n < 0:
                    raise TypeError(
                        f"expected_senders_per_dst entries must be non-negative "
                        f"int; got {n!r}"
                    )

    @property
    def routed_dims(self) -> tuple[int, ...]:
        """Sorted tuple of routed dim indices."""
        return tuple(d for d, _ in self.per_dim_indices)

    def indices_for(self, dim: int) -> IndicesSpec:
        """Look up the LOCAL ``IndicesSpec`` for a routed dim."""
        for d, spec in self.per_dim_indices:
            if d == dim:
                return spec
        raise KeyError(f"dim {dim} is not routed in this SrcSideView")


def _src_replicated_axes(src_spec: PhysicalLayout) -> tuple[str, ...]:
    """Mesh axes on which the src layout is replicated (not in partitions).

    A src rank's coord on these axes determines whether it is the
    canonical sender for the data fragment (coord 0 → owner). Empty tuple
    means src is fully partitioned — every rank is its own owner.
    """
    if src_spec is None or src_spec.mesh is None:
        return ()
    used: set[str] = set()
    for axes in src_spec.partitions.values():
        for ax in axes:
            used.add(ax)
    return tuple(name for name in src_spec.mesh.axis_names if name not in used)


def _is_owner_on_replicated_axes(
    src_spec: PhysicalLayout | None, src_local_rank: int
) -> bool:
    """True iff ``src_local_rank`` has coord 0 on every src-replicated axis.

    On the canonical (coord==0) rank we transmit real payload; on the
    other replicas we still participate in the SPMD round but emit empty
    packets. This collapses ``K`` duplicate sends to 1 without dropping
    SPMD safety.
    """
    if src_spec is None:
        return True
    if src_spec.mesh is None:
        raise ValueError("src_spec.mesh is required for replicated-axis owner gating.")
    rep_axes = _src_replicated_axes(src_spec)
    if not rep_axes:
        return True
    coords = src_spec.mesh.get_coordinate(src_local_rank)
    axis_names = list(src_spec.mesh.axis_names)
    for ax in rep_axes:
        if ax in axis_names and coords[axis_names.index(ax)] != 0:
            return False
    return True


def _is_data_driven_routing(target: PhysicalLayout) -> bool:
    """True iff *any* routed dim depends on runtime data values.

    When True the per-dst sender count cannot be derived statically and
    callers must fall back to always-full SPMD with ``N = src_world_size``.
    """
    if target is None:
        return False
    return target.is_content_addressable


def _src_local_indices_for_routed_dim(
    src_spec: PhysicalLayout | None,
    src_local_rank: int,
    dim: int,
    dst_global_dim: int,
    target: PhysicalLayout | None = None,
) -> IndicesSpec:
    """Derive the LOCAL IndicesSpec held by ``src_local_rank`` along ``dim``.

    Structural cases (no runtime data inspection):

    1. ``src_spec`` does not partition ``dim`` (src is replicated on this
       dim, or src has no spec at all): the rank holds the *full* global
       extent — ``IdentityIndices(length=dst_global_dim)``.
    2. ``src_spec`` partitions ``dim`` AND the dst chunk geometry is
       *isomorphic* (``src.shard_size(dim) == dst.shard_size(dim)``,
       i.e. the chunk boundary alignment maps each src shard to exactly
       one dst sub-rank): every local entity (row) on this rank shares
       the same dst sub-rank along ``dim``, so the per-axis routing
       decision degenerates to a constant.  We emit
       ``ConstantIndices(value=shard_rank * chunk, length=-1)`` — the
       length placeholder is finalised by ``_build_routing_domain`` to
       the primary axis length, and the planner uses an XLA broadcast
       fast path that never materialises a length-sized constant array.
       This is the Patch L fix for multi-routing-dim zip semantics.
    3. ``src_spec`` partitions ``dim`` with heterogeneous chunk geometry
       (``src.shard_size(dim) != dst.shard_size(dim)``): each rank owns a
       contiguous slice ``[shard_rank * chunk, (shard_rank+1) * chunk)``
       — expressed as ``RangeIndices.contiguous(...)``.  This case is
       only legal as the *primary* routed dim under current
       ``axes=(0,)`` row-binding; multi-routing-dim heterogeneous splits
       must be decomposed at the call-site (``_build_routing_domain``
       enforces zip-length consistency for non-primary axes).
    4. ``src_spec`` uses cyclic routing: each rank owns
       ``{shard_rank + k * sub_size}``, expressed as ``RangeIndices`` with
       ``stride=sub_size``.
    """
    if src_spec is None or dim not in src_spec.partitions:
        return IdentityIndices(length=dst_global_dim)

    if src_spec.is_content_addressable:
        return IdentityIndices(length=dst_global_dim)

    sub_size = src_spec.shard_size(dim)
    if sub_size <= 0:
        return IdentityIndices(length=dst_global_dim)

    shard_rank = src_spec.get_shard_rank(src_local_rank, dim)

    if isinstance(src_spec.strategy, CyclicPartition):
        if dst_global_dim <= shard_rank:
            length = 0
        else:
            length = ((dst_global_dim - 1 - shard_rank) // sub_size) + 1
        return RangeIndices.strided(
            start=shard_rank,
            length=length,
            stride=sub_size,
        )

    if not isinstance(src_spec.strategy, ChunkPartition):
        raise NotImplementedError(
            f"Static source-side index derivation for {type(src_spec.strategy).__name__} "
            "is not implemented. Supply explicit routing_indices for this tensor."
        )

    chunk = max(1, (dst_global_dim + sub_size - 1) // sub_size)
    start = shard_rank * chunk
    length = min(chunk, max(0, dst_global_dim - start))

    # Patch L: detect isomorphic chunk geometry so the auxiliary axis
    # collapses to a constant under zip semantics.  When ``target`` is
    # supplied and ``target.shard_size(dim) == src.shard_size(dim)`` along
    # this dim, every entity on this rank routes to the *same* dst
    # sub-rank; emit ``ConstantIndices`` with the GLOBAL chunk start as
    # the routing key (the strategy maps it to ``start // dst_chunk ==
    # shard_rank``).  ``length=-1`` defers length finalisation to
    # ``_build_routing_domain``.
    if target is not None and dim in target.partitions:
        try:
            dst_sub_size = target.shard_size(dim)
        except Exception:
            dst_sub_size = -1
        if dst_sub_size > 0 and dst_sub_size == sub_size:
            if length == 0:
                return RangeIndices.contiguous(start=start, length=0)
            return ConstantIndices(value=start, length=-1)

    return RangeIndices.contiguous(start=start, length=length)


def _expected_senders_per_dst(
    src_spec: PhysicalLayout | None,
    target: PhysicalLayout,
    dst_global_shape: tuple[int | None, ...],
) -> tuple[int, ...] | None:
    """Compute static N per dst rank, or None when routing is data-driven.

    Algorithm: for every owner src rank, derive its per-dim LOCAL
    ``[start, end)`` range, intersect it with each dst sub-rank's chunk
    boundaries to obtain the set of dst sub-ranks hit on that dim, then
    take the Cartesian product across routed dims to get the full set
    of dst ranks reached. Cost: O(src_world * ∏ sub_size) which is
    bounded by the mesh size and runs only at planning time.

    Returns None when any dst dim uses a data-driven strategy (Hash /
    Indexed) so the caller falls back to always-full SPMD.
    """
    if _is_data_driven_routing(target) or (
        src_spec is not None and src_spec.is_content_addressable
    ):
        return None
    if target is None or target.mesh is None:
        return None

    dst_world = target.mesh.size
    counts = [0] * dst_world

    if src_spec is None or src_spec.mesh is None:
        return None

    src_world = src_spec.mesh.size
    routed_dims = sorted(target.partitions.keys())
    if not routed_dims:
        return None

    mesh_axis_names = list(target.mesh.axis_names)

    for src_rank in range(src_world):
        if not _is_owner_on_replicated_axes(src_spec, src_rank):
            continue

        per_dim_subranks: list[list[int]] = []
        for dim in routed_dims:
            raw_dim = dst_global_shape[dim] if dim < len(dst_global_shape) else 0
            global_dim = raw_dim if raw_dim is not None else 0
            if global_dim <= 0:
                return None
            per_dim_subranks.append(
                _hit_dst_subranks_for_dim(src_spec, target, src_rank, dim, global_dim)
            )
            if not per_dim_subranks[-1]:
                break
        if not per_dim_subranks or any(not ranks for ranks in per_dim_subranks):
            continue

        hit_dsts: set[int] = set()

        def _walk(
            dim_idx: int,
            mesh_coords: list[int],
            *,
            hit_dsts: set[int] = hit_dsts,
            per_dim_subranks: list[list[int]] = per_dim_subranks,
        ) -> None:
            if dim_idx == len(routed_dims):
                dst_rank = 0
                for coord, size in zip(mesh_coords, target.mesh.shape, strict=True):
                    dst_rank = dst_rank * size + coord
                hit_dsts.add(dst_rank)
                return

            dim = routed_dims[dim_idx]
            sub_axes = tuple(
                ax for ax in target.partitions[dim] if ax in target.mesh.axis_names
            )
            sub_sizes = tuple(
                target.mesh.shape[mesh_axis_names.index(ax)] for ax in sub_axes
            )
            for sub_rank in per_dim_subranks[dim_idx]:
                rem = sub_rank
                next_coords = list(mesh_coords)
                for axis_pos, axis_size in enumerate(sub_sizes):
                    tail = math.prod(sub_sizes[axis_pos + 1 :])
                    coord = rem if tail == 1 else rem // tail
                    rem = 0 if tail == 1 else rem % tail
                    axis_name = sub_axes[axis_pos]
                    mesh_pos = mesh_axis_names.index(axis_name)
                    if next_coords[mesh_pos] not in (0, coord):
                        raise ValueError(
                            f"Overlapping target mesh axis {axis_name!r} maps to "
                            f"conflicting coords {next_coords[mesh_pos]} and {coord}."
                        )
                    next_coords[mesh_pos] = coord
                _walk(dim_idx + 1, next_coords)

        _walk(0, [0] * len(mesh_axis_names))
        for dst_rank in hit_dsts:
            if 0 <= dst_rank < dst_world:
                counts[dst_rank] += 1

    return tuple(counts)


def _hit_dst_subranks_for_dim(
    src_spec: PhysicalLayout,
    target: PhysicalLayout,
    src_rank: int,
    dim: int,
    global_dim: int,
) -> list[int]:
    src_strategy = src_spec.strategy
    dst_strategy = target.strategy
    dst_sub_size = target.shard_size(dim)
    if dst_sub_size <= 0:
        raise ValueError(f"target shard size for dim={dim} must be positive.")

    if dim not in src_spec.partitions:
        return list(range(dst_sub_size))

    src_sub_size = src_spec.shard_size(dim)
    src_sub_rank = src_spec.get_shard_rank(src_rank, dim)

    if isinstance(src_strategy, ChunkPartition) and isinstance(
        dst_strategy, ChunkPartition
    ):
        src_chunk = max(1, (global_dim + src_sub_size - 1) // src_sub_size)
        dst_chunk = max(1, (global_dim + dst_sub_size - 1) // dst_sub_size)
        src_start = src_sub_rank * src_chunk
        src_end = min(src_start + src_chunk, global_dim)
        if src_start >= src_end:
            return []
        first = src_start // dst_chunk
        last = (src_end - 1) // dst_chunk
        return list(range(max(0, first), min(dst_sub_size - 1, last) + 1))

    if isinstance(src_strategy, ChunkPartition) and isinstance(
        dst_strategy, CyclicPartition
    ):
        src_chunk = max(1, (global_dim + src_sub_size - 1) // src_sub_size)
        src_start = src_sub_rank * src_chunk
        src_end = min(src_start + src_chunk, global_dim)
        if src_start >= src_end:
            return []
        width = src_end - src_start
        if width >= dst_sub_size:
            return list(range(dst_sub_size))
        return sorted({idx % dst_sub_size for idx in range(src_start, src_end)})

    if isinstance(src_strategy, CyclicPartition) and isinstance(
        dst_strategy, ChunkPartition
    ):
        dst_chunk = max(1, (global_dim + dst_sub_size - 1) // dst_sub_size)
        hits = []
        for dst_sub_rank in range(dst_sub_size):
            dst_start = dst_sub_rank * dst_chunk
            dst_end = min(dst_start + dst_chunk, global_dim)
            if dst_start >= dst_end:
                continue
            first = dst_start + ((src_sub_rank - dst_start) % src_sub_size)
            if first < dst_end:
                hits.append(dst_sub_rank)
        return hits

    if isinstance(src_strategy, CyclicPartition) and isinstance(
        dst_strategy, CyclicPartition
    ):
        gcd = math.gcd(src_sub_size, dst_sub_size)
        return [
            dst_sub_rank
            for dst_sub_rank in range(dst_sub_size)
            if src_sub_rank % gcd == dst_sub_rank % gcd
        ]

    raise NotImplementedError(
        f"Static expected sender count is not implemented for "
        f"{type(src_strategy).__name__} -> {type(dst_strategy).__name__}."
    )


def _materialise_dst_rank(
    target: PhysicalLayout,
    dims: list[int],
    coord: dict[int, int],
    dst_global_shape: tuple[int | None, ...],
) -> int | None:
    """Closed-form dst rank for a single global entity coordinate.

    Retained as a helper for future call sites; ``_expected_senders_per_dst``
    no longer relies on point sampling.
    """
    if target.mesh is None:
        return None
    mesh_axis_names = list(target.mesh.axis_names)
    mesh_coords: list[int] = [0] * len(mesh_axis_names)
    for dim in dims:
        if dim not in target.partitions:
            continue
        raw_dim = dst_global_shape[dim] if dim < len(dst_global_shape) else 0
        global_dim = raw_dim if raw_dim is not None else 0
        sub_size = target.shard_size(dim)
        if sub_size <= 0 or global_dim <= 0:
            return None
        chunk = max(1, global_dim // sub_size)
        sub_rank = min(sub_size - 1, max(0, coord[dim] // chunk))
        sub_sizes = []
        sub_axes = []
        for ax in target.partitions[dim]:
            if ax in mesh_axis_names:
                sub_axes.append(ax)
                sub_sizes.append(target.mesh.shape[mesh_axis_names.index(ax)])
        if not sub_sizes:
            continue
        rem = sub_rank
        sub_coords = [0] * len(sub_sizes)
        for i, s in enumerate(sub_sizes):
            if i < len(sub_sizes) - 1:
                rest = 1
                for s2 in sub_sizes[i + 1 :]:
                    rest *= s2
                sub_coords[i] = rem // rest
                rem = rem % rest
            else:
                sub_coords[i] = rem
        for ax, c in zip(sub_axes, sub_coords, strict=True):
            mesh_coords[mesh_axis_names.index(ax)] = c
    dst_rank = 0
    for c, s in zip(mesh_coords, target.mesh.shape, strict=True):
        dst_rank = dst_rank * s + c
    return dst_rank


def derive_src_side_view(
    src_spec: PhysicalLayout | None,
    src_local_rank: int,
    target: PhysicalLayout,
    dst_global_shape: tuple[int | None, ...],
) -> SrcSideView:
    """Pure derivation of ``SrcSideView`` from layout + rank.

    Combines the F/G/J fixes:
      F: every routed dst dim gets its own LOCAL ``IndicesSpec`` reflecting
         the src rank's actual fragment along that dim.
      G: ``is_sender_active`` is False on src ranks that share data with
         the canonical (coord==0) replica on at least one src-replicated
         mesh axis.
      J: ``expected_senders_per_dst`` is computed when routing is fully
         static (Chunk/Cyclic) and returned as ``None`` otherwise so the
         caller falls back to always-full SPMD.
    """
    if target is None or not target.partitions:
        raise ValueError(
            "derive_src_side_view: target must be a routed PhysicalLayout "
            "(target.partitions must be non-empty); fully replicated dst "
            "should not reach the planner."
        )
    routed_dims = sorted(target.partitions.keys())
    per_dim: list[tuple[int, IndicesSpec]] = []
    for dim in routed_dims:
        raw_dim = dst_global_shape[dim] if dim < len(dst_global_shape) else 0
        global_dim = raw_dim if raw_dim is not None else 0
        per_dim.append(
            (
                dim,
                _src_local_indices_for_routed_dim(
                    src_spec, src_local_rank, dim, global_dim, target=target
                ),
            )
        )
    return SrcSideView(
        per_dim_indices=tuple(per_dim),
        is_sender_active=_is_owner_on_replicated_axes(src_spec, src_local_rank),
        expected_senders_per_dst=_expected_senders_per_dst(
            src_spec, target, dst_global_shape
        ),
    )


class BaseDataDistributor:
    """Base class for data distribution using TransportPlanner.

    Provides core logic for planning and sharding tensors across a distributed
    mesh network, handling both dense and ragged tensor layouts.
    """

    @staticmethod
    def _format_tuuids(tuuid, tuuid_to_name):
        def _fmt(u):
            if tuuid_to_name and u in tuuid_to_name:
                return f"{u} ({tuuid_to_name[u]})"
            return str(u)

        if isinstance(tuuid, list | tuple):
            return "[" + ", ".join(_fmt(u) for u in tuuid) + "]"
        return _fmt(tuuid)

    def __init__(self, local_rank: int):
        self.local_rank = local_rank
        from collections import defaultdict

        # Internal aggregation for "Late Aggregation"
        # {target_rank: [(tensors, counts, global_shapes, has_meta_tensor_state_list)]}
        self._dest_agg: dict[int, list[tuple]] = defaultdict(list)
        self._meta_sort_indices = None
        self._current_source_spec = None
        self._current_target_spec = None
        # Patch K-5: transport owns the src-side derivation, including the
        # static-routing per-dst expected sender count. Upper layers must
        # consult ``expected_n_for_dst`` (or ``last_src_view``) instead of
        # threading their own implicit N to ``ArrivalBarrier``.
        self._current_src_view: SrcSideView | None = None
        self._current_plan_manifest: TransportPlanManifest | None = None

    @property
    def last_src_view(self) -> SrcSideView | None:
        """Return the SrcSideView of the most recent ``plan_and_shard`` call.

        ``None`` until the first plan has been built. After ``distribute``
        clears internal state the view is retained until the next plan
        overwrites it.
        """
        return self._current_src_view

    def expected_n_for_dst(self, dst_rank: int, src_world_size: int) -> int:
        """Number of senders ``dst_rank`` will receive from in the next round.

        Returns the static count derived from
        ``SrcSideView.expected_senders_per_dst`` when routing is fully
        static (Chunk/Cyclic across every routed dim). Falls back to
        ``src_world_size`` (always-full SPMD) when:
          - no plan has been built yet, OR
          - the last plan contains a data-driven dim (Hash/Indexed),
            in which case the count cannot be proven without inspecting
            runtime tensor values.

        Callers should pass the result directly to ``ArrivalBarrier``'s
        ``wait_for_n_slices`` so the receiver does not need to encode
        any private agreement with the distributor about empty packets.
        """
        manifest = self._current_plan_manifest
        if manifest is None:
            return src_world_size
        return manifest.expected_count_for(dst_rank)

    def _plan_mirror_group(
        self,
        receipt,
        payloads,
        has_meta_tensor_state_list,
        compact_desc_info,
        global_shapes_list,
    ):
        """Internal logic for Mirror Backflow (using physical receipts)."""
        backend = get_backend()
        mirror_target_world_size = receipt.world_size
        sender_rank_tags = receipt.sender_rank_tags

        # Scalar fast-path: all data goes to one destination.
        # Handle before entering the JIT-compiled _stitch_with_prime to avoid
        # ConcretizationTypeError (int() on traced JAX values).
        is_scalar = isinstance(sender_rank_tags, int | np.integer) or (
            hasattr(sender_rank_tags, "shape")
            and np.asarray(sender_rank_tags).ndim == 0
        )
        if is_scalar:
            target_rank = int(sender_rank_tags)
            flat_leaves, treedef = backend.tree_flatten(payloads)
            per_tensor_counts = (
                [backend.shape(t)[0] for t in flat_leaves] if flat_leaves else []
            )

            for i in range(mirror_target_world_size):
                if i == target_rank and flat_leaves:
                    rank_flat_tensors = flat_leaves
                    rank_flat_counts = per_tensor_counts
                else:
                    rank_flat_tensors = (
                        [t[:0] for t in flat_leaves] if flat_leaves else []
                    )
                    rank_flat_counts = [0] * len(flat_leaves) if flat_leaves else []

                slice_params = []
                for meta_state, desc_obj in zip(
                    has_meta_tensor_state_list, compact_desc_info, strict=True
                ):
                    if meta_state & 4:
                        offset = 0
                        count = 1 if i == target_rank else 0
                        slice_params.append(
                            desc_obj.slice(offset, offset + count).to_compact_params()
                        )
                    else:
                        slice_params.append(None)

                yield (
                    [i],
                    rank_flat_tensors,
                    rank_flat_counts,
                    None,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    slice_params,
                    [int(EmptyKind.DATA_EMPTY_ALLOWED)]
                    * len(has_meta_tensor_state_list),
                )
            return

        sender_rank_tags = backend.as_tensor(sender_rank_tags)
        stitched_tensors, tensors_size, tensor_offsets = _stitch_with_prime(
            sender_rank_tags, payloads, mirror_target_world_size
        )

        counts_np = backend.to_numpy(tensors_size)
        offsets_np = backend.to_numpy(tensor_offsets)

        for i in range(mirror_target_world_size):
            count = int(counts_np[i])
            rank_flat_tensors = _get_rank_data_from_tree(
                i, stitched_tensors, counts_np, offsets_np
            )

            if count > 0:
                rank_flat_tensors = [t[:count] for t in rank_flat_tensors]
            rank_flat_counts = [count] * len(rank_flat_tensors)

            slice_params = []
            for meta_state, desc_obj in zip(  # type: ignore
                has_meta_tensor_state_list, compact_desc_info, strict=True
            ):
                if meta_state & 4:
                    offset = int(offsets_np[i])
                    slice_params.append(
                        desc_obj.slice(offset, offset + count).to_compact_params()
                    )
                else:
                    slice_params.append(None)

            yield (
                [i],
                rank_flat_tensors,
                rank_flat_counts,
                None,
                global_shapes_list,
                has_meta_tensor_state_list,
                slice_params,
                [
                    int(
                        EmptyKind.DATA_EMPTY_ALLOWED
                        if count > 0
                        else EmptyKind.PROTOCOL_EMPTY
                    )
                ]
                * len(has_meta_tensor_state_list),
            )

    def _plan_single_tensor_group(
        self,
        local_data: TransportTensor | list[TransportTensor],
        source: PhysicalLayout | None,
        target: PhysicalLayout | None,
        capacity: int | list[int] | None = None,
        tuuid: int | list[int] | None = None,
        tuuid_to_name: dict[int, str] | None = None,
    ):
        """Generates plans and padded sequences for a single distribution group.

        A group is defined as one or more tensors that SHARE the same routing indices
        (e.g. follow-sharding or explicitly aligned tensors).
        """
        if tuuid is None:
            raise ValueError(
                "[Fluxway Distributor Error] tuuid is required for TransportFrame planning. "
                "The frame ABI no longer supports implicit tensor identity."
            )

        # Pack all tensors and lengths
        # local_data can be a TransportTensor or List[TransportTensor]
        tensors_to_pack = (
            [local_data] if isinstance(local_data, TransportTensor) else local_data
        )

        # Primary tensor drives routing
        primary = tensors_to_pack[0]

        # Logical shape for routing (respecting Ragged/Padded boundaries)
        local_shape = primary.logical_shape

        # Padding-free global logical shape used by the routing strategy.
        # This is the *single source of truth* for ``ChunkPartition`` /
        # ``HashPartition`` ``dim_size`` arguments and MUST NOT be inflated by
        # any capacity bucketing applied to in-flight tensors below.
        # Mixing the two semantics produces silent routing corruption (e.g.
        # ``[3,4,8,9]`` against ``dim_size=16`` yields ``[1,1,2,3]`` instead
        # of ``[1,2,4,4]``).  When ``primary.global_shape`` is unavailable we
        # fall back to ``local_shape`` -- valid only for scalar / fully
        # replicated / single-rank cases (see Static->Static guard below).
        primary_logical_shape = (
            primary.global_shape if primary.global_shape is not None else local_shape
        )

        # Signal-driven Routing Detection (Referential vs Algebraic)
        is_referential = (source and source.is_referential) or (
            target and target.is_referential
        )
        is_mirror = is_referential
        source_is_content_addressable = bool(source and source.is_content_addressable)
        target_is_content_addressable = bool(target and target.is_content_addressable)
        # Pack Tensors and Metadata (Required for all paths)
        # All payloads/indices entering the eager pipeline below are
        # normalized to the active backend's native domain via
        # ``backend.adopt(x)`` here, so HybridBackend (which dispatches
        # on isinstance(np|cp)) never has to inspect or convert foreign
        # array types again.  This is the *single* chokepoint -- once
        # payloads cross this boundary, every downstream backend.* call
        # operates on a uniform domain.
        backend = get_backend()
        payloads = []
        has_meta_tensor_state_list = []
        nested_length_list: list[tuple | None] = []
        compact_desc_info: list[IndexDescriptor | None] = []

        for tt_idx, tt in enumerate(tensors_to_pack):
            payloads.append(backend.adopt(tt.flat_values))
            meta_tensor_state = 0

            if tt.is_ragged:
                meta_tensor_state |= 1
                meta_tensor_state = encode_ragged_level_count(
                    meta_tensor_state, len(tt.nested_splits)
                )
                level_lengths = []
                for splits in tt.nested_splits:
                    adopted_splits = backend.adopt(splits)
                    level_lengths.append(adopted_splits[1:] - adopted_splits[:-1])
                nested_length_list.append(tuple(level_lengths))
                if is_mirror:
                    for lengths in level_lengths:
                        payloads.append(lengths)
                        nested_length_list.append(None)
            else:
                nested_length_list.append(None)

            if tt.origin_indices is not None:
                if isinstance(tt.origin_indices, ContinuousIndex):
                    meta_tensor_state |= 4
                    compact_desc_info.append(tt.origin_indices)
                else:
                    meta_tensor_state |= 2
                    routing_len = routing_entity_count(tt.payload, tt.layout, backend)
                    idx_len = index_descriptor_length(tt.origin_indices, backend)
                    if idx_len is not None and idx_len != routing_len:
                        raise RuntimeError(
                            "[Fluxway TransportTensor Error] origin_indices "
                            f"length mismatch for tuuid(s)={self._format_tuuids(tuuid, tuuid_to_name)} "
                            f"(tt_idx={tt_idx}): "
                            f"flat_values.shape={tuple(getattr(tt.flat_values, 'shape', ()))}, "
                            f"nested_levels={len(tt.nested_splits)}, "
                            f"routing_axes={tt.routing_axes}, "
                            f"routing_entity_count={routing_len}, "
                            f"origin_indices type={type(tt.origin_indices).__name__}, "
                            f"origin_indices length={idx_len}. "
                            "For ragged tensors origin_indices describe outer routing "
                            "entities, not flat occurrences."
                        )
                    idx = tt.origin_indices.materialize()
                    payloads.append(backend.adopt(idx))
                    nested_length_list.append(None)
                    compact_desc_info.append(None)
            else:
                compact_desc_info.append(None)

            has_meta_tensor_state_list.append(meta_tensor_state)

        # Explicit Mirror Early-Exit
        # Hard interception: when the source/target declares mirror routing
        # (``is_referential``) we MUST have a SymmetryReceipt on the primary
        # tensor. Previously, a missing receipt silently fell through to the
        # algebraic branch below, which then misused ``origin_indices`` as a
        # routing source and surfaced as an opaque ``numpy.take`` IndexError.
        # We now fail fast with an actionable message.
        if is_referential:
            if primary.receipt is None:
                raise RuntimeError(
                    "[Fluxway Mirror Error] target/source declares mirror "
                    f"routing but primary.receipt is None for tuuid(s)="
                    f"{self._format_tuuids(tuuid, tuuid_to_name)}. "
                    "This means the mirror reference upstream did not "
                    "propagate a SymmetryReceipt onto this TransportTensor. "
                    "Common causes: (1) the producing OP rebuilt "
                    "TransportTensor without forwarding the upstream "
                    "receipt; (2) the ``reference=`` path of the mirror "
                    "output_sharding does not exist or is wrong; (3) the "
                    "producing OP used algebraic (non-mirror) routing whose "
                    "output cannot be mirrored. Fix the producer (forward "
                    "the upstream TransportTensor unchanged), correct the "
                    "reference= path, or change the target sharding to "
                    "dynamic/static."
                )
            global_shapes_list = [tt.global_shape for tt in tensors_to_pack]
            yield from self._plan_mirror_group(
                primary.receipt,
                payloads=payloads,
                has_meta_tensor_state_list=has_meta_tensor_state_list,
                compact_desc_info=compact_desc_info,
                global_shapes_list=global_shapes_list,
            )
            return

        if not source and primary.routing_indices is None:
            raise ValueError(
                f"[Fluxway Distributor Error] No sharding source available for TUUID(s): {self._format_tuuids(tuuid, tuuid_to_name)}.\n"
                f"  Problem: The algebraic path requires either a valid 'source' layout or explicit 'routing_indices'.\n"
                f"  Fix: Ensure the TransportTensor is initialized with a source PhysicalLayout."
            )

        src_geometry_indices, src_geometry_name = select_source_geometry_index_source(
            routing_indices=primary.routing_indices,
            origin_indices=primary.origin_indices,
            source_is_content_addressable=source_is_content_addressable,
            target_is_content_addressable=target_is_content_addressable,
        )
        if target_is_content_addressable and not is_referential:
            dst_content_routing_values = select_destination_content_index_source(
                routing_indices=primary.routing_indices,
                flat_values=primary.flat_values,
            )
        else:
            dst_content_routing_values = None

        # For dst layout, prefer explicit global_shape, then tensor's global_shape, then primary_logical_shape.
        dst_shape = (
            primary.global_shape
            if primary.global_shape is not None
            else primary_logical_shape
        )

        if (
            not is_referential
            and target
            and source
            and not target.is_content_addressable
            and not source.is_content_addressable
            and primary.shape != ()
        ):
            is_fallback_shape = (dst_shape == local_shape) or (
                primary.global_shape is None
            )
            src_mesh_size = source.mesh.size if source.mesh is not None else 1
            # axis=[] → source.partitions == {} → fully replicated; local_shape == global_shape is correct
            is_src_replicated = not source.partitions
            if dst_shape is None or (
                is_fallback_shape and src_mesh_size > 1 and not is_src_replicated
            ):
                raise ValueError(
                    f"[Distributor] Static-to-Static routing requires a valid 'global_shape' to determine shard boundaries. "
                    f"TUUID: {self._format_tuuids(tuuid, tuuid_to_name)}. "
                    f"Please ensure the source TransportTensor has 'global_shape' set explicitly."
                )

        if (
            not is_referential
            and target
            and source
            and not target.is_content_addressable
            and not source.is_content_addressable
            and dst_shape is not None
            and target.partitions
        ):
            for d, axes in target.partitions.items():
                if d >= len(dst_shape):
                    continue
                global_dim = dst_shape[d]
                sub_mesh_size = target.shard_size(d)
                if (
                    global_dim is not None
                    and global_dim > 0
                    and global_dim < sub_mesh_size
                ):
                    axis_names = list(axes)
                    raise ValueError(
                        f"\n[Fluxway Static Sharding Error] Data dimension {d} has size {global_dim}, "
                        f"but the target mesh needs {sub_mesh_size} elements (axes={axis_names}) "
                        f"to give every worker at least one element.\n"
                        f"  TUUID: {self._format_tuuids(tuuid, tuuid_to_name)}\n"
                        f"  global_shape={dst_shape}, target mesh axes={axis_names}, "
                        f"sub_mesh_size={sub_mesh_size}\n"
                        f"  Fix: Increase the data size along dimension {d} to >= {sub_mesh_size}, "
                        f"or reduce the target mesh size."
                    )

        if source is not None and target is not None:
            logger.debug(
                f"[Distributor] Planning: src_mesh={source.mesh.size if source.mesh is not None else 1}, "
                f"dst_mesh={target.mesh.size if target.mesh is not None else 1}, "
                f"local_rank={self.local_rank}, src_index_source={src_geometry_name}"
            )

        # Set capacity bucket. Bucketing is *only* needed when the active
        # backend requires static shapes (jax.jit / torch.compile). Pure
        # eager paths (HybridBackend default, TorchBackend eager) keep
        # capacity == actual_req so all padding branches below are no-ops.
        actual_req = local_shape[0] if local_shape else 1
        assert actual_req is not None, (
            "actual_req (primary_local_shape[0]) must be statically known"
        )
        bk_static = backend.requires_static_shapes()
        if isinstance(capacity, list | tuple):
            selected_capacity = None
            for cap in sorted(capacity):
                if cap >= actual_req:
                    selected_capacity = cap
                    break
            if selected_capacity is None:
                selected_capacity = actual_req
            capacity = selected_capacity
        elif capacity is None:
            if bk_static:
                if actual_req is None:
                    capacity = None
                elif actual_req == 0:
                    capacity = 0
                elif actual_req <= 1024:
                    capacity = 1 << (actual_req - 1).bit_length()
                else:
                    capacity = ((actual_req + 1023) // 1024) * 1024
            else:
                capacity = actual_req

        global_shapes_list = [tt.global_shape for tt in tensors_to_pack]

        is_src_replicated = source is not None and not source.partitions
        if primary.shape == () or is_src_replicated:
            target_world_size = target.mesh.size if target and target.mesh else 1
            src_world_size = source.mesh.size if source and source.mesh else 1

            # Replicated dispatch: responsible_src = floor(j * N / M)
            # Each target j receives exactly one copy from a designated source.
            # N < M → broadcast (contiguous block per source)
            # N > M → selection (only sources 0..M-1 send, others idle)
            # N == M → 1:1 bijection
            targets_to_send = []
            for j in range(target_world_size):
                responsible_src = (j * src_world_size) // target_world_size
                if responsible_src == self.local_rank:
                    targets_to_send.append(j)

            if not targets_to_send:
                return

            responsible = tuple(
                (j * src_world_size) // target_world_size == self.local_rank
                for j in range(target_world_size)
            )
            self._current_plan_manifest = TransportPlanManifest(
                expected_senders_per_dst=(1,) * target_world_size,
                sender_has_data_per_dst=responsible,
                emission_mode=SenderEmissionMode.STATIC_PRODUCERS_ONLY,
            )

            # Scalar payload has exactly 1 element; non-scalar carries actual length.
            counts = (
                [1] * len(payloads)
                if primary.shape == ()
                else [len(p) if hasattr(p, "__len__") else 1 for p in payloads]
            )
            yield (
                targets_to_send,
                payloads,
                counts,
                None,
                global_shapes_list,
                has_meta_tensor_state_list,
                [None] * len(has_meta_tensor_state_list),
                [int(EmptyKind.DATA_EMPTY_ALLOWED)] * len(has_meta_tensor_state_list),
            )
            return

        if is_mirror and primary.receipt is not None:
            yield from self._plan_mirror_group(
                primary.receipt,
                payloads,
                has_meta_tensor_state_list,
                compact_desc_info,
                global_shapes_list,
            )
            return

        # Pad payloads and lengths to capacity bucket shape to avoid JAX JIT recompilation
        # (only when the active backend requires static shapes; for pure
        # eager backends the capacity equals actual_req and all branches
        # below are no-ops).
        padded_payloads = []
        padded_nested_length_list: list[tuple[Any, ...] | None] = []
        assert capacity is not None, "capacity must be resolved to an int"
        static_shapes = [capacity]

        for p, p_len in zip(payloads, nested_length_list, strict=True):
            if p_len is not None:
                # Ragged payload
                padded_levels = []
                for level, lengths in enumerate(p_len):
                    if (
                        bk_static
                        and capacity is not None
                        and actual_req is not None
                        and capacity != actual_req
                        and level == 0
                    ):
                        pad_len_size = capacity - actual_req
                    else:
                        pad_len_size = 0

                    if pad_len_size > 0:
                        len_padding = backend.zeros(
                            (pad_len_size,), dtype=backend.dtype(lengths)
                        )
                        padded_levels.append(backend.cat([lengths, len_padding], dim=0))
                    else:
                        padded_levels.append(lengths)
                padded_nested_length_list.append(tuple(padded_levels))

                flat_req = p.shape[0]
                if bk_static and capacity != actual_req:
                    # Pad flat ragged values using same bucketing
                    if flat_req == 0:
                        flat_capacity = 0
                    elif flat_req <= 1024:
                        flat_capacity = 1 << (flat_req - 1).bit_length()
                    else:
                        flat_capacity = ((flat_req + 1023) // 1024) * 1024
                else:
                    flat_capacity = flat_req

                static_shapes.append(flat_capacity)
                pad_flat_size = flat_capacity - flat_req
                if pad_flat_size > 0:
                    flat_pad_shape = (pad_flat_size,) + backend.shape(p)[1:]
                    flat_padding = backend.zeros(flat_pad_shape, dtype=backend.dtype(p))
                    padded_p = backend.cat([p, flat_padding], dim=0)
                else:
                    padded_p = p
                padded_payloads.append(padded_p)
            else:
                # Dense payload
                padded_nested_length_list.append(None)

                payload_req = backend.shape(p)[0]
                if bk_static and capacity != actual_req:
                    if payload_req == 0:
                        payload_capacity = 0
                    elif payload_req <= 1024:
                        payload_capacity = 1 << (payload_req - 1).bit_length()
                    else:
                        payload_capacity = ((payload_req + 1023) // 1024) * 1024
                else:
                    payload_capacity = payload_req

                static_shapes.append(payload_capacity)

                pad_size = payload_capacity - payload_req
                if pad_size > 0:
                    pad_shape = (pad_size,) + backend.shape(p)[1:]
                    padding = backend.zeros(pad_shape, dtype=backend.dtype(p))
                    padded_p = backend.cat([p, padding], dim=0)
                else:
                    padded_p = p
                padded_payloads.append(padded_p)

        # NOTE: ``capacity`` is a per-rank *padding bucket* applied to in-flight
        # tensors so XLA can trace stable shapes; it is intentionally decoupled
        # from the routing domain (which drives the routing math).  Do *not*
        # derive a ``global_capacity = capacity * mesh_size`` here -- such an
        # inference is wrong whenever per-rank shard sizes are non-uniform
        # (e.g. a sender that holds 4 items in a 12/4 ChunkPartition) and would
        # corrupt the routing dim_size on the planner side.

        payloads_tuple = tuple(padded_payloads)
        # Each payload's binding axes (which axes of flat_values are bound to
        # the routing domain). Default (0,) — used for nearly all
        # DP/MP/EP/SP/CP/TP layouts. The fast-path `take(axis=0)` is taken
        # whenever the binding is exactly (0,).
        primary_routing_axes = primary.routing_axes
        payload_routing_axes = tuple(primary_routing_axes for _ in padded_payloads)
        static_shapes_tuple = tuple(static_shapes)

        # ---------- build RoutingDomain (the only thing the planner sees) ----------
        # The domain is built from the *target* layout. Per-axis sizes are the
        # routing-domain extents (== padding-free shape under eager backends;
        # capacity-padded extents under static-shape backends).
        #
        # Patch K: transport now owns full src-side reasoning via SrcSideView.
        # ``derive_src_side_view`` is the *only* legitimate constructor; it
        # encapsulates per-dim LOCAL IndicesSpec derivation (F), owner gating
        # on src-replicated mesh axes (G), and static expected_senders_per_dst
        # (J). Distribution layer consumes ``src_view.is_sender_active`` and
        # ``src_view.expected_senders_per_dst`` in K-4/K-5.
        if target is None:
            raise ValueError(
                f"[Fluxway Distributor Error] Cannot build routing domain for TUUID(s): "
                f"{self._format_tuuids(tuuid, tuuid_to_name)}. Target must be a valid PhysicalLayout."
            )

        src_view = derive_src_side_view(
            src_spec=source,
            src_local_rank=self.local_rank,
            target=target,
            dst_global_shape=tuple(s for s in dst_shape if s is not None)
            if dst_shape is not None
            else tuple(s for s in local_shape if s is not None),
        )
        # K-5: persist for upper layers (e.g. ArrivalBarrier N derivation).
        self._current_src_view = src_view

        assert capacity is not None, "capacity must be resolved to an integer"

        domain = _build_routing_domain(
            target=target,
            src_view=src_view,
            primary_local_shape=local_shape,
            src_geometry_indices=src_geometry_indices,
            dst_content_routing_values=dst_content_routing_values,
            bk_static=bk_static,
            capacity=capacity,
            actual_req=actual_req,
            dst_global_shape=tuple(s for s in dst_shape if s is not None)
            if dst_shape is not None
            else None,
        )

        # Split into (static, dynamic) legs so that under jax.jit the static
        # part is hashable and feeds static_argnums while the LiteralIndices
        # values flow as the dynamic leg.  RangeIndices / IdentityIndices
        # are fully hashable and remain inside ``domain_static`` untouched.
        def _split_axis(ax: RoutingAxis) -> tuple[RoutingAxis, Any]:
            spec = ax.indices
            if isinstance(spec, LiteralIndices):
                # Drop the runtime tensor from the static spec; tensor
                # flows via indices_dyn[i].
                static_spec = dataclasses.replace(spec, values=None)
                return dataclasses.replace(ax, indices=static_spec), spec.values
            return ax, None

        split_pairs = [_split_axis(ax) for ax in domain.axes]
        domain_static = dataclasses.replace(
            domain,
            axes=tuple(p[0] for p in split_pairs),
        )
        indices_dyn = tuple(p[1] for p in split_pairs)

        (
            flat_buffers_lists,
            counts_per_payload,
            offsets_per_payload,
            send_counts,
            send_offsets,
            dest_devices,
            replication_factor,
            sort_indices,
        ) = _jit_plan_and_pack(
            payloads_tuple,
            tuple(padded_nested_length_list),
            self.local_rank,
            domain_static,
            indices_dyn,
            payload_routing_axes,
            static_shapes_tuple,
        )

        if (
            sort_indices is not None
            and capacity is not None
            and actual_req is not None
            and capacity > actual_req
        ):
            # Static-shape path only: trim sentinel rows that were padded into
            # the routing domain. Eager path has capacity == actual_req so
            # this branch is skipped.
            mask = sort_indices < actual_req
            sort_indices = sort_indices[mask]

        # Materialise per-device send counts/offsets on the host once.  These are the
        # minimal scalars needed to slice the flat buffers; the heavy tensor data stays
        # on-device until the Python slice below.
        _backend = get_backend()
        counts_host = _backend.to_numpy(send_counts).flatten()
        offsets_host = _backend.to_numpy(send_offsets).flatten()
        counts_per_payload_host = [
            _backend.to_numpy(c).flatten() for c in counts_per_payload
        ]
        offsets_per_payload_host = [
            _backend.to_numpy(o).flatten() for o in offsets_per_payload
        ]

        def _get_slice_compact_desc_params(device_id):
            slice_compact_desc_params: list[list[int] | None] = []
            for meta_tensor_state, desc_obj in zip(  # type: ignore
                has_meta_tensor_state_list,
                compact_desc_info,
                strict=True,  # type: ignore
            ):  # type: ignore
                if meta_tensor_state & 4:
                    offset = int(offsets_host[device_id])
                    count = int(counts_host[device_id])
                    assert isinstance(desc_obj, ContinuousIndex)
                    slice_compact_desc_params.append(
                        desc_obj.slice(offset, offset + count).to_compact_params()
                    )
                else:
                    slice_compact_desc_params.append(None)
            return slice_compact_desc_params

        backend = get_backend()

        def _get_flat_payloads_for_rank(device_id):
            """Slice this device's segment out of each O(N) flat buffer.

            flat_buffers_lists holds globally-reordered 1-D arrays.  Each device's
            data occupies a contiguous window [offset, offset+count), so a single
            backend.get_contiguous_view call extracts the view.

            On backends with DMA support (NCCL/UCX), the returned view exposes
            __cuda_array_interface__ without any data copy; on Ray the default
            implementation falls back to a plain Python slice.
            """
            p_flat = []
            c_flat = []
            p_idx = 0
            c_idx = 0
            for meta_state in has_meta_tensor_state_list:
                p_data, p_len = flat_buffers_lists[p_idx]
                offset = int(offsets_per_payload_host[c_idx][device_id])
                count = int(counts_per_payload_host[c_idx][device_id])
                p_flat.append(backend.get_contiguous_view(p_data, offset, count))
                c_flat.append(count)

                for level_lengths in p_len or ():
                    len_offset = int(offsets_per_payload_host[c_idx + 1][device_id])
                    count_len = int(counts_per_payload_host[c_idx + 1][device_id])
                    p_flat.append(
                        backend.get_contiguous_view(
                            level_lengths, len_offset, count_len
                        )
                    )
                    c_flat.append(count_len)
                    c_idx += 1

                p_idx += 1
                c_idx += 1

                if meta_state & 2:  # materialized origin indices
                    p_data, _ = flat_buffers_lists[p_idx]
                    offset = int(offsets_per_payload_host[c_idx][device_id])
                    count = int(counts_per_payload_host[c_idx][device_id])
                    p_flat.append(backend.get_contiguous_view(p_data, offset, count))
                    c_flat.append(count)
                    p_idx += 1
                    c_idx += 1
            return p_flat, c_flat

        empty_flat_payloads_cache = None

        def _get_empty_flat_payloads():
            nonlocal empty_flat_payloads_cache
            if empty_flat_payloads_cache is not None:
                return empty_flat_payloads_cache
            p_flat = []
            p_idx = 0
            for meta_state in has_meta_tensor_state_list:
                p_data, p_len = flat_buffers_lists[p_idx]
                p_flat.append(backend.get_contiguous_view(p_data, 0, 0))
                for level_lengths in p_len or ():
                    p_flat.append(backend.get_contiguous_view(level_lengths, 0, 0))
                p_idx += 1
                if meta_state & 2:
                    p_data, _ = flat_buffers_lists[p_idx]
                    p_flat.append(backend.get_contiguous_view(p_data, 0, 0))
                    p_idx += 1
            empty_flat_payloads_cache = (p_flat, [0] * len(p_flat))
            return empty_flat_payloads_cache

        is_replicated = replication_factor > 1

        # Patch K-4: src_view drives SPMD-safe empty-packet emission.
        # ``expected_senders_per_dst`` enables the static optimization of
        # skipping a dst whose ``expected[d] == 0`` entirely (no sender
        # contributes — safe to short-circuit). When ``None`` the routing
        # graph contains a data-driven dim (Hash/Indexed) and we *cannot*
        # prove any dst is unreached, so we fall back to always-full SPMD
        # and emit empty packets to keep alltoall_v symmetric across
        # senders.
        # ``is_sender_active`` is False on src ranks that share data with
        # the canonical (coord==0) replica on at least one src-replicated
        # mesh axis; they MUST still participate in the SPMD round but
        # transmit zero-count payloads to avoid double-counting.
        expected_per_dst = src_view.expected_senders_per_dst
        is_sender_active = src_view.is_sender_active
        if expected_per_dst is None:
            if source is None or source.mesh is None:
                raise ValueError(
                    "[Fluxway Distributor Error] Dynamic transport manifest requires "
                    "a source PhysicalLayout with mesh to determine sender world size."
                )
            source_world_size = source.mesh.size
            manifest = TransportPlanManifest(
                expected_senders_per_dst=(source_world_size,) * len(counts_host),
                sender_has_data_per_dst=tuple(bool(count) for count in counts_host),
                emission_mode=SenderEmissionMode.ALL_SENDERS_WITH_PROTOCOL_EMPTY,
            )
        else:
            manifest = TransportPlanManifest(
                expected_senders_per_dst=expected_per_dst,
                sender_has_data_per_dst=tuple(
                    (is_sender_active and int(count) > 0) for count in counts_host
                ),
                emission_mode=SenderEmissionMode.STATIC_PRODUCERS_ONLY,
            )
        self._current_plan_manifest = manifest
        expected_counts = manifest.expected_senders_per_dst
        sender_has_data = manifest.sender_has_data_per_dst
        emit_all_senders = (
            manifest.emission_mode is SenderEmissionMode.ALL_SENDERS_WITH_PROTOCOL_EMPTY
        )

        if is_replicated:
            # Fan-out
            sorted_dest_host = _backend.to_numpy(dest_devices)

            for device_id, count in enumerate(counts_host):
                if expected_counts[device_id] <= 0 or (
                    not emit_all_senders and not sender_has_data[device_id]
                ):
                    continue

                targets = sorted_dest_host[device_id].flatten().tolist()
                if sender_has_data[device_id]:
                    p_flat, c_flat = _get_flat_payloads_for_rank(device_id)
                    empty_kind_list = [int(EmptyKind.DATA_EMPTY_ALLOWED)] * len(
                        has_meta_tensor_state_list
                    )
                else:
                    p_flat, c_flat = _get_empty_flat_payloads()
                    empty_kind_list = [int(EmptyKind.PROTOCOL_EMPTY)] * len(
                        has_meta_tensor_state_list
                    )

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    None,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
                    empty_kind_list,
                )
        else:
            # Scatter-send
            for device_id, count in enumerate(counts_host):
                if expected_counts[device_id] <= 0 or (
                    not emit_all_senders and not sender_has_data[device_id]
                ):
                    continue

                targets = [device_id]
                if sender_has_data[device_id]:
                    p_flat, c_flat = _get_flat_payloads_for_rank(device_id)
                    empty_kind_list = [int(EmptyKind.DATA_EMPTY_ALLOWED)] * len(
                        has_meta_tensor_state_list
                    )
                else:
                    p_flat, c_flat = _get_empty_flat_payloads()
                    empty_kind_list = [int(EmptyKind.PROTOCOL_EMPTY)] * len(
                        has_meta_tensor_state_list
                    )

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    sort_indices,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
                    empty_kind_list,
                )

    def plan_and_shard(
        self,
        local_data: TransportTensor | list[TransportTensor],
        source: PhysicalLayout | None = None,
        target: PhysicalLayout | None = None,
        capacity: int | list[int] | None = None,
        tuuid: int | list[int] | None = None,
        tuuid_to_name: dict[int, str] | None = None,
    ) -> None:
        """Computes sharding plan and buffers data for all destinations."""
        if tuuid is None:
            raise ValueError(
                "[Fluxway Distributor Error] tuuid is required for TransportFrame planning. "
                "The frame ABI no longer supports implicit tensor identity."
            )

        # PLAN ISOLATION:
        # We MUST NOT reuse plans across separate calls to plan_and_shard.
        # Each call represents a distinct group of tensors that might have
        # different routing requirements (e.g., different values/indices).
        self._meta_sort_indices = None
        self._current_source_spec = source
        self._current_target_spec = target
        self._current_plan_manifest = None

        for (
            targets,
            tensors,
            counts,
            sort_idx,
            global_shapes_list,
            has_meta_tensor_state_list,
            slice_compact_desc_params,
            empty_kind_list,
        ) in self._plan_single_tensor_group(
            local_data, source, target, capacity, tuuid, tuuid_to_name
        ):
            logger.debug(
                f"[Distributor Plan] [Rank {self.local_rank}] targets={targets} tensors={tensors} counts={counts} sort_idx={sort_idx} global_shapes_list={global_shapes_list} has_meta_tensor_state_list={has_meta_tensor_state_list} slice_compact_desc_params={slice_compact_desc_params}"
            )
            if self._meta_sort_indices is None:
                self._meta_sort_indices = sort_idx
            for target_id in targets:
                # Store tuuid along with data for correct reconstruction in send_fn
                self._dest_agg[target_id].append(
                    (
                        tuuid,
                        tensors,
                        counts,
                        global_shapes_list,
                        has_meta_tensor_state_list,
                        slice_compact_desc_params,
                        empty_kind_list,
                    )
                )


class AsyncDataDistributor(BaseDataDistributor):
    """Frame-only asynchronous data distributor."""

    def build_transport_frame_args(
        self,
        *,
        step: int,
        sender_spmd_code: int,
        sender_world_size: int,
    ) -> list[tuple[int, Any, Any, Any, Any, Any, Any, tuple[Any, ...]]]:
        frame_args = []
        for target_id in sorted(self._dest_agg.keys()):
            frame = self._build_transport_frame_for_target(
                target_id=target_id,
                groups=self._dest_agg[target_id],
                step=step,
                sender_spmd_code=sender_spmd_code,
                sender_world_size=sender_world_size,
            )
            if logger.isEnabledFor(logging.DEBUG):
                edge_summary = []
                for row in frame.edge_i32:
                    payload_begin = int(row[3])
                    payload_count = int(row[4])
                    valid_count = 0
                    if payload_count > 0:
                        valid_count = int(
                            frame.payload_i32[payload_begin][PAYLOAD_VALID_COUNT]
                        )
                    edge_summary.append(
                        (
                            int(row[EDGE_TUUID]),
                            valid_count,
                            EmptyKind(int(row[EDGE_EMPTY_KIND])).name,
                            int(row[EDGE_EXPECTED_COUNT]),
                        )
                    )
                logger.debug(
                    "[TransportFrame] sender_rank=%d target_rank=%d step=%d edges=%s",
                    self.local_rank,
                    target_id,
                    step,
                    edge_summary,
                )
            frame_args.append(
                (
                    target_id,
                    frame.frame_i32,
                    frame.edge_i32,
                    frame.payload_i32,
                    frame.shape_i32,
                    frame.global_shape_i64,
                    frame.compact_i64,
                    frame.payload_segments,
                )
            )
        return frame_args

    async def distribute(
        self,
        local_data: TransportTensor | list[TransportTensor] | None = None,
        source: PhysicalLayout | None = None,
        target: PhysicalLayout | None = None,
        send_fn: Callable | None = None,
        capacity: int | list[int] | None = None,
        tuuid: int | list[int] | None = None,
        tuuid_to_name: dict[int, str] | None = None,
        step: int = 0,
        sender_spmd_code: int = 0,
        sender_world_size: int = 1,
    ) -> tuple[list[Any], Any]:
        if local_data is not None:
            self.plan_and_shard(
                local_data, source, target, capacity, tuuid, tuuid_to_name
            )

        if send_fn is None:
            return [], self._meta_sort_indices

        frame_args = self.build_transport_frame_args(
            step=step,
            sender_spmd_code=sender_spmd_code,
            sender_world_size=sender_world_size,
        )
        tasks = []
        for (
            target_id,
            frame_i32,
            edge_i32,
            payload_i32,
            shape_i32,
            global_shape_i64,
            compact_i64,
            payload_segments,
        ) in frame_args:
            tasks.append(
                send_fn(
                    target_id,
                    frame_i32,
                    edge_i32,
                    payload_i32,
                    shape_i32,
                    global_shape_i64,
                    compact_i64,
                    *payload_segments,
                )
            )

        res = await asyncio.gather(*tasks) if tasks else []
        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return res, sort_indices

    def _build_transport_frame_for_target(
        self,
        *,
        target_id: int,
        groups: list[tuple],
        step: int,
        sender_spmd_code: int,
        sender_world_size: int,
    ):
        builder = TransportFrameBuilder(
            step=step,
            sender_spmd_code=sender_spmd_code,
            sender_rank=self.local_rank,
            sender_world_size=sender_world_size,
        )
        expected_count = self.expected_n_for_dst(target_id, sender_world_size)
        for (
            tuuids,
            tensors,
            counts,
            g_shapes,
            h_states,
            c_params,
            empty_kinds,
        ) in groups:
            if isinstance(tuuids, list | tuple):
                tuuid_seq = list(tuuids)
            else:
                tuuid_seq = [tuuids]
            if len(empty_kinds) != len(tuuid_seq):
                raise ValueError(
                    f"empty_kinds length {len(empty_kinds)} does not match "
                    f"tuuid count {len(tuuid_seq)} for target_id={target_id}."
                )
            cursor = 0
            for edge_idx, u in enumerate(tuuid_seq):
                meta_state = h_states[edge_idx]
                component_count = 1 + ragged_level_count(meta_state)
                if meta_state & 2:
                    component_count += 1
                edge_counts = counts[cursor : cursor + component_count]
                compact_params = (
                    c_params[edge_idx]
                    if c_params is not None and edge_idx < len(c_params)
                    else None
                )
                edge = builder.add_edge(
                    tuuid=int(u),
                    meta_state=int(meta_state),
                    empty_kind=EmptyKind(int(empty_kinds[edge_idx])),
                    min_rows=0,
                    expected_count=expected_count,
                    compact_params=compact_params,
                    global_shape=g_shapes[edge_idx],
                )
                builder.add_payload(
                    edge,
                    PayloadRole.VALUES,
                    tensors[cursor],
                    int(edge_counts[0]),
                )
                cursor += 1
                for _ in range(ragged_level_count(meta_state)):
                    builder.add_payload(
                        edge,
                        PayloadRole.RAGGED_LENGTHS,
                        tensors[cursor],
                        int(counts[cursor]),
                    )
                    cursor += 1
                if meta_state & 2:
                    builder.add_payload(
                        edge,
                        PayloadRole.ORIGIN,
                        tensors[cursor],
                        int(counts[cursor]),
                    )
                    cursor += 1
        return builder.build()


# ---------------------------------------------------------------------------
# Module-level JIT-compiled zero-copy write kernels.
#
# These are defined at module level so the @compile decorator traces them
# exactly once per unique (count, ndim) combination across all receivers,
# eliminating JAX eager-mode overhead on every receive_slices() call.
#
# JAX constraint: dynamic_slice requires slice_sizes to be statically known
# at trace time, so `count` is a static argnum.  The destination offset is
# passed as a runtime (jnp.int32) so it can vary per sender without
# triggering re-compilation.
# ---------------------------------------------------------------------------


@compile(static_argnums=(2,))
def _jit_write_at_offset(src, dst, count, offset):
    """Write `count` rows of `src` (which may be capacity-padded) into `dst`
    starting at row `offset`.  JIT-compiled; `count` is static."""
    backend = get_backend()
    ndim = len(backend.shape(src))
    start = (0,) * ndim
    sizes = (count,) + backend.shape(src)[1:]
    trimmed = backend.dynamic_slice(src, start, sizes)
    dst_start = (offset,) + (0,) * (ndim - 1)
    return backend.dynamic_slice_update(dst, trimmed, dst_start)


def _get_rank_data_from_tree(rank_id, stitched_tensors_tree, counts, offsets):
    """
    rank_id: target rank id
    stitched_tensors_tree: Pytree sorted by rank
    counts: counts of each rank
    offsets: offsets of each rank
    """
    backend = get_backend()

    start = offsets[rank_id]
    size = counts[rank_id]

    def slice_leaf(array):
        # assume all slaves are in the 0-th dimension (Batch dimension)
        return backend.dynamic_slice_in_dim(array, start=start, size=size, axis=0)

    # 3. 使用 tree_map 一次性处理 List 中的所有 Tensor
    return backend.tree_map(slice_leaf, stitched_tensors_tree)


@compile(static_argnums=(2,))
def _jit_write_at_zero(src, dst, count):
    """Write `count` rows of `src` (which may be capacity-padded) into `dst`
    starting at row 0.  JIT-compiled; `count` is static."""
    backend = get_backend()
    ndim = len(backend.shape(src))
    start = (0,) * ndim
    sizes = (count,) + backend.shape(src)[1:]
    trimmed = backend.dynamic_slice(src, start, sizes)
    return backend.dynamic_slice_update(dst, trimmed, (0,) * ndim)


class TransportReceiver:
    """Configurable receiver for zero-copy or buffered transport scenarios.

    **Modes**:

    Buffered (default):

        Example::
            receiver.register_per_sender_buffers({
                0: [np.zeros((256, 64))],
                1: [np.zeros((128, 64))],
            })

    **Actual written lengths**:
        After receive_slices(), call get_written_counts() to retrieve
        {source_rank: [count_payload_0, count_payload_1, ...]}.
        This is the number of valid rows written (the tensor may arrive
        capacity-padded, so the buffer holds more rows than were written).

    **Completion**:

    One-shot (default):  await receiver.wait()
    Periodic:            async for count in receiver: ...  +  receiver.close()
    """

    def __init__(self, wait_for_n_slices: int = 1, periodic: bool = False):
        from fluxway.common.sync import ArrivalBarrier

        self._barrier = ArrivalBarrier(wait_for_n_slices, periodic=periodic)
        self._periodic = periodic

        # Shared-buffer zero-copy (offset mode)
        self.target_buffers: list[Any] | None = None
        self.expect_offsets: dict = {}

        # Per-sender zero-copy (no-offset mode)
        self._per_sender_buffers: dict | None = None  # {rank: [buf, ...]}

        # Buffered mode
        self._rx_buffers: list = []

        # Actual written row counts: {source_rank: [count_per_payload_index]}
        self._written_counts: dict = {}
        _backend = get_backend()
        self._is_torch = _backend.is_torch
        self._is_jax = _backend.is_jax
        # Whether to route writes through the @compile JIT kernels
        # (_jit_write_at_offset / _jit_write_at_zero).  Pure eager
        # backends (numpy / cupy / torch eager) skip the JIT path and
        # use plain slice-assignment, which avoids capacity-padded
        # buffers entirely.
        self._requires_static_shapes = _backend.requires_static_shapes()

    # ------------------------------------------------------------------
    # Zero-copy registration
    # ------------------------------------------------------------------

    def register_zero_copy_buffers(
        self, target_buffers: list[Any], expect_offsets: dict
    ) -> None:
        """
        Offset mode: all senders write into a single shared buffer set.

        Args:
            target_buffers: List of destination tensors (one per payload index).
            expect_offsets: {source_rank: row_offset_or_list_of_offsets}
        """
        self.target_buffers = target_buffers
        self.expect_offsets = expect_offsets
        self._per_sender_buffers = None
        self._written_counts = {}
        self._barrier.reset()

    def register_per_sender_buffers(self, per_sender_buffers: dict) -> None:
        """
        No-offset mode: each sender has its own pre-allocated buffer list.

        Data is written at row 0; actual written rows are tracked in
        get_written_counts().

        Args:
            per_sender_buffers: {source_rank: [buf0, buf1, ...]}
        """
        self._per_sender_buffers = per_sender_buffers
        self.target_buffers = None
        self.expect_offsets = {}
        self._written_counts = {}

    # ------------------------------------------------------------------
    # Slice reception
    # ------------------------------------------------------------------

    def receive_slices(
        self, source_rank: int, tensors: list[Any], counts: list[int]
    ) -> bool:
        """Called by the network layer when a payload arrives.

        All JAX write operations go through the module-level @compile kernels
        (_jit_write_at_offset / _jit_write_at_zero) to avoid eager execution.
        """
        written = []

        if self._per_sender_buffers is not None:
            # No-offset zero-copy
            sender_bufs = self._per_sender_buffers.get(source_rank)
            if sender_bufs is not None:
                for i, tensor in enumerate(tensors):
                    if tensor is None or i >= len(sender_bufs):
                        written.append(0)
                        continue
                    if i < len(counts):
                        count = counts[i]
                    else:
                        raise RuntimeError(
                            f"Missing shard count for bundle element {i} from rank {source_rank}"
                        )
                    if self._requires_static_shapes and get_backend().is_tensor(tensor):
                        sender_bufs[i] = _jit_write_at_zero(
                            tensor, sender_bufs[i], count
                        )
                    else:
                        sender_bufs[i][:count] = tensor[:count]
                    written.append(count)

        elif self.target_buffers is not None:
            # Offset zero-copy
            if source_rank in self.expect_offsets:
                offsets = self.expect_offsets[source_rank]
                if not isinstance(offsets, list | tuple):
                    offsets = [offsets]

                for i, tensor in enumerate(tensors):
                    if tensor is None or i >= len(self.target_buffers):
                        written.append(0)
                        continue
                    count = counts[i] if i < len(counts) else counts[0]
                    offset = int(offsets[i] if i < len(offsets) else offsets[0])

                    # Bounds check (host-side, before JIT dispatch)
                    try:
                        buf_rows = np.shape(self.target_buffers[i])[0]
                        if offset + count > buf_rows:
                            raise RuntimeError(
                                f"Zero-copy buffer [{i}] overflow: "
                                f"offset={offset} + count={count} > buf_rows={buf_rows}"
                            )
                    except (TypeError, AttributeError):
                        pass  # opaque buffer, skip check

                    if self._requires_static_shapes and get_backend().is_tensor(tensor):
                        self.target_buffers[i] = _jit_write_at_offset(
                            tensor, self.target_buffers[i], count, offset
                        )
                    else:
                        self.target_buffers[i][offset : offset + count] = tensor[:count]
                    written.append(count)

        else:
            # Buffered mode: accumulate for later consumption.
            self._rx_buffers.append((tensors, counts))
            written = counts

        if written:
            self._written_counts[source_rank] = written
        self._barrier.arrive()
        return True

    # ------------------------------------------------------------------
    # Completion interface
    # ------------------------------------------------------------------

    async def wait(self) -> None:
        """One-shot mode: block until all expected slices have arrived."""
        await self._barrier.wait()

    async def wait_batch(self) -> int:
        """Periodic mode: block until the next batch fires."""
        return await self._barrier.wait_batch()

    def close(self) -> bool:
        """Periodic mode: flush remainder batch."""
        return self._barrier.close()

    def __aiter__(self):
        return self._barrier.__aiter__()

    def is_done(self) -> bool:
        """Non-blocking: True if the one-shot barrier has fired."""
        return self._barrier.is_done()

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_buffered_slices(self) -> list:
        """Consume and return all pending (tensors, counts) tuples."""
        ret = self._rx_buffers
        self._rx_buffers = []
        return ret

    def get_per_sender_buffers(self) -> dict | None:
        """Return the per-sender buffer dict (no-offset zero-copy) or None."""
        return self._per_sender_buffers

    def get_written_counts(self) -> dict:
        """Return {source_rank: [actual_row_count_per_payload_index]}.

        Available after receive_slices() for both zero-copy modes.
        Useful when tensors arrive capacity-padded: only the first
        `count` rows of each buffer are valid data.
        """
        return dict(self._written_counts)


@compile(static_argnums=(2,))
def _stitch_with_prime(prime_rank_tags, slave_tensors, world_size):
    """
    prime_rank_tags: (N,) or scalar - origin rang tags decide where to go
    slave_tensors: pytree (list/dict), with (N, ...extra_dims) tensors
    world_size: scalar
    """
    backend = get_backend()

    # Fast path: If prime_rank_tags is a scalar, everyone goes to one destination
    if (
        isinstance(prime_rank_tags, int | np.integer)
        or len(backend.shape(prime_rank_tags)) == 0
    ):
        target_rank = int(prime_rank_tags)
        counts = backend.zeros((world_size,), dtype=backend.int32(), device="cpu")

        if target_rank >= 0 and target_rank < world_size:
            first_tensor = backend.tree_flatten(slave_tensors)[0][0]
            rows = backend.shape(first_tensor)[0]
            counts = backend.index_update(counts, (target_rank,), rows)

        offsets = backend.zeros((world_size,), dtype=backend.int32(), device="cpu")
        return slave_tensors, counts, offsets

    # Sentinel Protocol:
    # Valid tags: 0 <= tag < world_size
    # Invalid tags (Padding): tag == world_size
    # We use minlength=world_size + 1 to naturally collect garbage in the last bin.

    sort_idx = backend.argsort(prime_rank_tags)

    counts_with_garbage = backend.bincount(prime_rank_tags, minlength=world_size + 1)
    counts = counts_with_garbage[:world_size]

    offsets_with_garbage = backend.cumsum(counts_with_garbage) - counts_with_garbage
    offsets = offsets_with_garbage[:world_size]

    def reorder_fn(x):
        return x[sort_idx]

    stitched_tensors = backend.tree_map(reorder_fn, slave_tensors)
    return stitched_tensors, counts, offsets


def _build_routing_domain(
    target: PhysicalLayout,
    src_view: SrcSideView,
    primary_local_shape: tuple[int | None, ...],
    src_geometry_indices: Any | None,
    dst_content_routing_values: Any | None,
    bk_static: bool,
    capacity: int,
    actual_req: int,
    dst_global_shape: tuple[int, ...] | None = None,
) -> RoutingDomain:
    """Build the RoutingDomain consumed by ``TransportPlanner``.

    L1's *single* contract with L2: the planner sees only routing-domain
    sizes and 1-D index tensors per axis — never tensor data shape.

    Sizing rule per axis:
      - ``axis.size`` is the strategy's ``dim_size`` (the GLOBAL routing
        extent over which the strategy partitions). For static targets we
        prefer ``dst_global_shape[dim]`` when available, otherwise we fall
        back to ``primary_local_shape[dim]`` (correct for fully replicated
        / single-rank cases).
      - ``axis.indices`` is an ``IndicesSpec`` instance (strict typing).
        For the *primary* (data-driving) routed dim we wrap the
        runtime tensor in ``LiteralIndices(values, length,
        valid_length=...)``; ``valid_length`` marks the real entries
        when capacity padding introduces ``-1`` sentinels.
        For *auxiliary* routed dims we consume the LOCAL
        ``IndicesSpec`` derived by ``derive_src_side_view`` (Patch K),
        so non-primary axes correctly reflect this src rank's actual
        fragment along that dim instead of the previous degenerate
        ``IdentityIndices(global_dim)`` placeholder.
    """
    spec = target.spec
    partitions = getattr(spec, "partitions", None) or {}
    ordered_dims = sorted(partitions.keys())
    if not ordered_dims:
        raise ValueError(
            "_build_routing_domain requires a target with non-empty partitions; "
            "fully replicated targets must be handled before reaching the planner."
        )

    # Strict invariant: SrcSideView.routed_dims must match target.partitions.
    # ``derive_src_side_view`` is the only legitimate constructor and it
    # iterates the same sorted target dims, so any mismatch points to a
    # caller wiring bug.
    if tuple(src_view.routed_dims) != tuple(ordered_dims):
        raise ValueError(
            f"SrcSideView.routed_dims={src_view.routed_dims} does not match "
            f"target routed dims {tuple(ordered_dims)}; caller must pass the"
            f" SrcSideView produced by derive_src_side_view(target=target)."
        )

    primary_dim = target.routing_dim
    if isinstance(primary_dim, tuple):
        primary_dim = primary_dim[0] if primary_dim else ordered_dims[0]
    if primary_dim not in partitions:
        primary_dim = ordered_dims[0]

    primary_indices = (
        dst_content_routing_values
        if dst_content_routing_values is not None
        else src_geometry_indices
    )

    # Encode capacity bucketing into the IndicesSpec: ``length`` is the
    # padded buffer length; ``valid_length`` is the real entry count.
    padded = bk_static and capacity > actual_req
    primary_length = capacity if padded else actual_req
    primary_valid_length: int | None = actual_req if padded else None

    def _global_dim(d: int) -> int:
        if dst_global_shape is not None and d < len(dst_global_shape):
            v = dst_global_shape[d]
            if v is not None:
                return v
        if d < len(primary_local_shape):
            v2 = primary_local_shape[d]
            if v2 is not None:
                return v2
        raise ValueError(
            f"_build_routing_domain: cannot determine global size of routing dim {d}; "
            f"dst_global_shape={dst_global_shape}, primary_local_shape={primary_local_shape}"
        )

    axis_sizes: list[int] = []
    axis_indices_list: list[IndicesSpec] = []
    for dim in ordered_dims:
        axis_sizes.append(_global_dim(dim))
        if dim == primary_dim:
            if primary_indices is not None:
                axis_indices_list.append(
                    index_source_to_indices_spec(
                        primary_indices,
                        length=primary_length,
                        valid_length=primary_valid_length,
                        backend=get_backend(),
                    )
                )
            else:
                src_spec = src_view.indices_for(dim)
                if isinstance(src_spec, ConstantIndices) and src_spec.length < 0:
                    axis_indices_list.append(
                        RangeIndices.contiguous(
                            start=src_spec.value,
                            length=primary_length,
                            valid_length=primary_valid_length,
                        )
                    )
                else:
                    axis_indices_list.append(src_spec)
        else:
            # Auxiliary routed axes: consume the LOCAL IndicesSpec from
            # SrcSideView (Patch K — fixes Patch F: previous code hard-
            # coded ``IdentityIndices(global_dim)`` and lost per-rank
            # locality on heterogeneous dp×tp layouts).
            axis_indices_list.append(src_view.indices_for(dim))

    # Patch L: finalise the entity count under zip semantics.
    #   - Primary axis defines the entity count (LiteralIndices.length
    #     under capacity bucketing, or the SrcSideView fallback length).
    #   - ``ConstantIndices(length=-1)`` placeholders inherit that
    #     length so the planner's zip-mode invariant holds.
    #   - Non-primary axes that are not ConstantIndices (RangeIndices /
    #     IdentityIndices / LiteralIndices) are required to already
    #     match the primary length — a mismatch surfaces in the
    #     planner's zip-invariant check with a clear diagnostic.
    # Choose the entity count for zip-mode finalisation.  Prefer any
    # non-ConstantIndices length (LiteralIndices under capacity bucketing,
    # or the SrcSideView RangeIndices/IdentityIndices fallback).  When all
    # axes are ConstantIndices placeholders the routing domain degenerates
    # to a single entity (one constant decision per rank).
    primary_eff_length: int | None = None
    for spec in axis_indices_list:
        if not (isinstance(spec, ConstantIndices) and spec.length < 0):
            primary_eff_length = spec.length
            break
    if primary_eff_length is None:
        primary_eff_length = 1
    finalised: list[IndicesSpec] = []
    for spec in axis_indices_list:
        if isinstance(spec, ConstantIndices) and spec.length < 0:
            finalised.append(
                ConstantIndices(
                    value=spec.value,
                    length=primary_eff_length,
                    valid_length=spec.valid_length,
                )
            )
        else:
            finalised.append(spec)
    axis_indices_list = finalised

    return RoutingDomain.from_layout(
        target=target,
        axis_sizes=tuple(axis_sizes),
        axis_indices=tuple(axis_indices_list),
        combine="zip",
    )


def _apply_with_binding(data: Any, sort_indices: Any, axes: tuple[int, ...]) -> Any:
    """Reorder ``data`` along ``axes`` using a 1-D ``sort_indices``.

    The fast path (``axes == (0,)``) is handled by the caller. This fallback
    is used by exotic layouts that bind the routing domain to a non-leading
    axis or to multiple axes simultaneously.  Implementing it across all
    three backends would require ``transpose`` / ``moveaxis`` ops that are
    currently not part of the common backend surface; we therefore emit a
    clear NotImplementedError here.  The default ``TransportTensor.routing_axes
    = (0,)`` covers every shipped layout (DP/MP/EP/SP/CP/TP), so this branch
    is only a contract placeholder.
    """
    raise NotImplementedError(
        "Non-leading routing_axes binding is not yet supported across all "
        f"backends. Got axes={axes}; only the default (0,) is implemented."
    )


def _ragged_project_rows_to_flat(
    backend,
    level_lengths: tuple[Any, ...],
    row_sort_indices,
    level_caps: tuple[int, ...],
):
    """Project routed outer rows through all ragged levels to flat values.

    Example:
      rows -> lists -> values
      splits0: [0, 2, 3]      row 0 owns child lists [0, 2)
      splits1: [0, 2, 3, 6]   those child lists own flat values [0, 3)

    The planner only sorts outer rows. This helper keeps transport open to
    deeper ragged topologies by repeatedly expanding sorted parent ids into
    child ids until the last level describes flat value segments.
    """
    zero = backend.zeros((1,), dtype=backend.int32(), device="cpu")
    parent_ids = row_sort_indices
    reordered_levels = []

    for lengths, total_children in zip(level_lengths, level_caps, strict=True):
        offsets = backend.cat([zero, backend.cumsum(lengths)[:-1]], dim=0)
        starts = backend.take(offsets, parent_ids)
        reordered_lengths = backend.take(lengths, parent_ids)
        reordered_levels.append(reordered_lengths)

        child_offsets = backend.cat([zero, backend.cumsum(reordered_lengths)], dim=0)
        child_count = backend.shape(reordered_lengths)[0]
        child_parent_ids = backend.repeat_interleave(
            backend.arange(0, child_count, dtype=backend.int32(), device="cpu"),
            reordered_lengths,
            total_repeat_length=total_children,
        )
        child_iota = backend.arange(
            0, total_children, dtype=backend.int32(), device="cpu"
        )
        child_starts = backend.take(child_offsets, child_parent_ids)
        child_inner_offsets = child_iota - child_starts
        parent_starts = backend.take(starts, child_parent_ids)
        parent_ids = parent_starts + child_inner_offsets

    return parent_ids, tuple(reordered_levels)


@compile(static_argnums=(2, 3, 5, 6))
def _jit_plan_and_pack(
    payloads_tuple: tuple[Any, ...],
    nested_length_list: tuple[Any | None, ...],
    src_local_rank: int,
    domain_static: RoutingDomain,
    indices_dyn: tuple[Any, ...],
    payload_routing_axes: tuple[tuple[int, ...], ...],
    static_shapes: tuple[int, ...],
):
    """JIT-friendly kernel: compute the transport plan and reorder payloads.

    The kernel takes the routing description as a (static, dynamic) split:
    ``domain_static`` carries every hashable, type-only field of the
    ``RoutingDomain`` (and is therefore safe as a ``jax.jit`` static arg);
    ``indices_dyn`` carries the per-axis runtime index tensors. They are
    re-merged before invoking ``TransportPlanner.compute_plan``.

    Returns O(N) flat reordered arrays together with per-device
    ``(offsets, counts)`` vectors.  The caller slices each device's window
    directly from the flat buffer, avoiding the O(W*N) vmap-padded
    intermediate that the previous implementation produced.
    """
    backend = get_backend()

    # Re-merge static + dynamic legs of the routing domain.  Only
    # `LiteralIndices` axes carry a runtime tensor; the spec for those is
    # rebuilt with `values=indices_dyn[i]`.  RangeIndices / IdentityIndices
    # axes are pass-through (their dyn entry is None).
    def _merge_axis(ax: RoutingAxis, dyn: Any) -> RoutingAxis:
        spec = ax.indices
        if isinstance(spec, LiteralIndices) and dyn is not None:
            new_spec = dataclasses.replace(spec, values=dyn)
            return dataclasses.replace(ax, indices=new_spec)
        return ax

    merged_axes = tuple(
        _merge_axis(ax, dyn)
        for ax, dyn in zip(domain_static.axes, indices_dyn, strict=True)
    )
    domain = dataclasses.replace(domain_static, axes=merged_axes)

    plan = TransportPlanner.compute_plan(src_local_rank, domain)
    sort_indices = plan.sort_indices
    dest_devices = plan.dest_devices
    send_counts = plan.send_counts
    replica_size = plan.replication_factor

    cs = backend.cumsum(send_counts, dim=0)
    zero = backend.zeros((1,), dtype=backend.int32(), device="cpu")
    send_offsets = backend.cat([zero, cs[:-1]], dim=0)

    # Reorder data per payload — O(N) flat reorder, no vmap padding.
    flat_buffers_lists = []
    counts_per_payload = []
    offsets_per_payload = []

    for i in range(len(payloads_tuple)):
        data = payloads_tuple[i]
        level_lengths = nested_length_list[i]
        is_ragged = level_lengths is not None
        axes = payload_routing_axes[i]

        if not is_ragged:
            # Dense path: one gather places all rows in send order.
            if axes == (0,):
                reordered_data = backend.take(data, sort_indices)
            else:
                reordered_data = _apply_with_binding(data, sort_indices, axes)
            flat_buffers_lists.append((reordered_data, None))
            counts_per_payload.append(send_counts)
            offsets_per_payload.append(send_offsets)

        else:
            # Ragged path: routing binds to outer rows, then projects row
            # order through every nested split level before gathering values.
            level_caps = tuple(
                int(backend.shape(level_lengths[j + 1])[0])
                if j + 1 < len(level_lengths)
                else static_shapes[i + 1]
                for j in range(len(level_lengths))
            )
            flat_indices, reordered_levels = _ragged_project_rows_to_flat(
                backend, level_lengths, sort_indices, level_caps
            )
            reordered_data = backend.take(data, flat_indices)

            level_counts = []
            level_offsets = []
            parent_offsets = send_offsets
            parent_counts = send_counts
            for reordered_lengths in reordered_levels:
                level_counts.append(parent_counts)
                level_offsets.append(parent_offsets)
                level_cs = backend.cat([zero, backend.cumsum(reordered_lengths)], dim=0)
                child_offsets = backend.take(level_cs, parent_offsets)
                child_counts = backend.take(
                    level_cs, parent_offsets + parent_counts
                ) - backend.take(level_cs, parent_offsets)
                parent_offsets = child_offsets
                parent_counts = child_counts

            send_flat_counts = parent_counts
            send_flat_offsets = parent_offsets

            flat_buffers_lists.append((reordered_data, reordered_levels))
            counts_per_payload.append(send_flat_counts)
            offsets_per_payload.append(send_flat_offsets)
            for child_counts, child_offsets in zip(
                level_counts, level_offsets, strict=True
            ):
                counts_per_payload.append(child_counts)
                offsets_per_payload.append(child_offsets)

    return (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        send_counts,
        send_offsets,
        dest_devices,
        replica_size,
        sort_indices,
    )
