"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

import dataclasses
from typing import Any
from collections.abc import Callable
import logging

import asyncio
import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.transport.planner import TransportPlanner
from fluxway.common.types.sharding import (
    PhysicalLayout,
    RoutingAxis,
    RoutingDomain,
    ConstantIndices,
    IdentityIndices,
    IndicesSpec,
    LiteralIndices,
    RangeIndices,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.execution import compile
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor

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


def _is_owner_on_replicated_axes(src_spec: PhysicalLayout, src_local_rank: int) -> bool:
    """True iff ``src_local_rank`` has coord 0 on every src-replicated axis.

    On the canonical (coord==0) rank we transmit real payload; on the
    other replicas we still participate in the SPMD round but emit empty
    packets. This collapses ``K`` duplicate sends to 1 without dropping
    SPMD safety.
    """
    rep_axes = _src_replicated_axes(src_spec)
    if not rep_axes:
        return True
    try:
        coords = src_spec.mesh.get_coordinate(src_local_rank)
    except (ValueError, AttributeError):
        # Unknown / out-of-range rank: treat as owner to preserve at-least-once.
        return True
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
    return bool(target.is_content_addressable)


def _src_local_indices_for_routed_dim(
    src_spec: PhysicalLayout | None,
    src_local_rank: int,
    dim: int,
    dst_global_dim: int,
    target: PhysicalLayout | None = None,
) -> IndicesSpec:
    """Derive the LOCAL IndicesSpec held by ``src_local_rank`` along ``dim``.

    Four structural cases (no runtime data inspection):

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
    4. Other partition strategies (Cyclic / Hash / Indexed): we currently
       fall back to ``IdentityIndices(length=local_chunk_size)`` because
       the strict-typed Cyclic/Hash spec extraction is part of the K-2
       follow-up.  The fallback is correct for chunk-style src layouts
       and is the dominant case in the codebase.
    """
    if src_spec is None or dim not in src_spec.partitions:
        return IdentityIndices(length=int(dst_global_dim))

    sub_size = src_spec.shard_size(dim)
    if sub_size <= 0:
        return IdentityIndices(length=int(dst_global_dim))

    chunk = max(1, dst_global_dim // sub_size)
    shard_rank = src_spec.get_shard_rank(src_local_rank, dim)
    start = shard_rank * chunk
    # The last shard absorbs the remainder when dst_global_dim is not
    # divisible by sub_size; conservative upper bound = dst_global_dim - start.
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
            dst_sub_size = int(target.shard_size(dim))
        except Exception:
            dst_sub_size = -1
        if dst_sub_size > 0 and dst_sub_size == int(sub_size):
            return ConstantIndices(value=int(start), length=-1)

    return RangeIndices.contiguous(start=int(start), length=int(length))


def _expected_senders_per_dst(
    src_spec: PhysicalLayout | None,
    target: PhysicalLayout,
    dst_global_shape: tuple[int, ...],
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
    if _is_data_driven_routing(target):
        return None
    if target is None or target.mesh is None:
        return None

    dst_world = int(target.mesh.size)
    counts = [0] * dst_world

    if src_spec is None or src_spec.mesh is None:
        return None

    src_world = int(src_spec.mesh.size)
    routed_dims = sorted(target.partitions.keys())
    if not routed_dims:
        return None

    # Pre-compute per-dim chunk geometry for the dst layout.
    dim_geom: dict[int, tuple[int, int, list[str]]] = {}
    for dim in routed_dims:
        global_dim = int(dst_global_shape[dim]) if dim < len(dst_global_shape) else 0
        sub_size = int(target.shard_size(dim))
        if sub_size <= 0 or global_dim <= 0:
            return None
        chunk = max(1, global_dim // sub_size)
        sub_axes = [ax for ax in target.partitions[dim] if ax in target.mesh.axis_names]
        dim_geom[dim] = (chunk, sub_size, sub_axes)

    mesh_axis_names = list(target.mesh.axis_names)

    for src_rank in range(src_world):
        # owner-gating: skip non-canonical replicas.
        if not _is_owner_on_replicated_axes(src_spec, src_rank):
            continue

        # Per-dim: range -> set of dst sub-ranks intersected.
        per_dim_subranks: list[list[int]] = []
        feasible = True
        for dim in routed_dims:
            global_dim = (
                int(dst_global_shape[dim]) if dim < len(dst_global_shape) else 0
            )
            spec = _src_local_indices_for_routed_dim(
                src_spec, src_rank, dim, global_dim, target=target
            )
            chunk, sub_size, _sub_axes = dim_geom[dim]
            if isinstance(spec, ConstantIndices):
                # Single-valued axis: the whole rank lives on one dst
                # sub-rank along this dim.  ``value`` is the GLOBAL chunk
                # start; ``value // chunk`` recovers the dst sub-rank
                # under isomorphic geometry.
                sub_rank = max(0, min(sub_size - 1, int(spec.value) // chunk))
                per_dim_subranks.append([sub_rank])
            elif isinstance(spec, RangeIndices):
                if spec.length == 0:
                    feasible = False
                    break
                lo = spec.start
                hi = spec.start + (spec.length - 1) * spec.stride
                lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                first_sub = max(0, min(sub_size - 1, lo // chunk))
                last_sub = max(0, min(sub_size - 1, hi // chunk))
                per_dim_subranks.append(list(range(first_sub, last_sub + 1)))
            elif isinstance(spec, IdentityIndices):
                if spec.length == 0:
                    feasible = False
                    break
                last = spec.length - 1
                first_sub = 0
                last_sub = max(0, min(sub_size - 1, last // chunk))
                per_dim_subranks.append(list(range(first_sub, last_sub + 1)))
            else:
                # LiteralIndices — data-driven; the outer guard should have
                # caught this. Be defensive.
                return None
        if not feasible:
            continue

        hit_dsts: set[int] = set()

        def _walk(
            d_idx: int,
            mesh_coords: list[int],
            out: set[int] = hit_dsts,
            ranges: list[list[int]] = per_dim_subranks,
            dims: list[int] = routed_dims,
        ) -> None:
            if d_idx == len(dims):
                dst_rank = 0
                for c, s in zip(mesh_coords, target.mesh.shape, strict=True):
                    dst_rank = dst_rank * s + c
                if 0 <= dst_rank < dst_world:
                    out.add(dst_rank)
                return
            dim = dims[d_idx]
            _chunk, sub_size, sub_axes = dim_geom[dim]
            sub_sizes = [
                target.mesh.shape[mesh_axis_names.index(ax)] for ax in sub_axes
            ]
            for sub_rank in ranges[d_idx]:
                # Unravel sub_rank into mesh coords on this dim's axes.
                rem = sub_rank
                local_coords = [0] * len(sub_sizes)
                for i in range(len(sub_sizes)):
                    rest = 1
                    for s2 in sub_sizes[i + 1 :]:
                        rest *= s2
                    if rest == 0:
                        local_coords[i] = 0
                    elif i < len(sub_sizes) - 1:
                        local_coords[i] = rem // rest
                        rem = rem % rest
                    else:
                        local_coords[i] = rem
                next_coords = list(mesh_coords)
                for ax, c in zip(sub_axes, local_coords, strict=True):
                    next_coords[mesh_axis_names.index(ax)] = c
                _walk(d_idx + 1, next_coords)

        _walk(0, [0] * len(mesh_axis_names))
        for d in hit_dsts:
            counts[d] += 1

    return tuple(counts)


def _materialise_dst_rank(
    target: PhysicalLayout,
    dims: list[int],
    coord: dict[int, int],
    dst_global_shape: tuple[int, ...],
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
        global_dim = int(dst_global_shape[dim]) if dim < len(dst_global_shape) else 0
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
    dst_global_shape: tuple[int, ...],
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
        global_dim = int(dst_global_shape[dim]) if dim < len(dst_global_shape) else 0
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
        view = self._current_src_view
        if view is None or view.expected_senders_per_dst is None:
            return int(src_world_size)
        if 0 <= dst_rank < len(view.expected_senders_per_dst):
            return int(view.expected_senders_per_dst[dst_rank])
        return int(src_world_size)

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

        # 1. Determine Addressability
        # Mirror is never content-addressable (it uses physical receipts)
        if is_referential:
            dst_is_content_addressable = False
        else:
            dst_is_content_addressable = getattr(
                target, "is_content_addressable", False
            )

        # 2. Pack Tensors and Metadata (Required for all paths)
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
        nested_length_list = []
        compact_desc_info: list[IndexDescriptor | None] = []

        for tt_idx, tt in enumerate(tensors_to_pack):
            payloads.append(backend.adopt(tt.flat_values))
            meta_tensor_state = 0

            if tt.is_ragged:
                meta_tensor_state |= 1
                splits = backend.adopt(tt.nested_splits[0])
                lengths = splits[1:] - splits[:-1]
                nested_length_list.append(lengths)
                if is_mirror:
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
                    idx = (
                        tt.origin_indices.materialize()
                        if hasattr(tt.origin_indices, "materialize")
                        else tt.origin_indices
                    )
                    # Hard interception: origin_indices length MUST match the
                    # primary routing axis length of flat_values. A common
                    # mistake is copying ``unique_ids.origin_indices`` (length
                    # K = batch x col) onto a *deduplicated* TransportTensor
                    # whose row count is N (unique count). The two lengths
                    # silently disagree and would later surface as an opaque
                    # ``numpy.take`` IndexError inside ``_jit_plan_and_pack``.
                    # Catching it here keeps the error in fluxway's concept
                    # space.
                    pri_axis = (tt.routing_axes or (0,))[0]
                    pri_shape = getattr(tt.flat_values, "shape", None)
                    pri_len = (
                        int(pri_shape[pri_axis])
                        if pri_shape is not None and pri_axis < len(pri_shape)
                        else None
                    )
                    idx_shape = getattr(idx, "shape", None)
                    idx_len = (
                        int(idx_shape[0])
                        if idx_shape is not None and len(idx_shape) >= 1
                        else None
                    )
                    if (
                        pri_len is not None
                        and idx_len is not None
                        and idx_len != pri_len
                    ):
                        raise RuntimeError(
                            "[Fluxway TransportTensor Error] origin_indices "
                            f"length mismatch for tuuid(s)={self._format_tuuids(tuuid, tuuid_to_name)} "
                            f"(tt_idx={tt_idx}): "
                            f"flat_values.shape={tuple(pri_shape) if pri_shape is not None else None}, "
                            f"routing_axes={tt.routing_axes}, primary_axis={pri_axis}, "
                            f"primary_axis_len={pri_len}, "
                            f"origin_indices type={type(tt.origin_indices).__name__}, "
                            f"origin_indices length={idx_len}. "
                            "Did you copy origin_indices from a different-length "
                            "TransportTensor (e.g. unique_ids -> sparse_features after dedup)? "
                            "Use TransportTensor(values) and rely on mirror reference "
                            "to inherit the upstream routing instead."
                        )
                    payloads.append(backend.adopt(idx))
                    nested_length_list.append(None)
                    compact_desc_info.append(None)
            else:
                compact_desc_info.append(None)

            has_meta_tensor_state_list.append(meta_tensor_state)

        # 3. Explicit Mirror Early-Exit
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

        # 1. Extract Initial Routing Signal from Primary Tensor
        src_geometry_indices = primary.origin_indices
        if src_geometry_indices is None:
            src_geometry_indices = primary.routing_indices

        # 2. Resolve Final Source Signal
        if src_geometry_indices is not None:
            # Case A: Explicitly provided indices (e.g. from a previous OP)
            if hasattr(src_geometry_indices, "materialize"):
                src_geometry_indices = src_geometry_indices.materialize()
            src_geometry_indices = backend.adopt(src_geometry_indices)
            # Hard interception: in the algebraic branch the routing source
            # is built from ``primary.origin_indices`` (or, as a fallback,
            # ``primary.routing_indices``). Its length defines the
            # routing-domain size; if it disagrees with the primary's
            # routing-axis length, ``sort_indices`` will index past the
            # tensor's first dimension inside ``_jit_plan_and_pack``.
            # Reject the mismatch here with a clear, fluxway-level error.
            pri_axis_alg = (primary.routing_axes or (0,))[0]
            pri_shape_alg = getattr(primary.flat_values, "shape", None)
            pri_len_alg = (
                int(pri_shape_alg[pri_axis_alg])
                if pri_shape_alg is not None and pri_axis_alg < len(pri_shape_alg)
                else None
            )
            src_shape_alg = getattr(src_geometry_indices, "shape", None)
            src_len_alg = (
                int(src_shape_alg[0])
                if src_shape_alg is not None and len(src_shape_alg) >= 1
                else None
            )
            if (
                pri_len_alg is not None
                and src_len_alg is not None
                and src_len_alg != pri_len_alg
            ):
                raise RuntimeError(
                    "[Fluxway Algebraic Routing Error] routing source "
                    "length does not match primary.flat_values routing axis "
                    f"for tuuid(s)={self._format_tuuids(tuuid, tuuid_to_name)}: "
                    f"primary.flat_values.shape="
                    f"{tuple(pri_shape_alg) if pri_shape_alg is not None else None}, "
                    f"routing_axes={primary.routing_axes}, "
                    f"primary_axis={pri_axis_alg}, primary_axis_len={pri_len_alg}, "
                    f"routing_source_len={src_len_alg} "
                    f"(from {'origin_indices' if primary.origin_indices is not None else 'routing_indices'}). "
                    "Algebraic routing requires len(routing_source) == "
                    "flat_values.shape[routing_axis]. Did you copy "
                    "origin_indices from an upstream TransportTensor with a "
                    "different row count? Either drop the stale "
                    "origin_indices (use TransportTensor(values) and rely on "
                    "mirror reference) or supply correctly-sized "
                    "routing_indices."
                )
            dst_content_routing_values = None
        elif source and source.is_content_addressable:
            # Case B: Content-addressable (Dynamic) routing
            src_geometry_indices = backend.adopt(primary.flat_values)
            dst_content_routing_values = None
        elif source:
            # Case C: Rank-based (Static) routing
            source_routing_dim = source.routing_dim
            if isinstance(source_routing_dim, tuple):
                source_routing_dim = source_routing_dim[0]

            dim_size = (
                local_shape[source_routing_dim]
                if source_routing_dim < len(local_shape)
                else 0
            )
            effective_rank = source.get_shard_rank(self.local_rank, source_routing_dim)
            start = effective_rank * dim_size
            src_geometry_indices = np.arange(start, start + dim_size, dtype=np.int32)
            dst_content_routing_values = None
        else:
            raise ValueError(
                f"[Fluxway Distributor Error] No sharding source available for TUUID(s): {self._format_tuuids(tuuid, tuuid_to_name)}.\n"
                f"  Problem: The algebraic path requires either a valid 'source' layout or explicit 'routing_indices'.\n"
                f"  Fix: Ensure the TransportTensor is initialized with a source PhysicalLayout."
            )

        # Resolve Destination Signal (Content-based Hash)
        dst_is_content_addressable = target.is_content_addressable if target else False
        if dst_is_content_addressable and not is_referential:
            dst_content_routing_values = (
                primary.routing_indices
                if primary.routing_indices is not None
                else primary.flat_values
            )
            if hasattr(dst_content_routing_values, "materialize"):
                dst_content_routing_values = dst_content_routing_values.materialize()
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
            src_mesh_size = source.mesh.size if source.mesh else 1
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
                if global_dim > 0 and global_dim < sub_mesh_size:
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

        if source and target:
            logger.debug(
                f"[Distributor] Planning: src_mesh={source.mesh.size if source.mesh else 1}, "
                f"dst_mesh={target.mesh.size if target.mesh else 1}, "
                f"local_rank={self.local_rank}, src_indices={src_geometry_indices}"
            )

        # Set capacity bucket. Bucketing is *only* needed when the active
        # backend requires static shapes (jax.jit / torch.compile). Pure
        # eager paths (HybridBackend default, TorchBackend eager) keep
        # capacity == actual_req so all padding branches below are no-ops.
        actual_req = local_shape[0] if local_shape else 1
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
                if actual_req == 0:
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
        padded_nested_length_list = []
        static_shapes = [capacity]

        for p, p_len in zip(payloads, nested_length_list, strict=True):
            if p_len is not None:
                # Ragged payload
                if bk_static and capacity != actual_req:
                    pad_len_size = capacity - actual_req
                else:
                    pad_len_size = 0

                if pad_len_size > 0:
                    len_padding = backend.zeros(
                        (pad_len_size,), dtype=backend.dtype(p_len)
                    )
                    padded_len = backend.cat([p_len, len_padding], dim=0)
                else:
                    padded_len = p_len
                padded_nested_length_list.append(padded_len)

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

        # Pad routing indices and routing values to capacity bucket shape with -1
        if src_geometry_indices is not None:
            src_geometry_indices = backend.as_tensor(
                src_geometry_indices, dtype=backend.int32()
            )
            if bk_static and capacity != actual_req:
                pad_size = capacity - actual_req
                if pad_size > 0:
                    padding = backend.full((pad_size,), -1, dtype=backend.int32())
                    src_geometry_indices = backend.cat(
                        [src_geometry_indices, padding], dim=0
                    )

        if dst_content_routing_values is not None:
            dst_content_routing_values = backend.as_tensor(
                dst_content_routing_values, dtype=backend.int32()
            )
            if bk_static and capacity != actual_req:
                pad_size = capacity - actual_req
                if pad_size > 0:
                    padding = backend.full((pad_size,), -1, dtype=backend.int32())
                    dst_content_routing_values = backend.cat(
                        [dst_content_routing_values, padding], dim=0
                    )

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
        src_view = derive_src_side_view(
            src_spec=source,
            src_local_rank=self.local_rank,
            target=target,
            dst_global_shape=tuple(dst_shape)
            if dst_shape is not None
            else tuple(local_shape),
        )
        # K-5: persist for upper layers (e.g. ArrivalBarrier N derivation).
        self._current_src_view = src_view

        domain = _build_routing_domain(
            target=target,
            src_view=src_view,
            primary_local_shape=local_shape,
            src_geometry_indices=src_geometry_indices,
            dst_content_routing_values=dst_content_routing_values,
            bk_static=bk_static,
            capacity=capacity,
            actual_req=actual_req,
            dst_global_shape=dst_shape,
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

        if sort_indices is not None and capacity > actual_req:
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
                    # TODO(He Jia): Support Cyclic
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

                if p_len is not None:  # ragged: also slice the item-length array
                    len_offset = int(offsets_per_payload_host[c_idx + 1][device_id])
                    count_len = int(counts_per_payload_host[c_idx + 1][device_id])
                    p_flat.append(
                        backend.get_contiguous_view(p_len, len_offset, count_len)
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

        if is_replicated:
            # Fan-out
            sorted_dest_host = _backend.to_numpy(dest_devices)

            for device_id, count in enumerate(counts_host):
                # Static optimization: skip a dst whose expected sender
                # count is 0 (proven unreached by the routing closed-form).
                if (
                    expected_per_dst is not None
                    and device_id < len(expected_per_dst)
                    and expected_per_dst[device_id] == 0
                ):
                    continue

                targets = sorted_dest_host[device_id].flatten().tolist()
                p_flat, c_flat = _get_flat_payloads_for_rank(device_id)

                # Owner gating: non-canonical replicas emit empty packets
                # but still participate in the SPMD round.
                if not is_sender_active:
                    c_flat = [0] * len(c_flat)

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    None,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
                )
        else:
            # Scatter-send
            for device_id, count in enumerate(counts_host):
                # Static optimization: skip a dst whose expected sender
                # count is 0 (proven unreached by the routing closed-form).
                if (
                    expected_per_dst is not None
                    and device_id < len(expected_per_dst)
                    and expected_per_dst[device_id] == 0
                ):
                    continue

                targets = [device_id]
                p_flat, c_flat = _get_flat_payloads_for_rank(device_id)

                # Owner gating: non-canonical replicas emit empty packets
                # but still participate in the SPMD round.
                if not is_sender_active:
                    c_flat = [0] * len(c_flat)

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    sort_indices,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
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

        # PLAN ISOLATION:
        # We MUST NOT reuse plans across separate calls to plan_and_shard.
        # Each call represents a distinct group of tensors that might have
        # different routing requirements (e.g., different values/indices).
        self._meta_sort_indices = None
        self._current_source_spec = source
        self._current_target_spec = target

        for (
            targets,
            tensors,
            counts,
            sort_idx,
            global_shapes_list,
            has_meta_tensor_state_list,
            slice_compact_desc_params,
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
                    )
                )


class SyncDataDistributor(BaseDataDistributor):
    """Synchronous Data Distributor."""

    def distribute(
        self,
        local_data: TransportTensor | list[TransportTensor] | None = None,
        source: PhysicalLayout | None = None,
        target: PhysicalLayout | None = None,
        send_fn: Callable | None = None,
        capacity: int | list[int] | None = None,
        tuuid: int | list[int] | None = None,
        tuuid_to_name: dict[int, str] | None = None,
    ) -> tuple[list[Any], Any]:
        """Performs synchronous distribution."""
        if local_data is not None:
            self.plan_and_shard(
                local_data, source, target, capacity, tuuid, tuuid_to_name
            )

        if send_fn is None:
            return [], self._meta_sort_indices

        results = []
        for target_id in sorted(self._dest_agg.keys()):
            groups = self._dest_agg[target_id]
            merged_tuuids = []
            merged_tensors = []
            merged_counts = []
            merged_g_shapes = []
            merged_h_states = []
            merged_c_params = []
            for tuuids, tensors, counts, g_shapes, h_states, c_params in groups:
                if isinstance(tuuids, list | tuple):
                    merged_tuuids.extend(tuuids)
                else:
                    merged_tuuids.append(tuuids)
                merged_tensors.extend(tensors)
                merged_counts.extend(counts)
                merged_g_shapes.extend(g_shapes)
                merged_h_states.extend(h_states)
                merged_c_params.extend(c_params)

            res = send_fn(
                target_id,
                merged_tuuids,
                merged_tensors,
                merged_counts,
                merged_g_shapes,
                merged_h_states,
                merged_c_params,
            )

            results.append(res)

        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return results, sort_indices


class AsyncDataDistributor(BaseDataDistributor):
    """Asynchronous Data Distributor (using asyncio)."""

    async def distribute(
        self,
        local_data: TransportTensor | list[TransportTensor] | None = None,
        source: PhysicalLayout | None = None,
        target: PhysicalLayout | None = None,
        send_fn: Callable | None = None,
        capacity: int | list[int] | None = None,
        tuuid: int | list[int] | None = None,
        tuuid_to_name: dict[int, str] | None = None,
    ) -> tuple[list[Any], Any]:
        """Performs asynchronous distribution."""
        if local_data is not None:
            self.plan_and_shard(
                local_data, source, target, capacity, tuuid, tuuid_to_name
            )

        if send_fn is None:
            return [], self._meta_sort_indices

        tasks = []
        for target_id in sorted(self._dest_agg.keys()):
            groups = self._dest_agg[target_id]
            merged_tuuids = []
            merged_tensors = []
            merged_counts = []
            merged_g_shapes = []
            merged_h_states = []
            merged_c_params = []
            for tuuids, tensors, counts, g_shapes, h_states, c_params in groups:
                if isinstance(tuuids, list | tuple):
                    merged_tuuids.extend(tuuids)
                else:
                    merged_tuuids.append(tuuids)
                merged_tensors.extend(tensors)
                merged_counts.extend(counts)
                merged_g_shapes.extend(g_shapes)
                merged_h_states.extend(h_states)
                merged_c_params.extend(c_params)

            tasks.append(
                send_fn(
                    target_id,
                    merged_tuuids,
                    merged_tensors,
                    merged_counts,
                    merged_g_shapes,
                    merged_h_states,
                    merged_c_params,
                )
            )

        res = await asyncio.gather(*tasks) if tasks else []
        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return res, sort_indices


# compatibility
DataDistributor = SyncDataDistributor


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
    primary_local_shape: tuple[int, ...],
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
    padded = bool(bk_static and capacity > actual_req)
    primary_length = int(capacity if padded else actual_req)
    primary_valid_length: int | None = int(actual_req) if padded else None

    def _global_dim(d: int) -> int:
        if dst_global_shape is not None and d < len(dst_global_shape):
            v = dst_global_shape[d]
            if v is not None:
                return int(v)
        if d < len(primary_local_shape):
            return int(primary_local_shape[d])
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
                    LiteralIndices(
                        length=primary_length,
                        values=primary_indices,
                        valid_length=primary_valid_length,
                    )
                )
            else:
                # No materialised indices supplied for the primary axis.
                # Fall back to the SrcSideView-derived LOCAL spec so the
                # planner still sees the rank's fragment correctly.
                axis_indices_list.append(src_view.indices_for(dim))
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
            primary_eff_length = int(spec.length)
            break
    if primary_eff_length is None:
        primary_eff_length = 1
    finalised: list[IndicesSpec] = []
    for spec in axis_indices_list:
        if isinstance(spec, ConstantIndices) and spec.length < 0:
            finalised.append(
                ConstantIndices(
                    value=int(spec.value),
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
        lengths = nested_length_list[i]
        is_ragged = lengths is not None
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
            # Ragged path: routing always binds to the outermost axis (0,).
            # ``flat_cap`` is the static upper bound on total flat elements
            # so XLA knows the output size of repeat_interleave at trace time.
            flat_cap = static_shapes[i + 1]
            item_offsets_orig = backend.cat([zero, backend.cumsum(lengths)[:-1]], dim=0)
            reordered_item_starts = backend.take(item_offsets_orig, sort_indices)
            reordered_item_lengths = backend.take(lengths, sort_indices)

            reordered_flat_offsets = backend.cat(
                [zero, backend.cumsum(reordered_item_lengths)], dim=0
            )

            num_items = backend.shape(reordered_item_lengths)[0]
            item_ids = backend.repeat_interleave(
                backend.arange(0, num_items, dtype=backend.int32(), device="cpu"),
                reordered_item_lengths,
                total_repeat_length=flat_cap,
            )

            iota = backend.arange(0, flat_cap, dtype=backend.int32(), device="cpu")
            reordered_flat_starts = backend.take(reordered_flat_offsets, item_ids)
            inner_offsets = iota - reordered_flat_starts

            orig_item_starts = backend.take(reordered_item_starts, item_ids)
            flat_indices = orig_item_starts + inner_offsets

            reordered_data = backend.take(data, flat_indices)
            send_flat_offsets = backend.take(reordered_flat_offsets, send_offsets)

            flat_cs = backend.cat([zero, backend.cumsum(reordered_item_lengths)], dim=0)
            send_flat_counts = backend.take(
                flat_cs, send_offsets + send_counts
            ) - backend.take(flat_cs, send_offsets)

            flat_buffers_lists.append((reordered_data, reordered_item_lengths))
            counts_per_payload.append(send_flat_counts)
            offsets_per_payload.append(send_flat_offsets)
            counts_per_payload.append(send_counts)
            offsets_per_payload.append(send_offsets)

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
