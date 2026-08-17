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
from fluxway.common.coherence import CoherenceInstance
from fluxway.transport.planner import DestinationAssignment, TransportPlanner
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
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.common.execution import compile
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor
from fluxway.common.ragged import encode_ragged_level_count, ragged_level_count
from fluxway.common.tensor_components import (
    index_descriptor_length,
    index_source_to_indices_spec,
    routing_entity_count,
)
from fluxway.transport.manifest import (
    COMPACT_PARAM_COUNT,
    ENTRY_AUX,
    ENTRY_EXPECTED_COUNT,
    ENTRY_PAYLOAD_BEGIN,
    ENTRY_PAYLOAD_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_STATE_FLAGS,
    PAYLOAD_VALID_COUNT,
    PayloadRole,
    SenderEmissionMode,
    TensorEntryKind,
    TransportFrameBuilder,
    TransportFrameHeader,
    TransportFrameTemplate,
    TransportPlanManifest,
    dtype_code,
)
from fluxway.transport.coherence import entry_coherence_action, entry_tensor_kind
from fluxway.common.coherence import CoherenceAction
from fluxway.transport.kernels import FrameTableCapacity
from fluxway.transport.routing import (
    RoutingSourceKind,
    SelectedRoutingSource,
    TransportRoutingMode,
    receipt_destination_assignment,
    select_routing_source,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class TransportDelegationDescriptor:
    transfer_code: int
    epoch: int


@dataclasses.dataclass(frozen=True, slots=True)
class TransportFrameBatch:
    context_index: int
    target_rank: int
    batch_seq: int
    frames: tuple[tuple[TransportFrameHeader, tuple[Any, ...]], ...]
    delegations: tuple[TransportDelegationDescriptor, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class TransportBatchPlan:
    frame_groups: tuple[
        tuple[int, tuple[int, TransportFrameHeader, tuple[Any, ...]]], ...
    ]

    @classmethod
    def from_frame_groups(
        cls,
        frame_groups: list[
            tuple[int, tuple[int, TransportFrameHeader, tuple[Any, ...]]]
        ],
    ) -> TransportBatchPlan:
        return cls(frame_groups=tuple(frame_groups))

    def batches(self, *, batch_max_frames: int) -> list[TransportFrameBatch]:
        return build_transport_frame_batches(
            frame_groups=list(self.frame_groups),
            batch_max_frames=batch_max_frames,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TransportPayloadPlan:
    role: PayloadRole
    segment: Any
    valid_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class TransportFrameEntryPlan:
    transfer_code: int
    layout_state: int
    tensor_kind: TensorEntryKind
    coherence_epoch: int
    expected_count: int
    compact_params: tuple[int, ...] | None
    global_shape: tuple[int, ...] | None
    payloads: tuple[TransportPayloadPlan, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class AggregatedTransportGroup:
    transfer_codes: tuple[int, ...]
    tensors: tuple[Any, ...]
    counts: tuple[int, ...]
    global_shapes: tuple[tuple[int, ...] | None, ...]
    layout_states: tuple[int, ...]
    compact_params: tuple[tuple[int, ...] | None, ...] | None
    tensor_kinds: tuple[int, ...]
    coherence_epochs: tuple[int, ...]
    expected_count: int

    def __post_init__(self) -> None:
        if self.expected_count <= 0:
            raise ValueError(
                f"aggregated transport expected_count must be positive, got {self.expected_count}."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TransportFrameTemplateKey:
    target_rank: int
    sender_spmd_code: int
    sender_rank: int
    sender_world_size: int
    entries: tuple[tuple[Any, ...], ...]


TransportFrameTemplateCache = dict[TransportFrameTemplateKey, TransportFrameTemplate]


def build_transport_frame_batches(
    frame_groups: list[tuple[int, tuple[int, TransportFrameHeader, tuple[Any, ...]]]],
    *,
    batch_max_frames: int,
) -> list[TransportFrameBatch]:
    if batch_max_frames <= 0:
        raise ValueError(f"batch_max_frames must be positive, got {batch_max_frames}.")
    frames_by_target: dict[
        tuple[int, int],
        list[tuple[TransportFrameHeader, tuple[Any, ...]]],
    ] = {}
    for context_index, frame_arg in frame_groups:
        target_rank, header, payload_segments = frame_arg
        if not isinstance(header, TransportFrameHeader):
            raise TypeError(
                f"frame_arg header must be TransportFrameHeader, got {type(header).__name__}."
            )
        key = (context_index, target_rank)
        frames_by_target.setdefault(key, []).append((header, tuple(payload_segments)))

    batches: list[TransportFrameBatch] = []
    for (context_index, target_rank), frames in frames_by_target.items():
        batch_seq = 0
        for start in range(0, len(frames), batch_max_frames):
            batch_frames = tuple(frames[start : start + batch_max_frames])
            delegations: list[TransportDelegationDescriptor] = []
            seen_delegations: set[TransportDelegationDescriptor] = set()
            for header, _payload_segments in batch_frames:
                for entry in header.entry_i32:
                    if (
                        TensorEntryKind(
                            entry_tensor_kind(int(entry[ENTRY_STATE_FLAGS]))
                        )
                        is not TensorEntryKind.TENSOR
                        or entry_coherence_action(int(entry[ENTRY_STATE_FLAGS]))
                        is not CoherenceAction.BUS_RD
                    ):
                        continue
                    descriptor = TransportDelegationDescriptor(
                        int(entry[ENTRY_ROUTE_CODE]),
                        int(entry[ENTRY_AUX]),
                    )
                    if descriptor not in seen_delegations:
                        seen_delegations.add(descriptor)
                        delegations.append(descriptor)
            batches.append(
                TransportFrameBatch(
                    context_index=context_index,
                    target_rank=target_rank,
                    batch_seq=batch_seq,
                    frames=batch_frames,
                    delegations=tuple(delegations),
                )
            )
            batch_seq += 1
    return batches


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
            transmits routing-miss entries.
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
        seen = [False] * dst_sub_size
        for idx in range(src_start, src_end):
            seen[idx % dst_sub_size] = True
        return [sub_rank for sub_rank, present in enumerate(seen) if present]

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
    def _format_transfer_codes(transfer_codes, transfer_code_to_name):
        def _format(transfer_code):
            if transfer_code_to_name and transfer_code in transfer_code_to_name:
                return f"{transfer_code} ({transfer_code_to_name[transfer_code]})"
            return str(transfer_code)

        if isinstance(transfer_codes, list | tuple):
            return "[" + ", ".join(_format(code) for code in transfer_codes) + "]"
        return _format(transfer_codes)

    def __init__(self, local_rank: int):
        self.local_rank = local_rank
        from collections import defaultdict

        # Internal aggregation retains plan-local ABI metadata per target.
        # {target_rank: [AggregatedTransportGroup]}
        self._dest_agg: dict[int, list[AggregatedTransportGroup]] = defaultdict(list)
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
        any private agreement with the distributor about routing-miss arrivals.
        """
        manifest = self._current_plan_manifest
        if manifest is None:
            return src_world_size
        return manifest.expected_count_for(dst_rank)

    def plan_routing_miss(
        self,
        *,
        target_rank: int,
        expected_count: int,
        transfer_code: int,
        payload: Any,
        coherence_instance: CoherenceInstance,
    ) -> None:
        self._plan_absent_tensor_entry(
            target_rank=target_rank,
            expected_count=expected_count,
            transfer_code=transfer_code,
            payload=payload,
            coherence_instance=coherence_instance,
            tensor_kind=TensorEntryKind.ROUTING_MISS,
        )

    def plan_completed_without_tensor(
        self,
        *,
        target_rank: int,
        expected_count: int,
        transfer_code: int,
        payload: Any,
        coherence_instance: CoherenceInstance,
    ) -> None:
        self._plan_absent_tensor_entry(
            target_rank=target_rank,
            expected_count=expected_count,
            transfer_code=transfer_code,
            payload=payload,
            coherence_instance=coherence_instance,
            tensor_kind=TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
        )

    def _plan_absent_tensor_entry(
        self,
        *,
        target_rank: int,
        expected_count: int,
        transfer_code: int,
        payload: Any,
        coherence_instance: CoherenceInstance,
        tensor_kind: TensorEntryKind,
    ) -> None:
        if tensor_kind not in (
            TensorEntryKind.ROUTING_MISS,
            TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
        ):
            raise ValueError(
                "Absent tensor entries require ROUTING_MISS or "
                "COMPLETED_WITHOUT_TENSOR, "
                f"got {tensor_kind!r}."
            )
        if target_rank < 0:
            raise ValueError(f"target_rank must be non-negative, got {target_rank}.")
        if expected_count <= 0:
            raise ValueError(f"expected_count must be positive, got {expected_count}.")
        if not isinstance(coherence_instance, CoherenceInstance):
            raise TypeError(
                "coherence_instance must be CoherenceInstance, "
                f"got {type(coherence_instance).__name__}."
            )
        target_world_size = target_rank + 1
        manifest = self._current_plan_manifest
        if manifest is not None:
            target_world_size = max(target_world_size, manifest.target_world_size)
        expected = [0] * target_world_size
        sender_has_data = [False] * target_world_size
        if manifest is not None:
            expected[: manifest.target_world_size] = manifest.expected_senders_per_dst
            sender_has_data[: manifest.target_world_size] = (
                manifest.sender_has_data_per_dst
            )
        existing = expected[target_rank]
        if existing not in (0, expected_count):
            raise ValueError(
                f"Conflicting absent-entry expected_count for target_rank={target_rank}: "
                f"existing={existing} new={expected_count}."
            )
        expected[target_rank] = expected_count
        self._current_plan_manifest = TransportPlanManifest(
            expected_senders_per_dst=tuple(expected),
            sender_has_data_per_dst=tuple(sender_has_data),
            emission_mode=SenderEmissionMode.ALL_EXPECTED_SENDERS,
        )
        self._dest_agg[target_rank].append(
            AggregatedTransportGroup(
                transfer_codes=(transfer_code,),
                tensors=(payload,),
                counts=(0,),
                global_shapes=(None,),
                layout_states=(0,),
                compact_params=None,
                tensor_kinds=(int(tensor_kind),),
                coherence_epochs=(coherence_instance.epoch,),
                expected_count=expected_count,
            )
        )

    def _plan_single_tensor_group(
        self,
        local_data: TransportTensor | list[TransportTensor],
        source: PhysicalLayout | None,
        target: PhysicalLayout | None,
        *,
        routing_source: SelectedRoutingSource,
        capacity: int | list[int] | None = None,
        transfer_codes: int | list[int] | None = None,
        transfer_code_to_name: dict[int, str] | None = None,
    ):
        """Generates plans and padded sequences for a single distribution group.

        A group is defined as one or more tensors that SHARE the same routing indices
        (e.g. follow-sharding or explicitly aligned tensors).
        """
        if transfer_codes is None:
            raise ValueError(
                "[Fluxway Distributor Error] transfer_codes is required for "
                "TransportFrame planning."
            )
        if not isinstance(routing_source, SelectedRoutingSource):
            raise TypeError("routing_source must be a SelectedRoutingSource.")
        if source is not None and source.is_referential:
            raise ValueError(
                "Transport source must use compiled non-referential geometry."
            )
        if target is not None and target.is_referential:
            raise ValueError(
                "Transport target must use compiled non-referential geometry."
            )

        tensors_to_pack = (
            [local_data] if isinstance(local_data, TransportTensor) else local_data
        )
        if not tensors_to_pack:
            raise ValueError("Transport planning requires at least one tensor.")
        if target is None:
            raise ValueError("Transport planning requires a target PhysicalLayout.")
        coherence_epochs_list = []
        for tt_idx, tt in enumerate(tensors_to_pack):
            instances = tt._coherence_instances
            if not instances:
                raise ValueError(
                    "[Fluxway Distributor Error] each TransportTensor must carry "
                    "at least one canonical coherence instance before frame planning: "
                    f"transfer_codes={self._format_transfer_codes(transfer_codes, transfer_code_to_name)} "
                    f"tt_idx={tt_idx} instances={instances!r}."
                )
            epochs = {instance.epoch for instance in instances}
            if len(epochs) != 1:
                raise ValueError(
                    "[Fluxway Distributor Error] one TransportTensor cannot carry "
                    "coherence instances from multiple epochs: "
                    f"transfer_codes={self._format_transfer_codes(transfer_codes, transfer_code_to_name)} "
                    f"tt_idx={tt_idx} instances={instances!r}."
                )
            coherence_epochs_list.append(next(iter(epochs)))

        primary = tensors_to_pack[0]
        local_shape = primary.logical_shape
        primary_logical_shape = (
            primary.global_shape if primary.global_shape is not None else local_shape
        )
        backend = get_backend()
        actual_req = primary.routing_entity_count
        if routing_source.entity_count != actual_req:
            raise ValueError(
                "Selected routing source entity count does not match the primary "
                f"tensor: routing_source={routing_source.entity_count} "
                f"tensor={actual_req}."
            )
        uses_receipt_routing = routing_source.kind is RoutingSourceKind.RECEIPT
        target_is_content_addressable = target.is_content_addressable
        receipt: SymmetryReceipt | None = None
        if uses_receipt_routing:
            receipt = routing_source.value
            if not isinstance(receipt, SymmetryReceipt):
                raise TypeError(
                    "Receipt routing requires a SymmetryReceipt routing source."
                )
            if receipt.world_size != target.mesh.size:
                raise ValueError(
                    "Receipt rank domain does not match the target mesh: "
                    f"receipt={receipt.world_size} target={target.mesh.size}."
                )
            receipt.validate_entity_count(actual_req)

        payloads = []
        payload_routing_axes: list[tuple[int, ...]] = []
        has_meta_tensor_state_list = []
        nested_length_list: list[tuple | None] = []
        compact_desc_info: list[IndexDescriptor | None] = []

        for tt_idx, tt in enumerate(tensors_to_pack):
            tensor_entity_count = tt.routing_entity_count
            if tensor_entity_count != actual_req:
                raise ValueError(
                    "Tensors in one routing batch must have the same entity count: "
                    f"primary={actual_req} tensor[{tt_idx}]={tensor_entity_count}."
                )
            payloads.append(backend.adopt(tt.flat_values))
            payload_routing_axes.append(tt.routing_axes)
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
            else:
                nested_length_list.append(None)

            if tt.origin_indices is not None:
                preserve_compact_origin = isinstance(
                    tt.origin_indices, ContinuousIndex
                ) and not (
                    uses_receipt_routing
                    and receipt is not None
                    and not receipt.is_scalar()
                )
                if preserve_compact_origin:
                    meta_tensor_state |= 4
                    compact_desc_info.append(tt.origin_indices)
                else:
                    meta_tensor_state |= 2
                    routing_len = routing_entity_count(tt.payload, tt.layout, backend)
                    idx_len = index_descriptor_length(tt.origin_indices, backend)
                    if idx_len is not None and idx_len != routing_len:
                        raise RuntimeError(
                            "[Fluxway TransportTensor Error] origin_indices "
                            "length mismatch for transfer code(s)="
                            f"{self._format_transfer_codes(transfer_codes, transfer_code_to_name)} "
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
                    payload_routing_axes.append((0,))
                    nested_length_list.append(None)
                    compact_desc_info.append(None)
            else:
                compact_desc_info.append(None)

            has_meta_tensor_state_list.append(meta_tensor_state)

        if (
            not uses_receipt_routing
            and routing_source.kind is RoutingSourceKind.SOURCE_LAYOUT
            and source is None
        ):
            raise ValueError(
                "Source-layout routing requires a source PhysicalLayout for "
                f"transfer code(s) {self._format_transfer_codes(transfer_codes, transfer_code_to_name)}."
            )

        if uses_receipt_routing:
            src_geometry_indices = None
            src_geometry_name = routing_source.kind.value
            dst_content_routing_values = None
        elif target_is_content_addressable:
            src_geometry_indices = None
            src_geometry_name = routing_source.kind.value
            dst_content_routing_values = routing_source.value
        else:
            src_geometry_indices = routing_source.value
            src_geometry_name = routing_source.kind.value
            dst_content_routing_values = None

        # For dst layout, prefer explicit global_shape, then tensor's global_shape, then primary_logical_shape.
        dst_shape = (
            primary.global_shape
            if primary.global_shape is not None
            else primary_logical_shape
        )
        if (
            actual_req == 0
            and primary.shape != ()
            and target
            and target.is_static
            and primary.global_shape is not None
        ):
            raise ValueError(
                f"[Fluxway Static Sharding Error] Static tensor has zero "
                "routing rows for transfer code(s): "
                f"{self._format_transfer_codes(transfer_codes, transfer_code_to_name)}. "
                f"local_shape={local_shape}, global_shape={dst_shape}, "
                f"target_partitions={target.partitions}. "
                "Static outputs must be non-empty; use an explicit control tensor "
                "for optional data semantics."
            )

        if (
            not uses_receipt_routing
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
                    "transfer code(s): "
                    f"{self._format_transfer_codes(transfer_codes, transfer_code_to_name)}. "
                    f"Please ensure the source TransportTensor has 'global_shape' set explicitly."
                )

        if (
            not uses_receipt_routing
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
                        "  transfer code(s): "
                        f"{self._format_transfer_codes(transfer_codes, transfer_code_to_name)}\n"
                        f"  global_shape={dst_shape}, target mesh axes={axis_names}, "
                        f"sub_mesh_size={sub_mesh_size}\n"
                        f"  Fix: Increase the data size along dimension {d} to >= {sub_mesh_size}, "
                        f"or reduce the target mesh size."
                    )

        if (
            source is not None
            and target is not None
            and logger.isEnabledFor(logging.DEBUG)
        ):
            logger.debug(
                "[Distributor] Planning: src_mesh=%s dst_mesh=%s "
                "local_rank=%s src_index_source=%s",
                source.mesh.size if source.mesh is not None else 1,
                target.mesh.size if target.mesh is not None else 1,
                self.local_rank,
                src_geometry_name,
            )

        # Set capacity bucket. Bucketing is *only* needed when the active
        # backend requires static shapes (jax.jit / torch.compile). Pure
        # eager paths (HybridBackend default, TorchBackend eager) retention
        # capacity == actual_req so all padding branches below are no-ops.
        bk_static = backend.requires_static_shapes()
        if isinstance(capacity, list | tuple):
            selected_capacity = None
            for cap in capacity:
                if cap >= actual_req and (
                    selected_capacity is None or cap < selected_capacity
                ):
                    selected_capacity = cap
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
        if not uses_receipt_routing and (primary.shape == () or is_src_replicated):
            if source is None:
                raise ValueError(
                    "Replicated transport requires a source PhysicalLayout."
                )
            target_world_size = target.mesh.size
            src_world_size = source.mesh.size

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
                [int(TensorEntryKind.TENSOR)] * len(has_meta_tensor_state_list),
                coherence_epochs_list,
            )
            return

        identity_target_rank = (
            None
            if uses_receipt_routing
            else _static_identity_target_rank(source, target, self.local_rank)
        )
        if identity_target_rank is not None and target is not None:
            target_world_size = target.mesh.size
            sender_has_data = [False] * target_world_size
            sender_has_data[identity_target_rank] = True
            self._current_plan_manifest = TransportPlanManifest(
                expected_senders_per_dst=(1,) * target_world_size,
                sender_has_data_per_dst=tuple(sender_has_data),
                emission_mode=SenderEmissionMode.STATIC_PRODUCERS_ONLY,
            )
            direct_payloads, counts, direct_compact_params = (
                _static_identity_payload_group(
                    payloads=payloads,
                    nested_length_list=nested_length_list,
                    has_meta_tensor_state_list=has_meta_tensor_state_list,
                    compact_desc_info=compact_desc_info,
                )
            )
            yield (
                [identity_target_rank],
                direct_payloads,
                counts,
                None,
                global_shapes_list,
                has_meta_tensor_state_list,
                direct_compact_params,
                [int(TensorEntryKind.TENSOR)] * len(has_meta_tensor_state_list),
                coherence_epochs_list,
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
        payload_routing_axes_tuple = tuple(payload_routing_axes)
        static_shapes_tuple = tuple(static_shapes)
        invalid_destination_count = None
        src_view: SrcSideView | None = None
        if uses_receipt_routing:
            if receipt is None:
                raise RuntimeError("Receipt routing lost its validated receipt.")
            assignment = receipt_destination_assignment(
                receipt,
                entity_count=actual_req,
                capacity=capacity,
            )
            (
                flat_buffers_lists,
                counts_per_payload,
                offsets_per_payload,
                send_counts,
                send_offsets,
                dest_devices,
                replication_factor,
                sort_indices,
                invalid_destination_count,
            ) = _destination_assignment_and_pack(
                payloads_tuple,
                tuple(padded_nested_length_list),
                assignment,
                payload_routing_axes_tuple,
                static_shapes_tuple,
            )
        else:
            src_view = derive_src_side_view(
                src_spec=source,
                src_local_rank=self.local_rank,
                target=target,
                dst_global_shape=tuple(s for s in dst_shape if s is not None)
                if dst_shape is not None
                else tuple(s for s in local_shape if s is not None),
            )
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
                dst_global_shape=tuple(s for s in dst_shape if s is not None)
                if dst_shape is not None
                else None,
            )

            def _split_axis(ax: RoutingAxis) -> tuple[RoutingAxis, Any]:
                spec = ax.indices
                if isinstance(spec, LiteralIndices):
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
                payload_routing_axes_tuple,
                static_shapes_tuple,
            )

        if invalid_destination_count is not None:
            invalid_count = int(backend.to_numpy(invalid_destination_count))
            if invalid_count:
                target_size = (
                    target.mesh.size
                    if target is not None and getattr(target, "mesh", None) is not None
                    else -1
                )
                raise ValueError(
                    "Destination assignment contains invalid rank entries: "
                    f"count={invalid_count} target_rank_count={target_size}."
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

        if source is None:
            raise ValueError(
                "Transport manifest requires a source PhysicalLayout to determine "
                "the sender rank domain."
            )
        if uses_receipt_routing:
            source_world_size = source.mesh.size
            manifest = TransportPlanManifest(
                expected_senders_per_dst=(source_world_size,) * len(counts_host),
                sender_has_data_per_dst=tuple(bool(count) for count in counts_host),
                emission_mode=SenderEmissionMode.ALL_EXPECTED_SENDERS,
            )
        else:
            if src_view is None:
                raise RuntimeError("Algebraic routing lost its source-side view.")
            expected_per_dst = src_view.expected_senders_per_dst
            is_sender_active = src_view.is_sender_active
            if expected_per_dst is None:
                source_world_size = source.mesh.size
                manifest = TransportPlanManifest(
                    expected_senders_per_dst=(source_world_size,) * len(counts_host),
                    sender_has_data_per_dst=tuple(bool(count) for count in counts_host),
                    emission_mode=SenderEmissionMode.ALL_EXPECTED_SENDERS,
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
            manifest.emission_mode is SenderEmissionMode.ALL_EXPECTED_SENDERS
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
                    tensor_kind_list = [int(TensorEntryKind.TENSOR)] * len(
                        has_meta_tensor_state_list
                    )
                else:
                    p_flat, c_flat = _get_empty_flat_payloads()
                    tensor_kind_list = [int(TensorEntryKind.ROUTING_MISS)] * len(
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
                    tensor_kind_list,
                    coherence_epochs_list,
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
                    tensor_kind_list = [int(TensorEntryKind.TENSOR)] * len(
                        has_meta_tensor_state_list
                    )
                else:
                    p_flat, c_flat = _get_empty_flat_payloads()
                    tensor_kind_list = [int(TensorEntryKind.ROUTING_MISS)] * len(
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
                    tensor_kind_list,
                    coherence_epochs_list,
                )

    def plan_and_shard(
        self,
        local_data: TransportTensor | list[TransportTensor],
        source: PhysicalLayout | None = None,
        target: PhysicalLayout | None = None,
        *,
        routing_source: SelectedRoutingSource,
        capacity: int | list[int] | None = None,
        transfer_codes: int | list[int] | None = None,
        transfer_code_to_name: dict[int, str] | None = None,
    ) -> None:
        """Computes sharding plan and buffers data for all destinations."""
        if transfer_codes is None:
            raise ValueError(
                "[Fluxway Distributor Error] transfer_codes is required for "
                "TransportFrame planning."
            )

        self._meta_sort_indices = None
        self._current_source_spec = source
        self._current_target_spec = target
        self._current_src_view = None
        self._current_plan_manifest = None

        for (
            targets,
            tensors,
            counts,
            sort_idx,
            global_shapes_list,
            has_meta_tensor_state_list,
            slice_compact_desc_params,
            tensor_kind_list,
            coherence_epochs_list,
        ) in self._plan_single_tensor_group(
            local_data,
            source,
            target,
            routing_source=routing_source,
            capacity=capacity,
            transfer_codes=transfer_codes,
            transfer_code_to_name=transfer_code_to_name,
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Distributor Plan] [Rank %s] targets=%s tensors=%s counts=%s "
                    "sort_idx=%s global_shapes_list=%s "
                    "has_meta_tensor_state_list=%s slice_compact_desc_params=%s "
                    "tensor_kind_list=%s",
                    self.local_rank,
                    targets,
                    tensors,
                    counts,
                    sort_idx,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    slice_compact_desc_params,
                    tensor_kind_list,
                    coherence_epochs_list,
                )
            if self._meta_sort_indices is None:
                self._meta_sort_indices = sort_idx
            for target_id in targets:
                manifest = self._current_plan_manifest
                if manifest is None:
                    raise RuntimeError(
                        "transport aggregation requires a plan manifest before staging frames."
                    )
                self._dest_agg[target_id].append(
                    AggregatedTransportGroup(
                        transfer_codes=(
                            tuple(transfer_codes)
                            if isinstance(transfer_codes, list | tuple)
                            else (transfer_codes,)
                        ),
                        tensors=tuple(tensors),
                        counts=tuple(counts),
                        global_shapes=tuple(global_shapes_list),
                        layout_states=tuple(has_meta_tensor_state_list),
                        compact_params=(
                            None
                            if slice_compact_desc_params is None
                            else tuple(
                                None if compact is None else tuple(compact)
                                for compact in slice_compact_desc_params
                            )
                        ),
                        tensor_kinds=tuple(tensor_kind_list),
                        coherence_epochs=tuple(coherence_epochs_list),
                        expected_count=manifest.expected_count_for(target_id),
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
        sender_rank: int,
        template_cache: TransportFrameTemplateCache | None = None,
    ) -> list[tuple[int, TransportFrameHeader, tuple[Any, ...]]]:
        frame_args = []
        for target_id, groups in self._dest_agg.items():
            frame = self._build_transport_frame_for_target(
                target_id=target_id,
                groups=groups,
                step=step,
                sender_spmd_code=sender_spmd_code,
                sender_rank=sender_rank,
                sender_world_size=sender_world_size,
                template_cache=template_cache,
            )
            if logger.isEnabledFor(logging.DEBUG):
                entry_summary = []
                for entry in frame.entry_i32:
                    payload_begin = int(entry[ENTRY_PAYLOAD_BEGIN])
                    payload_count = int(entry[ENTRY_PAYLOAD_COUNT])
                    valid_count = 0
                    if payload_count > 0:
                        valid_count = int(
                            frame.payload_i32[payload_begin][PAYLOAD_VALID_COUNT]
                        )
                    entry_summary.append(
                        (
                            int(entry[ENTRY_ROUTE_CODE]),
                            valid_count,
                            TensorEntryKind(
                                entry_tensor_kind(int(entry[ENTRY_STATE_FLAGS]))
                            ).name,
                            int(entry[ENTRY_EXPECTED_COUNT]),
                        )
                    )
                logger.debug(
                    "[TransportFrame] sender_rank=%d target_rank=%d step=%d entries=%s",
                    sender_rank,
                    target_id,
                    step,
                    entry_summary,
                )
            frame_args.append(
                (
                    target_id,
                    frame.header,
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
        transfer_codes: int | list[int] | None = None,
        transfer_code_to_name: dict[int, str] | None = None,
        step: int = 0,
        sender_spmd_code: int = 0,
        sender_rank: int | None = None,
        sender_world_size: int = 1,
        routing_mode: TransportRoutingMode = TransportRoutingMode.ALGEBRAIC,
    ) -> tuple[list[Any], Any]:
        if local_data is not None:
            tensors = (
                [local_data] if isinstance(local_data, TransportTensor) else local_data
            )
            if not tensors:
                raise ValueError("Transport distribution requires at least one tensor.")
            expected_receipt_world_size = (
                target.mesh.size if target is not None else None
            )
            routing_sources = [
                select_routing_source(
                    tensor=tensor,
                    source=source,
                    target=target,
                    mode=routing_mode,
                    expected_receipt_world_size=expected_receipt_world_size,
                )
                for tensor in tensors
            ]
            routing_source = routing_sources[0]
            if any(
                candidate.batch_key != routing_source.batch_key
                for candidate in routing_sources[1:]
            ):
                raise ValueError(
                    "One transport distribution call cannot combine tensors with "
                    "different destination assignments."
                )
            self.plan_and_shard(
                tensors,
                source,
                target,
                routing_source=routing_source,
                capacity=capacity,
                transfer_codes=transfer_codes,
                transfer_code_to_name=transfer_code_to_name,
            )

        if send_fn is None:
            return [], self._meta_sort_indices
        if sender_rank is None:
            raise ValueError("sender_rank is required when sending transport frames.")

        frame_args = self.build_transport_frame_args(
            step=step,
            sender_spmd_code=sender_spmd_code,
            sender_rank=sender_rank,
            sender_world_size=sender_world_size,
        )
        tasks = []
        for target_id, header, payload_segments in frame_args:
            tasks.append(send_fn(target_id, header, *payload_segments))

        res = await asyncio.gather(*tasks) if tasks else []
        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return res, sort_indices

    def _build_transport_frame_for_target(
        self,
        *,
        target_id: int,
        groups: list[AggregatedTransportGroup],
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
        template_cache: TransportFrameTemplateCache | None = None,
    ):
        if template_cache is not None:
            return self._build_transport_frame_for_target_from_template(
                target_id=target_id,
                groups=groups,
                step=step,
                sender_spmd_code=sender_spmd_code,
                sender_rank=sender_rank,
                sender_world_size=sender_world_size,
                template_cache=template_cache,
            )
        return self._build_transport_frame_for_target_with_builder(
            target_id=target_id,
            groups=groups,
            step=step,
            sender_spmd_code=sender_spmd_code,
            sender_rank=sender_rank,
            sender_world_size=sender_world_size,
        )

    def _build_transport_frame_for_target_with_builder(
        self,
        *,
        target_id: int,
        groups: list[AggregatedTransportGroup],
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
    ):
        builder = TransportFrameBuilder(
            step=step,
            sender_spmd_code=sender_spmd_code,
            sender_rank=sender_rank,
            sender_world_size=sender_world_size,
            capacity=_frame_table_capacity_for_groups(groups),
        )
        for entry_plan in _iter_transport_frame_entry_plans(
            groups,
            target_id=target_id,
        ):
            entry_index = builder.add_tensor_transfer_entry(
                transfer_code=entry_plan.transfer_code,
                layout_state=entry_plan.layout_state,
                tensor_kind=entry_plan.tensor_kind,
                coherence_epoch=entry_plan.coherence_epoch,
                expected_count=entry_plan.expected_count,
                compact_params=entry_plan.compact_params,
                global_shape=entry_plan.global_shape,
            )
            for payload in entry_plan.payloads:
                builder.add_payload(
                    entry_index,
                    payload.role,
                    payload.segment,
                    payload.valid_count,
                )
        return builder.build()

    def _build_transport_frame_for_target_from_template(
        self,
        *,
        target_id: int,
        groups: list[AggregatedTransportGroup],
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
        template_cache: TransportFrameTemplateCache,
    ):
        entry_plans = tuple(
            _iter_transport_frame_entry_plans(
                groups,
                target_id=target_id,
            )
        )
        key = _transport_frame_template_key(
            target_rank=target_id,
            sender_spmd_code=sender_spmd_code,
            sender_rank=sender_rank,
            sender_world_size=sender_world_size,
            entry_plans=entry_plans,
        )
        template = template_cache.get(key)
        if template is None:
            frame = self._build_transport_frame_for_target_with_builder(
                target_id=target_id,
                groups=groups,
                step=step,
                sender_spmd_code=sender_spmd_code,
                sender_rank=sender_rank,
                sender_world_size=sender_world_size,
            )
            template_cache[key] = TransportFrameTemplate.from_frame(frame)
            return frame
        return template.instantiate(
            step=step,
            payload_segments=tuple(
                payload.segment for entry in entry_plans for payload in entry.payloads
            ),
            payload_valid_counts=tuple(
                payload.valid_count
                for entry in entry_plans
                for payload in entry.payloads
            ),
            global_shapes=tuple(entry.global_shape for entry in entry_plans),
            coherence_epochs=tuple(entry.coherence_epoch for entry in entry_plans),
        )


def _iter_transport_frame_entry_plans(
    groups: list[AggregatedTransportGroup],
    *,
    target_id: int,
) -> tuple[TransportFrameEntryPlan, ...]:
    entries: list[TransportFrameEntryPlan] = []
    for group_index, group in enumerate(groups):
        transfer_codes = group.transfer_codes
        tensors = group.tensors
        counts = group.counts
        global_shapes = group.global_shapes
        layout_states = group.layout_states
        compact_params = group.compact_params
        tensor_kinds = group.tensor_kinds
        coherence_epochs = group.coherence_epochs
        transfer_code_seq = transfer_codes
        if len(tensor_kinds) != len(transfer_code_seq):
            raise ValueError(
                f"tensor_kinds length {len(tensor_kinds)} does not match "
                f"transfer code count {len(transfer_code_seq)} for target_id={target_id}."
            )
        if len(layout_states) != len(transfer_code_seq):
            raise ValueError(
                f"layout_states length {len(layout_states)} does not match "
                f"transfer code count {len(transfer_code_seq)} for target_id={target_id}."
            )
        if len(global_shapes) != len(transfer_code_seq):
            raise ValueError(
                f"global_shapes length {len(global_shapes)} does not match "
                f"transfer code count {len(transfer_code_seq)} for target_id={target_id}."
            )
        if len(coherence_epochs) != len(transfer_code_seq):
            raise ValueError(
                f"coherence_epochs length {len(coherence_epochs)} does not match "
                f"transfer code count {len(transfer_code_seq)} for target_id={target_id}."
            )
        cursor = 0
        for entry_index, transfer_code in enumerate(transfer_code_seq):
            layout_state = int(layout_states[entry_index])
            component_count = 1 + ragged_level_count(layout_state)
            if layout_state & 2:
                component_count += 1
            if cursor + component_count > len(tensors):
                raise ValueError(
                    f"group_index={group_index} entry_index={entry_index} requires "
                    f"{component_count} payload components from cursor={cursor}, "
                    f"but tensors length is {len(tensors)}."
                )
            payload_counts = counts[cursor : cursor + component_count]
            compact = (
                compact_params[entry_index]
                if compact_params is not None and entry_index < len(compact_params)
                else None
            )
            if compact is not None:
                compact = tuple(int(value) for value in compact)
            payloads = [
                TransportPayloadPlan(
                    role=PayloadRole.VALUES,
                    segment=tensors[cursor],
                    valid_count=int(payload_counts[0]),
                )
            ]
            cursor += 1
            for _ in range(ragged_level_count(layout_state)):
                payloads.append(
                    TransportPayloadPlan(
                        role=PayloadRole.RAGGED_LENGTHS,
                        segment=tensors[cursor],
                        valid_count=int(counts[cursor]),
                    )
                )
                cursor += 1
            if layout_state & 2:
                payloads.append(
                    TransportPayloadPlan(
                        role=PayloadRole.ORIGIN,
                        segment=tensors[cursor],
                        valid_count=int(counts[cursor]),
                    )
                )
                cursor += 1
            shape = global_shapes[entry_index]
            entries.append(
                TransportFrameEntryPlan(
                    transfer_code=transfer_code,
                    layout_state=layout_state,
                    tensor_kind=TensorEntryKind(int(tensor_kinds[entry_index])),
                    coherence_epoch=int(coherence_epochs[entry_index]),
                    expected_count=group.expected_count,
                    compact_params=compact,
                    global_shape=tuple(int(value) for value in shape)
                    if shape is not None
                    else None,
                    payloads=tuple(payloads),
                )
            )
        if cursor != len(tensors):
            raise ValueError(
                f"group_index={group_index} consumed {cursor} payload components, "
                f"but tensors length is {len(tensors)}."
            )
    return tuple(entries)


def _transport_frame_template_key(
    *,
    target_rank: int,
    sender_spmd_code: int,
    sender_rank: int,
    sender_world_size: int,
    entry_plans: tuple[TransportFrameEntryPlan, ...],
) -> TransportFrameTemplateKey:
    return TransportFrameTemplateKey(
        target_rank=target_rank,
        sender_spmd_code=sender_spmd_code,
        sender_rank=sender_rank,
        sender_world_size=sender_world_size,
        entries=tuple(
            (
                entry.transfer_code,
                entry.layout_state,
                int(entry.tensor_kind),
                entry.expected_count,
                entry.compact_params,
                0 if entry.global_shape is None else len(entry.global_shape),
                tuple(
                    (
                        int(payload.role),
                        dtype_code(payload.segment.dtype),
                        len(
                            tuple(int(x) for x in getattr(payload.segment, "shape", ()))
                        ),
                    )
                    for payload in entry.payloads
                ),
            )
            for entry in entry_plans
        ),
    )


def _static_identity_target_rank(
    source: PhysicalLayout | None,
    target: PhysicalLayout | None,
    local_rank: int,
) -> int | None:
    if source is None or target is None:
        return None
    if source.is_content_addressable or target.is_content_addressable:
        return None
    if source.is_referential or target.is_referential:
        return None
    if not source.is_static or not target.is_static:
        return None
    if not source.partitions or not target.partitions:
        return None
    if source.partitions != target.partitions:
        return None
    if not isinstance(source.spec.strategy, ChunkPartition):
        return None
    if not isinstance(target.spec.strategy, ChunkPartition):
        return None
    if source.mesh.axis_names != target.mesh.axis_names:
        return None
    if source.mesh.shape != target.mesh.shape:
        return None
    if tuple(source.mesh.flatten()) != tuple(target.mesh.flatten()):
        return None

    consumers = source.resolve_consumers(target, local_rank)
    if len(consumers) != 1:
        return None
    target_rank = consumers[0]
    producers = source.resolve_producers(target, target_rank)
    if producers != [local_rank]:
        return None
    return target_rank


def _payload_row_count(payload: Any) -> int:
    shape = get_backend().shape(payload)
    if not shape:
        return 1
    return int(shape[0])


def _static_identity_payload_group(
    *,
    payloads: list[Any],
    nested_length_list: list[tuple[Any, ...] | None],
    has_meta_tensor_state_list: list[int],
    compact_desc_info: list[IndexDescriptor | None],
) -> tuple[list[Any], list[int], list[list[int] | None]]:
    direct_payloads: list[Any] = []
    counts: list[int] = []
    compact_params: list[list[int] | None] = []
    payload_cursor = 0

    for meta_state, desc_obj in zip(
        has_meta_tensor_state_list,
        compact_desc_info,
        strict=True,
    ):
        if payload_cursor >= len(payloads):
            raise ValueError(
                "Static identity transport payload cursor exceeded payload list "
                f"at meta_state={meta_state}."
            )
        values = payloads[payload_cursor]
        nested_lengths = nested_length_list[payload_cursor]
        payload_cursor += 1
        direct_payloads.append(values)
        counts.append(_payload_row_count(values))

        for lengths in nested_lengths or ():
            direct_payloads.append(lengths)
            counts.append(_payload_row_count(lengths))

        if meta_state & 2:
            if payload_cursor >= len(payloads):
                raise ValueError(
                    "Static identity transport expected materialized origin payload "
                    f"for meta_state={meta_state}, but no payload remains."
                )
            origin = payloads[payload_cursor]
            payload_cursor += 1
            direct_payloads.append(origin)
            counts.append(_payload_row_count(origin))

        if meta_state & 4:
            if not isinstance(desc_obj, ContinuousIndex):
                raise TypeError(
                    "Static identity compact origin metadata requires ContinuousIndex, "
                    f"got {type(desc_obj).__name__}."
                )
            compact_params.append(desc_obj.to_compact_params())
        else:
            compact_params.append(None)

    if payload_cursor != len(payloads):
        raise ValueError(
            "Static identity transport consumed an unexpected number of payloads: "
            f"consumed={payload_cursor}, total={len(payloads)}."
        )
    return direct_payloads, counts, compact_params


def _frame_table_capacity_for_groups(
    groups: list[AggregatedTransportGroup],
) -> FrameTableCapacity:
    entry_count = 0
    payload_count = 0
    shape_count = 0
    global_shape_count = 0
    compact_count = 0

    for group_index, group in enumerate(groups):
        transfer_codes = group.transfer_codes
        tensors = group.tensors
        global_shapes = group.global_shapes
        layout_states = group.layout_states
        compact_params = group.compact_params
        tensor_kinds = group.tensor_kinds
        coherence_epochs = group.coherence_epochs
        transfer_code_seq = transfer_codes
        if len(tensor_kinds) != len(transfer_code_seq):
            raise ValueError(
                f"tensor_kinds length {len(tensor_kinds)} does not match "
                f"transfer code count {len(transfer_code_seq)} for group_index={group_index}."
            )
        if len(layout_states) != len(transfer_code_seq):
            raise ValueError(
                f"layout_states length {len(layout_states)} does not match "
                f"transfer code count {len(transfer_code_seq)} for group_index={group_index}."
            )
        if len(global_shapes) != len(transfer_code_seq):
            raise ValueError(
                f"global_shapes length {len(global_shapes)} does not match "
                f"transfer code count {len(transfer_code_seq)} for group_index={group_index}."
            )
        if len(coherence_epochs) != len(transfer_code_seq):
            raise ValueError(
                f"coherence_epochs length {len(coherence_epochs)} does not match "
                f"transfer code count {len(transfer_code_seq)} for group_index={group_index}."
            )

        cursor = 0
        entry_count += len(transfer_code_seq)
        for entry_index, layout_state in enumerate(layout_states):
            component_count = 1 + ragged_level_count(int(layout_state))
            if int(layout_state) & 2:
                component_count += 1
            if cursor + component_count > len(tensors):
                raise ValueError(
                    f"group_index={group_index} entry_index={entry_index} requires "
                    f"{component_count} payload components from cursor={cursor}, "
                    f"but tensors length is {len(tensors)}."
                )
            payload_count += component_count
            for tensor in tensors[cursor : cursor + component_count]:
                shape_count += len(tuple(int(x) for x in getattr(tensor, "shape", ())))
            cursor += component_count

            global_shape = global_shapes[entry_index]
            if global_shape is not None:
                global_shape_count += len(global_shape)
            if (
                compact_params is not None
                and entry_index < len(compact_params)
                and compact_params[entry_index] is not None
            ):
                compact_count += COMPACT_PARAM_COUNT

        if cursor != len(tensors):
            raise ValueError(
                f"group_index={group_index} consumed {cursor} payload components, "
                f"but tensors length is {len(tensors)}."
            )

    return FrameTableCapacity(
        entry_count=entry_count,
        payload_count=payload_count,
        shape_count=shape_count,
        global_shape_count=global_shape_count,
        compact_count=compact_count,
        segment_count=payload_count,
    )


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

    def __init__(
        self,
        wait_for_n_slices: int = 1,
        periodic: bool = False,
        profiler_recorder: Any | None = None,
    ):
        from fluxway.common.sync import ArrivalBarrier

        self._barrier = ArrivalBarrier(wait_for_n_slices, periodic=periodic)
        self._periodic = periodic
        self._wait_for_n_slices = wait_for_n_slices
        self._profiler_recorder = profiler_recorder

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
        if self._profiler_recorder is None:
            await self._barrier.wait()
            return
        async with self._profiler_recorder.transport_receiver_wait(
            expected_slice_count=self._wait_for_n_slices,
            periodic=self._periodic,
        ):
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

    The planner only sorts outer rows. This helper retentions transport open to
    deeper ragged topologies by repeatedly expanding sorted parent ids into
    child ids until the last level describes flat value segments.
    """
    zero = backend.zeros((1,), dtype=backend.int32())
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
            backend.arange(0, child_count, dtype=backend.int32()),
            reordered_lengths,
            total_repeat_length=total_children,
        )
        child_iota = backend.arange(0, total_children, dtype=backend.int32())
        child_starts = backend.take(child_offsets, child_parent_ids)
        child_inner_offsets = child_iota - child_starts
        parent_starts = backend.take(starts, child_parent_ids)
        parent_ids = parent_starts + child_inner_offsets

    return parent_ids, tuple(reordered_levels)


def _pack_payloads_by_plan(
    *,
    backend: Any,
    payloads_tuple: tuple[Any, ...],
    nested_length_list: tuple[Any | None, ...],
    payload_routing_axes: tuple[tuple[int, ...], ...],
    static_shapes: tuple[int, ...],
    sort_indices: Any | None,
    send_counts: Any,
):
    has_ragged_payload = any(levels is not None for levels in nested_length_list)
    if sort_indices is not None:
        sort_indices = backend.adopt(sort_indices)

    cs = backend.cumsum(send_counts, dim=0)
    zero = send_counts[:1] * 0
    send_offsets = backend.cat([zero, cs[:-1]], dim=0)
    packing_send_counts = (
        backend.adopt(send_counts) if has_ragged_payload else send_counts
    )
    packing_send_offsets = (
        backend.adopt(send_offsets) if has_ragged_payload else send_offsets
    )
    packing_zero = packing_send_counts[:1] * 0

    # Reorder data per payload — O(N) flat reorder, no vmap padding.
    flat_buffers_lists: list[tuple[Any, tuple[Any, ...] | None]] = []
    counts_per_payload = []
    offsets_per_payload = []

    for i in range(len(payloads_tuple)):
        data = payloads_tuple[i]
        raw_level_lengths = nested_length_list[i]
        level_lengths = (
            None
            if raw_level_lengths is None
            else tuple(backend.adopt(lengths) for lengths in raw_level_lengths)
        )
        is_ragged = level_lengths is not None
        axes = payload_routing_axes[i]

        if not is_ragged:
            if sort_indices is None:
                reordered_data = data
            elif axes == (0,):
                reordered_data = backend.take(data, sort_indices)
            else:
                reordered_data = _apply_with_binding(data, sort_indices, axes)
            flat_buffers_lists.append((reordered_data, None))
            counts_per_payload.append(send_counts)
            offsets_per_payload.append(send_offsets)

        else:
            if sort_indices is None:
                reordered_data = data
                reordered_levels = level_lengths
            else:
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
            parent_offsets = packing_send_offsets
            parent_counts = packing_send_counts
            for reordered_lengths in reordered_levels:
                level_counts.append(parent_counts)
                level_offsets.append(parent_offsets)
                level_cs = backend.cat(
                    [packing_zero, backend.cumsum(reordered_lengths)], dim=0
                )
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
        send_offsets,
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
    backend = get_backend()

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
    (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        send_offsets,
    ) = _pack_payloads_by_plan(
        backend=backend,
        payloads_tuple=payloads_tuple,
        nested_length_list=nested_length_list,
        payload_routing_axes=payload_routing_axes,
        static_shapes=static_shapes,
        sort_indices=plan.sort_indices,
        send_counts=plan.send_counts,
    )
    return (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        plan.send_counts,
        send_offsets,
        plan.dest_devices,
        plan.replication_factor,
        plan.sort_indices,
    )


@compile(static_argnums=(3, 4, 5, 6))
def _jit_destination_assignment_and_pack(
    payloads_tuple: tuple[Any, ...],
    nested_length_list: tuple[Any | None, ...],
    destination_ranks: Any,
    entity_count: int,
    target_rank_count: int,
    payload_routing_axes: tuple[tuple[int, ...], ...],
    static_shapes: tuple[int, ...],
):
    backend = get_backend()
    plan = TransportPlanner.compute_destination_assignment(
        DestinationAssignment(
            ranks=destination_ranks,
            entity_count=entity_count,
            target_rank_count=target_rank_count,
        )
    )
    (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        send_offsets,
    ) = _pack_payloads_by_plan(
        backend=backend,
        payloads_tuple=payloads_tuple,
        nested_length_list=nested_length_list,
        payload_routing_axes=payload_routing_axes,
        static_shapes=static_shapes,
        sort_indices=plan.sort_indices,
        send_counts=plan.send_counts,
    )
    return (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        plan.send_counts,
        send_offsets,
        plan.dest_devices,
        plan.replication_factor,
        plan.sort_indices,
        plan.invalid_destination_count,
    )


@compile(static_argnums=(4, 5))
def _jit_pack_destination_plan(
    payloads_tuple: tuple[Any, ...],
    nested_length_list: tuple[Any | None, ...],
    sort_indices: Any | None,
    send_counts: Any,
    payload_routing_axes: tuple[tuple[int, ...], ...],
    static_shapes: tuple[int, ...],
):
    return _pack_payloads_by_plan(
        backend=get_backend(),
        payloads_tuple=payloads_tuple,
        nested_length_list=nested_length_list,
        payload_routing_axes=payload_routing_axes,
        static_shapes=static_shapes,
        sort_indices=sort_indices,
        send_counts=send_counts,
    )


def _destination_assignment_and_pack(
    payloads_tuple: tuple[Any, ...],
    nested_length_list: tuple[Any | None, ...],
    assignment: DestinationAssignment,
    payload_routing_axes: tuple[tuple[int, ...], ...],
    static_shapes: tuple[int, ...],
):
    if isinstance(assignment.ranks, int | np.integer | np.ndarray):
        plan = TransportPlanner.compute_destination_assignment(assignment)
        (
            flat_buffers_lists,
            counts_per_payload,
            offsets_per_payload,
            send_offsets,
        ) = _jit_pack_destination_plan(
            payloads_tuple,
            nested_length_list,
            plan.sort_indices,
            plan.send_counts,
            payload_routing_axes,
            static_shapes,
        )
        return (
            flat_buffers_lists,
            counts_per_payload,
            offsets_per_payload,
            plan.send_counts,
            send_offsets,
            plan.dest_devices,
            plan.replication_factor,
            plan.sort_indices,
            plan.invalid_destination_count,
        )
    return _jit_destination_assignment_and_pack(
        payloads_tuple,
        nested_length_list,
        assignment.ranks,
        assignment.entity_count,
        assignment.target_rank_count,
        payload_routing_axes,
        static_shapes,
    )
