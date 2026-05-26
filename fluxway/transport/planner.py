"""Transport Planner — pure routing math over a `RoutingDomain`.

The planner is L2 in the four-layer communication stack. It consumes a
`RoutingDomain` (a Cartesian product of independent `RoutingAxis` objects
plus a `DeviceMesh` and an optional `ReplicaSpec`) and emits a flat 1-D
`TransportPlan`.

Design invariants
-----------------
1. Tensor data shape is *invisible* to the planner. The single input is the
   `RoutingDomain`, which carries only routing-domain sizes — never `(N, D)`
   data shape. Historical `routing_global_shape` / `dst_partitions` /
   `dst_routing_dim` parameters are intentionally absent so the
   `sort_indices.len = N × D` broadcast leak cannot be reintroduced.
2. `sort_indices` is always 1-D of length `domain.domain_size`. Cartesian
   broadcasting only ever happens *between routing axes*, not into tensor
   data dims.
3. Backend neutrality: only generic `backend.*` ops are used (where, argsort,
   bincount, cumsum, unravel_index, coordinate_to_device, cast, zeros, ones,
   reshape). Hybrid / Jax / Torch backends are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.common.types.sharding import (
    ConstantIndices,
    IdentityIndices,
    IndicesSpec,
    LiteralIndices,
    RangeIndices,
    RoutingDomain,
)

logger = logging.getLogger(__name__)


# ============================================================
# TransportPlan — Result of compute_plan(domain)
# ============================================================


@dataclass
class TransportPlan:
    """Pure tensor description of a transport schedule.

    Attributes:
        sort_indices: 1-D permutation of length `domain.domain_size`,
            grouping rows by destination device.
        dest_devices: Tensor of destination device IDs, shape
            `(domain.domain_size,)` for non-replicated plans, or
            `(domain.domain_size, replica_size)` when `replica_size > 1`.
        send_counts: 1-D int32 tensor of length `target_num_workers`.
        target_num_workers: Total mesh size (= product over mesh.shape).
        replication_factor: `domain.replica_size`, ≥ 1.
    """

    sort_indices: Any
    dest_devices: Any
    send_counts: Any
    target_num_workers: int
    replication_factor: int = 1

    @property
    def is_replicated(self) -> bool:
        return self.replication_factor > 1

    @property
    def send_offsets(self) -> Any:
        backend = get_backend()
        counts = backend.to_numpy(self.send_counts)
        cs = np.cumsum(counts)
        return np.concatenate([[0], cs[:-1]]).astype(np.int32)

    @property
    def send_counts_tensor(self) -> Any:
        return self.send_counts

    def iter_device_slices(self):
        """Iterate by device: yield (device_id, offset, count)."""
        backend = get_backend()
        counts = backend.to_numpy(self.send_counts)
        offset = 0
        for device_id in range(self.target_num_workers):
            count = int(counts[device_id])
            yield device_id, offset, count
            offset += count


# ============================================================
# TransportPlanner — single-entry routing math
# ============================================================


class TransportPlanner:
    """Computes a 1-D TransportPlan from a `RoutingDomain`.

    The public surface is the single static `compute_plan(src_local_rank,
    domain)` method. There is no longer a separate `plan(...)` entry —
    callers must construct a `RoutingDomain` (typically via
    `RoutingDomain.from_layout(...)`) before invoking the planner.
    """

    @staticmethod
    def compute_plan(
        src_local_rank: int,
        domain: RoutingDomain,
    ) -> TransportPlan:
        """Compute the transport plan for `src_local_rank` over `domain`.

        Args:
            src_local_rank: Logical rank of the calling worker in the
                source layout. Used only for diagnostics; the routing math
                is content-driven via `domain.axes[*].indices`.
            domain: Pure routing space — independent of any tensor data
                shape. Carries one `RoutingAxis` per routed mesh group,
                plus an optional `ReplicaSpec`.

        Returns:
            A `TransportPlan` whose `sort_indices` length equals
            `domain.domain_size`.
        """
        sort_indices, dest_devices, send_counts, target_num_workers, replica_size = (
            TransportPlanner._compute_plan_internal(domain, src_local_rank)
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            target_num_workers=target_num_workers,
            replication_factor=replica_size,
        )

    @staticmethod
    def _compute_plan_internal(
        domain: RoutingDomain,
        src_local_rank: int,
    ) -> tuple[Any, Any, Any, int, int]:
        """JIT-friendly core. See `compute_plan` for the contract.

        The body never touches tensor data shape: every working tensor has
        shape `(N,)` (single-axis case) or broadcast-shape
        `(s_0, s_1, ..., s_{k-1})` then ravelled to `(N,)`, where
        `N = ∏ axis.size` and `s_i = axis_i.size`. No tensor data dimension
        ever participates.
        """
        backend = get_backend()

        mesh = domain.mesh
        mesh_names = tuple(mesh.axis_names)
        dst_num_workers = mesh.size
        num_axes = len(domain.axes)

        if num_axes == 0:
            raise ValueError(
                "RoutingDomain must have at least one RoutingAxis (full "
                "replication should be handled before reaching the planner)"
            )

        logger.debug(
            "[Planner] src_local_rank=%d domain_size=%d num_axes=%d "
            "replica_size=%d mesh_shape=%s",
            src_local_rank,
            domain.domain_size,
            num_axes,
            domain.replica_size,
            mesh.shape,
        )

        mesh_coords: list[Any | None] = [None] * len(mesh.shape)
        valid_axis_tensors: list[Any] = []
        # Effective length per axis (== sort_indices length contribution).
        # Equals `axis.indices.length`, which is decoupled from `axis.size`
        # (the strategy `dim_size` — routing-domain GLOBAL extent).
        effective_lengths: list[int] = []

        # ---------- per-axis: indices → ranks → mesh coords ----------
        # Patch L: ``domain.combine`` selects between two semantics.
        #   - "cartesian" (default): every axis_pos's rank tensor is
        #     reshape-broadcast into its own dimension so the final
        #     ``primary_dest`` shape is ``∏ effective_lengths`` — the
        #     general routing-engine semantics retained for non-row-binding
        #     callers and existing planner unit tests.
        #   - "zip": all axes are zipped element-wise; the final shape is
        #     ``primary axis length``.  Required by L1 transport's
        #     ``backend.take(data, sort_indices, axes=(0,))`` row-binding,
        #     so ``sort_indices.len == row count`` and never the inflated
        #     Cartesian product.  ConstantIndices keeps a ``(1,)`` rank
        #     tensor and broadcasts via ``coordinate_to_device`` — zero
        #     materialisation cost for the constant axis.
        zip_mode = domain.combine == "zip"

        for axis_pos, axis in enumerate(domain.axes):
            spec = axis.indices
            if not isinstance(spec, IndicesSpec):
                raise TypeError(
                    f"RoutingAxis.indices must be IndicesSpec; got "
                    f"{type(spec).__name__} on axis {axis_pos}"
                )
            if isinstance(spec, ConstantIndices) and spec.length < 0:
                raise ValueError(
                    f"ConstantIndices on axis {axis_pos} has placeholder "
                    f"length=-1; ``_build_routing_domain`` must finalise "
                    f"the length before reaching the planner."
                )
            eff_len = int(spec.length)
            effective_lengths.append(eff_len)

            is_constant = isinstance(spec, ConstantIndices)

            # ConstantIndices fast path (zip-mode only): the rank tensor is
            # a single scalar broadcast to the primary axis length
            # downstream.  We never materialise a length-sized constant
            # array.  Cartesian mode falls through to the slow path so the
            # axis still inflates ``primary_dest`` correctly.
            if is_constant and zip_mode:
                if eff_len == 0:
                    ranks_axis = backend.zeros(
                        (0,), dtype=backend.int32(), device="cpu"
                    )
                    valid_axis = backend.zeros(
                        (0,), dtype=backend.int32(), device="cpu"
                    )
                else:
                    safe = backend.tensor(
                        [int(spec.value)], dtype=backend.int32(), device="cpu"
                    )
                    ranks_axis = axis.strategy(
                        safe,
                        dim_size=axis.size,
                        num_workers=axis.num_workers,
                        index_desc=axis.index_descriptor,
                    )
                    if spec.valid_length is not None and spec.valid_length == 0:
                        valid_axis = backend.zeros(
                            (1,), dtype=backend.int32(), device="cpu"
                        )
                    else:
                        valid_axis = backend.cast(safe >= 0, backend.int32())
            elif eff_len == 0:
                # Empty axis — short-circuit to avoid arange(0) edge cases.
                ranks_axis = backend.zeros((0,), dtype=backend.int32(), device="cpu")
                valid_axis = backend.zeros((0,), dtype=backend.int32(), device="cpu")
            else:
                # Materialise the spec into a 1-D tensor so the strategy's
                # generic `(indices, dim_size, num_workers, idx_desc) -> ranks`
                # contract holds.  TODO(perf): when the strategy implements
                # `routes_for_range(spec, ...)` we can skip materialisation
                # entirely — see Task 11 follow-up.
                materialised = _materialize(spec, backend)
                tensor_valid = materialised >= 0
                safe = backend.where(tensor_valid, materialised, 0)
                ranks_axis = axis.strategy(
                    safe,
                    dim_size=axis.size,
                    num_workers=axis.num_workers,
                    index_desc=axis.index_descriptor,
                )
                valid_axis = backend.cast(tensor_valid, backend.int32())

                # Apply spec.valid_length sentinel mask (Cartesian AND with
                # the materialised tensor mask).
                if spec.valid_length is not None and spec.valid_length < eff_len:
                    iota = backend.arange(
                        0, eff_len, dtype=backend.int32(), device="cpu"
                    )
                    spec_valid = backend.cast(iota < spec.valid_length, backend.int32())
                    valid_axis = valid_axis * spec_valid

            # Combine semantics:
            #   - cartesian: reshape this axis's rank/valid tensors into
            #     their own broadcast dimension so the final shape is
            #     ``∏ effective_lengths`` after coordinate_to_device.
            #   - zip: keep 1-D shapes; constant axes carry shape (1,) and
            #     non-constant axes must already share the same length
            #     (validated below).  XLA broadcast at
            #     ``coordinate_to_device`` time aligns shapes element-wise.
            if not zip_mode and num_axes > 1:
                bcast_shape = [1] * num_axes
                bcast_shape[axis_pos] = -1
                ranks_axis = backend.reshape(ranks_axis, tuple(bcast_shape))
                valid_axis = backend.reshape(valid_axis, tuple(bcast_shape))

            sub_idx = [mesh_names.index(name) for name in axis.mesh_axes]
            sub_shape = tuple(mesh.shape[i] for i in sub_idx)
            unraveled = backend.unravel_index(ranks_axis, sub_shape)
            if not isinstance(unraveled, tuple):
                unraveled = (unraveled,)
            for mesh_pos, coord in zip(sub_idx, unraveled, strict=True):
                mesh_coords[mesh_pos] = coord

            valid_axis_tensors.append(valid_axis)

        # Zip-mode invariant: all non-constant axes must share the same
        # entity count.  Detect mismatch early so callers can fix the
        # IndicesSpec rather than silently truncating via broadcast.
        if zip_mode:
            non_const_lens = [
                int(domain.axes[i].indices.length)
                for i in range(num_axes)
                if not isinstance(domain.axes[i].indices, ConstantIndices)
            ]
            if len(set(non_const_lens)) > 1:
                raise ValueError(
                    f"RoutingDomain(combine='zip') requires all non-constant "
                    f"axes to share length; got {non_const_lens}.  Multi-axis "
                    f"routing under heterogeneous chunk geometry must be "
                    f"split at the call-site — see ``_build_routing_domain``."
                )

        # Combine valid masks across axes (Cartesian AND).
        combined_valid = valid_axis_tensors[0]
        for v in valid_axis_tensors[1:]:
            combined_valid = combined_valid * v

        # Fill mesh axes that are neither routed nor replica-bound with 0.
        # Replica axes (if any) get rewritten by the replication step below.
        replica_axes = tuple(domain.replica.axes) if domain.replica is not None else ()
        for i, name in enumerate(mesh_names):
            if mesh_coords[i] is None:
                mesh_coords[i] = backend.zeros(
                    (1,), dtype=backend.int32(), device="cpu"
                )
            # Replica axes are filled below; leaving placeholder zero is fine
            # because they get overwritten by the replication broadcast.
            del name

        # primary_dest carries the broadcast-product shape over all routing
        # axes. coordinate_to_device evaluates the broadcast and returns a
        # device-id tensor of the same broadcast shape.
        primary_dest = mesh.coordinate_to_device(tuple(mesh_coords))

        # ---------- replication: Cartesian product with replica axes ----------
        num_replicas = domain.replica_size

        if replica_axes:
            primary_coords = mesh.unravel_index(primary_dest)
            expanded_coords = [backend.reshape(c, (-1, 1)) for c in primary_coords]

            rep_axis_idx = [mesh_names.index(ax) for ax in replica_axes]
            rep_shape = [mesh.shape[idx] for idx in rep_axis_idx]

            rep_flat_indices = backend.arange(
                0, num_replicas, dtype=backend.int32(), device="cpu"
            )

            if len(replica_axes) == 1:
                expanded_coords[rep_axis_idx[0]] = backend.reshape(
                    rep_flat_indices, (1, num_replicas)
                )
            else:
                rep_unraveled = backend.unravel_index(
                    rep_flat_indices, tuple(rep_shape)
                )
                if not isinstance(rep_unraveled, tuple):
                    rep_unraveled = (rep_unraveled,)
                for ax_id, r_coord in zip(rep_axis_idx, rep_unraveled, strict=True):
                    expanded_coords[ax_id] = backend.reshape(r_coord, (1, num_replicas))

            dest_devices = mesh.coordinate_to_device(tuple(expanded_coords))
        else:
            dest_devices = backend.reshape(primary_dest, (-1, 1))

        # ---------- sort by destination device ----------
        primary_dest_flat = (
            backend.reshape(dest_devices, (-1,))
            if num_replicas == 1
            else backend.reshape(backend.select(dest_devices, dim=1, index=0), (-1,))
        )

        valid_mask_flat = backend.reshape(combined_valid, (-1,))

        # Sentinel rows (valid_mask == 0) are routed to a synthetic
        # `dst_num_workers` bucket so they sort to the very end and never
        # corrupt the per-device offsets used downstream.
        primary_dest_flat_for_sort = backend.where(
            valid_mask_flat > 0, primary_dest_flat, dst_num_workers
        )
        sort_indices = backend.argsort(primary_dest_flat_for_sort, dim=0, stable=True)

        # ---------- send counts (sentinels excluded via weights) ----------
        safe_primary_dest = backend.where(valid_mask_flat > 0, primary_dest_flat, 0)
        send_counts = _bincount(
            backend,
            safe_primary_dest,
            weights=backend.cast(valid_mask_flat, backend.float32()),
            minlength=dst_num_workers,
        )
        send_counts = backend.cast(send_counts, backend.int32())

        return (
            sort_indices,
            dest_devices,
            send_counts,
            dst_num_workers,
            num_replicas,
        )


# ============================================================
# Internal helpers
# ============================================================


def _bincount(
    backend, tensor: Any, weights: Any | None = None, minlength: int = 0
) -> Any:
    """Cross-backend bincount."""
    return backend.bincount(tensor, weights=weights, minlength=minlength)


def _materialize(spec: IndicesSpec, backend: Any) -> Any:
    """Visitor: materialise an `IndicesSpec` into a 1-D index tensor.

    All type dispatch on `IndicesSpec` is concentrated here so call sites
    never need `isinstance` branches.  Sentinel tail (when
    `spec.valid_length is not None`) is left in-place; the planner masks
    it out via the dedicated `valid_axis` channel.

    - `IdentityIndices(length)`        -> `arange(0, length)`
    - `RangeIndices(start, length, s)` -> `start + arange(length) * s`
    - `LiteralIndices(values, length)` -> `values` (passthrough)
    """
    if isinstance(spec, LiteralIndices):
        return spec.values
    if isinstance(spec, IdentityIndices):
        return backend.arange(0, spec.length, dtype=backend.int32(), device="cpu")
    if isinstance(spec, RangeIndices):
        if spec.length == 0:
            return backend.zeros((0,), dtype=backend.int32(), device="cpu")
        if spec.stride == 1 and spec.start == 0:
            return backend.arange(0, spec.length, dtype=backend.int32(), device="cpu")
        iota = backend.arange(0, spec.length, dtype=backend.int32(), device="cpu")
        return spec.start + iota * spec.stride
    if isinstance(spec, ConstantIndices):
        # Slow-path materialisation — only used when a non-planner caller
        # explicitly invokes ``_materialize``.  The planner avoids this
        # path entirely (see ConstantIndices fast path).
        if spec.length <= 0:
            return backend.zeros((0,), dtype=backend.int32(), device="cpu")
        return backend.tensor(
            [int(spec.value)] * int(spec.length),
            dtype=backend.int32(),
            device="cpu",
        )
    raise TypeError(f"_materialize: unknown IndicesSpec subclass {type(spec).__name__}")
