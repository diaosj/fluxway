"""
Mailbox module for dual-stream convergence.

Data plane: tensor shards from upstream workers, indexed by (op_type, name, step).
Control plane: state scalars from Driver.

Completion tracking is event-driven, not polled:
  - Each (op_type, step) has a StepBarrier composed of ArrivalBarriers.
  - When all expected shards and states arrive, the barrier fires automatically.
  - Worker registers an on_ready callback.

Zero-copy shard writing is supported via TransportReceiver delegation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from collections.abc import Callable

import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.common.index_descriptor import ContinuousIndex, ExplicitIndex
from fluxway.common.ragged import ragged_level_count
from fluxway.common.sync import ArrivalBarrier
from fluxway.common.types.configs import BarrierConfig
from fluxway.common.types.tensor import Scalar, TransportTensor
from fluxway.transport.communication import TransportReceiver
from fluxway.transport.manifest import (
    EDGE_EMPTY_KIND,
    EDGE_COMPACT_BEGIN,
    EDGE_COMPACT_COUNT,
    EDGE_EXPECTED_COUNT,
    EDGE_GLOBAL_SHAPE_BEGIN,
    EDGE_GLOBAL_SHAPE_COUNT,
    EDGE_META_STATE,
    EDGE_PAYLOAD_BEGIN,
    EDGE_PAYLOAD_COUNT,
    EDGE_TUUID,
    EmptyKind,
    FRAME_SENDER_RANK,
    FRAME_SENDER_WORLD_SIZE,
    FRAME_STEP,
    PAYLOAD_ROLE,
    PAYLOAD_SEGMENT_ARG_INDEX,
    PAYLOAD_DTYPE,
    PAYLOAD_NDIM,
    PAYLOAD_SHAPE_BEGIN,
    PAYLOAD_SHAPE_COUNT,
    PAYLOAD_VALID_COUNT,
    PayloadRole,
    TransportFrame,
    dtype_from_code,
)

logger = logging.getLogger(__name__)


_MISSING = object()


@dataclass(slots=True)
class TransportShardView:
    payloads: tuple[Any, ...]
    counts: tuple[int, ...]
    meta_state: int
    sender_world_size: int
    sender_rank: int
    global_shape: tuple[int, ...] | None
    compact_params: tuple[int, ...] | None
    _flat_values_cache: Any = field(default=_MISSING, init=False, repr=False)
    _nested_splits_cache: Any = field(default=_MISSING, init=False, repr=False)
    _origin_indices_cache: Any = field(default=_MISSING, init=False, repr=False)
    _origin_values_cache: Any = field(default=_MISSING, init=False, repr=False)
    _receipt_cache: Any = field(default=_MISSING, init=False, repr=False)

    @property
    def data_count(self) -> int:
        return self.counts[0] if self.counts else 0

    def _unwrap(self, payload: Any) -> Any:
        return payload.tensor if isinstance(payload, Scalar) else payload

    def _payload_view(self, index: int) -> Any:
        payload = self._unwrap(self.payloads[index])
        count = self.counts[index]
        backend = get_backend()
        if backend.shape(payload):
            return payload[:count]
        return payload

    @property
    def flat_values(self) -> Any:
        if self._flat_values_cache is not _MISSING:
            return self._flat_values_cache
        data_payload = self._payload_view(0)
        self._flat_values_cache = data_payload
        return data_payload

    @property
    def nested_splits(self) -> list[Any]:
        if self._nested_splits_cache is not _MISSING:
            return list(self._nested_splits_cache)
        backend = get_backend()
        splits = []
        cursor = 1
        for _ in range(ragged_level_count(self.meta_state)):
            if cursor >= len(self.payloads):
                raise ValueError(
                    "TransportShardView is missing ragged lengths payload."
                )
            lengths = self._payload_view(cursor)
            zero = backend.zeros((1,), dtype=backend.dtype(lengths))
            splits.append(backend.cat([zero, backend.cumsum(lengths, dim=0)], dim=0))
            cursor += 1
        self._nested_splits_cache = tuple(splits)
        return list(splits)

    @property
    def origin_indices(self) -> Any | None:
        if self._origin_indices_cache is not _MISSING:
            return self._origin_indices_cache
        has_origin_indices = bool(self.meta_state & 2)
        has_compact_index = bool(self.meta_state & 4)
        if has_origin_indices and has_compact_index:
            raise ValueError(
                "TransportShardView cannot carry explicit and compact origin metadata."
            )
        if has_compact_index:
            if self.compact_params is None:
                raise ValueError("TransportShardView is missing compact metadata.")
            self._origin_indices_cache = ContinuousIndex.from_compact_params(
                list(self.compact_params)
            )
        elif has_origin_indices:
            self._origin_indices_cache = ExplicitIndex(self._origin_payload_view())
        else:
            self._origin_indices_cache = None
        return self._origin_indices_cache

    @property
    def origin_indices_materialized(self) -> Any | None:
        if self._origin_values_cache is not _MISSING:
            return self._origin_values_cache
        indices = self.origin_indices
        if indices is None:
            self._origin_values_cache = None
        elif isinstance(indices, ExplicitIndex):
            self._origin_values_cache = indices.indices
        else:
            self._origin_values_cache = indices.materialize()
        return self._origin_values_cache

    @property
    def routing_indices(self) -> None:
        return None

    @property
    def routing_indices_materialized(self) -> None:
        return None

    @property
    def receipt(self) -> Any:
        if self._receipt_cache is _MISSING:
            from fluxway.common.types.tensor import SymmetryReceipt

            self._receipt_cache = SymmetryReceipt(
                self.sender_world_size, self.sender_rank
            )
        return self._receipt_cache

    @property
    def routing_axes(self) -> tuple[int, ...]:
        return (0,)

    @property
    def shape(self) -> tuple[int, ...]:
        return get_backend().shape(self.flat_values)

    @property
    def dtype(self) -> Any:
        return get_backend().dtype(self.flat_values)

    @property
    def is_ragged(self) -> bool:
        return bool(self.meta_state & 1)

    @property
    def is_dense(self) -> bool:
        return not self.is_ragged

    def __len__(self) -> int:
        if self.is_ragged:
            if len(self.counts) < 2:
                raise ValueError("TransportShardView is missing ragged row counts.")
            return self.counts[1]
        shape = self.shape
        if not shape:
            return 1
        return shape[0]

    def _origin_payload_view(self) -> Any:
        origin_index = 1 + ragged_level_count(self.meta_state)
        if origin_index >= len(self.payloads):
            raise ValueError("TransportShardView is missing origin payload.")
        return self._payload_view(origin_index)


# ---------------------------------------------------------------------------
# StepBarrier — one per (op_type, step)
# ---------------------------------------------------------------------------


class StepBarrier:
    """
    Fires when all declared tensor names have received their expected shard
    counts AND all declared state names have been received.

    Internally composes one ArrivalBarrier per named stream.

    Args:
        expected_tensors: {tuuid: required_shard_count}
        expected_states:  list of required state names
    """

    __slots__ = (
        "_op_name",
        "_step",
        "_tensor_barriers",
        "_control_arg_barriers",
        "_combined",
        "_periodic",
    )

    def __init__(
        self,
        op_name: str,
        step: int,
        expected_tensors: dict[int, int],
        expected_states: list[str],
        periodic: bool = False,
    ) -> None:
        self._op_name = op_name
        self._step = step
        self._periodic = periodic
        self._tensor_barriers: dict[int, ArrivalBarrier] = {
            u: ArrivalBarrier(count, periodic=periodic)
            for u, count in expected_tensors.items()
        }
        self._control_arg_barriers: dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(1, periodic=periodic) for name in expected_states
        }
        # Combined completion event (one-shot mode only)
        self._combined = asyncio.Event()
        if not periodic:
            self._check()  # fire immediately if both dicts are empty

    @property
    def arrived_tensors(self) -> list[int]:
        return [u for u, b in self._tensor_barriers.items() if b.count >= b.target]

    @property
    def expected_tensors(self) -> list[int]:
        return list(self._tensor_barriers.keys())

    @property
    def tensor_progress(self) -> dict[int, str]:
        return {u: f"{b.count}/{b.target}" for u, b in self._tensor_barriers.items()}

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def tensor_arrived(self, tuuid: int, current_step: int | None = None) -> bool:
        """Record one shard arriving for tensor TUUID. Returns True if barrier just fired."""
        barrier = self._tensor_barriers.get(tuuid)
        if barrier is not None:
            fired = barrier.arrive()
            step_to_log = current_step if current_step is not None else self._step
            logger.debug(
                f"[StepBarrier {self._op_name}:{step_to_log}] TENSOR ARRIVED: tuuid={tuuid}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier] UNEXPECTED TENSOR: tuuid={tuuid}, expected={list(self._tensor_barriers.keys())}"
            )
        return False

    def control_arg_arrived(self, name: str, current_step: int | None = None) -> bool:
        """Record state data arriving for `name`. Returns True if barrier just fired."""
        barrier = self._control_arg_barriers.get(name)
        if barrier is not None:
            fired = barrier.arrive()
            step_to_log = current_step if current_step is not None else self._step
            logger.debug(
                f"[StepBarrier {self._op_name}:{step_to_log}] STATE ARRIVED: name={name}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier {self._op_name}:{self._step}] UNEXPECTED STATE: name={name}, expected={list(self._control_arg_barriers.keys())}"
            )
        return False

    def configure_expected_tensor_count(self, tuuid: int, expected_count: int) -> None:
        if expected_count < 0:
            raise ValueError(
                f"expected_count for tuuid={tuuid} must be non-negative, got {expected_count}."
            )
        barrier = self._tensor_barriers.get(tuuid)
        if barrier is None:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Cannot configure "
                f"unexpected tensor tuuid={tuuid}; expected={list(self._tensor_barriers.keys())}."
            )
        if self._combined.is_set() and barrier.target != expected_count:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Cannot change "
                f"tuuid={tuuid} expected count from {barrier.target} to "
                f"{expected_count} after the step barrier fired."
            )
        if barrier.count:
            if barrier.target != expected_count:
                raise ValueError(
                    f"[StepBarrier {self._op_name}:{self._step}] Cannot change "
                    f"tuuid={tuuid} expected count from {barrier.target} to "
                    f"{expected_count} after {barrier.count} arrivals."
                )
            return
        barrier._n = expected_count
        if not self._periodic and expected_count == 0:
            barrier._event.set()
        elif not self._periodic:
            barrier._event.clear()

    # ------------------------------------------------------------------
    # Completion query
    # ------------------------------------------------------------------

    def _check(self, current_step: int | None = None) -> bool:
        if self._combined.is_set():
            return False
        tensors_ok = all(b.is_done() for b in self._tensor_barriers.values())
        states_ok = all(b.is_done() for b in self._control_arg_barriers.values())

        step_to_log = current_step if current_step is not None else self._step

        if tensors_ok and states_ok:
            logger.debug(f"[StepBarrier {self._op_name}:{step_to_log}] FIRED.")
            self._combined.set()
            return True

        # Log why it's not ready
        waiting_tensors = [
            n for n, b in self._tensor_barriers.items() if not b.is_done()
        ]
        waiting_states = [
            n for n, b in self._control_arg_barriers.items() if not b.is_done()
        ]
        logger.debug(
            f"[StepBarrier {self._op_name}:{step_to_log}] NOT READY: waiting_tensors={waiting_tensors}, waiting_states={waiting_states}"
        )
        return False

    def is_ready(self) -> bool:
        return self._combined.is_set()

    def recompute_ready(self, current_step: int | None = None) -> bool:
        return self._check(current_step=current_step)

    async def wait(self) -> None:
        await self._combined.wait()


# ---------------------------------------------------------------------------
# WorkerMailbox
# ---------------------------------------------------------------------------


class WorkerMailbox:
    """
    Mailbox for caching incoming data in an SPMD Worker.

    Control plane (state):
        {op_type → {name → {step → data}}}

    Data plane (tensors):
        {op_type → {name → {step → {rank → (data, meta)}}}}

    Completion tracking:
        One StepBarrier per (op_type, step), registered via declare_step().
        When a step becomes ready, the on_ready callback registered via
        set_ready_callback() is called immediately — no external polling.

    Zero-copy slots:
        register_zero_copy_slot() activates a TransportReceiver for a slot
        so deposit_shards() writes directly into pre-allocated buffers.
    """

    def __init__(self) -> None:
        # Control plane
        self._control_arg_data: dict[str, dict[str, dict[int, Any]]] = {}

        # Data plane (buffered path)
        self._tensor_slots: dict[
            str, dict[int, dict[int, dict[tuple[str, int], list[TransportShardView]]]]
        ] = {}
        self._tensor_arrivals: dict[
            str, dict[int, dict[int, dict[tuple[str, int], dict[str, Any]]]]
        ] = {}

        # Completion barriers: {op_name: {step: StepBarrier}}
        self._barriers: dict[str, dict[int, StepBarrier]] = {}

        # Zero-copy receivers (activated on demand)
        self._zc_receivers: dict[str, dict[int, dict[int, TransportReceiver]]] = {}

        # Called with (op_name, step) when a step barrier fires
        self._on_ready: Callable[[str, int], None] | None = None

        # Periodic mode state: {op_name: current_virtual_step}
        self._periodic_steps: dict[str, int] = {}
        # Mapping from periodic op to its perpetual barrier
        self._periodic_barriers: dict[str, StepBarrier] = {}

        # Expected counts for satisfying staleness: {op_name: {name_or_uuid: expected_count}}
        self._expected_tensor_counts: dict[str, dict[int, int]] = {}
        self._expected_control_arg_names: dict[str, list[str]] = {}

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Step barrier lifecycle
    # ------------------------------------------------------------------

    def set_ready_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Register a callback fired when a step becomes ready.

        The callback is invoked inside the asyncio.Lock, synchronously —
        use asyncio.create_task() inside it for non-blocking work.

        Args:
            callback: callable(op_name, step)
        """
        self._on_ready = callback

    def declare_step(
        self,
        op_name: str,
        step: int,
        expected_tensors: dict[int, int],
        expected_states: list[str],
        barrier_config: Optional["BarrierConfig"] = None,
    ) -> None:
        """
        Declare the expected arrivals for a step.

        Must be called before the first deposit_shards() / deposit_control_arguments() for
        that step.  If not called, deposit_* operations still work but no
        barrier fires.

        Args:
            op_name: Operation name
            step: Execution step (ignored for periodic ops)
            expected_tensors: {tuuid: required_shard_count}
            expected_states: list of required state names
            barrier_config: Configuration for the step barrier
        """
        from fluxway.common.types.configs import BarrierType

        is_periodic = (
            barrier_config.type == BarrierType.PERIODIC if barrier_config else False
        )

        self._expected_tensor_counts[op_name] = expected_tensors
        self._expected_control_arg_names[op_name] = expected_states

        if is_periodic:
            if op_name not in self._periodic_barriers:
                # Use n from barrier_config if available
                # Note: StepBarrier currently assumes n is the same for all inputs in periodic mode.
                # For now we use barrier_config.n if it is PeriodicBarrierConfig.
                from fluxway.common.types.configs import PeriodicBarrierConfig

                n = (
                    barrier_config.n
                    if isinstance(barrier_config, PeriodicBarrierConfig)
                    else 1
                )
                self._periodic_barriers[op_name] = StepBarrier(
                    op_name, -1, expected_tensors, expected_states, periodic=True
                )
                # Ensure each ArrivalBarrier inside StepBarrier uses the correct n
                for b in self._periodic_barriers[op_name]._tensor_barriers.values():
                    b._n = n
                for b in self._periodic_barriers[
                    op_name
                ]._control_arg_barriers.values():
                    b._n = n

                self._periodic_steps[op_name] = 0
            return

        if op_name in self._barriers and step in self._barriers[op_name]:
            old_b = self._barriers[op_name][step]
            logger.warning(
                f"[WorkerMailbox] DUPLICATE declare_step: op={op_name}, step={step}. "
                f"Progress: {old_b.tensor_progress}"
            )

        barrier = StepBarrier(op_name, step, expected_tensors, expected_states)
        self._barriers.setdefault(op_name, {})[step] = barrier
        logger.debug(
            f"[WorkerMailbox] declare_step: op={op_name}, step={step}, expected_tensors={expected_tensors}, expected_states={expected_states}"
        )

        # Check if any data for THIS EXACT step has already arrived.
        # This prevents progress loss if declare_step is called after some deposit_* calls.
        if expected_tensors:
            for tuuid, target in expected_tensors.items():
                arrived_shards = (
                    self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
                )
                for sender_shard_list in arrived_shards.values():
                    for _ in range(len(sender_shard_list)):
                        barrier.tensor_arrived(tuuid)
        # Check for states
        for name in expected_states:
            if step in self._control_arg_data.get(op_name, {}).get(name, {}):
                barrier.control_arg_arrived(name)

        # In case expected counts are all zero the barrier is already ready
        if barrier.is_ready() and self._on_ready:
            self._on_ready(op_name, step)

    def configure_transport_expected_counts(
        self, op_name: str, step: int, expected_counts: dict[int, int]
    ) -> None:
        if not expected_counts:
            return
        if op_name in self._periodic_barriers:
            barrier = self._periodic_barriers[op_name]
            actual_step = self._periodic_steps[op_name]
        else:
            actual_step = step
            barrier = self._barriers.get(op_name, {}).get(step)
        if barrier is None:
            raise ValueError(
                f"[WorkerMailbox] Cannot configure transport expected counts "
                f"before declare_step: op={op_name} step={step}."
            )
        for tuuid, expected_count in expected_counts.items():
            barrier.configure_expected_tensor_count(tuuid, expected_count)
            self._expected_tensor_counts.setdefault(op_name, {})[tuuid] = expected_count
        if barrier.recompute_ready(current_step=actual_step) and self._on_ready:
            self._on_ready(op_name, actual_step)

    # ------------------------------------------------------------------
    # Control plane — state data
    # ------------------------------------------------------------------

    async def deposit_control_arguments(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> None:
        """Cache state data and notify the step barrier."""
        if len(names) != len(values):
            raise ValueError("names and values must have the same length")

        async with self._lock:
            # Handle periodic ops: use virtual step
            actual_step = step
            barrier = None
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                barrier = self._barriers.get(op_name, {}).get(step)

            any_fired = False
            for name, data in zip(names, values, strict=True):  # type: ignore
                self._control_arg_data.setdefault(op_name, {}).setdefault(name, {})[
                    actual_step
                ] = data
                if barrier is not None:
                    if barrier.control_arg_arrived(name, current_step=actual_step):
                        any_fired = True

            if any_fired:
                if op_name in self._periodic_barriers:
                    self._periodic_steps[op_name] += 1
                if self._on_ready:
                    self._on_ready(op_name, actual_step)
        logger.debug(
            "Mailbox deposit_control_arguments: op=%s names=%s step=%d",
            op_name,
            names,
            step,
        )

    async def get_control_argument(
        self, op_name: str, name: str, step: int
    ) -> Any | None:
        async with self._lock:
            return self._control_arg_data.get(op_name, {}).get(name, {}).get(step)

    async def get_all_control_arguments(
        self, op_name: str, step: int
    ) -> dict[str, Any]:
        async with self._lock:
            result = {}
            for name, step_dict in self._control_arg_data.get(op_name, {}).items():
                if step in step_dict:
                    result[name] = step_dict[step]
            return result

    async def has_control_argument(self, op_name: str, name: str, step: int) -> bool:
        async with self._lock:
            return step in self._control_arg_data.get(op_name, {}).get(name, {})

    # ------------------------------------------------------------------
    # Data plane — tensor shards
    # ------------------------------------------------------------------

    def _buffered_slot(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[tuple[str, int], list[TransportShardView]]:
        return (
            self._tensor_slots.setdefault(op_name, {})
            .setdefault(tuuid, {})
            .setdefault(step, {})
        )

    def _arrival_slot(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[tuple[str, int], dict[str, Any]]:
        return (
            self._tensor_arrivals.setdefault(op_name, {})
            .setdefault(tuuid, {})
            .setdefault(step, {})
        )

    async def deposit_shards(
        self,
        op_name: str,
        tuuids: list[int],
        step: int,
        sender_id: str,
        rank: int,
        sender_world_size: int,
        payloads: list[Any],
        counts: list[int],
        global_shapes: list[tuple[int, ...] | None] | None = None,
        meta_tensor_state_list: list[int] | None = None,
        slice_compact_desc_params: list[list[int] | None] | None = None,
    ) -> int:
        """
        Cache multiple tensor shards and notify the step barrier.

        Args:
            op_name: Operation type
            tuuids: List of TUUIDs
            step: Execution step
            sender_id: ID of the sender
            rank: Sender rank
            payload_groups: List of lists of raw data arrays (data, lengths, routing_idx, origin_idx)
            counts_groups: List of lists of counts for each payload
            meta_tensor_state_list: List of ints representing the bitmask state of each tensor
        """
        async with self._lock:
            # Determine actual step (periodic or standard)
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                # self._periodic_steps[op_name] += 1 # This is moved to after the loop
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)

            any_fired = False
            cursor = 0
            for i, u in enumerate(tuuids):
                logger.debug(
                    f"[WorkerMailbox] deposit_shards: op={op_name}, tuuid={u}, step={step}, sender={sender_id}:{rank}"
                )

                meta_state = meta_tensor_state_list[i] if meta_tensor_state_list else 0
                has_origin_indices = bool(meta_state & 2)
                compact_params = (
                    slice_compact_desc_params[i] if slice_compact_desc_params else None
                )

                # Payload protocol:
                #   [flat_values, level0_lengths, ..., levelN_lengths, origins?]
                # Planner routes outer rows; length levels preserve the full
                # topology needed to project that row slice to flat values.
                n_elements = 1
                n_elements += ragged_level_count(meta_state)
                if has_origin_indices:
                    n_elements += 1

                sub_payloads = payloads[cursor : cursor + n_elements]
                sub_counts = counts[cursor : cursor + n_elements]
                cursor += n_elements

                global_shape = (
                    global_shapes[i]
                    if global_shapes is not None and i < len(global_shapes)
                    else None
                )

                # Strict Duplicate Validation
                key = (sender_id, rank)
                arrival_slot = self._arrival_slot(op_name, u, actual_step)
                if key in arrival_slot:
                    msg = (
                        f"[WorkerMailbox Error] DUPLICATE SHARD RECEIVED: op={op_name} tuuid={u} step={actual_step} sender={sender_id} rank={rank}. "
                        "This indicates that the same worker is receiving the same shard from the same sender twice. "
                        f"Current arrival metadata: {arrival_slot[key]}"
                    )
                    raise ValueError(msg)
                arrival_slot[key] = {
                    "empty_kind": EmptyKind.DATA_EMPTY_ALLOWED,
                    "meta_state": meta_state,
                    "payload_schema": tuple(
                        {
                            "role": int(
                                PayloadRole.VALUES
                                if j == 0
                                else PayloadRole.ORIGIN
                                if has_origin_indices and j == n_elements - 1
                                else PayloadRole.RAGGED_LENGTHS
                            ),
                            "dtype": np.dtype(sub_payloads[j].dtype),
                            "shape": tuple(
                                int(x) for x in getattr(sub_payloads[j], "shape", ())
                            ),
                            "valid_count": sub_counts[j],
                        }
                        for j in range(n_elements)
                    ),
                    "global_shape": global_shape,
                    "compact_params": tuple(compact_params)
                    if compact_params is not None
                    else None,
                    "sender_world_size": sender_world_size,
                    "expected_count": self._expected_tensor_counts.get(op_name, {}).get(
                        u, 0
                    ),
                }

                # Route to Zero-Copy or Buffered path
                zc = self._zc_receivers.get(op_name, {}).get(u, {}).get(actual_step)
                if zc is not None:
                    # Note: Zero-copy bundling is not yet implemented; use flat sub-lists
                    zc.receive_slices(rank, sub_payloads, sub_counts)
                else:
                    self._deposit_buffered_shard_views(
                        op_name,
                        u,
                        actual_step,
                        sender_id,
                        rank,
                        sender_world_size,
                        sub_payloads,
                        sub_counts,
                        meta_state=meta_state,
                        global_shape=global_shape,
                        compact_desc_params=compact_params,
                    )

                # Notify barrier ONCE per sender for this tensor in the bundle
                if barrier is not None:
                    if barrier.tensor_arrived(u, current_step=actual_step):
                        any_fired = True

            # Fire readiness ONCE per bundle if any barrier satisfied
            if any_fired:
                if op_name in self._periodic_barriers:
                    # Periodic op consumes one virtual step for the entire bundle
                    self._periodic_steps[op_name] += 1

                if self._on_ready:
                    self._on_ready(op_name, actual_step)

        return self._get_size_lock_free()

    async def deposit_transport_frame(
        self,
        op_name: str,
        sender_id: str,
        frame: TransportFrame,
    ) -> int:
        frame.validate()
        step = int(frame.frame_i32[FRAME_STEP])
        rank = int(frame.frame_i32[FRAME_SENDER_RANK])
        sender_world_size = int(frame.frame_i32[FRAME_SENDER_WORLD_SIZE])

        async with self._lock:
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)

            any_fired = False
            watermark = 0
            for edge in frame.edge_i32:
                tuuid = int(edge[EDGE_TUUID])
                meta_state = int(edge[EDGE_META_STATE])
                empty_kind = EmptyKind(int(edge[EDGE_EMPTY_KIND]))
                expected_count = int(edge[EDGE_EXPECTED_COUNT])
                if expected_count <= 0:
                    raise ValueError(
                        f"[WorkerMailbox Error] Invalid EDGE_EXPECTED_COUNT={expected_count}: "
                        f"op={op_name} tuuid={tuuid} step={actual_step} "
                        f"sender={sender_id}:{rank}."
                    )
                if barrier is not None:
                    barrier.configure_expected_tensor_count(tuuid, expected_count)
                payload_begin = int(edge[EDGE_PAYLOAD_BEGIN])
                payload_count = int(edge[EDGE_PAYLOAD_COUNT])
                compact_begin = int(edge[EDGE_COMPACT_BEGIN])
                compact_count = int(edge[EDGE_COMPACT_COUNT])
                global_shape_begin = int(edge[EDGE_GLOBAL_SHAPE_BEGIN])
                global_shape_count = int(edge[EDGE_GLOBAL_SHAPE_COUNT])

                edge_payloads = []
                edge_counts = []
                payload_schema = []
                for payload_row in frame.payload_i32[
                    payload_begin : payload_begin + payload_count
                ]:
                    role = PayloadRole(int(payload_row[PAYLOAD_ROLE]))
                    segment_idx = int(payload_row[PAYLOAD_SEGMENT_ARG_INDEX])
                    valid_count = int(payload_row[PAYLOAD_VALID_COUNT])
                    shape_begin = int(payload_row[PAYLOAD_SHAPE_BEGIN])
                    shape_count = int(payload_row[PAYLOAD_SHAPE_COUNT])
                    shape = tuple(
                        int(x)
                        for x in frame.shape_i32[
                            shape_begin : shape_begin + shape_count
                        ]
                    )
                    if role in (
                        PayloadRole.VALUES,
                        PayloadRole.RAGGED_LENGTHS,
                        PayloadRole.ORIGIN,
                    ):
                        edge_payloads.append(frame.payload_segments[segment_idx])
                        edge_counts.append(valid_count)
                        payload_schema.append(
                            {
                                "role": int(role),
                                "dtype": dtype_from_code(
                                    int(payload_row[PAYLOAD_DTYPE])
                                ),
                                "shape": shape,
                                "valid_count": valid_count,
                                "ndim": int(payload_row[PAYLOAD_NDIM]),
                            }
                        )

                if not edge_payloads:
                    raise ValueError(
                        f"[WorkerMailbox Error] Transport frame edge has no payload schema: "
                        f"op={op_name} tuuid={tuuid} step={actual_step} "
                        f"sender={sender_id}:{rank} empty_kind={empty_kind.name}."
                    )

                data_count = edge_counts[0]
                if empty_kind == EmptyKind.DATA_REQUIRED and data_count == 0:
                    raise ValueError(
                        f"[WorkerMailbox Error] DATA_REQUIRED edge arrived empty: "
                        f"op={op_name} tuuid={tuuid} step={actual_step} "
                        f"sender={sender_id}:{rank}."
                    )

                key = (sender_id, rank)
                arrival_slot = self._arrival_slot(op_name, tuuid, actual_step)
                if key in arrival_slot:
                    raise ValueError(
                        f"[WorkerMailbox Error] DUPLICATE SHARD RECEIVED: "
                        f"op={op_name} tuuid={tuuid} step={actual_step} "
                        f"sender={sender_id} rank={rank}."
                    )

                compact_params = None
                if compact_count:
                    compact_params = tuple(
                        int(x)
                        for x in frame.compact_i64[
                            compact_begin : compact_begin + compact_count
                        ]
                    )
                global_shape = None
                if global_shape_count:
                    global_shape = tuple(
                        int(x)
                        for x in frame.global_shape_i64[
                            global_shape_begin : global_shape_begin + global_shape_count
                        ]
                    )
                arrival_slot[key] = {
                    "empty_kind": empty_kind,
                    "meta_state": meta_state,
                    "payload_schema": tuple(payload_schema),
                    "global_shape": global_shape,
                    "compact_params": compact_params,
                    "sender_world_size": sender_world_size,
                    "expected_count": expected_count,
                }

                if empty_kind != EmptyKind.PROTOCOL_EMPTY:
                    has_compact_index = bool(meta_state & 4)
                    if has_compact_index:
                        if compact_params is None:
                            raise ValueError(
                                f"[WorkerMailbox Error] missing compact metadata: "
                                f"op={op_name} tuuid={tuuid} step={actual_step} "
                                f"sender={sender_id}:{rank}."
                            )
                    watermark = self._deposit_buffered_shard_views(
                        op_name,
                        tuuid,
                        actual_step,
                        sender_id,
                        rank,
                        sender_world_size,
                        edge_payloads,
                        edge_counts,
                        meta_state=meta_state,
                        global_shape=global_shape,
                        compact_desc_params=list(compact_params)
                        if compact_params is not None
                        else None,
                    )

                if barrier is not None:
                    if barrier.tensor_arrived(tuuid, current_step=actual_step):
                        any_fired = True

            if any_fired:
                if op_name in self._periodic_barriers:
                    self._periodic_steps[op_name] += 1
                if self._on_ready:
                    self._on_ready(op_name, actual_step)

            return max(watermark, self._get_size_lock_free())

    async def deposit_local_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: Any,
    ) -> int:
        async with self._lock:
            barrier = self._barriers.get(op_name, {}).get(step)
            if barrier is None:
                raise ValueError(
                    f"[WorkerMailbox] Cannot deposit local tensor before declare_step: "
                    f"op={op_name} tuuid={tuuid} step={step}."
                )
            expected = self._expected_tensor_counts.setdefault(op_name, {}).get(
                tuuid, 1
            )
            if expected != 1:
                raise ValueError(
                    f"[WorkerMailbox] local_only tensor requires expected_count=1, "
                    f"op={op_name} tuuid={tuuid} step={step} expected={expected}."
                )
            self._expected_tensor_counts[op_name][tuuid] = 1
            barrier.configure_expected_tensor_count(tuuid, 1)

            key = (sender_id, rank)
            arrival_slot = self._arrival_slot(op_name, tuuid, step)
            if key in arrival_slot:
                raise ValueError(
                    f"[WorkerMailbox Error] DUPLICATE LOCAL SHARD RECEIVED: "
                    f"op={op_name} tuuid={tuuid} step={step} "
                    f"sender={sender_id} rank={rank}."
                )

            if not isinstance(tensor, TransportTensor):
                raise TypeError(
                    "WorkerMailbox.deposit_local_tensor requires a TransportTensor, "
                    f"got {type(tensor).__name__}."
                )
            values = tensor.flat_values
            backend = get_backend()
            shape = tuple(int(x) for x in backend.shape(values))
            dtype = backend.dtype(values)
            data_count = tensor.routing_entity_count
            global_shape = tensor.global_shape
            arrival_slot[key] = {
                "empty_kind": EmptyKind.DATA_EMPTY_ALLOWED,
                "meta_state": 0,
                "payload_schema": (
                    {
                        "role": int(PayloadRole.VALUES),
                        "dtype": dtype,
                        "shape": shape,
                        "valid_count": data_count,
                        "ndim": len(shape),
                    },
                ),
                "global_shape": global_shape,
                "compact_params": None,
                "sender_world_size": 1,
                "expected_count": 1,
            }
            self._tensor_slots.setdefault(op_name, {}).setdefault(tuuid, {}).setdefault(
                step, {}
            )[key] = [tensor]

            if barrier.tensor_arrived(tuuid, current_step=step) and self._on_ready:
                self._on_ready(op_name, step)
            return data_count

    def _deposit_buffered_shard_views(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        sender_id: str,
        rank: int,
        sender_world_size: int,
        payloads: list[Any],
        counts: list[int],
        meta_state: int,
        global_shape: tuple[int, ...] | None = None,
        compact_desc_params: list[int] | None = None,
    ) -> int:
        is_ragged = bool(meta_state & 1)
        has_origin_indices = bool(meta_state & 2)
        has_compact_index = bool(meta_state & 4)
        expected_payload_count = 1
        if is_ragged:
            expected_payload_count += ragged_level_count(meta_state)
        if has_origin_indices or has_compact_index:
            expected_payload_count += 0 if has_compact_index else 1
        if len(payloads) != expected_payload_count:
            raise ValueError(
                f"[WorkerMailbox Error] payload count mismatch: op={op_name} "
                f"tuuid={tuuid} step={step} sender={sender_id}:{rank} "
                f"payloads={len(payloads)} expected={expected_payload_count}."
            )
        if len(counts) != len(payloads):
            raise ValueError(
                f"[WorkerMailbox Error] count count mismatch: op={op_name} "
                f"tuuid={tuuid} step={step} sender={sender_id}:{rank} "
                f"counts={len(counts)} payloads={len(payloads)}."
            )
        if has_compact_index and compact_desc_params is None:
            raise ValueError(
                f"[WorkerMailbox Error] missing compact metadata: op={op_name} "
                f"tuuid={tuuid} step={step} sender={sender_id}:{rank}."
            )

        key = (sender_id, rank)
        self._buffered_slot(op_name, tuuid, step)[key] = [
            TransportShardView(
                payloads=tuple(payloads),
                counts=tuple(count for count in counts),
                meta_state=meta_state,
                sender_world_size=sender_world_size,
                sender_rank=rank,
                global_shape=global_shape,
                compact_params=tuple(compact_desc_params)
                if compact_desc_params is not None
                else None,
            )
        ]

        # Debug Trace for barrier status
        barrier = self._barriers.get(op_name, {}).get(step)
        if barrier and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[WorkerMailbox] Shard stored: op={op_name} tuuid={tuuid} step={step} rank={rank}"
            )
        return counts[0] if counts else 0

    async def get_all_tensor_shard_views(
        self, op_name: str, step: int
    ) -> dict[int, dict[tuple[str, int], list[TransportShardView]]]:
        async with self._lock:
            expected_counts = self._expected_tensor_counts.get(op_name, {})
            all_tuuids = set(self._tensor_slots.get(op_name, {}).keys()) | set(
                expected_counts.keys()
            )
            result: dict[int, dict[tuple[str, int], list[TransportShardView]]] = {}
            for tuuid in all_tuuids:
                slotted = self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step)
                if slotted and len(slotted) >= expected_counts.get(tuuid, 0):
                    result[tuuid] = dict(slotted)
                    continue

                if slotted:
                    target_count = expected_counts.get(tuuid, 0)
                    if target_count > 0 and len(slotted) < target_count:
                        raise ValueError(
                            f"[WorkerMailbox Error] Incomplete tensor shards for "
                            f"op={op_name} tuuid={tuuid} step={step}: "
                            f"arrived={len(slotted)} expected={target_count}."
                        )
                    result[tuuid] = dict(slotted)
            return result

    def _get_barrier(self, op_name: str, step: int) -> StepBarrier | None:
        """Internal helper to get barrier."""
        if op_name in self._periodic_barriers:
            return self._periodic_barriers[op_name]
        return self._barriers.get(op_name, {}).get(step)

    async def get_tensor_shard_views_for_rank(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        rank: int,
        sender_id: str | None = None,
    ) -> list[TransportShardView] | None:
        async with self._lock:
            slot = self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
            if sender_id is not None:
                return slot.get((sender_id, rank))

            # Search for the first key that matches the rank
            for (sid, r), tensors in slot.items():
                if r == rank:
                    return tensors
            return None

    async def get_tensor_shard_views(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[tuple[str, int], list[TransportShardView]]:
        async with self._lock:
            return dict(
                self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
            )

    async def get_tensor_arrivals(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[tuple[str, int], dict[str, Any]]:
        async with self._lock:
            return dict(
                self._tensor_arrivals.get(op_name, {}).get(tuuid, {}).get(step, {})
            )

    async def get_tensor_shard_views_and_arrivals(
        self, op_name: str, tuuid: int, step: int
    ) -> tuple[
        int,
        dict[tuple[str, int], list[TransportShardView]],
        dict[tuple[str, int], dict[str, Any]],
    ]:
        async with self._lock:
            shards = dict(
                self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
            )
            arrivals = dict(
                self._tensor_arrivals.get(op_name, {}).get(tuuid, {}).get(step, {})
            )
            return step, shards, arrivals

    async def get_tensor_shard_count(self, op_name: str, tuuid: int, step: int) -> int:
        async with self._lock:
            return len(self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {}))

    # ------------------------------------------------------------------
    # Step readiness
    # ------------------------------------------------------------------

    def is_step_ready(self, op_name: str, step: int) -> bool:
        """Non-blocking: True if the step barrier has fired."""
        barrier = self._barriers.get(op_name, {}).get(step)
        return barrier is not None and barrier.is_ready()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            barriers = {
                op_name: {
                    str(step): {
                        "ready": barrier.is_ready(),
                        "tensor_progress": dict(barrier.tensor_progress),
                        "control_progress": {
                            name: f"{arg_barrier.count}/{arg_barrier.target}"
                            for name, arg_barrier in barrier._control_arg_barriers.items()
                        },
                    }
                    for step, barrier in step_map.items()
                }
                for op_name, step_map in self._barriers.items()
            }
            tensor_arrivals = {
                op_name: {
                    str(tuuid): {
                        str(step): len(arrivals) for step, arrivals in step_map.items()
                    }
                    for tuuid, step_map in tuuid_map.items()
                }
                for op_name, tuuid_map in self._tensor_arrivals.items()
            }
            tensor_slots = {
                op_name: {
                    str(tuuid): {
                        str(step): sum(len(items) for items in sender_map.values())
                        for step, sender_map in step_map.items()
                    }
                    for tuuid, step_map in tuuid_map.items()
                }
                for op_name, tuuid_map in self._tensor_slots.items()
            }
            return {
                "barriers": barriers,
                "tensor_arrivals": tensor_arrivals,
                "tensor_slots": tensor_slots,
                "expected_tensor_counts": {
                    op_name: {str(k): v for k, v in values.items()}
                    for op_name, values in self._expected_tensor_counts.items()
                },
            }

    async def wait_step_ready(self, op_name: str, step: int) -> None:
        """Async wait until the step barrier fires.

        Raises RuntimeError if declare_step() was never called for this step.
        """
        barrier = self._get_barrier(op_name, step)
        if barrier is not None:
            await barrier.wait()
        else:
            raise RuntimeError(
                f"No barrier declared for op={op_name} step={step}. Call declare_step() first."
            )

    # ------------------------------------------------------------------
    # Zero-copy registration
    # ------------------------------------------------------------------

    async def register_zero_copy_slot(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        target_buffers: list[Any],
        expect_offsets: dict,
    ) -> None:
        """Enable zero-copy write-through for a tensor slot."""
        async with self._lock:
            receiver = TransportReceiver(wait_for_n_slices=0)
            receiver.register_zero_copy_buffers(target_buffers, expect_offsets)
            self._zc_receivers.setdefault(op_name, {}).setdefault(tuuid, {})[step] = (
                receiver
            )

    async def get_zero_copy_buffers(
        self, op_name: str, tuuid: int, step: int
    ) -> list[Any] | None:
        async with self._lock:
            receiver = self._zc_receivers.get(op_name, {}).get(tuuid, {}).get(step)
            return receiver.target_buffers if receiver is not None else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_step(self, op_name: str, step: int) -> None:
        """Release all data for a completed step."""
        async with self._lock:
            if op_name in self._tensor_slots:
                for name, step_dict in list(self._tensor_slots[op_name].items()):
                    step_dict.pop(step, None)
                    if not step_dict:
                        del self._tensor_slots[op_name][name]

            if op_name in self._zc_receivers:
                for name, step_dict in list(self._zc_receivers[op_name].items()):
                    step_dict.pop(step, None)
                    if not step_dict:
                        del self._zc_receivers[op_name][name]

            if op_name in self._control_arg_data:
                for name, step_dict in list(self._control_arg_data[op_name].items()):
                    step_dict.pop(step, None)
                    if not step_dict:
                        del self._control_arg_data[op_name][name]

            if op_name in self._barriers:
                self._barriers[op_name].pop(step, None)

    async def clear_all(self) -> None:
        async with self._lock:
            self._control_arg_data.clear()
            self._tensor_slots.clear()
            self._zc_receivers.clear()
            self._barriers.clear()

    # ------------------------------------------------------------------
    # Backpressure monitoring
    # ------------------------------------------------------------------

    async def get_size(self) -> int:
        async with self._lock:
            return self._get_size_lock_free()

    def _get_size_lock_free(self) -> int:
        """Internal helper to compute size without acquiring lock."""
        size = 0
        for op_name_dict in self._control_arg_data.values():
            for step_dict in op_name_dict.values():
                size += len(step_dict)
        for op_name_dict in self._tensor_slots.values():
            for step_dict in op_name_dict.values():
                for rank_dict in step_dict.values():
                    size += len(rank_dict)
        return size

    async def get_capacity(self) -> int:
        return 100
