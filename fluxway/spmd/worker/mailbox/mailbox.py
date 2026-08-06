"""
Mailbox module for dual-stream convergence.

Data plane: tensor shards from upstream workers, indexed by (op_type, name, step).
Control plane: state scalars from Driver.

Completion tracking is event-driven, not polled:
  - Each (op_type, step) has a StepBarrier composed of ArrivalBarriers.
  - When all expected shards and states arrive, the barrier fires automatically.
  - Worker registers an on_ready callback.

"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, cast
from collections.abc import Callable

from fluxway.common.tensor_backend import get_backend
from fluxway.common.index_descriptor import ContinuousIndex, ExplicitIndex
from fluxway.common.ragged import ragged_level_count
from fluxway.common.sync import ArrivalBarrier
from fluxway.common.operator_policy import BarrierConfig
from fluxway.common.types.tensor import Scalar, TransportTensor
from fluxway.transport.manifest import (
    DependencyDecision,
    DependencyDecisionMismatchError,
    ENTRY_AUX,
    ENTRY_COMPACT_BEGIN,
    ENTRY_EXPECTED_COUNT,
    ENTRY_GLOBAL_SHAPE_BEGIN,
    ENTRY_GLOBAL_SHAPE_COUNT,
    ENTRY_PAYLOAD_BEGIN,
    ENTRY_PAYLOAD_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_STATE_FLAGS,
    FRAME_FLAGS,
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
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrame,
    dtype_from_code,
)
from fluxway.transport.coherence import entry_layout_state, entry_tensor_kind

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
        backend = get_backend()
        payload = backend.adopt(self._unwrap(self.payloads[index]))
        count = self.counts[index]
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
        return cast(tuple[int, ...], get_backend().shape(self.flat_values))

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
        expected_transfers: {transfer_code: required_shard_count}
        expected_states:  list of required state names
    """

    __slots__ = (
        "_op_name",
        "_step",
        "_transfer_barriers",
        "_control_arg_barriers",
        "_ordering_barriers",
        "_transfer_dependencies",
        "_dependency_transfer_codes",
        "_dependency_codes",
        "_dependency_decision_barriers",
        "_combined",
        "_periodic",
        "_periodic_epoch",
        "_last_completion_skipped",
    )

    def __init__(
        self,
        op_name: str,
        step: int,
        expected_transfers: dict[int, int],
        expected_states: list[str],
        expected_orderings: dict[int, int] | None = None,
        transfer_dependencies: dict[int, int] | None = None,
        expected_dependency_decisions: dict[int, int] | None = None,
        periodic: bool = False,
    ) -> None:
        self._op_name = op_name
        self._step = step
        self._periodic = periodic
        self._periodic_epoch = 0
        self._last_completion_skipped = False
        self._transfer_barriers: dict[int, ArrivalBarrier] = {
            transfer_code: ArrivalBarrier(count)
            for transfer_code, count in expected_transfers.items()
        }
        self._control_arg_barriers: dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(1) for name in expected_states
        }
        self._ordering_barriers: dict[int, ArrivalBarrier] = {
            code: ArrivalBarrier(count)
            for code, count in (expected_orderings or {}).items()
        }
        dependency_codes = (
            {transfer_code: transfer_code for transfer_code in expected_transfers}
            if transfer_dependencies is None
            else dict(transfer_dependencies)
        )
        if set(dependency_codes) != set(expected_transfers):
            raise ValueError(
                f"StepBarrier {op_name}:{step} transfer dependency keys must "
                "exactly match expected transfer codes."
            )
        self._transfer_dependencies = dependency_codes
        grouped: dict[int, list[int]] = {}
        for transfer_code, dependency_code in dependency_codes.items():
            grouped.setdefault(dependency_code, []).append(transfer_code)
        self._dependency_transfer_codes = {
            dependency_code: tuple(sorted(transfer_codes))
            for dependency_code, transfer_codes in grouped.items()
        }
        condition_counts = (
            {}
            if expected_dependency_decisions is None
            else expected_dependency_decisions
        )
        transfer_dependency_codes = set(self._dependency_transfer_codes)
        ordering_dependency_codes = set(self._ordering_barriers)
        overlapping_dependencies = transfer_dependency_codes.intersection(
            ordering_dependency_codes
        )
        if overlapping_dependencies:
            raise ValueError(
                f"StepBarrier {op_name}:{step} dependencies cannot be both tensor "
                f"and ordering dependencies: {sorted(overlapping_dependencies)}."
            )
        self._dependency_decision_barriers: dict[int, _DependencyDecisionBarrier] = {
            dependency_code: _DependencyDecisionBarrier(expected_count)
            for dependency_code, expected_count in condition_counts.items()
        }
        self._dependency_codes = tuple(
            sorted(
                transfer_dependency_codes
                | ordering_dependency_codes
                | set(self._dependency_decision_barriers)
            )
        )
        # Combined completion event (one-shot mode only)
        self._combined = asyncio.Event()
        if not periodic:
            self._check()

    @property
    def arrived_transfers(self) -> list[int]:
        return [
            transfer_code
            for transfer_code, barrier in self._transfer_barriers.items()
            if barrier.count >= barrier.target
        ]

    @property
    def expected_transfers(self) -> list[int]:
        return list(self._transfer_barriers)

    @property
    def transfer_progress(self) -> dict[int, str]:
        return {
            transfer_code: f"{barrier.count}/{barrier.target}"
            for transfer_code, barrier in self._transfer_barriers.items()
        }

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def transfer_arrived(
        self, transfer_code: int, current_step: int | None = None
    ) -> bool:
        barrier = self._transfer_barriers.get(transfer_code)
        if barrier is not None:
            fired = barrier.arrive()
            step_to_log = current_step if current_step is not None else self._step
            logger.debug(
                f"[StepBarrier {self._op_name}:{step_to_log}] TRANSFER ARRIVED: "
                f"transfer_code={transfer_code}, count={barrier.count}/{barrier.target}, "
                f"fired={fired}"
            )
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier] UNEXPECTED TRANSFER: transfer_code={transfer_code}, "
                f"expected={list(self._transfer_barriers)}"
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
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier {self._op_name}:{self._step}] UNEXPECTED STATE: name={name}, expected={list(self._control_arg_barriers.keys())}"
            )
        return False

    def ordering_arrived(
        self, dependency_code: int, current_step: int | None = None
    ) -> bool:
        barrier = self._ordering_barriers.get(dependency_code)
        if barrier is None:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Unexpected ordering "
                f"dependency code={dependency_code}; "
                f"expected={list(self._ordering_barriers)}."
            )
        barrier.arrive()
        return self._check(current_step=current_step)

    def ordering_expected_count(self, dependency_code: int) -> int:
        barrier = self._ordering_barriers.get(dependency_code)
        if barrier is None:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Unexpected ordering "
                f"dependency code={dependency_code}."
            )
        return barrier.target

    def dependency_decision_arrived(
        self,
        dependency_code: int,
        decision: DependencyDecision,
        expected_count: int,
        current_step: int | None = None,
    ) -> bool:
        if dependency_code not in self._dependency_codes:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Unexpected dependency "
                "decision "
                f"dependency code={dependency_code}."
            )
        barrier = self._dependency_decision_barriers.get(dependency_code)
        if barrier is None:
            if decision is not DependencyDecision.SKIP:
                raise ValueError(
                    f"[StepBarrier {self._op_name}:{self._step}] Unconditional "
                    f"dependency code={dependency_code} can only receive propagated "
                    "SKIP decision."
                )
            barrier = _DependencyDecisionBarrier(expected_count)
            self._dependency_decision_barriers[dependency_code] = barrier
        if barrier.expected_count != expected_count:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Dependency decision expected "
                f"count mismatch for dependency_code={dependency_code}: "
                f"frame={expected_count} declared={barrier.expected_count}."
            )
        barrier.arrive(decision)
        return self._check(current_step=current_step)

    def dependency_decision_expected_count(self, dependency_code: int) -> int | None:
        barrier = self._dependency_decision_barriers.get(dependency_code)
        return None if barrier is None else barrier.expected_count

    def is_skipped(self) -> bool:
        return any(
            barrier.is_done and barrier.decision is DependencyDecision.SKIP
            for barrier in self._dependency_decision_barriers.values()
        )

    def transfer_expected_count(self, transfer_code: int) -> int:
        barrier = self._transfer_barriers.get(transfer_code)
        if barrier is None:
            raise ValueError(
                f"[StepBarrier {self._op_name}:{self._step}] Unexpected "
                f"transfer_code={transfer_code}; "
                f"expected={list(self._transfer_barriers)}."
            )
        return barrier.target

    # ------------------------------------------------------------------
    # Completion query
    # ------------------------------------------------------------------

    def _check(self, current_step: int | None = None) -> bool:
        if not self._periodic and self._combined.is_set():
            return False

        def arrival_is_done(barrier: ArrivalBarrier) -> bool:
            if not self._periodic:
                return barrier.is_done()
            return barrier.count >= barrier.target * (self._periodic_epoch + 1)

        dependencies_ok = True
        for dependency_code in self._dependency_codes:
            decision_barrier = self._dependency_decision_barriers.get(dependency_code)
            if decision_barrier is not None and not decision_barrier.is_done:
                dependencies_ok = False
                break
            if (
                decision_barrier is not None
                and decision_barrier.decision is DependencyDecision.SKIP
            ):
                continue
            transfer_codes = self._dependency_transfer_codes.get(dependency_code, ())
            if not all(
                arrival_is_done(self._transfer_barriers[transfer_code])
                for transfer_code in transfer_codes
            ):
                dependencies_ok = False
                break
            ordering_barrier = self._ordering_barriers.get(dependency_code)
            if ordering_barrier is not None and not arrival_is_done(ordering_barrier):
                dependencies_ok = False
                break
        skipped = self.is_skipped()
        states_ok = skipped or all(
            arrival_is_done(b) for b in self._control_arg_barriers.values()
        )
        step_to_log = current_step if current_step is not None else self._step

        if dependencies_ok and states_ok:
            logger.debug(f"[StepBarrier {self._op_name}:{step_to_log}] FIRED.")
            self._last_completion_skipped = skipped
            if self._periodic:
                self._combined.set()
                self._combined = asyncio.Event()
                self._periodic_epoch += 1
                for barrier in self._dependency_decision_barriers.values():
                    barrier.reset()
                return True
            self._combined.set()
            return True

        # Log why it's not ready
        waiting_transfers = [
            transfer_code
            for transfer_code, barrier in self._transfer_barriers.items()
            if not arrival_is_done(barrier)
        ]
        waiting_states = [
            n for n, b in self._control_arg_barriers.items() if not arrival_is_done(b)
        ]
        logger.debug(
            f"[StepBarrier {self._op_name}:{step_to_log}] NOT READY: "
            f"waiting_transfers={waiting_transfers}, waiting_states={waiting_states}"
        )
        return False

    def is_ready(self) -> bool:
        if self._periodic:
            return self._periodic_epoch > 0
        return self._combined.is_set()

    @property
    def last_completion_skipped(self) -> bool:
        return self._last_completion_skipped

    def recompute_ready(self, current_step: int | None = None) -> bool:
        return self._check(current_step=current_step)

    async def wait(self) -> None:
        await self._combined.wait()


@dataclass(slots=True)
class _DependencyDecisionBarrier:
    expected_count: int
    decision: DependencyDecision | None = None
    count: int = 0

    def __post_init__(self) -> None:
        if self.expected_count <= 0:
            raise ValueError(
                f"Dependency decision expected_count must be positive, got {self.expected_count}."
            )

    @property
    def is_done(self) -> bool:
        return self.count == self.expected_count

    def arrive(self, decision: DependencyDecision) -> None:
        if self.decision is not None and self.decision is not decision:
            raise DependencyDecisionMismatchError(
                f"Dependency decisions disagree: first={self.decision.name} "
                f"incoming={decision.name}."
            )
        if self.count >= self.expected_count:
            raise ValueError(
                "Dependency decision arrivals exceed the declared expected count."
            )
        self.decision = decision
        self.count += 1

    def reset(self) -> None:
        self.decision = None
        self.count = 0


# ---------------------------------------------------------------------------
# WorkerMailbox
# ---------------------------------------------------------------------------


@dataclass
class MailboxStepState:
    control_args: dict[str, Any] = field(default_factory=dict)
    transfer_slots: dict[
        int, dict[tuple[str, int], list[TransportShardView | TransportTensor]]
    ] = field(default_factory=dict)
    transfer_arrivals: dict[int, dict[tuple[str, int], dict[str, Any]]] = field(
        default_factory=dict
    )
    ordering_arrivals: dict[int, set[tuple[str, int]]] = field(default_factory=dict)
    dependency_decision_arrivals: dict[
        int, dict[tuple[str, int], DependencyDecision]
    ] = field(default_factory=dict)


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

    """

    def __init__(self) -> None:
        self._states: dict[str, dict[int, MailboxStepState]] = {}

        # Completion barriers: {op_name: {step: StepBarrier}}
        self._barriers: dict[str, dict[int, StepBarrier]] = {}

        # Called with (op_name, step) when a step barrier fires
        self._on_ready: Callable[[str, int], None] | None = None

        # Periodic mode state: {op_name: current_virtual_step}
        self._periodic_steps: dict[str, int] = {}
        # Mapping from periodic op to its perpetual barrier
        self._periodic_barriers: dict[str, StepBarrier] = {}
        self._periodic_skipped_steps: dict[str, set[int]] = {}

        # Expected counts for satisfying staleness: {op_name: {transfer_code: expected_count}}
        self._expected_transfer_counts: dict[str, dict[int, int]] = {}
        self._expected_control_arg_names: dict[str, list[str]] = {}
        self._expected_ordering_counts: dict[str, dict[int, int]] = {}
        self._expected_dependency_decisions: dict[str, dict[int, int]] = {}
        self._transfer_dependencies: dict[str, dict[int, int]] = {}

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Step barrier lifecycle
    # ------------------------------------------------------------------

    def _get_step_state(self, op_name: str, step: int) -> MailboxStepState:
        return self._states.setdefault(op_name, {}).setdefault(step, MailboxStepState())

    def set_ready_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Register a callback fired when a step becomes ready.

        The callback is invoked inside the asyncio.Lock, synchronously —
        use asyncio.create_task() inside it for non-blocking work.

        Args:
            callback: callable(op_name, step)
        """
        self._on_ready = callback

    def _step_completed(self, op_name: str, step: int, barrier: StepBarrier) -> None:
        if op_name in self._periodic_barriers:
            if barrier.last_completion_skipped:
                self._periodic_skipped_steps.setdefault(op_name, set()).add(step)
            self._periodic_steps[op_name] += 1
        if self._on_ready:
            self._on_ready(op_name, step)

    def declare_step(
        self,
        op_name: str,
        step: int,
        expected_transfers: dict[int, int],
        expected_states: list[str],
        barrier_config: Optional["BarrierConfig"] = None,
        expected_orderings: dict[int, int] | None = None,
        transfer_dependencies: dict[int, int] | None = None,
        expected_dependency_decisions: dict[int, int] | None = None,
    ) -> None:
        """
        Declare the expected arrivals for a step.

        Must be called before the first tensor frame or control argument for
        that step.

        Args:
            op_name: Operation name
            step: Execution step (ignored for periodic ops)
            expected_transfers: {transfer_code: required_shard_count}
            expected_states: list of required state names
            barrier_config: Configuration for the step barrier
        """
        from fluxway.common.operator_policy import BarrierType

        is_periodic = (
            barrier_config.type == BarrierType.PERIODIC if barrier_config else False
        )

        ordering_counts = {} if expected_orderings is None else expected_orderings
        condition_counts = (
            {}
            if expected_dependency_decisions is None
            else expected_dependency_decisions
        )
        dependency_codes = (
            {transfer_code: transfer_code for transfer_code in expected_transfers}
            if transfer_dependencies is None
            else transfer_dependencies
        )
        self._expected_transfer_counts[op_name] = expected_transfers
        self._expected_control_arg_names[op_name] = expected_states
        self._expected_ordering_counts[op_name] = ordering_counts
        self._expected_dependency_decisions[op_name] = condition_counts
        self._transfer_dependencies[op_name] = dict(dependency_codes)

        if is_periodic:
            if op_name not in self._periodic_barriers:
                # Use n from barrier_config if available
                # Note: StepBarrier currently assumes n is the same for all inputs in periodic mode.
                # For now we use barrier_config.n if it is PeriodicBarrierConfig.
                from fluxway.common.operator_policy import PeriodicBarrierConfig

                n = (
                    barrier_config.n
                    if isinstance(barrier_config, PeriodicBarrierConfig)
                    else 1
                )
                self._periodic_barriers[op_name] = StepBarrier(
                    op_name,
                    -1,
                    expected_transfers,
                    expected_states,
                    ordering_counts,
                    dependency_codes,
                    condition_counts,
                    periodic=True,
                )
                # Ensure each ArrivalBarrier inside StepBarrier uses the correct n
                for b in self._periodic_barriers[op_name]._transfer_barriers.values():
                    b._n = n
                for b in self._periodic_barriers[
                    op_name
                ]._control_arg_barriers.values():
                    b._n = n
                for b in self._periodic_barriers[op_name]._ordering_barriers.values():
                    b._n = n

                self._periodic_steps[op_name] = 0
            return

        if op_name in self._barriers and step in self._barriers[op_name]:
            old_b = self._barriers[op_name][step]
            logger.warning(
                f"[WorkerMailbox] DUPLICATE declare_step: op={op_name}, step={step}. "
                f"Progress: {old_b.transfer_progress}"
            )

        barrier = StepBarrier(
            op_name,
            step,
            expected_transfers,
            expected_states,
            ordering_counts,
            dependency_codes,
            condition_counts,
        )
        self._barriers.setdefault(op_name, {})[step] = barrier
        logger.debug(
            f"[WorkerMailbox] declare_step: op={op_name}, step={step}, "
            f"expected_transfers={expected_transfers}, expected_states={expected_states}"
        )

        # Check if any data for THIS EXACT step has already arrived.
        # This prevents progress loss if declare_step is called after some deposit_* calls.
        if expected_transfers:
            for transfer_code in expected_transfers:
                state = self._states.get(op_name, {}).get(step)
                transfer_arrivals = (
                    state.transfer_arrivals.get(transfer_code, {}) if state else {}
                )
                for _ in transfer_arrivals:
                    barrier.transfer_arrived(transfer_code)
        # Check for states
        for name in expected_states:
            state = self._states.get(op_name, {}).get(step)
            if state and name in state.control_args:
                barrier.control_arg_arrived(name)
        state = self._states.get(op_name, {}).get(step)
        if state is not None:
            for dependency_code, ordering_senders in state.ordering_arrivals.items():
                for _ in ordering_senders:
                    barrier.ordering_arrived(dependency_code)
            for (
                dependency_code,
                decision_senders,
            ) in state.dependency_decision_arrivals.items():
                expected_count = condition_counts.get(dependency_code)
                for decision in decision_senders.values():
                    barrier.dependency_decision_arrived(
                        dependency_code,
                        decision,
                        (
                            len(decision_senders)
                            if expected_count is None
                            else expected_count
                        ),
                    )

        # In case expected counts are all zero the barrier is already ready
        if barrier.is_ready() and self._on_ready:
            self._on_ready(op_name, step)

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
                state = self._get_step_state(op_name, actual_step)
                state.control_args[name] = data
                if barrier is not None:
                    if barrier.control_arg_arrived(name, current_step=actual_step):
                        any_fired = True

            if any_fired and barrier is not None:
                self._step_completed(op_name, actual_step, barrier)
        logger.debug(
            "Mailbox deposit_control_arguments: op=%s names=%s step=%d",
            op_name,
            names,
            step,
        )

    async def deposit_ordering(
        self,
        op_name: str,
        dependency_code: int,
        step: int,
        *,
        sender_id: str,
        rank: int,
        expected_count: int,
    ) -> int:
        async with self._lock:
            barrier: StepBarrier | None
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)
            if barrier is None:
                raise ValueError(
                    f"[WorkerMailbox] Cannot deposit ordering before declare_step: "
                    f"op={op_name} dependency_code={dependency_code} step={step}."
                )
            declared_count = barrier.ordering_expected_count(dependency_code)
            if expected_count != declared_count:
                raise ValueError(
                    f"[WorkerMailbox] Ordering expected count mismatch: op={op_name} "
                    f"dependency_code={dependency_code} step={actual_step} "
                    f"frame={expected_count} declared={declared_count}."
                )
            state = self._get_step_state(op_name, actual_step)
            arrivals = state.ordering_arrivals.setdefault(dependency_code, set())
            sender = (sender_id, rank)
            if sender in arrivals:
                raise ValueError(
                    f"[WorkerMailbox Error] DUPLICATE ORDERING RECEIVED: "
                    f"op={op_name} dependency_code={dependency_code} "
                    f"step={actual_step} sender={sender_id}:{rank}."
                )
            arrivals.add(sender)
            fired = barrier.ordering_arrived(dependency_code, current_step=actual_step)
            if fired:
                self._step_completed(op_name, actual_step, barrier)
            return self._get_size_lock_free()

    async def deposit_dependency_decision(
        self,
        op_name: str,
        dependency_code: int,
        step: int,
        *,
        sender_id: str,
        rank: int,
        expected_count: int,
        decision: DependencyDecision,
    ) -> int:
        if not isinstance(decision, DependencyDecision):
            raise TypeError(
                f"decision must be DependencyDecision, got {type(decision).__name__}."
            )
        async with self._lock:
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)
            if barrier is None:
                raise ValueError(
                    "[WorkerMailbox] Cannot deposit dependency decision before "
                    "declare_step: "
                    f"op={op_name} dependency_code={dependency_code} step={step}."
                )
            declared_count = barrier.dependency_decision_expected_count(dependency_code)
            if declared_count is not None and expected_count != declared_count:
                raise ValueError(
                    f"[WorkerMailbox] Dependency decision expected count mismatch: op={op_name} "
                    f"dependency_code={dependency_code} step={actual_step} "
                    f"frame={expected_count} declared={declared_count}."
                )
            state = self._get_step_state(op_name, actual_step)
            arrivals = state.dependency_decision_arrivals.setdefault(
                dependency_code, {}
            )
            sender = (sender_id, rank)
            if sender in arrivals:
                raise ValueError(
                    f"[WorkerMailbox Error] DUPLICATE DEPENDENCY DECISION RECEIVED: "
                    f"op={op_name} dependency_code={dependency_code} "
                    f"step={actual_step} sender={sender_id}:{rank}."
                )
            fired = barrier.dependency_decision_arrived(
                dependency_code,
                decision,
                expected_count,
                current_step=actual_step,
            )
            arrivals[sender] = decision
            if fired:
                self._step_completed(op_name, actual_step, barrier)
            return self._get_size_lock_free()

    async def get_control_argument(
        self, op_name: str, name: str, step: int
    ) -> Any | None:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return state.control_args.get(name) if state else None

    async def get_all_control_arguments(
        self, op_name: str, step: int
    ) -> dict[str, Any]:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return dict(state.control_args) if state else {}

    async def has_control_argument(self, op_name: str, name: str, step: int) -> bool:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return name in state.control_args if state else False

    # ------------------------------------------------------------------
    # Data plane — tensor shards
    # ------------------------------------------------------------------

    def _buffered_transfer_slot(
        self, op_name: str, transfer_code: int, step: int
    ) -> dict[tuple[str, int], list[TransportShardView | TransportTensor]]:
        state = self._get_step_state(op_name, step)
        return state.transfer_slots.setdefault(transfer_code, {})

    def _transfer_arrival_slot(
        self, op_name: str, transfer_code: int, step: int
    ) -> dict[tuple[str, int], dict[str, Any]]:
        state = self._get_step_state(op_name, step)
        return state.transfer_arrivals.setdefault(transfer_code, {})

    async def deposit_transport_frame(
        self,
        op_name: str,
        sender_id: str,
        frame: TransportFrame,
    ) -> int:
        frame.validate()
        frame_flag = FrameFlag(int(frame.frame_i32[FRAME_FLAGS]))
        if frame_flag is not FrameFlag.TENSOR_TRANSFER:
            raise ValueError(
                "WorkerMailbox.deposit_transport_frame requires FrameFlag.TENSOR_TRANSFER, "
                f"got {frame_flag.name}."
            )
        step = int(frame.frame_i32[FRAME_STEP])
        rank = int(frame.frame_i32[FRAME_SENDER_RANK])
        sender_world_size = int(frame.frame_i32[FRAME_SENDER_WORLD_SIZE])

        async with self._lock:
            barrier: StepBarrier | None
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)
            if barrier is None:
                raise ValueError(
                    f"[WorkerMailbox] Cannot deposit tensor frame before "
                    f"declare_step: op={op_name} step={step}."
                )

            any_fired = False
            watermark = 0
            for entry in frame.entry_i32:
                transfer_code = int(entry[ENTRY_ROUTE_CODE])
                state_flags = int(entry[ENTRY_STATE_FLAGS])
                meta_state = entry_layout_state(state_flags)
                tensor_kind = TensorEntryKind(entry_tensor_kind(state_flags))
                expected_count = int(entry[ENTRY_EXPECTED_COUNT])
                source_step = step
                if expected_count <= 0:
                    raise ValueError(
                        f"[WorkerMailbox Error] Invalid ENTRY_EXPECTED_COUNT={expected_count}: "
                        f"op={op_name} transfer_code={transfer_code} step={actual_step} "
                        f"sender={sender_id}:{rank}."
                    )
                declared_count = barrier.transfer_expected_count(transfer_code)
                if expected_count != declared_count:
                    raise ValueError(
                        f"[WorkerMailbox] Tensor expected count mismatch: "
                        f"op={op_name} transfer_code={transfer_code} "
                        f"step={actual_step} frame={expected_count} "
                        f"declared={declared_count}."
                    )
                payload_begin = int(entry[ENTRY_PAYLOAD_BEGIN])
                payload_count = int(entry[ENTRY_PAYLOAD_COUNT])
                compact_begin = int(entry[ENTRY_COMPACT_BEGIN])
                compact_count = 4 if meta_state & 4 else 0
                global_shape_begin = int(entry[ENTRY_GLOBAL_SHAPE_BEGIN])
                global_shape_count = int(entry[ENTRY_GLOBAL_SHAPE_COUNT])

                entry_payloads = []
                payload_valid_counts = []
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
                        entry_payloads.append(frame.payload_segments[segment_idx])
                        payload_valid_counts.append(valid_count)
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

                if not entry_payloads:
                    raise ValueError(
                        f"[WorkerMailbox Error] Transport frame entry has no payload schema: "
                        f"op={op_name} transfer_code={transfer_code} "
                        f"step={actual_step} "
                        f"sender={sender_id}:{rank} tensor_kind={tensor_kind.name}."
                    )

                key = (sender_id, rank)
                arrival_slot = self._transfer_arrival_slot(
                    op_name, transfer_code, actual_step
                )
                if key in arrival_slot:
                    raise ValueError(
                        f"[WorkerMailbox Error] DUPLICATE SHARD RECEIVED: "
                        f"op={op_name} transfer_code={transfer_code} "
                        f"step={actual_step} "
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
                arrival_metadata = {
                    "tensor_kind": tensor_kind,
                    "meta_state": meta_state,
                    "payload_schema": tuple(payload_schema),
                    "global_shape": global_shape,
                    "compact_params": compact_params,
                    "sender_world_size": sender_world_size,
                    "expected_count": expected_count,
                    "source_step": source_step,
                    "coherence_epoch": int(entry[ENTRY_AUX]),
                }

                if tensor_kind is TensorEntryKind.TENSOR:
                    has_compact_index = bool(meta_state & 4)
                    if has_compact_index:
                        if compact_params is None:
                            raise ValueError(
                                f"[WorkerMailbox Error] missing compact metadata: "
                                f"op={op_name} transfer_code={transfer_code} "
                                f"step={actual_step} "
                                f"sender={sender_id}:{rank}."
                            )
                    watermark = self._deposit_buffered_shard_views(
                        op_name,
                        transfer_code,
                        actual_step,
                        sender_id,
                        rank,
                        sender_world_size,
                        entry_payloads,
                        payload_valid_counts,
                        meta_state=meta_state,
                        global_shape=global_shape,
                        compact_desc_params=list(compact_params)
                        if compact_params is not None
                        else None,
                    )

                arrival_slot[key] = arrival_metadata

                if barrier.transfer_arrived(transfer_code, current_step=actual_step):
                    any_fired = True

            if any_fired:
                self._step_completed(op_name, actual_step, barrier)

            return max(watermark, self._get_size_lock_free())

    async def deposit_local_tensor(
        self,
        op_name: str,
        transfer_code: int,
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
                    f"op={op_name} transfer_code={transfer_code} step={step}."
                )
            expected = self._expected_transfer_counts.setdefault(op_name, {}).get(
                transfer_code, 1
            )
            if expected != 1:
                raise ValueError(
                    f"[WorkerMailbox] local_only tensor requires expected_count=1, "
                    f"op={op_name} transfer_code={transfer_code} step={step} "
                    f"expected={expected}."
                )
            key = (sender_id, rank)
            arrival_slot = self._transfer_arrival_slot(op_name, transfer_code, step)
            if key in arrival_slot:
                raise ValueError(
                    f"[WorkerMailbox Error] DUPLICATE LOCAL SHARD RECEIVED: "
                    f"op={op_name} transfer_code={transfer_code} step={step} "
                    f"sender={sender_id} rank={rank}."
                )

            if not isinstance(tensor, TransportTensor):
                raise TypeError(
                    "WorkerMailbox.deposit_local_tensor requires a TransportTensor, "
                    f"got {type(tensor).__name__}."
                )
            if tensor.source_step is None:
                raise ValueError(
                    "WorkerMailbox.deposit_local_tensor requires tensor.source_step, "
                    f"op={op_name} transfer_code={transfer_code} step={step} "
                    f"sender={sender_id}:{rank}."
                )
            values = tensor.flat_values
            backend = get_backend()
            shape = tuple(int(x) for x in backend.shape(values))
            dtype = backend.dtype(values)
            data_count = tensor.routing_entity_count
            global_shape = tensor.global_shape
            arrival_slot[key] = {
                "tensor_kind": TensorEntryKind.TENSOR,
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
                "source_step": tensor.source_step,
            }
            state = self._get_step_state(op_name, step)
            state.transfer_slots.setdefault(transfer_code, {})[key] = [tensor]

            if (
                barrier.transfer_arrived(transfer_code, current_step=step)
                and self._on_ready
            ):
                self._on_ready(op_name, step)
            return data_count

    def _deposit_buffered_shard_views(
        self,
        op_name: str,
        transfer_code: int,
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
                f"transfer_code={transfer_code} step={step} sender={sender_id}:{rank} "
                f"payloads={len(payloads)} expected={expected_payload_count}."
            )
        if len(counts) != len(payloads):
            raise ValueError(
                f"[WorkerMailbox Error] count count mismatch: op={op_name} "
                f"transfer_code={transfer_code} step={step} sender={sender_id}:{rank} "
                f"counts={len(counts)} payloads={len(payloads)}."
            )
        if has_compact_index and compact_desc_params is None:
            raise ValueError(
                f"[WorkerMailbox Error] missing compact metadata: op={op_name} "
                f"transfer_code={transfer_code} step={step} sender={sender_id}:{rank}."
            )

        key = (sender_id, rank)
        self._buffered_transfer_slot(op_name, transfer_code, step)[key] = [
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
                f"[WorkerMailbox] Shard stored: op={op_name} "
                f"transfer_code={transfer_code} step={step} rank={rank}"
            )
        return counts[0] if counts else 0

    async def get_all_tensor_shard_views(
        self, op_name: str, step: int
    ) -> dict[int, dict[tuple[str, int], list[TransportShardView | TransportTensor]]]:
        async with self._lock:
            expected_counts = self._expected_transfer_counts.get(op_name, {})
            state = self._states.get(op_name, {}).get(step)
            transfer_codes = set(expected_counts)
            if state:
                transfer_codes |= set(state.transfer_slots)
            result: dict[
                int, dict[tuple[str, int], list[TransportShardView | TransportTensor]]
            ] = {}
            for transfer_code in transfer_codes:
                slotted = state.transfer_slots.get(transfer_code) if state else None
                if slotted and len(slotted) >= expected_counts.get(transfer_code, 0):
                    result[transfer_code] = dict(slotted)
                    continue

                if slotted:
                    target_count = expected_counts.get(transfer_code, 0)
                    if target_count > 0 and len(slotted) < target_count:
                        raise ValueError(
                            f"[WorkerMailbox Error] Incomplete tensor shards for "
                            f"op={op_name} transfer_code={transfer_code} step={step}: "
                            f"arrived={len(slotted)} expected={target_count}."
                        )
                    result[transfer_code] = dict(slotted)
            return result

    def _get_barrier(self, op_name: str, step: int) -> StepBarrier | None:
        """Internal helper to get barrier."""
        if op_name in self._periodic_barriers:
            return self._periodic_barriers[op_name]
        return self._barriers.get(op_name, {}).get(step)

    async def get_tensor_shard_views_for_rank(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        rank: int,
        sender_id: str | None = None,
    ) -> list[TransportShardView | TransportTensor] | None:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            slot = state.transfer_slots.get(transfer_code, {}) if state else {}
            if sender_id is not None:
                return slot.get((sender_id, rank))

            # Search for the first key that matches the rank
            for (sid, r), tensors in slot.items():
                if r == rank:
                    return tensors
            return None

    async def get_tensor_shard_views(
        self, op_name: str, transfer_code: int, step: int
    ) -> dict[tuple[str, int], list[TransportShardView | TransportTensor]]:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return dict(state.transfer_slots.get(transfer_code, {})) if state else {}

    async def get_tensor_arrivals(
        self, op_name: str, transfer_code: int, step: int
    ) -> dict[tuple[str, int], dict[str, Any]]:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return dict(state.transfer_arrivals.get(transfer_code, {})) if state else {}

    async def get_tensor_shard_views_and_arrivals(
        self, op_name: str, transfer_code: int, step: int
    ) -> tuple[
        int,
        dict[tuple[str, int], list[TransportShardView | TransportTensor]],
        dict[tuple[str, int], dict[str, Any]],
    ]:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return self._snapshot_tensor_shard_views_and_arrivals(
                op_name, transfer_code, step, state
            )

    async def get_tensor_shard_views_and_arrivals_many(
        self,
        op_name: str,
        transfer_codes: list[int] | tuple[int, ...],
        step: int,
    ) -> list[
        tuple[
            int,
            dict[tuple[str, int], list[TransportShardView | TransportTensor]],
            dict[tuple[str, int], dict[str, Any]],
        ]
    ]:
        if len(transfer_codes) != len(set(transfer_codes)):
            raise ValueError(
                f"tensor snapshot transfer_codes must be unique: op={op_name} "
                f"step={step} transfer_codes={transfer_codes}."
            )
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return [
                self._snapshot_tensor_shard_views_and_arrivals(
                    op_name, transfer_code, step, state
                )
                for transfer_code in transfer_codes
            ]

    @staticmethod
    def _snapshot_tensor_shard_views_and_arrivals(
        op_name: str,
        transfer_code: int,
        step: int,
        state: MailboxStepState | None,
    ) -> tuple[
        int,
        dict[tuple[str, int], list[TransportShardView | TransportTensor]],
        dict[tuple[str, int], dict[str, Any]],
    ]:
        shards = dict(state.transfer_slots.get(transfer_code, {})) if state else {}
        arrivals = dict(state.transfer_arrivals.get(transfer_code, {})) if state else {}
        source_steps = {
            int(metadata["source_step"])
            for metadata in arrivals.values()
            if "source_step" in metadata
        }
        if not source_steps:
            raise ValueError(
                f"[WorkerMailbox Error] missing source_step arrival metadata: "
                f"op={op_name} transfer_code={transfer_code} step={step}."
            )
        if len(source_steps) != 1:
            raise ValueError(
                f"[WorkerMailbox Error] mixed source_step arrival metadata: "
                f"op={op_name} transfer_code={transfer_code} step={step} "
                f"source_steps={sorted(source_steps)}."
            )
        return next(iter(source_steps)), shards, arrivals

    async def get_tensor_shard_count(
        self, op_name: str, transfer_code: int, step: int
    ) -> int:
        async with self._lock:
            state = self._states.get(op_name, {}).get(step)
            return len(state.transfer_slots.get(transfer_code, {})) if state else 0

    # ------------------------------------------------------------------
    # Step readiness
    # ------------------------------------------------------------------

    def is_step_ready(self, op_name: str, step: int) -> bool:
        """Non-blocking: True if the step barrier has fired."""
        if op_name in self._periodic_steps:
            return step < self._periodic_steps[op_name]
        barrier = self._barriers.get(op_name, {}).get(step)
        return barrier is not None and barrier.is_ready()

    def is_step_skipped(self, op_name: str, step: int) -> bool:
        if op_name in self._periodic_steps:
            return step in self._periodic_skipped_steps.get(op_name, ())
        barrier = self._barriers.get(op_name, {}).get(step)
        return barrier is not None and barrier.is_skipped()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            barriers = {
                op_name: {
                    str(step): {
                        "ready": barrier.is_ready(),
                        "transfer_progress": dict(barrier.transfer_progress),
                        "control_progress": {
                            name: f"{arg_barrier.count}/{arg_barrier.target}"
                            for name, arg_barrier in barrier._control_arg_barriers.items()
                        },
                    }
                    for step, barrier in step_map.items()
                }
                for op_name, step_map in self._barriers.items()
            }
            transfer_arrivals: dict[str, dict[str, dict[str, int]]] = {}
            transfer_slots: dict[str, dict[str, dict[str, int]]] = {}
            for op_name, steps in self._states.items():
                transfer_arrivals[op_name] = {}
                transfer_slots[op_name] = {}
                for step, state in steps.items():
                    for (
                        transfer_code,
                        arrival_senders,
                    ) in state.transfer_arrivals.items():
                        transfer_arrivals[op_name].setdefault(str(transfer_code), {})[
                            str(step)
                        ] = len(arrival_senders)
                    for transfer_code, slot_senders in state.transfer_slots.items():
                        transfer_slots[op_name].setdefault(str(transfer_code), {})[
                            str(step)
                        ] = sum(len(x) for x in slot_senders.values())
            return {
                "barriers": barriers,
                "transfer_arrivals": transfer_arrivals,
                "transfer_slots": transfer_slots,
                "expected_transfer_counts": {
                    op_name: {str(k): v for k, v in values.items()}
                    for op_name, values in self._expected_transfer_counts.items()
                },
            }

    async def wait_step_ready(self, op_name: str, step: int) -> None:
        """Async wait until the step barrier fires.

        Raises RuntimeError if declare_step() was never called for this step.
        """
        barrier = self._get_barrier(op_name, step)
        if barrier is None:
            raise RuntimeError(
                f"No barrier declared for op={op_name} step={step}. Call declare_step() first."
            )
        while not self.is_step_ready(op_name, step):
            await barrier.wait()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_step(self, op_name: str, step: int) -> None:
        """Release all data for a completed step."""
        async with self._lock:
            if op_name in self._states:
                self._states[op_name].pop(step, None)
                if not self._states[op_name]:
                    del self._states[op_name]

            if op_name in self._barriers:
                self._barriers[op_name].pop(step, None)
            if op_name in self._periodic_skipped_steps:
                self._periodic_skipped_steps[op_name].discard(step)

    async def clear_all(self) -> None:
        async with self._lock:
            self._states.clear()
            self._barriers.clear()
            self._periodic_skipped_steps.clear()

    # ------------------------------------------------------------------
    # Backpressure monitoring
    # ------------------------------------------------------------------

    async def get_size(self) -> int:
        async with self._lock:
            return self._get_size_lock_free()

    def _get_size_lock_free(self) -> int:
        """Internal helper to compute size without acquiring lock."""
        size = 0
        for steps in self._states.values():
            for state in steps.values():
                size += len(state.control_args)
                for rank_dict in state.transfer_slots.values():
                    size += len(rank_dict)
        return size

    async def get_capacity(self) -> int:
        return 100
